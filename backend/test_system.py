"""驗證實測資料解析、模型評分與服務 API。"""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from openpyxl import Workbook

from .app import compare_landing, create_app, predict
from .excel_loader import EXCEL_HEADERS, DataError, ExcelSource, parse_integer, read_excel
from .storage import PublishedRun, read_json, save_uploads, write_json
from .training import evaluate, make_folds, make_model, score_axis


def workbook_bytes(rows, extra_sheets=None):
    """以指定資料列建立記憶體中的 Excel 測試檔。"""
    workbook = Workbook()
    workbook.active.title = "實測"
    for sheet, data in {"實測": rows, **(extra_sheets or {})}.items():
        target = workbook.active if sheet == "實測" else workbook.create_sheet(sheet)
        for row in data:
            target.append(row)
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


class DataTests(unittest.TestCase):
    """涵蓋 Excel 格式、資料類型及逐球展開規則。"""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def source(self, number, contents):
        """將測試 Excel 放進此案例的暫存目錄。"""
        path = self.directory / f"{number}.xlsx"
        path.write_bytes(contents)
        return ExcelSource(number, "同名.xlsx", path)

    def test_expansion_keeps_source_and_repeats(self):
        """確認重複資料有各自的球號，並可追溯來源。"""
        row = ("A", "13.0", 2, 3, 4, "(1, 2)", "(1,2)", "未過網", None, "(0,3)")
        data = workbook_bytes(
            [(*EXCEL_HEADERS, "補測順位"), ("預填",), row, row,
             ("全缺測", 1, 2, 3, 4, "(0,0)", "無效球", "（1，2）")],
            {"補測": [EXCEL_HEADERS, row], "說明": [("項目", "內容")]},
        )
        balls, counts = read_excel([self.source(n, data) for n in (1, 2)], lambda *args: None)
        self.assertEqual(len(balls), 18)
        self.assertEqual(len({ball.ball_id for ball in balls}), 18)
        self.assertEqual(balls[0].row, 3)
        self.assertEqual(balls[0].parameters, (13, 2, 3, 4))
        self.assertEqual(balls[0].test_id, "A")
        self.assertEqual([ball.landing for ball in balls[:3]], [(1, 2), (1, 2), (0, 0)])
        self.assertEqual(
            [
                counts[name]
                for name in ("total_balls", "valid_balls", "invalid_balls", "missing_balls")
            ],
            [40, 12, 6, 22],
        )
        self.assertEqual(counts["sources"][0]["sheets"], ["實測", "補測"])

    def test_invalid_parameter_even_if_all_landings_are_missing(self):
        """確認落點全缺測的資料列仍會檢查四參數。"""
        data = workbook_bytes([EXCEL_HEADERS, ("A", 13.5, 2, 3, 4)])
        with self.assertRaises(DataError) as failure:
            read_excel([self.source(1, data)], lambda *args: None)
        self.assertEqual(failure.exception.details, {
            "file": "同名.xlsx", "sheet": "實測", "row": 2, "field": "仰角角度", "raw_value": "13.5",
        })

    def test_every_file_needs_a_qualified_sheet(self):
        """確認每份檔案至少有一張格式正確的工作表。"""
        good = self.source(1, workbook_bytes([EXCEL_HEADERS]))
        bad = self.source(2, workbook_bytes([("說明",)]))
        with self.assertRaisesRegex(DataError, "找不到"):
            read_excel([good, bad], lambda *args: None)
        with self.assertRaisesRegex(DataError, "無法讀取"):
            read_excel([good, self.source(3, b"invalid xlsx")], lambda *args: None)

    def test_integer_validation(self):
        """檢查等於整數的數字可用，其他輸入會被拒絕。"""
        for accepted in (13, 13.0, " 13.00 ", "1.3e1"):
            self.assertEqual(parse_integer(accepted, "仰角", 28), 13)
        for rejected in (True, None, 13.5, "13.000000000000001", "NaN", "", 29, 0):
            with self.subTest(rejected=rejected), self.assertRaises(DataError):
                parse_integer(rejected, "仰角", 28)


class ModelTests(unittest.TestCase):
    """驗證分組五折、模型設定及評分規則。"""

    def test_fold_membership_and_model_settings(self):
        """確認分組沒有交叉，並核對候選模型設定。"""
        features = np.asarray([[group, 2, 3, 4] for group in range(1, 16) for _ in range(3)])
        coordinates = np.asarray([(1, 2), (3, 4), (0, 0)] * 15)
        folds = make_folds(features, coordinates)
        self.assertEqual(len(folds), 5)
        seen = []
        for training, validation in folds:
            self.assertTrue(
                {tuple(row) for row in features[training]}.isdisjoint(
                    tuple(row) for row in features[validation]
                )
            )
            seen.extend(validation.tolist())
        self.assertEqual(sorted(seen), list(range(len(features))))
        self.assertAlmostEqual(score_axis([0, 1], [0, 1]), 2 / 6)
        for algorithm in ("RF", "ET"):
            settings = make_model(algorithm).get_params()
            self.assertEqual(settings["n_estimators"], 500)
            self.assertEqual(settings["random_state"], 42)
            self.assertEqual(settings["max_features"], 2)
            self.assertIsNone(settings["max_depth"])
            self.assertEqual(settings["bootstrap"], algorithm == "RF")
            self.assertIsNone(settings["max_samples"])
            self.assertIsNone(settings["n_jobs"])

    def test_evaluation_and_zero_denominator(self):
        """核對命中率、有效性矩陣及零分母處理。"""
        records = [
            dict(actual_x=0, actual_y=0, predicted_x=0, predicted_y=3),
            dict(actual_x=4, actual_y=2, predicted_x=2, predicted_y=3),
            dict(actual_x=4, actual_y=3, predicted_x=4, predicted_y=3),
            dict(actual_x=3, actual_y=3, predicted_x=3, predicted_y=0),
            dict(actual_x=0, actual_y=0, predicted_x=1, predicted_y=1),
        ]
        report = evaluate(records)
        self.assertEqual(report["hit_rates"], {"x": 3 / 5, "y": 1 / 5, "ball": 1 / 5})
        self.assertEqual(report["validity_matrices"], {
            "x": [[3, 0], [1, 1]], "y": [[2, 1], [2, 0]], "ball": [[2, 1], [1, 1]],
        })
        self.assertEqual(report["actual_valid_given_predicted_valid"], 2 / 3)
        self.assertEqual(report["actual_invalid_given_predicted_invalid"], 1 / 2)
        self.assertEqual(report["distance_balls"], 2)
        self.assertAlmostEqual(report["mean_distance"], 5 ** 0.5 / 2)
        missing = evaluate(records[:1])
        self.assertIsNone(missing["actual_valid_given_predicted_valid"])
        self.assertIsNone(missing["mean_distance"])

    def test_comparison_cases_keep_mixed_zero(self):
        """確認單球預測保留一軸為零的原座標。"""
        class FixedModel:
            """以固定類別模擬單軸模型的輸出。"""

            def __init__(self, value):
                self.value = value

            def predict(self, _features):
                """回傳此案例設定的單一預測值。"""
                return [self.value]

        bundle = PublishedRun({"run_id": "example"}, {"x": FixedModel(0), "y": FixedModel(3)})
        result = predict(bundle, (1, 2, 3, 4), (0, 0))
        self.assertEqual((result["x"], result["y"], result["valid"]), (0, 3, False))
        self.assertEqual(result["comparison"]["message"], "同為無效")
        self.assertIsNone(result["comparison"]["distance"])
        self.assertIsNone(compare_landing((4, 2), None))
        self.assertEqual(compare_landing((4, 2), (0, 0))["message"], "有效性不一致")
        self.assertTrue(compare_landing((4, 2), (4, 2))["same_cell"])
        self.assertAlmostEqual(compare_landing((4, 2), (2, 3))["distance"], 5 ** 0.5)


async def request(app, method, path, *, body=b"", content_type=b"application/json"):
    """直接送出 ASGI 請求，收集 HTTP 狀態與 JSON。"""
    messages = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app({
        "type": "http", "asgi": {"version": "3.0"}, "method": method,
        "path": path, "root_path": "", "query_string": b"",
        "headers": [(b"content-type", content_type)],
    }, receive, send)
    return messages[0]["status"], json.loads(messages[-1]["body"])


def multipart(files):
    """產生 API 測試所需的 multipart 上傳內容。"""
    boundary = b"TableTennisTestBoundary"
    parts = []
    for name, contents in files:
        parts.append(
            b"--" + boundary + b'\r\nContent-Disposition: form-data; name="files"; filename="' +
            name.encode("utf-8") + b'"\r\n\r\n' + contents + b"\r\n"
        )
    return {
        "body": b"".join(parts) + b"--" + boundary + b"--\r\n",
        "content_type": b"multipart/form-data; boundary=" + boundary,
    }


class APITests(unittest.IsolatedAsyncioTestCase):
    """驗證訓練上傳、報表載入與獨立評估。"""

    async def test_independent_evaluation(self):
        """以固定預測核對獨立測試指標及換版處理。"""
        class FixedAxis:
            """以四參數為索引，提供已知答案的單軸模型。"""

            def __init__(self, values):
                self.values = {
                    (angle, 2, 3, 4): value
                    for angle, value in enumerate(values, start=1)
                }

            def predict(self, features):
                """為輸入的每組四參數回傳對應預測。"""
                return [self.values[tuple(settings)] for settings in features]

        app = create_app()
        app.state.published = None
        endpoint = "/api/evaluation"
        self.assertEqual(await request(app, "POST", endpoint), (409, {"error": "請先完成訓練"}))

        report = {"run_id": "test-run", "winners": {"x": "RFx", "y": "ETy"}}
        app.state.published = PublishedRun(
            report,
            {"x": FixedAxis([0, 2, 3, 1]), "y": FixedAxis([3, 3, 0, 1])},
        )
        rows = [
            EXCEL_HEADERS,
            ("A", 1, 2, 3, 4, "未過網", "(2,3)"),
            ("B", 2, 2, 3, 4, "(4,2)", "(2,3)"),
            ("C", 3, 2, 3, 4, "(3,3)"),
            ("D", 4, 2, 3, 4, "飛出台面"),
        ]
        files = [("已知答案.xlsx", workbook_bytes(rows))]
        with (
            patch("backend.app.train", side_effect=AssertionError("獨立測試進入了訓練")),
            patch("backend.app.save_uploads", side_effect=AssertionError("獨立測試進入了訓練上傳")),
            patch("backend.app.publish", side_effect=AssertionError("獨立測試發布了模型")),
        ):
            status, result = await request(app, "POST", endpoint, **multipart(files))
            self.assertEqual(status, 200)
            self.assertEqual(result["run_id"], "test-run")
            self.assertEqual(result["missing_balls"], 14)
            self.assertEqual(result["models"]["x"]["algorithm"], "Random Forest")
            self.assertEqual(result["models"]["y"]["algorithm"], "Extra Trees")
            self.assertAlmostEqual(result["models"]["x"]["macro_f1"], 1 / 3)
            self.assertAlmostEqual(result["models"]["y"]["macro_f1"], 2 / 21)
            metrics = result["evaluation"]
            self.assertEqual(metrics["test_balls"], 6)
            self.assertEqual(metrics["hit_rates"], {"x": 3 / 6, "y": 2 / 6, "ball": 1 / 6})
            self.assertEqual(metrics["validity_matrices"], {
                "x": [[3, 1], [1, 1]], "y": [[3, 1], [2, 0]], "ball": [[2, 2], [1, 1]],
            })
            self.assertAlmostEqual(metrics["actual_valid_given_predicted_valid"], 2 / 3)
            self.assertAlmostEqual(metrics["actual_invalid_given_predicted_invalid"], 1 / 3)
            self.assertEqual(metrics["distance_balls"], 2)
            self.assertAlmostEqual(metrics["mean_distance"], 5 ** 0.5 / 2)
            self.assertEqual(await request(app, "GET", "/api/reports/latest"), (200, report))

            single_invalid = [("無有效預測.xlsx", workbook_bytes([
                EXCEL_HEADERS, ("A", 1, 2, 3, 4, "未過網"),
            ]))]
            status, result = await request(app, "POST", endpoint, **multipart(single_invalid))
            self.assertEqual(status, 200)
            self.assertIsNone(result["evaluation"]["actual_valid_given_predicted_valid"])
            self.assertIsNone(result["evaluation"]["mean_distance"])
            self.assertEqual(result["evaluation"]["distance_balls"], 0)

        self.assertEqual((await request(app, "POST", endpoint, **multipart(files * 2)))[0], 400)
        empty = [("全缺測.xlsx", workbook_bytes([EXCEL_HEADERS, ("A", 1, 2, 3, 4)]))]
        status, error = await request(app, "POST", endpoint, **multipart(empty))
        self.assertEqual(status, 422)
        self.assertIn("沒有可評分的球", error["error"])

        def evaluate_during_publication(published, _source):
            app.state.published = PublishedRun({"run_id": "new-run"}, {})
            return {"run_id": published.run_id}

        with patch("backend.app.evaluate_test_workbook", side_effect=evaluate_during_publication):
            self.assertEqual(
                await request(app, "POST", endpoint, **multipart(files)),
                (409, {"error": "正式模型已更新，請重新執行獨立測試。"}),
            )

    async def test_uploaded_filenames_survive_restart(self):
        """既有批次、新上傳和重啟後都能回傳已保存的原檔名。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = str(uuid4())
            status_file = root / "artifacts" / "training_status.json"
            status_file.parent.mkdir()
            save_uploads(root, run_id, [("舊資料.xlsx", io.BytesIO(b"old data"))])
            write_json(status_file, {
                "run_id": run_id, "status": "completed", "phase": "completed", "message": "訓練完成",
            })

            endpoint = "/api/training/status"
            app = create_app(root)
            async with app.router.lifespan_context(app):
                status, job = await request(app, "GET", endpoint)
                self.assertEqual((status, job["uploaded_files"]), (200, ["舊資料.xlsx"]))

                new_files = [("新資料.xlsx", b"invalid Excel"), ("新資料.xlsx", b"another file")]
                status, job = await request(app, "POST", "/api/training", **multipart(new_files))
                expected = [filename for filename, _ in new_files]
                self.assertEqual((status, job["uploaded_files"]), (202, expected))
                await app.state.task
                _, job = await request(app, "GET", endpoint)
                self.assertEqual((job["status"], job["uploaded_files"]), ("failed", expected))

                status, _ = await request(app, "POST", "/api/training", **multipart([]))
                self.assertEqual(status, 400)

            restarted = create_app(root)
            async with restarted.router.lifespan_context(restarted):
                _, job = await request(restarted, "GET", endpoint)
                self.assertEqual(job["uploaded_files"], expected)

    async def test_upload_train_restart_and_preserve_previous_success(self):
        """驗證多檔訓練、重啟載入與失敗時保留正式成果。"""
        rows = [EXCEL_HEADERS] + [
            ("重複", n, 2, 3, 4, "(1,1)", "(2,2)", "飛出台面", "(1,1)", None)
            for n in range(1, 16)
        ]
        files = [("任意檔名.xlsx", workbook_bytes(rows))] * 4
        input_values = b'{"angle":13.0,"position":2,"spin":3,"strength":4}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = create_app(root)
            async with app.router.lifespan_context(app):
                self.assertEqual(
                    await request(app, "POST", "/api/predict"),
                    (409, {"error": "請先完成訓練"}),
                )
                self.assertEqual(
                    (await request(app, "POST", "/api/training", **multipart([])))[0], 400,
                )
                with patch("backend.training.TREE_COUNT", 4):
                    status, single_task = await request(
                        app, "POST", "/api/training", **multipart(files[:1]),
                    )
                    self.assertEqual(status, 202)
                    await app.state.task
                    status, single_report = await request(app, "GET", "/api/reports/latest")
                    self.assertEqual(
                        (status, single_report["run_id"]), (200, single_task["run_id"]),
                    )
                    self.assertEqual(single_report["data"]["training_balls"], 60)
                    self.assertEqual(len(single_report["data"]["sources"]), 1)

                    status, task = await request(app, "POST", "/api/training", **multipart(files))
                    self.assertEqual(status, 202)
                    self.assertEqual(
                        await request(app, "POST", "/api/training", **multipart(files)),
                        (409, {"error": "已有訓練進行中"}),
                    )
                    await app.state.task
                run_id = task["run_id"]
                status, task = await request(app, "GET", "/api/training/status")
                self.assertEqual((status, task["status"]), (200, "completed"))
                status, report = await request(app, "GET", "/api/reports/latest")
                self.assertEqual((status, report["run_id"]), (200, run_id))
                self.assertEqual(report["data"]["training_balls"], 240)
                self.assertEqual(len(report["data"]["sources"]), 4)
                with (root / "artifacts" / run_id / "fold_predictions.csv").open(
                    encoding="utf-8-sig", newline=""
                ) as stream:
                    records = list(csv.DictReader(stream))
                self.assertEqual(len({row["ball_id"] for row in records}), 240)
                self.assertEqual({row["fold"] for row in records}, set("ABCDE"))
                for row in records:
                    self.assertEqual(row["predicted_x"], row[report["winners"]["x"]])
                    self.assertEqual(row["predicted_y"], row[report["winners"]["y"]])
                self.assertEqual(
                    (await request(app, "POST", "/api/predict", body=input_values))[1]["run_id"],
                    run_id,
                )
                status, error = await request(
                    app, "POST", "/api/predict",
                    body=input_values.replace(b"13.0", b"13.000000000000001"),
                )
                self.assertEqual((status, error["field"]), (422, "仰角角度"))
                broken = [*files[:2], ("壞檔.xlsx", b"bad data")]
                self.assertEqual(
                    (await request(app, "POST", "/api/training", **multipart(broken)))[0], 202,
                )
                await app.state.task
                status, failed = await request(app, "GET", "/api/training/status")
                self.assertEqual(
                    (status, failed["status"], failed["file"]), (200, "failed", "壞檔.xlsx"),
                )
                self.assertEqual(failed["latest_success_run_id"], run_id)

                def reject_publication(path, payload):
                    if path.name == "latest_success.json":
                        raise OSError("simulated publication failure")
                    write_json(path, payload)

                with (
                    patch("backend.training.TREE_COUNT", 4),
                    patch("backend.storage.write_json", side_effect=reject_publication),
                    self.assertLogs("backend.app", level="ERROR"),
                ):
                    self.assertEqual(
                        (await request(app, "POST", "/api/training", **multipart(files)))[0],
                        202,
                    )
                    await app.state.task
                _, failed = await request(app, "GET", "/api/training/status")
                self.assertEqual((failed["status"], failed["phase"]), ("failed", "publishing"))
                self.assertEqual(failed["latest_success_run_id"], run_id)

            self.assertEqual(
                read_json(root / "artifacts" / "latest_success.json"), {"run_id": run_id},
            )
            restarted = create_app(root)
            async with restarted.router.lifespan_context(restarted):
                self.assertEqual(
                    await request(restarted, "GET", "/api/reports/latest"), (200, report),
                )
                status, result = await request(restarted, "POST", "/api/predict", body=input_values)
                self.assertEqual(
                    (status, result["run_id"], result["comparison"]), (200, run_id, None),
                )
