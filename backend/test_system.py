import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook

from .app import compare_landing, create_app, predict
from .excel_loader import EXCEL_HEADERS, DataError, ExcelSource, parse_integer, read_excel
from .storage import PublishedRun, read_json, write_json
from .training import evaluate, make_folds, make_model, score_axis


def workbook_bytes(rows, extra_sheets=None):
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
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def source(self, number, contents):
        path = self.directory / f"{number}.xlsx"
        path.write_bytes(contents)
        return ExcelSource(number, "同名.xlsx", path)

    def test_expansion_keeps_source_and_repeats(self):
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
            [counts[name] for name in ("total_balls", "valid_balls", "invalid_balls", "missing_balls")],
            [40, 12, 6, 22],
        )
        self.assertEqual(counts["sources"][0]["sheets"], ["實測", "補測"])

    def test_invalid_parameter_even_if_all_landings_are_missing(self):
        data = workbook_bytes([EXCEL_HEADERS, ("A", 13.5, 2, 3, 4)])
        with self.assertRaises(DataError) as failure:
            read_excel([self.source(1, data)], lambda *args: None)
        self.assertEqual(failure.exception.details, {
            "file": "同名.xlsx", "sheet": "實測", "row": 2, "field": "仰角角度", "raw_value": "13.5",
        })

    def test_every_file_needs_a_qualified_sheet(self):
        good = self.source(1, workbook_bytes([EXCEL_HEADERS]))
        bad = self.source(2, workbook_bytes([("說明",)]))
        with self.assertRaisesRegex(DataError, "找不到"):
            read_excel([good, bad], lambda *args: None)
        with self.assertRaisesRegex(DataError, "無法讀取"):
            read_excel([good, self.source(3, b"invalid xlsx")], lambda *args: None)

    def test_integer_validation(self):
        for accepted in (13, 13.0, " 13.00 ", "1.3e1"):
            self.assertEqual(parse_integer(accepted, "仰角", 28), 13)
        for rejected in (True, None, 13.5, "13.000000000000001", "NaN", "", 29, 0):
            with self.subTest(rejected=rejected), self.assertRaises(DataError):
                parse_integer(rejected, "仰角", 28)


class ModelTests(unittest.TestCase):
    def test_fold_membership_and_model_settings(self):
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
        class FixedModel:
            def __init__(self, value):
                self.value = value

            def predict(self, features):
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
    async def test_upload_train_restart_and_preserve_previous_success(self):
        rows = [EXCEL_HEADERS] + [
            ("重複", n, 2, 3, 4, "(1,1)", "(2,2)", "飛出台面", "(1,1)", None)
            for n in range(1, 16)
        ]
        files = [("任意檔名.xlsx", workbook_bytes(rows))] * 3
        input_values = b'{"angle":13.0,"position":2,"spin":3,"strength":4}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = create_app(root)
            async with app.router.lifespan_context(app):
                self.assertEqual(await request(app, "POST", "/api/predict"), (409, {"error": "請先完成訓練"}))
                self.assertEqual((await request(app, "POST", "/api/training", **multipart(files[:2])))[0], 400)
                with patch("backend.training.TREE_COUNT", 4):
                    status, task = await request(app, "POST", "/api/training", **multipart(files))
                    self.assertEqual(status, 202)
                    self.assertEqual(await request(app, "POST", "/api/training", **multipart(files)),
                                     (409, {"error": "已有訓練進行中"}))
                    await app.state.task
                run_id = task["run_id"]
                status, task = await request(app, "GET", "/api/training/status")
                self.assertEqual((status, task["status"]), (200, "completed"))
                status, report = await request(app, "GET", "/api/reports/latest")
                self.assertEqual((status, report["run_id"]), (200, run_id))
                self.assertEqual(report["data"]["training_balls"], 180)
                with (root / "artifacts" / run_id / "fold_predictions.csv").open(
                    encoding="utf-8-sig", newline=""
                ) as stream:
                    records = list(csv.DictReader(stream))
                self.assertEqual(len({row["ball_id"] for row in records}), 180)
                self.assertEqual({row["fold"] for row in records}, set("ABCDE"))
                for row in records:
                    self.assertEqual(row["predicted_x"], row[report["winners"]["x"]])
                    self.assertEqual(row["predicted_y"], row[report["winners"]["y"]])
                self.assertEqual(
                    (await request(app, "POST", "/api/predict", body=input_values))[1]["run_id"], run_id,
                )
                status, error = await request(
                    app, "POST", "/api/predict", body=input_values.replace(b"13.0", b"13.000000000000001")
                )
                self.assertEqual((status, error["field"]), (422, "仰角角度"))
                broken = [*files[:2], ("壞檔.xlsx", b"bad data")]
                self.assertEqual((await request(app, "POST", "/api/training", **multipart(broken)))[0], 202)
                await app.state.task
                status, failed = await request(app, "GET", "/api/training/status")
                self.assertEqual((status, failed["status"], failed["file"]), (200, "failed", "壞檔.xlsx"))
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
                    self.assertEqual((await request(app, "POST", "/api/training", **multipart(files)))[0], 202)
                    await app.state.task
                _, failed = await request(app, "GET", "/api/training/status")
                self.assertEqual((failed["status"], failed["phase"]), ("failed", "publishing"))
                self.assertEqual(failed["latest_success_run_id"], run_id)

            self.assertEqual(read_json(root / "artifacts" / "latest_success.json"), {"run_id": run_id})
            restarted = create_app(root)
            async with restarted.router.lifespan_context(restarted):
                self.assertEqual(await request(restarted, "GET", "/api/reports/latest"), (200, report))
                status, result = await request(restarted, "POST", "/api/predict", body=input_values)
                self.assertEqual((status, result["run_id"], result["comparison"]), (200, run_id, None))
