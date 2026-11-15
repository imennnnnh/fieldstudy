"""定義模型訓練、報表、獨立測試與單球預測的 HTTP 路由。"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from decimal import Decimal
from math import dist
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .evaluation import evaluate_test_workbook
from .excel_loader import PARAMETERS, DataError, ExcelSource, parse_integer, read_excel
from .storage import load_latest, load_uploaded_filenames, publish, read_json, save_uploads, write_json
from .training import TrainingError, is_valid, train


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
LOGGER = logging.getLogger(__name__)
ACTIVE_STATUSES = {"uploading", "running"}


def update_job(state, status, phase, message, **details):
    """更新目前訓練任務，並嘗試保存狀態。"""
    for field in ("counts", "uploaded_files"):
        if field in state.job:
            details.setdefault(field, state.job[field])
    state.job = {
        "run_id": state.job["run_id"],
        "status": status, "phase": phase, "message": message,
        **details,
    }
    try:
        write_json(state.artifacts / "training_status.json", state.job)
    except OSError:
        LOGGER.exception("Could not persist training status")


def create_app(data_dir=ROOT / "data"):
    """建立 Starlette 應用程式並設定正式模型的載入流程。"""
    @asynccontextmanager
    async def lifespan(application):
        state = application.state
        state.data_dir = Path(data_dir)
        state.artifacts = state.data_dir / "artifacts"
        state.artifacts.mkdir(parents=True, exist_ok=True)
        state.task = None
        state.published = None
        try:
            state.published = await run_in_threadpool(load_latest, state.artifacts)
        # 正式模型讀取出錯時，保留可啟動新訓練的服務狀態。
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.exception("Could not load the published model and report")

        state.job = {"run_id": None, "status": "idle", "phase": None, "message": ""}
        status_path = state.artifacts / "training_status.json"
        if status_path.is_file():
            try:
                saved = read_json(status_path)
                if (
                    not isinstance(saved, dict)
                    or not {"run_id", "status", "phase", "message"} <= saved.keys()
                ):
                    raise ValueError("Invalid training status")
                state.job = saved
            except (OSError, ValueError):
                LOGGER.exception("Could not load training status")
        uploaded_files = await run_in_threadpool(
            load_uploaded_filenames, state.data_dir, state.job["run_id"],
        ) or state.job.get("uploaded_files")
        if not uploaded_files and state.published:
            sources = state.published.report.get("data", {}).get("sources", [])
            uploaded_files = [source["filename"] for source in sources]
        state.job["uploaded_files"] = uploaded_files or []
        if state.job["status"] in ACTIVE_STATUSES:
            if state.published and state.published.run_id == state.job["run_id"]:
                update_job(state, "completed", "completed", "訓練完成")
            else:
                update_job(state, "failed", state.job["phase"], "訓練因服務重啟而中斷，請重新上傳並啟動。")
        try:
            yield
        finally:
            if state.task is not None:
                await state.task

    return Starlette(
        routes=[
            Route("/", homepage),
            Route("/api/training", start_training, methods=["POST"]),
            Route("/api/training/status", training_status),
            Route("/api/reports/latest", latest_report),
            Route("/api/evaluation", evaluate_model, methods=["POST"]),
            Route("/api/predict", predict_landing, methods=["POST"]),
            Mount("/static", app=StaticFiles(directory=FRONTEND), name="static"),
        ],
        lifespan=lifespan,
        exception_handlers={HTTPException: http_error},
    )


async def http_error(_request, error):
    """以 JSON 回傳 HTTP 錯誤，保留狀態碼與標頭。"""
    return JSONResponse(
        {"error": error.detail}, status_code=error.status_code, headers=error.headers,
    )


async def homepage(_request):
    """提供桌球落點預測系統的前端頁面。"""
    return FileResponse(FRONTEND / "index.html")


def job_payload(state):
    """回傳訓練狀態與最近成功發布的模型批次。"""
    return {
        **state.job,
        "latest_success_run_id": state.published.run_id if state.published else None,
    }


async def start_training(request):
    """接收當次完整的訓練檔案清單，並安排背景訓練。"""
    state = request.app.state
    if state.job["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "已有訓練進行中")

    run_id = str(uuid4())
    state.job = {"run_id": run_id, "uploaded_files": state.job["uploaded_files"]}
    update_job(state, "uploading", "uploading", "接收訓練用 Excel")
    try:
        async with request.form(max_files=float("inf"), max_fields=0) as form:
            files = form.getlist("files")
            if not files or any(
                not isinstance(file, UploadFile) or not file.filename for file in files
            ):
                raise HTTPException(400, "請至少上傳一份訓練用 .xlsx 檔案。")
            uploads = [(PureWindowsPath(file.filename).name, file.file) for file in files]
            for filename, _ in uploads:
                if Path(filename).suffix.lower() != ".xlsx":
                    raise HTTPException(400, f"「{filename}」須為 .xlsx 檔案。")
            sources = await run_in_threadpool(save_uploads, state.data_dir, run_id, uploads)
    except HTTPException as error:
        update_job(state, "failed", "uploading", error.detail)
        raise
    except asyncio.CancelledError:
        update_job(state, "failed", "uploading", "上傳已中斷。")
        raise
    except Exception as error:
        LOGGER.exception("Could not receive Excel uploads")
        message = "上傳失敗，請檢查檔案及資料資料夾是否可寫入。"
        update_job(state, "failed", "uploading", message)
        raise HTTPException(500, message) from error

    update_job(
        state, "running", "reading", "準備讀取 Excel",
        uploaded_files=[source.filename for source in sources],
    )
    state.task = asyncio.create_task(train_in_background(state, run_id, sources))
    return JSONResponse(job_payload(state), status_code=202)


async def train_in_background(state, run_id, sources):
    """讀取資料、訓練模型並在成果完整時發布。"""
    loop = asyncio.get_running_loop()

    def progress(phase, message, **details):
        def update():
            update_job(state, "running", phase, message, **details)
        loop.call_soon_threadsafe(update)

    def run():
        balls, counts = read_excel(sources, progress)
        progress("reading", "Excel 已展開為逐球資料", counts=counts)
        models, records, report = train(balls, counts, run_id, progress)
        progress("publishing", "保存模型、逐折驗證紀錄與報表")
        return publish(state.artifacts, run_id, models, records, report)

    try:
        published = await run_in_threadpool(run)
    except (DataError, TrainingError) as error:
        update_job(state, "failed", state.job["phase"], str(error), **getattr(error, "details", {}))
    # 未預期的背景錯誤也必須標成訓練失敗，讓使用者能重試。
    except Exception:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Training failed for run %s", run_id)
        phase = state.job["phase"]
        message = "成果保存失敗，請檢查資料資料夾是否可寫入及剩餘空間。"
        if phase != "publishing":
            message = "訓練失敗，請查看後端終端機訊息。"
        update_job(state, "failed", phase, message)
    else:
        state.published = published
        update_job(state, "completed", "completed", "訓練完成", counts=published.report["data"])


async def training_status(request):
    """提供前端輪詢所需的訓練狀態。"""
    return JSONResponse(job_payload(request.app.state), headers={"Cache-Control": "no-store"})


async def latest_report(request):
    """回傳目前正式模型批次的五折報表。"""
    published = request.app.state.published
    if published is None:
        raise HTTPException(404, "請先完成訓練")
    return JSONResponse(published.report, headers={"Cache-Control": "no-store"})


async def evaluate_model(request):
    """使用目前已發布的模型評估一份測試 Excel。"""
    state = request.app.state
    published = state.published
    if published is None:
        raise HTTPException(409, "請先完成訓練")

    try:
        async with request.form(max_files=1, max_fields=0) as form:
            files = form.getlist("files")
            if len(files) != 1 or not isinstance(files[0], UploadFile) or not files[0].filename:
                raise HTTPException(400, "請上傳一份測試用 .xlsx 檔案。")
            file = files[0]
            filename = PureWindowsPath(file.filename).name
            if Path(filename).suffix.lower() != ".xlsx":
                raise HTTPException(400, f"「{filename}」須為 .xlsx 檔案。")
            source = ExcelSource(1, filename, file.file)
            result = await run_in_threadpool(evaluate_test_workbook, published, source)
    except DataError as error:
        return JSONResponse({"error": str(error), **error.details}, status_code=422)
    except HTTPException as error:
        if str(error.detail).startswith("Too many files."):
            raise HTTPException(400, "每次只能上傳一份測試用 .xlsx 檔案。") from error
        raise
    except Exception as error:
        LOGGER.exception("Independent evaluation failed for run %s", published.run_id)
        raise HTTPException(500, "評估失敗，請查看後端終端機訊息。") from error

    if state.published is not published:
        raise HTTPException(409, "正式模型已更新，請重新執行獨立測試。")
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


def parse_measured(measured):
    """將單球表單的實測狀態轉為座標或空值。"""
    if measured is None:
        return None
    if isinstance(measured, dict):
        status = measured.get("status")
        if status == "unfilled":
            return None
        if status == "invalid":
            return (0, 0)
        if status == "valid":
            return tuple(
                parse_integer(measured.get(axis), f"實測 {axis.upper()}", 5) for axis in "xy"
            )
    raise DataError("實測狀態須為未填、有效球或無效球。", field="measured")


def compare_landing(predicted, measured):
    """比較單球預測與實測，兩者皆有效時才計算距離。"""
    if measured is None:
        return None
    result = {"measured": measured, "same_cell": None, "distance": None}
    predicted_valid, measured_valid = is_valid(predicted), is_valid(measured)
    if predicted_valid and measured_valid:
        return {**result, "status": "both_valid", "same_cell": predicted == measured,
                "distance": dist(predicted, measured)}
    if not predicted_valid and not measured_valid:
        return {**result, "status": "both_invalid", "message": "同為無效"}
    return {**result, "status": "validity_mismatch", "message": "有效性不一致"}


def predict(published, parameters, measured):
    """以正式 X、Y 模型產生單球預測。"""
    landing = tuple(int(published.models[axis].predict([parameters])[0]) for axis in "xy")
    return {
        "run_id": published.run_id, "x": landing[0], "y": landing[1],
        "valid": is_valid(landing), "comparison": compare_landing(landing, measured),
    }


async def predict_landing(request):
    """驗證單球請求並回傳預測結果。"""
    published = request.app.state.published
    if published is None:
        raise HTTPException(409, "請先完成訓練")
    try:
        payload = json.loads(await request.body(), parse_float=Decimal)
    except (ValueError, ArithmeticError) as error:
        raise HTTPException(400, "請提供有效的 JSON。") from error
    if not isinstance(payload, dict):
        raise HTTPException(400, "請提供 JSON 物件。")
    try:
        parameters = tuple(
            parse_integer(payload.get(key), label, maximum)
            for key, label, maximum in PARAMETERS
        )
        measured = parse_measured(payload.get("measured"))
    except DataError as error:
        return JSONResponse({"error": str(error), **error.details}, status_code=422)
    result = await run_in_threadpool(predict, published, parameters, measured)
    return JSONResponse(result)


app = create_app()
