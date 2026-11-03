# ============================================================
# app.py
# Starlette 後端：提供網頁與 API
#
# 啟動方式（在 VS Code 終端機輸入）：
#     uvicorn app:app --host 127.0.0.1 --port 8000
# 然後用瀏覽器開啟 http://127.0.0.1:8000
#
# 因為「訓練」和「完整掃描 147 萬組」都要跑很久，
# 這些工作會放到「背景執行緒」去跑，網頁每隔 1 秒詢問進度。
# ============================================================

import json
import math
import tempfile
from pathlib import Path
import threading
from contextlib import asynccontextmanager
import time
import traceback
import uuid

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

from core.config import DATA_DIR, OUTPUT_DIR, FRONTEND_DIR, ROUTE_NAMES_FOR_WEB
from core.sampling import generate_from_training_data
from core import predictor
from core.reverse_search import (
    search_best_parameters,
    normalize_target_labels,
    normalize_ball_conditions,
    validate_spin_direction_value,
)
from core.multi_route import recommend_multi_route
from train_model import train_and_save


# ============================================================
# 一、背景工作管理
# ============================================================

# 所有工作都記在這個字典：工作編號 → 工作狀態
class BusyError(Exception):
    pass


jobs = {}

# 鎖：避免兩個執行緒同時改字典
jobs_lock = threading.Lock()


def find_running_job():
    """找找看有沒有正在跑的工作。有就回傳它，沒有回傳 None。"""
    with jobs_lock:
        for job_id, job in jobs.items():
            if job["status"] == "running":
                return job_id, job
    return None


def create_job(kind):
    """建立一個新工作，回傳工作編號。"""
    job_id = uuid.uuid4().hex[:8]    # 隨機 8 碼編號
    with jobs_lock:
        if any(j["status"] == "running" for j in jobs.values()):
            raise BusyError("目前已有工作正在執行，請等它完成後再試。")
        jobs[job_id] = {
            "kind": kind,            # 工作種類：train / reverse / multi
            "status": "running",     # running / done / error
            "progress": 0,           # 0~100
            "message": "開始執行...",
            "log": [],               # 訓練時的文字紀錄
            "result": None,          # 完成後的結果
            "error": None,           # 失敗時的錯誤訊息
            "started_at": time.time(),
            "finished_at": None,
        }
    return job_id


def update_job(job_id, **changes):
    """更新工作狀態。"""
    with jobs_lock:
        jobs[job_id].update(changes)


def start_background(job_id, work_function):
    """
    在背景執行緒執行 work_function。
    work_function 回傳的東西會存成 job 的 result。
    """

    def runner():
        try:
            result = work_function()
            update_job(job_id, status="done", progress=100,
                       message="完成", result=result,
                       finished_at=time.time())
        except Exception as error:
            # 把完整錯誤印在終端機，方便你除錯
            traceback.print_exc()
            update_job(job_id, status="error", error=str(error),
                       message="失敗", finished_at=time.time())

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()


def dataframe_to_records(df):
    """
    把 DataFrame 轉成網頁可以讀的 list。
    用 pandas 內建的 to_json，可以處理 numpy 數字與空值。
    """
    json_text = df.to_json(orient="records", force_ascii=False)
    return json.loads(json_text)


def error_response(message, status_code=400):
    """回傳錯誤訊息給網頁。"""
    return JSONResponse({"ok": False, "error": message}, status_code=status_code)


def busy_response():
    """已經有工作在跑時的回應。"""
    return error_response("目前已有工作正在執行，請等它完成後再試。", 409)


# ============================================================
# 二、網頁與模型狀態
# ============================================================

async def homepage(request):
    """首頁：回傳 index.html。"""
    return FileResponse(FRONTEND_DIR / "index.html")


async def model_info(request):
    """查詢目前載入的模型狀態。"""
    info = predictor.get_model_info()
    info["route_names"] = ROUTE_NAMES_FOR_WEB
    active = find_running_job()
    info["active_job"] = {"id": active[0], "kind": active[1]["kind"]} if active else None
    return JSONResponse({"ok": True, "info": info})


async def job_status(request):
    """查詢背景工作的進度與結果。"""
    job_id = request.path_params["job_id"]

    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return error_response("找不到這個工作編號", 404)
        # 複製一份，避免讀取時被改到
        job_copy = dict(job)
        job_copy["log"] = list(job["log"])

    # 已完成的工作用「完成時間」計算耗時；還在跑的用現在時間
    end_time = job_copy["finished_at"] or time.time()
    job_copy["elapsed_seconds"] = round(end_time - job_copy["started_at"], 1)
    return JSONResponse({"ok": True, "job": job_copy})


# ============================================================
# 三、上傳資料並訓練
# ============================================================

async def upload_and_train(request):
    """
    接收上傳的 Excel，存到 data/，然後在背景重新訓練。
    表單欄位：
        files        一個或多個 Excel
        replace_old  "true" 代表先清掉 data/ 裡的舊 Excel
    """

    if find_running_job() is not None:
        return busy_response()

    form = await request.form()
    uploaded_files = form.getlist("files")
    replace_old = str(form.get("replace_old", "false")).lower() == "true"

    # 檢查有沒有真的上傳檔案
    real_uploads = [f for f in uploaded_files if getattr(f, "filename", "")]

    if len(real_uploads) == 0 and replace_old:
        return error_response("選了「只用這次上傳的檔案」，但沒有上傳任何檔案。")

    payloads = []
    for upload in real_uploads:
        name = upload.filename.replace("\\", "/").split("/")[-1]
        if Path(name).suffix.lower() not in {".xlsx", ".xls"} or name.startswith("~$"):
            raise ValueError("訓練僅接受 .xlsx 或 .xls 檔案")
        payloads.append((name, await upload.read()))
    existing = [p for p in DATA_DIR.iterdir()
                if p.suffix.lower() in {".xlsx", ".xls"} and not p.name.startswith("~$")]
    if not payloads and not existing:
        raise ValueError("請先上傳 Excel 資料")
    job_id = create_job("train")
    try:
        # 先驗證新資料；無效上傳不會清掉舊資料。
        from core.data_loader import prepare_training_data
        if payloads:
            with tempfile.TemporaryDirectory(prefix="fieldstudy-check-") as folder:
                paths = []
                for name, content in payloads:
                    path = Path(folder) / name
                    path.write_bytes(content)
                    paths.append(path)
                prepare_training_data(paths)
        if replace_old:
            for old in existing:
                old.unlink()
        saved_names = []
        for name, content in payloads:
            (DATA_DIR / name).write_bytes(content)
            saved_names.append(name)
        excel_paths = [p for p in sorted(DATA_DIR.iterdir())
                       if p.suffix.lower() in {".xlsx", ".xls"} and not p.name.startswith("~$")]
    except Exception as error:
        update_job(job_id, status="error", error=str(error), finished_at=time.time())
        raise

    # 訓練時會不斷呼叫這個函數回報進度
    def progress_callback(message, percent):
        with jobs_lock:
            jobs[job_id]["log"].append(message)
            jobs[job_id]["message"] = message.strip().split("\n")[-1]
            if percent is not None:
                jobs[job_id]["progress"] = percent

    def work():
        summary = train_and_save(excel_paths, progress_callback=progress_callback)
        # 訓練完成後，立刻載入新模型給 API 使用
        predictor.load_model_from_disk()
        from core.multi_route import reset_history
        reset_history()
        summary["used_files"] = [p.name for p in excel_paths]
        return summary

    start_background(job_id, work)

    return JSONResponse({
        "ok": True,
        "job_id": job_id,
        "saved_files": saved_names,
        "training_files": [p.name for p in excel_paths],
    })


# ============================================================
# 四、模組一：四參數 → 預測落點（很快，直接回傳）
# ============================================================

async def api_predict(request):
    try:
        body = await read_body(request)

        prediction, probability_df = predictor.predict_landing(
            elevation=body.get("elevation"),
            horizontal=body.get("horizontal"),
            spin_angle=body.get("spin"),
            power=body.get("power"),
            top_n=10,
        )

        return JSONResponse({
            "ok": True,
            "prediction": str(prediction),
            "top": dataframe_to_records(probability_df),
        })

    except ValueError as error:
        return error_response(str(error))


# ============================================================
# 五、模組二：目標落點反推四參數（要掃描 147 萬組，放背景）
# ============================================================

def get_target_list(body):
    """
    從網頁傳來的資料取得目標落點 list。
    網頁傳 targets: ["(4,3)"]，只能指定單一格。
    """
    targets = body.get("targets")
    if not isinstance(targets, list) or len(targets) == 0:
        raise ValueError("請至少選一個目標落點")
    return targets


def get_integer(body, key, default, minimum, maximum):
    value = body.get(key, default)
    try:
        if isinstance(value, bool):
            raise ValueError()
        number = float(value)
        if not math.isfinite(number) or not number.is_integer() or not minimum <= number <= maximum:
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError(f"{key} 必須是 {minimum}~{maximum} 的整數")
    return int(number)


async def read_body(request):
    try:
        body = await request.json()
    except (ValueError, UnicodeError):
        raise ValueError("請提供有效的 JSON")
    if not isinstance(body, dict):
        raise ValueError("JSON 必須是物件")
    return body


def scan_progress(job_id):
    def callback(done, total):
        update_job(job_id, progress=int(done * 100 / total) if total else 100,
                   message=f"已掃描 {done:,} / {total:,} 組")
    return callback


async def api_reverse_search(request):
    predictor.get_package()
    body = await read_body(request)
    targets = normalize_target_labels(get_target_list(body))
    top_n = get_integer(body, "top_n", 20, 1, 100)
    job_id = create_job("reverse")

    def work():
        # 使用者即使直接呼叫 API 也不能降低門檻或附加球路條件。
        result, status, searched = search_best_parameters(
            target_landing=targets, top_n=top_n,
            progress_callback=scan_progress(job_id))
        return {"status_text": status, "searched_count": int(searched),
                "rows": dataframe_to_records(result)}

    start_background(job_id, work)
    return JSONResponse({"ok": True, "job_id": job_id})


async def api_multi_route(request):
    predictor.get_package()
    body = await read_body(request)
    targets = normalize_target_labels(get_target_list(body))
    routes = body.get("routes")
    if not isinstance(routes, list) or not routes or any(r not in ROUTE_NAMES_FOR_WEB for r in routes):
        raise ValueError("請至少選擇一種有效的球路選項")
    normalize_ball_conditions(routes)
    n_solutions = get_integer(body, "n_solutions", 5, 1, 20)
    job_id = create_job("multi")

    def work():
        results, statuses, searched = recommend_multi_route(
            target_landing=targets, ball_conditions=routes, n_solutions=n_solutions,
            progress_callback=scan_progress(job_id))
        return {"searched_count": int(searched), "routes": [
            {"route": name, "status_text": statuses[name], "rows": dataframe_to_records(df)}
            for name, df in results.items()]}

    start_background(job_id, work)
    return JSONResponse({"ok": True, "job_id": job_id})


async def api_sampling(request):
    body = await read_body(request)
    generate = body.get("generate", True)
    if not isinstance(generate, bool):
        raise ValueError("generate 必須是 true 或 false")
    if not generate:
        return JSONResponse({"ok": True, "skipped": True, "message": "已選擇暫不產生補測資料。"})
    if any(key in body for key in ["files", "observations", "complete_data"]):
        raise ValueError("補測只使用目前模型的訓練資料，不接受額外上傳或指定其他資料。")
    if find_running_job() is not None:
        return busy_response()
    observations = predictor.get_training_observations()
    tested_parameters = set(predictor.get_tested_parameters())
    source_info = predictor.get_sampling_info()
    job_id = create_job("sampling")

    def work():
        result, strategy, summary = generate_from_training_data(
            observations, tested_parameters=tested_parameters,
            progress_callback=lambda message, percent: update_job(job_id, message=message, progress=percent))
        summary["source_files"] = source_info.get("source_files", [])
        csv_path = OUTPUT_DIR / f"supplement_{job_id}.csv"
        xlsx_path = OUTPUT_DIR / f"supplement_{job_id}.xlsx"
        result.to_csv(csv_path, index=False, encoding="utf-8-sig")
        import pandas as pd
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            result.to_excel(writer, sheet_name="推薦補測300組", index=False)
            strategy.to_excel(writer, sheet_name="採樣策略", index=False)
            pd.DataFrame([summary]).to_excel(writer, sheet_name="摘要", index=False)
            ws = writer.sheets["推薦補測300組"]
            ws.freeze_panes = "F2"
            ws.auto_filter.ref = ws.dimensions
            for column in ws.columns:
                ws.column_dimensions[column[0].column_letter].width = 19
        return {"summary": summary, "rows": dataframe_to_records(result),
                "downloads": {"csv": f"/api/sampling/{job_id}/download/csv",
                              "xlsx": f"/api/sampling/{job_id}/download/xlsx"}}

    start_background(job_id, work)
    return JSONResponse({"ok": True, "job_id": job_id})


async def download_sampling(request):
    job_id, extension = request.path_params["job_id"], request.path_params["extension"]
    with jobs_lock:
        job = jobs.get(job_id)
        valid = job and job["kind"] == "sampling" and job["status"] == "done"
    if not valid or extension not in {"csv", "xlsx"}:
        return error_response("找不到已完成的補測檔案", 404)
    path = OUTPUT_DIR / f"supplement_{job_id}.{extension}"
    if not path.exists():
        return error_response("補測檔案不存在", 404)
    return FileResponse(path, filename=f"fieldstudy_supplement_300.{extension}")


# ============================================================
# 七、組裝網站
# ============================================================

@asynccontextmanager
async def lifespan(app):
    """
    伺服器「啟動時」會先執行 yield 之前的程式；
    「關閉時」會執行 yield 之後的程式（這裡不需要做事）。
    """
    loaded = predictor.load_model_from_disk()
    if loaded:
        print(" 已載入現有模型：", predictor.get_package()["best_model_name"])
    else:
        print("目前還沒有模型，請在網頁上傳資料並訓練。")
    yield


routes = [
    Route("/", homepage),
    Route("/api/model-info", model_info),
    Route("/api/job/{job_id}", job_status),
    Route("/api/upload-train", upload_and_train, methods=["POST"]),
    Route("/api/predict", api_predict, methods=["POST"]),
    Route("/api/reverse-search", api_reverse_search, methods=["POST"]),
    Route("/api/multi-route", api_multi_route, methods=["POST"]),
    Route("/api/sampling", api_sampling, methods=["POST"]),
    Route("/api/sampling/{job_id}/download/{extension}", download_sampling),
    Mount("/static", app=StaticFiles(directory=FRONTEND_DIR), name="static"),
]

async def validation_error(request, error):
    return error_response(str(error))


async def busy_error(request, error):
    return busy_response()


app = Starlette(routes=routes, lifespan=lifespan,
                exception_handlers={ValueError: validation_error, BusyError: busy_error})
