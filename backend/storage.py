import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import UUID

import joblib
import sklearn

from .excel_loader import ExcelSource


@dataclass(frozen=True)
class PublishedRun:
    report: dict
    models: dict

    @property
    def run_id(self):
        return self.report["run_id"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    # 暫存檔與目標放在同一磁碟區，以一次取代發布完整 JSON。
    with NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_uploads(data_dir, run_id, files):
    directory = data_dir / "uploads" / run_id
    directory.mkdir(parents=True)
    sources = []
    for number, (filename, stream) in enumerate(files, start=1):
        path = directory / f"{number}.xlsx"
        with path.open("wb") as destination:
            shutil.copyfileobj(stream, destination)
        sources.append(ExcelSource(number, filename, path))
    write_json(directory / "sources.json", [
        {"file_id": source.file_id, "filename": source.filename, "stored_as": source.path.name}
        for source in sources
    ])
    return sources


def load_uploaded_filenames(data_dir, run_id):
    """讀取既有訓練批次的上傳原檔名。"""
    try:
        directory = data_dir / "uploads" / str(UUID(str(run_id)))
        manifest = read_json(directory / "sources.json")
    except (OSError, ValueError):
        return []
    if not isinstance(manifest, list):
        return []
    return [
        source["filename"] for source in manifest
        if isinstance(source, dict) and isinstance(source.get("filename"), str)
    ]


def load_bundle(directory, run_id):
    report = read_json(directory / "report.json")
    if report["run_id"] != run_id or report["scikit_learn_version"] != sklearn.__version__:
        raise ValueError("訓練批次或 scikit-learn 版本不符。")
    with (directory / "fold_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "ball_id" not in (reader.fieldnames or []) or sum(1 for _ in reader) != report["data"]["training_balls"]:
            raise ValueError("逐折驗證紀錄不完整。")
    models = {axis: joblib.load(directory / f"model_{axis}.joblib") for axis in "xy"}
    return PublishedRun(report, models)


def publish(artifacts, run_id, models, records, report):
    with TemporaryDirectory(dir=artifacts, prefix=".staging-") as name:
        staging = Path(name)
        bundle = staging / run_id
        bundle.mkdir()
        for axis, model in models.items():
            joblib.dump(model, bundle / f"model_{axis}.joblib", compress=3)
        with (bundle / "fold_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        write_json(bundle / "report.json", {**report, "scikit_learn_version": sklearn.__version__})
        published = load_bundle(bundle, run_id)
        bundle.rename(artifacts / run_id)
    write_json(artifacts / "latest_success.json", {"run_id": run_id})
    return published


def load_latest(artifacts):
    pointer = artifacts / "latest_success.json"
    if not pointer.is_file():
        return None
    run_id = str(UUID(read_json(pointer)["run_id"]))
    return load_bundle(artifacts / run_id, run_id)
