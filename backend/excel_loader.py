"""驗證實測 Excel 的欄位並展開逐球落點資料。"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import BinaryIO

from openpyxl import load_workbook


PARAMETERS = (
    ("angle", "仰角角度", 28),
    ("position", "左右位置", 9),
    ("spin", "旋球角度", 59),
    ("strength", "出球強度", 99),
)
PARAMETER_KEYS = tuple(key for key, _, _ in PARAMETERS)
EXCEL_HEADERS = (
    "測試編號",
    *(label for _, label, _ in PARAMETERS),
    *(f"發球{ball}_落點" for ball in range(1, 6)),
)
INVALID_LABELS = {"未過網", "球碰網後沒有進入有效區", "飛出台面", "碰桌後彈出"}
COORDINATE = re.compile(r"\(\s*([1-5])\s*,\s*([1-5])\s*\)")
NUMBER_TEXT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


class DataError(ValueError):
    """描述資料錯誤及可供定位的欄位資訊。"""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class ExcelSource:
    """表示一份實測 Excel 的識別碼、原檔名及內容來源。"""

    file_id: int
    filename: str
    path: Path | BinaryIO


@dataclass(frozen=True)
class Ball:
    """保存一顆有實測落點的球及其原始位置。"""

    ball_id: str
    file_id: int
    filename: str
    sheet: str
    row: int
    test_id: object
    serve: int
    parameters: tuple[int, int, int, int]
    raw_landing: str
    landing: tuple[int, int]


def parse_integer(value, label, maximum):
    """將 Excel 或 API 的數值解析為指定範圍內的整數。"""
    try:
        if isinstance(value, str):
            stripped = value.strip()
            number = Decimal(stripped) if NUMBER_TEXT.fullmatch(stripped) else None
        elif type(value) in (int, float, Decimal):
            number = Decimal(str(value))
        else:
            number = None
    except InvalidOperation:
        number = None
    if (
        number is None
        or not number.is_finite()
        or not 1 <= number <= maximum
        or number != number.to_integral_value()
    ):
        raise DataError(f"{label}須為 1～{maximum} 的整數。", field=label)
    return int(number)


def is_blank(value):
    """判斷 Excel 儲存格是否為空值或純空白字串。"""
    return value is None or isinstance(value, str) and not value.strip()


def parse_landing(value):
    """將落點解析為座標、無效球座標或缺測。"""
    if not isinstance(value, str):
        return None
    label = value.strip()
    if label in INVALID_LABELS:
        return (0, 0)
    match = COORDINATE.fullmatch(label)
    return tuple(map(int, match.groups())) if match else None


def read_sheets(workbook, source, balls, counts):
    """處理表頭符合規範的工作表，累加逐球資料與計數。"""
    sheets = []
    for sheet in workbook.worksheets:
        if sheet.title in {"說明", "下拉選單"}:
            continue
        rows = sheet.values
        headers = [
            value.strip() if isinstance(value, str) else value
            for value in next(rows, ())
        ]
        if any(headers.count(header) != 1 for header in EXCEL_HEADERS):
            continue
        columns = [headers.index(header) for header in EXCEL_HEADERS]
        sheets.append(sheet.title)
        for row_number, row in enumerate(rows, start=2):
            test_id, *values = [row[i] if i < len(row) else None for i in columns]
            if all(is_blank(value) for value in values):
                continue
            settings = []
            for value, (_, label, maximum) in zip(values[:4], PARAMETERS, strict=True):
                try:
                    settings.append(parse_integer(value, label, maximum))
                except DataError as error:
                    raw = repr(value) if not is_blank(value) else "空白"
                    location = f"「{source.filename}」／「{sheet.title}」第 {row_number} 列"
                    raise DataError(
                        f"{location}，「{label}」原值為 {raw}。{error}",
                        file=source.filename, sheet=sheet.title, row=row_number,
                        field=label, raw_value=repr(value),
                    ) from error
            counts["total_balls"] += 5
            parameters = tuple(settings)
            for serve, value in enumerate(values[4:], start=1):
                landing = parse_landing(value)
                if landing is None:
                    counts["missing_balls"] += 1
                    continue
                counts["invalid_balls" if landing == (0, 0) else "valid_balls"] += 1
                balls.append(Ball(
                    f"{source.file_id}:{sheet.title}:{row_number}:{serve}",
                    source.file_id, source.filename, sheet.title, row_number,
                    test_id, serve, parameters, value, landing,
                ))
    return sheets


def read_excel(sources, progress):
    """合併來源的逐球紀錄並回傳實測球數統計。"""
    balls = []
    counts = dict(total_balls=0, valid_balls=0, invalid_balls=0, missing_balls=0)
    used_sources = []
    for source in sources:
        progress("reading", f"讀取並驗證「{source.filename}」")
        try:
            workbook = load_workbook(source.path, read_only=True, data_only=True)
            try:
                sheets = read_sheets(workbook, source, balls, counts)
            finally:
                workbook.close()
        except DataError:
            raise
        except Exception as error:
            raise DataError(
                f"無法讀取「{source.filename}」的 Excel 內容（{type(error).__name__}）。",
                file=source.filename,
            ) from error
        if not sheets:
            raise DataError(
                f"「{source.filename}」找不到包含 10 個必要欄位的工作表。",
                file=source.filename,
            )
        used_sources.append({
            "file_id": source.file_id,
            "filename": source.filename,
            "sheets": sheets,
        })
    return balls, {**counts, "training_balls": len(balls), "sources": used_sources}
