# ============================================================
# config.py
# 放「整個專案共用的設定」：資料夾路徑、參數範圍、29 類落點、球路條件表。
# 其他檔案都從這裡拿設定，這樣要改設定只要改這一個檔案。
# ============================================================

import re                      # 用來處理文字格式（正規表達式）
from pathlib import Path       # 用來處理資料夾路徑

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 一、資料夾與檔案路徑
# ------------------------------------------------------------

# __file__ 是這個檔案本身的位置（fieldstudy/core/config.py）
# .parent 是上一層資料夾，連用兩次就回到專案根目錄 fieldstudy/
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"            # 上傳的 Excel 放這裡
MODEL_DIR = BASE_DIR / "models"         # 訓練好的模型放這裡
OUTPUT_DIR = BASE_DIR / "outputs"       # 訓練報告 CSV 放這裡
FRONTEND_DIR = BASE_DIR / "frontend"    # index.html 放這裡

# 模型存檔的完整路徑
MODEL_PATH = MODEL_DIR / "table_tennis_ball_landing_model.pkl"

# 如果資料夾不存在就自動建立（exist_ok=True 代表已存在也不會報錯）
for folder in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 二、模型訓練設定
# ------------------------------------------------------------

N_ESTIMATORS = 500     # 決策樹數量：依需求維持 500，不要降低
N_SPLITS = 5           # 交叉驗證折數：5 折
RANDOM_STATE = 42      # 固定隨機種子，讓每次結果可以重現


# ------------------------------------------------------------
# 三、欄位名稱
# ------------------------------------------------------------

# 四個機器參數（模型的輸入 X）
FEATURE_COLUMNS = [
    "仰角角度",
    "左右位置",
    "旋球角度",     # Excel 欄位名稱；實際是球機旋球設定值 1~59，不是度數
    "出球強度",
]

# 每組參數會發 5 球，這是 5 個落點欄位
SHOT_COLUMNS = [
    "發球1_落點",
    "發球2_落點",
    "發球3_落點",
    "發球4_落點",
    "發球5_落點",
]

# 測試編號欄位
ID_COLUMN = "測試編號"

# Excel 工作表一定要有的欄位（少一個就不會被匯入）
IMPORT_REQUIRED_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + SHOT_COLUMNS


# ------------------------------------------------------------
# 四、四個參數的合法範圍（球機設備限制）
# ------------------------------------------------------------

PARAMETER_LIMITS = {
    "仰角角度": (1, 28),
    "左右位置": (1, 9),
    "旋球角度": (1, 59),   # 旋球設定值 1~59（範圍涵蓋 360°）
    "出球強度": (1, 99),
}


# ------------------------------------------------------------
# 五、29 類落點
# ------------------------------------------------------------

# 25 個有效落點：(1,1) (1,2) ... (5,5)
VALID_LABELS = [
    f"({x},{y})"
    for x in range(1, 6)
    for y in range(1, 6)
]

# 4 種無效結果
INVALID_LABELS = [
    "飛出台面",
    "未過網",
    "碰桌後彈出",
    "球碰網後沒有進入有效區",
]

# 全部 25 + 4 = 29 類
ALL_LABELS = VALID_LABELS + INVALID_LABELS


# ------------------------------------------------------------
# 六、球路條件表
#
# 格式： "條件名稱": [ (要看的欄位, 方向), ... ]
# 方向 "max"            = 這個參數越大越好
# 方向 "min"            = 這個參數越小越好
# 方向 "target_circular" = 越接近使用者指定的設定值越好（1 與 59 視為相鄰）
# ------------------------------------------------------------

# 固定推薦條件；API 不接受降低門檻。
MIN_TARGET_PROBABILITY = 1.0
MAX_INVALID_PROBABILITY = 0.0

# 所有名稱都代表機器設定偏好，並非實測球速或已校正的旋轉方向。
BALL_CONDITION_MAP = {
    "標準": [],
    "均衡方案": [(col, "mid") for col in FEATURE_COLUMNS],
    "多樣化方案": [],
    "高速球": [("出球強度", "max")],
    "中速球": [("出球強度", "mid")],
    "低速球": [("出球強度", "min")],
    "高仰角": [("仰角角度", "max")],
    "中仰角": [("仰角角度", "mid")],
    "低仰角": [("仰角角度", "min")],
    "左右設定值偏小": [("左右位置", "min")],
    "左右設定值居中": [("左右位置", "mid")],
    "左右設定值偏大": [("左右位置", "max")],
    "旋球設定值偏小": [("旋球角度", "min")],
    "旋球設定值居中": [("旋球角度", "mid")],
    "旋球設定值偏大": [("旋球角度", "max")],
    "高速＋高仰角": [("出球強度", "max"), ("仰角角度", "max")],
    "高速＋低仰角": [("出球強度", "max"), ("仰角角度", "min")],
    "低速＋高仰角": [("出球強度", "min"), ("仰角角度", "max")],
    "低速＋低仰角": [("出球強度", "min"), ("仰角角度", "min")],
}
ROUTE_NAMES_FOR_WEB = [name for name in BALL_CONDITION_MAP if name != "標準"]


# ------------------------------------------------------------
# 七、落點文字整理的小工具
# ------------------------------------------------------------

def clean_landing_point(value):
    """
    把 Excel 裡各種寫法的落點，整理成統一格式。
    例如： "（4，3）"、"4,3"、" (4,3) "  都會變成  "(4,3)"
    無法辨識的就回傳 NaN（空值）。
    """

    # 空值直接回傳空值
    if pd.isna(value):
        return np.nan

    # 轉成文字並去掉前後空白
    value = str(value).strip()

    # 全形符號換成半形符號
    value = value.replace("（", "(").replace("）", ")").replace("，", ",")

    # 移除所有空白
    value = re.sub(r"\s+", "", value)

    # 兩種名稱代表同一類；沿用現有模型的「碰桌後彈出」編碼。
    if value == "其他無效":
        value = "碰桌後彈出"

    # 如果是 4 種無效結果之一，直接回傳
    if value in INVALID_LABELS:
        return value

    # 座標格式：(4,3) 或 4,3
    match = re.fullmatch(r"\(([1-5]),([1-5])\)|([1-5]),([1-5])", value)

    if match:
        if match.group(1) is not None:
            x = match.group(1)
            y = match.group(2)
        else:
            x = match.group(3)
            y = match.group(4)
        return f"({x},{y})"

    # 都不是，無法辨識
    return np.nan


def parse_coordinate(label):
    """
    把 "(4,3)" 變成數字 (4, 3)。
    如果不是座標（例如 "飛出台面"），回傳 None。
    """

    if not isinstance(label, str):
        return None

    match = re.fullmatch(r"\(([1-5]),([1-5])\)", label)

    if not match:
        return None

    return (int(match.group(1)), int(match.group(2)))
