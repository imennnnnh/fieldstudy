# ============================================================
# predictor.py
# 負責：載入訓練好的模型，並提供「四參數 → 預測落點」功能
# （原本 Colab 第 22、23 格的功能）
# ============================================================

import math

import joblib
import pandas as pd

from core.config import MODEL_PATH, DATA_DIR, PARAMETER_LIMITS, FEATURE_COLUMNS


# ------------------------------------------------------------
# 用一個字典保存「目前載入的模型」
# 整個後端程式共用這一份，不用每次預測都重新讀檔
# ------------------------------------------------------------
_state = {
    "package": None,        # 模型存檔裡的全部內容
    "class_to_index": None  # 落點名稱 → 機率欄位編號
}


def load_model_from_disk():
    """
    從 models/ 資料夾讀取 .pkl 模型。
    成功回傳 True；找不到檔案回傳 False。
    """

    if not MODEL_PATH.exists():
        _state["package"] = None
        _state["class_to_index"] = None
        return False

    package = joblib.load(MODEL_PATH)

    _state["package"] = package

    # 建立「落點名稱 → 第幾欄機率」的對照表
    best_model = package["best_model"]
    _state["class_to_index"] = {
        label: index
        for index, label in enumerate(best_model.classes_)
    }

    return True


def is_model_loaded():
    """目前有沒有載入模型？"""
    return _state["package"] is not None


def get_package():
    """取得模型存檔內容（沒有模型就丟錯誤）。"""
    if _state["package"] is None:
        raise ValueError("目前沒有模型，請先到「訓練」頁上傳資料並訓練。")
    return _state["package"]


def get_best_model():
    """取得 best_model（必須有 predict_proba 方法）。"""
    return get_package()["best_model"]


def get_class_to_index():
    """取得 落點名稱 → 機率欄位編號 的對照表。"""
    get_package()   # 先確認有模型
    return _state["class_to_index"]


def get_tested_parameters():
    """取得訓練資料裡已經測過的四參數組合（set）。"""
    return get_package().get("tested_parameters", set())


def get_training_observations():
    """取得目前模型的實測訓練快照；舊模型從保留的 data 檔重建並核對。"""
    package = get_package()
    snapshot = package.get("training_observations")
    if snapshot is not None:
        if not isinstance(snapshot, pd.DataFrame) or snapshot.empty or not set(FEATURE_COLUMNS + ["落點"]).issubset(snapshot.columns):
            raise ValueError("模型內的訓練資料快照不完整，請回到上傳與訓練頁重新訓練。")
        return snapshot.copy()

    # 相容更新前已訓練的模型；不自動改用無關的資料或另行上傳。
    paths = [p for p in sorted(DATA_DIR.iterdir())
             if p.suffix.lower() in {".xlsx", ".xls"} and not p.name.startswith("~$")]
    if not paths:
        raise ValueError("目前模型沒有訓練資料快照，data 資料夾也沒有原訓練檔；請回到上傳與訓練頁重新訓練。")
    from core.data_loader import prepare_training_data
    data = prepare_training_data(paths)
    expected = package.get("tested_parameters")
    info = package.get("training_info", {})
    if (expected is None or set(data["tested_parameters"]) != set(expected)
            or info.get("total_samples") != len(data["y"])
            or set(package.get("class_names", [])) != set(data["y"].unique())):
        raise ValueError("data 資料夾的檔案與目前模型的訓練紀錄不一致，請先重新訓練模型。")
    package["training_observations"] = data["training_observations"]
    package["training_source_files"] = [p.name for p in paths]
    package["sampling_source_kind"] = "legacy_files"
    return package["training_observations"].copy()


def get_sampling_info():
    if not is_model_loaded():
        return {"ready": False, "message": "請先在上傳與訓練頁完成模型訓練。"}
    try:
        data = get_training_observations()
    except ValueError as error:
        return {"ready": False, "message": str(error)}
    package = get_package()
    return {
        "ready": True,
        "source_files": package.get("training_source_files", []),
        "observed_shots": len(data),
        "observed_settings": len(data[FEATURE_COLUMNS].drop_duplicates()),
        "excluded_settings": len(package.get("tested_parameters", set())),
        "source_kind": package.get("sampling_source_kind", "model_snapshot"),
        "message": "使用目前模型的訓練資料，不需再次上傳。",
    }


def get_model_info():
    """回傳給網頁顯示的「模型狀態」。"""

    if not is_model_loaded():
        return {"has_model": False, "sampling": get_sampling_info()}

    package = get_package()

    return {
        "has_model": True,
        "best_model_name": package["best_model_name"],
        "num_classes": package["num_classes"],
        "class_names": package["class_names"],
        "tested_parameter_count": package["tested_parameter_count"],
        "training_info": package["training_info"],
        "sampling": get_sampling_info(),
    }


# ------------------------------------------------------------
# 檢查使用者輸入的四個參數
# ------------------------------------------------------------

def validate_parameter_values(elevation, horizontal, spin_angle, power):
    """
    檢查四個參數是否為「範圍內的整數」。
    通過後回傳整理好的字典；有問題就丟出 ValueError。
    """

    values = {
        "仰角角度": elevation,
        "左右位置": horizontal,
        "旋球角度": spin_angle,
        "出球強度": power,
    }

    cleaned = {}

    for col, value in values.items():

        min_value, max_value = PARAMETER_LIMITS[col]

        # 沒填
        if value is None or pd.isna(value):
            raise ValueError(f"{col} 不可以是空值")

        if isinstance(value, bool):
            raise ValueError(f"{col} 必須是整數數值")

        # 轉成數字（網頁傳來的可能是文字）
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{col} 必須是數字")

        if not math.isfinite(number):
            raise ValueError(f"{col} 必須是有限整數")

        # 範圍
        if number < min_value or number > max_value:
            raise ValueError(f"{col} 必須在 {min_value}~{max_value} 之間")

        # 整數
        if int(number) != number:
            raise ValueError(f"{col} 必須是整數")

        cleaned[col] = int(number)

    return cleaned


# ------------------------------------------------------------
# 單組參數預測
# ------------------------------------------------------------

def predict_landing(elevation, horizontal, spin_angle, power, top_n=10):
    """
    輸入四個參數，回傳：
      prediction     模型預測的落點（機率最高的那一類）
      probability_df 前 top_n 個最可能的結果與機率
    """

    values = validate_parameter_values(elevation, horizontal, spin_angle, power)

    best_model = get_best_model()

    # 模型需要 DataFrame，欄位順序要和訓練時一樣
    input_df = pd.DataFrame([values])[FEATURE_COLUMNS]

    prediction = best_model.predict(input_df)[0]
    probabilities = best_model.predict_proba(input_df)[0]

    probability_df = pd.DataFrame({
        "落點": best_model.classes_,
        "預測機率": probabilities,
    })

    probability_df = (
        probability_df
        .sort_values("預測機率", ascending=False)
        .reset_index(drop=True)
    )

    return prediction, probability_df.head(top_n)
