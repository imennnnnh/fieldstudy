# ============================================================
# reverse_search.py
# 負責：「指定目標落點 → 反推四個參數」（完整掃描 1,471,932 組）
# （原本 Colab 第 22 格共用函數 + 第 25 格）
#
# 這個檔案同時放了「球路條件的共用函數」，
# multi_route.py 也會從這裡匯入使用。
#
# 搜尋原則：固定機率篩選，沒有不合格候選回傳。
#   第一階段：落點精準度是硬條件
#       ① 模型第一名預測必須落在目標
#       ② 目標格機率必須為 100%
#       ③ 無效球與所有非目標類別機率必須為 0%
#   第二階段：只有通過第一階段的候選，才依球路條件排序
# ============================================================

import numpy as np
import pandas as pd

from core.config import (
    FEATURE_COLUMNS,
    PARAMETER_LIMITS,
    VALID_LABELS,
    INVALID_LABELS,
    BALL_CONDITION_MAP,
    MIN_TARGET_PROBABILITY, MAX_INVALID_PROBABILITY,
    clean_landing_point,
    parse_coordinate,
)
from core.predictor import (
    get_best_model,
    get_class_to_index,
    get_tested_parameters,
)


# ============================================================
# 第一部分：共用小函數
# ============================================================

def calculate_invalid_probability(probabilities, class_to_index):
    """
    計算「無效球總機率」= 4 種無效結果的機率加總。
    probabilities 是一個表格：每列一組參數，每欄一個落點的機率。
    """

    # 一開始每組參數都是 0
    invalid_prob = np.zeros(len(probabilities))

    for invalid_label in class_to_index:
        if invalid_label in VALID_LABELS:
            continue
        if invalid_label in class_to_index:
            column_index = class_to_index[invalid_label]
            invalid_prob = invalid_prob + probabilities[:, column_index]

    return invalid_prob


def normalize_target_labels(target_landing):
    """
    把目標落點整理成乾淨的 list。
    接受字串 "(4,3)" 或單元素清單 ["(4,3)"]。
    """

    if isinstance(target_landing, str):
        labels = [target_landing]
    elif isinstance(target_landing, (list, tuple, set)):
        labels = list(target_landing)
    else:
        raise ValueError("目標落點必須是 '(4,3)' 或 ['(4,3)'] 這種格式")

    normalized = []

    for label in labels:
        cleaned = clean_landing_point(label)

        if pd.isna(cleaned) or cleaned not in VALID_LABELS:
            raise ValueError(f"無效目標落點：{label}，必須是 (1,1)~(5,5)")

        normalized.append(cleaned)

    if len(normalized) != 1:
        raise ValueError("請指定一個目標格子；100% 指的是該格，不能合併多格機率。")
    return normalized


def distance_to_targets(predicted_label, target_labels):
    """
    計算「預測落點」到「目標範圍」的最小曼哈頓距離。
    曼哈頓距離 = |x差| + |y差|。
    預測落點如果不是座標（例如飛出台面），距離視為無限大。
    """

    pred_xy = parse_coordinate(predicted_label)

    if pred_xy is None:
        return np.inf

    distances = []

    for target in target_labels:
        target_xy = parse_coordinate(target)

        if target_xy is None:
            continue

        dx = pred_xy[0] - target_xy[0]
        dy = pred_xy[1] - target_xy[1]
        distances.append(abs(dx) + abs(dy))

    if len(distances) == 0:
        return np.inf

    return min(distances)


# ============================================================
# 第二部分：球路條件（高速球、低速球、旋轉方向…）
# ============================================================

def normalize_ball_conditions(ball_condition):
    """把球路條件整理成 list，並檢查名稱是否存在。"""

    if ball_condition is None:
        return ["標準"]

    if isinstance(ball_condition, str):
        conditions = [ball_condition]
    elif isinstance(ball_condition, (list, tuple, set)):
        conditions = list(ball_condition)
    else:
        raise ValueError("球路條件必須是字串或 list")

    for condition in conditions:
        if condition not in BALL_CONDITION_MAP:
            raise ValueError(
                f"未知球路條件：{condition}，可使用：{list(BALL_CONDITION_MAP.keys())}"
            )

    return list(dict.fromkeys(conditions))


def validate_spin_direction_value(spin_direction_value):
    """檢查旋球設定值：必須是 1~59 的整數。"""

    if spin_direction_value is None or pd.isna(spin_direction_value):
        raise ValueError("使用『旋轉方向』時，必須指定旋球設定值 1~59")

    try:
        number = float(spin_direction_value)
    except (TypeError, ValueError):
        raise ValueError("旋球設定值必須是數字")

    if int(number) != number:
        raise ValueError("旋球設定值必須是 1~59 的整數")

    number = int(number)

    if number < 1 or number > 59:
        raise ValueError("旋球設定值必須在 1~59 之間")

    return number


def circular_spin_setting_distance(spin_values, target_value):
    """
    旋球設定值 1~59 是一整圈，所以 1 和 59 視為相鄰。
    回傳「設定值的環狀距離」（不是角度差）。
    """

    target_value = validate_spin_direction_value(target_value)

    values = np.asarray(spin_values, dtype=float)

    direct_distance = np.abs(values - target_value)

    # 繞另一邊走比較近的話，取較小的距離
    circular_distance = np.minimum(direct_distance, 59 - direct_distance)

    return circular_distance


def normalized_parameter_series(df_input):
    """把四個參數都縮放到 0~1，方便比較。"""

    result = pd.DataFrame(index=df_input.index)

    for col, (min_value, max_value) in PARAMETER_LIMITS.items():
        result[col] = (df_input[col] - min_value) / (max_value - min_value)

    return result.clip(0, 1)


def calculate_condition_score(candidate_df, ball_condition, spin_direction_value=None):
    """
    計算球路條件分數：0 = 完全不符合，1 = 最符合。
    這個分數只用在「落點已經合格」之後的第二階段排序。
    """

    conditions = normalize_ball_conditions(ball_condition)

    if "旋轉方向" in conditions:
        spin_direction_value = validate_spin_direction_value(spin_direction_value)

    norm_df = normalized_parameter_series(candidate_df)

    score_list = []   # 每個球路條件各一個分數

    for condition in conditions:

        rules = BALL_CONDITION_MAP[condition]

        if len(rules) == 0:
            # 標準 / 一般：全部給 0.5，不偏向任何方向
            score = pd.Series(0.5, index=candidate_df.index)

        else:
            rule_scores = []

            for col, direction in rules:

                if direction == "max":
                    rule_score = norm_df[col]

                elif direction == "min":
                    rule_score = 1 - norm_df[col]

                elif direction == "mid":
                    rule_score = 1 - 2 * (norm_df[col] - 0.5).abs()

                elif direction == "target_circular":
                    distance = circular_spin_setting_distance(
                        candidate_df[col].to_numpy(),
                        spin_direction_value,
                    )
                    # 59 個環狀位置，最遠的最短距離是 29
                    rule_score = pd.Series(
                        1 - (distance / 29.0),
                        index=candidate_df.index,
                    ).clip(0, 1)

                else:
                    raise ValueError(f"未知方向：{direction}")

                rule_scores.append(rule_score)

            score = sum(rule_scores) / len(rule_scores)

        score_list.append(score)

    # 多個條件取平均
    return sum(score_list) / len(score_list)


# ============================================================
# 第三部分：篩選與排序候選參數
# ============================================================

def filter_accurate_candidates(candidate_df, min_target_probability=1.0,
                               max_invalid_probability=0.0):
    """不使用四捨五入或 isclose；拒絕任何低於 100% 或非零無效機率。"""
    if candidate_df.empty:
        return candidate_df.copy()
    mask = ((candidate_df["命中目標"] == 1)
            & (candidate_df["目標落點機率"] >= MIN_TARGET_PROBABILITY)
            & (candidate_df["無效總機率"] == MAX_INVALID_PROBABILITY))
    if "非目標總機率" in candidate_df:
        mask &= candidate_df["非目標總機率"] == 0.0
    return candidate_df.loc[mask].copy()


def rank_candidate_dataframe(candidate_df, ball_condition="標準"):
    """第二階段：依球路條件排序（已通過落點門檻的候選）。"""

    if candidate_df.empty:
        return candidate_df.copy()

    conditions = normalize_ball_conditions(ball_condition)

    # 有沒有指定特別的球路？（不是「標準」或「一般」）
    has_specific_route = any(c not in ["標準", "一般"] for c in conditions)

    if has_specific_route:
        # 球路條件分數最優先
        sort_columns = ["球路條件分數", "目標落點機率", "無效總機率", "目標距離_曼哈頓"]
        ascending = [False, False, True, True]
    else:
        # 標準方案：以落點精準度排序
        sort_columns = ["目標落點機率", "無效總機率", "目標距離_曼哈頓"]
        ascending = [False, True, True]

    return (
        candidate_df
        .sort_values(by=sort_columns, ascending=ascending)
        .reset_index(drop=True)
    )


# ============================================================
# 第四部分：建立搜尋空間（所有可能的四參數組合）
# ============================================================

def build_search_grid(ranges=None, exclude_tested_parameters=False):
    """
    產生所有四參數組合，回傳 DataFrame（每列一組參數）。

    ranges 是測試用的小範圍，平常保持 None 就是完整範圍：
        仰角 1~28 × 左右 1~9 × 旋球 1~59 × 強度 1~99 = 1,471,932 組
    """

    if ranges is None:
        ranges = {}

    # 每個參數的所有可能數值
    elevation_values = np.array(ranges.get("仰角角度", range(1, 29)), dtype=np.int64)
    horizontal_values = np.array(ranges.get("左右位置", range(1, 10)), dtype=np.int64)
    spin_values = np.array(ranges.get("旋球角度", range(1, 60)), dtype=np.int64)
    power_values = np.array(ranges.get("出球強度", range(1, 100)), dtype=np.int64)

    # meshgrid 會把四個數列「全部互相配對」
    elevation_grid, horizontal_grid, spin_grid, power_grid = np.meshgrid(
        elevation_values,
        horizontal_values,
        spin_values,
        power_values,
        indexing="ij",
    )

    grid_df = pd.DataFrame({
        "仰角角度": elevation_grid.ravel(),
        "左右位置": horizontal_grid.ravel(),
        "旋球角度": spin_grid.ravel(),
        "出球強度": power_grid.ravel(),
    })

    # 探索模式：排除已經測過的參數
    if exclude_tested_parameters:
        tested_parameters = get_tested_parameters()

        if len(tested_parameters) > 0:
            # 把四個數字合成一個編碼，方便比對
            def make_key(e, h, s, p):
                return ((e * 10 + h) * 100 + s) * 1000 + p

            tested_keys = np.array(
                [make_key(e, h, s, p) for (e, h, s, p) in tested_parameters],
                dtype=np.int64,
            )

            grid_keys = make_key(
                grid_df["仰角角度"].to_numpy(),
                grid_df["左右位置"].to_numpy(),
                grid_df["旋球角度"].to_numpy(),
                grid_df["出球強度"].to_numpy(),
            )

            grid_df = grid_df.loc[~np.isin(grid_keys, tested_keys)].reset_index(drop=True)

    return grid_df[FEATURE_COLUMNS]


# ============================================================
# 第五部分：計算一批候選的預測結果
# ============================================================

def build_base_result(batch_df, target_labels):
    """
    對一批參數（batch_df）呼叫模型，算出：
      預測落點、命中目標、目標落點機率、無效總機率、目標距離
    （球路條件分數由呼叫端另外加上去）
    """

    best_model = get_best_model()
    class_to_index = get_class_to_index()

    # 模型預測 29 類的機率（這一步最花時間）
    probabilities = best_model.predict_proba(batch_df)

    # 目標落點機率 = 目標範圍內每個格子的機率加總
    target_probability = np.zeros(len(batch_df))

    for target_label in target_labels:
        target_index = class_to_index[target_label]
        target_probability = target_probability + probabilities[:, target_index]

    # 機率最高的那一類 = 模型預測落點
    best_index = np.argmax(probabilities, axis=1)
    predicted_labels = best_model.classes_[best_index]

    # 預測有沒有落在目標範圍內
    hit_target = np.isin(predicted_labels, target_labels).astype(int)

    # 無效球總機率
    invalid_probability = calculate_invalid_probability(probabilities, class_to_index)

    # 距離：每個「類別」的距離先算一次，再用 best_index 對應過去（比逐筆算快很多）
    class_distance = np.array([
        distance_to_targets(label, target_labels)
        for label in best_model.classes_
    ])
    distances = class_distance[best_index]

    base_result = batch_df.copy()
    base_result["預測落點"] = predicted_labels
    base_result["命中目標"] = hit_target
    base_result["目標落點機率"] = target_probability
    base_result["無效總機率"] = invalid_probability
    outside = [i for label, i in class_to_index.items() if label not in target_labels]
    base_result["非目標總機率"] = probabilities[:, outside].sum(axis=1)
    base_result["目標距離_曼哈頓"] = distances

    return base_result


def check_targets_in_model(target_labels):
    """確認目標落點有在模型學過的類別裡。"""

    class_to_index = get_class_to_index()

    missing_targets = [label for label in target_labels if label not in class_to_index]

    if len(missing_targets) > 0:
        raise ValueError(
            "目前模型沒有學到目標類別：" + ", ".join(missing_targets)
            + "。請確認訓練資料中有這些落點。"
        )


# ============================================================
# 第六部分：主功能 — 目標落點反推四參數
# ============================================================

def search_best_parameters(
    target_landing, ball_condition="標準", top_n=20, chunk_size=50000,
    pool_size=None, min_target_probability=1.0, max_invalid_probability=0.0,
    spin_direction_value=None, exclude_tested_parameters=False,
    grid_ranges=None, progress_callback=None,
):
    """四參數完整搜尋，只回傳符合固定條件的組合；沒有備用回傳。"""
    targets = normalize_target_labels(target_landing)
    check_targets_in_model(targets)
    if not isinstance(top_n, int) or not 1 <= top_n <= 100:
        raise ValueError("顯示組數必須在 1~100 之間")
    grid = build_search_grid(grid_ranges, exclude_tested_parameters)
    pool = pd.DataFrame()
    for start in range(0, len(grid), chunk_size):
        end = min(start + chunk_size, len(grid))
        batch = filter_accurate_candidates(build_base_result(grid.iloc[start:end], targets))
        if not batch.empty:
            pool = pd.concat([pool, batch], ignore_index=True).head(top_n)
        if progress_callback:
            progress_callback(end, len(grid))
    pool.insert(0, "目標落點", targets[0])
    status = (f"找到 {len(pool)} 組符合固定模型機率條件的方案。" if not pool.empty
              else "沒有符合條件的方案：模型預測目標格 100%、無效球 0%。")
    return pool, status, len(grid)
