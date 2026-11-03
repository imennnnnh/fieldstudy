"""同一目標格的多種設定偏好。只在嚴格合格候選中挑選。"""
import threading

import numpy as np
import pandas as pd

from core.config import FEATURE_COLUMNS, ROUTE_NAMES_FOR_WEB
from core.predictor import get_best_model
from core.reverse_search import (
    normalize_target_labels, normalize_ball_conditions, check_targets_in_model,
    build_search_grid, build_base_result, filter_accurate_candidates,
    calculate_condition_score, normalized_parameter_series,
)

# 只記錄這次服務啟動後已顯示的方案。重訓會更換模型，歷史自動分開。
_history = {}
_history_lock = threading.Lock()


def reset_history():
    with _history_lock:
        _history.clear()


def select_diverse_solutions(result_df, n=5, min_normalized_distance=0.08):
    if result_df.empty:
        return result_df.copy()
    vectors = normalized_parameter_series(result_df).to_numpy()
    selected = []
    for i, vector in enumerate(vectors):
        if not selected or np.min(np.linalg.norm(vectors[selected] - vector, axis=1)) >= min_normalized_distance:
            selected.append(i)
        if len(selected) >= n:
            break
    # 多樣性是偏好；候選相近時仍可提供其他「不同且合格」的組合。
    if len(selected) < n:
        selected += [i for i in range(len(result_df)) if i not in selected][:n-len(selected)]
    return result_df.iloc[selected].reset_index(drop=True)


def recommend_multi_route(
    target_landing, ball_conditions, n_solutions=5, pool_for_diversity=1000,
    min_target_probability=1.0, max_invalid_probability=0.0,
    spin_direction_value=None, exclude_tested_parameters=False,
    grid_ranges=None, progress_callback=None, chunk_size=50000, random_state=None,
):
    """掃描一次。未顯示過的組合優先，再依設定偏好與隨機同分排序。"""
    targets = normalize_target_labels(target_landing)
    conditions = normalize_ball_conditions(ball_conditions)
    if not conditions or any(c not in ROUTE_NAMES_FOR_WEB for c in conditions):
        raise ValueError("請選擇有效的球路選項")
    if not isinstance(n_solutions, int) or not 1 <= n_solutions <= 20:
        raise ValueError("每種球路的方案數必須在 1~20 之間")
    check_targets_in_model(targets)
    rng = np.random.default_rng(random_state)
    model = get_best_model()
    keys = {c: (id(model), targets[0], c) for c in conditions}
    with _history_lock:
        used = {c: set(_history.get(keys[c], set())) for c in conditions}
    pool_size = max(pool_for_diversity, n_solutions * 50)
    pools = {c: pd.DataFrame() for c in conditions}
    counts = {c: 0 for c in conditions}
    grid = build_search_grid(grid_ranges, exclude_tested_parameters)

    def rank(df):
        return df.sort_values(["_fresh", "球路條件分數", "_tie"], ascending=[False, False, False]).head(pool_size)

    for start in range(0, len(grid), chunk_size):
        end = min(start + chunk_size, len(grid))
        base = filter_accurate_candidates(build_base_result(grid.iloc[start:end], targets))
        if not base.empty:
            tuples = list(base[FEATURE_COLUMNS].itertuples(index=False, name=None))
            for condition in conditions:
                batch = base.copy()
                batch["球路條件分數"] = calculate_condition_score(batch, condition).to_numpy()
                batch["_fresh"] = [t not in used[condition] for t in tuples]
                batch["_tie"] = rng.random(len(batch))
                counts[condition] += len(batch)
                pools[condition] = rank(pd.concat([pools[condition], rank(batch)], ignore_index=True))
        if progress_callback:
            progress_callback(end, len(grid))
    results, statuses = {}, {}
    for condition in conditions:
        pool = pools[condition]
        # 候選足夠時，先把尚未顯示的挑完；不足才重用舊組合。
        if not pool.empty:
            fresh = pool.loc[pool["_fresh"]]
            selected = select_diverse_solutions(fresh, n_solutions)
            if len(selected) < n_solutions:
                previous = pool.loc[~pool["_fresh"]]
                selected = pd.concat([selected, select_diverse_solutions(previous, n_solutions-len(selected))], ignore_index=True)
            shown = set(selected[FEATURE_COLUMNS].itertuples(index=False, name=None))
            reused = sum(t in used[condition] for t in shown)
            with _history_lock:
                _history.setdefault(keys[condition], set()).update(shown)
            selected = selected.drop(columns=["_fresh", "_tie"])
            status = f"提供 {len(selected)} 組；共 {counts[condition]:,} 組符合固定模型機率條件。"
            if reused:
                status += f" 未顯示過的組合不足，{reused} 組曾在本次服務中顯示。"
        else:
            selected = pool.copy()
            status = "沒有符合條件的方案：模型預測目標格 100%、無效球 0%。"
        selected.insert(0, "目標落點", targets[0])
        selected.insert(1, "球路條件", condition)
        results[condition], statuses[condition] = selected, status
    return results, statuses, len(grid)
