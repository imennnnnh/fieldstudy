"""依完整實測分布推薦 300 組補測。輸出不包含虛構落點。"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from core.config import (
    FEATURE_COLUMNS, SHOT_COLUMNS, ID_COLUMN, IMPORT_REQUIRED_COLUMNS,
    PARAMETER_LIMITS, ALL_LABELS, clean_landing_point,
)
from core.data_loader import validate_parameters

LOW = np.array([PARAMETER_LIMITS[c][0] for c in FEATURE_COLUMNS])
HIGH = np.array([PARAMETER_LIMITS[c][1] for c in FEATURE_COLUMNS])
WIDTH = HIGH - LOW
SHAPE = tuple(HIGH - LOW + 1)


def normalize(values):
    return (np.asarray(values, dtype=float) - LOW) / WIDTH


def load_complete_observations(paths):
    """只接受每列五球均完整且可辨識的實測表。CSV、Excel 可混合。"""
    frames = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == '.csv':
            try:
                sheets = {path.name: pd.read_csv(path, encoding='utf-8-sig')}
            except UnicodeDecodeError:
                sheets = {path.name: pd.read_csv(path, encoding='big5')}
        elif path.suffix.lower() in {'.xlsx', '.xls'}:
            # 暫存資料夾離開前，必須先關閉工作簿與底層檔案。
            with pd.ExcelFile(path) as excel_file:
                sheets = excel_file.parse(sheet_name=None)
        else:
            raise ValueError('補測資料僅接受 Excel 或 CSV')
        for name, frame in sheets.items():
            frame = frame.dropna(how='all').copy()
            frame.columns = frame.columns.astype(str).str.strip()
            if frame.empty:
                continue
            if not set(FEATURE_COLUMNS).intersection(frame.columns):
                continue
            missing = set(IMPORT_REQUIRED_COLUMNS) - set(frame.columns)
            if missing:
                raise ValueError(f'{path.name} / {name} 缺少欄位：{", ".join(sorted(missing))}')
            frame = validate_parameters(frame)
            if frame[ID_COLUMN].isna().any():
                raise ValueError('完整實測資料不可以有空白測試編號')
            for col in SHOT_COLUMNS:
                cleaned = frame[col].map(clean_landing_point)
                if cleaned.isna().any():
                    raise ValueError(f'{path.name} / {name} 的 {col} 有空白或未知落點；請上傳五球完整的實測資料。')
                frame[col] = cleaned
            frames.append(frame[IMPORT_REQUIRED_COLUMNS])
    if not frames:
        raise ValueError('尚未提供完整實測資料，無法產生補測表')
    return pd.concat(frames, ignore_index=True)


def _codes(values):
    return np.ravel_multi_index((np.asarray(values, dtype=int) - LOW).T, SHAPE)


def _decode(codes):
    return np.array(np.unravel_index(codes, SHAPE)).T + LOW


def _farthest(candidates, references, count):
    """先挑離實測最近鄰最遠的點，逐點更新與已選點的最短距離。"""
    if count == 0 or len(candidates) == 0:
        return np.array([], dtype=int)
    candidates_norm = normalize(candidates)
    distance = cKDTree(normalize(references)).query(candidates_norm)[0]
    chosen = []
    for _ in range(min(count, len(candidates))):
        i = int(np.argmax(distance))
        chosen.append(i)
        distance = np.minimum(distance, np.linalg.norm(candidates_norm - candidates_norm[i], axis=1))
        distance[chosen] = -np.inf
    return np.array(chosen, dtype=int)


def generate_supplement(observations, random_state=None, progress_callback=None):
    """150 空間探索 + 120 分類邊界補強 + 30 隨機驗證；不足邊界改探索。"""
    # 公開函數也驗證，避免由其他入口繞過完整資料要求。
    missing = set(IMPORT_REQUIRED_COLUMNS) - set(observations.columns)
    if missing or observations.empty:
        raise ValueError('必須提供完整實測資料')
    data = validate_parameters(observations.copy())
    for col in SHOT_COLUMNS:
        data[col] = data[col].map(clean_landing_point)
        if data[col].isna().any():
            raise ValueError('必須提供五球落點完整且可辨識的實測資料')
    long = data.melt(id_vars=FEATURE_COLUMNS, value_vars=SHOT_COLUMNS, value_name='落點')
    return generate_from_training_data(long, random_state=random_state, progress_callback=progress_callback)


def generate_from_training_data(observations, tested_parameters=None, random_state=None, progress_callback=None):
    """直接使用模型真正參與訓練的長資料；不補造原本空白的球。"""
    if not isinstance(observations, pd.DataFrame) or observations.empty or not set(FEATURE_COLUMNS + ['落點']).issubset(observations.columns):
        raise ValueError('目前模型缺少可用的實測訓練資料')
    long = validate_parameters(observations.copy())
    long['落點'] = long['落點'].map(clean_landing_point)
    if long['落點'].isna().any():
        raise ValueError('模型訓練快照包含空白或未知落點，請重新訓練')
    rng = np.random.default_rng(random_state)
    counts = long.groupby(FEATURE_COLUMNS + ['落點']).size().unstack(fill_value=0).reindex(columns=ALL_LABELS, fill_value=0)
    distributions = counts.div(counts.sum(axis=1), axis=0).to_numpy()
    measured = counts.index.to_frame(index=False).to_numpy(dtype=int)
    excluded_values = measured
    if tested_parameters is not None and len(tested_parameters):
        extra = pd.DataFrame(list(tested_parameters), columns=FEATURE_COLUMNS)
        extra = validate_parameters(extra)
        excluded_values = np.vstack([measured, extra.to_numpy()])
    tested_codes = np.unique(_codes(excluded_values))
    available = np.prod(SHAPE) - len(tested_codes)
    if available < 300:
        raise ValueError(f'僅剩 {available} 組未測參數，無法產生 300 組')
    if progress_callback:
        progress_callback('檢查分類邊界與已測參數', 10)
    pairs = cKDTree(normalize(measured)).query_pairs(0.15, output_type='ndarray')
    if len(pairs):
        different = np.any(np.abs(distributions[pairs[:, 0]] - distributions[pairs[:, 1]]) > 1e-12, axis=1)
        pairs = pairs[different]
    boundary_pair_count = len(pairs)
    if len(pairs) > 4000:
        pairs = pairs[rng.choice(len(pairs), 4000, replace=False)]
    boundary = np.empty((0, 4), dtype=int)
    if len(pairs):
        midpoints = (measured[pairs[:, 0]] + measured[pairs[:, 1]]) / 2
        centers = np.repeat(midpoints, 24, axis=0)
        # 在近鄰異分布點對的中點附近提出候選；不是模型保證的真正邊界。
        proposed = np.rint(centers + rng.uniform(-0.035, 0.035, centers.shape) * WIDTH).astype(int)
        proposed = np.clip(proposed, LOW, HIGH)
        codes = np.unique(_codes(proposed))
        codes = codes[~np.isin(codes, tested_codes)]
        rng.shuffle(codes)
        candidates = _decode(codes)
        boundary = candidates[_farthest(candidates, measured, min(120, len(candidates)))]
    deficit = 120 - len(boundary)
    if progress_callback:
        progress_callback('挑選低密度區域與隨機驗證組合', 35)
    # 全域均勻、不放回的候選池；排除完整實測及已選邊界。
    excluded = np.concatenate([tested_codes, _codes(boundary)])
    pool_codes = rng.choice(int(np.prod(SHAPE)), min(int(np.prod(SHAPE)), 40000), replace=False)
    pool_codes = pool_codes[~np.isin(pool_codes, excluded)]
    if len(pool_codes) < 180 + deficit:
        pool_codes = np.setdiff1d(np.arange(np.prod(SHAPE)), excluded)
        rng.shuffle(pool_codes)
        pool_codes = pool_codes[:40000]
    pool = _decode(pool_codes)
    references = np.vstack([measured, boundary])
    spatial_idx = _farthest(pool, references, 150 + deficit)
    spatial = pool[spatial_idx]
    remaining = np.delete(pool, spatial_idx, axis=0)
    random_rows = remaining[rng.choice(len(remaining), 30, replace=False)]
    values = np.vstack([spatial, boundary, random_rows])
    result = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    result.insert(0, ID_COLUMN, [f'補測{i:03d}' for i in range(1, 301)])
    for col in SHOT_COLUMNS:
        result[col] = ''
    strategy = pd.DataFrame({
        ID_COLUMN: result[ID_COLUMN],
        '採樣策略': ['空間探索']*len(spatial) + ['分類邊界補強']*len(boundary) + ['隨機驗證']*30,
    })
    summary = {
        'total': 300, 'measured_count': len(measured), 'excluded_count': len(tested_codes), 'observed_shots': len(long), 'boundary_pair_count': boundary_pair_count,
        'spatial_count': len(spatial), 'boundary_count': len(boundary), 'random_count': 30,
        'boundary_fallback_count': deficit, 'normalized_radius': 0.15,
    }
    if progress_callback:
        progress_callback('300 組補測參數已產生', 95)
    return result[IMPORT_REQUIRED_COLUMNS], strategy, summary
