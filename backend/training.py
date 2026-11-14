from math import dist
from statistics import mean

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from .excel_loader import PARAMETER_KEYS


FOLDS = "ABCDE"
TREE_COUNT = 500
RANDOM_SEED = 42
AXIS_LABELS = list(range(6))
ALGORITHMS = {"RF": RandomForestClassifier, "ET": ExtraTreesClassifier}
CANDIDATES = {
    f"{algorithm}{axis}": (algorithm, column)
    for column, axis in enumerate("xy")
    for algorithm in ALGORITHMS
}
REPORT_NOTE = "同一批資料選模後的驗證表現，可能偏樂觀"


class TrainingError(ValueError):
    pass


def make_folds(features, coordinates):
    if not len(features):
        raise TrainingError("這批資料沒有可訓練的逐球紀錄。")
    settings, groups = np.unique(features, axis=0, return_inverse=True)
    groups = groups.ravel()
    if len(settings) < len(FOLDS):
        raise TrainingError("五折驗證至少需要五組不同的四參數設定。")

    landing_classes = coordinates[:, 0] * 6 + coordinates[:, 1]
    splitter = StratifiedGroupKFold(
        n_splits=len(FOLDS), shuffle=True, random_state=RANDOM_SEED,
    )
    try:
        folds = list(splitter.split(features, landing_classes, groups))
    except ValueError as error:
        raise TrainingError(f"無法依整球落點完成分組五折切分（{error}）。") from error

    seen = np.zeros(len(features), dtype=int)
    for training, validation in folds:
        if not len(training) or not len(validation):
            raise TrainingError("切分後至少一折缺少訓練或驗證資料。")
        if np.intersect1d(groups[training], groups[validation]).size:
            raise TrainingError("切分後同一組四參數出現在同折的訓練集與驗證集。")
        seen[validation] += 1
    if not np.all(seen == 1):
        raise TrainingError("切分後每球必須恰好被驗證一次。")
    return folds


def make_model(algorithm):
    return ALGORITHMS[algorithm](
        n_estimators=TREE_COUNT,
        max_features=2,
        max_depth=None,
        bootstrap=algorithm == "RF",
        random_state=RANDOM_SEED,
    )


def score_axis(actual, predicted):
    return float(f1_score(
        actual, predicted, labels=AXIS_LABELS, average="macro", zero_division=0,
    ))


def is_valid(landing):
    return all(1 <= coordinate <= 5 for coordinate in landing)


def evaluate(records):
    # 矩陣以實際為列、預測為欄，索引 0 為有效，1 為無效。
    matrices = {name: [[0, 0], [0, 0]] for name in ("x", "y", "ball")}
    hits = dict(x=0, y=0, ball=0)
    distances = []
    for record in records:
        actual = (record["actual_x"], record["actual_y"])
        predicted = (record["predicted_x"], record["predicted_y"])
        for axis, real, estimate in zip("xy", actual, predicted, strict=True):
            hits[axis] += real == estimate
            matrices[axis][int(real == 0)][int(estimate == 0)] += 1
        hits["ball"] += actual == predicted
        actual_valid, predicted_valid = is_valid(actual), is_valid(predicted)
        matrices["ball"][int(not actual_valid)][int(not predicted_valid)] += 1
        if actual_valid and predicted_valid:
            distances.append(dist(actual, predicted))

    (valid_valid, valid_invalid), (invalid_valid, invalid_invalid) = matrices["ball"]
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "validation_balls": len(records),
        "hit_rates": {name: ratio(count, len(records)) for name, count in hits.items()},
        "validity_matrices": matrices,
        "actual_valid_given_predicted_valid": ratio(valid_valid, valid_valid + invalid_valid),
        "actual_invalid_given_predicted_invalid": ratio(invalid_invalid, valid_invalid + invalid_invalid),
        "mean_distance": mean(distances) if distances else None,
        "distance_balls": len(distances),
    }


def validation_record(ball, fold, predictions, winners):
    return {
        "ball_id": ball.ball_id,
        "file_id": ball.file_id,
        "filename": ball.filename,
        "sheet": ball.sheet,
        "excel_row": ball.row,
        "test_id": ball.test_id,
        "serve": ball.serve,
        **dict(zip(PARAMETER_KEYS, ball.parameters, strict=True)),
        "raw_landing": ball.raw_landing,
        "fold": fold,
        "actual_x": ball.landing[0],
        "actual_y": ball.landing[1],
        **predictions,
        "predicted_x": predictions[winners["x"]],
        "predicted_y": predictions[winners["y"]],
    }


def train(balls, counts, run_id, progress):
    progress("splitting", "建立 A～E 五折")
    features = np.asarray([ball.parameters for ball in balls], dtype=int)
    coordinates = np.asarray([ball.landing for ball in balls], dtype=int)
    folds = make_folds(features, coordinates)
    ball_ids = [ball.ball_id for ball in balls]
    if len(set(ball_ids)) != len(ball_ids):
        raise TrainingError("逐球紀錄包含重複的唯一球號。")

    scores = {name: {} for name in CANDIDATES}
    predictions = {name: {} for name in CANDIDATES}
    fold_by_id = {}
    completed = 0
    for fold, (training, validation) in zip(FOLDS, folds, strict=True):
        ids = [ball_ids[index] for index in validation]
        fold_by_id.update(dict.fromkeys(ids, fold))
        for name, (algorithm, column) in CANDIDATES.items():
            message = f"{fold} 折 {name}"
            progress("cross_validation", message, fold=fold, model=name, completed=completed, total=20)
            model = make_model(algorithm).fit(features[training], coordinates[training, column])
            predicted = model.predict(features[validation])
            scores[name][fold] = score_axis(coordinates[validation, column], predicted)
            predictions[name].update(zip(ids, map(int, predicted), strict=True))
            completed += 1

    progress("evaluating", "選出 X、Y 模型並計算整批評估", completed=20, total=20)
    selection = {
        name: {"folds": fold_scores, "mean": mean(fold_scores.values())}
        for name, fold_scores in scores.items()
    }
    winners = {
        axis: max((f"RF{axis}", f"ET{axis}"), key=lambda name: selection[name]["mean"])
        for axis in "xy"
    }
    records = [
        validation_record(
            ball, fold_by_id[ball.ball_id],
            {name: by_id[ball.ball_id] for name, by_id in predictions.items()},
            winners,
        )
        for ball in balls
    ]
    evaluation = evaluate(records)

    models = {}
    for column, axis in enumerate("xy"):
        winner = winners[axis]
        progress("refitting", f"重訓 {axis.upper()} 軸 {winner}", model=winner, completed=column, total=2)
        models[axis] = make_model(CANDIDATES[winner][0]).fit(features, coordinates[:, column])
    return models, records, {
        "run_id": run_id,
        "data": counts,
        "selection": selection,
        "winners": winners,
        "evaluation": evaluation,
        "note": REPORT_NOTE,
    }
