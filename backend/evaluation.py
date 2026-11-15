"""用已發布的 X、Y 模型評估一份獨立測試活頁簿。"""

from .excel_loader import DataError, read_excel
from .training import CANDIDATES, evaluate, score_axis


ALGORITHM_NAMES = {"RF": "Random Forest", "ET": "Extra Trees"}


def evaluate_test_workbook(published, source):
    """依四參數組合預測，並對每顆有實測結果的球評分。"""
    balls, counts = read_excel([source], lambda *args: None)
    if not balls:
        raise DataError(
            f"「{source.filename}」沒有可評分的球。請填入至少一顆球的落點，"
            "格式為半形座標（例如 (2,3)）或表單指定的無效球文字（例如未過網）。",
            file=source.filename,
        )

    features = list(dict.fromkeys(ball.parameters for ball in balls))
    predicted_x = published.models["x"].predict(features)
    predicted_y = published.models["y"].predict(features)
    by_parameters = {
        parameters: (int(x), int(y))
        for parameters, x, y in zip(features, predicted_x, predicted_y, strict=True)
    }

    records = []
    for ball in balls:
        x, y = by_parameters[ball.parameters]
        records.append({
            "actual_x": ball.landing[0],
            "actual_y": ball.landing[1],
            "predicted_x": x,
            "predicted_y": y,
        })

    models = {}
    for axis in "xy":
        winner = published.report["winners"][axis]
        algorithm, _ = CANDIDATES[winner]
        models[axis] = {
            "algorithm": ALGORITHM_NAMES[algorithm],
            "macro_f1": score_axis(
                [record[f"actual_{axis}"] for record in records],
                [record[f"predicted_{axis}"] for record in records],
            ),
        }

    metrics = evaluate(records)
    metrics["test_balls"] = metrics.pop("validation_balls")
    return {
        "run_id": published.run_id,
        "filename": source.filename,
        "missing_balls": counts["missing_balls"],
        "models": models,
        "evaluation": metrics,
    }
