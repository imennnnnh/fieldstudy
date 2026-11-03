# ============================================================
# train_model.py
# 訓練腳本：5 折分層分組交叉驗證 → 選出最佳演算法 → 全資料重訓 → 存檔
#
# 兩種用法：
#   1. 終端機直接執行：   python train_model.py
#      （會讀取 data/ 資料夾裡所有 Excel）
#   2. 由 app.py 呼叫：   train_and_save(...)
#      （網頁上傳資料後觸發）
#
# 流程：
#   步驟 1  讀取資料
#   步驟 2  (A) 5 折交叉驗證：比較 Random Forest 與 Extra Trees   ← 計時 A
#   步驟 3  選出平均 Macro F1 較高的演算法
#   步驟 4  (B) 用 100% 資料重訓最佳模型                          ← 計時 B
#   步驟 5  存檔成 .pkl
# ============================================================

import time          # 計時用
import warnings      # 控制警告訊息
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

from core.config import (
    DATA_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
    N_ESTIMATORS,
    N_SPLITS,
    RANDOM_STATE,
    FEATURE_COLUMNS,
    PARAMETER_LIMITS,
    VALID_LABELS,
    INVALID_LABELS,
    ALL_LABELS,
    BALL_CONDITION_MAP,
)
from core.data_loader import prepare_training_data


# ------------------------------------------------------------
# 建立兩種模型（參數和原本 Colab 完全相同，樹數量維持 500）
# ------------------------------------------------------------

def make_random_forest():
    """建立一個全新的 Random Forest。"""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,          # 500 棵樹
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,                          # 用全部 CPU 核心
    )


def make_extra_trees():
    """建立一個全新的 Extra Trees。"""
    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,          # 500 棵樹
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# 模型名稱 → 建立模型的函數
MODEL_BUILDERS = {
    "Random Forest": make_random_forest,
    "Extra Trees": make_extra_trees,
}


# ------------------------------------------------------------
# 小工具：同時印在終端機，也回報給網頁
# ------------------------------------------------------------

def say(message, progress_callback=None, percent=None):
    """印出訊息；如果有 progress_callback 也一起通知網頁。"""
    print(message, flush=True)
    if progress_callback is not None:
        progress_callback(message, percent)


# ------------------------------------------------------------
# 步驟 2：5 折分層分組交叉驗證（只做評估，不拿來上線）
# ------------------------------------------------------------

def run_cross_validation(X, y, groups, progress_callback=None):
    """
    用 StratifiedGroupKFold(n_splits=5) 評估 RF 與 ET。

    - Stratified：盡量讓 29 類落點在每個 Fold 分佈平均
    - Group：同一組四參數的 5 筆資料一定在同一個 Fold（避免資料洩漏）

    回傳 (fold_df, summary_df)
      fold_df     每個模型、每個 Fold 的成績
      summary_df  每個模型 5 折平均成績
    """

    # 檢查：有哪些落點的「組數」少於 5？（太少的類別無法平均分到 5 折）
    groups_per_class = pd.DataFrame({"落點": y.values, "組": groups.values}) \
        .groupby("落點")["組"].nunique()
    rare_classes = groups_per_class[groups_per_class < N_SPLITS]

    if len(rare_classes) > 0:
        say(" 以下落點的參數組數少於 5，無法平均分到每個 Fold（仍會繼續）：", progress_callback)
        for label, count in rare_classes.items():
            say(f"    {label}：只有 {count} 組", progress_callback)

    # 建立切分器
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 先把 5 折的切法算好，RF 與 ET 使用「完全相同的切法」，比較才公平
    # （此步驟 sklearn 會對稀少類別發出警告，上面已經提醒過了，所以這裡關掉）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fold_splits = list(splitter.split(X, y, groups=groups))

    # 保險：再檢查一次「同一組參數沒有同時出現在訓練與測試」
    for fold_number, (train_idx, test_idx) in enumerate(fold_splits, start=1):
        train_groups = set(groups.iloc[train_idx].unique())
        test_groups = set(groups.iloc[test_idx].unique())
        overlap = train_groups & test_groups
        if len(overlap) > 0:
            raise ValueError(f" Fold {fold_number} 發生資料洩漏：訓練與測試有相同參數組合！")

    say(" 已確認 5 個 Fold 都沒有資料洩漏（訓練/測試參數組合完全不重疊）", progress_callback)

    fold_records = []
    total_steps = len(MODEL_BUILDERS) * N_SPLITS
    finished_steps = 0

    for model_name, build_function in MODEL_BUILDERS.items():

        for fold_number, (train_idx, test_idx) in enumerate(fold_splits, start=1):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # 每個 Fold 都建立「全新」的模型，不能沿用上一折的
            model = build_function()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)

            fold_records.append({
                "模型": model_name,
                "Fold": fold_number,
                "訓練球數": len(train_idx),
                "測試球數": len(test_idx),
                "Macro F1": macro_f1,
                "Balanced Accuracy": balanced_acc,
            })

            finished_steps += 1
            percent = int(5 + 70 * finished_steps / total_steps)   # 進度 5%~75%

            say(
                f"  [{model_name}] Fold {fold_number}/{N_SPLITS}  "
                f"Macro F1 = {macro_f1:.4f}   Balanced Acc = {balanced_acc:.4f}",
                progress_callback,
                percent,
            )

    fold_df = pd.DataFrame(fold_records)

    # 每個模型 5 折平均
    summary_df = (
        fold_df
        .groupby("模型", as_index=False)[["Macro F1", "Balanced Accuracy"]]
        .mean()
        .rename(columns={
            "Macro F1": "平均 Macro F1",
            "Balanced Accuracy": "平均 Balanced Accuracy",
        })
    )

    return fold_df, summary_df


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def train_and_save(excel_paths, progress_callback=None):
    """
    完整訓練流程。
    progress_callback(message, percent) 可選，網頁用來顯示進度。
    回傳一個字典（摘要），讓網頁顯示結果。
    """

    total_start = time.time()

    # ========== 步驟 1：讀取資料 ==========
    say("========== 步驟 1：讀取與檢查資料 ==========", progress_callback, 2)

    data = prepare_training_data(excel_paths)

    X = data["X"]
    y = data["y"]
    groups = data["groups"]

    say(f" 可用球數：{len(X)}", progress_callback, 4)
    say(f" 不同的四參數組合（Group）數量：{groups.nunique()}", progress_callback, 4)
    say(f" 實際出現的落點類別數：{y.nunique()} / {len(ALL_LABELS)}", progress_callback, 5)

    if groups.nunique() < N_SPLITS:
        raise ValueError(
            f"參數組合只有 {groups.nunique()} 組，少於 {N_SPLITS}，無法做 {N_SPLITS} 折交叉驗證。"
        )

    # ========== 步驟 2：(A) 5 折交叉驗證 ==========
    say("\n========== 步驟 2：5 折分層分組交叉驗證 ==========", progress_callback, 5)

    time_a_start = time.time()

    fold_df, summary_df = run_cross_validation(X, y, groups, progress_callback)

    time_a = time.time() - time_a_start

    say("\n各模型 5 折平均成績：", progress_callback)
    say(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"), progress_callback)

    # ========== 步驟 3：選出最佳演算法 ==========
    # 比較「平均 Macro F1」，數字大的獲勝
    best_row = summary_df.loc[summary_df["平均 Macro F1"].idxmax()]
    best_model_name = best_row["模型"]

    say(f"\n 平均 Macro F1 較高的演算法：{best_model_name}", progress_callback, 76)

    # ========== 步驟 4：(B) 用 100% 資料重訓 ==========
    say("\n========== 步驟 3：用全部 100% 資料重新訓練最佳模型 ==========", progress_callback, 78)

    time_b_start = time.time()

    best_model = MODEL_BUILDERS[best_model_name]()   # 建立全新的最佳模型
    best_model.fit(X, y)                             # 用全部資料訓練（不切 train/test）

    time_b = time.time() - time_b_start

    say(" 全資料重訓完成", progress_callback, 92)

    # 檢查：一定要有 predict_proba（後面 147 萬組掃描要用）
    if not hasattr(best_model, "predict_proba"):
        raise ValueError("最佳模型沒有 predict_proba 方法，無法進行後續搜尋")

    # 檢查模型學到幾類
    missing_model_classes = [
        label for label in ALL_LABELS if label not in best_model.classes_
    ]

    # ========== 步驟 5：存檔 ==========
    model_package = {
        # ----- 模型本體（變數名稱依需求命名）-----
        "best_model": best_model,
        "best_model_name": best_model_name,

        # ----- 後續模組需要的資訊 -----
        "feature_columns": FEATURE_COLUMNS,
        "class_names": list(best_model.classes_),
        "num_classes": len(best_model.classes_),
        "parameter_limits": PARAMETER_LIMITS,
        "valid_labels": VALID_LABELS,
        "invalid_labels": INVALID_LABELS,
        "all_labels": ALL_LABELS,
        "ball_condition_map": BALL_CONDITION_MAP,
        "spin_setting_note": "旋球角度為既有 Excel 欄位名稱；實際代表球機 1~59 設定值，範圍涵蓋 360°，不是 1~59 度或旋轉強度。",
        "tested_parameters": data["tested_parameters"],
        "tested_parameter_count": len(data["tested_parameters"]),
        # 與模型一起保存真正參與訓練的球，不依賴另外上傳或日後的檔案內容。
        "training_observations": data["training_observations"],
        "training_source_files": [Path(p).name for p in excel_paths],

        # ----- 訓練資訊 -----
        "training_info": {
            "total_samples": int(len(X)),
            "total_groups": int(groups.nunique()),
            "cv_folds": N_SPLITS,
            "cv_mean_macro_f1": float(best_row["平均 Macro F1"]),
            "cv_mean_balanced_accuracy": float(best_row["平均 Balanced Accuracy"]),
            "time_cv_seconds": float(time_a),
            "time_retrain_seconds": float(time_b),
        },
    }

    joblib.dump(model_package, MODEL_PATH)

    say(f" 模型已儲存：{MODEL_PATH}", progress_callback, 95)

    # 儲存報告 CSV
    fold_df.to_csv(OUTPUT_DIR / "cv_fold_results.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False, encoding="utf-8-sig")

    importance_df = pd.DataFrame({
        "參數": FEATURE_COLUMNS,
        "重要程度": best_model.feature_importances_,
    }).sort_values("重要程度", ascending=False).reset_index(drop=True)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    data["class_counts"].rename("球數").rename_axis("落點").reset_index() \
        .to_csv(OUTPUT_DIR / "class_distribution.csv", index=False, encoding="utf-8-sig")

    # ========== 計時總結 ==========
    total_time = time.time() - total_start

    say("\n==================== 計時結果 ====================", progress_callback)
    say(f"(A) 5 折交叉驗證評估總耗時：{time_a:.1f} 秒（{time_a / 60:.2f} 分鐘）", progress_callback)
    say(f"(B) 最佳模型全資料重訓耗時：{time_b:.1f} 秒（{time_b / 60:.2f} 分鐘）", progress_callback)
    say(f"    整個流程總耗時：{total_time:.1f} 秒", progress_callback)
    say("==================================================", progress_callback, 100)

    # 回傳給網頁的摘要
    return {
        "best_model_name": best_model_name,
        "total_samples": int(len(X)),
        "total_groups": int(groups.nunique()),
        "classes_in_model": int(len(best_model.classes_)),
        "missing_classes": missing_model_classes,
        "cv_summary": summary_df.to_dict("records"),
        "cv_folds": fold_df.to_dict("records"),
        "feature_importance": importance_df.to_dict("records"),
        "time_cv_seconds": round(time_a, 1),
        "time_retrain_seconds": round(time_b, 1),
        "time_total_seconds": round(total_time, 1),
        "import_report": data["import_report"],
        "blank_count": data["blank_count"],
    }


# ------------------------------------------------------------
# 直接在終端機執行 python train_model.py 時，會跑這一段
# ------------------------------------------------------------

if __name__ == "__main__":

    # 找出 data/ 資料夾裡所有的 Excel（不含 Excel 暫存檔 ~$xxx.xlsx）
    excel_files = []
    for path in sorted(DATA_DIR.iterdir()):
        is_excel = path.suffix.lower() in [".xlsx", ".xls"]
        is_temp = path.name.startswith("~$")
        if is_excel and not is_temp:
            excel_files.append(path)

    if len(excel_files) == 0:
        print(f" {DATA_DIR} 裡面沒有 Excel 檔案，請先把實測資料放進去。")
    else:
        print(f"找到 {len(excel_files)} 個 Excel：")
        for path in excel_files:
            print("  -", path.name)
        train_and_save(excel_files)
