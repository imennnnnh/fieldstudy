# ============================================================
# data_loader.py
# 負責：讀 Excel → 檢查格式 → 合併 → 整理落點 → 寬資料轉長資料
# （原本 Colab 第 2~12 格的功能）
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from core.config import (
    ID_COLUMN,
    FEATURE_COLUMNS,
    SHOT_COLUMNS,
    IMPORT_REQUIRED_COLUMNS,
    PARAMETER_LIMITS,
    ALL_LABELS,
    clean_landing_point,
)


# ------------------------------------------------------------
# 步驟 1：讀取所有 Excel，找出格式正確的工作表並合併
# ------------------------------------------------------------

def load_excel_files(excel_paths):
    """
    讀取多個 Excel 檔案。
    - 檔名、工作表名稱都不限制
    - 只要工作表有「必要欄位」就會被匯入
    - 回傳 (合併後的 df, 匯入報告 list)
    """

    all_dataframes = []   # 存放每個合格工作表的資料
    import_records = []   # 存放匯入報告（成功或失敗的原因）

    for excel_path in excel_paths:

        filename = Path(excel_path).name   # 只取檔名，不要整串路徑

        # 使用 with 確保釋放檔案控制代碼，讓 Windows 能清理暫存檔。
        try:
            with pd.ExcelFile(excel_path) as excel_file:
                # ---------- 逐一檢查每個工作表 ----------
                for sheet_name in excel_file.sheet_names:

                    try:
                        temp_df = excel_file.parse(sheet_name=sheet_name)

                        # 欄位名稱去掉前後空白
                        temp_df.columns = temp_df.columns.astype(str).str.strip()

                        # 工作表完全空白
                        if temp_df.empty:
                            import_records.append({
                                "檔案": filename,
                                "工作表": sheet_name,
                                "狀態": " 空白工作表",
                                "資料列數": 0,
                                "錯誤內容": "工作表沒有資料",
                            })
                            continue

                        # 找出缺少的必要欄位
                        missing_columns = []
                        for col in IMPORT_REQUIRED_COLUMNS:
                            if col not in temp_df.columns:
                                missing_columns.append(col)

                        if len(missing_columns) > 0:
                            import_records.append({
                                "檔案": filename,
                                "工作表": sheet_name,
                                "狀態": " 欄位不完整",
                                "資料列數": len(temp_df),
                                "錯誤內容": "缺少欄位：" + ", ".join(missing_columns),
                            })
                            continue

                        # 移除「整列都空白」的資料列
                        temp_df = temp_df.dropna(how="all").copy()

                        if temp_df.empty:
                            import_records.append({
                                "檔案": filename,
                                "工作表": sheet_name,
                                "狀態": " 無實際資料",
                                "資料列數": 0,
                                "錯誤內容": "欄位存在，但資料列為空",
                            })
                            continue

                        # 保留原本的測試編號
                        temp_df["原始測試編號"] = temp_df[ID_COLUMN]

                        # 記錄資料來源
                        temp_df["資料來源檔案"] = filename
                        temp_df["資料來源工作表"] = str(sheet_name)

                        # 建立「全域唯一」的測試編號：檔名|工作表|原始編號
                        # 因為不同 Excel 可能都有編號 1、2、3...，不加上檔名會被當成同一組
                        temp_df[ID_COLUMN] = (
                            filename
                            + "|"
                            + str(sheet_name)
                            + "|"
                            + temp_df["原始測試編號"].astype(str)
                        )

                        all_dataframes.append(temp_df)

                        import_records.append({
                            "檔案": filename,
                            "工作表": sheet_name,
                            "狀態": " 成功匯入",
                            "資料列數": len(temp_df),
                            "錯誤內容": "",
                        })

                    except Exception as error:
                        import_records.append({
                            "檔案": filename,
                            "工作表": sheet_name,
                            "狀態": " 讀取錯誤",
                            "資料列數": 0,
                            "錯誤內容": str(error),
                        })
        except Exception as error:
            import_records.append({
                "檔案": filename,
                "工作表": None,
                "狀態": " 檔案錯誤",
                "資料列數": 0,
                "錯誤內容": str(error),
            })

    # ---------- 沒有任何合格資料 → 停止 ----------
    if len(all_dataframes) == 0:
        raise ValueError(
            "所有上傳檔案都沒有找到可用的模型資料。\n"
            "匯入報告：\n"
            + "\n".join(
                f"  {r['檔案']} / {r['工作表']} : {r['狀態']} {r['錯誤內容']}"
                for r in import_records
            )
        )

    # 合併所有合格資料
    df = pd.concat(all_dataframes, ignore_index=True)

    return df, import_records


# ------------------------------------------------------------
# 步驟 2：檢查四個參數是否合法（整數、沒空白、在範圍內）
# ------------------------------------------------------------

def validate_parameters(df):
    """
    檢查四個機器參數。有問題就直接丟出錯誤（ValueError）。
    通過後，會把四欄都轉成整數（int）。
    """

    for col, (min_value, max_value) in PARAMETER_LIMITS.items():

        # 轉成數字；轉不了的會變成空值 NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # 空值數量
        missing_count = int(df[col].isna().sum())

        # 超出範圍的數量
        out_of_range = int(((df[col] < min_value) | (df[col] > max_value)).sum())

        # 非整數的數量（例如 3.5）
        non_integer_count = int((~df[col].isna() & (df[col] % 1 != 0)).sum())

        if missing_count > 0:
            raise ValueError(f"{col} 有 {missing_count} 筆空白或不是數字的資料")

        if out_of_range > 0:
            raise ValueError(
                f"{col} 有 {out_of_range} 筆超出設備範圍 {min_value}~{max_value}"
            )

        if non_integer_count > 0:
            raise ValueError(f"{col} 有 {non_integer_count} 筆非整數資料")

        df[col] = df[col].astype(int)

    return df


# ------------------------------------------------------------
# 步驟 3：整理落點文字 + 寬資料轉長資料
# ------------------------------------------------------------

def make_long_data(df):
    """
    把「一列 = 一組參數 + 5 個落點」，拆成「一列 = 一組參數 + 1 個落點」。
    例如 1 組參數 × 5 球 → 5 列。
    """

    # 整理 5 個落點欄位的文字格式
    for col in SHOT_COLUMNS:
        raw = df[col]
        cleaned = raw.apply(clean_landing_point)
        nonblank = raw.notna() & raw.astype(str).str.strip().ne("")
        unknown = nonblank & cleaned.isna()
        if unknown.any():
            values = raw.loc[unknown].astype(str).unique()[:5]
            raise ValueError(f"{col} 發現未知落點標籤：" + ", ".join(values))
        df[col] = cleaned

    # 寬 → 長
    long_df = df.melt(
        id_vars=[ID_COLUMN] + FEATURE_COLUMNS,
        value_vars=SHOT_COLUMNS,
        var_name="發球次數",
        value_name="落點",
    )

    # 沒有落點結果的球不拿來訓練
    blank_count = int(long_df["落點"].isna().sum())
    long_df = long_df.dropna(subset=["落點"]).copy()

    # 檢查有沒有不在 29 類裡的奇怪標籤
    actual_labels = long_df["落點"].unique().tolist()
    unknown_labels = [label for label in actual_labels if label not in ALL_LABELS]

    if len(unknown_labels) > 0:
        raise ValueError("發現未知落點標籤：" + ", ".join(unknown_labels))

    return long_df, blank_count


# ------------------------------------------------------------
# 步驟 4：把以上全部串起來，輸出訓練需要的 X / y / groups
# ------------------------------------------------------------

def prepare_training_data(excel_paths):
    """
    從 Excel 檔案清單，一路做到可以丟進模型訓練的資料。

    回傳一個字典，內容：
      X                 四個參數（DataFrame）
      y                 落點答案（Series）
      groups            分組依據：「四參數組合」（Series）
      tested_parameters 已測過的四參數組合（set）
      import_report     匯入報告（list）
      blank_count       被丟掉的空白落點數量
      class_counts      各落點球數（Series）
    """

    # 1. 讀取並合併
    df, import_report = load_excel_files(excel_paths)

    # 2. 檢查四個參數
    df = validate_parameters(df)

    # 3. 記錄「已測過的參數組合」（後面反推的探索模式會用到）
    tested_parameters = set(
        zip(
            df["仰角角度"],
            df["左右位置"],
            df["旋球角度"],
            df["出球強度"],
        )
    )

    # 4. 整理落點 + 轉長資料
    long_df, blank_count = make_long_data(df)

    if len(long_df) == 0:
        raise ValueError("沒有任何有效的落點資料可以訓練")

    # 5. 建立 X / y
    X = long_df[FEATURE_COLUMNS].copy()
    y = long_df["落點"].copy()

    # 6. 建立 groups（分組依據）
    #
    # 為什麼用「四參數組合」而不是「測試編號」？
    #   同一組四參數發 5 球，這 5 球一定要在同一個 Fold，否則會資料洩漏。
    #   但如果不同測試編號剛好用了「完全相同的四參數」（例如穩定性重複確認），
    #   用測試編號分組會讓它們被分到不同 Fold → 還是洩漏。
    #   用「四參數組合」分組，這兩種情況都能避免洩漏。
    groups = (
        long_df["仰角角度"].astype(str)
        + "_"
        + long_df["左右位置"].astype(str)
        + "_"
        + long_df["旋球角度"].astype(str)
        + "_"
        + long_df["出球強度"].astype(str)
    )

    # 7. 各落點球數
    class_counts = long_df["落點"].value_counts().reindex(ALL_LABELS, fill_value=0)

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "tested_parameters": tested_parameters,
        "import_report": import_report,
        "blank_count": blank_count,
        "class_counts": class_counts,
        "training_observations": long_df[[ID_COLUMN] + FEATURE_COLUMNS + ["落點"]].reset_index(drop=True),
    }
