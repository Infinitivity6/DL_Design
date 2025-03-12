# services/dataNormalization.py
import os
import pandas as pd
from flask import current_app

def normalize_all_datasets(method):
    """
    对 train/validation/test 三个数据集:
    1) 若未上传 => {"message":"xx 数据集未上传"}
    2) 若全部非数值列 => {"message":"xx 数据集无需标准化"}
    3) 否则 => 对数值列执行 zscore 或 minmax
    返回:
    {
      "train": {...},
      "validation": {...},
      "test": {...}
    }
    """
    results = {}
    for ds in ['train','validation','test']:
        results[ds] = normalize_single_dataset(ds, method)
    return results

def normalize_single_dataset(ds, method):
    preproc_folder = current_app.config['PREPROCESSED_FOLDER']
    upload_folder = current_app.config['UPLOAD_FOLDER']

    preproc_path = os.path.join(preproc_folder, f"{ds}.csv")
    upload_path = os.path.join(upload_folder, f"{ds}.csv")

    if not os.path.exists(preproc_path) and not os.path.exists(upload_path):
        return {"message": f"{ds} 数据集未上传"}

    # 读最新版本
    src_path = preproc_path if os.path.exists(preproc_path) else upload_path
    try:
        df = pd.read_csv(src_path)
    except:
        return {"message": f"读取 {ds} 数据集失败"}

    # 假设最后一列是标签, 我们只对前面列(或非标签列)做标准化
    # 也可根据需求改
    label_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    if len(numeric_cols)==0:
        return {"message": f"{ds} 数据集无需标准化 (无数值列)"}

    # 执行标准化
    num_processed = 0
    skipped = 0
    for col in numeric_cols:
        series = df[col]
        if method=='zscore':
            # Z-score
            mean_val = series.mean()
            std_val = series.std()
            if std_val < 1e-8:
                # 若标准差几乎为0 => 跳过
                skipped += 1
                continue
            df[col] = (series - mean_val) / std_val
            num_processed += 1
        elif method=='minmax':
            # Min-Max
            min_val = series.min()
            max_val = series.max()
            if abs(max_val - min_val) < 1e-8:
                skipped += 1
                continue
            df[col] = (series - min_val) / (max_val - min_val)
            num_processed += 1
        else:
            return {"message": f"未知标准化方式: {method}"}

    if num_processed==0:
        return {"message": f"{ds} 数据集无需标准化 (数值列={len(numeric_cols)}, 但全部跳过)"}

    # 写回 preProcess/ds.csv
    df.to_csv(preproc_path, index=False)

    return {
      "status":"ok",
      "num_processed": num_processed,
      "skipped": skipped
    }
