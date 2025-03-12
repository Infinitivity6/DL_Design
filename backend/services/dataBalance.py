# services/dataBalance.py
# 该文件的主要作用是执行数据预处理数据平衡的操作，并返回给后端业务逻辑层，并由其将结果传送至前端渲染


import os
import pandas as pd
from flask import current_app

def balance_all_datasets(method):
    """
    对 train/validation/test 三个数据集:
    1) 若未上传 => {"message":"xx 数据集未上传"}
    2) 若已上传 => 先判断是否足够平衡(自定义阈值)
       - 如果已较为平衡 => {"message":"无需数据平衡"}
       - 否则 => 根据method('undersampling'/'oversampling')做操作
                 并写回 preProcess/ds.csv
       - 返回 {status:"ok", before:..., after:..., removed:..., added:...}
    最终组装成:
    {
      "train": {...},
      "validation": {...},
      "test": {...}
    }
    """
    results = {}
    datasets = ['train','validation','test']
    for ds in datasets:
        result = balance_single_dataset(ds, method)
        results[ds] = result
    return results

def balance_single_dataset(ds, method):
    preproc_folder = current_app.config['PREPROCESSED_FOLDER']
    upload_folder = current_app.config['UPLOAD_FOLDER']

    preproc_path = os.path.join(preproc_folder, f"{ds}.csv")
    upload_path = os.path.join(upload_folder, f"{ds}.csv")

    if not os.path.exists(preproc_path) and not os.path.exists(upload_path):
        return {"message": f"{ds} 数据集未上传"}

    # 读取最新版本
    src_path = preproc_path if os.path.exists(preproc_path) else upload_path
    try:
        df = pd.read_csv(src_path)
    except:
        return {"message": f"读取 {ds} 数据集失败"}

    # 判断是否平衡：这里简单用 "最大类别 / 最小类别 < 2" 作为平衡阈值
    # (实际可以更精细)
    label_col = df.columns[-1]  # 假设最后一列是标签
    label_counts = df[label_col].value_counts()
    max_count = label_counts.max()
    min_count = label_counts.min()
    if min_count == 0:
        # 说明有类别完全没有样本, 这里也可做其它逻辑
        return {"message": f"{ds} 存在某些类别为0, 无法平衡"}

    ratio = max_count / min_count
    if ratio < 2:
        return {"message": f"{ds} 数据集无需数据平衡操作 (类别比率约为 {ratio:.2f})"}

    before_len = len(df)

    # 开始平衡
    if method == 'undersampling':
        new_df, removed = undersample(df, label_col)
        after_len = len(new_df)
        added = 0
        new_df.to_csv(preproc_path, index=False)
        return {
          "status":"ok",
          "before": before_len,
          "after": after_len,
          "removed": removed,
          "added": added
        }
    elif method == 'oversampling':
        new_df, added = oversample(df, label_col)
        after_len = len(new_df)
        removed = 0
        new_df.to_csv(preproc_path, index=False)
        return {
          "status":"ok",
          "before": before_len,
          "after": after_len,
          "removed": removed,
          "added": added
        }
    else:
        return {"message": f"未知的平衡方式: {method}"}


def undersample(df, label_col):
    """
    简单随机欠采样:
    找出最少类的样本数 min_count
    对所有类别都随机采样 min_count
    """
    counts = df[label_col].value_counts()
    min_count = counts.min()
    frames = []
    for cls in counts.index:
        cls_df = df[df[label_col] == cls]
        # 随机采样 min_count
        sampled = cls_df.sample(n=min_count, random_state=42)
        frames.append(sampled)
    new_df = pd.concat(frames, ignore_index=True)
    removed = len(df) - len(new_df)
    return new_df, removed

def oversample(df, label_col):
    """
    简单随机过采样:
    找出最多类的样本数 max_count
    对所有类别都随机重复采样 max_count
    """
    counts = df[label_col].value_counts()
    max_count = counts.max()
    frames = []
    for cls in counts.index:
        cls_df = df[df[label_col] == cls]
        size = len(cls_df)
        if size < max_count:
            # 随机重复
            repeat_df = cls_df.sample(n=(max_count - size), replace=True, random_state=42)
            new_cls_df = pd.concat([cls_df, repeat_df], ignore_index=True)
            frames.append(new_cls_df)
        else:
            # size==max_count
            frames.append(cls_df)
    new_df = pd.concat(frames, ignore_index=True)
    added = len(new_df) - len(df)
    return new_df, added
