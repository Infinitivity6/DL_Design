# services/dataDisplay.py
# 该文件的主要作用是执行数据展示的操作，并返回给后端业务逻辑层，并由其将结果传送至前端渲染

import os
import pandas as pd
from flask import current_app

def get_dataset_info(dataset_type=None):
    """
    根据 dataset_type ('train', 'validation', 'test' 或 'all')
    读取对应的 CSV 文件，返回数据概览和统计信息。
    如果 dataset_type = 'all', 则一次性返回所有已上传的数据集信息（示例中可合并或分别返回）。
    """
    upload_folder = current_app.config['UPLOAD_FOLDER']

    # 如果您有更复杂的命名规则，比如 userID_前缀，也可在此处理
    # 这里假设只存 train.csv / validation.csv / test.csv
    def file_path_for(dt):
        return os.path.join(upload_folder, f"{dt}.csv")

    # 如果 dataset_type = 'all'，收集多个数据集的信息
    if dataset_type == 'all':
        result = {}
        for dt in ['train', 'validation', 'test']:
            fp = file_path_for(dt)
            if os.path.exists(fp):
                result[dt] = _analyze_csv(fp)
            else:
                result[dt] = {"message": f"{dt} 数据集尚未上传"}
        return result

    # 否则就只读取某个数据集
    if dataset_type not in ['train', 'validation', 'test']:
        return {"error": f"未知的数据集类型: {dataset_type}"}

    fp = file_path_for(dataset_type)
    if not os.path.exists(fp):
        return {"error": f"尚未上传 {dataset_type} 数据集"}
    
    return _analyze_csv(fp)


def _analyze_csv(file_path):
    """
    读取 CSV 并返回：
    1. 前10行数据
    2. 基本信息（行数、列数、每列类型、缺失值、重复值）
    3. 最后一列标签分布
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"读取 CSV 失败: {str(e)}"}

    # 前10行
    preview_rows = df.head(10).values.tolist()
    preview_cols = list(df.columns)

    # 行数、列数
    row_count, col_count = df.shape

    # 每列类型
    dtypes_info = df.dtypes.apply(lambda x: str(x)).to_dict()

    # 缺失值统计
    missing_info = df.isnull().sum().to_dict()

    # 重复值统计
    duplicate_count = df.duplicated().sum()

    # 最后一列标签分布（若至少一列）
    label_distribution = {}
    if col_count > 0:
        last_col = df.columns[-1]
        label_distribution = df[last_col].value_counts().to_dict()

    # 组装返回
    return {
        "preview": {
            "columns": preview_cols,
            "rows": preview_rows
        },
        "info": {
            "row_count": row_count,
            "col_count": col_count,
            "dtypes": dtypes_info,
            "missing": missing_info,
            "duplicate_count": int(duplicate_count)
        },
        "label_distribution": label_distribution
    }
