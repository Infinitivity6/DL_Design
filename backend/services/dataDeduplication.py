# services/dataPreprocess.py
# 该文件的主要作用是执行数据预处理去重的操作，并返回给后端业务逻辑层，并由其将结果传送至前端渲染

import os
import pandas as pd
from flask import current_app

def remove_duplicates_from_all():
    """
    1) 优先从 preProcess/<dataset>.csv 读取，如果没有则从 uploads/<dataset>.csv 读取
    2) 对读到的数据进行去重
    3) 将结果写回 preProcess/<dataset>.csv
    4) 若三种文件都不存在, 返回 error
    """

    upload_folder = current_app.config['UPLOAD_FOLDER']
    preprocessed_folder = current_app.config['PREPROCESSED_FOLDER']

    datasets = ['train', 'validation', 'test']
    results = {}
    file_found = False

    for ds in datasets:
        # 构建“最新版本”路径和“原始”路径
        preproc_path = os.path.join(preprocessed_folder, f"{ds}.csv")
        upload_path = os.path.join(upload_folder, f"{ds}.csv")

        # 决定 src_path (要读的文件)
        if os.path.exists(preproc_path):
            # 已有最新版本 -> 在此基础上去重
            src_path = preproc_path
        elif os.path.exists(upload_path):
            # 没有最新版本, 但有原始文件 -> 用原始文件
            src_path = upload_path
        else:
            # 都没有 -> 说明此 ds 尚未上传
            results[ds] = {"message": f"{ds} 数据集尚未上传"}
            continue  # 跳过

        file_found = True
        try:
            df = pd.read_csv(src_path)
            before_count = len(df)
            df.drop_duplicates(inplace=True)
            after_count = len(df)
            removed_count = before_count - after_count

            # 把结果写回 preProcess/<ds>.csv (覆盖)
            df.to_csv(preproc_path, index=False)

            results[ds] = {
                "before": before_count,
                "after": after_count,
                "removed": removed_count
            }
        except Exception as e:
            results[ds] = {"error": f"去重时出错: {str(e)}"}

    if not file_found:
        # 三个数据集都没有任何文件 -> 提示
        return {"error": "请上传至少一个数据集后再进行去重"}

    return results
