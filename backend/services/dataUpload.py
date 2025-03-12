# services/dataUpload.py

import os
from flask import current_app

def save_file(file_obj, dataset_type=None):
    """
    接收一个 file_obj (FileStorage) 和可选的 dataset_type ('train', 'validation', 'test')
    如果 dataset_type 在 [train, validation, test]，则存成对应的 train.csv / validation.csv / test.csv
    否则就用原文件名。
    返回最终保存的文件名。
    """

    if dataset_type in ['train', 'validation', 'test']:
        # 如果用户上传训练集，就强制命名为 train.csv，验证集 -> validation.csv，测试集 -> test.csv
        filename = f"{dataset_type}.csv"
    else:
        # 如果 dataset_type 不在 [train, validation, test]，就用原文件名
        filename = file_obj.filename

    save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file_obj.save(save_path)
    return filename
