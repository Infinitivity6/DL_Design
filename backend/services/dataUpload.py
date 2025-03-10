# services/data_service.py

import os
from flask import current_app

def save_file(file_obj, dataset_type=None):
    """
    接收一个 file_obj (FileStorage)，以及可选的 dataset_type
    在后端的 UPLOAD_FOLDER 中保存文件，并返回最终保存的文件名
    """
    filename = file_obj.filename
    if dataset_type:
        filename = f"{dataset_type}_{filename}"

    save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file_obj.save(save_path)
    return filename
