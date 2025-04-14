# routes/classification_image_train_routes.py
from flask import Blueprint, request, jsonify
import os
import time
from services.dataUploadClassificationText import handle_file_upload

# 创建蓝图，设置 URL 前缀为 /api/classification/text/upload
classification_text_upload_bp = Blueprint('classification_text_upload_bp', __name__)

@classification_text_upload_bp.route('', methods=['POST'])
def upload_text_file():
    """
    处理文本分类任务中上传的文件请求
    参数：
      - dataset_type: 文件类型，'train'、'validation' 或 'test'
      - file: 上传的文件对象
    返回：
      - JSON 格式的上传结果信息，包括保存的文件路径
    """
    # 检查是否提供了文件
    if 'file' not in request.files:
        return jsonify({"message": "未找到上传的文件"}), 400

    file = request.files['file']
    dataset_type = request.form.get('dataset_type', None)

    if file.filename == "":
        return jsonify({"message": "未选择文件"}), 400

    if not dataset_type:
        return jsonify({"message": "未指定文件类型（train/validation/test）"}), 400

    try:
        # 调用服务函数处理文件上传
        saved_path = handle_file_upload(file, dataset_type)
        return jsonify({"message": "文件上传成功", "file_path": saved_path})
    except Exception as e:
        return jsonify({"message": f"上传文件出错：{str(e)}"}), 500

