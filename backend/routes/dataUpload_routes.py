# routes/data_routes.py
# 该文件的主要作用是响应前端对于数据上传的请求

from flask import Blueprint, request, jsonify
from services.dataUpload import save_file

data_bp = Blueprint('data_bp', __name__)

@data_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    文件上传的路由
    """
    dataset_type = request.form.get('dataset_type')
    file = request.files.get('file')

    if not file:
        return jsonify({"message": "未上传文件"}), 400

    # 调用业务逻辑函数
    filename = save_file(file, dataset_type)
    return jsonify({"message": f"File '{filename}' 上传成功!"})
