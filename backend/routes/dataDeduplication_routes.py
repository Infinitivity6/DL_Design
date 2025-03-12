# routes/dataPreprocess_routes.py
# 该文件的主要左右是响应前端对于数据预处理中去重的请求

import os
from flask import Blueprint, jsonify, current_app, request
from services.dataDeduplication import remove_duplicates_from_all

data_deduplication_bp = Blueprint('data_deduplication_bp', __name__)

@data_deduplication_bp.route('/remove_duplicates', methods=['POST'])
def remove_duplicates():
    """
    接收前端请求，对所有已上传的数据集执行去重操作
    并将结果存到 preProcess/ 文件夹中
    返回一个 JSON，包含每个数据集去重前后行数和删除行数
    如果没有任何文件则提示“请上传至少一个数据集”
    """
    result = remove_duplicates_from_all()

    if result.get('error'):
        return jsonify({"message": result['error']}), 400
    else:
        # 成功, 返回一个结构, 比如:
        # {
        #   "train": {"before":100, "after":95, "removed":5},
        #   "validation": {...},
        #   "test": {...}
        # }
        return jsonify(result), 200
