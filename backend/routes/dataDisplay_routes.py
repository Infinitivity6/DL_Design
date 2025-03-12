# routes/dataDisplay_routes.py
# 该文件的主要作用是响应前端对于数据展示的请求

from flask import Blueprint, request, jsonify
from services.dataDisplay import get_dataset_info

data_display_bp = Blueprint('data_display_bp', __name__)

@data_display_bp.route('/dataset', methods=['GET'])
def dataset_info():
    """
    前端通过 /dataDisplay/dataset?type=train/validation/test/all 来获取数据集信息
    """
    dataset_type = request.args.get('type', 'train')
    result = get_dataset_info(dataset_type)

    if 'error' in result:
        return jsonify({"message": result['error']}), 400

    return jsonify(result)
