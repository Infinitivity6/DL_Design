# routes/dataNormalization_routes.py
# 该文件的主要左右是响应前端对于数据预处理中数据标准化的请求

from flask import Blueprint, request, jsonify
from services.dataNormalization import normalize_all_datasets

data_normalization_bp = Blueprint('data_normalization_bp', __name__)

@data_normalization_bp.route('/normalize', methods=['POST'])
def normalize():
    """
    前端POST /api/dataPreprocess/normalize
    body: { "method": "zscore" 或 "minmax" }
    返回:
    {
      "train": {
        "message": "...未上传..." 或 "...无需标准化..." 或
        "status":"ok",
        "num_processed": X,  # 标准化了多少数值列
        "skipped": Y         # 跳过多少列(非数值列/无差异列等)
      },
      "validation": {...},
      "test": {...}
    }
    """
    data = request.json
    method = data.get('method', None)  # "zscore" or "minmax"
    result = normalize_all_datasets(method)
    return jsonify(result)
