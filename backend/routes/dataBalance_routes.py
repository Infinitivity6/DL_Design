# routes/dataBalance_routes.py
# 该文件的主要左右是响应前端对于数据预处理中数据平衡的请求


from flask import Blueprint, request, jsonify
from services.dataBalance import balance_all_datasets

data_balance_bp = Blueprint('data_balance_bp', __name__)

@data_balance_bp.route('/balance', methods=['POST'])
def balance():
    """
    前端POST /api/dataPreprocess/balance
    body: { "method": "undersampling" 或 "oversampling" }
    返回:
    {
      "train": {
        "message": "...未上传..." 或 "...无需平衡..." 或
        "status":"ok",
        "removed": X,
        "added": Y,
        "before": old_count,
        "after": new_count
      },
      "validation": {...},
      "test": {...}
    }
    """
    data = request.json
    method = data.get('method', None)  # "undersampling" or "oversampling"
    result = balance_all_datasets(method)
    return jsonify(result)
