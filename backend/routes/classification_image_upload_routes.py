# routes/classification_image_upload_routes.py
# 该文件的主要作用是响应前端对于图像分类数据上传的请求

from flask import Blueprint, request, jsonify
from services.dataUploadClassificationImage import save_images, get_all_categories, delete_category, delete_image

# 创建蓝图
image_upload_bp = Blueprint('image_upload_bp', __name__)

@image_upload_bp.route('/upload', methods=['POST'])
def upload_images():
    """
    接收上传的图像文件，并按类别保存
    """
    if 'category_name' not in request.form:
        return jsonify({"message": "缺少类别名称"}), 400
    
    category_name = request.form['category_name']
    
    if 'images' not in request.files:
        return jsonify({"message": "未找到图片文件"}), 400
    
    images = request.files.getlist('images')
    if len(images) == 0 or images[0].filename == '':
        return jsonify({"message": "未选择任何图片"}), 400
    
    # 调用业务逻辑函数保存图片
    result = save_images(images, category_name)
    
    if 'error' in result:
        return jsonify({"message": result['error']}), 400
    
    return jsonify({
        "message": f"成功上传 {len(result['saved_files'])} 张图片到类别 '{category_name}'",
        "category": category_name,
        "saved_files": result['saved_files']
    })

@image_upload_bp.route('/categories', methods=['GET'])
def get_categories():
    """
    获取所有已上传的类别和图片信息
    """
    categories = get_all_categories()
    return jsonify({"categories": categories})

@image_upload_bp.route('/category/<category_name>', methods=['DELETE'])
def remove_category(category_name):
    """
    删除指定类别及其中的所有图片
    """
    result = delete_category(category_name)
    
    if 'error' in result:
        return jsonify({"message": result['error']}), 400 if result.get('not_found', False) else 500
    
    return jsonify({"message": result['message']})

@image_upload_bp.route('/image', methods=['DELETE'])
def remove_image():
    """
    删除指定类别中的指定图片
    """
    data = request.get_json(silent=True) or {}
    
    if 'category_name' not in data or 'image_name' not in data:
        return jsonify({"message": "缺少类别名称或图片名称"}), 400
    
    result = delete_image(data['category_name'], data['image_name'])
    
    if 'error' in result:
        return jsonify({"message": result['error']}), 400 if result.get('not_found', False) else 500
    
    return jsonify({"message": result['message']})