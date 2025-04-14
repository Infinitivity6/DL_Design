# services/dataUploadClassificationImage.py
# 该文件的主要作用是执行图像分类数据上传的业务逻辑

import os
import time
import uuid
from flask import current_app
from werkzeug.utils import secure_filename

# 允许的图像文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    """
    检查文件是否是允许的图像文件类型
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_dir(category_name):
    """
    确保上传目录存在，如果不存在则创建
    """
    # 使用相对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    upload_folder = os.path.join(base_dir, 'uploads', 'classification', 'Image', category_name)
    
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    return upload_folder

def save_images(images_list, category_name):
    """
    保存上传的图像文件到对应类别目录
    
    Args:
        images_list: 上传的图像文件列表
        category_name: 类别名称
        
    Returns:
        dict: 包含保存结果的字典
    """
    # 类别名称安全处理
    category_name = secure_filename(category_name)
    
    # 获取上传目录
    try:
        upload_folder = ensure_upload_dir(category_name)
    except Exception as e:
        return {"error": f"创建上传目录失败: {str(e)}"}
    
    # 保存所有上传的图片
    saved_files = []
    for image in images_list:
        if image.filename == '' or not allowed_file(image.filename):
            continue
        
        try:
            # 生成安全的文件名，避免文件名冲突
            filename = secure_filename(image.filename)
            # 添加时间戳前缀，确保唯一性
            unique_filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{filename}"
            file_path = os.path.join(upload_folder, unique_filename)
            
            # 保存文件
            image.save(file_path)
            saved_files.append({
                "original_name": filename,
                "saved_name": unique_filename,
                "path": os.path.join('classification', 'Image', category_name, unique_filename)
            })
        except Exception as e:
            return {"error": f"保存图片 '{image.filename}' 失败: {str(e)}"}
    
    return {
        "saved_files": saved_files
    }

def get_all_categories():
    """
    获取所有已上传的类别和图片信息
    
    Returns:
        list: 包含所有类别信息的列表
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(base_dir, 'uploads', 'classification', 'Image')
    
    if not os.path.exists(image_dir):
        return []
    
    categories = []
    for category_name in os.listdir(image_dir):
        category_path = os.path.join(image_dir, category_name)
        if os.path.isdir(category_path):
            # 获取该类别下的图片文件
            image_files = [f for f in os.listdir(category_path) 
                          if os.path.isfile(os.path.join(category_path, f)) and 
                          allowed_file(f)]
            
            # 为每个图片创建URL
            images_info = []
            for img_file in image_files[:10]:  # 只返回前10张图片
                images_info.append({
                    "name": img_file,
                    "path": os.path.join('classification', 'Image', category_name, img_file)
                })
            
            categories.append({
                "name": category_name,
                "image_count": len(image_files),
                "images": images_info
            })
    
    return categories

def delete_category(category_name):
    """
    删除指定类别及其中的所有图片
    
    Args:
        category_name: 要删除的类别名称
        
    Returns:
        dict: 包含删除结果的字典
    """
    # 类别名称安全处理
    category_name = secure_filename(category_name)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    category_path = os.path.join(base_dir, 'uploads', 'classification', 'Image', category_name)
    
    if not os.path.exists(category_path):
        return {"error": f"类别 '{category_name}' 不存在", "not_found": True}
    
    try:
        # 删除该类别下的所有文件
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # 删除类别目录
        os.rmdir(category_path)
        return {"message": f"类别 '{category_name}' 已成功删除"}
    except Exception as e:
        return {"error": f"删除类别失败: {str(e)}"}

def delete_image(category_name, image_name):
    """
    删除指定类别中的指定图片
    
    Args:
        category_name: 类别名称
        image_name: 图片文件名
        
    Returns:
        dict: 包含删除结果的字典
    """
    # 安全处理输入
    category_name = secure_filename(category_name)
    image_name = secure_filename(image_name)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, 'uploads', 'classification', 'Image', category_name, image_name)
    
    if not os.path.exists(image_path):
        return {"error": f"图片 '{image_name}' 不存在", "not_found": True}
    
    try:
        os.remove(image_path)
        return {"message": f"图片 '{image_name}' 已成功删除"}
    except Exception as e:
        return {"error": f"删除图片失败: {str(e)}"}