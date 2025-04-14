# services/dataUploadClassificationText.py
# 该文件的主要作用是执行文字分类数据上传的业务逻辑


import os
import time
from werkzeug.utils import secure_filename

# 定义允许上传的文件扩展名
ALLOWED_EXTENSIONS = {'txt', 'csv', 'json'}

def allowed_file(filename):
    """
    检查文件扩展名是否在允许范围内
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_file_upload(file, dataset_type):
    """
    处理文本分类任务的文件上传
    参数：
      file: 文件对象
      dataset_type: 文件类型，应该为 'train'、'validation' 或 'test'
    返回：
      保存的完整文件路径
    """
    # 定义上传基础目录，相对于 backend 文件夹
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads", "classification", "Text")
    base_dir = os.path.normpath(base_dir)
    
    # 定义 dataset_type 到子目录的映射（验证集存到 "val"）
    type_to_subdir = {
        'train': 'train',
        'validation': 'val',
        'test': 'test'
    }
    
    if dataset_type not in type_to_subdir:
        raise ValueError("无效的文件类型，应为 'train'、'validation' 或 'test'")
    
    # 构造目标子目录
    sub_dir = os.path.join(base_dir, type_to_subdir[dataset_type])
    os.makedirs(sub_dir, exist_ok=True)
    
    # 确保文件名安全并添加时间戳以避免重复
    filename = secure_filename(file.filename)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    filename = f"{timestamp}_{filename}"
    
    # 检查文件扩展名是否合法
    if not allowed_file(filename):
        raise ValueError("文件类型不允许上传")
    
    # 构造最终保存路径
    save_path = os.path.join(sub_dir, filename)
    
    # 保存文件
    file.save(save_path)
    
    return save_path
