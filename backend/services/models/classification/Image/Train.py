# services/models/classification/Image/Train.py

import os
from services.models.classification.Image.ResNet18 import train_resnet18
from services.models.classification.Image.VGG16 import train_vgg16
from services.models.classification.Image.MobileNetV2 import train_mobilenetv2
from services.task_manager import TaskManager

def train_model(model_choice, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager, categories):
    """
    根据用户选择调用对应的训练函数。
    直接使用uploads/classification/Image目录下的子文件夹作为类别
    """
    # 获取训练数据根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # 修正为正确的上传目录路径
    data_dir = os.path.join(base_dir, "..", "uploads", "classification", "Image")
    data_dir = os.path.normpath(data_dir)  # 规范化路径
    
    print(f"数据目录: {data_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return {
            "status": "failed",
            "message": "数据目录不存在，请确保上传了图像文件",
            "training_logs": []
        }
    
    # 获取目录中的所有子文件夹（作为类别）
    try:
        actual_categories = [f for f in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, f))]
        print(f"检测到的类别: {actual_categories}")
        
        if len(actual_categories) < 2:
            print(f"类别数量不足: {len(actual_categories)}")
            return {
                "status": "failed",
                "message": f"检测到的类别数量不足，至少需要2个类别，当前只有{len(actual_categories)}个",
                "training_logs": []
            }
    except Exception as e:
        print(f"获取类别列表失败: {str(e)}")
        return {
            "status": "failed",
            "message": f"获取类别列表失败: {str(e)}",
            "training_logs": []
        }
    
    try:
        # 根据选择的模型调用对应的训练函数
        if model_choice.lower() == 'resnet18':
            print("使用ResNet18模型")
            result = train_resnet18(data_dir, actual_categories, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
        elif model_choice.lower() == 'vgg16':
            print("使用VGG16模型")
            result = train_vgg16(data_dir, actual_categories, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
        elif model_choice.lower() == 'mobilenetv2':
            print("使用MobileNetV2模型")
            result = train_mobilenetv2(data_dir, actual_categories, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
        else:
            print(f"未知模型类型: {model_choice}")
            result = {
                "status": "failed",
                "message": f"未知模型类型: {model_choice}",
                "training_logs": [],
                "categories": actual_categories
            }
        
        # 确保结果包含所有必要字段
        if not isinstance(result, dict):
            print(f"训练函数返回非字典结果: {result}")
            result = {
                "status": "failed",
                "message": "训练函数返回无效结果",
                "training_logs": [],
                "categories": actual_categories
            }
        elif "training_logs" not in result:
            print(f"训练结果中缺少training_logs字段，添加空列表")
            result["training_logs"] = []
        
        # 确保结果中包含实际使用的类别
        result["categories"] = actual_categories
        
        return result
    except Exception as e:
        print(f"训练异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "message": f"训练异常: {str(e)}",
            "training_logs": [],
            "categories": actual_categories
        }