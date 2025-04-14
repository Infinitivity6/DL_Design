# backend/services/models/classification/Text/Train.py
import os
import time
from services.models.classification.Text.Bert import train_bert
from services.models.classification.Text.LSTM import train_lstm
from services.models.classification.Text.Transformer import train_transformer

def train_model(model_choice, epochs, batch_size, learning_rate, eval_metric, language, task_id, task_manager):
    """
    根据用户选择的模型和语言调用对应的文本分类训练函数
    参数：
      model_choice: 模型选择 ('bert', 'lstm', 'transformer')
      epochs: 训练轮数
      batch_size: 批处理大小
      learning_rate: 学习率
      eval_metric: 评价指标
      language: 文本语言 ('zh' 或 'en')
      task_id: 当前任务的ID
      task_manager: 任务管理器实例，用于更新任务状态
    返回：
      dict：包含训练结果与日志的字典
    """
    # 获取训练、验证、测试数据的根目录（这里采用相对目录，使用 train 为必需，验证集与测试集可选）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 训练集目录
    train_dir = os.path.join(current_dir, "../../../../uploads/classification/Text/train")
    # 验证集目录（如果存在）
    val_dir = os.path.join(current_dir, "../../../../uploads/classification/Text/val")
    # 测试集目录（如果存在）
    test_dir = os.path.join(current_dir, "../../../../uploads/classification/Text/test")

    # 规范化路径
    train_dir = os.path.normpath(train_dir)
    val_dir = os.path.normpath(val_dir)
    test_dir = os.path.normpath(test_dir)

    # 根据模型选择调用对应的训练函数
    if model_choice.lower() == 'bert':
        result = train_bert(train_dir, val_dir, test_dir, language, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
    elif model_choice.lower() == 'lstm':
        result = train_lstm(train_dir, val_dir, test_dir, language, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
    elif model_choice.lower() == 'transformer':
        result = train_transformer(train_dir, val_dir, test_dir, language, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
    else:
        result = {
            "status": "failed",
            "message": f"未知模型类型：{model_choice}",
            "training_logs": []
        }
    return result
