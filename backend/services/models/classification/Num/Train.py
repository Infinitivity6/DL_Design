# services/models/classification/Num/Train.py

import os
from services.models.classification.Num.MLP import train_mlp  # 调用 MLP 训练代码
from services.models.classification.Num.TabNet import train_tabnet
from services.models.classification.Num.CNN1D import train_cnn1d
from services.task_manager import TaskManager  # 引入共享的 TaskManager 实例


def train_task(task_id, model_choice, epochs, batch_size, learning_rate, eval_metric, task_manager):
    """
    异步执行训练任务
    """
    result = train_model(model_choice, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
    
    # 更新任务状态
    task_manager.update_task(task_id, {
        'status': 'completed',  # 任务完成，设置状态为 completed
        'result': result,
    })



def train_model(model_choice, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager):
    """
    根据用户选择调用对应的训练函数。
    训练数据统一存放在 uploads 文件夹下的 train.csv
    """
    # 获取训练文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, "../../../../uploads/train.csv")

    # 调用 MLP 训练
    if model_choice.lower() == 'mlp':
        result = train_mlp(train_file, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
    elif model_choice.lower() == 'tabnet':
        result = train_tabnet(train_file, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
    elif model_choice.lower() == 'cnn1d':
        result = train_cnn1d(train_file, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
    else:
        result = {"message": "未知模型类型"}
    return result
