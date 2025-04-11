# routes/classification_num_train_routes.py

from flask import Blueprint, request, jsonify
import threading
import time
from services.models.classification.Num.Train import train_model  # 确保导入了 train_model
from services.task_manager import TaskManager

# 创建任务管理类实例
task_manager = TaskManager()

# 定义 train_task 函数
def train_task(task_id, model_choice, epochs, batch_size, learning_rate, eval_metric):
    """
    异步执行训练任务
    """
    result = train_model(model_choice, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager)
    
    # 更新任务状态
    task_manager.update_task(task_id, {
        'status': 'completed',
        'result': result,
    })

# 创建蓝图
classification_num_bp = Blueprint('classification_num_bp', __name__)

@classification_num_bp.route('/num', methods=['POST'])
def start_training():
    data = request.get_json(silent=True) or {}
    if 'model_choice' not in data or data.get('model_choice') is None:
        return jsonify({"message": "缺失参数：model_choice"}), 400
    
    task_id = str(time.time())  # 使用时间戳生成任务ID
    
    # 初始化任务状态，并保存所有参数
    task_manager.add_task(task_id, {
        'status': 'started',
        'model_choice': data['model_choice'],
        'epochs': data['epochs'],
        'batch_size': data['batch_size'],
        'learning_rate': data['learning_rate'],
        'eval_metric': data['eval_metric'],
        'training_logs': []
    })

    # 启动后台线程进行训练
    threading.Thread(target=train_task, args=(task_id, data['model_choice'], data['epochs'], data['batch_size'], data['learning_rate'], data['eval_metric'])).start()

    return jsonify({'task_id': task_id})

@classification_num_bp.route('/status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    task_info = task_manager.get_task(task_id)
    if not task_info:
        return jsonify({"message": "任务ID无效"}), 400

    if task_info['status'] == 'completed':
        return jsonify({
            'status': 'completed',
            'training_logs': task_info['result']['training_logs'],
            'final_accuracy': task_info['result']['final_metric'],
            'model': task_info['result']['model'],
            'epochs': task_info['result']['epochs'],
            'batch_size': task_info['result']['batch_size'],
            'learning_rate': task_info['result']['learning_rate'],
            'eval_metric': task_info['result']['eval_metric']
        })

    return jsonify({
        'status': 'running',
        'training_logs': task_info['training_logs'],
        'model': task_info['model_choice'],
        'epochs': task_info['epochs'],
        'batch_size': task_info['batch_size'],
        'learning_rate': task_info['learning_rate'],
        'eval_metric': task_info['eval_metric']
    })


