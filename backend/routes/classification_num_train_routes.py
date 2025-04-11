from flask import Blueprint, request, jsonify
import threading
import time
from services.models.classification.Num.Train import train_model  # 确保导入了 train_model

# 用于存储任务状态的字典
task_status = {}

# 定义 train_task 函数
def train_task(task_id, model_choice, epochs, batch_size, learning_rate, eval_metric):
    """
    异步执行训练任务
    """
    result = train_model(model_choice, epochs, batch_size, learning_rate, eval_metric, task_id)
    task_status[task_id] = {
        'status': 'completed',
        'result': result,
    }

# 创建蓝图
classification_num_bp = Blueprint('classification_num_bp', __name__)

@classification_num_bp.route('/num', methods=['POST'])
def start_training():
    data = request.get_json(silent=True) or {}
    if 'model_choice' not in data or data.get('model_choice') is None:
        return jsonify({"message": "缺失参数：model_choice"}), 400
    
    task_id = str(time.time())  # 使用时间戳生成任务ID
    task_status[task_id] = {'status': 'started', 'training_logs': []}

    # 启动后台线程进行训练
    threading.Thread(target=train_task, args=(task_id, data['model_choice'], data['epochs'], data['batch_size'], data['learning_rate'], data['eval_metric'])).start()

    return jsonify({'task_id': task_id})

@classification_num_bp.route('/status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    if task_id not in task_status:
        return jsonify({"message": "任务ID无效"}), 400
    
    status = task_status[task_id]
    if status['status'] == 'completed':
        return jsonify({
            'status': 'completed',
            'training_logs': status['result']['training_logs'],
            'final_accuracy': status['result']['final_metric']
        })
    
    return jsonify({'status': 'running', 'training_logs': status['training_logs']})
