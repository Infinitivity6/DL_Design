# backend/routes/classification_text_train_routes.py
from flask import Blueprint, request, jsonify
import threading
import time
from services.models.classification.Text.Train import train_model  # 从 Train 模块导入训练函数
from services.task_manager import TaskManager  # 导入全局共享的任务管理器实例

# 创建任务管理器实例（全局共享此实例）
task_manager = TaskManager()

# 定义后台训练任务函数
def train_task(task_id, model_choice, epochs, batch_size, learning_rate, eval_metric, language):
    """
    异步执行文本分类训练任务
    参数：
      task_id: 任务ID
      model_choice: 模型选择（如 'bert', 'lstm', 'transformer'）
      epochs: 训练轮数
      batch_size: 批处理大小
      learning_rate: 学习率
      eval_metric: 评价指标
      language: 文本语言（'zh' 或 'en'）
    """
    # 调用训练函数
    result = train_model(model_choice, epochs, batch_size, learning_rate, eval_metric, language, task_id, task_manager)
    # 更新任务状态为完成
    task_manager.update_task(task_id, {
        'status': 'completed',
        'result': result
    })

# 创建蓝图，并指定 URL 前缀为 /api/classification/text
classification_text_bp = Blueprint('classification_text_bp', __name__)

# 训练任务请求（POST 请求）
@classification_text_bp.route('/text', methods=['POST'])
def start_training():
    data = request.get_json(silent=True) or {}
    # 检查必要参数
    if 'model_choice' not in data or not data.get('model_choice'):
        return jsonify({"message": "缺失参数：model_choice"}), 400
    if 'language' not in data or not data.get('language'):
        return jsonify({"message": "缺失参数：language"}), 400
    # if 'train_file' not in data:
    #     return jsonify({"message": "缺失训练集上传信息"}), 400

    task_id = str(time.time())  # 使用时间戳生成任务ID

    # 初始化任务状态（保存所有参数，训练日志初始为空）
    task_manager.add_task(task_id, {
        'status': 'started',
        'model_choice': data['model_choice'],
        'language': data['language'],
        'epochs': data.get('epochs', 50),
        'batch_size': data.get('batch_size', 16),
        'learning_rate': data.get('learning_rate', 0.001),
        'eval_metric': data.get('eval_metric', 'accuracy'),
        'training_logs': []
    })

    # 启动后台线程进行训练
    threading.Thread(target=train_task, args=(
        task_id,
        data['model_choice'],
        data.get('epochs', 50),
        data.get('batch_size', 16),
        data.get('learning_rate', 0.001),
        data.get('eval_metric', 'accuracy'),
        data['language']
    )).start()

    return jsonify({'task_id': task_id})

# 状态查询路由（GET 请求），返回训练状态及最新日志
@classification_text_bp.route('/text/status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    task_info = task_manager.get_task(task_id)
    if not task_info:
        return jsonify({"message": "任务ID无效"}), 400

    if task_info.get('status') == 'completed' and 'result' in task_info:
        result = task_info['result']
        return jsonify({
            'status': 'completed',
            'training_logs': result.get('training_logs', []),
            'final_accuracy': result.get('final_metric', 0),
            'model': result.get('model', task_info.get('model_choice', '')),
            'language': result.get('language', task_info.get('language', '')),
            'epochs': result.get('epochs', task_info.get('epochs', 0)),
            'batch_size': result.get('batch_size', task_info.get('batch_size', 0)),
            'learning_rate': result.get('learning_rate', task_info.get('learning_rate', 0)),
            'eval_metric': result.get('eval_metric', task_info.get('eval_metric', ''))
        })
    else:
        # 任务仍在运行时返回已记录的训练日志
        return jsonify({
            'status': task_info.get('status', 'unknown'),
            'training_logs': task_info.get('training_logs', []),
            'model': task_info.get('model_choice', ''),
            'language': task_info.get('language', ''),
            'epochs': task_info.get('epochs', 0),
            'batch_size': task_info.get('batch_size', 0),
            'learning_rate': task_info.get('learning_rate', 0),
            'eval_metric': task_info.get('eval_metric', '')
        })
