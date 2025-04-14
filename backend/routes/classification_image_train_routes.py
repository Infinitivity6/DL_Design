# routes/classification_image_train_routes.py

from flask import Blueprint, request, jsonify
import threading
import time
from services.models.classification.Image.Train import train_model  # 确保导入了 train_model
from services.task_manager import TaskManager

# 创建任务管理类实例（如果已存在全局实例，可以导入共享的实例）
task_manager = TaskManager()

# 定义 train_task 函数
def train_task(task_id, model_choice, epochs, batch_size, learning_rate, eval_metric, categories):
    """
    异步执行训练任务
    """
    try:
        # 打印开始训练的信息
        print(f"开始训练任务: {task_id}, 模型: {model_choice}")
        
        # 调用训练函数
        result = train_model(model_choice, epochs, batch_size, learning_rate, eval_metric, task_id, task_manager, categories)
        
        # 确保结果包含必要的字段
        if not result:
            print(f"训练返回空结果，设置默认失败状态")
            result = {
                "status": "failed",
                "message": "训练函数返回空结果",
                "training_logs": [],
                "final_metric": 0,
                "model": model_choice,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "eval_metric": eval_metric,
                "categories": categories
            }
        elif "training_logs" not in result:
            print(f"训练结果中缺少training_logs字段，添加空列表")
            result["training_logs"] = []
        
        # 更新任务状态
        print(f"训练完成，更新任务状态: {result.get('status', 'unknown')}")
        task_manager.update_task(task_id, {
            'status': 'completed',
            'result': result,
        })
        
    except Exception as e:
        print(f"训练任务异常: {str(e)}")
        # 发生异常时，确保任务状态更新为失败，并包含必要的字段
        error_result = {
            "status": "failed",
            "message": f"训练过程中发生错误: {str(e)}",
            "training_logs": [],
            "final_metric": 0,
            "model": model_choice,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "eval_metric": eval_metric,
            "categories": categories
        }
        task_manager.update_task(task_id, {
            'status': 'failed',
            'error': str(e),
            'result': error_result
        })

# 创建蓝图
classification_image_bp = Blueprint('classification_image_bp', __name__)

@classification_image_bp.route('/image', methods=['POST'])
def start_training():
    try:
        data = request.get_json(silent=True) or {}
        print(f"收到训练请求: {data}")
        
        if 'model_choice' not in data or data.get('model_choice') is None:
            return jsonify({"message": "缺失参数：model_choice"}), 400
        
        # 创建任务ID
        task_id = str(time.time())
        
        # 初始化任务状态，保留前端传递的类别名称，但实际训练会使用后端检测到的文件夹名称
        task_manager.add_task(task_id, {
            'status': 'started',
            'model_choice': data['model_choice'],
            'epochs': data.get('epochs', 50),
            'batch_size': data.get('batch_size', 16),
            'learning_rate': data.get('learning_rate', 0.001),
            'eval_metric': data.get('eval_metric', 'accuracy'),
            'categories': data.get('categories', []),  # 保留前端传递的类别，但不实际使用
            'training_logs': []
        })
        
        print(f"任务 {task_id} 已添加到任务管理器")
        
        # 启动后台线程进行训练
        try:
            thread = threading.Thread(
                target=train_task, 
                args=(
                    task_id, 
                    data['model_choice'], 
                    data.get('epochs', 50), 
                    data.get('batch_size', 16), 
                    data.get('learning_rate', 0.001), 
                    data.get('eval_metric', 'accuracy'),
                    data.get('categories', [])  # 传递类别，但训练时会以实际文件夹为准
                )
            )
            thread.daemon = True  # 设置为守护线程，避免主程序退出时线程还在运行
            thread.start()
            print(f"训练线程已启动")
        except Exception as e:
            print(f"启动训练线程失败: {str(e)}")
            return jsonify({"message": f"启动训练线程失败: {str(e)}"}), 500

        return jsonify({'task_id': task_id})
    except Exception as e:
        print(f"处理训练请求时出错: {str(e)}")
        return jsonify({"message": f"处理请求出错: {str(e)}"}), 500

@classification_image_bp.route('/image/status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    try:
        task_info = task_manager.get_task(task_id)
        if not task_info:
            return jsonify({"message": "任务ID无效"}), 400

        print(f"获取任务状态: {task_id}, 当前状态: {task_info.get('status', 'unknown')}")

        if task_info.get('status') == 'completed' and 'result' in task_info:
            result = task_info['result']
            return jsonify({
                'status': 'completed',
                'training_logs': result.get('training_logs', []),
                'final_accuracy': result.get('final_metric', 0),
                'model': result.get('model', task_info.get('model_choice', '')),
                'epochs': result.get('epochs', task_info.get('epochs', 0)),
                'batch_size': result.get('batch_size', task_info.get('batch_size', 0)),
                'learning_rate': result.get('learning_rate', task_info.get('learning_rate', 0)),
                'eval_metric': result.get('eval_metric', task_info.get('eval_metric', '')),
                'categories': result.get('categories', task_info.get('categories', []))
            })
        elif task_info.get('status') == 'failed':
            result = task_info.get('result', {})
            return jsonify({
                'status': 'failed',
                'message': result.get('message', task_info.get('error', '未知错误')),
                'training_logs': result.get('training_logs', []),
                'model': task_info.get('model_choice', ''),
                'epochs': task_info.get('epochs', 0),
                'batch_size': task_info.get('batch_size', 0),
                'learning_rate': task_info.get('learning_rate', 0),
                'eval_metric': task_info.get('eval_metric', ''),
                'categories': result.get('categories', task_info.get('categories', []))
            })

        # 对于运行中的任务
        return jsonify({
            'status': task_info.get('status', 'unknown'),
            'training_logs': task_info.get('training_logs', []),
            'model': task_info.get('model_choice', ''),
            'epochs': task_info.get('epochs', 0),
            'batch_size': task_info.get('batch_size', 0),
            'learning_rate': task_info.get('learning_rate', 0),
            'eval_metric': task_info.get('eval_metric', ''),
            'categories': task_info.get('categories', [])
        })
    except Exception as e:
        print(f"获取任务状态异常: {str(e)}")
        return jsonify({"message": f"获取任务状态异常: {str(e)}"}), 500