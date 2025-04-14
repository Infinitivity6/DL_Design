# app.py
# 注册所有业务并规定基本配置信息

import os
import time
import threading
from flask import Flask
from flask_cors import CORS
from routes.dataUpload_routes import data_bp
from routes.dataDisplay_routes import data_display_bp
from routes.dataDeduplication_routes import data_deduplication_bp
from routes.dataFill_routes import data_fill_bp
from routes.dataBalance_routes import data_balance_bp
from routes.dataNormalization_routes import data_normalization_bp
from routes.classification_num_train_routes import classification_num_bp
from routes.classification_image_upload_routes import image_upload_bp
from routes.classification_image_train_routes import classification_image_bp
from routes.classification_text_upload_routes import classification_text_upload_bp
from routes.classification_text_train_routes import classification_text_bp


# 用于存储任务状态的字典
task_status = {}

def train_task(task_id, model_choice, epochs, batch_size, learning_rate, eval_metric):
    """
    异步执行训练任务
    """
    result = train_model(model_choice, epochs, batch_size, learning_rate, eval_metric)
    task_status[task_id] = {
        'status': 'completed',
        'result': result,
    }


def create_app():
    app = Flask(__name__)
    CORS(app)  # 如果前后端端口不同，需要跨域

    # 配置上传目录 为同目录下uploads文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_folder = os.path.join(base_dir, 'uploads')
    preprocessed_folder = os.path.join(base_dir, 'preProcess')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # 确保图像分类的上传目录存在
    image_upload_folder = os.path.join(upload_folder, 'classification', 'Image')
    if not os.path.exists(image_upload_folder):
        os.makedirs(image_upload_folder)

    app.config['UPLOAD_FOLDER'] = upload_folder
    app.config['PREPROCESSED_FOLDER'] = preprocessed_folder

    # 注册Blueprint(数据上传功能)
    app.register_blueprint(data_bp, url_prefix='/api/data')

    # 注册Blueprint(数据展示功能)
    app.register_blueprint(data_display_bp,url_prefix='/api/dataDisplay')

    # 注册数据预处理_数据去重功能
    app.register_blueprint(data_deduplication_bp, url_prefix='/api/dataPreprocess')

    # 注册数据预处理_数据填充缺失值功能
    app.register_blueprint(data_fill_bp, url_prefix='/api/dataPreprocess')

    # 注册数据预处理_数据填充缺失值功能
    app.register_blueprint(data_balance_bp, url_prefix='/api/dataPreprocess')

    # 注册数据预处理_数据标准化功能
    app.register_blueprint(data_normalization_bp, url_prefix='/api/dataPreprocess')

    # 注册数值分类功能的模型功能
    app.register_blueprint(classification_num_bp, url_prefix='/api/classification')

    # 注册图像上传的功能
    app.register_blueprint(image_upload_bp, url_prefix='/api/image')

    # 注册图像分类的功能
    app.register_blueprint(classification_image_bp, url_prefix='/api/classification')

    # 注册文本上传的功能
    app.register_blueprint(classification_text_upload_bp, url_prefix='/api/classification/text/upload')

    # 注册文本分类的功能
    app.register_blueprint(classification_text_bp, url_prefix='/api/classification')




    



    @app.route('/')
    def index():
        return "Hello from Flask"

    return app

if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(debug=True, port=5000)
