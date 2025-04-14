<!-- ClassificationTextData.vue -->
<template>
  <div class="background-wrapper">
    <div class="container">
      <div class="card shadow-sm mb-5">
        <div class="card-header text-white text-center">
          <h2>文字分类任务</h2>
        </div>
        <div class="card-body">
          <!-- 任务语言选择 -->
          <div class="row mb-4">
            <div class="col-md-6 offset-md-3">
              <label for="task-type">选择语言:</label>
              <!-- 此处选择 "text-classification" 表示中文，"ner" 表示英文（你也可以根据实际需要调整） -->
              <select id="task-type" v-model="taskType" class="form-control">
                <option value="text-classification">中文</option>
                <option value="ner">英文</option>
              </select>
            </div>
          </div>

          <!-- 文件上传 -->
          <div class="row mb-4">
            <div class="col-md-4">
              <label for="train-file">上传训练集:</label>
              <input type="file" id="train-file" class="form-control" @change="onTrainFileChange">
              <button class="btn btn-primary mt-2" @click="uploadTrainFile">上传训练集</button>
            </div>
            <div class="col-md-4">
              <label for="val-file">上传验证集:</label>
              <input type="file" id="val-file" class="form-control" @change="onValFileChange">
              <button class="btn btn-primary mt-2" @click="uploadValFile">上传验证集</button>
            </div>
            <div class="col-md-4">
              <label for="test-file">上传测试集:</label>
              <input type="file" id="test-file" class="form-control" @change="onTestFileChange">
              <button class="btn btn-primary mt-2" @click="uploadTestFile">上传测试集</button>
            </div>
          </div>

          <!-- 查看数据集按钮 -->
          <div class="row mb-4 text-center">
            <div class="col-md-12">
              <button class="btn btn-custom" @click="viewDataset">
                <i class="fas fa-database"></i> 查看数据集
              </button>
            </div>
          </div>

          <!-- 模型训练配置 -->
          <div class="card mt-5">
            <div class="card-header text-white">
              <h4>设置训练参数</h4>
            </div>
            <div class="card-body">
              <!-- 模型选择 -->
              <div class="form-group mb-3">
                <label for="model-selection">选择使用模型:</label>
                <select id="model-selection" v-model="selectedModel" class="form-control">
                  <option value="bert">BERT</option>
                  <option value="transformer">Transformer</option>
                  <option value="lstm">LSTM</option>
                </select>
              </div>

              <div class="form-group mb-3">
                <label>Epochs:</label>
                <input type="number" class="form-control" v-model="epochs" />
              </div>

              <div class="form-group mb-3">
                <label>Batch Size:</label>
                <input type="number" class="form-control" v-model="batchSize" />
              </div>

              <div class="form-group mb-3">
                <label>Learning Rate:</label>
                <input type="number" step="0.001" class="form-control" v-model="learningRate" />
              </div>

              <div class="form-group mb-3">
                <label for="evaluation-metrics">选择评价指标:</label>
                <select id="evaluation-metrics" v-model="selectedMetrics" class="form-control">
                  <option value="accuracy">准确率 (Accuracy)</option>
                  <option value="precision">精确率 (Precision)</option>
                  <option value="recall">召回率 (Recall)</option>
                  <option value="f1-score">F1 值 (F1 Score)</option>
                </select>
              </div>

              <!-- 开始训练按钮；在至少上传训练集后启用 -->
              <button class="btn btn-success btn-lg mt-3 w-100" @click="trainModel" :disabled="!trainFileUploaded">
                <i class="fas fa-play"></i> 训练模型
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
export default {
  data() {
    return {
      taskType: 'text-classification', // 默认中文
      selectedModel: 'bert',             // 默认使用 BERT 模型
      epochs: 50,
      batchSize: 16,
      learningRate: 0.001,
      selectedMetrics: 'accuracy',

      // 用于存储用户选择的三个文件
      trainFile: null,
      valFile: null,
      testFile: null,

      // 标记训练集是否已上传
      trainFileUploaded: false
    };
  },
  methods: {
    // 处理文件选择
    onTrainFileChange(event) {
      this.trainFile = event.target.files[0];
      console.log('Train File selected:', this.trainFile);
    },
    onValFileChange(event) {
      this.valFile = event.target.files[0];
      console.log('Val File selected:', this.valFile);
    },
    onTestFileChange(event) {
      this.testFile = event.target.files[0];
      console.log('Test File selected:', this.testFile);
    },
    // 上传训练集文件
    uploadTrainFile() {
      if (!this.trainFile) {
        alert("请先选择训练集文件");
        return;
      }
      const formData = new FormData();
      formData.append('dataset_type', 'train');
      formData.append('file', this.trainFile);
      // 文件上传接口 (假设接口为 /api/data/uploadClassificationText)
      axios.post('/api/classification/text/upload', formData)
        .then(res => {
          alert(res.data.message);
          this.trainFileUploaded = true; // 标记训练集已上传
        })
        .catch(err => {
          console.error(err);
          alert('上传训练集失败');
        });
    },
    // 上传验证集文件
    uploadValFile() {
      if (!this.valFile) {
        alert("请先选择验证集文件");
        return;
      }
      const formData = new FormData();
      formData.append('dataset_type', 'validation');
      formData.append('file', this.valFile);
      axios.post('/api/classification/text/upload', formData)
        .then(res => {
          alert(res.data.message);
        })
        .catch(err => {
          console.error(err);
          alert('上传验证集失败');
        });
    },
    // 上传测试集文件
    uploadTestFile() {
      if (!this.testFile) {
        alert("请先选择测试集文件");
        return;
      }
      const formData = new FormData();
      formData.append('dataset_type', 'test');
      formData.append('file', this.testFile);
      axios.post('/api/classification/text/upload', formData)
        .then(res => {
          alert(res.data.message);
        })
        .catch(err => {
          console.error(err);
          alert('上传测试集失败');
        });
    },
    // 查看数据集页面，假设路由名称为 'DatasetDisplay'
    viewDataset() {
      this.$router.push({ name: 'DatasetDisplay' });
    },
    // 开始训练模型
    async trainModel() {
      // 检查训练集是否上传
      if (!this.trainFileUploaded) {
        alert("请先上传训练集文件");
        return;
      }
      // 根据 taskType 字段确定语言参数；这里 "text-classification" 代表中文，"ner" 代表英文
      const language = (this.taskType === 'text-classification') ? 'zh' : 'en';

      // 构建训练请求的 payload 数据
      const payload = {
        model_choice: this.selectedModel,  // 模型选择，例如 "bert"
        language: language,
        epochs: this.epochs,
        batch_size: this.batchSize,
        learning_rate: this.learningRate,
        eval_metric: this.selectedMetrics
      };

      try {
        // 发送训练任务请求到后端文字分类训练接口（假设接口为 /api/classification/text）
        const response = await axios.post('/api/classification/text', payload, {
          headers: {
            "Content-Type": "application/json"
          }
        });
        const taskId = response.data.task_id;
        // 跳转到训练结果显示页面，并将任务ID作为 query 参数传递，假设路由名称为 'TrainingOutcome_ClassificationText'
        this.$router.push({
          name: 'TrainingOutcome_ClassificationText',
          query: { taskId: taskId }
        });
      } catch (error) {
        console.error("启动训练任务失败:", error);
        alert("启动训练任务失败:" + (error.response?.data?.message || "未知错误"));
      }
    }
  }
}
</script>

<style scoped>
.background-wrapper {
  background-image: url('../assets/img/background/bak5.jpg');
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}
.container {
  max-width: 800px;
  padding: 1.5rem;
  background-color: rgba(255, 255, 255, 0.9);
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
.card {
  border-radius: 20px;
  overflow: hidden;
}
.card-header {
  background-color: #3B6695;
  color: white;
  font-weight: bold;
  text-align: center;
  padding: 1rem;
}
.card h2, .card h4 {
  font-family: 'Roboto', sans-serif;
  color: #ffffff;
  margin: 0.5rem 0;
}
.form-control {
  border: none;
  background-color: #f7f7f7;
  padding: 0.5rem;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
  transition: background 0.3s ease;
}
.btn-custom {
  background-color: #4CAF50;
  color: white;
  font-weight: bold;
  transition: background 0.3s ease;
}
.btn-custom:hover {
  background-color: #45a049;
}
</style>
