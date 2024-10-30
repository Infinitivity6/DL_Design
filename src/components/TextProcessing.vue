<template>
  <div class="background-wrapper">
    <div class="container">
      <div class="card shadow-sm mb-5">
        <div class="card-header text-white text-center">
          <h2>文字处理任务</h2>
        </div>
        <div class="card-body">
          <!-- 任务类型选择 -->
          <div class="row mb-4">
            <div class="col-md-6 offset-md-3">
              <label for="task-type">选择任务类型:</label>
              <select id="task-type" v-model="taskType" class="form-control">
                <option value="text-classification">文本分类</option>
                <option value="ner">命名实体识别</option>
              </select>
            </div>
          </div>

          <!-- 文件上传 -->
          <div class="row mb-4">
            <div class="col-md-4">
              <label for="train-file">上传训练集:</label>
              <input type="file" id="train-file" class="form-control" @change="uploadTrainFile">
            </div>
            <div class="col-md-4">
              <label for="val-file">上传验证集:</label>
              <input type="file" id="val-file" class="form-control" @change="uploadValFile">
            </div>
            <div class="col-md-4">
              <label for="test-file">上传测试集:</label>
              <input type="file" id="test-file" class="form-control" @change="uploadTestFile">
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

          <!-- 模型选择 -->
          <div class="row mb-4">
            <div class="col-md-6 offset-md-3">
              <label for="model-selection">选择使用模型:</label>
              <select id="model-selection" v-model="selectedModel" class="form-control">
                <option value="bert">BERT</option>
                <option value="gpt">GPT</option>
                <option value="lstm">LSTM</option>
              </select>
            </div>
          </div>

          <!-- 训练参数设置 -->
          <div class="card mt-5">
            <div class="card-header text-white">
              <h4>设置训练参数</h4>
            </div>
            <div class="card-body">
              <div class="form-group">
                <label>Epochs:</label>
                <input type="number" class="form-control" v-model="epochs" />
              </div>
              <div class="form-group">
                <label>Batch Size:</label>
                <input type="number" class="form-control" v-model="batchSize" />
              </div>
              <div class="form-group">
                <label>Learning Rate:</label>
                <input type="number" step="0.001" class="form-control" v-model="learningRate" />
              </div>

            <!-- 10.30新开发增加用户选择评价指标框 -->
            <div class="form-group mt-4">
              <label for="evaluation-metrics">选择评价指标:</label>
              <select id="evaluation-metrics" v-model="selectedMetrics" class="form-control">
                <option value="accuracy">准确率 (Accuracy)</option>
                <option value="precision">精确率 (Precision)</option>
                <option value="recall">召回率 (Recall)</option>
                <option value="f1-score">F1 值 (F1 Score)</option>
              </select>
            </div>

              <button class="btn btn-success btn-lg mt-3 w-100" @click="trainModel">
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
export default {
  data() {
    return {
      taskType: 'text-classification',
      selectedModel: 'bert',
      epochs: 50,
      batchSize: 16,
      learningRate: 0.001,
      selectedMetrics: 'accuracy' // 默认评价指标
    };
  },
  methods: {
    uploadTrainFile(event) {
      const file = event.target.files[0];
      console.log('Train File:', file);
    },
    uploadValFile(event) {
      const file = event.target.files[0];
      console.log('Validation File:', file);
    },
    uploadTestFile(event) {
      const file = event.target.files[0];
      console.log('Test File:', file);
    },
    trainModel() {
      alert(`开始训练模型，使用的评价指标为：${this.selectedMetrics}`);
      // 在此处实现模型训练逻辑，并将 selectedMetrics 发送给后端
    },
    viewDataset() {
      this.$router.push({ name: 'DatasetDisplay' });
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
  /* background-color: #007bff; 改为柔和的蓝色
  border-radius: 12px 12px 0 0;
  font-weight: bold; */
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
