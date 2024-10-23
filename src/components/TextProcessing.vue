<!-- 文字项目的选择文件上传和参数设置界面 -->
<template>
    <div class="text-processing-container">
      <h2>文字处理任务</h2>
  
      <!-- 选择任务类型 -->
      <div>
        <label for="task-type">选择任务类型:</label>
        <select id="task-type" v-model="taskType">
          <option value="text-classification">文本分类</option>
          <option value="ner">命名实体识别</option>
        </select>
      </div>
  
      <!-- 上传训练集、验证集、测试集 -->
      <div>
        <label for="train-file">上传训练集:</label>
        <input type="file" id="train-file" @change="uploadTrainFile">
  
        <label for="val-file">上传验证集:</label>
        <input type="file" id="val-file" @change="uploadValFile">
  
        <label for="test-file">上传测试集:</label>
        <input type="file" id="test-file" @change="uploadTestFile">
      </div>
  
      <!-- 选择使用模型 -->
      <div>
        <label for="model-selection">选择使用模型:</label>
        <select id="model-selection" v-model="selectedModel">
          <option value="bert">BERT</option>
          <option value="gpt">GPT</option>
          <option value="lstm">LSTM</option>
        </select>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        taskType: 'text-classification', // 默认任务类型
        selectedModel: 'bert', // 默认模型
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
      }
    }
  }
  </script>
  
  <style scoped>
  .text-processing-container {
    padding: 20px;
  }
  </style>
  