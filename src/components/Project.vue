<!-- 默认的选择文件上传和参数设置界面 -->
<template>
  <div class="container text-center">
    <h2>{{ projectType }} 项目</h2>

    <!-- 类别部分 -->
    <div v-for="(classItem, index) in classes" :key="index" class="my-3">
      <h4>类别 {{ index + 1 }}</h4>
      <div class="d-flex justify-content-center align-items-center">
        <input type="file" class="form-control m-2" @change="uploadFile($event, index)" />
        <button class="btn btn-danger large-btn m-2" @click="removeClass(index)">删除类别</button>
      </div>
    </div>
    <button class="btn btn-secondary my-3" @click="addClass">添加类别</button>

    <!-- 箭头 -->
    <div class="arrow-container my-5">
      <i class="fas fa-arrow-down"></i>
    </div>

    <!-- 训练参数部分 -->
    <div class="training-params my-5">
      <h4>设置训练参数</h4>
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
      <button class="btn btn-success btn-lg mt-3" @click="trainModel">训练模型</button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ProjectComponent', // 多词组合名称,vue3要求
  data() {
    return {
      projectType: this.$route.params.type,
      classes: [],
      epochs: 50,
      batchSize: 16,
      learningRate: 0.001
    };
  },
  methods: {
    addClass() {
      this.classes.push({ files: [] });
    },
    removeClass(index) {
      this.classes.splice(index, 1);
    },
    uploadFile(event, index) {
      const files = event.target.files;
      this.classes[index].files = files;
    },
    trainModel() {
      alert('Training Model...');
      // Add actual training logic here
      // 实际训练模型逻辑处理
    }
  }
};
</script>

<style scoped>

/* 删除类别按钮尺寸 */
.large-btn{
  width: 120px;    
  font-size: 1.25rem;  
  padding: 10px 20px;
  white-space: nowrap;
}

.container {
  padding: 20px;
  max-width: 600px;
  margin: 0 auto; /* 居中对齐 */
}

.input {
  width: 300px;
}

.arrow-container {
  text-align: center;
  font-size: 2rem;
}

.arrow-container i {
  color: #007bff; /* 箭头颜色 */
}

.training-params {
  margin-top: 40px;
}
</style>
