<template>
    <div class="container text-center">
      <h1 class="my-5">创建一个新项目</h1>
      <div class="d-flex justify-content-center flex-wrap">
  
        <!-- 文字项目 -->
        <div class="card m-3" style="width: 18rem;" @click="goToTextProcessing()">
          <div class="card-img-container">
            <img src="../assets/img/text/sample1.png" class="sample-img" alt="Text Sample 1">
            <img src="../assets/img/text/sample2.png" class="sample-img" alt="Text Sample 2">
            <img src="../assets/img/text/sample3.png" class="sample-img" alt="Text Sample 3">
          </div>
          <div class="card-body">
            <h5 class="card-title">文字项目</h5>
            <p class="card-text">基于文字进行训练，可以上传文本文件或输入文字。</p>
          </div>
        </div>
  
        <!-- 图片项目 -->
        <div class="card m-3" style="width: 18rem;" @click="goToProject('图片')">
          <div class="card-img-container">
            <img src="../assets/img/pic/sample1.jpg" class="sample-img" alt="Image Sample 1">
            <img src="../assets/img/pic/sample2.jpg" class="sample-img" alt="Image Sample 2">
            <img src="../assets/img/pic/sample3.jpg" class="sample-img" alt="Image Sample 3">
          </div>
          <div class="card-body">
            <h5 class="card-title">图片项目</h5>
            <p class="card-text">基于图片进行训练，支持文件上传或使用你的摄像头。</p>
          </div>
        </div>
  
        <!-- 视频项目 -->
        <div class="card m-3" style="width: 18rem;" @click="goToProject('视频')">
          <div class="card-img-container">
            <img src="../assets/img/video/sample1.jpg" class="sample-img" alt="Video Sample 1">
            <img src="../assets/img/video/sample2.jpg" class="sample-img" alt="Video Sample 2">
            <img src="../assets/img/video/sample3.jpg" class="sample-img" alt="Video Sample 3">
          </div>
          <div class="card-body">
            <h5 class="card-title">视频项目</h5>
            <p class="card-text">基于视频进行训练，支持上传视频文件或使用摄像头录制视频。</p>
          </div>
        </div>
  
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'HomeComponent', // 多词组合名称
    methods: {
      goToProject(type) {
        this.$router.push({ name: 'Project', params: { type } });
      },
      goToTextProcessing(){
        this.$router.push({ name: 'TextProcessing' });
      }

    }
  }
  </script>
  
  <style scoped>
  .container {
    padding: 20px;
  }
  

  /* 卡片点击动画 */
  .card {
  cursor: pointer; /* 鼠标悬浮时变为手指 */
  transition: transform 0.3s ease, box-shadow 0.3s ease; /* 平滑的动画过渡 */
}

.card:hover {
  transform: translateY(-10px); /* 悬浮时卡片上移 */
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); /* 四周阴影 */
}

.card-img-container img {
  transition: transform 0.3s ease; /* 图片变换的平滑动画 */
}

.card:hover .card-img-container img {
  transform: scale(1.05); /* 悬浮时图片稍微放大 */
}




  .card-img-container {
    display: flex;
    justify-content: space-between;
    padding: 10px;
  }
  
  .sample-img {
    width: 30%;
    height: auto; /* 自动适应高度以保持宽高比例 */
    object-fit: cover; /* 确保图片不会变形 */
  }
  
  .card-body {
    text-align: center;
  }
  
  h5.card-title {
    font-size: 1.25rem;
    font-weight: bold;
  }
  
  p.card-text {
    font-size: 0.9rem;
    color: #6c757d;
  }
  </style>
  