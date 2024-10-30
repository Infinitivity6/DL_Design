<template> 
    <!-- 10.25开发新内容，增加导航栏 -->
  <div>
      <!-- 左上角的菜单 -->
    <div class="menu-icon-wrapper">
      <div class="menu-icon" @click="toggleMenu">
        <i class="fas fa-bars"></i> 深度学习多任务学习平台
      </div>
  </div>

    <!-- 菜单内容（可折叠） -->
    <div v-if="isMenuOpen" class="menu-content">
      <ul>
        <li @click="chooseFunction('classification')">
          <img src="../assets/img/index_menubar/classification.png" alt="分类图标">分类
        </li>
        <li @click="chooseFunction('regression')">
          <img src="../assets/img/index_menubar/linear_regression.png" alt="回归图标">回归
        </li>
        <li @click="chooseFunction('data_generation')">
          <img src="../assets/img/index_menubar/data_generation.png" alt="数据生=生成图标">数据生成
        </li>
        <!-- 其他任务类型 -->
      </ul>
    </div>
  </div>

    <!-- 主体内容 -->
    <div class="container text-center">
      <h1 class="my-5">创建一个新项目</h1>
      <div class="d-flex justify-content-center flex-wrap">
  
        <!-- 文字项目 -->
        <div class="card m-3" style="width: 21rem; height: 21rem;" @click="goToTextProcessing()">
          <div class="card-img-container">
            <img src="../assets/img/index_cards/text/sample1.png" class="sample-img" alt="Text Sample 1">
            <img src="../assets/img/index_cards/text/sample2.png" class="sample-img" alt="Text Sample 2">
            <img src="../assets/img/index_cards/text/sample3.png" class="sample-img" alt="Text Sample 3">
          </div>
          <div class="card-body">
            <h5 class="card-title">文字项目</h5>
            <p class="card-text">基于文字进行训练，可以上传文本文件或输入文字。</p>
          </div>
        </div>
  
        <!-- 图片项目 -->
        <div class="card m-3" style="width: 21rem; height: 21rem" @click="goToProject('image-processing')">
          <div class="card-img-container">
            <img src="../assets/img/index_cards/pic/sample1.jpg" class="sample-img" alt="Image Sample 1">
            <img src="../assets/img/index_cards/pic/sample2.jpg" class="sample-img" alt="Image Sample 2">
            <img src="../assets/img/index_cards/pic/sample3.jpg" class="sample-img" alt="Image Sample 3">
          </div>
          <div class="card-body">
            <h5 class="card-title">图片项目</h5>
            <p class="card-text">基于图片进行训练，支持文件上传或使用你的摄像头。</p>
          </div>
        </div>
  
        <!-- 视频项目 -->
        <div class="card m-3" style="width: 21rem; height: 21rem" @click="goToProject('video-processing')">
          <div class="card-img-container">
            <img src="../assets/img/index_cards/video/sample1.jpg" class="sample-img" alt="Video Sample 1">
            <img src="../assets/img/index_cards/video/sample2.jpg" class="sample-img" alt="Video Sample 2">
            <img src="../assets/img/index_cards/video/sample3.jpg" class="sample-img" alt="Video Sample 3">
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
    data(){
      return {
        isMenuOpen: false
      };
    },
    methods: {
      toggleMenu(){
        this.isMenuOpen = !this.isMenuOpen;
      },
      chooseFunction(){
        console.log("这里功能是选择功能");
        //功能选择逻辑
      },
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
/* 10.25新增菜单样式 */
.menu-icon-wrapper {
  background-color: white;
  padding: 15px 30px; /* 增大框框 */
  position: fixed;
  top: 10px; /* 下移一点以避免顶栏紧贴浏览器顶部 */
  left: 10px; /* 保证边距 */
  z-index: 999;
  border-radius: 15px; /* 圆角 */
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); /* 增加阴影效果 */
  display: flex;
  align-items: center;
}

.menu-icon {
  cursor: pointer;
  display: flex;
  align-items: center;
  font-size: 1.3rem;
  /* 应用谷歌的某个艺术字体 */
  font-family: 'Pacifico', cursive;
}

.menu-icon i {
  margin-right: 10px;
}

.menu-content {
  background-color: white;
  position: absolute;
  top: 80px; /* 调整下方内容位置 */
  left: 10px;
  padding: 15px 20px;
  border-radius: 10px; /* 圆角 */
  box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
  font-family: 'Roboto', sans-serif; /* 使用Roboto作为正文字体 */
}

.menu-content ul {
  list-style: none;
  padding: 0;
  margin: 0;
  text-align: left;
}

/* .menu-content li {
  margin: 15px 0;
  font-size: 1.2rem;
}   原来的最初草稿菜单样式，比较简陋，遗弃*/

.menu-content li {
  display: flex;
  align-items: center; /* 图片与文字对齐 */
  margin: 10px 0;
  padding: 10px;
  font-size: 1.2rem; /* 调整字体大小 */
  font-family: 'Pacifico', cursive; /* 使用艺术字体 */
  color: #333; /* 默认字体颜色 */
  cursor: pointer;
  transition: background-color 0.3s, color 0.3s; /* 平滑过渡效果 */
}

.menu-content li:hover {
  background-color: #f0f0f0; /* 悬停时背景变灰 */
  color: #000; /* 悬停时字体变黑 */
}

.menu-content li img {
  margin-right: 10px; /* 图片与文字之间的距离 */
  width: 20px; /* 图片大小 */
  height: 20px;
}

.menu-content li:active {
  background-color: #e0e0e0; /* 点击时背景色稍微加深 */
  color: #007bff; /* 点击时文字变色 */
}

  .container {
    padding: 100px;
    margin-top: 0;
  }
  

  /* 卡片点击动画 */
  .card {
  cursor: pointer; /* 鼠标悬浮时变为手指 */
  transition: transform 0.3s ease, box-shadow 0.3s ease; /* 平滑的动画过渡 */
  display: flex;
  flex-direction: column;
  justify-content: space-between;
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
    padding: 20px 10px 0 10px;
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
    margin-top: 10px;
    margin-bottom: 30px;
  }
  
  p.card-text {
    font-size: 0.9rem;
    color: #6c757d;
    text-align: left;
    margin-bottom: 10px;
  }
  </style>
  