<!-- ClassificationImageData.vue -->
<template>
    <div class="background-wrapper">
      <div class="container">
        <div class="card shadow-sm mb-5">
          <div class="card-header text-white text-center">
            <h2>图像数据分类任务</h2>
          </div>
          <div class="card-body">
            <!-- 类别管理和图像上传 -->
            <div class="mb-4">
              <h4 class="text-center mb-3">管理分类类别与图像</h4>
              <p class="text-muted text-center mb-4">请为每个类别上传图像，至少需要两个类别，每个类别至少上传若干张图片</p>
              
              <!-- 类别卡片列表 -->
              <div class="row">
                <div class="col-md-6 mb-3" v-for="(category, index) in categories" :key="index">
                  <div class="category-card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                      <input 
                        type="text" 
                        class="form-control-sm category-name-input" 
                        v-model="category.name" 
                        placeholder="输入类别名称" 
                      />
                      <button class="btn btn-sm btn-danger" @click="removeCategory(index)" :disabled="categories.length <= 2">
                        <i class="fas fa-times"></i>
                      </button>
                    </div>
                    <div class="card-body">
                      <div class="upload-area">
                        <label :for="'image-upload-'+index" class="btn btn-outline-primary w-100">
                          <i class="fas fa-cloud-upload-alt"></i> 选择图像
                        </label>
                        <input 
                          type="file" 
                          :id="'image-upload-'+index" 
                          class="d-none" 
                          @change="handleImageUpload($event, index)" 
                          accept="image/*"
                          multiple
                        />
                      </div>
                      
                      <!-- 已上传图像预览 -->
                      <div class="image-preview-container mt-3" v-if="category.images.length > 0">
                        <p class="text-muted small">
                          已上传 {{ category.images.length }} 张图片
                          <span v-if="category.images.length > 15">（仅显示前16张预览）</span>
                        </p>
                        <div class="image-preview-grid">
                          <div class="image-preview-item" v-for="(image, imgIndex) in displayedImages(category.images)" :key="imgIndex">
                            <img :src="image.preview" class="img-thumbnail" />
                            <button class="remove-image-btn" @click="removeImage(index, imgIndex)">
                              <i class="fas fa-times"></i>
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <!-- 添加类别按钮 -->
              <div class="text-center mt-3">
                <button class="btn btn-outline-success" @click="addCategory">
                  <i class="fas fa-plus"></i> 添加更多类别
                </button>
              </div>
              
              <!-- 上传数据集按钮 -->
              <div class="text-center mt-4">
                <button 
                  class="btn btn-primary btn-lg" 
                  @click="uploadCategoriesAndImages"
                  :disabled="!hasImagesToUpload"
                >
                  <i class="fas fa-upload"></i> 上传所有图像类别到服务器
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
                    <option value="ResNet18">ResNet18</option>
                    <option value="VGG16">VGG16</option>
                    <option value="MobileNetV2">MobileNetV2</option>
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
                
                <button 
                  class="btn btn-success btn-lg mt-3 w-100" 
                  @click="startTraining"
                  :disabled="!canStartTraining || !allImagesUploaded"
                >
                  <i class="fas fa-play"></i> 训练模型
                </button>
                
                <div class="alert alert-warning mt-3" v-if="!canStartTraining">
                  <p class="mb-0"><i class="fas fa-exclamation-triangle"></i> 请确保至少有两个类别，且每个类别至少上传了若干张图片，之后方可开始训练你的模型</p>
                </div>
                
                <div class="alert alert-warning mt-3" v-if="canStartTraining && !allImagesUploaded">
                  <p class="mb-0"><i class="fas fa-exclamation-triangle"></i> 请先上传所有选择的图片</p>
                </div>
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
        // 分类类别
        categories: [
          { name: '类别1', images: [] },
          { name: '类别2', images: [] }
        ],
        
        // 是否所有图片都已上传
        allImagesUploaded: false,
        
        // 训练参数
        selectedModel: 'ResNet18',
        epochs: 50,
        batchSize: 16,
        learningRate: 0.001,
        selectedMetrics: 'accuracy'
      };
    },
    computed: {
      // 判断是否可以开始训练
      canStartTraining() {
        // 至少需要两个类别
        if (this.categories.length < 2) return false;
        
        // 每个类别至少需要有图片
        for (const category of this.categories) {
          if (category.images.length === 0) return false;
        }
        
        return true;
      },
      
      // 判断是否有未上传的图片
      hasImagesToUpload() {
        for (const category of this.categories) {
          // 检查是否有图片需要上传
          if (category.images.some(img => !img.uploaded)) {
            return true;
          }
        }
        return false;
      }
    },
    methods: {
      // 限制显示的图片数量
      displayedImages(images) {
        return images.slice(0, 16); // 只显示前15张
      },
      
      // 添加新类别
      addCategory() {
        this.categories.push({
          name: `类别${this.categories.length + 1}`,
          images: []
        });
        this.allImagesUploaded = false;
      },
      
      // 移除类别
      removeCategory(index) {
        // 至少保留两个类别
        if (this.categories.length <= 2) return;
        this.categories.splice(index, 1);
        this.allImagesUploaded = false;
      },
      
      // 处理图片上传
      async handleImageUpload(event, categoryIndex) {
        const files = event.target.files;
        if (!files || files.length === 0) return;
        
        // 处理每个文件
        for (const file of files) {
          // 创建预览
          const reader = new FileReader();
          reader.onload = (e) => {
            this.categories[categoryIndex].images.push({
              file: file,
              preview: e.target.result,
              uploaded: false
            });
          };
          reader.readAsDataURL(file);
        }
        
        // 清空input，以便再次选择相同文件
        event.target.value = '';
        
        // 设置为未上传状态
        this.allImagesUploaded = false;
      },
      
      // 移除已上传图片
      removeImage(categoryIndex, imageIndex) {
        // 从预览数组中计算实际索引（因为我们可能只显示前15张）
        const actualIndex = imageIndex;
        this.categories[categoryIndex].images.splice(actualIndex, 1);
        this.allImagesUploaded = false;
      },
      
      // 上传所有图片到服务器
      async uploadAllImages() {
        try {
          // 创建上传任务列表
          const uploadTasks = [];
  
          for (let i = 0; i < this.categories.length; i++) {
            const category = this.categories[i];
            
            // 过滤出未上传的图片
            const imagesToUpload = category.images.filter(img => !img.uploaded);
            
            if (imagesToUpload.length === 0) continue;
            
            const formData = new FormData();
            
            // 添加类别名称
            formData.append('category_name', category.name);
            
            // 添加所有图片
            for (const image of imagesToUpload) {
              formData.append('images', image.file);
            }
            
            // 创建上传任务
            uploadTasks.push(
              axios.post('/api/image/upload', formData)
                .then(response => {
                  // 标记图片为已上传
                  imagesToUpload.forEach(img => {
                    img.uploaded = true;
                  });
                  
                  console.log(`成功上传到类别 ${category.name}: `, response.data);
                  return response.data;
                })
            );
          }
          
          // 等待所有上传任务完成
          if (uploadTasks.length > 0) {
            await Promise.all(uploadTasks);
            return true;
          }
          
          return true;  // 如果没有图片需要上传，也返回成功
        } catch (error) {
          console.error('上传图片失败:', error);
          alert('上传图片失败: ' + (error.response?.data?.message || '未知错误'));
          return false;
        }
      },
      
      // 上传类别和图片
      async uploadCategoriesAndImages() {
        // 显示上传中提示
        const uploadingMsg = "正在上传图片，请稍候...";
        alert(uploadingMsg);
        
        // 执行上传
        const success = await this.uploadAllImages();
        
        if (success) {
          this.allImagesUploaded = true;
          alert("所有图片上传成功！现在您可以开始训练模型了。");
        }
      },
      
      // 开始训练模型
      async startTraining() {
        // 如果有未上传的图片，提示先上传
        if (!this.allImagesUploaded) {
          alert("请先上传所有选择的图片到服务器");
          return;
        }
        
        // 构建训练请求
        const payload = {
          model_choice: this.selectedModel,
          epochs: this.epochs,
          batch_size: this.batchSize,
          learning_rate: this.learningRate,
          eval_metric: this.selectedMetrics,
          categories: this.categories.map(c => c.name)
        };
        
        try {
          // 发送训练请求
          const response = await axios.post('/api/classification/image', payload, {
            headers: {
              "Content-Type": "application/json"
            }
          });
          
          const taskId = response.data.task_id;
          
          // 跳转到训练结果页面
          this.$router.push({
            name: 'TrainingOutcome_ClassificationImage',
            query: { taskId: taskId }
          });
        } catch (error) {
          console.error('启动训练任务失败:', error);
          alert('启动训练任务失败: ' + (error.response?.data?.message || '未知错误'));
        }
      }
    }
  };
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
    padding: 2rem 0;
  }
  
  .container {
    max-width: 1000px;
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
  
  .category-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    overflow: hidden;
    height: 100%;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  }
  
  .category-card .card-header {
    background-color: #f7f7f7;
    color: #333;
    padding: 0.5rem;
    border-bottom: 1px solid #ddd;
  }
  
  .category-name-input {
    border: 1px solid #ddd;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    width: auto;
    flex-grow: 1;
    margin-right: 0.5rem;
  }
  
  .upload-area {
    border: 2px dashed #ddd;
    padding: 1.5rem;
    text-align: center;
    border-radius: 8px;
    background-color: #f9f9f9;
  }
  
  .image-preview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    gap: 10px;
  }
  
  .image-preview-item {
    position: relative;
  }
  
  .image-preview-item img {
    width: 100%;
    height: 80px;
    object-fit: cover;
    border-radius: 4px;
  }
  
  .remove-image-btn {
    position: absolute;
    top: -5px;
    right: -5px;
    background-color: #ff5252;
    color: white;
    border: none;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    font-size: 10px;
    cursor: pointer;
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