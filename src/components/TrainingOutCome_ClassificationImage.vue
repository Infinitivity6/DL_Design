<!-- TrainingOutcome_ClassificationImage.vue -->
<template>
    <div class="training-page container">
      <h2 class="text-center my-3">图像分类模型训练过程与结果</h2>
      
      <!-- 显示训练参数 -->
      <div class="card p-3 mb-4">
        <h4>训练参数</h4>
        <p><strong>模型选择：</strong>{{ selectedModel }}</p>
        <p><strong>迭代轮数：</strong>{{ epochs }}</p>
        <p><strong>批处理大小：</strong>{{ batchSize }}</p>
        <p><strong>学习率：</strong>{{ learningRate }}</p>
        <p><strong>评价指标：</strong>{{ evalMetric }}</p>
        <p><strong>分类类别：</strong>{{ categories.join(', ') }}</p>
      </div>
      
      <!-- 训练日志图表 -->
      <div class="card p-3 mb-4">
        <h4>训练过程日志</h4>
        <div v-if="isTraining" class="text-center my-3">
          <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">加载中...</span>
          </div>
          <p class="mt-2">模型训练中，请稍候...</p>
        </div>
        <div id="trainingLogChart" class="chart"></div>
      </div>
      
      <!-- 最终结果图表 -->
      <div class="card p-3 mb-4">
        <h4>最终评估结果</h4>
        <div v-if="isTraining" class="text-center my-3">
          <p>训练完成后将显示评估结果</p>
        </div>
        <div v-else>
          <p><strong>最终{{ evalMetricName }}：</strong>{{ finalAccuracy }}</p>
          <div id="resultChart" class="chart"></div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import axios from 'axios'
  import * as echarts from 'echarts'
  
  export default {
    name: "TrainingOutcome_ClassificationImage",
    data() {
      return {
        // 从路由 query 参数中获取
        selectedModel: "ResNet18",
        epochs: 50,
        batchSize: 16,
        learningRate: 0.001,
        evalMetric: "accuracy",
        categories: [],
        trainingLogs: [],
        finalAccuracy: 0,
        timer: null,  // 用于定时轮询
        isTraining: true,  // 训练状态标志
        trainingChart: null,
        resultChart: null

      }
    },
    computed: {
      // 评价指标的中文名称
      evalMetricName() {
        const metricMap = {
          'accuracy': '准确率',
          'precision': '精确率',
          'recall': '召回率',
          'f1-score': 'F1值'
        };
        return metricMap[this.evalMetric] || '准确率';
      }
    },
    methods: {
      fetchTrainingStatus(taskId) {
        axios.get(`/api/classification/image/status/${taskId}`)
          .then(res => {
            const data = res.data;
            console.log("从后端收到的数据：", data);

            // 确保从后端接收到正确的状态字段
            if (!data.status) {
              console.error("返回数据中没有 status 字段");
              return;
            }

            // 获取训练参数并更新界面
            this.selectedModel = data.model || this.selectedModel;
            this.epochs = data.epochs || this.epochs;
            this.batchSize = data.batch_size || this.batchSize;
            this.learningRate = data.learning_rate || this.learningRate;
            this.evalMetric = data.eval_metric || this.evalMetric;
            this.categories = data.categories || [];

            // 获取训练日志和最终准确率
            this.trainingLogs = data.training_logs || [];
            this.finalAccuracy = data.final_accuracy;

            // 渲染训练过程日志图表和结果图表
            this.renderTrainingLogChart();

            // 如果任务已完成，停止轮询
            if (data.status === "completed") {
              console.log("训练完成，停止轮询...");
              this.isTraining = false;
              clearInterval(this.timer);  // 停止轮询
              // 确保训练完成后渲染结果图表
              this.$nextTick(() => {
              this.renderResultChart();
            });

            }
          })
          .catch(err => {
            console.error(err);
            clearInterval(this.timer);  // 停止轮询
            alert("训练状态获取失败：" + (err.response?.data?.message || "未知错误"));
          });
      },
      
      renderTrainingLogChart() {
        const chartDom = document.getElementById('trainingLogChart');
        if (!chartDom) return;
        
        // 如果图表实例不存在，创建新实例
        if (!this.trainingChart) {
            this.trainingChart = echarts.init(chartDom);
        }
        
        // 如果没有训练日志数据，不进行渲染
        if (!this.trainingLogs || this.trainingLogs.length === 0) return;
        
        const epochs = this.trainingLogs.map(log => log.epoch);
        const losses = this.trainingLogs.map(log => log.loss);
        const metrics = this.trainingLogs.map(log => log.metric);

        const option = {
            title: { text: '训练过程日志', left: 'center' },
            tooltip: { trigger: 'axis' },
            legend: { data: ['Loss', this.evalMetricName], top: 30 },
            xAxis: { type: 'category', data: epochs, name: 'Epoch' },
            yAxis: [
            { type: 'value', name: 'Loss' },
            { type: 'value', name: this.evalMetricName, max: 1 }
            ],
            series: [
            { name: 'Loss', type: 'line', data: losses },
            { name: this.evalMetricName, type: 'line', yAxisIndex: 1, data: metrics }
            ]
        };
        
        // 使用setOption更新图表，不完全替换选项
        this.trainingChart.setOption(option);
        },
    
        renderResultChart() {
            // 如果训练还在进行中，不渲染结果图表
            if (this.isTraining) return;
            
            const chartDom = document.getElementById('resultChart');
            if (!chartDom) return;
            
            // 如果图表实例不存在，创建新实例
            if (!this.resultChart) {
                this.resultChart = echarts.init(chartDom);
            }
            
            const option = {
                title: { text: '最终评估结果', left: 'center' },
                tooltip: { trigger: 'item', formatter: '{a} <br/>{b}: {c} ({d}%)' },
                series: [
                {
                    name: this.evalMetricName,
                    type: 'pie',
                    radius: '50%',
                    center: ['50%', '60%'],
                    data: [
                    { value: this.finalAccuracy, name: this.evalMetricName },
                    { value: (1 - this.finalAccuracy).toFixed(4), name: `非${this.evalMetricName}` }
                    ],
                    emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0,0,0,0.5)'
                    }
                    },
                    label: {
                    normal: {
                        formatter: '{b}: {c} ({d}%)'
                    }
                    }
                }
                ]
            };
            
            this.resultChart.setOption(option);
         },  
    
    },
    mounted() {
      // 从路由中获取任务ID
      const taskId = this.$route.query.taskId;
      if (!taskId) {
        alert("缺少任务ID，无法获取训练状态");
        return;
      }
      
      // 初始获取一次训练状态
      this.fetchTrainingStatus(taskId);
      
      // 设置定时轮询，每秒获取一次训练状态
      this.timer = setInterval(() => {
        this.fetchTrainingStatus(taskId);
      }, 500);
      // 添加窗口大小变化监听，自动调整图表大小
      window.addEventListener('resize', this.handleResize);

      
    },
    beforeUnmount() {
        // 组件销毁时清除定时器
        if (this.timer) {
        clearInterval(this.timer);
        }
        
        // 销毁图表实例
        if (this.trainingChart) {
        this.trainingChart.dispose();
        this.trainingChart = null;
        }
        
        if (this.resultChart) {
        this.resultChart.dispose();
        this.resultChart = null;
        }
        
        // 移除窗口大小变化监听
        window.removeEventListener('resize', this.handleResize);
    },
    handleResize() {
        if (this.trainingChart) {
        this.trainingChart.resize();
        }
        
        if (this.resultChart) {
        this.resultChart.resize();
        }
    },
  
  }
  </script>
  
  <style scoped>
  .container {
    padding: 20px;
    max-width: 1000px;
  }
  .chart {
  width: 100%;
  height: 400px;
  margin-top: 20px;
  border: 1px solid #eee;
  background-color: #fff;
  }
  .card {
    margin-bottom: 20px;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 5px;
  }
  </style>