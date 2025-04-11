<!-- TrainingOutcome_ClassificationNum.vue -->
<template>
    <div class="training-page container">
      <h2 class="text-center my-3">模型训练过程与结果</h2>
      <!-- 显示训练参数 -->
      <div class="card p-3 mb-4">
        <h4>训练参数</h4>
        <p><strong>模型选择：</strong>{{ selectedModel }}</p>
        <p><strong>迭代轮数：</strong>{{ epochs }}</p>
        <p><strong>批处理大小：</strong>{{ batchSize }}</p>
        <p><strong>学习率：</strong>{{ learningRate }}</p>
        <p><strong>评价指标：</strong>{{ evalMetric }}</p>
      </div>
      <!-- 训练日志图表 -->
      <div class="card p-3 mb-4">
        <h4>训练过程日志</h4>
        <div id="trainingLogChart" class="chart"></div>
      </div>
      <!-- 最终结果图表 -->
      <div class="card p-3 mb-4">
        <h4>最终评估结果</h4>
        <p><strong>最终准确率：</strong>{{ finalAccuracy }}</p>
        <div id="resultChart" class="chart"></div>
      </div>
    </div>
  </template>
  
  <script>
  import axios from 'axios'
  import * as echarts from 'echarts'
  
  export default {
    name: "TrainingOutcome_ClassificationNum",
    data() {
      return {
        // 从路由 query 参数中获取
        selectedModel: "MLP",
        epochs: 50,
        batchSize: 16,
        learningRate: 0.001,
        evalMetric: "accuracy",
        trainingLogs: [],
        finalAccuracy: 0,
        timer: null  // 用于定时轮询
      }
    },
    methods: {
        fetchTrainingStatus(taskId) {
            axios.get(`/api/classification/status/${taskId}`)
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

                // 获取训练日志和最终准确率
                this.trainingLogs = data.training_logs;
                this.finalAccuracy = data.final_accuracy;

                // 渲染训练过程日志图表
                this.renderTrainingLogChart();
                this.renderResultChart();

                // 如果任务已完成，停止轮询
                if (data.status === "completed") {
                    console.log("训练完成，停止轮询...");
                    clearInterval(this.timer);  // 停止轮询
                }
            })
            .catch(err => {
                console.error(err);
                clearInterval(this.timer);  // 停止轮询
                alert("训练状态获取失败：" + (err.response?.data?.message || "未知错误"));
            });
        },
      renderTrainingLogChart() {
        const chartDom = document.getElementById('trainingLogChart')
        if (!chartDom) return
        const myChart = echarts.init(chartDom)
        const epochs = this.trainingLogs.map(log => log.epoch)
        const losses = this.trainingLogs.map(log => log.loss)
        const accuracies = this.trainingLogs.map(log => log.accuracy)
        const option = {
          title: { text: '训练过程日志', left: 'center' },
          tooltip: { trigger: 'axis' },
          legend: { data: ['Loss', 'Accuracy'], top: 30 },
          xAxis: { type: 'category', data: epochs, name: 'Epoch' },
          yAxis: [
            { type: 'value', name: 'Loss' },
            { type: 'value', name: 'Accuracy', max: 1 }
          ],
          series: [
            { name: 'Loss', type: 'line', data: losses },
            { name: 'Accuracy', type: 'line', yAxisIndex: 1, data: accuracies }
          ]
        }
        myChart.setOption(option)
      },
      renderResultChart() {
        const chartDom = document.getElementById('resultChart')
        if (!chartDom) return
        const myChart = echarts.init(chartDom)
        const option = {
          title: { text: '最终评估结果', left: 'center' },
          tooltip: { trigger: 'item' },
          series: [
            {
              name: '准确率',
              type: 'pie',
              radius: '50%',
              data: [
                { value: this.finalAccuracy, name: '准确率' },
                { value: (1 - this.finalAccuracy).toFixed(4), name: '不准确率' }
              ],
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowOffsetX: 0,
                  shadowColor: 'rgba(0,0,0,0.5)'
                }
              }
            }
          ]
        }
        myChart.setOption(option)
      },
    },
    mounted() {
        // 从路由中获取任务ID
        const taskId = this.$route.query.taskId;
        this.fetchTrainingStatus(taskId);
        this.timer = setInterval(() => {
            this.fetchTrainingStatus(taskId);
        }, 500);  // 每0.5秒获取一次训练状态
    },
    beforeUnmount() {
      // 组件销毁时清除定时器
      clearInterval(this.timer)
    }
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
  }
  .card {
    margin-bottom: 20px;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 5px;
  }
  </style>
  