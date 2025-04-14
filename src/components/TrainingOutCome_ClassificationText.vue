<!-- TrainingOutcome_ClassificationText.vue -->
<template>
    <div class="training-page container">
      <h2 class="text-center my-3">文字分类模型训练过程与结果</h2>
      <!-- 显示训练参数 -->
      <div class="card p-3 mb-4">
        <h4>训练参数</h4>
        <p><strong>模型选择：</strong>{{ selectedModel }}</p>
        <p><strong>语言：</strong>{{ language }}</p>
        <p><strong>Epochs:</strong>{{ epochs }}</p>
        <p><strong>Batch Size:</strong>{{ batchSize }}</p>
        <p><strong>Learning Rate:</strong>{{ learningRate }}</p>
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
        <div v-if="isTraining" class="text-center my-3">
          <p>正在训练中，请稍候...</p>
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
    name: "TrainingOutcome_ClassificationText",
    data() {
      return {
        // 训练参数，从路由或默认值获取
        selectedModel: "bert",
        language: "zh",       // 默认语言，可为 zh/en/other
        epochs: 50,
        batchSize: 16,
        learningRate: 0.001,
        evalMetric: "accuracy",
        trainingLogs: [],
        finalAccuracy: 0,
        timer: null,         // 定时轮询计时器
        isTraining: true     // 任务是否仍在运行
      }
    },
    computed: {
      // 评价指标的中文名称映射
      evalMetricName() {
        const map = {
          'accuracy': '准确率',
          'precision': '精确率',
          'recall': '召回率',
          'f1-score': 'F1值'
        }
        return map[this.evalMetric] || '准确率'
      }
    },
    methods: {
      // 定时向后端获取训练状态
      fetchTrainingStatus(taskId) {
        axios.get(`/api/classification/text/status/${taskId}`)
          .then(res => {
            const data = res.data
            console.log("从后端收到的数据：", data)
            // 更新前端显示的训练参数（这里也可由前端保存，不必从后端更新）
            this.selectedModel = data.model || this.selectedModel
            this.language = data.language || this.language
            this.epochs = data.epochs || this.epochs
            this.batchSize = data.batch_size || this.batchSize
            this.learningRate = data.learning_rate || this.learningRate
            this.evalMetric = data.eval_metric || this.evalMetric
  
            // 更新训练日志和最终准确率
            this.trainingLogs = data.training_logs || []
            this.finalAccuracy = data.final_accuracy
  
            // 渲染训练日志图表
            this.renderTrainingLogChart()
  
            // 如果任务状态为"completed"，标记停止轮询
            if (data.status === "completed") {
              console.log("训练完成，停止轮询...")
              this.isTraining = false
              clearInterval(this.timer)
              // 渲染最终结果图表
              this.$nextTick(() => {
                this.renderResultChart()
              })
            }
          })
          .catch(err => {
            console.error(err)
            clearInterval(this.timer)
            alert("获取训练状态失败：" + (err.response?.data?.message || "未知错误"))
          })
      },
      renderTrainingLogChart() {
        const chartDom = document.getElementById('trainingLogChart')
        if (!chartDom) return
        if (!this.trainingChart) {
          this.trainingChart = echarts.init(chartDom)
        }
        if (!this.trainingLogs || this.trainingLogs.length === 0) return
  
        const epochs = this.trainingLogs.map(log => log.epoch)
        const losses = this.trainingLogs.map(log => log.loss)
        const metrics = this.trainingLogs.map(log => log.metric)
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
        }
        this.trainingChart.setOption(option)
      },
      renderResultChart() {
        const chartDom = document.getElementById('resultChart')
        if (!chartDom) return
        if (!this.resultChart) {
          this.resultChart = echarts.init(chartDom)
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
              }
            }
          ]
        }
        this.resultChart.setOption(option)
      }
    },
    mounted() {
      // 从路由中获取任务ID
      const taskId = this.$route.query.taskId
      if (!taskId) {
        alert("缺少任务ID，无法获取训练状态")
        return
      }
      // 初始获取一次训练状态
      this.fetchTrainingStatus(taskId)
      // 设置定时轮询，每500毫秒获取一次
      this.timer = setInterval(() => {
        this.fetchTrainingStatus(taskId)
      }, 500)
      // 监听窗口大小变化以调整图表尺寸
      window.addEventListener('resize', this.handleResize)
    },
    beforeUnmount() {
      if (this.timer) clearInterval(this.timer)
      if (this.trainingChart) {
        this.trainingChart.dispose()
        this.trainingChart = null
      }
      if (this.resultChart) {
        this.resultChart.dispose()
        this.resultChart = null
      }
      window.removeEventListener('resize', this.handleResize)
    },

    handleResize() {
        if (this.trainingChart) this.trainingChart.resize()
        if (this.resultChart) this.resultChart.resize()
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
  