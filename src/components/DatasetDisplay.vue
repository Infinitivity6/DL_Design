<template>

    <div class="container">
         <!-- 独立的标题行，使标题居中 -->
        <div class="text-center my-3">
            <h2>数据集展示</h2>
        </div>

        <!-- 下拉菜单部分 -->
      <div class="d-flex justify-content-end mb-5">
        <div class="dropdown">
          <button
            class="btn btn-secondary dropdown-toggle"
            type="button"
            id="datasetDropdown"
            data-bs-toggle="dropdown"
            aria-expanded="false"
          >
            选择数据集
          </button>
          <ul class="dropdown-menu" aria-labelledby="datasetDropdown">
            <li><a class="dropdown-item" @click="selectDataset('train')">训练集</a></li>
            <li><a class="dropdown-item" @click="selectDataset('validation')">验证集</a></li>
            <li><a class="dropdown-item" @click="selectDataset('test')">测试集</a></li>
            <li><a class="dropdown-item" @click="selectDataset('all')">所有数据</a></li>
          </ul>
        </div>
      </div>
  
      <!-- 数据集表格展示 -->
      <table class="table table-bordered">
        <thead>
          <tr>
            <th v-for="(column, index) in columns" :key="index">{{ column }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, rowIndex) in rows" :key="rowIndex">
            <td v-for="(cell, cellIndex) in row" :key="cellIndex">{{ cell }}</td>
          </tr>
        </tbody>
      </table>
  
      <!-- 按钮组 -->
      <div class="text-center my-4">
        <button class="btn btn-warning mx-2" @click="removeDuplicates">去除重复值</button>
        <button class="btn btn-info mx-2" @click="fillMissingValues">缺失值填充</button>
        <button class="btn btn-success mx-2" @click="balanceData">数据平衡</button>
        <button class="btn btn-primary mx-2" @click="normalizeData">数据标准化</button>
      </div>
  
      <!-- 数据统计图展示 -->
      <div class="charts-container mt-5">
        <h4 class="text-center">数据统计图</h4>
  
        <!-- 扇形图 -->
        <div class="chart-placeholder">
          <h5>扇形图</h5>
          <div id="pieChart" class="chart"></div>
        </div>
  
        <!-- 条形图 -->
        <div class="chart-placeholder mt-4">
          <h5>条形图</h5>
          <div id="barChart" class="chart"></div>
        </div>
      </div>


      <!-- 10.30新开发 缺失值填充/数据平衡选项 -->
    <!-- 模态框：缺失值填充选项 -->
    <div class="modal fade" id="fillMissingModal" tabindex="-1" aria-labelledby="fillMissingModalLabel" aria-hidden="true">
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="fillMissingModalLabel">缺失值填充选项</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <label for="columnSelect">选择列名:</label>
            <select id="columnSelect" v-model="selectedColumn" class="form-control">
              <option v-for="column in columns" :key="column" :value="column">{{ column }}</option>
            </select>

            <label class="mt-3" for="fillMethod">选择填充方式:</label>
            <select id="fillMethod" v-model="fillMethod" class="form-control">
              <option value="interpolation">插值</option>
              <option value="mean">均值填充</option>
              <option value="specific">填充特定值</option>
            </select>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
            <button type="button" class="btn btn-primary" @click="confirmFillMissing">确认</button>
          </div>
        </div>
      </div>
    </div>

    <!-- 模态框：数据平衡选项 -->
    <div class="modal fade" id="balanceDataModal" tabindex="-1" aria-labelledby="balanceDataModalLabel" aria-hidden="true">
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="balanceDataModalLabel">数据平衡选项</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <label>选择平衡方式:</label>
            <select v-model="balanceMethod" class="form-control">
              <option value="undersampling">欠采样</option>
              <option value="oversampling">过采样</option>
            </select>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
            <button type="button" class="btn btn-primary" @click="confirmBalanceData">确认</button>
          </div>
        </div>
      </div>
    </div>
   </div>
  </template>
  
  <script>
  import { Modal, Dropdown } from 'bootstrap';
  import * as echarts from 'echarts';
    
  export default {
    data() {
      return {
        columns: ['列1', '列2', '列3'], // 假设有三列
        rows: [
          ['数据1', '数据2', '数据3'],
          ['数据4', '数据5', '数据6'],
          // 更多数据行...
        ],
        isNormalized: false // 数据是否经过标准化
      };
    },
    methods: {
      initializeCharts() {
        // 初始化扇形图
        this.pieChart = echarts.init(document.getElementById('pieChart'));
        const pieOption = {
          title: {
            text: '类别分布',
            left: 'center'
          },
          tooltip: {
            trigger: 'item'
          },
          series: [
            {
              name: '数据量',
              type: 'pie',
              radius: '50%',
              data: [
                { value: 10, name: '类别1' },
                { value: 20, name: '类别2' },
                { value: 30, name: '类别3' }
              ],
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowOffsetX: 0,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
              }
            }
          ]
        };
        this.pieChart.setOption(pieOption);
  
        // 初始化条形图
        this.barChart = echarts.init(document.getElementById('barChart'));
        const barOption = {
          title: {
            text: '类别数据量',
            left: 'center'
          },
          tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' }
          },
          xAxis: {
            type: 'category',
            data: ['类别1', '类别2', '类别3']
          },
          yAxis: {
            type: 'value'
          },
          series: [
            {
              data: [10, 20, 30],
              type: 'bar',
              itemStyle: {
                color: '#4f81bd'
              }
            }
          ]
        };
        this.barChart.setOption(barOption);
      },
      selectDataset(datasetType) {
        alert(`展示${datasetType}数据`);
        // 此处添加后端请求代码以获取指定数据集
        // 后端请求获取指定数据集的逻辑在此添加
      },
      removeDuplicates() {
        alert('去除重复值功能待实现');
        // 添加与后端交互逻辑以删除重复值
        // 请求后端删除重复值并刷新数据
      },
      fillMissingValues() {
        const fillModal = new Modal(document.getElementById('fillMissingModal'));
        fillModal.show();
     },
      confirmFillMissing() {
        alert(`对列 ${this.selectedColumn} 使用 ${this.fillMethod} 填充缺失值`);
      // 填充缺失值的请求逻辑
    },
      balanceData() {
        const balanceModal = new Modal(document.getElementById('balanceDataModal'));
        balanceModal.show();
        // 实现数据平衡逻辑
      },
      confirmBalanceData() {
        alert(`数据平衡方式：${this.balanceMethod}`);
      // 数据平衡请求逻辑
    },
      normalizeData() {
        this.isNormalized = true;
        alert('数据标准化后图表展示');
        this.pieChart.setOption({
          series: [{ data: [{ value: 15, name: '类别1' }, { value: 25, name: '类别2' }, { value: 35, name: '类别3' }] }]
        });
        this.barChart.setOption({
          series: [{ data: [15, 25, 35] }]
        });
      }
    },
    mounted() {
      this.initializeCharts();

      // 手动初始化 dropdown
      const dropdownElement = document.getElementById('datasetDropdown');
      new Dropdown(dropdownElement);
    },
  };
  </script>
  
  <style scoped>

  h2 {
    text-align: center;
  }

  .container {
    padding: 20px;
    max-width: 1000px;
  }
  
  .table {
    margin-top: 20px;
  }
  
  .button-group {
    margin-top: 20px;
  }
  
  .charts-container {
    margin-top: 40px;
  }
  
  .chart-placeholder {
    text-align: center;
    padding: 20px;
    border-radius: 8px;
    background-color: #f9f9f9;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
  }
  
  .chart-placeholder h5 {
    font-weight: bold;
    margin-bottom: 15px;
  }
  
  .chart {
    width: 100%;
    height: 400px;
  }
  </style>
  