<!-- DatasetDisplay.vue 代码 -->
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
            {{ selectDatasetName}}
          </button>
          <ul class="dropdown-menu" aria-labelledby="datasetDropdown">
            <li><a class="dropdown-item" @click="selectDataset('train')">训练集</a></li>
            <li><a class="dropdown-item" @click="selectDataset('validation')">验证集</a></li>
            <li><a class="dropdown-item" @click="selectDataset('test')">测试集</a></li>
            <li><a class="dropdown-item" @click="selectDataset('all')">所有数据</a></li>
          </ul>
        </div>
      </div>
  

      <!-- 完善数据集展示功能 -->
        <!-- 单个数据集的展示 (v-if="!isAll") -->
    <div v-if="!isAll">
      <!-- 表格展示 -->
      <div class="text-center my-3">
            <h4>部分数据展示</h4>
        </div>
      <div class="table-responsive">
         <table class="table table-bordered table-striped table-hover">
          <caption class="text-muted small">以上仅展示前 10 行数据</caption>

            <thead>
              <tr>
                <th v-for="(col, cidx) in singleData.columns" :key="cidx">{{ col }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, ridx) in singleData.rows" :key="ridx">
                <td v-for="(cell, ccidx) in row" :key="ccidx">{{ cell }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      




      <!-- 基本信息 -->
      <div class="dataset-info mt-4 card p-3" v-if="singleData.info">
        <h4 class="mb-3">数据基本信息</h4>
        <div class="row mb-2">
          <div class="col-md-4 text-center">
            <h6>行数</h6>
            <p>{{ singleData.info.row_count }}</p>
          </div>
          <div class="col-md-4 text-center">
            <h6>列数</h6>
            <p>{{ singleData.info.col_count }}</p>
          </div>
          <div class="col-md-4 text-center">
            <h6>重复值</h6>
            <p>{{ singleData.info.duplicate_count }}</p>
          </div>
        </div>

        <div class="row">
          <!-- 每列类型 -->
          <div class="col-md-6">
            <h6>每列类型</h6>
            <table class="table table-sm table-bordered">
              <thead>
                <tr><th>列名</th><th>类型</th></tr>
              </thead>
              <tbody>
                <tr v-for="(dtype, colName) in singleData.info.dtypes" :key="colName">
                  <td>{{ colName }}</td>
                  <td>{{ dtype }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 缺失值 -->
          <div class="col-md-6">
            <h6>缺失值</h6>
            <table class="table table-sm table-bordered">
              <thead>
                <tr><th>列名</th><th>缺失数量</th></tr>
              </thead>
              <tbody>
                <tr v-for="(missingCount, colName) in singleData.info.missing" :key="colName">
                  <td>{{ colName }}</td>
                  <td>{{ missingCount }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- 图表展示 -->
      <div class="charts-container mt-5">
        <h4 class="text-center">数据统计图</h4>
        <div class="chart-placeholder">
          <h5>扇形图</h5>
          <div id="pieChartSingle" class="chart"></div>
        </div>
        <div class="chart-placeholder mt-4">
          <h5>条形图</h5>
          <div id="barChartSingle" class="chart"></div>
        </div>
      </div>
    </div>

    <!-- 多个数据集的展示 (v-if="isAll") -->
    <div v-else>
      <div
        v-for="ds in allDatasetInfo"
        :key="ds.name"
        class="mb-5"
      >
        <h3 class="mt-3">{{ ds.name }} 数据集</h3>

        <!-- 如果 ds.error 或 ds.message，说明未上传或出错 -->
        <div v-if="ds.error || ds.message" class="alert alert-warning">
          {{ ds.error || ds.message }}
        </div>

        <div v-else>
          <!-- 表格 -->
          <div class="table-responsive">
            <table class="table table-bordered table-striped table-hover">
              <caption class="text-muted small">以下仅展示前 10 行数据</caption>

              <thead>
                <tr>
                  <th v-for="(col, cidx) in ds.preview.columns" :key="cidx">{{ col }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, ridx) in ds.preview.rows" :key="ridx">
                  <td v-for="(cell, ccidx) in row" :key="ccidx">{{ cell }}</td>
                </tr>
              </tbody>
            </table>
          </div>



          <!-- 基本信息 -->
          <div class="dataset-info mt-4" v-if="ds.info">
            <h5 class="mb-3">数据基本信息</h5>
            <div class="row mb-2">
              <div class="col-md-4 text-center">
                <h6>行数</h6>
                <p>{{ ds.info.row_count }}</p>
              </div>
              <div class="col-md-4 text-center">
                <h6>列数</h6>
                <p>{{ ds.col_count }}</p>
              </div>
              <div class="col-md-4 text-center">
                <h6>重复值</h6>
                <p>{{ ds.info.duplicate_count }}</p>
              </div>
            </div>

            <div class="row">
              <!-- 每列类型 -->
              <div class="col-md-6">
                <h6>每列类型</h6>
                <table class="table table-sm table-bordered">
                  <thead>
                    <tr><th>列名</th><th>类型</th></tr>
                  </thead>
                  <tbody>
                    <tr v-for="(dtype, colName) in ds.info.dtypes" :key="colName">
                      <td>{{ colName }}</td>
                      <td>{{ dtype }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <!-- 缺失值 -->
              <div class="col-md-6">
                <h6>缺失值</h6>
                <table class="table table-sm table-bordered">
                  <thead>
                    <tr><th>列名</th><th>缺失数量</th></tr>
                  </thead>
                  <tbody>
                    <tr v-for="(missingCount, colName) in ds.info.missing" :key="colName">
                      <td>{{ colName }}</td>
                      <td>{{ missingCount }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- 图表 -->
          <div class="charts-container mt-4">
            <h4>数据统计图</h4>
            <div class="chart-placeholder">
              <h5>扇形图</h5>
              <!-- 每个数据集使用不同的id -->
              <div :id="'pieChart-' + ds.name" class="chart"></div>
            </div>
            <div class="chart-placeholder mt-4">
              <h5>条形图</h5>
              <div :id="'barChart-' + ds.name" class="chart"></div>
            </div>
          </div>
        </div>
      </div>
    </div>






  
      <!-- 按钮组 数据预处理 -->
      <div class="text-center my-4">
        <button class="btn btn-warning mx-2" @click="removeDuplicates">去除重复值</button>
        <button class="btn btn-info mx-2" @click="fillMissingValues">缺失值填充</button>
        <button class="btn btn-success mx-2" @click="balanceData">数据平衡</button>
        <button class="btn btn-primary mx-2" @click="normalizeData">数据标准化</button>
        <button class="btn btn-primary mx-2" @click="test">测试</button>
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

    <!-- 3.13新增: 缺失值详情 Modal -->
<div class="modal fade" id="missingDetailModal" tabindex="-1" aria-labelledby="missingDetailModalLabel" aria-hidden="true">
  <div class="modal-dialog modal-lg">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="missingDetailModalLabel">缺失值详情</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <!-- 这里分 train/validation/test 三个面板 -->
        <div v-for="ds in ['train','validation','test']" :key="ds" class="mb-4">
          <h5>{{ ds }} 数据集</h5>
          <div v-if="missingDetails[ds]?.message">
            <p class="text-info">{{ missingDetails[ds].message }}</p>
          </div>
          <div v-else-if="missingDetails[ds]?.missingCells">
            <div class="d-flex justify-content-end mb-2">
              <!-- 一键填充 / 删除按钮 -->
              <button class="btn btn-sm btn-success me-2" @click="setFillMode(ds,'auto')">一键填充</button>
              <button class="btn btn-sm btn-danger me-2" @click="setFillMode(ds,'delete')">删除所有缺失行</button>
              <button class="btn btn-sm btn-secondary" @click="setFillMode(ds,'manual')">逐个填充</button>
            </div>
            <p>当前模式: {{ fillMode[ds] }}</p>
            <!-- 如果 fillMode=='manual', 显示逐个cell的选择UI -->
            <div v-if="fillMode[ds]==='manual'">
              <!-- 3x3 网格显示 missingCells -->
              <div class="row row-cols-3 g-2">
                <div class="col" v-for="(cell, idx) in missingDetails[ds].missingCells" :key="idx">
                  <div class="border p-2">
                    <p>行: {{ cell.row }}, 列: {{ cell.colName }}</p>
                    <!-- 下拉选择 method -->
                    <select class="form-select form-select-sm mb-1" @change="onMethodChange(ds, cell, $event)">
                      <option value="mean">均值</option>
                      <option value="interpolation">插值</option>
                      <option value="specific">特定值</option>
                    </select>
                    <!-- 如果 method == 'specific', 再出现 input -->
                    <input type="text" class="form-control form-control-sm" placeholder="填充值" v-if="cell._method==='specific'" @input="onFillValueChange(ds, cell, $event)"/>
                  </div>
                </div>
              </div>
              <!-- 加载更多 -->
              <div v-if="missingDetails[ds].hasMore" class="mt-2">
                <button class="btn btn-sm btn-outline-primary" @click="loadMoreMissing(ds)">加载更多</button>
              </div>
            </div>
            <!-- 如果 fillMode=='auto' or 'delete', 就提示用户: 不需逐个选 -->
            <div v-else>
              <p>已选择{{ fillMode[ds]==='auto'?'一键填充':'删除缺失行' }}模式</p>
            </div>
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <!-- 点击确认后, 调用 confirmFillAll() -->
        <button type="button" class="btn btn-primary" @click="confirmFillAll">确认填充</button>
      </div>
    </div>
  </div>
</div>


    <!-- 模态框：数据标准化选项 -->
<div class="modal fade" id="normalizeDataModal" tabindex="-1" aria-labelledby="normalizeDataModalLabel" aria-hidden="true">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="normalizeDataModalLabel">数据标准化</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <label>选择标准化方式:</label>
        <select v-model="normalizeMethod" class="form-control">
          <option value="zscore">Z-score 标准化</option>
          <option value="minmax">Min-Max 归一化</option>
        </select>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
        <button type="button" class="btn btn-primary" @click="confirmNormalizeData">确认</button>
      </div>
    </div>
  </div>
</div>








   </div>
  </template>
  
  <script>
  import axios from 'axios';
  import Swal from 'sweetalert2'
  import { Modal, Dropdown } from 'bootstrap';
  import * as echarts from 'echarts';
    
  export default {
    data() {
      return {
        selectDatasetName: '请选择数据集', //下拉按钮显示的文字，默认为请选择数据集

         // 单个数据集展示
        isAll: false,
        singleData: {
        columns: [],
        rows: [],
        info: null,
        label_distribution: {}
      },

      // 多个数据集展示
      allDatasetInfo: [],

      // ECharts 实例,扇形图和条形图
      singlePieChart: null,
      singleBarChart: null,
      isNormalized: false, // 数据是否经过标准化

      // 缺失值填充相关
      // ----------缺失值填充相关开始
      missingModal: null, // 用来存储 '缺失值详情' 模态的实例
      missingDetails: {
      // 形如: {
      //   train: {
      //     missingCells: [],
      //     totalMissingCount: 0,
      //     hasMore: false,
      //     currentPage: 1,
      //     message: "...",
      //   },
      //   validation: {...},
      //   test: {...}
      // }
    },

    // 当前选中的 datasetType，用于获取缺失值
    fillSelectedDataset: null,

    // 用户选择的填充方式: auto/delete/manual
    fillMode: {
      train: 'manual',
      validation: 'manual',
      test: 'manual'
    },

    // 如果 fillMode=='manual', 需要逐个cell存储用户选择
    // 例如 fillCells[train] = [ {row, colName, method, fillValue}, ... ]
    fillCells: {
      train: [],
      validation: [],
      test: []
    },

    // 分页大小
    pageSize: 9,
    // -----------缺失值填充相关结束

    balanceMethod: 'undersampling', // 默认欠采样 or oversampling

    normalizeMethod: 'zscore' // 默认zscore或minmax







      };
    },
    methods: {
      test(){
      this.$router.push({ name: 'TrainingOutcome' });
 
      },
      // 缺失值填充相关函数
      onMethodChange(ds, cell, event) {
        const method = event.target.value; // "mean"/"interpolation"/"specific"
        cell._method = method;

        // 记录到 fillCells
        this.chooseCellMethod(ds, cell.row, cell.colName, method, null);
      },
      onFillValueChange(ds, cell, event) {
        const val = event.target.value;
        // 只当 _method=='specific'
        this.chooseCellMethod(ds, cell.row, cell.colName, 'specific', val);
      },

      chooseCellMethod(ds, row, colName, method, fillValue) {
        if(!this.fillCells[ds]) {
          this.fillCells[ds] = [];
         }
        const arr = this.fillCells[ds];
        const idx = arr.findIndex(c => c.row===row && c.colName===colName);
        if(idx>=0) {
          arr[idx].method = method;
          arr[idx].fillValue = fillValue;
        } else {
          arr.push({ row, colName, method, fillValue });
        }
      },

      // 展示数据集的相关函数
      selectDataset(datasetType) {

        // 0) 在此之前可以更新选择栏的显示内容
        const mapTypeToName = {
          train: '训练集',
          validation: '验证集',
          test: '测试集',
          all: '所有数据'
        };
        this.selectDatasetName = mapTypeToName[datasetType] || '请选择数据集';


         // 1) 先清空旧数据
      this.isAll = (datasetType === 'all');
      this.singleData = { columns: [], rows: [], info: null, label_distribution: {} };
      this.allDatasetInfo = [];

      // 2) 发请求
      axios.get('/api/dataDisplay/dataset', {
        params: { type: datasetType }
      })
      .then(res => {
        const data = res.data;
        if (data.message) {
          // 说明后端返回 { "message": "...(错误/未上传)" }
          alert(data.message);
          return;
        }

        if (datasetType === 'all') {
          // data = { train: {...}, validation: {...}, test: {...} }
          this.handleAllDataset(data);
        } else {
          // data = { preview, info, label_distribution }
          this.handleSingleDataset(data);
        }
      })
      .catch(err => {
        console.error(err);
        alert(err.response?.data?.message || '获取数据失败');
      });     
      },

      // 处理单个数据集情况
       handleSingleDataset(data) {
      if (data.error) {
        alert(data.error);
        return;
      }
      // data.preview.columns, data.preview.rows
      this.singleData.columns = data.preview.columns;
      this.singleData.rows = data.preview.rows;
      this.singleData.info = data.info;
      this.singleData.label_distribution = data.label_distribution || {};

      // 更新图表
      this.updateSingleCharts();
    },


    updateSingleCharts() {
      // 初始化 / 更新单个数据集图表
      if (!this.singlePieChart) {
        // 第一次时初始化
        this.singlePieChart = echarts.init(document.getElementById('pieChartSingle'));
        this.singleBarChart = echarts.init(document.getElementById('barChartSingle'));

        // 设置基本option
        this.singlePieChart.setOption({
          title: { text: '标签分布', left: 'center' },
          tooltip: { trigger: 'item' },
          series: [{ type: 'pie', radius: '50%', data: [] }]
        });

        this.singleBarChart.setOption({
          title: { text: '标签数量', left: 'center' },
          tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
          xAxis: { type: 'category', data: [] },
          yAxis: { type: 'value' },
          series: [{ type: 'bar', data: [] }]
        });
      }

      // 将 label_distribution => ECharts data
      const dist = this.singleData.label_distribution;
      const categories = Object.keys(dist);
      const values = Object.values(dist);
      const pieData = categories.map((k, i) => ({ name: k, value: values[i] }));

      this.singlePieChart.setOption({
        series: [{
          data: pieData
        }]
      });

      this.singleBarChart.setOption({
        xAxis: { data: categories },
        series: [{ data: values }]
      });
    },

    handleAllDataset(allObj) {
      // allObj = { train: {...}, validation: {...}, test: {...} }
      // 我们要把train/validation/test转换为一个数组, 方便 v-for 渲染
      this.allDatasetInfo = [];

      const dsNames = ['train', 'validation', 'test'];
      dsNames.forEach(dsName => {
        const dsData = allObj[dsName];
        // dsData可能是 { preview, info, label_distribution } 或 { message: "...尚未上传" }
        if (dsData.error || dsData.message) {
          // 说明未上传或读取失败
          this.allDatasetInfo.push({
            name: dsName,
            error: dsData.error,
            message: dsData.message
          });
        } else {
          // 正常数据集
          this.allDatasetInfo.push({
            name: dsName,
            preview: dsData.preview,
            info: dsData.info,
            label_distribution: dsData.label_distribution
          });
        }
      });

      // 等DOM更新后初始化每个数据集的图表
      this.$nextTick(() => {
        this.allDatasetInfo.forEach(ds => {
          if (!ds.error && !ds.message) {
            this.initAllChartsForDataset(ds);
          }
        });
      });
    },
    


      




    initAllChartsForDataset(ds) {
      // ds.name: train / validation / test
      // ds.label_distribution: {A:10, B:5, ...}
      const pieId = 'pieChart-' + ds.name;
      const barId = 'barChart-' + ds.name;

      const pieChart = echarts.init(document.getElementById(pieId));
      const barChart = echarts.init(document.getElementById(barId));

      // 设置初始option
      pieChart.setOption({
        title: { text: ds.name + ' 标签分布', left: 'center' },
        tooltip: { trigger: 'item' },
        series: [{
          type: 'pie', radius: '50%', data: []
        }]
      });
      barChart.setOption({
        title: { text: ds.name + ' 标签数量', left: 'center' },
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        xAxis: { type: 'category', data: [] },
        yAxis: { type: 'value' },
        series: [{ type: 'bar', data: [] }]
      });

      const dist = ds.label_distribution || {};
      const categories = Object.keys(dist);
      const values = Object.values(dist);
      const pieData = categories.map((k, i) => ({ name: k, value: values[i] }));

      pieChart.setOption({
        series: [{
          data: pieData
        }]
      });
      barChart.setOption({
        xAxis: { data: categories },
        series: [{ data: values }]
      });
    },
      
    removeDuplicates() {
      axios.post('/api/dataPreprocess/remove_duplicates')
        .then(res => {
          // 后端返回 {train:{before, after, removed}, validation:{...}, test:{...}}
          // 或 {error:"请上传至少一个数据集..."}
          const data = res.data
          if (data.message || data.error) {
            // 如果是400 或 error
            // data.error 可能是 "请上传至少一个数据集后再进行去重"
            this.showSweetAlert('去重失败', data.message || data.error, 'error')
          } else {
            // 成功, data形如:
            // {
            //   "train": { "before": 100, "after":95, "removed":5 },
            //   "validation": { "message": "validation 数据集尚未上传" },
            //   "test": { "before":200, "after":198, "removed":2 }
            // }
            let htmlStr = ''
            Object.keys(data).forEach(ds => {
              const info = data[ds]
              if (info.error) {
                htmlStr += `<p><b>${ds}</b> 出错: ${info.error}</p>`
              } else if (info.message) {
                htmlStr += `<p><b>${ds}</b>: ${info.message}</p>`
              } else {
                // {before, after, removed}
                htmlStr += `<p><b>${ds}</b>: 去重前 ${info.before} 行, 去重后 ${info.after} 行, 删除 ${info.removed} 行</p>`
              }
            })
            // 显示成功弹窗
            this.showSweetAlert('去重成功', htmlStr, 'success')
          }
        })
        .catch(err => {
          console.error(err)
          const msg = err.response?.data?.message || '去重时发生错误'
          this.showSweetAlert('去重失败', msg, 'error')
        })
    },

    showSweetAlert(title, html, icon) {
      // SweetAlert2
      Swal.fire({
        title,
        html,
        icon,   // 'success', 'error', 'warning', 'info'...
        showCloseButton: true,
        focusConfirm: false,
        confirmButtonText: '确定'
      })
    },

      //填充缺失值
      fillMissingValues() {
    // 1) 决定要获取 train/validation/test 全部, 还是让用户再选?
    //   如果只想一次获取全部, 
    axios.get('/api/dataPreprocess/missing_details', {
      params: {
        // dataset: null, // 表示获取 train/validation/test
        currentPage: 1,
        pageSize: this.pageSize
      }
    })
    .then(res => {
      this.missingDetails = res.data; 
      // 形如 { train: {...}, validation: {...}, test: {...} }
      // 后端可能返回 { train:{ message:"没缺失" }, validation:{...}, test:{...} }
      // 下面可以展示一个新的 Modal 或 切换到一个"缺失值管理"视图
      this.showMissingModal(); 
    })
    .catch(err => {
      console.error(err);
      this.showSweetAlert('获取缺失值失败', err.response?.data?.message || '请求错误', 'error');
    });
   },

  showMissingModal() {
    const modalEl = document.getElementById('missingDetailModal');
   // 创建并显示实例
    this.missingModal = new Modal(modalEl);
    this.missingModal.show();
  },

  // 请求下一页
  loadMoreMissing(ds) {
    const info = this.missingDetails[ds];
    if (!info || !info.hasMore) return;
    const nextPage = info.currentPage + 1;
    axios.get('/api/dataPreprocess/missing_details', {
      params: {
        dataset: ds,
        currentPage: nextPage,
        pageSize: this.pageSize
      }
    })
    .then(res => {
      // 合并 missingCells
      const newInfo = res.data[ds];
      if (newInfo.missingCells) {
        info.missingCells = info.missingCells.concat(newInfo.missingCells);
        info.currentPage = newInfo.currentPage;
        info.hasMore = newInfo.hasMore;
      }
    })
    .catch(err => {
      console.error(err);
      this.showSweetAlert('加载更多失败', err.response?.data?.message || '请求错误', 'error');
    });
  },

  // 当用户选择 "一键填充" / "删除"
  setFillMode(ds, mode) {
    this.fillMode[ds] = mode; // 'auto' or 'delete'
  },


  // 最后用户点击"确认填充"
  confirmFillAll() {
    // 构建 instructions
    const instructions = {};
    for (const ds of ['train','validation','test']) {
      const mode = this.fillMode[ds];
      if (!this.missingDetails[ds]) continue; // 说明这个ds可能没上传
      if (mode==='auto' || mode==='delete') {
        instructions[ds] = { fillMode: mode };
      } else {
        instructions[ds] = {
          fillMode: 'manual',
          cells: this.fillCells[ds] || []
        };
      }
    }

    axios.post('/api/dataPreprocess/apply_fill', { instructions })
    .then(res => {
      const data = res.data;
      // 构建一个更易读的 htmlStr
      let htmlStr = '';
      // 遍历 train/validation/test
      for (const ds in data) {
        // data[ds] 可能是 { "message": "..."} / { "status":"ok", "filledCount":5 } / ...
        if (data[ds].error) {
          htmlStr += `<p><b>${ds}</b> 出错: ${data[ds].error}</p>`;
        } else if (data[ds].message) {
          // 例如 "test 数据集尚未上传，无法填充"
          htmlStr += `<p><b>${ds}</b>: ${data[ds].message}</p>`;
        } else if (data[ds].status === 'ok') {
          // 可能是一键填充 autoFilled / 删除 deleteRows / 手动 filledCount
          if (data[ds].autoFilled) {
            htmlStr += `<p><b>${ds}</b> 一键填充成功，共填充了 ${data[ds].autoFilled} 个缺失值</p>`;
          } else if (data[ds].deletedRows) {
            htmlStr += `<p><b>${ds}</b> 已删除所有含缺失值的行，共删除 ${data[ds].deletedRows} 行</p>`;
          } else if (data[ds].filledCount) {
            htmlStr += `<p><b>${ds}</b> 手动填充了 ${data[ds].filledCount} 个缺失值</p>`;
          } else {
            htmlStr += `<p><b>${ds}</b> 填充完成</p>`;
          }
        }
      }

  // 用这个 htmlStr 显示成功弹窗
  this.showSweetAlert('填充完成', htmlStr, 'success');

  // 关闭模态
  if (this.missingModal) {
    this.missingModal.hide();
    this.missingModal = null;
  }

  // 若需要刷新表格或重新获取数据，也可以在这里调用（后续优化考虑）
})
      .catch(err => {
        console.error(err);
        this.showSweetAlert('填充失败', err.response?.data?.message || '请求错误', 'error');
      });
  },


      balanceData() {
        const balanceModal = new Modal(document.getElementById('balanceDataModal'));
        balanceModal.show();
        // 实现数据平衡逻辑
      },

      confirmBalanceData() {
      // 用户在下拉中选了 "undersampling" or "oversampling"
      // 发请求给后端
      axios.post('/api/dataPreprocess/balance', {
        method: this.balanceMethod
      })
      .then(res => {
        const data = res.data;
        // data 形如:
        // {
        //   "train": { "message":"无需数据平衡" } 或 { "status":"ok", "before":1000,"after":600,"removed":400,"added":0 },
        //   "validation": {...},
        //   "test": {...}
        // }
        let htmlStr = '';
        for(const ds of ['train','validation','test']) {
          if(data[ds]) {
            if(data[ds].message) {
              htmlStr += `<p><b>${ds}</b>: ${data[ds].message}</p>`;
            } else if(data[ds].status==='ok') {
              // "before","after","removed","added"
              const bef = data[ds].before;
              const aft = data[ds].after;
              const rem = data[ds].removed || 0;
              const add = data[ds].added || 0;
              htmlStr += `<p><b>${ds}</b>: 平衡前 ${bef} 行, 平衡后 ${aft} 行`;
              if(rem>0) {
                htmlStr += `, 删除 ${rem} 行`;
              }
              if(add>0) {
                htmlStr += `, 增加 ${add} 行`;
              }
              htmlStr += `</p>`;
            } else {
              // 其它情况
              htmlStr += `<p><b>${ds}</b>: 未知状态</p>`;
            }
          } else {
            // data里没这个ds
            htmlStr += `<p><b>${ds}</b>: 未上传或无信息</p>`;
          }
        }

        // 用 SweetAlert2 显示
        this.showSweetAlert('数据平衡完成', htmlStr, 'success');

            // 关闭Modal
          const modalEl = document.getElementById('balanceDataModal');
          const bsModal = Modal.getInstance(modalEl);
          if (bsModal) {
            bsModal.hide();
          }

      })
      .catch(err => {
        console.error(err);
        const msg = err.response?.data?.message || '数据平衡时发生错误';
        this.showSweetAlert('数据平衡失败', msg, 'error');
      });
    },




      
      normalizeData() {
      // 打开 "数据标准化" 模态框
      const normModal = new Modal(document.getElementById('normalizeDataModal'));
      normModal.show();
      },

      confirmNormalizeData() {
      // 用户在下拉里选了 zscore 或 minmax
      axios.post('/api/dataPreprocess/normalize', {
        method: this.normalizeMethod
      })
      .then(res => {
        const data = res.data;
        // data 形如:
        // {
        //   "train": {"message":"xx 数据集未上传"} 或 { "status":"ok","num_processed":3,"skipped":2 },
        //   "validation": {...},
        //   "test": {...}
        // }
        let htmlStr = '';
        for(const ds of ['train','validation','test']) {
          if(data[ds]) {
            if(data[ds].message) {
              htmlStr += `<p><b>${ds}</b>: ${data[ds].message}</p>`;
            } else if(data[ds].status==='ok') {
              // num_processed, skipped
              const np = data[ds].num_processed;
              const sk = data[ds].skipped;
              htmlStr += `<p><b>${ds}</b>: 成功对 ${np} 个数值列进行标准化, 跳过 ${sk} 个列</p>`;
            } else {
              htmlStr += `<p><b>${ds}</b>: 未知状态</p>`;
            }
          } else {
            htmlStr += `<p><b>${ds}</b>: 无信息</p>`;
          }
        }

        // 显示弹窗
        this.showSweetAlert('标准化完成', htmlStr, 'success');
        // 自动关闭
        const modalEl = document.getElementById('normalizeDataModal');
        const bsModal = Modal.getInstance(modalEl);
        if(bsModal) {
          bsModal.hide();
        }
      })
      .catch(err => {
        console.error(err);
        const msg = err.response?.data?.message || '标准化时发生错误';
        this.showSweetAlert('标准化失败', msg, 'error');
      });
    },


    },
    mounted() {
      

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

  .dataset-info.card {
  background-color: #cfd1d7;
  border: 1px solid #0bbbcb;
  border-radius: 8px;
}

  </style>
  