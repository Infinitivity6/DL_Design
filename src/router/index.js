import { createRouter, createWebHistory } from 'vue-router';
import HomeComponent from '../components/Home.vue';
import DatasetDisplay from '@/components/DatasetDisplay.vue';
import TrainingOutcome_ClassificationNum from '@/components/TrainingOutcome_ClassificationNum.vue';
import ClassificationNumData from '../components/ClassificationNumData.vue';
import ClassificationImageData from '@/components/ClassificationImageData.vue';
import TrainingOutCome_ClassificationImage from '@/components/TrainingOutCome_ClassificationImage.vue';
import ClassificationTextData from '@/components/ClassificationTextData.vue';
import TrainingOutCome_ClassificationText from '@/components/TrainingOutCome_ClassificationText.vue';

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomeComponent,
    meta: {title: '首页'}
  },

  {
    path: '/training-outcome-classification-num',
    name: 'TrainingOutcome_ClassificationNum',
    component: TrainingOutcome_ClassificationNum,
    meta: {title: '数值分类训练结果'}
  },
  {
    path: '/training-outcome-classification-image',
    name: 'TrainingOutcome_ClassificationImage',
    component: TrainingOutCome_ClassificationImage,
    meta: {title: '图像分类训练结果'}
  },
  {
    path: '/training-outcome-classification-text',
    name: 'TrainingOutcome_ClassificationText',
    component: TrainingOutCome_ClassificationText,
    meta: {title: '文字分类训练结果'}
  },


  {
    path: '/num-data-processing-classification',
    name: 'ClassificationNumData',
    component: ClassificationNumData,
    meta: {title: '数值项目分类处理'}
  },
  {
    path: '/image-data-processing-classification',
    name: 'ClassificationImageData',
    component: ClassificationImageData,
    meta: {title: '图像分类处理'}
  },
  {
    path: '/text-data-processing-classification',
    name: 'ClassificationTextData',
    component: ClassificationTextData,
    meta: {title: '文本分类处理'}
  },
  {
    path: '/dataset-display',
    name: 'DatasetDisplay',
    component: DatasetDisplay,
    meta: {title: '数据集展示'}
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});


router.afterEach((to) => {
  // 在路由配置好之后，添加一个 afterEach 守卫
  // 如果该路由设置了 meta.title，就用它，否则用一个默认值
  document.title = to.meta.title || '多任务通用学习平台';
});

export default router;
  