import { createRouter, createWebHistory } from 'vue-router';
import HomeComponent from '../components/Home.vue';
import ProjectComponent from '../components/Project.vue';
import TextProcessing from '@/components/TextProcessing.vue';
import DatasetDisplay from '@/components/DatasetDisplay.vue';

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomeComponent,
    meta: {title: '首页'}
  },
  {
    path: '/project/:type',
    name: 'Project',
    component: ProjectComponent,
    props: true, // 允许路由参数作为 props 传递给组件
    meta: {title: '项目'}
  },
  {
    path: '/text-processing',
    name: 'TextProcessing',
    component: TextProcessing,
    meta: {title: '文本处理'}
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
  