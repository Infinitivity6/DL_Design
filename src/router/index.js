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
  },
  {
    path: '/project/:type',
    name: 'Project',
    component: ProjectComponent,
    props: true, // 允许路由参数作为 props 传递给组件
  },
  {
    path: '/text-processing',
    name: 'TextProcessing',
    component: TextProcessing
  },
  {
    path: '/dataset-display',
    name: 'DatasetDisplay',
    component: DatasetDisplay
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
  