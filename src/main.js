import { createApp } from 'vue'
import App from './App.vue'
import router from './router'; // 引入你刚刚创建的 router

// 引入 Bootstrap 的 CSS
import 'bootstrap/dist/css/bootstrap.min.css'

// 引入 Bootstrap 的 JS
import 'bootstrap/dist/js/bootstrap.bundle.min.js'

import 'bootstrap/dist/css/bootstrap.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import 'bootstrap/dist/js/bootstrap.bundle'; // 包含 Popper 和 Bootstrap JS
import '@fortawesome/fontawesome-free/css/all.css';

createApp(App)
.use(router) //使用路由
.mount('#app')
