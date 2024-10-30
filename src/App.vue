<template>
  <div id="app">
    <!-- 路由出口，将显示匹配的组件 -->
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App',
};
</script>

<style>



#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;

  /* 10.25开发 新增背景色 */
  background-color: #E8EAED; /* 可以把主页面背景换成别的颜色 */
  min-height: 100vh; /* 确保背景色覆盖整个页面 */
  margin: 0; /* 移除默认的外边距 */
  padding: 0; /* 移除默认的内边距 */
}
</style>
