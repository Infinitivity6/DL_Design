const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true
      },

      // // 当请求地址以 /data 开头时，代理到后端 127.0.0.1:5000
      // '/data': {
      //   target: 'http://127.0.0.1:5000', // 后端地址
      //   changeOrigin: true              // 允许改变 Origin头
      //   // pathRewrite: { '^/data': '/data' } 
      // },
      // '/dataDisplay': {
      //   target: 'http://127.0.0.1:5000',
      //   changeOrigin: true
      // }
      
    }
  }
})
