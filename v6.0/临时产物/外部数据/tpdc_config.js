/*
 * @Author: panrong
 * @Date: 2022-01-11 10:19:12
 * @LastEditTime: 2023-10-13 14:09:11
 * @LastEditors: yintao
 * @Description: 线上部署环境配置文件
 * @FilePath: \QZGY_WebView\public\config.js
 */
window.proConfig = {
  VUE_APP_BASEURL_DATA: '/view',
  VUE_APP_BASEURL_PERSONAL: '/my',
  VUE_APP_BASEURL_OPERATION: '/admin',
  VUE_APP_BASEURL_ENCLOSURE: '/file',
  VUE_APP_BASEURL_MAP: '/geoserver',
  VUE_APP_BASEURL_ROR: '/ror',

  // 一段时间没请求接口，则退出登录状态   单位是 秒   24小时
  noOperationTime: 24 * 60 * 60,

  // 登录时后端返回的token剩余有效时间和实际刷新token的时间预留的时间差   单位是 秒   5分钟
  lossTime: 5 * 60,

  // 数据中心/个人中心跳转运控中心时是否需要从域名(公网)跳转到IP(内网)
  isDomainToIP: true,

  // 各中心地址的经纬度
  mapCenters: {
    qz: [116.37009, 40.00408,],
    // qz: [116.37609, 40.00538,],
    qh: [101.755971,36.749079,],
    tibet: [91.062429,29.600718,],
    cd: [104.116307,30.408475,],
  },

  // 是否需要第三方登录   true 需要    false 不需要
  needThirdPartyLogin: true,
  // 首页是否需要关键数据集模块   true 需要    false 不需要
  needKeyDataset: true,

  doiCodes: {
    qz: '.tpdc.',
    qh: '.tpdc.qh.',
    tibet: '.tpdc.tibet.',
    cd: '.tpdc.cd.',
  },

  // 当前中心  必须是 'qz' | 'qh' | 'tibet' | 'cd'
  curCenter: 'qz',
};
