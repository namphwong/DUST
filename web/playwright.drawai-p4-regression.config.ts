import {defineConfig} from '@playwright/test';
export default defineConfig({
  testDir:'./tests',testMatch:['drawai.spec.ts','drawai-phase1.spec.ts'],timeout:120000,workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/phase4/product-regression.json'}]],
  use:{baseURL:'http://127.0.0.1:4186',browserName:'chromium'},
  webServer:{command:'npm run preview -- --port 4186 --strictPort',url:'http://127.0.0.1:4186',reuseExistingServer:false},
});
