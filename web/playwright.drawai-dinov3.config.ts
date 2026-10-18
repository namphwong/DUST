import {defineConfig} from '@playwright/test';
process.env.DRAWAI_P2_LOCAL_SMOKE='1';
export default defineConfig({
  testDir:'./tests',testMatch:'drawai-dinov3.spec.ts',timeout:180000,workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/phase2/browser-tests.json'}]],
  use:{baseURL:'http://127.0.0.1:4188',browserName:'chromium'},
  webServer:{command:'npm run preview -- --port 4188 --strictPort',url:'http://127.0.0.1:4188',reuseExistingServer:false},
});
