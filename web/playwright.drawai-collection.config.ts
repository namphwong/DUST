import {defineConfig} from '@playwright/test';
export default defineConfig({
  testDir:'./tests',testMatch:'drawai-collection.spec.ts',timeout:120000,workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/phase3/browser-tests.json'}]],
  use:{baseURL:'http://127.0.0.1:4193',browserName:'chromium',acceptDownloads:true},
  webServer:{command:'npm run preview -- --port 4193 --strictPort',url:'http://127.0.0.1:4193',reuseExistingServer:false},
});
