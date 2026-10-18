import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir:'./tests',testMatch:'drawai-phase1.spec.ts',timeout:120000,workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/phase1/route-tests.json'}]],
  use:{baseURL:'http://127.0.0.1:4187',browserName:'chromium'},
  webServer:{command:'npm run preview -- --port 4187 --strictPort',url:'http://127.0.0.1:4187',reuseExistingServer:false},
});
