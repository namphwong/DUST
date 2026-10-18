import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir:'./tests',testMatch:'drawai-feasibility.spec.ts',timeout:120000,workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/feasibility/playwright-report.json'}]],
  use:{baseURL:'http://127.0.0.1:4184',browserName:'chromium'},
  webServer:{command:'npm run preview -- --port 4184 --strictPort',url:'http://127.0.0.1:4184',reuseExistingServer:false},
});
