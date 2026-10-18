import { defineConfig } from '@playwright/test';
process.env.DRAWAI_REGRESSION_REPORT_DIR='../artifacts/drawai/export/d2-regression';
export default defineConfig({
  testDir:'./tests',testMatch:['drawai-release.spec.ts','drawai-feasibility.spec.ts'],timeout:120000,workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/export/playwright-report.json'}]],
  use:{baseURL:'http://127.0.0.1:4185',browserName:'chromium'},
  webServer:{command:'npm run preview -- --port 4185 --strictPort',url:'http://127.0.0.1:4185',reuseExistingServer:false},
});
