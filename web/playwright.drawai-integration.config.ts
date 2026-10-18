import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir:'./tests',testMatch:['drawai-integration.spec.ts','drawai.spec.ts'],timeout:120000,workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/verification/playwright-report.json'}]],
  use:{baseURL:'http://127.0.0.1:4186'},
  projects:[{name:'chrome',use:{browserName:'chromium',channel:'chrome'}},{name:'edge',use:{browserName:'chromium',channel:'msedge'}}],
  webServer:{command:'npm run preview -- --port 4186 --strictPort',url:'http://127.0.0.1:4186',reuseExistingServer:false},
});
