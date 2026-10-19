import {defineConfig} from '@playwright/test';
export default defineConfig({
  testDir:'./tests',testMatch:['drawai.spec.ts','drawai-phase1.spec.ts','drawai-release.spec.ts','drawai-integration.spec.ts','drawai-feasibility.spec.ts','drawai-p6.spec.ts'],
  timeout:120000,workers:1,reporter:[['list'],['json',{outputFile:'../artifacts/drawai/phase6-image64/product-regression.json'}]],
  use:{baseURL:'http://127.0.0.1:4186',browserName:'chromium'},
  webServer:{command:'npm run preview -- --port 4186 --strictPort',env:{DRAWAI_P6_LOCAL_SMOKE:'1'},url:'http://127.0.0.1:4186',reuseExistingServer:false},
});
