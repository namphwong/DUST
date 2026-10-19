import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',testMatch:['drawai-p6-closed.spec.ts'],workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/phase6-image64/gate-regression.json'}]],
  use:{baseURL:'http://127.0.0.1:4198'},
  webServer:{command:'npm run preview -- --port 4198 --strictPort',env:{DRAWAI_P6_LOCAL_SMOKE:'0'},url:'http://127.0.0.1:4198',reuseExistingServer:false}});
