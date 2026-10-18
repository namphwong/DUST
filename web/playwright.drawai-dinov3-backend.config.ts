import {defineConfig} from '@playwright/test';
import {resolve} from 'node:path';
process.env.DRAWAI_P2_LOCAL_SMOKE='1';
export default defineConfig({
  testDir:'./tests',testMatch:'drawai-dinov3-backend.spec.ts',timeout:180000,workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/phase2/backend-tests.json'}]],
  use:{baseURL:'http://127.0.0.1:4188',browserName:'chromium'},
  webServer:[
    {command:`"${resolve('../.venv-dinov3/Scripts/python.exe')}" -m training.drawai.dinov3_backend`,cwd:resolve('..'),url:'http://127.0.0.1:4190/health',reuseExistingServer:false,timeout:30000},
    {command:'npm run preview -- --port 4188 --strictPort',url:'http://127.0.0.1:4188',reuseExistingServer:false},
  ],
});
