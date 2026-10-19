import {defineConfig} from '@playwright/test';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
export default defineConfig({
  testDir:'./tests',testMatch:['drawai-p7.spec.ts','drawai-phase1.spec.ts'],timeout:90000,workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/phase7-image64/browser-tests.json'}]],
  use:{baseURL:'http://127.0.0.1:4186',browserName:'chromium'},
  webServer:[
    {command:`"${resolve(root,'.venv-dinov3/Scripts/python.exe')}" -m training.drawai.paint_app_runtime`,cwd:root,url:'http://127.0.0.1:4197/health',timeout:60000,reuseExistingServer:false},
    {command:'npm run preview -- --port 4186 --strictPort',url:'http://127.0.0.1:4186',reuseExistingServer:false},
  ],
});
