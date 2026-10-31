import {defineConfig} from '@playwright/test';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const python=process.env.DRAWAI_TEST_PYTHON??resolve(root,'.venv-dinov3/Scripts/python.exe');
export default defineConfig({
  testDir:'./tests',testMatch:'drawai-public.spec.ts',timeout:120000,workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/public-deployment/browser-tests.json'}]],
  use:{baseURL:'http://127.0.0.1:4188',browserName:'chromium'},
  webServer:[
    {command:`"${python}" -m uvicorn training.drawai.public_api:app --host 127.0.0.1 --port 4198 --no-access-log`,cwd:root,
     env:{ALLOWED_ORIGINS:'http://127.0.0.1:4188'},url:'http://127.0.0.1:4198/health',timeout:90000,reuseExistingServer:false},
    {command:'npm run preview -- --port 4188 --strictPort',url:'http://127.0.0.1:4188',reuseExistingServer:false},
  ],
});
