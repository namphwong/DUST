import {defineConfig} from '@playwright/test';

const baseURL=process.env.DRAWAI_CLOUD_URL;
if(!baseURL)throw new Error('DRAWAI_CLOUD_URL is required');

export default defineConfig({
  testDir:'./tests',
  testMatch:'drawai-cloud.spec.ts',
  timeout:180000,
  workers:1,
  reporter:[['list'],['json',{outputFile:'../artifacts/drawai/public-deployment/cloud-browser-tests.json'}]],
  use:{baseURL,browserName:'chromium',trace:'retain-on-failure'},
});
