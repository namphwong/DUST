import { defineConfig } from '@playwright/test';
export default defineConfig({
  reporter: [['list'], ['json', { outputFile: '../artifacts/phase6/playwright-results.json' }]],
  testDir: './tests', timeout: 120000, workers: 1,
  use: { baseURL: 'http://127.0.0.1:4174', browserName: 'chromium' },
  webServer: { command: 'npm run preview -- --port 4174 --strictPort', url: 'http://127.0.0.1:4174', reuseExistingServer: false },
});
