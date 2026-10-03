import { test, expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
test('real WASM inference matches all PyTorch fixtures twice', async ({ page }, testInfo) => {
  const failures: string[] = [];
  page.on('pageerror', error => failures.push(error.message));
  await page.goto('/?phase=1');
  await expect(page.getByText('Mô hình chưa huấn luyện.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Chạy kiểm chứng', exact: true }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-passed', 'true', { timeout: 90000 });
  await expect(page.locator('tbody tr')).toHaveCount(8);
  const report = await page.locator('pre').textContent();
  await mkdir('../docs', { recursive: true });
  await writeFile('../docs/phase1-browser-results.json', JSON.stringify({
    measuredAt: new Date().toISOString(), userAgent: await page.evaluate(() => navigator.userAgent),
    ...JSON.parse(report!),
  }, null, 2));
  await testInfo.attach('browser-report', { body: report!, contentType: 'application/json' });
  await page.screenshot({ path: 'test-results/phase1-browser.png', fullPage: true });
  await page.getByRole('button', { name: 'Chạy kiểm chứng lại' }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-passed', 'true', { timeout: 90000 });
  expect(failures).toEqual([]);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Chạy kiểm chứng lại' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
test('wrong golden logits fail parity instead of showing PASS', async ({ page }) => {
  await page.route('**/models/phase1/fixtures.json', async route => {
    const response = await route.fetch();
    const fixtures = await response.json();
    fixtures.cases[0].expected[0] += 1;
    await route.fulfill({ response, json: fixtures });
  });
  await page.goto('/?phase=1');
  await page.getByRole('button', { name: 'Chạy kiểm chứng', exact: true }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-passed', 'false', { timeout: 90000 });
  await expect(page.getByTestId('status')).toContainText('FAIL');
});
test('missing model produces a useful error', async ({ page }) => {
  await page.route('**/models/phase1/model.onnx', route => route.fulfill({ status: 404, body: 'missing' }));
  await page.goto('/?phase=1');
  await page.getByRole('button', { name: 'Chạy kiểm chứng', exact: true }).click();
  await expect(page.getByTestId('status')).toContainText('Thiếu artifact Phase 1');
  await expect(page.getByRole('button', { name: 'Chạy kiểm chứng', exact: true })).toBeEnabled();
});
