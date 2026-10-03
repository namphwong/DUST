import { test, expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const base = '**/models/har-swin-kaggle-v1/';
test('trained release: normalization, WASM parity, demo labels, benchmark and rerun', async ({ page, browser }, testInfo) => {
  const errors: string[] = [];
  const uploads: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.method() !== 'GET') uploads.push(request.url()); });
  await page.goto('/?phase=4');
  await expect(page.getByText('Mô hình đã train trên Kaggle.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Chạy kiểm chứng', exact: true }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-passed', 'true', { timeout: 90000 });
  await expect(page.getByTestId('result').locator('table').first().locator('tbody tr')).toHaveCount(14);
  await expect(page.getByTestId('activities').locator('tbody tr')).toHaveCount(6);
  const result = JSON.parse((await page.locator('pre').textContent())!);
  expect(result.report.trained).toBe(true);
  expect(result.report.onnx_test.macro_f1).toBeCloseTo(0.8704888484455225, 12);
  expect(result.normalizationMaxError).toBeLessThanOrEqual(1e-6);
  expect(result.modelBytes).toBeLessThanOrEqual(10*1024*1024);
  expect(result.medianMs).toBeLessThanOrEqual(500);
  for (const activity of result.activities) {
    expect(activity.scores).toHaveLength(6);
    expect(activity.scores.reduce((a: number, b: number) => a+b, 0)).toBeCloseTo(1, 12);
  }
  await mkdir('../artifacts/phase4', { recursive: true });
  await writeFile('../artifacts/phase4/browser-verification.json', JSON.stringify({
    measuredAt: new Date().toISOString(), userAgent: await page.evaluate(() => navigator.userAgent),
    browserVersion: browser.version(), runtime: 'onnxruntime-web 1.20.1, WASM one thread',
    benchmark: { warmup: 5, iterations: 30, source: 'localhost, not Vercel/Internet latency' },
    ...result,
  }, null, 2));
  await testInfo.attach('phase4-browser-report', { body: JSON.stringify(result), contentType: 'application/json' });
  await page.screenshot({ path: '../artifacts/phase4/browser-proof.png', fullPage: true });
  await page.getByRole('button', { name: 'Chạy kiểm chứng lại' }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-passed', 'true', { timeout: 90000 });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: '../artifacts/phase4/browser-mobile-proof.png', fullPage: true });
  expect(errors).toEqual([]);
  expect(uploads).toEqual([]);
});

test('missing trained model errors and can retry successfully', async ({ page }) => {
  await page.route(base + 'model.onnx', route => route.fulfill({ status: 404, body: 'missing' }));
  await page.goto('/?phase=4');
  await page.getByRole('button', { name: 'Chạy kiểm chứng', exact: true }).click();
  await expect(page.getByTestId('status')).toContainText('Thiếu artifact Phase 4: model.onnx');
  await expect(page.getByTestId('result')).toHaveCount(0);
  await page.unroute(base + 'model.onnx');
  await page.getByRole('button', { name: 'Chạy kiểm chứng', exact: true }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-passed', 'true', { timeout: 90000 });
});

test('mixed metadata hash is rejected before inference', async ({ page }) => {
  await page.route(base + 'preprocessing.json', async route => {
    const response = await route.fetch();
    const json = await response.json();
    json.mean[0] += 1;
    await route.fulfill({ response, json });
  });
  await page.goto('/?phase=4');
  await page.getByRole('button', { name: 'Chạy kiểm chứng', exact: true }).click();
  await expect(page.getByTestId('status')).toContainText('sai hash: preprocessing.json');
  await expect(page.getByTestId('result')).toHaveCount(0);
});

test('stale release or untrained release is rejected', async ({ page }) => {
  await page.route(base + 'release.json', async route => {
    const response = await route.fetch();
    const json = await response.json();
    json.version = 'phase1'; json.trained = false;
    await route.fulfill({ response, json });
  });
  await page.goto('/?phase=4');
  await page.getByRole('button', { name: 'Chạy kiểm chứng', exact: true }).click();
  await expect(page.getByTestId('status')).toContainText('Release không đúng version/trained=true');
});

test('wrong golden logits still fail parity when manifest hash matches', async ({ page }) => {
  const response = await page.request.get('/models/har-swin-kaggle-v1/fixtures.json');
  const fixtures = await response.json();
  fixtures.cases[0].expected[0] += 1;
  const body = JSON.stringify(fixtures);
  await page.route(base + 'fixtures.json', route => route.fulfill({ body, contentType: 'application/json' }));
  await page.route(base + 'release.json', async route => {
    const response = await route.fetch();
    const json = await response.json();
    json.files['fixtures.json'] = { sha256: createHash('sha256').update(body).digest('hex'), bytes: Buffer.byteLength(body) };
    await route.fulfill({ response, json });
  });
  await page.goto('/?phase=4');
  await page.getByRole('button', { name: 'Chạy kiểm chứng', exact: true }).click();
  await expect(page.getByTestId('result')).toHaveAttribute('data-passed', 'false', { timeout: 90000 });
  await expect(page.getByTestId('status')).toContainText('FAIL');
  await expect(page.getByTestId('activities')).toHaveCount(0);
});
