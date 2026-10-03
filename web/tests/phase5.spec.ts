import { test, expect } from '@playwright/test';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

test('sample in two actions: real local model, chart, provenance, six scores and result CSV', async ({ page }) => {
  await page.setViewportSize({ width:1366, height:768 });
  const networkWrites: string[] = [], errors: string[] = [], modelLoads: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('request', r => { if (r.method() !== 'GET') networkWrites.push(r.url()); if (r.url().endsWith('model.onnx')) modelLoads.push(r.url()); });
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Hiểu chuyển động');
  await expect(page.getByRole('button', { name: /01.*Đi bộ/ })).toBeEnabled();
  expect(modelLoads).toHaveLength(0);
  await page.getByRole('button', { name: /01.*Đi bộ/ }).click();
  await expect(page.getByRole('img', { name: /Biểu đồ gia tốc/ })).toBeVisible();
  await expect(page.getByText(/dòng nguồn 79/)).toBeVisible();
  await page.getByRole('button', { name: /^Phân tích mẫu/ }).click();
  await expect(page.getByTestId('prediction')).toBeVisible({ timeout: 90000 });
  await expect(page.locator('.score-row')).toHaveCount(6);
  const demoName = await page.getByTestId('selected-window').locator('h3').textContent();
  const scoreTexts = await page.locator('.score-row b').allTextContents();
  await mkdir('../artifacts/phase5', { recursive: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: '../artifacts/phase5/desktop-proof.png', fullPage: true });
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Tải CSV kết quả ↓' }).click();
  const download = await downloadPromise;
  const csv = await readFile((await download.path())!, 'utf-8');
  expect(csv).toContain('segment_id,window_index,start_ms,end_ms,predicted_id,predicted_label');
  expect(csv).toContain('har-swin-kaggle-v1,1,g');
  expect(csv.trim().split('\n')).toHaveLength(2);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: '../artifacts/phase5/mobile-proof.png', fullPage: true });
  expect(errors).toEqual([]); expect(networkWrites).toEqual([]);
  await writeFile('../artifacts/phase5/browser-verification.json', JSON.stringify({
    status: 'passed', measuredAt: new Date().toISOString(), userAgent: await page.evaluate(() => navigator.userAgent),
    demo: 'walking.csv', selectionActions: 2, predictedLabel: demoName, displayedScores: scoreTexts,
    provenance: 'Official train row 79, subject 1, reference label WALKING',
    inference: 'Actual verified float32 ONNX, client-side WASM', csvExport: 'one real predicted window',
    noNetworkWrites: true, modelNotLoadedBeforeAnalysis: true, mobileNoHorizontalOverflow: true,
    uploadedCsvScope: 'Phase 5 historical scope: preview. Current worker/full pipeline verified separately in artifacts/phase6.',
  }, null, 2));
});

test('all six predeclared demo CSVs have preview and true prediction; source change clears result', async ({ page }) => {
  const golden = JSON.parse(await readFile('../artifacts/phase5/demo-reference.json', 'utf-8'));
  await page.goto('/');
  for (let i=1; i<=6; i++) {
    const button = page.locator('.sample-card').nth(i-1);
    await expect(button).toBeEnabled(); await button.click();
    await expect(page.getByTestId('prediction')).toHaveCount(0);
    await expect(page.locator('.data-stats')).toContainText('128');
    await page.getByRole('button', { name: /^Phân tích mẫu/ }).click();
    await expect(page.getByTestId('prediction')).toBeVisible({ timeout: 90000 });
    await expect(page.locator('.score-row')).toHaveCount(6);
    const promised = page.waitForEvent('download');
    await page.getByRole('button', { name:'Tải CSV kết quả ↓' }).click();
    const file = await promised;
    const csv = await readFile((await file.path())!, 'utf-8');
    const values = csv.trim().split('\n')[1].split(',');
    expect(Number(values[4])).toBe(golden.cases[i-1].predicted_id);
    for (let c=0; c<6; c++) expect(Math.abs(Number(values[6+c])-golden.cases[i-1].scores[c])).toBeLessThanOrEqual(1e-5);
  }
});

test('valid uploaded CSV preview, unit and gravity controls; no fake analysis or stale result', async ({ page }) => {
  const requests: string[] = [];
  await page.goto('/');
  await page.getByRole('button', { name: /01.*Đi bộ/ }).click();
  await page.getByRole('button', { name: /^Phân tích mẫu/ }).click();
  await expect(page.getByTestId('prediction')).toBeVisible({ timeout: 90000 });
  page.on('request', request => requests.push(request.url()));
  await page.getByLabel('Chọn file CSV', { exact: true }).setInputFiles(resolve('../examples/phase0/valid-session-g.csv'));
  await expect(page.getByTestId('csv-preview')).toBeVisible();
  await expect(page.getByTestId('prediction')).toHaveCount(0);
  await expect(page.locator('.data-stats')).toContainText('192');
  await expect(page.getByRole('button', { name: /Suy luận CSV/ })).toBeDisabled();
  await page.getByLabel('Đơn vị nguồn', { exact: true }).selectOption('ms2');
  await page.getByRole('checkbox', { name: 'Gia tốc có bao gồm trọng lực' }).check();
  await expect(page.getByRole('button', { name: /Suy luận CSV/ })).toBeDisabled();
  await page.getByRole('button', { name:'Kiểm tra resample' }).click();
  await expect(page.getByTestId('preparation-report')).toContainText('2 cửa sổ');
  // Phase 6 loads a static worker script; no CSV or model fetch occurs during preview/preparation.
  expect(requests.every(url=>/\/assets\/pipeline-worker-[^/]+\.js$/.test(new URL(url).pathname))).toBe(true);
});

test('row limit accepts 100000 plus trailing blanks and rejects 100001', async ({ page }) => {
  await page.goto('/');
  const input = page.getByLabel('Chọn file CSV', { exact: true });
  const header = 'timestamp_ms,acc_x,acc_y,acc_z\n';
  const text = header + Array.from({ length:100000 }, (_, i)=>`${i*20},1,0,0`).join('\n');
  await input.setInputFiles({ name:'limit.csv', mimeType:'text/csv', buffer:Buffer.from(text+'\n\n') });
  await expect(page.getByTestId('csv-preview')).toBeVisible();
  await expect(page.locator('.data-stats')).toContainText('100.000');
  await expect(page.getByRole('button', { name:/Suy luận CSV/ })).toBeDisabled();
  await input.setInputFiles({ name:'too-many.csv', mimeType:'text/csv', buffer:Buffer.from(text+'\n2000000,1,0,0') });
  await expect(page.getByRole('alert')).toContainText('TOO_MANY_ROWS');
});

for (const [file, code] of [
  ['invalid-missing-column.csv','MISSING_COLUMN'], ['invalid-number.csv','INVALID_NUMBER'],
  ['invalid-duplicate-time.csv','INVALID_TIMESTAMP'], ['invalid-reversed-time.csv','INVALID_TIMESTAMP'],
]) test(`CSV ${file} has actionable ${code}`, async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Chọn file CSV', { exact: true }).setInputFiles(resolve('../examples/phase0/'+file));
  await expect(page.getByRole('alert')).toContainText(code);
  await expect(page.getByRole('alert')).toContainText('dòng');
  await expect(page.getByTestId('prediction')).toHaveCount(0);
});

test('CSV quoting, BOM and reordered columns with extra column are supported', async ({ page }) => {
  const text = '\uFEFFacc_z,note,timestamp_ms,acc_y,acc_x\r\n"0.03","comma, and ""quote""",0,0.98,0.12\r\n0.04,"next",20,0.97,0.15\r\n\r\n';
  await page.goto('/');
  await page.getByLabel('Chọn file CSV', { exact: true }).setInputFiles({ name:'quoted.csv', mimeType:'text/csv', buffer:Buffer.from(text) });
  await expect(page.getByTestId('csv-preview')).toContainText('Bỏ qua cột thêm: note');
  await expect(page.getByTestId('csv-preview')).toContainText('Dưới 128 điểm nguồn');
  await expect(page.getByTestId('csv-preview').locator('tbody tr')).toHaveCount(2);
});

test('oversize file, duplicated header and broken quoting are rejected', async ({ page }) => {
  await page.goto('/');
  const input = page.getByLabel('Chọn file CSV', { exact: true });
  await input.setInputFiles({ name:'large.csv', mimeType:'text/csv', buffer:Buffer.alloc(10*1024*1024+1) });
  await expect(page.getByRole('alert')).toContainText('FILE_TOO_LARGE');
  await input.setInputFiles({ name:'duplicate.csv', mimeType:'text/csv', buffer:Buffer.from('timestamp_ms,acc_x,acc_x,acc_y,acc_z\n0,1,1,1,1') });
  await expect(page.getByRole('alert')).toContainText('DUPLICATE_COLUMN');
  await input.setInputFiles({ name:'quote.csv', mimeType:'text/csv', buffer:Buffer.from('timestamp_ms,acc_x,acc_y,acc_z\n0,"1,2,3') });
  await expect(page.getByRole('alert')).toContainText('INVALID_CSV');
});

test('keyboard selects demo; model loading failure is retryable and never shows a prediction', async ({ page }) => {
  await page.route('**/models/har-swin-kaggle-v1/model.onnx', route=>route.fulfill({ status:404, body:'missing' }));
  await page.goto('/');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name:'Bỏ qua đến khu vực phân tích' })).toBeFocused();
  await expect(page.getByRole('link', { name:'Bỏ qua đến khu vực phân tích' })).toBeVisible();
  const button = page.getByRole('button', { name: /01.*Đi bộ/ });
  await expect(button).toBeEnabled(); await button.focus(); await page.keyboard.press('Enter');
  await page.getByRole('button', { name: /^Phân tích mẫu/ }).click();
  await expect(page.getByRole('alert')).toContainText('Thiếu artifact Phase 4');
  await expect(page.getByTestId('prediction')).toHaveCount(0);
  await page.unroute('**/models/har-swin-kaggle-v1/model.onnx');
  await page.getByRole('button', { name: /^Phân tích mẫu/ }).click();
  await expect(page.getByTestId('prediction')).toBeVisible({ timeout:90000 });
});
