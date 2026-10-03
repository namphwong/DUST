import { test, expect } from '@playwright/test';
import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

for (const [name,body,code] of [
  ['empty','timestamp_ms,acc_x,acc_y,acc_z\n','EMPTY_FILE'],
  ['nan','timestamp_ms,acc_x,acc_y,acc_z\n0,NaN,0,0','INVALID_NUMBER'],
  ['infinity','timestamp_ms,acc_x,acc_y,acc_z\n0,1,Infinity,0','INVALID_NUMBER'],
  ['empty-cell','timestamp_ms,acc_x,acc_y,acc_z\n0,1,,0','INVALID_NUMBER'],
  ['negative-time','timestamp_ms,acc_x,acc_y,acc_z\n-1,1,0,0','INVALID_TIMESTAMP'],
  ['row-width','timestamp_ms,acc_x,acc_y,acc_z\n0,1,0','INVALID_CSV'],
]) test(`worker rejects ${name} with ${code}`, async ({ page })=>{
  await page.goto('/');
  await page.getByLabel('Chọn file CSV',{exact:true}).setInputFiles({name:name+'.csv',mimeType:'text/csv',buffer:Buffer.from(body)});
  await expect(page.getByRole('alert')).toContainText(code);
  if (code === 'INVALID_NUMBER') {
    await expect(page.getByRole('alert')).toContainText('dòng 2');
    await expect(page.getByRole('alert')).toContainText('cột');
  }
  await expect(page.getByTestId('prediction')).toHaveCount(0);
});

test('invalid UTF-8 and preparation errors are actionable in UI, without prediction',async ({page})=>{
  await page.goto('/');
  const input = page.getByLabel('Chọn file CSV',{exact:true});
  await input.setInputFiles({name:'bad-utf8.csv',mimeType:'text/csv',buffer:Buffer.from([0xff,0xfe])});
  await expect(page.getByRole('alert')).toContainText('INVALID_CSV');
  await expect(page.getByRole('alert')).toContainText('UTF-8');
  for (const [name,code] of [['boundary-127','TOO_SHORT'],['unsupported-30hz','UNSUPPORTED_FREQUENCY'],['expanded-row-limit','TOO_MANY_ROWS']]) {
    await input.setInputFiles(resolve('../artifacts/phase6/fixtures/'+name+'.csv'));
    await expect(page.getByTestId('csv-preview')).toBeVisible();
    await page.getByRole('checkbox',{name:'Gia tốc có bao gồm trọng lực'}).check();
    await page.getByRole('button',{name:'Kiểm tra resample'}).click();
    await expect(page.getByRole('alert')).toContainText(code);
    await expect(page.getByTestId('prediction')).toHaveCount(0);
    await expect(page.getByRole('button',{name:'Suy luận CSV'})).toBeDisabled();
  }
});

test('17 independent Python CSV fixtures: errors, resample, layout, units, normalized tensor and real model parity', async ({ page }) => {
  const golden = JSON.parse(await readFile('../artifacts/phase6/python-reference.json','utf-8'));
  const asset = (await readdir('dist/assets')).find(name=>/^pipeline-worker-.*\.js$/.test(name))!;
  await page.goto('/');
  const cases = [];
  for (const fixture of golden.cases) {
    const csv = await readFile('../artifacts/phase6/fixtures/'+fixture.file,'utf-8');
    const result = await page.evaluate(async ({ asset, csv, unit, gravity })=>{
      const worker = new Worker('/assets/'+asset,{type:'module'});
      let id = 0;
      function request(action: string, payload: Record<string,unknown> = {}): Promise<any> {
        const current = ++id;
        return new Promise((resolve,reject)=>{
          worker.onerror = e=>reject(new Error(e.message));
          worker.onmessage = e=>{
            if (e.data.id !== current) return;
            if (e.data.type === 'done') resolve(e.data.data);
            else if (e.data.type === 'error') reject(new Error(e.data.data));
          };
          worker.postMessage({id:current,action,...payload});
        });
      }
      try {
        const preview = await request('parse',{file:new File([csv],'fixture.csv')});
        const summary = await request('prepare',{unit,gravity});
        const output = await request('inspect');
        return {preview,summary,...output};
      } catch(e) { return { error: e instanceof Error ? e.message : 'unknown' }; }
      finally { worker.terminate(); }
    },{asset,csv,unit:fixture.unit,gravity:fixture.gravity});
    if (fixture.expectedError) {
      expect(result.error,fixture.name).toContain(fixture.expectedError);
      cases.push({name:fixture.name,error:fixture.expectedError,passed:true});
      continue;
    }
    expect(result.error,fixture.name).toBeUndefined();
    expect(result.summary,fixture.name).toEqual(fixture.summary);
    expect(result.results.length).toBe(fixture.windows.length);
    let inputMaxError = 0, logitMaxError = 0;
    for (let w=0; w<fixture.windows.length; w++) {
      const expected = fixture.windows[w], actual = result.results[w];
      expect([actual.segmentId,actual.windowIndex,actual.startMs,actual.endMs]).toEqual([expected.segmentId,expected.windowIndex,expected.startMs,expected.endMs]);
      expect(actual.predictedId).toBe(expected.predictedId);
      for (let i=0;i<384;i++) {
        const error = Math.abs(result.inputs[w][i]-expected.input[i]);
        inputMaxError = Math.max(inputMaxError,error);
        expect(error).toBeLessThanOrEqual(1e-6);
      }
      for (let i=0;i<6;i++) {
        const error = Math.abs(actual.logits[i]-expected.logits[i]);
        logitMaxError = Math.max(logitMaxError,error);
        expect(error).toBeLessThanOrEqual(1e-5+1e-4*Math.abs(expected.logits[i]));
        expect(Math.abs(actual.scores[i]-expected.scores[i])).toBeLessThanOrEqual(1e-5);
      }
      expect(actual.scores.reduce((a:number,b:number)=>a+b,0)).toBeCloseTo(1,12);
    }
    cases.push({name:fixture.name,windows:result.results.length,inputMaxError,logitMaxError,passed:true});
  }
  await mkdir('../artifacts/phase6',{recursive:true});
  await writeFile('../artifacts/phase6/browser-parity.json',JSON.stringify({status:'passed',measuredAt:new Date().toISOString(),userAgent:await page.evaluate(()=>navigator.userAgent),cases},null,2));
});

test('user CSV whole flow, g/ms², confirmation invalidation, export all windows, responsive layout', async ({ page }) => {
  const writes: string[] = [], errors: string[] = [];
  page.on('request',r=>{if(r.method() !== 'GET') writes.push(r.url());});
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');
  await page.getByLabel('Chọn file CSV',{exact:true}).setInputFiles(resolve('../artifacts/phase6/fixtures/two-segments.csv'));
  await expect(page.getByTestId('csv-preview')).toBeVisible();
  await expect(page.getByRole('button',{name:'Suy luận CSV'})).toBeDisabled();
  const gravity = page.getByRole('checkbox',{name:'Gia tốc có bao gồm trọng lực'});
  await gravity.focus(); await page.keyboard.press('Space');
  const prepare = page.getByRole('button',{name:'Kiểm tra resample'});
  await prepare.focus(); await page.keyboard.press('Enter');
  await expect(page.getByTestId('preparation-report')).toContainText('3 cửa sổ');
  await expect(page.getByRole('button',{name:'Suy luận CSV'})).toBeDisabled();
  await page.getByRole('checkbox',{name:/Tôi đã kiểm tra/}).focus(); await page.keyboard.press('Space');
  await page.getByRole('button',{name:'Suy luận CSV'}).focus(); await page.keyboard.press('Enter');
  await expect(page.getByTestId('prediction')).toBeVisible({timeout:90000});
  const pending = page.waitForEvent('download');
  await page.getByRole('button',{name:'Tải CSV kết quả ↓'}).click();
  const file = await pending;
  const csv = await readFile((await file.path())!,'utf-8');
  const rows = csv.trim().split('\n');
  expect(rows.length).toBe(4);
  expect(rows[3]).toMatch(/^1,2,5000,7540,/);
  await page.getByTestId('prediction').locator('tbody button').last().click();
  await expect(page.locator('.score-row')).toHaveCount(6);
  await page.setViewportSize({width:1366,height:768});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path:'../artifacts/phase6/desktop-proof.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path:'../artifacts/phase6/mobile-proof.png',fullPage:true});
  await page.getByLabel('Đơn vị nguồn',{exact:true}).selectOption('ms2');
  await expect(page.getByTestId('prediction')).toHaveCount(0);
  await expect(page.getByTestId('preparation-report')).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Suy luận CSV'})).toBeDisabled();
  await page.getByLabel('Chọn file CSV',{exact:true}).setInputFiles(resolve('../artifacts/phase6/fixtures/units-ms2.csv'));
  await expect(page.getByTestId('csv-preview')).toBeVisible();
  await page.getByLabel('Đơn vị nguồn',{exact:true}).selectOption('ms2');
  await page.getByRole('checkbox',{name:'Gia tốc có bao gồm trọng lực'}).check();
  await page.getByRole('button',{name:'Kiểm tra resample'}).click();
  await expect(page.getByTestId('preparation-report')).toBeVisible();
  await page.getByRole('checkbox',{name:/Tôi đã kiểm tra/}).check();
  await page.getByRole('button',{name:'Suy luận CSV'}).click();
  await expect(page.getByTestId('prediction')).toBeVisible({timeout:90000});
  const msDownload = page.waitForEvent('download');
  await page.getByRole('button',{name:'Tải CSV kết quả ↓'}).click();
  const msFile = await msDownload;
  expect(await readFile((await msFile.path())!,'utf-8')).toContain('har-swin-kaggle-v1,1,m/s²');
  expect(writes).toEqual([]);expect(errors).toEqual([]);
  await writeFile('../artifacts/phase6/ui-verification.json',JSON.stringify({status:'passed',measuredAt:new Date().toISOString(),noNetworkWrites:true,confirmationRequired:true,unitChangeClearsResults:true,allWindowsExported:true,mobileNoHorizontalOverflow:true,desktopNoHorizontalOverflow:true},null,2));
});

test('100000 rows: worker progress, responsive main thread, cancel within 1 second, rerun and stale-source isolation', async ({ page }) => {
  await page.goto('/');
  const csv = 'timestamp_ms,acc_x,acc_y,acc_z\n'+Array.from({length:100000},(_,i)=>`${i*20},1,0,0`).join('\n');
  const input = page.getByLabel('Chọn file CSV',{exact:true});
  await page.evaluate(()=>{
    const state = {ticks:0,maxGap:0,last:performance.now()};
    (window as any).responsiveness = state;
    const timer = setInterval(()=>{ const now = performance.now();state.ticks++;state.maxGap=Math.max(state.maxGap,now-state.last);state.last=now;},20);
    (window as any).responsivenessTimer = timer;
  });
  await input.setInputFiles({name:'100k.csv',mimeType:'text/csv',buffer:Buffer.from(csv)});
  await expect(page.getByTestId('csv-preview')).toBeVisible();
  await page.getByRole('checkbox',{name:'Gia tốc có bao gồm trọng lực'}).check();
  await page.getByRole('button',{name:'Kiểm tra resample'}).click();
  await expect(page.getByTestId('preparation-report')).toContainText('1561 cửa sổ');
  await page.getByRole('checkbox',{name:/Tôi đã kiểm tra/}).check();
  await page.getByRole('button',{name:'Suy luận CSV'}).click();
  await expect(page.getByRole('progressbar')).toHaveAttribute('value',/^(?:[1-9]\d*)$/,{timeout:90000});
  const before = Date.now();
  await page.getByRole('button',{name:'Hủy tác vụ'}).click();
  await expect(page.getByRole('status')).toContainText('Đã hủy tác vụ');
  const cancelMs = Date.now()-before;
  expect(cancelMs).toBeLessThanOrEqual(1000);
  await expect(page.getByTestId('prediction')).toHaveCount(0);
  await page.getByRole('button',{name:'Kiểm tra resample'}).click();
  await expect(page.getByTestId('preparation-report')).toContainText('1561 cửa sổ');
  await page.getByRole('checkbox',{name:/Tôi đã kiểm tra/}).check();
  const analysisStarted = Date.now();
  await page.getByRole('button',{name:'Suy luận CSV'}).click();
  await expect(page.getByTestId('prediction')).toBeVisible({timeout:90000});
  const completeAnalysisMs = Date.now()-analysisStarted;
  await expect(page.getByTestId('prediction')).toContainText('1561 cửa sổ');
  await expect(page.getByTestId('prediction').locator('tbody tr')).toHaveCount(20);
  const promised = page.waitForEvent('download');
  await page.getByRole('button',{name:'Tải CSV kết quả ↓'}).click();
  const download = await promised;
  expect((await readFile((await download.path())!,'utf-8')).trim().split('\n')).toHaveLength(1562);
  // Replace the source while an actual second inference pass is running.
  await page.getByRole('button',{name:'Suy luận CSV'}).click();
  await expect(page.getByRole('progressbar')).toHaveAttribute('value',/^(?:[1-9]\d*)$/);
  await input.setInputFiles(resolve('../artifacts/phase6/fixtures/boundary-128.csv'));
  await expect(page.getByTestId('csv-preview')).toBeVisible();
  await expect(page.locator('.data-stats')).toContainText('128');
  await expect(page.getByTestId('prediction')).toHaveCount(0);
  await expect(page.getByRole('checkbox',{name:'Gia tốc có bao gồm trọng lực'})).not.toBeChecked();
  const responsive = await page.evaluate(()=>{clearInterval((window as any).responsivenessTimer); return (window as any).responsiveness;});
  expect(responsive.ticks).toBeGreaterThan(100);
  expect(responsive.maxGap).toBeLessThan(500);
  await writeFile('../artifacts/phase6/performance.json',JSON.stringify({status:'passed',measuredAt:new Date().toISOString(),sourceRows:100000,resampledRows:100000,windows:1561,cancelMs,completeAnalysisMs,mainThreadHeartbeatIntervalMs:20,mainThreadHeartbeatTicks:responsive.ticks,maxHeartbeatGapMs:responsive.maxGap,heartbeatAcceptanceLimitMs:500,staleResultsAbsent:true,renderedRows:20,userAgent:await page.evaluate(()=>navigator.userAgent)},null,2));
});
