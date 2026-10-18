import { test, expect, type Page } from '@playwright/test';
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { cpus, totalmem } from 'node:os';

const reportDirectory=process.env.DRAWAI_REGRESSION_REPORT_DIR ?? '../artifacts/drawai/feasibility';

async function tamperWithConsistentHash(page:Page,file:string,mutate:(value:any)=>void) {
  const base='/models/drawai-feasibility-l64/';
  const release=await (await page.request.get(base+'release.json')).json();
  const artifact=await (await page.request.get(base+file)).json();
  mutate(artifact);
  const body=JSON.stringify(artifact);
  release.files[file]={bytes:Buffer.byteLength(body),sha256:createHash('sha256').update(body).digest('hex')};
  await page.route('**'+base+file,route=>route.fulfill({status:200,body,contentType:'application/json'}));
  await page.route('**'+base+'release.json',route=>route.fulfill({status:200,body:JSON.stringify(release),contentType:'application/json'}));
}

test('both lengths match Python preprocessing and PyTorch logits in real WASM worker',async({page})=> {
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/drawai/check');
  await expect(page.getByText('Mô hình kỹ thuật chưa huấn luyện.',{exact:true})).toBeVisible();
  const reports=[];
  for (const length of [64,128]) {
    await page.getByLabel('Chiều dài chuỗi').selectOption(String(length));
    await page.getByRole('button',{name:'Chạy kiểm chứng DrawAI'}).click();
    await expect(page.getByTestId('drawai-result')).toHaveAttribute('data-passed','true',{timeout:90000});
    const report=JSON.parse((await page.locator('pre').textContent())!);
    expect(report.trained).toBe(false);
    expect(report.cases).toHaveLength(20);
    expect(report.invalidCases).toHaveLength(14);
    expect(report.invalidCases.every((c:{passed:boolean})=>c.passed)).toBe(true);
    expect(report.targetsMet).toBe(true);
    expect(report.benchmark).toMatchObject({warmup:5,iterations:100,input:'random'});
    expect(report.cases.filter((c:{preprocessingError:number})=>c.preprocessingError>2e-7)).toHaveLength(0);
    reports.push(report);
  }
  expect(errors).toEqual([]);
  await mkdir(reportDirectory,{recursive:true});
  await writeFile(`${reportDirectory}/browser-report.json`,JSON.stringify({
    status:'passed',userAgent:await page.evaluate(()=>navigator.userAgent),
    machine:{cpu:cpus()[0].model,logicalCores:cpus().length,totalMemoryBytes:totalmem()},reports,
  },null,2));
  await page.screenshot({path:`${reportDirectory}/desktop.png`,fullPage:true});
});

test('missing model returns error and supports retry',async({page})=> {
  const pattern='**/models/drawai-feasibility-l64/model.onnx';
  await page.route(pattern,route=>route.fulfill({status:404,body:'missing'}));
  await page.goto('/drawai/check');
  await page.getByRole('button',{name:'Chạy kiểm chứng DrawAI'}).click();
  await expect(page.getByTestId('drawai-status')).toContainText('Thiếu artifact DrawAI: model.onnx');
  await page.unroute(pattern);
  await page.getByRole('button',{name:'Chạy kiểm chứng DrawAI'}).click();
  await expect(page.getByTestId('drawai-result')).toHaveAttribute('data-passed','true',{timeout:90000});
});

test('corrupted artifact cannot produce PASS',async({page})=> {
  await page.route('**/models/drawai-feasibility-l64/fixtures.json',route=>route.fulfill({status:200,body:'{}',contentType:'application/json'}));
  await page.goto('/drawai/check');
  await page.getByRole('button',{name:'Chạy kiểm chứng DrawAI'}).click();
  await expect(page.getByTestId('drawai-status')).toContainText('Sai hash/bytes: fixtures.json');
  await expect(page.getByTestId('drawai-result')).toHaveCount(0);
});

test('trained release is rejected by technical page',async({page})=> {
  const manifest=await (await page.request.get('/models/drawai-feasibility-l64/release.json')).json();
  manifest.trained=true;
  await page.route('**/models/drawai-feasibility-l64/release.json',route=>route.fulfill({status:200,body:JSON.stringify(manifest),contentType:'application/json'}));
  await page.goto('/drawai/check');
  await page.getByRole('button',{name:'Chạy kiểm chứng DrawAI'}).click();
  await expect(page.getByTestId('drawai-status')).toContainText('Sai version hoặc trạng thái');
  await expect(page.getByTestId('drawai-result')).toHaveCount(0);
});

test('wrong preprocessing contract fails even with consistent artifact hash',async({page})=> {
  await tamperWithConsistentHash(page,'preprocessing.json',value=>{value.layout='NTC';});
  await page.goto('/drawai/check');
  await page.getByRole('button',{name:'Chạy kiểm chứng DrawAI'}).click();
  await expect(page.getByTestId('drawai-status')).toContainText('Metadata/fixture không đúng hợp đồng');
  await expect(page.getByTestId('drawai-result')).toHaveCount(0);
});

test('non-numeric fixture fails even with consistent artifact hash',async({page})=> {
  await tamperWithConsistentHash(page,'fixtures.json',value=>{value.cases[0].input[0]=null;});
  await page.goto('/drawai/check');
  await page.getByRole('button',{name:'Chạy kiểm chứng DrawAI'}).click();
  await expect(page.getByTestId('drawai-status')).toContainText('Fixture sai shape hoặc có giá trị không hữu hạn');
  await expect(page.getByTestId('drawai-result')).toHaveCount(0);
});
