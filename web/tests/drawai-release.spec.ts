import { test, expect, type Page } from '@playwright/test';
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { cpus, totalmem } from 'node:os';

const base='/models/drawai-swin-v1/';
async function run(page:Page) {
  await page.goto('/drawai/release-check');
  await page.getByRole('button',{name:'Chạy kiểm chứng DrawAI'}).click();
}
async function tamper(page:Page,file:string,mutate:(value:any)=>void) {
  const release=await (await page.request.get(base+'release.json')).json();
  const value=await (await page.request.get(base+file)).json(); mutate(value);
  const body=JSON.stringify(value);
  release.files[file]={bytes:Buffer.byteLength(body),sha256:createHash('sha256').update(body).digest('hex')};
  await page.route('**'+base+file,r=>r.fulfill({status:200,body,contentType:'application/json'}));
  await page.route('**'+base+'release.json',r=>r.fulfill({status:200,body:JSON.stringify(release),contentType:'application/json'}));
}

test('trained Swin release matches 20 fixtures in real WASM worker',async({page})=> {
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await run(page);
  await expect(page.getByTestId('drawai-result')).toHaveAttribute('data-passed','true',{timeout:90000});
  const report=JSON.parse((await page.locator('pre').textContent())!);
  expect(report.trained).toBe(true);expect(report.length).toBe(128);
  expect(report.cases).toHaveLength(20);expect(report.invalidCases).toHaveLength(14);
  expect(report.invalidCases.every((c:any)=>c.passed)).toBe(true);
  expect(report.targetsMet).toBe(true);expect(errors).toEqual([]);
  const release=await (await page.request.get(base+'release.json')).json();
  await mkdir('../artifacts/drawai/export',{recursive:true});
  await writeFile('../artifacts/drawai/export/browser-report.json',JSON.stringify({status:'passed',
    releaseSha256:createHash('sha256').update(await (await page.request.get(base+'release.json')).body()).digest('hex'),
    userAgent:await page.evaluate(()=>navigator.userAgent),
    machine:{cpu:cpus()[0].model,logicalCores:cpus().length,totalMemoryBytes:totalmem()},
    modelSha256:release.files['model.onnx'].sha256,report},null,2));
  await page.screenshot({path:'../artifacts/drawai/export/browser-proof.png',fullPage:true});
});

test('missing labels rejects release and retry succeeds',async({page})=> {
  const pattern='**'+base+'labels.json';
  await page.route(pattern,r=>r.fulfill({status:404,body:'missing'}));
  await run(page);
  await expect(page.getByTestId('drawai-status')).toContainText('Thiếu artifact DrawAI: labels.json');
  await expect(page.getByTestId('drawai-result')).toHaveCount(0);
  await page.unroute(pattern);
  await page.getByRole('button',{name:'Chạy kiểm chứng DrawAI'}).click();
  await expect(page.getByTestId('drawai-result')).toHaveAttribute('data-passed','true',{timeout:90000});
});

test('corrupt trained model rejected before inference',async({page})=> {
  await page.route('**'+base+'model.onnx',r=>r.fulfill({status:200,body:'corrupt'}));
  await run(page);
  await expect(page.getByTestId('drawai-status')).toContainText('Sai hash/bytes: model.onnx');
  await expect(page.getByTestId('drawai-result')).toHaveCount(0);
});

for (const [file,mutate,error] of [
  ['labels.json',(v:any)=>{v.classes[0].name='9';},'Sai nhãn hoặc provenance'],
  ['report.json',(v:any)=>{v.checkpoint_sha256='0'.repeat(64);},'Sai nhãn hoặc provenance'],
  ['preprocessing.json',(v:any)=>{v.length=64;},'Metadata/fixture không đúng hợp đồng'],
] as const) {
  test(`reject ${file} mismatch even with consistent hash`,async({page})=> {
    await tamper(page,file,mutate);await run(page);
    await expect(page.getByTestId('drawai-status')).toContainText(error);
    await expect(page.getByTestId('drawai-result')).toHaveCount(0);
  });
}

test('trained=false release cannot produce result',async({page})=> {
  const release=await (await page.request.get(base+'release.json')).json();release.trained=false;
  await page.route('**'+base+'release.json',r=>r.fulfill({status:200,body:JSON.stringify(release),contentType:'application/json'}));
  await run(page);
  await expect(page.getByTestId('drawai-status')).toContainText('Sai version hoặc trạng thái');
  await expect(page.getByTestId('drawai-result')).toHaveCount(0);
});
