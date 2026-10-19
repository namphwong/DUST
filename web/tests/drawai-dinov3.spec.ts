import {test,expect} from '@playwright/test';
import {mkdir,readFile,writeFile,stat} from 'node:fs/promises';
import {resolve} from 'node:path';
import {execFile,spawn} from 'node:child_process';
import {promisify} from 'node:util';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';

const run=promisify(execFile),base='/__drawai_p2__/',modelName=JSON.parse(readFileSync('../artifacts/drawai/phase2/technical/manifest.json','utf8')).model_filename;
const python=resolve('../.venv-dinov3/Scripts/python.exe');
const rss=async()=>JSON.parse((await run(python,[resolve('../tools/browser_rss.py'),String(process.pid)])).stdout);

test('full DINOv3 smoke graph, raster hashes, WASM parity and runtime budgets',async({page})=>{
  await page.goto('/drawai/dinov3-check');
  await expect(page.getByText('Artifact smoke chưa phải model chữ số đã train.',{exact:false})).toBeVisible();
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  const before=await rss(),samples:any[]=[before];let collecting=false;
  const sampler=setInterval(()=>{if(!collecting){collecting=true;rss().then(r=>samples.push(r)).finally(()=>collecting=false);}},750);
  try{
    await page.getByRole('button',{name:'Chạy kiểm chứng DINOv3',exact:true}).click();
    await expect(page.getByTestId('dinov3-report')).toBeVisible({timeout:150000});
  }finally{clearInterval(sampler);}
  samples.push(await rss());
  const report=JSON.parse((await page.getByTestId('dinov3-report').textContent())!);
  const peak=Math.max(...samples.map(r=>r.rss)),incremental=Math.max(0,peak-before.rss);
  const result={...report,browser:await page.context().browser()!.version(),memory:{method:'sampled aggregate browser descendant RSS; shared pages may be counted twice',baselineBytes:before.rss,peakSampledBytes:peak,incrementalBytes:incremental,samples},memoryBudgetPassed:incremental<=805306368};
  await mkdir('../artifacts/drawai/phase2',{recursive:true});
  await writeFile('../artifacts/drawai/phase2/browser-report.json',JSON.stringify(result,null,2));
  await page.screenshot({path:'../artifacts/drawai/phase2/technical-page.png',fullPage:true});
  expect(errors).toEqual([]);expect(report.parityPassed).toBe(true);
  // Evaluate the fixed budget honestly; a measured failure selects the backend
  // fallback, it must not be hidden by changing the 500ms threshold.
  expect(report.passed).toBe(report.modelBytes<=157286400&&report.coldMs<=15000&&report.p95Ms<=500);
  const artifact=JSON.parse(await readFile('../artifacts/drawai/phase2/technical/manifest.json','utf8'));
  expect(report.pretrainedLoaded).toBe(artifact.pretrained_loaded);expect(report.releaseReady).toBe(false);
  expect(report.cases).toHaveLength(19);expect(report.invalid).toHaveLength(8);
  expect(before.processes.length).toBeGreaterThan(0);expect(samples.length).toBeGreaterThan(2);
  expect(incremental).toBeLessThanOrEqual(805306368);
});

test('missing local technical model gives an explicit error',async({page})=>{
  await page.route('**'+base+modelName,r=>r.fulfill({status:404,body:'missing'}));
  await page.goto('/drawai/dinov3-check');await page.getByRole('button',{name:'Chạy kiểm chứng DINOv3'}).click();
  await expect(page.getByTestId('dinov3-status')).toContainText('FETCH:');
  await expect(page.getByTestId('dinov3-report')).toHaveCount(0);
});

test('raster metadata tamper is rejected even with matching manifest hash',async({page})=>{
  const folder='../artifacts/drawai/phase2/technical/';
  const manifest=JSON.parse(await readFile(folder+'manifest.json','utf8'));
  const prep=JSON.parse(await readFile(folder+'preprocessing.json','utf8'));prep.image_std=[1,1,1];
  const body=JSON.stringify(prep);manifest.files['preprocessing.json']={bytes:Buffer.byteLength(body),sha256:createHash('sha256').update(body).digest('hex')};
  await page.route('**'+base+'preprocessing.json',r=>r.fulfill({status:200,body,contentType:'application/json'}));
  await page.route('**'+base+'manifest.json',r=>r.fulfill({status:200,body:JSON.stringify(manifest),contentType:'application/json'}));
  await page.goto('/drawai/dinov3-check');await page.getByRole('button',{name:'Chạy kiểm chứng DINOv3'}).click();
  await expect(page.getByTestId('dinov3-status')).toContainText('CONTRACT:');
});

test('technical artifacts are absent from public/dist; local route is allowlisted',async({request})=>{
  expect(await stat('public/models/drawai-swin-v1/drawai_swin1d_pendigits_l128_fp32_v1.onnx')).toBeTruthy();
  await expect(stat('public/'+modelName)).rejects.toThrow();
  await expect(stat('dist/'+modelName)).rejects.toThrow();
  expect((await request.get(base+'not-allowed.json')).status()).toBe(404);
});

test('technical manifest cannot pretend to be a trained release',async({page})=>{
  const manifest=JSON.parse(await readFile('../artifacts/drawai/phase2/technical/manifest.json','utf8'));
  manifest.trained=true;manifest.release_ready=true;
  await page.route('**'+base+'manifest.json',r=>r.fulfill({status:200,body:JSON.stringify(manifest),contentType:'application/json'}));
  await page.goto('/drawai/dinov3-check');await page.getByRole('button',{name:'Chạy kiểm chứng DINOv3'}).click();
  await expect(page.getByTestId('dinov3-status')).toContainText('CONTRACT:');
});

test('ordinary preview cannot expose local smoke artifacts',async({request})=>{
  const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','preview','--host','127.0.0.1','--port','4189','--strictPort'],
    {env:{...process.env,DRAWAI_P2_LOCAL_SMOKE:'0'},windowsHide:true,stdio:'ignore'});
  try{
    await expect.poll(async()=>{try{return(await request.get('http://127.0.0.1:4189/')).status();}catch{return 0;}}).toBe(200);
    expect((await request.get('http://127.0.0.1:4189'+base+'manifest.json')).status()).toBe(404);
    expect((await request.get('http://127.0.0.1:4189'+base+modelName)).status()).toBe(404);
  }finally{server.kill();}
});
