import { test,expect,type Page } from '@playwright/test';
import { mkdir,writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { cpus,totalmem } from 'node:os';

test.beforeEach(async({page})=>{
  await page.addInitScript(()=>{
    const Native=window.Worker;(window as any).d6Workers=[];(window as any).d6Messages=[];
    window.Worker=class extends Native{constructor(url:string|URL,options?:WorkerOptions){super(url,options);(window as any).d6Workers.push(this);this.addEventListener('message',e=>(window as any).d6Messages.push(e.data));}};
  });
});
async function sample(page:Page,id:number){await page.getByRole('button',{name:`Thử mẫu ${id}`,exact:true}).click();await expect(page.getByTestId('drawai-reference')).toContainText(`Nhãn mẫu: ${id}`);}
async function ready(page:Page){await expect(page.getByTestId('drawai-prediction')).toBeVisible({timeout:90000});}

test('full JSON/tensor/PyTorch/ORT/product WASM parity and measured 100-run benchmark',async({page,browser},info)=>{
  const requests:{url:string;method:string}[]=[];page.on('request',r=>requests.push({url:r.url(),method:r.method()}));
  await page.goto('/drawai/integration-check');await page.getByRole('button',{name:'Chạy kiểm chứng D6'}).click();
  await expect(page.getByTestId('d6-report')).toHaveAttribute('data-passed','true',{timeout:90000});
  const report=JSON.parse((await page.getByTestId('d6-report').textContent())!);
  expect(report.cases).toHaveLength(17);expect(report.invalid).toHaveLength(7);expect(report.timings).toHaveLength(100);
  expect(report.p95Ms).toBeLessThanOrEqual(100);expect(requests.every(r=>r.method==='GET'&&r.url.startsWith('http://127.0.0.1:4186/'))).toBe(true);
  const release=await(await page.request.get('/models/drawai-swin-v1/release.json')).json();expect(release.files['model.onnx'].bytes).toBeLessThanOrEqual(10*1024*1024);
  await mkdir('../artifacts/drawai/verification',{recursive:true});await mkdir('../artifacts/drawai/benchmark',{recursive:true});
  const evidence={status:'passed',browser:info.project.name,browserVersion:browser.version(),userAgent:await page.evaluate(()=>navigator.userAgent),
    machine:{cpu:cpus()[0].model,logicalCores:cpus().length,totalMemoryBytes:totalmem()},report,requests,
    releaseSha256:createHash('sha256').update(await(await page.request.get('/models/drawai-swin-v1/release.json')).body()).digest('hex')};
  await writeFile(`../artifacts/drawai/verification/${info.project.name}-parity.json`,JSON.stringify(evidence,null,2));
  await writeFile(`../artifacts/drawai/benchmark/${info.project.name}.json`,JSON.stringify({browser:info.project.name,version:browser.version(),modelBytes:release.files['model.onnx'].bytes,
    stages:report.stages,loadAndInitMs:report.loadAndInitMs,medianMs:report.medianMs,p95Ms:report.p95Ms,timings:report.timings,benchmark:report.benchmark},null,2));
});

for(const action of ['draw','undo','sample'] as const)test(`editing via ${action} during delayed load discards stale result`,async({page})=>{
  let release!:()=>void;const gate=new Promise<void>(r=>release=r);
  await page.route('**/drawai-swin-v1/model.onnx',async r=>{await gate;try{await r.continue();}catch{}});
  await page.goto('/drawai');await sample(page,8);await page.getByRole('button',{name:'Nhận dạng →'}).click();
  await expect(page.getByTestId('drawai-app-status')).toContainText('Đang tải');
  if(action==='sample')await sample(page,9);
  else if(action==='undo')await page.getByRole('button',{name:'↶ Hoàn tác nét'}).click();
  else {const canvas=page.getByRole('img',{name:'Vùng vẽ chữ số'});await canvas.scrollIntoViewIfNeeded();const b=(await canvas.boundingBox())!;await page.mouse.move(b.x+50,b.y+50);await page.mouse.down();await page.mouse.move(b.x+100,b.y+100,{steps:5});await page.mouse.up();}
  release();await expect(page.getByRole('button',{name:'Nhận dạng →',exact:true})).toBeVisible({timeout:90000});
  await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);
  await sample(page,2);await page.getByRole('button',{name:'Nhận dạng →'}).click();await ready(page);
  await expect(page.getByTestId('drawai-prediction')).toHaveAttribute('data-digit','2');
});

test('rapid clicks run once; worker crash resets and allows retry',async({page})=>{
  await page.goto('/drawai');await sample(page,5);await page.getByRole('button',{name:'Nhận dạng →'}).dblclick();await ready(page);
  expect(await page.evaluate(()=>(window as any).d6Messages.filter((m:any)=>m.type==='result').length)).toBe(1);
  await sample(page,6);await page.getByRole('button',{name:'Nhận dạng →'}).click();
  await page.evaluate(()=>(window as any).d6Workers[0].dispatchEvent(new ErrorEvent('error',{message:'D6 simulated crash'})));
  await expect(page.getByRole('alert')).toContainText('WORKER');await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);
  await page.getByRole('button',{name:'Thử nhận dạng lại →'}).click();await ready(page);
  expect(await page.evaluate(()=>(window as any).d6Workers.length)).toBe(2);
});

test('60-second timeout with accelerated page clock resets hung worker and retries',async({page})=>{
  let release!:()=>void;let first=true;const gate=new Promise<void>(r=>release=r);
  await page.route('**/drawai-swin-v1/model.onnx',async r=>{if(first){first=false;await gate;}try{await r.continue();}catch{}});
  await page.goto('/drawai');await sample(page,0);await page.clock.install();await page.getByRole('button',{name:'Nhận dạng →'}).click();
  await expect(page.getByTestId('drawai-app-status')).toContainText('Đang tải');await page.clock.fastForward(60001);
  await expect(page.getByRole('alert')).toContainText('60 giây');release();
  await page.getByRole('button',{name:'Thử nhận dạng lại →'}).click();await ready(page);
});

test('cached manifest from old release is rejected and corrected retry succeeds',async({page})=>{
  const pattern='**/drawai-swin-v1/release.json';const manifest=await(await page.request.get('/models/drawai-swin-v1/release.json')).json();manifest.version='drawai-swin-old';
  await page.route(pattern,r=>r.fulfill({status:200,body:JSON.stringify(manifest),headers:{'Cache-Control':'public,max-age=31536000'},contentType:'application/json'}));
  await page.goto('/drawai');await sample(page,1);await page.getByRole('button',{name:'Nhận dạng →'}).click();await expect(page.getByRole('alert')).toContainText('CONTRACT');
  await page.unroute(pattern);await page.getByRole('button',{name:'Thử nhận dạng lại →'}).click();await ready(page);
});

test('WASM missing returns init error, preserves drawing and can retry',async({page})=>{
  const pattern='**/ort-wasm*.wasm';await page.route(pattern,r=>r.fulfill({status:404,body:'missing'}));
  await page.goto('/drawai');await sample(page,7);await page.getByRole('button',{name:'Nhận dạng →'}).click();
  await expect(page.getByRole('alert')).toContainText('INIT',{timeout:90000});await expect(page.locator('.da-count')).not.toHaveText('0 nét');
  await page.unroute(pattern);await page.getByRole('button',{name:'Thử nhận dạng lại →'}).click();await ready(page);
});
