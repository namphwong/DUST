import { test,expect,type Page } from '@playwright/test';
import { mkdir,readFile,writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

test.beforeEach(async({page})=>{
  await page.addInitScript(()=>{
    const Native=window.Worker;
    (window as any).drawaiMessages=[];(window as any).drawaiWorkers=[];
    window.Worker=class extends Native{constructor(url:string|URL,options?:WorkerOptions){super(url,options);(window as any).drawaiWorkers.push(String(url));this.addEventListener('message',e=>(window as any).drawaiMessages.push(e.data));}};
  });
});
async function sample(page:Page,id:number){await page.getByRole('button',{name:`Thử mẫu ${id}`,exact:true}).click();await expect(page.getByTestId('drawai-reference')).toContainText(`Nhãn mẫu: ${id}`);}
async function recognize(page:Page){await page.getByRole('button',{name:'Nhận dạng →',exact:true}).click();await expect(page.getByTestId('drawai-prediction')).toBeVisible({timeout:90000});}
async function draw(page:Page){const canvas=page.getByRole('img',{name:'Vùng vẽ chữ số'});await canvas.scrollIntoViewIfNeeded();const b=(await canvas.boundingBox())!;await page.mouse.move(b.x+b.width*.3,b.y+b.height*.2);await page.mouse.down();await page.mouse.move(b.x+b.width*.65,b.y+b.height*.7,{steps:15});await page.mouse.up();}

test('all ten demos use real Swin, match D4 logits and reuse one worker/session',async({page})=>{
  const requests:{url:string;method:string}[]=[];page.on('request',r=>requests.push({url:r.url(),method:r.method()}));
  let modelRequests=0;page.on('request',r=>{if(r.url().endsWith('/drawai-swin-v1/drawai_swin1d_pendigits_l128_fp32_v1.onnx'))modelRequests++;});
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/drawai');
  await expect(page.getByRole('button',{name:'Nhận dạng →'})).toBeDisabled();
  const fixtures=JSON.parse(await readFile('public/models/drawai-swin-v1/fixtures.json','utf8')).cases;
  const observations=[];
  for(let digit=0;digit<10;digit++){
    await sample(page,digit);await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);await recognize(page);
    const result=await page.evaluate(()=>(window as any).drawaiMessages.filter((m:any)=>m.type==='result').at(-1));
    const expected=fixtures.find((f:any)=>f.name===`train-digit-${digit}`);
    expect(result.logits).toHaveLength(10);expect(result.scores.reduce((a:number,b:number)=>a+b,0)).toBeCloseTo(1,6);
    result.logits.forEach((v:number,i:number)=>expect(Math.abs(v-expected.expected[i])).toBeLessThanOrEqual(1e-5+1e-4*Math.abs(expected.expected[i])));
    expect(result.predicted_id).toBe(expected.expected.indexOf(Math.max(...expected.expected)));
    const scores=await page.locator('.da-scores li strong').allTextContents();expect(scores).toHaveLength(3);
    observations.push({reference:digit,...result});
  }
  expect(modelRequests).toBe(1);expect(await page.evaluate(()=>(window as any).drawaiWorkers.length)).toBe(1);expect(errors).toEqual([]);
  expect(requests.every(r=>r.method==='GET'&&r.url.startsWith('http://127.0.0.1:4186/'))).toBe(true);
  await mkdir('../artifacts/drawai/ui',{recursive:true});
  await writeFile('../artifacts/drawai/ui/demo-parity.json',JSON.stringify({status:'passed',observations,modelRequests,workers:1,requests},null,2));
});

test('worker validates timestamp/schema and product rejects metadata tamper with consistent hashes',async({page})=>{
  await page.goto('/drawai');await sample(page,1);await recognize(page);
  const invalid=await page.evaluate(async()=>{
    const worker=new Worker((window as any).drawaiWorkers[0],{type:'module'});
    const base={schema_version:'drawai-strokes-v1',coordinate_system:'canvas-y-down',canvas:{width:400,height:400},strokes:[[{x:0,y:0},{x:2,y:2}]]};
    const cases=[{...base,unexpected:1},{...base,source:null},{...base,strokes:[[{x:0,y:0,t:2},{x:2,y:2,t:1}]]},{...base,strokes:[[{x:0,y:0,t:0},{x:2,y:2}]]}];
    const codes=[];
    for(let i=0;i<cases.length;i++){
      const response=await new Promise<any>(resolve=>{worker.onmessage=e=>{if(e.data.type==='error'||e.data.type==='result')resolve(e.data);};worker.postMessage({protocol_version:'drawai-worker-v1',request_id:i+1,drawing_revision:0,release_version:'drawai-swin-v1',type:'predict',drawing:cases[i]});});codes.push(response.code);
    }
    worker.terminate();return codes;
  });
  expect(invalid).toEqual(['SCHEMA','SCHEMA','TIMESTAMP','TIMESTAMP']);
  const base='/models/drawai-swin-v1/';const release=await(await page.request.get(base+'release.json')).json();const prep=await(await page.request.get(base+'preprocessing.json')).json();prep.channels=['y','x','stroke_start'];const body=JSON.stringify(prep);
  release.files['preprocessing.json']={bytes:Buffer.byteLength(body),sha256:createHash('sha256').update(body).digest('hex')};
  await page.route('**'+base+'preprocessing.json',r=>r.fulfill({status:200,body,contentType:'application/json'}));
  await page.route('**'+base+'release.json',r=>r.fulfill({status:200,body:JSON.stringify(release),contentType:'application/json'}));
  await page.reload();await sample(page,1);await page.getByRole('button',{name:'Nhận dạng →'}).click();
  await expect(page.getByRole('alert')).toContainText('CONTRACT');await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);
});

test('canvas inference, undo, clear and JSON keep reference independent of prediction',async({page})=>{
  await page.goto('/drawai');await draw(page);await recognize(page);
  await expect(page.getByTestId('drawai-prediction')).toBeVisible();
  await page.locator('.da-label summary').click();await page.getByLabel('Chữ số bạn muốn viết').selectOption('7');
  const download=page.waitForEvent('download');await page.getByRole('button',{name:'Xuất JSON ↓'}).click();
  const file=await download;await file.saveAs('../artifacts/drawai/ui/user-strokes.json');
  const exported=JSON.parse(await readFile('../artifacts/drawai/ui/user-strokes.json','utf8'));
  expect(exported.reference_label).toBe(7);expect(exported.source.kind).toBe('user');expect(exported.strokes).toHaveLength(1);expect(exported.predicted_id).toBeUndefined();
  await draw(page);await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);
  await page.getByRole('button',{name:'↶ Hoàn tác nét'}).click();await expect(page.locator('.da-count')).toHaveText('1 nét');
  await page.getByRole('button',{name:'Xóa hết'}).click();await expect(page.getByRole('button',{name:'Nhận dạng →'})).toBeDisabled();
});

test('clear during delayed model load discards stale result and allows a new request',async({page})=>{
  let release!:()=>void;const gate=new Promise<void>(r=>release=r);
  await page.route('**/drawai-swin-v1/drawai_swin1d_pendigits_l128_fp32_v1.onnx',async r=>{await gate;await r.continue();});
  await page.goto('/drawai');await sample(page,3);await page.getByRole('button',{name:'Nhận dạng →'}).click();
  await expect(page.getByTestId('drawai-app-status')).toContainText('Đang tải');
  await page.getByRole('button',{name:'Xóa hết'}).click();release();
  await expect(page.getByRole('button',{name:'Nhận dạng →'})).toBeVisible({timeout:90000});
  await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);
  await sample(page,4);await recognize(page);await expect(page.getByTestId('drawai-prediction')).toHaveAttribute('data-digit','4');
});

test('load failure preserves strokes and retry runs real model',async({page})=>{
  const pattern='**/drawai-swin-v1/drawai_swin1d_pendigits_l128_fp32_v1.onnx';await page.route(pattern,r=>r.fulfill({status:404,body:'missing'}));
  await page.goto('/drawai');await sample(page,2);await page.getByRole('button',{name:'Nhận dạng →'}).click();
  await expect(page.getByRole('alert')).toContainText('FETCH');await expect(page.locator('.da-count')).not.toHaveText('0 nét');
  await page.unroute(pattern);await page.getByRole('button',{name:'Thử nhận dạng lại →'}).click();
  await expect(page.getByTestId('drawai-prediction')).toHaveAttribute('data-digit','2',{timeout:90000});
  expect(await page.evaluate(()=>(window as any).drawaiWorkers.length)).toBe(2);
});

test('hash mismatch blocks product inference',async({page})=>{
  await page.route('**/drawai-swin-v1/labels.json',r=>r.fulfill({status:200,body:'{}'}));
  await page.goto('/drawai');await sample(page,0);await page.getByRole('button',{name:'Nhận dạng →'}).click();
  await expect(page.getByRole('alert')).toContainText('HASH');await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);
});

test('single tap cannot trigger recognition; pointer cancel discards incomplete stroke',async({page})=>{
  await page.goto('/drawai');const canvas=page.getByRole('img',{name:'Vùng vẽ chữ số'});await canvas.click();
  await expect(page.getByRole('button',{name:'Nhận dạng →'})).toBeDisabled();
  await page.getByRole('button',{name:'Xóa hết'}).click();const b=(await canvas.boundingBox())!;
  await page.mouse.move(b.x+20,b.y+20);await page.mouse.down();await page.mouse.move(b.x+120,b.y+120);
  await canvas.dispatchEvent('pointercancel',{pointerId:1});await page.mouse.up();
  await expect(page.locator('.da-count')).toHaveText('0 nét');
});

test('responsive resize retains drawing JSON; samples work by keyboard',async({page})=>{
  await page.goto('/drawai');await page.setViewportSize({width:1366,height:768});
  await page.getByRole('button',{name:'Thử mẫu 6',exact:true}).focus();await page.keyboard.press('Enter');
  await expect(page.getByTestId('drawai-reference')).toContainText('Nhãn mẫu: 6');await recognize(page);
  await page.screenshot({path:'../artifacts/drawai/ui/desktop.png',fullPage:true});
  async function exported(){if(!await page.locator('.da-label').evaluate((node:HTMLDetailsElement)=>node.open))await page.locator('.da-label summary').click();const p=page.waitForEvent('download');await page.getByRole('button',{name:'Xuất JSON ↓'}).click();const d=await p;return JSON.parse(await readFile((await d.path())!,'utf8'));}
  const before=await exported();await page.setViewportSize({width:390,height:844});const after=await exported();expect(after).toEqual(before);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.locator('.da-label summary').click();
  await page.screenshot({path:'../artifacts/drawai/ui/mobile.png',fullPage:true});
});

test('DPR 2 and competing pointer preserve coordinate scale and stroke boundary',async({browser})=>{
  const context=await browser.newContext({viewport:{width:390,height:844},deviceScaleFactor:2});const page=await context.newPage();
  await page.goto('http://127.0.0.1:4186/drawai');const canvas=page.getByRole('img',{name:'Vùng vẽ chữ số'});await canvas.scrollIntoViewIfNeeded();const b=(await canvas.boundingBox())!;
  await page.mouse.move(b.x+b.width/4,b.y+b.height/4);await page.mouse.down();
  await canvas.dispatchEvent('pointerdown',{pointerId:22,isPrimary:false,button:0,clientX:b.x+10,clientY:b.y+10});
  await page.mouse.move(b.x+b.width*3/4,b.y+b.height*3/4,{steps:10});await page.mouse.up();
  await expect(page.locator('.da-count')).toHaveText('1 nét');
  await page.locator('.da-label summary').click();const download=page.waitForEvent('download');await page.getByRole('button',{name:'Xuất JSON ↓'}).click();const file=await download;
  const data=JSON.parse(await readFile((await file.path())!,'utf8'));expect(data.strokes).toHaveLength(1);
  expect(data.strokes[0][0].x).toBeCloseTo(100,0);expect(data.strokes[0].at(-1).x).toBeCloseTo(300,0);
  expect(await canvas.evaluate((node:HTMLCanvasElement)=>node.width)).toBeCloseTo(b.width*2,0);
  await context.close();
});
