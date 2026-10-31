import {test,expect,type Page} from '@playwright/test';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
const API='https://drawai-public-test.invalid';
const OUT='../artifacts/drawai/public-deployment';
const fixtures=JSON.parse(await readFile('../artifacts/drawai/phase7-image64/canvas-fixtures.json','utf8')).cases;
const baseline=JSON.parse(await readFile('../artifacts/drawai/phase7-image64/canvas-parity.json','utf8')).records;
test.beforeEach(async({page})=>{
  await mkdir(OUT+'/ui',{recursive:true});
  await page.addInitScript(()=>{
    const Native=Worker;(window as any).paintMessages=[];
    window.Worker=class extends Native {constructor(url:string|URL,options?:WorkerOptions){super(url,options);(window as any).paintWorkerURL=String(url);this.addEventListener('message',e=>(window as any).paintMessages.push(e.data));}};
  });
  // Real local production API via intercepted HTTPS test hostname, not a cloud claim.
  await page.route(API+'/**',async route=>{
    const response=await route.fetch({url:'http://127.0.0.1:4198'+new URL(route.request().url()).pathname});
    await route.fulfill({response});
  });
});
async function draw(page:Page){
  const canvas=page.getByRole('img',{name:'Vùng vẽ chữ số'});await canvas.scrollIntoViewIfNeeded();const b=(await canvas.boundingBox())!;
  await page.mouse.move(b.x+b.width*.25,b.y+b.height*.2);await page.mouse.down();await page.mouse.move(b.x+b.width*.7,b.y+b.height*.75,{steps:15});await page.mouse.up();
}
async function start(page:Page){await page.goto('/');await draw(page);await page.getByRole('button',{name:'Nhận dạng →',exact:true}).click();await expect(page.getByTestId('drawai-prediction')).toBeVisible();}
test('public worker matches P7 frozen logits and pixels for both real models',async({page})=>{
  await start(page);
  const records=await page.evaluate(async(cases)=>{
    const w=new Worker((window as any).paintWorkerURL,{type:'module'}),records=[];
    for(const f of cases)for(const model of ['swin1d-image','dinov3-vits16']){
      const id=records.length+100;
      const result=await new Promise<any>(resolve=>{
        w.onmessage=e=>{if(e.data.type!=='progress')resolve(e.data);};
        w.postMessage({protocol_version:'drawai-worker-v1',type:'predict',request_id:id,drawing_revision:77,model_id:model,drawing:f.drawing});
      });
      records.push({name:f.name,...result});
    }
    w.terminate();return records;
  },fixtures);
  for(const r of records){
    expect(r.type).toBe('result');const expected=baseline.find((b:any)=>b.name===r.name&&b.model_version===r.model_version);
    expect(r.pixels_sha256).toBe(expected.pixels_sha256);expect(r.predicted_id).toBe(expected.predicted_id);
    const [atol,rtol]=r.model_version==='swin1d-image'?[1e-5,1e-4]:[1e-4,1e-3];
    for(let i=0;i<10;i++)expect(Math.abs(r.logits[i]-expected.logits[i])).toBeLessThanOrEqual(atol+rtol*Math.abs(expected.logits[i]));
  }
  await writeFile(OUT+'/worker-parity.json',JSON.stringify({passed:true,scope:'local public-profile integration; no cloud deployment',records},null,2));
});
test('comparison runs Swin in browser and sends only DINO to API',async({page})=>{
  const requests:any[]=[];page.on('request',r=>{if(r.url()===API+'/predict')requests.push(r.postDataJSON());});
  await page.goto('/drawai/compare');await draw(page);await page.getByRole('button',{name:'So sánh hai model',exact:true}).click();
  await expect(page.getByTestId('compare-result')).toHaveCount(2);
  expect(requests.map(r=>r.model_id)).toEqual(['dinov3-vits16']);
  await expect(page.getByText('Swin chạy trong trình duyệt',{exact:false})).toBeVisible();
});
test('Swin hash corruption fails; retry loads verified graph',async({page})=>{
  await start(page);
  const pattern='**/drawai_swin1d-image_paint35-main-v1_rows64_fp32_v1.onnx';
  await page.route(pattern,r=>r.fulfill({body:Buffer.from('corrupt graph')}));
  const result=await page.evaluate(async drawing=>{
    const w=new Worker((window as any).paintWorkerURL,{type:'module'});
    const r=await new Promise<any>(resolve=>{w.onmessage=e=>{if(e.data.type!=='progress')resolve(e.data);};w.postMessage({protocol_version:'drawai-worker-v1',type:'predict',request_id:99,drawing_revision:9,model_id:'swin1d-image',drawing});});w.terminate();return r;
  },fixtures[0].drawing);
  expect(result.type).toBe('error');expect(result.message).toContain('HASH');
  await page.unroute(pattern);await page.goto('/drawai/compare');await draw(page);await page.getByRole('button',{name:'So sánh hai model',exact:true}).click();
  await expect(page.getByTestId('compare-result')).toHaveCount(2);
});
test('busy API preserves strokes and manual retry succeeds',async({page})=>{
  await page.route(API+'/predict',r=>r.fulfill({status:429,headers:{'Access-Control-Allow-Origin':'http://127.0.0.1:4188'},json:{error:'BUSY'}}));
  await page.goto('/');await draw(page);await page.getByRole('button',{name:'Nhận dạng →',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('đang bận');await expect(page.locator('.da-count')).toHaveText('1 nét');
  await page.unroute(API+'/predict');await page.getByRole('button',{name:'Thử nhận dạng lại →',exact:true}).click();await expect(page.getByTestId('drawai-prediction')).toBeVisible();
});
test('sleep readiness retry, clear cancels wait and ignores stale response',async({page})=>{
  let health=0;await page.route(API+'/health',async r=>{health++;await r.fulfill({status:503,headers:{'Access-Control-Allow-Origin':'http://127.0.0.1:4188'},json:{error:'sleeping'}});});
  await page.goto('/');await draw(page);await page.getByRole('button',{name:'Nhận dạng →',exact:true}).click();
  await expect(page.getByRole('status')).toContainText('đánh thức');await expect.poll(()=>health).toBeGreaterThanOrEqual(2);
  await page.getByRole('button',{name:'Xóa hết',exact:true}).click();await expect(page.locator('.da-count')).toHaveText('0 nét');
  await page.unroute(API+'/health');await draw(page);await page.getByRole('button',{name:'Nhận dạng →',exact:true}).click();await expect(page.getByTestId('drawai-prediction')).toBeVisible();
});
test('wrong backend identity fails before any prediction',async({page})=>{
  let predicted=0;page.on('request',r=>{if(r.url()===API+'/predict')predicted++;});
  await page.route(API+'/health',r=>r.fulfill({headers:{'Access-Control-Allow-Origin':'http://127.0.0.1:4188'},json:{status:'ready',default_model:'wrong'}}));
  await page.goto('/');await draw(page);await page.getByRole('button',{name:'Nhận dạng →',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('CONTRACT');expect(predicted).toBe(0);
});
test('public routes and unchanged layout on desktop/mobile',async({page})=>{
  for(const route of ['/','/drawai','/drawai/']){await page.goto(route);await expect(page.getByRole('heading',{name:'Vẽ một chữ số.'})).toBeVisible();}
  for(const route of ['/collect','/drawai/legacy','/drawai/dinov3-check']){await page.goto(route);await expect(page.getByRole('heading',{name:'Không tìm thấy trang'})).toBeVisible();}
  await page.goto('/');await page.setViewportSize({width:1366,height:768});await expect(page.locator('.da-card')).toHaveCount(2);await page.screenshot({path:OUT+'/ui/desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);await page.screenshot({path:OUT+'/ui/mobile.png',fullPage:true});
});
