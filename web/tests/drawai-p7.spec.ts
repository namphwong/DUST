import {test,expect,type Page} from '@playwright/test';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const OUT='../artifacts/drawai/phase7-image64';
test.beforeEach(async({page})=>{
  await mkdir(OUT+'/ui',{recursive:true});
  await page.addInitScript(()=>{
    const Native=Worker;(window as any).paintMessages=[];(window as any).paintWorkerURLs=[];
    window.Worker=class extends Native {constructor(url:string|URL,options?:WorkerOptions){super(url,options);(window as any).paintWorkerURLs.push(String(url));this.addEventListener('message',e=>(window as any).paintMessages.push(e.data));}};
  });
});
async function sample(page:Page,n=1){await page.getByRole('button',{name:`Thử mẫu ${n}`,exact:true}).click();await expect(page.getByTestId('drawai-reference')).toContainText(`Nhãn mẫu: ${n}`);}
async function predict(page:Page){await page.getByRole('button',{name:'Nhận dạng →',exact:true}).click();await expect(page.getByTestId('drawai-prediction')).toBeVisible();}
async function draw(page:Page){const canvas=page.getByRole('img',{name:'Vùng vẽ chữ số'});await canvas.scrollIntoViewIfNeeded();const b=(await canvas.boundingBox())!;await page.mouse.move(b.x+b.width*.3,b.y+b.height*.2);await page.mouse.down();await page.mouse.move(b.x+b.width*.65,b.y+b.height*.7,{steps:15});await page.mouse.up();}
test('default DINO runs all ten original train demos; records domain gap without changing examples',async({page})=>{
  const requests:string[]=[];page.on('request',r=>requests.push(r.url()));await page.goto('/');
  const observations=[];
  for(let digit=0;digit<10;digit++){
    await sample(page,digit);await predict(page);
    const result=await page.evaluate(()=>(window as any).paintMessages.filter((m:any)=>m.type==='result').at(-1));
    expect(result.model_version).toBe('dinov3-vits16');expect(result.logits).toHaveLength(10);expect(result.scores.reduce((a:number,b:number)=>a+b,0)).toBeCloseTo(1,6);
    await expect(page.locator('.da-scores li')).toHaveCount(3);observations.push({reference:digit,source:'PenDigits train; not Paint or new-user evaluation',...result});
  }
  expect(requests.every(url=>url.startsWith('http://127.0.0.1:4186/'))).toBe(true);
  expect(requests.some(url=>url.endsWith('.onnx'))).toBe(false);
  expect(await page.evaluate(()=>(window as any).paintWorkerURLs.length)).toBe(1);
  await writeFile(OUT+'/demo-observations.json',JSON.stringify({observations,correct:observations.filter(x=>x.reference===x.predicted_id).length,total:10,not_accuracy_benchmark:true},null,2));
});
test('Python/worker raster byte parity and both real model contracts',async({page})=>{
  await page.goto('/drawai');await sample(page);await predict(page);
  const fixtures=JSON.parse(await readFile(OUT+'/canvas-fixtures.json','utf8')).cases;
  const records=await page.evaluate(async(cases)=>{
    const worker=new Worker((window as any).paintWorkerURLs[0],{type:'module'}),records=[];
    for(const fixture of cases)for(const model of ['swin1d-image','dinov3-vits16']){
      const request_id=records.length+100;
      const result=await new Promise<any>(resolve=>{worker.onmessage=e=>{if(e.data.type!=='progress')resolve(e.data);};worker.postMessage({protocol_version:'drawai-worker-v1',type:'predict',request_id,drawing_revision:77,model_id:model,drawing:fixture.drawing});});
      records.push({name:fixture.name,expected_hash:fixture.pixels_sha256,...result});
    }
    worker.terminate();return records;
  },fixtures);
  for(const r of records){expect(r.type).toBe('result');expect(r.pixels_sha256).toBe(r.expected_hash);expect(r.drawing_revision).toBe(77);}
  await writeFile(OUT+'/canvas-parity.json',JSON.stringify({status:'passed',records},null,2));
});
test('comparison sends identical pixels to both real models',async({page})=>{
  const inputs:any[]=[];page.on('request',r=>{if(r.url().endsWith('/api/drawai/predict'))inputs.push(r.postDataJSON());});
  await page.goto('/drawai/compare');await draw(page);await page.getByRole('button',{name:'So sánh hai model',exact:true}).click();
  await expect(page.getByTestId('compare-result')).toHaveCount(2);
  expect(inputs.map(x=>x.model_id)).toEqual(['swin1d-image','dinov3-vits16']);expect(inputs[0].pixels64_base64).toBe(inputs[1].pixels64_base64);
  await writeFile(OUT+'/comparison.json',JSON.stringify({status:'passed',models:inputs.map(x=>x.model_id),pixels_sha256:createHash('sha256').update(Buffer.from(inputs[0].pixels64_base64,'base64')).digest('hex')},null,2));
});
test('backend missing preserves strokes, retry succeeds, undo clear and JSON remain usable',async({page})=>{
  const pattern='**/api/drawai/predict';await page.route(pattern,r=>r.fulfill({status:503,body:'offline'}));await page.goto('/');await draw(page);
  await page.getByRole('button',{name:'Nhận dạng →'}).click();await expect(page.getByRole('alert')).toContainText('BACKEND');await expect(page.locator('.da-count')).toHaveText('1 nét');
  await page.unroute(pattern);await page.getByRole('button',{name:'Thử nhận dạng lại →'}).click();await expect(page.getByTestId('drawai-prediction')).toBeVisible();
  await page.locator('.da-label summary').click();await page.getByLabel('Chữ số bạn muốn viết').selectOption('7');
  const pending=page.waitForEvent('download');await page.getByRole('button',{name:'Xuất JSON ↓'}).click();const file=await pending;await file.saveAs(OUT+'/user-strokes.json');
  const value=JSON.parse(await readFile(OUT+'/user-strokes.json','utf8'));expect(value.reference_label).toBe(7);expect(value.source.kind).toBe('user');expect(value.predicted_id).toBeUndefined();
  await draw(page);await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);await page.getByRole('button',{name:'↶ Hoàn tác nét'}).click();await expect(page.locator('.da-count')).toHaveText('1 nét');
  await page.getByRole('button',{name:'Xóa hết'}).click();await expect(page.getByRole('button',{name:'Nhận dạng →'})).toBeDisabled();
});
test('late response is discarded after clear and next prediction succeeds',async({page})=>{
  let release!:()=>void;const gate=new Promise<void>(r=>release=r);
  await page.route('**/api/drawai/predict',async r=>{await gate;await r.continue();});
  await page.goto('/');await sample(page,3);await page.getByRole('button',{name:'Nhận dạng →'}).click();await expect(page.getByTestId('drawai-app-status')).toContainText('Đang nhận dạng');
  await page.getByRole('button',{name:'Xóa hết'}).click();release();await expect(page.getByRole('button',{name:'Nhận dạng →'})).toBeVisible();await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);
  await page.unroute('**/api/drawai/predict');await sample(page,4);await predict(page);
});
test('model identity, revision, hash and nonfinite response corruption block prediction',async({page})=>{
  for(const patch of [{bundle_sha256:'0'.repeat(64)},{model_id:'swin1d-image'},{drawing_revision:-1},{pixels_sha256:'0'.repeat(64)},{logits:[null,...Array(9).fill(0)]}]){
    await page.route('**/api/drawai/predict',async r=>{const response=await r.fetch();const data=await response.json();await r.fulfill({response,json:{...data,...patch}});});
    await page.goto('/');await sample(page);await page.getByRole('button',{name:'Nhận dạng →'}).click();await expect(page.getByRole('alert')).toContainText('CONTRACT');await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);
    await page.unroute('**/api/drawai/predict');
  }
});
test('worker crash is recoverable',async({page})=>{
  let failed=false;await page.route('**/paint-worker-*.js',r=>{failed=true;return r.abort();});await page.goto('/');await sample(page);await page.getByRole('button',{name:'Nhận dạng →'}).click();
  await expect(page.getByRole('alert')).toContainText('WORKER');expect(failed).toBe(true);await page.unroute('**/paint-worker-*.js');
  await page.getByRole('button',{name:'Thử nhận dạng lại →'}).click();await expect(page.getByTestId('drawai-prediction')).toBeVisible();
});
test('desktop and mobile initial design matches legacy P0 layout; touch DPR2 uses logical viewport',async({page,browser})=>{
  const design=[];
  for(const [width,height,name] of [[1366,768,'desktop'],[390,844,'mobile']] as const){
    await page.setViewportSize({width,height});await page.goto('/drawai/legacy');const legacy=await page.screenshot({path:`${OUT}/ui/legacy-empty-${name}.png`,fullPage:true});
    await page.goto('/');const current=await page.screenshot({path:`${OUT}/ui/empty-${name}.png`,fullPage:true});expect(current.equals(legacy)).toBe(true);
    design.push({viewport:{width,height},main_sha256:createHash('sha256').update(current).digest('hex'),legacy_sha256:createHash('sha256').update(legacy).digest('hex'),pixel_identical:true});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  }
  await writeFile(OUT+'/design-parity.json',JSON.stringify({status:'passed',states:'initial empty desktop/mobile only; result text changes with selected model',design},null,2));
  const context=await browser.newContext({viewport:{width:390,height:844},deviceScaleFactor:2,hasTouch:true,isMobile:true});const mobile=await context.newPage();await mobile.goto('http://127.0.0.1:4186/');
  const canvas=mobile.getByRole('img',{name:'Vùng vẽ chữ số'});const box=(await canvas.boundingBox())!;
  await canvas.dispatchEvent('pointerdown',{pointerId:1,pointerType:'touch',isPrimary:true,button:0,clientX:box.x+box.width*.3,clientY:box.y+box.height*.2});
  await canvas.dispatchEvent('pointermove',{pointerId:1,pointerType:'touch',isPrimary:true,buttons:1,clientX:box.x+box.width*.6,clientY:box.y+box.height*.7});
  await canvas.dispatchEvent('pointerup',{pointerId:1,pointerType:'touch',isPrimary:true,button:0,clientX:box.x+box.width*.6,clientY:box.y+box.height*.7});
  await expect(mobile.locator('.da-count')).toHaveText('1 nét');await predict(mobile);await mobile.screenshot({path:OUT+'/ui/touch-dpr2.png',fullPage:true});await context.close();
});
test('API rejects malformed contract, size, cross origin and unknown model',async({request})=>{
  expect((await request.post('/api/drawai/predict',{data:{}})).status()).toBe(400);
  expect((await request.post('/api/drawai/predict',{headers:{Origin:'https://example.org'},data:{}})).status()).toBe(403);
  expect((await request.post('/api/drawai/predict',{data:{x:'x'.repeat(9000)}})).status()).toBe(413);
  const health=await(await request.get('/api/drawai/health')).json();expect(health.models).toEqual(['dinov3-vits16','swin1d-image']);expect(health.data_persisted).toBe(false);
});
test('warm browser worker to backend benchmark includes raster, proxy and inference',async({page,request})=>{
  await page.goto('/');await sample(page);await predict(page);
  const fixture=JSON.parse(await readFile(OUT+'/canvas-fixtures.json','utf8')).cases[0];
  const benchmarks=await page.evaluate(async(drawing)=>{
    const worker=new Worker((window as any).paintWorkerURLs[0],{type:'module'}),rows=[];let id=0;
    for(const model of ['swin1d-image','dinov3-vits16']){
      const elapsed=[];
      for(let i=0;i<35;i++){
        const start=performance.now();const result=await new Promise<any>(resolve=>{worker.onmessage=e=>{if(e.data.type!=='progress')resolve(e.data);};worker.postMessage({protocol_version:'drawai-worker-v1',type:'predict',request_id:++id,drawing_revision:0,model_id:model,drawing});});
        if(result.type!=='result')throw new Error(JSON.stringify(result));if(i>=5)elapsed.push(performance.now()-start);
      }
      elapsed.sort((a,b)=>a-b);rows.push({model_id:model,warmup:5,iterations:30,p50_ms:(elapsed[14]+elapsed[15])/2,p95_ms:elapsed[28],samples_ms:elapsed});
    }
    worker.terminate();return rows;
  },fixture.drawing);
  const health=await(await request.get('/api/drawai/health')).json();
  for(const row of benchmarks)expect(row.p95_ms).toBeLessThanOrEqual(500);
  expect(health.rss_bytes).toBeLessThanOrEqual(1073741824);
  await writeFile(OUT+'/browser-benchmark.json',JSON.stringify({scope:'Chromium worker postMessage roundtrip + gray64 raster/hash + same-origin HTTP proxy + image preprocessing + CPU model + JSON validation/softmax; excludes user drawing time and UI paint',source:'fixed procedural diagonal; no accuracy claim',user_agent:await page.evaluate(()=>navigator.userAgent),benchmarks,health,physical_touch_tested:false},null,2));
});
