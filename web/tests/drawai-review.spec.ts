import {test,expect,type Page} from '@playwright/test';
async function sample(page:Page){await page.getByRole('button',{name:'Thử mẫu 2',exact:true}).click();await expect(page.getByTestId('drawai-reference')).toContainText('Nhãn mẫu: 2');}
for(const fault of ['null','logits-object','scores-object','post-throw']){
  test(`malformed worker ${fault} rejects immediately and retry recovers`,async({page})=>{
    const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
    await page.addInitScript(fault=>{
      const RealWorker=window.Worker;
      window.Worker=class extends EventTarget {
        onmessage:any;onerror:any;onmessageerror:any;
        constructor(){super();window.Worker=RealWorker;}
        terminate(){}
        postMessage(value:any){
          if(fault==='post-throw')throw new Error('Cannot post');
          const result={protocol_version:'drawai-worker-v1',request_id:value.request_id,type:'result',model_version:'dinov3-vits16',preprocessing_version:'drawai-paint-image64-v1',drawing_revision:value.drawing_revision,
            logits:Array(10).fill(0),scores:Array(10).fill(.1),predicted_id:0,timing_ms:{preprocessing:0,inference:0,total:0}};
          const data=fault==='null'?null:{...result,[fault==='logits-object'?'logits':'scores']:{length:10}};
          queueMicrotask(()=>this.onmessage?.({data}));
        }
      } as unknown as typeof Worker;
    },fault);
    await page.goto('/');await sample(page);await page.getByRole('button',{name:'Nhận dạng →'}).click();
    await expect(page.getByRole('alert')).toContainText('WORKER');await expect(page.getByTestId('drawai-prediction')).toHaveCount(0);
    await page.getByRole('button',{name:'Thử nhận dạng lại →'}).click();await expect(page.getByTestId('drawai-prediction')).toBeVisible();expect(errors).toEqual([]);
  });
}
test('worker rejects null request without crashing',async({page})=>{
  await page.addInitScript(()=>{
    const Native=Worker;(window as any).urls=[];
    window.Worker=class extends Native{constructor(url:string|URL,options?:WorkerOptions){super(url,options);(window as any).urls.push(String(url));}};
  });
  await page.goto('/');await sample(page);await page.getByRole('button',{name:'Nhận dạng →'}).click();await expect(page.getByTestId('drawai-prediction')).toBeVisible();
  const result=await page.evaluate(async()=>{
    const worker=new Worker((window as any).urls[0],{type:'module'});
    const result=await new Promise<any>((resolve,reject)=>{worker.onmessage=e=>resolve(e.data);worker.onerror=()=>reject(new Error('Worker crashed'));worker.postMessage(null);});worker.terminate();return result;
  });expect(result.type).toBe('error');expect(result.message).toContain('CONTRACT');
});
test('clear during comparison prevents second stale inference and allows a fresh comparison',async({page})=>{
  let release!:()=>void;const gate=new Promise<void>(r=>release=r),seen:any[]=[];
  await page.route('**/api/drawai/predict',async r=>{seen.push(r.request().postDataJSON());await gate;await r.continue();});
  await page.goto('/drawai/compare');const canvas=page.getByRole('img',{name:'Vùng vẽ chữ số'}),box=(await canvas.boundingBox())!;
  const draw=async()=>{await page.mouse.move(box.x+60,box.y+50);await page.mouse.down();await page.mouse.move(box.x+180,box.y+230,{steps:8});await page.mouse.up();};
  await draw();await page.getByRole('button',{name:'So sánh hai model',exact:true}).click();await expect.poll(()=>seen.length).toBe(1);
  await page.getByRole('button',{name:'Xóa hết'}).click();release();await expect(page.getByRole('status')).toContainText('Vẽ một chữ số');
  expect(seen).toHaveLength(1);await expect(page.getByTestId('compare-result')).toHaveCount(0);
  await draw();await page.getByRole('button',{name:'So sánh hai model',exact:true}).click();await expect(page.getByTestId('compare-result')).toHaveCount(2);expect(seen).toHaveLength(3);
});
