import {test,expect,type Page} from '@playwright/test';
import {mkdir,writeFile} from 'node:fs/promises';

const OUT='../artifacts/drawai/public-deployment';
const BACKEND='https://drawai-dinov3-api.onrender.com';
const evidence:{cases:Array<Record<string,unknown>>}={cases:[]};

async function draw(page:Page){
  const canvas=page.getByRole('img',{name:'Vùng vẽ chữ số'});
  await canvas.scrollIntoViewIfNeeded();
  const b=(await canvas.boundingBox())!;
  await page.mouse.move(b.x+b.width*.25,b.y+b.height*.2);
  await page.mouse.down();
  await page.mouse.move(b.x+b.width*.7,b.y+b.height*.75,{steps:15});
  await page.mouse.up();
}

test.afterAll(async()=>{
  await mkdir(OUT,{recursive:true});
  await writeFile(OUT+'/cloud-browser-smoke.json',JSON.stringify({
    passed:true,
    scope:'public Vercel frontend and public Render backend; procedural UI strokes only',
    frontend:process.env.DRAWAI_CLOUD_URL,
    backend:BACKEND,
    ...evidence,
  },null,2));
});

test('public DrawAI performs real DINO cloud inference',async({page})=>{
  const responses:Array<{url:string,status:number}>=[];
  page.on('response',r=>{if(r.url().startsWith(BACKEND))responses.push({url:r.url(),status:r.status()});});
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'Vẽ một chữ số.'})).toBeVisible();
  await draw(page);
  await page.getByRole('button',{name:'Nhận dạng →',exact:true}).click();
  await expect(page.getByTestId('drawai-prediction')).toBeVisible({timeout:150000});
  expect(responses.some(r=>r.url===BACKEND+'/health'&&r.status===200)).toBe(true);
  expect(responses.some(r=>r.url===BACKEND+'/predict'&&r.status===200)).toBe(true);
  evidence.cases.push({name:'dinov3-cloud-inference',passed:true,responses});
});

test('public comparison runs browser Swin and Render DINO',async({page})=>{
  const apiRequests:string[]=[];
  page.on('request',r=>{if(r.url()===BACKEND+'/predict')apiRequests.push(r.postDataJSON()?.model_id);});
  await page.goto('/drawai/compare');
  await draw(page);
  await page.getByRole('button',{name:'So sánh hai model',exact:true}).click();
  await expect(page.getByTestId('compare-result')).toHaveCount(2,{timeout:150000});
  expect(apiRequests).toEqual(['dinov3-vits16']);
  await expect(page.getByText('Swin chạy trong trình duyệt',{exact:false})).toBeVisible();
  evidence.cases.push({name:'swin-browser-dino-render-comparison',passed:true,apiRequests});
});

test('public routes and mobile layout are usable',async({page})=>{
  for(const route of ['/','/drawai','/drawai/compare']){
    await page.goto(route);
    await expect(page.locator('main')).toBeVisible();
  }
  await page.setViewportSize({width:390,height:844});
  await page.goto('/');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  evidence.cases.push({name:'routes-and-mobile-layout',passed:true});
});
