import {test,expect,type Page} from '@playwright/test';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {resolve} from 'node:path';
const OUT=process.env.DRAWAI_EVIDENCE_DIR?`${process.env.DRAWAI_EVIDENCE_DIR}/collection`:'../artifacts/drawai/phase3';

async function enroll(page:Page){
  await page.goto('/collect');await page.getByLabel('Mã người tham gia',{exact:true}).fill('P001');
  await page.getByLabel('Mã hồ sơ đồng ý',{exact:true}).fill('C-P001');
  for(const checkbox of await page.getByRole('checkbox').all())await checkbox.check();
  await page.getByRole('button',{name:'Bắt đầu phiên',exact:true}).click();
}
async function draw(page:Page){
  const b=(await page.getByRole('img',{name:'Vùng vẽ chữ số'}).boundingBox())!;
  await page.mouse.move(b.x+40,b.y+60);await page.mouse.down();
  await page.mouse.move(b.x+110,b.y+170,{steps:8});await page.mouse.move(b.x+250,b.y+270,{steps:8});await page.mouse.up();
}
async function download(page:Page){
  const waiting=page.waitForEvent('download');await page.getByRole('button',{name:'Tải JSON phiên thu'}).click();
  const file=await waiting;return JSON.parse(await readFile((await file.path())!,'utf8'));
}

test('consent and reserved participant codes gate collection; no inference or upload',async({page})=>{
  const requests:string[]=[];page.on('request',r=>requests.push(r.url()));await page.goto('/collect');
  await expect(page.getByRole('button',{name:'Bắt đầu phiên',exact:true})).toBeDisabled();
  await page.getByLabel('Mã người tham gia',{exact:true}).fill('M001');await page.getByLabel('Mã hồ sơ đồng ý',{exact:true}).fill('C-P001');
  for(const checkbox of await page.getByRole('checkbox').all())await checkbox.check();
  await expect(page.getByRole('button',{name:'Bắt đầu phiên',exact:true})).toBeDisabled();
  await page.getByLabel('Mã người tham gia',{exact:true}).fill('P001');await page.getByRole('checkbox').last().uncheck();
  await expect(page.getByRole('button',{name:'Bắt đầu phiên',exact:true})).toBeDisabled();
  expect(requests.some(url=>/\.onnx|infer|__drawai_p2/.test(url))).toBe(false);
  expect(await page.evaluate(()=>localStorage.length)).toBe(0);
});

test('actual canvas session exports valid Python schema and prompt order then resumes',async({page})=>{
  await enroll(page);await draw(page);await page.getByRole('button',{name:'Ghi mẫu và tiếp tục'}).click();
  await expect(page.getByTestId('collection-progress')).toHaveText('1/50');
  const value=await download(page);expect(value.samples).toHaveLength(1);expect(value.samples[0].actual_pointer_types).toEqual(['mouse']);
  await mkdir(`${OUT}/technical-only`,{recursive:true});
  const path=resolve(`${OUT}/technical-only/browser-session-synthetic.json`);
  await writeFile(path,JSON.stringify(value));
  const python=resolve('../.venv-dinov3/Scripts/python.exe');
  const code='import sys;from training.drawai.private_data import load,plan_content,validate_session,schedule;v=load(sys.argv[1]);validate_session(v,plan_content());print(schedule(v["participant_id"],v["stage"])[1]["label"])';
  const next=execFileSync(python,['-c',code,path],{cwd:resolve('..'),encoding:'utf8'}).trim();
  await page.goto('/collect');await page.getByLabel('Tiếp tục từ JSON đã tải').setInputFiles(path);
  await expect(page.getByTestId('collection-progress')).toHaveText('1/50');
  await expect(page.getByTestId('collection-prompt')).toHaveText('Viết chữ số '+next);
  const resumed=await download(page);expect(resumed).toEqual(value);
  await page.screenshot({path:`${OUT}/collector-desktop.png`,fullPage:true});
});

test('tap is not a sample; incorrect real pointer type is blocked',async({page})=>{
  await enroll(page);const canvas=page.getByRole('img',{name:'Vùng vẽ chữ số'});await canvas.click();
  await page.getByRole('button',{name:'Ghi mẫu và tiếp tục'}).click();await expect(page.getByRole('alert')).toContainText('TOO_SHORT');
  await page.getByRole('button',{name:'Xóa bản nháp'}).click();await draw(page);
  await canvas.dispatchEvent('pointerdown',{pointerType:'pen',pointerId:99,isPrimary:false,button:0});
  await page.getByRole('button',{name:'Ghi mẫu và tiếp tục'}).click();
  await expect(page.getByRole('alert')).toContainText('Thiết bị thực khác');await expect(page.getByTestId('collection-progress')).toHaveText('0/50');
});

test('resume rejects metadata/PII tamper and cannot reset quota',async({page})=>{
  const original=JSON.parse(await readFile(`${OUT}/technical-only/browser-session-synthetic.json`,'utf8'));
  for(const changed of [{...original,email:'private@example.test'},{...original,participant_id:'M001'},
    {...original,samples:[{...original.samples[0],label:(original.samples[0].label+1)%10}]}]){
    await page.goto('/collect');await page.getByLabel('Tiếp tục từ JSON đã tải').setInputFiles({name:'bad.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(changed))});
    await expect(page.getByRole('alert')).not.toBeEmpty();await expect(page.getByTestId('collection-progress')).toHaveCount(0);
  }
});

test('mobile collector stays within viewport; protocol available',async({page,request})=>{
  await page.setViewportSize({width:390,height:844});await enroll(page);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:`${OUT}/collector-mobile.png`,fullPage:true});
  expect((await request.get('/collection-protocol.md')).status()).toBe(200);
  const value=await download(page);expect(value.samples).toHaveLength(0);
});
