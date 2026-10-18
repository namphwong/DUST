import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';

test('pretrained backend raw-stroke parity and end-to-end runtime budget',async({page})=>{
  await page.goto('/drawai/dinov3-check');
  await page.getByRole('button',{name:'Kiểm chứng backend local',exact:true}).click();
  await expect(page.getByTestId('dinov3-report')).toBeVisible({timeout:150000});
  const report=JSON.parse((await page.getByTestId('dinov3-report').textContent())!);
  await writeFile('../artifacts/drawai/phase2/backend-report.json',JSON.stringify(report,null,2));
  expect(report.parityPassed).toBe(true);expect(report.pretrainedLoaded).toBe(true);
  expect(report.cases).toHaveLength(19);expect(report.passed).toBe(true);expect(report.dataPersisted).toBe(false);
  await page.screenshot({path:'../artifacts/drawai/phase2/backend-page.png',fullPage:true});
});

test('backend rejects malformed strokes, stale model and oversized requests',async({request})=>{
  const health=await(await request.get('/__drawai_p2_backend__/health')).json();
  const drawing={strokes:[[{x:0,y:0},{x:10,y:10}]],y_axis:'down',model_sha256:health.model_sha256};
  for(const value of [{...drawing,strokes:[]},{...drawing,model_sha256:'wrong'},{...drawing,unexpected:true}]){
    expect((await request.post('/__drawai_p2_backend__/infer',{data:value})).status()).toBe(400);
  }
  expect((await request.post('/__drawai_p2_backend__/infer',{data:'x'.repeat(1048577),headers:{'Content-Type':'application/json'}})).status()).toBe(413);
  expect((await request.post('http://127.0.0.1:4190/infer',{data:'x'.repeat(1048577),headers:{'Content-Type':'application/json'}})).status()).toBe(413);
});

test('backend recovers after invalid request on localhost',async({request})=>{
  const health=await(await request.get('http://127.0.0.1:4190/health')).json();
  expect(health.pretrained_loaded).toBe(true);expect(health.release_ready).toBe(false);expect(health.data_persisted).toBe(false);
  const response=await request.post('/__drawai_p2_backend__/infer',{data:{strokes:[[{x:0,y:0},{x:10,y:10}]],y_axis:'down',model_sha256:health.model_sha256}});
  expect(response.status()).toBe(200);expect((await response.json()).logits).toHaveLength(10);
});
