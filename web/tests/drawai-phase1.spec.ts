import {test,expect} from '@playwright/test';
import {mkdir} from 'node:fs/promises';

test('root, alias and trailing slash all open DrawAI',async({page})=>{
  for(const route of ['/','/drawai','/drawai/','/?phase=1','/?phase=4']){
    await page.goto(route);
    await expect(page).toHaveTitle('DrawAI · Nhận dạng chữ số');
    await expect(page.getByRole('heading',{name:'Vẽ một chữ số.'})).toBeVisible();
    await expect(page.locator('.motion-app')).toHaveCount(0);
  }
});

test('root performs real inference with the existing design',async({page})=>{
  const requests:string[]=[];page.on('request',r=>requests.push(r.url()));
  await page.setViewportSize({width:1366,height:768});await page.goto('/');
  await page.getByRole('button',{name:'Thử mẫu 6',exact:true}).click();
  await expect(page.getByTestId('drawai-reference')).toContainText('Nhãn mẫu: 6');
  await page.getByRole('button',{name:'Nhận dạng →',exact:true}).click();
  await expect(page.getByTestId('drawai-prediction')).toHaveAttribute('data-digit','6',{timeout:90000});
  expect(requests.some(url=>url.endsWith('/drawai-swin-v1/model.onnx'))).toBe(true);
  expect(requests.some(url=>/uci-har|har-swin|motionsense|csv-pipeline/i.test(url))).toBe(false);
  await mkdir('../artifacts/drawai/phase1/ui',{recursive:true});
  await page.screenshot({path:'../artifacts/drawai/phase1/ui/root-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:'../artifacts/drawai/phase1/ui/root-mobile.png',fullPage:true});
});

test('unknown route has an explicit return to DrawAI',async({page})=>{
  await page.goto('/missing-page');
  await expect(page.getByRole('heading',{name:'Không tìm thấy trang'})).toBeVisible();
  await page.getByRole('link',{name:'Về DrawAI',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Vẽ một chữ số.'})).toBeVisible();
});
