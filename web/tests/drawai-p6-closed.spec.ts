import {test,expect} from '@playwright/test';
test('P6 artifact gate is closed by default',async({request})=>{
  for(const name of ['fixtures.json','drawai_swin1d-image_paint35-main-v1_rows64_fp32_v1.onnx']){
    const response=await request.get('/__drawai_p6__/swin1d-image/'+name);expect(response.status()).toBe(404);
  }
});
