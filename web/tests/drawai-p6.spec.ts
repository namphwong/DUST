import {test,expect} from '@playwright/test';
import {readFile,writeFile,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';

test('legacy canonical filename and compatibility alias preserve exact graph bytes',async({request})=>{
  const base='/models/drawai-swin-v1/';const release=await(await request.get(base+'release.json')).json();
  expect(release.model_filename).toBe('drawai_swin1d_pendigits_l128_fp32_v1.onnx');
  const canonical=await(await request.get(base+release.model_filename)).body();
  const alias=await(await request.get(base+'model.onnx')).body();
  expect(canonical.equals(alias)).toBe(true);
  expect(createHash('sha256').update(canonical).digest('hex')).toBe('28100de69dec82303a954af5c18936788e7c3dee623d70ec7a89ad9f7ef02366');
});

test('trained row64 Swin full graph runs in WASM and matches procedural fixtures',async({page})=>{
  const root='../models/exported/paint35-v1/swin1d-image/';
  const report=JSON.parse(await readFile(root+'report.json','utf8'));
  const assets=await readdir('dist/assets');
  const wasm='/assets/'+assets.find(n=>n.startsWith('ort-wasm-simd-threaded')&&n.endsWith('.wasm'));
  const mjs='/assets/'+assets.find(n=>n.startsWith('ort-wasm-simd-threaded')&&n.endsWith('.mjs'));
  await page.goto('/');
  await page.addScriptTag({path:'node_modules/onnxruntime-web/dist/ort.wasm.min.js'});
  const result=await page.evaluate(async({expectedHash,expectedBytes,wasm,mjs})=>{
    const ort=(window as any).ort;
    ort.env.wasm.numThreads=1;
    ort.env.wasm.wasmPaths={wasm:location.origin+wasm,mjs:location.origin+mjs};
    const fixtures=await(await fetch('/__drawai_p6__/swin1d-image/fixtures.json')).json();
    const response=await fetch('/__drawai_p6__/swin1d-image/drawai_swin1d-image_paint35-main-v1_rows64_fp32_v1.onnx');
    if(!response.ok)throw new Error('P6 model unavailable');
    const buffer=await response.arrayBuffer();
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),v=>v.toString(16).padStart(2,'0')).join('');
    if(hash!==expectedHash||buffer.byteLength!==expectedBytes)throw new Error('P6 artifact hash mismatch');
    const init=performance.now();
    const session=await ort.InferenceSession.create(buffer,{executionProviders:['wasm']});
    const init_ms=performance.now()-init;
    let maxError=0;const cases=[];let lastInput:any;
    for(const fixture of fixtures.cases){
      const bytes=Uint8Array.from(atob(fixture.png64_base64),(c:string)=>c.charCodeAt(0));
      const bitmap=await createImageBitmap(new Blob([bytes],{type:'image/png'}));
      const canvas=document.createElement('canvas');canvas.width=64;canvas.height=64;
      const ctx=canvas.getContext('2d')!;ctx.drawImage(bitmap,0,0);const rgba=ctx.getImageData(0,0,64,64).data;bitmap.close();
      const input=new Float32Array(4096);
      for(let y=0;y<64;y++)for(let x=0;x<64;x++)input[x*64+y]=Math.fround((255-rgba[(y*64+x)*4])/255);
      const output=await session.run({row_pixels:new ort.Tensor('float32',input,[1,64,64])});
      lastInput=new ort.Tensor('float32',input,[1,64,64]);
      const actual=Array.from(output.logits.data) as number[];
      const passed=actual.every((v,i)=>Number.isFinite(v)&&Math.abs(v-fixture.expected_logits[i])<=fixtures.atol+fixtures.rtol*Math.abs(fixture.expected_logits[i]))&&
        actual.indexOf(Math.max(...actual))===fixture.expected_logits.indexOf(Math.max(...fixture.expected_logits));
      maxError=Math.max(maxError,...actual.map((v,i)=>Math.abs(v-fixture.expected_logits[i])));
      cases.push({name:fixture.name,passed});if(!passed)throw new Error('WASM fixture mismatch '+fixture.name);
    }
    const timings=[];
    for(let i=0;i<35;i++){const t=performance.now();await session.run({row_pixels:lastInput});if(i>=5)timings.push(performance.now()-t);}
    timings.sort((a,b)=>a-b);
    await session.release();return {passed:true,cases,max_abs_error:maxError,model_sha256:hash,bytes:buffer.byteLength,init_ms,
      warm_inference_p50_ms:timings[14],warm_inference_p95_ms:timings[28],warmup:5,iterations:30,timing_scope:'WASM inference only; procedural noise tensor'};
  },{expectedHash:report.model_sha256,expectedBytes:report.model_bytes,wasm,mjs});
  expect(result.passed).toBe(true);expect(result.cases).toHaveLength(9);
  await writeFile('../artifacts/drawai/phase6-image64/browser-swin.json',JSON.stringify(result,null,2));
});
