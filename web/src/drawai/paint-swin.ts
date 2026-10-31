import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import wasmModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import {PAINT_BUNDLE_SHA256,PAINT_PREPROCESSING} from './paint-registry';

const FILENAME='drawai_swin1d-image_paint35-main-v1_rows64_fp32_v1.onnx';
const MODEL_HASH='9bb8bf5478cb2d8131936a79888505dcbc4f05e331027ca4e2117db7715df59d';
let pending:Promise<ort.InferenceSession>|undefined;
async function load(progress:(stage:string)=>void){
  progress('fetching');const base=`${import.meta.env.BASE_URL}models/drawai-paint35-swin-v1/`;
  const manifestResponse=await fetch(base+'release.json',{cache:'no-cache',signal:AbortSignal.timeout(20000)});
  if(!manifestResponse.ok)throw new Error('FETCH: Không tải được thông tin Swin.');
  const manifest=await manifestResponse.json();
  if(manifest.version!=='drawai-paint35-swin-web-v1'||manifest.bundle_sha256!==PAINT_BUNDLE_SHA256||manifest.preprocessing_version!==PAINT_PREPROCESSING||
     manifest.model_filename!==FILENAME||manifest.sha256!==MODEL_HASH||manifest.bytes!==2391351)throw new Error('CONTRACT: Swin web release không khớp.');
  const response=await fetch(base+FILENAME,{signal:AbortSignal.timeout(30000)});
  if(!response.ok)throw new Error('FETCH: Không tải được Swin. Hãy thử lại.');
  const bytes=await response.arrayBuffer();progress('verifying');
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
  if(bytes.byteLength!==2391351||hash!==MODEL_HASH)throw new Error('HASH: Swin không khớp bản đã xuất.');
  progress('initializing');ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths={wasm:wasmUrl,mjs:wasmModuleUrl};
  const session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  if(JSON.stringify(session.inputNames)!=='["row_pixels"]'||JSON.stringify(session.outputNames)!=='["logits"]'){
    await session.release();throw new Error('CONTRACT: Swin input/output sai.');
  }
  return session;
}
export async function predictSwin(pixels:Uint8Array,progress:(stage:string)=>void){
  if(!pending)pending=load(progress).catch(e=>{pending=undefined;throw e;});
  const session=await pending,input=new Float32Array(4096);
  const pre=performance.now();
  // Match paint_data.swin_rows: (255-gray)/255 float32, then transpose to CL.
  for(let y=0;y<64;y++)for(let x=0;x<64;x++)input[x*64+y]=Math.fround((255-pixels[y*64+x])/255);
  const preprocessing_ms=performance.now()-pre;progress('inference');const t=performance.now();
  const outputs=await session.run({row_pixels:new ort.Tensor('float32',input,[1,64,64])});
  if(outputs.logits.type!=='float32'||JSON.stringify(outputs.logits.dims)!=='[1,10]')throw new Error('CONTRACT: Swin output shape/type sai.');
  const logits=Array.from(outputs.logits.data as Float32Array);
  if(logits.length!==10||!logits.every(Number.isFinite))throw new Error('CONTRACT: Swin logits sai.');
  return {logits,preprocessing_ms,inference_ms:performance.now()-t};
}
