import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import wasmModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import { GEOMETRY_VERSION } from './preprocessing';
import { RELEASE } from './types';

export async function sha256(bytes:ArrayBuffer) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
}
export async function loadRelease(progress:(stage:string)=>void) {
  const base=`${import.meta.env.BASE_URL}models/${RELEASE}/`;
  const fetchBytes=async(name:string)=>{
    let response:Response;
    try {response=await fetch(base+name,{cache:'no-cache'});}
    catch {throw new Error(`FETCH: Lỗi mạng khi tải ${name}. Hãy thử lại.`);}
    if (!response.ok) throw new Error(`FETCH: Không tải được ${name}. Hãy thử lại.`);
    return response.arrayBuffer();
  };
  progress('fetching');
  const manifest=JSON.parse(new TextDecoder().decode(await fetchBytes('release.json')));
  if(manifest.version!==RELEASE||manifest.trained!==true||!manifest.files) throw new Error('CONTRACT: Release chưa train hoặc sai version.');
  const names=['model.onnx','preprocessing.json','labels.json','report.json'];
  const bytes=await Promise.all(names.map(fetchBytes));progress('verifying');
  for(let i=0;i<names.length;i++) {
    const meta=manifest.files[names[i]];
    if (!meta||meta.bytes!==bytes[i].byteLength||await sha256(bytes[i])!==meta.sha256) throw new Error(`HASH: Sai hash/bytes ${names[i]}.`);
  }
  const decode=(i:number)=>JSON.parse(new TextDecoder().decode(bytes[i]));
  const prep=decode(1),labels=decode(2),report=decode(3);
  const expected={version:GEOMETRY_VERSION,length:128,layout:'NCT',dtype:'float32',y_axis:'down',padding:false,learned_normalization:false,timestamps_used:false,max_strokes:16,max_points:8192,
    channels:['x','y','stroke_start'],coordinate_range:[-.5,.5],normalization:'bbox-center/max-span; preserve aspect ratio',resampling:'per-stroke arc length; minimum endpoints; largest remainder allocation'};
  if(Object.entries(expected).some(([key,value])=>JSON.stringify(prep[key])!==JSON.stringify(value))||
    JSON.stringify(labels)!==JSON.stringify({classes:Array.from({length:10},(_,id)=>({id,name:String(id)}))})||
    report.status!=='passed'||report.trained!==true||report.version!==RELEASE||report.opset!==17||
    report.checkpoint_sha256!=='a5ccb65dd4699b6451bf03ff23b6071382c29443213263f1fbc9ef6089fcf324'||
    report.input_name!=='drawing'||report.output_name!=='logits'||JSON.stringify(report.input_shape)!=='[1,3,128]'||JSON.stringify(report.output_shape)!=='[1,10]'||
    report.sha256!==manifest.files['model.onnx'].sha256||report.model_bytes!==bytes[0].byteLength) throw new Error('CONTRACT: Nhãn, preprocessing hoặc model không khớp release.');
  ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths={wasm:wasmUrl,mjs:wasmModuleUrl};
  progress('initializing');
  let session:ort.InferenceSession;
  try {session=await ort.InferenceSession.create(bytes[0],{executionProviders:['wasm'],graphOptimizationLevel:'all'});}
  catch {throw new Error('INIT: Không khởi tạo được WASM/model. Hãy thử lại.');}
  if(JSON.stringify(session.inputNames)!=='["drawing"]'||JSON.stringify(session.outputNames)!=='["logits"]') {
    await session.release();throw new Error('CONTRACT: Input/output của model sai.');
  }
  return session;
}
