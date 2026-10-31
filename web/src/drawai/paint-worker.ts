/// <reference lib="webworker" />
import {paintPixels,pixelsBase64} from './paint-raster';
import {CANVAS_ADAPTER,PAINT_BUNDLE_SHA256,PAINT_MODELS,PAINT_PREPROCESSING} from './paint-registry';
import {PROTOCOL} from './types';
import {PUBLIC_PAINT,publicApiBase} from './paint-deployment';
let busy=false;
async function ready(base:string,progress:()=>void){
  const deadline=performance.now()+90000;progress();
  while(performance.now()<deadline){
    try{
      const response=await fetch(base+'/health',{cache:'no-store',signal:AbortSignal.timeout(Math.max(1,Math.floor(Math.min(10000,deadline-performance.now()))))});
      if(response.ok){
        const value=await response.json();
        if(value.status!=='ready'||value.default_model!=='dinov3-vits16'||value.bundle_sha256!==PAINT_BUNDLE_SHA256||
           value.model_sha256!=='5846bd2331f546d1a97e617a0733f7cafd72a953ee8988dd7538940ff2cc6baa'||value.artifact_verified!==true)
          throw new Error('CONTRACT: Backend không khớp model đã khóa.');
        return;
      }
      if(![408,429,502,503,504].includes(response.status))throw new Error(`BACKEND: Không thể kết nối (${response.status}).`);
    }catch(e){if(e instanceof Error&&/^(CONTRACT|BACKEND):/.test(e.message))throw e;}
    await new Promise(resolve=>setTimeout(resolve,Math.min(3000,Math.max(0,deadline-performance.now()))));
  }
  throw new Error('BACKEND: Máy chủ chưa sẵn sàng sau 90 giây. Nét vẫn được giữ; hãy thử lại.');
}
self.onmessage=async({data})=>{
  const send=(value:object)=>self.postMessage({protocol_version:PROTOCOL,request_id:data?.request_id,...value});
  if(busy){send({type:'error',message:'BUSY: Đang nhận dạng.'});return;}
  busy=true;
  try{
    if(!data||typeof data!=='object'||data.protocol_version!==PROTOCOL||data.type!=='predict'||!Number.isSafeInteger(data.request_id)||data.request_id<0||!Number.isSafeInteger(data.drawing_revision)||data.drawing_revision<0||typeof data.model_id!=='string'||!Object.hasOwn(PAINT_MODELS,data.model_id))throw new Error('CONTRACT: Yêu cầu không hợp lệ.');
    send({type:'progress',stage:'preprocessing'});const start=performance.now(),pixels=paintPixels(data.drawing);
    const digest=await crypto.subtle.digest('SHA-256',pixels),pixelHash=Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join(''),pre=performance.now()-start;
    let r:any;
    if(PUBLIC_PAINT&&data.model_id==='swin1d-image'){
      const {predictSwin}=await import('./paint-swin');
      const result=await predictSwin(pixels,stage=>send({type:'progress',stage}));
      r={...result,protocol:'drawai-paint-api-v1',request_id:data.request_id,drawing_revision:data.drawing_revision,model_id:data.model_id,
        bundle_sha256:PAINT_BUNDLE_SHA256,pixels_sha256:pixelHash,canvas_adapter:CANVAS_ADAPTER,preprocessing_version:PAINT_PREPROCESSING,
        artifact_format:'onnx',artifact_verified:true,trained:true,data_persisted:false,predicted_id:result.logits.indexOf(Math.max(...result.logits)),
        server_total_ms:result.preprocessing_ms+result.inference_ms};
    }else{
    const base=PUBLIC_PAINT?publicApiBase():'/api/drawai';
    if(PUBLIC_PAINT)await ready(base,()=>send({type:'progress',stage:'waking'}));
    send({type:'progress',stage:'inference'});
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);
    try{
      const response=await fetch(base+'/predict',{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,
        body:JSON.stringify({protocol:'drawai-paint-api-v1',request_id:data.request_id,drawing_revision:data.drawing_revision,model_id:data.model_id,canvas_adapter:CANVAS_ADAPTER,pixels64_base64:pixelsBase64(pixels)})});
      if(!response.ok)throw new Error(response.status===429?'BACKEND: Máy chủ đang bận. Nét vẫn được giữ; hãy thử lại.':`BACKEND: Không thể nhận dạng (${response.status}). Kiểm tra kết nối rồi thử lại.`);
      r=await response.json();
    }finally{clearTimeout(timer);}
    }
      const format=PAINT_MODELS[data.model_id as keyof typeof PAINT_MODELS].format;
      if(r.protocol!=='drawai-paint-api-v1'||r.request_id!==data.request_id||r.drawing_revision!==data.drawing_revision||r.model_id!==data.model_id||r.bundle_sha256!==PAINT_BUNDLE_SHA256||r.pixels_sha256!==pixelHash||r.canvas_adapter!==CANVAS_ADAPTER||r.preprocessing_version!==PAINT_PREPROCESSING||r.artifact_format!==format||r.artifact_verified!==true||r.trained!==true||r.data_persisted!==false||!Array.isArray(r.logits)||r.logits.length!==10||!r.logits.every(Number.isFinite)||r.predicted_id!==r.logits.indexOf(Math.max(...r.logits))||![r.preprocessing_ms,r.inference_ms,r.server_total_ms].every(v=>Number.isFinite(v)&&v>=0))throw new Error('CONTRACT: Phản hồi model không khớp gói đã khóa.');
      const exp=r.logits.map((x:number)=>Math.exp(x-Math.max(...r.logits))),sum=exp.reduce((a:number,b:number)=>a+b,0);
      send({type:'result',model_version:r.model_id,preprocessing_version:PAINT_PREPROCESSING,drawing_revision:data.drawing_revision,
        logits:r.logits,scores:exp.map((x:number)=>x/sum),predicted_id:r.predicted_id,pixels_sha256:pixelHash,
        timing_ms:{preprocessing:pre+r.preprocessing_ms,inference:r.inference_ms,total:performance.now()-start}});
  }catch(e){send({type:'error',message:e instanceof Error?e.message:String(e)});}
  finally{busy=false;}
};
