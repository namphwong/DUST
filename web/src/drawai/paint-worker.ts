/// <reference lib="webworker" />
import {paintPixels,pixelsBase64} from './paint-raster';
import {CANVAS_ADAPTER,PAINT_BUNDLE_SHA256,PAINT_MODELS,PAINT_PREPROCESSING} from './paint-registry';
import {PROTOCOL} from './types';
self.onmessage=async({data})=>{
  const send=(value:object)=>self.postMessage({protocol_version:PROTOCOL,request_id:data.request_id,...value});
  try{
    if(data.protocol_version!==PROTOCOL||data.type!=='predict'||!Number.isSafeInteger(data.request_id)||data.request_id<0||!Number.isSafeInteger(data.drawing_revision)||data.drawing_revision<0||typeof data.model_id!=='string'||!Object.hasOwn(PAINT_MODELS,data.model_id))throw new Error('CONTRACT: Yêu cầu không hợp lệ.');
    send({type:'progress',stage:'preprocessing'});const start=performance.now(),pixels=paintPixels(data.drawing);
    const digest=await crypto.subtle.digest('SHA-256',pixels),pixelHash=Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join(''),pre=performance.now()-start;
    send({type:'progress',stage:'inference'});
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);
    try{
      const response=await fetch('/api/drawai/predict',{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,
        body:JSON.stringify({protocol:'drawai-paint-api-v1',request_id:data.request_id,drawing_revision:data.drawing_revision,model_id:data.model_id,canvas_adapter:CANVAS_ADAPTER,pixels64_base64:pixelsBase64(pixels)})});
      if(!response.ok)throw new Error(`BACKEND: Không thể nhận dạng (${response.status}). Kiểm tra backend DrawAI local rồi thử lại.`);
      const r=await response.json(),format=PAINT_MODELS[data.model_id as keyof typeof PAINT_MODELS].format;
      if(r.protocol!=='drawai-paint-api-v1'||r.request_id!==data.request_id||r.drawing_revision!==data.drawing_revision||r.model_id!==data.model_id||r.bundle_sha256!==PAINT_BUNDLE_SHA256||r.pixels_sha256!==pixelHash||r.canvas_adapter!==CANVAS_ADAPTER||r.preprocessing_version!==PAINT_PREPROCESSING||r.artifact_format!==format||r.artifact_verified!==true||r.trained!==true||r.data_persisted!==false||!Array.isArray(r.logits)||r.logits.length!==10||!r.logits.every(Number.isFinite)||r.predicted_id!==r.logits.indexOf(Math.max(...r.logits))||![r.preprocessing_ms,r.inference_ms,r.server_total_ms].every(v=>Number.isFinite(v)&&v>=0))throw new Error('CONTRACT: Phản hồi model không khớp gói đã khóa.');
      const exp=r.logits.map((x:number)=>Math.exp(x-Math.max(...r.logits))),sum=exp.reduce((a:number,b:number)=>a+b,0);
      send({type:'result',model_version:r.model_id,preprocessing_version:PAINT_PREPROCESSING,drawing_revision:data.drawing_revision,
        logits:r.logits,scores:exp.map((x:number)=>x/sum),predicted_id:r.predicted_id,pixels_sha256:pixelHash,
        timing_ms:{preprocessing:pre+r.preprocessing_ms,inference:r.inference_ms,total:performance.now()-start}});
    }finally{clearTimeout(timer);}
  }catch(e){send({type:'error',message:e instanceof Error?e.message:String(e)});}
};
