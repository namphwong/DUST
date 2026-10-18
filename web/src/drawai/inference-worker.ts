import * as ort from 'onnxruntime-web/wasm';
import { GEOMETRY_VERSION } from './preprocessing';
import { inferDrawing } from './infer-drawing';
import { loadRelease } from './release-loader';
import { PROTOCOL, RELEASE, validateDrawing } from './types';

let session:ort.InferenceSession|undefined,busy=false;
self.onmessage=async({data})=>{
  data=data??{};
  const identity={protocol_version:PROTOCOL,request_id:data.request_id,drawing_revision:data.drawing_revision};
  const send=(payload:object)=>self.postMessage({...identity,...payload});
  if(data.protocol_version!==PROTOCOL||!Number.isInteger(data.request_id)||data.request_id<1||data.release_version!==RELEASE||!['load','predict'].includes(data.type)) {
    send({type:'error',code:'PROTOCOL',message:'PROTOCOL: Yêu cầu không hợp lệ.',retryable:false});return;
  }
  if(busy){send({type:'error',code:'BUSY',message:'BUSY: Worker đang xử lý.',retryable:true});return;}
  busy=true;const start=performance.now();
  try {
    if(data.type==='predict') {
      if(!Number.isInteger(data.drawing_revision)||data.drawing_revision<0)throw new Error('PROTOCOL: Revision không hợp lệ.');
      validateDrawing(data.drawing);
    }
    if(!session)session=await loadRelease(stage=>send({type:'progress',stage}));
    if(data.type==='load')send({type:'ready',model_version:RELEASE,preprocessing_version:GEOMETRY_VERSION,input_shape:[1,3,128],labels:Array.from({length:10},(_,id)=>String(id))});
    else {
      send({type:'progress',stage:'preprocessing'});
      const {logits,scores,predicted_id,preprocessing,inference}=await inferDrawing(session,data.drawing,()=>send({type:'progress',stage:'inference'}));
      send({type:'result',model_version:RELEASE,preprocessing_version:GEOMETRY_VERSION,logits,scores,predicted_id,timing_ms:{preprocessing,inference,total:performance.now()-start}});
    }
  }catch(error){const message=error instanceof Error?error.message:String(error);const code=message.split(':')[0];
    send({type:'error',message,code,retryable:['FETCH','INIT','WORKER','INFERENCE'].includes(code)});
  }finally{busy=false;}
};
