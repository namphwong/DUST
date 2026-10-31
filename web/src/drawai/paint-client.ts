import { PROTOCOL, type Drawing, type Prediction } from './types';
import {DEFAULT_PAINT_MODEL,PAINT_PREPROCESSING,type PaintModel} from './paint-registry';
import {PUBLIC_PAINT} from './paint-deployment';
export class PaintInferenceClient {
  constructor(private model:PaintModel=DEFAULT_PAINT_MODEL){}
  private worker?:Worker;private next=0;private active?:{id:number;revision:number;resolve:(r:Prediction)=>void;reject:(e:Error)=>void;progress:(s:string)=>void;timer:ReturnType<typeof setTimeout>};
  predict(drawing:Drawing,revision:number,progress:(stage:string)=>void):Promise<Prediction> {
    if(this.active)return Promise.reject(new Error('BUSY: Đang nhận dạng.'));
    if(!this.worker){
      const worker=new Worker(new URL('./paint-worker.ts',import.meta.url),{type:'module'});this.worker=worker;
      worker.onmessage=({data})=>{
        if(this.worker!==worker||!this.active)return;
        if(!data||typeof data!=='object'){this.reset('WORKER: Phản hồi không hợp lệ.');return;}
        if(data.request_id!==this.active.id)return;
        if(data.protocol_version!==PROTOCOL){this.reset('WORKER: Protocol không hợp lệ.');return;}
        if(data.type==='progress'){this.active.progress(data.stage);return;}
        const active=this.active;clearTimeout(active.timer);this.active=undefined;
        if(data.type==='result'&&data.model_version===this.model&&data.preprocessing_version===PAINT_PREPROCESSING&&data.drawing_revision===active.revision&&
          Array.isArray(data.logits)&&data.logits.length===10&&data.logits.every(Number.isFinite)&&Array.isArray(data.scores)&&data.scores.length===10&&
          data.scores.every((n:number)=>Number.isFinite(n)&&n>=0&&n<=1)&&Math.abs(data.scores.reduce((a:number,b:number)=>a+b,0)-1)<1e-6&&
          data.predicted_id===data.logits.indexOf(Math.max(...data.logits))&&data.timing_ms&&
          ['preprocessing','inference','total'].every(k=>Number.isFinite(data.timing_ms[k])&&data.timing_ms[k]>=0))active.resolve(data);
        else {active.reject(new Error(data.message??'WORKER: Response không hợp lệ.'));this.reset();}
      };
      worker.onerror=()=>this.reset('WORKER: Không thể chạy model. Hãy thử lại.');
      worker.onmessageerror=()=>this.reset('WORKER: Không đọc được phản hồi.');
    }
    return new Promise((resolve,reject)=>{
      const id=++this.next;
      this.active={id,revision,resolve,reject,progress,timer:setTimeout(()=>this.reset('WORKER: Quá thời gian chờ. Nét vẫn được giữ; hãy thử lại.'),PUBLIC_PAINT?150000:60000)};
      try{this.worker!.postMessage({protocol_version:PROTOCOL,type:'predict',request_id:id,drawing_revision:revision,model_id:this.model,drawing:structuredClone(drawing)});}
      catch{this.reset('WORKER: Không gửi được bản vẽ. Hãy thử lại.');}
    });
  }
  reset(message='WORKER: Đã dừng phiên nhận dạng.') {
    this.worker?.terminate();this.worker=undefined;
    if(this.active){clearTimeout(this.active.timer);this.active.reject(new Error(message));this.active=undefined;}
  }
}
