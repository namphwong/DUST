import { PROTOCOL, RELEASE, type Drawing, type Prediction } from './types';
import { GEOMETRY_VERSION } from './preprocessing';
export class InferenceClient {
  private worker?:Worker;private next=0;private active?:{id:number;revision:number;resolve:(r:Prediction)=>void;reject:(e:Error)=>void;progress:(s:string)=>void;timer:ReturnType<typeof setTimeout>};
  predict(drawing:Drawing,revision:number,progress:(stage:string)=>void):Promise<Prediction> {
    if(this.active)return Promise.reject(new Error('BUSY: Đang nhận dạng.'));
    if(!this.worker){
      const worker=new Worker(new URL('./inference-worker.ts',import.meta.url),{type:'module'});this.worker=worker;
      worker.onmessage=({data})=>{
        if(this.worker!==worker||!this.active||data.protocol_version!==PROTOCOL||data.request_id!==this.active.id)return;
        if(data.type==='progress'){this.active.progress(data.stage);return;}
        const active=this.active;clearTimeout(active.timer);this.active=undefined;
        if(data.type==='result'&&data.model_version===RELEASE&&data.preprocessing_version===GEOMETRY_VERSION&&data.drawing_revision===active.revision&&
          data.logits?.length===10&&data.logits.every(Number.isFinite)&&data.scores?.length===10&&
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
      this.active={id,revision,resolve,reject,progress,timer:setTimeout(()=>this.reset('WORKER: Quá thời gian 60 giây. Hãy thử lại.'),60000)};
      this.worker!.postMessage({protocol_version:PROTOCOL,type:'predict',request_id:id,drawing_revision:revision,release_version:RELEASE,drawing:structuredClone(drawing)});
    });
  }
  reset(message='WORKER: Đã dừng phiên nhận dạng.') {
    this.worker?.terminate();this.worker=undefined;
    if(this.active){clearTimeout(this.active.timer);this.active.reject(new Error(message));this.active=undefined;}
  }
}
