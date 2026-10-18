import {preprocess,type Stroke} from './preprocessing';
export const PROTOCOL='drawai-private-protocol-v1';
export type Stage='pilot'|'main';
export type Device='mouse'|'touch'|'pen';
export function schedule(participant:string,stage:Stage){
  const result:{label:number;repetition:number}[]=[];
  for(let repetition=1;repetition<=(stage==='pilot'?5:20);repetition++){
    let state=2166136261;
    for(const c of `${participant}:${stage}:${repetition}`)state=Math.imul(state^c.charCodeAt(0),16777619)>>>0;
    const digits=Array.from({length:10},(_,i)=>i);
    for(let i=9;i>0;i--){state=(Math.imul(state,1664525)+1013904223)>>>0;const j=state%(i+1);[digits[i],digits[j]]=[digits[j],digits[i]];}
    result.push(...digits.map(label=>({label,repetition})));
  }return result;
}
export function participantValid(id:string,stage:Stage){return stage==='pilot'?/^P00[1-5]$/.test(id):/^M0(0[1-9]|[12][0-9]|30)$/.test(id);}
export function sampleDrawing(strokes:Stroke[],label:number){
  preprocess(strokes,128);
  return {schema_version:'drawai-strokes-v1',coordinate_system:'canvas-y-down',canvas:{width:400,height:400},
    strokes:structuredClone(strokes),reference_label:label,source:{kind:'user'}};
}
export function validateResume(value:any){
  const exact=(o:any,keys:string[])=>o&&typeof o==='object'&&!Array.isArray(o)&&Object.keys(o).length===keys.length&&keys.every(k=>Object.hasOwn(o,k));
  const uuid=(prefix:string,v:any)=>typeof v==='string'&&new RegExp(`^${prefix}_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`).test(v);
  const timestamp=(v:any)=>typeof v==='string'&&/^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/.test(v)&&Number.isFinite(Date.parse(v));
  if(!exact(value,['schema_version','protocol_version','session_id','participant_id','stage','input_device','started_at','consent','samples'])||
    !exact(value.consent,['version','record_ref','confirmed_at','training','evaluation','kaggle_publication']))throw new Error('Các trường JSON phiên thu không đúng schema.');
  if(value?.schema_version!=='drawai-private-session-v1'||value.protocol_version!==PROTOCOL||
    !['pilot','main'].includes(value.stage)||!participantValid(value.participant_id,value.stage)||
    !uuid('session',value.session_id)||!['mouse','touch','pen'].includes(value.input_device)||
    !/^C-[A-Z0-9-]{3,32}$/.test(value.consent?.record_ref)||
    value.consent.version!=='drawai-consent-v1'||!['training','evaluation','kaggle_publication'].every(k=>value.consent[k]===true)||
    !timestamp(value.consent.confirmed_at)||!timestamp(value.started_at)||
    !Array.isArray(value.samples))throw new Error('Session/protocol/consent không hợp lệ.');
  const tasks=schedule(value.participant_id,value.stage),ids=new Set();
  let previous=Date.parse(value.started_at);const now=Date.now();
  if(Date.parse(value.consent.confirmed_at)>previous||previous>now)throw new Error('Thời gian consent/phiên thu không hợp lệ.');
  if(value.samples.length>tasks.length)throw new Error('Session vượt quota.');
  for(const [i,s] of value.samples.entries()){
    if(!exact(s,['schema_version','protocol_version','sample_id','session_id','participant_id','stage','label','repetition','input_device','actual_pointer_types','source_kind','collected_at','drawing'])||
      s.schema_version!=='drawai-private-sample-v1'||s.protocol_version!==PROTOCOL||
      !uuid('sample',s.sample_id)||ids.has(s.sample_id)||s.session_id!==value.session_id||
      s.participant_id!==value.participant_id||s.stage!==value.stage||s.input_device!==value.input_device||
      s.label!==tasks[i].label||s.repetition!==tasks[i].repetition||s.source_kind!=='human-collected'||
      !timestamp(s.collected_at)||Date.parse(s.collected_at)<previous||Date.parse(s.collected_at)>now||!Array.isArray(s.actual_pointer_types)||s.actual_pointer_types.length!==1||s.actual_pointer_types[0]!==value.input_device||
      !exact(s.drawing,['schema_version','coordinate_system','canvas','strokes','reference_label','source'])||
      !exact(s.drawing.canvas,['width','height'])||!exact(s.drawing.source,['kind'])||
      s.drawing?.reference_label!==s.label||s.drawing.coordinate_system!=='canvas-y-down'||
      s.drawing.schema_version!=='drawai-strokes-v1'||s.drawing.source?.kind!=='user'||
      s.drawing.canvas?.width!==400||s.drawing.canvas?.height!==400)throw new Error(`Mẫu ${i+1} không hợp lệ.`);
    sampleDrawing(s.drawing.strokes,s.label);
    for(const stroke of s.drawing.strokes)for(const point of stroke)
      if(!exact(point,['x','y'])||point.x<0||point.x>400||point.y<0||point.y>400)throw new Error('Tọa độ không hợp lệ.');
    ids.add(s.sample_id);previous=Date.parse(s.collected_at);
  }return value;
}
