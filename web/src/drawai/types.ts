import { type Stroke } from './preprocessing';
export const RELEASE='drawai-swin-v1';
export const PROTOCOL='drawai-worker-v1';
export type Drawing={schema_version:'drawai-strokes-v1';coordinate_system:'canvas-y-down';canvas:{width:number;height:number};strokes:Stroke[];reference_label?:number|null;source?:{kind:'user'|'dataset'|'synthetic';sample_id?:string;split?:string}};
export type Prediction={protocol_version:typeof PROTOCOL;type:'result';request_id:number;drawing_revision:number;model_version:string;preprocessing_version:string;logits:number[];scores:number[];predicted_id:number;timing_ms:{preprocessing:number;inference:number;total:number}};
export function validateDrawing(value:unknown):asserts value is Drawing {
  const d=value as Drawing;
  const only=(o:object,keys:string[])=>Object.keys(o).every(k=>keys.includes(k));
  const number=(v:unknown)=>typeof v==='number'&&Number.isFinite(v);
  if (!d||typeof d!=='object'||!only(d,['schema_version','coordinate_system','canvas','strokes','reference_label','source'])||d.schema_version!=='drawai-strokes-v1'||d.coordinate_system!=='canvas-y-down'||
      !d.canvas||!only(d.canvas,['width','height'])||![d.canvas.width,d.canvas.height].every(v=>number(v)&&v>0&&v<=1e6)||!Array.isArray(d.strokes)||!d.strokes.length||d.strokes.length>16) throw new Error('SCHEMA: Bản vẽ không đúng hợp đồng.');
  if (d.reference_label!==undefined&&d.reference_label!==null&&(!Number.isInteger(d.reference_label)||d.reference_label<0||d.reference_label>9)) throw new Error('SCHEMA: Nhãn tham chiếu phải từ 0 đến 9.');
  if (d.source!==undefined&&(!d.source||typeof d.source!=='object'||!only(d.source,['kind','sample_id','split'])||!['user','dataset','synthetic'].includes(d.source.kind)||
    (d.source.sample_id!==undefined&&(typeof d.source.sample_id!=='string'||!d.source.sample_id.length||d.source.sample_id.length>256))||
    (d.source.split!==undefined&&!['train','validation','test','canvas-evaluation'].includes(d.source.split)))) throw new Error('SCHEMA: Nguồn bản vẽ không hợp lệ.');
  let count=0, timed:number|undefined, previous=-1;
  for (const stroke of d.strokes) {
    if (!Array.isArray(stroke)||!stroke.length) throw new Error('EMPTY_STROKE: Nét rỗng.');
    for (const p of stroke) {
      if (++count>8192) throw new Error('POINTS: Tối đa 8.192 điểm.');
      if (!p||!only(p,['x','y','t'])||![p.x,p.y].every(v=>number(v)&&Math.abs(v)<=1e6)) throw new Error('NUMBER: Tọa độ không hợp lệ.');
      const has=p.t===undefined?0:1;
      if (timed!==undefined&&timed!==has) throw new Error('TIMESTAMP: Thời gian phải có ở tất cả điểm hoặc không có.');
      timed=has;
      if (has&&(!number(p.t)||p.t!<0||p.t!<previous)) throw new Error('TIMESTAMP: Thời gian không được giảm.');
      if (has) previous=p.t!;
    }
  }
}
