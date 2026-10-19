import {useEffect,useRef,useState} from 'react';
import {DrawingCanvas} from './drawing-canvas';
import {PaintInferenceClient} from './paint-client';
import {PAINT_MODELS,type PaintModel} from './paint-registry';
import {paintPixels} from './paint-raster';
import {type Stroke} from './preprocessing';
import {type Drawing,type Prediction} from './types';
import './drawai.css';
export function PaintCompare(){
  const [strokes,setStrokes]=useState<Stroke[]>([]),[busy,setBusy]=useState(false),[drawing,setDrawing]=useState(false),[error,setError]=useState('');
  const [results,setResults]=useState<Prediction[]>([]),[imageHash,setImageHash]=useState('');
  const revision=useRef(0),clients=useRef(Object.keys(PAINT_MODELS).map(id=>new PaintInferenceClient(id as PaintModel)));
  useEffect(()=>()=>clients.current.forEach(c=>c.reset()),[]);
  function change(next:Stroke[]){revision.current++;setStrokes(next);setResults([]);setImageHash('');setError('');}
  async function compare(){
    const current=revision.current;setBusy(true);setError('');setResults([]);
    const value:Drawing={schema_version:'drawai-strokes-v1',coordinate_system:'canvas-y-down',canvas:{width:400,height:400},strokes:structuredClone(strokes),source:{kind:'user'}};
    try{
      const bytes=paintPixels(value),hash=await crypto.subtle.digest('SHA-256',bytes);
      const results=[];for(const client of clients.current)results.push(await client.predict(value,current,()=>{}));
      if(revision.current===current){setResults(results);setImageHash(Array.from(new Uint8Array(hash),v=>v.toString(16).padStart(2,'0')).join(''));}
    }catch(e){if(revision.current===current)setError(e instanceof Error?e.message:String(e));}
    finally{setBusy(false);}
  }
  return <main className="drawai-shell"><h1>Kiểm chứng hai model DrawAI</h1><p><a href="/drawai">Về DrawAI</a> · DINOv3 là mặc định, chọn từ validation P5. Không fusion.</p>
    <p>Cùng nét → cùng ảnh xám 64×64. Swin nhận hàng pixel; DINOv3 nhận RGB224. Cả hai chạy backend CPU local, không lưu ảnh.</p>
    <div className="da-canvas-wrap" style={{maxWidth:400}}><DrawingCanvas strokes={strokes} onChange={change} onDrawing={setDrawing}/></div>
    <button disabled={!strokes.length||drawing||busy} onClick={compare}>So sánh hai model</button> <button disabled={drawing} onClick={()=>change([])}>Xóa hết</button>
    <p role="status">{busy?'Đang chạy hai model…':results.length?'Đã chạy cùng ảnh qua hai model.':'Vẽ một chữ số để kiểm chứng.'}</p>{error&&<p role="alert">{error}</p>}
    {!!results.length&&<><table><thead><tr><th>Model</th><th>Dự đoán</th><th>Top 3</th><th>Tổng ms</th></tr></thead><tbody>{results.map(r=><tr key={r.model_version} data-testid="compare-result"><td>{PAINT_MODELS[r.model_version as PaintModel].name}</td><td>{r.predicted_id}</td><td>{r.scores.map((score,id)=>({score,id})).sort((a,b)=>b.score-a.score).slice(0,3).map(x=>`${x.id}: ${(x.score*100).toFixed(1)}%`).join(' · ')}</td><td>{r.timing_ms.total.toFixed(1)}</td></tr>)}</tbody></table><p>SHA-256 ảnh đầu vào: <code>{imageHash}</code></p></>}
    <p>Điểm softmax không phải độ chính xác bảo đảm. Chất lượng Paint P5 không đại diện chất lượng canvas; còn giới hạn tương đồng qua split và chưa thử người/thiết bị cảm ứng thật.</p>
  </main>;
}
