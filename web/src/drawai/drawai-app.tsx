import { useEffect,useRef,useState } from 'react';
import { DrawingCanvas } from './drawing-canvas';
import { InferenceClient } from './inference-client';
import { preprocess, type Stroke } from './preprocessing';
import { sha256 } from './release-loader';
import { validateDrawing, type Drawing, type Prediction } from './types';
import './drawai.css';

const stages:Record<string,string>={fetching:'Đang tải bộ nhận dạng…',verifying:'Đang chuẩn bị…',initializing:'Đang khởi động…',preprocessing:'Đang đọc nét…',inference:'Đang nhận dạng…'};
export function DrawAIApp(){
  const [strokes,setStrokes]=useState<Stroke[]>([]),[drawing,setDrawing]=useState(false),[busy,setBusy]=useState(false),[loadingSample,setLoadingSample]=useState(false);
  const [result,setResult]=useState<Prediction>(),[status,setStatus]=useState('Vẽ một chữ số hoặc chọn mẫu để bắt đầu.'),[error,setError]=useState('');
  const [reference,setReference]=useState<number>(),[userLabel,setUserLabel]=useState(''),[source,setSource]=useState<Drawing['source']>({kind:'user'});
  const [modelReady,setModelReady]=useState(false);
  const revision=useRef(0),client=useRef(new InferenceClient()),sampleRequest=useRef(0),mounted=useRef(true);
  useEffect(()=>{document.title='DrawAI · Nhận dạng chữ số';mounted.current=true;return()=>{mounted.current=false;client.current.reset();};},[]);
  function changed(next:Stroke[],ref?:number,origin:Drawing['source']={kind:'user'}){
    revision.current++;sampleRequest.current++;setLoadingSample(false);setStrokes(next);setResult(undefined);setError('');setReference(ref);setSource(origin);setUserLabel('');
    setStatus(next.length?'Nét đã thay đổi. Bấm Nhận dạng để xem kết quả.':'Vẽ một chữ số hoặc chọn mẫu để bắt đầu.');
  }
  let inputError='';try{if(strokes.length)preprocess(strokes,128);}catch(e){inputError=e instanceof Error?e.message:String(e);}
  const valid=strokes.length>0&&!inputError;
  function snapshot():Drawing{return {schema_version:'drawai-strokes-v1',coordinate_system:'canvas-y-down',canvas:{width:400,height:400},strokes:structuredClone(strokes),
    reference_label:userLabel!==''?Number(userLabel):reference??null,source};}
  async function predict(){
    if(!valid||busy||drawing||loadingSample)return;
    const current=revision.current;setBusy(true);setError('');setResult(undefined);
    try {
      const value=snapshot();validateDrawing(value);
      const prediction=await client.current.predict(value,current,stage=>{
        if(mounted.current&&revision.current===current)setStatus(stages[stage]??'Đang xử lý…');
      });
      if(!mounted.current)return;setModelReady(true);
      if(revision.current===current){setResult(prediction);setStatus(`Đã nhận dạng: ${prediction.predicted_id}.`);}
    }catch(e){if(mounted.current){setModelReady(false);if(revision.current===current){setError(e instanceof Error?e.message:String(e));setStatus('Không thể nhận dạng. Nét vẽ được giữ lại; hãy thử lại.');}}}
    finally{if(mounted.current)setBusy(false);}
  }
  async function sample(label:number){
    const token=++sampleRequest.current;setLoadingSample(true);setError('');setStatus('Đang tải mẫu PenDigits…');
    try{
      const base=`${import.meta.env.BASE_URL}samples/drawai/`,response=await fetch(base+'manifest.json',{cache:'no-cache'});
      if(!response.ok)throw new Error('Không tải được danh sách mẫu.');const manifest=await response.json();
      const entry=manifest.samples.find((s:{label:number})=>s.label===label);
      if(!entry||entry.file!==`digit-${label}.json`||entry.split!=='train')throw new Error('Mẫu không đúng nguồn train.');
      const res=await fetch(base+entry.file);if(!res.ok)throw new Error('Không tải được mẫu.');const bytes=await res.arrayBuffer();
      if(bytes.byteLength!==entry.bytes||await sha256(bytes)!==entry.sha256)throw new Error('Hash mẫu không khớp.');
      const value=JSON.parse(new TextDecoder().decode(bytes));validateDrawing(value);
      if(value.reference_label!==label||value.source?.sample_id!==entry.sample_id||value.source?.split!=='train')throw new Error('Nhãn/nguồn mẫu không khớp.');
      // Uniformly fit the demo into the stable logical viewport; preserve aspect ratio.
      const scale=300/Math.max(value.canvas.width,value.canvas.height);
      const next=value.strokes.map(s=>s.map(p=>({x:50+(400-100-value.canvas.width*scale)/2+p.x*scale,y:50+(400-100-value.canvas.height*scale)/2+p.y*scale})));
      if(mounted.current&&sampleRequest.current===token){changed(next,label,value.source);setStatus(`Mẫu ${label} từ PenDigits train. Bấm Nhận dạng để thử.`);}
    }catch(e){if(mounted.current&&sampleRequest.current===token){setError(e instanceof Error?e.message:String(e));setStatus('Không tải được mẫu. Hãy chọn lại.');}}
    finally{if(mounted.current&&sampleRequest.current===token)setLoadingSample(false);}
  }
  function download(){
    const value=snapshot();validateDrawing(value);const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='drawai-strokes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  const top=result?result.scores.map((score,id)=>({score,id})).sort((a,b)=>b.score-a.score||a.id-b.id).slice(0,3):[];
  const state=error?'error':busy?(modelReady?'predicting':'loading'):drawing?'drawing':result?'result':valid?'ready':'empty';
  return <div className="drawai-shell" data-state={state}>
    <header className="da-header"><a href="/drawai" className="da-brand" aria-label="DrawAI trang chủ"><span className="da-logo">d.</span>DrawAI</a><span className="da-local"><i/> Chạy trên thiết bị của bạn</span></header>
    <main className="da-main">
      <section className="da-heading"><h1>Vẽ một chữ số.</h1><p>Viết từ 0 đến 9 vào ô bên dưới, rồi bấm Nhận dạng.</p></section>
      <div className="da-workspace">
        <section className="da-card da-input"><div className="da-card-title"><h2>Vùng vẽ</h2><span className="da-count">{strokes.length} nét</span></div>
          <div className="da-canvas-wrap"><DrawingCanvas strokes={strokes} onChange={s=>changed(s)} onDrawing={setDrawing}/>{!strokes.length&&<div className="da-placeholder" aria-hidden="true"><span>✎</span><strong>Vẽ chữ số ở đây</strong><small>Dùng chuột, cảm ứng hoặc bút</small></div>}</div>
          <div className="da-tools"><button disabled={!strokes.length||drawing} onClick={()=>changed(strokes.slice(0,-1))}>↶ Hoàn tác nét</button><button disabled={!strokes.length||drawing} onClick={()=>changed([])}>Xóa hết</button></div>
          <button className="da-predict" disabled={!valid||drawing||busy||loadingSample} onClick={predict}>{busy?'Đang nhận dạng…':error?'Thử nhận dạng lại →':'Nhận dạng →'}</button>
          <p className="da-status" role="status" data-testid="drawai-app-status">{status}</p>
          {inputError&&!drawing&&<p className="da-error">Nét quá ngắn hoặc không hợp lệ. Hãy vẽ thêm hoặc xóa để thử lại.</p>}
          {error&&<p className="da-error" role="alert">{error}</p>}
          <details className="da-label"><summary>Tùy chọn</summary><label>Chữ số bạn muốn viết <select value={userLabel} onChange={e=>setUserLabel(e.target.value)}><option value="">{reference===undefined?'Không gắn nhãn':`Nhãn mẫu: ${reference}`}</option>{Array.from({length:10},(_,i)=><option key={i} value={i}>{i}</option>)}</select></label><p>Nhãn này do bạn cung cấp, độc lập với dự đoán.</p><button disabled={!valid||drawing} onClick={download}>Xuất JSON ↓</button></details>
        </section>
        <section className="da-card da-output"><div className="da-card-title"><h2>Kết quả</h2></div>
          {result?<div data-testid="drawai-prediction" data-digit={result.predicted_id} data-revision={result.drawing_revision}><div className="da-result-label">CHỮ SỐ DỰ ĐOÁN</div><div className="da-digit">{result.predicted_id}</div><details className="da-alternatives"><summary>Xem các dự đoán khác</summary><ol className="da-scores">{top.map(({id,score})=><li key={id}><b>{id}</b><div><div className="da-track"><span style={{width:`${score*100}%`}}/></div></div><strong>{(score*100).toFixed(1)}%</strong></li>)}</ol><p className="da-disclaimer">Điểm dự đoán, không phải độ chính xác bảo đảm.</p></details></div>:<div className="da-waiting"><div className="da-empty-digit">—</div><p>{busy?'Đang đọc nét của bạn…':'Kết quả sẽ xuất hiện ở đây.'}</p></div>}
          <div className="da-samples"><h3>Hoặc thử một mẫu</h3><div>{Array.from({length:10},(_,i)=><button key={i} aria-label={`Thử mẫu ${i}`} disabled={drawing} onClick={()=>sample(i)}>{i}</button>)}</div>{reference!==undefined&&<small data-testid="drawai-reference">Nhãn mẫu: {reference} · nguồn: PenDigits train</small>}</div>
        </section>
      </div>
      <footer className="da-footer"><p>Viết lớn, rõ và đủ nét. Bạn có thể xóa để thử lại.</p><details><summary>Về DrawAI</summary><p>Nhận dạng một chữ số bằng Swin Transformer 1D. Mẫu có sẵn từ PenDigits. Chất lượng nét người dùng thực chưa được đánh giá.</p><a href="/drawai/release-check">Thông tin kiểm chứng</a></details></footer>
    </main>
  </div>;
}
