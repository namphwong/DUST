import {useEffect,useRef,useState} from 'react';
import {DrawingCanvas} from './drawing-canvas';
import {type Stroke} from './preprocessing';
import {PROTOCOL,participantValid,schedule,sampleDrawing,validateResume,type Stage,type Device} from './collection';
import './collect.css';

export function CollectApp(){
  const [stage,setStage]=useState<Stage>('pilot'),[participant,setParticipant]=useState(''),[device,setDevice]=useState<Device>('mouse');
  const [consents,setConsents]=useState([false,false,false]),[record,setRecord]=useState('');
  const [session,setSession]=useState<any>(),[strokes,setStrokes]=useState<Stroke[]>([]),[drawing,setDrawing]=useState(false),[error,setError]=useState('');
  const pointers=useRef(new Set<string>()),saved=useRef(true);
  useEffect(()=>{document.title='DrawAI · Thu dữ liệu';
    const unload=(e:BeforeUnloadEvent)=>{if(!saved.current){e.preventDefault();e.returnValue='';}};
    window.addEventListener('beforeunload',unload);return()=>window.removeEventListener('beforeunload',unload);},[]);
  const tasks=session?schedule(session.participant_id,session.stage):[],index=session?.samples.length??0,task=tasks[index];
  const eligible=participantValid(participant,stage)&&consents.every(Boolean)&&/^C-[A-Z0-9-]{3,32}$/.test(record);
  function begin(){
    if(!eligible)return;const now=new Date().toISOString();
    setSession({schema_version:'drawai-private-session-v1',protocol_version:PROTOCOL,session_id:`session_${crypto.randomUUID()}`,
      participant_id:participant,stage,input_device:device,started_at:now,
      consent:{version:'drawai-consent-v1',record_ref:record,confirmed_at:now,training:true,evaluation:true,kaggle_publication:true},samples:[]});
    setError('');saved.current=false;
  }
  function clear(){setStrokes([]);pointers.current.clear();setError('');}
  function accept(){
    if(!session||!task||drawing)return;
    try{
      if(pointers.current.size!==1||!pointers.current.has(session.input_device))throw new Error('Thiết bị thực khác thiết bị đã đăng ký; xóa nét rồi dùng đúng thiết bị.');
      const drawingValue=sampleDrawing(strokes,task.label);
      const sample={schema_version:'drawai-private-sample-v1',protocol_version:PROTOCOL,sample_id:`sample_${crypto.randomUUID()}`,
        session_id:session.session_id,participant_id:session.participant_id,stage:session.stage,label:task.label,repetition:task.repetition,
        input_device:session.input_device,actual_pointer_types:[...pointers.current],source_kind:'human-collected',collected_at:new Date().toISOString(),drawing:drawingValue};
      setSession({...session,samples:[...session.samples,sample]});saved.current=false;clear();
    }catch(e){setError(e instanceof Error?e.message:String(e));}
  }
  function download(){
    const blob=new Blob([JSON.stringify(session,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download=`drawai_${session.stage}_${session.participant_id}_${session.session_id}_${index}samples.json`;a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);saved.current=true;
  }
  async function resume(file?:File){
    if(!file)return;
    try{if(file.size>20*1024*1024)throw new Error('File vượt20MiB.');const value=validateResume(JSON.parse(await file.text()));
      setSession(value);clear();saved.current=true;
    }catch(e){setError(e instanceof Error?e.message:String(e));}
  }
  return <main className="drawai-collect">
    <a href="/">Về DrawAI</a><h1>Thu dữ liệu chữ số</h1>
    <p>Viết tự nhiên đúng chữ số được yêu cầu. Không có dự đoán AI trong lúc thu. Nét chỉ nằm trong bộ nhớ trình duyệt cho tới khi tải JSON; không gửi lên server.</p>
    <p>Đọc <a href="/collection-protocol.md" target="_blank" rel="noreferrer">hướng dẫn thu</a>. Không nhập tên, email, số điện thoại. Người điều phối giữ hồ sơ đồng ý và kiểm tra mỗi mã ứng với một người thật.</p>
    {!session?<section aria-label="Đăng ký phiên thu">
      <label>Đợt thu<select value={stage} onChange={e=>setStage(e.target.value as Stage)}><option value="pilot">Pilot — 50 mẫu/người</option><option value="main">Đợt chính — 200 mẫu/người</option></select></label>
      <label>Mã người tham gia<input value={participant} onChange={e=>setParticipant(e.target.value.trim().toUpperCase())} placeholder="P001" maxLength={4}/></label>
      <label>Thiết bị<select value={device} onChange={e=>setDevice(e.target.value as Device)}><option value="mouse">Chuột</option><option value="touch">Cảm ứng</option><option value="pen">Bút</option></select></label>
      <label>Mã hồ sơ đồng ý<input value={record} onChange={e=>setRecord(e.target.value.trim().toUpperCase())} placeholder="C-P001" maxLength={34}/></label>
      {['Tôi đồng ý dùng nét viết để huấn luyện mô hình.','Tôi đồng ý dùng nét viết để đánh giá và báo cáo nghiên cứu.','Tôi đồng ý công bố nét viết và metadata đã ẩn danh lên Kaggle public.'].map((label,i)=><label key={label} className="collection-consent"><input type="checkbox" checked={consents[i]} onChange={e=>setConsents(consents.map((v,j)=>j===i?e.target.checked:v))}/>{label}</label>)}
      <p>Nếu không đồng ý một mục, không bắt đầu thu cho dataset public này. P001–P005 dành cho pilot; M001–M030 dành cho người mới của đợt chính.</p>
      <button disabled={!eligible} onClick={begin}>Bắt đầu phiên</button>
      <label>Tiếp tục từ JSON đã tải<input type="file" accept="application/json,.json" onChange={e=>resume(e.target.files?.[0])}/></label>
    </section>:<section aria-label="Phiên thu đang mở">
      <p>{session.participant_id} · {session.input_device} · <span data-testid="collection-progress">{index}/{tasks.length}</span> mẫu đã ghi trong bộ nhớ.</p>
      <p>Tải bản sao sau mỗi vòng10 mẫu và trước khi đóng trang. Muốn nghỉ: tải JSON rồi mở lại bằng “Tiếp tục”. Không gửi nhiều snapshot của cùng phiên vào inbox; giữ bản mới nhất.</p>
      {task?<>
        <h2 data-testid="collection-prompt">Viết chữ số {task.label}</h2><p>Vòng {task.repetition}/{session.stage==='pilot'?5:20}. Nhãn là chữ số được yêu cầu, không suy từ model.</p>
        <div className="collection-canvas" onPointerDownCapture={e=>pointers.current.add(e.pointerType)}>
          <DrawingCanvas strokes={strokes} onChange={setStrokes} onDrawing={setDrawing}/>
        </div>
        <button disabled={drawing||!strokes.length} onClick={()=>setStrokes(strokes.slice(0,-1))}>Hoàn tác nét</button>{' '}
        <button disabled={drawing} onClick={clear}>Xóa bản nháp</button>{' '}
        <button disabled={drawing||!strokes.length} onClick={accept}>Ghi mẫu và tiếp tục</button>
      </>:<h2>Đủ quota phiên. Tải JSON để bàn giao.</h2>}
      <p><button disabled={drawing} onClick={download}>Tải JSON phiên thu</button></p>
      <p>JSON phiên chứa mã hồ sơ đồng ý: giữ trong inbox private, không upload nguyên file lên Kaggle.</p>
    </section>}
    <p role="alert">{error}</p>
  </main>;
}
