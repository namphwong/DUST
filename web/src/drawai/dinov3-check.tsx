import {useRef,useState,useEffect} from 'react';

export function DinoV3Check(){
  const worker=useRef<Worker>(),[busy,setBusy]=useState(false),[status,setStatus]=useState('Sẵn sàng'),[report,setReport]=useState<any>();
  useEffect(()=>{document.title='DrawAI · DINOv3 P2 technical check';return()=>worker.current?.terminate();},[]);
  function run(backend=false){
    setBusy(true);setReport(undefined);setStatus('Đang kiểm chứng kỹ thuật…');worker.current?.terminate();
    const w=backend?new Worker(new URL('./dinov3-backend-worker.ts',import.meta.url),{type:'module'}):new Worker(new URL('./dinov3-worker.ts',import.meta.url),{type:'module'});worker.current=w;
    w.onmessage=({data})=>{
      if(data.type==='progress'){setStatus(data.message);return;}
      setBusy(false);setReport(data.report);setStatus(data.type==='result'&&data.report.passed?'PASS · technical only':`FAIL · ${data.message??'parity or runtime budget'}`);w.terminate();
    };
    w.onerror=()=>{setBusy(false);setStatus('FAIL · worker error');w.terminate();};w.postMessage({});
  }
  return <main>
    <a href="/">Về DrawAI</a><h1>DINOv3 · Kiểm chứng P2</h1>
    <aside><strong>Chỉ kiểm chứng kỹ thuật.</strong> Artifact smoke chưa phải model chữ số đã train. Nguồn backbone (random-init hoặc pretrained) được ghi trong báo cáo. Không dùng kết quả để nhận dạng hoặc báo độ chính xác.</aside>
    <p>Renderer Python/TypeScript → full backbone + head ONNX/WASM. Artifact chỉ được phục vụ local khi bật DRAWAI_P2_LOCAL_SMOKE.</p>
    <button onClick={()=>run(false)} disabled={busy}>Chạy kiểm chứng DINOv3</button>{' '}
    <button onClick={()=>run(true)} disabled={busy}>Kiểm chứng backend local</button>
    <p>Backend local gửi nét tới API 127.0.0.1 trên máy này, không lưu nét. Trang DrawAI chính vẫn chạy Swin trong trình duyệt.</p>
    <p role="status" data-testid="dinov3-status">{status}</p>
    {report&&<pre data-testid="dinov3-report" data-passed={String(report.passed)}>{JSON.stringify(report,null,2)}</pre>}
  </main>;
}
