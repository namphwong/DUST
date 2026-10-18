import { useEffect,useRef,useState } from 'react';
export function IntegrationCheck(){
  const worker=useRef<Worker>();const [busy,setBusy]=useState(false),[status,setStatus]=useState('Sẵn sàng'),[report,setReport]=useState<any>();
  useEffect(()=>()=>worker.current?.terminate(),[]);
  function run(){setBusy(true);setReport(undefined);setStatus('Đang kiểm tra toàn tuyến và benchmark…');worker.current?.terminate();
    const w=new Worker(new URL('./integration-worker.ts',import.meta.url),{type:'module'});worker.current=w;
    w.onmessage=({data})=>{setBusy(false);if(data.type==='result'){setReport(data.report);setStatus(data.report.passed?'PASS':'FAIL');}else setStatus(data.message);w.terminate();};
    w.onerror=()=>{setBusy(false);setStatus('Lỗi worker');w.terminate();};w.postMessage({});
  }
  return <main><h1>DrawAI · Kiểm chứng D6</h1><p>JSON → Python/TypeScript → PyTorch/ONNX/WASM; đây là fixture kỹ thuật, không phải dữ liệu người tham gia.</p><button disabled={busy} onClick={run}>Chạy kiểm chứng D6</button><p role="status">{status}</p>{report&&<pre data-testid="d6-report" data-passed={String(report.passed)}>{JSON.stringify(report,null,2)}</pre>}</main>;
}
