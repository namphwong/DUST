import { useEffect, useRef, useState } from 'react';

export function DrawAIFeasibility({trained=false}:{trained?:boolean}) {
  const [length,setLength] = useState(trained ? 128 : 64);
  const [busy,setBusy] = useState(false);
  const [status,setStatus] = useState('Sẵn sàng kiểm chứng');
  const [result,setResult] = useState<Record<string,unknown>>();
  const workerRef = useRef<Worker>();
  useEffect(()=> { document.title=trained?'DrawAI · Kiểm chứng D4':'DrawAI · Kiểm chứng D2'; return ()=>workerRef.current?.terminate(); },[trained]);
  function run() {
    workerRef.current?.terminate();
    setBusy(true);setResult(undefined);setStatus('Đang khởi động worker…');
    const worker = new Worker(new URL('./feasibility-worker.ts',import.meta.url),{type:'module'});
    workerRef.current=worker;
    worker.onmessage = event=> {
      if (event.data.type==='progress') setStatus(event.data.message);
      else {
        if (event.data.type==='result') {
          setResult(event.data.result);
          setStatus(event.data.result.passed ? 'PASS · Python, TypeScript và WASM khớp' : 'FAIL · Có fixture không đạt');
        } else setStatus(`Lỗi: ${event.data.message}`);
        setBusy(false);worker.terminate();
      }
    };
    worker.onerror=event=> {setStatus(`Lỗi worker: ${event.message}`);setBusy(false);worker.terminate();};
    worker.postMessage({length,trained});
  }
  return <main>
    <div className="eyebrow">DrawAI / {trained?'D4 / Trained release':'D2 / Technical verification'}</div>
    <h1>1D Swin Transformer<br/><span>cho chuỗi nét bút.</span></h1>
    <p className="intro">Kiểm tra tiền xử lý Python–TypeScript và suy luận ONNX thật trong Web Worker.</p>
    {trained ? <aside><strong>Mô hình Swin đã huấn luyện.</strong> L128 · seed 44 · PenDigits test accuracy 98,77%, macro-F1 98,79%. Kiểm chứng artifact; chất lượng nét canvas thuộc D6.</aside> : <aside><strong>Mô hình kỹ thuật chưa huấn luyện.</strong> Trang này kiểm chứng pipeline, chưa cung cấp nhận dạng chữ số. Các nhánh attention/MLP được kích hoạt riêng cho phép thử export.</aside>}
    <section className="contract"><div>INPUT<strong>1 × 3 × {length}</strong></div><div>OUTPUT<strong>10 logits</strong></div><div>FEATURES<strong>x / y / đầu nét</strong></div></section>
    <label>Chiều dài chuỗi <select aria-label="Chiều dài chuỗi" value={length} disabled={busy||trained} onChange={e=>{setLength(Number(e.target.value));setResult(undefined);setStatus('Sẵn sàng kiểm chứng');}}><option value={64}>64 điểm</option><option value={128}>128 điểm</option></select></label>{' '}
    <button onClick={run} disabled={busy}>{busy?'Đang kiểm chứng…':'Chạy kiểm chứng DrawAI'}</button>
    <p data-testid="drawai-status" role="status">{status}</p>
    {result && <section data-testid="drawai-result" data-passed={String(result.passed)}>
      <p><strong>{result.passed?'PASS':'FAIL'}</strong> · {(result.cases as unknown[]).length} fixture · WASM trong worker</p>
      <p>Model {(Number(result.modelBytes)/1024/1024).toFixed(2)} MiB · p95 {Number(result.p95Ms).toFixed(2)} ms</p>
      <details open><summary>Báo cáo kiểm chứng</summary><pre>{JSON.stringify(result,null,2)}</pre></details>
    </section>}
    <footer>{trained?'drawai-swin-v1 · trained=true · Model đã train; canvas sản phẩm thuộc D5.':'drawai-geometry-v1 · Hai cấu hình thử L=64/128 · trained=false · Không dùng logits này làm kết quả nhận dạng.'}</footer>
  </main>;
}
