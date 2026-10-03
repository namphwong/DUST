import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { runSmokeTest, type Result } from './inference-smoke-test';
import { runArtifactTest, type ArtifactResult } from './inference-artifact';
import './style.css';
import { MotionApp } from './motion-app';
import './motion-style.css';

function App() {
  const phase1 = new URLSearchParams(location.search).get('phase') === '1';
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('Sẵn sàng kiểm chứng');
  const [result, setResult] = useState<Result>();
  async function run() {
    setBusy(true); setResult(undefined);
    try {
      const next = await (phase1 ? runSmokeTest(setStatus) : runArtifactTest(setStatus));
      setResult(next);
      setStatus(next.passed ? 'PASS · Đầu ra trình duyệt khớp PyTorch' : 'FAIL · Có mẫu vượt ngưỡng sai số');
    } catch (error) { setStatus(`Lỗi: ${error instanceof Error ? error.message : String(error)}`); }
    finally { setBusy(false); }
  }
  return <main>
    <div className="eyebrow">DuST / {phase1 ? 'PHASE 01 / BROWSER FEASIBILITY' : 'PHASE 04 / TRAINED ARTIFACT'}</div>
    <h1>Swin Transformer 1D<br/><span>chạy ngay trên trình duyệt.</span></h1>
    <p className="intro">Đối chiếu cùng tín hiệu giữa PyTorch, ONNX Python và ONNX Runtime Web. Tính toán bằng WASM CPU trên thiết bị của bạn.</p>
    {phase1 ? <aside><strong>Mô hình chưa huấn luyện.</strong> Đây là phép thử kỹ thuật với tín hiệu tổng hợp; logits không phải kết quả nhận dạng hoạt động. Các nhánh attention và MLP được kích hoạt để kiểm tra export.</aside> : <aside><strong>Mô hình đã train trên Kaggle.</strong> Swin seed 44 · test macro-F1 87,05%, chưa đạt mục tiêu 90%. Demo nghiên cứu UCI HAR; có thể nhầm ngồi–đứng và không bảo đảm trên cảm biến ngoài dataset. Điểm softmax không phải độ chính xác bảo đảm.</aside>}
    <section className="contract"><div>INPUT<strong>1 × 3 × 128</strong></div><div>OUTPUT<strong>1 × 6 logits</strong></div><div>BACKEND<strong>WASM · 1 thread</strong></div></section>
    <button onClick={run} disabled={busy}>{busy ? 'Đang kiểm chứng…' : result ? 'Chạy kiểm chứng lại' : 'Chạy kiểm chứng'}</button>
    <p role="status" aria-live="polite" data-testid="status">{status}</p>
    {result && <section data-testid="result" data-passed={String(result.passed)}>
      <div className="metrics"><div>MODEL<strong>{(result.modelBytes / 1024 / 1024).toFixed(2)} MiB</strong></div><div>TẢI ARTIFACT<strong>{result.loadMs.toFixed(0)} ms</strong></div><div>KHỞI TẠO<strong>{result.initMs.toFixed(0)} ms</strong></div><div>SUY LUẬN MEDIAN<strong>{result.medianMs.toFixed(2)} ms</strong></div></div>
      <h2>Đối chiếu {result.cases.length} mẫu</h2>
      <p>Điều kiện: |web − PyTorch| ≤ 1e-5 + 1e-4 × |PyTorch|. Đo suy luận 30 lần sau 5 lần warm-up.</p>
      <table><thead><tr><th>Mẫu</th><th>Sai số tuyệt đối lớn nhất</th><th>Kết quả</th></tr></thead><tbody>{result.cases.map(item => <tr key={item.name}><td>{item.name}</td><td>{item.maxError.toExponential(3)}</td><td>{item.passed ? 'PASS' : 'FAIL'}</td></tr>)}</tbody></table>
      {!phase1 && result.passed && <section data-testid="activities"><h2>Dự đoán 6 cửa sổ UCI tham chiếu</h2><p>Mẫu đầu tiên của mỗi lớp trong test, không chọn theo dự đoán đúng. Không nối các cửa sổ này thành timeline.</p><table><thead><tr><th>Nhãn nguồn</th><th>Model dự đoán</th><th>Điểm softmax cao nhất</th></tr></thead><tbody>{(result as ArtifactResult).activities.map(item => <tr key={item.name}><td>{item.reference}</td><td>{item.predicted}</td><td>{(Math.max(...item.scores)*100).toFixed(1)}%</td></tr>)}</tbody></table></section>}
      <details><summary>Xem báo cáo đầy đủ và logits</summary><pre>{JSON.stringify(result, null, 2)}</pre></details>
    </section>}
    <footer>{phase1 ? 'Fixture chưa train của Phase 1.' : 'Release har-swin-kaggle-v1 · trained=true · total acceleration có gravity, đơn vị g, 50 Hz. Chưa có upload CSV hoặc worker (Phase 5–6).'} Dữ liệu tham chiếu nằm trong static assets. Không gửi tín hiệu đến server để dự đoán.</footer>
  </main>;
}
const technical = ['1', '4'].includes(new URLSearchParams(location.search).get('phase') ?? '');
createRoot(document.getElementById('root')!).render(technical ? <App/> : <MotionApp/>);
