import { useEffect, useRef, useState } from 'react';
import { MAX_BYTES } from './csv-preview';
import { RELEASE } from './inference-artifact';
import { PipelineClient } from './pipeline-client';
import { resultsCsv, type CompactPreview, type PreparationSummary, type CsvResult } from './csv-pipeline';
import { SignalChart } from './signal-chart';

const names = ['Đi bộ', 'Lên cầu thang', 'Xuống cầu thang', 'Ngồi', 'Đứng', 'Nằm'];
const codes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'];
type Demo = { file: string; reference_label_id: number; subject_id: number; source_line_one_based: number; prepared_split: string; original_split: string; csv_sha256: string };
type Source = { name: string; preview: CompactPreview; file: File; demo?: Demo };
const sampleBase = `${import.meta.env.BASE_URL}samples/uci-har-3ch-v1/`;

function saveText(text: string, filename: string) {
  const url = URL.createObjectURL(new Blob(['\uFEFF'+text], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a'); link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function MotionApp() {
  const [demos, setDemos] = useState<Demo[]>([]);
  const [indexError, setIndexError] = useState('');
  const [source, setSource] = useState<Source>();
  const [unit, setUnit] = useState<'g'|'ms2'>('g');
  const [gravity, setGravity] = useState(false);
  const [busy, setBusy] = useState(false);
  const [taskStage, setTaskStage] = useState<'read'|'prepare'|'analyze'>('read');
  const [status, setStatus] = useState('Chọn một mẫu để bắt đầu.');
  const [error, setError] = useState('');
  const [prediction, setPrediction] = useState<CsvResult>();
  const [inputMode, setInputMode] = useState<'file'|'paste'>('file');
  const [pasted, setPasted] = useState('');
  const [dragging, setDragging] = useState(false);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const [prepared, setPrepared] = useState<PreparationSummary>();
  const [confirmed, setConfirmed] = useState(false);
  const [results, setResults] = useState<CsvResult[]>([]);
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [resultPage, setResultPage] = useState(0);
  const client = useRef<PipelineClient>();
  const sequence = useRef(0);
  useEffect(()=>{
    if (!results.length) return;
    resultHeading.current?.focus({preventScroll:true});
    if (window.matchMedia('(max-width: 720px)').matches) resultHeading.current?.scrollIntoView({block:'start',behavior:'auto'});
  },[results]);

  function resetWorker() {
    client.current?.terminate();
    client.current = new PipelineClient();
    setPrepared(undefined); setConfirmed(false); setResults([]); setResultPage(0); setProgress({completed:0,total:0});
    return client.current;
  }
  function invalidateSettings() {
    setPrepared(undefined); setConfirmed(false); setResults([]); setPrediction(undefined); setError('');
  }
  function cancel() {
    ++sequence.current; client.current?.terminate(); client.current = undefined;
    setBusy(false); setPrepared(undefined); setConfirmed(false); setPrediction(undefined); setResults([]);
    setStatus('Đã hủy tác vụ. Không hiển thị/xuất kết quả dở dang; có thể kiểm tra và chạy lại.');
  }
  useEffect(()=>()=>client.current?.terminate(), []);

  async function loadIndex() {
    setIndexError('');
    try {
      const response = await fetch(sampleBase+'index.json');
      if (!response.ok) throw new Error('Không tải được danh sách mẫu UCI.');
      const index = await response.json();
      if (index.artifact_version !== 'uci-har-3ch-v1' || index.samples.length !== 6 || index.samples.some((d: Demo, i: number) => d.reference_label_id !== i || d.file !== codes[i].toLowerCase()+'.csv')) throw new Error('Danh sách mẫu không đúng version/thứ tự lớp.');
      setDemos(index.samples);
    } catch (e) { setIndexError(e instanceof Error ? e.message : 'Không tải được mẫu.'); }
  }
  useEffect(() => { void loadIndex(); }, []);

  async function chooseDemo(demo: Demo) {
    setTaskStage('read');
    const id = ++sequence.current;
    const worker = resetWorker();
    setBusy(true); setError(''); setPrediction(undefined); setSource(undefined); setStatus('Đang tải mẫu UCI…');
    try {
      const response = await fetch(sampleBase+demo.file);
      if (!response.ok) throw new Error('Không tải được CSV mẫu. Hãy thử lại.');
      const bytes = await response.arrayBuffer();
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), n => n.toString(16).padStart(2,'0')).join('');
      if (hash !== demo.csv_sha256) throw new Error('CSV mẫu sai hash, không phân tích dữ liệu đã thay đổi.');
      if (id !== sequence.current) return;
      const file = new File([bytes], demo.file, {type:'text/csv'});
      const preview = await worker.request<CompactPreview>('parse',{file});
      await worker.request('prepare',{unit:'g',gravity:true,demo:true});
      if (id !== sequence.current) return;
      setSource({ name: names[demo.reference_label_id], preview, file, demo }); setUnit('g'); setGravity(true);
      setStatus('Mẫu đã sẵn sàng. Bấm Phân tích mẫu để chạy model trên thiết bị.');
    } catch (e) { if (id === sequence.current) { setError(e instanceof Error ? e.message : 'Lỗi tải mẫu.'); setStatus('Không tải được mẫu.'); } }
    finally { if (id === sequence.current) setBusy(false); }
  }

  async function chooseFile(file: File | undefined) {
    if (!file) return;
    setTaskStage('read');
    const id = ++sequence.current;
    const worker = resetWorker();
    setSource(undefined); setPrediction(undefined); setError(''); setGravity(false); setUnit('g'); setBusy(true); setStatus('Đang đọc CSV cục bộ…');
    try {
      if (file.size > MAX_BYTES) throw new Error('FILE_TOO_LARGE: File phải không quá 10 MiB.');
      const preview = await worker.request<CompactPreview>('parse',{file});
      if (id !== sequence.current) return;
      setSource({ name: file.name, preview, file });
      setStatus('CSV hợp lệ. Chọn đơn vị, xác nhận trọng lực rồi kiểm tra resample trước khi phân tích.');
    } catch (e) { if (id === sequence.current) { setError(e instanceof Error ? e.message : 'Không đọc được UTF-8 CSV.'); setStatus('CSV chưa hợp lệ. Vui lòng sửa rồi chọn lại.'); } }
    finally { if (id === sequence.current) setBusy(false); }
  }

  async function prepare() {
    if (!source || busy || !gravity) return;
    setTaskStage('prepare');
    const id = sequence.current;
    setBusy(true); setError(''); setPrediction(undefined); setResults([]); setConfirmed(false);
    try {
      if (!client.current) {
        const worker = resetWorker();
        await worker.request('parse',{file:source.file});
      }
      const report = await client.current!.request<PreparationSummary>('prepare',{unit,gravity,demo:!!source.demo});
      if (id === sequence.current) { setPrepared(report); setStatus('Đã đổi đơn vị/tách đoạn/resample. Kiểm tra thống kê và xác nhận để phân tích.'); }
    } catch(e) { if (id === sequence.current) setError(e instanceof Error ? e.message : 'Không chuẩn bị được CSV.'); }
    finally { if (id === sequence.current) setBusy(false); }
  }

  async function analyze() {
    if (!source || busy || (!source.demo && (!prepared || !confirmed))) return;
    setTaskStage('analyze');
    const id = sequence.current;
    setBusy(true); setError(''); setPrediction(undefined); setResults([]);
    try {
      if (!client.current) {
        const worker = resetWorker();
        await worker.request('parse',{file:source.file});
        await worker.request('prepare',{unit,gravity,demo:!!source.demo});
      }
      setProgress({completed:0,total:prepared?.windowCount ?? 1});
      const output = await client.current!.request<{results:CsvResult[]}>('analyze',{}, (type,data)=>{
        if (id !== sequence.current) return;
        if (type === 'status') setStatus(data);
        else if (type === 'progress') { setProgress(data); setStatus(`Đang phân tích ${data.completed}/${data.total} cửa sổ…`); }
      });
      if (id === sequence.current) { setResults(output.results); setResultPage(0); setPrediction(output.results[0]); setStatus(`Phân tích hoàn tất · ${output.results.length} cửa sổ · dữ liệu không rời thiết bị.`); }
    } catch (e) { if (id === sequence.current) { setError(e instanceof Error ? e.message : 'Không chạy được model.'); setStatus('Phân tích thất bại. Có thể thử lại.'); } }
    finally { if (id === sequence.current) setBusy(false); }
  }

  function exportResult() {
    if (!results.length || busy) return;
    saveText(resultsCsv(results,unit),'motionsense-result.csv');
  }


  const activeStep = results.length ? 4 : busy && taskStage === 'analyze' ? 3 : source ? 2 : 1;
  const distribution = names.map((name,id)=>({name,id,count:results.filter(r=>r.predictedId===id).length}));
  const ready = !!source && (!!source.demo || (!!prepared && confirmed));
  const nextAction = !source ? 'Chọn CSV hoặc một mẫu UCI để bắt đầu.' : source.demo ? 'Mẫu sẵn sàng. Bấm Phân tích mẫu.' : !gravity ? 'Chọn đơn vị và xác nhận gia tốc bao gồm trọng lực.' : !prepared ? 'Bấm Kiểm tra resample để kiểm tra dữ liệu.' : !confirmed ? 'Đọc thống kê và xác nhận dữ liệu đã xử lý.' : 'Bấm Suy luận CSV để nhận kết quả.';
  function downloadExample() {
    saveText('timestamp_ms,acc_x,acc_y,acc_z\r\n'+Array.from({length:128},(_,i)=>`${i*20},0,0,1`).join('\r\n')+'\r\n','motionsense-example-g.csv');
  }
  return <div className="motion-app">
    <a className="skip-link" href="#workspace">Bỏ qua đến khu vực phân tích</a>
    <header className="site-header"><a className="brand" href="#"><span className="brand-symbol" aria-hidden="true">m</span>MotionSense<span className="brand-tag">DUST / 1D</span></a><nav aria-label="Điều hướng chính"><a className="nav-active" href="#workspace">Phân tích dữ liệu</a><a href="#guide">Hướng dẫn CSV</a><a href="#model-card">Về mô hình</a></nav><span className="private-badge"><i/>Xử lý trên thiết bị</span></header>
    <main className="motion-main">
      <section className="page-intro"><div><p className="overline">GIA TỐC 3 TRỤC → 6 HOẠT ĐỘNG</p><h1>Hiểu chuyển động từ dữ liệu của bạn.</h1><p>Nhập tín hiệu gia tốc. Kiểm tra dữ liệu. Nhận dự đoán từng khoảng thời gian.</p></div><a className="model-chip" href="#model-card"><span className="live-dot"/>Swin Transformer 1D<small>Chạy cục bộ · không tài khoản</small></a></section>
      <ol className="flow-steps" aria-label="Các bước phân tích">{['Nhập dữ liệu','Kiểm tra đầu vào','Phân tích','Nhận kết quả'].map((label,i)=><li key={label} className={activeStep===i+1?'current':activeStep>i+1?'complete':''} aria-current={activeStep===i+1?'step':undefined}><span>{activeStep>i+1?'✓':`0${i+1}`}</span><div>{label}<small>{['CSV hoặc mẫu có sẵn','Đơn vị & chất lượng tín hiệu','Swin 1D trên thiết bị','Xem chi tiết & tải CSV'][i]}</small></div></li>)}</ol>
      <section id="workspace" className="workspace" aria-labelledby="workspace-title">
        <section className="input-panel">
          <div className="panel-heading"><div><p className="section-label">ĐẦU VÀO</p><h2 id="workspace-title">Dữ liệu của bạn</h2></div><span className="panel-icon" aria-hidden="true">↗</span></div><p className="muted">CSV gia tốc X, Y, Z có timestamp mili giây. Dữ liệu chỉ ở thiết bị của bạn.</p>
          <div className="input-tabs" role="tablist" aria-label="Cách nhập dữ liệu">
            {(['file','paste'] as const).map(mode=><button key={mode} role="tab" id={mode+'-tab'} aria-selected={inputMode===mode} aria-controls={mode+'-input-panel'} tabIndex={inputMode===mode?0:-1} onKeyDown={e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();const next=mode==='file'?'paste':'file';setInputMode(next);document.getElementById(next+'-tab')?.focus();}}} onClick={()=>setInputMode(mode)}>{mode==='file'?'Tải file CSV':'Dán dữ liệu'}</button>)}
          </div>
          <div id="file-input-panel" role="tabpanel" aria-labelledby="file-tab" hidden={inputMode!=='file'}>
            <label className={`upload-zone ${dragging?'dragging':''}`} onDragOver={e=>{e.preventDefault();setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);if(e.dataTransfer.files.length!==1){setError('Vui lòng chọn một file CSV mỗi lần.');return;}void chooseFile(e.dataTransfer.files[0]);}}>
              <span className="upload-icon" aria-hidden="true">↑</span><strong>Kéo thả CSV vào đây</strong><span>hoặc bấm để chọn file từ thiết bị</span><span className="upload-limit">Tối đa 10 MiB · 100.000 dòng · UTF-8</span><input aria-label="Chọn file CSV" type="file" accept=".csv,text/csv" onChange={e=>{void chooseFile(e.target.files?.[0]);e.target.value='';}}/>
            </label>
          </div>
          <div id="paste-input-panel" role="tabpanel" aria-labelledby="paste-tab" hidden={inputMode!=='paste'}>
            <label className="paste-label" htmlFor="csv-text">Dán CSV gồm header và các dòng dữ liệu</label><textarea id="csv-text" value={pasted} onChange={e=>setPasted(e.target.value)} placeholder={'timestamp_ms,acc_x,acc_y,acc_z\n0,0.12,0.98,0.03\n20,0.15,1.01,0.02'} spellCheck={false}/>
            <div className="paste-actions"><small>Cùng giới hạn như file CSV.</small><button className="secondary-button" disabled={!pasted.trim()} onClick={()=>void chooseFile(new File([pasted],'du-lieu-da-dan.csv',{type:'text/csv'}))}>Sử dụng dữ liệu đã dán</button></div><p className="field-hint">Bấm Sử dụng để áp dụng nội dung mới và hủy tác vụ cũ.</p>
          </div>
          <div className="input-help"><button className="text-button" onClick={downloadExample}>Tải CSV ví dụ</button><a href="#guide">Xem định dạng ↗</a></div><p className="field-hint">CSV ví dụ: 128 điểm tổng hợp đơn vị g, không có nhãn hoạt động thật.</p>
          {source && <div className="source-file" data-testid="source-file"><span className="file-symbol" aria-hidden="true">≡</span><div><strong>{source.name}</strong><small>{source.preview.sourceRows.toLocaleString('vi-VN')} điểm · {source.demo?'Mẫu UCI tham chiếu':'CSV cục bộ'}</small></div><span className="valid-dot" aria-label="Đã đọc CSV">✓</span></div>}
          {source && !source.demo && <fieldset className="csv-settings" disabled={busy}><legend>Thiết lập tín hiệu</legend><label htmlFor="source-unit">Đơn vị nguồn<select id="source-unit" aria-label="Đơn vị nguồn" value={unit} onChange={e=>{setUnit(e.target.value as 'g'|'ms2');invalidateSettings();}}><option value="g">g — gia tốc trọng trường</option><option value="ms2">m/s² — mét trên giây bình phương</option></select></label><label className="check-label"><input type="checkbox" checked={gravity} onChange={e=>{setGravity(e.target.checked);invalidateSettings();}}/>Gia tốc có bao gồm trọng lực</label><p className="field-hint">Dùng total acceleration, không dùng body acceleration đã tách trọng lực. Không tự đoán đơn vị.</p>{!gravity && <p className="warning-text">INCOMPATIBLE_SIGNAL: Cần xác nhận dữ liệu bao gồm trọng lực để tiếp tục.</p>}<button className="secondary-button prepare-button" disabled={!gravity} onClick={()=>void prepare()}>Kiểm tra resample<span aria-hidden="true">→</span></button></fieldset>}
          {prepared && source && !source.demo && <div className="preparation-report" data-testid="preparation-report"><div className="report-heading"><h3>✓ Dữ liệu đã kiểm tra</h3><span>50 Hz / g</span></div><p>{prepared.sourceRows} dòng nguồn → {prepared.resampledRows} điểm · {prepared.segmentCount} đoạn · {prepared.interpolatedPoints} điểm nội suy · {prepared.windowCount} cửa sổ.</p><p>Bỏ {prepared.droppedSegments} đoạn ngắn ({prepared.droppedPoints} điểm) và {prepared.tailPoints} điểm đuôi. Không nối qua mất mẫu.</p><label className="check-label"><input type="checkbox" checked={confirmed} disabled={busy} onChange={e=>{setConfirmed(e.target.checked);setPrediction(undefined);setResults([]);}}/>Tôi đã kiểm tra và đồng ý dữ liệu resample</label></div>}
          {source && <div className="analysis-actions"><button className="primary-button" disabled={busy||!ready} onClick={()=>void analyze()}>{busy?'Đang phân tích…':source.demo?prediction?'Phân tích mẫu lại':'Phân tích mẫu':'Suy luận CSV'}<span aria-hidden="true">→</span></button>{source.demo && <a download href={sampleBase+source.demo.file} className="secondary-button">Tải CSV mẫu</a>}</div>}
          {!busy && <p className="next-action">ⓘ {results.length?<><a href="#output-title">Xem kết quả phân tích ↗</a> hoặc chọn dữ liệu mới.</>:nextAction}</p>}
          <div className="demo-section"><div className="divider"><span>CHƯA CÓ DỮ LIỆU? THỬ MẪU UCI</span></div><div className="sample-grid" aria-label="Sáu mẫu UCI HAR">{names.map((name,i)=><button key={name} className={`sample-card ${source?.demo?.reference_label_id===i?'selected':''}`} aria-pressed={source?.demo?.reference_label_id===i} disabled={busy||!demos.length} onClick={()=>void chooseDemo(demos[i])}><span className="sample-number">0{i+1}</span><span>{name}</span><span aria-hidden="true">↗</span></button>)}</div><p className="field-hint">Mỗi mẫu là một cửa sổ độc lập có nhãn tham chiếu.</p>{indexError && <div role="alert" className="error-box">{indexError}<button className="text-button" onClick={()=>void loadIndex()}>Tải lại danh sách mẫu</button></div>}</div>
        </section>
        <section className="analysis-panel output-panel" aria-labelledby="output-title" aria-busy={busy}>
          <div className="panel-heading"><div><p className="section-label">ĐẦU RA</p><h2 id="output-title" ref={resultHeading} tabIndex={-1}>Kết quả phân tích</h2></div><span className={`output-status ${results.length?'done':''}`}>{busy?'Đang xử lý':results.length?'Hoàn tất':'Chờ dữ liệu'}</span></div>
          {error && <div className="error-box" role="alert"><strong>Chưa thể xử lý dữ liệu</strong><p>{error}</p><span>Kiểm tra định dạng/thiết lập hoặc chọn lại CSV. Không có kết quả mới được tạo.</span></div>}
          <p className="analysis-status" role="status" aria-live="polite">{status}</p>
          {busy && <div className="worker-progress"><div className="processing-icon" aria-hidden="true">◎</div><h3>{taskStage==='read'?'Đang đọc và kiểm tra CSV':taskStage==='prepare'?'Đang chuẩn bị tín hiệu':'Đang phân tích trên thiết bị'}</h3><p>Bạn có thể hủy hoặc chọn CSV mới bất kỳ lúc nào.</p><progress aria-label="Tiến trình phân tích" max={progress.total||1} value={taskStage==='analyze'?progress.completed:undefined}/><div><span>{taskStage==='analyze'?`${progress.completed}/${progress.total||'—'} cửa sổ`:'Đang kiểm tra dữ liệu đầu vào…'}</span><button className="secondary-button" onClick={cancel}>Hủy tác vụ</button></div></div>}
          {!prediction && !busy && <div className="empty-state" data-testid="output-empty"><div className="empty-output-icon" aria-hidden="true"><span>▥</span><i>↗</i></div><h3>Kết quả của bạn sẽ xuất hiện ở đây</h3><p>{nextAction}</p><ul><li><span>01</span>Nhãn hoạt động theo cửa sổ 2,54 s</li><li><span>02</span>Sáu điểm dự đoán để so sánh</li><li><span>03</span>CSV có timestamp và phiên bản model</li></ul></div>}
          {prediction && source && <section className="prediction-panel" data-testid="prediction">
            <div className="result-summary"><div><small>CỬA SỔ ĐÃ PHÂN TÍCH</small><strong>{results.length.toLocaleString('vi-VN')}</strong></div><div><small>ĐOẠN CÓ KẾT QUẢ</small><strong>{new Set(results.map(r=>r.segmentId)).size}</strong></div><div><small>ĐƠN VỊ NGUỒN</small><strong>{unit==='ms2'?'m/s²':'g'}</strong></div></div>
            <div className="output-distribution"><h3>Phân bố dự đoán của cả file</h3><p>Số cửa sổ theo nhãn, không phải thời lượng; các cửa sổ có chồng lấn.</p><div className="distribution-bar" aria-hidden="true">{distribution.filter(d=>d.count).map(d=><span key={d.id} className={`activity-${d.id}`} style={{width:`${d.count/results.length*100}%`}}/>)}</div><div className="distribution-labels">{distribution.map(d=><span key={d.id}><i className={`activity-${d.id}`}/>{d.name}<b>{d.count}</b></span>)}</div></div>
            <div className="selected-window" data-testid="selected-window"><div className="window-heading"><span>CHI TIẾT CỬA SỔ #{prediction.windowIndex}</span><span>Đoạn {prediction.segmentId} · {(prediction.startMs/1000).toFixed(2)}–{(prediction.endMs/1000).toFixed(2)} s</span></div><div className="prediction-heading"><div><span>HOẠT ĐỘNG DỰ ĐOÁN</span><h3>{prediction.predictedLabel}</h3></div><div><span>ĐIỂM CAO NHẤT</span><strong>{(Math.max(...prediction.scores)*100).toFixed(1)}<small>%</small></strong></div></div><div className="score-list">{names.map((name,i)=><div className={`score-row ${i===prediction.predictedId?'winner':''}`} key={name}><span>{name}</span><div className="score-track"><i style={{width:`${prediction.scores[i]*100}%`}}/></div><b>{(prediction.scores[i]*100).toFixed(1)}%</b></div>)}</div><p className="score-disclaimer">Điểm softmax không phải độ chính xác bảo đảm. Model chỉ phân loại sáu hoạt động này.</p></div>
            {source.demo?<p className="comparison">Nhãn nguồn: <strong>{names[source.demo.reference_label_id]}</strong> · {prediction.predictedId===source.demo.reference_label_id?'Khớp trên cửa sổ này; không phải đo accuracy.':'Không khớp nhãn nguồn. Giữ kết quả sai để minh bạch.'}</p>:<p className="comparison">CSV ngoài bộ đánh giá · không có nhãn tham chiếu. Chất lượng có thể khác theo thiết bị, vị trí và hệ trục cảm biến.</p>}
            <div className="window-list-heading"><h3>Dự đoán theo thời gian</h3><span>{results.length} cửa sổ · stride 64</span></div>
            <div className="table-scroll" tabIndex={0} aria-label="Bảng kết quả cửa sổ có thể cuộn"><table><caption>Timestamp tương đối từ đầu phiên (ms). Chọn hàng để xem sáu điểm ở trên.</caption><thead><tr><th>Đoạn / cửa sổ</th><th>Bắt đầu–kết thúc</th><th>Dự đoán / chi tiết</th></tr></thead><tbody>{results.slice(resultPage*20,(resultPage+1)*20).map(r=><tr key={r.windowIndex} className={r.windowIndex===prediction.windowIndex?'selected-row':''}><td>{r.segmentId} / {r.windowIndex}</td><td>{r.startMs}–{r.endMs}</td><td><button className="text-button" aria-pressed={prediction===r} onClick={()=>setPrediction(r)}>{r.predictedLabel}</button></td></tr>)}</tbody></table></div>
            {results.length>20 && <div className="pagination"><button className="secondary-button" disabled={!resultPage} onClick={()=>setResultPage(p=>p-1)}>Trang trước</button><span>Trang {resultPage+1}/{Math.ceil(results.length/20)}</span><button className="secondary-button" disabled={(resultPage+1)*20>=results.length} onClick={()=>setResultPage(p=>p+1)}>Trang sau</button></div>}
            <div className="result-footer"><div><strong>Kết quả sẵn sàng để tải</strong><p>{results.length} cửa sổ · đủ sáu điểm và timestamp<br/>{prediction.version} · contract v1</p></div><button className="primary-button" onClick={exportResult}>Tải CSV kết quả ↓</button></div>
          </section>}
          <div className="research-note"><span aria-hidden="true">ⓘ</span><p>Demo nghiên cứu · test macro-F1 87,05%, chưa đạt 90%. Có thể nhầm ngồi–đứng. Không dùng cho y tế hoặc giám sát an toàn.</p></div>
        </section>
        {source && <section className="signal-panel" aria-labelledby="signal-title"><div className="panel-heading"><div><p className="section-label">KIỂM TRA TÍN HIỆU NGUỒN</p><h2 id="signal-title">{source.name}</h2></div><span className="source-badge">{source.demo?'UCI HAR · mẫu tham chiếu':'CSV cục bộ · không gửi lên server'}</span></div><div className="preview-layout"><div><div className="data-stats"><div><span>SỐ ĐIỂM</span><strong>{source.preview.sourceRows.toLocaleString('vi-VN')}</strong></div><div><span>KHOẢNG THỜI GIAN</span><strong>{(source.preview.durationMs/1000).toFixed(2)} s</strong></div><div><span>TẦN SỐ ƯỚC LƯỢNG</span><strong>{source.preview.estimatedHz?.toFixed(1)??'—'} Hz</strong></div></div><SignalChart points={source.preview.points} unit={unit} sourceRows={source.preview.sourceRows} gaps={source.preview.gapBounds}/></div><div>{source.demo?<p className="provenance">Nguồn: UCI HAR · split gốc {source.demo.original_split} / split chuẩn bị {source.demo.prepared_split} · subject {source.demo.subject_id} · dòng nguồn {source.demo.source_line_one_based}. Nhãn tham chiếu: <strong>{names[source.demo.reference_label_id]}</strong>. Cửa sổ độc lập; timestamp chỉ để hiển thị. Mẫu chọn trước, không dựa vào dự đoán đúng.</p>:<div className="csv-preview" data-testid="csv-preview"><div className="preview-note"><strong>Kiểm tra định dạng thành công.</strong><p>{source.preview.gapCount?`${source.preview.gapCount} khoảng >100 ms: tách đoạn, không nội suy qua mất mẫu.`:'Không phát hiện khoảng mất dữ liệu >100 ms.'} {source.preview.sourceRows<128&&'Dưới 128 điểm nguồn; cần kiểm tra resample để xác định có đủ cửa sổ không.'} {source.preview.extraColumns.length>0&&`Bỏ qua cột thêm: ${source.preview.extraColumns.join(', ')}.`}</p></div><div className="table-scroll" tabIndex={0} aria-label="Preview CSV có thể cuộn"><table><caption>5 dòng đầu — giá trị gốc, chưa đổi đơn vị</caption><thead><tr><th>timestamp_ms</th><th>acc_x</th><th>acc_y</th><th>acc_z</th></tr></thead><tbody>{source.preview.firstRows.map(p=><tr key={p.timestampMs}><td>{p.timestampMs}</td>{p.acc.map((v,i)=><td key={i}>{v.toFixed(5)}</td>)}</tr>)}</tbody></table></div></div>}</div></div></section>}
      </section>
      <section id="guide" className="info-section"><div className="panel-heading"><div><p className="section-label">HƯỚNG DẪN NHANH</p><h2>Đầu vào đúng. Đầu ra dễ hiểu.</h2></div><button className="secondary-button" onClick={downloadExample}>Tải CSV ví dụ</button></div><div className="info-grid"><article><span className="guide-number">01</span><h3>Chuẩn bị dữ liệu</h3><p>Bốn cột bắt buộc; thứ tự có thể thay đổi. Timestamp mili giây tăng nghiêm ngặt, dấu phẩy phân cách, dấu chấm thập phân.</p><pre>timestamp_ms,acc_x,acc_y,acc_z{'\n'}0,0.12,0.98,0.03{'\n'}20,0.15,1.01,0.02</pre></article><article><span className="guide-number">02</span><h3>Kiểm tra trước khi chạy</h3><p>Chọn g hoặc m/s² và xác nhận trọng lực. Nguồn 40–100 Hz được nội suy về 50 Hz. Mất mẫu &gt;100 ms tách đoạn; cửa sổ 128 điểm, stride 64, không pad đuôi.</p><p>Đọc thống kê rồi xác nhận dữ liệu resample. Mẫu UCI giữ nguyên.</p></article><article><span className="guide-number">03</span><h3>Nhận và sử dụng kết quả</h3><p>Chọn cửa sổ để xem nhãn và sáu điểm. Tải CSV chứa kết quả toàn file, timestamp, đơn vị nguồn và phiên bản model.</p><p>Đây là dự đoán, không phải nhãn thực tế. Không tự tạo lớp “khác” cho dữ liệu ngoài sáu lớp.</p></article></div></section>
      <section id="model-card" className="model-section"><div><p className="section-label">VỀ MÔ HÌNH</p><h2>Swin Transformer 1D</h2><p>Train trên Kaggle, chạy bằng ONNX/WASM một thread trên trình duyệt. Chỉ dùng total acceleration XYZ; không cần backend hoặc GPU của người dùng.</p><a className="text-link" href="?phase=4">Mở kiểm chứng kỹ thuật ↗</a></div><dl><div><dt>Dataset / input</dt><dd><a href="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones" target="_blank" rel="noreferrer">UCI HAR ↗</a> · 3 trục · 50 Hz · 128 điểm</dd></div><div><dt>Split theo subject</dt><dd>5392 train / 1960 val / 2947 test</dd></div><div><dt>Test accuracy / macro-F1</dt><dd>87,17% / 87,05% · chưa đạt F1 90%</dd></div><div><dt>Release / giới hạn</dt><dd>{RELEASE} · nhầm ngồi–đứng, ngoài UCI chưa bảo đảm</dd></div></dl></section>
    </main><footer className="site-footer"><span className="footer-brand">MotionSense <small>BY DUST</small></span><span>Không backend suy luận · không lưu CSV lên server · chưa deploy Vercel</span><a href="#">Về đầu trang ↑</a></footer>
  </div>;
}
