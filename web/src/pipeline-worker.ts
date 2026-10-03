import { MAX_BYTES, parseCsvPreview, type CsvPreview } from './csv-preview';
import { compactPreview, prepareCsv, summary, windows, type Unit, type Prepared, type CsvResult } from './csv-pipeline';
import { createClassifier } from './inference-artifact';
let preview: CsvPreview | undefined;
let prepared: Prepared | undefined;
let classifier: Awaited<ReturnType<typeof createClassifier>> | undefined;
let running = false;
// Requests are serialized by the UI; worker termination also cancels fetch/inference.
self.onmessage = async (event: MessageEvent<{ id: number; action: string; file?: File; unit?: Unit; gravity?: boolean; demo?: boolean }>) => {
  const { id, action, file, unit = 'g', gravity = false, demo = false } = event.data;
  const send = (type: string, data: unknown) => self.postMessage({ id, type, data });
  if (running) { send('error', 'Tác vụ trước chưa hoàn tất.'); return; }
  running = true;
  try {
    if (action === 'parse') {
      if (!file || file.size > MAX_BYTES) throw new Error('FILE_TOO_LARGE: File phải không quá 10 MiB.');
      preview = undefined; prepared = undefined;
      let text: string;
      try { text = new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer()); }
      catch { throw new Error('INVALID_CSV: Không đọc được UTF-8; hãy lưu CSV bằng UTF-8.'); }
      preview = parseCsvPreview(text);
      send('done', compactPreview(preview));
    } else if (action === 'prepare') {
      prepared = undefined;
      if (!preview) throw new Error('Chọn CSV trước.');
      prepared = prepareCsv(preview, unit, gravity, demo);
      send('done', summary(prepared));
    } else if (action === 'analyze' || action === 'inspect') {
      if (!prepared) throw new Error('Kiểm tra và xác nhận resample trước.');
      classifier ??= await createClassifier(message=>send('status',message));
      const results: CsvResult[] = [], inputs: number[][] = [];
      const start = performance.now();
      if (action === 'inspect' && prepared.windowCount > 32) throw new Error('Diagnostics giới hạn 32 cửa sổ.');
      for (const window of windows(prepared)) {
        if (action === 'inspect') inputs.push(Array.from(classifier.normalize(window.raw)));
        const { raw, ...metadata } = window;
        results.push({ ...metadata, ...await classifier.predict(raw) });
        if (results.length === 1 || results.length%16 === 0 || results.length === prepared.windowCount) send('progress',{ completed: results.length, total: prepared.windowCount });
      }
      send('done', { results, inputs, elapsedMs: performance.now()-start });
    } else throw new Error('Tác vụ không được hỗ trợ.');
  } catch (error) { send('error', error instanceof Error ? error.message : 'Không xử lý được dữ liệu.'); }
  finally { running = false; }
};
