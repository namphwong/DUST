import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import wasmModuleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';

export type CaseResult = { name: string; passed: boolean; maxError: number; logits: number[] };
export type Result = { passed: boolean; loadMs: number; medianMs: number; initMs: number; modelBytes: number; cases: CaseResult[]; report: Record<string, unknown> };
type Fixtures = { atol: number; rtol: number; cases: { name: string; input: number[]; expected: number[] }[] };
const base = `${import.meta.env.BASE_URL}models/phase1/`;

export async function runSmokeTest(progress: (message: string) => void): Promise<Result> {
  ort.env.wasm.numThreads = 1; // Works without cross-origin isolation headers.
  ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: wasmModuleUrl };
  progress('Đang tải mô hình và các mẫu tham chiếu…');
  const start = performance.now();
  const responses = await Promise.all(['model.onnx', 'fixtures.json', 'report.json'].map(name => fetch(base + name)));
  if (responses.some(response => !response.ok)) throw new Error('Thiếu artifact Phase 1. Hãy chạy training/export_smoke_test.py.');
  const [bytes, fixtures, report] = await Promise.all([
    responses[0].arrayBuffer(), responses[1].json() as Promise<Fixtures>, responses[2].json(),
  ]);
  const loadMs = performance.now() - start;
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const sha256 = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
  if (sha256 !== report.sha256 || report.trained !== false) throw new Error('Artifact Phase 1 không đồng bộ hoặc không phải mô hình thử untrained.');
  if (fixtures.cases.length !== 8) throw new Error('Phase 1 yêu cầu đúng 8 mẫu tham chiếu.');
  progress('Đang khởi tạo ONNX Runtime · WASM CPU…');
  const initStart = performance.now();
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  const initMs = performance.now() - initStart;
  try {
    const cases: CaseResult[] = [];
    for (const fixture of fixtures.cases) {
      progress(`Đang đối chiếu mẫu ${fixture.name}…`);
      if (fixture.input.length !== 384 || fixture.expected.length !== 6) throw new Error('Fixture không đúng shape [1,3,128] → [1,6].');
      const output = await session.run({ signals: new ort.Tensor('float32', Float32Array.from(fixture.input), [1, 3, 128]) });
      const tensor = output.logits;
      const logits = Array.from(tensor.data as Float32Array);
      const validShape = tensor.dims.length === 2 && tensor.dims[0] === 1 && tensor.dims[1] === 6;
      const errors = logits.map((value, index) => Math.abs(value - fixture.expected[index]));
      cases.push({ name: fixture.name, logits, maxError: Math.max(...errors),
        passed: validShape && logits.length === 6 && logits.every((value, index) => Number.isFinite(value) && errors[index] <= fixtures.atol + fixtures.rtol * Math.abs(fixture.expected[index])) });
    }
    progress('Đang đo tốc độ sau warm-up…');
    const input = new ort.Tensor('float32', Float32Array.from(fixtures.cases[2].input), [1, 3, 128]);
    for (let i = 0; i < 5; i++) await session.run({ signals: input });
    const timings: number[] = [];
    for (let i = 0; i < 30; i++) {
      const before = performance.now();
      await session.run({ signals: input });
      timings.push(performance.now() - before);
    }
    timings.sort((a, b) => a - b);
    return { passed: cases.every(item => item.passed), loadMs, initMs, medianMs: (timings[14] + timings[15]) / 2, modelBytes: bytes.byteLength, cases, report };
  } finally { await session.release(); }
}
