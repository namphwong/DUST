import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import wasmModuleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import type { Result, CaseResult } from './inference-smoke-test';

export const RELEASE = 'har-swin-kaggle-v1';
type Fixture = { name: string; kind: string; raw: number[]; input: number[]; expected: number[]; reference_label?: number };
type Fixtures = { version: string; atol: number; rtol: number; cases: Fixture[] };
type Labels = { classes: { id: number; code: string; name_vi: string }[] };
type Preprocessing = { channels: string[]; mean: number[]; std: number[]; sample_rate_hz: number; window_size: number; input_layout: string; input_unit: string; includes_gravity: boolean };
export type ActivityResult = { name: string; reference: string; predicted: string; scores: number[] };
export type ArtifactResult = Result & { activities: ActivityResult[]; version: string; normalizationMaxError: number; labelCodes: string[] };

export function normalizeWindow(raw: number[], prep: Preprocessing): Float32Array {
  if (raw.length !== 384 || raw.some(value => !Number.isFinite(value))) throw new Error('INVALID_NUMBER: Input phải có 384 số hữu hạn, layout channel-major.');
  const normalized = Float32Array.from(raw, (value, index) => {
    const channel = Math.floor(index / 128);
    return Math.fround(Math.fround(Math.fround(value) - Math.fround(prep.mean[channel])) / Math.fround(Math.max(prep.std[channel], 1e-8)));
  });
  if (normalized.some(v => !Number.isFinite(v))) throw new Error('INVALID_NUMBER: Input vượt miền float32 sau chuẩn hóa.');
  return normalized;
}

function softmax(logits: number[]): number[] {
  const maximum = Math.max(...logits);
  const exponents = logits.map(value => Math.exp(value - maximum));
  const sum = exponents.reduce((a, b) => a + b, 0);
  return exponents.map(value => value / sum);
}

async function sha256(bytes: ArrayBuffer): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), value => value.toString(16).padStart(2, '0')).join('');
}

export async function loadArtifact(progress: (message: string) => void) {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: wasmModuleUrl };
  const base = `${import.meta.env.BASE_URL}models/${RELEASE}/`;
  progress('Đang tải release đã train và kiểm tra SHA-256…');
  const started = performance.now();
  const manifestResponse = await fetch(base + 'release.json');
  if (!manifestResponse.ok) throw new Error('Thiếu manifest Phase 4. Hãy export artifact trước.');
  const manifest = await manifestResponse.json() as { version: string; trained: boolean; files: Record<string, { sha256: string; bytes: number }> };
  if (manifest.version !== RELEASE || manifest.trained !== true) throw new Error('Release không đúng version/trained=true.');
  const names = ['model.onnx', 'fixtures.json', 'preprocessing.json', 'labels.json', 'report.json'];
  const buffers = await Promise.all(names.map(async name => {
    const response = await fetch(base + name);
    if (!response.ok) throw new Error(`Thiếu artifact Phase 4: ${name}. Có thể thử lại.`);
    const bytes = await response.arrayBuffer();
    const expected = manifest.files[name];
    if (!expected || bytes.byteLength !== expected.bytes || await sha256(bytes) !== expected.sha256) throw new Error(`Artifact không đồng bộ hoặc sai hash: ${name}.`);
    return bytes;
  }));
  const [bytes] = buffers;
  const decode = (index: number) => JSON.parse(new TextDecoder().decode(buffers[index]));
  const fixtures = decode(1) as Fixtures;
  const prep = decode(2) as Preprocessing;
  const labels = decode(3) as Labels;
  const report = decode(4) as Record<string, unknown>;
  const codes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'];
  if (report.version !== RELEASE || report.trained !== true || report.status !== 'passed' || report.sha256 !== manifest.files['model.onnx'].sha256 || report.model_bytes !== bytes.byteLength ||
      JSON.stringify(report.input_shape) !== '[1,3,128]' || JSON.stringify(report.output_shape) !== '[1,6]' || report.input_name !== 'signals' || report.output_name !== 'logits' || report.input_dtype !== 'float32' || report.input_is_normalized !== true || report.quantized !== false) throw new Error('Metadata model không đúng hợp đồng Phase 4.');
  if (prep.input_layout !== 'NCT' || prep.window_size !== 128 || prep.sample_rate_hz !== 50 || prep.input_unit !== 'g' || prep.includes_gravity !== true ||
      JSON.stringify(prep.channels) !== '["total_acc_x","total_acc_y","total_acc_z"]' || prep.mean.length !== 3 || prep.std.length !== 3 ||
      prep.mean.some(v => !Number.isFinite(v)) || prep.std.some(v => !Number.isFinite(v) || v <= 0)) throw new Error('Preprocessing không đúng dữ liệu acceleration train-only.');
  if (labels.classes.length !== 6 || labels.classes.some((c, i) => c.id !== i || c.code !== codes[i] || typeof c.name_vi !== 'string')) throw new Error('Nhãn không đúng thứ tự 0–5.');
  if (fixtures.version !== RELEASE || fixtures.cases.length !== 14 || report.fixture_count !== 14 || fixtures.atol !== 1e-5 || fixtures.rtol !== 1e-4) throw new Error('Fixture không đồng bộ với release.');
  const loadMs = performance.now() - started;
  return { bytes, fixtures, prep, labels, report, loadMs };
}

export type WindowPrediction = { predictedId: number; predictedLabel: string; predictedCode: string; scores: number[]; logits: number[]; version: string };

// One hash-verified session per worker; normalization happens once per window.
export async function createClassifier(progress: (message: string) => void) {
  const { bytes, prep, labels } = await loadArtifact(progress);
  progress('Đang khởi tạo model đã train…');
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  return {
    normalize: (raw: number[]) => normalizeWindow(raw, prep),
    async predict(raw: number[]): Promise<WindowPrediction> {
      const output = await session.run({ signals: new ort.Tensor('float32', normalizeWindow(raw, prep), [1,3,128]) });
      const logits = Array.from(output.logits.data as Float32Array);
      if (JSON.stringify(output.logits.dims) !== '[1,6]' || logits.length !== 6 || logits.some(v => !Number.isFinite(v))) throw new Error('Model trả đầu ra không hợp lệ.');
      const predictedId = logits.indexOf(Math.max(...logits));
      return { predictedId, predictedLabel: labels.classes[predictedId].name_vi, predictedCode: labels.classes[predictedId].code, scores: softmax(logits), logits, version: RELEASE };
    },
  };
}

export async function predictDemoWindow(raw: number[], progress: (message: string) => void): Promise<WindowPrediction> {
  const { bytes, prep, labels } = await loadArtifact(progress);
  const input = normalizeWindow(raw, prep);
  progress('Đang khởi tạo model đã train…');
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  try {
    progress('Đang phân tích cửa sổ UCI trên thiết bị…');
    const output = await session.run({ signals: new ort.Tensor('float32', input, [1, 3, 128]) });
    const logits = Array.from(output.logits.data as Float32Array);
    if (JSON.stringify(output.logits.dims) !== '[1,6]' || logits.length !== 6 || logits.some(v => !Number.isFinite(v))) throw new Error('Model trả đầu ra không hợp lệ.');
    const predictedId = logits.indexOf(Math.max(...logits));
    return { predictedId, predictedLabel: labels.classes[predictedId].name_vi, predictedCode: labels.classes[predictedId].code,
      scores: softmax(logits), logits, version: RELEASE };
  } finally { await session.release(); }
}

export async function runArtifactTest(progress: (message: string) => void): Promise<ArtifactResult> {
  const { bytes, fixtures, prep, labels, report, loadMs } = await loadArtifact(progress);
  progress('Đang khởi tạo model đã train · WASM CPU…');
  const initStart = performance.now();
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  const initMs = performance.now() - initStart;
  try {
    const cases: CaseResult[] = [];
    const activities: ActivityResult[] = [];
    let normalizationMaxError = 0;
    for (const fixture of fixtures.cases) {
      progress(`Đang kiểm tra preprocessing và logits: ${fixture.name}…`);
      if (fixture.input.length !== 384 || fixture.expected.length !== 6 || [...fixture.input, ...fixture.expected].some(v => !Number.isFinite(v))) throw new Error('Fixture sai shape hoặc không hữu hạn.');
      const input = normalizeWindow(fixture.raw, prep);
      const preprocessingError = Math.max(...Array.from(input, (v, i) => Math.abs(v - fixture.input[i])));
      normalizationMaxError = Math.max(normalizationMaxError, preprocessingError);
      if (preprocessingError > 1e-6) throw new Error('Chuẩn hóa browser không khớp Python; không chạy inference.');
      const output = await session.run({ signals: new ort.Tensor('float32', input, [1, 3, 128]) });
      const tensor = output.logits;
      const logits = Array.from(tensor.data as Float32Array);
      const errors = logits.map((value, i) => Math.abs(value - fixture.expected[i]));
      const passed = JSON.stringify(tensor.dims) === '[1,6]' && logits.length === 6 && logits.every((v, i) => Number.isFinite(v) && errors[i] <= fixtures.atol + fixtures.rtol * Math.abs(fixture.expected[i]));
      cases.push({ name: fixture.name, logits, maxError: Math.max(...errors), passed });
      if (fixture.kind === 'uci-reference') {
        if (!Number.isInteger(fixture.reference_label) || fixture.reference_label! < 0 || fixture.reference_label! > 5) throw new Error('Nhãn tham chiếu fixture không hợp lệ.');
        const predicted = logits.indexOf(Math.max(...logits));
        activities.push({ name: fixture.name, reference: labels.classes[fixture.reference_label!].name_vi, predicted: labels.classes[predicted].name_vi, scores: softmax(logits) });
      }
    }
    progress('Đang benchmark: 5 warm-up và 30 lần suy luận…');
    const input = new ort.Tensor('float32', normalizeWindow(fixtures.cases[8].raw, prep), [1, 3, 128]);
    for (let i = 0; i < 5; i++) await session.run({ signals: input });
    const timings: number[] = [];
    for (let i = 0; i < 30; i++) {
      const before = performance.now();
      await session.run({ signals: input });
      timings.push(performance.now() - before);
    }
    timings.sort((a, b) => a - b);
    return { passed: cases.every(c => c.passed), loadMs, initMs, medianMs: (timings[14] + timings[15]) / 2,
      modelBytes: bytes.byteLength, cases, report, activities, version: RELEASE, normalizationMaxError, labelCodes: labels.classes.map(c => c.code) };
  } finally { await session.release(); }
}
