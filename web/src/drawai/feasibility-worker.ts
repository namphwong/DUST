import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import wasmModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import { preprocess, GEOMETRY_VERSION, type Stroke } from './preprocessing';

type Fixture = { name: string; input: number[]; expected: number[]; strokes?: Stroke[]; y_axis?: string };
type Manifest = { version: string; trained: boolean; files: Record<string,{sha256:string;bytes:number}> };

async function sha256(bytes: ArrayBuffer) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
}

async function run(length: number, trained = false) {
  if (![64,128].includes(length)) throw new Error('Chiều dài không hợp lệ.');
  if (trained && length !== 128) throw new Error('Release trained yêu cầu L128.');
  const progress = (message: string) => self.postMessage({type:'progress',message});
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = {wasm:wasmUrl,mjs:wasmModuleUrl};
  const version = trained ? 'drawai-swin-v1' : `drawai-feasibility-l${length}`;
  const base = `${import.meta.env.BASE_URL}models/${version}/`;
  progress('Đang tải và kiểm tra SHA-256 của artifact kỹ thuật…');
  const before = performance.now();
  const manifestResponse = await fetch(base+'release.json');
  if (!manifestResponse.ok) throw new Error('Thiếu release DrawAI. Hãy chạy verify_feasibility.');
  const manifest = await manifestResponse.json() as Manifest;
  if (manifest.version !== version || manifest.trained !== trained) throw new Error('Sai version hoặc trạng thái artifact kỹ thuật/release.');
  const names = ['model.onnx','fixtures.json','preprocessing.json','report.json',...(trained ? ['labels.json'] : [])];
  const buffers = await Promise.all(names.map(async name=> {
    const response = await fetch(base+name);
    if (!response.ok) throw new Error(`Thiếu artifact DrawAI: ${name}. Có thể thử lại.`);
    const bytes = await response.arrayBuffer();
    const expected = manifest.files[name];
    if (!expected || bytes.byteLength !== expected.bytes || await sha256(bytes) !== expected.sha256) throw new Error(`Sai hash/bytes: ${name}.`);
    return bytes;
  }));
  const loadMs = performance.now()-before;
  const decode = (i: number)=>JSON.parse(new TextDecoder().decode(buffers[i]));
  const fixtures = decode(1) as {atol:number;rtol:number;cases:Fixture[]};
  const prep = decode(2), report = decode(3);
  if (trained && (JSON.stringify(decode(4)) !== JSON.stringify({classes:Array.from({length:10},(_,id)=>({id,name:String(id)}))}) ||
      report.version !== version || report.run_id !== 'official-swin-l128-aug1-seed44' ||
      report.checkpoint_sha256 !== 'a5ccb65dd4699b6451bf03ff23b6071382c29443213263f1fbc9ef6089fcf324' ||
      report.argmax_disagreements !== 0 || report.test_samples !== 3498 ||
      !Number.isFinite(report.macro_f1_degradation) || report.macro_f1_degradation > .001)) {
    throw new Error('Sai nhãn hoặc provenance/parity release DrawAI.');
  }
  if (prep.version !== GEOMETRY_VERSION || prep.length !== length || prep.layout !== 'NCT' ||
      prep.dtype !== 'float32' || prep.y_axis !== 'down' || prep.padding !== false ||
      prep.learned_normalization !== false || prep.timestamps_used !== false ||
      prep.max_strokes !== 16 || prep.max_points !== 8192 ||
      JSON.stringify(prep.coordinate_range) !== '[-0.5,0.5]' ||
      prep.normalization !== 'bbox-center/max-span; preserve aspect ratio' ||
      prep.resampling !== 'per-stroke arc length; minimum endpoints; largest remainder allocation' ||
      JSON.stringify(prep.channels) !== '["x","y","stroke_start"]' ||
      report.status !== 'passed' || report.trained !== trained || report.opset !== 17 ||
      report.input_name !== 'drawing' || report.output_name !== 'logits' ||
      report.model_bytes !== buffers[0].byteLength || report.sha256 !== manifest.files['model.onnx'].sha256 ||
      JSON.stringify(report.input_shape) !== JSON.stringify([1,3,length]) ||
      JSON.stringify(report.output_shape) !== '[1,10]' || report.fixture_count !== fixtures.cases.length ||
      fixtures.atol !== 1e-5 || fixtures.rtol !== 1e-4 || fixtures.cases.length !== 20) throw new Error('Metadata/fixture không đúng hợp đồng DrawAI D2.');
  progress('Đang khởi tạo ONNX Runtime WASM trong Web Worker…');
  const initStart = performance.now();
  const session = await ort.InferenceSession.create(buffers[0],{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  const initMs = performance.now()-initStart;
  try {
    if (JSON.stringify(session.inputNames) !== '["drawing"]' || JSON.stringify(session.outputNames) !== '["logits"]') {
      throw new Error('Sai input/output names của model.');
    }
    const cases = [];
    for (const fixture of fixtures.cases) {
      progress(`Đối chiếu ${fixture.name}…`);
      if (!Array.isArray(fixture.input) || fixture.input.length !== 3*length ||
          !Array.isArray(fixture.expected) || fixture.expected.length !== 10 ||
          fixture.input.some(v=>!Number.isFinite(v)) || fixture.expected.some(v=>!Number.isFinite(v))) {
        throw new Error('Fixture sai shape hoặc có giá trị không hữu hạn.');
      }
      let input = Float32Array.from(fixture.input), preprocessingError = 0;
      if (fixture.strokes) {
        input = preprocess(fixture.strokes,length,fixture.y_axis);
        preprocessingError = Math.max(...input.map((v,i)=>Math.abs(v-fixture.input[i])));
      }
      const output = await session.run({drawing:new ort.Tensor('float32',input,[1,3,length])});
      const logits = Array.from(output.logits.data as Float32Array);
      const errors = logits.map((v,i)=>Math.abs(v-fixture.expected[i]));
      cases.push({name:fixture.name,preprocessingError,maxError:Math.max(...errors),logits,
                  passed:preprocessingError <= 2e-7 && JSON.stringify(output.logits.dims) === '[1,10]' &&
                         logits.length === 10 && logits.every((v,i)=>Number.isFinite(v) && errors[i]<=fixtures.atol+fixtures.rtol*Math.abs(fixture.expected[i]))});
    }
    const invalidInputs: {name:string;strokes:Stroke[];code:string;length?:number;y_axis?:string}[] = [
      {name:'blank',strokes:[],code:'EMPTY'}, {name:'empty-stroke',strokes:[[]],code:'EMPTY_STROKE'},
      {name:'dot',strokes:[[{x:0,y:0}]],code:'TOO_SHORT'},
      {name:'isolated-dots',strokes:[[{x:0,y:0}],[{x:1,y:1}]],code:'TOO_SHORT'},
      {name:'nan',strokes:[[{x:0,y:0},{x:NaN,y:1}]],code:'NUMBER'},
      {name:'too-many-strokes',strokes:Array.from({length:17},()=>[{x:0,y:0},{x:1,y:1}]),code:'STROKES'},
      {name:'too-many-points',strokes:[Array.from({length:8193},(_,x)=>({x,y:0}))],code:'POINTS'},
      {name:'infinity',strokes:[[{x:0,y:0},{x:Infinity,y:1}]],code:'NUMBER'},
      {name:'string-coordinate',strokes:[[{x:'1',y:0}]] as unknown as Stroke[],code:'NUMBER'},
      {name:'bool-coordinate',strokes:[[{x:true,y:0}]] as unknown as Stroke[],code:'NUMBER'},
      {name:'out-of-range',strokes:[[{x:0,y:0},{x:1e7,y:1}]],code:'NUMBER'},
      {name:'total-points',strokes:[Array.from({length:4097},(_,x)=>({x,y:0})),Array.from({length:4096},(_,x)=>({x,y:1}))],code:'POINTS'},
      {name:'bad-length',strokes:[[{x:0,y:0},{x:1,y:1}]],length:63,code:'LENGTH'},
      {name:'bad-axis',strokes:[[{x:0,y:0},{x:1,y:1}]],y_axis:'unknown',code:'AXIS'},
    ];
    const invalidCases = invalidInputs.map(f=> {
      try { preprocess(f.strokes,f.length??length,f.y_axis); return {name:f.name,passed:false}; }
      catch (error) { return {name:f.name,passed:error instanceof Error && error.message.startsWith(f.code+':')}; }
    });
    progress('Đang đo 100 lần suy luận sau 5 warm-up…');
    const benchmarkFixture = fixtures.cases.find(f=>f.name==='random');
    if (!benchmarkFixture) throw new Error('Thiếu fixture benchmark.');
    const tensor = new ort.Tensor('float32',Float32Array.from(benchmarkFixture.input),[1,3,length]);
    for (let i=0;i<5;i++) await session.run({drawing:tensor});
    const timings = [];
    for (let i=0;i<100;i++) {
      const start = performance.now();
      await session.run({drawing:tensor});
      timings.push(performance.now()-start);
    }
    timings.sort((a,b)=>a-b);
    return {passed:cases.every(c=>c.passed)&&invalidCases.every(c=>c.passed),trained,length,version,
            cases,invalidCases,loadMs,initMs,medianMs:(timings[49]+timings[50])/2,p95Ms:timings[94],
            modelBytes:buffers[0].byteLength,backend:'WASM worker / CPU 1 thread',
            benchmark:{warmup:5,iterations:100,input:'random',percentileMethod:'nearest rank'},
            modelSha256:manifest.files['model.onnx'].sha256,preprocessingVersion:prep.version,
            targetsMet:buffers[0].byteLength<=10*1024*1024 && timings[94]<=100};
  } finally { await session.release(); }
}

self.onmessage = async (event: MessageEvent<{length:number;trained?:boolean}>) => {
  try { self.postMessage({type:'result',result:await run(event.data.length,event.data.trained)}); }
  catch (error) { self.postMessage({type:'error',message:error instanceof Error ? error.message : String(error)}); }
};
