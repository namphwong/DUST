import * as ort from 'onnxruntime-web/wasm';
import { preprocess } from './preprocessing';
import { validateDrawing, type Drawing } from './types';

/** Shared inference path for product and D6 instrumentation. */
export async function inferDrawing(session:ort.InferenceSession,drawing:Drawing,onInference?:()=>void) {
  validateDrawing(drawing);const start=performance.now();
  const input=preprocess(drawing.strokes,128);const preprocessing=performance.now()-start;
  onInference?.();const inferenceStart=performance.now();
  const output=await session.run({drawing:new ort.Tensor('float32',input,[1,3,128])});
  const inference=performance.now()-inferenceStart;
  const logits=Array.from(output.logits.data as Float32Array);
  if(JSON.stringify(output.logits.dims)!=='[1,10]'||logits.length!==10||!logits.every(Number.isFinite))throw new Error('INFERENCE: Output không hợp lệ.');
  const exp=logits.map(v=>Math.exp(v-Math.max(...logits))),sum=exp.reduce((a,b)=>a+b,0);
  return {input,logits,scores:exp.map(v=>v/sum),predicted_id:logits.indexOf(Math.max(...logits)),preprocessing,inference};
}
