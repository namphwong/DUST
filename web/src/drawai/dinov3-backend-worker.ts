import {renderRgb,normalizeRgb} from './raster';
const base='/__drawai_p2_backend__/';
const hash=async(value:ArrayBuffer)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',value)),b=>b.toString(16).padStart(2,'0')).join('');
const p95=(values:number[])=>[...values].sort((a,b)=>a-b)[Math.ceil(.95*values.length)-1];
self.onmessage=async()=>{
  try{
    const get=async(path:string)=>{const response=await fetch(path,{cache:'no-store'});if(!response.ok)throw new Error(`FETCH: ${path}`);return response.json();};
    const manifest=await get('/__drawai_p2__/manifest.json'),fixtures=await get('/__drawai_p2__/fixtures.json');
    const initial=await get(base+'health');
    if(!initial.pretrained_loaded||initial.trained||initial.release_ready||initial.model_sha256!==manifest.files[manifest.model_filename].sha256)
      throw new Error('CONTRACT: backend/model provenance differs');
    const infer=async(fixture:any)=>{
      const response=await fetch(base+'infer',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({strokes:fixture.strokes,y_axis:fixture.y_axis,model_sha256:initial.model_sha256})});
      if(!response.ok)throw new Error(`INFER: ${response.status}`);return response.json();
    };
    const cases:any[]=[];
    for(const fixture of fixtures.cases){
      const value=await infer(fixture),rgb=renderRgb(fixture.strokes,fixture.y_axis),tensor=normalizeRgb(rgb);
      if(await hash(rgb.buffer as ArrayBuffer)!==value.rgb_sha256||await hash(tensor.buffer as ArrayBuffer)!==value.tensor_sha256||
        value.rgb_sha256!==fixture.rgb_sha256||value.tensor_sha256!==fixture.tensor_sha256)throw new Error('RASTER_PARITY: '+fixture.name);
      if(value.logits.length!==10||value.logits.some((n:number,i:number)=>!Number.isFinite(n)||Math.abs(n-fixture.expected_logits[i])>fixtures.atol+fixtures.rtol*Math.abs(fixture.expected_logits[i])))
        throw new Error('LOGITS_PARITY: '+fixture.name);
      const reference=fixture.expected_logits.indexOf(Math.max(...fixture.expected_logits));
      if(value.predicted_id!==reference)throw new Error('ARGMAX_PARITY: '+fixture.name);
      cases.push({name:fixture.name,rgbHash:value.rgb_sha256,tensorHash:value.tensor_sha256,maxAbsError:Math.max(...value.logits.map((n:number,i:number)=>Math.abs(n-fixture.expected_logits[i]))),predicted:value.predicted_id});
    }
    for(let i=0;i<5;i++)await infer(fixtures.cases[0]);
    const times:number[]=[],serverTimes:any[]=[];
    for(let i=0;i<30;i++){const start=performance.now(),value=await infer(fixtures.cases[0]);times.push(performance.now()-start);serverTimes.push({rasterMs:value.raster_ms,inferenceMs:value.inference_ms,totalMs:value.server_total_ms});}
    const health=await get(base+'health'),latency=p95(times);
    const report={passed:latency<=500&&health.cold_start_ms<=15000&&health.peak_sampled_rss_bytes<=1073741824,
      technicalOnly:true,pretrainedLoaded:true,trained:false,releaseReady:false,backend:'Local ONNX Runtime CPU 4 threads',
      modelSha256:health.model_sha256,p95EndToEndMs:latency,medianEndToEndMs:[...times].sort((a,b)=>a-b)[14],
      iterations:30,warmup:5,timingSamplesMs:times,serverTimes,coldStartMs:health.cold_start_ms,
      peakSampledRssBytes:health.peak_sampled_rss_bytes,memoryMethod:health.memory_method,cases,parityPassed:true,dataPersisted:health.data_persisted};
    self.postMessage({type:'result',report});
  }catch(error){self.postMessage({type:'error',message:error instanceof Error?error.message:String(error)});}
};
