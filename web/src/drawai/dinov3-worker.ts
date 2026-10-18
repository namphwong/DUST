import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import wasmModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import {renderRgb,normalizeRgb,RASTER_VERSION} from './raster';

const base='/__drawai_p2__/';
const hash=async(value:ArrayBuffer)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',value)),b=>b.toString(16).padStart(2,'0')).join('');
const p95=(values:number[])=>[...values].sort((a,b)=>a-b)[Math.ceil(.95*values.length)-1];

self.onmessage=async()=>{
  let session:ort.InferenceSession|undefined;
  try{
    const coldStart=performance.now();
    const get=async(name:string)=>{const r=await fetch(base+name,{cache:'no-store'});if(!r.ok)throw new Error(`FETCH: ${name}`);return r.arrayBuffer();};
    const manifest=JSON.parse(new TextDecoder().decode(await get('manifest.json')));
    const expectedName=manifest.model_filename;
    const pretrainedName='drawai_dinov3-vits16-timm-qkvb_pretrained-backbone_rgb224_fp32_p2.onnx';
    if(manifest.smoke_only!==true||manifest.trained!==false||manifest.release_ready!==false||
      !['drawai_dinov3-vits16_architecture-smoke_rgb224_fp32_v0.onnx',pretrainedName].includes(expectedName))
      throw new Error('CONTRACT: not the explicit P2 technical artifact');
    const files:Record<string,ArrayBuffer>={};
    for(const name of [expectedName,'fixtures.json','preprocessing.json','python-report.json']){
      self.postMessage({type:'progress',message:`Đang tải và kiểm tra ${name}…`});
      const value=await get(name),meta=manifest.files[name];
      if(!meta||value.byteLength!==meta.bytes||await hash(value)!==meta.sha256)throw new Error(`HASH: ${name}`);
      files[name]=value;
    }
    const parse=(name:string)=>JSON.parse(new TextDecoder().decode(files[name]));
    const preprocessing=parse('preprocessing.json'),python=parse('python-report.json');
    if(preprocessing.version!==RASTER_VERSION||preprocessing.size!==224||preprocessing.padding_pixels!==24||preprocessing.stroke_width_pixels!==8||
       JSON.stringify(preprocessing.image_mean)!=='[0.485,0.456,0.406]'||JSON.stringify(preprocessing.image_std)!=='[0.229,0.224,0.225]'||
       JSON.stringify(preprocessing.channels)!=='["R","G","B"]'||python.full_backbone_exported!==true||python.smoke_only!==true||python.trained!==false||
       python.parameters!==(expectedName===pretrainedName?21603850:21604234)||JSON.stringify(python.input_shape)!=='[1,3,224,224]'||JSON.stringify(python.output_shape)!=='[1,10]')
      throw new Error('CONTRACT: preprocessing/full backbone differs');
    if(expectedName===pretrainedName&&(python.pretrained_loaded!==true||manifest.pretrained_loaded!==true||
       python.provenance.model_id!=='timm/vit_small_patch16_dinov3_qkvb.lvd1689m'||
       python.provenance.revision!=='2c7705788ac282557562465d6443606664a55f05'||
       python.provenance.weights_sha256!=='71aedea5ea914657b8fd641beadbd97c012f87e3116d4d081eb143aa520ab53c'||
       python.provenance.native_state_all_tensors_equal!==true))throw new Error('CONTRACT: unverified pretrained provenance');
    const fetchVerifyMs=performance.now()-coldStart;
    ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths={wasm:wasmUrl,mjs:wasmModuleUrl};
    self.postMessage({type:'progress',message:'Đang khởi tạo full DINOv3 ONNX…'});
    const initStart=performance.now();
    session=await ort.InferenceSession.create(files[expectedName],{executionProviders:['wasm'],graphOptimizationLevel:'all'});
    if(JSON.stringify(session.inputNames)!=='["pixel_values"]'||JSON.stringify(session.outputNames)!=='["logits"]')throw new Error('CONTRACT: ONNX IO');
    const initMs=performance.now()-initStart,coldMs=performance.now()-coldStart;
    const fixtures=parse('fixtures.json'),cases:any[]=[];
    let first:Float32Array|undefined;
    for(const fixture of fixtures.cases){
      const rgb=renderRgb(fixture.strokes,fixture.y_axis),values=normalizeRgb(rgb);first??=values;
      const rgbHash=await hash(rgb.buffer as ArrayBuffer),tensorHash=await hash(values.buffer as ArrayBuffer);
      if(rgbHash!==fixture.rgb_sha256||tensorHash!==fixture.tensor_sha256)throw new Error(`RASTER_PARITY: ${fixture.name}`);
      const outputs=await session.run({pixel_values:new ort.Tensor('float32',values,[1,3,224,224])});
      const logits=Array.from(outputs.logits.data as Float32Array),expected=fixture.expected_logits;
      if(logits.length!==10||logits.some((n,i)=>!Number.isFinite(n)||Math.abs(n-expected[i])>fixtures.atol+fixtures.rtol*Math.abs(expected[i])))
        throw new Error(`LOGITS_PARITY: ${fixture.name}`);
      const predicted=logits.indexOf(Math.max(...logits)),reference=expected.indexOf(Math.max(...expected));
      if(predicted!==reference)throw new Error(`ARGMAX_PARITY: ${fixture.name}`);
      cases.push({name:fixture.name,rgbHash,tensorHash,maxAbsError:Math.max(...logits.map((n,i)=>Math.abs(n-expected[i]))),predicted,expectedArgmax:reference});
    }
    const invalid:any[]=[];
    for(const fixture of fixtures.invalid){
      let code='NONE';try{renderRgb(fixture.strokes);}catch(e){code=String((e as Error).message).split(':')[0];}
      if(code!==fixture.code)throw new Error(`INVALID_PARITY: ${fixture.name}`);invalid.push({name:fixture.name,code});
    }
    const feed={pixel_values:new ort.Tensor('float32',first!,[1,3,224,224])};
    for(let i=0;i<5;i++)await session.run(feed);
    const times:number[]=[];
    for(let i=0;i<30;i++){const t=performance.now();await session.run(feed);times.push(performance.now()-t);}
    const median=[...times].sort((a,b)=>a-b)[14],latencyP95=p95(times);
    const report={passed:files[expectedName].byteLength<=157286400&&coldMs<=15000&&latencyP95<=500,
      technicalOnly:true,pretrainedLoaded:manifest.pretrained_loaded,releaseReady:false,trained:false,
      backend:'WASM CPU 1 thread',modelBytes:files[expectedName].byteLength,modelSha256:manifest.files[expectedName].sha256,
      fetchVerifyMs,initMs,coldMs,medianMs:median,p95Ms:latencyP95,warmup:5,iterations:30,cases,invalid,
      parityPassed:true,memoryBudgetAssessedByExternalRunner:true,timingSamplesMs:times};
    self.postMessage({type:'result',report});
  }catch(error){self.postMessage({type:'error',message:error instanceof Error?error.message:String(error)});}
  finally{if(session)await session.release();}
};
