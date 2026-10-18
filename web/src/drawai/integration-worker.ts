import { loadRelease } from './release-loader';
import { inferDrawing } from './infer-drawing';
import { InferenceClient } from './inference-client';

self.onmessage=async()=>{
  let session:Awaited<ReturnType<typeof loadRelease>>|undefined;
  try{
    const fixtures=await(await fetch(`${import.meta.env.BASE_URL}verification/drawai/d6-fixtures.json`)).json();
    const stages:{stage:string;at:number}[]=[];const start=performance.now();
    session=await loadRelease(stage=>stages.push({stage,at:performance.now()-start}));
    const loaded=performance.now()-start,cases=[];
    // A real product client/worker receives the same JSON as the instrumented shared path.
    const client=new InferenceClient();
    try {
      for(let i=0;i<fixtures.cases.length;i++){
        const fixture=fixtures.cases[i];const result=await inferDrawing(session,fixture.drawing);
        const product=await client.predict(fixture.drawing,i,()=>{});
        cases.push({name:fixture.name,input:Array.from(result.input),logits:result.logits,product,
          tensorError:Math.max(...result.input.map((v,j)=>Math.abs(v-fixture.input[j]))),
          passed:result.input.every((v,j)=>Math.abs(v-fixture.input[j])<=2e-7)&&[result.logits,product.logits].every(logits=>logits.every((v,j)=>Math.abs(v-fixture.pytorch_logits[j])<=fixtures.atol+fixtures.rtol*Math.abs(fixture.pytorch_logits[j])))&&
            product.predicted_id===fixture.pytorch_logits.indexOf(Math.max(...fixture.pytorch_logits))});
      }
    }finally{client.reset();}
    const invalid=[];
    for(const f of fixtures.invalid){try{await inferDrawing(session,f.drawing);invalid.push({name:f.name,passed:false});}catch(e){invalid.push({name:f.name,passed:e instanceof Error&&e.message.startsWith(f.code+':')});}}
    const drawing=fixtures.cases.find((f:any)=>f.name==='train-digit-0').drawing;
    for(let i=0;i<5;i++)await inferDrawing(session,drawing);
    const timings=[];for(let i=0;i<100;i++){const r=await inferDrawing(session,drawing);timings.push({preprocessing:r.preprocessing,inference:r.inference});}
    const sorted=timings.map(t=>t.inference).sort((a,b)=>a-b);
    self.postMessage({type:'result',report:{passed:cases.every(c=>c.passed)&&invalid.every(c=>c.passed),cases,invalid,
      fixtureReleaseHash:fixtures.release_sha256,modelSha256:fixtures.model_sha256,loadAndInitMs:loaded,stages,timings,
      medianMs:(sorted[49]+sorted[50])/2,p95Ms:sorted[94],benchmark:{warmup:5,iterations:100,input:'train-digit-0',threads:1,percentile:'nearest rank'}}});
  }catch(e){self.postMessage({type:'error',message:e instanceof Error?e.message:String(e)});}
  finally{await session?.release();}
};
