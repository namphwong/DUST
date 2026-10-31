import {readFile,writeFile,mkdir,rename} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {resolve,dirname} from 'node:path';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const dir=resolve(root,'public/models/drawai-paint35-swin-v1');
const name='drawai_swin1d-image_paint35-main-v1_rows64_fp32_v1.onnx';
const sha='9bb8bf5478cb2d8131936a79888505dcbc4f05e331027ca4e2117db7715df59d';
const valid=b=>b.length===2391351&&createHash('sha256').update(b).digest('hex')===sha;
if(process.env.VITE_DRAWAI_PUBLIC!=='true')throw new Error('Set VITE_DRAWAI_PUBLIC=true for a public build');
const api=new URL(process.env.VITE_API_BASE_URL??'');
if(api.protocol!=='https:'||api.username||api.password||api.pathname!=='/'||api.search||api.hash||/^(localhost|127\.|0\.|\[::1\])/.test(api.hostname))throw new Error('VITE_API_BASE_URL must be a public HTTPS origin');
let bytes;try{bytes=await readFile(resolve(dir,name));}catch(e){if(e.code!=='ENOENT')throw e;}
if(bytes&&!valid(bytes))throw new Error('Existing Swin bytes/hash mismatch; remove corrupt asset and retry');
if(!bytes){
  const url=`https://github.com/namphwong/DUST/releases/download/drawai-models-paint35-v1/${name}`;
  const response=await fetch(url,{signal:AbortSignal.timeout(90000)});
  if(!response.ok)throw new Error(`Swin download failed: ${response.status}`);
  bytes=Buffer.from(await response.arrayBuffer());if(!valid(bytes))throw new Error('Downloaded Swin hash/bytes mismatch');
  await mkdir(dir,{recursive:true});await writeFile(resolve(dir,name+'.partial'),bytes);await rename(resolve(dir,name+'.partial'),resolve(dir,name));
}
console.log('Verified public Swin asset:',name,sha);
