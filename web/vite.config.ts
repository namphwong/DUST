import {defineConfig,type Plugin} from 'vite';
import {createReadStream,statSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {request as httpRequest} from 'node:http';

/** Local feasibility files are served only when explicitly enabled.
 * They are never copied into public/ or dist/ and are not production assets.
 */
function localPhase2():Plugin{
  const root=resolve(dirname(fileURLToPath(import.meta.url)),'../artifacts/drawai/phase2/technical');
  const allowed=new Set(['manifest.json','fixtures.json','preprocessing.json','python-report.json',
    'drawai_dinov3-vits16_architecture-smoke_rgb224_fp32_v0.onnx',
    'drawai_dinov3-vits16-timm-qkvb_pretrained-backbone_rgb224_fp32_p2.onnx']);
  const attach=(server:any)=>{
    server.middlewares.use('/__drawai_p2_backend__',(req:any,res:any)=>{
      if(process.env.DRAWAI_P2_LOCAL_SMOKE!=='1'){res.statusCode=404;res.end('P2 local backend disabled');return;}
      if(!((req.method==='GET'&&req.url==='/health')||(req.method==='POST'&&req.url==='/infer'))){res.statusCode=404;res.end('Not found');return;}
      if(req.method==='POST'&&Number(req.headers['content-length'])>1048576){
        req.resume();res.writeHead(413,{'Content-Type':'application/json'});res.end('{"error":"SIZE"}');return;
      }
      const upstream=httpRequest({hostname:'127.0.0.1',port:4190,path:req.url,method:req.method,
        headers:{'Content-Type':'application/json',...(req.headers['content-length']?{'Content-Length':req.headers['content-length']}:{})}},(response)=>{
          res.writeHead(response.statusCode??502,{'Content-Type':'application/json','Cache-Control':'no-store'});response.pipe(res);
        });
      upstream.setTimeout(10000,()=>upstream.destroy());
      upstream.on('error',()=>{if(!res.headersSent)res.writeHead(502,{'Content-Type':'application/json'});res.end('{"error":"Local backend unavailable"}');});
      req.pipe(upstream);
    });
    server.middlewares.use('/__drawai_p2__',(req:any,res:any)=>{
      if(process.env.DRAWAI_P2_LOCAL_SMOKE!=='1'){res.statusCode=404;res.end('P2 local smoke disabled');return;}
      const name=(req.url??'').split('?')[0].replace(/^\//,'');
      if(req.method!=='GET'||!allowed.has(name)){res.statusCode=404;res.end('Not found');return;}
      try{
        const path=resolve(root,name),stat=statSync(path);
        res.setHeader('Content-Type',name.endsWith('.json')?'application/json':'application/octet-stream');
        res.setHeader('Content-Length',stat.size);res.setHeader('Cache-Control','no-store');
        const stream=createReadStream(path);stream.on('error',()=>res.destroy());stream.pipe(res);
      }catch{res.statusCode=404;res.end('Run the P2 feasibility exporter locally first');}
    });
  };
  return {name:'drawai-local-p2-only',configureServer:attach,configurePreviewServer:attach};
}
export default defineConfig({plugins:[localPhase2()]});
