"""Production ASGI API for the verified DINOv3 TorchScript artifact only."""
import base64
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from .native_runtime import configure_windows_dlls
configure_windows_dlls()
import numpy as np
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from .paint_package import verify_bundle,spec
from .paint_data import dino_rgb
from .raster import normalize_rgb

BUNDLE_HASH='86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2'
MODEL_ID='dinov3-vits16';ADAPTER='drawai-canvas400-gray64-v1';PROTOCOL='drawai-paint-api-v1'


def decode_request(payload):
    def invalid(_):raise ValueError('Nonfinite JSON')
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    value=json.loads(payload,parse_constant=invalid,object_pairs_hook=pairs)
    if not isinstance(value,dict) or set(value)!={'protocol','request_id','drawing_revision','model_id','canvas_adapter','pixels64_base64'}:raise ValueError('Invalid request fields')
    if value['protocol']!=PROTOCOL or value['model_id']!=MODEL_ID or value['canvas_adapter']!=ADAPTER:raise ValueError('Contract/model mismatch')
    if any(type(value[k]) is not int or not 0<=value[k]<=9007199254740991 for k in ('request_id','drawing_revision')):raise ValueError('Invalid request/revision ID')
    encoded=value['pixels64_base64']
    if not isinstance(encoded,str) or len(encoded)!=5464:raise ValueError('Expected gray64 pixels')
    pixels=base64.b64decode(encoded,validate=True)
    if len(pixels)!=4096 or min(pixels)==255:raise ValueError('Empty/invalid pixels')
    return value,np.frombuffer(pixels,dtype=np.uint8).reshape(64,64)


class DinoRuntime:
    def __init__(self,bundle):
        start=time.perf_counter();self.bundle=Path(bundle)
        self.record,_=verify_bundle(self.bundle,BUNDLE_HASH)
        if hasattr(os,'posix_fadvise'):
            for path in self.bundle.rglob('*'):
                if path.is_file() and path.suffix in ('.pt','.onnx'):
                    try:
                        with path.open('rb') as stream:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                    except OSError:pass
        # Import torch only after hash verification, never a training factory or ONNX Runtime.
        import torch
        self.torch=torch;torch.set_num_threads(1);torch.set_num_interop_threads(1)
        self.model=torch.jit.load(str(self.bundle/MODEL_ID/spec(MODEL_ID)['model_filename']),map_location='cpu').eval()
        self.start_ms=(time.perf_counter()-start)*1000

    def health(self):
        resources={}
        for name,file in [('memory_peak_bytes','memory.peak'),('memory_limit_bytes','memory.max')]:
            try:resources[name]=int((Path('/sys/fs/cgroup')/file).read_text().strip())
            except (OSError,ValueError):resources[name]=None
        try:resources['memory_events']={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())}
        except (OSError,ValueError):resources['memory_events']=None
        return dict(status='ready',models=[MODEL_ID],default_model=MODEL_ID,artifact_format='torchscript',threads=1,
            trained=True,artifact_verified=True,release_ready=False,technical_only=True,data_persisted=False,
            bundle_sha256=BUNDLE_HASH,model_sha256=self.record['files'][MODEL_ID+'/'+spec(MODEL_ID)['model_filename']]['sha256'],
            protocol=PROTOCOL,canvas_adapter=ADAPTER,preprocessing_version='drawai-paint-image64-v1',
            code_revision=os.getenv('VCS_REF','development'),runtime_cold_ms=self.start_ms,resources=resources)

    def predict(self,value,pixels):
        start=time.perf_counter();x=normalize_rgb(dino_rgb(pixels))[None];pre=(time.perf_counter()-start)*1000;t=time.perf_counter()
        with self.torch.inference_mode(),self.torch.jit.optimized_execution(False):
            logits=self.model(self.torch.from_numpy(x)).numpy()
        inference=(time.perf_counter()-t)*1000
        if logits.shape!=(1,10) or not np.isfinite(logits).all():raise RuntimeError('Invalid model output')
        return dict(self.health(),request_id=value['request_id'],drawing_revision=value['drawing_revision'],model_id=MODEL_ID,
            pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),logits=logits[0].tolist(),predicted_id=int(logits.argmax()),
            preprocessing_ms=pre,inference_ms=inference,server_total_ms=(time.perf_counter()-start)*1000)


def create_app(runtime_factory=None,allowed_origins=None):
    origins=set(allowed_origins if allowed_origins is not None else filter(None,os.getenv('ALLOWED_ORIGINS','').split(',')))
    for origin in origins:
        u=urlsplit(origin)
        if u.scheme not in ('http','https') or not u.hostname or u.path or u.query or u.fragment or u.username or '*' in origin:raise ValueError('Origins must be exact scheme://host[:port]')
    gate=threading.Lock()
    @asynccontextmanager
    async def lifespan(app):
        factory=runtime_factory or (lambda:DinoRuntime(os.getenv('MODEL_BUNDLE_DIR','models/exported/paint35-v1')))
        app.state.runtime=await run_in_threadpool(factory)
        yield
    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    @app.middleware('http')
    async def policy(request,call_next):
        origin=request.headers.get('origin')
        if origin and origin not in origins:return JSONResponse({'error':'ORIGIN'},403)
        if request.method=='OPTIONS':
            response=JSONResponse({},200) if request.url.path=='/predict' else JSONResponse({'error':'NOT_FOUND'},404)
        else:response=await call_next(request)
        response.headers['Cache-Control']='no-store'
        if origin:
            response.headers['Access-Control-Allow-Origin']=origin;response.headers['Vary']='Origin'
            response.headers['Access-Control-Allow-Methods']='GET, POST, OPTIONS';response.headers['Access-Control-Allow-Headers']='Content-Type'
        return response
    @app.get('/health')
    async def health():return app.state.runtime.health()
    @app.post('/predict')
    async def predict(request:Request):
        if request.headers.get('content-type','').split(';')[0].strip()!='application/json':return JSONResponse({'error':'CONTENT_TYPE'},415)
        if not gate.acquire(blocking=False):return JSONResponse({'error':'BUSY'},429)
        try:
            data=bytearray()
            async with asyncio.timeout(10):
                async for block in request.stream():
                    data.extend(block)
                    if len(data)>8192:return JSONResponse({'error':'SIZE'},413)
            value,pixels=decode_request(bytes(data))
            return await run_in_threadpool(app.state.runtime.predict,value,pixels)
        except TimeoutError:return JSONResponse({'error':'BODY_TIMEOUT'},408)
        except (ValueError,TypeError,KeyError,OverflowError):return JSONResponse({'error':'INVALID_INPUT'},400)
        except Exception:return JSONResponse({'error':'INFERENCE'},500)
        finally:gate.release()
    return app


app=create_app()
