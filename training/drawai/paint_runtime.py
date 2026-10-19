"""P6 local CPU handoff: ONNX Swin + TorchScript DINO. PNG64 input only."""
import argparse
import base64
import binascii
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time

from .native_runtime import configure_windows_dlls
configure_windows_dlls()
import numpy as np
import onnxruntime as ort
from PIL import Image
import psutil
from .private_data import ROOT
from .paint_package import verify_bundle,spec
from . import paint_data as paint
from .raster import normalize_rgb


def decode_png(value):
    if not isinstance(value,str) or not 0<len(value)<=24576:raise ValueError('PNG64 base64 size/type invalid')
    try:payload=base64.b64decode(value,validate=True)
    except (ValueError,binascii.Error):raise ValueError('Invalid base64') from None
    if len(payload)>16384 or payload[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Expected bounded PNG64')
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format!='PNG' or image.mode!='L' or image.size!=(64,64):raise ValueError('Expected grayscale PNG64')
            image.load();return np.asarray(image).copy()
    except (OSError,SyntaxError):raise ValueError('Invalid PNG payload') from None


class Runtime:
    def __init__(self,directory,expected_bundle_sha256):
        start=time.perf_counter();self.directory=Path(directory).resolve()
        self.record,self.registry=verify_bundle(self.directory,expected_bundle_sha256)
        self.lock=threading.Lock();self.sessions={};self.process=psutil.Process();self.torch=None
        for family in self.registry['models']:
            info=spec(family);options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
            if info['runtime_format']=='torchscript':
                import torch
                torch.set_num_threads(4);self.torch=torch
                self.sessions[family]=torch.jit.load(str(self.directory/family/info['model_filename']),map_location='cpu').eval()
                continue
            session=ort.InferenceSession(str(self.directory/family/info['model_filename']),sess_options=options,providers=['CPUExecutionProvider'])
            a,b=session.get_inputs(),session.get_outputs()
            if len(a)!=1 or len(b)!=1 or (a[0].name,a[0].shape,a[0].type,b[0].name,b[0].shape,b[0].type)!=(info['input_name'],info['input_shape'],'tensor(float)','logits',[1,10],'tensor(float)'):
                raise ValueError('ONNX input/output contract mismatch')
            self.sessions[family]=session
        self.cold_ms=(time.perf_counter()-start)*1000

    def health(self):
        memory=self.process.memory_info()
        return dict(status='ready',technical_only=True,trained=True,artifact_verified=True,release_ready=False,
            models=list(self.sessions),bundle_sha256=self.record['bundle_sha256'],provider='CPUExecutionProvider + TorchScript CPU',threads=4,
            runtime_cold_ms=self.cold_ms,rss_bytes=memory.rss,process_peak_rss=getattr(memory,'peak_wset',memory.rss),
            torch_imported='torch' in __import__('sys').modules,timm_imported='timm' in __import__('sys').modules,
            transformers_imported='transformers' in __import__('sys').modules,
            runtime_formats={family:spec(family)['runtime_format'] for family in self.sessions},
            providers={family:('CPUExecutionProvider' if spec(family)['runtime_format']=='onnx' else 'TorchScript CPU') for family in self.sessions},
            data_persisted=False,input_protocol='grayscale PNG64, not strokes')

    def infer(self,value):
        if not isinstance(value,dict) or set(value)!={'model_id','png64_base64'}:raise ValueError('Expected model_id/png64_base64')
        family=value['model_id']
        if not isinstance(family,str) or family not in self.sessions:raise ValueError('Unknown model_id')
        start=time.perf_counter();pixels=decode_png(value['png64_base64'])
        x=(paint.swin_rows(pixels) if family=='swin1d-image' else normalize_rgb(paint.dino_rgb(pixels)))[None]
        pre=(time.perf_counter()-start)*1000;t=time.perf_counter()
        if spec(family)['runtime_format']=='torchscript':
            with self.torch.inference_mode(),self.torch.jit.optimized_execution(False):
                logits=self.sessions[family](self.torch.from_numpy(x)).numpy()
        else:logits=self.sessions[family].run(None,{spec(family)['input_name']:x})[0]
        if logits.shape!=(1,10) or not np.isfinite(logits).all():raise ValueError('Nonfinite/invalid ONNX output')
        return dict(model_id=family,artifact_format=spec(family)['runtime_format'],trained=True,artifact_verified=True,release_ready=False,technical_only=True,
            bundle_sha256=self.record['bundle_sha256'],logits=logits[0].tolist(),predicted_id=int(logits.argmax(1)[0]),
            preprocessing_ms=pre,inference_ms=(time.perf_counter()-t)*1000,server_total_ms=(time.perf_counter()-start)*1000,data_persisted=False)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send_json(self,status,value):
        payload=json.dumps(value,allow_nan=False).encode();self.send_response(status)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)))
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        if self.path!='/health':self.send_json(404,dict(error='NOT_FOUND'));return
        self.send_json(200,self.server.runtime.health())
    def do_POST(self):
        if self.path!='/infer':self.send_json(404,dict(error='NOT_FOUND'));return
        # Same-origin proxy/client integration is P7; no permissive browser CORS.
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:4186','http://localhost:4186'):
            self.send_json(403,dict(error='ORIGIN'));return
        try:size=int(self.headers.get('Content-Length','0'))
        except ValueError:self.send_json(400,dict(error='LENGTH'));return
        if not 0<size<=32768:self.send_json(413,dict(error='SIZE'));return
        if not self.server.runtime.lock.acquire(blocking=False):self.send_json(429,dict(error='BUSY'));return
        try:
            def invalid(value):raise ValueError('Nonfinite JSON')
            def pairs(items):
                obj={}
                for key,item in items:
                    if key in obj:raise ValueError('Duplicate JSON key')
                    obj[key]=item
                return obj
            value=json.loads(self.rfile.read(size),parse_constant=invalid,object_pairs_hook=pairs)
            self.send_json(200,self.server.runtime.infer(value))
        except (ValueError,KeyError,TypeError,OverflowError) as error:self.send_json(400,dict(error=str(error)))
        finally:self.server.runtime.lock.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,default=ROOT/'models/exported/paint35-v1')
    parser.add_argument('--expected-bundle-sha256',required=True);parser.add_argument('--port',type=int,default=4196)
    args=parser.parse_args();runtime=Runtime(args.bundle,args.expected_bundle_sha256)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.runtime=runtime
    print(json.dumps(runtime.health()),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
