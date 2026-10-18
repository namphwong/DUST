"""Local P2 CPU backend: same pretrained ONNX, raw strokes -> raster -> logits.

Technical verification only. Digit head is untrained; never advertises a release.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time

START=time.perf_counter()
_dll=None
if os.name=='nt':
    p=Path(os.environ.get('DUST_DLL_DIRECTORY','C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'))
    if p.is_dir():_dll=os.add_dll_directory(str(p))

import numpy as np
import onnxruntime as ort
import psutil
from .raster import render_rgb,normalize_rgb
from .prepare_data import ROOT


class Runtime:
    def __init__(self,directory):
        self.lock=threading.Lock();self.process=psutil.Process();self.peak_rss=0
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        self.report=json.loads((directory/'python-report.json').read_text(encoding='utf-8'))
        if not manifest['pretrained_loaded'] or manifest['trained'] or manifest['release_ready']:
            raise ValueError('Backend requires verified pretrained backbone + explicitly untrained P2 head')
        name=manifest['model_filename'];meta=manifest['files'][name];path=directory/name
        if path.stat().st_size!=meta['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:
            raise ValueError('ONNX hash mismatch')
        self.model_sha=meta['sha256'];self.provenance=self.report['provenance']
        self.stop=threading.Event()
        def sample():
            while not self.stop.is_set():
                self.peak_rss=max(self.peak_rss,self.process.memory_info().rss)
                self.stop.wait(.02)
        self.sampler=threading.Thread(target=sample,daemon=True);self.sampler.start()
        options=ort.SessionOptions();options.intra_op_num_threads=4
        self.session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        self.cold_ms=(time.perf_counter()-START)*1000

    def health(self):
        return dict(status='ready',technical_only=True,pretrained_loaded=True,trained=False,release_ready=False,
                    model_sha256=self.model_sha,provenance=self.provenance,threads=4,cold_start_ms=self.cold_ms,
                    current_rss_bytes=self.process.memory_info().rss,peak_sampled_rss_bytes=self.peak_rss,
                    memory_method='process RSS sampled every 20 ms; not guaranteed exact peak',data_persisted=False)

    def infer(self,value):
        if not isinstance(value,dict) or set(value)!={'strokes','y_axis','model_sha256'}:
            raise ValueError('SCHEMA: expected strokes/y_axis/model_sha256')
        if value['model_sha256']!=self.model_sha:raise ValueError('MODEL: stale or wrong model')
        start=time.perf_counter();rgb=render_rgb(value['strokes'],value['y_axis']);values=normalize_rgb(rgb)[None]
        raster_ms=(time.perf_counter()-start)*1000;t=time.perf_counter()
        logits=self.session.run(None,{'pixel_values':values})[0][0]
        inference_ms=(time.perf_counter()-t)*1000
        if not np.isfinite(logits).all():raise ValueError('OUTPUT: nonfinite logits')
        return dict(technical_only=True,pretrained_loaded=True,release_ready=False,trained=False,
            model_sha256=self.model_sha,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            tensor_sha256=hashlib.sha256(values.tobytes()).hexdigest(),logits=logits.tolist(),
            predicted_id=int(logits.argmax()),raster_ms=raster_ms,inference_ms=inference_ms,
            server_total_ms=(time.perf_counter()-start)*1000,data_persisted=False)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass # Never persist request bodies/handwriting.
    def send_json(self,status,value):
        payload=json.dumps(value,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        if self.path!='/health':self.send_json(404,{'error':'NOT_FOUND'});return
        self.send_json(200,self.server.runtime.health())
    def do_POST(self):
        if self.path!='/infer':self.send_json(404,{'error':'NOT_FOUND'});return
        try:size=int(self.headers.get('Content-Length','0'))
        except ValueError:self.send_json(400,{'error':'LENGTH'});return
        if not 0<size<=1048576:self.send_json(413,{'error':'SIZE'});return
        if not self.server.runtime.lock.acquire(blocking=False):self.send_json(429,{'error':'BUSY'});return
        try:
            def invalid_constant(value):raise ValueError('NUMBER: nonfinite JSON constant')
            value=json.loads(self.rfile.read(size),parse_constant=invalid_constant)
            self.send_json(200,self.server.runtime.infer(value))
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            self.send_json(400,{'error':str(error)})
        finally:self.server.runtime.lock.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=4190)
    parser.add_argument('--artifact-dir',type=Path,default=ROOT/'artifacts/drawai/phase2/technical')
    args=parser.parse_args();runtime=Runtime(args.artifact_dir.resolve())
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.runtime=runtime
    print(json.dumps(runtime.health()),flush=True)
    try:server.serve_forever()
    finally:runtime.stop.set();server.server_close()
