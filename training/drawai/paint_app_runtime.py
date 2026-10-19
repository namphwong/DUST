"""P7 local application adapter over the hash-verified P6 inference runtime."""
import argparse
import base64
import binascii
import hashlib
import io
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

from .paint_runtime import Runtime, Handler, Image, np
from .private_data import ROOT

BUNDLE_SHA256='86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2'
ADAPTER='drawai-canvas400-gray64-v1'
PROTOCOL='drawai-paint-api-v1'


def adapt_request(value):
    keys={'protocol','request_id','drawing_revision','model_id','canvas_adapter','pixels64_base64'}
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError('CONTRACT: Unexpected fields')
    if value['protocol']!=PROTOCOL or value['canvas_adapter']!=ADAPTER:raise ValueError('CONTRACT: Version mismatch')
    for key in ('request_id','drawing_revision'):
        if type(value[key]) is not int or not 0<=value[key]<=9007199254740991:raise ValueError('CONTRACT: Invalid revision/request')
    if value['model_id'] not in ('swin1d-image','dinov3-vits16'):raise ValueError('CONTRACT: Unknown model')
    text=value['pixels64_base64']
    if not isinstance(text,str) or len(text)!=5464:raise ValueError('SIZE: Expected 4096 pixels')
    try:pixels=base64.b64decode(text,validate=True)
    except (ValueError,binascii.Error):raise ValueError('PIXELS: Invalid base64') from None
    if len(pixels)!=4096 or min(pixels)==255:raise ValueError('PIXELS: Empty/invalid image')
    stream=io.BytesIO();Image.fromarray(np.frombuffer(pixels,dtype=np.uint8).reshape(64,64)).save(stream,format='PNG')
    return dict(model_id=value['model_id'],png64_base64=base64.b64encode(stream.getvalue()).decode()),hashlib.sha256(pixels).hexdigest()


class AppHandler(Handler):
    def do_GET(self):
        if self.path!='/health':self.send_json(404,dict(error='NOT_FOUND'));return
        health=self.server.runtime.health()
        health.update(protocol=PROTOCOL,canvas_adapter=ADAPTER,preprocessing_version='drawai-paint-image64-v1',
            default_model='dinov3-vits16',process_id=os.getpid(),input_protocol='4096 grayscale uint8 row-major pixels, base64; no strokes')
        self.send_json(200,health)

    def do_POST(self):
        if self.path!='/predict':self.send_json(404,dict(error='NOT_FOUND'));return
        if self.headers.get('Origin') is not None or self.headers.get_content_type()!='application/json':
            self.send_json(403,dict(error='USE_SAME_ORIGIN_LOCAL_PROXY'));return
        try:size=int(self.headers.get('Content-Length','0'))
        except ValueError:self.send_json(400,dict(error='LENGTH'));return
        if not 0<size<=8192:self.send_json(413,dict(error='SIZE'));return
        if not self.server.runtime.lock.acquire(blocking=False):self.send_json(429,dict(error='BUSY'));return
        try:
            def invalid(_):raise ValueError('Nonfinite JSON')
            def pairs(items):
                value={}
                for key,item in items:
                    if key in value:raise ValueError('Duplicate JSON key')
                    value[key]=item
                return value
            value=json.loads(self.rfile.read(size),parse_constant=invalid,object_pairs_hook=pairs)
            request,image_hash=adapt_request(value)
            result=self.server.runtime.infer(request)
            result.update(protocol=PROTOCOL,request_id=value['request_id'],drawing_revision=value['drawing_revision'],
                canvas_adapter=ADAPTER,preprocessing_version='drawai-paint-image64-v1',pixels_sha256=image_hash)
            # Technical local integration; physical touch/new-user acceptance is still pending.
            self.send_json(200,result)
        except (ValueError,KeyError,TypeError,OverflowError) as error:self.send_json(400,dict(error=str(error)))
        except Exception:self.send_json(500,dict(error='INFERENCE: Runtime failure; retry or restart backend'))
        finally:self.server.runtime.lock.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT/'models/exported/paint35-v1')
    parser.add_argument('--port',type=int,default=4197)
    args=parser.parse_args()
    runtime=Runtime(args.bundle,BUNDLE_SHA256)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),AppHandler);server.runtime=runtime
    print(json.dumps(runtime.health()),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
