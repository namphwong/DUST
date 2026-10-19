"""P7 fresh-process API parity against P6 procedural fixtures; no private data/test access."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai.native_runtime import configure_windows_dlls
configure_windows_dlls()
import numpy as np
import psutil
from PIL import Image
from training.drawai.private_data import load,dump
from training.drawai.paint_package import TOLERANCES
from training.drawai.paint_app_runtime import ADAPTER,PROTOCOL,BUNDLE_SHA256
OUT=ROOT/'artifacts/drawai/phase7-image64'


def request(path,value=None,origin=None):
    headers={'Content-Type':'application/json'}
    if origin:headers['Origin']=origin
    payload=value if isinstance(value,bytes) else json.dumps(value).encode() if value is not None else None
    req=urllib.request.Request('http://127.0.0.1:4197'+path,data=payload,headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=20) as response:return response.status,json.load(response)
    except urllib.error.HTTPError as error:return error.code,json.load(error)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'backend-server.log').open('w',encoding='utf-8') as log:
        start=time.perf_counter();process=subprocess.Popen([sys.executable,'-m','training.drawai.paint_app_runtime'],cwd=ROOT,stdout=log,stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            while True:
                if process.poll() is not None:raise RuntimeError('Backend exited; inspect log')
                try:status,health=request('/health');break
                except (OSError,urllib.error.URLError):
                    if time.perf_counter()-start>15:raise RuntimeError('Cold process startup exceeded 15s')
                    time.sleep(.1)
            cold=(time.perf_counter()-start)*1000
            ancestry={health['process_id'],*(p.pid for p in psutil.Process(health['process_id']).parents())}
            assert process.pid in ancestry,'Port 4197 belongs to another backend; run verification sequentially'
            assert health['bundle_sha256']==BUNDLE_SHA256 and not health['timm_imported'] and not health['transformers_imported']
            records=[];request_id=0
            for family,tolerance in TOLERANCES.items():
                fixtures=load(ROOT/'models/exported/paint35-v1'/family/'fixtures.json')['cases']
                for fixture in fixtures:
                    with Image.open(io.BytesIO(base64.b64decode(fixture['png64_base64']))) as image:pixels=np.asarray(image,dtype=np.uint8).tobytes()
                    if min(pixels)==255:continue  # P7 rejects blank images; separate negative probe below.
                    request_id+=1;value=dict(protocol=PROTOCOL,request_id=request_id,drawing_revision=7,canvas_adapter=ADAPTER,model_id=family,pixels64_base64=base64.b64encode(pixels).decode())
                    status,result=request('/predict',value)
                    assert status==200 and result['request_id']==request_id and result['drawing_revision']==7 and result['data_persisted'] is False,(family,fixture.get('name'),status,result)
                    actual=np.asarray(result['logits']);expected=np.asarray(fixture['expected_logits'])
                    assert np.allclose(actual,expected,**tolerance) and int(expected.argmax())==result['predicted_id']
                    records.append(dict(model_id=family,max_abs_error=float(np.max(np.abs(actual-expected))),passed=True))
            valid=value
            probes=[('missing-fields',{},400),('unknown-model',dict(valid,model_id='unknown'),400),('wrong-adapter',dict(valid,canvas_adapter='old'),400),
                ('boolean-request',dict(valid,request_id=True),400),('invalid-base64',dict(valid,pixels64_base64='!'*5464),400),
                ('blank-image',dict(valid,pixels64_base64=base64.b64encode(bytes([255])*4096).decode()),400),
                ('oversized',b' '*8193,413),('duplicate-json',b'{"x":0,"x":1}',400)]
            errors=[]
            for name,value,expected_status in probes:
                status,_=request('/predict',value);assert status==expected_status,(name,status);errors.append(dict(probe=name,status=status))
            assert request('/predict',valid,origin='https://example.com')[0]==403
            assert request('/predict',valid)[0]==200
            _,health=request('/health');assert cold<=15000 and health['rss_bytes']<=1073741824
            dump(OUT/'backend-verification.json',dict(passed=True,cold_process_start_ms=cold,health=health,procedural_parity=records,
                error_probes=errors,origin_rejected=True,recovery_passed=True,private_data_accessed=False,test_accessed=False,
                no_timm_or_transformers_at_runtime=True,public_deployment=False))
            print(f'P7 backend PASS: {len(records)} procedural parity cases; cold {cold:.1f} ms; {len(errors)} invalid requests + origin/recovery.')
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
