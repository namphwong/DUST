"""Launch the portable ONNX backend, test HTTP parity/errors and P2 budgets."""
import base64
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai.private_data import load,dump
from training.drawai.paint_package import TOLERANCES,LOCK_HASH
from training.drawai import paint_dataset as data
import numpy as np

OUT=ROOT/'artifacts/drawai/phase6-image64';BUNDLE=ROOT/'models/exported/paint35-v1'


def request(path,value=None,origin=None):
    headers={'Content-Type':'application/json'}
    if origin:headers['Origin']=origin
    raw=None if value is None else value if isinstance(value,bytes) else json.dumps(value).encode()
    req=urllib.request.Request('http://127.0.0.1:4196'+path,data=raw,headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=10) as response:return response.status,json.load(response)
    except urllib.error.HTTPError as error:return error.code,json.load(error)


def main():
    anchor=load(OUT/'trusted-bundle.json');_,rows=data.verify_lock(ROOT/'data/private/locked/paint35-main-v1',LOCK_HASH)
    samples=[]
    for label in range(10):
        row=next(r for r in rows if r['split']=='validation' and int(r['label'])==label)
        samples.append(base64.b64encode((ROOT/'data/private/locked/paint35-main-v1/development'/row['png64']).read_bytes()).decode())
    with (OUT/'backend-server.log').open('w',encoding='utf-8') as log:
        start=time.perf_counter();process=subprocess.Popen([sys.executable,'-m','training.drawai.paint_runtime','--expected-bundle-sha256',anchor['bundle_sha256']],
            cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            while True:
                if process.poll() is not None:raise RuntimeError('Backend exited; inspect backend-server.log')
                try:status,health=request('/health');break
                except (OSError,urllib.error.URLError):
                    if time.perf_counter()-start>15:raise RuntimeError('Cold-start budget exceeded')
                    time.sleep(.1)
            cold_ms=(time.perf_counter()-start)*1000
            assert status==200 and health['timm_imported'] is False and health['transformers_imported'] is False and health['bundle_sha256']==anchor['bundle_sha256']
            parity={};benchmarks={}
            for family,tolerance in TOLERANCES.items():
                fixtures=load(BUNDLE/family/'fixtures.json')['cases'];maximum=0.
                for fixture in fixtures:
                    status,value=request('/infer',dict(model_id=family,png64_base64=fixture['png64_base64']))
                    assert status==200 and value['trained'] and value['data_persisted'] is False
                    expected=np.asarray(fixture['expected_logits']);actual=np.asarray(value['logits'])
                    assert np.allclose(actual,expected,**tolerance) and int(expected.argmax())==value['predicted_id']
                    maximum=max(maximum,float(np.max(np.abs(expected-actual))))
                parity[family]=dict(cases=len(fixtures),max_abs_error=maximum,passed=True)
                times=[]
                for i in range(35):
                    before=time.perf_counter();status,value=request('/infer',dict(model_id=family,png64_base64=samples[i%10]));elapsed=(time.perf_counter()-before)*1000
                    assert status==200 and np.isfinite(value['logits']).all()
                    if i>=5:times.append(elapsed)
                times.sort();benchmarks[family]=dict(warmup=5,iterations=30,p50_ms=float(np.median(times)),p95_ms=times[math.ceil(.95*len(times))-1],
                    timing_scope='JSON HTTP roundtrip + PNG64 decode/preprocessing + full exported model inference (ONNX/TorchScript) + response parse',source='ten validation images covering digits0-9')
            valid=dict(model_id='swin1d-image',png64_base64=samples[0])
            probes=[('missing-field',{},400),('unknown-model',dict(valid,model_id='unknown'),400),('invalid-base64',dict(valid,png64_base64='!!'),400),
                    ('extra-field',dict(valid,extra=True),400),('oversize-body',b' '*32769,413),('duplicate-key',b'{"model_id":"x","model_id":"y","png64_base64":"x"}',400)]
            errors=[]
            for name,value,expected in probes:
                status,body=request('/infer',value);assert status==expected,(name,status);errors.append(dict(probe=name,status=status))
            assert request('/infer',valid,origin='https://example.com')[0]==403
            assert request('/missing')[0]==404 and request('/infer',valid)[0]==200
            _,health=request('/health');budget=load(ROOT/'docs/drawai/phase2-backend-budget.json')
            assert cold_ms<=budget['cold_start_max_ms'] and health['rss_bytes']<=budget['process_rss_max_bytes']
            assert all(b['p95_ms']<=budget['end_to_end_p95_max_ms'] for b in benchmarks.values())
            dump(OUT/'backend-verification.json',dict(passed=True,health=health,cold_process_start_ms=cold_ms,
                procedural_parity=parity,benchmarks=benchmarks,error_probes=errors,origin_rejected=True,recovery_passed=True,
                existing_P2_cpu_budgets_passed=True,no_timm_or_pretrained_snapshot_dependency_at_runtime=True,public_deployment=False))
            print('P6 backend PASS:ONNX Swin and TorchScript DINO, HTTP parity/errors and CPU budgets; no timm/pretrained snapshot required.')
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
