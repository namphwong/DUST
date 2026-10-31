"""Real model HTTP parity smoke against public procedural P6 fixtures; no private dataset."""
import argparse
import base64
import io
import json
import math
from pathlib import Path
import time
import urllib.request
import urllib.error
import numpy as np
from PIL import Image

BUNDLE_HASH='86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2'


def request(url,value=None,timeout=30):
    req=urllib.request.Request(url,data=None if value is None else json.dumps(value).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)


def wait_ready(url,expected_revision=None,timeout=120):
    start=time.perf_counter()
    while True:
        remaining=timeout-(time.perf_counter()-start)
        if remaining<=0:raise RuntimeError('Readiness/revision timed out')
        try:
            health=request(url+'/health',timeout=min(30,remaining))
            if health.get('status')=='ready' and (not expected_revision or health.get('code_revision')==expected_revision):return health
        except (OSError,ValueError):pass
        time.sleep(min(1,max(0,timeout-(time.perf_counter()-start))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',required=True);parser.add_argument('--bundle',type=Path,default=Path('deployment/docker/model-bundle'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--expected-revision');parser.add_argument('--enforce-container-budget',action='store_true')
    parser.add_argument('--readiness-timeout',type=float,default=120);args=parser.parse_args()
    if not 0<args.readiness_timeout<=600:parser.error('readiness-timeout must be >0 and <=600 seconds')
    start=time.perf_counter()
    health=wait_ready(args.url,args.expected_revision,args.readiness_timeout)
    readiness_ms=(time.perf_counter()-start)*1000
    assert health['models']==['dinov3-vits16'] and health['bundle_sha256']==BUNDLE_HASH
    if args.expected_revision:assert health['code_revision']==args.expected_revision
    fixtures=json.loads((args.bundle/'dinov3-vits16/fixtures.json').read_text(encoding='utf-8'))['cases'];rows=[]
    samples=[]
    for fixture in fixtures:
        with Image.open(io.BytesIO(base64.b64decode(fixture['png64_base64']))) as image:pixels=np.asarray(image,dtype=np.uint8).tobytes()
        if min(pixels)==255:continue
        value=dict(protocol='drawai-paint-api-v1',request_id=len(rows)+1,drawing_revision=7,model_id='dinov3-vits16',canvas_adapter='drawai-canvas400-gray64-v1',pixels64_base64=base64.b64encode(pixels).decode())
        result=request(args.url+'/predict',value);a=np.asarray(result['logits']);b=np.asarray(fixture['expected_logits'])
        assert a.shape==(10,) and np.isfinite(a).all() and np.allclose(a,b,atol=1e-4,rtol=1e-3)
        assert result['predicted_id']==int(b.argmax()) and result['request_id']==value['request_id'] and result['drawing_revision']==7
        assert result['data_persisted'] is False and result['artifact_verified'] is True and result['bundle_sha256']==BUNDLE_HASH
        rows.append(dict(name=fixture.get('name'),max_abs_error=float(np.max(np.abs(a-b))),predicted_id=result['predicted_id']));samples.append(value)
    times=[]
    for i in range(35):
        t=time.perf_counter();request(args.url+'/predict',samples[i%len(samples)])
        if i>=5:times.append((time.perf_counter()-t)*1000)
    times.sort()
    health=request(args.url+'/health')
    p95=times[math.ceil(.95*len(times))-1]
    if args.enforce_container_budget:
        assert health['resources']['memory_peak_bytes'] is not None and health['resources']['memory_peak_bytes']<=450000000,health['resources']
        assert health['resources']['memory_events']['oom_kill']==0
        assert p95<=5000 and readiness_ms<=90000,(p95,readiness_ms)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True,health=health,procedural_cases=rows,source='public P6 procedural fixtures; no private test',warmup=5,iterations=30,warm_http_p95_ms=p95,readiness_ms=readiness_ms,container_budget_enforced=args.enforce_container_budget),indent=2)+'\n',encoding='utf-8')
    print(f"Model HTTP smoke PASS: {len(rows)} fixtures; p95={times[28]:.1f}ms")


if __name__=='__main__':main()
