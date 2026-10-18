"""D2 technical tests only; exported model is untrained, not a digit recognizer.

python -m training.drawai.verify_feasibility
Optional Windows DLL search directory: DUST_DLL_DIRECTORY (process-local).
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

_dll_handle = None
if sys.platform == 'win32' and os.environ.get('DUST_DLL_DIRECTORY'):
    _dll_handle = os.add_dll_directory(os.environ['DUST_DLL_DIRECTORY'])

import numpy as np
import torch
import onnx
import onnxruntime as ort

from .models import build_model, swin_config
from .preprocessing import preprocess, contract
from .prepare_data import ROOT, digest, write_json

ATOL, RTOL = 1e-5, 1e-4


def geometry_cases(length):
    def pts(*pairs):
        return [dict(x=x,y=y) for x,y in pairs]
    shapes = dict(horizontal=[pts((0,0),(10,0))], vertical=[pts((1,0),(1,10))],
                  two_strokes=[pts((0,0),(1,0)),pts((9,0),(10,0))],
                  duplicates=[pts((0,0),(0,0),(1,2),(1,2),(3,0))],
                  singleton=[pts((0,0)),pts((2,1),(3,2))],
                  equal_three=[pts((i*3,0),(i*3+1,0)) for i in range(3)],
                  corners=[pts((0,0),(0,2),(3,2),(3,0))])
    cases = [dict(name=name,strokes=strokes,y_axis='down',
                  input=preprocess(strokes,length).reshape(-1).tolist()) for name,strokes in shapes.items()]
    train_cases = json.loads((ROOT/f'data/processed/pendigits-drawai-v1/l{length}/fixtures.json').read_text())['cases']
    return cases + train_cases


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    out = ROOT / 'artifacts/drawai/feasibility'
    out.mkdir(parents=True, exist_ok=True)
    d1 = json.loads((ROOT/'artifacts/drawai/data/verification.json').read_text())
    if d1['status'] != 'passed' or d1['verified_tensor_count'] != 21984:
        raise ValueError('Need verified D1 data for both lengths before D2')
    # Use the untouched training initialization for backward/optimizer checks.
    checks = []
    for length in [64,128]:
        model = build_model('swin',length)
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.003)
        X = torch.randn(4,3,length)
        y = torch.tensor([0,3,7,9])
        losses = []
        gradient_history = []
        for _ in range(8):
            optimizer.zero_grad()
            logits = model(X)
            assert logits.shape == (4,10) and torch.isfinite(logits).all()
            loss = torch.nn.functional.cross_entropy(logits,y)
            assert torch.isfinite(loss)
            loss.backward()
            for parameter in [model.patch_embed.proj.weight,model.layers[0].blocks[1].attn.qkv.weight]:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            monitored = {'patch_embedding':model.patch_embed.proj.weight,
                         'shifted_attention':model.layers[0].blocks[1].attn.qkv.weight,
                         'mlp':model.layers[0].blocks[1].mlp.fc1.weight,'head':model.head.weight}
            for parameter in model.parameters():
                assert parameter.grad is None or torch.isfinite(parameter.grad).all()
            gradient_history.append({name:float(p.grad.abs().sum()) for name,p in monitored.items()})
            losses.append(float(loss.detach()))
            optimizer.step()
        assert model.patch_embed.proj.weight.grad.abs().sum() > 0
        assert model.layers[0].blocks[1].attn.qkv.weight.grad.abs().sum() > 0
        model.eval()
        with torch.inference_mode():
            assert model(torch.zeros(1,3,length)).shape == (1,10)
        assert all(value > 0 for value in gradient_history[-1].values())
        checks.append(dict(length=length,backward='passed',steps=8,losses=losses,
                           batch_shapes={'1':[1,10],'4':[4,10]},gradient_l1_by_step=gradient_history,
                           original_initialization=True))
    # Learn one actual TRAIN sample per class; never use validation/test here.
    torch.manual_seed(42)
    data = np.load(ROOT/'data/processed/pendigits-drawai-v1/l64/train.npz',allow_pickle=False)
    ids = [int(np.flatnonzero(data['y']==i)[0]) for i in range(10)]
    X,y = torch.from_numpy(data['X'][ids]),torch.from_numpy(data['y'][ids])
    tiny_sample_ids = data['sample_ids'][ids].tolist()
    manifest = json.loads((ROOT/'data/processed/pendigits-drawai-v1/l64/manifest.json').read_text())
    train_hash = hashlib.sha256(data['X'].tobytes()+data['y'].tobytes()+'\n'.join(data['sample_ids']).encode()).hexdigest()
    assert train_hash == manifest['splits']['train']['tensor_sha256']
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0)
    history = []
    for step in range(200):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X),y)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        if step == 0 or (step+1)%20==0:
            model.eval()
            with torch.inference_mode():
                accuracy = float((model(X).argmax(1)==y).float().mean())
            history.append(dict(step=step+1,loss=float(loss.detach()),accuracy=accuracy))
            print(f"Tiny-train check {step+1}: loss={loss.item():.4f}, acc={accuracy:.2f}",flush=True)
            if step >= 39 and accuracy == 1.0:
                break
    assert history[-1]['accuracy'] >= 0.9, 'Cannot learn tiny train subset'
    reports = []
    for length in [64,128]:
        torch.manual_seed(42)
        model = build_model('swin',length).eval()
        # D2 fixture deliberately activates attention/MLP residual branches.
        # This modification is only for untrained export parity, never training.
        for stage in model.layers:
            for block in stage.blocks:
                torch.nn.init.ones_(block.norm1.weight)
                torch.nn.init.ones_(block.norm2.weight)
        target = ROOT/f'web/public/models/drawai-feasibility-l{length}'
        target.mkdir(parents=True,exist_ok=True)
        path = target/'model.onnx'
        torch.onnx.export(model,torch.zeros(1,3,length),str(path),input_names=['drawing'],
                          output_names=['logits'],opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(str(path)))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        assert session.get_inputs()[0].shape == [1,3,length]
        assert session.get_outputs()[0].shape == [1,10]
        cases = geometry_cases(length)
        max_error = 0
        for case in cases:
            values = np.asarray(case['input'],np.float32).reshape(1,3,length)
            with torch.inference_mode():
                expected = model(torch.from_numpy(values)).numpy()[0]
            observed = session.run(None,dict(drawing=values))[0][0]
            np.testing.assert_allclose(observed,expected,atol=ATOL,rtol=RTOL)
            case['expected'] = expected.tolist()
            max_error = max(max_error,float(np.abs(observed-expected).max()))
        # Additional non-geometric tensors exercise boundaries in the graph.
        rng = np.random.default_rng(42)
        for name,values in [('zeros',np.zeros((1,3,length),np.float32)),
                            ('ones',np.ones((1,3,length),np.float32)),
                            ('random',rng.normal(size=(1,3,length)).astype(np.float32))]:
            with torch.inference_mode():
                expected = model(torch.from_numpy(values)).numpy()[0]
            actual = session.run(None,dict(drawing=values))[0][0]
            np.testing.assert_allclose(actual,expected,atol=ATOL,rtol=RTOL)
            max_error = max(max_error,float(np.abs(actual-expected).max()))
            cases.append(dict(name=name,input=values.reshape(-1).tolist(),expected=expected.tolist()))
        timings = []
        for _ in range(5):
            session.run(None,dict(drawing=values))
        for _ in range(100):
            start = time.perf_counter()
            session.run(None,dict(drawing=values))
            timings.append((time.perf_counter()-start)*1000)
        report = dict(status='passed',trained=False,length=length,config=swin_config(length),
                      input_shape=[1,3,length],output_shape=[1,10],input_name='drawing',output_name='logits',
                      atol=ATOL,rtol=RTOL,fixture_count=len(cases),max_abs_error=max_error,
                      model_bytes=path.stat().st_size,sha256=digest(path),
                      parameters=sum(p.numel() for p in model.parameters()),
                      benchmark=dict(warmup=5,iterations=100,input='random',threads=4,
                                     percentile_method='numpy linear interpolation'),
                      checkpoint_source='fresh seed-42 model; tiny-train weights not exported',
                      opset=17,graph_checker='passed',
                      python_ort_median_ms=float(np.median(timings)),python_ort_p95_ms=float(np.percentile(timings,95)),
                      note='Untrained technical fixture; residual post-norm weights set to one for parity only.')
        write_json(target/'fixtures.json',dict(atol=ATOL,rtol=RTOL,cases=cases))
        write_json(target/'preprocessing.json',contract(length))
        write_json(target/'report.json',report)
        write_json(target/'release.json',dict(version=f'drawai-feasibility-l{length}',trained=False,
                   files={n:dict(sha256=digest(target/n),bytes=(target/n).stat().st_size) for n in
                          ['model.onnx','fixtures.json','preprocessing.json','report.json']}))
        reports.append(report)
    final = dict(status='passed',trained_classifier=False,backward=checks,tiny_train=history,exports=reports,
                 tiny_train_provenance=dict(split='train',selection='First source-ordered train sample per class',
                                            sample_ids=tiny_sample_ids,train_tensor_sha256=train_hash,
                                            maximum_steps=200,criterion='accuracy >= 0.9',checkpoint_saved=False),
                 d1_verification_sha256=digest(ROOT/'artifacts/drawai/data/verification.json'),
                 runtime=dict(executable=sys.executable,dll_search_directory=os.environ.get('DUST_DLL_DIRECTORY'),
                              import_status='passed',providers=ort.get_available_providers(),device='CPU'),
                 versions={p:importlib.metadata.version(p) for p in ['torch','numpy','onnx','onnxruntime','unlzw3']},
                 python=platform.python_version(),platform=platform.platform(),torch_threads=4)
    write_json(out/'python-report.json',final)
    print(json.dumps(dict(status='passed',tiny_train=history[-1],
                         exports=[{k:r[k] for k in ['length','model_bytes','fixture_count','max_abs_error','python_ort_p95_ms']} for r in reports]),indent=2))


if __name__ == '__main__':
    main()
