"""P2 full-architecture smoke/export. Not a pretrained or released classifier."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

_dll=None
if os.name=='nt':
    p=Path(os.environ.get('DUST_DLL_DIRECTORY','C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'))
    if p.is_dir(): _dll=os.add_dll_directory(str(p))

import numpy as np
import torch
from PIL import Image
import onnx
import onnxruntime as ort
import psutil
from .models import build_model
from .prepare_data import ROOT,write_json,digest
from .raster import render_rgb,normalize_rgb,contract

NAME='drawai_dinov3-vits16_architecture-smoke_rgb224_fp32_v0.onnx'
PRETRAINED_NAME='drawai_dinov3-vits16-timm-qkvb_pretrained-backbone_rgb224_fp32_p2.onnx'


def fixture_drawings():
    rows=[]
    for digit in range(10):
        p=ROOT/f'web/public/samples/drawai/digit-{digit}.json'
        value=json.loads(p.read_text(encoding='utf-8'))
        if value['source']['split']!='train': raise ValueError('Only train demos allowed in P2')
        rows.append(dict(name=f'train-digit-{digit}',strokes=value['strokes'],y_axis='down',
                         reference_label=digit,source_sample_id=value['source']['sample_id']))
    geometries=dict(horizontal=[[dict(x=0,y=0),dict(x=10,y=0)]],
        vertical=[[dict(x=0,y=0),dict(x=0,y=10)]],
        diagonal=[[dict(x=0,y=0),dict(x=10,y=10)]],
        gap=[[dict(x=0,y=0),dict(x=0,y=10)],[dict(x=10,y=0),dict(x=10,y=10)]],
        singleton=[[dict(x=0,y=0),dict(x=10,y=10)],[dict(x=10,y=0)]],
        duplicate=[[dict(x=0,y=0),dict(x=0,y=0),dict(x=3,y=10)]])
    rows += [dict(name=name,strokes=strokes,y_axis='down') for name,strokes in geometries.items()]
    base=rows[6]['strokes']
    for name,transform in [('translated',lambda p:dict(x=p['x']+31,y=p['y']-22)),
                           ('scaled',lambda p:dict(x=p['x']*2,y=p['y']*2)),
                           ('y-up',lambda p:dict(x=p['x'],y=-p['y']))]:
        rows.append(dict(name=name,strokes=[[transform(p) for p in stroke] for stroke in base],y_axis='up' if name=='y-up' else 'down'))
    return rows


def tiny_head_overfit(model,inputs):
    """Train-only ten-example probe; no private/test data, no release mutation."""
    model.train()
    assert not model.backbone.training
    model.zero_grad(set_to_none=True)
    labels=torch.arange(10)
    loss=torch.nn.functional.cross_entropy(model(inputs),labels)
    loss.backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    assert all(p.grad is None for p in model.backbone.parameters())
    with torch.no_grad(): features=model.forward_features(inputs).detach()
    # Explicit train-only feature conditioning for the technical ten-example test.
    center=features.mean(0);scale=features.std(0).clamp_min(1e-3)
    x=(features-center)/scale
    probe=copy.deepcopy(model.head)
    with torch.no_grad(): initial_loss=float(torch.nn.functional.cross_entropy(probe(x),labels))
    optimizer=torch.optim.Adam(probe.parameters(),lr=0.03)
    history=[]
    for step in range(500):
        optimizer.zero_grad(set_to_none=True)
        logits=probe(x);value=torch.nn.functional.cross_entropy(logits,labels)
        value.backward();optimizer.step()
        with torch.no_grad():
            logits=probe(x);accuracy=float((logits.argmax(1)==labels).float().mean())
            updated=float(torch.nn.functional.cross_entropy(logits,labels))
        history.append(dict(step=step+1,loss=updated,train_accuracy=accuracy))
        if accuracy==1 and updated<0.03: break
    assert history[-1]['train_accuracy']==1 and history[-1]['loss']<initial_loss,history[-1]
    model.zero_grad(set_to_none=True);model.eval()
    return dict(status='passed',examples=10,source='one PenDigits train demo per class',
                pretrained_loaded=model.provenance['pretrained_loaded'],test_opened=False,
                feature_conditioning='mean/std from these ten training examples only; min scale 0.001',
                exported_head_modified=False,initial_loss=initial_loss,history=history,backbone_gradients_absent=True,
                note='Tiny probe overfit is an optimization check, not recognition accuracy or a trained release')


def prepare(out,weights_dir=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model=build_model('dinov3_vits16',weights_dir=weights_dir,smoke_only=weights_dir is None).eval()
    total=sum(p.numel() for p in model.parameters())
    is_timm=model.provenance.get('library')=='timm'
    expected_backbone=21596160 if is_timm else 21596544
    assert sum(p.numel() for p in model.backbone.parameters())==expected_backbone
    assert total==expected_backbone+7690
    fixtures=fixture_drawings()
    tensors=[];previews=[]
    for fixture in fixtures:
        rgb=render_rgb(fixture['strokes'],fixture['y_axis']);tensor=normalize_rgb(rgb)
        fixture['rgb_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
        fixture['tensor_sha256']=hashlib.sha256(tensor.tobytes()).hexdigest()
        fixture['tensor_shape']=[1,3,224,224]
        tensors.append(tensor);previews.append(Image.fromarray(rgb))
    pixels=torch.from_numpy(np.stack(tensors))
    tiny=tiny_head_overfit(model,pixels[:10])
    write_json(out/'tiny-overfit.json',tiny)
    board=Image.new('RGB',(224*5,224*2),'white')
    for i,image in enumerate(previews[:10]): board.paste(image,((i%5)*224,(i//5)*224))
    board.save(out/'train-digits-rendered.png')
    with torch.inference_mode(): expected=model(pixels).numpy()
    name=PRETRAINED_NAME if is_timm else NAME
    path=out/name
    torch.onnx.export(model,pixels[:1],str(path),input_names=['pixel_values'],output_names=['logits'],
                      opset_version=17,dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(path)),full_check=True)
    options=ort.SessionOptions();options.intra_op_num_threads=4
    start=time.perf_counter()
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    init_ms=(time.perf_counter()-start)*1000
    actual=np.stack([session.run(None,{'pixel_values':row.numpy()[None]})[0][0] for row in pixels])
    np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
    assert np.array_equal(actual.argmax(1),expected.argmax(1))
    for i,fixture in enumerate(fixtures): fixture['expected_logits']=expected[i].tolist()
    invalid=[dict(name='empty',strokes=[],code='EMPTY'),dict(name='empty-stroke',strokes=[[]],code='EMPTY_STROKE'),
             dict(name='tap',strokes=[[dict(x=1,y=1)]],code='TOO_SHORT'),
             dict(name='isolated',strokes=[[dict(x=0,y=0)],[dict(x=1,y=1)]],code='TOO_SHORT'),
             dict(name='boolean',strokes=[[dict(x=True,y=0),dict(x=1,y=1)]],code='NUMBER'),
             dict(name='string',strokes=[[dict(x='0',y=0),dict(x=1,y=1)]],code='NUMBER'),
             dict(name='large',strokes=[[dict(x=0,y=0),dict(x=1000001,y=1)]],code='NUMBER'),
             dict(name='strokes',strokes=[[dict(x=0,y=0)] for _ in range(17)],code='STROKES')]
    write_json(out/'fixtures.json',dict(cases=fixtures,invalid=invalid,atol=1e-4,rtol=1e-3))
    write_json(out/'preprocessing.json',contract())
    for _ in range(5): session.run(None,{'pixel_values':pixels[:1].numpy()})
    timings=[]
    for _ in range(30):
        start=time.perf_counter();session.run(None,{'pixel_values':pixels[:1].numpy()});timings.append((time.perf_counter()-start)*1000)
    report=dict(status='passed',scope='technical feasibility only',trained=False,release_ready=False,
        smoke_only=True,pretrained_loaded=model.provenance['pretrained_loaded'],provenance=model.provenance,
        parameters=total,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_backbone=True,pooling='CLS concatenated with mean patch tokens; register tokens excluded',
        input_shape=[1,3,224,224],output_shape=[1,10],opset=17,full_backbone_exported=True,
        fixture_count=len(fixtures),argmax_disagreements=int(np.count_nonzero(actual.argmax(1)!=expected.argmax(1))),
        max_abs_error=float(np.abs(actual-expected).max()),atol=1e-4,rtol=1e-3,
        model_bytes=path.stat().st_size,model_sha256=digest(path),python_init_ms=init_ms,
        python_median_ms=float(np.median(timings)),python_p95_ms=sorted(timings)[math.ceil(.95*len(timings))-1],
        python_threads=4,iterations=30,warmup=5,process_rss_bytes=psutil.Process().memory_info().rss,
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),
            transformers=importlib.metadata.version('transformers'),onnx=onnx.__version__,onnxruntime=ort.__version__),
        test_or_private_data_opened=False)
    write_json(out/'python-report.json',report)
    write_json(out/'manifest.json',dict(version='drawai-dinov3-p2-technical-v1',trained=False,smoke_only=True,
        pretrained_loaded=model.provenance['pretrained_loaded'],release_ready=False,model_filename=name,
        files={name:dict(bytes=(out/name).stat().st_size,sha256=digest(out/name))
               for name in [name,'fixtures.json','preprocessing.json','python-report.json']}))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/drawai/phase2/technical')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--weights-dir',type=Path,default=ROOT/'models/pretrained/dinov3-vits16-timm-qkvb',help='Pinned public timm or authorized Meta HF snapshot; defaults to the verified timm source')
    modes.add_argument('--architecture-smoke',action='store_true',help='Explicit random initialization, never a pretrained fallback')
    args=parser.parse_args();prepare(args.output_dir,None if args.architecture_smoke else args.weights_dir)
