"""P6 full FP32 ONNX export of validation-selected P5 checkpoints; no retuning."""
import argparse
import base64
from datetime import datetime,timezone
import io
from pathlib import Path
import re
import tempfile
import time

from .native_runtime import configure_windows_dlls
configure_windows_dlls()
import numpy as np
from PIL import Image
import torch
import onnx
import onnxruntime as ort

from . import paint_train as train,paint_data as paint
from .metrics import classification_metrics
from .private_data import ROOT,canonical,sha,load,dump
from .private_dataset import separate_output,safe_file
from .raster import normalize_rgb
from .paint_package import spec,verify_bundle,TOLERANCES

DEFAULT_OUTPUT=ROOT/'models/exported/paint35-v1'
DEFAULT_REPORT=ROOT/'artifacts/drawai/phase6-image64'


def encode_png(pixels):
    buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG');return base64.b64encode(buffer.getvalue()).decode()


def fixtures():
    """Procedural images ONLY; never publish held-out or private source images."""
    rows=[];rng=np.random.default_rng(6006)
    for name in ['white','black','single-pixel','vertical','horizontal','diagonal','border','checker','noise']:
        pixels=np.full((64,64),255,dtype=np.uint8)
        if name=='black':pixels[:]=0
        if name=='single-pixel':pixels[32,32]=0
        if name=='vertical':pixels[8:56,30:34]=0
        if name=='horizontal':pixels[30:34,8:56]=0
        if name=='diagonal':np.fill_diagonal(pixels,0)
        if name=='border':pixels[[0,-1],:]=0;pixels[:,[0,-1]]=0
        if name=='checker':pixels=((np.indices((64,64)).sum(0)%2)*255).astype(np.uint8)
        if name=='noise':pixels=rng.integers(0,256,(64,64),dtype=np.uint8)
        rows.append((name,pixels))
    return rows


def tensor(pixels,family):
    return (paint.swin_rows(pixels) if family=='swin1d-image' else normalize_rgb(paint.dino_rgb(pixels)))[None]


def check_parity(expected,actual,labels,family):
    if expected.shape!=actual.shape or expected.ndim!=2 or expected.shape[1]!=10 or not len(expected):raise ValueError('Invalid logits shape')
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():raise ValueError('Nonfinite export logits')
    tolerance=TOLERANCES[family]
    if not np.allclose(actual,expected,**tolerance):raise ValueError('Full-logit tolerance failed')
    disagreements=int(np.count_nonzero(expected.argmax(1)!=actual.argmax(1)))
    if disagreements:raise ValueError('Argmax changed after export')
    value=dict(samples=len(expected),max_abs_error=float(np.max(np.abs(expected-actual))),argmax_disagreements=disagreements,**tolerance)
    if labels is not None:
        py=classification_metrics(labels,expected.argmax(1));ox=classification_metrics(labels,actual.argmax(1))
        if py!=ox:raise ValueError('Export metrics changed')
        value.update(pytorch_metrics=py,onnx_metrics=ox,accuracy_degradation=0.,macro_f1_degradation=0.)
    return value


def create_session(path,family):
    info=spec(family);options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    if len(session.get_inputs())!=1 or len(session.get_outputs())!=1:raise ValueError('Wrong ONNX input/output count')
    a,b=session.get_inputs()[0],session.get_outputs()[0]
    if (a.name,a.shape,a.type,b.name,b.shape,b.type)!=(info['input_name'],info['input_shape'],'tensor(float)',info['output_name'],[1,10],'tensor(float)'):
        raise ValueError('Wrong ONNX tensor contract')
    return session


def export_all(dataset,experiment,weights,output,report_dir):
    output=separate_output(output,dataset);report_dir=Path(report_dir).resolve();train.private_output(report_dir)
    if output==report_dir or output in report_dir.parents or report_dir in output.parents:raise ValueError('Bundle and report must be separate')
    selection,runs=train.checked_selection(experiment);final=load(Path(experiment)/'evaluation/final-test.json')
    handoff=load(Path(experiment)/'handoff.json');p5=load(Path(experiment)/'verification.json')
    if final['selection_sha256']!=selection['selection_sha256'] or handoff['selection_sha256']!=selection['selection_sha256'] or not p5['phase_complete']:
        raise ValueError('P5 verification/selection missing')
    if handoff['evaluation_sha256']!=final['evaluation_sha256'] or final['evaluation_sha256']!=sha(canonical({k:v for k,v in final.items() if k!='evaluation_sha256'})):
        raise ValueError('P5 evaluation changed')
    train.seed_all(42);pixels,labels,rows=train.test_images(dataset,experiment)
    output.parent.mkdir(parents=True,exist_ok=True);staging=Path(tempfile.mkdtemp(prefix=output.name+'.staging-',dir=output.parent))
    report_dir.mkdir(parents=True,exist_ok=True);(report_dir/'evaluation').mkdir(exist_ok=True)
    registry=dict(version='drawai-paint-model-registry-v1',models={});reports={}
    for family in train.FAMILIES:
        winner=selection['winners'][family];run=next(r for r in runs if r['family']==family and r['seed']==winner['seed'])
        folder=Path(experiment)/'runs'/f"{family}-seed{run['seed']}"
        if handoff['models'][family]['checkpoint_sha256']!=run['checkpoint_sha256']:raise ValueError('Handoff checkpoint changed')
        model=train.load_fitted(folder,run,weights);info=spec(family);target=staging/family;target.mkdir()
        path=target/info['model_filename'];onnx_path=target/info['diagnostic_onnx_filename']
        torch.onnx.export(model,torch.zeros(info['input_shape']),str(onnx_path),input_names=[info['input_name']],
            output_names=['logits'],opset_version=17,dynamo=False,do_constant_folding=True,external_data=False)
        graph=onnx.load(str(onnx_path),load_external_data=False);onnx.checker.check_model(graph,full_check=True)
        if any(i.data_location==onnx.TensorProto.EXTERNAL for i in graph.graph.initializer):raise ValueError('External tensor data not supported')
        elements=sum(int(np.prod(i.dims)) for i in graph.graph.initializer)
        if elements<(20_000_000 if family=='dinov3-vits16' else 500_000):raise ValueError('Expected full neural network, not head-only export')
        session=create_session(onnx_path,family);expected=[];actual=[];onnx_actual=[];started=time.perf_counter();scripted=None
        if family=='dinov3-vits16':
            with torch.inference_mode(),torch.jit.optimized_execution(False):
                traced=torch.jit.trace(model,torch.zeros(info['input_shape']),strict=True,check_trace=True)
                torch.jit.save(traced,str(path))
                scripted=torch.jit.load(str(path),map_location='cpu').eval()
        with torch.inference_mode(),torch.jit.optimized_execution(False):
            for index,image in enumerate(pixels):
                values=tensor(image,family);expected.append(model(torch.from_numpy(values)).numpy()[0])
                ox=session.run(None,{info['input_name']:values})[0][0];onnx_actual.append(ox)
                actual.append(ox if scripted is None else scripted(torch.from_numpy(values)).numpy()[0])
                if (index+1)%200==0:print(f'{family} full-test parity {index+1}/1000',flush=True)
        expected=np.asarray(expected);actual=np.asarray(actual);onnx_actual=np.asarray(onnx_actual);parity=check_parity(expected,actual,labels,family)
        tolerance=TOLERANCES[family];failed=np.any(np.abs(onnx_actual-expected)>tolerance['atol']+tolerance['rtol']*np.abs(expected),axis=1)
        diagnostic=dict(numeric_parity_passed=not bool(failed.any()),failed_logit_rows=int(failed.sum()),
            max_abs_error=float(np.max(np.abs(onnx_actual-expected))),argmax_disagreements=int(np.count_nonzero(onnx_actual.argmax(1)!=expected.argmax(1))),
            approved_runtime=scripted is None,model_filename=info['diagnostic_onnx_filename'],**tolerance)
        prior_result=next(r for r in final['results'] if r['family']==family and r['seed']==run['seed'])
        prior_path=Path(experiment)/'evaluation'/prior_result['predictions_file']
        if train.file_sha(prior_path)!=prior_result['predictions_sha256']:raise ValueError('P5 prediction hash changed')
        with np.load(prior_path,allow_pickle=False) as prior:
            if prior['sample_ids'].tolist()!=[r['sample_id'] for r in rows] or not np.array_equal(prior['labels'],labels):raise ValueError('P5/P6 sample mismatch')
            check_parity(prior['logits'],expected,labels,family)
        if parity['onnx_metrics']!={k:v for k,v in prior_result['metrics'].items() if k!='loss'}:raise ValueError('P5 test metrics not reproduced')
        np.savez_compressed(report_dir/'evaluation'/f'{family}-full-test-parity.npz',sample_ids=np.asarray([r['sample_id'] for r in rows]),labels=labels,pytorch_logits=expected,artifact_logits=actual,onnx_logits=onnx_actual)
        cases=[]
        with torch.inference_mode(),torch.jit.optimized_execution(False):
            for name,image in fixtures():
                values=tensor(image,family);py=model(torch.from_numpy(values)).numpy()
                ox=session.run(None,{info['input_name']:values})[0] if scripted is None else scripted(torch.from_numpy(values)).numpy()
                check_parity(py,ox,None,family)
                cases.append(dict(name=name,png64_base64=encode_png(image),expected_logits=py[0].tolist()))
        dump(target/'fixtures.json',dict(version='drawai-paint-procedural-fixtures-v1',source='Procedural pixels only, no private/test images',cases=cases,**TOLERANCES[family]))
        dump(target/'labels.json',dict(classes=[dict(id=i,name=str(i)) for i in range(10)]));dump(target/'preprocessing.json',paint.contract())
        report=dict(version='drawai-paint-export-report-v1',family=family,spec=info,trained=True,artifact_verified=True,release_ready=False,
            seed=run['seed'],best_epoch=run['best']['epoch'],checkpoint_sha256=run['checkpoint_sha256'],
            selection_sha256=selection['selection_sha256'],lock_sha256=train.LOCK_HASH,plan_sha256=selection['plan_sha256'],
            model_sha256=train.file_sha(path),model_bytes=path.stat().st_size,opset=17,full_backbone_exported=family=='dinov3-vits16',
            graph_nodes=len(graph.graph.node),initializer_elements=elements,parameters=run['parameters'],
            parity=parity,onnx_diagnostic=diagnostic,p5_test_argmax_reproduced=True,procedural_fixtures=len(cases),full_test_seconds=time.perf_counter()-started,
            runtime=dict(torch=str(torch.__version__),onnx=onnx.__version__,onnxruntime=ort.__version__,provider='CPUExecutionProvider' if scripted is None else 'TorchScript CPU (optimized_execution=False)',threads=4),
            similarity_independence_verified=False,main_cross_split_near_groups=37,product_default_changed=False)
        dump(target/'report.json',report)
        (target/'model-card.md').write_text(f'''# DrawAI {family} - Paint35 FP32 {info["runtime_format"]}

Trained checkpoint seed{run['seed']}, epoch{run['best']['epoch']}, selected using validation before test.
Input {info['input_name']} float32 {info['input_shape']}; output logits float32[1,10], classes0-9.
Preprocessing is external: see preprocessing.json. Source uint8 grayscale64 with white background/black ink.
Swin uses spatial rows, not writing trajectories. DINOv3 includes the full pretrained backbone and fitted head.
Test1000: accuracy {parity['onnx_metrics']['accuracy']:.4f}, macro-F1 {parity['onnx_metrics']['macro_f1']:.6f}; same argmax as P5.
37 near-groups across main splits remain. No leakage-free/generalization claim; owner-attested Paint origin/identities/permissions.
Snapshot split is retrospective; original source already public. No device metadata or pen trajectories.
FP32 artifact format: {info["runtime_format"]}; verified full CPU parity. ONNX diagnostic status is in report.json. No canvas integration in P6.
trained=true, artifact_verified=true, release_ready=false until P7 product/domain/runtime acceptance.
Do not use the P2 untrained-head artifact in place of this model. Do not tune from final test results.
Checkpoint SHA256: {run['checkpoint_sha256']}. Artifact SHA256: {report['model_sha256']}.
''',encoding='utf-8')
        if family=='dinov3-vits16':
            (target/'LICENSE-DINOv3.md').write_bytes((Path(weights)/'LICENSE.md').read_bytes())
            dump(target/'pretrained-provenance.json',run['provenance'])
        files={p.name:dict(bytes=p.stat().st_size,sha256=train.file_sha(p)) for p in sorted(target.iterdir()) if p.is_file()}
        release=dict(version='drawai-paint-model-release-v1',spec=info,trained=True,artifact_verified=True,release_ready=False,
            lock_sha256=train.LOCK_HASH,selection_sha256=selection['selection_sha256'],checkpoint_sha256=run['checkpoint_sha256'],
            files=files,test_accuracy=parity['onnx_metrics']['accuracy'],test_macro_f1=parity['onnx_metrics']['macro_f1'])
        release['release_sha256']=sha(canonical(release));dump(target/'release.json',release)
        registry['models'][family]=dict(release_directory=family,release_sha256=release['release_sha256'],spec=info)
        reports[family]=report;dump(report_dir/f'export-{family}.json',report)
        del model,session,graph
    dump(staging/'registry.json',registry)
    files={p.relative_to(staging).as_posix():dict(bytes=p.stat().st_size,sha256=train.file_sha(p)) for p in sorted(staging.rglob('*')) if p.is_file()}
    record=dict(version='drawai-paint-onnx-bundle-v1',trained=True,artifact_verified=True,release_ready=False,product_default_changed=False,
        created_at=datetime.now(timezone.utc).isoformat(),selection_sha256=selection['selection_sha256'],lock_sha256=train.LOCK_HASH,
        files=files,code_sha256={name:train.file_sha(ROOT/name) for name in ['training/drawai/paint_export.py','training/drawai/paint_package.py','training/drawai/native_runtime.py','training/drawai/paint_data.py','training/drawai/dinov3_model.py','training/drawai/models.py']})
    record['bundle_sha256']=sha(canonical(record));dump(staging/'bundle-manifest.json',record)
    verify_bundle(staging,record['bundle_sha256']);staging.rename(output)
    dump(report_dir/'trusted-bundle.json',dict(bundle_sha256=record['bundle_sha256'],bundle_path=output.relative_to(ROOT).as_posix(),note='External integrity anchor; not a signature'))
    print('P6 full ONNX bundle verified: '+record['bundle_sha256'],flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['export','verify'])
    parser.add_argument('--dataset',type=Path,default=train.DEFAULT_DATA);parser.add_argument('--experiment',type=Path,default=train.DEFAULT_OUT)
    parser.add_argument('--weights',type=Path,default=train.DEFAULT_WEIGHTS);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--report-dir',type=Path,default=DEFAULT_REPORT);parser.add_argument('--expected-bundle-sha256')
    args=parser.parse_args()
    if args.command=='export':export_all(args.dataset,args.experiment,args.weights,args.output,args.report_dir)
    else:verify_bundle(args.output,args.expected_bundle_sha256);print('Bundle integrity verified')


if __name__=='__main__':main()
