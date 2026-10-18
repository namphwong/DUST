"""D4: export the validation-locked trained Swin and audit every official test sample."""
import argparse
import json
import os
import platform
import sys
import time

_dll_handle = None
if sys.platform == 'win32' and os.environ.get('DUST_DLL_DIRECTORY'):
    _dll_handle = os.add_dll_directory(os.environ['DUST_DLL_DIRECTORY'])

import numpy as np
import torch
import onnx
import onnxruntime as ort

from .models import build_model, swin_config
from .prepare_data import ROOT, digest, write_json, parse_unipen
from .preprocessing import contract, preprocess
from .train import load_split, tensor_hash
from .metrics import classification_metrics
from .verify_training import verify_selected_swin
from .verify_feasibility import geometry_cases

VERSION = 'drawai-swin-v1'
ATOL, RTOL = 1e-5, 1e-4


def export(experiment, release_dir=None, report_dir=None, samples_dir=None):
    experiment = experiment.resolve()
    verify_selected_swin(experiment)
    selection = json.loads((experiment/'selection.json').read_text())
    evaluation = json.loads((experiment/'evaluation/evaluation.json').read_text())
    run = next(r for r in selection['runs'] if r['run_id'] == selection['deployment_run'])
    source = experiment/'runs'/run['run_id']/'best.pth'
    if digest(source) != selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash mismatch')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    config = saved['config']; length = config['length']
    data_dir = ROOT/f'data/processed/pendigits-drawai-v1/l{length}'
    labels = json.loads((data_dir/'labels.json').read_text())
    if (saved['trained'] is not True or saved['test_used_for_training'] or
        config != run['config'] or config['model'] != 'swin' or
        saved['preprocessing'] != contract(length) or saved['labels'] != labels or
        saved['model_kwargs'] != swin_config(length)):
        raise ValueError('Checkpoint contract differs from selected trained Swin')
    for split in ['train', 'validation']:
        if tensor_hash(*load_split(data_dir, split, length)) != saved['data_hashes'][split]:
            raise ValueError('Training provenance mismatch')
    torch.set_num_threads(4)
    model = build_model('swin', length).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    target = release_dir or ROOT/f'web/public/models/{VERSION}'
    out = report_dir or ROOT/'artifacts/drawai/export'
    target.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    path = target/'model.onnx'
    torch.onnx.export(model, torch.zeros(1,3,length), str(path), input_names=['drawing'],
                      output_names=['logits'], opset_version=17, dynamo=False)
    graph = onnx.load(str(path)); onnx.checker.check_model(graph, full_check=True)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    if (session.get_inputs()[0].shape != [1,3,length] or session.get_outputs()[0].shape != [1,10] or
        session.get_inputs()[0].type != 'tensor(float)' or session.get_outputs()[0].type != 'tensor(float)'):
        raise ValueError('Export tensor contract mismatch')
    test = load_split(data_dir, 'test', length)
    expected, actual = [], []
    with torch.inference_mode():
        for i, row in enumerate(test[0]):
            values = row[None]
            expected.append(model(torch.from_numpy(values)).numpy()[0])
            actual.append(session.run(None, {'drawing': values})[0][0])
            if (i+1) % 500 == 0: print(f'Full-test parity {i+1}/{len(test[0])}', flush=True)
    expected = np.asarray(expected); actual = np.asarray(actual)
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite inference')
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    disagreements = int(np.count_nonzero(actual.argmax(1) != expected.argmax(1)))
    pytorch_metrics = classification_metrics(test[1], expected.argmax(1))
    onnx_metrics = classification_metrics(test[1], actual.argmax(1))
    d3 = next(r for r in evaluation['results'] if r['run_id'] == run['run_id'])
    with np.load(experiment/'evaluation'/f"{run['run_id']}-predictions.npz", allow_pickle=False) as prior:
        np.testing.assert_array_equal(prior['sample_ids'], test[2])
        np.testing.assert_array_equal(prior['y_true'], test[1])
        np.testing.assert_array_equal(prior['y_pred'], expected.argmax(1))
    if pytorch_metrics['macro_f1'] != d3['test']['macro_f1']:
        raise ValueError('D3 test metrics not reproduced')
    degradation = pytorch_metrics['macro_f1']-onnx_metrics['macro_f1']
    if disagreements or degradation > .001:
        raise ValueError('Full-test argmax/metric parity failed')
    np.savez_compressed(out/'full-test-parity.npz', pytorch_logits=expected, onnx_logits=actual,
                        y_true=test[1], sample_ids=test[2])
    cases = geometry_cases(length)
    rng = np.random.default_rng(42)
    for name, values in [('zeros',np.zeros(3*length,np.float32)), ('ones',np.ones(3*length,np.float32)),
                         ('random',rng.normal(size=3*length).astype(np.float32))]:
        cases.append(dict(name=name, input=values.tolist()))
    for case in cases:
        values = np.asarray(case['input'],np.float32).reshape(1,3,length)
        with torch.inference_mode(): case['expected'] = model(torch.from_numpy(values)).numpy()[0].tolist()
    write_json(target/'fixtures.json',dict(atol=ATOL,rtol=RTOL,cases=cases))
    write_json(target/'preprocessing.json',contract(length)); write_json(target/'labels.json',labels)
    # Select samples by source ID, without consulting model predictions.
    manifest = json.loads((data_dir/'manifest.json').read_text())
    raw = ROOT/'data/raw/pendigits/pendigits-orig.tra'
    if not raw.exists():
        raw = next((ROOT/'data/raw').rglob('pendigits-orig.tra'))
    if digest(raw) != manifest['sources']['pendigits-orig.tra']['uncompressed_sha256']:
        raise ValueError('Demo source hash mismatch')
    samples = {s['id']:s for s in parse_unipen(raw.read_text(),raw.name)}
    train_data = load_split(data_dir,'train',length)
    sample_dir = samples_dir or ROOT/'web/public/samples/drawai'; sample_dir.mkdir(parents=True,exist_ok=True)
    demo_entries = []
    for label in range(10):
        entry = min((e for e in manifest['splits']['train']['entries'] if e['label']==label),key=lambda e:e['id'])
        sample = samples[entry['id']]
        values = preprocess(sample['strokes'],length,'up')
        index = list(train_data[2]).index(entry['id'])
        np.testing.assert_array_equal(values,train_data[0][index])
        # Browser canvas contract has y downward; translate source into a positive viewport.
        xs = [p['x'] for stroke in sample['strokes'] for p in stroke]
        ys = [p['y'] for stroke in sample['strokes'] for p in stroke]
        strokes = [[dict(x=p['x']-min(xs), y=max(ys)-p['y']) for p in stroke] for stroke in sample['strokes']]
        np.testing.assert_array_equal(preprocess(strokes,length,'down'),values)
        filename = f'digit-{label}.json'
        write_json(sample_dir/filename,dict(schema_version='drawai-strokes-v1',
                   canvas=dict(width=max(1,max(xs)-min(xs)),height=max(1,max(ys)-min(ys))),
                   coordinate_system='canvas-y-down',strokes=strokes,reference_label=label,
                   source=dict(kind='dataset',sample_id=entry['id'],split='train')))
        demo_entries.append(dict(label=label,file=filename,sample_id=entry['id'],split='train',
                                 sha256=digest(sample_dir/filename),bytes=(sample_dir/filename).stat().st_size,
                                 provenance=dict(**entry,source_sha256=digest(raw),
                                   selection='Lowest source sample ID in train for each class',
                                   coordinate_transform='x-min(x), max(y)-y',original_y_axis='up')))
    write_json(sample_dir/'manifest.json',dict(version='drawai-train-demos-v1',samples=demo_entries))
    timings=[]; values=test[0][0:1]
    for _ in range(5): session.run(None,{'drawing':values})
    for _ in range(100):
        start=time.perf_counter(); session.run(None,{'drawing':values}); timings.append((time.perf_counter()-start)*1000)
    report=dict(status='passed',trained=True,version=VERSION,length=length,config=swin_config(length),
                input_shape=[1,3,length],output_shape=[1,10],input_name='drawing',output_name='logits',
                opset=17,dtype='float32',graph_checker='passed',fixture_count=len(cases),atol=ATOL,rtol=RTOL,
                model_bytes=path.stat().st_size,sha256=digest(path),parameters=saved['parameters'],
                checkpoint_sha256=digest(source),run_id=run['run_id'],seed=config['seed'],best_epoch=run['best']['epoch'],
                selection_sha256=digest(experiment/'selection.json'),test_sha256=tensor_hash(*test),
                test_samples=len(test[0]),max_abs_error=float(np.max(np.abs(expected-actual))),
                argmax_disagreements=disagreements,macro_f1_degradation=degradation,
                pytorch_metrics=pytorch_metrics,onnx_metrics=onnx_metrics,
                python_ort_median_ms=float(np.median(timings)),python_ort_p95_ms=float(np.percentile(timings,95)),
                benchmark=dict(warmup=5,iterations=100,threads=4,input='First official test sample'),
                runtime=dict(python=platform.python_version(),torch=torch.__version__,onnx=onnx.__version__,
                             onnxruntime=ort.__version__,device='cpu'),
                demo_manifest_sha256=digest(sample_dir/'manifest.json'))
    write_json(target/'report.json',report)
    write_json(target/'release.json',dict(version=VERSION,trained=True,
               files={n:dict(sha256=digest(target/n),bytes=(target/n).stat().st_size) for n in
                      ['model.onnx','fixtures.json','preprocessing.json','labels.json','report.json']}))
    write_json(out/'python-report.json',report)
    print(json.dumps({k:report[k] for k in ['status','test_samples','max_abs_error','argmax_disagreements','macro_f1_degradation','model_bytes']},indent=2))


if __name__ == '__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,default=ROOT/'artifacts/drawai/d3-local-v1')
    parser.add_argument('--release-dir',type=Path,help='Optional candidate export folder; defaults to the deployed release')
    parser.add_argument('--report-dir',type=Path,help='Optional verification report folder')
    parser.add_argument('--samples-dir',type=Path,help='Optional demo output folder')
    args = parser.parse_args()
    export(args.experiment_dir,args.release_dir,args.report_dir,args.samples_dir)
