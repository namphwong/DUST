"""Export the verified Kaggle validation winner; never modify trained weights."""
import hashlib
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
import onnx
import onnxruntime as ort
import torch
from torch.torch_version import TorchVersion
from har_models import build_model
from har_metrics import classification_metrics
from prepare_uci_har import ROOT, content_hash, write_json
from train_har import load_data, validate_data

VERSION = 'har-swin-kaggle-v1'
ATOL, RTOL = 1e-5, 1e-4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    origin = ROOT / 'artifacts/phase3/kaggle'
    verified = json.loads((origin / 'verification.json').read_text(encoding='utf-8'))
    evaluation = json.loads((origin / 'artifacts/phase3/evaluation/evaluation.json').read_text(encoding='utf-8'))
    selection = evaluation['selection']
    run = selection['deployment_run']
    if verified['status'] != 'passed' or not verified['kaggle_gpu_verified'] or run != verified['deployment_run']:
        raise ValueError('Need verified Kaggle deployment selection')
    candidate = next(c for c in selection['candidates'] if c['run'] == run)
    checkpoint_path = origin / f'artifacts/phase3/runs/{run}/best.pth'
    if digest(checkpoint_path) != candidate['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since validation selection')
    # Trusted, hash-verified artifact from our own run; allow only installed TorchVersion.
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    train, val = validate_data(ROOT / 'data/processed/uci-har-3ch-v1')
    if not checkpoint['trained'] or checkpoint['config']['model'] != 'swin' or checkpoint['data_hash'] != content_hash(*train, *val):
        raise ValueError('Wrong checkpoint family/training dataset')
    data_dir = ROOT / 'data/processed/uci-har-3ch-v1'
    for name in ['preprocessing', 'labels']:
        if checkpoint[name] != json.loads((data_dir / f'{name}.json').read_text(encoding='utf-8')):
            raise ValueError('Checkpoint metadata differs from Phase 2')
    torch.set_num_threads(4)
    model = build_model(checkpoint['config']).eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    target = ROOT / 'web/public/models' / VERSION
    target.mkdir(parents=True, exist_ok=True)
    path = target / 'model.onnx'
    torch.onnx.export(model, torch.zeros(1, 3, 128), str(path),
                      input_names=['signals'], output_names=['logits'], opset_version=17,
                      do_constant_folding=True, dynamo=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape == [1, 3, 128]
    assert session.get_outputs()[0].shape == [1, 6]
    X, y, subjects, indices = load_data(data_dir, 'test')
    reference_result = next(r for r in evaluation['results'] if r['run'] == run)
    if content_hash(X, y, subjects, indices) != reference_result['test_sha256']:
        raise ValueError('Evaluation data differs from the verified Kaggle test')
    with np.load(origin / f'artifacts/phase3/evaluation/{run}-predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['y_true'], y)
        np.testing.assert_array_equal(saved['subject_id'], subjects)
        np.testing.assert_array_equal(saved['source_index'], indices)
        kaggle_predictions = saved['y_pred'].copy()
    expected, actual = [], []
    started = time.perf_counter()
    for i, window in enumerate(X):
        with torch.inference_mode():
            golden = model(torch.from_numpy(window[None])).numpy()[0]
        observed = session.run(['logits'], {'signals': window[None]})[0][0]
        if not np.isfinite(golden).all() or not np.isfinite(observed).all():
            raise ValueError('Non-finite evaluation logits')
        np.testing.assert_allclose(observed, golden, atol=ATOL, rtol=RTOL)
        expected.append(golden)
        actual.append(observed)
        if (i + 1) % 500 == 0:
            print(f'Parity: {i+1}/{len(X)}', flush=True)
    expected, actual = np.asarray(expected), np.asarray(actual)
    pt_metrics = classification_metrics(y, expected.argmax(1))
    ort_metrics = classification_metrics(y, actual.argmax(1))
    reference = reference_result['test']
    delta = reference['macro_f1'] - ort_metrics['macro_f1']
    if delta > 0.005:
        raise ValueError('ONNX macro-F1 degradation exceeds 0.005')
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2*np.pi, 128, dtype=np.float32)
    raw_cases = [('zeros', np.zeros((3, 128), np.float32)), ('ones', np.ones((3, 128), np.float32)),
                 ('sine', np.stack([np.sin(t), np.cos(t), np.sin(2*t)])),
                 ('impulse', np.pad(np.ones((3, 1), np.float32), ((0, 0), (63, 64)))),
                 *[(f'random-{i}', rng.normal(size=(3, 128)).astype(np.float32)) for i in range(4)]]
    prep = checkpoint['preprocessing']
    mean = np.asarray(prep['mean'], dtype=np.float32)[:, None]
    std = np.asarray(prep['std'], dtype=np.float32)[:, None]
    cases = []
    for name, raw in raw_cases:
        normalized = ((raw - mean) / std).astype(np.float32)
        with torch.inference_mode():
            golden = model(torch.from_numpy(normalized[None])).numpy()[0]
        observed = session.run(['logits'], {'signals': normalized[None]})[0][0]
        if not np.isfinite(golden).all() or not np.isfinite(observed).all():
            raise ValueError('Non-finite fixture logits')
        np.testing.assert_allclose(observed, golden, atol=ATOL, rtol=RTOL)
        cases.append(dict(name=name, kind='synthetic', raw=raw.reshape(-1).tolist(),
                          input=normalized.reshape(-1).tolist(), expected=golden.tolist()))
    with np.load(data_dir / 'test.npz', allow_pickle=False) as data:
        for label in range(6):
            index = int(np.flatnonzero(y == label)[0])
            raw = data['X_raw'][index]
            cases.append(dict(name=f'uci-{checkpoint["labels"]["classes"][label]["code"]}', kind='uci-reference',
                              raw=raw.reshape(-1).tolist(), input=X[index].reshape(-1).tolist(),
                              expected=expected[index].tolist(), reference_label=int(label),
                              subject_id=int(subjects[index]), source_index=int(indices[index]),
                              selection='First test window of each class; independent of prediction'))
    write_json(target / 'fixtures.json', dict(version=VERSION, atol=ATOL, rtol=RTOL, cases=cases))
    write_json(target / 'preprocessing.json', prep)
    write_json(target / 'labels.json', checkpoint['labels'])
    report = dict(status='passed', version=VERSION, trained=True, deployment_run=run,
                  checkpoint_sha256=digest(checkpoint_path), sha256=digest(path), model_bytes=path.stat().st_size,
                  config=checkpoint['config'], input_shape=[1, 3, 128], output_shape=[1, 6],
                  input_name='signals', output_name='logits', input_dtype='float32', opset=17,
                  input_is_normalized=True, normalization_applied_by='caller exactly once',
                  quantized=False, weights_modified=False, atol=ATOL, rtol=RTOL, fixture_count=len(cases),
                  evaluation_samples=len(y), evaluation_data_hash=content_hash(X, y, subjects, indices),
                  full_test_max_abs_error=float(np.max(np.abs(actual-expected))),
                  pytorch_onnx_prediction_agreement=int(np.sum(expected.argmax(1) == actual.argmax(1))),
                  kaggle_onnx_prediction_agreement=int(np.sum(kaggle_predictions == actual.argmax(1))),
                  pytorch_test=pt_metrics, onnx_test=ort_metrics,
                  kaggle_test_macro_f1=reference['macro_f1'], macro_f1_degradation=delta,
                  max_allowed_degradation=0.005, target_macro_f1=0.9,
                  target_met=ort_metrics['macro_f1'] >= 0.9,
                  evaluation_seconds=time.perf_counter()-started,
                  intended_use='Limited UCI HAR research demo, not guaranteed real-world accuracy',
                  versions=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                                onnx=onnx.__version__, onnxruntime=ort.__version__),
                  exporter='Pinned torch 2.5.1 legacy fixed-shape exporter; no dynamic batch or sequence')
    write_json(target / 'report.json', report)
    files = ['model.onnx', 'fixtures.json', 'preprocessing.json', 'labels.json', 'report.json']
    write_json(target / 'release.json', dict(version=VERSION, trained=True,
               files={name: dict(sha256=digest(target / name), bytes=(target / name).stat().st_size) for name in files}))
    write_json(ROOT / 'artifacts/phase4/export-verification.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'model_bytes', 'sha256', 'evaluation_samples',
          'full_test_max_abs_error', 'pytorch_onnx_prediction_agreement', 'kaggle_onnx_prediction_agreement',
          'macro_f1_degradation']}, indent=2))


if __name__ == '__main__':
    main()
