"""Verify current DrawAI release and save independent Phase 0 evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Match the existing project's documented Windows runtime workaround.
_dll = None
if sys.platform == 'win32':
    dll_dir = Path(os.environ.get('DUST_DLL_DIRECTORY',
        'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'))
    if dll_dir.is_dir():
        _dll = os.add_dll_directory(str(dll_dir))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from training.drawai.models import build_model
from training.drawai.preprocessing import preprocess, contract


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / 'artifacts/drawai/phase0'
    release_dir = ROOT / 'web/public/models/drawai-swin-v1'
    manifest = json.loads((release_dir / 'release.json').read_text())
    assert manifest['trained'] is True
    for name, meta in manifest['files'].items():
        path = release_dir / name
        assert path.stat().st_size == meta['bytes'] and digest(path) == meta['sha256'], name
    onnx.checker.check_model(onnx.load(str(release_dir / 'model.onnx')), full_check=True)
    report = json.loads((release_dir / 'report.json').read_text())
    checkpoint = ROOT / 'artifacts/drawai/d3-local-v1/runs/official-swin-l128-aug1-seed44/best.pth'
    assert digest(checkpoint) == report['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['trained'] is True and saved['preprocessing'] == contract(128)
    model = build_model('swin', 128).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    torch.set_num_threads(4)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(release_dir / 'model.onnx'), options, providers=['CPUExecutionProvider'])
    fixtures = json.loads((release_dir / 'fixtures.json').read_text())['cases']
    browser = json.loads((ROOT / 'artifacts/drawai/ui/demo-parity.json').read_text())
    tests = json.loads((ROOT / 'artifacts/drawai/ui/playwright-report.json').read_text())
    assert tests['stats']['expected'] == 9
    assert all(tests['stats'][key] == 0 for key in ['unexpected', 'flaky', 'skipped'])
    assert len(browser['observations']) == 10 and browser['workers'] == browser['modelRequests'] == 1
    observations = []
    for label in range(10):
        drawing = json.loads((ROOT / f'web/public/samples/drawai/digit-{label}.json').read_text())
        assert drawing['reference_label'] == label and drawing['source']['split'] == 'train'
        tensor = preprocess(drawing['strokes'], 128)[None]
        with torch.inference_mode():
            expected = model(torch.from_numpy(tensor)).numpy()[0]
        actual = session.run(None, {'drawing': tensor})[0][0]
        fixture = next(f for f in fixtures if f['name'] == f'train-digit-{label}')
        web = browser['observations'][label]
        np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(expected, fixture['expected'], atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(expected, web['logits'], atol=1e-5, rtol=1e-4)
        assert int(expected.argmax()) == int(actual.argmax()) == web['predicted_id']
        observations.append(dict(reference_label=label, prediction=int(actual.argmax()),
                                 max_abs_pytorch_onnx=float(np.max(np.abs(expected - actual))),
                                 pytorch_logits=expected.tolist(), onnx_logits=actual.tolist(),
                                 browser_logits=web['logits']))
    ui = out / 'ui'; ui.mkdir(exist_ok=True)
    for name in ['desktop.png', 'mobile.png', 'demo-parity.json', 'playwright-report.json', 'user-strokes.json']:
        shutil.copy2(ROOT / 'artifacts/drawai/ui' / name, ui / name)
    fingerprint_paths = ['web/src/main.tsx', 'web/src/style.css', 'web/src/motion-style.css',
                         'models/SwinTransformer_cls.py']
    fingerprint_paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'web/src/drawai').glob('*') if p.is_file()]
    fingerprints = {p: digest(ROOT / p) for p in sorted(fingerprint_paths)}
    baseline = dict(status='passed', phase='P0', release=manifest['version'],
                    model_sha256=digest(release_dir / 'model.onnx'), checkpoint_sha256=digest(checkpoint),
                    parameters=sum(p.numel() for p in model.parameters()),
                    input_shape=session.get_inputs()[0].shape, output_shape=session.get_outputs()[0].shape,
                    browser_tests=tests['stats'], tested_viewports=[[1366, 768], [390, 844]],
                    real_touch_device_tested=False, real_private_data_evaluated=False,
                    samples_are_train_demos_not_test_accuracy=True,
                    sample_parity=observations, source_fingerprints=fingerprints,
                    onnx_manifest_all_files_verified=True, onnx_checker=True)
    (out / 'baseline.json').write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'PASS: trained checkpoint/ONNX/browser parity on {len(observations)} train demos; '
          f'{baseline["parameters"]} parameters; release hashes verified; 9 browser tests, screenshots copied.')


if __name__ == '__main__':
    main()
