"""Export the repository's real Swin 1D and check fixed-shape inference.

This generates an UNTRAINED technical fixture, never an activity classifier.
Run from any directory: python training/export_smoke_test.py
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

# Optional process-local runtime directory for Windows machines with old VC DLLs.
# No system DLLs or global Python environment are modified.
_dll_handle = None
if sys.platform == 'win32' and os.environ.get('DUST_DLL_DIRECTORY'):
    _dll_handle = os.add_dll_directory(os.environ['DUST_DLL_DIRECTORY'])

import numpy as np
import torch
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.SwinTransformer_cls import SwinTransformer_1D

CONFIG = dict(seq_len=128, in_chans=3, num_classes=6, patch_size=2,
              window_size=8, embed_dim=32, depths=[2, 2, 2, 2],
              num_heads=[2, 4, 8, 16], drop_path_rate=0.1)
ATOL, RTOL = 1e-5, 1e-4


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = SwinTransformer_1D(**CONFIG).eval()
    # Original residual post-norms initialize to zero. Activate both branches
    # for the export fixture so parity checks actually exercise attention/MLP.
    for stage in model.layers:
        for block in stage.blocks:
            torch.nn.init.ones_(block.norm1.weight)
            torch.nn.init.ones_(block.norm2.weight)

    target = ROOT / 'web/public/models/phase1'
    target.mkdir(parents=True, exist_ok=True)
    path = target / 'model.onnx'
    example = torch.zeros(1, 3, 128)
    torch.onnx.export(model, example, str(path), input_names=['signals'],
                      output_names=['logits'], opset_version=17,
                      do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    fixtures = {
        'zeros': np.zeros((1, 3, 128), dtype=np.float32),
        'ones': np.ones((1, 3, 128), dtype=np.float32),
        'sine': np.stack([np.sin(t), np.cos(t), np.sin(2*t)])[None],
        'impulse': np.pad(np.ones((1, 3, 1), dtype=np.float32), ((0, 0), (0, 0), (63, 64))),
        **{f'random-{i}': rng.normal(size=(1, 3, 128)).astype(np.float32) for i in range(4)},
    }
    cases, errors = [], []
    for name, values in fixtures.items():
        with torch.inference_mode():
            expected = model(torch.from_numpy(values)).numpy()
        actual = session.run(['logits'], {'signals': values})[0]
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
        error = float(np.max(np.abs(actual - expected)))
        errors.append(error)
        cases.append(dict(name=name, input=values.reshape(-1).tolist(),
                          expected=expected.reshape(-1).tolist(), python_max_abs_error=error))
    # Check training path can produce gradients; no optimization/training here.
    model.train()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 3, 128)), torch.tensor([0, 5]))
    loss.backward()
    for parameter in [model.patch_embed.proj.weight, model.layers[0].blocks[1].attn.qkv.weight]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    timings = []
    for _ in range(5):
        session.run(None, {'signals': fixtures['sine']})
    for _ in range(30):
        start = time.perf_counter()
        session.run(None, {'signals': fixtures['sine']})
        timings.append((time.perf_counter() - start) * 1000)
    report = dict(status='passed', trained=False, seed=42, config=CONFIG,
                  input_shape=[1, 3, 128], output_shape=[1, 6], opset=17,
                  atol=ATOL, rtol=RTOL, cases=len(cases),
                  max_abs_error=max(errors), model_bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  parameters=sum(p.numel() for p in model.parameters()),
                  python_ort_median_ms=float(np.median(timings)), backward_pass='passed',
                  versions=dict(python=platform.python_version(), torch=torch.__version__,
                                numpy=np.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__),
                  fixture_note='Untrained; residual norm weights set to 1 to exercise attention and MLP.')
    (target / 'fixtures.json').write_text(json.dumps(dict(atol=ATOL, rtol=RTOL, cases=cases)), encoding='utf-8')
    (target / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
