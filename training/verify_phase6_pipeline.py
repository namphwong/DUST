"""Build synthetic CSV fixtures and real checkpoint golden tensors/logits."""
import hashlib
import json
import numpy as np
import torch
from torch.torch_version import TorchVersion
from prepare_uci_har import ROOT, write_json
from har_models import build_model
from csv_pipeline import parse_csv, prepare_csv, normalize


def signal_csv(times, unit='g', reorder=False):
    scale = 9.80665 if unit == 'ms2' else 1
    rows = []
    for i, time in enumerate(times):
        # Deterministic distinct axes; no artificial reference activity label.
        x, y, z = [v*scale for v in (0.8+0.15*np.sin(i/9), -0.02+0.08*np.cos(i/13), 0.07+i*0.0001)]
        rows.append(f'{z:.17g},"extra, ignored",{time:.17g},{y:.17g},{x:.17g}' if reorder else f'{time:.17g},{x:.17g},{y:.17g},{z:.17g}')
    header = '\ufeffacc_z,note,timestamp_ms,acc_y,acc_x' if reorder else 'timestamp_ms,acc_x,acc_y,acc_z'
    return header+'\r\n'+'\r\n'.join(rows)+'\r\n\r\n'


def fixture_specs():
    fixtures = [(f'boundary-{n}', np.arange(n)*20, 'g', False, 'TOO_SHORT' if n == 127 else None) for n in (127,128,191,192)]
    fixtures += [
        ('units-ms2', np.arange(192)*20, 'ms2', False, None),
        ('absolute-reordered', 1700000000000+np.arange(192)*20, 'g', True, None),
        ('jitter-linear', np.arange(220)*15+np.where(np.arange(220)%2,2,0), 'g', False, None),
        ('100hz-downsample', np.arange(400)*10, 'g', False, None),
        ('40hz-upsample', np.arange(160)*25, 'g', False, None),
        ('two-segments', np.r_[np.arange(192)*20,5000+np.arange(128)*20], 'g', False, None),
        ('drop-short-segment', np.r_[np.arange(12)*20,1000+np.arange(128)*20], 'g', False, None),
        ('gap-100-not-split', np.r_[np.arange(64)*20,1360+np.arange(64)*20], 'g', False, None),
        ('gap-101-split-short', np.r_[np.arange(64)*20,1361+np.arange(64)*20], 'g', False, 'TOO_SHORT'),
        ('unsupported-30hz', np.arange(128)*33, 'g', False, 'UNSUPPORTED_FREQUENCY'),
        ('unsupported-200hz', np.arange(600)*5, 'g', False, 'UNSUPPORTED_FREQUENCY'),
        ('expanded-row-limit', np.arange(90000)*25, 'g', False, 'TOO_MANY_ROWS'),
        ('incompatible-gravity', np.arange(128)*20, 'g', False, 'INCOMPATIBLE_SIGNAL'),
    ]
    return fixtures


def main():
    release = ROOT/'web/public/models/har-swin-kaggle-v1'
    report = json.loads((release/'report.json').read_text(encoding='utf-8'))
    checkpoint = ROOT/'artifacts/phase3/kaggle/artifacts/phase3/runs/swin-seed44/best.pth'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report['checkpoint_sha256']
    with torch.serialization.safe_globals([TorchVersion]):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    torch.set_num_threads(4)
    model = build_model(saved['config']).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    target = ROOT/'artifacts/phase6/fixtures'
    target.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, times, unit, reorder, error in fixture_specs():
        text = signal_csv(times, unit, reorder)
        (target/f'{name}.csv').write_text(text, encoding='utf-8', newline='')
        case = dict(name=name, file=f'{name}.csv', unit=unit, gravity=name != 'incompatible-gravity', expectedError=error,
                    csv_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
        try:
            info, windows = prepare_csv(parse_csv(text), unit, case['gravity'])
            assert error is None, name
            case['summary'] = info
            case['windows'] = []
            for window in windows:
                normalized = normalize(window.pop('raw'),saved['preprocessing'])
                with torch.inference_mode():
                    logits = model(torch.from_numpy(normalized[None]))[0]
                    scores = torch.softmax(logits, dim=0)
                case['windows'].append(dict(**window,input=normalized.flatten().tolist(), logits=logits.tolist(),
                                            predictedId=int(logits.argmax()), scores=scores.tolist()))
        except ValueError as exc:
            assert str(exc) == error, (name,str(exc),error)
        cases.append(case)
    write_json(ROOT/'artifacts/phase6/python-reference.json',dict(status='passed',
               version=report['version'], checkpoint_sha256=report['checkpoint_sha256'],
               scope='Synthetic CSV pipeline parity only, not accuracy or new training', cases=cases))
    print(f'PASS: {len(cases)} CSV fixtures with NumPy normalization / actual PyTorch golden predictions.')


if __name__ == '__main__':
    main()
