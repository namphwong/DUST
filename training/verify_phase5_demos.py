"""Golden predictions for the six predeclared Phase 2 demo CSVs, not CSV resampling."""
import csv
import hashlib
import json
import torch
import numpy as np
from torch.torch_version import TorchVersion
from har_models import build_model
from prepare_uci_har import ROOT, write_json


def main():
    sample_root = ROOT / 'web/public/samples/uci-har-3ch-v1'
    index = json.loads((sample_root / 'index.json').read_text(encoding='utf-8'))
    artifact = ROOT / 'artifacts/phase3/kaggle'
    verified = json.loads((artifact / 'verification.json').read_text(encoding='utf-8'))
    run = verified['deployment_run']
    checkpoint_path = artifact / f'artifacts/phase3/runs/{run}/best.pth'
    report = json.loads((ROOT / 'web/public/models/har-swin-kaggle-v1/report.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == report['checkpoint_sha256']
    with torch.serialization.safe_globals([TorchVersion]):
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    torch.set_num_threads(4)
    model = build_model(saved['config']).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    prep = saved['preprocessing']
    mean = np.asarray(prep['mean'], np.float32)[:, None]
    std = np.asarray(prep['std'], np.float32)[:, None]
    results = []
    for demo in index['samples']:
        path = sample_root / demo['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == demo['csv_sha256']
        with path.open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 128 and [float(r['timestamp_ms']) for r in rows] == list(range(0, 2560, 20))
        raw = np.asarray([[float(r[f'acc_{axis}']) for r in rows] for axis in 'xyz'], np.float32)
        normalized = (raw-mean)/std
        with torch.inference_mode():
            logits = model(torch.from_numpy(normalized[None]))[0]
            scores = torch.softmax(logits, dim=0)
        results.append(dict(file=demo['file'], reference_id=demo['reference_label_id'],
                            predicted_id=int(logits.argmax()), logits=logits.tolist(), scores=scores.tolist(),
                            csv_sha256=demo['csv_sha256']))
    write_json(ROOT / 'artifacts/phase5/demo-reference.json', dict(status='passed',
               version=report['version'], checkpoint_sha256=report['checkpoint_sha256'],
               scope='Six fixed, sourced 128-point demo CSVs only; no resample/worker validation',
               cases=results))
    print('PASS: six demo CSV hashes/shape checked; PyTorch golden logits and scores saved.')


if __name__ == '__main__':
    main()
