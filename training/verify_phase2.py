"""Verify actual prepared UCI arrays, provenance, demos and reproducibility."""
import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from prepare_uci_har import ROOT, Source, load_split, prepare, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/raw/uci-har.zip')
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/processed/uci-har-3ch-v1')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'artifacts/phase2')
    parser.add_argument('--demos', type=Path, default=ROOT / 'web/public/samples/uci-har-3ch-v1')
    parser.add_argument('--config', type=Path, default=ROOT / 'training/configs/data.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((args.metadata / 'split-manifest.json').read_text(encoding='utf-8'))
    preprocessing = json.loads((args.metadata / 'preprocessing.json').read_text(encoding='utf-8'))
    report = json.loads((args.metadata / 'data-report.json').read_text(encoding='utf-8'))
    source = Source(args.source)
    try:
        original = {name: load_split(source, name, config) for name in ['train', 'test']}
    finally:
        source.close()
    arrays = {}
    for split in ['train', 'val', 'test']:
        with np.load(args.prepared / f'{split}.npz', allow_pickle=False) as file:
            arrays[split] = {name: file[name] for name in file.files}
        data = arrays[split]
        info = manifest['splits'][split]
        indices = np.asarray(info['source_indices_zero_based'])
        raw, labels, subjects, _ = original[info['original_split']]
        np.testing.assert_array_equal(data['X_raw'], raw[indices])
        np.testing.assert_array_equal(data['y'], labels[indices])
        np.testing.assert_array_equal(data['subject_id'], subjects[indices])
        np.testing.assert_array_equal(data['source_index'], indices)
        assert data['X'].dtype == np.float32 and data['y'].dtype == np.int64
        assert data['X'].shape[1:] == (3, 128) and np.isfinite(data['X']).all()
        assert set(np.unique(data['y'])) == set(range(6))
        # Independent transformation with metadata as a future browser would do.
        expected = ((raw[indices].astype(np.float64) - np.array(preprocessing['mean'])[None, :, None])
                    / np.array(preprocessing['std'])[None, :, None]).astype(np.float32)
        np.testing.assert_allclose(data['X'], expected, rtol=1e-6, atol=1e-6)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not set(arrays[a]['subject_id']) & set(arrays[b]['subject_id'])
    combined = np.sort(np.concatenate([arrays['train']['source_index'], arrays['val']['source_index']]))
    np.testing.assert_array_equal(combined, np.arange(7352))
    np.testing.assert_array_equal(arrays['test']['source_index'], np.arange(2947))
    train = arrays['train']['X_raw'].astype(np.float64)
    np.testing.assert_array_equal(np.asarray(preprocessing['mean'], dtype=np.float32), train.mean(axis=(0, 2)).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(preprocessing['std'], dtype=np.float32), train.std(axis=(0, 2)).astype(np.float32))
    np.testing.assert_allclose(arrays['train']['X'].mean(axis=(0, 2), dtype=np.float64), 0, atol=1e-6)
    np.testing.assert_allclose(arrays['train']['X'].std(axis=(0, 2), dtype=np.float64), 1, atol=1e-6)

    demo_index = json.loads((args.demos / 'index.json').read_text(encoding='utf-8'))
    assert len(demo_index['samples']) == 6
    for demo in demo_index['samples']:
        path = args.demos / demo['file']
        with path.open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        values = np.asarray([[float(row[name]) for row in rows] for name in ['acc_x', 'acc_y', 'acc_z']], dtype=np.float32)
        index = demo['source_index_zero_based']
        np.testing.assert_array_equal(values, original['train'][0][index])
        np.testing.assert_array_equal([int(row['timestamp_ms']) for row in rows], np.arange(128)*20)
        assert demo['reference_label_id'] == int(original['train'][1][index])
        assert demo['subject_id'] == int(original['train'][2][index])
        assert index == int(np.flatnonzero(original['train'][1] == demo['reference_label_id'])[0])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == demo['csv_sha256']

    # Rebuild all artifacts in a disposable task directory; compare numerical
    # payloads because ZIP container timestamps need not be byte-identical.
    with tempfile.TemporaryDirectory(prefix='dust-phase2-verify-') as directory:
        target = Path(directory)
        repeat_report = prepare(args.source, target / 'data', target / 'metadata', target / 'demos', config)
        assert report == repeat_report
        for filename in ['split-manifest.json', 'preprocessing.json', 'labels.json']:
            assert (args.metadata / filename).read_bytes() == (target / 'metadata' / filename).read_bytes()
        for split, data in arrays.items():
            with np.load(target / 'data' / f'{split}.npz', allow_pickle=False) as repeat:
                for name, expected in data.items():
                    np.testing.assert_array_equal(expected, repeat[name])
        for file in args.demos.iterdir():
            if file.suffix in ['.csv', '.json']:
                assert file.read_bytes() == (target / 'demos' / file.name).read_bytes()
    verification = dict(status='passed', criteria=['P2-01', 'P2-02', 'P2-03', 'P2-04'],
                        checks=['source pairing', 'train/val/test subject disjoint', 'complete source indices',
                                'train-only mean/std', 'normalized tensor matches metadata',
                                'six CSVs roundtrip exactly to source float32', 'demo selection independent of model',
                                'second build identical arrays/metadata/demos'],
                        archive_sha256=report['source_archive_sha256'], numpy_version=np.__version__)
    write_json(args.metadata / 'verification.json', verification)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
