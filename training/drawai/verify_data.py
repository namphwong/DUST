"""Check source identity, split coverage, labels and every exported tensor."""
import hashlib
import json
import platform
import zipfile

import numpy as np
from unlzw3 import unlzw

from .prepare_data import ROOT, digest, parse_unipen, trajectory_hash, write_json
from .preprocessing import contract, preprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    source = ROOT/'data/raw/pendigits'
    parsed = {}
    for name, count in [('pendigits-orig.tra', 7494), ('pendigits-orig.tes', 3498)]:
        original = unlzw((source/(name+'.Z')).read_bytes())
        require(original == (source/name).read_bytes(), f'{name}: raw differs from compressed source')
        parsed[name] = parse_unipen(original.decode('ascii'), name)
        require(len(parsed[name]) == count, f'{name}: source sample count mismatch')
    # Reconstruct the split independently of the production split function.
    rng = np.random.default_rng(42)
    labels = np.asarray([s['label'] for s in parsed['pendigits-orig.tra']])
    train_ids, val_ids = [], []
    for label in range(10):
        indexes = rng.permutation(np.flatnonzero(labels == label)).tolist()
        boundary = int(np.floor(len(indexes) * 0.2))
        val_ids.extend(indexes[:boundary]); train_ids.extend(indexes[boundary:])
    train_ids.sort(); val_ids.sort()
    groups = dict(train=[parsed['pendigits-orig.tra'][i] for i in train_ids],
                  validation=[parsed['pendigits-orig.tra'][i] for i in val_ids],test=parsed['pendigits-orig.tes'])
    checks = []
    baseline_entries = None
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        require(not {s['id'] for s in groups[a]} & {s['id'] for s in groups[b]}, f'ID overlap {a}/{b}')
    require(set(train_ids) | set(val_ids) == set(range(7494)), 'Development coverage mismatch')
    for length in [64,128]:
        target = ROOT/f'data/processed/pendigits-drawai-v1/l{length}'
        manifest = json.loads((target/'manifest.json').read_text())
        if manifest['archive_sha256'] is not None:
            require(digest(source/'source.zip') == manifest['archive_sha256'], 'Archive hash mismatch')
            with zipfile.ZipFile(source/'source.zip') as archive:
                for name in ['pendigits-orig.tra.Z', 'pendigits-orig.tes.Z', 'pendigits-orig.names', 'pendigits.names']:
                    require(archive.read(name) == (source/name).read_bytes(), f'{name}: archive/extracted mismatch')
        for name,expected in manifest['source_documents'].items():
            require(digest(source/name) == expected, f'{name}: document hash mismatch')
        for name,meta in manifest['sources'].items():
            require(digest(source/name) == meta['uncompressed_sha256'], f'{name}: raw hash mismatch')
            require(digest(source/(name+'.Z')) == meta['compressed_sha256'], f'{name}: compressed hash mismatch')
        require(manifest['seed'] == 42 and manifest['validation_fraction'] == 0.2, 'Split config mismatch')
        require(manifest['official_test_unchanged'] is True and manifest['writer_id_verified'] is False, 'Split declarations mismatch')
        require(json.loads((target/'preprocessing.json').read_text()) == contract(length), 'Preprocessing metadata mismatch')
        require(json.loads((target/'labels.json').read_text()) == dict(classes=[dict(id=i,name=str(i)) for i in range(10)]), 'Label metadata mismatch')
        entries = {split: manifest['splits'][split]['entries'] for split in groups}
        if baseline_entries is None:
            baseline_entries = entries
        else:
            require(entries == baseline_entries, 'L64/L128 do not have identical source provenance/split')
        for split,samples in groups.items():
            with np.load(target/f'{split}.npz',allow_pickle=False) as data:
                X,y,ids = data['X'],data['y'],data['sample_ids']
                require(X.dtype == np.float32 and X.shape == (len(samples),3,length), f'{split}: tensor shape/dtype')
                require(y.dtype == np.int64, f'{split}: label dtype')
                require(np.isfinite(X).all(), f'{split}: non-finite tensor')
                require(np.max(np.abs(X[:,:2])) <= 0.5, f'{split}: coordinates outside normalized range')
                require(np.isin(X[:,2], [0,1]).all(), f'{split}: invalid stroke marker')
                np.testing.assert_array_equal(y,[s['label'] for s in samples])
                np.testing.assert_array_equal(ids,[s['id'] for s in samples])
                require(manifest['splits'][split]['samples'] == len(samples), f'{split}: manifest sample count')
                for i,sample in enumerate(samples):
                    expected_entry = dict(id=sample['id'],label=sample['label'],source_line=sample['source_line'],
                                          source=sample['source'],source_index=sample['source_index'],
                                          segment_range=sample['segment_range'],trajectory_sha256=trajectory_hash(sample),
                                          writer_id=None,comment=sample['comment'])
                    require(entries[split][i] == expected_entry, f'{split}: provenance mismatch at {i}')
                    require(int(X[i,2].sum()) == len(sample['strokes']), f'{split}: lost stroke at {i}')
                    np.testing.assert_array_equal(X[i],preprocess(sample['strokes'],length,'up'))
                observed = hashlib.sha256(X.tobytes()+y.tobytes()+'\n'.join(ids).encode()).hexdigest()
                require(observed == manifest['splits'][split]['tensor_sha256'], f'{split}: tensor hash mismatch')
                checks.append(dict(length=length,split=split,samples=len(samples),tensor_sha256=observed))
        fixtures = json.loads((target/'fixtures.json').read_text())
        require(fixtures['version'] == contract(length)['version'] and fixtures['length'] == length, 'Fixture contract mismatch')
        require(len(fixtures['cases']) == 10, 'Need one fixture per class')
        for label, fixture in enumerate(fixtures['cases']):
            sample = next(s for s in groups['train'] if s['label'] == label)
            require(fixture['sample_id'] == sample['id'] and fixture['label'] == label
                    and fixture['strokes'] == sample['strokes'] and fixture['y_axis'] == 'up', 'Fixture provenance mismatch')
            np.testing.assert_array_equal(np.asarray(fixture['input'],np.float32).reshape(3,length),
                                          preprocess(sample['strokes'],length,'up'))
    write_json(ROOT/'artifacts/drawai/data/verification.json',dict(status='passed',checks=checks,
               verified_tensor_count=sum(c['samples'] for c in checks),writer_id_verified=False,
               independent_split_reconstruction=True, compressed_raw_archive_equivalence=True,
               provenance_and_fixture_verification=True, cross_length_split_equal=True,
               python=platform.python_version(), numpy=np.__version__))
    print(f"PASS: {len(checks)} splits, {sum(c['samples'] for c in checks)} tensors rebuilt exactly from source.")


if __name__ == '__main__':
    main()
