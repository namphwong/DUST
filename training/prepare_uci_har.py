"""Prepare original UCI HAR windows. Requires only numpy, works on Kaggle/CPU.

Example: python training/prepare_uci_har.py --download
Alternatively --archive PATH or --raw-dir '/kaggle/input/.../UCI HAR Dataset'.
"""
import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABEL_CODES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
LABEL_VI = ['Đi bộ', 'Lên cầu thang', 'Xuống cầu thang', 'Ngồi', 'Đứng', 'Nằm']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def content_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class Source:
    """Read only required members; never extract arbitrary ZIP paths."""
    def __init__(self, path):
        self.path = Path(path)
        self.archive = None
        self.nested_buffer = None
        if self.path.is_dir():
            if not (self.path / 'train').is_dir():
                self.path = self.path / 'UCI HAR Dataset'
            if not (self.path / 'train').is_dir():
                raise ValueError('raw-dir must contain the original UCI HAR Dataset/train directory')
        else:
            self.archive = zipfile.ZipFile(self.path)
            if not any(name.endswith('/train/y_train.txt') for name in self.archive.namelist()):
                nested = [name for name in self.archive.namelist() if name.endswith('UCI HAR Dataset.zip')]
                if len(nested) != 1:
                    raise ValueError('Archive does not contain the original UCI HAR windows')
                self.nested_buffer = io.BytesIO(self.archive.read(nested[0]))
                self.archive.close()
                self.archive = zipfile.ZipFile(self.nested_buffer)

    def read(self, relative):
        if self.archive is None:
            return (self.path / relative).read_bytes()
        matches = [name for name in self.archive.namelist()
                   if name == relative or name.endswith('/' + relative)]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one archive member: {relative}')
        return self.archive.read(matches[0])

    def close(self):
        if self.archive:
            self.archive.close()
        if self.nested_buffer:
            self.nested_buffer.close()


def load_split(source, split, config):
    signal_files = [f'{split}/Inertial Signals/{channel}_{split}.txt' for channel in config['channels']]
    blobs = [source.read(name) for name in signal_files]
    label_blob = source.read(f'{split}/y_{split}.txt')
    subject_blob = source.read(f'{split}/subject_{split}.txt')
    X = np.stack([np.loadtxt(io.BytesIO(blob), dtype=np.float32, ndmin=2) for blob in blobs], axis=1)
    # Read IDs as floats first to reject non-integral IDs instead of truncating.
    labels = np.loadtxt(io.BytesIO(label_blob), ndmin=1)
    subjects = np.loadtxt(io.BytesIO(subject_blob), ndmin=1)
    if X.shape != (len(labels), 3, 128) or subjects.shape != labels.shape:
        raise ValueError(f'{split}: mismatched signal, label or subject shape')
    if not np.isfinite(X).all() or not np.isfinite(labels).all() or not np.isfinite(subjects).all():
        raise ValueError(f'{split}: non-finite values')
    if not np.isin(labels, np.arange(1, 7)).all():
        raise ValueError(f'{split}: activity IDs must be integers 1..6')
    if not np.isin(subjects, np.arange(1, 31)).all():
        raise ValueError(f'{split}: subject IDs must be integers 1..30')
    hashes = dict(zip(signal_files, [hashlib.sha256(blob).hexdigest() for blob in blobs]))
    hashes[f'{split}/y_{split}.txt'] = hashlib.sha256(label_blob).hexdigest()
    hashes[f'{split}/subject_{split}.txt'] = hashlib.sha256(subject_blob).hexdigest()
    return X, labels.astype(np.int64) - 1, subjects.astype(np.int64), hashes


def split_subjects(subjects, seed, validation_subject_count):
    unique = np.unique(subjects)
    if not 0 < validation_subject_count < len(unique):
        raise ValueError('validation_subject_count must leave both train and validation subjects')
    selected = np.sort(np.random.default_rng(seed).permutation(unique)[:validation_subject_count])
    mask = np.isin(subjects, selected)
    return np.flatnonzero(~mask), np.flatnonzero(mask), selected


def fit_preprocessing(train, config):
    if train.ndim != 3 or train.shape[1:] != (3, 128) or not len(train) or not np.isfinite(train).all():
        raise ValueError('Expected finite non-empty training windows [N,3,128]')
    # Accumulate in float64, store float32 values: same numbers used by web.
    mean = train.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0, 2), dtype=np.float64, ddof=0).astype(np.float32)
    std = np.maximum(std, np.float32(config['epsilon']))
    return mean, std


def normalize(X, mean, std):
    return ((X - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def prepare(source_path, output, metadata_dir, demo_dir, config):
    if config['channels'] != ['total_acc_x', 'total_acc_y', 'total_acc_z'] or config['window_size'] != 128:
        raise ValueError('Phase 2 contract requires total_acc X/Y/Z and 128-point original windows')
    if config['input_unit'] != 'g' or config['includes_gravity'] is not True or config['sample_rate_hz'] != 50:
        raise ValueError('Original UCI HAR uses total acceleration in g at 50 Hz')
    if not np.isfinite(config['epsilon']) or config['epsilon'] <= 0:
        raise ValueError('epsilon must be positive and finite')
    source = Source(source_path)
    try:
        activity_blob = source.read('activity_labels.txt')
        activities = [line.split() for line in activity_blob.decode('ascii').splitlines() if line.strip()]
        if activities != [[str(i+1), code] for i, code in enumerate(LABEL_CODES)]:
            raise ValueError('Source activity mapping differs from the Phase 0 contract')
        original_X, original_y, original_subjects, train_hashes = load_split(source, 'train', config)
        test_X, test_y, test_subjects, test_hashes = load_split(source, 'test', config)
    finally:
        source.close()
    if len(original_X) != 7352 or len(test_X) != 2947:
        raise ValueError('Expected original UCI HAR split: 7352 train and 2947 test windows')
    if len(np.unique(original_subjects)) != 21 or len(np.unique(test_subjects)) != 9:
        raise ValueError('Expected original subject split: 21 train and 9 test subjects')
    if set(original_subjects) & set(test_subjects):
        raise ValueError('Official train/test subjects overlap')

    train_indices, val_indices, val_subjects = split_subjects(original_subjects, config['seed'], config['validation_subject_count'])
    mean, std = fit_preprocessing(original_X[train_indices], config)
    specs = {'train': (original_X[train_indices], original_y[train_indices], original_subjects[train_indices], train_indices, 'train'),
             'val': (original_X[val_indices], original_y[val_indices], original_subjects[val_indices], val_indices, 'train'),
             'test': (test_X, test_y, test_subjects, np.arange(len(test_y)), 'test')}
    output.mkdir(parents=True, exist_ok=True)
    manifest_splits, report_splits = {}, {}
    for name, (raw, y, subject_ids, source_indices, original_split) in specs.items():
        if set(np.unique(y)) != set(range(6)):
            raise ValueError(f'{name}: every activity must be represented')
        X = normalize(raw, mean, std)
        np.savez(output / f'{name}.npz', X=X, X_raw=raw, y=y, subject_id=subject_ids,
                 source_index=source_indices.astype(np.int64))
        manifest_splits[name] = dict(original_split=original_split, subject_ids=np.unique(subject_ids).tolist(),
                                    source_indices_zero_based=source_indices.tolist(), count=len(y))
        report_splits[name] = dict(shape=list(X.shape), dtype=str(X.dtype), label_dtype=str(y.dtype),
                                  windows=len(y), subjects=np.unique(subject_ids).tolist(), finite=bool(np.isfinite(X).all()),
                                  class_counts={LABEL_CODES[i]: int((y == i).sum()) for i in range(6)},
                                  normalized_mean=X.mean(axis=(0, 2), dtype=np.float64).tolist(),
                                  normalized_std=X.std(axis=(0, 2), dtype=np.float64).tolist(),
                                  raw_min=raw.min(axis=(0, 2)).tolist(), raw_max=raw.max(axis=(0, 2)).tolist(),
                                  content_sha256=content_hash(X, raw, y, subject_ids, source_indices.astype(np.int64)))

    preprocessing = {key: config[key] for key in ['contract_version', 'artifact_version', 'channels', 'input_unit', 'includes_gravity', 'sample_rate_hz', 'window_size', 'stride', 'epsilon']}
    preprocessing.update(mean=mean.tolist(), std=std.tolist(), input_layout='NCT', fitted_on='train only after subject split',
                         fit_subject_ids=np.unique(original_subjects[train_indices]).tolist(), std_ddof=0,
                         statistics_weighting='All time points across training windows, including original overlapping windows',
                         transformed_dtype='float32')
    labels = dict(contract_version=config['contract_version'], artifact_version=config['artifact_version'],
                  classes=[dict(id=i, uci_id=i+1, code=code, name_vi=LABEL_VI[i]) for i, code in enumerate(LABEL_CODES)])
    manifest = dict(artifact_version=config['artifact_version'], config=config, splits=manifest_splits,
                    source_file_sha256={**train_hashes, **test_hashes,
                        'activity_labels.txt': hashlib.sha256(activity_blob).hexdigest()}, source_order_preserved=True)
    write_json(metadata_dir / 'preprocessing.json', preprocessing)
    write_json(metadata_dir / 'labels.json', labels)
    write_json(metadata_dir / 'split-manifest.json', manifest)
    write_json(output / 'preprocessing.json', preprocessing)
    write_json(output / 'labels.json', labels)
    write_json(output / 'split-manifest.json', manifest)

    # First official TRAIN row for each label, chosen without any model prediction.
    # Keep demo subjects out of test to avoid showcasing the held-out evaluation.
    demo_dir.mkdir(parents=True, exist_ok=True)
    demos = []
    for label_id, code in enumerate(LABEL_CODES):
        index = int(np.flatnonzero(original_y == label_id)[0])
        filename = code.lower() + '.csv'
        with (demo_dir / filename).open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp_ms', 'acc_x', 'acc_y', 'acc_z'])
            for timestep in range(128):
                writer.writerow([timestep*20, *[format(float(value), '.9g') for value in original_X[index, :, timestep]]])
        demos.append(dict(file=filename, dataset='UCI HAR', original_split='train',
                          prepared_split='val' if index in val_indices else 'train',
                          subject_id=int(original_subjects[index]), source_index_zero_based=index,
                          source_line_one_based=index+1, reference_label_id=label_id, reference_label=code,
                          input_unit='g', includes_gravity=True, window_size=128,
                          timestamp_note='Relative display timestamps for one independent window; not a reconstructed continuous trial',
                          csv_sha256=hashlib.sha256((demo_dir / filename).read_bytes()).hexdigest()))
    write_json(demo_dir / 'index.json', dict(artifact_version=config['artifact_version'],
               dataset_url='https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones',
               selection_rule='First official train row per class, no model/predictions used', samples=demos))
    report = dict(status='passed', artifact_version=config['artifact_version'], source_url=config['source_url'],
                  source_archive_sha256=hashlib.sha256(Path(source_path).read_bytes()).hexdigest() if Path(source_path).is_file() else None,
                  seed=config['seed'], validation_subjects=val_subjects.tolist(),
                  checks=dict(expected_official_counts=True, subject_disjoint=True, all_classes_present=True,
                              finite_inputs=True, original_order_preserved=True, fit_train_only=True),
                  splits=report_splits, preprocessing=preprocessing, demo_selection='First official train row per class')
    write_json(metadata_dir / 'data-report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--download', action='store_true')
    source_group.add_argument('--archive', type=Path)
    source_group.add_argument('--raw-dir', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'training/configs/data.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/uci-har-3ch-v1')
    parser.add_argument('--metadata-dir', type=Path, default=ROOT / 'artifacts/phase2')
    parser.add_argument('--demo-dir', type=Path, default=ROOT / 'web/public/samples/uci-har-3ch-v1')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    cache = ROOT / 'data/raw/uci-har.zip'
    source_path = args.raw_dir or args.archive or cache
    if args.download and not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_suffix('.zip.partial')
        print('Downloading official UCI archive...', flush=True)
        request = urllib.request.Request(config['source_url'], headers={'User-Agent': 'DuST-HAR/0.1'})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        if not zipfile.is_zipfile(partial):
            raise ValueError('Download is not a ZIP; cached file was not replaced')
        partial.replace(cache)
    if not source_path.exists():
        parser.error('Provide --download, --archive ZIP, or --raw-dir original dataset folder')
    report = prepare(source_path, args.output_dir, args.metadata_dir, args.demo_dir, config)
    print(json.dumps({name: {'shape': value['shape'], 'subjects': value['subjects']} for name, value in report['splits'].items()}, indent=2))
    print('PASS: saved prepared windows, metadata, manifest and 6 demo CSVs.')


if __name__ == '__main__':
    main()
