"""Verify/downloaded Kaggle artifacts without unpickling checkpoints or unsafe ZIP extraction."""
import argparse
import ast
import hashlib
import io
import json
import math
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from har_metrics import classification_metrics
from evaluate_har import seed_statistics
from prepare_uci_har import ROOT, content_hash, write_json
from train_har import load_data, validate_data


def verify_and_import(archive_path, output):
    notebook_path = ROOT / 'notebooks/kaggle_train_har_gpu.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    source_cell = next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('SOURCES = '))
    sources = ast.literal_eval(ast.parse(source_cell).body[0].value)
    expected_sources = {name: hashlib.sha256(source.encode('utf-8')).hexdigest() for name, source in sources.items()}
    train, val = validate_data(ROOT / 'data/processed/uci-har-3ch-v1')
    training_hash = content_hash(*train, *val)
    _, truth, test_subjects, test_indices = load_data(ROOT / 'data/processed/uci-har-3ch-v1', 'test')
    expected_runs = [f'{family}-seed{seed}' for family in ['cnn', 'swin'] for seed in [42, 43, 44]]
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError('Duplicate ZIP members')
        for member in members:
            path = PurePosixPath(member)
            if path.is_absolute() or '..' in path.parts or '\\' in member or ':' in member:
                raise ValueError('Unsafe ZIP path: ' + member)
            if member != 'source-hashes.json' and (not member.startswith('artifacts/') or path.suffix not in ['.json', '.png', '.npz', '.pth']):
                raise ValueError('Unexpected artifact: ' + member)
        required = {'source-hashes.json', 'artifacts/phase3/evaluation/evaluation.json',
                    'artifacts/phase3/evaluation/selection.json'}
        if not required.issubset(members):
            raise ValueError('Archive is missing required metadata')
        def read_json(name):
            return json.loads(archive.read(name).decode('utf-8'))
        if read_json('source-hashes.json') != expected_sources:
            raise ValueError('Kaggle sources differ from uploaded notebook')
        report = read_json('artifacts/phase3/evaluation/evaluation.json')
        selection = read_json('artifacts/phase3/evaluation/selection.json')
        if report['selection'] != selection or selection['test_used_for_selection']:
            raise ValueError('Selection/report mismatch')
        if sorted(c['run'] for c in selection['candidates']) != sorted(expected_runs):
            raise ValueError('Unexpected experiment candidates')
        if sorted(r['run'] for r in report['results']) != sorted(expected_runs):
            raise ValueError('Need evaluation of all six predeclared runs')
        labels = json.loads((ROOT / 'artifacts/phase2/labels.json').read_text(encoding='utf-8'))
        preprocessing = json.loads((ROOT / 'artifacts/phase2/preprocessing.json').read_text(encoding='utf-8'))
        for result in report['results']:
            run = result['run']
            family, seed = run.split('-seed')
            summary = read_json(f'artifacts/phase3/runs/{run}/summary.json')
            config = json.loads((ROOT / f'training/configs/train-{family}.json').read_text(encoding='utf-8'))
            config['seed'] = int(seed)
            if summary['config'] != config or summary['data_hash'] != training_hash:
                raise ValueError('Run configuration/training data mismatch')
            if summary['labels'] != labels or summary['preprocessing'] != preprocessing:
                raise ValueError('Label/normalization mismatch')
            if summary['environment']['device'] != 'cuda' or not summary['trained'] or summary['test_used_for_training']:
                raise ValueError('Not a genuine CUDA training run')
            if result['environment'] != summary['environment']:
                raise ValueError('Result environment mismatch')
            digest = hashlib.sha256(archive.read(f'artifacts/phase3/runs/{run}/best.pth')).hexdigest()
            candidate = next(c for c in selection['candidates'] if c['run'] == run)
            if not (digest == summary['checkpoint_sha256'] == result['checkpoint_sha256'] == candidate['checkpoint_sha256']):
                raise ValueError('Checkpoint hash mismatch')
            if candidate['validation_macro_f1'] != summary['best_validation_macro_f1'] or result['best_epoch'] != summary['best_epoch']:
                raise ValueError('Validation selection metadata mismatch')
            with np.load(io.BytesIO(archive.read(f'artifacts/phase3/evaluation/{run}-predictions.npz')), allow_pickle=False) as predictions:
                np.testing.assert_array_equal(predictions['y_true'], truth)
                np.testing.assert_array_equal(predictions['subject_id'], test_subjects)
                np.testing.assert_array_equal(predictions['source_index'], test_indices)
                if predictions['logits'].shape != (len(truth), 6) or not np.isfinite(predictions['logits']).all():
                    raise ValueError('Invalid logits')
                np.testing.assert_array_equal(predictions['logits'].argmax(axis=1), predictions['y_pred'])
                metrics = classification_metrics(truth, predictions['y_pred'])
            for key in ['accuracy', 'macro_f1']:
                if not math.isclose(metrics[key], result['test'][key], abs_tol=1e-12, rel_tol=0):
                    raise ValueError('Test metric does not match saved predictions')
            if metrics['confusion_matrix'] != result['test']['confusion_matrix']:
                raise ValueError('Confusion matrix mismatch')
            for expected, observed in zip(metrics['per_class'], result['test']['per_class']):
                for key in ['precision', 'recall', 'f1', 'support']:
                    if not math.isclose(expected[key], observed[key], abs_tol=1e-12, rel_tol=0):
                        raise ValueError('Per-class metric mismatch')
        for family in ['cnn', 'swin']:
            candidates = [c for c in selection['candidates'] if c['model'] == family]
            chosen = max(candidates, key=lambda c: (c['validation_macro_f1'], -c['seed']))['run']
            if selection['selected'][family] != chosen:
                raise ValueError('Selected run is not validation winner')
        if selection['deployment_run'] != selection['selected']['swin']:
            raise ValueError('Deployment family must remain Swin')
        if report['seed_statistics'] != seed_statistics(report['results']):
            raise ValueError('Seed statistics mismatch')
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output directory already contains files; preserve existing artifacts')
        output.mkdir(parents=True, exist_ok=True)
        # Copy only expected artifact types to validated relative destinations.
        for member in members:
            relative = PurePosixPath(member)
            target = output.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(output.resolve()):
                raise ValueError('Artifact escapes output root')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination)
    verification = dict(status='passed', kaggle_gpu_verified=True, seeds=[42, 43, 44],
                        notebook_url='https://www.kaggle.com/code/namphwong/kaggle-train-har-gpu',
                        run_id='354801477', notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                        downloaded_zip_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                        deployment_run=selection['deployment_run'],
                        checks=['embedded source hashes', 'six predeclared CUDA runs', 'config and train/val data hashes',
                                'train-only preprocessing', 'checkpoint hashes', 'test predictions vs official labels',
                                'recomputed metrics/confusion matrix', 'validation-only selection', 'mean/sample std'],
                        official_test_already_viewed_in_local_run=True)
    write_json(output / 'verification.json', verification)
    print(json.dumps(verification, indent=2))
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/phase3/kaggle')
    args = parser.parse_args()
    verify_and_import(args.archive, args.output_dir)
