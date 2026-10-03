"""Freeze validation-only selection BEFORE opening the official test split."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from har_models import build_model
from prepare_uci_har import ROOT, content_hash, write_json
from train_har import load_data, predict, seed_everything, validate_data


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_selection(runs, output, data_dir):
    candidates = []
    train_data, val_data = validate_data(data_dir)
    expected_hash = content_hash(*train_data, *val_data)
    preprocessing = json.loads((data_dir / 'preprocessing.json').read_text(encoding='utf-8'))
    labels = json.loads((data_dir / 'labels.json').read_text(encoding='utf-8'))
    for run in runs:
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        digest = sha256(run / 'best.pth')
        if digest != summary['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from completed run: ' + str(run))
        if summary['data_hash'] != expected_hash or summary['preprocessing'] != preprocessing or summary['labels'] != labels:
            raise ValueError('Run trained with a different dataset or contract')
        candidates.append(dict(run=run.name, model=summary['config']['model'], seed=summary['config']['seed'],
                               validation_macro_f1=summary['best_validation_macro_f1'], best_epoch=summary['best_epoch'],
                               checkpoint_sha256=digest))
    if len({c['run'] for c in candidates}) != len(candidates):
        raise ValueError('Run names must be unique')
    selected = {}
    for family in ['cnn', 'swin']:
        eligible = [c for c in candidates if c['model'] == family]
        if not eligible:
            raise ValueError('Need both CNN baseline and Swin runs')
        selected[family] = max(eligible, key=lambda c: (c['validation_macro_f1'], -c['seed']))['run']
    selection = dict(rule='Best validation macro-F1 per family; tie uses smallest seed. Deployment family fixed to Swin.',
                     test_used_for_selection=False, candidates=candidates, selected=selected,
                     deployment_run=selected['swin'])
    path = output / 'selection.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != selection:
        raise ValueError('Selection already frozen. Do not change candidates after viewing test results.')
    write_json(path, selection)
    return selection


def plot_run(run, metrics, output):
    history = json.loads((run / 'history.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epochs = [h['epoch'] for h in history]
    axes[0].plot(epochs, [h['train_loss'] for h in history], label='Train')
    axes[0].plot(epochs, [h['validation']['loss'] for h in history], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy', title=run.name)
    axes[0].legend()
    axes[1].plot(epochs, [h['validation']['macro_f1'] for h in history])
    axes[1].set(xlabel='Epoch', ylabel='Validation macro-F1', ylim=(0, 1))
    fig.savefig(output / f'{run.name}-learning-curves.png', dpi=140)
    plt.close(fig)
    matrix = np.asarray(metrics['confusion_matrix'])
    codes = [c['code'] for c in metrics['per_class']]
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=codes, yticklabels=codes,
           xlabel='Predicted', ylabel='True', title=run.name + ' — official test (counts)')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    for row in range(6):
        for col in range(6):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center',
                    color='white' if matrix[row, col] > matrix.max()/2 else 'black')
    fig.colorbar(im, ax=ax)
    fig.savefig(output / f'{run.name}-confusion-matrix.png', dpi=140)
    plt.close(fig)


def seed_statistics(results):
    aggregates = {}
    for family in ['cnn', 'swin']:
        members = [result for result in results if result['model'] == family]
        if not members:
            continue
        aggregates[family] = dict(seeds=[r['seed'] for r in members], count=len(members),
                                 std_ddof=1 if len(members) > 1 else None)
        for metric in ['accuracy', 'macro_f1']:
            values = [result['test'][metric] for result in members]
            aggregates[family][metric] = dict(mean=float(np.mean(values)),
                                              std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    return aggregates


def evaluate(runs, data_dir, output, device='cpu', all_seeds=False):
    seed_everything(42, 4)
    output.mkdir(parents=True, exist_ok=True)
    selection = freeze_selection(runs, output, data_dir)
    # Test is not opened until selection has been validated and written to disk.
    X, y, subjects, indices = load_data(data_dir, 'test')
    train_data, val_data = validate_data(data_dir)
    manifest = json.loads((data_dir / 'split-manifest.json').read_text(encoding='utf-8'))['splits']['test']
    if set(subjects) & (set(train_data[2]) | set(val_data[2])):
        raise ValueError('Test subjects overlap train/validation')
    if indices.tolist() != manifest['source_indices_zero_based'] or np.unique(subjects).tolist() != manifest['subject_ids']:
        raise ValueError('Test differs from manifest')
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=64, shuffle=False)
    results = []
    for run in runs:
        if not all_seeds and run.name not in selection['selected'].values():
            continue
        checkpoint = torch.load(run / 'best.pth', map_location=device, weights_only=False)
        model = build_model(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        metrics, logits, predictions = predict(model, loader, torch.device(device))
        for record, label in zip(metrics['per_class'], checkpoint['labels']['classes']):
            record.update(code=label['code'], name_vi=label['name_vi'])
        result = dict(run=run.name, model=checkpoint['config']['model'], seed=checkpoint['config']['seed'],
                      checkpoint_sha256=sha256(run / 'best.pth'), best_epoch=checkpoint['epoch'],
                      validation=checkpoint['validation'], test=metrics, test_sha256=content_hash(X, y, subjects, indices),
                      environment=checkpoint['environment'], trained=True, test_used_for_selection=False)
        write_json(output / f'{run.name}-metrics.json', result)
        np.savez_compressed(output / f'{run.name}-predictions.npz', y_true=y, y_pred=predictions,
                            logits=logits, subject_id=subjects, source_index=indices)
        plot_run(run, metrics, output)
        results.append(result)
        print(f"{run.name}: test accuracy={metrics['accuracy']:.6f} macro_f1={metrics['macro_f1']:.6f}", flush=True)
    report = dict(selection=selection, results=results, target_macro_f1=0.90,
                  target_met=next(r['test']['macro_f1'] for r in results if r['run'] == selection['deployment_run']) >= 0.90)
    if all_seeds:
        report['seed_statistics'] = seed_statistics(results)
        report['test_evaluation_scope'] = 'All predeclared seeds; selection remains validation-only. Official test was already viewed in prior local run.'
    write_json(output / 'evaluation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/uci-har-3ch-v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/phase3/evaluation')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--all-seeds', action='store_true', help='Report all predeclared seeds, never select by test')
    args = parser.parse_args()
    evaluate(args.runs, args.data_dir, args.output_dir, args.device, args.all_seeds)


if __name__ == '__main__':
    main()
