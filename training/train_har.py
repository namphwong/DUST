"""Train using train/val ONLY. Evaluate official test in evaluate.py later."""
import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from har_metrics import classification_metrics
from har_models import build_model
from prepare_uci_har import ROOT, content_hash, write_json


def load_data(data_dir, split):
    with np.load(data_dir / f'{split}.npz', allow_pickle=False) as archive:
        X, y = archive['X'].copy(), archive['y'].copy()
        subject, indices = archive['subject_id'].copy(), archive['source_index'].copy()
    if X.dtype != np.float32 or X.shape[1:] != (3, 128) or y.dtype != np.int64:
        raise ValueError(f'{split}: tensor contract mismatch')
    if len(X) != len(y) or not np.isfinite(X).all() or not np.isin(y, range(6)).all():
        raise ValueError(f'{split}: invalid inputs or labels')
    return X, y, subject, indices


def validate_data(data_dir):
    train, val = load_data(data_dir, 'train'), load_data(data_dir, 'val')
    if set(train[2]) & set(val[2]):
        raise ValueError('Train and validation subjects overlap')
    manifest = json.loads((data_dir / 'split-manifest.json').read_text(encoding='utf-8'))
    for split, data in [('train', train), ('val', val)]:
        entry = manifest['splits'][split]
        if np.unique(data[2]).tolist() != entry['subject_ids'] or data[3].tolist() != entry['source_indices_zero_based']:
            raise ValueError('Dataset differs from split manifest')
    return train, val


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    predicted, truth, logits = [], [], []
    total_loss = 0.0
    for inputs, labels in loader:
        output = model(inputs.to(device))
        if not torch.isfinite(output).all():
            raise ValueError('Non-finite logits')
        total_loss += torch.nn.functional.cross_entropy(output, labels.to(device), reduction='sum').item()
        predicted.extend(output.argmax(dim=1).cpu().tolist())
        truth.extend(labels.tolist())
        logits.append(output.cpu().numpy())
    metrics = classification_metrics(truth, predicted)
    metrics['loss'] = total_loss / len(truth)
    return metrics, np.concatenate(logits), np.array(predicted, dtype=np.int64)


def train(config, data_dir, output, device_name='auto', threads=4, resume=False):
    seed_everything(config['seed'], threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else torch.device(device_name)
    train_data, val_data = validate_data(data_dir)
    preprocessing = json.loads((data_dir / 'preprocessing.json').read_text(encoding='utf-8'))
    labels = json.loads((data_dir / 'labels.json').read_text(encoding='utf-8'))
    data_hash = content_hash(*train_data, *val_data)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pth').exists() and not resume:
        raise ValueError('Run already exists. Use --resume or a new --output-dir.')
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    def schedule(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch+1)/config['warmup_epochs']
        progress = (epoch-config['warmup_epochs'])/max(1, config['epochs']-config['warmup_epochs'])
        return 0.05 + 0.95*(1+math.cos(math.pi*progress))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch, best_f1, best_epoch, stale = 0, -1.0, 0, 0
    history = []
    previous_seconds = 0
    if resume:
        # Only resume checkpoints produced locally/by this notebook, not untrusted uploads.
        saved = torch.load(output / 'last.pth', map_location=device, weights_only=False)
        if saved['config'] != config or saved['data_hash'] != data_hash or saved['preprocessing'] != preprocessing:
            raise ValueError('Resume config/data/preprocessing differs from checkpoint')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda' and saved['cuda_rng']:
            torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda_rng']])
        start_epoch, best_f1, best_epoch, stale = saved['epoch'], saved['best_f1'], saved['best_epoch'], saved['stale']
        history = saved['history']
        previous_seconds = saved['training_seconds']
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_data[0]), torch.from_numpy(val_data[1])),
                            batch_size=config['batch_size'], shuffle=False)
    train_dataset = TensorDataset(torch.from_numpy(train_data[0]), torch.from_numpy(train_data[1]))
    metadata = dict(config=config, data_hash=data_hash, preprocessing=preprocessing, labels=labels,
                    trained=True, parameters=sum(p.numel() for p in model.parameters()),
                    environment=dict(device=str(device), hardware=torch.cuda.get_device_name() if device.type == 'cuda' else platform.processor(),
                                     python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, threads=threads))
    write_json(output / 'run-config.json', metadata)
    started = time.perf_counter()
    for epoch in range(start_epoch, config['epochs']):
        if stale >= config['patience']:
            break
        epoch_started = time.perf_counter()
        # Epoch-specific shuffling survives interruption without hidden loader state.
        loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                            generator=torch.Generator().manual_seed(config['seed']+epoch))
        model.train()
        loss_sum = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()*len(targets)
        val_metrics, _, _ = predict(model, val_loader, device)
        record = dict(epoch=epoch+1, train_loss=loss_sum/len(train_dataset), validation=val_metrics,
                      lr=optimizer.param_groups[0]['lr'], epoch_seconds=time.perf_counter()-epoch_started)
        history.append(record)
        improved = val_metrics['macro_f1'] > best_f1 + 1e-6
        if improved:
            best_f1, best_epoch, stale = val_metrics['macro_f1'], epoch+1, 0
            torch.save(dict(**metadata, epoch=epoch+1, validation=val_metrics, model_state_dict=model.state_dict()), output / 'best.pth')
        else:
            stale += 1
        scheduler.step()
        elapsed = previous_seconds + time.perf_counter()-started
        # Atomic checkpoint promotion preserves the previous last checkpoint if interrupted.
        checkpoint_path = output / 'last.pth.partial'
        torch.save(dict(**metadata, epoch=epoch+1, best_f1=best_f1, best_epoch=best_epoch, stale=stale,
                        history=history, training_seconds=elapsed, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []), checkpoint_path)
        checkpoint_path.replace(output / 'last.pth')
        write_json(output / 'history.json', history)
        print(f"{config['model']} seed={config['seed']} epoch={epoch+1} train_loss={record['train_loss']:.4f} val_f1={best_f1:.4f} current_f1={val_metrics['macro_f1']:.4f} seconds={record['epoch_seconds']:.1f}", flush=True)
    summary = dict(**metadata, best_epoch=best_epoch, best_validation_macro_f1=best_f1,
                   epochs_completed=len(history), training_seconds=previous_seconds+time.perf_counter()-started,
                   stopped_early=stale >= config['patience'], test_used_for_training=False,
                   checkpoint_sha256=hashlib.sha256((output / 'best.pth').read_bytes()).hexdigest())
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/uci-har-3ch-v1')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    train(config, args.data_dir, args.output_dir, args.device, args.threads, args.resume)


if __name__ == '__main__':
    main()
