"""Train DrawAI on train/validation only; no test loading in this module."""
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .metrics import classification_metrics
from .models import build_model, swin_config
from .prepare_data import ROOT, digest, write_json
from .preprocessing import contract


def tensor_hash(X, y, ids):
    return hashlib.sha256(X.tobytes() + y.tobytes() + '\n'.join(ids).encode()).hexdigest()


def load_split(data_dir, split, length):
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown split')
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    with np.load(data_dir / f'{split}.npz', allow_pickle=False) as archive:
        X, y, ids = archive['X'].copy(), archive['y'].copy(), archive['sample_ids'].copy()
    if (X.dtype != np.float32 or X.shape != (len(y), 3, length) or y.dtype != np.int64
            or len(ids) != len(y) or not len(y) or not np.isfinite(X).all()
            or not np.isin(y, range(10)).all() or len(set(ids)) != len(ids)):
        raise ValueError(f'{split}: data contract mismatch')
    entry = manifest['splits'][split]
    if (entry['samples'] != len(y) or [s['id'] for s in entry['entries']] != ids.tolist()
            or [s['label'] for s in entry['entries']] != y.tolist()
            or tensor_hash(X, y, ids) != entry['tensor_sha256']):
        raise ValueError(f'{split}: manifest/hash mismatch')
    if json.loads((data_dir / 'preprocessing.json').read_text()) != contract(length):
        raise ValueError('Preprocessing metadata mismatch')
    labels = json.loads((data_dir / 'labels.json').read_text())
    if labels != dict(classes=[dict(id=i, name=str(i)) for i in range(10)]):
        raise ValueError('Label metadata mismatch')
    return X, y, ids


def validate_config(config):
    if config['model'] != 'swin' or config['length'] not in (64, 128):
        raise ValueError('Unsupported model/length')
    for key in ('seed', 'epochs', 'patience', 'batch_size', 'warmup_epochs'):
        if type(config[key]) is not int or config[key] < (0 if key in ('seed', 'warmup_epochs') else 1):
            raise ValueError(f'Invalid {key}')
    if config['warmup_epochs'] >= config['epochs']:
        raise ValueError('Warmup must be shorter than total training')
    for key in ('lr', 'weight_decay', 'clip_grad_norm'):
        if not math.isfinite(config[key]) or config[key] < (0 if key == 'weight_decay' else 1e-12):
            raise ValueError(f'Invalid {key}')
    if type(config['augmentation']) is not bool:
        raise ValueError('augmentation must be boolean')
    if config.get('augmentation_spec') != dict(rotation_degrees=8, scale=[0.9, 1.1], translation=0.03):
        raise ValueError('Unknown augmentation specification')


def learning_rate(config, epoch):
    warm = config['warmup_epochs']
    if epoch < warm:
        return config['lr'] * (epoch + 1) / warm
    progress = (epoch - warm) / max(1, config['epochs'] - warm - 1)
    return config['lr'] * (0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2)


def augment(inputs):
    """Small train-only affine changes; marker is untouched; no reflection."""
    out = inputs.clone()
    random = torch.rand(len(out), 4, device=out.device)
    angle = (random[:, 0] * 2 - 1) * (8 * math.pi / 180)
    scale = 0.9 + random[:, 1] * 0.2
    x, y = inputs[:, 0], inputs[:, 1]
    out[:, 0] = scale[:, None] * (angle.cos()[:, None] * x - angle.sin()[:, None] * y) + (random[:, 2:3] * 2 - 1) * 0.03
    out[:, 1] = scale[:, None] * (angle.sin()[:, None] * x + angle.cos()[:, None] * y) + (random[:, 3:4] * 2 - 1) * 0.03
    return out


@torch.inference_mode()
def predict(model, data, batch_size, device):
    X, y, _ = data
    model.eval()
    logits = []
    loss_sum = 0.0
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size)
    for inputs, labels in loader:
        output = model(inputs.to(device))
        if output.shape != (len(labels), 10) or not torch.isfinite(output).all():
            raise ValueError('Nonfinite/invalid model output')
        loss_sum += float(torch.nn.functional.cross_entropy(output, labels.to(device), reduction='sum'))
        logits.append(output.cpu().numpy())
    logits = np.concatenate(logits)
    metrics = classification_metrics(y, logits.argmax(1))
    metrics['loss'] = loss_sum / len(y)
    return metrics, logits


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def train(config, data_dir, output, device_name='auto', threads=4, resume=False, stop_after=None):
    validate_config(config)
    torch.set_num_threads(threads)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    train_data = load_split(data_dir, 'train', config['length'])
    validation = load_split(data_dir, 'validation', config['length'])
    if set(train_data[2]) & set(validation[2]):
        raise ValueError('Train/validation ID overlap')
    hashes = {name: tensor_hash(*data) for name, data in [('train', train_data), ('validation', validation)]}
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pth').exists() and not resume:
        raise ValueError('Run already exists; use --resume')
    if resume and not (output / 'last.pth').exists():
        raise ValueError('Cannot resume without last checkpoint')
    model = build_model(config['model'], config['length']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    start_epoch, stale, history, previous_seconds = 0, 0, [], 0.0
    best = dict(macro_f1=-1.0, loss=float('inf'), epoch=0)
    metadata = dict(config=config, data_hashes=hashes, preprocessing=contract(config['length']),
                    labels=dict(classes=[dict(id=i, name=str(i)) for i in range(10)]),
                    model_kwargs=swin_config(config['length']),
                    parameters=sum(p.numel() for p in model.parameters()), trained=True,
                    device=str(device), torch_version=str(torch.__version__), python=platform.python_version(),
                    threads=threads, test_used_for_training=False)
    if resume:
        saved = torch.load(output / 'last.pth', map_location=device, weights_only=True)
        if saved['config'] != config or saved['data_hashes'] != hashes or saved['preprocessing'] != metadata['preprocessing']:
            raise ValueError('Resume config/data/preprocessing mismatch')
        if saved['device'] != str(device):
            raise ValueError('Resume device changed; deterministic replay requires same device')
        if saved['threads'] != threads or saved['torch_version'] != str(torch.__version__):
            raise ValueError('Resume thread count/Torch version changed')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        start_epoch, stale, history, best = saved['epoch'], saved['stale'], saved['history'], saved['best']
        previous_seconds = saved['training_seconds']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    started = time.perf_counter()
    interrupted = False
    dataset = TensorDataset(torch.from_numpy(train_data[0]), torch.from_numpy(train_data[1]))
    for epoch in range(start_epoch, config['epochs']):
        if stale >= config['patience']:
            break
        lr = learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        generator = torch.Generator().manual_seed(config['seed'] + epoch)
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, generator=generator, num_workers=0)
        model.train()
        epoch_start, loss_sum, correct = time.perf_counter(), 0.0, 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if config['augmentation']:
                inputs = augment(inputs)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
        val_metrics, val_logits = predict(model, validation, config['batch_size'], device)
        improved = (val_metrics['macro_f1'] > best['macro_f1'] or
                    (val_metrics['macro_f1'] == best['macro_f1'] and val_metrics['loss'] < best['loss']))
        record = dict(epoch=epoch+1, lr=lr, train_loss=loss_sum/len(dataset),
                      train_accuracy=correct/len(dataset), validation=val_metrics,
                      epoch_seconds=time.perf_counter()-epoch_start)
        history.append(record)
        if improved:
            best = dict(macro_f1=val_metrics['macro_f1'], loss=val_metrics['loss'], epoch=epoch+1)
            stale = 0
            save_checkpoint(output/'best.pth', dict(**metadata, epoch=epoch+1, validation=val_metrics, model_state_dict=model.state_dict()))
            np.savez_compressed(output/'best-validation.npz', logits=val_logits, y=validation[1], sample_ids=validation[2])
        else:
            stale += 1
        elapsed = previous_seconds + time.perf_counter()-started
        save_checkpoint(output/'last.pth', dict(**metadata, epoch=epoch+1, stale=stale, history=history, best=best,
                        training_seconds=elapsed, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
        write_json(output/'history.json', history)
        print(f"{output.name} epoch={epoch+1} loss={record['train_loss']:.4f} val_f1={val_metrics['macro_f1']:.4f} best={best['macro_f1']:.4f} seconds={record['epoch_seconds']:.2f}", flush=True)
        if stop_after is not None and epoch+1 >= stop_after:
            interrupted = epoch+1 < config['epochs'] and stale < config['patience']
            break
    summary = dict(**metadata, status='interrupted' if interrupted else 'completed',
                   best=best, epochs_completed=len(history), stopped_early=stale>=config['patience'],
                   training_seconds=previous_seconds+time.perf_counter()-started,
                   checkpoint_sha256=digest(output/'best.pth'))
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    data_dir=args.data_dir or ROOT/f"data/processed/pendigits-drawai-v1/l{config['length']}"
    train(config,data_dir,args.output_dir,args.device,args.threads,args.resume,args.stop_after)


if __name__=='__main__':
    main()
