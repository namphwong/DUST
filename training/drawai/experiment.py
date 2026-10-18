"""Validation-only pilot selection, locked Swin-only design, checkpoint selection."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .prepare_data import ROOT, digest, write_json
from .train import train

PLAN = dict(version='drawai-swin-experiment-v2', pilot_seed=42, seeds=[42,43,44],
            lengths=[64,128], augmentation_candidates=[False,True], pilot_epochs=12,
            official_epochs=50, pilot_patience=12, official_patience=10,
            batch_size=128, learning_rate=0.001, weight_decay=0.01, gradient_clip=1.0,
            augmentation_spec=dict(rotation_degrees=8,scale=[0.9,1.1],translation=0.03),
            selection_rule='Highest validation macro-F1; lowest validation loss; shorter L; no augmentation; lowest seed; run ID',
            checkpoint_rule='Highest validation macro-F1; lowest validation loss; earlier epoch',
            test_access='evaluate module only after all three official Swin checkpoints are locked')


def config_for(model,length,augmentation,seed,official):
    if model != 'swin':
        raise ValueError('P1 training supports only Swin; DINOv3 is planned for P2')
    return dict(model=model,length=length,augmentation=augmentation,seed=seed,
                augmentation_spec=PLAN['augmentation_spec'],
                epochs=PLAN['official_epochs'] if official else PLAN['pilot_epochs'],
                patience=PLAN['official_patience'] if official else PLAN['pilot_patience'],
                batch_size=PLAN['batch_size'],lr=PLAN['learning_rate'],weight_decay=PLAN['weight_decay'],
                warmup_epochs=3 if official else 1,clip_grad_norm=PLAN['gradient_clip'])


def ranked(summary):
    c=summary['config']; b=summary['best']
    return (-b['macro_f1'],b['loss'],c['length'],c['augmentation'],c['seed'],summary['run_id'])


def run_one(config,folder,device,threads):
    data_dir=ROOT/f"data/processed/pendigits-drawai-v1/l{config['length']}"
    if (folder/'summary.json').exists():
        saved=json.loads((folder/'summary.json').read_text())
        if saved['config']!=config:
            raise ValueError('Existing run has different config')
        if saved['status']=='completed':
            if digest(folder/'best.pth')!=saved['checkpoint_sha256']:
                raise ValueError('Existing checkpoint hash mismatch')
            return dict(saved,run_id=folder.name)
    result=train(config,data_dir,folder,device,threads,resume=(folder/'last.pth').exists())
    return dict(result,run_id=folder.name)


def run_experiment(stage,output,device='cpu',threads=4,workers=1):
    if workers < 1 or (workers > 1 and device != 'cpu'):
        raise ValueError('Parallel runs require explicit CPU device and workers >= 1')
    output.mkdir(parents=True,exist_ok=True)
    plan_path=output/'experiment-plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=PLAN:
        raise ValueError('Plan changed; use a new experiment directory before test access')
    write_json(plan_path,PLAN)
    if stage=='pilots':
        candidates=[]
        for model in ['swin']:
            for length in PLAN['lengths']:
                for augmentation in PLAN['augmentation_candidates']:
                    name=f'pilot-{model}-l{length}-aug{int(augmentation)}-seed42'
                    config=config_for(model,length,augmentation,42,False)
                    candidates.append(run_one(config,output/'runs'/name,device,threads))
        winners={model:min([c for c in candidates if c['config']['model']==model],key=ranked)
                 for model in ['swin']}
        lock=dict(plan_sha256=digest(plan_path),test_accessed=False,selection_rule=PLAN['selection_rule'],
                  candidates=candidates,winners={m:w['run_id'] for m,w in winners.items()},
                  configs={m:w['config'] for m,w in winners.items()})
        path=output/'configuration-lock.json'
        if path.exists() and json.loads(path.read_text())!=lock:
            raise ValueError('Existing configuration lock differs')
        write_json(path,lock)
        print('Locked validation-selected configs: '+json.dumps(lock['winners']),flush=True)
    elif stage=='official':
        lock=json.loads((output/'configuration-lock.json').read_text())
        if lock['plan_sha256']!=digest(plan_path):
            raise ValueError('Configuration lock/plan mismatch')
        runs=[]
        jobs=[]
        for model in ['swin']:
            chosen=lock['configs'][model]
            for seed in PLAN['seeds']:
                c=config_for(model,chosen['length'],chosen['augmentation'],seed,True)
                name=f"official-{model}-l{c['length']}-aug{int(c['augmentation'])}-seed{seed}"
                jobs.append((c,output/'runs'/name,device,threads))
        if workers == 1:
            runs=[run_one(*job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures=[executor.submit(run_one,*job) for job in jobs]
                runs=[future.result() for future in futures]
        deployment=min([r for r in runs if r['config']['model']=='swin'],key=ranked)
        selection=dict(plan_sha256=digest(plan_path),configuration_lock_sha256=digest(output/'configuration-lock.json'),
                       selection_rule=PLAN['selection_rule'],test_accessed_for_selection=False,
                       deployment_run=deployment['run_id'],checkpoint_sha256=deployment['checkpoint_sha256'],
                       runs=runs)
        path=output/'selection.json'
        if path.exists() and json.loads(path.read_text())!=selection:
            raise ValueError('Selection changed; do not overwrite locked test experiment')
        write_json(path,selection)
        print('Locked deployment checkpoint: '+deployment['run_id'],flush=True)
    else:
        raise ValueError('Unknown stage')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['pilots','official'],required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/drawai/swin-pendigits-v2')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args(); run_experiment(args.stage,args.output_dir,args.device,args.threads,args.workers)


if __name__=='__main__':
    main()
