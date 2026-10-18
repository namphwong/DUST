"""Evaluate locked D3 runs; official test cannot select configs/checkpoints."""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .experiment import PLAN, config_for, ranked
from .models import build_model
from .prepare_data import ROOT, digest, parse_unipen, write_json
from .train import load_split, predict, tensor_hash


def plots(folder, run_id, history, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    epochs=[h['epoch'] for h in history]
    axes[0].plot(epochs,[h['train_loss'] for h in history],label='train')
    axes[0].plot(epochs,[h['validation']['loss'] for h in history],label='validation')
    axes[0].set(xlabel='Epoch',ylabel='Cross entropy'); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(epochs,[h['validation']['macro_f1'] for h in history],label='validation macro-F1')
    axes[1].set(xlabel='Epoch',ylabel='Macro-F1'); axes[1].grid(alpha=.2)
    fig.suptitle(run_id); fig.tight_layout(); fig.savefig(folder/f'{run_id}-learning-curves.png',dpi=140); plt.close(fig)
    matrix=np.asarray(metrics['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(6,6)); ax.imshow(matrix,cmap='Blues')
    for i in range(10):
        for j in range(10):
            if matrix[i,j]: ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black',fontsize=8)
    ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted',ylabel='True',title=run_id)
    fig.tight_layout(); fig.savefig(folder/f'{run_id}-confusion-matrix.png',dpi=140); plt.close(fig)


def evaluate(output,threads=4):
    torch.set_num_threads(threads)
    selection_path=output/'selection.json'
    selection=json.loads(selection_path.read_text())
    if selection['plan_sha256']!=digest(output/'experiment-plan.json') or selection['configuration_lock_sha256']!=digest(output/'configuration-lock.json'):
        raise ValueError('Plan/configuration changed after checkpoint selection')
    if json.loads((output/'experiment-plan.json').read_text()) != PLAN:
        raise ValueError('Experiment plan differs; historical D3 artifacts are read-only')
    runs=selection['runs']
    if len(runs)!=3 or {(r['config']['model'],r['config']['seed']) for r in runs}!={(m,s) for m in ['swin'] for s in [42,43,44]}:
        raise ValueError('Need exactly three official Swin completed runs')
    config_lock=json.loads((output/'configuration-lock.json').read_text())
    for run in runs:
        chosen=config_lock['configs'][run['config']['model']]
        expected=config_for(run['config']['model'],chosen['length'],chosen['augmentation'],run['config']['seed'],True)
        if run['config']!=expected or run['status']!='completed' or run['test_used_for_training']:
            raise ValueError('Official config/status contract mismatch')
        if digest(output/'runs'/run['run_id']/'best.pth')!=run['checkpoint_sha256']:
            raise ValueError('Locked checkpoint changed')
    deployed=min([r for r in runs if r['config']['model']=='swin'],key=ranked)
    if deployed['run_id']!=selection['deployment_run'] or deployed['checkpoint_sha256']!=selection['checkpoint_sha256']:
        raise ValueError('Deployment checkpoint was not selected using declared validation rule')
    gate=output/'test-access.json'
    selection_hash=digest(selection_path)
    if gate.exists() and json.loads(gate.read_text())['selection_sha256']!=selection_hash:
        raise ValueError('Selection changed after official test was opened')
    folder=output/'evaluation'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'evaluation.json').exists():
        prior=json.loads((folder/'evaluation.json').read_text())
        if prior['selection_sha256']!=selection_hash:
            raise ValueError('Existing evaluation belongs to another selection')
        print('Evaluation already completed; returning immutable existing results.',flush=True)
        return prior
    # Lock all validation decisions BEFORE loading any official test tensor.
    if not gate.exists():
        write_json(gate,dict(selection_sha256=selection_hash,opened_utc=datetime.now(timezone.utc).isoformat(),
                            declaration='All three Swin run configs and best checkpoints locked before official test access'))
    results=[]
    for run in runs:
        name=run['run_id']; c=run['config']; source=output/'runs'/name
        saved=torch.load(source/'best.pth',map_location='cpu',weights_only=True)
        if saved['config']!=c or saved['trained'] is not True or saved['test_used_for_training']:
            raise ValueError('Checkpoint metadata differs from locked run')
        model=build_model(c['model'],c['length']).eval(); model.load_state_dict(saved['model_state_dict'],strict=True)
        data_dir=ROOT/f"data/processed/pendigits-drawai-v1/l{c['length']}"
        val=load_split(data_dir,'validation',c['length'])
        observed_val,val_logits=predict(model,val,c['batch_size'],torch.device('cpu'))
        if abs(observed_val['macro_f1']-run['best']['macro_f1'])>1e-12:
            raise ValueError('Validation score not reproduced from checkpoint')
        with np.load(source/'best-validation.npz',allow_pickle=False) as expected:
            np.testing.assert_array_equal(expected['y'],val[1]); np.testing.assert_array_equal(expected['sample_ids'],val[2])
            np.testing.assert_allclose(val_logits,expected['logits'],atol=1e-5,rtol=1e-4)
        test=load_split(data_dir,'test',c['length'])
        metrics,logits=predict(model,test,c['batch_size'],torch.device('cpu'))
        predictions=logits.argmax(1)
        np.savez_compressed(folder/f'{name}-predictions.npz',logits=logits,y_true=test[1],y_pred=predictions,sample_ids=test[2])
        history=json.loads((source/'history.json').read_text())
        plots(folder,name,history,metrics)
        failures=np.flatnonzero(predictions!=test[1])
        examples=[dict(sample_id=str(test[2][i]),true=int(test[1][i]),predicted=int(predictions[i])) for i in failures[:30]]
        result=dict(run_id=name,config=c,checkpoint_sha256=run['checkpoint_sha256'],
                    parameters=run['parameters'],best_epoch=run['best']['epoch'],validation=observed_val,
                    test=metrics,test_sha256=tensor_hash(*test),training_seconds=run['training_seconds'],
                    epochs_completed=run['epochs_completed'],errors=examples)
        write_json(folder/f'{name}-metrics.json',result); results.append(result)
        print(f"TEST {name}: accuracy={metrics['accuracy']:.6f}, macro_f1={metrics['macro_f1']:.6f}",flush=True)
    aggregate={}
    for family in ['swin']:
        group=[r for r in results if r['config']['model']==family]
        aggregate[family]={metric:dict(mean=float(np.mean([r['test'][metric] for r in group])),
                                      sample_std=float(np.std([r['test'][metric] for r in group],ddof=1)))
                           for metric in ['accuracy','macro_f1']}
    chosen=next(r for r in results if r['run_id']==selection['deployment_run'])
    final=dict(status='passed',selection_sha256=selection_hash,deployment_run=chosen['run_id'],
               selection_rule=PLAN['selection_rule'],results=results,aggregate=aggregate,
               target_accuracy=0.98,target_macro_f1=0.97,
               target_met=chosen['test']['accuracy']>=0.98 and chosen['test']['macro_f1']>=0.97,
               writer_id_verified=False,validation_limitation='Stratified sample split may share writers with train',
               test_used_for_selection=False,device='cpu',torch_version=str(torch.__version__),python=platform.python_version())
    write_json(folder/'evaluation.json',final)
    error_grid(folder,chosen)
    return final


def error_grid(folder,chosen):
    raw=ROOT/'data/raw/pendigits/pendigits-orig.tes'
    if not raw.exists():
        return # Kaggle bundles may omit raw data; IDs remain in error manifest.
    source={s['id']:s for s in parse_unipen(raw.read_text(),raw.name)}
    errors=chosen['errors'][:10]
    if not errors: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,5,figsize=(14,6))
    for ax,error in zip(axes.flat,errors):
        sample=source[error['sample_id']]
        for stroke in sample['strokes']:
            ax.plot([p['x'] for p in stroke],[-p['y'] for p in stroke])
        ax.invert_yaxis(); ax.set_aspect('equal',adjustable='datalim')
        ax.set_title(f"True {error['true']} / predicted {error['predicted']}\nsource #{sample['source_index']}")
    for ax in list(axes.flat)[len(errors):]: ax.axis('off')
    fig.suptitle('Validation-selected Swin / first test errors in source order')
    fig.tight_layout(); fig.savefig(folder/'deployment-error-examples.png',dpi=140); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/drawai/swin-pendigits-v2')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(); evaluate(args.output_dir,args.threads)


if __name__=='__main__': main()
