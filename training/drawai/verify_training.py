"""Independently audit D3 configs, selection, predictions and published metrics."""
import argparse
import json
from pathlib import Path

import numpy as np

from .experiment import PLAN, config_for, ranked
from .prepare_data import ROOT, digest, write_json
from .train import load_split, tensor_hash


def verify_selected_swin(output):
    """Audit the deployed Swin without requiring archived CNN checkpoints.

    Historical experiment metadata stays immutable. This is explicitly a
    selected-Swin audit, not a fresh verification of the archived CNN runs.
    """
    import torch
    from .models import build_model, swin_config
    from .preprocessing import contract
    from .train import predict

    output = Path(output).resolve()
    plan = json.loads((output/'experiment-plan.json').read_text())
    lock = json.loads((output/'configuration-lock.json').read_text())
    selection = json.loads((output/'selection.json').read_text())
    evaluation = json.loads((output/'evaluation/evaluation.json').read_text())
    if plan.get('version') not in ('drawai-experiment-v1', PLAN['version']):
        raise ValueError('Unknown experiment plan')
    if (selection['plan_sha256'] != digest(output/'experiment-plan.json') or
        lock['plan_sha256'] != selection['plan_sha256'] or
        selection['configuration_lock_sha256'] != digest(output/'configuration-lock.json') or
        evaluation['selection_sha256'] != digest(output/'selection.json') or
        json.loads((output/'test-access.json').read_text())['selection_sha256'] != evaluation['selection_sha256'] or
        selection['test_accessed_for_selection'] is not False or lock['test_accessed'] is not False):
        raise ValueError('Selection/provenance chain mismatch')
    swins = [r for r in selection['runs'] if r['config']['model'] == 'swin']
    if len(swins) != 3 or {r['config']['seed'] for r in swins} != {42,43,44}:
        raise ValueError('Need three locked Swin seeds')
    winner = min(swins, key=ranked)
    if winner['run_id'] != selection['deployment_run'] or winner['run_id'] != evaluation['deployment_run']:
        raise ValueError('Selected Swin does not match validation ranking')
    pilots = [r for r in lock['candidates'] if r['config']['model'] == 'swin']
    if len(pilots) != 4 or min(pilots, key=ranked)['run_id'] != lock['winners']['swin']:
        raise ValueError('Pilot validation ranking mismatch')
    if any(r['config']['length'] != lock['configs']['swin']['length'] or
           r['config']['augmentation'] != lock['configs']['swin']['augmentation'] for r in swins):
        raise ValueError('Official Swin differs from locked configuration')
    folder = output/'runs'/winner['run_id']
    source = folder/'best.pth'
    summary = json.loads((folder/'summary.json').read_text())
    if digest(source) != winner['checkpoint_sha256'] or digest(source) != selection['checkpoint_sha256'] or digest(source) != summary['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash mismatch')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    length = winner['config']['length']
    if (saved['trained'] is not True or saved['test_used_for_training'] or
        summary['status'] != winner['status'] or summary['status'] != 'completed' or
        summary['test_used_for_training'] or saved['config'] != winner['config'] or
        summary['config'] != saved['config'] or saved['model_kwargs'] != swin_config(length) or
        saved['preprocessing'] != contract(length)):
        raise ValueError('Selected checkpoint metadata mismatch')
    history = json.loads((folder/'history.json').read_text())
    best = min(history, key=lambda h: (-h['validation']['macro_f1'],h['validation']['loss'],h['epoch']))
    if (len(history) != summary['epochs_completed'] or best['epoch'] != winner['best']['epoch'] or
        best['epoch'] != saved['epoch'] or best['validation']['macro_f1'] != winner['best']['macro_f1'] or
        best['validation']['loss'] != winner['best']['loss'] or saved['validation'] != best['validation']):
        raise ValueError('Selected best epoch does not match validation history')
    data_dir = ROOT/f'data/processed/pendigits-drawai-v1/l{length}'
    for split in ('train', 'validation'):
        if tensor_hash(*load_split(data_dir,split,length)) != saved['data_hashes'][split] or saved['data_hashes'][split] != winner['data_hashes'][split]:
            raise ValueError('Training data provenance mismatch')
    if saved['labels'] != json.loads((data_dir/'labels.json').read_text()):
        raise ValueError('Checkpoint labels mismatch')
    torch.set_num_threads(4)
    model = build_model('swin',length).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    validation = load_split(data_dir,'validation',length)
    observed, logits = predict(model,validation,winner['config']['batch_size'],torch.device('cpu'))
    if abs(observed['macro_f1']-winner['best']['macro_f1']) > 1e-12 or abs(observed['loss']-winner['best']['loss']) > 1e-5:
        raise ValueError('Validation metrics cannot be reproduced')
    with np.load(folder/'best-validation.npz',allow_pickle=False) as expected:
        np.testing.assert_array_equal(expected['sample_ids'],validation[2])
        np.testing.assert_array_equal(expected['y'],validation[1])
        np.testing.assert_allclose(logits,expected['logits'],atol=1e-5,rtol=1e-4)
    report = dict(status='passed',scope='selected-swin-only',historical_other_models_reaudited=False,
                  deployment_run=winner['run_id'], checkpoint_sha256=digest(source),
                  selection_sha256=digest(output/'selection.json'), validation_samples=len(validation[1]),
                  validation=observed, archived_checkpoints_required=False, test_opened=False)
    write_json(output/'selected-swin-verification.json',report)
    print('PASS: selected Swin provenance, strict weights, validation and selection chain; no archived CNN checkpoint required.')
    return report


def independent_metrics(truth,predicted):
    matrix=[[0]*10 for _ in range(10)]
    for true,pred in zip(truth,predicted): matrix[int(true)][int(pred)]+=1
    f1=[]; per=[]
    for label in range(10):
        correct=matrix[label][label]; support=sum(matrix[label]); selected=sum(row[label] for row in matrix)
        precision=correct/selected if selected else 0
        recall=correct/support if support else 0
        score=2*correct/(support+selected) if support+selected else 0
        f1.append(score); per.append((precision,recall,score,support))
    return matrix,sum(matrix[i][i] for i in range(10))/len(truth),sum(f1)/10,per


def verify(output):
    plan=json.loads((output/'experiment-plan.json').read_text())
    if plan.get('version') == 'drawai-experiment-v1':
        return verify_selected_swin(output)
    assert plan==PLAN
    lock=json.loads((output/'configuration-lock.json').read_text())
    selection=json.loads((output/'selection.json').read_text())
    evaluation=json.loads((output/'evaluation/evaluation.json').read_text())
    assert lock['plan_sha256']==digest(output/'experiment-plan.json')
    assert selection['configuration_lock_sha256']==digest(output/'configuration-lock.json')
    assert evaluation['selection_sha256']==digest(output/'selection.json')
    assert json.loads((output/'test-access.json').read_text())['selection_sha256']==evaluation['selection_sha256']
    candidates=lock['candidates']; assert len(candidates)==4
    for family in ['swin']:
        pilots=[c for c in candidates if c['config']['model']==family]
        assert {(p['config']['length'],p['config']['augmentation']) for p in pilots}=={(l,a) for l in [64,128] for a in [False,True]}
        assert lock['winners'][family]==min(pilots,key=ranked)['run_id']
    runs=selection['runs']; assert len(runs)==3
    assert {(r['config']['model'],r['config']['seed']) for r in runs}=={(m,s) for m in ['swin'] for s in [42,43,44]}
    winner=min([r for r in runs if r['config']['model']=='swin'],key=ranked)
    assert winner['run_id']==selection['deployment_run']==evaluation['deployment_run']
    data={}
    for length in {r['config']['length'] for r in runs}:
        folder=ROOT/f'data/processed/pendigits-drawai-v1/l{length}'
        data[length]={s:load_split(folder,s,length) for s in ['train','validation','test']}
    for run in candidates+runs:
        folder=output/'runs'/run['run_id']; saved=json.loads((folder/'summary.json').read_text())
        assert run['status']==saved['status']=='completed'
        assert digest(folder/'best.pth')==run['checkpoint_sha256']==saved['checkpoint_sha256']
        history=json.loads((folder/'history.json').read_text()); assert len(history)==saved['epochs_completed']
        best=min(history,key=lambda h:(-h['validation']['macro_f1'],h['validation']['loss'],h['epoch']))
        assert best['epoch']==run['best']['epoch']
        assert best['validation']['macro_f1']==run['best']['macro_f1']
        assert best['validation']['loss']==run['best']['loss']
        assert not saved['test_used_for_training']
    checks=[]
    for run in runs:
        name=run['run_id']; c=run['config']; entry=next(r for r in evaluation['results'] if r['run_id']==name)
        assert c==config_for(c['model'],lock['configs'][c['model']]['length'],lock['configs'][c['model']]['augmentation'],c['seed'],True)
        for split in ['train','validation']:
            assert tensor_hash(*data[c['length']][split])==run['data_hashes'][split]
        test=data[c['length']]['test']
        with np.load(output/f'evaluation/{name}-predictions.npz',allow_pickle=False) as p:
            assert p['logits'].shape==(3498,10) and np.isfinite(p['logits']).all()
            np.testing.assert_array_equal(p['y_true'],test[1]); np.testing.assert_array_equal(p['sample_ids'],test[2])
            np.testing.assert_array_equal(p['y_pred'],p['logits'].argmax(1))
            matrix,accuracy,f1,per=independent_metrics(p['y_true'],p['y_pred'])
        assert matrix==entry['test']['confusion_matrix']
        assert abs(accuracy-entry['test']['accuracy'])<1e-12 and abs(f1-entry['test']['macro_f1'])<1e-12
        for expected,observed in zip(per,entry['test']['per_class']):
            assert all(abs(expected[i]-observed[key])<1e-12 for i,key in enumerate(['precision','recall','f1','support']))
        assert entry['test_sha256']==tensor_hash(*test)
        checks.append(dict(run_id=name,test_samples=3498,accuracy=accuracy,macro_f1=f1))
    for family in ['swin']:
        group=[c for c in checks if family in c['run_id']]
        for metric in ['accuracy','macro_f1']:
            assert abs(float(np.mean([c[metric] for c in group]))-evaluation['aggregate'][family][metric]['mean'])<1e-12
            assert abs(float(np.std([c[metric] for c in group],ddof=1))-evaluation['aggregate'][family][metric]['sample_std'])<1e-12
    report=dict(status='passed',pilot_runs=4,official_runs=3,verified_test_predictions=3*3498,
                selected_using_validation=True,selection_sha256=digest(output/'selection.json'),
                evaluation_sha256=digest(output/'evaluation/evaluation.json'),checks=checks)
    write_json(output/'verification.json',report)
    print('PASS: audited 4 Swin pilots, 3 official runs and 10,494 test predictions independently.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/drawai/d3-local-v1')
    args=parser.parse_args(); verify(args.output_dir)


if __name__=='__main__': main()
