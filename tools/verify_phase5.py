"""Verify six fitted checkpoints, saved predictions, frozen selection and UI."""
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageChops
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai import paint_train as p, paint_dataset as data
from training.drawai.private_data import load, dump
from training.drawai.metrics import classification_metrics
from training.drawai.models import build_model
from training.drawai.dinov3_model import DinoDigitClassifier
OUT=p.DEFAULT_OUT


def read_predictions(path,expected_split):
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={'sample_ids','labels','logits'}:raise ValueError('Unexpected prediction schema')
        ids=archive['sample_ids'];y=archive['labels'];logits=archive['logits']
    assert ids.shape==y.shape==(1000,) and logits.shape==(1000,10) and logits.dtype==np.float32
    assert y.dtype==np.int64 and np.isfinite(logits).all() and len(set(ids))==1000
    people=range(26,31) if expected_split=='validation' else range(31,36)
    expected={f'paint35_p{person:03}_d{label}_r{rep:02}':label for person in people for label in range(10) for rep in range(1,21)}
    assert set(ids)==set(expected) and [expected[sid] for sid in ids]==y.tolist()
    metrics=classification_metrics(y,logits.argmax(1))
    metrics['loss']=float(torch.nn.functional.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y)))
    rows=[dict(sample_id=sid,person_id=int(sid[9:12])) for sid in ids]
    return metrics,p.grouped_metrics(y,logits,rows),(ids,y,logits)


def main():
    p.seed_all(42);selection,runs=p.checked_selection(OUT)
    receipt=load(OUT/'evaluation/test-access.json');final=load(OUT/'evaluation/final-test.json')
    assert receipt['selection_sha256']==selection['selection_sha256']==final['selection_sha256']
    assert final['evaluation_sha256']==p.sha(p.canonical({k:v for k,v in final.items() if k!='evaluation_sha256'}))
    assert datetime.fromisoformat(selection['created_at'])<=datetime.fromisoformat(receipt['opened_at'])<=datetime.fromisoformat(final['evaluated_at'])
    assert final['candidate_default_unchanged']==selection['candidate_default'] and final['test_used_for_tuning'] is False
    _,devrows=data.verify_lock(p.DEFAULT_DATA,p.LOCK_HASH)
    valrows=[r for r in devrows if r['split']=='validation']
    valx=np.stack([p.paint.swin_rows(data.check_image(p.DEFAULT_DATA/'development',r)) for r in valrows])
    pretrained=DinoDigitClassifier.from_pretrained(p.DEFAULT_WEIGHTS)
    cachemeta=load(p.DEFAULT_CACHE/'development.json')
    assert p.file_sha(p.DEFAULT_CACHE/'development.npy')==cachemeta['sha256']
    assert cachemeta['key']==p.cache_key(devrows,'development')
    features=np.load(p.DEFAULT_CACHE/'development.npy',allow_pickle=False)
    assert features.shape==(5000,768) and features.dtype==np.float32 and np.isfinite(features).all()
    # Independently recompute ten full-backbone features from locked PNGs.
    probe_indices=list(range(5))+[i for i,r in enumerate(devrows) if r['split']=='validation'][:5]
    inputs=torch.from_numpy(np.stack([p.normalize_rgb(p.paint.dino_rgb(data.check_image(
        p.DEFAULT_DATA/'development',devrows[i]))) for i in probe_indices]))
    with torch.inference_mode():actual_features=pretrained.forward_features(inputs).numpy()
    assert np.allclose(actual_features,features[probe_indices],rtol=1e-4,atol=1e-4)
    valfeatures=features[np.asarray([r['split']=='validation' for r in devrows])]
    reference_val=None;reference_test=None;checked=[]
    for run in runs:
        folder=OUT/'runs'/f"{run['family']}-seed{run['seed']}"
        assert p.file_sha(folder/'history.json')==run['history_sha256']
        history=load(folder/'history.json');config=p.plan()['swin' if run['family']=='swin1d-image' else 'dinov3']
        assert len(history)==run['epochs_completed']<=config['epochs']
        best=min(history,key=lambda e:(-e['validation_macro_f1'],e['validation_loss'],e['epoch']))
        assert best['epoch']==run['best']['epoch'] and best['validation_macro_f1']==run['validation']['macro_f1']
        metrics,by_writer,pred=read_predictions(folder/'validation-predictions.npz','validation')
        assert metrics==run['validation'] and by_writer==run['validation_by_writer']
        if reference_val is None:reference_val=pred[:2]
        else:assert all(np.array_equal(a,b) for a,b in zip(reference_val,pred[:2]))
        saved=torch.load(folder/run['checkpoint'],map_location='cpu',weights_only=True)
        assert saved['family']==run['family'] and saved['seed']==run['seed'] and saved['lock_sha256']==p.LOCK_HASH
        assert saved['trained'] is True and saved['release_ready'] is False and saved['test_used_for_selection'] is False
        assert saved['plan_sha256']==selection['plan_sha256'] and saved['preprocessing']==p.paint.contract()
        if run['family']=='swin1d-image':
            assert saved['model_kwargs']==p.swin_image_config()
            model=build_model('swin1d_image');model.load_state_dict(saved['model_state_dict'],strict=True)
            _,replay=p.predict(model,valx,np.asarray([int(r['label']) for r in valrows],dtype=np.int64))
            assert [r['sample_id'] for r in valrows]==pred[0].tolist()
        else:
            assert saved['provenance']['weights_sha256']==p.TIMM_WEIGHTS_SHA256 and saved['provenance']['pretrained_loaded'] is True
            # Ensure frozen backbone stayed exactly equal to the real pinned weights.
            for name,tensor in pretrained.backbone.state_dict().items():
                assert torch.equal(tensor,saved['model_state_dict']['backbone.'+name]),name
            head=torch.nn.Linear(768,10)
            head.load_state_dict({k[5:]:v for k,v in saved['model_state_dict'].items() if k.startswith('head.')},strict=True)
            _,replay=p.predict(head,valfeatures,np.asarray([int(r['label']) for r in valrows],dtype=np.int64))
            assert [r['sample_id'] for r in valrows]==pred[0].tolist()
        assert np.allclose(replay,pred[2],rtol=1e-6,atol=1e-6)
        test=next(r for r in final['results'] if r['family']==run['family'] and r['seed']==run['seed'])
        assert test['checkpoint_sha256']==run['checkpoint_sha256']
        assert p.file_sha(OUT/'evaluation'/test['predictions_file'])==test['predictions_sha256']
        testmetrics,testwriters,testpred=read_predictions(OUT/'evaluation'/test['predictions_file'],'test')
        assert testmetrics==test['metrics'] and testwriters==test['by_writer']
        if reference_test is None:reference_test=testpred[:2]
        else:assert all(np.array_equal(a,b) for a,b in zip(reference_test,testpred[:2]))
        assert not set(pred[0]) & set(testpred[0])
        checked.append(dict(family=run['family'],seed=run['seed'],best_epoch=run['best']['epoch'],
            checkpoint_sha256=run['checkpoint_sha256'],validation=metrics,test=testmetrics,
            training_seconds=run['training_seconds'],parameters=run['parameters'],trainable_parameters=run['trainable_parameters']))
    assert len(checked)==6
    for family in p.FAMILIES:
        for metric in ('accuracy','macro_f1'):
            values=[r['test'][metric] for r in checked if r['family']==family]
            assert final['aggregate'][family][metric]==dict(mean=float(np.mean(values)),std_population=float(np.std(values,ddof=0)))
        bench=load(OUT/f'benchmark-{family}.json')
        assert bench['checkpoint_sha256']==selection['winners'][family]['sha256'] and bench['test_data_used'] is False
        assert bench['iterations']==50 and bench['milliseconds']['p95']>0
    protected=load(ROOT/'artifacts/drawai/phase4/protected-files.json')
    for name,digest in protected.items():assert p.file_sha(ROOT/name)==digest,name
    design={}
    for name in ('desktop','mobile','root-desktop','root-mobile'):
        before=Image.open(ROOT/f'artifacts/drawai/phase0/ui/{name.removeprefix("root-")}.png').convert('RGB')
        current=Image.open(OUT/f'ui/{name}.png').convert('RGB')
        assert before.size==current.size and ImageChops.difference(before,current).getbbox() is None
        design[name]=True
    log=(OUT/'python-tests.log').read_text(encoding='utf-8-sig')
    assert re.search(r'\nOK\s*$',log);count=int(re.search(r'Ran (\d+) tests',log)[1]);assert count>=113
    stats=load(OUT/'product-regression.json')['stats'];assert stats['expected']==12 and all(stats[k]==0 for k in ('unexpected','flaky','skipped'))
    assert 'built in' in (OUT/'build.log').read_text(encoding='utf-8-sig')
    result=dict(phase='P5',status='DONE_LOCAL_CPU_TRAINING_AND_FINAL_COMPARISON',phase_complete=True,
        lock_sha256=p.LOCK_HASH,plan_sha256=selection['plan_sha256'],selection_sha256=selection['selection_sha256'],
        evaluation_sha256=final['evaluation_sha256'],real_runs=6,seeds=[42,43,44],
        train_samples=4000,validation_samples=1000,test_samples=1000,
        checkpoints_checked=checked,validation_logits_replayed_all_runs=True,dino_backbone_unchanged_exactly=True,
        full_backbone_feature_cache_probe_images=10,
        selected_by_validation=selection['candidate_default'],test_used_for_selection=False,product_default_changed=False,
        onnx_exported_in_phase5=False,python_tests_passed=count,browser_tests_passed=12,protected_files_verified=len(protected),design=design,
        code_sha256_at_verification={n:p.file_sha(ROOT/n) for n in ['training/drawai/paint_train.py','training/drawai/models.py','tools/verify_phase5.py']},
        limitations=['37 near groups across main splits; no clean leakage-free/generalization claim',
                    'Human origin/person IDs/permissions owner-attested; not independently proven',
                    'CPU only; DINO frozen-feature linear probe, Swin from scratch; pretraining/compute differ',
                    'Test original publicly available; procedural isolation, not secret holdout',
                    'Per-device unavailable; five writers per validation/test split, seeds are not independent datasets',
                    'Checkpoint exports/runtime integration and canvas-to-image parity remain P6/P7'])
    dump(OUT/'verification.json',result)
    print('P5 PASS:6 fitted checkpoints, all saved metrics and validation logits checked; fixed selection before test; unchanged UI.')


if __name__=='__main__':main()
