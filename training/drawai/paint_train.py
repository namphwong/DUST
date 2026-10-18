"""P5 Paint35 CPU experiment: fixed plan, three seeds, validation selection.

Training reads only the verified development package. Test evaluation is a
separate explicit command and requires an immutable validation selection first.
Frozen DINO features are cached privately with hashes; full checkpoints include
the pretrained backbone and fitted head. No release/ONNX export in this phase.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gc
import io
import os
from pathlib import Path
import platform
import random
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import paint_dataset as data, paint_data as paint
from .private_data import ROOT, canonical, sha, load, dump, new_output
from .private_dataset import safe_file
from .models import build_model, swin_image_config
from .dinov3_model import DinoDigitClassifier, TIMM_WEIGHTS_SHA256
from .metrics import classification_metrics
from .raster import normalize_rgb

LOCK_HASH = '2c3a86c5be2b2e0f952a9c3390d56785473a82634338cec805903cbb1b3ef47a'
DEFAULT_DATA = ROOT/'data/private/locked/paint35-main-v1'
DEFAULT_OUT = ROOT/'artifacts/drawai/phase5-image64'
DEFAULT_WEIGHTS = ROOT/'models/pretrained/dinov3-vits16-timm-qkvb'
DEFAULT_CACHE = ROOT/'data/private/features/paint35-main-v1-dinov3-vits16'
FAMILIES = ('swin1d-image','dinov3-vits16')


def plan():
    return dict(version='drawai-paint-experiment-v1',dataset='paint35-main-v1',lock_sha256=LOCK_HASH,
        source_sha256=paint.SOURCE_SHA256,preprocessing=paint.contract(),seeds=[42,43,44],
        device='cpu',threads=4,batch_size=128,augmentation=False,auxiliary_digit_data=False,
        swin=dict(architecture=swin_image_config(),initialization='scratch',epochs=30,patience=8,
                  lr=.001,weight_decay=.01,warmup_epochs=2,gradient_clip=1.0),
        dinov3=dict(backbone_weights_sha256=TIMM_WEIGHTS_SHA256,backbone_frozen=True,
                    pooling='CLS concatenated with mean patches; exclude four registers',
                    head='Linear(768,10)',epochs=100,patience=15,lr=.003,weight_decay=.01,
                    warmup_epochs=2,gradient_clip=1.0),
        checkpoint_rule='Max validation macro-F1, min validation cross-entropy, earlier epoch',
        selection_rule='Max validation macro-F1, min validation loss, lower seed, lexical family',
        tuning_budget='One configuration per family, three seeds; no partial backbone fine-tuning or test tuning',
        final_test='Only after all six runs complete and selection is frozen; all six runs once, no retuning',
        similarity_independence_verified=False,main_cross_split_near_groups=37,
        device_metadata_available=False,compare_architecture_only=False)


def file_sha(path):return sha(Path(path).read_bytes())


def seed_all(seed, threads=4):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False


def runtime():
    import PIL, timm
    return dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                pillow=PIL.__version__,timm=timm.__version__,device='cpu',threads=torch.get_num_threads())


def atomic_save(path, value):
    temporary=Path(str(path)+'.partial');torch.save(value,temporary);temporary.replace(path)


def sealed(value, field):
    result=dict(value);result[field]=sha(canonical(result));return result


def private_output(output):
    output=Path(output).resolve()
    for forbidden in (ROOT/'web/public',ROOT/'web/dist'):
        if output==forbidden or forbidden in output.parents:
            raise ValueError('Experiment output cannot be public/dist')
    return output


def read_plan(output):
    output=private_output(output)
    value=load(Path(output)/'experiment-plan.json')
    if value != plan():raise ValueError('Experiment plan changed; use a new experiment version')
    return file_sha(Path(output)/'experiment-plan.json')


def development(dataset):
    _,rows=data.verify_lock(dataset,LOCK_HASH)
    pixels=[]
    for row in rows:pixels.append(data.check_image(Path(dataset)/'development',row))
    return np.stack(pixels),np.asarray([int(r['label']) for r in rows],dtype=np.int64),rows


def cache_key(rows, partition):
    return dict(version='drawai-dino-feature-cache-v1',lock_sha256=LOCK_HASH,partition=partition,
        backbone_sha256=TIMM_WEIGHTS_SHA256,preprocessing=paint.contract(),
        code_sha256={n:file_sha(ROOT/n) for n in ['training/drawai/dinov3_model.py','training/drawai/paint_data.py','training/drawai/raster.py']},
        sample_ids=[r['sample_id'] for r in rows],labels=[int(r['label']) for r in rows],
        pixel_hashes=[r['image_sha256'] for r in rows],features_shape=[len(rows),768],dtype='float32')


def frozen_features(model,pixels,rows,folder,partition,batch_size=16):
    folder=Path(folder).resolve()
    for forbidden in (ROOT/'web/public',ROOT/'web/dist'):
        if folder==forbidden or forbidden in folder.parents:raise ValueError('Feature cache cannot be public')
    key=cache_key(rows,partition);meta_path=folder/f'{partition}.json';path=folder/f'{partition}.npy'
    if meta_path.exists():
        meta=load(meta_path)
        if meta.get('key')!=key or file_sha(path)!=meta.get('sha256'):raise ValueError('Stale/tampered feature cache')
        features=np.load(path,allow_pickle=False)
        if features.shape!=(len(rows),768) or features.dtype!=np.float32 or not np.isfinite(features).all():raise ValueError('Invalid feature cache')
        return features,meta
    if path.exists():raise ValueError('Incomplete feature cache; preserve/debug before rebuilding')
    model.eval();folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter();parts=[]
    with torch.inference_mode():
        for start in range(0,len(pixels),batch_size):
            inputs=torch.from_numpy(np.stack([normalize_rgb(paint.dino_rgb(x)) for x in pixels[start:start+batch_size]]))
            features=model.forward_features(inputs).cpu().numpy()
            if features.shape!=(len(inputs),768) or not np.isfinite(features).all():raise ValueError('Invalid pretrained features')
            parts.append(features)
            if start//batch_size % 20 == 0:print(f'DINO {partition} features {min(start+batch_size,len(pixels))}/{len(pixels)}',flush=True)
    result=np.ascontiguousarray(np.concatenate(parts),dtype=np.float32)
    temp=Path(str(path)+'.partial')
    with temp.open('wb') as handle:np.save(handle,result,allow_pickle=False)
    temp.replace(path)
    meta=dict(key=key,sha256=file_sha(path),seconds=time.perf_counter()-started,feature_extraction_batch_size=batch_size,
              pretrained_provenance=model.provenance,source_pixels_used=True)
    dump(meta_path,meta);return result,meta


@torch.inference_mode()
def predict(model, x, y, batch_size=128):
    model.eval();parts=[]
    for start in range(0,len(x),batch_size):
        value=model(torch.from_numpy(x[start:start+batch_size])).cpu().numpy()
        if value.shape!=(min(batch_size,len(x)-start),10) or not np.isfinite(value).all():raise ValueError('Invalid logits')
        parts.append(value)
    logits=np.concatenate(parts);metrics=classification_metrics(y,logits.argmax(1))
    metrics['loss']=float(nn.functional.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y)))
    return metrics,logits


def grouped_metrics(y,logits,rows):
    result={}
    for person in sorted({int(r['person_id']) for r in rows}):
        idx=np.asarray([int(r['person_id'])==person for r in rows])
        result[str(person)]=classification_metrics(y[idx],logits[idx].argmax(1))
    return result


def learning_rate(config,epoch):
    warm=config['warmup_epochs']
    if epoch<warm:return config['lr']*(epoch+1)/warm
    progress=(epoch-warm)/max(1,config['epochs']-warm-1)
    return config['lr']*(.05+.95*(1+np.cos(np.pi*progress))/2)


def checkpoint_name(family,seed):
    representation='rows64' if family=='swin1d-image' else 'rgb224'
    return f'drawai_{family}_paint35-main-v1_{representation}_fp32_seed{seed}_best.pth'


def rank(summary):
    return (-summary['validation']['macro_f1'],summary['validation']['loss'],summary['seed'],summary['family'])


def train_one(family,seed,x,y,rows,output,plan_sha,backbone_model=None):
    output=private_output(output);output.mkdir(parents=True,exist_ok=True)
    summary_path=output/'summary.json'
    if summary_path.exists():
        value=load(summary_path)
        if value['plan_sha256']!=plan_sha or value['family']!=family or value['seed']!=seed or file_sha(output/value['checkpoint'])!=value['checkpoint_sha256']:
            raise ValueError('Completed run changed')
        return value
    config=plan()['swin' if family=='swin1d-image' else 'dinov3'];seed_all(seed)
    model=build_model('swin1d_image') if family=='swin1d-image' else nn.Linear(768,10)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    train_idx=np.asarray([r['split']=='train' for r in rows]);val_idx=~train_idx
    if set(Counter(r['split'] for r in rows))!={'train','validation'}:raise ValueError('Test/pilot in training inputs')
    trainx,trainy=x[train_idx],y[train_idx];valx,valy=x[val_idx],y[val_idx]
    trainset=TensorDataset(torch.from_numpy(trainx),torch.from_numpy(trainy))
    history=[];best=None;stale=0;elapsed=0.;start_epoch=0;last=output/'last.pth'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=True)
        if saved['plan_sha256']!=plan_sha or saved['seed']!=seed or saved['family']!=family:raise ValueError('Resume binding mismatch')
        model.load_state_dict(saved['state'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best'];stale=saved['stale'];elapsed=saved['seconds'];start_epoch=len(history)
        torch.set_rng_state(saved['rng'])
    started=time.perf_counter()
    for epoch in range(start_epoch,config['epochs']):
        if stale>=config['patience']:break
        for group in optimizer.param_groups:group['lr']=float(learning_rate(config,epoch))
        generator=torch.Generator().manual_seed(seed+epoch)
        loader=DataLoader(trainset,batch_size=plan()['batch_size'],shuffle=True,generator=generator,num_workers=0)
        model.train();total=0.;correct=0;epoch_start=time.perf_counter()
        for inputs,labels in loader:
            optimizer.zero_grad(set_to_none=True);logits=model(inputs);loss=nn.functional.cross_entropy(logits,labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(labels);correct+=int((logits.argmax(1)==labels).sum())
        metrics,_=predict(model,valx,valy)
        entry=dict(epoch=epoch+1,train_loss=total/len(trainy),train_accuracy=correct/len(trainy),
                   validation_macro_f1=metrics['macro_f1'],validation_accuracy=metrics['accuracy'],validation_loss=metrics['loss'],
                   learning_rate=optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-epoch_start)
        history.append(entry)
        candidate=(-metrics['macro_f1'],metrics['loss'],epoch+1)
        if best is None or candidate<(-best['macro_f1'],best['loss'],best['epoch']):
            best=dict(epoch=epoch+1,macro_f1=metrics['macro_f1'],loss=metrics['loss']);stale=0
            atomic_save(output/'best-trainable.pth',dict(state=model.state_dict(),best=best,plan_sha256=plan_sha))
        else:stale+=1
        atomic_save(last,dict(state=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),history=history,best=best,
                             stale=stale,seconds=elapsed+time.perf_counter()-started,plan_sha256=plan_sha,seed=seed,family=family))
        dump(output/'history.json',history)
        print(f"{family} seed{seed} epoch{epoch+1}: val acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f} ({entry['seconds']:.1f}s)",flush=True)
    seconds=elapsed+time.perf_counter()-started
    fitted=torch.load(output/'best-trainable.pth',map_location='cpu',weights_only=True);model.load_state_dict(fitted['state'],strict=True)
    metrics,logits=predict(model,valx,valy)
    valrows=[r for r in rows if r['split']=='validation']
    np.savez_compressed(output/'validation-predictions.npz',sample_ids=np.asarray([r['sample_id'] for r in valrows]),labels=valy,logits=logits)
    if family=='dinov3-vits16':
        if backbone_model is None or backbone_model.provenance['pretrained_loaded'] is not True:raise ValueError('Verified pretrained backbone required')
        backbone_model.head.load_state_dict(model.state_dict());full=backbone_model
        provenance=full.provenance
    else:full=model;provenance=dict(initialization='scratch',pretrained_loaded=False,auxiliary_data=False)
    checkpoint=checkpoint_name(family,seed)
    atomic_save(output/checkpoint,dict(version='drawai-paint-trained-checkpoint-v1',family=family,seed=seed,
        model_state_dict=full.state_dict(),plan_sha256=plan_sha,lock_sha256=LOCK_HASH,preprocessing=paint.contract(),
        model_kwargs=swin_image_config() if family=='swin1d-image' else dict(frozen_backbone=True,head_in_features=768),
        provenance=provenance,best=best,trained=True,release_ready=False,test_used_for_selection=False))
    summary=dict(version='drawai-paint-run-v1',family=family,seed=seed,status='completed',plan_sha256=plan_sha,
        lock_sha256=LOCK_HASH,best=best,epochs_completed=len(history),training_seconds=seconds,
        checkpoint=checkpoint,checkpoint_sha256=file_sha(output/checkpoint),checkpoint_bytes=(output/checkpoint).stat().st_size,
        validation=metrics,validation_by_writer=grouped_metrics(valy,logits,valrows),per_device=None,
        predictions_sha256=file_sha(output/'validation-predictions.npz'),history_sha256=file_sha(output/'history.json'),
        parameters=sum(t.numel() for t in full.parameters()),trainable_parameters=sum(t.numel() for t in full.parameters() if t.requires_grad),
        provenance=provenance,runtime=runtime(),trained=True,release_ready=False,test_used_for_selection=False)
    dump(summary_path,summary);return summary


def prepare(output):
    output=private_output(output)
    if not output.exists():new_output(output)
    path=output/'experiment-plan.json'
    if path.exists():read_plan(output)
    else:dump(path,plan())
    return file_sha(path)


def train_experiment(dataset,output,weights,cache):
    dataset=Path(dataset).resolve();output=private_output(output);cache=private_output(cache)
    for left,right in ((dataset,output),(dataset,cache),(output,cache)):
        if left==right or left in right.parents or right in left.parents:
            raise ValueError('Dataset, experiment output and feature cache must not overlap')
    plan_sha=prepare(output);seed_all(42)
    if (Path(output)/'selection.json').exists():raise ValueError('Selection already frozen; verify or evaluate, do not retrain')
    pixels,y,rows=development(dataset)
    swin_x=np.stack([paint.swin_rows(x) for x in pixels])
    summaries=[]
    for seed in plan()['seeds']:
        summaries.append(train_one(FAMILIES[0],seed,swin_x,y,rows,Path(output)/'runs'/f'swin1d-image-seed{seed}',plan_sha))
    del swin_x;gc.collect()
    seed_all(42);dino=DinoDigitClassifier.from_pretrained(weights)
    if any(t.requires_grad for t in dino.backbone.parameters()):raise ValueError('Backbone not frozen')
    features,cache_meta=frozen_features(dino,pixels,rows,cache,'development')
    dump(Path(output)/'feature-extraction-summary.json',dict(seconds=cache_meta['seconds'],sample_count=len(rows),
        cache_sha256=cache_meta['sha256'],pretrained_provenance=dino.provenance,cache_contains_test=False))
    for seed in plan()['seeds']:
        summaries.append(train_one(FAMILIES[1],seed,features,y,rows,Path(output)/'runs'/f'dinov3-vits16-seed{seed}',plan_sha,dino))
    if len(summaries)!=6:raise ValueError('Six completed runs required')
    winners={family:min([r for r in summaries if r['family']==family],key=rank) for family in FAMILIES}
    selected=min(winners.values(),key=rank)
    selection=sealed(dict(version='drawai-paint-selection-v1',created_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=plan_sha,lock_sha256=LOCK_HASH,runs=[dict(family=r['family'],seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],
            validation_macro_f1=r['validation']['macro_f1'],validation_loss=r['validation']['loss']) for r in summaries],
        winners={f:dict(seed=r['seed'],checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']) for f,r in winners.items()},
        candidate_default=dict(family=selected['family'],seed=selected['seed']),
        test_accessed=False,rule=plan()['selection_rule'],product_default_changed=False), 'selection_sha256')
    dump(Path(output)/'selection.json',selection)
    print('Validation selection FROZEN; no test opened. '+selection['selection_sha256'],flush=True)
    return selection


def checked_selection(output):
    plan_sha=read_plan(output);selection=load(Path(output)/'selection.json')
    unsigned={k:v for k,v in selection.items() if k!='selection_sha256'}
    if selection['selection_sha256']!=sha(canonical(unsigned)) or selection['plan_sha256']!=plan_sha or selection['test_accessed'] is not False:
        raise ValueError('Selection integrity mismatch')
    actual=[]
    for family in FAMILIES:
        for seed in plan()['seeds']:
            folder=Path(output)/'runs'/f'{family}-seed{seed}';s=load(folder/'summary.json')
            if s['status']!='completed' or s['plan_sha256']!=plan_sha or s['lock_sha256']!=LOCK_HASH or s['test_used_for_selection'] is not False:
                raise ValueError('Run provenance mismatch')
            if file_sha(folder/s['checkpoint'])!=s['checkpoint_sha256'] or file_sha(folder/'validation-predictions.npz')!=s['predictions_sha256']:
                raise ValueError('Run checksum mismatch')
            actual.append(s)
    expected=[dict(family=r['family'],seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],
                   validation_macro_f1=r['validation']['macro_f1'],validation_loss=r['validation']['loss']) for r in actual]
    # Compare by key, not family iteration ordering.
    if sorted(expected,key=lambda r:(r['family'],r['seed']))!=sorted(selection['runs'],key=lambda r:(r['family'],r['seed'])):
        raise ValueError('Selection/run mismatch')
    winners={f:min([r for r in actual if r['family']==f],key=rank) for f in FAMILIES}
    if selection['winners']!={f:dict(seed=r['seed'],checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']) for f,r in winners.items()}:
        raise ValueError('Selection rule violated')
    selected=min(winners.values(),key=rank)
    if selection['candidate_default']!=dict(family=selected['family'],seed=selected['seed']):raise ValueError('Default selection mismatch')
    return selection,actual


def test_images(dataset,output):
    """Explicit final-evaluation reader, never called by train_experiment."""
    checked_selection(output);data.verify_lock(dataset,LOCK_HASH)
    with zipfile.ZipFile(Path(dataset)/'heldout-test.zip') as archive:
        rows=list(csv.DictReader(io.StringIO(archive.read('manifest.csv').decode())))
        data.quotas(rows,('test',))
        if len(archive.namelist())!=1002 or archive.testzip() is not None:raise ValueError('Invalid held-out archive')
        from PIL import Image
        pixels=[]
        for row in rows:
            raw=archive.read(row['png64'])
            if sha(raw)!=row['png_sha256']:raise ValueError('Test PNG checksum mismatch')
            with Image.open(io.BytesIO(raw)) as image:
                if image.mode!='L' or image.size!=(64,64):raise ValueError('Invalid test PNG')
                value=np.asarray(image).copy()
            if sha(value.tobytes())!=row['image_sha256']:raise ValueError('Test pixel checksum mismatch')
            pixels.append(value)
    return np.stack(pixels),np.asarray([int(r['label']) for r in rows],dtype=np.int64),rows


def load_fitted(folder,summary,weights):
    saved=torch.load(Path(folder)/summary['checkpoint'],map_location='cpu',weights_only=True)
    if saved['lock_sha256']!=LOCK_HASH or saved['plan_sha256']!=summary['plan_sha256'] or saved['trained'] is not True:
        raise ValueError('Checkpoint provenance mismatch')
    model=build_model('swin1d_image') if summary['family']=='swin1d-image' else DinoDigitClassifier.from_pretrained(weights)
    model.load_state_dict(saved['model_state_dict'],strict=True);model.eval();return model


def evaluate(dataset,output,weights,cache):
    output=Path(output);selection,runs=checked_selection(output);seed_all(42)
    if (output/'evaluation/final-test.json').exists():raise ValueError('Final test already evaluated; verify saved evidence, do not tune/reopen')
    evaluation=output/'evaluation';evaluation.mkdir(exist_ok=True)
    receipt=evaluation/'test-access.json'
    if receipt.exists():
        old=load(receipt)
        if old['selection_sha256']!=selection['selection_sha256']:raise ValueError('Selection changed after test access')
    else:dump(receipt,dict(opened_at=datetime.now(timezone.utc).isoformat(),selection_sha256=selection['selection_sha256'],purpose='Predeclared final evaluation; resume only for same selection',retuning_allowed=False))
    pixels,y,rows=test_images(dataset,output);results=[];features=None
    for s in runs:
        folder=output/'runs'/f"{s['family']}-seed{s['seed']}";model=load_fitted(folder,s,weights)
        if s['family']=='swin1d-image':
            x=np.stack([paint.swin_rows(image) for image in pixels]);predictor=model
        else:
            if features is None:features,_=frozen_features(model,pixels,rows,cache,'final-test')
            x=features;predictor=model.head
        metrics,logits=predict(predictor,x,y)
        path=evaluation/f"{s['family']}-seed{s['seed']}-test-predictions.npz"
        np.savez_compressed(path,sample_ids=np.asarray([r['sample_id'] for r in rows]),labels=y,logits=logits)
        results.append(dict(family=s['family'],seed=s['seed'],checkpoint_sha256=s['checkpoint_sha256'],
            metrics=metrics,by_writer=grouped_metrics(y,logits,rows),per_device=None,
            predictions_file=path.name,predictions_sha256=file_sha(path)))
        print(f"FINAL TEST {s['family']} seed{s['seed']}: acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f}",flush=True)
        del model;gc.collect()
    aggregate={}
    for f in FAMILIES:
        chosen=[r for r in results if r['family']==f]
        aggregate[f]={metric:dict(mean=float(np.mean([r['metrics'][metric] for r in chosen])),
                    std_population=float(np.std([r['metrics'][metric] for r in chosen],ddof=0))) for metric in ('accuracy','macro_f1')}
    value=sealed(dict(version='drawai-paint-final-test-v1',selection_sha256=selection['selection_sha256'],lock_sha256=LOCK_HASH,
        evaluated_at=datetime.now(timezone.utc).isoformat(),results=results,aggregate=aggregate,
        test_used_for_tuning=False,similarity_independence_verified=False,main_cross_split_near_groups=37,
        candidate_default_unchanged=selection['candidate_default'],product_default_changed=False), 'evaluation_sha256')
    dump(evaluation/'final-test.json',value);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['plan','train','evaluate','verify-selection'])
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATA);parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS);parser.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    args=parser.parse_args()
    if args.command=='plan':print(prepare(args.output))
    elif args.command=='train':train_experiment(args.dataset,args.output,args.weights,args.cache)
    elif args.command=='evaluate':evaluate(args.dataset,args.output,args.weights,args.cache)
    else:print(checked_selection(args.output)[0]['selection_sha256'])


if __name__=='__main__':main()
