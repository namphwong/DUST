"""Fresh-process CPU latency on validation images, for a selected fitted model."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai import paint_train as p, paint_data as paint, paint_dataset as data
from training.drawai.private_data import load, dump
from training.drawai.raster import normalize_rgb


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--family',choices=p.FAMILIES,required=True)
    args=parser.parse_args();p.seed_all(42);output=p.DEFAULT_OUT
    selection,runs=p.checked_selection(output);winner=selection['winners'][args.family]
    summary=next(r for r in runs if r['family']==args.family and r['seed']==winner['seed'])
    _,rows=data.verify_lock(p.DEFAULT_DATA,p.LOCK_HASH)
    val=[next(r for r in rows if r['split']=='validation' and int(r['label'])==label) for label in range(10)]
    pixels=[data.check_image(p.DEFAULT_DATA/'development',r) for r in val]
    process=psutil.Process();gc.collect();before=process.memory_info().rss;start=time.perf_counter()
    model=p.load_fitted(output/'runs'/f"{args.family}-seed{summary['seed']}",summary,p.DEFAULT_WEIGHTS)
    load_seconds=time.perf_counter()-start;after_load=process.memory_info().rss
    times=[];preprocess=[];network=[]
    with torch.inference_mode():
        for i in range(55):
            t0=time.perf_counter()
            value=paint.swin_rows(pixels[i%10]) if args.family=='swin1d-image' else normalize_rgb(paint.dino_rgb(pixels[i%10]))
            x=torch.from_numpy(value[None]);t1=time.perf_counter();logits=model(x);t2=time.perf_counter()
            if logits.shape!=(1,10) or not torch.isfinite(logits).all():raise ValueError('Invalid benchmark logits')
            if i>=5:
                preprocess.append((t1-t0)*1000);network.append((t2-t1)*1000);times.append((t2-t0)*1000)
    memory=process.memory_info()
    result=dict(family=args.family,seed=summary['seed'],checkpoint_sha256=summary['checkpoint_sha256'],
        selection_sha256=selection['selection_sha256'],runtime=p.runtime(),batch_size=1,
        input='ten validation PNG64 images, repeated; disk decoding outside timed section',
        validation_class_coverage=list(range(10)),
        timing_scope='source uint8 image -> preprocessing -> full fitted PyTorch model; no HTTP/browser/ONNX',
        warmup=5,iterations=50,load_seconds=load_seconds,
        milliseconds=dict(p50=float(np.median(times)),p95=float(np.percentile(times,95)),mean=float(np.mean(times)),
                          preprocessing_mean=float(np.mean(preprocess)),network_mean=float(np.mean(network))),
        memory=dict(rss_before_model=before,rss_after_model_load=after_load,rss_after_inference=memory.rss,
                    process_lifetime_peak_rss=getattr(memory,'peak_wset',None),
                    caveat='Process peak includes imports, dataset verification, checkpoint load and inference; not model-only allocation'),
        checkpoint_bytes=summary['checkpoint_bytes'],test_data_used=False)
    dump(output/f'benchmark-{args.family}.json',result);print(json.dumps(result['milliseconds']))


if __name__=='__main__':main()
