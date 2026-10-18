"""Validate operator-attested human canvas data and evaluate the locked D4 ONNX release.

This program never generates human samples. Consent/source authenticity needs human attestation.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

_dll_handle=None
if sys.platform=='win32' and os.environ.get('DUST_DLL_DIRECTORY'):
    _dll_handle=os.add_dll_directory(os.environ['DUST_DLL_DIRECTORY'])

import jsonschema
import numpy as np

from .prepare_data import ROOT,digest,write_json
from .preprocessing import preprocess
from .metrics import classification_metrics


def validate_manifest(path):
    path=Path(path).resolve(); root=path.parent
    manifest=json.loads(path.read_text())
    if set(manifest)!={'protocol_version','release_version','human_source_attested','participants','samples'}:
        raise ValueError('Unexpected or missing manifest fields')
    if manifest.get('protocol_version')!='drawai-canvas-eval-v1' or manifest.get('release_version')!='drawai-swin-v1' or manifest.get('human_source_attested') is not True:
        raise ValueError('Need fixed protocol/release and explicit human-source attestation')
    people=manifest.get('participants',[])
    if len(people)<5:raise ValueError('Need at least five consented human participants')
    participants={}
    for person in people:
        pid=person.get('id','')
        if not re.fullmatch(r'P\d{3}',pid) or pid in participants or person.get('consent') is not True or person.get('input_device') not in ['mouse','touch','pen']:
            raise ValueError('Participant ID/consent/input device invalid or duplicated')
        if not all(isinstance(person.get(k),str) and person[k].strip() for k in ['device_description','browser']):
            raise ValueError('Need participant device/browser metadata')
        if set(person)!={'id','consent','input_device','device_description','browser'}:raise ValueError('Unexpected participant metadata')
        participants[pid]=person
    rows=[];seen=set();hashes=set();counts=Counter()
    schema=json.loads((ROOT/'docs/drawai/strokes.schema.json').read_text())
    for sample in manifest.get('samples',[]):
        if set(sample)!={'participant_id','label','repetition','file'}:raise ValueError('Unexpected sample metadata')
        pid,label,repeat=sample.get('participant_id'),sample.get('label'),sample.get('repetition')
        if pid not in participants or type(label) is not int or label not in range(10) or type(repeat) is not int or repeat not in range(1,6):
            raise ValueError('Invalid participant/label/repetition')
        key=(pid,label,repeat)
        if key in seen:raise ValueError('Duplicate participant/label/repetition')
        seen.add(key)
        filename=sample.get('file')
        if not isinstance(filename,str) or Path(filename).is_absolute():raise ValueError('Need dataset-relative sample path')
        source=(root/filename).resolve()
        if not source.is_relative_to(root):raise ValueError('Sample path escapes dataset root')
        hash_value=digest(source)
        if hash_value in hashes:raise ValueError('Duplicate JSON content; investigate before evaluation')
        hashes.add(hash_value)
        drawing=json.loads(source.read_text());jsonschema.validate(drawing,schema)
        if drawing.get('reference_label')!=label or drawing.get('source',{}).get('kind')!='user' or drawing.get('source',{}).get('split') not in [None,'canvas-evaluation']:
            raise ValueError('Need independent writer label and human user source')
        points=[p for stroke in drawing['strokes'] for p in stroke]
        if len(points)>8192:raise ValueError('Too many total points')
        flags=['t' in p for p in points]
        if any(flags) and (not all(flags) or any(not np.isfinite(p['t']) for p in points) or any(a['t']>b['t'] for a,b in zip(points,points[1:]))):
            raise ValueError('Inconsistent/nonfinite/backward timestamps')
        tensor=preprocess(drawing['strokes'],128)
        counts[(pid,label)]+=1
        rows.append(dict(participant_id=pid,label=label,repetition=repeat,file=filename,sha256=hash_value,input_device=participants[pid]['input_device'],tensor=tensor))
    if len(rows)!=50*len(participants) or any(counts[(pid,label)]!=5 for pid in participants for label in range(10)):
        raise ValueError('Need all ten digits five times per participant')
    return manifest,rows


def evaluate(path,output):
    manifest,rows=validate_manifest(path)
    if (output/'evaluation.json').exists():raise ValueError('Output already evaluated; preserve results and use a new output directory')
    import onnxruntime as ort
    target=ROOT/'web/public/models/drawai-swin-v1'
    release=json.loads((target/'release.json').read_text())
    assert release['version']=='drawai-swin-v1' and release['trained'] is True
    for name,meta in release['files'].items():
        if digest(target/name)!=meta['sha256'] or (target/name).stat().st_size!=meta['bytes']:raise ValueError('Release file mismatch')
    session=ort.InferenceSession(str(target/'model.onnx'),providers=['CPUExecutionProvider'])
    predictions=[]
    for row in rows:
        logits=session.run(None,dict(drawing=row['tensor'][None]))[0][0]
        if not np.isfinite(logits).all():raise ValueError('Nonfinite logits')
        predictions.append(dict(**{k:v for k,v in row.items() if k!='tensor'},predicted=int(logits.argmax()),logits=logits.tolist()))
    def metrics(values):return classification_metrics([r['label'] for r in values],[r['predicted'] for r in values])
    report=dict(status='evaluated-human-source-attested',human_source_authenticity='operator attestation, not automatically verified',
                protocol_version=manifest['protocol_version'],release_sha256=digest(target/'release.json'),
                model_sha256=digest(target/'model.onnx'),manifest_sha256=digest(Path(path)),
                participants=len(manifest['participants']),samples=len(rows),metrics=metrics(predictions),
                by_participant={p['id']:metrics([r for r in predictions if r['participant_id']==p['id']]) for p in manifest['participants']},
                by_input_device={device:metrics([r for r in predictions if r['input_device']==device]) for device in sorted({r['input_device'] for r in rows})},
                errors=[r for r in predictions if r['label']!=r['predicted']],predictions=predictions)
    write_json(output/'evaluation.json',report);print(f'Evaluated {len(rows)} operator-attested human samples')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();evaluate(args.manifest,args.output_dir)
