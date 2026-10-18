"""P3 private collection planning, strict audit and preliminary two-model build.

No training/test selection, no publication and no claim of verified human origin.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from jsonschema import Draft202012Validator,FormatChecker
from .preprocessing import preprocess
from .raster import render_rgb

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL='drawai-private-protocol-v1'
FIELDS=['sample_id','participant_id','session_id','stage','split','label','repetition','input_device',
        'raw_json','raw_sha256','png','png_sha256','swin_npy','swin_sha256','geometry_sha256','raster_sha256']

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(value):return hashlib.sha256(value).hexdigest()
def load(path):
    def invalid(value):raise ValueError('Nonfinite JSON value: '+value)
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Duplicate JSON key: '+key)
            value[key]=item
        return value
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=invalid,object_pairs_hook=unique)
def dump(path,value):Path(path).write_bytes(canonical(value)+b'\n')

def schedule(participant,stage):
    tasks=[]
    for repetition in range(1,(5 if stage=='pilot' else 20)+1):
        state=2166136261
        for char in f'{participant}:{stage}:{repetition}':state=((state^ord(char))*16777619)&0xffffffff
        digits=list(range(10))
        for i in range(9,0,-1):
            state=(state*1664525+1013904223)&0xffffffff;j=state%(i+1)
            digits[i],digits[j]=digits[j],digits[i]
        tasks.extend(dict(label=label,repetition=repetition) for label in digits)
    return tasks

def plan_content():
    participants=[dict(participant_id=f'P{i:03}',stage='pilot',split='pilot-dev',samples_per_class=5) for i in range(1,6)]
    participants += [dict(participant_id=f'M{i:03}',stage='main',split='train' if i<=20 else 'validation' if i<=25 else 'test',samples_per_class=20) for i in range(1,31)]
    value=dict(version='drawai-private-plan-v1',protocol_version=PROTOCOL,dataset='drawai-private-v1',
        participants=participants,assignment='Reserved codes assigned before collection; identity dedup by coordinator outside public files',
        pilot_excluded_from_main=True,augmentation='train only; not applied by P3 builder',
        release_ready=False,target_pilot_samples=250,target_main_samples=6000)
    return {**value,'plan_sha256':sha(canonical(value))}

def read_plan(path):
    plan=load(path)
    if plan!=plan_content():raise ValueError('Plan differs from locked P3 policy; version and review a new policy before collection')
    return plan

def new_output(path):
    path=Path(path).resolve()
    for forbidden in (ROOT/'web/public',ROOT/'web/dist'):
        if path==forbidden or forbidden in path.parents:raise ValueError('Private output cannot be placed in public/dist')
    if path.exists() and any(path.iterdir()):raise ValueError('Output must be new or empty; never overwrite prior review/build')
    path.mkdir(parents=True,exist_ok=True)
    return path

def validator():
    schema=load(ROOT/'docs/drawai/collection/session.schema.json')
    return Draft202012Validator(schema,format_checker=FormatChecker())

def validate_session(value,plan):
    errors=sorted(validator().iter_errors(value),key=lambda e:str(list(e.path)))
    if errors:raise ValueError('; '.join(f'{list(e.path)}: {e.message}' for e in errors[:5]))
    reserved={p['participant_id']:p for p in plan['participants']}
    participant=reserved.get(value['participant_id'])
    if participant is None or participant['stage']!=value['stage']:raise ValueError('Participant/stage not reserved in plan')
    tasks=schedule(value['participant_id'],value['stage'])
    if len(value['samples'])>len(tasks):raise ValueError('Session exceeds quota')
    start=datetime.fromisoformat(value['started_at'].replace('Z','+00:00'))
    consent=datetime.fromisoformat(value['consent']['confirmed_at'].replace('Z','+00:00'))
    now=datetime.now(timezone.utc)
    if consent>start or start>now:raise ValueError('Consent/start time invalid or in the future')
    previous=start
    for i,sample in enumerate(value['samples']):
        for field in ('participant_id','session_id','stage','input_device'):
            if sample[field]!=value[field]:raise ValueError('Session metadata mismatch: '+field)
        if sample['label']!=tasks[i]['label'] or sample['repetition']!=tasks[i]['repetition']:
            raise ValueError('Prompt order/label/repetition mismatch')
        if sample['actual_pointer_types']!=[value['input_device']]:raise ValueError('Declared/actual device mismatch')
        drawing=sample['drawing']
        if drawing['reference_label']!=sample['label']:raise ValueError('Label/drawing mismatch')
        if drawing['canvas']!={'width':400,'height':400}:raise ValueError('Expected unscaled canonical 400px canvas')
        for stroke in drawing['strokes']:
            for point in stroke:
                if set(point)!={'x','y'} or any(not 0<=point[k]<=400 for k in ('x','y')):raise ValueError('Invalid canvas point')
        preprocess(drawing['strokes'],128)
        time=datetime.fromisoformat(sample['collected_at'].replace('Z','+00:00'))
        if not previous<=time<=now:raise ValueError('Sample time out of order/in future')
        previous=time
    return participant

def geometry_hash(strokes):
    # Numeric 1 and 1.0 are identical; timestamps/labels never hide an exact copy.
    return sha(canonical([[[float(p['x']),float(p['y'])] for p in s] for s in strokes]))

def audit(input_dir,plan,stage):
    input_dir=Path(input_dir).resolve()
    if not input_dir.is_dir():raise ValueError('Input directory does not exist')
    errors=[];warnings=[];rows=[];samples=[];sessions=set();sample_ids=set();participants=set();geometry={};raster_seen={}
    perceptual_buckets={};near=[];near_total=0;counts={};device_counts={};split_counts={};consent_refs={};parents={}
    def find(sid):
        parents.setdefault(sid,sid)
        while parents[sid]!=sid:
            parents[sid]=parents[parents[sid]];sid=parents[sid]
        return sid
    files=sorted(input_dir.rglob('*.json'))
    for path in files:
        try:
            if path.is_symlink() or input_dir not in path.resolve().parents:raise ValueError('Symlink/outside input rejected')
            if path.stat().st_size>20*1024*1024:raise ValueError('Session exceeds 20MiB')
            value=load(path);participant=validate_session(value,plan)
            if value['stage']!=stage:raise ValueError('Do not mix pilot/main in one dataset build')
            if value['session_id'] in sessions:raise ValueError('Duplicate session/snapshot: keep newest version only')
            if value['participant_id'] in participants:raise ValueError('Multiple sessions for participant: resume original session, no quota reset')
            ref=value['consent']['record_ref']
            if ref in consent_refs and consent_refs[ref]!=value['participant_id']:raise ValueError('Consent record reused by different participants')
            consent_refs[ref]=value['participant_id'];sessions.add(value['session_id']);participants.add(value['participant_id'])
            for sample in value['samples']:
                sid=sample['sample_id'];pid=sample['participant_id'];label=sample['label'];strokes=sample['drawing']['strokes']
                if sid in sample_ids:raise ValueError('Duplicate sample_id: '+sid)
                sample_ids.add(sid);geom=geometry_hash(strokes)
                if geom in geometry:raise ValueError(f'Exact duplicate strokes: {sid} and {geometry[geom]}')
                geometry[geom]=sid
                rgb=render_rgb(strokes);rhash=sha(rgb.tobytes());swin=preprocess(strokes,128)
                small=np.asarray(Image.fromarray(rgb).convert('L').resize((9,8),Image.Resampling.BILINEAR))
                bits=0
                for bit in (small[:,1:]>small[:,:-1]).flat:bits=(bits<<1)|int(bit)
                candidates={}
                for chunk in range(4):
                    for item in perceptual_buckets.get((chunk,(bits>>(chunk*16))&65535),[]):candidates[item[0]]=item
                for other_id,other_bits,other_split in candidates.values():
                    distance=(bits^other_bits).bit_count()
                    if distance<=2:
                        near_total+=1
                        parents[find(sid)]=find(other_id)
                        if len(near)<5000:near.append(dict(sample_id=sid,other_sample_id=other_id,hamming_distance=distance,
                            cross_split=participant['split']!=other_split,action='manual review; no automatic exclusion or tuning'))
                for chunk in range(4):perceptual_buckets.setdefault((chunk,(bits>>(chunk*16))&65535),[]).append((sid,bits,participant['split']))
                if rhash in raster_seen:warnings.append(dict(code='IDENTICAL_RASTER',sample_id=sid,other_sample_id=raster_seen[rhash]))
                raster_seen[rhash]=sid
                raw=canonical(sample)+b'\n';png_bytes=io.BytesIO();Image.fromarray(rgb).save(png_bytes,format='PNG')
                npy_bytes=io.BytesIO();np.save(npy_bytes,swin,allow_pickle=False)
                row=dict(sample_id=sid,participant_id=pid,session_id=value['session_id'],stage=stage,split=participant['split'],
                    label=label,repetition=sample['repetition'],input_device=sample['input_device'],
                    raw_json=f'raw/{sid}.json',raw_sha256=sha(raw),png=f'images/{sid}.png',png_sha256=sha(png_bytes.getvalue()),
                    swin_npy=f'swin-l128/{sid}.npy',swin_sha256=sha(npy_bytes.getvalue()),geometry_sha256=geom,raster_sha256=rhash)
                rows.append(row);samples.append((sample,raw,png_bytes.getvalue(),npy_bytes.getvalue()))
                counts.setdefault(pid,{str(k):0 for k in range(10)})[str(label)]+=1
                device_counts[sample['input_device']]=device_counts.get(sample['input_device'],0)+1
                split_counts[participant['split']]=split_counts.get(participant['split'],0)+1
        except (ValueError,TypeError,KeyError,OverflowError,OSError) as error:errors.append(dict(file=str(path.relative_to(input_dir)),error=str(error)))
    target=[p for p in plan['participants'] if p['stage']==stage]
    missing={p['participant_id']:{str(k):max(0,p['samples_per_class']-counts.get(p['participant_id'],{}).get(str(k),0)) for k in range(10)} for p in target}
    complete=not errors and all(all(n==0 for n in labels.values()) for labels in missing.values())
    if not files:warnings.append(dict(code='NO_HUMAN_DATA',message='No session JSON found; pilot has not been performed'))
    components={}
    for sid in parents:components.setdefault(find(sid),[]).append(sid)
    groups=[dict(group_id='near_'+sha(canonical(sorted(ids))),sample_ids=sorted(ids)) for ids in components.values() if len(ids)>1]
    groups.sort(key=lambda g:g['group_id'])
    result=dict(version='drawai-private-audit-v1',protocol_version=PROTOCOL,plan_sha256=plan['plan_sha256'],stage=stage,
        input_files=len(files),accepted_samples=len(rows) if not errors else 0,parsed_samples=len(rows),errors=errors,warnings=warnings,
        near_duplicates=near,near_duplicate_pairs_total=near_total,near_duplicate_report_truncated=near_total>len(near),
        near_duplicate_policy='Review original strokes without prediction; record reasons; never auto-remove by model output; P4 must resolve all flags, including truncated reports',
        near_duplicate_groups=groups,near_groups_include_all_pairs=True,
        counts_by_participant_and_class=counts,counts_by_device=device_counts,counts_by_split=split_counts,
        missing_by_participant_and_class=missing,quota_complete=complete,dataset_locked=False,release_ready=False,
        human_origin='self-declared by collection session; coordinator must verify identities/consent outside public files',
        verified_human_origin_by_software=False,test_used_for_tuning=False)
    return result,rows,samples

def build(input_dir,plan,stage,output,require_complete=False):
    report,rows,samples=audit(input_dir,plan,stage)
    if report['errors']:raise ValueError('Audit errors; run audit and resolve before build: '+str(report['errors'][:3]))
    if not rows:raise ValueError('No samples; never create a fabricated pilot dataset')
    if require_complete and not report['quota_complete']:raise ValueError('Quota incomplete; no complete-pilot claim allowed')
    output=new_output(output)
    for folder in ('raw','images','swin-l128'):(output/folder).mkdir()
    for row,(_,raw,png,npy) in zip(rows,samples):
        for path,data in ((row['raw_json'],raw),(row['png'],png),(row['swin_npy'],npy)):(output/path).write_bytes(data)
    with (output/'manifest.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    dump(output/'audit.json',report);dump(output/'plan.json',plan)
    dump(output/'dataset.json',dict(dataset='drawai-private-v1',stage=stage,preliminary=True,dataset_locked=False,
        release_ready=False,sample_count=len(rows),plan_sha256=plan['plan_sha256'],manifest_sha256=sha((output/'manifest.csv').read_bytes()),
        same_sample_ids_for_both_models=True,augmentation_applied=False,
        excluded_private_fields=['consent record reference','consent timestamps','session start time'],
        public_release='P8 only after manual privacy/license/consent/data-card review; P3 builder never publishes'))
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('plan');p.add_argument('--output',type=Path,default=ROOT/'data/private/plans/private-v1')
    for command in ('audit','build'):
        p=commands.add_parser(command);p.add_argument('--input',type=Path,required=True)
        p.add_argument('--plan',type=Path,default=ROOT/'data/private/plans/private-v1/plan.json')
        p.add_argument('--stage',choices=['pilot','main'],default='pilot');p.add_argument('--output',type=Path,required=True)
        if command=='build':p.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    try:
        if args.command=='plan':
            output=new_output(args.output);plan=plan_content();dump(output/'plan.json',plan)
            with (output/'participants-reserved.csv').open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=plan['participants'][0].keys());writer.writeheader();writer.writerows(plan['participants'])
            dump(output/'prompt-parity.json',{pid:schedule(pid,stage) for pid,stage in [('P001','pilot'),('P005','pilot'),('M001','main'),('M030','main')]})
            print('Plan reserved; no participant enrolled and no handwriting collected. '+plan['plan_sha256'])
        elif args.command=='audit':
            report,_,_=audit(args.input,read_plan(args.plan),args.stage);output=new_output(args.output);dump(output/'audit.json',report)
            print(json.dumps({key:report[key] for key in ('accepted_samples','quota_complete','dataset_locked','errors')}))
            if report['errors']:return 1
        else:
            report=build(args.input,read_plan(args.plan),args.stage,args.output,args.require_complete)
            print(f"Preliminary build: {report['accepted_samples']} samples; not locked, not published.")
    except (ValueError,OSError) as error:print('ERROR: '+str(error),file=sys.stderr);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
