"""P4 review-bound dataset locking; manual attestations are not identity proof.

Commands never publish data. Held-out ZIP is hash sealed, NOT encrypted.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np
from PIL import Image
from . import private_data as collection
from .private_data import ROOT,canonical,sha,load,dump,read_plan,new_output,audit
from .raster import contract as raster_contract,normalize_rgb

DEFAULT_PLAN=ROOT/'data/private/plans/private-v1/plan.json'
CONFIRMATIONS=['human_origin_and_consent_verified','unique_people_verified',
    'split_assignment_before_collection_verified','no_model_based_exclusions',
    'protocol_exclusions_logged_and_recollected']
LOCK_FIELDS=[field for field in collection.FIELDS if field!='session_id']

def exact(value,keys,context):
    if not isinstance(value,dict) or set(value)!=set(keys):raise ValueError(context+': unexpected/missing fields')

def stamp(value):
    if not isinstance(value,str):raise ValueError('Review date requires actual timezone-aware ISO timestamp')
    try:result=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError:raise ValueError('Invalid review timestamp') from None
    if result.tzinfo is None or result>datetime.now(timezone.utc):raise ValueError('Review timestamp is naive/future')
    return result

def inventory(directory):
    directory=Path(directory).resolve()
    if not directory.is_dir():raise ValueError('Input directory does not exist')
    files=[]
    for p in sorted(directory.rglob('*.json')):
        if p.is_symlink() or directory not in p.resolve().parents:raise ValueError('Symlink/outside input rejected')
        if p.stat().st_size>20*1024*1024:raise ValueError('Session exceeds 20MiB')
        files.append(dict(path=p.relative_to(directory).as_posix(),bytes=p.stat().st_size,sha256=sha(p.read_bytes())))
    return dict(files=files,sha256=sha(canonical(files)))

def source_state(directory,policy,stage):
    before=inventory(directory);report,rows,samples=audit(directory,policy,stage)
    if before!=inventory(directory):raise ValueError('Source changed during audit')
    return before,report,rows,samples

def separate_output(output,input_dir):
    output=Path(output).resolve();source=Path(input_dir).resolve()
    if output==source or source in output.parents or output in source.parents:
        raise ValueError('Input and output must be separate, non-overlapping directories')
    for forbidden in (ROOT/'web/public',ROOT/'web/dist'):
        if output==forbidden or forbidden in output.parents:raise ValueError('Private output cannot be public/dist')
    if output.exists():raise ValueError('P4 output must be new; do not overwrite a review/lock')
    return output

def expected_participants(policy,stage):return [p for p in policy['participants'] if p['stage']==stage]

def checked_approval(path,policy):
    value=load(path)
    exact(value,['version','protocol_version','plan_sha256','source_sha256','review_sha256','approved_at',
                 'sample_count','participant_count','person_token_hashes','geometry_hashes','sample_ids','approval_sha256'],'Pilot approval')
    unsigned={k:v for k,v in value.items() if k!='approval_sha256'}
    pilot=expected_participants(policy,'pilot')
    if value['version']!='drawai-pilot-approval-v1' or value['protocol_version']!=policy['protocol_version'] or value['plan_sha256']!=policy['plan_sha256']:
        raise ValueError('Pilot approval protocol/plan mismatch')
    if value['approval_sha256']!=sha(canonical(unsigned)):raise ValueError('Pilot approval hash mismatch')
    if value['sample_count']!=policy['target_pilot_samples'] or value['participant_count']!=len(pilot):raise ValueError('Pilot quota not approved')
    if len(set(value['person_token_hashes']))!=len(pilot) or len(value['person_token_hashes'])!=len(pilot):raise ValueError('Pilot people invalid')
    for key in ('person_token_hashes','geometry_hashes'):
        if not isinstance(value[key],list) or any(not isinstance(s,str) or not re.fullmatch('[0-9a-f]{64}',s) for s in value[key]):raise ValueError('Invalid pilot hashes')
    if len(value['geometry_hashes'])!=value['sample_count'] or len(set(value['geometry_hashes']))!=value['sample_count']:
        raise ValueError('Pilot geometry inventory invalid')
    if len(value['sample_ids'])!=value['sample_count'] or len(set(value['sample_ids']))!=value['sample_count']:raise ValueError('Pilot sample inventory invalid')
    stamp(value['approved_at']);return value

def review_template(input_dir,stage,output,plan_path=DEFAULT_PLAN,pilot_approval=None):
    output=separate_output(output,input_dir);policy=read_plan(plan_path)
    sources,report,_,_=source_state(input_dir,policy,stage)
    approval=checked_approval(pilot_approval,policy) if pilot_approval else None
    review=dict(version='drawai-dataset-review-v1',protocol_version=policy['protocol_version'],stage=stage,
        plan_sha256=policy['plan_sha256'],source_sha256=sources['sha256'],audit_sha256=sha(canonical(report)),
        pilot_approval_sha256=approval['approval_sha256'] if approval else None,
        reviewer_ref='',reviewed_at=None,confirmations={key:False for key in CONFIRMATIONS},
        participants=[dict(participant_id=p['participant_id'],person_token='',human_origin_verified=False,
                           consent_record_verified=False,withdrawn=False) for p in expected_participants(policy,stage)],
        near_groups=[dict(group_id=g['group_id'],decision='pending',reason='') for g in report['near_duplicate_groups']],
        exclusion_log=[])
    new_output(output);dump(output/'review.json',review);dump(output/'source-inventory.json',sources);dump(output/'audit.json',report)
    dump(output/'readiness.json',dict(stage=stage,source_files=len(sources['files']),samples=report['accepted_samples'],
        quota_complete=report['quota_complete'],pilot_approval_present=approval is not None,
        review_complete=False,dataset_locked=False,
        blockers=(['No session JSON provided'] if not sources['files'] else [])+
            (['Quota incomplete'] if not report['quota_complete'] else [])+
            (['Audit errors'] if report['errors'] else [])+
            (['Approved human pilot required'] if stage=='main' and approval is None else [])+['Coordinator review not signed']))
    return review

def validate_review(input_dir,stage,review_path,policy,pilot_approval=None):
    sources,report,rows,samples=source_state(input_dir,policy,stage)
    if report['errors']:raise ValueError('Collection audit errors must be resolved')
    if not report['quota_complete'] or not rows:raise ValueError('Full planned quota required; empty/partial dataset cannot be locked')
    target=policy['target_pilot_samples' if stage=='pilot' else 'target_main_samples']
    if len(rows)!=target:raise ValueError('Unexpected sample count')
    review=load(review_path)
    exact(review,['version','protocol_version','stage','plan_sha256','source_sha256','audit_sha256','pilot_approval_sha256',
                  'reviewer_ref','reviewed_at','confirmations','participants','near_groups','exclusion_log'],'Review')
    if review['version']!='drawai-dataset-review-v1' or review['stage']!=stage or review['protocol_version']!=policy['protocol_version']:
        raise ValueError('Review stage/protocol mismatch')
    if review['plan_sha256']!=policy['plan_sha256'] or review['source_sha256']!=sources['sha256'] or review['audit_sha256']!=sha(canonical(report)):
        raise ValueError('Stale review: plan/source/audit changed; regenerate review')
    if not isinstance(review['reviewer_ref'],str) or not re.fullmatch('R-[A-Z0-9-]{3,32}',review['reviewer_ref']):raise ValueError('Coordinator reviewer reference required')
    reviewed_at=stamp(review['reviewed_at'])
    if any(stamp(sample['collected_at'])>reviewed_at for sample,_,_,_ in samples):raise ValueError('Review predates collected samples')
    exact(review['confirmations'],CONFIRMATIONS,'Confirmations')
    if any(value is not True for value in review['confirmations'].values()):raise ValueError('All coordinator confirmations required')
    expected={p['participant_id'] for p in expected_participants(policy,stage)}
    people=review['participants'];seen=set();tokens=set()
    if not isinstance(people,list):raise ValueError('Invalid participants review')
    for person in people:
        exact(person,['participant_id','person_token','human_origin_verified','consent_record_verified','withdrawn'],'Participant review')
        pid=person['participant_id'];token=person['person_token']
        if pid not in expected or pid in seen:raise ValueError('Participant review duplicate/unknown')
        if not isinstance(token,str) or not re.fullmatch('person_[0-9a-f]{32}',token):raise ValueError('Use stable private pseudonymous person token')
        if token in tokens:raise ValueError('Same actual person appears under multiple codes')
        if person['human_origin_verified'] is not True or person['consent_record_verified'] is not True or person['withdrawn'] is not False:
            raise ValueError('Human/consent review missing or participant withdrawn; recollect before lock')
        seen.add(pid);tokens.add(token)
    if seen!=expected:raise ValueError('Missing participant review')
    group_ids={g['group_id'] for g in report['near_duplicate_groups']};decisions=set()
    if not isinstance(review['near_groups'],list):raise ValueError('Invalid near group review')
    for group in review['near_groups']:
        exact(group,['group_id','decision','reason'],'Near group review')
        if group['group_id'] not in group_ids or group['group_id'] in decisions:raise ValueError('Unknown/duplicate near group')
        if group['decision']!='keep_distinct_verified' or not isinstance(group['reason'],str) or not 10<=len(group['reason'])<=1000:
            raise ValueError('Near duplicates require explicit human keep-distinct rationale or recollection')
        decisions.add(group['group_id'])
    if decisions!=group_ids:raise ValueError('Unreviewed near groups')
    if not isinstance(review['exclusion_log'],list):raise ValueError('Invalid exclusion log')
    for item in review['exclusion_log']:
        exact(item,['session_id','original_file_sha256','reason','replacement_session_id'],'Exclusion log')
        if item['reason'] not in ['corrupt_file','empty_drawing','label_error_verified','protocol_violation','exact_duplicate','consent_withdrawn']:
            raise ValueError('Exclusion reason cannot be model prediction/accuracy')
        if not isinstance(item['original_file_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',item['original_file_sha256']):raise ValueError('Exclusion requires original hash')
        for field in ('session_id','replacement_session_id'):
            if not isinstance(item[field],str) or not re.fullmatch('session_[0-9a-f-]{36}',item[field]):raise ValueError('Exclusion requires traceable session IDs')
    token_hashes=sorted(sha(token.encode()) for token in tokens)
    if stage=='main':
        if not pilot_approval:raise ValueError('Approved human pilot required before main lock')
        approval=checked_approval(pilot_approval,policy)
        if review['pilot_approval_sha256']!=approval['approval_sha256']:raise ValueError('Pilot approval differs from review')
        if set(token_hashes)&set(approval['person_token_hashes']):raise ValueError('Pilot participant reused in main dataset')
        if {r['geometry_sha256'] for r in rows}&set(approval['geometry_hashes']):raise ValueError('Pilot strokes copied into main dataset')
        if {r['sample_id'] for r in rows}&set(approval['sample_ids']):raise ValueError('Pilot sample reused in main dataset')
    elif review['pilot_approval_sha256'] is not None:raise ValueError('Pilot review cannot reference another approval')
    return sources,report,rows,samples,review,token_hashes

def approve_pilot(input_dir,review_path,output,plan_path=DEFAULT_PLAN):
    output=separate_output(output,input_dir);policy=read_plan(plan_path)
    sources,report,rows,_,review,tokens=validate_review(input_dir,'pilot',review_path,policy)
    value=dict(version='drawai-pilot-approval-v1',protocol_version=policy['protocol_version'],plan_sha256=policy['plan_sha256'],
        source_sha256=sources['sha256'],review_sha256=sha(Path(review_path).read_bytes()),approved_at=review['reviewed_at'],
        sample_count=len(rows),participant_count=len(tokens),person_token_hashes=tokens,
        geometry_hashes=sorted(r['geometry_sha256'] for r in rows),sample_ids=sorted(r['sample_id'] for r in rows))
    value['approval_sha256']=sha(canonical(value))
    if inventory(input_dir)!=sources:raise ValueError('Pilot source changed during approval')
    new_output(output);dump(output/'pilot-approval.json',value)
    return value

def csv_bytes(rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=LOCK_FIELDS);writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode()

def sanitized(sample):
    return dict(schema_version='drawai-private-drawing-v1',sample_id=sample['sample_id'],participant_id=sample['participant_id'],
        stage=sample['stage'],label=sample['label'],repetition=sample['repetition'],input_device=sample['input_device'],drawing=sample['drawing'])

def zip_write(archive,name,data):
    info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o600<<16
    archive.writestr(info,data)

def lock(input_dir,review_path,pilot_approval,output,plan_path=DEFAULT_PLAN):
    output=separate_output(output,input_dir);policy=read_plan(plan_path)
    sources,report,rows,samples,review,tokens=validate_review(input_dir,'main',review_path,policy,pilot_approval)
    review_hash=sha(Path(review_path).read_bytes());approval_hash=sha(Path(pilot_approval).read_bytes())
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.staging-',dir=output.parent))
    development=[];heldout=[]
    with zipfile.ZipFile(staging/'heldout-test.zip','w') as test:
        for row,(sample,_,png,npy) in zip(rows,samples):
            raw=canonical(sanitized(sample))+b'\n';row={k:v for k,v in row.items() if k in LOCK_FIELDS};row['raw_sha256']=sha(raw)
            blobs=[(row['raw_json'],raw),(row['png'],png),(row['swin_npy'],npy)]
            if row['split']=='test':
                heldout.append(row)
                for name,data in blobs:zip_write(test,name,data)
            else:
                development.append(row)
                for name,data in blobs:
                    path=staging/'development'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        zip_write(test,'manifest.csv',csv_bytes(heldout))
        zip_write(test,'CUSTODY.txt',b'Held-out labels and representations. Hash-sealed, NOT encrypted. Do not open for tuning; coordinator custody until final evaluation.\n')
    (staging/'development/manifest.csv').write_bytes(csv_bytes(development))
    dump(staging/'plan.json',policy);dump(staging/'preprocessing.json',dict(swin='drawai-geometry-v1 L128',dinov3=raster_contract()))
    split_counts={s:sum(r['split']==s for r in rows) for s in ('train','validation','test')}
    writers={s:sorted({r['participant_id'] for r in rows if r['split']==s}) for s in split_counts}
    class_counts={s:{str(c):sum(r['split']==s and r['label']==c for r in rows) for c in range(10)} for s in split_counts}
    devices={s:{d:sum(r['split']==s and r['input_device']==d for r in rows) for d in ('mouse','touch','pen')} for s in split_counts}
    dump(staging/'review-summary.json',dict(reviewer_ref=review['reviewer_ref'],reviewed_at=review['reviewed_at'],
        confirmations=review['confirmations'],source_sha256=sources['sha256'],review_sha256=review_hash,
        pilot_approval_file_sha256=approval_hash,near_groups_reviewed=len(review['near_groups']),near_pairs=report['near_duplicate_pairs_total'],
        exclusion_log_count=len(review['exclusion_log']),person_tokens_and_consent_refs_excluded=True,
        manual_attestation='Software checks supplied attestations; cannot independently prove identities or consent'))
    (staging/'DATASET_CARD.md').write_text(f'''# DrawAI private-v1 — locked internal dataset

Samples: {len(rows)}. Split counts: {json.dumps(split_counts)}. Writers: {json.dumps({k:len(v) for k,v in writers.items()})}.
Per-class counts: {json.dumps(class_counts)}. Per-device counts: {json.dumps(devices)}.
Plan: {policy['plan_sha256']}. Original collection source inventory: {sources['sha256']}.
Common sample IDs/splits for Swin1D L128 and DINOv3 RGB224; no augmentation applied.
Manual coordinator human/consent/identity/near-duplicate review bound to source hashes; not independently proven by software.
Raw derived drawings exclude precise collection timestamps, session IDs, consent references and person tokens.
Test ZIP is not encrypted or access controlled. Coordinator must restrict access outside this tool until evaluation.
No model quality metrics in P4. No publication or dataset license granted by this lock; privacy/license/Kaggle review remains P8.
''',encoding='utf-8')
    files={p.relative_to(staging).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(staging.rglob('*')) if p.is_file()}
    code_hashes={name:sha((ROOT/name).read_bytes()) for name in ['training/drawai/preprocessing.py','training/drawai/raster.py','training/drawai/private_dataset.py']}
    record=dict(version='drawai-private-lock-v1',dataset='drawai-private-v1',dataset_locked=True,release_ready=False,published=False,
        sample_count=len(rows),split_counts=split_counts,writers=writers,class_counts=class_counts,device_counts=devices,
        plan_sha256=policy['plan_sha256'],source_sha256=sources['sha256'],review_sha256=review_hash,
        created_at=datetime.now(timezone.utc).isoformat(),files=files,code_sha256=code_hashes,
        test_custody='heldout-test.zip; hash integrity only, NOT encryption or OS access control',
        development_manifest='development/manifest.csv',test_used_for_tuning=False)
    record['lock_sha256']=sha(canonical(record));dump(staging/'dataset-lock.json',record)
    if inventory(input_dir)!=sources or sha(Path(review_path).read_bytes())!=review_hash or sha(Path(pilot_approval).read_bytes())!=approval_hash:
        raise ValueError('Source/review changed while writing; staging is not a final lock')
    verify_lock(staging,record['lock_sha256'],plan_path)
    staging.rename(output)
    return record

def safe_file(root,name):
    if not isinstance(name,str) or '\\' in name or ':' in name or name.startswith('/') or any(p in ('','..','.') for p in name.split('/')):
        raise ValueError('Unsafe package path')
    path=root/name
    if path.is_symlink() or root not in path.resolve().parents:raise ValueError('Symlink/outside package rejected')
    return path

def verify_lock(root,expected_lock_sha256,plan_path=DEFAULT_PLAN):
    root=Path(root).resolve();record=load(root/'dataset-lock.json');policy=read_plan(plan_path)
    unsigned={k:v for k,v in record.items() if k!='lock_sha256'}
    if not isinstance(expected_lock_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_lock_sha256):raise ValueError('Trusted lock hash required')
    if record.get('lock_sha256')!=expected_lock_sha256 or record.get('lock_sha256')!=sha(canonical(unsigned)):raise ValueError('Dataset lock metadata hash mismatch')
    if record.get('version')!='drawai-private-lock-v1' or record.get('dataset')!='drawai-private-v1' or record.get('dataset_locked') is not True:
        raise ValueError('Invalid dataset lock')
    if record.get('release_ready') is not False or record.get('published') is not False or record.get('test_used_for_tuning') is not False:raise ValueError('Not a P4 internal lock')
    if record.get('plan_sha256')!=policy['plan_sha256'] or record.get('sample_count')!=policy['target_main_samples']:raise ValueError('Lock policy/quota mismatch')
    if load(root/'plan.json')!=policy:raise ValueError('Packaged policy changed')
    expected_counts={split:sum(p['samples_per_class']*10 for p in policy['participants'] if p['stage']=='main' and p['split']==split) for split in ('train','validation','test')}
    if record['split_counts']!=expected_counts:raise ValueError('Split quota mismatch')
    expected_writers={split:sorted(p['participant_id'] for p in policy['participants'] if p['stage']=='main' and p['split']==split) for split in expected_counts}
    if record['writers']!=expected_writers:raise ValueError('Writer assignment mismatch')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(record['files'])|{'dataset-lock.json'}:raise ValueError('Unexpected/missing package file')
    for name,meta in record['files'].items():
        path=safe_file(root,name)
        if path.stat().st_size!=meta['bytes'] or sha(path.read_bytes())!=meta['sha256']:raise ValueError('Package hash mismatch: '+name)
    if record.get('development_manifest')!='development/manifest.csv':raise ValueError('Invalid development manifest')
    with (root/'development/manifest.csv').open(encoding='utf-8',newline='') as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=LOCK_FIELDS:raise ValueError('Manifest schema mismatch')
        rows=list(reader)
    if len(rows)!=expected_counts['train']+expected_counts['validation']:raise ValueError('Development count mismatch')
    ids=set()
    for row in rows:
        split=row['split'];sid=row['sample_id']
        if split not in ('train','validation') or row['participant_id'] not in expected_writers[split] or sid in ids:raise ValueError('Test leakage or duplicate sample/writer')
        ids.add(sid)
        for key,folder in [('raw_json','raw'),('png','images'),('swin_npy','swin-l128')]:
            suffix={'raw_json':'json','png':'png','swin_npy':'npy'}[key]
            if row[key]!=f'{folder}/{sid}.{suffix}':raise ValueError('Manifest path mismatch')
            path=safe_file(root,'development/'+row[key]);hash_key={'raw_json':'raw_sha256','png':'png_sha256','swin_npy':'swin_sha256'}[key]
            if sha(path.read_bytes())!=row[hash_key]:raise ValueError('Manifest content hash mismatch')
    # Deliberately hash the sealed ZIP above without extracting held-out labels.
    return record,rows

def load_development(root,split,expected_lock_sha256,plan_path=DEFAULT_PLAN):
    if split not in ('train','validation'):raise ValueError('Test remains sealed; development loader only accepts train/validation')
    _,rows=verify_lock(root,expected_lock_sha256,plan_path)
    selected=[r for r in rows if r['split']==split];root=Path(root).resolve()
    values=[np.load(safe_file(root,'development/'+r['swin_npy']),allow_pickle=False) for r in selected]
    if any(v.shape!=(3,128) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in values):raise ValueError('Invalid Swin tensor')
    return np.stack(values),np.asarray([int(r['label']) for r in selected],dtype=np.int64),selected

def iter_development(root,split,expected_lock_sha256,representation='swin1d',plan_path=DEFAULT_PLAN):
    """Lazy common-ID inputs; avoid loading all RGB224 tensors into RAM."""
    if split not in ('train','validation'):raise ValueError('Test remains sealed; development iterator only accepts train/validation')
    if representation not in ('swin1d','dinov3'):raise ValueError('Use swin1d or dinov3 representation')
    _,rows=verify_lock(root,expected_lock_sha256,plan_path);root=Path(root).resolve()
    for row in rows:
        if row['split']!=split:continue
        if representation=='swin1d':
            value=np.load(safe_file(root,'development/'+row['swin_npy']),allow_pickle=False)
            shape=(3,128)
        else:
            with Image.open(safe_file(root,'development/'+row['png'])) as image:
                if image.mode!='RGB' or image.size!=(224,224):raise ValueError('Invalid RGB224 PNG')
                value=normalize_rgb(np.asarray(image))
            shape=(3,224,224)
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Invalid development tensor')
        yield value,int(row['label']),row

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('review');p.add_argument('--stage',choices=['pilot','main'],required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot-approval',type=Path)
    for command in ('approve-pilot','lock'):
        p=sub.add_parser(command);p.add_argument('--input',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        if command=='lock':p.add_argument('--pilot-approval',type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--dataset',type=Path,required=True);p.add_argument('--expected-lock-sha256',required=True)
    for p in sub.choices.values():p.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    args=parser.parse_args()
    try:
        if args.command=='review':
            review_template(args.input,args.stage,args.output,args.plan,args.pilot_approval)
            print('Unsigned review generated. No consent/identity approval or dataset lock created.')
        elif args.command=='approve-pilot':
            result=approve_pilot(args.input,args.review,args.output,args.plan);print('Pilot approval recorded from coordinator attestations: '+result['approval_sha256'])
        elif args.command=='lock':
            result=lock(args.input,args.review,args.pilot_approval,args.output,args.plan);print('Internal dataset locked: '+result['lock_sha256']+'; not published, no training performed.')
        else:
            result,_=verify_lock(args.dataset,args.expected_lock_sha256,args.plan);print('Package integrity verified; held-out labels not opened: '+result['lock_sha256'])
    except (ValueError,OSError,KeyError,TypeError,zipfile.BadZipFile) as error:print('ERROR: '+str(error));return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
