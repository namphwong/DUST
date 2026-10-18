"""P3 image-only Paint dataset preparation, separate from captured stroke sessions.

No image-to-stroke reconstruction, model training or P4 lock is performed.
"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image,ImageDraw
from .private_data import ROOT,canonical,sha,load,dump,new_output
from .raster import normalize_rgb

SOURCE_SHA256='c594927b59509bb62cb185cf5e64034ee641b046798f0f192d6155fd91f71a64'
VERSION='drawai-paint-image64-v1'
MANIFEST_FIELDS=['sample_id','source_index','person_id','label','repetition','split','png64','png_sha256','image_sha256','swin_rows_sha256','dinov3_rgb_sha256','dinov3_tensor_sha256']

def contract():
    return dict(version=VERSION,source=dict(shape=[64,64],dtype='uint8',range=[0,255],background='white',ink='black'),
        swin=dict(shape=[64,64],layout='CL',channels='64 column pixel features',length='64 row positions in top-to-bottom order',
            dtype='float32',normalization='(255-image)/255 in float32, then transpose; no learned statistics',
            representation='spatial row sequence, NOT temporal writing trajectory',existing_stroke_checkpoint_compatible=False),
        dinov3=dict(shape=[3,224,224],dtype='float32',resize='Pillow bilinear 64->224; no crop, no bbox normalization',
            channels='repeat grayscale into RGB',rescale_factor=1/255,mean=[.485,.456,.406],std=[.229,.224,.225]),
        image_contract_not_stroke_raster=True,augmentation_applied=False,original_strokes_available=False)

def swin_rows(image):
    if image.shape!=(64,64) or image.dtype!=np.uint8:raise ValueError('Expected uint8 grayscale64 image')
    return np.ascontiguousarray(((np.float32(255)-image.astype(np.float32))/np.float32(255)).T)

def dino_rgb(image):
    if image.shape!=(64,64) or image.dtype!=np.uint8:raise ValueError('Expected uint8 grayscale64 image')
    rgb=np.repeat(image[:,:,None],3,axis=2)
    return np.asarray(Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR))

def policy(source_sha256=SOURCE_SHA256):
    value=dict(version='drawai-paint-image64-plan-v1',protocol_version=VERSION,source_sha256=source_sha256,
        assignment='Retrospective person-ID split fixed before model training/evaluation; not the prospective stroke-collection plan',
        pilot_qc=dict(person_ids=list(range(1,6)),repetitions=list(range(1,6)),samples=250,
            meaning='post-collection image/provenance QC; not a live usability pilot of /collect'),
        pilot_reserved=dict(person_ids=list(range(1,6)),repetitions=list(range(6,21)),samples=750),
        train=dict(person_ids=list(range(6,26)),repetitions=list(range(1,21)),samples=4000),
        validation=dict(person_ids=list(range(26,31)),repetitions=list(range(1,21)),samples=1000),
        test=dict(person_ids=list(range(31,36)),repetitions=list(range(1,21)),samples=1000),
        selection_uses_model_predictions=False,pilot_people_excluded_from_main=True,dataset_locked=False)
    return {**value,'plan_sha256':sha(canonical(value))}

def split_for(person,repetition):
    if not 1<=person<=35 or not 1<=repetition<=20:raise ValueError('Person/repetition outside Paint35 policy')
    return ('pilot-qc' if repetition<=5 else 'pilot-reserved') if person<=5 else 'train' if person<=25 else 'validation' if person<=30 else 'test'

def read_source(path,expected_sha=SOURCE_SHA256):
    path=Path(path);payload=path.read_bytes();digest=sha(payload)
    if expected_sha and digest!=expected_sha:raise ValueError('Source SHA mismatch; review/version the new source explicitly')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info=archive.infolist();expected={'images.npy','labels.npy','person_ids.npy','sample_ids.npy','description.npy'}
        if len(info)!=5 or {i.filename for i in info}!=expected:raise ValueError('Unexpected/duplicate NPZ members')
        if sum(i.file_size for i in info)>128*1024*1024 or any(i.file_size>64*1024*1024 for i in info):raise ValueError('NPZ uncompressed size exceeds limit')
    with np.load(io.BytesIO(payload),allow_pickle=False) as source:
        arrays={key:source[key] for key in source.files}
    images=arrays['images'];n=len(images)
    if images.ndim!=3 or images.shape[1:]!=(64,64) or images.dtype!=np.uint8 or not 1<=n<=10000:raise ValueError('Invalid images N64x64 uint8')
    for key in ('labels','person_ids','sample_ids'):
        a=arrays[key]
        if a.shape!=(n,) or a.dtype.kind not in 'iu':raise ValueError('Invalid integer vector: '+key)
    if np.any((arrays['labels']<0)|(arrays['labels']>9)):raise ValueError('Invalid digit labels')
    if np.any((arrays['person_ids']<1)|(arrays['person_ids']>35)) or np.any((arrays['sample_ids']<1)|(arrays['sample_ids']>20)):
        raise ValueError('Person/repetition IDs outside source policy')
    description=arrays['description']
    if description.shape!=() or description.dtype.kind!='U' or len(description.item())>4096:raise ValueError('Invalid scalar Unicode description')
    return arrays,dict(filename=path.name,bytes=len(payload),sha256=digest,original_description=description.item())

def check_provenance(value,digest):
    required={'version','source_sha256','source_kind','capture_tool','person_ids_are_distinct_people','permissions',
              'metadata_description_is_incorrect','original_strokes_available','attestation_source','confirmed_date'}
    if not isinstance(value,dict) or set(value)!=required or value['version']!='drawai-paint-provenance-v1':raise ValueError('Invalid provenance schema')
    if value['source_sha256']!=digest or value['source_kind']!='human-user-attested' or value['capture_tool']!='Paint':raise ValueError('Provenance source mismatch')
    if value['person_ids_are_distinct_people'] is not True or value['metadata_description_is_incorrect'] is not True or value['original_strokes_available'] is not False:
        raise ValueError('Missing explicit origin/metadata/stroke confirmation')
    if value['permissions']!={'training':True,'evaluation':True,'kaggle_publication':True}:raise ValueError('Missing confirmed data-use scope')
    if value['attestation_source']!='Explicit human user confirmation in this project chat' or value['confirmed_date']!='2026-10-18':raise ValueError('Unrecognized attestation record')

def audit_arrays(arrays,source,provenance=None):
    if provenance is not None:check_provenance(provenance,source['sha256'])
    images=arrays['images'];labels=arrays['labels'];people=arrays['person_ids'];reps=arrays['sample_ids']
    counts=Counter((int(p),int(y),int(r)) for p,y,r in zip(people,labels,reps))
    full={(p,y,r) for p in range(1,36) for y in range(10) for r in range(1,21)}
    duplicate_keys=[list(k) for k,n in counts.items() if n>1]
    missing=sorted(full-set(counts));errors=[]
    if duplicate_keys:errors.append('Duplicate composite person/label/repetition keys')
    if missing:errors.append('Missing source quota entries')
    exact={};duplicates=[];blank=[];split_counts=Counter();class_counts=defaultdict(Counter);person_counts=defaultdict(Counter)
    near_buckets=defaultdict(list);parents={};pairs=0
    def find(i):
        parents.setdefault(i,i)
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,(image,label,person,rep) in enumerate(zip(images,labels,people,reps)):
        split=split_for(int(person),int(rep));split_counts[split]+=1;class_counts[split][int(label)]+=1;person_counts[int(person)][int(label)]+=1
        digest=sha(image.tobytes())
        if digest in exact:
            other=exact[digest];duplicates.append(dict(source_index=i,other_index=other,cross_person=bool(people[i]!=people[other]),cross_label=bool(labels[i]!=labels[other])))
        else:exact[digest]=i
        if int((image<192).sum())<4 or int(image.max())-int(image.min())<16:blank.append(i)
        # Background dominates a raw whole-image dHash for sparse digits.
        # Review on an ink ROI, within a label, with a second cosine gate.
        ys,xs=np.where(image<192)
        if len(xs):roi=image[max(0,int(ys.min())-2):min(64,int(ys.max())+3),max(0,int(xs.min())-2):min(64,int(xs.max())+3)]
        else:roi=image
        roi=Image.fromarray(roi)
        descriptor=(255-np.asarray(roi.resize((32,32),Image.Resampling.BILINEAR)).astype(np.float32)).ravel()/255
        descriptor[descriptor<31/255]=0
        norm=float(np.linalg.norm(descriptor));descriptor=descriptor/max(norm,1e-12)
        small=np.asarray(roi.resize((9,8),Image.Resampling.BILINEAR));bits=0
        for bit in (small[:,1:]>small[:,:-1]).flat:bits=(bits<<1)|int(bit)
        candidates={}
        for chunk in range(4):
            for other,b,other_descriptor in near_buckets[(int(label),chunk,(bits>>(chunk*16))&65535)]:candidates[other]=(b,other_descriptor)
        for other,(b,other_descriptor) in candidates.items():
            if (bits^b).bit_count()<=2 and float(descriptor@other_descriptor)>=.995:
                pairs+=1;parents[find(i)]=find(other)
        for chunk in range(4):near_buckets[(int(label),chunk,(bits>>(chunk*16))&65535)].append((i,bits,descriptor))
    if duplicates:errors.append('Exact duplicate images require review')
    if blank:errors.append('Blank/low-contrast candidates require review')
    components=defaultdict(list)
    for index in parents:components[find(index)].append(index)
    groups=[dict(group_id='near_'+sha(canonical(sorted(indices))),source_indices=sorted(indices),
        cross_split=len({split_for(int(people[i]),int(reps[i])) for i in indices})>1) for indices in components.values() if len(indices)>1]
    groups.sort(key=lambda g:g['group_id'])
    return dict(version='drawai-paint-image64-audit-v1',source=source,sample_count=len(images),person_id_count=len(set(people.tolist())),
        source_kind='human-user-attested' if provenance else 'unverified',independent_human_verification=False,
        metadata_conflict=dict(original_mentions_synthetic='synthetic' in source['original_description'].lower(),
            resolution='User explicitly confirmed incorrect packaging description; original bytes preserved' if provenance else 'Unresolved'),
        source_quota_complete=len(images)==7000 and not missing and not duplicate_keys,
        duplicate_composite_keys=duplicate_keys,missing_composite_keys_count=len(missing),exact_duplicates=duplicates,blank_or_low_contrast_indices=blank,
        counts_by_split=dict(split_counts),counts_by_split_and_class={s:{str(k):n for k,n in sorted(c.items())} for s,c in class_counts.items()},
        counts_by_person_and_class={str(p):{str(k):n for k,n in sorted(c.items())} for p,c in sorted(person_counts.items())},
        near_duplicate_pairs=pairs,near_duplicate_groups=groups,
        near_duplicate_policy='Within-label ink-ROI dHash Hamming<=2 AND cosine>=0.995 on32x32 ink; heuristic only, no automatic exclusions or model filtering',
        errors=errors,original_strokes_available=False,legacy_swin_checkpoint_compatible=False,dataset_locked=False,
        test_used_for_model_selection=False,provenance_confirmation_present=provenance is not None)

def build(source_path,provenance_path,output):
    output=Path(output).resolve();source_path=Path(source_path).resolve()
    if output==source_path.parent or source_path.parent in output.parents:raise ValueError('Keep source and prepared output separate')
    arrays,source=read_source(source_path);provenance=load(provenance_path);report=audit_arrays(arrays,source,provenance)
    if report['errors']:raise ValueError('Source audit errors: '+str(report['errors']))
    out=new_output(output);(out/'images64').mkdir();(out/'pilot-qc').mkdir()
    for key in ('images','labels','person_ids','sample_ids'):np.save(out/(key+'.npy'),arrays[key],allow_pickle=False)
    with (out/'manifest.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=MANIFEST_FIELDS);writer.writeheader()
        for i,(image,label,person,rep) in enumerate(zip(arrays['images'],arrays['labels'],arrays['person_ids'],arrays['sample_ids'])):
            sid=f'paint35_p{person:03}_d{label}_r{rep:02}';png=f'images64/{sid}.png';Image.fromarray(image).save(out/png)
            rgb=dino_rgb(image)
            writer.writerow(dict(sample_id=sid,source_index=i,person_id=int(person),label=int(label),repetition=int(rep),split=split_for(int(person),int(rep)),
                png64=png,png_sha256=sha((out/png).read_bytes()),image_sha256=sha(image.tobytes()),swin_rows_sha256=sha(swin_rows(image).tobytes()),
                dinov3_rgb_sha256=sha(rgb.tobytes()),dinov3_tensor_sha256=sha(normalize_rgb(rgb).tobytes())))
    # Five real source cohorts, five repetitions each, all 250 QC images.
    for person in range(1,6):
        board=Image.new('RGB',(1000,530),'white');draw=ImageDraw.Draw(board)
        for rep in range(1,6):
            for label in range(10):
                index=int(np.flatnonzero((arrays['person_ids']==person)&(arrays['labels']==label)&(arrays['sample_ids']==rep))[0])
                tile=Image.fromarray(arrays['images'][index]).convert('RGB').resize((80,80),Image.Resampling.NEAREST)
                x=label*100+10;y=(rep-1)*106+20;board.paste(tile,(x,y));draw.text((x,y-16),f'p{person} d{label} r{rep}',fill='black')
        board.save(out/f'pilot-qc/person-{person:03}.png')
    dump(out/'audit.json',report);dump(out/'provenance.json',provenance);dump(out/'plan.json',policy(source['sha256']));dump(out/'preprocessing.json',contract())
    dump(out/'dataset.json',dict(version=VERSION,source_sha256=source['sha256'],sample_count=7000,main_sample_count=6000,pilot_qc_samples=250,pilot_reserved_samples=750,
        manifest_sha256=sha((out/'manifest.csv').read_bytes()),plan_sha256=policy(source['sha256'])['plan_sha256'],
        source_kind='human-user-attested',precise_collection_time_available=False,input_device_available=False,
        raw_stroke_sessions_available=False,retrospective_partition=True,dataset_locked=False,training_started=False,
        same_ids_for_both_image_representations=True,legacy_stroke_pipeline_compatible=False))
    return report

def iter_prepared(root,split,representation):
    if split not in ('pilot-qc','train','validation'):raise ValueError('P3 loader excludes held-out test and pilot-reserved')
    if representation not in ('swin-rows64','dinov3-rgb224'):raise ValueError('Unsupported image representation')
    root=Path(root);meta=load(root/'dataset.json')
    if sha((root/'manifest.csv').read_bytes())!=meta['manifest_sha256']:raise ValueError('Manifest hash mismatch')
    with (root/'manifest.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    for row in rows:
        if row['split']!=split:continue
        sid=row['sample_id'];path=root/'images64'/(sid+'.png')
        if row['png64']!=f'images64/{sid}.png' or path.parent.resolve()!=(root/'images64').resolve() or path.is_symlink():raise ValueError('Invalid image path')
        if sha(path.read_bytes())!=row['png_sha256']:raise ValueError('PNG hash mismatch')
        with Image.open(path) as image:
            if image.mode!='L' or image.size!=(64,64):raise ValueError('Invalid source PNG')
            pixels=np.asarray(image)
        values=swin_rows(pixels) if representation=='swin-rows64' else normalize_rgb(dino_rgb(pixels))
        expected=row['swin_rows_sha256'] if representation=='swin-rows64' else row['dinov3_tensor_sha256']
        if sha(values.tobytes())!=expected:raise ValueError('Preprocessing hash mismatch')
        yield values,int(row['label']),row

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--provenance',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=build(args.source,args.provenance,args.output)
    print(f"Prepared {report['sample_count']} Paint images; {report['person_id_count']} user-attested person IDs; no P4 lock/training.")

if __name__=='__main__':main()
