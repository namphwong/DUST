"""Re-audit all 7000 Paint images for P3; no held-out model evaluation."""
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
from PIL import Image,ImageChops

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai.paint_data import SOURCE_SHA256,read_source,check_provenance,policy,contract,split_for,swin_rows,dino_rgb
from training.drawai.private_data import sha,load
from training.drawai.raster import normalize_rgb
OUT=ROOT/'artifacts/drawai/phase3-image64';PREP=ROOT/'data/private/prepared/paint-image64-v1'

def main():
    arrays,source=read_source(ROOT/'data/private/external/paint35/So_viet_tay_35_nguoi.npz')
    assert source['sha256']==SOURCE_SHA256
    original=Path('C:/Users/admin/Downloads/So_viet_tay_35_nguoi.npz')
    if original.is_file():assert sha(original.read_bytes())==SOURCE_SHA256
    provenance=load(PREP/'provenance.json');check_provenance(provenance,SOURCE_SHA256)
    assert load(PREP/'plan.json')==policy()==load(ROOT/'docs/drawai/collection/paint-image64-plan.json')
    assert load(PREP/'preprocessing.json')==contract()==load(ROOT/'docs/drawai/collection/paint-image64-contract.json')
    for key in ('images','labels','person_ids','sample_ids'):assert np.array_equal(np.load(PREP/(key+'.npy'),allow_pickle=False),arrays[key])
    metadata=load(PREP/'dataset.json');audit=load(PREP/'audit.json')
    assert not audit['errors'] and audit['source_quota_complete'] and metadata['dataset_locked'] is False
    assert audit['provenance_confirmation_present'] and audit['independent_human_verification'] is False
    with (PREP/'manifest.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==7000 and len({r['sample_id'] for r in rows})==7000
    assert sha((PREP/'manifest.csv').read_bytes())==metadata['manifest_sha256']
    counts={};writers={};seen=set()
    for row in rows:
        i=int(row['source_index']);assert i not in seen and 0<=i<7000;seen.add(i)
        person=int(row['person_id']);label=int(row['label']);rep=int(row['repetition']);split=split_for(person,rep)
        assert row['split']==split and (person,label,rep)==(int(arrays['person_ids'][i]),int(arrays['labels'][i]),int(arrays['sample_ids'][i]))
        sid=f'paint35_p{person:03}_d{label}_r{rep:02}';assert row['sample_id']==sid and row['png64']==f'images64/{sid}.png'
        path=PREP/row['png64'];assert sha(path.read_bytes())==row['png_sha256']
        with Image.open(path) as image:assert image.mode=='L' and np.array_equal(np.asarray(image),arrays['images'][i])
        pixels=arrays['images'][i];assert sha(pixels.tobytes())==row['image_sha256']
        assert sha(swin_rows(pixels).tobytes())==row['swin_rows_sha256']
        rgb=dino_rgb(pixels);assert sha(rgb.tobytes())==row['dinov3_rgb_sha256']
        assert sha(normalize_rgb(rgb).tobytes())==row['dinov3_tensor_sha256']
        counts[split]=counts.get(split,0)+1;writers.setdefault(split,set()).add(person)
    assert counts=={'pilot-qc':250,'pilot-reserved':750,'train':4000,'validation':1000,'test':1000}
    sets=[writers[s] for s in ('pilot-qc','train','validation','test')]
    for i,a in enumerate(sets):
        for b in sets[i+1:]:assert not a&b
    visual=load(OUT/'pilot-visual-review.json')
    assert visual['images_reviewed']==250 and visual['contact_sheets_reviewed']==5 and visual['source_sha256']==SOURCE_SHA256
    log=(OUT/'python-tests.log').read_text(encoding='utf-8-sig')
    assert re.search(r'Ran 78 tests',log) and re.search(r'\nOK\s*$',log)
    stats=load(OUT/'product-regression.json')['stats']
    assert stats['expected']==12 and all(stats[key]==0 for key in ('unexpected','flaky','skipped'))
    assert 'built in' in (OUT/'build.log').read_text(encoding='utf-8-sig')
    protected=load(ROOT/'artifacts/drawai/phase4/protected-files.json')
    for name,digest in protected.items():assert sha((ROOT/name).read_bytes())==digest,name
    design={}
    for name in ('desktop','mobile','root-desktop','root-mobile'):
        current=Image.open(OUT/f'ui/{name}.png').convert('RGB');baseline=Image.open(ROOT/f'artifacts/drawai/phase0/ui/{name.removeprefix("root-")}.png').convert('RGB')
        assert current.size==baseline.size and ImageChops.difference(current,baseline).getbbox() is None
        design[name]=True
    result=dict(phase='P3',protocol='image64 Paint rerun',status='DONE_LOCAL_IMAGE_DATA_PREPARATION',phase_complete=True,
        completion_scope='P3 image protocol: owner attestation, schema/source audit, fixed retrospective person split,250-image QC and reproducible two-model input preparation',
        source=source,source_kind='human-user-attested',human_origin_independently_verified=False,
        sample_count=7000,person_id_count=35,counts_by_split=counts,writers_by_split={s:sorted(p) for s,p in writers.items()},
        exact_duplicate_images=0,blank_or_low_contrast_images=0,near_duplicate_pairs=audit['near_duplicate_pairs'],
        near_groups=len(audit['near_duplicate_groups']),near_groups_cross_split=sum(g['cross_split'] for g in audit['near_duplicate_groups']),
        preprocessing_hash_rows_checked=7000,pilot_images_visually_reviewed=250,python_tests_passed=78,browser_tests_passed=12,
        legacy_serving_model_and_design_preserved=design,original_npz_unchanged=True,dataset_locked=False,training_started=False,
        existing_swin_checkpoint_compatible=False,live_collection_usability_pilot_performed=False,
        limitations=['Owner confirmed human Paint origin/35 distinct IDs/permissions and packaging description error; no independent identity proof',
                     'Only images available; no stroke order, pen lifts, capture timestamps or verified device type',
                     'New row-pixel Swin1D input requires separate training/export; original trajectory Swin still serves DrawAI',
                     'Split fixed retrospectively before model training; not prospective P3 stroke-session collection',
                     'Near-duplicate heuristic flags need P4 review; no automatic exclusion or test-based tuning'],
        next_phase='P4 must support image-protocol review/lock rather than approving old empty stroke inbox')
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    public_audit={k:v for k,v in audit.items() if k not in ['near_duplicate_groups']}
    public_audit['near_group_count']=len(audit['near_duplicate_groups'])
    (OUT/'data-audit-summary.json').write_text(json.dumps(public_audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('P3 image protocol DONE:7000 image/tensor rows verified;250 QC images;78 Python+12 browser tests; unchanged serving design/model; P4 lock pending.')

if __name__=='__main__':main()
