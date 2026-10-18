"""Audit actual P4 image snapshot, not the historical empty stroke inbox."""
import csv
import io
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image, ImageChops
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai import paint_dataset as p
from training.drawai.private_data import load, sha
OUT=ROOT/'artifacts/drawai/phase4-image64'
LOCK=ROOT/'data/private/locked/paint35-main-v1'
REVIEW=ROOT/'data/private/reviews/paint35-p4-v1'


def main():
    anchor=load(OUT/'trusted-lock.json')
    # Verification/development iteration must not open held-out labels.
    with patch.object(zipfile,'ZipFile',side_effect=AssertionError('Development opened test ZIP')):
        record,rows=p.verify_lock(LOCK,anchor['lock_sha256'])
        values=[]
        for representation in ('swin-rows64','dinov3-rgb224'):
            value,label,row=next(p.iter_development(LOCK,'validation',anchor['lock_sha256'],representation))
            assert value.shape==((64,64) if representation=='swin-rows64' else (3,224,224))
            assert value.dtype==np.float32 and np.isfinite(value).all()
            values.append((label,row['sample_id']))
        assert values[0]==values[1]
    for split in ('test','pilot-qc','pilot-reserved'):
        try:next(p.iter_development(LOCK,split,anchor['lock_sha256']))
        except ValueError:pass
        else:raise AssertionError('Forbidden split accepted')
    # P4 packaging QC only. Reading labels here is disclosed; no inference,
    # model selection, augmentation, or P5 loader access to held-out samples.
    with zipfile.ZipFile(LOCK/'heldout-test.zip') as archive:
        assert archive.testzip() is None
        test=list(csv.DictReader(io.StringIO(archive.read('manifest.csv').decode())))
        p.quotas(test,('test',))
        names=archive.namelist()
        assert len(names)==len(set(names))==1002
        assert set(names)=={r['png64'] for r in test}|{'manifest.csv','CUSTODY.txt'}
        for row in test:
            raw=archive.read(row['png64']);assert sha(raw)==row['png_sha256']
            with Image.open(io.BytesIO(raw)) as image:
                assert image.mode=='L' and image.size==(64,64)
                assert sha(np.asarray(image).tobytes())==row['image_sha256']
    review=load(REVIEW/'review.json');visual=load(OUT/'visual-review.json')
    assert len(review['near_groups'])==534 and sum(g['cross_split'] for g in review['near_groups'])==98
    assert visual['contact_sheets_viewed']==5 and visual['cross_split_groups_viewed']==98
    for sheet in review['cross_split_contact_sheets']:
        assert sha((REVIEW/sheet['path']).read_bytes())==sheet['sha256']
    assert visual['review_file_sha256']==sha((REVIEW/'review.json').read_bytes())
    protected=load(ROOT/'artifacts/drawai/phase4/protected-files.json')
    for name,digest in protected.items():assert sha((ROOT/name).read_bytes())==digest,name
    design={}
    for name in ('desktop','mobile','root-desktop','root-mobile'):
        current=Image.open(OUT/f'ui/{name}.png').convert('RGB')
        before=Image.open(ROOT/f'artifacts/drawai/phase0/ui/{name.removeprefix("root-")}.png').convert('RGB')
        assert current.size==before.size and ImageChops.difference(current,before).getbbox() is None
        design[name]=True
    log=(OUT/'python-tests.log').read_text(encoding='utf-8-sig')
    assert re.search(r'\nOK\s*$',log)
    count=int(re.search(r'Ran (\d+) tests',log)[1]);assert count>=95
    assert 'built in' in (OUT/'build.log').read_text(encoding='utf-8-sig')
    stats=load(OUT/'product-regression.json')['stats']
    assert stats['expected']==12 and all(stats[k]==0 for k in ('unexpected','flaky','skipped'))
    result=dict(phase='P4',status='DONE_LOCAL_IMAGE_SNAPSHOT_WITH_DISCLOSED_SIMILARITY',phase_complete=True,
        completion_scope='Reproducible integrity-checked snapshot and train/validation loader; NOT certification of independent source drawings',
        dataset_locked=True,dataset=record['dataset'],lock_sha256=record['lock_sha256'],
        dataset_path=LOCK.relative_to(ROOT).as_posix(),sample_count=6000,
        split_counts=record['split_counts'],writers={'train':20,'validation':5,'test':5},
        samples_per_class={'train':400,'validation':100,'test':100},pilot_excluded=1000,
        near_groups_audited=534,cross_split_groups_visual_qc=98,near_groups_retained=534,exclusions=0,
        similarity_independence_verified=False,test_labels_inspected_for_packaging_qc=True,
        test_used_for_model_selection=False,test_used_for_tuning=False,training_started=False,
        development_loader_does_not_open_test_zip=True,test_archive_members=1002,
        python_tests_passed=count,browser_tests_passed=12,protected_files_verified=len(protected),design=design,
        source_already_public=True,source_kind='human-user-attested',independent_identity_verification=False,
        limitations=['Retrospective fixed writer split; group similarity remains across splits; no clean generalization claim',
                    'No exact pixel duplicates does not prove independent handwriting or eliminate near-copy leakage',
                    'Owner attestation corrects original Synthetic metadata; no independent identity/source proof',
                    'Test ZIP is not encrypted/access-controlled; original NPZ already public',
                    'P4 inspected test images/labels only for source integrity and similarity QC, not model inference',
                    'Snapshot is unaugmented; no new trained/exported classifier; P5 remains separate'])
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'P4 PASS:6000 locked; {count} Python +12 browser tests. Similarity unresolved and disclosed; no training.')


if __name__=='__main__':main()
