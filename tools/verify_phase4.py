"""P4 technical audit; do not infer a collected/locked dataset from synthetic tests."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from PIL import Image,ImageChops

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai.private_data import audit,plan_content,load
OUT=ROOT/'artifacts/drawai/phase4'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protected=load(OUT/'protected-files.json')
    for name,expected in protected.items():assert digest(ROOT/name)==expected,name
    for name in ['locking-guide.md','review-guide.md']:
        assert (ROOT/'docs/drawai/dataset'/name).stat().st_size>0
    log=(OUT/'python-tests.log').read_text(encoding='utf-8-sig')
    assert re.search(r'Ran 68 tests',log) and re.search(r'\nOK\s*$',log)
    assert 'built in' in (OUT/'build.log').read_text(encoding='utf-8-sig')
    tests=load(OUT/'product-regression.json');stats=tests['stats']
    assert stats['expected']==12 and not tests.get('errors')
    assert all(stats[key]==0 for key in ('unexpected','flaky','skipped'))
    designs={}
    for name in ['desktop','mobile','root-desktop','root-mobile']:
        current=Image.open(OUT/f'ui/{name}.png').convert('RGB')
        baseline=Image.open(ROOT/f'artifacts/drawai/phase0/ui/{name.removeprefix("root-")}.png').convert('RGB')
        assert current.size==baseline.size and ImageChops.difference(current,baseline).getbbox() is None
        designs[name]=dict(pixel_identical_to_P0=True,size=list(current.size))
    pilot,_,_=audit(ROOT/'data/private/inbox/pilot',plan_content(),'pilot')
    main_data,_,_=audit(ROOT/'data/private/inbox/main',plan_content(),'main')
    probes=[]
    commands=[['approve-pilot','--input','data/private/inbox/pilot','--review','data/private/reviews/p4-pilot-draft-v1/review.json','--output','data/private/approvals/p4-negative-probe'],
        ['lock','--input','data/private/inbox/main','--review','data/private/reviews/p4-main-draft-v1/review.json',
         '--pilot-approval','data/private/approvals/not-yet-approved/pilot-approval.json','--output','data/private/locked/p4-negative-probe']]
    # These probes are valid only for the currently empty inbox. Do not attempt
    # approvals on later human data just to regenerate a historical P4 report.
    if pilot['input_files']==0 and main_data['input_files']==0:
        for command in commands:
            result=subprocess.run([sys.executable,'-m','training.drawai.private_dataset',*command],cwd=ROOT,capture_output=True,text=True)
            assert result.returncode==1 and 'Full planned quota required' in result.stdout
            assert not (ROOT/command[-1]).exists()
            probes.append(dict(command=command[0],exit_code=result.returncode,output=result.stdout.strip(),no_output_created=True))
    summary=dict(phase='P4',status='TECHNICAL_READY_WAITING_FOR_HUMAN_DATA',phase_complete=False,technical_complete=True,
        python_tests_passed=68,p4_new_tests_passed=12,browser_tests=stats,design=designs,protected_files_verified=len(protected),
        plan_sha256=plan_content()['plan_sha256'],human_pilot_declared_samples=pilot['accepted_samples'],human_main_declared_samples=main_data['accepted_samples'],
        pilot_quota_complete=pilot['quota_complete'],main_quota_complete=main_data['quota_complete'],
        dataset_locked=False,locked_dataset_path=None,kaggle_published=False,training_started=False,
        negative_real_inbox_probes=probes,synthetic_tests_are_not_human_data=True,
        test_custody='Implemented hash-sealed ZIP; no real test archive exists yet; not encryption/access control',
        pending=['Actual consenting human pilot (250 samples) and coordinator approval',
                 'Actual main collection (6000 samples, 30 new writers) under fixed plan',
                 'Source-bound human/consent/identity/exclusion/near-duplicate review and actual final lock'],
        limitations=['Positive locking tests use a miniature synthetic contract with read_plan patched only inside temporary unit tests',
                     'Production policy remains 250 pilot /6000 main; no clean install or full-size6000 performance benchmark claimed',
                     'Manual attestations and checksums cannot independently prove identity or consent; keep external review evidence',
                     'Coordinator must hold test ZIP and trusted lock hash separately; no cryptographic signature or OS ACL enforcement'])
    (OUT/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,value in [('current-pilot-audit',pilot),('current-main-audit',main_data)]:
        (OUT/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"P4 technical PASS:68 Python and12 browser tests; model/design preserved. Actual pilot {pilot['accepted_samples']}/250; main {main_data['accepted_samples']}/6000; no real dataset lock.")

if __name__=='__main__':main()
