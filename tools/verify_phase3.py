"""P3 technical acceptance; never count automated pointer events as human pilot."""
import hashlib
import json
from pathlib import Path
import re
import sys
from PIL import Image,ImageChops

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai.private_data import audit,plan_content,load
OUT=ROOT/'artifacts/drawai/phase3'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def suite(path,count):
    value=load(path);stats=value['stats']
    assert stats['expected']==count and not value.get('errors')
    assert all(stats[k]==0 for k in ('unexpected','flaky','skipped'))
    return stats

def main():
    protected=load(OUT/'pre-p3-protected.json')
    for path,digest in protected.items():assert sha(ROOT/path)==digest,path
    plan=plan_content()
    assert load(ROOT/'docs/drawai/collection/split-policy.json')==plan
    assert load(ROOT/'data/private/plans/private-v1/plan.json')==plan
    assert sha(ROOT/'web/public/collection-protocol.md')==sha(ROOT/'docs/drawai/collection/protocol.md')
    for name in ['session.schema.json','protocol.md','consent-template.md','operator-guide.md','dataset-card.md',
                 'registry-template.csv','rejection-log-template.csv','pilot-report-template.md','manifest-template.csv']:
        assert (ROOT/'docs/drawai/collection'/name).stat().st_size>0
    log=(OUT/'python-tests.log').read_text(encoding='utf-8-sig')
    assert re.search(r'Ran 56 tests',log) and re.search(r'\nOK\s*$',log)
    assert 'built in' in (OUT/'build.log').read_text(encoding='utf-8-sig')
    suites=dict(collection=suite(OUT/'browser-tests.json',5),product=suite(OUT/'product-regression.json',12))
    designs={}
    for name in ['desktop','mobile','root-desktop','root-mobile']:
        current=Image.open(OUT/f'ui/{name}.png').convert('RGB')
        reference=Image.open(ROOT/f'artifacts/drawai/phase0/ui/{name.removeprefix("root-")}.png').convert('RGB')
        assert current.size==reference.size and ImageChops.difference(current,reference).getbbox() is None,name
        designs[name]=dict(pixel_identical_to_P0=True,size=list(current.size))
    report,_,_=audit(ROOT/'data/private/inbox/pilot',plan,'pilot')
    # Human provenance/consent/pilot review cannot be established by software.
    # Even quota_complete=True would still require coordinator evidence/review.
    result=dict(phase='P3',status='TECHNICAL_READY_WAITING_FOR_HUMAN_PILOT',phase_complete=False,
        technical_complete=True,all_executable_checks_passed=True,python_tests_passed=56,browser_tests=suites,
        design=designs,protected_files_verified=len(protected),plan_sha256=plan['plan_sha256'],
        planned_pilot=dict(participants=5,samples=250),planned_main=dict(participants=30,samples=6000,train_writers=20,validation_writers=5,test_writers=5),
        pilot_session_files=report['input_files'],pilot_declared_samples=report['accepted_samples'],pilot_quota_complete=report['quota_complete'],
        pilot_human_origin_verified=False,automated_test_fixtures_count_as_human_data=False,
        dataset_locked=False,published_to_kaggle=False,main_design_and_serving_model_unchanged=True,
        pending=['Collect pilot with five distinct consenting real participants (250 valid samples)',
                 'Coordinator verify identities/consent; review duplicates, collection issues and pilot report before P4'],
        limitations=['Local mouse and simulated mobile viewport tested; real touch/pen not tested',
                     'Collection browser memory requires JSON download/resume; no persistent autosave',
                     'Participant uniqueness and actual consent require coordinator evidence; schema is not proof of identity',
                     'P3 builds are preliminary staging, never locked or automatically published'])
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'current-pilot-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"P3 technical PASS:56 Python,17 browser tests; design/model preserved. Pilot {report['accepted_samples']}/250 claimed samples; human pilot still requires coordinator review.")

if __name__=='__main__':main()
