"""Check current repository review evidence without rewriting historical phase results."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.audit_repo import audit
from training.drawai.paint_package import verify_bundle
OUT=ROOT/'artifacts/drawai/repo-review'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def main():
    inventory=audit()
    assert not inventory['syntax_errors'] and not inventory['broken_local_links'] and not inventory['retired_project_code']
    tests=read(OUT/'browser-tests.json')['stats']
    assert tests['expected']==33 and tests['unexpected']==tests['flaky']==tests['skipped']==0
    log=(OUT/'python-tests.log').read_text(encoding='utf-8-sig',errors='replace')
    assert re.search(r'Ran 133 tests',log) and re.search(r'\nOK\s*$',log)
    assert 'built in' in (OUT/'build.log').read_text(encoding='utf-8-sig',errors='replace')
    dependencies=read(OUT/'npm-audit.json');assert dependencies['metadata']['vulnerabilities']['total']==0
    anchor='86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2'
    verify_bundle(ROOT/'models/exported/paint35-v1',anchor)
    source_design={}
    with (ROOT/'artifacts/drawai/phase0/file-inventory.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['path'] in ('web/src/drawai/drawai.css','web/src/drawai/drawing-canvas.tsx'):
                assert digest(ROOT/row['path'])==row['sha256'];source_design[row['path']]=row['sha256']
    assert len(source_design)==2
    images=read(OUT/'browser/design-parity.json')['design']
    assert len(images)==2 and all(r['main_sha256']==r['legacy_sha256'] and r['pixel_identical'] for r in images)
    for row in images:
        name='desktop' if row['viewport']['width']==1366 else 'mobile'
        assert digest(OUT/f'browser/ui/empty-{name}.png')==row['main_sha256']
        assert digest(OUT/f'browser/ui/legacy-empty-{name}.png')==row['legacy_sha256']
    frozen_changes=subprocess.check_output(['git','diff','8673f72','--name-only','--','artifacts/drawai'],cwd=ROOT,text=True).splitlines()
    assert not [p for p in frozen_changes if p.startswith('artifacts/drawai/phase')], 'Historical phase evidence was changed'
    cleanup=read(OUT/'cache-cleanup.json');assert cleanup['applied'] and len(cleanup['directories'])==5
    assert all(r['removed'] for r in cleanup['directories'])
    proof=dict(status='PASS',base_commit='8673f724cda03383f074286127f539c5f4af4089',
        python_tests_passed=133,browser_tests_passed=33,browser_new_regression_cases=6,build_passed=True,
        npm_advisories_at_review=0,tracked_files=inventory['tracked_files'],python_syntax_checked=inventory['python_syntax_checked'],
        json_checked=inventory['json_checked'],markdown_checked=inventory['markdown_checked'],active_broken_local_links=0,
        snapshot_links_kept_in_original_form=len(inventory['historical_snapshot_relative_links']),
        local_only_log_links=len(inventory['local_only_link_targets']),duplicate_groups_preserved=len(inventory['duplicate_groups']),
        removed_cache_directories=5,first_cleanup_bytes=cleanup['total_bytes'],
        bundle_sha256=anchor,model_bundle_unchanged=True,design_sources_equal_P0=source_design,empty_design_pixel_identical=True,
        historical_phase_artifacts_unchanged=True,raw_data_deleted=False,retraining=False,private_test_reopened=False,
        product_release_ready=False,physical_touch_tested=False,clean_checkout_accepted=False,
        limitations=['P8 large-model distribution/clean checkout/dataset card still pending',
                    'DINO correct5/10 original PenDigits demos; no new-user canvas accuracy',
                    '37 retained similarity groups across main splits; provenance owner-attested'],
        evidence_sha256={name:digest(OUT/name) for name in ('browser-tests.json','npm-audit.json','cache-cleanup.json','browser/design-parity.json','browser/demo-observations.json')})
    code_paths=['tools/audit_repo.py','tools/clean_repo_cache.ps1','tools/verify_repo_review.py','web/src/drawai/paint-client.ts',
        'web/src/drawai/paint-worker.ts','web/src/drawai/paint-compare.tsx','training/drawai/paint_canvas.py',
        'web/playwright.drawai-review.config.ts','web/tests/drawai-review.spec.ts']
    proof['review_code_sha256']={name:digest(ROOT/name) for name in code_paths}
    write(OUT/'audit.json',inventory);write(OUT/'verification.json',proof)
    print(f"REPO REVIEW PASS: {inventory['tracked_files']} tracked files; {inventory['python_syntax_checked']} Python / {inventory['json_checked']} JSON; 133 Python +33 browser tests; no active broken doc links.")


if __name__=='__main__':main()
