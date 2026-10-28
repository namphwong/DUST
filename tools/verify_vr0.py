"""Verify the VR0 deployment specification and local anchors; never run/train on private test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai.private_data import canonical,sha
from training.drawai.paint_package import verify_bundle,spec,TOLERANCES,LOCK_HASH

CONTRACT=ROOT/'deployment/vr0/public-contract.json'
ANCHOR=ROOT/'deployment/vr0/trusted-contract.json'
OUT=ROOT/'artifacts/drawai/vr0'
BUNDLE_HASH='86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2'
SELECTION_HASH='827f063bb6fc814d94406204f2c45ac91668988c0273329a59c3f5aaeccbec51'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def local_path(relative):
    require(isinstance(relative,str),'Expected local path')
    path=(ROOT/relative).resolve()
    require(path.is_relative_to(ROOT),'Path outside project')
    return path


def validate_contract(value,expected_sha256):
    require(isinstance(value,dict),'Expected contract object')
    unsigned={k:v for k,v in value.items() if k!='contract_sha256'}
    require(value.get('contract_sha256')==expected_sha256==sha(canonical(unsigned)),'Contract anchor/hash mismatch')
    require(value['version']=='drawai-public-deployment-contract-v1' and value['phase']=='VR0' and value['state']=='SPEC_LOCKED_NOT_DEPLOYED','Wrong contract version/state')
    architecture=value['architecture']
    require(architecture['default_model']=='dinov3-vits16' and architecture['baseline_model']=='swin1d-image','Default/baseline changed')
    require(architecture['silent_fallback'] is False and architecture['model_fusion'] is False and architecture['public_inference_ready'] is False,'Silent fallback/fusion/ready claim')
    require(set(value['models'])==set(TOLERANCES),'Model inventory changed')
    for family,model in value['models'].items():
        require(model['spec']==spec(family),'Artifact format/shape changed')
        require(model['tolerance']==TOLERANCES[family],'Numerical tolerance changed')
        require(model['role']==('default' if family=='dinov3-vits16' else 'baseline'),'Model role changed')
    anchors=value['source_anchors']
    require(anchors['bundle_sha256']==BUNDLE_HASH and anchors['selection_sha256']==SELECTION_HASH and anchors['dataset_lock_sha256']==LOCK_HASH,'Source anchor changed')
    public=value['profiles']['public_target']
    require(public['implemented'] is False and public['vite_base']=='/','VR0 public profile is specification only')
    require(public['public_routes']==['/','/drawai','/drawai/compare'],'Public route scope changed')
    require(public['production_urls']==dict(frontend=None,backend=None),'Unverified public URL claim')
    require(public['backend_model_ids']==['dinov3-vits16'] and public['browser_model_ids']==['swin1d-image'],'Deployment model placement changed')
    require(value['input_contract']['api_routes']==dict(health='/health',predict='/predict'),'Public endpoint contract changed')
    budgets=value['budgets']['backend']
    require(0<budgets['peak_cgroup_memory_max_bytes']<512000000 and budgets['container_memory_limit_bytes']==536870912,'Memory guard exceeds free target')
    require(value['security_privacy']['worker_processes']==value['security_privacy']['max_concurrent_inferences']==1,'Unbounded production concurrency')
    require(value['security_privacy']['request_body_logged'] is False and value['security_privacy']['image_persistence'] is False,'Image persistence/logging changed')
    for name in ('product_release_ready','physical_touch_tested','cloud_memory_tested','cloud_latency_tested','clean_checkout_verified','model_download_public','onnx_dinov3_approved'):
        require(value['claims'][name] is False,'Unmeasured completion claim: '+name)
    return value


def verify():
    anchor=read(ANCHOR);value=validate_contract(read(CONTRACT),anchor['contract_sha256'])
    require(file_sha(CONTRACT)==anchor['public_contract_file_sha256'],'Contract file bytes changed')
    bundle,registry=verify_bundle(ROOT/'models/exported/paint35-v1',BUNDLE_HASH)
    selection=read(ROOT/'artifacts/drawai/phase5-image64/selection.json')
    require(selection['selection_sha256']==SELECTION_HASH==sha(canonical({k:v for k,v in selection.items() if k!='selection_sha256'})),'P5 selection hash mismatch')
    require(selection['candidate_default']==dict(family='dinov3-vits16',seed=42) and selection['test_accessed'] is False,'Selection protocol changed')
    dataset=read(ROOT/'data/private/locked/paint35-main-v1/dataset-lock.json')
    require(dataset['lock_sha256']==LOCK_HASH==sha(canonical({k:v for k,v in dataset.items() if k!='lock_sha256'})),'P4 dataset lock mismatch')
    # P5 binds the exact plan file bytes (including newline), not canonical JSON.
    require(file_sha(ROOT/'artifacts/drawai/phase5-image64/experiment-plan.json')==selection['plan_sha256']==value['source_anchors']['experiment_plan_sha256'],'Experiment plan binding changed')
    require(dataset['bindings']['source_sha256']==value['source_anchors']['original_npz_sha256'] and selection['lock_sha256']==LOCK_HASH,'Dataset/source lineage changed')
    checked={}
    for section in ('protected_files_sha256','reference_files_sha256'):
        for relative,expected in value[section].items():
            require(file_sha(local_path(relative))==expected,'Frozen file changed: '+relative);checked[relative]=expected
    for family,model in value['models'].items():
        winner=selection['winners'][family]
        require(model['seed']==winner['seed'] and model['checkpoint']['sha256']==winner['sha256'],'Wrong selected checkpoint')
        for key in ('checkpoint','artifact'):
            item=model[key];require(file_sha(local_path(item['path']))==item['sha256'],'Model/checkpoint bytes mismatch: '+family);checked[item['path']]=item['sha256']
        require(model['artifact']['sha256']==bundle['files'][family+'/'+spec(family)['model_filename']]['sha256'],'Runtime artifact not in verified bundle')
    historical=read(ROOT/'artifacts/drawai/repo-review/verification.json')
    require(historical['status']=='PASS' and historical['python_tests_passed']==133 and historical['browser_tests_passed']==33,'Missing historical technical baseline')
    screenshots={}
    for name,relative in value['ui_contract']['initial_empty_screenshots'].items():
        screenshots[relative]=file_sha(local_path(relative))
        comparison=ROOT/f'artifacts/drawai/repo-review/browser/ui/legacy-empty-{name}.png'
        require(screenshots[relative]==file_sha(comparison),'Baseline empty UI comparison mismatch')
    contract_source=subprocess.check_output(['git','rev-parse',value['baseline_commit']],cwd=ROOT,text=True).strip()
    require(contract_source==value['baseline_commit'],'Baseline commit unavailable')
    runtime_changes=subprocess.check_output(['git','diff',contract_source,'--name-only','--','web/src','web/vite.config.ts','training/drawai','models'],cwd=ROOT,text=True).splitlines()
    checklist=read(ROOT/'deployment/vr0/acceptance-checklist.json')
    require(checklist['contract_sha256']==anchor['contract_sha256'],'Checklist uses wrong contract')
    require(file_sha(ROOT/'deployment/vr0/acceptance-checklist.json')==anchor['acceptance_checklist_file_sha256'],'Checklist file bytes changed')
    require(len(checklist['checks'])==19 and len({r['id'] for r in checklist['checks']})==19,'Checklist inventory changed')
    require(sum(r['status']=='PASS' and r['gate']=='VR0' for r in checklist['checks'])==5,'VR0 checklist evidence incomplete')
    require(all(r['status'] in ('PASS','NOT_RUN') for r in checklist['checks']),'Invalid acceptance status')
    require(all(r['status']=='NOT_RUN' for r in checklist['checks'] if r['gate']!='VR0'),'Future acceptance was claimed without execution')
    return dict(phase='VR0',status='DONE_SCOPE_AND_ACCEPTANCE_LOCKED',date='2026-10-28',contract_sha256=anchor['contract_sha256'],
        baseline_commit=contract_source,default_model='dinov3-vits16',baseline_model='swin1d-image',local_anchor_checks='PASS',
        checked_files_sha256=checked,ui_baseline_screenshots_sha256=screenshots,dataset_lock_metadata_verified=True,
        checkpoint_hashes_verified=True,bundle_inventory_and_hashes_verified=True,selection_verified=True,
        private_test_images_read=False,heldout_zip_opened=False,training_executed=False,app_runtime_modified=bool(runtime_changes),runtime_changes_since_baseline=runtime_changes,
        historical_tests=dict(python=133,browser=33,rerun_in_vr0=False,source='artifacts/drawai/repo-review/verification.json'),
        checklist_status_counts={s:sum(r['status']==s for r in checklist['checks']) for s in ('PASS','NOT_RUN')},
        checklist_sha256=file_sha(ROOT/'deployment/vr0/acceptance-checklist.json'),
        public_target_profile_implemented=False,cloud_tests_run=False,product_release_ready=False,deployed=False,
        next_phase='VR1',limitations=value['claims'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-evidence',action='store_true');args=parser.parse_args()
    result=verify()
    if args.write_evidence:
        import io
        import unittest
        from time import perf_counter
        suite=unittest.defaultTestLoader.loadTestsFromName('training.tests.test_drawai_deployment_contract')
        before=perf_counter();run=unittest.TextTestRunner(stream=io.StringIO(),verbosity=2).run(suite)
        require(run.wasSuccessful() and run.testsRun==10 and not run.skipped,'Contract gate tests failed or skipped')
        result['contract_gate_tests']=dict(passed=run.testsRun,failed=len(run.failures),errors=len(run.errors),skipped=len(run.skipped),
            seconds=perf_counter()-before,source='training/tests/test_drawai_deployment_contract.py',
            source_sha256=file_sha(ROOT/'training/tests/test_drawai_deployment_contract.py'))
        result['verifier_source_sha256']=file_sha(Path(__file__))
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'verification.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
    print('VR0 PASS: selected checkpoints, full model bundle, P4/P5 anchors, frozen preprocessing/design and public specification; cloud NOT RUN.')


if __name__=='__main__':main()
