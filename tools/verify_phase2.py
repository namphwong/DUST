"""P2 auditor: require real pretrained weights, parity and a measured eligible runtime."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/drawai/phase2'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path,count):
    value=read(path)
    assert value['stats']['expected']==count and not value.get('errors')
    assert all(value['stats'][key]==0 for key in ('unexpected','flaky','skipped'))
    return value['stats']


def main():
    from PIL import Image,ImageChops
    import numpy as np
    from training.drawai.dinov3_model import TIMM_MODEL_ID,TIMM_REVISION,TIMM_WEIGHTS_SHA256,TIMM_CONFIG_BLOB
    from training.drawai.raster import contract
    technical=OUT/'technical';manifest=read(technical/'manifest.json');python=read(technical/'python-report.json')
    for name,metadata in manifest['files'].items():
        assert (technical/name).stat().st_size==metadata['bytes'] and digest(technical/name)==metadata['sha256'],name
    assert manifest['smoke_only'] and not manifest['release_ready'] and not manifest['trained']
    assert python['full_backbone_exported'] and python['parameters']==21603850 and python['trainable_parameters']==7690
    assert python['argmax_disagreements']==0 and python['fixture_count']==19
    assert read(technical/'preprocessing.json')==contract()
    browser=read(OUT/'browser-report.json');budget=read(ROOT/'docs/drawai/phase2-runtime-budget.json')['browser']
    assert browser['parityPassed'] and browser['modelSha256']==python['model_sha256']
    assert browser['pretrainedLoaded']==python['pretrained_loaded']
    assert browser['memoryBudgetPassed']
    assert browser['modelBytes']<=budget['model_max_bytes'] and browser['coldMs']<=budget['cold_local_load_verify_init_max_ms']
    assert browser['passed']==(browser['p95Ms']<=budget['warm_inference_p95_max_ms'])
    assert browser['memory']['incrementalBytes']<=budget['incremental_browser_process_tree_rss_max_bytes']
    fixtures=read(technical/'fixtures.json')['cases']
    assert len(browser['cases'])==len(fixtures)==19 and len(browser['invalid'])==8
    for expected,actual in zip(fixtures,browser['cases']):
        assert expected['name']==actual['name'] and expected['rgb_sha256']==actual['rgbHash']
        assert expected['tensor_sha256']==actual['tensorHash'] and actual['predicted']==actual['expectedArgmax']
    tiny=read(technical/'tiny-overfit.json')
    assert tiny['status']=='passed' and tiny['test_opened'] is False and tiny['exported_head_modified'] is False
    assert tiny['history'][-1]['train_accuracy']==1 and tiny['backbone_gradients_absent']
    backend=read(OUT/'backend-report.json');backend_budget=read(ROOT/'docs/drawai/phase2-backend-budget.json')
    assert backend['passed'] and backend['parityPassed'] and backend['pretrainedLoaded']
    assert not backend['trained'] and not backend['releaseReady'] and not backend['dataPersisted']
    assert backend['modelSha256']==python['model_sha256'] and len(backend['cases'])==19
    assert backend['p95EndToEndMs']<=backend_budget['end_to_end_p95_max_ms']
    assert backend['coldStartMs']<=backend_budget['cold_start_max_ms']
    assert backend['peakSampledRssBytes']<=backend_budget['process_rss_max_bytes']
    assert backend['iterations']==30 and backend['warmup']==5
    for expected,actual in zip(fixtures,backend['cases']):
        assert expected['name']==actual['name'] and expected['rgb_sha256']==actual['rgbHash']
        assert expected['tensor_sha256']==actual['tensorHash']
        assert actual['predicted']==int(np.argmax(expected['expected_logits']))
        assert actual['maxAbsError']<=1e-4+1e-3*max(abs(v) for v in expected['expected_logits'])
    tests=dict(dinov3=suite(OUT/'browser-tests.json',6),backend=suite(OUT/'backend-tests.json',3),product=suite(OUT/'product-regression.json',12))
    log=(OUT/'python-tests.log').read_text(encoding='utf-8-sig')
    assert re.search(r'Ran 46 tests',log) and re.search(r'\nOK\s*$',log)
    # Preserve the serving Swin, all existing DrawAI components, and PenDigits.
    p1=read(ROOT/'artifacts/drawai/phase1/verification.json')
    for name,sha in p1['unchanged_product_sources'].items():assert digest(ROOT/name)==sha,name
    protected=read(OUT/'pre-phase2-baseline.json')['protected_files']
    for row in protected:
        if row['path'].startswith(('data/','web/public/')):assert digest(ROOT/row['path'])==row['sha256'],row['path']
    designs={}
    for view in ('desktop','mobile'):
        original=Image.open(ROOT/f'artifacts/drawai/phase0/ui/{view}.png').convert('RGB')
        for prefix in ('','root-'):
            current=Image.open(OUT/f'ui/{prefix}{view}.png').convert('RGB')
            assert original.size==current.size and ImageChops.difference(original,current).getbbox() is None
            designs[prefix+view]=dict(pixel_identical_to_P0=True,size=list(current.size))
    assert not list((ROOT/'web/public').rglob('*dinov3*.onnx'))
    assert not list((ROOT/'web/dist').rglob('*.onnx')) or all('dinov3' not in p.name for p in (ROOT/'web/dist').rglob('*.onnx'))
    baseline_py=ROOT/'.venv-phase1/Scripts/python.exe'
    baseline_info=json.loads(subprocess.check_output([str(baseline_py),'-c',
        'import importlib.util,json,importlib.metadata;print(json.dumps({"transformers_present":importlib.util.find_spec("transformers") is not None,"packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}}))'],text=True))
    assert baseline_info['transformers_present'] is False
    old_packages=read(ROOT/'artifacts/drawai/phase0/python-packages.json')['packages']
    assert all(baseline_info['packages'].get(p['name'])==p['version'] for p in old_packages)
    pretrained=python['pretrained_loaded'] is True
    if pretrained:
        provenance=python['provenance']
        assert provenance['model_id']==TIMM_MODEL_ID and provenance['revision']==TIMM_REVISION
        assert provenance['weights_sha256']==TIMM_WEIGHTS_SHA256 and provenance['official_config_verified']
        assert provenance['config_git_blob']==TIMM_CONFIG_BLOB and provenance['native_state_all_tensors_equal']
        source=Path(provenance['source_directory'])
        assert digest(source/'model.safetensors')==TIMM_WEIGHTS_SHA256
        config=(source/'config.json').read_bytes()
        assert hashlib.sha1(b'blob '+str(len(config)).encode()+b'\0'+config).hexdigest()==TIMM_CONFIG_BLOB
        assert digest(source/'LICENSE.md')=='25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e'
    assert pretrained
    result=dict(phase='P2',status='DONE_LOCAL',
        phase_complete=pretrained,all_executable_checks_passed=True,pretrained_loaded=pretrained,
        public_weights_metadata=dict(model_id=TIMM_MODEL_ID,revision=TIMM_REVISION,sha256=TIMM_WEIGHTS_SHA256,config_git_blob=TIMM_CONFIG_BLOB),
        pending=[],
        runtime_decision='local-onnx-cpu-backend',wasm_eligible=browser['passed'],
        backend_fallback_required=not browser['passed'],production_model='unchanged Swin PenDigits release',
        digit_head_trained=False,release_ready=False,
        design=designs,python_tests_passed=46,browser_tests=tests,
        parity=dict(fixtures=19,invalid_inputs=8,rgb_and_tensor_hashes_identical=True,python_max_abs_error=python['max_abs_error'],browser_max_abs_error=max(c['maxAbsError'] for c in browser['cases'])),
        benchmark=dict(model_bytes=python['model_bytes'],python_p95_ms=python['python_p95_ms'],python_threads=4,
            wasm_p95_ms=browser['p95Ms'],wasm_threads=1,cold_local_ms=browser['coldMs'],browser_incremental_sampled_rss_bytes=browser['memory']['incrementalBytes'],
            backend_p95_end_to_end_ms=backend['p95EndToEndMs'],backend_cold_ms=backend['coldStartMs'],backend_peak_sampled_rss_bytes=backend['peakSampledRssBytes']),
        baseline_environment_unchanged=True,private_or_test_data_used=False,
        tiny_overfit_is_quality_metric=False,smoke_artifact_not_in_public_or_dist=True,
        limitations=['No private data or digit quality evaluation in P2','Memory is sampled aggregate RSS, not exact peak or unique physical memory','No phone, clean install, GPU or trained-classifier performance claim without explicit evidence'])
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result['status']+': 46 Python and 21 browser tests pass; real pretrained gate passed; 19 raster/logit fixtures; CPU backend verified; serving model/design preserved.')


if __name__=='__main__':main()
