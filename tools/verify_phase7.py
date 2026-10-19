"""Final P7 audit of existing evidence, model anchors and preserved design; no private test read."""
import csv
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai.private_data import load,dump,sha,canonical
from training.drawai.paint_package import verify_bundle
OUT=ROOT/'artifacts/drawai/phase7-image64'
BUNDLE_SHA256='86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2'


def main():
    bundle,registry=verify_bundle(ROOT/'models/exported/paint35-v1',BUNDLE_SHA256)
    selection=load(ROOT/'artifacts/drawai/phase5-image64/selection.json')
    assert selection['selection_sha256']==sha(canonical({k:v for k,v in selection.items() if k!='selection_sha256'}))==bundle['selection_sha256']
    assert selection['candidate_default']==dict(family='dinov3-vits16',seed=42) and selection['test_accessed'] is False
    p6=load(ROOT/'artifacts/drawai/phase6-image64/verification.json')
    for path,expected in p6['code_sha256_at_verification'].items():assert sha((ROOT/path).read_bytes())==expected
    source_design={}
    with (ROOT/'artifacts/drawai/phase0/file-inventory.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['path'] in ('web/src/drawai/drawai.css','web/src/drawai/drawing-canvas.tsx'):
                assert sha((ROOT/row['path']).read_bytes())==row['sha256'];source_design[row['path']]=row['sha256']
    assert len(source_design)==2
    design=load(OUT/'design-parity.json')
    for row in design['design']:
        name='desktop' if row['viewport']['width']==1366 else 'mobile'
        assert row['main_sha256']==sha((OUT/f'ui/empty-{name}.png').read_bytes())==row['legacy_sha256']==sha((OUT/f'ui/legacy-empty-{name}.png').read_bytes())
    assert len(design['design'])==2
    reports=[load(OUT/name) for name in ('browser-tests.json','legacy-tests.json')]
    assert [r['stats']['expected'] for r in reports]==[13,9]
    assert all(r['stats']['unexpected']==r['stats']['flaky']==r['stats']['skipped']==0 for r in reports)
    python_log=(OUT/'python-tests.log').read_text(encoding='utf-8-sig',errors='replace')
    assert re.search(r'Ran 133 tests',python_log) and re.search(r'\nOK\s*$',python_log)
    parity=load(OUT/'canvas-parity.json');assert len(parity['records'])==10
    assert all(r['type']=='result' and r['pixels_sha256']==r['expected_hash'] for r in parity['records'])
    api=load(OUT/'backend-verification.json');assert api['passed'] and len(api['procedural_parity'])==16 and len(api['error_probes'])==8
    assert api['private_data_accessed'] is False and api['test_accessed'] is False and api['origin_rejected'] and api['recovery_passed']
    assert api['health']['default_model']=='dinov3-vits16' and api['health']['canvas_adapter']=='drawai-canvas400-gray64-v1'
    benchmark=load(OUT/'browser-benchmark.json');assert len(benchmark['benchmarks'])==2
    assert all(r['p95_ms']<=500 and r['warmup']==5 and r['iterations']==30 for r in benchmark['benchmarks'])
    assert api['cold_process_start_ms']<=15000 and benchmark['health']['rss_bytes']<=1073741824
    demos=load(OUT/'demo-observations.json');assert len(demos['observations'])==10 and demos['not_accuracy_benchmark']
    correct=sum(r['reference']==r['predicted_id'] for r in demos['observations']);assert demos['correct']==correct==5
    assert all(r['model_version']=='dinov3-vits16' for r in demos['observations'])
    app=(ROOT/'web/src/drawai/drawai-app.tsx').read_text(encoding='utf-8-sig')
    assert 'legacy=false' in app and 'legacy?new InferenceClient():new PaintInferenceClient()' in app
    assert '86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2' in (ROOT/'web/src/drawai/paint-registry.ts').read_text()
    assert sha((ROOT/'web/public/models/drawai-swin-v1/drawai_swin1d_pendigits_l128_fp32_v1.onnx').read_bytes())=='28100de69dec82303a954af5c18936788e7c3dee623d70ec7a89ad9f7ef02366'
    evidence_files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.png') and p.name not in ('verification.json','handoff.json')]
    verification=dict(phase='P7',status='DONE_LOCAL_TECHNICAL_ACCEPTANCE_PENDING',technical_integration_complete=True,
        full_product_acceptance_complete=False,product_release_ready=False,default_model='dinov3-vits16',selected_seed=42,
        selection_sha256=selection['selection_sha256'],bundle_sha256=BUNDLE_SHA256,canvas_adapter='drawai-canvas400-gray64-v1',
        runtime_formats={k:v['spec']['runtime_format'] for k,v in registry['models'].items()},python_tests_passed=133,browser_tests_passed=22,
        api_procedural_parity=16,raster_cross_language_parity=10,design_sources_equal_P0=source_design,empty_design_pixel_identical=True,
        cold_process_start_ms=api['cold_process_start_ms'],browser_p95_ms={r['model_id']:r['p95_ms'] for r in benchmark['benchmarks']},
        rss_bytes=benchmark['health']['rss_bytes'],original_demo_observation=dict(correct=correct,total=10,source='PenDigits train',accuracy_benchmark=False),
        private_test_reopened=False,retraining=False,adapter_tuned_on_test_or_demo_results=False,physical_touch_tested=False,new_user_quality_measured=False,
        dinov3_onnx_approved=False,preexisting_P6_runtime_unchanged=True,public_deployment=False,
        limitations=['Paint-to-canvas domain gap: DINO correct5/10 original PenDigits demos; no new-user accuracy claim',
            '37 retained similarity groups across main splits; no leakage-free/generalization claim',
            'Human provenance/IDs/consent owner-attested, no independent identity proof',
            'Physical touch/new-user acceptance and P8 clean checkout/large-weight distribution still pending'],
        evidence_sha256={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sorted(evidence_files)})
    code_paths=['training/drawai/paint_app_runtime.py','training/drawai/paint_canvas.py','training/tests/test_drawai_paint_app.py',
        'tools/prepare_phase7_fixtures.py','tools/verify_phase7_backend.py','tools/verify_phase7.py','tools/run_drawai.ps1',
        'web/src/drawai/paint-registry.ts','web/src/drawai/paint-raster.ts','web/src/drawai/paint-worker.ts','web/src/drawai/paint-client.ts',
        'web/src/drawai/paint-compare.tsx','web/src/drawai/drawai-app.tsx','web/src/main.tsx','web/vite.config.ts',
        'web/playwright.drawai-p7.config.ts','web/tests/drawai-p7.spec.ts']
    verification['code_sha256_at_verification']={path:sha((ROOT/path).read_bytes()) for path in code_paths}
    dump(OUT/'verification.json',verification)
    dump(OUT/'handoff.json',dict(phase='P7',next_phase='P8',status=verification['status'],technical_integration_complete=True,
        product_release_ready=False,report='docs/drawai/phase7-report.md',guide='docs/drawai/paint-app-guide.md',
        next_steps=['Distribute full pinned bundle with license/checksums, validate clean checkout',
                    'Finalize Kaggle dataset card/source attestation/license and submission links',
                    'Collect independent canvas validation/acceptance data and test physical touch; report actual quality before claiming product readiness'],
        do_not_change=['P4 lock/splits','P5 validation selection','P6 bundle/strict tolerances','Original demo source/labels/results','DrawAI visual design']))
    print('P7 AUDIT PASS: 133 Python +22 browser tests; anchored mixed-runtime integration; design preserved; product acceptance pending.')


if __name__=='__main__':main()
