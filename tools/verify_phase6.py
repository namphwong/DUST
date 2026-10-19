"""Final P6 audit: primary artifacts, numerical gates, runtime and legacy migration."""
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai.native_runtime import configure_windows_dlls
configure_windows_dlls()
import torch
import numpy as np
from PIL import Image,ImageChops
from training.drawai import paint_export as p,paint_train as train
from training.drawai.private_data import load,dump,sha,canonical
from training.drawai.paint_package import verify_bundle,spec

OUT=ROOT/'artifacts/drawai/phase6-image64';BUNDLE=ROOT/'models/exported/paint35-v1'


def main():
    train.seed_all(42);anchor=load(OUT/'trusted-bundle.json');bundle,registry=verify_bundle(BUNDLE,anchor['bundle_sha256'])
    selection,runs=train.checked_selection(train.DEFAULT_OUT);assert selection['selection_sha256']==bundle['selection_sha256']
    p5_final=load(train.DEFAULT_OUT/'evaluation/final-test.json')
    checked={}
    for family in train.FAMILIES:
        info=spec(family);report=load(BUNDLE/family/'report.json');release=load(BUNDLE/family/'release.json')
        assert report==load(OUT/f'export-{family}.json')
        winner=selection['winners'][family];run=next(r for r in runs if r['family']==family and r['seed']==winner['seed'])
        assert report['checkpoint_sha256']==winner['sha256'] and report['seed']==run['seed']
        with np.load(OUT/'evaluation'/f'{family}-full-test-parity.npz',allow_pickle=False) as z:
            assert z['pytorch_logits'].shape==z['artifact_logits'].shape==z['onnx_logits'].shape==(1000,10)
            parity=p.check_parity(z['pytorch_logits'],z['artifact_logits'],z['labels'],family)
            assert parity==report['parity']
            expected={f'paint35_p{person:03}_d{label}_r{rep:02}' for person in range(31,36) for label in range(10) for rep in range(1,21)}
            assert set(z['sample_ids'])==expected and len(set(z['sample_ids']))==1000
            tolerance=p.TOLERANCES[family]
            failed=np.any(np.abs(z['onnx_logits']-z['pytorch_logits'])>tolerance['atol']+tolerance['rtol']*np.abs(z['pytorch_logits']),axis=1)
            assert int(failed.sum())==report['onnx_diagnostic']['failed_logit_rows']
            assert np.count_nonzero(z['onnx_logits'].argmax(1)!=z['pytorch_logits'].argmax(1))==0
        p5_result=next(r for r in p5_final['results'] if r['family']==family and r['seed']==run['seed'])
        assert parity['onnx_metrics']=={k:v for k,v in p5_result['metrics'].items() if k!='loss'}
        if family=='dinov3-vits16':
            assert info['runtime_format']=='torchscript' and report['onnx_diagnostic']['approved_runtime'] is False
            script=torch.jit.load(str(BUNDLE/family/info['model_filename']),map_location='cpu').eval()
            checkpoint=torch.load(train.DEFAULT_OUT/'runs'/f"{family}-seed{run['seed']}"/run['checkpoint'],map_location='cpu',weights_only=True)
            state=script.state_dict()
            assert all(name in state and torch.equal(value,state[name]) for name,value in checkpoint['model_state_dict'].items())
            assert sum(t.numel() for t in script.parameters())==run['parameters']==21603850
        else:
            assert report['onnx_diagnostic']['numeric_parity_passed'] and info['runtime_format']=='onnx'
        assert train.file_sha(BUNDLE/family/info['model_filename'])==report['model_sha256']
        checked[family]=dict(filename=info['model_filename'],format=info['runtime_format'],bytes=report['model_bytes'],sha256=report['model_sha256'],
            seed=run['seed'],test_accuracy=parity['onnx_metrics']['accuracy'],test_macro_f1=parity['onnx_metrics']['macro_f1'],
            max_abs_error=parity['max_abs_error'],argmax_disagreements=0,onnx_diagnostic=report['onnx_diagnostic'])
    baseline=load(OUT/'protected-before.json');allowed={'web/src/drawai/types.ts','web/src/drawai/release-loader.ts','web/src/drawai/feasibility-worker.ts','web/public/models/drawai-swin-v1/release.json'}
    changes={}
    for name,digest in baseline.items():
        current=train.file_sha(ROOT/name)
        if current!=digest:
            assert name in allowed,name
            changes[name]=dict(before_sha256=digest,after_sha256=current,reason='Canonical legacy filename/manifest migration; graph bytes unchanged')
    assert set(changes)==allowed
    legacy=ROOT/'web/public/models/drawai-swin-v1';canonical_name='drawai_swin1d_pendigits_l128_fp32_v1.onnx'
    assert train.file_sha(legacy/canonical_name)==train.file_sha(legacy/'model.onnx')=='28100de69dec82303a954af5c18936788e7c3dee623d70ec7a89ad9f7ef02366'
    assert load(legacy/'release.json')['model_filename']==canonical_name
    dump(OUT/'legacy-migration.json',dict(canonical_name=canonical_name,alias='model.onnx',same_graph_bytes=True,changes=changes,remaining_protected_files_unchanged=len(baseline)-len(changes)))
    design={}
    for name in ('desktop','mobile','root-desktop','root-mobile'):
        current=Image.open(OUT/f'ui/{name}.png').convert('RGB');before=Image.open(ROOT/f'artifacts/drawai/phase0/ui/{name.removeprefix("root-")}.png').convert('RGB')
        assert current.size==before.size and ImageChops.difference(current,before).getbbox() is None
        design[name]=True
    log=(OUT/'python-tests.log').read_text(encoding='utf-8-sig');assert re.search(r'\nOK\s*$',log);count=int(re.search(r'Ran (\d+) tests',log)[1]);assert count>=128
    product=load(OUT/'product-regression.json')['stats'];gate=load(OUT/'gate-regression.json')['stats']
    for stats in (product,gate):assert all(stats[k]==0 for k in ('unexpected','flaky','skipped'))
    assert product['expected']>=20 and gate['expected']==1 and 'built in' in (OUT/'build.log').read_text(encoding='utf-8-sig')
    browser=load(OUT/'browser-swin.json');assert browser['passed'] and len(browser['cases'])==9
    assert browser['model_sha256']==checked['swin1d-image']['sha256'] and all(c['passed'] for c in browser['cases'])
    backend=load(OUT/'backend-verification.json');assert backend['passed'] and backend['existing_P2_cpu_budgets_passed'] and backend['no_timm_or_pretrained_snapshot_dependency_at_runtime']
    # Exported weights must remain local; build may contain only the unchanged legacy model.
    exported_names={entry['filename'] for entry in checked.values()}|{spec(family)['diagnostic_onnx_filename'] for family in train.FAMILIES}
    for root in (ROOT/'web/public',ROOT/'web/dist'):
        assert not any(path.name in exported_names for path in root.rglob('*') if path.is_file())
    for name,digest in bundle['code_sha256'].items():assert train.file_sha(ROOT/name)==digest,name
    result=dict(phase='P6',status='DONE_LOCAL_VERIFIED_MIXED_RUNTIME_EXPORTS',phase_complete=True,
        bundle_path=BUNDLE.relative_to(ROOT).as_posix(),bundle_sha256=anchor['bundle_sha256'],models=checked,
        completion_scope='ONNX Swin + full TorchScript DINO primary artifacts, exact P5 selection, full-test parity and local CPU/browser verification',
        dinov3_onnx_approved=False,all_onnx_parity_passed=False,onnx_tolerance_relaxed=False,
        full_test_samples_per_model=1000,procedural_fixtures_per_model=9,python_tests_passed=count,
        browser_tests_passed=product['expected']+gate['expected'],product_regression=product,default_asset_gate_closed=True,
        design=design,legacy_graph_unchanged=True,product_model_replaced=False,product_release_ready=False,
        no_retraining=True,test_used_for_model_selection=False,
        code_sha256_at_verification={name:train.file_sha(ROOT/name) for name in ['training/drawai/paint_runtime.py','tools/verify_phase6_backend.py','tools/verify_phase6.py']},
        limitations=['DINO ONNX fails strict per-logit tolerance on2/1000 samples despite identical argmax; diagnostic only',
                    'TorchScript CPU selected as DINO deployment artifact; requires torch, no browser claim',
                    '37 near-groups across main splits; no leakage-free/generalization claim',
                    'Human origin/IDs/permissions owner-attested; no independent identity proof',
                    'P7 still must connect canvas to image preprocessing, test UX/domain gap and deployment'])
    dump(OUT/'verification.json',result);print(f"P6 PASS:{count} Python,{result['browser_tests_passed']} browser tests; verified ONNX Swin + TorchScript DINO. DINO ONNX remains diagnostic.")


if __name__=='__main__':main()
