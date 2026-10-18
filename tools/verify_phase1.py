"""Verify P1 cleanup, unchanged DrawAI design/model and executed evidence."""
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'artifacts/drawai/phase1'
_dll = None
if sys.platform == 'win32':
    dll = Path(os.environ.get('DUST_DLL_DIRECTORY',
        'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'))
    if dll.is_dir():
        _dll = os.add_dll_directory(str(dll))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path, count):
    report = read(path)
    stats = report['stats']
    assert stats['expected'] == count, (path, stats)
    assert all(stats[key] == 0 for key in ('unexpected','skipped','flaky'))
    assert not report.get('errors')
    return stats


def main():
    import numpy as np
    from PIL import Image, ImageChops
    before = read(OUT/'pre-phase1-backup.json')
    archive = Path(before['backup_directory'])/'removed-files'
    with (OUT/'deletion-plan.csv').open(encoding='utf-8-sig') as f:
        removed = list(csv.DictReader(f))
    for row in removed:
        assert digest(archive/row['path']) == row['pre_phase1_sha256'], row['path']
    forbidden = ['motionsense','motion-app','har-swin-kaggle-v1','uci-har-3ch-v1','csv-pipeline']
    for path in list((ROOT/'web/src').rglob('*'))+list((ROOT/'web/dist').rglob('*')):
        if path.suffix in ('.ts','.tsx','.js','.css','.html'):
            text = path.read_text(encoding='utf-8').lower()
            assert not any(token in text for token in forbidden), path
    assert not list((ROOT/'training').glob('*.py'))
    assert {p.name for p in (ROOT/'models').glob('*.py')} == {'SwinTransformer_cls.py'}
    assert all(p.name.startswith('drawai') for p in (ROOT/'web/tests').glob('*.ts'))
    assert all('drawai' in p.name for p in (ROOT/'web/public/models').iterdir())
    assert not list((ROOT/'training/drawai/configs').glob('*cnn*'))
    assert not list((ROOT/'artifacts/drawai/d3-local-v1/runs').glob('*cnn*'))
    baseline = read(ROOT/'artifacts/drawai/phase0/baseline.json')
    unchanged_sources = {}
    for path, expected in baseline['source_fingerprints'].items():
        if path.startswith('web/src/drawai/') or path == 'models/SwinTransformer_cls.py':
            assert digest(ROOT/path) == expected, path
            unchanged_sources[path] = expected
    with (ROOT/'artifacts/drawai/phase0/file-inventory.csv').open(encoding='utf-8-sig') as f:
        old_css = next(r for r in csv.DictReader(f) if r['path'] == 'web/src/style.css')
    assert digest(ROOT/'web/src/drawai/base.css') == old_css['sha256']
    design = {}
    for view in ('desktop','mobile'):
        original = Image.open(ROOT/f'artifacts/drawai/phase0/ui/{view}.png').convert('RGB')
        for prefix in ('','root-'):
            path = OUT/f'ui/{prefix}{view}.png'
            current = Image.open(path).convert('RGB')
            assert original.size == current.size and ImageChops.difference(original,current).getbbox() is None, path
            design[prefix+view] = dict(pixel_identical=True,size=list(current.size))
    model_dir = ROOT/'web/public/models/drawai-swin-v1'
    for name, metadata in read(model_dir/'release.json')['files'].items():
        assert digest(model_dir/name) == metadata['sha256']
        assert (model_dir/name).stat().st_size == metadata['bytes']
    assert digest(model_dir/'model.onnx') == baseline['model_sha256']
    checkpoint = ROOT/'artifacts/drawai/d3-local-v1/runs/official-swin-l128-aug1-seed44/best.pth'
    assert digest(checkpoint) == baseline['checkpoint_sha256']
    protected = [r for r in before['files'] if r['path'].startswith(('data/raw/pendigits/','data/processed/pendigits-drawai-v1/'))]
    assert all(digest(ROOT/r['path']) == r['sha256'] for r in protected)
    candidate = read(OUT/'export-audit/python-report.json')
    assert candidate['status'] == 'passed' and candidate['test_samples'] == 3498
    assert candidate['argmax_disagreements'] == 0 and candidate['macro_f1_degradation'] == 0
    assert digest(OUT/'export-candidate/model.onnx') == baseline['model_sha256']
    fixtures = read(model_dir/'fixtures.json')['cases']
    parity = read(OUT/'ui/demo-parity.json')
    for label, observation in enumerate(parity['observations']):
        fixture = next(f for f in fixtures if f['name'] == f'train-digit-{label}')
        np.testing.assert_allclose(observation['logits'],fixture['expected'],atol=1e-5,rtol=1e-4)
        assert observation['predicted_id'] == int(np.argmax(fixture['expected']))
    tests = dict(app=suite(OUT/'ui/playwright-report.json',9),
                 routes=suite(OUT/'route-tests.json',3),
                 release=suite(ROOT/'artifacts/drawai/export/playwright-report.json',13),
                 chrome_edge=suite(ROOT/'artifacts/drawai/verification/playwright-report.json',34))
    python_log = (OUT/'python-tests.log').read_text(encoding='utf-8-sig')
    assert re.search(r'Ran 34 tests',python_log) and re.search(r'\nOK\s*$',python_log)
    notebook = read(ROOT/'artifacts/drawai/notebook-smoke/verification.json')
    assert notebook['status'] == 'passed' and notebook['test_opened'] is False
    assert notebook['source_sha256'] == digest(ROOT/'notebooks/kaggle_train_drawai.ipynb')
    assert notebook['full_branch_executed'] is False and notebook['kaggle_gpu_executed'] is False
    bundle_manifest = read(ROOT/'artifacts/drawai/kaggle/bundle-manifest.json')
    bundle = ROOT/'artifacts/drawai/kaggle/drawai-source.zip'
    assert digest(bundle) == bundle_manifest['sha256']
    with zipfile.ZipFile(bundle) as z:
        for name in z.namelist():
            assert digest(ROOT/name) == hashlib.sha256(z.read(name)).hexdigest(), name
            assert 'cnn' not in Path(name).name.lower()
    # Imports detect dangling Python references without running module CLIs.
    modules = []
    for path in sorted((ROOT/'training/drawai').glob('*.py')):
        name = 'training.drawai.'+path.stem
        importlib.import_module(name)
        modules.append(name)
    result = dict(phase='P1',status='DONE_LOCAL',archived_files_verified=len(removed),
                  backup_directory=before['backup_directory'],model_sha256=baseline['model_sha256'],
                  checkpoint_sha256=baseline['checkpoint_sha256'],private_data_or_dinov3_implemented=False,
                  protected_pendigits_files_verified=len(protected), unchanged_product_sources=unchanged_sources,
                  design=design,browser_tests=tests,python_tests_passed=34,modules_imported=modules,
                  full_test_export_samples=3498,argmax_disagreements=0,macro_f1_degradation=0,
                  notebook_cpu_smoke=notebook,main_routes=['/','/drawai','/drawai/'],
                  runtime_legacy_scan='passed',exported_onnx_identical_to_P0=True,
                  limitations=['No clean install, private data, DINOv3, GPU full run or real touch-device evaluation in P1',
                               'Legacy files were moved to an external recovery archive after permanent delete commands were rejected'])
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS P1: {len(removed)} archived files verified; 34 Python + 59 browser tests; '
          'pixel-identical design, unchanged model/backbone/data, 3498 export predictions match; notebook CPU smoke passed.')


if __name__ == '__main__':
    main()
