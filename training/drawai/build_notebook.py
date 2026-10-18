"""Build a portable Kaggle notebook and source/data bundle from this checkout."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path

import nbformat

from .prepare_data import ROOT, write_json


def main():
    notebook=nbformat.v4.new_notebook()
    notebook.metadata.update(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
                             language_info=dict(name='python',version='3.11'),
                             kaggle=dict(accelerator='gpu',isInternetEnabled=True))
    notebook.cells=[
        nbformat.v4.new_markdown_cell('''# DrawAI â€” Swin 1D baseline (P1)

Attach the generated `drawai-source.zip` as a Kaggle input dataset. Enable a GPU accelerator for the full experiment. Source and processed tensors are bundled; the tensor hashes are validated before training. Only DrawAI data and the Swin baseline are included. DINOv3 is planned for P2.

The full mode runs 4 Swin validation-only pilots, freezes the Swin configuration, trains seeds 42/43/44, locks all checkpoints, then evaluates official test once. GPU runs have not been performed merely by generating this notebook.

For local verification set `DRAWAI_NOTEBOOK_MODE=smoke`, `DRAWAI_REQUIRE_CUDA=0`, and `DRAWAI_SOURCE_ROOT` to the checkout. Smoke runs two epochs on full train/validation for each model without opening test. Versions are printed and saved; use a compatible PyTorch environment. The pinned local requirements are in `training/drawai/requirements.txt`; install dependencies in a separate setup cell only if needed. Do not install a CPU-only Torch wheel into a GPU Kaggle runtime.'''),
        nbformat.v4.new_code_cell('''import os, sys, json, platform, zipfile, subprocess, importlib.util
from pathlib import Path
if importlib.util.find_spec('unlzw3') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'unlzw3==0.2.3'], check=True)
MODE = os.environ.get('DRAWAI_NOTEBOOK_MODE', 'full')
if MODE not in ('smoke', 'full'):
    raise ValueError('MODE must be smoke or full')
if os.environ.get('DRAWAI_SOURCE_ROOT'):
    ROOT = Path(os.environ['DRAWAI_SOURCE_ROOT']).resolve()
else:
    inputs = list(Path('/kaggle/input').rglob('drawai-source.zip'))
    if len(inputs) != 1:
        raise RuntimeError('Attach exactly one drawai-source.zip dataset')
    ROOT = Path('/kaggle/working/drawai').resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(inputs[0]) as archive:
        for name in archive.namelist():
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError('Archive member outside workspace')
        archive.extractall(ROOT)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
require_cuda = os.environ.get('DRAWAI_REQUIRE_CUDA', '1') == '1'
if MODE == 'full' and require_cuda and not torch.cuda.is_available():
    raise RuntimeError('Full Kaggle mode requires a GPU; CPU smoke must be explicit')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THREADS = int(os.environ.get('DRAWAI_THREADS', '4'))
OUTPUT = Path(os.environ.get('DRAWAI_NOTEBOOK_OUTPUT', str(ROOT / 'artifacts/drawai' / ('swin-notebook-smoke-v2' if MODE == 'smoke' else 'swin-gpu-v2'))))
OUTPUT.mkdir(parents=True, exist_ok=True)
versions = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, device=DEVICE)
print(json.dumps(versions, indent=2))
from training.drawai.prepare_data import write_json
write_json(OUTPUT / 'notebook-runtime.json', versions)
def command(*args):
    subprocess.run([sys.executable, '-m', *args], cwd=ROOT, check=True)
'''),
        nbformat.v4.new_code_cell('''from training.drawai.experiment import config_for
if MODE == 'smoke':
    for model in ['swin']:
        config = config_for(model, 64, False, 42, False)
        config.update(epochs=2, patience=2, warmup_epochs=0)
        path = OUTPUT / (model + '-config.json')
        write_json(path, config)
        folder = OUTPUT / model
        args = ['training.drawai.train', '--config', str(path), '--output-dir', str(folder), '--device', DEVICE, '--threads', str(THREADS)]
        if (folder / 'last.pth').exists():
            args.append('--resume')
        command(*args)
else:
    command('training.drawai.experiment', '--stage', 'pilots', '--output-dir', str(OUTPUT), '--device', DEVICE, '--threads', str(THREADS))
'''),
        nbformat.v4.new_code_cell('''if MODE == 'full':
    command('training.drawai.experiment', '--stage', 'official', '--output-dir', str(OUTPUT), '--device', DEVICE, '--threads', str(THREADS))
    command('training.drawai.evaluate', '--output-dir', str(OUTPUT), '--threads', str(THREADS))
    command('training.drawai.verify_training', '--output-dir', str(OUTPUT))
else:
    for model in ['swin']:
        summary = json.loads((OUTPUT / model / 'summary.json').read_text())
        assert summary['status'] == 'completed' and summary['epochs_completed'] == 2
        assert summary['test_used_for_training'] is False
    write_json(OUTPUT / 'smoke-report.json', dict(status='passed', mode=MODE, device=DEVICE, test_opened=False, models=['swin']))
    print('PASS: notebook CPU/GPU smoke; official test unopened')
'''),
        nbformat.v4.new_code_cell('''if MODE == 'full':
    import shutil
    archive = shutil.make_archive(str(OUTPUT.parent / (OUTPUT.name + '-results')), 'zip', root_dir=OUTPUT.parent, base_dir=OUTPUT.name)
    print('Download results:', archive)
else:
    print('Smoke evidence:', OUTPUT / 'smoke-report.json')
'''),
        nbformat.v4.new_markdown_cell('''## Interruption and reporting

Run the same cells with the same output directory to resume interrupted runs. Do not change the experiment plan after official test access; use a new experiment ID and disclose any test-driven changes. Download the complete results including configs, logs, checkpoint hashes and evaluation files. Local CPU smoke is technical validation and does not establish Kaggle GPU correctness or recognition quality.''')
    ]
    # Stable cell IDs keep repeated builds byte-identical for provenance checks.
    for index, cell in enumerate(notebook.cells):
        cell['id'] = hashlib.sha256(f'{index}:{cell.source}'.encode('utf-8')).hexdigest()[:16]
        if cell.cell_type=='code': ast.parse(cell.source)
    nbformat.validate(notebook)
    destination=ROOT/'notebooks/kaggle_train_drawai.ipynb'
    nbformat.write(notebook,destination)
    bundle=ROOT/'artifacts/drawai/kaggle/drawai-source.zip'
    bundle.parent.mkdir(parents=True,exist_ok=True)
    paths=list((ROOT/'training/drawai').glob('*.py'))+list((ROOT/'training/drawai/configs').glob('*.json'))
    paths += [ROOT/'models/SwinTransformer_cls.py', ROOT/'training/drawai/requirements.txt', ROOT/'artifacts/drawai/data/verification.json']
    for length in [64,128]: paths += list((ROOT/f'data/processed/pendigits-drawai-v1/l{length}').glob('*'))
    paths += [ROOT/'data/raw/pendigits/pendigits-orig.tes']
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if path.is_file(): archive.write(path,path.relative_to(ROOT).as_posix())
    write_json(bundle.parent/'bundle-manifest.json',dict(notebook=str(destination.relative_to(ROOT)),
               sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),bytes=bundle.stat().st_size,
               files=[str(p.relative_to(ROOT).as_posix()) for p in sorted(paths) if p.is_file()],
               gpu_executed=False))
    print('Built notebook and source/data bundle:',bundle)


if __name__=='__main__': main()
