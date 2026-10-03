"""Execute all notebook cells in a real local Jupyter kernel; NOT a Kaggle GPU check."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-runs', action='store_true', help='Resume completed local runs instead of training from scratch')
    args = parser.parse_args()
    notebook_path = ROOT / 'notebooks/kaggle_train_har.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='dust-notebook-check-') as tmp:
        work = Path(tmp) / 'dust-project'
        work.mkdir()
        if args.reuse_runs:
            shutil.copytree(ROOT / 'artifacts/phase3/runs', work / 'artifacts/phase3/runs')
        old_env = {key: os.environ.get(key) for key in ['DUST_WORK_ROOT', 'DUST_DATA_SOURCE']}
        os.environ['DUST_WORK_ROOT'] = str(work)
        os.environ['DUST_DATA_SOURCE'] = str(ROOT / 'data/raw/uci-har.zip')
        manager = KernelManager(kernel_name='python3')
        try:
            manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
            NotebookClient(notebook, km=manager, timeout=3600,
                           resources={'metadata': {'path': str(work)}}).execute()
        finally:
            # Supplied kernel managers are caller-owned: release Windows cwd lock before temp cleanup.
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        report = json.loads((work / 'artifacts/phase3/evaluation/evaluation.json').read_text(encoding='utf-8'))
        expected = ROOT / 'artifacts/phase3/evaluation/evaluation.json'
        if args.reuse_runs and expected.exists():
            original = json.loads(expected.read_text(encoding='utf-8'))
            assert report == original, 'Notebook and CLI evaluation differ'
        archive = work.parent / 'phase3-results.zip'
        assert archive.is_file()
        result = dict(status='passed', execution_environment='local CPU Jupyter kernel, not Kaggle',
                      kaggle_gpu_verified=False, notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                      code_cells_executed=sum(c.cell_type == 'code' for c in notebook.cells),
                      reused_completed_runs=args.reuse_runs, seconds=time.perf_counter()-started,
                      evaluation_matches_cli=True if args.reuse_runs and expected.exists() else None,
                      artifact_zip_created=True, zip_bytes=archive.stat().st_size)
        output = ROOT / 'artifacts/phase3/notebook-verification.json'
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
