"""Inventory and verify a non-destructive, external recovery snapshot (stdlib only)."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/drawai/phase0'
EXCLUDED = {'.git', 'node_modules', '.venv-phase1', '.venv-drawai', '.venv',
            '__pycache__', '.pytest_cache', 'dist', 'test-results', '.vite', '.ipynb_checkpoints'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8', errors='replace').strip()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def decision(p):
    if p in {'.gitignore', '.gitattributes', 'README.md', 'notebooks/README.md',
             'web/src/main.tsx', 'web/package.json', 'web/package-lock.json',
             'web/playwright.config.ts'}:
        return 'UPDATE', 'Shared entrypoint/config/docs: remove HAR references while retaining DrawAI'
    if p in {'web/src/style.css', 'web/src/motion-style.css'}:
        return 'REVIEW_SHARED_CSS', 'Currently imported globally; inspect cascade and compare baseline screenshots before removal'
    if p == 'models/SwinTransformer_cls.py':
        return 'KEEP', 'Direct dependency of training/drawai/models.py'
    if p.startswith(('training/drawai/', 'web/src/drawai/')) and any(x in p for x in ['models.py', 'experiment.py', 'verify_training.py', 'evaluate.py', 'export_onnx.py', 'build_notebook.py']):
        return 'KEEP_REFACTOR', 'DrawAI pipeline; historical CNN/selection/export coupling must be migrated before deleting CNN support'
    if (p.startswith(('docs/drawai/', 'artifacts/drawai/', 'examples/drawai/',
                      'training/drawai/', 'web/src/drawai/', 'web/public/samples/drawai/',
                      'web/public/verification/', 'tools/')) or
        p.startswith(('web/public/models/drawai-', 'data/raw/pendigits/',
                      'data/processed/pendigits-drawai-v1/', 'training/tests/test_drawai',
                      'web/tests/drawai', 'web/playwright.drawai')) or
        p in {'docs/drawai-project-plan.md', 'notebooks/kaggle_train_drawai.ipynb'}):
        return 'KEEP', 'DrawAI source/data/model/evidence; retain provenance, review generated copies separately'
    if p in {'LICENSE', 'web/index.html', 'web/tsconfig.json', 'web/src/vite-env.d.ts'}:
        return 'KEEP', 'License or shared web infrastructure'
    if p.startswith('training/requirements-'):
        return 'REVIEW_DEPENDENCY', 'Compare installed packages and DrawAI requirements before removing shared setup'
    if (p.startswith(('artifacts/phase', 'artifacts/ui-redesign/', 'data/processed/uci-har',
                      'data/raw/uci-har', 'web/public/models/phase1/',
                      'web/public/models/har-', 'web/public/samples/uci-har',
                      'examples/phase0/', 'saved_models/', 'video_features/', 'logs/')) or
        p.startswith(('step_fusion', 'notebooks/kaggle_train_har')) or
        p in {'gen_data.py', 'models/SwinTransformer_fusion.py', 'models/video_swin_transformer.py',
              'Shi_DuST_Dual_Swin_Transformer_for_Multi-modal_Video_and_Time-Series_Modeling_CVPRW_2024_paper.pdf',
              'examples/test-input-g.csv'}):
        return 'REMOVE_PHASE1', 'HAR/MotionSense or original video/fusion project; preserved in external backup'
    if p.startswith(('docs/', 'training/', 'web/src/', 'web/tests/')):
        return 'REMOVE_PHASE1', 'Legacy HAR path outside DrawAI; confirm reference inventory before removal'
    return 'REVIEW', 'Unclassified infrastructure: inspect before any Phase 1 action'


def main():
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d-%H%M%S')
    backup = ROOT.parent / 'DUST-backups' / ('phase0-' + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True, exist_ok=True)
    tracked = set(git('ls-files', '-z').split('\0'))
    ignored = set(git('ls-files', '--others', '--ignored', '--exclude-standard', '-z').split('\0'))
    status = git('status', '--porcelain=v1', '--untracked-files=all')
    (backup / 'git-status-before.txt').write_text(status, encoding='utf-8')
    (backup / 'tracked-files.txt').write_text('\n'.join(sorted(tracked)), encoding='utf-8')
    rows, references = [], []
    excluded = []
    for base, dirs, files in os.walk(ROOT):
        for d in list(dirs):
            full = Path(base) / d
            if d in EXCLUDED or full == OUT:
                dirs.remove(d)
                excluded.append(full.relative_to(ROOT).as_posix())
        for name in sorted(files):
            path = Path(base) / name
            p = path.relative_to(ROOT).as_posix()
            action, reason = decision(p)
            rows.append(dict(path=p, bytes=path.stat().st_size, sha256=sha(path),
                             git_state='tracked' if p in tracked else 'ignored' if p in ignored else 'untracked',
                             phase1_action=action, reason=reason))
            if path.suffix in {'.py', '.ts', '.tsx', '.css', '.json', '.ipynb'}:
                source = path.read_text(encoding='utf-8-sig', errors='replace')
                if path.suffix == '.py':
                    try:
                        for node in ast.walk(ast.parse(source)):
                            if isinstance(node, ast.ImportFrom):
                                references.append(dict(source=p, line=node.lineno, kind='python-import',
                                                       reference='.' * node.level + (node.module or '')))
                            elif isinstance(node, ast.Import):
                                for a in node.names:
                                    references.append(dict(source=p, line=node.lineno, kind='python-import', reference=a.name))
                    except SyntaxError:
                        pass
                for i, line in enumerate(source.splitlines(), 1):
                    if (path.suffix in {'.ts', '.tsx', '.css'} and
                        re.search(r'\bimport\b|new URL\(|fetch\(|/models/|/samples/', line)):
                        references.append(dict(source=p, line=i, kind='web-reference', reference=line.strip()))
                    elif path.suffix in {'.json', '.ipynb'} and re.search(r'uci.har|drawai|models\.|training\.', line, re.I):
                        references.append(dict(source=p, line=i, kind='config-or-notebook', reference=line.strip()[:1200]))
    rows.sort(key=lambda r: r['path'])
    with (OUT / 'file-inventory.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUT / 'dependency-references.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'line', 'kind', 'reference']); writer.writeheader(); writer.writerows(references)
    archive = backup / 'working-tree.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for row in rows:
            z.write(ROOT / row['path'], row['path'])
    bundle = backup / 'history.bundle'
    subprocess.run(['git', 'bundle', 'create', str(bundle), '--all'], cwd=ROOT, check=True, capture_output=True)
    checked = subprocess.run(['git', 'bundle', 'verify', str(bundle)], cwd=ROOT, check=True, capture_output=True, text=True)
    (backup / 'bundle-verification.txt').write_text(checked.stdout + checked.stderr, encoding='utf-8')
    # Restore to a separate folder, then compare every payload file by SHA-256.
    restore = backup / 'restore-check'
    subprocess.run(['git', 'clone', '--no-checkout', str(bundle), str(restore)], check=True, capture_output=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(restore)
    # Populate the clone's index without changing restored working files.
    subprocess.run(['git', 'reset', '--mixed', 'HEAD'], cwd=restore, check=True, capture_output=True)
    for row in rows:
        assert sha(restore / row['path']) == row['sha256'], row['path']
        assert sha(ROOT / row['path']) == row['sha256'], 'Source changed during snapshot: ' + row['path']
    from collections import Counter
    summary = dict(created_at=datetime.now(timezone(timedelta(hours=7))).isoformat(),
                   root=str(ROOT), backup_directory=str(backup), git_branch=git('branch', '--show-current'),
                   git_head=git('rev-parse', 'HEAD'), source_status_file=str(backup / 'git-status-before.txt'),
                   files=len(rows), source_bytes=sum(r['bytes'] for r in rows),
                   git_states=dict(Counter(r['git_state'] for r in rows)),
                   phase1_actions=dict(Counter(r['phase1_action'] for r in rows)),
                   excluded_directories=sorted(excluded),
                   archive=dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)),
                   history_bundle=dict(path=str(bundle), bytes=bundle.stat().st_size, sha256=sha(bundle), verified=True),
                   restore_check=dict(path=str(restore), files_verified=len(rows), all_hashes_match=True),
                   dependencies='static reference inventory; dynamic paths and CSS need manual review',
                   exclusions_note='Runtimes/dependencies/caches/generated dist/test-results excluded; no model/data payload excluded. Phase0 output excluded to avoid recursion. Git history preserved separately; working tree index staging is not recreated automatically.')
    (OUT / 'snapshot.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (backup / 'snapshot.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (backup / 'file-inventory.csv').write_bytes((OUT / 'file-inventory.csv').read_bytes())
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
