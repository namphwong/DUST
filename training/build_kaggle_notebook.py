"""Generate a self-contained notebook from tracked project sources (no GitHub dependency)."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, index):
    value = dict(cell_type=kind, id=f'phase3-{index:02d}', metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def build(gpu=False):
    files = ['models/SwinTransformer_cls.py', 'training/har_models.py', 'training/har_metrics.py',
             'training/train_har.py', 'training/evaluate_har.py', 'training/prepare_uci_har.py',
             'training/verify_phase2.py', 'training/configs/data.json', 'training/configs/train-cnn.json',
             'training/configs/train-swin.json']
    sources = {name: (ROOT / name).read_text(encoding='utf-8') for name in files}
    cells = []
    def add(kind, source):
        cells.append(cell(kind, source, len(cells)))
    add('markdown', '''# DuST — UCI HAR, CNN baseline + Swin Transformer 1D

Notebook độc lập, không cần clone repo/private token. Bật GPU nếu tài khoản còn quota;
CPU cũng chạy được. Input chỉ `total_acc_x/y/z`, 50 Hz, 128 điểm, 6 lớp.
Train/validation/test không trùng subject; normalization fit train-only.
Checkpoint chọn bằng **validation macro-F1**. Test chỉ mở ở bước cuối sau khi đóng băng lựa chọn.

Nguồn: [UCI HAR chính thức](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
Bật Internet để tải ZIP, hoặc Add Input ZIP/folder gốc rồi sửa `DATA_SOURCE`.
Không tải checkpoint không rõ nguồn: `torch.load` dùng pickle cho artifact tự tạo.
Mặc định một seed để tiết kiệm quota; muốn ba seed phải chọn trước khi chạy evaluation.
Notebook chưa có kết quả GPU Kaggle nhúng sẵn; tất cả output sinh từ lần chạy của bạn.
''')
    add('code', '''import os, sys, json, subprocess, zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# Chỉ sửa cấu hình trước khi chạy. Không thay candidates sau khi xem test.
WORK_ROOT = Path(os.environ.get('DUST_WORK_ROOT', '/kaggle/working/dust-project'))
DATA_SOURCE = os.environ.get('DUST_DATA_SOURCE') or None  # ZIP hoặc folder UCI HAR gốc
SEEDS = [42]  # Có quota: [42, 43, 44], chốt trước evaluation.
DEVICE = 'auto'  # Kaggle GPU nếu khả dụng, nếu không CPU.
if not SEEDS or len(set(SEEDS)) != len(SEEDS):
    raise ValueError('SEEDS must be non-empty and unique')
WORK_ROOT.mkdir(parents=True, exist_ok=True)
def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=WORK_ROOT, check=True)
''')
    add('markdown', '## 1. Khôi phục mã nguồn/config đã đóng gói\nKhông ghi đè một file có nội dung khác: dùng thư mục mới nếu đã chỉnh source ở run trước.\n')
    add('code', 'SOURCES = ' + repr(sources) + '''
import hashlib
source_hashes = {}
for relative, source in SOURCES.items():
    path = WORK_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8') != source:
        raise ValueError(f'Source differs: {relative}. Use a new WORK_ROOT.')
    path.write_text(source, encoding='utf-8', newline='\\n')
    source_hashes[relative] = hashlib.sha256(source.encode('utf-8')).hexdigest()
(WORK_ROOT / 'source-hashes.json').write_text(json.dumps(source_hashes, indent=2), encoding='utf-8')
print('Embedded project files:', len(SOURCES))
''')
    add('markdown', '''## 2. Môi trường
Kaggle thường có sẵn PyTorch. Không cài lại CUDA/PyTorch để tránh tải nặng.
Ghi version thực tế; bản CPU tham chiếu của repo: Python 3.11, torch 2.5.1, numpy 1.26.4.
Môi trường khác có thể cho kết quả khác dù seed giống nhau. Matplotlib chỉ dùng vẽ báo cáo.
''')
    add('code', '''import importlib.util
if importlib.util.find_spec('matplotlib') is None:
    run('-m', 'pip', 'install', 'matplotlib==3.9.4')
import torch, numpy as np
assert int(torch.__version__.split('.')[0]) >= 2, 'Need PyTorch 2+'
print({'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
       'cuda': torch.cuda.is_available(),
       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
''')
    add('markdown', '## 3. Chuẩn bị và kiểm chứng dữ liệu\nChạy lại Phase 2; kiểm tra source pairing, normalization và tính tái lập.\n')
    add('code', '''data_dir = WORK_ROOT / 'data/processed/uci-har-3ch-v1'
source_args = ['--download'] if DATA_SOURCE is None else [
    '--raw-dir' if Path(DATA_SOURCE).is_dir() else '--archive', DATA_SOURCE]
run('training/prepare_uci_har.py', *source_args)
source_path = Path(DATA_SOURCE) if DATA_SOURCE else WORK_ROOT / 'data/raw/uci-har.zip'
run('training/verify_phase2.py', '--source', source_path)
''')
    add('markdown', '''## 4. Train CNN và Swin 1D
60 epochs tối đa, patience 12; AdamW, warmup/cosine, gradient clipping.
Không augmentation/sampling theo test. `X` trong NPZ đã normalize, không normalize lần hai.
Lưu best.pth, last.pth, config, history, summary từng run. Chạy lại cùng thư mục dùng resume.
Muốn resume giữa hai phiên Kaggle: tải và khôi phục cả thư mục project (đặc biệt last.pth),
hoặc giữ output bằng Save Version. Notebook không tự tìm checkpoint ở /kaggle/input.
''')
    add('code', '''runs = []
for family in ['cnn', 'swin']:
    base = json.loads((WORK_ROOT / f'training/configs/train-{family}.json').read_text(encoding='utf-8'))
    for seed in SEEDS:
        config = dict(base, seed=int(seed))
        config_path = WORK_ROOT / f'training/configs/{family}-seed{seed}.json'
        if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Run config changed: use a new run directory')
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        output = WORK_ROOT / f'artifacts/phase3/runs/{family}-seed{seed}'
        args = ['training/train_har.py', '--config', config_path, '--output-dir', output, '--device', DEVICE]
        if (output / 'last.pth').exists():
            args.append('--resume')
        run(*args)
        runs.append(output)
''')
    add('markdown', '''## 5. Đóng băng lựa chọn, đánh giá test một lần
Chọn seed tốt nhất theo validation trong mỗi family, giữ Swin làm model triển khai.
selection.json không cho thay candidates sau khi đã xem test. Nếu dưới 0.90 macro-F1,
đọc confusion matrix, ghi giới hạn demo nghiên cứu; không tune lại theo test để tuyên bố benchmark.
''')
    add('code', '''evaluation = WORK_ROOT / 'artifacts/phase3/evaluation'
run('training/evaluate_har.py', '--runs', *runs, '--output-dir', evaluation)
report = json.loads((evaluation / 'evaluation.json').read_text(encoding='utf-8'))
for result in report['results']:
    print(result['run'], 'accuracy', result['test']['accuracy'], 'macro-F1', result['test']['macro_f1'])
    print('Recall per class:', [(c['code'], c['recall']) for c in result['test']['per_class']])
print('Swin target >=0.90:', report['target_met'])
from IPython.display import display, Image
for path in sorted(evaluation.glob('*.png')):
    display(Image(filename=str(path)))
''')
    add('markdown', '''## 6. Tải artifact cho Phase 4
ZIP gồm best checkpoints, config/metadata, logs, predictions, biểu đồ và source hash.
Không gồm dữ liệu gốc hay last.pth/optimizer để nhẹ hơn; **không phải ZIP resume**.
best.pth chứa trained=true, labels và normalization thật. Chưa phải ONNX production.
Save Version để giữ output; tải ZIP về repo và kiểm tra hash trước bước export.
''')
    add('code', '''archive = WORK_ROOT.parent / 'phase3-results.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted((WORK_ROOT / 'artifacts').rglob('*')):
        if path.is_file() and path.name not in ['last.pth', 'last.pth.partial']:
            z.write(path, path.relative_to(WORK_ROOT))
    z.write(WORK_ROOT / 'source-hashes.json', 'source-hashes.json')
from IPython.display import FileLink
display(FileLink(str(archive)))
print('Checkpoint for deployment:', report['selection']['deployment_run'])
''')
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                              language_info=dict(name='python', version='3.11')),
                    nbformat=4, nbformat_minor=5)
    if gpu:
        for entry in cells:
            source = ''.join(entry['source'])
            source = source.replace('SEEDS = [42]  # Có quota: [42, 43, 44], chốt trước evaluation.',
                                    'SEEDS = [42, 43, 44]  # Predeclared, no changes after test evaluation.')
            source = source.replace("DEVICE = 'auto'  # Kaggle GPU nếu khả dụng, nếu không CPU.",
                                    "DEVICE = 'cuda'  # Fail explicitly if Kaggle GPU is unavailable.")
            source = source.replace("assert int(torch.__version__.split('.')[0]) >= 2, 'Need PyTorch 2+'",
                                    "assert int(torch.__version__.split('.')[0]) >= 2, 'Need PyTorch 2+'\nassert torch.cuda.is_available(), 'GPU required: enable Kaggle accelerator before running'")
            source = source.replace("run('training/evaluate_har.py', '--runs', *runs, '--output-dir', evaluation)",
                                    "run('training/evaluate_har.py', '--runs', *runs, '--output-dir', evaluation, '--all-seeds')")
            source = source.replace("print('Swin target >=0.90:', report['target_met'])",
                                    "print('Seed statistics:', json.dumps(report['seed_statistics'], indent=2))\nprint('Swin target >=0.90:', report['target_met'])")
            source = source.replace('Mặc định một seed để tiết kiệm quota; muốn ba seed phải chọn trước khi chạy evaluation.',
                                    'GPU run: ba seed 42/43/44 đã chốt, cùng hyperparameters CPU. Test chính thức đã xem ở lần local; đây là replication/multi-seed report, không phải benchmark mù mới.')
            entry['source'] = source.splitlines(keepends=True)
    target = ROOT / ('notebooks/kaggle_train_har_gpu.ipynb' if gpu else 'notebooks/kaggle_train_har.ipynb')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for entry in cells:
        if entry['cell_type'] == 'code':
            compile(''.join(entry['source']), entry['id'], 'exec')
    print(f'Generated {target}: {len(cells)} cells; all code cells compile.')
    return notebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Generate the predeclared CUDA three-seed replication notebook')
    build(parser.parse_args().gpu)
