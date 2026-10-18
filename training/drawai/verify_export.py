"""Independently verify D4 release hashes, stored full-test logits, demos and browser evidence."""
import json

import numpy as np
import jsonschema

from .prepare_data import ROOT, digest, write_json, parse_unipen
from .preprocessing import preprocess, contract
from .train import load_split, tensor_hash
from .verify_training import independent_metrics


def verify():
    target=ROOT/'web/public/models/drawai-swin-v1'; out=ROOT/'artifacts/drawai/export'
    experiment=ROOT/'artifacts/drawai/d3-local-v1'
    release=json.loads((target/'release.json').read_text())
    assert release['version']=='drawai-swin-v1' and release['trained'] is True
    assert set(release['files'])=={'model.onnx','fixtures.json','preprocessing.json','labels.json','report.json'}
    for name,metadata in release['files'].items():
        assert digest(target/name)==metadata['sha256'] and (target/name).stat().st_size==metadata['bytes']
    report=json.loads((target/'report.json').read_text())
    assert report==json.loads((out/'python-report.json').read_text())
    selection=json.loads((experiment/'selection.json').read_text())
    assert report['checkpoint_sha256']==selection['checkpoint_sha256']
    assert report['selection_sha256']==digest(experiment/'selection.json')
    assert digest(experiment/'runs'/report['run_id']/'best.pth')==report['checkpoint_sha256']
    assert json.loads((target/'preprocessing.json').read_text())==contract(128)
    assert json.loads((target/'labels.json').read_text())=={'classes':[dict(id=i,name=str(i)) for i in range(10)]}
    data_dir=ROOT/'data/processed/pendigits-drawai-v1/l128'
    test=load_split(data_dir,'test',128)
    with np.load(out/'full-test-parity.npz',allow_pickle=False) as values:
        np.testing.assert_array_equal(values['sample_ids'],test[2]); np.testing.assert_array_equal(values['y_true'],test[1])
        p,o=values['pytorch_logits'],values['onnx_logits']
        assert p.shape==o.shape==(3498,10) and p.dtype==o.dtype==np.float32
        assert np.isfinite(p).all() and np.isfinite(o).all()
        np.testing.assert_allclose(o,p,atol=1e-5,rtol=1e-4)
        assert float(np.max(np.abs(p-o)))==report['max_abs_error']
        assert np.count_nonzero(p.argmax(1)!=o.argmax(1))==report['argmax_disagreements']==0
        for logits,name in [(p,'pytorch_metrics'),(o,'onnx_metrics')]:
            matrix,accuracy,f1,per=independent_metrics(test[1],logits.argmax(1))
            assert matrix==report[name]['confusion_matrix']
            assert abs(accuracy-report[name]['accuracy'])<1e-12 and abs(f1-report[name]['macro_f1'])<1e-12
            for expected,row in zip(per,report[name]['per_class']):
                assert all(abs(a-b)<1e-12 for a,b in zip(expected,[row['precision'],row['recall'],row['f1'],row['support']]))
    assert report['test_sha256']==tensor_hash(*test) and report['macro_f1_degradation']==0
    manifest=json.loads((data_dir/'manifest.json').read_text())
    demo_dir=ROOT/'web/public/samples/drawai'; demos=json.loads((demo_dir/'manifest.json').read_text())
    assert digest(demo_dir/'manifest.json')==report['demo_manifest_sha256']
    assert len(demos['samples'])==10 and {e['label'] for e in demos['samples']}==set(range(10))
    schema=json.loads((ROOT/'docs/drawai/strokes.schema.json').read_text())
    raw=next((ROOT/'data/raw').rglob('pendigits-orig.tra'))
    assert digest(raw)==manifest['sources'][raw.name]['uncompressed_sha256']
    original={s['id']:s for s in parse_unipen(raw.read_text(),raw.name)}
    train=load_split(data_dir,'train',128)
    for entry in demos['samples']:
        value=json.loads((demo_dir/entry['file']).read_text()); jsonschema.validate(value,schema)
        assert digest(demo_dir/entry['file'])==entry['sha256'] and (demo_dir/entry['file']).stat().st_size==entry['bytes']
        first=min((e for e in manifest['splits']['train']['entries'] if e['label']==entry['label']),key=lambda e:e['id'])
        assert entry['sample_id']==first['id'] and entry['split']=='train'
        assert value['reference_label']==entry['label'] and value['source']['sample_id']==first['id']
        assert entry['provenance']['source_sha256']==digest(raw)
        raw_tensor=preprocess(original[first['id']]['strokes'],128,'up')
        np.testing.assert_array_equal(preprocess(value['strokes'],128,'down'),raw_tensor)
        np.testing.assert_array_equal(raw_tensor,train[0][list(train[2]).index(first['id'])])
    browser=json.loads((out/'browser-report.json').read_text()); result=browser['report']
    assert browser['status']=='passed' and browser['releaseSha256']==digest(target/'release.json')
    assert browser['modelSha256']==report['sha256'] and result['modelSha256']==report['sha256']
    assert result['passed'] and result['trained'] and result['length']==128
    assert len(result['cases'])==20 and all(c['passed'] for c in result['cases'])
    assert len(result['invalidCases'])==14 and all(c['passed'] for c in result['invalidCases'])
    fixtures=json.loads((target/'fixtures.json').read_text())['cases']
    for actual,expected in zip(result['cases'],fixtures):
        assert actual['name']==expected['name'] and actual['preprocessingError']<=2e-7
        np.testing.assert_allclose(actual['logits'],expected['expected'],atol=1e-5,rtol=1e-4)
        assert np.argmax(actual['logits'])==np.argmax(expected['expected'])
    playwright=json.loads((out/'playwright-report.json').read_text())
    assert playwright['stats']['expected']==13 and playwright['stats']['unexpected']==0
    assert playwright['stats']['skipped']==0 and playwright['stats']['flaky']==0
    verification=dict(phase='D4',status='passed-local-cpu-chromium',
                      tasks={f'D4-{i:02}':'passed-local' for i in range(1,8)},
                      full_test_samples=3498,demo_samples=10,browser_fixtures=20,browser_tests_passed=13,
                      release_sha256=digest(target/'release.json'),checkpoint_sha256=report['checkpoint_sha256'],
                      model_sha256=report['sha256'],parity_npz_sha256=digest(out/'full-test-parity.npz'),
                      browser_report_sha256=digest(out/'browser-report.json'),
                      python_report_sha256=digest(out/'python-report.json'),
                      build='npm run build passed',kaggle_gpu_executed=False,canvas_evaluated=False)
    write_json(out/'verification.json',verification)
    docs=ROOT/'docs/drawai'
    commands="""```powershell
# Dependency kiểm toán schema, không cần cho inference:
./.venv-phase1/Scripts/python.exe -m pip install -r training/drawai/requirements-verification.txt
$env:DUST_DLL_DIRECTORY = 'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
./.venv-phase1/Scripts/python.exe -m training.drawai.export_onnx
Set-Location web
npm run build
npx playwright test --config playwright.drawai-release.config.ts
Set-Location ..
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_export
```"""
    (docs/'artifact-report.md').write_text(f"""# DrawAI — Báo cáo D4

Nghiệm thu local ngày 11/10/2026: **D4-01–D4-07 đạt** trên CPU và Chromium WASM. Release `drawai-swin-v1`, `trained=true`, dùng nguyên trọng số checkpoint D3 chọn bằng validation.

## Checkpoint và export

Run `{report['run_id']}`, seed {report['seed']}, best epoch {report['best_epoch']}; SHA-256 `{report['checkpoint_sha256']}`. Đối chiếu selection lock, config, nhãn, preprocessing, dữ liệu train/validation và load strict trước export. Không thay residual norm hoặc train thêm.

Float32, fixed shape `[1,3,128]` → `[1,10]`, input `drawing`, output `logits`, opset 17. ONNX checker full check đạt; không quantize. Tracer warnings của backbone liên quan shape cố định; không hỗ trợ dynamic batch/length.

## Full-test parity

| Chỉ số | Kết quả |
|---|---:|
| Test chính thức | 3.498 mẫu |
| Sai số logits tuyệt đối tối đa | {report['max_abs_error']:.8g} |
| Argmax khác nhau | 0 |
| Accuracy PyTorch / ORT | {report['onnx_metrics']['accuracy']*100:.4f}% / {report['onnx_metrics']['accuracy']*100:.4f}% |
| Macro-F1 PyTorch / ORT | {report['onnx_metrics']['macro_f1']:.6f} / {report['onnx_metrics']['macro_f1']:.6f} |
| Giảm macro-F1 | 0 |
| Dung lượng ONNX | {report['model_bytes']:,} bytes ({report['model_bytes']/1024**2:.2f} MiB) |

Mọi logits đạt `abs(ORT−PyTorch) ≤ 1e-5 + 1e-4 × abs(PyTorch)`. Max absolute error có thể lớn hơn 1e-5 vì ngưỡng bao gồm sai số tương đối. Predictions tái tạo kết quả D3. Verifier độc lập đối chiếu nhãn/ID/test hash và tái tính confusion matrix, accuracy, macro-F1, per-class metrics từ NPZ.

## Browser và kiểm tra lỗi

Chromium thực, WASM CPU 1 thread trong Web Worker: 20 fixture gồm 10 chữ số train, 7 hình học và 3 tensor; tensor TypeScript khớp Python với tolerance 2e-7, logits/argmax khớp checkpoint. 14 input lỗi trả mã đúng. Build và 13 browser tests đạt (7 release, 6 regression D2); thiếu labels rồi retry, model corrupt, nhãn/provenance/preprocessing sai dù hash đồng bộ, trained=false đều bị từ chối.

Browser {browser['userAgent']}; máy {browser['machine']['cpu']}. Median {result['medianMs']:.2f} ms, p95 {result['p95Ms']:.2f} ms; load {result['loadMs']:.2f} ms, init {result['initMs']:.2f} ms. Benchmark 5 warm-up + 100 inference với tensor random, nearest-rank p95. ORT Python p95 {report['python_ort_p95_ms']:.2f} ms, 4 threads, mẫu test đầu tiên; hai benchmark khác input/thread count nên không dùng như so tốc độ công bằng. Dung lượng ONNX không gồm WASM runtime. Chưa đo Edge/điện thoại hoặc production.

## Release và demo

`web/public/models/drawai-swin-v1/` chứa model, preprocessing, labels, fixtures, report và release manifest hash/bytes. Client kiểm tra file bắt buộc, hash/bytes, shape, nhãn 0–9, version và nguồn checkpoint; SHA-256 bảo đảm tính nhất quán file, không phải chữ ký xác thực.

`web/public/samples/drawai/` chứa 10 JSON theo schema D0 và manifest provenance. Chọn sample ID thấp nhất trong train mỗi lớp, không chọn theo dự đoán đúng. Đổi tọa độ `x−min(x), max(y)−y` để giữ hình học và dùng canvas y xuống. Đã đối chiếu nét gốc, nhãn, nguồn, train tensor và schema cho cả 10 mẫu.

Trang kiểm chứng `/drawai/release-check` dùng model đã train; `/drawai/check` vẫn dùng fixture D2 untrained. Canvas sản phẩm thuộc D5; chất lượng nét người dùng thuộc D6.

## Tái lập

{commands}

Trên Linux hoặc máy có VC runtime đúng, bỏ biến DLL Windows. Verifier cần `jsonschema` để kiểm tra demo theo schema; đây là dependency kiểm toán riêng. Kết quả tại `artifacts/drawai/export/`: full-test NPZ, Python/browser reports, Playwright report, ảnh và verification.json. [Model card](model-card.md) · [Báo cáo D3](experiment-report.md).
""",encoding='utf-8')
    (docs/'model-card.md').write_text(f"""# DrawAI — Model card drawai-swin-v1

Ngày 11/10/2026. Swin Transformer 1D cho một chữ số 0–9 từ chuỗi tọa độ nét bút; phát hành artifact local, chưa deploy canvas.

- Backbone `models/SwinTransformer_cls.py::SwinTransformer_1D`, 545.038 tham số, embed32, depths [2,2,2], heads [2,4,8], patch2/window8; 3 input channels x/y/stroke_start, 10 logits.
- PenDigits gốc: train 5.999, validation 1.495, official test 3.498. Validation phân tầng seed42; writer ID chưa xác minh, train/val không bảo đảm khác người. Test chính thức giữ nguyên theo D1.
- `drawai-geometry-v1`: tâm bounding box/max-span giữ tỷ lệ, resample theo từng nét thành L128; tensor float32 NCT `[1,3,128]`, hệ y xuống. Không dùng timestamp, padding hoặc learned normalization. Nhãn có thứ tự 0–9.
- Training local CPU Torch {report['runtime']['torch']}: AdamW lr .001, batch128, weight decay .01; augmentation train nhẹ; seeds42/43/44 và ngân sách pilot CNN/Swin ngang nhau. Run `{report['run_id']}`, best epoch {report['best_epoch']}, chọn bằng validation trước test.
- Checkpoint SHA-256 `{report['checkpoint_sha256']}`; ONNX SHA-256 `{report['sha256']}`. Release manifest ghi hash/bytes từng file.
- Official test accuracy **{report['onnx_metrics']['accuracy']*100:.2f}%**, macro-F1 **{report['onnx_metrics']['macro_f1']*100:.2f}%**; D4 full-test PyTorch/ORT có 0 argmax khác, macro-F1 không giảm. CNN baseline trung bình tốt hơn Swin; xem [D3](experiment-report.md) để so đủ ba seed.
- ONNX float32 opset17 fixed batch1/L128, {report['model_bytes']/1024**2:.2f} MiB. Chromium WASM 1 thread: 20 fixture đạt, p95 {result['p95Ms']:.2f} ms trên laptop local. Chưa quantize hoặc benchmark điện thoại/Edge.
- Demo là 10 mẫu train có nhãn nguồn/provenance, chọn độc lập dự đoán. Softmax sau này là điểm model, không phải độ chính xác bảo đảm.
- Giới hạn: chỉ một chữ số hoàn chỉnh, không OCR ảnh/nhiều chữ số; PenDigits không đại diện đầy đủ chuột/cảm ứng/thiết bị/người mới. Nét canvas thực chưa được thu/đánh giá; Kaggle GPU chưa chạy. Không tuyên bố metric này là chất lượng ứng dụng ngoài dataset.

[Dataset/preprocessing](data-report.md) · [Protocol training](experiment-protocol.md) · [Export, parity và lệnh tái lập](artifact-report.md). Nguồn dataset, archive hashes và điều kiện nguồn nằm trong báo cáo D1; không bổ sung tuyên bố giấy phép khi chưa có xác minh.
""",encoding='utf-8')
    verification['report_sha256']=digest(docs/'artifact-report.md')
    verification['model_card_sha256']=digest(docs/'model-card.md')
    verification['source_files']={str(p.relative_to(ROOT)):digest(p) for p in [
        ROOT/'training/drawai/export_onnx.py',ROOT/'training/drawai/verify_export.py',
        ROOT/'web/src/drawai/feasibility-worker.ts',ROOT/'web/src/drawai/feasibility-app.tsx',
        ROOT/'web/tests/drawai-release.spec.ts',ROOT/'web/playwright.drawai-release.config.ts']}
    write_json(out/'d4-review.json',verification)
    print('PASS: independently verified full-test parity, 10 schema/provenance demos and browser release evidence.')


if __name__=='__main__': verify()
