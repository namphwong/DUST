# DrawAI — Báo cáo D4

Nghiệm thu local ngày 11/10/2026: **D4-01–D4-07 đạt** trên CPU và Chromium WASM. Release `drawai-swin-v1`, `trained=true`, dùng nguyên trọng số checkpoint D3 chọn bằng validation.

## Checkpoint và export

Run `official-swin-l128-aug1-seed44`, seed 44, best epoch 49; SHA-256 `a5ccb65dd4699b6451bf03ff23b6071382c29443213263f1fbc9ef6089fcf324`. Đối chiếu selection lock, config, nhãn, preprocessing, dữ liệu train/validation và load strict trước export. Không thay residual norm hoặc train thêm.

Float32, fixed shape `[1,3,128]` → `[1,10]`, input `drawing`, output `logits`, opset 17. ONNX checker full check đạt; không quantize. Tracer warnings của backbone liên quan shape cố định; không hỗ trợ dynamic batch/length.

## Full-test parity

| Chỉ số | Kết quả |
|---|---:|
| Test chính thức | 3.498 mẫu |
| Sai số logits tuyệt đối tối đa | 1.1444092e-05 |
| Argmax khác nhau | 0 |
| Accuracy PyTorch / ORT | 98.7707% / 98.7707% |
| Macro-F1 PyTorch / ORT | 0.987896 / 0.987896 |
| Giảm macro-F1 | 0 |
| Dung lượng ONNX | 2,384,716 bytes (2.27 MiB) |

Mọi logits đạt `abs(ORT−PyTorch) ≤ 1e-5 + 1e-4 × abs(PyTorch)`. Max absolute error có thể lớn hơn 1e-5 vì ngưỡng bao gồm sai số tương đối. Predictions tái tạo kết quả D3. Verifier độc lập đối chiếu nhãn/ID/test hash và tái tính confusion matrix, accuracy, macro-F1, per-class metrics từ NPZ.

## Browser và kiểm tra lỗi

Chromium thực, WASM CPU 1 thread trong Web Worker: 20 fixture gồm 10 chữ số train, 7 hình học và 3 tensor; tensor TypeScript khớp Python với tolerance 2e-7, logits/argmax khớp checkpoint. 14 input lỗi trả mã đúng. Build và 13 browser tests đạt (7 release, 6 regression D2); thiếu labels rồi retry, model corrupt, nhãn/provenance/preprocessing sai dù hash đồng bộ, trained=false đều bị từ chối.

Browser Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/153.0.8010.12 Safari/537.36; máy AMD Ryzen 7 H 255 w/ Radeon 780M Graphics. Median 1.50 ms, p95 2.10 ms; load 113.20 ms, init 388.90 ms. Benchmark 5 warm-up + 100 inference với tensor random, nearest-rank p95. ORT Python p95 0.92 ms, 4 threads, mẫu test đầu tiên; hai benchmark khác input/thread count nên không dùng như so tốc độ công bằng. Dung lượng ONNX không gồm WASM runtime. Chưa đo Edge/điện thoại hoặc production.

## Release và demo

`web/public/models/drawai-swin-v1/` chứa model, preprocessing, labels, fixtures, report và release manifest hash/bytes. Client kiểm tra file bắt buộc, hash/bytes, shape, nhãn 0–9, version và nguồn checkpoint; SHA-256 bảo đảm tính nhất quán file, không phải chữ ký xác thực.

`web/public/samples/drawai/` chứa 10 JSON theo schema D0 và manifest provenance. Chọn sample ID thấp nhất trong train mỗi lớp, không chọn theo dự đoán đúng. Đổi tọa độ `x−min(x), max(y)−y` để giữ hình học và dùng canvas y xuống. Đã đối chiếu nét gốc, nhãn, nguồn, train tensor và schema cho cả 10 mẫu.

Trang kiểm chứng `/drawai/release-check` dùng model đã train; `/drawai/check` vẫn dùng fixture D2 untrained. Canvas sản phẩm thuộc D5; chất lượng nét người dùng thuộc D6.

## Tái lập

```powershell
# Dependency kiểm toán schema, không cần cho inference:
./.venv-phase1/Scripts/python.exe -m pip install -r training/drawai/requirements-verification.txt
$env:DUST_DLL_DIRECTORY = 'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
./.venv-phase1/Scripts/python.exe -m training.drawai.export_onnx
Set-Location web
npm run build
npx playwright test --config playwright.drawai-release.config.ts
Set-Location ..
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_export
```

Trên Linux hoặc máy có VC runtime đúng, bỏ biến DLL Windows. Verifier cần `jsonschema` để kiểm tra demo theo schema; đây là dependency kiểm toán riêng. Kết quả tại `artifacts/drawai/export/`: full-test NPZ, Python/browser reports, Playwright report, ảnh và verification.json. [Model card](model-card.md) · [Báo cáo D3](experiment-report.md).

P6 filename migration: canonical `drawai_swin1d_pendigits_l128_fp32_v1.onnx`; `model.onnx` remains a byte-identical compatibility alias. Weights, PenDigits metrics and product design are unchanged. Paint35 exports are documented separately in phase6-report.md.
