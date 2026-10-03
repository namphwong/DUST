# Phase 4 — Artifact Swin đã train cho trình duyệt

Ngày nghiệm thu: 03/10/2026. Trạng thái **PASS P4-01–P4-04**. Không train lại, không đổi checkpoint theo test, không đổi kiến trúc hoặc kích hoạt norm bằng tay như fixture Phase 1.

## Artifact phát hành

Release `har-swin-kaggle-v1` tại `web/public/models/har-swin-kaggle-v1/`. Nguồn là validation winner `swin-seed44`, best epoch 31 của Kaggle run 354801477. Checkpoint SHA-256 `f45501770c7a66b0cb5e29a5b3f7a84bd5ad204eb63dab80fdcad865fe9bc749`.

- `model.onnx`: float32, opset 17, 9.036.087 bytes (~8,62 MiB), **trained=true** trong metadata; chưa quantize.
- `preprocessing.json`: mean/std Phase 2 fit train-only, total acceleration có gravity, g, 50 Hz; không thay bằng body acceleration.
- `labels.json`: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING, theo ID 0–5.
- `fixtures.json`: 8 mẫu tổng hợp + cửa sổ test đầu tiên của mỗi lớp, không chọn theo dự đoán đúng. Bao gồm raw, normalized tensor, logits PyTorch và provenance của mẫu thật.
- `report.json`: config, metric, shape, hash checkpoint/ONNX, version môi trường và giới hạn.
- `release.json`: version cùng SHA-256/byte size của cả năm file; frontend từ chối file thiếu, sai hash, sai version hoặc trained=false trước inference.

Model SHA-256: `73c12455363fd3f80d9f2ffff44c29d35d5b9aba5861db6d288f1a910dd0d75b`.

## Hợp đồng inference

Input `signals`: float32 **[1,3,128]**, layout NCT/channel-major, đã chuẩn hóa đúng một lần bởi caller: `(raw_g - mean[channel]) / std[channel]`. Output `logits`: float32 **[1,6]**; argmax là ID nhãn, softmax là điểm model không được coi là độ chính xác bảo đảm. ONNX không tự chuẩn hóa. Trình duyệt kiểm tra raw → normalized so với fixture Python trước khi đối chiếu logits.

Batch/sequence **cố định**, không tuyên bố support dynamic batch. Export bằng torch 2.5.1 legacy exporter đã kiểm chứng trong Phase 1 (`dynamo=False`), opset 17. TracerWarning của assert chiều dài/window_reverse phù hợp với hợp đồng cố định; không bỏ qua để tuyên bố shape động. Đọc checkpoint hash-verified với weights_only=True, chỉ allowlist TorchVersion của thư viện đã cài. Không thay trọng số đã học.

## Parity và chất lượng

| Kiểm tra | Kết quả |
|---|---:|
| Full official test, PyTorch CPU ↔ ONNX Python | 2947/2947 logits trong atol=1e-5, rtol=1e-4 |
| Sai số tuyệt đối lớn nhất full test | 9,5367e-6 |
| Argmax PyTorch ↔ ONNX | 2947/2947 |
| Argmax Kaggle CUDA đã lưu ↔ ONNX | 2947/2947 |
| ONNX test accuracy | 0,871734 |
| ONNX test macro-F1 | 0,870489 |
| Giảm macro-F1 so với Kaggle checkpoint | 0,000000; giới hạn 0,005 |
| Browser ↔ PyTorch | 14/14 fixture PASS |
| Browser preprocessing ↔ Python | max error 0 |

Metric không tăng do export, và **vẫn dưới mục tiêu macro-F1 0,90**. Các cửa sổ demo có cả dự đoán sai (xuống cầu thang → lên cầu thang, ngồi → đứng); giữ nguyên để minh bạch, không chọn lại mẫu đẹp hơn. Xem [model card](model-card.md) và [báo cáo training](kaggle-training-report.md).

## Benchmark và test browser

Chromium headless 153.0.8010.12, Windows/máy tham chiếu Phase 1, ONNX Runtime Web 1.20.1 WASM CPU **một thread**. Đo 30 inference sau 5 warm-up; số thực tế mới nhất nằm trong [browser-verification.json](../artifacts/phase4/browser-verification.json). Mục tiêu model ≤10 MiB và median inference ≤500 ms đều được assert bằng browser test. Tải/khởi tạo là số đo localhost, không đại diện cho mạng Internet/Vercel; viewport mobile chỉ kiểm layout, không phải benchmark điện thoại.

Lượt đo nghiệm thu: median **2,90 ms/cửa sổ**, tải artifact **324 ms**, khởi tạo **426 ms**; browser logit max error **4,2915e-6**. Không bảo đảm các số này cho mọi thiết bị/mạng. [Checklist tổng hợp](../artifacts/phase4/verification.json).

Runtime .wasm/.mjs được bundle cùng phiên bản package, không cần CDN/API trả phí; model tải khi bấm chạy. Giữ numThreads=1 để không yêu cầu cross-origin isolation; cấu hình dựa trên [hướng dẫn ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html). WASM khoảng 10,72 MiB không nén, còn có JS/module/fixture và artifact Phase 1: mục tiêu 10 MiB chỉ dành cho model, không phải tổng download/build. Chưa tuyên bố đã tối ưu tải/cache Vercel.

Trang kỹ thuật hiện ở `/?phase=4`; trang mặc định là MotionSense Phase 5 dùng artifact đã train. `/?phase=1` giữ kiểm chứng untrained cũ. Test gồm chạy lại/release session, parity sai phải FAIL và không hiện dự đoán, missing model/retry, metadata sai hash, stale/untrained release, desktop/mobile không tràn ngang và không gửi tín hiệu qua POST. Lượt nghiệm thu Phase 4: **8/8 browser tests**, **14/14 Python unit tests**, TypeScript/Vite build/compileall pass; regression tiếp tục được chạy trong bộ test Phase 5.

## Nghiệm thu

| Tiêu chí | Bằng chứng | Trạng thái |
|---|---|---|
| P4-01 | Full test parity + 14 fixture browser, export-verification.json | PASS |
| P4-02 | ONNX macro-F1 bằng checkpoint, degradation=0; không quantize | PASS |
| P4-03 | Versioned release, SHA-256 mọi file, labels/preprocessing thật, trained=true | PASS |
| P4-04 | Browser WASM warm-up/benchmark, manifest check, browser-verification.json | PASS |

Bằng chứng: [Python](../artifacts/phase4/export-verification.json), [browser](../artifacts/phase4/browser-verification.json), [desktop screenshot](../artifacts/phase4/browser-proof.png), [mobile screenshot](../artifacts/phase4/browser-mobile-proof.png).

## Chạy lại và handoff

Từ repo, cần checkpoint Kaggle đã import và dữ liệu Phase 2. Dùng requirements-phase1.txt đã pin; không nâng exporter/runtime trong lần nghiệm thu này.

```powershell
# Chỉ trên máy Windows hiện tại nếu ONNX cần runtime DLL cục bộ:
$env:DUST_DLL_DIRECTORY = 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'
.venv-phase1/Scripts/python.exe training/export_trained_har.py
.venv-phase1/Scripts/python.exe -m unittest discover -s training/tests -v
cd web
npm run build
npm test
npm run preview -- --port 4173
```

Mở http://127.0.0.1:4173/?phase=4, bấm “Chạy kiểm chứng”. Từ Phase 5, tests tự bật/tắt preview port 4174, không đóng preview người dùng 4173. Với Linux/Kaggle, bỏ biến DLL. Exporter assert source checkpoint/data hash, shape/checker, logits hữu hạn/parity; không ghi report PASS nếu lỗi. Re-export cùng version chỉ tái tạo cùng checkpoint/config, chạy lại browser tests sau export; thay model/metadata contract phải tạo version mới, không ghi đè release đã deploy. Phase 7 sẽ thêm cache/rollback; giữ Phase 1 để kiểm tra lịch sử, không dùng nhận dạng.

Phase 5–6 chưa thực hiện: CSV người dùng, UI MVP, worker, hủy tác vụ, resample/gaps và toàn pipeline. Phase 7 chưa deploy Vercel. **Phase 4 hoàn tất artifact/integration kỹ thuật, không có nghĩa toàn bộ web đã hoàn tất.**
