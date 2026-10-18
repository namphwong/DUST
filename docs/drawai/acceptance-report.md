# DrawAI — Nghiệm thu D6

Ngày 11/10/2026. **Kiểm chứng kỹ thuật local đạt; D6 chưa hoàn thành nghiệm thu người thật.** Theo lựa chọn của người dùng, chưa có người/thiết bị để thu nét; không thay dữ liệu đó bằng fixture hoặc Pointer Events tự động.

| Task | Trạng thái |
|---|---|
| D6-01 | passed-local |
| D6-02 | passed-local |
| D6-03 | passed-local |
| D6-04 | passed-local |
| D6-05 | desktop-passed; real-touch-not-run |
| D6-06 | not-run; no participants |
| D6-07 | not-run; no human data |
| D6-08 | local-report-complete; full-acceptance-pending |

## Kiểm chứng toàn tuyến và lỗi

Build TypeScript/Vite và 34 browser tests đạt trên Chrome/Edge cài trên máy (headless, profile mới). 17 fixture JSON gồm 7 hình học và 10 train PenDigits: một/nhiều nét, dọc/ngang, điểm trùng, singleton trong bản vẽ có chuyển động; thêm 7 JSON lỗi. Cùng JSON được xử lý bằng Python, đường preprocessing/inference dùng chung với sản phẩm và một InferenceClient/worker sản phẩm thật. Verifier độc lập so tensor, logits PyTorch/ORT/WASM, argmax và softmax.
Race: nhận dạng rồi vẽ/xóa/undo/nạp mẫu trong lúc tải trì hoãn; response cũ bị bỏ, request tiếp theo hoạt động. Nhấn nhanh không chạy trùng, cancel/resize/DPR2/pointer cạnh tranh đạt. Worker crash được mô phỏng bằng ErrorEvent; timeout dùng đồng hồ page chạy nhanh qua 60 giây, sau reset/retry có inference thật. Đây là phép thử điều khiển lỗi, không phải đợi stall thực 60 giây hoặc chứng minh mọi tình huống browser crash.
Tải model 404, hash sai, hợp đồng sai dù hash nhất quán, release cache cũ và WASM 404 đều bị từ chối/giữ nét; retry sau khi sửa nguồn hoạt động. Request inference/demo/technical chỉ GET static assets cùng origin, không có POST nét; worker giữ session cho request tiếp theo. Loader đã chuẩn hóa lỗi mạng thành FETCH, lỗi khởi tạo thành INIT; client kiểm tra thêm preprocessing version và timing hữu hạn; worker từ chối message null thay vì crash.

## Benchmark laptop

| Browser / version | Load ms | Verify ms | Init ms | Preprocess median ms | Inference median ms | Inference p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| chrome 154.0.8037.98 | 146.50 | 4.80 | 670.90 | 0.10 | 2.60 | 3.90 |
| edge 155.0.4283.45 | 129.60 | 4.60 | 661.90 | 0.00 | 2.50 | 3.60 |

ONNX 2,384,716 bytes (2.27 MiB), đạt ≤10 MiB; p95 đạt ≤100 ms ở cả hai browser. Máy AMD Ryzen 7 H 255 w/ Radeon 780M Graphics. WASM CPU 1 thread, cùng fixture train-digit-0, 5 warm-up + 100 inference; p95 nearest rank. Load/verify/init là một lần cold session trong profile mới, có thể chịu OS/network cache; không phải thống kê nhiều lần cold load. Benchmark dùng shared inference path trong worker, đo session.run riêng; request sản phẩm cùng đường này đã được đối chiếu logits. Không suy latency laptop thành latency điện thoại, không tính WASM runtime vào dung lượng ONNX.

| Browser | Tensor max error | Logits max error (ORT/WASM vs PyTorch) |
|---|---:|---:|
| chrome | 0 | 5.01e-06 |
| edge | 0 | 5.01e-06 |

Tensor tolerance 2e-7; logits abs(error) ≤ 1e-5 + 1e-4×abs(PyTorch). Mọi argmax khớp. Chrome/Edge desktop, viewport 1366×768/390×844 và DPR1/2 đã kiểm tra; mobile viewport và synthetic pointer không phải điện thoại/cảm ứng/bút thật.

## Nét người thật và điều kiện còn thiếu

**0 mẫu người tham gia; accuracy/macro-F1 canvas: NOT RUN.** [Protocol cố định](canvas-protocol.md), template manifest và CLI validate/evaluate đã chuẩn bị. Bốn tests dùng dữ liệu synthetic trong thư mục tạm kiểm tra đủ ma trận 250 mẫu, thiếu/duplicate, source/consent/attestation, timestamp và path traversal; chúng không tạo artifact người thật hoặc metric canvas.
Cần ít nhất 5 người × 10 chữ số × 5 lần = 250 JSON với consent, ID ẩn danh và device metadata. Người cung cấp phải xác nhận nguồn người thật; công cụ không tự chứng minh tính xác thực. Toàn bộ tập giữ làm evaluation cho release đã khóa, không fine-tune hoặc chọn lại model bằng kết quả này. Cần thử thiết bị cảm ứng/bút thật riêng trước nghiệm thu D6 đầy đủ. D7 có thể chuẩn bị demo nghiên cứu nhưng không gọi D6 hoàn thành toàn bộ.

## Tái lập

```powershell
$env:DUST_DLL_DIRECTORY = 'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
./.venv-phase1/Scripts/python.exe -m training.drawai.prepare_integration
./.venv-phase1/Scripts/python.exe -m unittest training.tests.test_drawai_canvas -v
Set-Location web
npm run build
npx playwright test --config playwright.drawai-integration.config.ts
Set-Location ..
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_integration
```

Cần Chrome/Edge cài local cho channels chrome/msedge; không tự đổi sang Chromium để gọi đó là Chrome/Edge. Python verification dùng requirements-verification.txt; Linux bỏ biến DLL. Tests port4186; preview4183 có `/drawai` và `/drawai/integration-check`. Evidence tại artifacts/drawai/verification/ và benchmark/; d6-review.json ghi hash, PASS/NOT RUN và phase_complete=false.
