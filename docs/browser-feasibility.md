# Phase 1 — Báo cáo khả năng chạy Swin 1D trong trình duyệt

Ngày nghiệm thu: 02/10/2026. Kết quả: **PASS**, có thể tiếp tục Phase 2.

Đã kiểm chứng mô hình SwinTransformer_1D thực tế từ models/SwinTransformer_cls.py chạy qua PyTorch → ONNX Runtime Python → ONNX Runtime Web WASM CPU. Mô hình dùng trọng số khởi tạo, chưa học hoạt động; đây là bằng chứng triển khai kỹ thuật, chưa phải bằng chứng độ chính xác phân loại.

## Cấu hình và kiểm tra kiến trúc

- Input float32 [1,3,128], output [1,6] logits; channel-major theo Phase 0.
- Patch size 2, window size 8, embed_dim 32, depths [2,2,2,2], heads [2,4,8,16].
- Chiều dài token qua bốn stage: 64 → 32 → 16 → 8; chiều rộng đặc trưng: 32 → 64 → 128 → 256.
- Stage cuối dài bằng window nên tự tắt shift; các stage trước có block shifted-window để kiểm chứng torch.roll/mask.
- 2.191.594 tham số; export ONNX opset 17, shape cố định.
- Attention, MLP, patch merging, normalization và average pooling đều xuất được với cấu hình này; chưa cần sửa kiến trúc gốc.

Repo khởi tạo residual post-norm bằng zero. Riêng fixture export đặt norm1.weight/norm2.weight thành 1 để đầu ra thực sự phụ thuộc attention/MLP; thay đổi nằm trong script thử, không thay mô hình gốc hoặc quy tắc training. Backward với batch 2 kiểm tra gradient hữu hạn và khác 0 ở patch embedding và shifted attention.

Exporter phát TracerWarning vì assert chiều dài và phép chuyển batch thành int trong window_reverse. Điều này phù hợp với input cố định [1,3,128] đang nghiệm thu; không tuyên bố ONNX hỗ trợ batch/sequence động. Khi train dùng PyTorch batch bình thường, web xử lý từng cửa sổ batch 1.

## Kết quả đo

Máy tham chiếu: Windows, AMD Ryzen 7 H 255, RAM khoảng 15.3 GB. Browser kiểm thử Chromium headless 153.0.8010.12. Browser dùng WASM một thread, không yêu cầu COOP/COEP hoặc GPU.

| Hạng mục | Kết quả |
|---|---|
| Mẫu kiểm chứng | 8: zeros, ones, sine, impulse và 4 mẫu random |
| PyTorch ↔ ONNX Python, sai số lớn nhất | 4.7684e-7 |
| PyTorch ↔ browser WASM, sai số lớn nhất | 5.6624e-7 |
| Điều kiện parity | abs(actual−expected) ≤ 1e-5 + 1e-4 × abs(expected) |
| Kích thước model | 8.998.779 bytes = 8,58 MiB |
| Median inference browser | Khoảng 3,60 ms/cửa sổ |
| Tải artifact từ localhost | Khoảng 446 ms |
| Khởi tạo session/WASM | Khoảng 513 ms |
| Benchmark | 5 warm-up, 30 lần đo; median |
| Browser tests | 3/3 pass |

Thời gian localhost không đại diện cho Internet/Vercel. Benchmark là một lượt đo trên máy tham chiếu, không bảo đảm cho mọi thiết bị. Model đạt hai mục tiêu Phase 0: ≤10 MiB và inference ≤500 ms/cửa sổ.

WASM runtime còn khoảng 10,72 MiB không nén (build báo gzip khoảng 2,85 MiB), cùng module khởi tạo khoảng 24,62 kB và bundle JS khoảng 197,52 kB. Giới hạn 10 MiB là mục tiêu riêng cho model; tổng tải lần đầu còn gồm runtime. Tất cả runtime assets được bundle cục bộ, không dựa vào CDN bên ngoài.

## Các lỗi đã xử lý

1. Python toàn cục trên máy lỗi torch DLL: dùng .venv-phase1 riêng với torch CPU 2.5.1.
2. ONNX trên máy cần VC runtime mới hơn: dùng DUST_DLL_DIRECTORY để thêm DLL search path chỉ trong tiến trình. Không thay DLL hệ thống. Phiên bản ONNX đã kiểm chứng là 1.16.2.
3. Production build thiếu ort-wasm-simd-threaded.mjs khiến browser không khởi tạo backend: import cả .mjs và .wasm bằng Vite ?url, khai báo cả hai đường dẫn ở ort.env.wasm.wasmPaths.
4. Residual norm bằng zero làm fixture ít nhạy với attention: kích hoạt nhánh trong mô hình thử và kiểm tra gradient.

## Nghiệm thu

| Tiêu chí | Bằng chứng | Trạng thái |
|---|---|---|
| P1-01 | Forward input/output đúng shape | PASS |
| P1-02 | Gradient hữu hạn, khác 0, được assert trong export script | PASS |
| P1-03 | onnx.checker, opset 17 và metadata trong report.json | PASS |
| P1-04 | 8 mẫu Python/browser và test cố ý thay logit cho FAIL | PASS |
| P1-05 | Chromium chạy thật, chạy lại session và test missing model | PASS |
| P1-06 | trained=false và thông báo rõ trên giao diện | PASS |
| P1-07 | Báo cáo size/tải/init/benchmark có đủ số lần đo | PASS |

Build TypeScript/Vite thành công. npm audit đã không báo lỗ hổng ở bộ dependency cài đặt khi cập nhật; kiểm tra runtime dependencies cuối cũng báo 0. Browser test còn kiểm tra viewport 390×844 không tràn ngang; đây là kiểm tra layout, không phải benchmark thiết bị di động.

## Bằng chứng và giới hạn

- [Report browser](phase1-browser-results.json): số đo, user-agent, logits và error từng mẫu.
- [Report Python](../web/public/models/phase1/report.json): cấu hình, version, SHA-256, parity và backward.
- [Script export](../training/export_smoke_test.py).
- [Trang kiểm chứng](../web/src/main.tsx) và [browser tests](../web/tests/browser.spec.ts).
- Ảnh sau test: web/test-results/phase1-browser.png (artifact local, không commit).

Phase 1 chưa train UCI, chưa đo accuracy, chưa triển khai CSV/worker, chưa test WebGPU/mobile inference và chưa deploy Vercel. Những phần đó thuộc các phase sau. Khi đã train, phải xuất lại checkpoint và kiểm tra parity theo Phase 4.

## Cách chạy lại

Từ thư mục repo, dùng uv (đã có trên máy này):

```powershell
uv venv .venv-phase1 --python 3.11
uv pip install --python .venv-phase1/Scripts/python.exe torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-phase1/Scripts/python.exe -r training/requirements-phase1.txt
.venv-phase1/Scripts/python.exe training/export_smoke_test.py
```

Trên máy Windows hiện tại, đặt biến sau trước khi chạy export nếu gặp lỗi DLL ONNX; đây là runtime có sẵn trên máy, không phải đường dẫn portable:

```powershell
$env:DUST_DLL_DIRECTORY = 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'
.venv-phase1/Scripts/python.exe training/export_smoke_test.py
```

Máy khác cần Visual C++ runtime tương thích hoặc cung cấp thư mục runtime qua DUST_DLL_DIRECTORY. Linux/Kaggle không sử dụng biến này. Không thay Python toàn cục; .venv-phase1 là môi trường riêng.

Chạy web sau khi export:

```powershell
cd web
npm ci
npm run dev
```

Mở http://127.0.0.1:5173 và bấm “Chạy kiểm chứng”. Artifact được tải khi bấm nút. Để kiểm tra production build:

```powershell
npm run build
npx playwright install chromium
npm test
```

Tests tự khởi động preview tại port 4173 rồi đóng server, xuất report browser vào docs/phase1-browser-results.json. Đóng server khác dùng port 4173 trước khi chạy. Nếu chỉ xem production build, dùng npm run preview và mở http://127.0.0.1:4173.
