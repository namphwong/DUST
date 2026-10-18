# DrawAI — Môi trường và lệnh hiện có

Cập nhật P1 17/10/2026: `/` và `/drawai` đều mở DrawAI, chỉ giữ tests DrawAI. Pipeline train/notebook hiện hành là Swin-only; DINOv3 thuộc P2. Xem [training guide](training-guide.md) và [báo cáo P1](phase1-report.md). Các số liệu D0–D6 dưới đây là lịch sử; môi trường local hiện không có pip, danh mục package ở artifacts/drawai/phase0/python-packages.json.

Đã kiểm tra ngày 11/10/2026 trên Windows, Python 3.11.9, Node 24.18.0, npm 11.16.0. Python riêng hiện dùng `.venv-phase1`; không thay Python toàn cục. Requirements DrawAI nằm ở `training/drawai/requirements.txt`: NumPy 1.26.4, unlzw3 0.2.3, Matplotlib 3.9.4, Torch 2.5.1 (local CPU build), ONNX 1.16.2, ORT 1.20.1. Web dùng dependencies và lockfile sẵn có.

Rà soát D0 đã đọc phiên bản từ interpreter và package metadata cục bộ: Torch thực cài `2.5.1+cpu`, React/React DOM 18.3.1, TypeScript 5.7.3, Vite 6.4.3, ORT Web 1.20.1, Playwright 1.63.0. Kiểm kê metadata không thay thế phép import/export runtime; report chạy D2 đã có ở `artifacts/drawai/feasibility/`, không chạy lại trong đợt nghiệm thu D0 này.

## Bootstrap môi trường mới

Từ root repo trong PowerShell, có thể dùng venv riêng cho DrawAI:

```powershell
py -3.11 -m venv .venv-drawai
./.venv-drawai/Scripts/python.exe -m pip install --upgrade pip
# Chọn CPU cho local; trên GPU/Kaggle chọn wheel phù hợp thay vì lệnh CPU này.
./.venv-drawai/Scripts/python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
./.venv-drawai/Scripts/python.exe -m pip install -r training/drawai/requirements.txt
Set-Location web
npm ci
```

Nếu dùng `.venv-drawai`, thay interpreter trong các lệnh bên dưới. Không ghi venv vào Git; requirements riêng DrawAI pin direct dependencies, chưa phải lockfile toàn bộ phụ thuộc Python. `web/package-lock.json` khóa dependencies web. Các lệnh bootstrap là hướng dẫn cho môi trường mới, chưa được chạy lại bằng clean install trong đợt D0 này.

Schema D0 đã kiểm tra bằng thư viện `jsonschema` có trong môi trường hiện tại; không thêm thư viện này vào dependencies inference/training chỉ để phục vụ kiểm tra tài liệu. Cần `jsonschema` nếu tái kiểm tra schema bằng Python ở máy mới; D5 triển khai validator ứng dụng riêng theo hợp đồng.

Tất cả lệnh Python chạy từ root repo; ví dụ PowerShell:

```powershell
# Dataset đã tải; --download khi cần tải lại từ nguồn.
./.venv-phase1/Scripts/python.exe -m training.drawai.prepare_data --length 64
./.venv-phase1/Scripts/python.exe -m training.drawai.prepare_data --length 128
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_data
./.venv-phase1/Scripts/python.exe -m unittest training.tests.test_drawai_data -v
./.venv-phase1/Scripts/python.exe -m unittest training.tests.test_drawai_model -v

# Tùy chọn chỉ cho Windows trên máy hiện tại khi ONNX lỗi VC runtime DLL:
$env:DUST_DLL_DIRECTORY = 'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_feasibility

Set-Location web
npm run build
npx playwright test --config playwright.drawai.config.ts
npm run dev -- --port 4183 --strictPort
```

Trang kiểm chứng: `http://127.0.0.1:4183/drawai/check`. Playwright dùng preview port 4184, không dùng hoặc tắt server MotionSense 4173. Trước chạy test phải build và tạo artifact D2.

## Máy khác / Kaggle

D6 local: `python -m training.drawai.prepare_integration` từ root (biến DLL như D4 trên máy này), build web rồi `npx playwright test --config playwright.drawai-integration.config.ts` từ web. Config chạy channels Chrome/Edge đã cài, port 4186. Kiểm toán bằng `python -m training.drawai.verify_integration`; xem [nghiệm thu D6](acceptance-report.md). `/drawai/integration-check` là trang kỹ thuật parity/benchmark, không dùng để thu người thật.

Ứng dụng D5 local: `http://127.0.0.1:4183/drawai`. Chạy `npm run build` rồi `npx playwright test --config playwright.drawai-app.config.ts` từ thư mục web (tests port 4186); từ root chạy `python -m training.drawai.verify_ui` với dependencies verification. Xem [báo cáo D5](ui-report.md).

Tạo venv riêng Python 3.11, cài requirements DrawAI; chọn Torch CPU/GPU đúng máy. Node cài dependencies web bằng lockfile (`npm ci`). Trên Linux/Kaggle không dùng biến DLL Windows. Runtime directory nói trên là đường dẫn cục bộ đã xác minh; máy khác cần Visual C++ runtime tương thích hoặc cung cấp runtime directory riêng.

Giải pháp DLL chỉ thêm search directory qua `os.add_dll_directory` trong tiến trình export, không thay DLL hệ thống. Đã import và export thật thành công. Export hiện fixed shape; tracer warnings về shape trong backbone đã được kiểm chứng trên cả L64/L128, không tuyên bố hỗ trợ dynamic length.

D3 đã có training pipeline, notebook Kaggle (CPU smoke đã chạy) và checkpoint PyTorch đã train local. Xem [hướng dẫn D3](training-guide.md) và [kết quả](experiment-report.md); dependencies notebook riêng tại `training/drawai/requirements-notebook.txt`. Kaggle GPU chưa chạy. D4 đã export ONNX trained và kiểm chứng CPU/Chromium: [báo cáo/lệnh D4](artifact-report.md), trang `/drawai/release-check` trên cùng server 4183; Playwright D4 dùng port 4185. Artifact D2 là `trained=false`; thay residual post-norm weights thành 1 chỉ để kiểm tra attention/MLP khi export, không áp dụng thay đổi này cho training mặc định.
