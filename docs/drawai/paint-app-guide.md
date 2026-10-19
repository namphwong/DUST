# Chạy DrawAI P7

P7 dùng DINOv3 TorchScript CPU mặc định và Swin ONNX CPU đối chứng. Không cần pretrained snapshot/timm/Transformers để suy luận. Cần **gói P6 đầy đủ**, không chỉ hai file primary: validator kiểm tra cả inventory/metadata và DINO ONNX diagnostic theo bundle đã khóa. Không đổi manifest để bỏ kiểm tra.

## Trên máy hiện tại

Từ `D:\ChuyenDe\DUST`, sau khi chạy `npm ci` trong `web`:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_drawai.ps1
```

Mở `http://127.0.0.1:4186/`. `/drawai/compare` chạy hai model từ cùng bản vẽ; `/drawai/legacy` giữ Swin PenDigits cũ. Ctrl+C dừng phiên. Script dùng `.venv-dinov3`, có thể truyền `-PythonPath C:\duong-dan\python.exe`; `-CheckOnly` kiểm tra startup rồi dừng backend. Cổng backend4197 cố định, frontend có thể đổi bằng `-Port`.

## Cài môi trường mới / chạy hai terminal

```powershell
py -3.11 -m venv .venv-dinov3
./.venv-dinov3/Scripts/python.exe -m pip install -r training/drawai/requirements-paint-inference.txt
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_app_runtime
```

Terminal thứ hai, trong `web`:

```powershell
npm ci
npm run dev -- --port 4186 --strictPort
```

Môi trường mới còn phải có bundle `models/exported/paint35-v1/` hợp lệ. Các weights lớn chưa được phân phối trên GitHub ở P7; xem [export guide](paint-export-guide.md) để tái tạo từ checkpoint local. Clean install/checkout trên máy khác chưa được nghiệm thu, thuộc P8. Windows DLL helper dùng đường dẫn process-local; nếu máy không có runtime DLL cần thiết, xem `native_runtime.py`/biến `DUST_DLL_DIRECTORY`, không sao chép DLL tùy tiện vào hệ thống.

## Hợp đồng suy luận

1. Canvas giữ viewport logic400×400 và độ rộng nét6. CSS, màu hiển thị và pointer không thay đổi.
2. Worker tạo ảnh xám64×64 bằng adapter `drawai-canvas400-gray64-v1`: nét đen đầu tròn/nền trắng, coverage4×, làm tròn half-up, không crop/center/bbox-fit, không nối qua lần nhấc bút. DPR/resize không đổi tọa độ logic. Đây là biểu diễn mới của canvas, không phải trajectory khôi phục từ NPZ Paint.
3. `POST /api/drawai/predict` qua Vite proxy cùng origin, chỉ chuyển tới `127.0.0.1:4197/predict`. Protocol `drawai-paint-api-v1`, gồm `request_id`, `drawing_revision`, `model_id`, `canvas_adapter`, `pixels64_base64` (4.096 byte uint8 row-major).
4. Backend chuyển lossless thành PNG L64 và gọi runtime P6: Swin `(255-image)/255` transpose → `[1,64,64]`; DINO gray→RGB, Pillow bilinear224, ImageNet normalize → `[1,3,224,224]`. Hai model nhận cùng pixels, không dùng nhãn demo làm input.
5. Backend xác minh bundle/hash trước khi load; worker kiểm tra model/format/bundle/pixel hash/revision/logits rồi tính softmax/top3. Client và UI chỉ nhận kết quả đúng phiên/revision. Backend không ghi ảnh/nét/request body.

Giới hạn: body8KiB, ảnh4.096 byte, không nhận unknown model/protocol/fields, duplicate JSON, ảnh trắng; mutex một inference tại một thời điểm. Proxy timeout15s, worker HTTP20s, client60s. Khi lỗi giữ bản vẽ để thử lại; không fallback ngầm. Gói artifact giữ `release_ready=false`, `technical_only=true` vì chưa nghiệm thu chất lượng canvas/người mới/cảm ứng thật.

## Kiểm tra và xử lý lỗi

- `GET /api/drawai/health`: trạng thái, default model, adapter, format và bundle hash; không phải phép đo accuracy.
- `BACKEND`: chạy backend trước, kiểm tra port4197, model bundle và log ở thư mục TEMP khi dùng launcher.
- `CONTRACT`: dữ liệu/identity/hash/phản hồi không khớp; không bỏ validator để chạy tiếp.
- `WORKER`: lỗi tải/chạy worker hoặc timeout; thử lại sẽ tạo worker mới.
- Model thiếu/corrupt: backend từ chối startup; tái tạo/khôi phục đúng bundle P6, không sửa checksum.

`npm run build` tạo static frontend nhưng không đóng gói Python/backend/weights mới. `vite preview` vẫn có proxy local; static hosting thông thường không có API này. Chưa triển khai production.

## Bằng chứng và giới hạn chất lượng

[Báo cáo P7](phase7-report.md), [API/parity](../../artifacts/drawai/phase7-image64/backend-verification.json), [browser benchmark](../../artifacts/drawai/phase7-image64/browser-benchmark.json), [quan sát demo](../../artifacts/drawai/phase7-image64/demo-observations.json).

DINO nhận đúng5/10 demo PenDigits train hiện có. Những nét này khác miền ảnh Paint; không dùng để báo accuracy tổng quát hoặc thay đổi model/adapter. Giữ nguyên quan sát và37 nhóm tương đồng qua split main. P7 chưa thu thêm ảnh/nét người thật, chưa đo quality canvas người mới và chưa thử thiết bị cảm ứng vật lý.
