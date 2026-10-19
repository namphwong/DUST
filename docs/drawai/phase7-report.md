# P7 — Tích hợp hai model vào DrawAI

Ngày19/10/2026. **DONE local phần kỹ thuật; nghiệm thu sản phẩm còn chờ.** Trang chính đã dùng DINOv3, có trang đối chứng Swin, giữ thiết kế. Không đánh dấu đã đạt chất lượng canvas người mới hoặc thử thiết bị cảm ứng thật. `release_ready=false` được giữ, không sửa metadata P6 để che phần chưa nghiệm thu.

## 1. Đã thực hiện

| Phần | Kết quả |
|---|---|
| Default | DINOv3 ViT-S/16 seed42, head chọn bằng validation P5 trước final test; P7 áp dụng đúng checkpoint đã chọn, không chọn lại từ test/demo |
| Đối chứng | Swin Transformer1D ảnh seed43 qua ONNX CPU; không dùng nhầm checkpoint PenDigits [3,128] |
| Trang chính | `/`, `/drawai`, `/drawai/`; không thêm bộ chọn model |
| So sánh | `/drawai/compare`, hai inference tuần tự từ cùng snapshot/ảnh; hiển thị dự đoán, top3, thời gian và SHA ảnh |
| Legacy | `/drawai/legacy` giữ Swin PenDigits WASM; các trang release/integration lịch sử tiếp tục dùng artifact cũ |
| Design | CSS `drawai.css` và component `drawing-canvas.tsx` khớp SHA mốc P0; ảnh trạng thái empty desktop/mobile khớp pixel giữa default mới và legacy |
| UX | Giữ vẽ/undo/clear,10 demo, top3, nhãn người dùng/export JSON, retry; bỏ phản hồi cũ sau khi revision thay đổi |
| Lỗi | Worker lỗi → reset/retry; backend thiếu → giữ nét/báo lỗi; response sai hash/model/revision/pixel/logits → không hiển thị dự đoán |
| Startup | Bundle/artifact thiếu hoặc corrupt → từ chối load; launcher PowerShell có kiểm tra readiness và cleanup |

Chỉ thay nội dung phần “Về DrawAI” khi mở details để nói đúng model/backend/nguồn mẫu; link kiểm chứng trỏ trang so sánh mới. Bố cục, màu, typography, độ rộng/màu nét hiển thị, nhãn/nút và canvas giữ nguyên. Không khẳng định screenshot trạng thái kết quả mới giống P0 vì model/dự đoán đã đổi.

## 2. Pipeline và phiên bản

```text
DrawingCanvas400×400 (design/nét6px giữ nguyên)
 → paint-worker → adapter drawai-canvas400-gray64-v1
 → 4096 uint8 grayscale pixels + SHA-256
 → same-origin /api/drawai/predict → loopback Python4197
 → runtime P6 xác minh bundle/artifact
     Swin: invert/255 transpose → [1,64,64] → ONNX CPU
     DINO: gray→RGB, bilinear224, ImageNet norm → [1,3,224,224] → TorchScript CPU
 → logits10 + model/format/hash/request/revision/pixel hash
 → worker kiểm tra + stable softmax → client kiểm tra → top3 UI
```

Adapter mới cố định trước khi đo demo: coverage4×, nét đen đầu tròn6 logicalpx trên nền trắng, làm tròn half-up; giữ nguyên viewport, không crop/center/normalize bbox, không nối nét qua pen-up. Pixel không phụ thuộc DPR/CSS canvas/browser rasterizer. Đây là lựa chọn kỹ thuật để biểu diễn canvas thành ảnh, **không phải hợp đồng trajectory của Paint**, không chứng minh canvas và Paint cùng phân phối.

NPZ không có tọa độ/thứ tự nét/pen-up; P7 không tạo trajectory giả từ ảnh. Python reference trong `paint_canvas.py` chỉ kiểm tra renderer bằng fixture procedural, không dùng để tuyên bố người dùng đã viết các nét đó.

Bundle vẫn khóa SHA `86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2`; P4 split/P5 selection/P6 runtime và model binaries không đổi. DINO ONNX vẫn diagnostic, không được chọn lại làm runtime. Backend cần torch cho TorchScript, không import timm/Transformers hoặc đọc snapshot pretrained khi chạy. Gói P6 đầy đủ vẫn cần để qua inventory validator.

API `drawai-paint-api-v1`: request ID/revision, model ID, adapter ID và4096 pixels base64. Hạn mức body8KiB, đúng schema, không duplicate JSON/nonfinite/blank/unknown model. Backend chỉ bind127.0.0.1, proxy cùng origin; không mở CORS permissive. Không ghi request body/ảnh/nét vào log hoặc dataset. Backend mutex một inference; client có timeout và retry, không fallback ngầm. Các flags `technical_only=true`, `release_ready=false` giữ nguyên ý nghĩa kỹ thuật local.

## 3. Kiểm chứng đã chạy

| Kiểm tra | Kết quả thực tế |
|---|---|
| Python toàn bộ DrawAI | **133/133 đạt**; gồm5 test P7 về ảnh lossless, sai contract, roundcap/pen-up, model missing/corrupt |
| Build | `tsc --noEmit && vite build` đạt |
| P7 + route browser | **13/13 đạt**, Chromium153.0.8010.12 trên WindowsCPU |
| Legacy browser hồi quy | **9/9 đạt** trên route legacy;10 demo logits khớp fixture release cũ |
| Python/TypeScript renderer |5 fixture procedural ×2 model = **10/10 pixel SHA khớp** |
| API/preprocessing/model parity |8 fixture procedural không rỗng P6 ×2 model = **16/16 đạt**; white-blank fixture được kiểm tra bằng nhánh reject riêng |
| Sai số API so logits P6 | Swin maxabs1,90735e-6; DINO0,0; dùng tolerance P6 đã khóa, không nới |
| Invalid API |8 probe sai schema/ID/adapter/base64/blank/size/duplicateJSON, cộng origin rejection và recovery đạt |
| Identity corruption | Sai bundle/model/revision/pixel hash/nonfinite logits bị chặn |
| Trải nghiệm | Undo/clear, export nhãn độc lập, late response, retry, worker crash đạt |
| Giao diện | Empty1366×768 và390×844 khớp pixel legacy; không tràn ngang; touch events/DPR2 giả lập đạt |
| Launcher | `tools/run_drawai.ps1 -CheckOnly` startup/hash/readiness/cleanup đạt trên môi trường local hiện tại |
| Final audit | `tools/verify_phase7.py` đạt; anchors, dữ liệu bằng chứng và design sources được kiểm tra |

Tổng **133 Python +22 browser tests**. Những test mang tên Phase1/D5 được chuyển rõ sang route legacy khi kiểm tra contract PenDigits; alias/root vẫn có kiểm tra riêng. Không lấy việc legacy pass thay cho test DINO mặc định. Bằng chứng mới nằm ở `artifacts/drawai/phase7-image64/`, không ghi đè báo cáo/ảnh các phase trước.

## 4. Runtime đo thực tế

Máy local Windows, AMD Ryzen7 H255, CPU4 threads/model, CUDA không dùng. Chromium headless153.0.8010.12; Python3.11.9, Torch2.5.1+cpu, ORT1.20.1. Cả Swin và DINO chạy backend CPU trong phép đo P7; không so p95 này trực tiếp với inference-only WASM P6.

| Phép đo | Kết quả | Budget local P2 |
|---|---:|---:|
| Swin browser worker roundtrip p95 | **8,10ms** | ≤500ms |
| DINO browser worker roundtrip p95 | **130,50ms** | ≤500ms |
| Cold fresh-process → health ready | **2.645,48ms** | ≤15.000ms |
| RSS backend sau browser benchmark | **421.761.024byte (~402,22MiB)** | ≤1GiB |

Mỗi model5 warmup +30 đo, p95 nearest-rank; nguồn là một đường chéo procedural cố định. Timer browser bao gồm postMessage, raster/hash, HTTP qua proxy, preprocessing, inference, JSON validation/softmax; loại thời gian người vẽ và UI paint. Cold đo riêng một process mới; xác minh PID/ancestry để tránh đo nhầm backend đang chạy. Số liệu không đại diện điện thoại/thiết bị cảm ứng thật/production hoặc accuracy. [Benchmark](../../artifacts/drawai/phase7-image64/browser-benchmark.json) · [Backend](../../artifacts/drawai/phase7-image64/backend-verification.json).

## 5. Khoảng cách chất lượng đã quan sát

Giữ nguyên10 demo PenDigits train, không thay bằng test Paint hoặc tạo stroke giả. DINO nhận đúng **5/10** theo adapter cố định; đây là bộ demo ít mẫu khác miền, **không phải accuracy benchmark**.

| Nhãn demo | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| DINO dự đoán | 0 | 7 | 2 | 2 | 9 | 7 | 6 | 7 | 8 | 0 |

Điều này cho thấy không thể suy từ accuracy98,80% trên ảnh Paint sang chất lượng nét canvas. Không sửa adapter/demo/checkpoint sau khi xem kết quả; không mở lại test, không train/tune thêm ở P7. Giới hạn37 nhóm ảnh tương đồng qua split main và provenance theo xác nhận chủ dữ liệu vẫn giữ nguyên. Nhãn demo/người dùng độc lập với prediction và không dùng làm input model.

## 6. File cập nhật và bàn giao

- Frontend mới: `paint-registry.ts`, `paint-raster.ts`, `paint-worker.ts`, `paint-client.ts`, `paint-compare.tsx`; sửa tối thiểu `drawai-app.tsx`/`main.tsx` để đổi inference/default/routes.
- Backend: `paint_app_runtime.py` bọc runtime P6 không đổi; `paint_canvas.py` reference parity; Vite proxy chỉ loopback; `tools/run_drawai.ps1` chạy local.
- Tests/audit: `test_drawai_paint_app.py`, `drawai-p7.spec.ts`, config P7/default scripts; `prepare_phase7_fixtures.py`, `verify_phase7_backend.py`, `verify_phase7.py`. Legacy tests ghi bằng chứng vào thư mục P7 riêng.
- Docs: README đã sửa thông tin model/training/runtime cũ, bảng vị trí update; contractv8, status, roadmap, [hướng dẫn chạy](paint-app-guide.md).
- Bằng chứng: [verification](../../artifacts/drawai/phase7-image64/verification.json), [handoff](../../artifacts/drawai/phase7-image64/handoff.json), screenshots, raster parity, API parity, quan sát demo, benchmark và test reports.

## 7. Phần còn chờ

1. Nghiệm thu cảm ứng/bút trên thiết bị vật lý và dữ liệu canvas người viết mới độc lập; browser giả lập không thay thế phần này. Chưa có dữ liệu để báo accuracy canvas thực.
2. Nếu cần cải thiện canvas, lập protocol/validation mới, giữ evaluation độc lập và version adapter/model; không tuning trên private test P5 hoặc10 demo đang báo.
3. P8: phân phối bundle lớn kèm license/checksum, thử clean checkout/máy khác, hoàn thiện dataset card/license/link và gói nộp. Static hosting/production chưa triển khai.

P7 đã bàn giao luồng tích hợp chạy được local cùng bằng chứng và giới hạn; chưa bàn giao một sản phẩm đã được nghiệm thu chất lượng thực địa.
