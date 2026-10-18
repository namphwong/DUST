# DrawAI — Báo cáo P2 hoàn thành local

Ngày 18/10/2026, Asia/Bangkok. **P2 DONE_LOCAL**, xác nhận bằng `artifacts/drawai/phase2/verification.json`. Đã dùng pretrained DINOv3 thật, export toàn bộ backbone + head, kiểm chứng preprocessing/logits và chọn backend CPU theo benchmark. Design và model Swin đang phục vụ giữ nguyên.

## 1. Tìm nguồn có pretrained weights công khai

| Nguồn | Truy cập đã kiểm tra | Quyết định |
|---|---|---|
| [timm ViT-S/16 DINOv3 QKV-bias](https://huggingface.co/timm/vit_small_patch16_dinov3_qkvb.lvd1689m) | Anonymous HTTP200; gated=false; safetensors | **Chọn**: native PyTorch, giữ QKV bias, phù hợp fine-tune/export |
| [timm ViT-S/16 DINOv3](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m) | Anonymous HTTP200; gated=false; safetensors | Phương án khác; bỏ bias bằng0, không trộn checkpoint với adapter đã chọn |
| [ONNX Community DINOv3 ViT-S/16](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX) | Metadata/file công khai; có ONNX và external data | Tham khảo inference; không chọn để train backbone PyTorch |
| [Meta HF gốc](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | Gated401 khi chưa được cấp quyền | Giữ loader tùy chọn, không cần cho pipeline timm hiện tại |

Đây là DINOv3 pretrained/distilled trên LVD-1689M, không thay bằng DINOv2. Nguồn được chọn là bản phát hành công khai của maintainer timm; không tuyên bố đó là file Meta gated hoặc outputs trùng bit. Code timm open source; **weights chịu DINOv3 custom license**, không gọi là weights OSI. License tải kèm, SHA-256 `25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e`.

Snapshot đã tải tại `models/pretrained/dinov3-vits16-timm-qkvb/` (Git ignore):

- Revision: `2c7705788ac282557562465d6443606664a55f05`.
- Safetensors: 86.401.352 bytes; SHA-256 `71aedea5ea914657b8fd641beadbd97c012f87e3116d4d081eb143aa520ab53c`.
- Config Git blob: `719caee8b50053d542aebd3bcb04d515ce89172e`.
- Strict load tất cả 186 tensors, không missing/unexpected keys; từng tensor native bằng checkpoint nguồn. Hash/config sai bị từ chối trước khi khởi tạo model.
- API metadata, log tải và thông tin nguồn: `artifacts/drawai/phase2/source-search/`, `references.json`, local `drawai-provenance.json`.
- Downloader đã được chạy lại vào thư mục mới ngoài project, tải đủ4 files công khai và strict-load thành công; log `source-search/fresh-downloader.log`. Không chỉ kiểm tra nhánh dùng lại cache local.

## 2. Kiến trúc và phần thay backbone

```text
JSON nét bút → drawai-raster-v1 → float32 [B,3,224,224]
 → DINOv3 ViT-S/16 pretrained (frozen)
 → CLS384 + mean 196 patch tokens384 (loại 4 register tokens)
 → Linear(768,10) → logits [B,10]
```

Backbone **21.596.160** tham số; head **7.690**; tổng **21.603.850**. Ít hơn implementation HF gốc 384 tham số do timm không có mask token không sử dụng. Native timm sinh RoPE periods FP32; Meta gốc lưu BF16, nên có khác biệt số học đã được publisher công bố. Không chuyển đổi/lấp tensor thiếu để giả vờ dùng checkpoint gốc.

`training/drawai/dinov3_model.py` strict-load native timm rồi bọc pooling/head; `training/drawai/models.py` factory hai model. Backbone ở eval và không gradient khi train head. Native forward ra `[1,201,384]`, head ra `[1,10]`. Swin 1D giữ nguyên tại `models/SwinTransformer_cls.py`.

## 3. Renderer và normalization đã xác minh

`training/drawai/raster.py` và `web/src/drawai/raster.ts`: ảnh RGB224, trắng/đen, padding24, nét8px, đầu tròn, giữ tỷ lệ/y-down, không nối qua chỗ nhấc bút; không screenshot, font engine, resize/crop thêm. Pixel centers0..223; byte rounding floor(x+0.5). Float32 NCHW chia255, mean[0.485,0.456,0.406], std[0.229,0.224,0.225].

Mean/std đã kiểm chứng với **config timm đúng hash**. Publisher mặc định256px, hỗ trợ kích thước thay đổi từ128px; 224px là adaptation raster chữ số đã khóa trong contract, không tuyên bố áp nguyên crop/resize ảnh tự nhiên. Gated Meta HF processor không thuộc đường chạy native timm đã chọn.

19 fixtures gồm 10 train demo và geometry/multi-stroke/y-up/scale/translation; RGB bytes và tensor SHA-256 Python/TypeScript/backend khớp toàn bộ. 8 browser invalid fixtures bị từ chối. Contract: `docs/drawai/raster-contract.json`.

## 4. Gradient, tiny probe và ONNX thật

Forward/backward hữu hạn, head có gradient, backbone không gradient. Bản sao head tiny probe dùng đúng 10 train demo, feature conditioning từ chính train: initial loss2.4834342 → loss0.000047977, train10/10 sau1 bước. Đây là kiểm tra tối ưu trên mẫu đã thấy, không phải accuracy tổng quát; không mở validation/test/private, không sửa head trong export.

Artifact: `artifacts/drawai/phase2/technical/drawai_dinov3-vits16-timm-qkvb_pretrained-backbone_rgb224_fp32_p2.onnx`.

- Full backbone + head, opset17, fixed batch1 FP32; **86,634,209 bytes** (~82,62 MiB).
- SHA-256 `cf998559a0f9d06f102efaf41879d391369b0852883c37bf483060ef6802921e`.
- ONNX checker full-check đạt; ORT/PyTorch 19 fixtures không khác argmax, max abs error **1.78813934e-06**.
- Tolerance khóa trước: atol1e-4 + rtol1e-3. Python ORT4threads p95 **80.61 ms**.
- `pretrained_loaded=true`, `smoke_only=true`, `trained=false`, `release_ready=false`. Pretrained áp cho backbone; head phân loại chữ số trong artifact còn random. Huấn luyện/đánh giá đầy đủ thuộc P5, export release P6.

Artifact random trước đây được lưu ở backup ngoài project; active manifest chỉ trỏ bản pretrained với tên rõ ràng.

## 5. Runtime: browser vượt budget, backend local đạt

Budget browser giữ nguyên từ trước phép đo: model≤150MiB, cold≤15s, warm p95≤500ms, incremental sampled RSS≤768MiB. WASM CPU1thread, warmup5/30iterations, nearest-rank p95.

| Chỉ số | Browser WASM | Backend ONNX CPU local |
|---|---:|---:|
| Threads | 1 | 4 |
| Median | 607.7 ms inference | 83.6 ms end-to-end |
| p95 | **661.6 ms — FAIL500ms** | **90.2 ms — PASS500ms** |
| Cold local | 1241.4 ms fetch/hash/init | 503.3 ms process imports/load/init |
| Sampled RSS | incremental 401.26 MiB | process peak 241.93 MiB |
| Parity | 19/19 | 19/19 |

Backend budget được ghi trước benchmark backend tại `phase2-backend-budget.json`: end-to-end p95≤500ms, cold≤15s, process RSS≤1GiB. Đã đo qua browser worker → Vite local proxy → HTTP JSON → Python raster → chính graph ONNX có hash trên → response parse, không lấy riêng tốc độ model để gọi end-to-end. CPU backend vượt qua đủ budget. Lần WASM trước cũng vượt500ms (~616.5ms); bằng chứng thất bại được giữ trong source-search, không nới ngưỡng.

**Chốt runtime DINOv3: backend ONNX Runtime CPU local** để tích hợp tiếp ở P7. `dinov3_backend.py` chỉ bind127.0.0.1:4190; không lưu nét/request body. Schema sai, model hash cũ, input quá lớn bị từ chối; backend hoạt động lại sau input lỗi. Trang kỹ thuật ghi rõ nét được gửi tới backend local. Trang chính vẫn suy luận Swin trong browser.

Số đo trên Windows/AMD Ryzen7 H255, Chromium headless, localhost; không đại diện điện thoại/Internet/GPU. Thread count khác nhau nên không kết luận so sánh tốc độ công bằng. Browser RSS cộng process tree có thể đếm lặp shared pages; backend RSS sampling20ms không bảo đảm peak tuyệt đối. Cold chỉ một lượt, OS cache có thể còn. Deploy/consent/mô tả xử lý dữ liệu và đo lại artifact sau train thuộc P6/P7.

## 6. Kiểm chứng và bảo toàn DrawAI

| Hạng mục | Kết quả |
|---|---|
| TypeScript/Vite build | PASS |
| Python tests | **46/46 PASS** |
| Browser DINOv3 WASM/guards | **6/6 PASS**: ghi nhận budget fail đúng, parity đạt |
| Browser backend/parity/API errors/recovery | **3/3 PASS** |
| Browser product/root/alias | **12/12 PASS** |
| Pretrained hash/config/strict tensor load | PASS |
| Full ONNX parity + backend runtime budget | PASS |
| Thiết kế desktop/mobile root/alias | Pixel-identical P0 |
| Swin weights/checkpoint, code app/CSS, PenDigits | Hash giữ nguyên |
| Baseline Python P1 | Danh mục package giữ nguyên |
| Audit P2 | **DONE_LOCAL; phase_complete=true** |

46 tests gồm kiểm tra positive pretrained, frozen gradients/pooling và reject config tamper. 21 browser tests không có skipped/flaky. WASM test kiểm tra phép đánh giá budget phản ánh số đo; không khẳng định WASM đã đạt500ms. Không đưa DINOv3 ONNX vào public/dist; middleware/proxy cần cờ explicit, preview mặc định đóng. Fake trained/release metadata bị từ chối.

## 7. Code đã cập nhật và bàn giao

| File | Vai trò |
|---|---|
| `training/drawai/dinov3_model.py` | Native timm pretrained loader, source/hash/config guards, freeze, pooling/head |
| `training/drawai/dinov3_weights.py` | Public default downloader pin revision, local verify/idempotent, optional Meta loader |
| `training/drawai/dinov3_feasibility.py` | Real-weight gradient/tiny/export/parity/benchmark |
| `training/drawai/dinov3_backend.py` | Local CPU inference, JSON/schema/model hash guards, RSS/cold timing |
| `training/drawai/raster.py`, `web/src/drawai/raster.ts` | Deterministic preprocessing Python/TS |
| `web/src/drawai/dinov3-check.tsx`, `dinov3-worker.ts`, `dinov3-backend-worker.ts` | Trang riêng, WASM/backend 19-fixture parity và benchmark |
| `web/vite.config.ts` | Allowlist local files và proxy cố định127.0.0.1; không phục vụ mặc định |
| `training/tests/test_drawai_dinov3.py`, `web/tests/drawai-dinov3*.spec.ts` | Positive pretrained/guards/parity/API/budget |
| `tools/verify_phase2.py` | Audit nguồn pretrained, bytes/hash, số đo, suites, design/model/data |
| `README.md`, contract/status/guide/report/NOTICE | Chỉ ra thay đổi, nguồn weights và phần tiếp theo |

Môi trường `.venv-dinov3`: Python3.11.9/Torch2.5.1+cpu, timm1.0.22, torchvision0.20.1+cpu, Transformers4.56.2, ONNX1.16.2, ORT1.20.1. Dependencies nền P1 được tham chiếu chỉ đọc; chưa nghiệm thu clean-install trên máy khác. P2 không thu private data/full-train/deploy.

Bằng chứng: `artifacts/drawai/phase2/verification.json`, `technical/manifest.json`, `technical/python-report.json`, `browser-report.json`, `backend-report.json`, suite JSON/logs và `ui/`. Tái tạo theo [dinov3-guide.md](dinov3-guide.md). P3 tiếp theo: công cụ/protocol thu dữ liệu theo roadmap; không còn chờ quyền Meta để tiếp tục.
