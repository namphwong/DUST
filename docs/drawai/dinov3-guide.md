# DrawAI — DINOv3 P2: setup và tái tạo

Cập nhật 18/10/2026. P2 hoàn thành local với pretrained weights công khai từ [timm](https://huggingface.co/timm/vit_small_patch16_dinov3_qkvb.lvd1689m). Không cần tài khoản/token cho nguồn đã chọn. Head chữ số trong artifact P2 chưa train đầy đủ; release Swin trên trang chính giữ nguyên.

## 1. Môi trường

Máy hiện tại dùng `.venv-dinov3`: Python3.11.9, Torch2.5.1+cpu, torchvision0.20.1+cpu, timm1.0.22, Transformers4.56.2, ONNX1.16.2, ORT1.20.1. Package nền được tham chiếu chỉ đọc từ `.venv-phase1` qua `.pth`; P1 không bị cài đè. Đây không phải clean install đã nghiệm thu.

Tạo env riêng trên máy khác, từ root:

```powershell
py -3.11 -m venv .venv-dinov3
./.venv-dinov3/Scripts/python.exe -m pip install -r training/drawai/requirements-dinov3.txt
cd web
npm ci
npx playwright install chromium
cd ..
```

Chọn cặp Torch2.5.1/torchvision0.20.1 CPU hoặc CUDA phù hợp máy. Trên máy Windows hiện tại, exporter/backend tự thêm DLL directory Poppler bundled nếu có; có thể đặt `DUST_DLL_DIRECTORY` tới DLL directory hợp lệ nếu ONNX báo thiếu DLL. Không cần sửa env Swin để thử DINOv3. Chi tiết package thực tế tại `artifacts/drawai/phase2/environment.json`.

## 2. Tải public pretrained và export

Từ root:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.dinov3_weights
./.venv-dinov3/Scripts/python.exe -m training.drawai.dinov3_feasibility
```

Downloader mặc định chọn `--source timm`, pin revision `2c7705788ac282557562465d6443606664a55f05`, tải config/weights/card/license về `models/pretrained/dinov3-vits16-timm-qkvb/`. SHA-256 weights `71aedea5ea914657b8fd641beadbd97c012f87e3116d4d081eb143aa520ab53c`. File đã tồn tại thì strict verify và không tải lại. Không có random fallback khi thiếu/sai weights. Code timm open source; weights theo DINOv3 custom license, xem LICENSE.md tải kèm.

Feasibility mặc định dùng snapshot timm local, chạy gradient/freeze, tiny probe trên bản sao head, full ONNX export/checker, 19-fixture PyTorch/ORT parity và benchmark. Output:

```text
artifacts/drawai/phase2/technical/
  drawai_dinov3-vits16-timm-qkvb_pretrained-backbone_rgb224_fp32_p2.onnx
  manifest.json
  python-report.json
  preprocessing.json
  fixtures.json
  tiny-overfit.json
  train-digits-rendered.png
```

ONNX chứa toàn bộ backbone pretrained + head ngẫu nhiên, không chỉ head. `pretrained_loaded=true`, `trained=false`, `release_ready=false`; tên `pretrained-backbone` để phân biệt với model chữ số sẽ train ở P5. Tiny10/10 không phải metric test và không sửa head export.

Nếu chủ động thử kiến trúc random, dùng `--architecture-smoke --output-dir artifacts/drawai/phase2/random-experiment`; không ghi đè evidence pretrained ở `technical/`. Loader Meta gốc vẫn có qua `--source meta` với quyền/xác thực HF local; không dùng token trong source/chat. Đường chạy đó không bắt buộc và chưa được nghiệm thu positive load bằng snapshot gated.

## 3. Kiểm chứng backend CPU local

WASM hiện vượt p95≤500ms, nên P2 chốt backend ONNX Runtime CPU4threads. Mở terminal tại root:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.dinov3_backend
```

Backend chỉ bind `127.0.0.1:4190`; `/health` trả provenance/metadata, POST `/infer` nhận `{strokes,y_axis,model_sha256}`. Schema/model hash sai bị400, body>1MiB bị413, busy429; không lưu nét/request body. Đây là server kỹ thuật local; deploy/auth/quyền riêng tư và tích hợp release thuộc P7.

Terminal thứ hai:

```powershell
cd web
npm run build
$env:DRAWAI_P2_LOCAL_SMOKE='1'
npm run preview -- --port 4188 --strictPort
```

Mở `http://127.0.0.1:4188/drawai/dinov3-check`, chọn **Kiểm chứng backend local**. Trang ghi rõ dữ liệu gửi tới backend127.0.0.1. Nút **Chạy kiểm chứng DINOv3** thử riêng WASM; report latency fail500ms là kết quả đã ghi nhận. Trang chính `/` và `/drawai` giữ Swin/browser/design cũ.

ONNX nằm ngoài public/dist. Không bật cờ thì file local và backend proxy đều404. Kết thúc preview, tắt cờ bằng `$env:DRAWAI_P2_LOCAL_SMOKE='0'` trong terminal đó.

## 4. Tái tạo toàn bộ kiểm thử

Từ root chạy Python tests; cần có snapshot pretrained thật để chạy positive load:

```powershell
./.venv-dinov3/Scripts/python.exe -m unittest discover -s training/tests -p "test_drawai*.py" -v
```

Để lưu đúng evidence audit, PowerShell redirect `*> artifacts/drawai/phase2/python-tests.log`. Từ web chạy tuần tự, dừng preview/backend thủ công trước vì configs tự khởi tạo server:

```powershell
npm run build
$env:DRAWAI_P2_LOCAL_SMOKE='0'
npx playwright test --config playwright.drawai-p2-regression.config.ts
npx playwright test --config playwright.drawai-dinov3.config.ts
npx playwright test --config playwright.drawai-dinov3-backend.config.ts
```

Các suite ghi JSON reports tự động:12 product +6 WASM/guards +3 backend. Hai config DINOv3 tự bật cờ trong process của chúng. Test WASM kiểm tra parity và tính đúng của kết luận budget, không bắt ép kết quả benchmark phải pass. Threshold500ms giữ nguyên; auditor yêu cầu backend thật đạt budget.

Sau suite product, từ root lưu ảnh và audit:

```powershell
New-Item -ItemType Directory -Force artifacts/drawai/phase2/ui | Out-Null
Copy-Item artifacts/drawai/ui/desktop.png,artifacts/drawai/ui/mobile.png artifacts/drawai/phase2/ui
Copy-Item artifacts/drawai/phase1/ui/root-desktop.png,artifacts/drawai/phase1/ui/root-mobile.png artifacts/drawai/phase2/ui
./.venv-dinov3/Scripts/python.exe tools/verify_phase2.py
```

Audit đối chiếu hash/config/license pretrained, full graph, metadata,19 fixtures,46 Python/21 browser tests, fixed budgets, design với P0, model/data/code Swin và package baseline. Kết quả hiện tại **DONE_LOCAL**, `phase_complete=true`. Nếu rerun chỉ một phần thì không sử dụng reports cũ để tuyên bố đã chạy lại toàn bộ.

## 5. Các file thay đổi

- `training/drawai/dinov3_model.py`: vị trí gắn backbone DINOv3 native timm, strict-load/hash/config, freeze, pooling/head.
- `dinov3_weights.py`: downloader pin public source; `dinov3_feasibility.py`: full export/parity/benchmark; `dinov3_backend.py`: server CPU.
- `training/drawai/raster.py`, `web/src/drawai/raster.ts`: preprocessing cùng JSON, cùng hash/tensor.
- `web/src/drawai/dinov3-check.tsx`, `dinov3-worker.ts`, `dinov3-backend-worker.ts`: trang kỹ thuật/parity/runtime.
- `web/vite.config.ts`: cờ/allowlist/proxy local; `tools/verify_phase2.py`: auditor; `tools/phase2_documentation.py`: cập nhật báo cáo từ evidence đã audit.

[Báo cáo chi tiết](phase2-report.md) có bảng nguồn thay thế, khác biệt timm/Meta, số đo và toàn bộ bằng chứng. P3 tiếp theo là protocol/công cụ thu dữ liệu; P5 train hai model trên dataset đã khóa, P6 xuất model cuối cùng.
