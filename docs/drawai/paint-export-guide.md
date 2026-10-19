# P6 — dùng gói model đã xuất

Artifact chính: Swin1D ảnh **ONNX** và DINOv3 **TorchScript FP32**. DINO ONNX có sẵn để chẩn đoán nhưng chưa đạt strict logit parity2/1.000 mẫu, không dùng để tích hợp. [Báo cáo](phase6-report.md) ghi đầy đủ kết quả và giới hạn.

## Cấu trúc và trạng thái

`models/exported/paint35-v1/` gồm registry.json, bundle-manifest.json và hai thư mục `swin1d-image/`, `dinov3-vits16/`. Mỗi thư mục có `release.json`/`model-card.md`/`labels.json`/`preprocessing.json`/`fixtures.json`/`report.json` cùng artifact. DINO có thêm license/provenance pretrained và ONNX diagnostic. `spec.runtime_format` + `spec.model_filename` là pointer chạy chính; không tự thay bằng file khác có cùng stem.

Các trường `opset`, `graph_nodes`, `initializer_elements` trong report mô tả graph ONNX đã xuất, kể cả DINO dùng runtime TorchScript. `parity` là kiểm tra artifact chính; key metric phân loại `onnx_metrics` giữ tên kỹ thuật cũ (argmax của cả hai format vẫn giống nhau). Xem `onnx_diagnostic.approved_runtime` để phân biệt numerical gate ONNX. Flags sản phẩm vẫn `release_ready=false`; P7 chưa tích hợp.

## Verify

Từ repository root, dùng `.venv-dinov3` đã kiểm chứng:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_export verify --expected-bundle-sha256 86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2
./.venv-dinov3/Scripts/python.exe tools/verify_phase6.py
```

Anchor gốc nằm ngoài package tại `artifacts/drawai/phase6-image64/trusted-bundle.json`; phải so với nguồn tin cậy ngoài package, không dùng self-hash trong package thay cho trust anchor. Hash không phải chữ ký tác giả. Verifier cuối cần evidence/logs và checkpoint/data local tương ứng; runtime server không cần dữ liệu training.

## Chạy backend local

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_runtime --expected-bundle-sha256 86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2 --port 4196
```

Địa chỉ `http://127.0.0.1:4196`, GET `/health`, POST `/infer`. Dependencies inference pin riêng tại `training/drawai/requirements-paint-inference.txt`: torch cho TorchScript, ONNX Runtime cho Swin; không cần timm/Transformers/pretrained snapshot. Giữ preprocessing code và metadata cùng version. Bản ONNX package verification không cần model factory training; runtime không đọc NPZ/cache/checkpoint P5.

Body chỉ có `model_id` (`swin1d-image` hoặc `dinov3-vits16`) và `png64_base64`. PNG phải grayscale modeL64×64, nền trắng/mực đen,≤16KiB; JSON body≤32KiB. Response có logits10 lớp0–9, predicted_id, model_id, artifact_format, checksum bundle và timing. Schema/dtype/shape/hash sai bị từ chối; không trả prediction giả. Không có CORS mở rộng hay logging request, server chỉ bind localhost; UI/proxy/public hosting thuộc P7.

Ví dụ dùng một ảnh train làm smoke input (không lấy test làm demo):

```python
import base64, json, urllib.request
from pathlib import Path
png = Path('data/private/locked/paint35-main-v1/development/images64/paint35_p006_d0_r01.png')
body = json.dumps(dict(model_id='dinov3-vits16', png64_base64=base64.b64encode(png.read_bytes()).decode())).encode()
req = urllib.request.Request('http://127.0.0.1:4196/infer', data=body, headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req) as response:
    print(json.load(response))
```

Khi load TorchScript trực tiếp, dùng `torch.jit.load(..., map_location='cpu').eval()`, `torch.inference_mode()` và `torch.jit.optimized_execution(False)` như đường đã kiểm chứng. Chuẩn bị RGB224 float32 đúng contract trước inference; không feed PNG bytes/feature cache trực tiếp vào graph. Swin ONNX nhận float32[1,64,64] (`row_pixels`), DINO TorchScript float32[1,3,224,224]; output float32[1,10]. Shape cố định batch1. Không áp crop/bbox normalization khác contract.

## Export lại / kiểm thử

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_export export --output models/exported/paint35-rebuild-v1 --report-dir artifacts/drawai/p6-rebuild
./.venv-dinov3/Scripts/python.exe -c "from training.drawai.native_runtime import configure_windows_dlls; configure_windows_dlls(); import unittest; unittest.main(module=None,argv=['unittest','discover','-s','training/tests','-p','test_drawai*.py','-v'])"
```

Export từ đúng P5 selection/lock, không ghi đè output hiện có; không train, đổi seed hoặc tune bằng test. Timestamp/serialization có thể khiến hash bundle rebuild khác; kiểm tra file/checkpoint/preprocessing/parity rồi đăng ký anchor mới. DLL helper chỉ thêm search directory trong process Windows; có thể chỉ định `DUST_DLL_DIRECTORY` khi dùng wheel native của môi trường hiện tại. Không tự sửa môi trường Swin gốc hay phát sinh token/network download.

Trong `web/`: `npm run build`, `npx playwright test --config playwright.drawai-p6.config.ts`, rồi `npx playwright test --config playwright.drawai-p6-closed.config.ts`. Cờ `DRAWAI_P6_LOCAL_SMOKE=1` chỉ mở allowlist model/fixtures local cho test; default404. Model Paint không được đưa vào public/dist. Backend benchmark: `python tools/verify_phase6_backend.py` (port4196 phải trống); script tự khởi động/tắt server, dùng ảnh validation và so budget P2.

## Migration và bàn giao

Model PenDigits cũ có canonical filename `drawai_swin1d_pendigits_l128_fp32_v1.onnx`; alias `model.onnx` giữ nguyên bytes. Release-loader/manifest/exporter/verifier/tests đã chuyển cùng nhau, không sửa design/weights cũ. Snapshot/audit trước P6 phải replay ở commit trước migration; kết quả đã ghi vẫn là lịch sử.

ONNX/TorchScript lớn bị Git ignore; cần giữ/copy cả thư mục bundle cùng trusted anchor riêng và runtime code để bàn giao. GitHub chưa chứa download weights mới. [Handoff P7](../../artifacts/drawai/phase6-image64/handoff.json) chỉ rõ format/path/hash thực. P7 mới nối canvas với input ảnh, kiểm tra domain gap và quyết định release.
