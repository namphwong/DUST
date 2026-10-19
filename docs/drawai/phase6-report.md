# P6 — xuất và kiểm chứng artifact Paint35

Ngày 19/10/2026. **DONE local trong phạm vi hai artifact chạy đã kiểm chứng: Swin ONNX và DINOv3 TorchScript. Mục tiêu hai ONNX đều đạt strict parity chưa đạt: DINO ONNX chỉ là chẩn đoán.** Giữ nguyên ngưỡng P2, không train/chọn lại model từ test. [Bằng chứng](../../artifacts/drawai/phase6-image64/verification.json) · [Hướng dẫn](paint-export-guide.md) · [Handoff P7](../../artifacts/drawai/phase6-image64/handoff.json).

## Hai artifact chính

| Họ | Seed P5 | Format chạy | Dung lượng | Max abs error so PyTorch | Test accuracy |
|---|---|---|---|---|---|
| swin1d-image | 43 | onnx | 2.28 MiB | 8.35955143e-06 | 84.40% |
| dinov3-vits16 | 42 | torchscript | 82.88 MiB | 0 | 98.80% |

Cả hai dùng đúng checkpoint đã chọn theo validation ở P5, cùng1.000 ảnh test,0 khác argmax và0 giảm accuracy/macro-F1. Swin macro-F1 **84,28%**, DINO **98,80%**. DINO artifact chứa đủ **21.603.850 tham số**, backbone+head; state tensors đối chiếu với checkpoint P5. TorchScript được load lại từ file, chạy CPU với `optimized_execution(False)` để giữ đường tính đã kiểm chứng.

Thư mục local: `models/exported/paint35-v1/`. Mỗi họ có model, labels.json, preprocessing.json,9 fixture ảnh sinh bằng code, report.json, model-card.md, release.json. DINO kèm license/provenance pretrained. Không có ảnh private/test trong gói; full-test predictions ở thư mục evidence bị Git ignore.

- `drawai_dinov3-vits16_paint35-main-v1_rgb224_fp32_v1.torchscript.pt`
  SHA256: `5846bd2331f546d1a97e617a0733f7cafd72a953ee8988dd7538940ff2cc6baa`
- `drawai_swin1d-image_paint35-main-v1_rows64_fp32_v1.onnx`
  SHA256: `9bb8bf5478cb2d8131936a79888505dcbc4f05e331027ca4e2117db7715df59d`

Anchor riêng ngoài package, tại `artifacts/drawai/phase6-image64/trusted-bundle.json`:

```text
86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2
```

## DINO ONNX: đã xuất full graph, chưa duyệt chạy

`drawai_dinov3-vits16_paint35-main-v1_rgb224_fp32_v1.onnx` chứa full backbone+head FP32/opset17, qua ONNX checker/shape/finite output và giữ100% argmax so checkpoint. Tuy nhiên **2/1.000 mẫu vượt atol=1e-4,rtol=1e-3**, max abs error **0,000542641**. Tắt graph optimizations vẫn chưa giải quyết; các probe RoPE/LayerNorm không được đưa vào model bàn giao. Không nới tolerance, không sửa head/weights hay dùng test để retrain. Bản ONNX này nằm trong package để chẩn đoán, `approved_runtime=false`.

Artifact chạy được chọn là TorchScript FP32, **max error0** trên1.000 test trong lần kiểm tra này;0 khác argmax, logits P5 được tái hiện. Không gọi DINO ONNX là model đã đạt parity hay quảng bá DINO chạy browser. Đây là thay đổi định dạng runtime theo phương án backend P6, không phải thay kiến trúc hay chọn seed lại.

## Runtime thực và budget

- **Swin mới trong Chromium/WASM:**9 fixture nguồn sinh bằng code đều đạt, hash/shape đúng; init **604.80 ms**, inference-only p95 **2.10 ms** (5 warmup,30 lần, một tensor noise thủ tục). Chưa phải UX/canvas end-to-end của model mới.
- **Backend localhost:** Swin ONNX + DINO TorchScript,4 threads CPU; JSON HTTP+PNG64 decode+preprocessing+full inference+parse response. Dùng10 ảnh validation phủ0–9,5 warmup+30 lần; p95 nearest-rank: Swin **16.65 ms**, DINO **133.03 ms**.
- Cold process **2195.46 ms**; RSS cuối **381.13 MiB**, peak process **382.02 MiB**. Đạt budget P2 backend: cold≤15s,p95≤500ms,RSS≤1GiB. Máy Ryzen7 H255/CPU đã dùng ở P5; không có phép đo điện thoại/GPU/public hosting.
- Backend cần torch để load TorchScript DINO; **không cần timm/Transformers, checkpoint training hay snapshot pretrained** khi chạy. Chỉ nhận PNG L64, không nhận JSON nét bút cũ. Bind127.0.0.1, không lưu request; kiểm tra hash trước load, body/schema/base64/shape/finite output, origin/size/duplicate keys, lỗi và phục hồi.

## Swin PenDigits cũ: đổi tên, giữ nguyên sản phẩm

Tên canonical: `web/public/models/drawai-swin-v1/drawai_swin1d_pendigits_l128_fp32_v1.onnx`. Giữ `model.onnx` làm alias byte-identical cho link/evidence lịch sử. Loader chính và worker kiểm chứng dùng tên canonical; manifest, exporter/verifier Python và tests liên quan cùng cập nhật. SHA của cả hai vẫn là:

```text
28100de69dec82303a954af5c18936788e7c3dee623d70ec7a89ad9f7ef02366
```

Model/weights/preprocessing/release version đang phục vụ DrawAI không đổi; chỉ thay URL canonical. Audit ghi rõ4 file bảo vệ đổi do migration và54 file còn lại không đổi. Bốn screenshot `/` và `/drawai` desktop/mobile giống pixel P0. Verifier D4 cũng đã chạy lại thành công; bản browser-report cập nhật gắn manifest tên mới. Audit pha cũ dùng snapshot/commit tương ứng nếu muốn replay baseline trước migration.

## Code và nghiệm thu

- `paint_export.py`: đọc selection P5, strict-load checkpoint, ONNX full graph, TorchScript full DINO, đối chiếu toàn test+9 fixtures; promote staging sau verify; không ghi đè output cũ.
- `paint_package.py`: registry/filename/shape/format, external anchor, exact inventory/hash/metadata; không tự lấy self hash để gọi là trusted.
- `paint_runtime.py`, `native_runtime.py`, `requirements-paint-inference.txt`: backend CPU và DLL search trong tiến trình Windows, không sửa môi trường Python gốc.
- `legacy_release.py`, exporter/verifier/integration cũ, release-loader/types/feasibility-worker/tests: tên rõ cho graph cũ, giữ alias tương thích.
- `web/vite.config.ts`: gate kiểm chứng local P6 mặc định đóng; không copy model mới vào public/dist và không thêm luồng UI. Gate allowlist chỉ phục vụ Swin mới khi bật cờ kiểm chứng.
- **128 Python tests +36 browser tests đạt, TypeScript/Vite build và final audit đạt.**35 tests gồm product/release/feasibility/integration/new WASM;1 test xác nhận gate mặc định404. Backend có18 fixture parity qua HTTP, kiểm tra lỗi và budget. Logits PyTorch/artifact/ONNX diagnostic lưu riêng để truy nguyên.

## Giới hạn và chuyển P7

`trained=true`, `artifact_verified=true`, **`release_ready=false` cho sản phẩm**. Hai model mới chưa thay model trên trang chính. P7 cần đưa canvas về PNG64 đúng contract, parity preprocessing, registry/worker/API, UX/error/retry, domain-gap và thiết bị thật mà giữ design. Nguồn Paint vẫn có37 near-groups qua split main và provenance dựa trên xác nhận chủ dữ liệu; metric chưa chứng minh benchmark sạch rò rỉ hay tổng quát hóa hoàn toàn. NPZ gốc đã public, test isolation theo quy trình.

Model lớn ONNX/TorchScript ở local, GitHub chứa code/docs/metadata/evidence. Chưa tạo release tải weights trên GitHub/Kaggle, chưa public deploy hoặc tuyên bố clean-install toàn bộ pipeline.
