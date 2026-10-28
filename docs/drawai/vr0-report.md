# VR0 — Khóa phạm vi bản public và điều kiện nghiệm thu

Ngày28/10/2026. **DONE: phạm vi/hợp đồng/checklist đã khóa và đối chiếu local.** Chưa deploy, chưa triển khai public build/API, chưa kiểm tra Render hoặc thiết bị cảm ứng thật. Mốc code trước VR0: `23c52f6`.

## 1. Bàn giao

| File | Vai trò |
|---|---|
| `deployment/vr0/public-contract.json` | Public deployment contractv1: model, input, profile, route, API, privacy, browser, budget và những claim chưa được phép |
| `deployment/vr0/trusted-contract.json` | Anchor SHA-256 độc lập để kiểm tra contract; là kiểm tra integrity, không phải chữ ký số |
| `deployment/vr0/acceptance-checklist.json` |19 mục:5 PASS thuộc VR0,14 NOT_RUN thuộc các phase public/đánh giá sản phẩm |
| `tools/verify_vr0.py` | Đối chiếu contract, P4/P5/P6, checkpoint thực, preprocessing/design và bằng chứng baseline |
| `training/tests/test_drawai_deployment_contract.py` |10 ca: integrity, đổi default, silent fallback, format sai, tolerance, route, RAM, claim cloud giả và path traversal |
| `artifacts/drawai/vr0/verification.json` | Kết quả chạy verifier và gate tests thực, kèm hash các file đã kiểm tra |

**Contract SHA-256:** `82142d9a61ee765ace27a7c15982ed8faf0ed2cd394d8727977e1f065c67e793`.

## 2. Model và nguồn đã khóa

- **DINOv3 seed42 là mặc định**, pretrained backbone + head10 lớp, primary TorchScript CPU trên Render; không chuyển sang ONNX diagnostic.
- **Swin1D ảnh seed43 là đối chứng**, primary ONNX; target ONNX Runtime Web/WASM trên thiết bị. Không dùng nhầm Swin PenDigits `[1,3,128]`.
- Giữ selection P5 bằng validation, không chọn lại theo test/demo. Đã hash lại hai checkpoint thật và kiểm tra toàn bộ inventory/hashes gói export P6.
- P4 dataset lock: `2c3a86c5be2b2e0f952a9c3390d56785473a82634338cec805903cbb1b3ef47a`.
- P5 selection: `827f063bb6fc814d94406204f2c45ac91668988c0273329a59c3f5aaeccbec51`.
- P6 bundle: `86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2`.
- DINO primary artifact: `5846bd2331f546d1a97e617a0733f7cafd72a953ee8988dd7538940ff2cc6baa`.
- Swin primary artifact: `9bb8bf5478cb2d8131936a79888505dcbc4f05e331027ca4e2117db7715df59d`.

Verifier đọc metadata dataset-lock và hash checkpoint/artifact; không đọc ảnh private test, không mở ZIP held-out, không train/evaluate lại. Không sao chép inventory ảnh riêng vào contract public.

## 3. Phạm vi web public đã chốt

**Kiến trúc:** Vercel frontend, Swin browser, Render DINO-only, model release có tag/hash cố định. Không fusion, không đổi model khi lỗi mà không thông báo.

**Public routes:** `/`, `/drawai`, `/drawai/compare`.

**Local/research only:** `/collect`, `/drawai/legacy`, `/drawai/check`, `/drawai/release-check`, `/drawai/integration-check`, `/drawai/dinov3-check`. VR0 chỉ phân tách profile trong đặc tả; không xóa các route đang có hoặc tuyên bố public build đã lọc route.

**Production API:** chốt `GET /health`, `POST /predict`; frontend gọi trực tiếp `VITE_API_BASE_URL` qua HTTPS. Đây là target VR2/VR3; API local `/api/drawai/*` hiện có không bị đổi trong VR0. Hostname production để trống, chỉ điền vào cấu hình triển khai sau khi có URL thực.

**Request:** protocol `drawai-paint-api-v1`, request/revision ID, `model_id`, adapter ID và4.096 uint8 gray pixels base64; body≤8KiB. Backend public chỉ nhận DINO model ID, một worker/một inference, busy429. Không queue không giới hạn, không log/lưu ảnh mặc định, không token trong frontend; CORS exact origins sẽ cấu hình ở VR4.

## 4. Design và preprocessing

Giữ layout, màu, font, canvas, undo/clear/top3; không thêm selector ở trang chính. Khóa hash CSS/base CSS/canvas, renderer TypeScript/Python và code preprocessing. Đã đối chiếu cặp screenshot empty desktop1366×768/mobile390×844 của mốc repo-review với bản legacy; đây là kiểm tra bằng chứng baseline, không phải chạy browser mới tại VR0.

Canvas logic400×400, nét6, coverage4× → gray64, không bbox crop/center, không nối qua lần nhấc bút. Swin invert/float32/transpose → `[1,64,64]`; DINO gray→RGB/Pillow bilinear224/ImageNet normalization → `[1,3,224,224]`.

Cho phép đổi câu trạng thái/cold start/privacy trong vị trí hiện có để nói đúng DINO gửi ảnh ra backend. Không được giữ câu “hoàn toàn chạy trên thiết bị” cho DINO cloud. Thay đổi chính sách/model/input phải có contract version mới và evidence, không sửa ngầm metadata P4–P6.

## 5. Gate chốt trước khi đo

| Gate tương lai | Ngân sách/điều kiện |
|---|---|
| Numerical parity Swin | `atol=1e-5`, `rtol=1e-4`, giữ tolerance P6 |
| Numerical parity DINO | `atol=1e-4`, `rtol=1e-3`, dùng TorchScript đã duyệt |
| Linux backend | Container cap512MiB, CPU quota0,1; cgroup peak≤450.000.000byte; không OOM, một process |
| Backend warm HTTP | p95≤5.000ms,5 warmup+30 lần đo, nearest-rank |
| Container cold-ready |≤90 giây; đo startup process mới dưới quota |
| Cloud wake-to-ready |≤120 giây mục tiêu; readiness chờ hữu hạn; inference timeout30 giây |
| Swin worker browser | p95≤200ms gồm raster/hash/tensor/WASM/softmax/postMessage;5 warmup+30 lần; loại download và thời gian người vẽ |
| Browser | Chrome/Chromium, Edge desktop; Chrome Android, Safari iOS mobile; WASM một thread baseline, không bắt buộc WebGPU |

Đây là **mục tiêu deployment mới, chưa có số đo cloud**; không thay budget local P2/P6 hoặc nới numerical tolerance cũ. Điều kiện512MiB của container không tự chứng minh cấu hình host đã đáp ứng; phải đối chiếu thực ở VR2/VR5 và để dư địa RAM.

Checklist tách nghiệm thu kỹ thuật public và nghiệm thu chất lượng sản phẩm. Public demo phải nói đúng các phần chưa chạy; không suy từ deploy thành công sang accuracy người dùng thật.

## 6. Kết quả VR0 và phần chưa chạy

- Verifier local: **PASS** cho contract/metadata/P4 lock/P5 selection, hai checkpoint thực, toàn bộ bundle P6 và preprocessing/design baseline.
- Gate tests mới: **10/10 PASS**, không skip. Test thay nội dung và tự tính lại hash vẫn bị chặn nếu vi phạm vai trò model, artifact, tolerance, route, RAM hoặc claim cloud.
- App/runtime/design/model không chỉnh trong VR0; chỉ thêm contract/verifier/tests/docs/evidence.
-133 Python+33 browser tests là kết quả **đợt repo-review trước**, có liên kết bằng chứng; không chạy lại toàn bộ ở VR0 và không gọi đó là kết quả mới.
- Model download public, Docker512MiB, API production, Swin web integration, Vercel/Render thật, cloud latency/memory, clean checkout và cảm ứng thật: **NOT_RUN**.
- Giữ các giới hạn: DINO đúng5/10 demo PenDigits,37 nhóm ảnh tương đồng qua split, nguồn theo xác nhận chủ dữ liệu; chưa có benchmark canvas người viết mới.

## 7. Chuyển tiếp VR1

Có thể bắt đầu VR1 để đóng gói model/tag/checksum và script download/verify. Chưa cần tài khoản/hostname thật để làm bước đó. Không tự tạo deployment cloud trong VR0.

Chạy lại gate từ root với model/checkpoint local hiện có:

```powershell
./.venv-dinov3/Scripts/python.exe -B tools/verify_vr0.py
./.venv-dinov3/Scripts/python.exe -B -m unittest training.tests.test_drawai_deployment_contract -v
```

`--write-evidence` chạy thêm10 gate tests và ghi lại bằng chứng VR0. Các phase sau giữ evidence VR0 làm mốc; bổ sung evidence của phase mới, không đổi các cờ NOT_RUN lịch sử thành PASS ngầm.
