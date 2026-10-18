# DrawAI

Nhận dạng **một chữ số viết tay 0–9 từ nét vẽ trên canvas**. DrawAI đang chạy bằng Swin Transformer 1D đã train trên PenDigits; DINOv3 đã load pretrained weights công khai từ timm, export full ONNX và kiểm chứng backend CPU local.

**Tiến độ 18/10/2026:** **P0–P2 hoàn thành local**; head chữ số DINOv3 sẽ train đầy đủ ở P5 sau khi thu/khóa dữ liệu riêng P3–P4. MotionSense/HAR và video/fusion đã được chuyển ra khỏi project. Giữ nguyên design DrawAI.

- [Roadmap P0–P8](docs/drawai/two-model-roadmap.md)
- [Báo cáo P0](docs/drawai/phase0-report.md) · [Báo cáo P1](docs/drawai/phase1-report.md)
- [Báo cáo P2](docs/drawai/phase2-report.md) · [Setup DINOv3 và hướng dẫn weights](docs/drawai/dinov3-guide.md)
- [Contract hai model](docs/drawai/two-model-contract.json) · [Tiến độ](docs/drawai/status.md)

## Hai mô hình

| Model | Vai trò | Input | Trạng thái |
|---|---|---|---|
| Swin Transformer 1D | Đối chứng | float32 [1,3,128], x/y/stroke_start | Đã train và export ONNX, chạy browser WASM |
| DINOv3 ViT-S/16 + classifier | Backbone ảnh + head 10 lớp | float32 [1,3,224,224] RGB từ cùng nét | Pretrained timm đã load; full ONNX/backend CPU đạt; head chữ số chưa train đầy đủ |

Hai model sẽ dùng cùng sample ID và split theo người của dữ liệu tự thu. Không fusion. Trang chính giữ design hiện tại, dùng một model mặc định; so sánh hai model qua công cụ/trang kiểm chứng riêng.

## Chạy giao diện

Cần Node.js/npm; môi trường đã kiểm chứng là Node 24.18.0. Từ thư mục `web`:

```powershell
npm ci
npm run dev -- --port 4183 --strictPort
```

Mở `http://127.0.0.1:4183/` hoặc `/drawai`. Các trang kỹ thuật:

- `/drawai/release-check`: kiểm chứng Swin đã train.
- `/drawai/check`: phép thử kỹ thuật artifact chưa train, không dùng để đo chất lượng nhận dạng.
- `/drawai/integration-check`: kiểm chứng pipeline/benchmark.

Không cần Python để chạy demo browser. Model được tải và chạy trên thiết bị bằng ONNX Runtime Web; không gửi nét tới server suy luận trong P1.

## Python, dữ liệu và training

Cần Python 3.11; có thể tạo môi trường riêng:

```powershell
py -3.11 -m venv .venv-drawai
./.venv-drawai/Scripts/python.exe -m pip install -r training/drawai/requirements.txt
./.venv-drawai/Scripts/python.exe -m pip install -r training/drawai/requirements-verification.txt
./.venv-drawai/Scripts/python.exe -m training.drawai.prepare_data --download --length 64
./.venv-drawai/Scripts/python.exe -m training.drawai.prepare_data --length 128
```

Chọn wheel PyTorch CPU/CUDA phù hợp thiết bị. Môi trường local hiện hữu `.venv-phase1` dùng Torch 2.5.1+cpu và không có pip; không cần thay môi trường đó để chạy các lệnh kiểm chứng đã có. P1 chưa kiểm chứng clean install trên máy mới.

[Training guide](docs/drawai/training-guide.md) có lệnh smoke, thí nghiệm Swin-only, evaluate, export và notebook Kaggle. P1 không train lại checkpoint release. Pipeline DINOv3 pretrained đã kiểm chứng ở P2; chưa train classifier đầy đủ hoặc thu dữ liệu riêng.

## Model đang có

| Artifact | Input → output | Nguồn | Tình trạng |
|---|---|---|---|
| `web/public/models/drawai-swin-v1/model.onnx` | [1,3,128] → [1,10] logits | Swin seed44, PenDigits | Trained; 2.27 MiB |

SHA-256 ONNX: `28100de69dec82303a954af5c18936788e7c3dee623d70ec7a89ad9f7ef02366`.

Checkpoint nguồn local: `artifacts/drawai/d3-local-v1/runs/official-swin-l128-aug1-seed44/best.pth`. Checkpoint/raw data lớn bị ignore; artifact ONNX và metadata nằm trong web/public. Chưa có link public tải checkpoint nguồn; clone GitHub chưa tự có checkpoint local. Có thể chạy demo bằng ONNX hoặc tái tạo dữ liệu/train theo hướng dẫn.

Tên model rõ nghĩa sẽ được áp dụng đồng bộ ở P6:

```text
drawai_swin1d_pendigits_l128_fp32_v1.onnx
drawai_swin1d_private-v1_l128_fp32_v1.onnx
drawai_dinov3-vits16_private-v1_rgb224_fp32_v1.onnx
```

Các tên trên là đích kế hoạch, không phải ba model đã tồn tại. [Model card](docs/drawai/model-card.md) và release.json ghi preprocessing, labels, hashes và giới hạn.

Metric lịch sử PenDigits: accuracy 98,77%, macro-F1 98,79% trên 3.498 mẫu test. Không phải metric trên dữ liệu tự thu hoặc bảo đảm chất lượng nét người dùng. Dữ liệu tự thu và link Kaggle public sẽ được hoàn thành ở P3–P4/P8; hiện chưa có.

## Kiểm chứng

Build trước khi chạy browser tests, từ `web`:

```powershell
npm run build
npm test
npm run test:release
npm run test:integration
npx playwright test --config playwright.drawai-phase1.config.ts
```

`npm test` chạy 9 tests DrawAI; release suite gồm 13 tests; integration dùng Chrome/Edge cài local. P1 route suite kiểm tra trang gốc, alias, inference thật và ảnh design. Cài Chromium của Playwright nếu môi trường mới chưa có browser.

Từ root, dùng interpreter đã cài dependencies:

```powershell
./.venv-phase1/Scripts/python.exe -m unittest discover -s training/tests -p "test_drawai*.py" -v
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_training
```

Lệnh verify mặc định chỉ audit Swin được chọn trong thí nghiệm lịch sử, không cần checkpoint CNN đã lưu ngoài project. Không gọi kết quả đó là audit lại toàn bộ thí nghiệm CNN cũ.

## DINOv3 P2 — hoàn thành local

- Nguồn đã chọn: [timm DINOv3 ViT-S/16 QKV-bias](https://huggingface.co/timm/vit_small_patch16_dinov3_qkvb.lvd1689m), tải công khai không cần token; pin revision/hash, strict-load đủ 186 tensors và so sánh từng tensor với checkpoint.
- Code timm là open source; **weights dùng DINOv3 custom license**. Hai nguồn khác và lý do chọn nằm trong [báo cáo P2](docs/drawai/phase2-report.md).
- Backbone đóng băng + CLS/mean-patch pooling + head 10 lớp: **21.603.850 tham số**, 7.690 trainable. Raster Python/TypeScript khớp 19 fixtures.
- Artifact full graph: `artifacts/drawai/phase2/technical/drawai_dinov3-vits16-timm-qkvb_pretrained-backbone_rgb224_fp32_p2.onnx`, **86,634,209 bytes** (~82,62 MiB).
- **46 Python + 21 browser tests đạt**; build và audit đạt. WASM p95 661.6 ms vượt ngưỡng 500 ms; backend CPU local end-to-end p95 **90.2 ms** đạt. Chốt phương án backend DINOv3 cho tích hợp sau; Swin vẫn chạy browser.
- `pretrained_loaded=true`, `trained=false`, `release_ready=false`: backbone pretrained thật, head export chưa train chữ số. Tiny overfit chỉ kiểm tra tối ưu; chưa phải model DINOv3 nhận dạng để nộp cuối kỳ.
- Giữ nguyên design và model Swin trên trang chính. Trang kỹ thuật riêng `/drawai/dinov3-check` chỉ truy cập artifact/backend khi bật `DRAWAI_P2_LOCAL_SMOKE=1`; ONNX không nằm trong public/dist.

Từ root:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.dinov3_weights
./.venv-dinov3/Scripts/python.exe -m training.drawai.dinov3_feasibility
./.venv-dinov3/Scripts/python.exe tools/verify_phase2.py
```

[Hướng dẫn tái tạo, backend và kiểm thử](docs/drawai/dinov3-guide.md). Audit cần các suite đã chạy; bộ bằng chứng ở `artifacts/drawai/phase2/verification.json`.

## Thu dữ liệu riêng P3

**P3 đã sẵn sàng về kỹ thuật; pilot người thật chưa thực hiện (0/250 mẫu).** Mở `/collect` trên cùng server web: trang thu riêng, không thay design/trang nhận dạng DrawAI, không prediction hoặc upload nét. Có xác nhận ba mục đồng ý, kiểm tra thiết bị, lịch chữ số theo vòng, tải JSON và tiếp tục phiên từ bản đã tải. Tải sau mỗi vòng để tránh mất dữ liệu trong bộ nhớ.

- Pilot: P001–P005, mỗi người50 mẫu; chính: M001–M030, mỗi người200 mẫu. Người pilot không tham gia chính; split chính20/5/5 người được khóa trước thu.
- [Hướng dẫn điều phối và lệnh audit/build](docs/drawai/collection/operator-guide.md) · [Protocol cho người viết](docs/drawai/collection/protocol.md) · [Mẫu đồng ý](docs/drawai/collection/consent-template.md).
- [Schema phiên](docs/drawai/collection/session.schema.json), [policy split có hash](docs/drawai/collection/split-policy.json), template manifest/registry/rejection log và dataset card dự thảo tại `docs/drawai/collection/`.
- Code: `web/src/drawai/collect-app.tsx`, `collection.ts`, `training/drawai/private_data.py`. Validator kiểm tra schema/nhãn/prompt/thiết bị/time/duplicates/quota; build tạo raw JSON, RGB224 PNG, Swin NPY cùng sample ID/split và SHA-256. Output luôn staging, chưa khóa/công bố.
- Dữ liệu/consent/mapping người ở `data/private/` hoặc kho private của điều phối, bị Git ignore; không đặt vào public/dist. Không upload nguyên JSON phiên có mã hồ sơ consent lên Kaggle.
- **56 Python +17 browser tests đạt**, design/root/alias giữ nguyên. [Báo cáo P3](docs/drawai/phase3-report.md), evidence `artifacts/drawai/phase3/verification.json`. Fixture tự động không được tính vào số người/mẫu thật.

## Thu chính thức và khóa dataset P4

**P4 sẵn sàng về kỹ thuật; hiện chưa có dữ liệu người thật để khóa**: pilot0/250, main0/6.000. Không chuyển dữ liệu test tự động/PenDigits thành private data.

- Code mới `training/drawai/private_dataset.py`: draft review gắn hash source/audit/plan; phê duyệt pilot; khóa main khi đủ quota và xác nhận rà soát thật; verify theo trusted lock hash độc lập.
- Chặn thiếu quota/pilot, thiếu consent hoặc withdrawn, trùng người/mẫu giữa pilot/main, source thay sau review và các nhóm gần trùng chưa rà soát. Lý do loại mẫu không được là dự đoán AI.
- Package chung JSON/PNG/Swin NPY có manifest/hash; development chỉ train/validation, test nằm ZIP riêng có hash. ZIP **không mã hóa/không tự áp quyền truy cập**; điều phối phải giữ riêng.
- Loader/iterator hai representation dùng chung sample ID/split, chỉ nhận train/validation; chưa train model hoặc công bố Kaggle ở P4. Design và Swin release giữ nguyên.
- [Hướng dẫn rà soát](docs/drawai/dataset/review-guide.md) · [Khóa/verify và bàn giao P5](docs/drawai/dataset/locking-guide.md) · [Báo cáo P4](docs/drawai/phase4-report.md).
- Kiểm chứng:68 Python tests (12 mới P4) và12 browser regression đạt. Evidence ở `artifacts/drawai/phase4/verification.json`; positive-lock tests dùng policy giả lập nhỏ trong thư mục tạm, production vẫn250/6.000 mẫu.

## Đã cập nhật ở đâu

| Phase / file | Thay đổi |
|---|---|
| P0: `tools/phase0_*.py`, `artifacts/drawai/phase0/` | Snapshot phục hồi, inventory, contract và ảnh mốc |
| P1: `web/src/main.tsx`, `web/index.html` | `/`, `/drawai`, `/drawai/` mở DrawAI; bỏ route/metadata MotionSense |
| P1: `web/src/drawai/base.css` | Chuyển CSS nền từ file chung, giữ nguyên cascade/design |
| P1: `web/package*.json`, `web/playwright*.ts` | Package DrawAI, default tests chỉ DrawAI |
| P1: `training/drawai/models.py`, `train.py`, `experiment.py`, `evaluate.py` | Bỏ CNN runtime; thí nghiệm Swin-only version mới, output riêng |
| P1: `verify_training.py`, `export_onnx.py` | Audit checkpoint Swin độc lập checkpoint CNN; giữ selection/provenance lịch sử |
| P1: `build_notebook.py`, `notebooks/kaggle_train_drawai.ipynb` | Notebook/source bundle Swin-only, smoke riêng |
| P1: `.gitignore`, `data/.gitignore`, `NOTICE.md` | Chính sách dữ liệu/checkpoint và giữ ghi công nguồn |
| P1: `artifacts/drawai/phase1/deletion-plan.csv` | 309 file ngoài phạm vi đã chuyển ra ngoài DUST, có hash để khôi phục |
| P1: `docs/drawai/phase1-report.md` | Báo cáo thay đổi và kết quả nghiệm thu thực tế |
| P2: `training/drawai/dinov3_model.py`, `dinov3_weights.py`, `models.py` | Native timm pretrained backbone + head, factory hai họ, tải public snapshot pin revision/hash/config và strict-load |
| P2: `training/drawai/raster.py`, `web/src/drawai/raster.ts` | Renderer và normalization chia sẻ Python/TypeScript |
| P2: `dinov3_feasibility.py`, `requirements-dinov3.txt` | Gradient/freeze, tiny probe train-only, full ONNX export, dependency riêng |
| P2: `training/drawai/dinov3_backend.py`, `web/src/drawai/dinov3-backend-worker.ts` | Backend CPU local, raw-stroke parity, đo HTTP/raster/inference end-to-end và RSS |
| P2: `web/src/drawai/dinov3-*`, `web/vite.config.ts`, `main.tsx` | Trang kiểm chứng riêng, artifact chỉ local, giữ nguyên trang chính |
| P2: `training/tests/test_drawai_dinov3.py`, `test_drawai_raster.py`, `web/tests/drawai-dinov3*.spec.ts` | Positive pretrained, renderer/gradients, tamper, parity, API errors, recovery, runtime và release guards |
| P2: `tools/verify_phase2.py`, `docs/drawai/phase2-report.md` | Audit nguồn pretrained và runtime thật; P2 DONE_LOCAL, head chữ số chưa train |
| P3: `web/src/drawai/collect*`, `web/src/main.tsx` | Thêm trang /collect riêng, consent/prompt/canvas/export/resume, không đổi trang chính |
| P3: `training/drawai/private_data.py`, tests, `tools/verify_phase3.py` | Plan/hash, audit duplicates/quota, staging JSON/PNG/NPY/manifest, kiểm chứng P3 |
| P3: `docs/drawai/collection/`, `web/public/collection-protocol.md` | Protocol, consent, schema, split policy, template log/manifest/registry và dataset card |
| P4: `training/drawai/private_data.py` | Reject duplicate JSON keys; nhóm near-duplicate đầy đủ kể cả khi danh sách cặp bị cắt |
| P4: `training/drawai/private_dataset.py`, tests | Review/approval/source binding, quota/human gates, atomic final lock, held-out ZIP, verify và loader hai representation |
| P4: `tools/verify_phase4.py`, `docs/drawai/dataset/` | Audit tình trạng thật, hướng dẫn rà soát/khóa và trusted hash/test custody |

## Cấu trúc hiện hành

```text
models/SwinTransformer_cls.py       backbone Swin giữ nguyên
training/drawai/                    data, train, audit, export, notebook
training/tests/test_drawai*.py      tests DrawAI
web/src/drawai/                     canvas, design và inference workers
web/public/models/drawai-*/         model/metadata DrawAI
web/public/samples/drawai/          10 demo nguồn train
notebooks/                         notebook Swin (DINOv3 bổ sung P2)
docs/drawai/                       roadmap, báo cáo, protocol
artifacts/drawai/                   bằng chứng, checkpoint và kết quả
```

Một số metadata/báo cáo D0–D6 nhắc CNN hoặc route cũ được giữ làm lịch sử của checkpoint Swin, không thuộc pipeline training hiện hành. Xem [NOTICE](NOTICE.md) và [LICENSE](LICENSE) để biết nguồn gốc/ghi công.
