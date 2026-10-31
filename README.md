# DrawAI

Nhận dạng **một chữ số viết tay 0–9 từ canvas**. Trang chính dùng **DINOv3 ViT-S/16 làm backbone + head 10 lớp**, chọn checkpoint bằng validation P5. **Swin Transformer 1D** là mô hình đối chứng. Hai model nhận cùng ảnh xám 64×64 được tạo từ nét vẽ, không fusion.

**Cập nhật 19/10/2026:** P0–P6 hoàn thành local; **P7 hoàn thành tích hợp kỹ thuật local**, giữ design. Chưa nghiệm thu chất lượng trên người viết mới/cảm ứng thật; P8 và phân phối model lớn cho checkout mới còn chờ. [Báo cáo P7](docs/drawai/phase7-report.md) · [Hướng dẫn chạy](docs/drawai/paint-app-guide.md) · [Tiến độ](docs/drawai/status.md) · [Roadmap](docs/drawai/two-model-roadmap.md).

**Rà soát repo:**133 Python +33 browser tests đạt; đã củng cố worker/retry và hủy inference cũ, dọn cache, tách kết quả kiểm tra khỏi snapshot lịch sử. [Báo cáo tổng hợp](docs/drawai/repo-review-report.md) · [Mục lục tài liệu](docs/drawai/README.md). Các báo cáo phase bên dưới vẫn giữ số đo theo thời điểm thực hiện phase đó.

**Kế hoạch web public ưu tiên (28/10/2026, chưa triển khai):** [Vercel frontend + Render backend](docs/drawai/vercel-render-plan.md), chia VR0–VR6; DINO mặc định qua Render, Swin chạy browser, kiểm tra512MB trước khi deploy Free. [Phương án GitHub Pages](docs/drawai/github-pages-plan.md) giữ để tham khảo.

**VR0 đã hoàn thành:** [khóa scope/model/input/design và tiêu chí nghiệm thu](docs/drawai/vr0-report.md);10 gate tests mới PASS,5 checklist PASS/14 NOT_RUN. Chưa thay runtime hoặc deploy; bước tiếp theo VR1 là đóng gói/phân phối model.

**CI/CD và Docker đã thực thi (31/10/2026):** [Báo cáo từng bước/chỗ cập nhật](docs/drawai/cicd-docker-report.md) · [CI PASS](https://github.com/namphwong/DUST/actions/runs/41164031905) · [Docker Hub public v1.0.0](https://hub.docker.com/r/namphwong/drawai-dinov3-api) · [Model Release public](https://github.com/namphwong/DUST/releases/tag/drawai-models-paint35-v1). Build/test inference dưới giới hạn512MiB/0,1CPU, pull digest trên runner mới không login đã PASS. Continuous delivery tới registry hoàn thành; **Render/Vercel cloud chưa deploy**. Các đoạn VR0/P7 phía trên là mốc lịch sử.

## Hai model hiện hành

| Model | Vai trò / input | Artifact đã xuất | Test Paint P5 |
|---|---|---|---|
| Swin Transformer 1D seed43 | Đối chứng; float32 [1,64,64], 64 vị trí hàng, 64 đặc trưng cột; không phải thời gian nét bút | `drawai_swin1d-image_paint35-main-v1_rows64_fp32_v1.onnx` | Accuracy84,40%; macro-F184,28% |
| DINOv3 ViT-S/16 seed42 + linear head | Backbone pretrained đóng băng; float32 [1,3,224,224] | `drawai_dinov3-vits16_paint35-main-v1_rgb224_fp32_v1.torchscript.pt` | Accuracy98,80%; macro-F198,80% |

Cả hai chạy **CPU trên backend local** trong P7. Swin ONNX đã kiểm chứng thêm WASM ở P6. DINO ONNX chỉ diagnostic:2/1.000 mẫu vượt ngưỡng logits, nên dùng TorchScript đạt parity; không nới tolerance. [P5](docs/drawai/phase5-report.md) · [P6](docs/drawai/phase6-report.md).

**Giới hạn dữ liệu/chất lượng:**37 nhóm ảnh tương đồng qua split main chưa giải quyết; nguồn/người/đồng ý theo xác nhận chủ dữ liệu, chưa xác minh độc lập. Metric Paint không đại diện canvas. Giữ10 mẫu PenDigits train hiện có; DINO nhận đúng5/10 qua adapter mới. Đây là quan sát demo khác miền, không phải benchmark. Không sửa mẫu/adapter/checkpoint để tăng số đúng và không đánh giá lại private test trong P7.

## Chạy DrawAI local

Cần Node/npm, Python3.11 và **bundle P6 đầy đủ** trong `models/exported/paint35-v1/` (hai artifact primary, DINO ONNX diagnostic và metadata để kiểm tra inventory/hash). Model lớn có tại [GitHub Release](https://github.com/namphwong/DUST/releases/tag/drawai-models-paint35-v1); clone code cần tải thêm bundle để chạy default DINO. `python tools/prepare_public_models.py fetch` tải/verify vào `deployment/docker/model-bundle/` để build container; runtime local P7 vẫn dùng đường dẫn `models/exported/paint35-v1/`. Nếu đã có môi trường `.venv-dinov3` và bundle P6 trên máy hiện tại:

```powershell
cd D:\ChuyenDe\DUST\web
npm ci
cd ..
powershell -ExecutionPolicy Bypass -File tools/run_drawai.ps1
```

Mở `http://127.0.0.1:4186/`. Lệnh trên khởi động backend loopback4197, kiểm tra bundle trước khi mở Vite4186; Ctrl+C dừng phiên. Backend không lưu ảnh/nét và không cần timm/Transformers khi suy luận. Cần torch cho TorchScript. Giao diện và proxy chỉ local, chưa triển khai production.

- `/`, `/drawai`: DINOv3 mặc định; canvas, undo/clear, top3, retry và export JSON giữ nguyên.
- `/drawai/compare`: cùng nét/ảnh chạy Swin và DINO; trang riêng, không thêm selector trên trang chính.
- `/drawai/legacy`: Swin PenDigits cũ chạy WASM, không cần backend; nhãn nguồn demo vẫn PenDigits train.
- `/drawai/release-check`, `/drawai/integration-check`: kiểm chứng release PenDigits lịch sử.
- `/collect`: công cụ thu nét mới, không phải nguồn trajectory của NPZ Paint.

Backend tắt/weights thiếu hoặc sai hash thì hiển thị lỗi, giữ nét để retry; không đổi ngầm sang model cũ. Static `dist/` đơn lẻ không cung cấp API DINO. [Cài dependency/chạy hai terminal/tái tạo bundle](docs/drawai/paint-app-guide.md).

## Dữ liệu, training và export

[Dataset Kaggle public](https://www.kaggle.com/datasets/namphwong/drawai-handwritten-digits-placeholder) đã có `So_viet_tay_35_nguoi.npz`:7.000 ảnh64×64,35 mã người theo xác nhận chủ dữ liệu; **không có trajectory**. Slug vẫn giữ tên cũ để không đổi link. Dòng Synthetic trong NPZ giữ nguyên bytes; [provenance/attestation](docs/drawai/collection/paint-image64-provenance.json) giải thích xác nhận của chủ dữ liệu. Không tự gán license mới.

Split khóa: pilot250 +reserved750 (người1–5, loại khỏi main), train4.000 (6–25), validation1.000 (26–30), test1.000 (31–35). Test nguồn đã public, custody chỉ bảo vệ quy trình/integrity. [P3](docs/drawai/phase3-report.md) · [P4](docs/drawai/phase4-report.md).

- [Train hai model, chọn checkpoint và giới hạn](docs/drawai/paint-training-guide.md).
- [Export/verify bundle, filenames/checksum và mixed runtime](docs/drawai/paint-export-guide.md).
- [Pretrained DINOv3/provenance/license](docs/drawai/dinov3-guide.md). DINOv3 dùng license riêng, không mặc định coi weights là OSI open source.
- [Contract hai model](docs/drawai/two-model-contract.json), [registry bundle](models/exported/paint35-v1/registry.json).

Dữ liệu/checkpoint/weights mới bị ignore; metadata và bằng chứng kiểm chứng có trong Git. Model PenDigits lịch sử vẫn có sẵn tại `web/public/models/drawai-swin-v1/drawai_swin1d_pendigits_l128_fp32_v1.onnx`, alias `model.onnx` byte-identical.

## Kiểm chứng P7

Từ root, dùng Python đã cài [requirements inference](training/drawai/requirements-paint-inference.txt):

```powershell
./.venv-dinov3/Scripts/python.exe -m unittest discover -s training/tests -p "test_drawai*.py" -v
./.venv-dinov3/Scripts/python.exe tools/prepare_phase7_fixtures.py
./.venv-dinov3/Scripts/python.exe tools/verify_phase7_backend.py
cd web
npm run build
npm test
npm run test:legacy
cd ..
./.venv-dinov3/Scripts/python.exe tools/verify_phase7.py
```

Chạy tuần tự các kiểm tra dùng cùng port4197; dừng dịch vụ local cũ trước. `npm test`/`test:app` chạy13 tests (10P7 +3route); `test:legacy` chạy9 tests. Tổng133 Python +22 browser tests đạt; bộ Python cần dữ liệu/artifact local của các phase trước. Browser cần Chromium Playwright. Thời gian đo chi tiết và phần chưa chạy nằm trong báo cáo P7.

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
| P3: `docs/drawai/collection/`, `web/public/collection-protocol.md` | Protocol nét bút v1 lịch sử và protocol ảnh Paint hiện tại |
| P3 rerun: `training/drawai/paint_data.py`, `tools/verify_phase3_image64.py`, tests | Audit7.000 ảnh, provenance, writer split mới, pilot250 và đầu vào ảnh chung cho hai model |
| P4: `training/drawai/private_data.py` | Reject duplicate JSON keys; nhóm near-duplicate đầy đủ kể cả khi danh sách cặp bị cắt |
| P4: `training/drawai/private_dataset.py`, tests | Review/approval/source binding, quota/human gates, atomic final lock, held-out ZIP, verify và loader hai representation |
| P4: `tools/verify_phase4.py`, `docs/drawai/dataset/` | Audit tình trạng thật, hướng dẫn rà soát/khóa và trusted hash/test custody |
| P4 ảnh: `paint_dataset.py`, `test_drawai_paint_lock.py`, `verify_phase4_image64.py` | Khóa main6.000 ảnh thực, review giữ cờ tương đồng, loaders/hash/test guards, báo cáo/evidence riêng |
| P5: `paint_train.py`, `models.py`, `test_drawai_paint_train.py` | Factory Swin row64, train/resume sáu run, pretrained feature cache, validation selection, final-test guards và checkpoint đầy đủ |
| P5: `benchmark_phase5.py`, `verify_phase5.py`, `docs/drawai/phase5-report.md` | Metrics replay, kiểm tra frozen weights, benchmark CPU fresh process, report và handoff riêng |
| P6: `paint_export.py`, `paint_package.py`, `paint_runtime.py`, `native_runtime.py` | Full export, primary-format registry/anchor, full-test parity, mixed CPU runtime, giữ DINO ONNX diagnostic |
| P6: legacy exporter/verifier/loaders/manifests/tests | Đổi canonical filename Swin PenDigits, giữ alias byte-identical và nghiệm thu model/design cũ |
| P6: `verify_phase6*.py`, `drawai-p6*.spec.ts`, `phase6-report.md` | Browser WASM, localhost backend/errors/budgets, default-closed gate, migration ledger và báo cáo |
| P7: `paint-registry.ts`, `paint-raster.ts`, `paint-worker.ts`, `paint-client.ts` | Registry khóa bundle, canvas400 → gray64, worker/API và kiểm tra hash/revision/softmax |
| P7: `paint_app_runtime.py`, `vite.config.ts`, `run_drawai.ps1` | Backend CPU local, proxy cùng origin, giới hạn input và lệnh chạy hai dịch vụ |
| P7: `drawai-app.tsx`, `paint-compare.tsx`, `main.tsx` | DINO mặc định; trang so sánh riêng; giữ design, legacy chuyển sang route riêng |
| P7: `test_drawai_paint_app.py`, `drawai-p7.spec.ts`, `verify_phase7*.py` | 133 Python +22 browser tests, parity, lỗi/retry, benchmark và kiểm tra giao diện |
| Rà soát: `paint-client.ts`, `paint-worker.ts`, `paint-compare.tsx`, `drawai-review.spec.ts` | Chặn response sai dạng, reset khi postMessage lỗi, ngừng model thứ hai khi bản vẽ đã đổi;6 ca hồi quy mới |
| Dọn repo: `audit_repo.py`, `clean_repo_cache.ps1`, tài liệu/index và review config | Audit cú pháp/link/duplicates, xóa cache theo allowlist, tách bằng chứng mới, sửa hướng dẫn lỗi thời |

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
