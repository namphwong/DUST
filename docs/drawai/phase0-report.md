# DrawAI — Báo cáo Phase 0: kiểm kê và lưu mốc phục hồi

Ngày nghiệm thu: 17/10/2026 (Asia/Bangkok). **P0 hoàn thành local. P1 chưa bắt đầu.**

P0 thuộc kế hoạch hai mô hình mới, khác các phase D0–D7 của đợt DrawAI trước. Chưa xóa MotionSense, chưa đổi route, CSS, model, preprocessing hoặc code sản phẩm. Chỉ thêm công cụ/tài liệu và chạy lại kiểm chứng; các file bằng chứng UI cũ được cập nhật bởi tests sau khi đã sao lưu.

## 1. Phạm vi đã khóa

- Sản phẩm duy nhất sau P1: DrawAI, nhận dạng một chữ số 0–9 từ nét canvas.
- Swin Transformer 1D là mô hình đối chứng; DINOv3 ViT-S/16 pretrained là backbone ảnh với classifier 10 lớp.
- Cùng JSON nét gốc sinh chuỗi cho Swin và ảnh RGB cho DINOv3. Không fusion, không thêm OCR ảnh tải lên.
- Giữ nguyên bố cục, màu, typography, canvas, nút nhận dạng, undo/clear, top-3 và luồng người dùng hiện tại.
- Trang chính dùng một model mặc định; hai model so sánh qua script/trang kiểm chứng riêng. Không tự thêm bộ chọn model vào design.
- Dữ liệu tự thu chia theo người; chốt cấu hình/model bằng validation và feasibility trước khi mở test.

Hợp đồng máy đọc được: [two-model-contract.json](two-model-contract.json). Input DINOv3 224×224 là cấu hình thử P2; renderer, normalization chính xác, weights revision và ngân sách runtime sẽ được khóa trong P2. Không có DINOv3 đã train ở P0.

| Model | Input inference | Output | Tình trạng |
|---|---|---|---|
| Swin 1D | float32 NCT [1,3,128], x/y/stroke_start | [1,10] logits, lớp 0–9 | Đã train PenDigits; kiểm chứng lại P0 |
| DINOv3 ViT-S/16 + head | float32 NCHW [1,3,224,224], RGB | [1,10] logits, lớp 0–9 | Kế hoạch P2, chưa có artifact |

Tên export đích: `drawai_swin1d_pendigits_l128_fp32_v1.onnx`, `drawai_swin1d_private-v1_l128_fp32_v1.onnx`, `drawai_dinov3-vits16_private-v1_rgb224_fp32_v1.onnx`. Chưa đổi tên model cũ trong P0.

## 2. Mốc phục hồi đã tạo và kiểm tra

- Git branch ban đầu: `main`.
- Commit nền: `ac092dc451ce0ae14adb587b5cbb209ddb44ce59`.
- Backup ngoài project: `D:\ChuyenDe\DUST-backups\phase0-20261017-112405\`.
- `working-tree.zip`: 292.117.263 bytes (~278,58 MiB), gồm code hiện tại, dữ liệu, model, checkpoint, notebooks và artifact thuộc phạm vi payload.
- `history.bundle`: 34.687.662 bytes, lưu lịch sử và refs Git bằng `git bundle create --all`; `git bundle verify` đạt.
- Đã clone bundle vào `restore-check`, giải nén payload và khôi phục index của clone bằng `git reset --mixed HEAD` (chỉ trong clone kiểm chứng, không chạy ở project gốc).
- **551/551 file khôi phục có SHA-256 khớp bản nguồn.** Tổng payload 347.816.890 bytes.
- Phân bố file: 220 tracked, 218 untracked, 113 ignored. Đây là số file payload, không bao gồm runtime/cache.
- Checkpoint Swin, dữ liệu PenDigits raw/processed và model ONNX đều nằm trong payload, kể cả file bị ignore.

Chi tiết/hash ZIP và bundle: [snapshot.json](../../artifacts/drawai/phase0/snapshot.json).

Không backup venv, node_modules, caches, web/dist hoặc web/test-results. Đã lưu environment, 60 package Python, web lockfile và kết quả build; đây chưa phải kiểm chứng cài môi trường sạch. Git config cục bộ/hooks/reflog và staging index không được clone nguyên trạng từ bundle; snapshot này phục hồi lịch sử và nội dung working tree. Trạng thái trước backup lưu riêng, không tạo commit/tag hay thay staging của người dùng.

### Cách phục hồi khi cần

1. Dùng thư mục mới ngoài DUST, tránh ghi đè code đang làm.
2. Clone `history.bundle` với `git clone --no-checkout` tới thư mục đó.
3. Giải nén `working-tree.zip` vào clone; chạy `git reset --mixed HEAD` ở clone để tạo index mà không đổi file working tree.
4. Đối chiếu mỗi đường dẫn/hash với `file-inventory.csv`, tạo lại dependencies từ môi trường đã ghi nhận rồi build/test.
5. Có thể xem trực tiếp bản đã phục hồi tại `D:\ChuyenDe\DUST-backups\phase0-20261017-112405\restore-check`.

Không chạy lệnh reset/clean hoặc giải nén đè vào project gốc để thử phục hồi.

## 3. Kiểm kê và danh mục P1

- [file-inventory.csv](../../artifacts/drawai/phase0/file-inventory.csv): đủ 551 đường dẫn, bytes, SHA-256, trạng thái Git và phân loại ban đầu.
- [dependency-references.csv](../../artifacts/drawai/phase0/dependency-references.csv): imports Python qua AST và tham chiếu web/config/notebook tĩnh. Không coi regex/static scan là bằng chứng đầy đủ cho mọi đường dẫn động.
- [phase1-file-decisions.csv](../../artifacts/drawai/phase0/phase1-file-decisions.csv): danh mục hành động sau rà soát; dùng file này làm nguồn lập thao tác P1, không xóa trực tiếp dựa trên phân loại ban đầu.

| Nhóm | Hành động P1 | Ghi chú |
|---|---|---|
| training/drawai, web/src/drawai, tests/config DrawAI | Giữ, refactor điểm phụ thuộc lịch sử | Không viết lại phần đã chạy được |
| models/SwinTransformer_cls.py | Giữ | models.py của DrawAI import trực tiếp |
| data/raw/pendigits, data/processed/pendigits-drawai-v1 | Giữ | Dữ liệu gốc, split và provenance |
| web/public/models/drawai-*, samples/drawai, verification/drawai | Giữ; rename release ở P6 | Phân biệt model chưa train D2 với release Swin thật |
| Artifacts Swin và báo cáo lựa chọn cũ | Giữ truy nguyên | Không đổi nhãn dataset cũ thành private |
| Checkpoint CNN/smoke cũ | Archive ngoài project, rồi dọn bản active | Backup đã có; tách verifier trước |
| main.tsx, README, package/config, .gitignore | Cập nhật | / sẽ mở DrawAI ở P1 |
| web/src/style.css | Di chuyển phần CSS nền cần thiết sang DrawAI | Có global selectors ảnh hưởng design |
| web/src/motion-style.css | Gỡ import rồi xóa | Scoped .motion-app; vẫn phải đối chiếu ảnh |
| MotionSense UI và CSV/worker/chart | Xóa sau gỡ imports | motion-app, pipeline-client/worker, csv-pipeline/preview, signal-chart |
| Training/notebooks/tests/data/models/docs HAR | Xóa sau kiểm tra tham chiếu | Bao gồm artifacts/phase*, artifacts/ui-redesign, HAR ONNX |
| step_fusion*, gen_data, video/fusion models | Xóa | Không dùng trong DrawAI |
| LICENSE và ghi công backbone | Giữ | Không mất nguồn gốc khi đổi tên project |
| Requirements cũ | Xóa sau chuyển tham chiếu | DrawAI có requirements riêng; không xóa venv hiện dùng |

Các quyết định có điều kiện là thứ tự migration cần thực hiện trong P1, không phải thiếu phê duyệt. Chưa xóa file nào ở P0.

### Phụ thuộc cần chú ý

1. `training/drawai/models.py → models/SwinTransformer_cls.py`: không xóa cả thư mục models.
2. `main.tsx` import cả CSS chung, MotionSense và DrawAI. CSS chung có body margin, font, main/button/status/details/footer rules. Phải bảo toàn cascade cần thiết và so ảnh trước/sau khi dọn.
3. `training/drawai/export_onnx.py → verify_training.py → experiment.py`: verifier hiện yêu cầu sáu run CNN/Swin và khóa selection lịch sử. Nếu xóa CNN ngay sẽ làm hỏng tái kiểm chứng export. Giữ legacy provenance hoặc chuyển verifier sang kiểm chứng checkpoint/selection độc lập trước khi bỏ CNN runtime.
4. `web/src/drawai/release-loader.ts`, `types.ts`, worker, tests và exporter đang cố định `drawai-swin-v1`, `model.onnx` và hash checkpoint. P6 phải đổi đồng bộ, không chỉ rename file.
5. `build_notebook.py` hiện sinh thí nghiệm CNN/Swin. Khi chuyển sang hai model mới phải cập nhật generator và notebook cùng lúc.
6. `data/.gitignore` đang ignore toàn bộ nội dung ngoài chính nó. P1 cần giữ chính sách không commit raw/private data, đồng thời ghi rõ manifest công khai đặt ở đâu.
7. Venv hiện tại không có pip. Đã lấy danh mục package bằng importlib.metadata, không cài thêm vào môi trường để phục vụ P0. P2 cần môi trường riêng/đã kiểm chứng cho DINOv3.

## 4. Kiểm chứng đã chạy lại

| Kiểm chứng | Kết quả | Bằng chứng |
|---|---|---|
| TypeScript + Vite production build | PASS | phase0/build.log |
| Python: data/model/training/canvas gate | 32 tests PASS | phase0/python-tests.log |
| Chromium DrawAI UI | 9 tests PASS, 0 fail/flaky/skip | phase0/browser-tests.log và ui/playwright-report.json |
| Release manifest | Hash/bytes đủ 5 file khớp | phase0/baseline.json |
| ONNX checker full_check | PASS | phase0/baseline-verification.log |
| Checkpoint load strict + provenance | PASS; 545.038 tham số | phase0/baseline.json |
| Python PyTorch / ONNX CPU / browser WASM | 10 demo train khớp logits trong tolerance và cùng argmax | phase0/baseline.json |
| Snapshot restoration | 551/551 hash khớp | phase0/snapshot.json |

Lệnh đã chạy từ root, trừ npm/npx chạy trong web:

```powershell
./.venv-phase1/Scripts/python.exe tools/phase0_snapshot.py
npm run build
./.venv-phase1/Scripts/python.exe -m unittest training.tests.test_drawai_data training.tests.test_drawai_model training.tests.test_drawai_training training.tests.test_drawai_canvas -v
npx playwright test --config playwright.drawai-app.config.ts
./.venv-phase1/Scripts/python.exe tools/phase0_verify_baseline.py
```

Browser tests đã kiểm tra 10 demo thật, reuse worker/session, schema/metadata/timestamp lỗi, canvas/undo/clear/export JSON, bỏ kết quả cũ, lỗi tải và retry, hash sai, pointer cancel, keyboard, resize và DPR2. Không chạy lại toàn bộ D2/D4/D6 cũ vì P0 chưa đổi runtime; không lấy kết quả cũ làm kiểm chứng mới.

Tolerance parity: abs(error) <= 1e-5 + 1e-4 * abs(expected). 10 mẫu là demo nguồn train, không phải phép đo test accuracy mới. Không retrain hoặc thay checkpoint.

## 5. Mốc giao diện

Đã chụp lại và xem trực quan:

- [Desktop 1366×768, ảnh toàn trang](../../artifacts/drawai/phase0/ui/desktop.png).
- [Mobile viewport 390×844, ảnh toàn trang](../../artifacts/drawai/phase0/ui/mobile.png).

Desktop giữ hai cột; mobile một cột, nút và mẫu hiển thị đầy đủ, không tràn ngang. Ảnh mốc ở trạng thái nhận dạng mẫu 6. Viewport giả lập và DPR2 không phải thử điện thoại/bút/cảm ứng thật.

`baseline.json` lưu hash code giao diện/CSS/backbone làm mốc. Phase cuối đối chiếu lại để bảo đảm P0 không sửa code sản phẩm. Thay đổi UI bằng chứng do chạy tests đã được sao lưu bản cũ trước đó.

## 6. Điều kiện chuyển P1

- [x] Khóa phạm vi hai model và nguyên tắc giữ design.
- [x] Có danh mục file/trạng thái Git và tham chiếu phụ thuộc.
- [x] Có danh mục giữ/xóa/archive/migration theo đường dẫn.
- [x] Backup code + dữ liệu + checkpoint; phục hồi thử và kiểm tra hash.
- [x] Build, unit tests, browser tests và inference mẫu đạt.
- [x] Có ảnh desktop/mobile và source fingerprints.
- [x] Có contract model/input/output/naming và tiêu chí chọn mặc định trước test.

**Sẵn sàng P1.** Công việc tiếp theo: dọn theo danh mục, chuyển / sang DrawAI, bảo toàn CSS nền và tách phụ thuộc verifier CNN. P0 không bao gồm huấn luyện DINOv3, thu dữ liệu người thật, Kaggle public hoặc deploy.
