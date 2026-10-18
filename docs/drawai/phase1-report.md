# DrawAI — Báo cáo P1: chỉ giữ DrawAI

Ngày nghiệm thu: 17/10/2026, Asia/Bangkok. **P1 hoàn thành local. P2 chưa bắt đầu.**

DrawAI hiện mở trực tiếp tại `/`, `/drawai` và `/drawai/`. Đã đưa MotionSense/HAR, video/fusion DuST và CNN runtime ra khỏi project đang sử dụng. Design và model Swin đang phục vụ người dùng được giữ nguyên.

## 1. Mốc phục hồi và phạm vi dọn

- Tạo backup ngay trước thay đổi: `D:\ChuyenDe\DUST-backups\pre-phase1-20261017-113455`.
- `working-tree.zip`: 573 file payload trước P1, gồm toàn bộ tài liệu/bằng chứng P0, code, data và checkpoints. Đã đọc lại từng member và đối chiếu hash.
- `history.bundle`: lịch sử Git được lưu và `git bundle verify` đạt. Không commit/push, reset hay thay staging của checkout gốc.
- Runtime/dependency/cache không nằm trong ZIP; metadata môi trường P0 vẫn được giữ.
- [pre-phase1-backup.json](../../artifacts/drawai/phase1/pre-phase1-backup.json) chứa đầy đủ đường dẫn/hash và thông tin phục hồi.

Lệnh xóa bằng PowerShell bị kiểm tra an toàn tự động từ chối với thông báo `blocked by policy`, kể cả khi chỉ định một file cụ thể. Phương án hoàn thành được sử dụng là **Move-Item sang archive ngoài DUST**, kiểm tra đường dẫn nguồn/đích nằm trong phạm vi cho phép, hash khớp backup và không ghi đè file archive. Không tìm cách xóa qua shell/ngôn ngữ khác.

**309 file ngoài phạm vi** đã chuyển tới `removed-files/` trong backup trên. Ngoài 297 file theo lượt dọn đầu còn có 12 file đánh giá CNN cũ được phát hiện khi rà soát cuối. Tất cả 309 file archive đã đối chiếu lại SHA-256. Cache sinh tự động và thư mục legacy rỗng cũng được lưu ngoài project.

- [deletion-plan.csv](../../artifacts/drawai/phase1/deletion-plan.csv): danh sách 309 file, hành động và hash trước P1.
- [archived-directories.json](../../artifacts/drawai/phase1/archived-directories.json), [additional-empty-dirs.json](../../artifacts/drawai/phase1/additional-empty-dirs.json): thư mục cache/rỗng đã chuyển.
- Dữ liệu PenDigits raw/processed: **23 file được giữ nguyên hash**.
- Model ONNX, checkpoint Swin và backbone cũng giữ nguyên hash.

Có thể khôi phục từng file bằng cách lấy đúng đường dẫn tương đối trong `removed-files/` hoặc ZIP pre-P1 vào một thư mục phục hồi riêng. Không giải nén đè vào DUST đang làm việc để thử phục hồi.

## 2. Các thay đổi chính

| Thành phần | Thay đổi |
|---|---|
| `web/src/main.tsx` | Chỉ route DrawAI, alias và ba trang kỹ thuật DrawAI; đường dẫn lạ có liên kết về trang chính |
| `web/index.html` | Title, description là DrawAI, bỏ metadata MotionSense |
| `web/src/drawai/base.css` | Chuyển CSS nền cũ nguyên byte để bảo toàn cascade và trang kỹ thuật |
| MotionSense UI/CSV/worker/chart | Đưa toàn bộ source và tests liên quan ra ngoài project |
| Data/model/artifact/notebook HAR | Đưa ra ngoài project; production build không chứa token/worker/model HAR |
| Video/fusion/toy scripts | Đưa ra ngoài project, giữ license và ghi công nguồn |
| `web/package.json`, lockfile | Đổi package name thành `drawai`, thêm các lệnh test DrawAI rõ nghĩa |
| Default Playwright config | Chạy 9 tests sản phẩm DrawAI; không còn suite HAR |
| `training/drawai/models.py`, `train.py` | Bỏ CNN class/factory/config; Swin còn nguyên backbone và cấu hình |
| `experiment.py`, `evaluate.py` | Plan mới `drawai-swin-experiment-v2`: 4 pilot + 3 official Swin run, thư mục output riêng |
| `verify_training.py`, `export_onnx.py` | Audit nguồn Swin được chọn độc lập checkpoint CNN; exporter có tùy chọn output candidate riêng |
| `report_experiment.py` | Báo cáo Swin có scope rõ, không tái tạo tuyên bố kiểm chứng CNN đã archive |
| Notebook/builder | Swin-only; cell IDs ổn định, source bundle có hash; output tách khỏi thí nghiệm lịch sử |
| Ignore/requirements | Giữ raw/private data và file lớn ngoài Git; khai báo Pillow cho so sánh ảnh; bỏ rule HAR cũ |
| README/docs/NOTICE | README riêng DrawAI, chỉ dẫn file cập nhật, cách chạy/train/export và ranh giới lịch sử/hiện hành |

Vẫn giữ các JSON selection/config/evaluation tổng hợp và báo cáo D3 lịch sử để truy nguyên checkpoint Swin; có thể còn nhắc CNN ở các tài liệu đó. CNN không còn class, config, notebook train, checkpoint hoặc file đánh giá riêng trong pipeline active. Không sửa lịch sử để biến kết quả CNN thành DINOv3.

## 3. Tách phụ thuộc CNN mà không thay checkpoint

Trước P1, exporter gọi verifier yêu cầu đủ toàn bộ checkpoints của 8 pilot + 6 run CNN/Swin. Sau P1:

1. Đọc chain plan/configuration lock/selection/evaluation/test-access bất biến.
2. Kiểm tra lựa chọn Swin theo validation trên ba seed và cấu hình thắng pilot Swin.
3. Đối chiếu checkpoint hash, summary, epoch tốt nhất và validation history.
4. Load trọng số strict; kiểm tra input/labels/preprocessing và hash train/validation.
5. Tái tạo 1.495 validation predictions và đối chiếu best-validation logits/IDs/labels.
6. Ghi `selected-swin-verification.json`, scope `selected-swin-only`, không ghi đè chứng cứ audit CNN lịch sử.

Lệnh mặc định `python -m training.drawai.verify_training` kiểm chứng selected Swin từ thí nghiệm cũ mà không cần checkpoint CNN. Plan Swin-only mới có output riêng để không ghi đè test/selection lịch sử. Full 4+3 training mới chưa thực hiện ở P1.

Hai regression tests mới chặn model CNN đi vào training hiện hành và từ chối provenance chain hỏng trước khi load weights. Test về test-access gate được chuyển sang yêu cầu ba official Swin run.

## 4. Design và suy luận được bảo toàn

- `drawai-app.tsx`, `drawai.css`, canvas, inference/preprocessing/worker và backbone Swin có hash như P0.
- `base.css` bằng nguyên nội dung `style.css` cũ; stylesheet MotionSense đã bỏ khỏi build.
- Ảnh desktop và mobile của `/drawai` **khớp từng pixel** với P0.
- Ảnh trang gốc `/` ở cùng trạng thái mẫu 6 cũng khớp từng pixel với P0.
- Không thêm dropdown model, tab, dashboard hoặc form thu dữ liệu vào giao diện chính.

Ảnh mốc sau P1:

- [Desktop trang gốc](../../artifacts/drawai/phase1/ui/root-desktop.png).
- [Mobile trang gốc](../../artifacts/drawai/phase1/ui/root-mobile.png).
- [Kết quả so sánh ảnh](../../artifacts/drawai/phase1/design-parity.json).

Viewport 1366×768 và 390×844; full-page PNG có kích thước 1366×920 và 390×1165. Đây là kiểm tra browser/viewport/DPR, không phải nghiệm thu thiết bị cảm ứng thật.

## 5. Kết quả kiểm chứng thực chạy

| Kiểm chứng | Kết quả |
|---|---|
| TypeScript + Vite production build | PASS; 38 modules thay vì 46 ở P0 |
| Python DrawAI | **34/34 PASS** |
| Default sản phẩm Chromium | **9/9 PASS** |
| Route/root/alias/unknown + inference root | **3/3 PASS** |
| D2/D4 release và feasibility | **13/13 PASS** |
| Integration + UI Chrome/Edge | **34/34 PASS** |
| Tổng lượt browser tests | **59 PASS**, không fail/flaky/skip |
| Notebook hiện hành | 4 code cells CPU smoke PASS; 2 epoch Swin, test unopened; rerun hỗ trợ resume |
| Source bundle notebook | Hash từng file khớp source hiện hành |
| Selected checkpoint audit | PASS; không cần CNN checkpoints |
| ONNX export candidate | PASS; full-test 3.498 mẫu, **0 khác argmax**, macro-F1 degradation **0** |
| Candidate ONNX | Hash giống ONNX đang phục vụ; không thay deployed model |
| Import mọi module training/drawai | PASS; không còn dangling import HAR/video/CNN |
| Archive integrity | **309/309 file** khớp hash backup |

Export kiểm chứng sử dụng thư mục riêng `phase1/export-candidate`, `phase1/export-audit`, `phase1/export-samples`, không ghi đè release đang chạy. Max absolute logits error 1,1444092e-05, nằm trong tolerance `1e-5 + 1e-4*abs(expected)`. Accuracy PenDigits tái tạo 98,7707%, macro-F1 98,7896%; không có training hoặc chọn checkpoint mới từ test.

Production CSS giảm từ 34,00 kB xuống 10,80 kB; JS main từ 251,26 kB xuống 213,38 kB theo Vite report. Model và WASM runtime có dung lượng riêng; số này không phải tổng tải của ứng dụng.

## 6. Lệnh và bằng chứng

Từ `web`:

```powershell
npm run build
npm test
npx playwright test --config playwright.drawai-phase1.config.ts
npm run test:release
npm run test:integration
```

Từ root với interpreter/dependencies hiện có:

```powershell
./.venv-phase1/Scripts/python.exe -m unittest discover -s training/tests -p "test_drawai*.py" -v
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_training
./.venv-phase1/Scripts/python.exe -m training.drawai.build_notebook
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_notebook
$env:DUST_DLL_DIRECTORY = 'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
./.venv-phase1/Scripts/python.exe -m training.drawai.export_onnx --release-dir artifacts/drawai/phase1/export-candidate --report-dir artifacts/drawai/phase1/export-audit --samples-dir artifacts/drawai/phase1/export-samples
./.venv-phase1/Scripts/python.exe tools/verify_phase1.py
```

Lệnh cuối tổng hợp archive hashes, source fingerprints, pixel parity, hashes model/data, các suite reports, export/parity và notebook bundle. Dependencies audit gồm requirements training, verification và notebook. Nó dựa vào archive/bằng chứng local đã tạo, không phải lệnh deploy trên máy người dùng cuối.

- [verification.json](../../artifacts/drawai/phase1/verification.json): nghiệm thu máy đọc được.
- [verification.log](../../artifacts/drawai/phase1/verification.log): kết quả audit tổng hợp.
- Các logs build/python/browser/notebook/export nằm trong `artifacts/drawai/phase1/`.
- [README](../../README.md): bảng chỉ ra chỗ update và hướng dẫn chạy hiện hành.

## 7. Giới hạn và bước tiếp theo

P1 chưa cài môi trường sạch trên máy khác, chưa full-train 4+3 run mới, chưa chạy Kaggle GPU, chưa thu dữ liệu riêng, chưa có link Kaggle public, chưa tích hợp DINOv3 và chưa deploy. Không suy rộng test PenDigits thành chất lượng người viết mới.

Filename release đang chạy vẫn là `drawai-swin-v1/model.onnx`, giữ nguyên trong P1 để tránh thay đồng thời contract release. P6 sẽ đổi tên đồng bộ exporter/loader/manifest/tests theo quy ước đã chốt, không chỉ rename file.

**Đủ điều kiện sang P2:** thêm DINOv3 ViT-S/16 backbone + head, renderer nét → ảnh và kiểm chứng runtime/export, giữ design hiện tại. Không cần dọn lại MotionSense trước P2.
