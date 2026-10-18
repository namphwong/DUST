# DrawAI — Báo cáo P4: thu chính thức, rà soát và khóa dataset

**Ghi chú sau P3 ảnh (18/10/2026):** Báo cáo này là lịch sử của protocol nét bút v1. Hiện NPZ7.000 ảnh đã được chuẩn bị theo xác nhận chủ dữ liệu,6.000 ảnh main staging có sẵn; lock image-protocol chưa triển khai. Xem [P3 mới](phase3-report.md). Các số0 mẫu bên dưới mô tả mốc trước file NPZ.

Ngày18/10/2026. **P4 hoàn thành phần công cụ kỹ thuật; chưa hoàn thành thu dữ liệu thực hoặc khóa dataset thật.** Trạng thái `TECHNICAL_READY_WAITING_FOR_HUMAN_DATA`, `phase_complete=false`. Pilot P3 hiện0/250 mẫu, main0/6.000 mẫu; không có hồ sơ pilot approval hoặc dataset-lock thật.

## 1. Công việc đã thực hiện

- Kiểm tra inbox pilot/main thực trên máy: chưa có JSON phiên người tham gia. Giữ P3 ở trạng thái chờ pilot, không tự xác nhận thay người hoặc dùng dữ liệu sinh để lấp quota.
- Triển khai `training/drawai/private_dataset.py`: draft review gắn với nguồn, phê duyệt pilot sau kiểm tra, khóa main, verify integrity và loader/iterator cho hai model.
- Nâng `private_data.py`: từ chối JSON có key trùng; nhóm gần trùng đầy đủ kể cả khi danh sách cặp bị cắt ở5.000.
- Tạo draft chưa xác nhận tại `data/private/reviews/p4-pilot-draft-v1` và `p4-main-draft-v1`; tất cả confirmations=false. Readiness hiển thị thiếu dữ liệu/quota/pilot approval và chưa có review thực.
- Bổ sung hướng dẫn review, lock, trusted hash và giữ tập test; cập nhật README/status/roadmap.

## 2. Điều kiện nghiệm thu dataset không thay đổi

| Phần | Quota | Thực có | Trạng thái |
|---|---:|---:|---|
| Pilot |5 người×10 chữ số×5 lần=250|0|Chưa thu/chưa phê duyệt |
| Main train |20 người×200=4.000|0|Chưa thu |
| Main validation |5 người×200=1.000|0|Chưa thu |
| Main test |5 người×200=1.000|0|Chưa thu |

Plan P3 vẫn là `drawai-private-plan-v1`: P001–P005 chỉ pilot-dev; M001–M020 train, M021–M025 validation, M026–M030 test. Không đổi split theo nét hoặc model, không dùng người pilot làm người main. Phương án2.000 mẫu chưa được kích hoạt. Các mã dự kiến không chứng minh đã có người tham gia.

## 3. Review ràng buộc với nguồn dữ liệu

Lệnh `review` tạo review.json, source-inventory.json, audit.json, readiness.json vào thư mục mới. Hash nguồn gồm đường dẫn tương đối, bytes và SHA-256 từng JSON; review còn gắn plan và toàn bộ kết quả audit. Thay nguồn, thêm snapshot hoặc sửa policy làm review cũ không hợp lệ; không sửa lại hash để bỏ qua lỗi.

Điều phối phải xác nhận thực tế:

- Nguồn người thật và consent; người không trùng; mã/split được cấp trước thu; không loại theo dự đoán AI; phiên lỗi được log/cách ly và thu lại.
- Mỗi người có token private ổn định dùng qua pilot/main, không chứa tên/email trong gói chia sẻ. Công cụ so token trùng trong main và pilot/main; đổi mã không được coi là người mới.
- Người đã rút hoặc chưa kiểm tra consent không thể khóa bằng review hợp lệ.
- Từng nhóm gần trùng có quyết định `keep_distinct_verified` cùng lý do sau xem nét gốc; nếu là copy/lỗi thì cách ly/thu lại, sinh review mới.
- Exclusion log giữ session ID, file hash nguồn, lý do protocol, replacement session. Không chấp nhận reason do model dự đoán sai/accuracy thấp.

Nhóm gần trùng là connected component của mọi cặp dHash/Hamming≤2. Dù report chỉ lưu5.000 cặp chi tiết, nhóm bao phủ toàn bộ cặp đã quét; không còn bỏ sót review khi report bị giới hạn. Kiểm thử trường hợp110 hình dịch chuyển cùng glyph có5.995 cặp xác nhận nhóm vẫn đủ110 sample IDs. Gần trùng chỉ là cờ review, không phải bằng chứng sao chép hay cùng người.

Các xác nhận thủ công được lưu và kiểm tra tính nhất quán; phần mềm không độc lập chứng minh danh tính hoặc consent. Hồ sơ thật/mapping/token phải được điều phối quản lý ngoài Git/Kaggle. Không tự điền review để làm lệnh thành công.

## 4. Pilot approval và khóa main

`approve-pilot` chỉ tạo approval sau khi đủ250 mẫu theo policy, audit không lỗi, review đầy đủ và gắn đúng nguồn. Approval ghi hash nguồn/review, token người đã băm, geometry hashes và sample IDs. Main phải tham chiếu đúng approval đó; chặn token/người hoặc mẫu/geometry pilot bị tái dùng trong main. Approval là bản ghi xác nhận nội bộ, không phải chữ ký số/chứng thư danh tính.

`lock` không có cờ bỏ qua quota/pilot/consent/review. Nó audit lại đầy đủ, kiểm tra source/review bindings và mọi gate trước khi ghi. Output không chồng input, không ghi đè thư mục cũ và không vào public/dist. Ghi trong thư mục staging rồi verify toàn bộ trước khi đổi tên thành output cuối; nếu lỗi giữa chừng, staging có thể được giữ để điều tra nhưng không phải lock cuối.

Gói sau khi khóa **thành công trên dữ liệu thật sau này**:

```text
data/private/locked/drawai-private-v1/
  dataset-lock.json
  plan.json
  preprocessing.json
  review-summary.json
  DATASET_CARD.md
  development/
    manifest.csv
    raw/sample_<uuid>.json
    images/sample_<uuid>.png
    swin-l128/sample_<uuid>.npy
  heldout-test.zip
```

Development chỉ chứa train/validation; ZIP chứa test manifest/raw/PNG/NPY. Cùng sample_id, label, writer và split dùng cho cả Swin1D và DINOv3; manifest có paths/hash từng file. Không augmentation trong P4. Derived raw schema `drawai-private-drawing-v1` bỏ timestamp thu chi tiết, session ID và consent refs; token và hồ sơ consent không vào package. Dữ liệu gốc giữ private để truy nguyên.

Lock metadata có file inventory bytes/hash, plan/quota/writers/class/device counts, hash nguồn/review, code preprocessing/renderer/locker và lock_sha256. DATASET_CARD được tạo từ số liệu thật khi lock, không bịa metric. Gói chỉ khóa nội bộ: `release_ready=false`, `published=false`; privacy/license và Kaggle public vẫn thuộc P8.

Hiện **chưa có thư mục dataset khóa thật**. Cấu trúc trên là hợp đồng output, không phải khẳng định đã có6.000 mẫu.

## 5. Verify và giữ test khỏi tuning

Verifier yêu cầu trusted lock SHA-256 đã giữ độc lập, kiểm tra metadata/file inventory/hashes, plan/quota/writers và development manifest/path. Thêm file, sửa raw/PNG/NPY/ZIP hoặc metadata làm verify lỗi. Không dùng hash lấy từ gói đang bị nghi ngờ như nguồn tin cậy; checksum không phải chữ ký và không chống người thay cả trusted hash.

Held-out ZIP chỉ **niêm phong bằng hash, không mã hóa hoặc áp ACL tự động**. Điều phối phải giữ quyền truy cập/ZIP ở nơi riêng, không mở để tune. Verifier hash nguyên ZIP mà không giải nén hoặc đọc nhãn. Dataset QC theo nhãn đã yêu cầu không phải đánh giá model trên test. Luồng đánh giá cuối sau khóa lựa chọn model thuộc P5, chưa triển khai ở P4.

`load_development` và `iter_development` chỉ nhận train/validation; yêu cầu test bị từ chối. Iterator trả từng mẫu Swin[3,128] hoặc DINOv3[3,224,224] float32 với cùng label/row, tránh giữ toàn bộ tensor ảnh trong RAM. Kiểm thử chứng minh hai representation có cùng sample IDs, và loader/iterator không mở ZIP test. Chưa bắt đầu training P5.

## 6. Kiểm chứng phần mềm thực chạy

| Hạng mục | Kết quả |
|---|---|
| Python toàn DrawAI | **68/68 PASS**, gồm12 tests mới P4 |
| Build TypeScript/Vite | PASS |
| Browser hồi quy root/alias/DrawAI | **12/12 PASS** |
| Positive review→pilot approval→lock→verify | PASS với hợp đồng giả lập nhỏ trong thư mục tạm |
| Metadata/file tamper, stale source, quota/consent/review/pilot/identity gates | PASS |
| Test sealed; Swin/DINOv3 IDs/labels và shape | PASS |
| Near-group đầy đủ vượt giới hạn5.000 cặp | PASS |
| CLI thực với inbox rỗng | Từ chối approve-pilot/lock, không tạo output giả |
| Design root/alias desktop/mobile | Pixel-identical P0 |
| Code/model/data được bảo vệ | Hash giữ nguyên |
| Pilot/main người thật, review và lock thật | **NOT RUN — chưa có dữ liệu** |

Tests positive dùng policy miniature được patch **chỉ trong unit test** để chạy toàn pipeline nhanh. Production `read_plan` vẫn chỉ chấp nhận policy250 pilot/6.000 main, không có chế độ bypass. Test fixture dùng tạm, không đưa vào inbox hoặc số liệu người thật. Chưa benchmark gói6.000 mẫu, chưa clean-install máy khác, chưa nghiệm thu touch/pen thật và chưa có dataset accuracy.

Bằng chứng: `artifacts/drawai/phase4/verification.json`, `python-tests.log`, `product-regression.json`, `build.log`, `current-pilot-audit.json`, `current-main-audit.json`, `ui/`; các log kiểm chứng chưa phải dữ liệu nghiên cứu. Snapshot code trước P4 tại `artifacts/drawai/phase4/pre-p4`.

## 7. File thay đổi và bàn giao

| File | Thay đổi |
|---|---|
| `training/drawai/private_data.py` | Duplicate JSON-key guard và complete near groups |
| `training/drawai/private_dataset.py` | Source inventory, draft review, validation, pilot approval, atomic final lock, verify, loaders/iterators |
| `training/tests/test_drawai_private_lock.py` |12 tests P4: positive pipeline và các negative gates |
| `training/tests/test_drawai_private.py` | Đóng file manifest bằng context manager trong test cũ |
| `web/playwright.drawai-p4-regression.config.ts` | Chạy suite12 tests sản phẩm với report riêng P4 |
| `tools/verify_phase4.py` | Audit bằng chứng, nguồn thực, negative CLI probes, model/design preservation |
| `docs/drawai/dataset/review-guide.md` | Hướng dẫn review có người chịu trách nhiệm, không tự ký |
| `docs/drawai/dataset/locking-guide.md` | Lệnh lock/verify, trusted hash, package/test custody và bàn giao P5 |
| `README.md`, `status.md`, `two-model-roadmap.md` | Ghi rõ technical-ready và thiếu dữ liệu thật |

Không thay giao diện/trang thu/model release. Không công bố Kaggle hay ghi consent thay người dùng.

## 8. Việc còn lại để đóng P4

1. Thu pilot250 mẫu từ5 người và thực hiện review P3 thực tế; phê duyệt pilot bằng CLI sau xác minh hồ sơ.
2. Thu main6.000 mẫu từ30 người mới theo quota/split đã khóa; chỉ dùng nguồn người thật và thiết bị ghi đúng thực tế.
3. Audit, kiểm tra consent/danh tính/trùng/gần trùng; log/cách ly/thu lại các lỗi protocol, không loại theo model.
4. Hoàn tất review gắn đúng source hashes; lock nội bộ, giữ trusted hash/test ZIP độc lập và verify trước P5.

Chi tiết lệnh: [Review guide](dataset/review-guide.md), [Locking guide](dataset/locking-guide.md). Trạng thái hiện tại **TECHNICAL_READY_WAITING_FOR_HUMAN_DATA**, không có dataset-lock thật và chưa đóng toàn bộ P4.
