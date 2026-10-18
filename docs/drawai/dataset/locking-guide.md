# DrawAI P4 — Khóa dataset và bàn giao cho P5

Hiện phần kỹ thuật đã có, nhưng chưa có dataset người thật để khóa. Mục tiêu không đổi:250 mẫu pilot đã duyệt và6.000 mẫu main/30 người mới. [Hướng dẫn review](review-guide.md) là điều kiện trước lệnh lock.

## Khóa nội bộ sau khi đủ dữ liệu và review thật

Từ root:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.private_dataset lock --input data/private/inbox/main --review data/private/reviews/main-real-001/review.json --pilot-approval data/private/approvals/pilot-v1/pilot-approval.json --output data/private/locked/drawai-private-v1
```

Không có cờ bỏ qua quota/pilot/consent/review; output phải mới, không chồng inbox và không nằm trong web/public/dist. Chỉ lock đợt chính, không gọi pilot là train/test chính.

Các guard: schema/nhãn/thứ tự vòng/device/time; exact duplicate; quota từng người/lớp; plan/split chuẩn; review gắn đúng source/audit hash; đầy đủ xác nhận và near-group decisions; không trùng person_token trong main hoặc giữa pilot/main; không sao chép geometry/sample_id pilot; lý do loại hợp lệ. Kiểm tra nguồn/review chưa thay đổi sau khi ghi. Ghi trong thư mục staging riêng, verify đầy đủ rồi đổi tên thành thư mục cuối. Nếu lỗi giữa chừng, staging có thể được giữ để điều tra, không được coi là lock cuối.

## Gói sau khóa

```text
data/private/locked/drawai-private-v1/
  dataset-lock.json
  plan.json
  preprocessing.json
  review-summary.json
  DATASET_CARD.md
  development/
    manifest.csv                 chỉ train và validation
    raw/sample_<uuid>.json
    images/sample_<uuid>.png     RGB224 cho DINOv3
    swin-l128/sample_<uuid>.npy  float32[3,128]
  heldout-test.zip               test manifest + raw/PNG/NPY + custody note
```

Train4.000, validation1.000, test1.000; writer-disjoint20/5/5. Manifest dùng cùng sample_id/nhãn/split cho hai representation, SHA-256 raw/PNG/NPY và geometry/raster hashes. Không augmentation trong P4. Derived raw schema `drawai-private-drawing-v1` bỏ thời gian thu chi tiết, session_id và consent refs; không copy person_token/phiếu đồng ý. Hồ sơ gốc vẫn giữ private ngoài gói. Dataset card thật khi lock ghi counts/lớp/thiết bị/split và giới hạn, không bịa metric.

`dataset-lock.json` chứa bytes/hash toàn bộ file ngoài chính nó, metadata/hash plan, split/writers/class/device counts, hash nguồn/review, code preprocessing/renderer/locker và lock_sha256. Lock là mốc toàn vẹn nội bộ; `release_ready=false`, `published=false`. Kaggle/license/privacy review còn thuộc P8, không tự publish chỉ vì đã lock.

## Trusted hash và xác minh

Lưu lock_sha256 do lệnh lock in ra ở nơi độc lập, ví dụ run config đã rà soát/mốc bàn giao ngoài thư mục dataset. Không lấy hash từ file vừa bị nghi ngờ rồi coi là nguồn tin cậy. Checksum không phải chữ ký số và không chống được người sửa cả gói lẫn trusted hash.

```powershell
$drawaiLockHash='THAY_BANG_64_KY_TU_SHA256_DA_LUU_DOC_LAP'
./.venv-dinov3/Scripts/python.exe -m training.drawai.private_dataset verify --dataset data/private/locked/drawai-private-v1 --expected-lock-sha256 $drawaiLockHash
```

Verifier kiểm tra trusted metadata hash, file inventory không có dư/thiếu, hash từng file, plan/quota/writers và development manifest/paths. Nó hash nguyên heldout-test.zip mà **không giải nén/đọc nhãn test**. Thay PNG/NPY/raw/ZIP, thêm file hoặc đổi metadata làm verify lỗi; không sửa lock.json để hợp thức hóa dữ liệu mới. Sửa dataset cần phiên bản/review/lock mới trước training.

## Đưa vào training P5

```python
from training.drawai.private_dataset import iter_development

# trusted_hash đọc từ run config đã chốt độc lập.
for tensor, label, row in iter_development(dataset_root, 'train', trusted_hash, 'swin1d'):
    assert tensor.shape == (3, 128)
for tensor, label, row in iter_development(dataset_root, 'validation', trusted_hash, 'dinov3'):
    assert tensor.shape == (3, 224, 224)
```

Iterator đọc từng PNG/NPY để tránh giữ toàn bộ RGB224 tensor trong RAM. DINOv3 chuẩn hóa đúng contract raster đã lưu; labels/sample IDs khớp Swin. `load_development` là helper đọc toàn bộ Swin nhỏ vào RAM. Cả hai chỉ nhận train/validation; yêu cầu test bị chặn. Đây là nền cho P5, chưa train model nào ở P4.

## Giữ tập test

ZIP chỉ được **niêm phong bằng hash**, không mã hóa, không tự áp ACL hệ điều hành. Điều phối giữ ZIP ở nơi hạn chế truy cập và giữ trusted hash độc lập; không mở/giải nén để tune hoặc loại mẫu theo model. P5 phải khóa lựa chọn model/cấu hình trước đánh giá cuối; luồng đánh giá test sau lựa chọn chưa triển khai ở P4. Dataset QC theo nhãn yêu cầu không phải đánh giá model trên test. Quyền đọc file trực tiếp ngoài loader vẫn là trách nhiệm tổ chức thực nghiệm.

## Kiểm thử và trạng thái

```powershell
./.venv-dinov3/Scripts/python.exe -m unittest discover -s training/tests -p "test_drawai*.py" -v
cd web
npm run build
npx playwright test --config playwright.drawai-p4-regression.config.ts
cd ..
./.venv-dinov3/Scripts/python.exe tools/verify_phase4.py
```

Auditor cần log Python ở artifacts/drawai/phase4/python-tests.log, build.log và ảnh regression trong phase4/ui. Dữ liệu positive-lock test là hợp đồng miniature giả lập trong thư mục tạm; production read_plan vẫn yêu cầu250/6.000. Không dùng kết quả unit tests làm bằng chứng đã thu dữ liệu thật hoặc benchmark gói6.000 mẫu.
