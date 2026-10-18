# P4 — sử dụng snapshot ảnh Paint35

Protocol hiện tại dùng ảnh 64×64, code `training/drawai/paint_dataset.py`. `private_dataset.py` là protocol JSON nét bút cũ. Dùng môi trường `.venv-dinov3` hiện có; không cần cài thêm dependencies.

## File bàn giao

`data/private/locked/paint35-main-v1/` có `development/manifest.csv`, 5.000 PNG train/validation, `heldout-test.zip`, `plan.json`, `preprocessing.json`, `provenance.json`, `review-summary.json`, `dataset-lock.json`. Pilot 1.000 ảnh không nằm trong main. Không copy NPZ gốc/arrays tổng vào development. Source, review chi tiết, contact sheets và package bị Git ignore; GitHub chỉ có code/docs/summary/hash.

## Kiểm tra bản đã khóa

Chạy từ repository root:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_dataset verify --dataset data/private/locked/paint35-main-v1 --expected-lock-sha256 2c3a86c5be2b2e0f952a9c3390d56785473a82634338cec805903cbb1b3ef47a
./.venv-dinov3/Scripts/python.exe tools/verify_phase4.py
```

Lệnh thứ hai cần bằng chứng tests/build/screenshots đã chạy trong `artifacts/drawai/phase4-image64/`. External anchor tại `trusted-lock.json` phải giữ riêng ngoài package. Không lấy hash từ chính lock cần kiểm tra để coi là chứng minh toàn vẹn. Checksum không phải chữ ký tác giả.

## Tái tạo review/lock mới

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_dataset review --prepared data/private/prepared/paint-image64-v1 --source data/private/external/paint35/So_viet_tay_35_nguoi.npz --output data/private/reviews/paint35-p4-v2
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_dataset lock --prepared data/private/prepared/paint-image64-v1 --source data/private/external/paint35/So_viet_tay_35_nguoi.npz --review data/private/reviews/paint35-p4-v2/review.json --output data/private/locked/paint35-main-rebuild-v2
```

Output phải mới, không chồng input và nằm ngoài public/dist; không ghi đè bản cũ. Lần tạo mới có timestamp/hash khác; đối chiếu source/plan/manifest, kiểm tra rồi lưu anchor mới. Đây là rebuild cùng policy v1, không tự đổi dataset protocol theo tên thư mục. Review kiểm tra nguồn NPZ cố định, quota, PNG/pixel/hashes cả hai preprocessing cho 7.000 ảnh; ràng buộc source/manifest/audit/plan/provenance/contract. Lock ghi staging, kiểm tra đầu vào lần nữa rồi verify và rename final. `.staging-*` sau lỗi không phải lock hoàn chỉnh.

## Dùng cho P5

```python
from training.drawai.paint_dataset import iter_development

anchor = '2c3a86c5be2b2e0f952a9c3390d56785473a82634338cec805903cbb1b3ef47a'
for x, label, row in iter_development(
    'data/private/locked/paint35-main-v1', 'train', anchor, 'swin-rows64'
):
    pass  # float32[64,64]: 64 column features × 64 spatial row positions
```

DINOv3 dùng `dinov3-rgb224`, float32[3,224,224], cùng ID/label/split. Swin ảnh không tương thích checkpoint nét bút cũ [3,128]. Loader chỉ nhận train/validation, kiểm tra toàn bộ package và từng tensor; không mở ZIP test. Iterator kiểm tra lock một lần mỗi lượt, không tạo iterator mới cho từng batch. Verify là kiểm tra toàn vẹn đầy đủ, không phải đường đọc tối ưu cho mỗi batch. Không dùng NPZ/staging P3 thay loader snapshot khi train.

## Near groups và test custody

534 nhóm được audit pixel từng cặp. Giữ với trạng thái `retain_flagged_similarity_unresolved`: không có bản trùng pixel hoàn toàn nhưng chưa xác minh độc lập nguồn vẽ. 98 nhóm qua split được xem trong 5 bảng ảnh; 37 nhóm nối các tập main, gồm 34 train–test và 3 train–validation. Không dùng model để lọc, không loại hay đổi split. Chủ dữ liệu đã xác nhận nguồn con người/35 mã người riêng biệt/quyền dùng, không có chứng minh danh tính độc lập. Báo cáo accuracy phải nêu giới hạn tương đồng; chưa đủ chứng minh tổng quát hóa sang nét viết hoàn toàn mới.

P4 mở test để QC đóng gói/tương đồng; không chạy inference. Từ P5 chỉ tuning bằng train/validation, chốt lựa chọn trước đánh giá test. ZIP chỉ hash-sealed, không mã hóa/ACL; nguồn NPZ đã public Kaggle nên không có test bí mật. Giữ archive ngoài training input và quản lý quyền đọc nếu làm nhóm. Không tự gán license dữ liệu mới.
