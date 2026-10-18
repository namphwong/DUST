# DrawAI — Protocol P3 cho ảnh Paint

Phiên bản `drawai-paint-image64-v1`,18/10/2026. Áp dụng cho `So_viet_tay_35_nguoi.npz` SHA-256 `c594927b59509bb62cb185cf5e64034ee641b046798f0f192d6155fd91f71a64`.

## Nguồn và đính chính metadata

Chủ dữ liệu xác nhận trực tiếp trong chat:35 mã person_id là35 người thật khác nhau, ảnh do người viết bằng Paint, đồng ý training/evaluation/Kaggle; dòng “Synthetic handwritten digits” trong description gốc là lỗi đóng gói. File gốc được giữ nguyên, xác nhận lưu trong provenance riêng với source SHA. Đây là lời xác nhận của chủ dữ liệu, không phải xác minh danh tính độc lập hoặc chữ ký điện tử. Không tự tạo thời gian thu, thiết bị hay tọa độ nét bút còn thiếu.

NPZ chứa images uint8[7000,64,64], labels int[7000], person_ids1..35, sample_ids1..20 (lần viết), description Unicode scalar. Mỗi người có20 ảnh/mỗi chữ số,10 chữ số; composite(person,label,repetition) duy nhất. `sample_ids` không duy nhất toàn bộ file; sample ID chuẩn là paint35_pNNN_dD_rRR, đi kèm source_index để truy nguyên chính xác.

## Policy mới, riêng với protocol nét bút v1

| Người nguồn | Lần viết | Nhóm | Số ảnh |
|---|---|---|---:|
|1–5|1–5|pilot-qc|250|
|1–5|6–20|pilot-reserved|750|
|6–25|1–20|train|4.000|
|26–30|1–20|validation|1.000|
|31–35|1–20|test|1.000|

Split được chốt hồi cứu sau khi đã có ảnh, trước huấn luyện/chọn mô hình. Không nói rằng mã/split này đã gán trước lúc thu trong Paint. Pilot là kiểm tra chất lượng ảnh/metadata sau thu; không phải pilot thao tác live của trang `/collect`. Toàn bộ người1–5 không vào train/val/test.750 ảnh reserved không tự bổ sung vào main. Không augmentation/test tuning/PenDigits mixing ở P3.

## Hai biểu diễn từ cùng ảnh

- Swin1D ảnh: chuẩn hóa float32 `(255-image)/255`, transpose thành[64,64] theo CL;64 đặc trưng cột cho mỗi token hàng,64 vị trí hàng từ trên xuống. Đây là chuỗi không gian, không phải thứ tự viết. SwinTransformer_1D với seq_len64/in_chans64/patch2/window8/embed32/depths[2,2,2]/heads[2,4,8] đã forward/backward được. Cần train từ đầu; checkpoint Swin nét bút x/y/stroke_start không tương thích input.
- DINOv3: lặp grayscale thành RGB64, Pillow bilinear lên224, không crop/bbox; float32 NCHW chia255 và mean[.485,.456,.406]/std[.229,.224,.225]. Tách version khỏi `drawai-raster-v1` (renderer từ nét bút).
- Cùng sample IDs/labels/split cho cả hai. Giữ nguyên design và model Swin nét bút đang phục vụ; P5/P6/P7 train/export/tích hợp nhánh ảnh riêng, cần kiểm tra lại runtime sau export.

## Tái tạo từ root

Local source không sửa nằm trong `data/private/external/paint35/So_viet_tay_35_nguoi.npz`. Plan/provenance local ở `data/private/plans/paint-image64-v1/`; bản xác nhận nguồn không có định danh cá nhân ở `docs/drawai/collection/paint-image64-provenance.json` để tái tạo từ source đã public. Cả raw và prepared bị Git ignore.

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_data --source data/private/external/paint35/So_viet_tay_35_nguoi.npz --provenance docs/drawai/collection/paint-image64-provenance.json --output data/private/prepared/paint-image64-v1
```

Output phải trống/mới; không ghi đè dữ liệu cũ và không đặt vào public/dist. Clone repo không có source riêng bị ignore; tải NPZ từ link Kaggle đã công bố và đối chiếu SHA trước khi chuẩn bị. Provenance thật phải do chủ dữ liệu xác nhận, không dùng fixture unit tests.

Output gồm images.npy/labels.npy/person_ids.npy/sample_ids.npy,7.000 PNG grayscale64, manifest.csv, audit.json, provenance.json, plan.json, preprocessing.json, dataset.json và5 contact sheets của250 ảnh pilot. PNG giữ nguyên pixel nguồn. Manifest có SHA từng PNG/pixel/tensor Swin/RGB DINO/tensor DINO. Không xuất JSON nét bút, không suy thứ tự nhấc bút từ ảnh.

## Kiểm tra và đọc input

```powershell
./.venv-dinov3/Scripts/python.exe -m unittest discover -s training/tests -p "test_drawai*.py" -v
./.venv-dinov3/Scripts/python.exe tools/verify_phase3_image64.py
```

Auditor cần evidence test/build/browser mới tại artifacts/drawai/phase3-image64. Verify7.000 PNG/tensor rows với source và manifest; bảo toàn code/model/design. Đây là kiểm toán dataset/QC, không evaluate model trên test. Không dùng các ảnh test để chọn preprocessing/hyperparameters/model.

```python
from training.drawai.paint_data import iter_prepared
for values,label,row in iter_prepared('data/private/prepared/paint-image64-v1','train','swin-rows64'):
    assert values.shape==(64,64)
for values,label,row in iter_prepared('data/private/prepared/paint-image64-v1','validation','dinov3-rgb224'):
    assert values.shape==(3,224,224)
```

Iterator chỉ cho pilot-qc/train/validation, chặn test/pilot-reserved. Staging còn toàn bộ source/manifest, chưa có test ZIP/OS access control/trusted P4 lock; điều phối vẫn phải hạn chế truy cập trực tiếp. P4 sẽ thêm nhánh ảnh, không chạy CLI lock JSON nét bút v1 lên dữ liệu ảnh.

## QC và phần P4

Đã kiểm tra counts/quota/composite IDs, toàn bộ7.000 hash rows, không ảnh exact duplicate/blank-low contrast. Near-duplicate dùng ink ROI, trong cùng nhãn, dHash≤2 kèm cosine≥0,995 trên32×32. Có1.063 cặp/534 nhóm,98 nhóm qua các split. Đây là cờ rà soát; nét của cùng chữ số có thể tương tự nhau hợp lệ. Không tự loại, không dùng prediction để lọc. P4 cần xem nhóm, xử lý/cách ly lỗi protocol theo bằng chứng, khóa manifest và tách test trước P5.
