# P4 — snapshot dữ liệu ảnh Paint35

Ngày 18/10/2026. **DONE local: đã khóa snapshot có kiểm tra toàn vẹn, với giới hạn tương đồng giữa các split được công khai.** Chưa train/export classifier mới. [Bằng chứng](../../artifacts/drawai/phase4-image64/verification.json) · [Hướng dẫn sử dụng](dataset/paint-image64-locking-guide.md).

## Kết quả thực tế

| Thành phần | Kết quả |
|---|---|
| Nguồn | 7.000 ảnh 64×64, 35 mã người; nguồn và đồng ý do chủ dữ liệu xác nhận |
| Main khóa | 6.000 ảnh, 30 người theo ID; cùng mẫu/split cho Swin1D ảnh và DINOv3 |
| Train | 4.000 ảnh, người 6–25; 400 ảnh/lớp |
| Validation | 1.000 ảnh, người 26–30; 100 ảnh/lớp |
| Test | 1.000 ảnh, người 31–35; 100 ảnh/lớp; ZIP riêng |
| Pilot ngoài main | 250 QC + 750 reserved, người 1–5 |
| Exact duplicates / ảnh trống | 0 / 0 |
| Near groups | 534 kiểm tra bằng pixel; 98 nhóm qua split được xem trong 5 contact sheets |
| Quyết định | Giữ 534 nhóm với cờ chưa xác minh độc lập; 0 loại, 0 đổi split |
| Model/design cũ | 58 file bảo vệ và 4 screenshot so với P0 được kiểm chứng |

Local: `data/private/locked/paint35-main-v1/`. Development chỉ chứa 5.000 PNG train/validation và manifest. Test ZIP có 1.002 members: 1.000 PNG, manifest, custody note. Không đưa NPZ gốc/array tổng 7.000 mẫu vào development.

Anchor nằm ngoài package tại `artifacts/drawai/phase4-image64/trusted-lock.json`:

```text
2c3a86c5be2b2e0f952a9c3390d56785473a82634338cec805903cbb1b3ef47a
```

## Đã thay đổi và kiểm chứng

- `training/drawai/paint_dataset.py`: review gắn SHA nguồn/manifest/plan/audit/provenance/preprocessing; kiểm tra 7.000 PNG và hai biểu diễn; khóa main bằng staging/rename; verify bằng external anchor; loader train/validation dùng chung ID cho hai model.
- `training/tests/test_drawai_paint_lock.py`: 17 tests mới về corruption, stale review, thiếu quyết định, split drift, quota thiếu, test/pilot guard, output/path và tuyên bố độc lập sai.
- `tools/verify_phase4_image64.py`: audit lock thật 6.000 mẫu, toàn bộ 5.000 development và 1.000 test PNG; ZIP CRC/inventory; loader không mở test ZIP; kiểm tra model/design.
- `tools/verify_phase4.py`: chọn audit ảnh khi có staging Paint; audit stroke cũ giữ làm lịch sử.
- **95 Python tests và 12 browser regression tests đạt; TypeScript/Vite build đạt.** Bốn ảnh desktop/mobile của `/` và `/drawai` giống pixel mốc P0. Không dùng test dữ liệu giả để tuyên bố khóa 6.000 ảnh nguồn người dùng cung cấp.

## Giới hạn của snapshot

98 nhóm qua split tính cả hai phần pilot. Trong main có **37 nhóm qua split: 34 train–test và 3 train–validation**. Một số mẫu rất giống dù khác mã người. Khác pixel không chứng minh khác lần viết; P4 giữ cờ `retain_flagged_similarity_unresolved`, không xác nhận các nhóm độc lập. Snapshot v1 dùng để tái lập so sánh hai model với giới hạn này; **chưa thể gọi là benchmark sạch rò rỉ**. Không loại/đổi test để nâng metric. Nếu cần benchmark nghiêm ngặt hơn, thu nguồn có truy vết hoặc lập protocol/group split mới trước thí nghiệm và giữ bản v1 làm lịch sử.

Người dùng xác nhận ảnh viết trong Paint, 35 người riêng biệt, quyền training/evaluation/Kaggle và dòng “Synthetic” là metadata đóng gói sai. Giữ nguyên NPZ và provenance riêng; không có xác minh danh tính độc lập, stroke order, timestamps hay dữ liệu nhấc bút.

Split theo người chốt hồi cứu trước training. P4 xem ảnh/nhãn test để QC toàn vẹn/tương đồng, không infer/tuning. ZIP không mã hóa hoặc kiểm soát truy cập; NPZ gốc đã public Kaggle. Test isolation từ P5 là quy trình, không phải bảo mật kỹ thuật. Chưa có accuracy/F1 của hai classifier mới.

## Chuyển P5

Train Swin1D ảnh từ đầu và DINOv3 pretrained backbone+head qua loader snapshot đã khóa, chỉ train/validation. Chọn hyperparameters/model bằng validation; test dành cho đánh giá cuối. Artifact xuất riêng dự kiến:

- `drawai_swin1d-image_paint35-main-v1_rows64_fp32_v1.onnx`
- `drawai_dinov3-vits16_paint35-main-v1_rgb224_fp32_v1.onnx`

Đây là tên dự kiến, chưa phải artifact đã tồn tại. [P4 stroke v1 trước khi có NPZ](../../artifacts/drawai/phase4-image64/pre-rerun/phase4-report.md) giữ làm lịch sử.
