# DrawAI — Báo cáo P3 làm lại với dữ liệu ảnh Paint

Ngày18/10/2026. **P3 hoàn thành local trong phạm vi protocol ảnh `drawai-paint-image64-v1`**: đã nhận và kiểm toán7.000 ảnh, ghi xác nhận nguồn từ người dùng, chốt split theo người, xem250 ảnh pilot QC và chuẩn bị input có hash cho hai mô hình. Trạng thái `DONE_LOCAL_IMAGE_DATA_PREPARATION`, `phase_complete=true`. Chưa khóa dataset P4 hoặc train model P5.

Báo cáo P3 trước file NPZ được lưu tại `artifacts/drawai/phase3-image64/pre-rerun/phase3-report.md`; kết quả nét bút v1 không dùng để mô tả provenance/đầu vào của bộ ảnh này.

## 1. Nguồn dữ liệu và xác nhận của chủ dữ liệu

File nhận: `C:/Users/admin/Downloads/So_viet_tay_35_nguoi.npz`; bản sao giữ nguyên tại `data/private/external/paint35/So_viet_tay_35_nguoi.npz`.

- Dung lượng **2.828.016 bytes**; SHA-256 `c594927b59509bb62cb185cf5e64034ee641b046798f0f192d6155fd91f71a64`.
- Người dùng xác nhận ảnh do người thật viết bằng Paint, chỉ còn ảnh NPZ; mỗi person_id1–35 là một người khác nhau.
- Người dùng xác nhận đồng ý sử dụng cho training, evaluation và công bố Kaggle; dòng “Synthetic” trong metadata gốc là thông tin đóng gói sai.
- Nguyên văn description gốc được giữ trong audit, không chỉnh file nguồn: `Synthetic handwritten digits: 35 people, each wrote every digit 0-9 twenty times. Shape of images: (7000, 64, 64), uint8, white bg (255), black ink (0).`.
- Provenance riêng gắn source SHA ghi đính chính/xác nhận trong chat ngày18/10/2026. Phần mềm không độc lập xác minh danh tính hoặc hồ sơ consent; không bịa chữ ký/thời gian thu/thiết bị.
- NPZ đã public ở [Kaggle](https://www.kaggle.com/datasets/namphwong/drawai-handwritten-digits-placeholder), version2. Việc upload trước đây không tự chứng minh đã hoàn thành P4/P5.

## 2. Cấu trúc và số liệu thực đo

| Trường | Shape / dtype | Ý nghĩa |
|---|---|---|
| images |7000×64×64, uint8|Ảnh grayscale64; nền trắng/nét đen |
| labels |7000, int8|Nhãn0–9, mỗi nhãn700 ảnh |
| person_ids |7000, int8|35 mã1–35, mỗi mã200 ảnh |
| sample_ids |7000, int8|Lần viết1–20 trong mỗi người/nhãn; không duy nhất toàn cục |
| description |Unicode scalar|Metadata gốc có dòng lỗi đã được chủ dữ liệu xác nhận |

Mỗi cặp người/chữ số đủ20 ảnh; composite `(person_id,label,repetition)` duy nhất7.000/7.000. Range pixel0..255, đúng uint8. Không original stroke coordinates, thứ tự nét, pen-lift hoặc timestamp từng điểm.

## 3. Điều chỉnh phạm vi input và kế hoạch

Bộ dữ liệu thực là ảnh nên không thể dùng input x/y/stroke_start của Swin PenDigits cũ như thể còn nét bút gốc. P3 mới tạo **hai biểu diễn ảnh của cùng mẫu**:

| Nhánh | Input | Tiền xử lý | Hệ quả |
|---|---|---|---|
| Swin Transformer1D ảnh |float32[64,64] CL|`(255-image)/255`, transpose;64 đặc trưng cột cho64 token hàng|Chuỗi không gian theo hàng, không phải thời gian viết; phải train từ đầu |
| DINOv3 ViT-S/16 |float32[3,224,224] NCHW|Repeat grayscale→RGB, bilinear64→224, chia255/mean-std ImageNet|Backbone pretrained P2 dùng tiếp; head sẽ train ở P5 |

Không phục dựng nét bút giả. Khóa contract riêng [paint-image64-contract.json](collection/paint-image64-contract.json), không gắn nhãn ảnh resize là `drawai-raster-v1`. Model Swin1D cấu hình seq_len64/in_chans64/patch2/window8/embed32/depths[2,2,2]/heads[2,4,8] đã forward/backward ra[2,10] hữu hạn. Đây là kiểm tra khả thi input, không phải training đầy đủ/export release.

Design, routes nhận dạng và checkpoint/ONNX Swin nét bút đang phục vụ giữ nguyên. Model ảnh tương lai có tên riêng `drawai_swin1d-image_paint35-main-v1_rows64_fp32_v1.onnx`; DINOv3 `drawai_dinov3-vits16_paint35-main-v1_rgb224_fp32_v1.onnx`. P6/P7 cần export/benchmark và chuyển canvas sang input ảnh đúng contract trước khi đổi model mặc định.

## 4. Split theo người, chốt trước huấn luyện

| Person ID nguồn | Repetition | Split | Mẫu |
|---|---|---|---:|
|1–5|1–5|pilot-qc|250|
|1–5|6–20|pilot-reserved|750|
|6–25|1–20|train|4.000|
|26–30|1–20|validation|1.000|
|31–35|1–20|test|1.000|

Mỗi nhóm chính có20/5/5 người, train400 ảnh/lớp, val100/lớp, test100/lớp. Không overlap người giữa pilot và main hoặc giữa train/val/test.750 ảnh pilot-reserved để riêng, không bổ sung vào main. Đây là **partition hồi cứu sau thu**, được chốt trước training/model selection, không tuyên bố gán split trước khi người dùng viết trong Paint.

Policy [paint-image64-plan.json](collection/paint-image64-plan.json), SHA `cef746927eb0588ad1bc42feafbe565e81d38bff6b613f8fe6452746924f3fa0`. Chưa augmentation, chưa dùng prediction/test metric chọn split/cấu hình, chưa full-train.

## 5. Kiểm tra dữ liệu và pilot QC

- Quota/schema/dtype/nhãn/range/composite IDs đạt trên toàn bộ7.000 mẫu.
- **0 exact duplicate ảnh**, không có blank/low-contrast candidate theo tiêu chí ink<192 ít hơn4 pixels hoặc dynamic range<16.
- Đã xem đủ250 ảnh pilot qua5 contact sheets, mỗi người50 ảnh/10 nhãn/5 lần. Không thấy ảnh trống, render mất/cắt chữ số rõ rệt; đây là QC thị giác, không phải accuracy/model benchmark và không phải pilot live của `/collect`.
- Kiểm tra gần trùng trên ink ROI trong cùng nhãn: dHash Hamming≤2 và cosine≥0,995 của ink32×32. Có **1,063 cặp / 534 nhóm**, **98 nhóm** xuất hiện qua các split.
- Near-duplicate là cờ cho review P4: các lần viết cùng chữ số có thể gần nhau hợp lệ. Không tự coi các nhóm là copy/leakage hoặc loại mẫu theo model. Chưa có quyết định giữ/loại nguồn ảnh trong P3.

Toàn bộ nhóm nằm trong `data/private/prepared/paint-image64-v1/audit.json`. Không công bố tên/định danh người từ các mã số và không tự suy tuổi/thiết bị/thời điểm thu.

## 6. Artifact và khả năng tái tạo

```text
data/private/prepared/paint-image64-v1/
  images.npy / labels.npy / person_ids.npy / sample_ids.npy
  images64/                          7.000 PNG giữ nguyên pixel nguồn
  manifest.csv                       sample_id/source_index/label/person/repetition/split/hash
  audit.json / provenance.json
  plan.json / preprocessing.json / dataset.json
  pilot-qc/person-001.png ... person-005.png
```

Mỗi dòng manifest có hash PNG, pixel nguồn, Swin row tensor, RGB DINO224 và DINO normalized tensor. Auditor đã tái tạo/đối chiếu **7.000/7.000 dòng** và so arrays/PNG với NPZ gốc. Không xuất tọa độ JSON/timestamp không có trong nguồn. Dữ liệu raw/prepared ở `data/private/`, bị Git ignore; không copy vào public/dist.

`training/drawai/paint_data.py` safe-load NPZ bằng `allow_pickle=False`, giới hạn nội dung archive và pin source hash; xác minh provenance, quota và dữ liệu trước build. Hai iterator lấy cùng sample IDs/labels; chỉ cho pilot-qc/train/validation, từ chối test/pilot-reserved. Staging còn toàn bộ source/manifest, chưa là gói test niêm phong P4 và không tự áp ACL.

Tái tạo theo [paint-image64-guide.md](collection/paint-image64-guide.md); [dataset card](collection/paint-image64-dataset-card.md) ghi nguồn, xác nhận đính chính và giới hạn. Output tồn tại sẽ không bị ghi đè. Bản build khảo sát dHash toàn ảnh trước khi chốt QC được giữ riêng dưới `_archive/`, không phải bản đang dùng.

## 7. Kiểm chứng đã chạy

| Hạng mục | Kết quả |
|---|---|
| Python toàn DrawAI |**78/78 PASS**, gồm10 tests ảnh mới |
| Build TypeScript/Vite |PASS |
| Browser DrawAI/root/alias |**12/12 PASS** |
| Source SHA/NPZ schema/quota/metadata confirmation |PASS |
| Arrays/PNG/two-model tensors |7.000/7.000 hash rows khớp |
| Pilot QC |250 ảnh/5 contact sheets đã xem |
| Row-pixel Swin1D forward/backward |PASS, chưa train/export |
| Guard test/manifest/PNG tamper |PASS |
| Model/design cũ |Hash giữ nguyên, desktop/mobile root/alias pixel-identical P0 |
| P4 image lock / P5 training / private test evaluation |Chưa thực hiện |

Bằng chứng: `artifacts/drawai/phase3-image64/verification.json`, `data-audit-summary.json`, `pilot-visual-review.json`, logs/suite JSON và `ui/`. Không dùng fixture unit tests làm mẫu người thật; số7.000 được lấy từ NPZ nguồn do người dùng xác nhận, không từ giả lập.

## 8. Code/tài liệu đã thay

- `training/drawai/paint_data.py`: audit, plan, preprocessing và build/iterators ảnh.
- `training/tests/test_drawai_paint_data.py`: shape/axis, RGB/norm, safe NPZ, metadata scope, writer split, tamper/test guards, Swin forward/backward.
- `tools/verify_phase3_image64.py`: audit nguồn đầy đủ và report P3 mới.
- `web/playwright.drawai-p3-image64.config.ts`: regression report riêng; không sửa code trang chính.
- `docs/drawai/collection/paint-image64-*`: protocol, input contract, split policy và dataset card; `two-model-contract.json`v4 ghi hai nhánh dữ liệu ảnh.
- `README.md`, `status.md`, `two-model-roadmap.md`, ghi chú lịch sử P4 được cập nhật: hiện đã có6.000 main staging, không còn ghi “chưa có dữ liệu” cho trạng thái hiện tại.

## 9. P4 tiếp theo

Cần triển khai image-protocol review/lock trên manifest mới, rà534 near-groups (98 cross-split), đối chiếu provenance/xác nhận nguồn, khóa dataset/manifest hash và tách test trước training. CLI `private_dataset.py` hiện có dành cho JSON nét bút v1, không dùng trực tiếp để khóa file ảnh này. Không chạy P5 bằng checkpoint/loader nét bút cũ như thể input tương thích.

P3 mới kết thúc ở **chuẩn bị dữ liệu ảnh đã kiểm toán**, không đồng nghĩa P4 khóa/P5 train/P6 export/P7 deploy/P8 hoàn tất dataset public.
