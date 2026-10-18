# DrawAI private-v1 — Dataset card DRAFT (P3)

Trạng thái: chưa có dữ liệu người thật, chưa khóa dataset, chưa public Kaggle. Tên private-v1 chỉ nguồn tự thu trước công bố, không hàm ý giữ bí mật sau phát hành.

## Bài toán và nguồn

Phân loại một chữ số0–9 từ nét canvas người thật. Swin1D dùng chuỗi float32[3,128]; DINOv3 dùng cùng JSON→RGB224 với drawai-raster-v1. PenDigits và test fixture không được tính vào bộ tự thu. Không có ảnh/font/synthetic thay người tham gia.

## Quy mô dự kiến

Pilot5 người/250 mẫu, chỉ pilot-dev. Chính30 người mới/6.000 mẫu:20 train/5 validation/5 test,4.000/1.000/1.000 mẫu. Chưa có số liệu thu thực; đếm theo người/lớp/thiết bị sau P4. Cùng người không được xuất hiện ở hai split; kiểm tra danh tính là trách nhiệm điều phối, không suy từ mã khác nhau.

## Thu và kiểm tra

Protocol-v1, consent-v1, plan hash trong split-policy.json. Không prediction khi thu, không loại mẫu vì model sai. Schema/geometry/labels/duplicates/quota được audit; near-duplicate cần người rà soát và ghi lý do. Rejection log giữ truy nguyên phiên gốc. Timestamp chỉ từ thời gian thu thực, không bịa point timestamps. Test không dùng tuning; augmentation chỉ train.

## Bản ghi dự kiến

JSON nét gốc; PNG224; manifest.csv với sample_id, participant_id ẩn danh, stage/split, label, repetition, thiết bị, đường dẫn/hash file; artifact Swin NPY cho nội bộ. Chính sách phiên bản và hash bảo đảm hai model dùng cùng sample_id/split. P3 output chỉ staging, không phải gói sẵn sàng phát hành.

## Riêng tư và đồng ý

Ba phạm vi đồng ý tách riêng: training, evaluation/reporting, Kaggle publication. Hồ sơ/mapping danh tính/consent refs ở kho private ngoài Git/public. Trước P8 loại trường thời gian chi tiết, mã hồ sơ và thông tin không cần thiết, rà soát raw strokes/metadata; xác nhận phạm vi đồng ý và license phát hành. Chưa chọn/chấp thuận license dataset cuối cùng; không tự gán license weights DINOv3 cho dataset người viết.

## Giới hạn

Mẫu nhỏ theo nhóm tuyển thuận tiện, chuột/touch/pen có domain gap; mobile giả lập chưa thay thiết bị thật. Người lớn/độ tuổi, ngôn ngữ, tuyển mẫu và phân bố thiết bị thực cần ghi khi có thông tin phù hợp và được đồng ý; không tự suy đoán/thu định danh. Chưa có accuracy/model kết quả trên dataset này. Chỉ một chữ số hoàn chỉnh; không OCR chữ/câu hoặc xác định danh tính người viết.

## Còn phải điền trước public

Số liệu người/mẫu thực và quota; split/manifest hashes; provenance và exclusion log; kết quả rà soát consent/privacy; license được chấp thuận; thông tin liên hệ quản lý; bản1.0 và link Kaggle public. Chưa có link Kaggle tại P3.
