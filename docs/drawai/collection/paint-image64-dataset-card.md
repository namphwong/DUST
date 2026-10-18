# DrawAI Paint35 — Dataset card (P3 prepared, chưa P4 lock)

## Nguồn và phạm vi

File chủ dữ liệu cung cấp: So_viet_tay_35_nguoi.npz,2.828.016 bytes; SHA-256 c594927b59509bb62cb185cf5e64034ee641b046798f0f192d6155fd91f71a64. Link public: https://www.kaggle.com/datasets/namphwong/drawai-handwritten-digits-placeholder (file ở phiên bản2).

Ngày18/10/2026, chủ dữ liệu xác nhận ảnh do35 người thật viết bằng Paint, person_ids là35 người khác nhau, đã đồng ý training/evaluation/public Kaggle và description “Synthetic” là lỗi đóng gói. Nguyên văn metadata gốc vẫn được giữ trong audit; provenance riêng ghi xác nhận, không sửa ngầm file đã tải lên. Không có xác minh danh tính/consent độc lập từ phần mềm. License dataset chưa chọn; quyền sử dụng đã được chủ dữ liệu xác nhận không tự thay cho license phát hành chuẩn hóa.

## Cấu trúc và thống kê thật

7.000 ảnh grayscale uint8 64×64, nền trắng/nét đen; labels0..9 mỗi lớp700;35 person IDs mỗi người200; mỗi cặp người/chữ số có20 lần. Composite người/nhãn/lần là duy nhất7.000, không dùng sample_ids1..20 làm ID toàn cục. Không có raw stroke JSON, timestamps thu, device metadata, phiếu consent hoặc tên người trong archive này.

## Split

Người1–5 là nhóm kiểm tra pilot hậu thu:250 ảnh reps1–5,750 reserved reps6–20. Train người6–25:4.000; validation26–30:1.000; test31–35:1.000. Train400 ảnh/lớp, validation100 ảnh/lớp, test100 ảnh/lớp;20/5/5 người. Cùng người không vào hai split chính hoặc pilot/main. Đây là partition hồi cứu chốt trước model training, không tuyên bố prospective split lúc người viết trong Paint.

## Biểu diễn và chất lượng

Ảnh gốc/PNG64/hash được bảo toàn. Swin1D dùng row-pixel[64,64]; DINOv3 RGB224 bilinear/ImageNet normalization. Cùng sample ID/label/split, không augmentation. Swin PenDigits nét bút cũ vẫn độc lập, không gọi weights đó là đã train trên Paint35.

Không exact image duplicate, không blank/low-contrast candidate theo ngưỡng đã ghi.250 ảnh pilot xem qua5 contact sheets, không thấy render trống/mất/cắt chữ số rõ rệt.1.063 near pairs/534 groups (98 cross-split) cần P4 review; không tự coi tất cả là data leakage, không tự loại. Chưa có accuracy/model comparison/test evaluation.

## Giới hạn và trạng thái

Không khôi phục được thứ tự nét/pen lifts từ ảnh; input row sequence không là trình tự viết. Nguồn người/consent dựa lời xác nhận của chủ dữ liệu; metadata gốc còn dòng lỗi đã được đính chính riêng. Không có thống kê tuổi/thiết bị/điều kiện thu và không tự suy đoán. Bộ ảnh cân bằng không đảm bảo domain coverage hoặc hiệu quả trên canvas người dùng.

P3 preparation hoàn thành local; P4 source/near-group review và image lock chưa chạy, P5 training/P6 export/P7 tích hợp/P8 hoàn thiện công bố còn lại. Public file đã tồn tại từ upload người dùng yêu cầu; không suy ra đã hoàn thành các phase sau chỉ vì link Kaggle hoạt động.
