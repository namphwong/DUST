# DrawAI — Giao diện tối giản cho người dùng

Cập nhật 11/10/2026. Trang sản phẩm tại `/drawai`: hướng dẫn một dòng, canvas, Nhận dạng và chữ số kết quả. Hoàn tác/Xóa luôn ở cạnh vùng vẽ; mẫu 0–9 có nút riêng để thử ngay.

Đưa xuất JSON và nhãn người viết vào “Tùy chọn”; top-3 vào “Xem các dự đoán khác”; thông tin mô hình/kiểm chứng vào “Về DrawAI”. Bỏ metric dataset, thông số model, latency và ba khối hướng dẫn dài khỏi màn hình chính. Giữ nguyên model, worker, revision, validation và retry.

Desktop hai cột trong một khung, mobile một cột; mẫu mobile có vùng bấm cao 44px. Các chi tiết mở thêm bằng bàn phím; status/lỗi vẫn có live announcement. Nét được giữ khi resize và không có dữ liệu gửi lên API.

Validation: production build và 9 UI browser tests (10 mẫu khớp logits D4, session reuse, canvas/undo/clear/export, stale response, retry/hash, schema/timestamp, resize, keyboard, DPR2/pointer). Ảnh hiện tại trong `artifacts/drawai/ui/desktop.png` và `mobile.png`. Báo cáo D5/D6 mô tả các lần nghiệm thu trước; lượt sửa này chỉ thay giao diện, không đổi kết quả nghiên cứu hoặc biến D6 người thật thành hoàn thành.
