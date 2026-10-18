# DrawAI — Phạm vi D0

Cập nhật 11/10/2026. Đặc tả được chốt cho MVP; giao diện sản phẩm và model đã train chưa triển khai.

## Người dùng và mục tiêu

Demo học tập/nghiên cứu nhận dạng một chữ số 0–9 từ chuyển động nét bút. Người dùng vẽ bằng chuột, bút hoặc cảm ứng, hoàn tất chữ số và bấm Nhận dạng. “Online” mô tả đầu vào chuỗi nét có thứ tự; chưa có dự đoán liên tục khi đang viết.

## Luồng và trạng thái

1. Mở `/drawai`: xem hướng dẫn, vùng vẽ và nút thử mẫu. Model tải khi cần.
2. Vẽ nhiều nét của một chữ số hoặc chọn một mẫu có nguồn.
3. Nhận dạng: kiểm tra input, tải/kiểm tra model nếu cần, xử lý trong worker.
4. Xem chữ số, top-3 điểm softmax và thời gian inference.
5. Vẽ mới, hoàn tác nét cuối, xóa hoặc nạp mẫu để thử lại; kết quả trước đó hết hiệu lực ngay khi dữ liệu thay đổi.
6. Xuất JSON chứa nét gốc, hệ tọa độ và phiên bản schema. Nhãn người dùng cung cấp, nếu có, độc lập với dự đoán của model.

| Trạng thái | Hành vi |
|---|---|
| Empty | Hướng dẫn vẽ/chọn mẫu; chưa có kết quả; Nhận dạng disabled |
| Drawing | Ghi tọa độ và ranh giới nét; chỉ nhận một pointer đang hoạt động |
| Loading | Hiển thị tiến trình tải/khởi tạo; vẫn có thể chỉnh nét; không chạy inference sớm |
| Ready | Nét hợp lệ; model sẵn sàng hoặc có thể tải khi bấm Nhận dạng |
| Predicting | Công việc chạy trong worker; phản hồi UI; không nhận thêm yêu cầu đồng thời |
| Result | Chữ số và top-3 đúng revision nét; điểm softmax không phải độ chính xác bảo đảm |
| Error | Lý do cụ thể; giữ nét; retry sau lỗi tải; sửa nét sau lỗi input |

## Kiến trúc và phạm vi

- Python: parser PenDigits gốc → split → preprocessing → CNN/Swin train/evaluate/export.
- React/TypeScript: canvas → cùng preprocessing trong worker → ONNX → UI.
- Worker giao tiếp theo `request_id`, `drawing_revision`, `type`; các response cũ bị bỏ khi sửa nét. Request inference gồm nét gốc và version; response gồm 10 logits/scores, nhãn, latency và model version. D5 triển khai hợp đồng này; worker D2 chỉ dùng kiểm chứng kỹ thuật.
- Kênh cơ bản `[x,y,stroke_start]`; nhãn theo thứ tự 0–9. Backbone dùng lại `models/SwinTransformer_cls.py`; không dùng checkpoint/mean/std/labels HAR.
- D2 có trang kỹ thuật `/drawai/check`, không hiển thị logits untrained như dự đoán chữ số.
- File nét được xử lý tại thiết bị; không có API/database bắt buộc. Hosting static sẽ kiểm tra quota khi đến D7.
- Các module đặt trong `training/drawai/`, `web/src/drawai/`, `docs/drawai/`, `artifacts/drawai/`.

Thiết kế chi tiết module/worker và quy tắc loại kết quả cũ: [architecture.md](architecture.md). Schema và ví dụ nét: [data-contract.md](data-contract.md).

## Điều kiện hoàn thành và giới hạn sản phẩm

MVP hoàn thành khi người dùng vẽ/nạp mẫu, nhận kết quả bằng artifact đã train tại URL phát hành và có report nghiệm thu. Canvas giữ nét sau lỗi tải; cho phép xóa/undo khi inference đang chạy; kết quả chỉ thuộc bản vẽ hiện tại. Clear/undo/nạp mẫu không tự khởi chạy inference.

Sản phẩm chỉ có 10 nhãn và không có lớp “khác”; kiểm tra hình học loại input rỗng/hỏng nhưng không bảo đảm từ chối mọi hình vẽ ngoài chữ số. Không tự đặt ngưỡng độ tin cậy để gọi hình là chữ số; nếu thêm tính năng từ chối phải thiết kế và đánh giá riêng.

Thu nét phục vụ D6 là hoạt động có nhãn do người tham gia cung cấp, dùng ID ẩn danh và ghi loại thiết bị. Nét vẽ thử trong ứng dụng không tự trở thành training data. Export JSON là hành động chủ động của người dùng, mặc định giữ dữ liệu trên thiết bị.

## Quyết định và nội dung còn mở

| Nội dung | Quyết định / phase giải quyết |
|---|---|
| Nhãn và đặc trưng | 0–9; x/y/stroke_start; không dùng timestamp/pressure |
| L=64 hay L=128 | Hai ứng viên kỹ thuật; D3 chọn bằng validation, D4 khóa trong release |
| CNN/Swin config và augmentation | D3 sàng lọc bằng validation; không chọn theo test |
| Trạng thái worker và stale result | Đặc tả D0 ở architecture.md; D5 triển khai, D6 kiểm thử |
| Retry/giới hạn thời gian worker | D5 chọn timeout theo benchmark/tình trạng tải; D6 xác minh lỗi và reset |
| Trang mặc định/hosting | `/drawai` là route sản phẩm; D7 chốt trang mặc định, target và quota |
| Canvas/điện thoại thật | D6 thu người tham gia/thiết bị; các mục thiếu ghi NOT RUN |

Ngoài MVP: nhiều chữ số, chữ cái/từ, OCR ảnh, đăng nhập/cloud history, tự học từ nét người dùng, suy luận khi chưa viết xong và quantization. Các mục này chỉ vào backlog sau MVP.
