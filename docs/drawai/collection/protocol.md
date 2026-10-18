# DrawAI — Hướng dẫn người viết và protocol thu dữ liệu v1

Phiên bản: drawai-private-protocol-v1, chốt ngày18/10/2026.

## Trước khi bắt đầu

Người điều phối cấp mã P001–P005 cho pilot; M001–M030 cho đợt chính. Mỗi mã là một người thật, không dùng hai mã cho cùng người. Người của pilot không tham gia đợt chính. Không nhập tên/email/điện thoại vào trang hoặc JSON. Hồ sơ đồng ý và danh sách đối chiếu do điều phối giữ riêng ngoài dữ liệu công khai.

Đọc và xác nhận riêng ba mục: dùng nét cho huấn luyện; đánh giá/báo cáo; công bố nét và metadata ẩn danh trên Kaggle public. Đây là ghi nhận trên công cụ, không thay hồ sơ đồng ý mà người điều phối giữ. Không đồng ý mục nào thì không thu vào dataset public này. Chưa công bố dữ liệu trong P3.

## Cách viết

1. Dùng đúng thiết bị đăng ký: chuột, cảm ứng hoặc bút. Không chọn cảm ứng nếu thực tế đang dùng chuột trên viewport mobile. Một phiên giữ một thiết bị.
2. Viết đúng một chữ số hiển thị, theo cách tự nhiên; không tô lại mẫu, không dán ảnh, không dùng JSON/demo của PenDigits hay dữ liệu sinh. Không viết tên/chữ ký/thông tin cá nhân.
3. Có thể nhấc bút nhiều nét. Viết trong khung; kiểm tra bằng mắt trước khi bấm “Ghi mẫu và tiếp tục”. Sai bản nháp thì hoàn tác/xóa rồi viết lại. Không đánh giá hoặc loại mẫu theo dự đoán AI.
4. Mỗi vòng có đủ0–9 với thứ tự đổi theo mã/vòng. Pilot5 vòng=50 mẫu/người; chính20 vòng=200 mẫu/người. Nghỉ sau mỗi vòng hoặc khi mỏi.
5. Tải JSON sau mỗi vòng10 mẫu và trước khi nghỉ/đóng trang. Dữ liệu chưa tải chỉ ở bộ nhớ và có thể mất khi reload. Công cụ không tự upload/không lưu vào localStorage.
6. Muốn tiếp tục: mở /collect và nạp JSON mới nhất. Không tạo phiên mới để tăng/đặt lại quota. Không sửa nhãn, thời gian hoặc mã trong JSON.
7. Tên file có mã phiên và số mẫu. Bàn giao bản mới nhất cho điều phối qua kênh đã thống nhất. Không upload nguyên JSON phiên có mã hồ sơ consent lên Kaggle. Điều phối chỉ để một snapshot mới nhất/phiên trong inbox.

## Nếu có sự cố

Lưu JSON hiện tại nếu còn mở được. Ghi mã phiên, số mẫu, loại sự cố và thiết bị; không ghi thông tin định danh vào log public. Không sửa nét để làm model đoán đúng. Nếu phát hiện mẫu đã ghi sai protocol, điều phối cách ly phiên, lưu lý do và hướng dẫn thu lại phiên theo cùng mã người; không tự xóa một phần JSON làm sai lịch prompt. Giữ bản bị cách ly để truy nguyên, không đưa vào inbox build.

## Quy mô và split khóa trước thu

- Pilot: P001–P005,5 mẫu/chữ số/người, tổng250; chỉ pilot-dev, không dùng làm test cuối.
- Chính:30 người mới,20 mẫu/chữ số/người, tổng6.000.
- Train: M001–M020=4.000; validation: M021–M025=1.000; test: M026–M030=1.000.
- Người điều phối cấp mã theo tuyển người trước khi xem nét/kết quả model, không đổi split theo chất lượng nét. Cân đối thiết bị giữa các nhóm trong lúc tuyển; ghi đúng thiết bị thực.
- Nếu đổi quy mô/split, phải tạo phiên bản protocol/plan mới trước khi thu chính; không âm thầm dùng phương án2.000 mẫu hoặc chuyển người sau khi xem test.

## Quyền thay đổi ý định trước phát hành

Liên hệ điều phối bằng kênh đã nhận lúc tham gia, cung cấp mã người/phiên. Điều phối ghi yêu cầu và cách ly dữ liệu liên quan trước khi build/phát hành. Không đưa danh tính người yêu cầu vào dataset public. Thời điểm phát hành và điều kiện sử dụng cuối cùng sẽ được giải thích, xác nhận khi chuẩn bị Kaggle ở P8.
