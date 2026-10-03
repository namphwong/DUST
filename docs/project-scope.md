# Phase 0 — Phạm vi dự án

Ngày chốt: 02/10/2026. Trạng thái: hoàn thành đặc tả Phase 0; các yêu cầu dưới đây chưa đồng nghĩa sản phẩm đã được triển khai.

## Mục tiêu

Xây web cá nhân phục vụ học tập/nghiên cứu: phân loại hoạt động từ gia tốc ba trục bằng Swin Transformer 1D. Huấn luyện trên Kaggle, suy luận bằng ONNX Runtime Web trong trình duyệt, giao diện React + TypeScript + Vite và hosting Vercel Hobby. Hướng đến chi phí dịch vụ 0 đồng trong hạn mức miễn phí, dùng tên miền *.vercel.app.

## Người dùng và phạm vi sử dụng

Người học và người nghiên cứu có thể thử mẫu UCI HAR, nhập CSV tương thích, xem tín hiệu và tải dự đoán. Dữ liệu ngoài UCI HAR được đánh dấu “dữ liệu ngoài bộ đánh giá”; chất lượng trên tập test UCI không đại diện cho mọi thiết bị/vị trí cảm biến.

Thiết bị tham chiếu local: Ryzen 7 H 255, Radeon 780M, RAM khoảng 15.3 GB. Dùng CPU/WASM làm đường chạy cơ bản; không yêu cầu GPU NVIDIA trên máy người dùng.

## Các quyết định đã chốt

| Thành phần | Quyết định |
|---|---|
| Dataset đầu tiên | UCI HAR chính thức, các file total_acc trong Inertial Signals |
| Đầu vào | float32 [B, 3, 128], thứ tự X, Y, Z; gia tốc bao gồm trọng lực |
| Đầu ra | 6 logits; softmax chỉ là điểm dự đoán, không phải xác suất đúng đã hiệu chuẩn |
| Tần số và cửa sổ | 50 Hz; 128 điểm; stride 64; không pad phần đuôi |
| Chuẩn hóa | Mean/std từng kênh tính trên train sau khi tách validation; xuất cùng model |
| Split | Test chính thức giữ riêng; train/validation tách theo subject_id |
| Export web | ONNX input cố định [1,3,128], mỗi lần xử lý một cửa sổ |
| Suy luận | WASM CPU 1 thread; WebGPU là tối ưu tùy chọn sau khi kiểm chứng |
| Lưu trữ | CSV và kết quả ở thiết bị người dùng; không cần backend/database |
| Trọng số | Chỉ mô hình đã train và đánh giá mới hiển thị kết quả hoạt động |
| Hosting | Vercel Hobby cho dự án cá nhân phi thương mại; theo dõi quota |

Tên sản phẩm tạm: MotionSense. Repo nghiên cứu vẫn là DuST. Tên giao diện có thể đổi mà không ảnh hưởng hợp đồng dữ liệu.

## MVP bắt buộc

1. Trang chính có nút thử mẫu và nhập CSV.
2. Có sáu mẫu UCI, mỗi lớp một mẫu có nhãn tham chiếu, nguồn và split rõ ràng. Chọn mẫu theo ID xác định trước, không chọn theo dự đoán đúng; phải hiển thị cả dự đoán sai nếu xảy ra.
3. CSV có header, preview, lựa chọn đơn vị và kiểm tra lỗi theo data-contract.md.
4. Biểu đồ gia tốc XYZ, các đoạn thiếu dữ liệu và khoảng thời gian dự đoán.
5. Kết quả từng cửa sổ: nhãn và điểm dự đoán sáu lớp; mô hình luôn thuộc bài toán sáu lớp, không tự suy ra lớp “khác”.
6. Trạng thái tải model, tiến trình phân tích, hủy tác vụ và thử lại sau lỗi.
7. Xuất CSV kết quả với phiên bản model và thông tin tiền xử lý.
8. Trang hướng dẫn và model card ghi metric, split, nguồn dữ liệu, giới hạn sử dụng.
9. Giao diện tiếng Việt, hỗ trợ laptop/điện thoại, bàn phím và thông báo trạng thái.

## Ngoài phạm vi MVP

Thu cảm biến trực tiếp từ điện thoại, đăng nhập, cloud history, backend suy luận, train trên web, thu phí, video, phát hiện té ngã, chẩn đoán y tế và tự động nhận dạng hoạt động ngoài sáu lớp. Cấu hình chín kênh chỉ là thí nghiệm mở rộng, không phải input của MVP.

## Luồng sử dụng

Mở web → chọn mẫu hoặc CSV → kiểm tra dữ liệu/đơn vị → xem preview → phân tích → xem kết quả → tải CSV.

Việc chọn đơn vị và xác nhận loại gia tốc phải diễn ra trước khi phân tích CSV người dùng. Tín hiệu đã tách trọng lực không tương thích với mô hình total acceleration. Không đưa phép xử lý bộ lọc tách trọng lực vào MVP.

## Giới hạn tài nguyên của sản phẩm

- CSV tối đa 10 MiB và 100.000 dòng dữ liệu (không tính header). Đây là giới hạn ứng dụng, không phải giới hạn Vercel.
- Phân tích tuần tự trong Web Worker; cập nhật tiến trình theo số cửa sổ đã xử lý.
- Model chỉ tải khi cần; cache theo phiên bản và kiểm tra đồng bộ artifact.
- Mục tiêu model ≤10 MiB; benchmark WASM sau warm-up ≤500 ms/cửa sổ trên laptop tham chiếu. Nếu không đạt cần điều chỉnh/ghi nhận trước phát hành.
- Giới hạn miễn phí có thể thay đổi; trước deploy phải kiểm tra quota tài khoản. Model và WASM vẫn tiêu thụ băng thông khi tải.

## Nguồn

- [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [Tải bản chính thức](https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [Vercel Hobby](https://vercel.com/docs/plans/hobby)

Tài liệu liên quan: [hợp đồng dữ liệu](data-contract.md), [nghiệm thu](acceptance-criteria.md), [tiến độ](phase-status.md).
