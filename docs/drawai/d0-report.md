# DrawAI — Báo cáo nghiệm thu D0

> Tài liệu lịch sử D0–D6 (11/10/2026). CNN và route cũ nếu được nhắc đến không thuộc pipeline P1 hiện hành. Xem [báo cáo P1](phase1-report.md) và [training guide hiện tại](training-guide.md).

Ngày: 11/10/2026. Phạm vi: rà soát và hoàn thiện đặc tả D0 trên workspace hiện tại, đã có tài liệu/artifact D1–D2 từ trước.

## Kết quả

| Task | Trạng thái | Bằng chứng |
|---|---|---|
| D0-01 | PASS | Kế hoạch tổng thể và checklist 8 phase |
| D0-02 | PASS | scope.md: MVP, luồng/state, lỗi, giới hạn và quyết định còn mở |
| D0-03 | PASS | data-contract.md, strokes.schema.json và hai ví dụ JSON có/không timestamp |
| D0-04 | PASS | architecture.md: ranh giới module, worker protocol, revision/version, loại response cũ |
| D0-05 | PASS | environment.md, requirements DrawAI, phiên bản runtime/package thực đọc |
| D0-06 | PASS | acceptance-criteria.md: ma trận task/kiểm tra/bằng chứng và ngưỡng |

Report máy đọc: `artifacts/drawai/d0/verification.json`. Report ghi SHA-256 các tài liệu/schema/ví dụ dùng để nghiệm thu; thay đổi các file sau đó cần rà soát lại các mục liên quan.

## Nội dung hoàn thiện trong đợt này

- Schema JSON v1 có kiểu dữ liệu, version, hệ tọa độ, canvas, nét, nhãn tham chiếu và provenance tùy chọn.
- Tách kiểm tra schema khỏi ràng buộc runtime: tổng điểm, timestamp toàn bản vẽ, số hữu hạn và đủ chuyển động.
- Timestamp đồng nhất toàn bản vẽ, hoặc hoàn toàn vắng; không sinh timestamp giả cho nguồn thiếu thời gian.
- Thiết kế worker sản phẩm với request ID/revision/release; snapshot lúc bấm Nhận dạng; bỏ response cũ sau sửa nét.
- Phân biệt input ready và model ready; tải model lần đầu khi cần, không tự nhận dạng sau clear/undo.
- Kiểm kê lại phiên bản Python/Node/web dependencies; bổ sung bootstrap venv riêng và giới hạn tái lập.

## Kiểm tra thực hiện

Kiểm tra schema Draft 2020-12; parse JSON theo chế độ từ chối NaN/Infinity; xác minh hai ví dụ hợp lệ. Kiểm tra schema từ chối phiên bản sai, hệ tọa độ sai, thiếu trường, canvas width=0, tọa độ bool/chuỗi, stroke rỗng, nhiều hơn 16 nét và reference label ngoài 0–9. Kiểm tra liên kết Markdown cục bộ trong các tài liệu D0, tính duy nhất/đủ của mã task D0 và kiểm tra whitespace của diff.

Kiểm kê Python/package metadata và Node/npm thực chạy trong workspace. Không clean-install hoặc chạy lại train/ONNX/browser trong nghiệm thu D0 vì đây là phase đặc tả; các bằng chứng D1–D2 đang có không bị xem là kết quả của đợt D0 này.

## Điểm bàn giao

D0 hoàn thành ở mức đặc tả/schema. Giao thức worker và kiểm tra toàn bản vẽ còn được triển khai ở D5; nghiệm thu chức năng ở D6. L64/L128, augmentation và cấu hình training sẽ được chọn bằng validation ở D3. Artifact D2 vẫn untrained; model sản phẩm phải qua D3–D4.

Writer ID chưa được xác minh và validation có thể chung người viết theo báo cáo D1 hiện có. Điện thoại/người tham gia/hosting cần bằng chứng riêng ở D6–D7. D0 không tuyên bố chất lượng nhận dạng đã đạt hoặc ứng dụng sản phẩm đã hoạt động.
