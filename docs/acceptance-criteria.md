# Tiêu chí nghiệm thu

Mã tiêu chí được dùng trong test và checklist phát hành. Trạng thái mặc định là chưa kiểm chứng, ngoại trừ các đầu ra đặc tả Phase 0. Mục tiêu hiệu năng/metric không phải số đo đã đạt.

## Nghiệm thu Phase 0

- P0-01: project-scope.md nêu mục tiêu, người dùng, MVP, ngoài phạm vi, kiến trúc và chi phí.
- P0-02: data-contract.md chốt sáu nhãn, input/output, đơn vị, thứ tự tensor và cửa sổ.
- P0-03: có hợp đồng CSV, lỗi đầu vào, resample, khoảng mất dữ liệu và CSV kết quả.
- P0-04: có CSV hợp lệ 128 điểm, phiên 192 điểm, mẫu m/s² và ít nhất bốn loại CSV không hợp lệ.
- P0-05: mẫu tổng hợp được ghi rõ nguồn và không gắn nhãn hoạt động giả.
- P0-06: nghiệm thu các phase tiếp theo có điều kiện cụ thể và trạng thái minh bạch.

## Phase 1 — Khả năng triển khai

- P1-01: mô hình thật từ repo forward với [1,3,128] → [1,6].
- P1-02: backward tạo gradient hữu hạn, không bằng 0 ở patch embedding và attention được kích hoạt.
- P1-03: xuất ONNX kiểm tra hợp lệ, input cố định và opset ghi trong report.
- P1-04: ≥8 fixture tổng hợp; kiểm tra PyTorch/ONNX Python/web với atol=1e-5, rtol=1e-4; sai lệch phải hiển thị FAIL.
- P1-05: chạy WASM trong Chromium thật; có báo cáo và version môi trường. Kiểm tra tải model lỗi và chạy lại.
- P1-06: model thử ghi trained=false, không hiển thị nhận dạng hoạt động.
- P1-07: đo dung lượng, tải/khởi tạo, median inference 30 lần sau 5 warm-up; đánh giá mục tiêu ≤10 MiB và ≤500 ms trên máy tham chiếu.

## Phase 2 — Dữ liệu

- P2-01: tensor/numeric type, pairing label/subject và phân bố lớp đúng nguồn.
- P2-02: train/validation không trùng subject; test chính thức giữ riêng.
- P2-03: mean/std chỉ từ train; split và preprocessing tái lập được.
- P2-04: demo có nguồn, ID và nhãn tham chiếu; không nối cửa sổ chồng lấn thành timeline giả.

## Phase 3 — Huấn luyện

- P3-01: notebook Kaggle chạy lại được; lưu seed/config/checkpoint/log.
- P3-02: có baseline CNN và Swin, chọn checkpoint bằng validation.
- P3-03: báo accuracy, macro-F1, recall từng lớp và confusion matrix trên test cuối.
- P3-04: mục tiêu tham khảo macro-F1 ≥0.90 trên UCI test; nếu không đạt phải phân tích lỗi và quyết định cải tiến hoặc phát hành như demo nghiên cứu với giới hạn rõ. Không tuyên bố đạt khi chưa đo.
- P3-05: không chọn siêu tham số bằng test; thử nhiều seed khi quota cho phép và báo số seed đã chạy.

## Phase 4 — Artifact

- P4-01: ONNX từ best checkpoint khớp PyTorch trên fixture và tập đánh giá.
- P4-02: macro-F1 ONNX không giảm quá 0.005 tuyệt đối so với checkpoint; với quantization phải kiểm tra lại.
- P4-03: có version/hash, mean/std thật, nhãn đúng thứ tự và trained=true.
- P4-04: model/warm-up được benchmark trong browser; metadata đồng bộ cùng release.

## Phase 5–6 — Sản phẩm và tích hợp

- WEB-01: người dùng mới thử được mẫu trong tối đa ba thao tác sau mở trang.
- WEB-02: CSV hợp lệ được preview, chọn đơn vị/loại gia tốc, phân tích và xuất kết quả.
- WEB-03: tất cả lỗi trong data-contract có thông báo tên lỗi, dòng/cột nếu phù hợp.
- WEB-04: fixture kiểm tra 127/128/191/192 điểm, channel-major, g/m/s² và đoạn mất dữ liệu.
- WEB-05: pipeline Python và web cho tensor chuẩn hóa gần tương đương và dự đoán khớp trên fixture.
- WEB-06: hủy tác vụ trong ≤1 giây sau cửa sổ đang xử lý trên máy tham chiếu; nhập file mới không nhận kết quả cũ.
- WEB-07: công việc nặng chạy trong worker; giao diện vẫn phản hồi khi phân tích 100.000 dòng.
- WEB-08: không gửi CSV qua mạng; kiểm tra Network khi chạy bằng file cục bộ.
- WEB-09: sáu điểm dự đoán không bị ghi thành độ chính xác bảo đảm; có thông tin ngoài miền dữ liệu.
- WEB-10: viewport 390×844 và 1366×768 không tràn ngang; thao tác chính dùng bàn phím được.

## Phase 7–8 — Phát hành

- REL-01: production build thành công; URL Vercel public tải đủ model/WASM từ browser mới.
- REL-02: WASM chạy được trên Chrome/Edge laptop; kiểm tra thêm mobile và ghi rõ browser đã thử.
- REL-03: thử mạng chậm, tải thất bại, cache mới/cũ và artifact khác version; lỗi có thể thử lại.
- REL-04: README, hướng dẫn CSV, model card, báo cáo metric và rollback đầy đủ.
- REL-05: dùng Vercel Hobby phi thương mại và tên miền miễn phí; không có backend/API trả phí bắt buộc.
- REL-06: ghi quota thực tế lúc deploy; model tải khi cần và cache theo version.

## Bằng chứng cần lưu

Report JSON cho parity và benchmark; manifest split/config; metric và confusion matrix; test frontend; checklist URL/browser/version/ngày nghiệm thu. Mỗi phase chỉ được đánh dấu hoàn thành khi có đầu ra và bằng chứng tương ứng.
