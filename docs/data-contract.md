# Hợp đồng dữ liệu MVP — phiên bản 1

Hợp đồng này đã được triển khai và kiểm chứng local ở [Phase 6](phase6-pipeline-report.md). Các CSV ví dụ trong examples/phase0 là dữ liệu tổng hợp để kiểm tra định dạng, không phải dữ liệu train hay mẫu UCI.

## Nhãn và tensor

| ID web/PyTorch | ID UCI | Mã lớp | Tên tiếng Việt |
|---|---|---|---|
| 0 | 1 | WALKING | Đi bộ |
| 1 | 2 | WALKING_UPSTAIRS | Lên cầu thang |
| 2 | 3 | WALKING_DOWNSTAIRS | Xuống cầu thang |
| 3 | 4 | SITTING | Ngồi |
| 4 | 5 | STANDING | Đứng |
| 5 | 6 | LAYING | Nằm |

Input float32 [B,3,128], thứ tự kênh X/Y/Z. Bộ nhớ ONNX web [1,3,128] là toàn bộ 128 giá trị X, rồi 128 Y, rồi 128 Z. Không flatten xen kẽ X/Y/Z theo từng dòng. Output [1,6] là logits theo bảng trên. Softmax được tính ổn định bằng cách trừ logit lớn nhất trước exp.

## Định dạng CSV người dùng

UTF-8, cho phép BOM, dấu phân cách phẩy, dấu thập phân chấm. Header bắt buộc gồm timestamp_ms,acc_x,acc_y,acc_z; thứ tự cột có thể khác. Tên cột được trim nhưng phải khớp đúng; chưa có mapping tên tùy ý trong MVP. Cột thêm được bỏ qua với thông báo. Không hỗ trợ chọn cột nhãn làm input.

```csv
timestamp_ms,acc_x,acc_y,acc_z
0,0.12,0.98,0.03
20,0.15,1.01,0.02
40,0.11,0.97,0.04
```

Mỗi CSV chỉ chứa một phiên liên tục của một nguồn cảm biến, cùng đơn vị và hệ tọa độ. Timestamp là mili giây tính từ đầu phiên hoặc timestamp tuyệt đối; trừ timestamp đầu để xử lý. Không nối nhiều người hoặc trial độc lập trong một CSV.

Người dùng chọn đơn vị g hoặc m/s², không tự đoán từ biên độ. Gia tốc bắt buộc bao gồm trọng lực. Vị trí/hệ trục tương thích UCI HAR cho kết quả có cơ sở đánh giá hơn; hệ trục khác hoặc vị trí khác được xem là ngoài miền dữ liệu. MVP không tự xoay hệ trục. Không clamp biên độ tùy ý.

## Quy tắc kiểm tra

| Mã lỗi | Điều kiện | Cách xử lý |
|---|---|---|
| EMPTY_FILE | Không có dòng dữ liệu | Từ chối, hướng dẫn chọn file khác |
| FILE_TOO_LARGE | >10 MiB | Từ chối trước khi parse |
| TOO_MANY_ROWS | >100.000 dòng | Từ chối, hướng dẫn cắt một phiên nhỏ hơn |
| MISSING_COLUMN | Thiếu header bắt buộc | Nêu tên cột thiếu |
| DUPLICATE_COLUMN | Trùng tên cột sau trim | Từ chối |
| INVALID_NUMBER | Ô trống, không phải số hữu hạn, NaN/Infinity | Từ chối và nêu dòng/cột; không tự điền 0 |
| INVALID_TIMESTAMP | Timestamp âm, trùng hoặc không tăng | Từ chối; không tự sort/gộp |
| TOO_SHORT | Không có đoạn đủ 128 điểm sau resample | Từ chối và nêu số điểm hợp lệ |
| INCOMPATIBLE_SIGNAL | Người dùng chọn gia tốc đã tách trọng lực | Giải thích cần total acceleration |
| UNSUPPORTED_FREQUENCY | Median từng đoạn đủ dài ngoài 10–25 ms | Từ chối, cần nguồn 40–100 Hz |
| INVALID_CSV | Quoting/số ô không hợp lệ hoặc không đọc được UTF-8 | Từ chối, hướng dẫn sửa định dạng |

MVP dùng parser tuân thủ CSV quoting. Dòng trắng cuối file bỏ qua; dòng có ô tín hiệu trống bên trong vẫn là lỗi. Timestamp số thực được chấp nhận nếu hữu hạn, không âm và tăng nghiêm ngặt. Giới hạn dòng được áp dụng cả sau resample để tránh timestamp tạo quá nhiều điểm.

## Resample và khoảng mất dữ liệu

1. Kiểm tra CSV trước khi biến đổi; chuyển m/s² sang g bằng cách chia 9.80665.
2. Tính khoảng thời gian giữa các mẫu. Nếu bất kỳ khoảng nào >100 ms, tách phiên thành các đoạn tại khoảng đó; không nội suy vượt khoảng mất dữ liệu.
3. Với mỗi đoạn, lấy lưới thời gian 20 ms bắt đầu từ điểm đầu đoạn, đến điểm cuối; nội suy tuyến tính từng trục. Không ngoại suy.
4. Nếu khoảng lấy mẫu median không nằm trong [10,25] ms (nguồn khoảng 40–100 Hz), từ chối resample trong MVP và giải thích cần nguồn gần 50 Hz. Kiểm tra riêng từng đoạn đủ dài; đoạn ngắn bị bỏ và báo rõ.
5. Nguồn có khoảng cách 20 ms đều giữ nguyên giá trị; phải có kiểm thử identity. Không resample dữ liệu UCI đã chia cửa sổ.
6. Preview báo tần số ước lượng, số đoạn, số điểm nội suy và các đoạn bị bỏ. Người dùng xác nhận trước khi phân tích dữ liệu đã resample.

Resample chỉ giải quyết thời gian; không bảo đảm tín hiệu giống phân bố train. Hỗ trợ downsample giới hạn như trên là lựa chọn MVP, chưa bao gồm thiết kế bộ lọc cho nguồn tần số cao.

## Chia cửa sổ và chuẩn hóa

Với mỗi đoạn n điểm: số cửa sổ = max(0, floor((n−128)/64)+1). Không lấy cửa sổ vượt ranh giới đoạn. Start index là 0,64,128,...; mỗi cửa sổ có đúng 128 điểm. Phần đuôi chưa đủ cửa sổ được bỏ và thông báo.

Ví dụ: 127 điểm → 0 cửa sổ; 128 → 1; 191 → 1 và 63 điểm đuôi; 192 → 2. Với 50 Hz, 128 mẫu chiếm khoảng hỗ trợ 2,56 giây; timestamp của mẫu đầu/cuối cách nhau 2,54 giây. Kết quả dùng timestamp thực của mẫu đầu/cuối, không cộng thêm 20 ms giả.

Sau chuyển đơn vị/resample, áp dụng (x−mean[channel])/std[channel]. Mean/std chỉ lấy từ train sau split validation. Với std <1e-8 dùng 1e-8. Giá trị mean/std chưa có ở Phase 0; không phát hành metadata giả như đã được fit.

## Dữ liệu train và demo UCI

Đọc total_acc_x/y/z_train.txt và các file test tương ứng, y_train/y_test và subject ID. Mỗi dòng tín hiệu đã là một cửa sổ; giữ nguyên pairing. Validation tách theo subject từ train chính thức và lưu manifest. Test không dùng chọn checkpoint hay ngưỡng.

Demo UCI là cửa sổ độc lập có metadata dataset/split/subject/window ID/reference label. Timestamp tương đối 0..2540 ms chỉ phục vụ biểu đồ cửa sổ, không tuyên bố là một timeline liên tục của người đó. Dự đoán và nhãn tham chiếu được hiển thị riêng.

## Artifact triển khai

- model.onnx: fixed input signals [1,3,128], output logits [1,6].
- labels.json: sáu lớp theo bảng, có phiên bản.
- preprocessing.json: contract_version, channels, input_unit=g, includes_gravity=true, sample_rate_hz=50, window_size=128, stride=64, mean/std có thật, epsilon=1e-8.
- model-card.json: version, trained=true/false, dataset, split, kiến trúc, seed, sha256 và nguồn.
- metrics.json: các số đo thật kèm tập đánh giá, không dùng số placeholder.
- golden-samples.json: input đã chuẩn hóa và output PyTorch; có thêm fixture từ CSV để kiểm tra toàn pipeline ở Phase 6.

Tất cả artifact phải thuộc cùng release. Model thử Phase 1 có trained=false; chỉ hiển thị logits và sai số kỹ thuật.

## CSV kết quả

Các cột bắt buộc: segment_id,window_index,start_ms,end_ms,predicted_id,predicted_label,score_0,score_1,score_2,score_3,score_4,score_5,model_version,contract_version,source_unit.

Điểm softmax có tổng gần 1; không đặt tên là “accuracy”. Lấy argmax, nếu bằng nhau chọn ID nhỏ hơn. Chưa đặt ngưỡng “không chắc chắn”; chỉ thêm khi đã chọn và đánh giá trên validation. Không dùng lớp thứ bảy để che input không phù hợp.
