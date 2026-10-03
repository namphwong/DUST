# CSV mẫu cho Phase 0

Tất cả tín hiệu ở đây được tạo tổng hợp, không phải UCI HAR và không có nhãn hoạt động. Mục đích là minh họa hợp đồng dữ liệu và làm fixture cho parser/preprocessing sau này.

| File | Đơn vị lựa chọn | Kỳ vọng |
|---|---|---|
| valid-window-g.csv | g | 128 điểm đều 50 Hz → 1 cửa sổ |
| valid-session-g.csv | g | 192 điểm đều 50 Hz → 2 cửa sổ |
| valid-window-ms2.csv | m/s² | Chuyển đơn vị → tương đương valid-window-g trong sai số làm tròn |
| invalid-missing-column.csv | g | MISSING_COLUMN: thiếu acc_z |
| invalid-number.csv | g | INVALID_NUMBER: acc_y ở dòng 3 |
| invalid-duplicate-time.csv | g | INVALID_TIMESTAMP: timestamp trùng ở dòng 4 |
| invalid-reversed-time.csv | g | INVALID_TIMESTAMP: timestamp giảm ở dòng 4 |
| invalid-too-short.csv | g | TOO_SHORT: chỉ có 127 điểm |

Không sử dụng các file này để huấn luyện hoặc minh họa độ chính xác mô hình. Gia tốc tổng hợp có thành phần nền ở trục Y; đây không phải phép mô phỏng hành động có nhãn.
