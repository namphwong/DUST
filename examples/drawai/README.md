# Ví dụ hợp đồng D0

Hai JSON là hình học tổng hợp để kiểm tra schema, không phải nét người thật hoặc mẫu PenDigits và không có nhãn chữ số.

- `strokes-valid.json`: hai nét có timestamp tương đối.
- `strokes-no-time.json`: một nét không có timestamp; không tự thêm thời gian giả.

Schema: `docs/drawai/strokes.schema.json`. Ràng buộc hình học/timestamp toàn bản vẽ: `docs/drawai/data-contract.md`. Schema chấp nhận nét một điểm về cấu trúc; preprocessing quyết định bản vẽ có đủ chuyển động để nhận dạng hay không. Chấp nhận schema không đồng nghĩa hình đó là chữ số hoặc model sẽ nhận đúng.
