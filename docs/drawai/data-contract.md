# DrawAI — Hợp đồng dữ liệu v1

Đã chốt qua kiểm toán D1 ngày 11/10/2026. Phiên bản tiền xử lý: `drawai-geometry-v1`.

## Nét gốc

Schema máy đọc: [strokes.schema.json](strokes.schema.json). Ví dụ tổng hợp: [examples/drawai](../../examples/drawai/README.md). Hợp đồng worker: [architecture.md](architecture.md).

Các trường bắt buộc: `schema_version`, `coordinate_system`, `canvas`, `strokes`. `reference_label` tùy chọn, nhận null hoặc số nguyên 0–9. `source` tùy chọn để ghi provenance; mẫu demo dataset phải có kind/sample_id/split và metadata nguồn có thể đối chiếu. Hai ví dụ D0 là tổng hợp, không có nhãn chữ số.

Schema kiểm tra kiểu/cấu trúc/giới hạn từng nét. Trước preprocessing, ứng dụng D5 còn kiểm tra tổng số điểm toàn bản vẽ ≤8.192, mọi số hữu hạn, timestamp nhất quán và không có thuộc tính lạ. JSON NaN/Infinity không hợp lệ; không đổi chuỗi số hoặc bool thành tọa độ.

Nếu có timestamp, mọi điểm của bản vẽ phải có `t`, đơn vị ms từ cùng mốc bắt đầu bản vẽ; `t` không giảm theo thứ tự điểm và nét (cho phép bằng nhau). Nếu nguồn không có timestamp thì bỏ `t` ở tất cả điểm. Schema không tự thực thi ràng buộc này; D5 cần validator bổ sung. Timestamp không phải đặc trưng mô hình v1.

Canvas width/height là kích thước CSS pixel tại lúc bắt đầu bản vẽ, hữu hạn và >0; giữ nguyên hệ tọa độ dữ liệu khi resize, chỉ đổi phép render. Điểm nằm ngoài biên vùng vẽ do pointer capture được phép trong giới hạn số của tiền xử lý. Chuyển mẫu y-up sang canvas-y-down trước khi xuất schema v1; không chỉ đổi tên hệ tọa độ mà giữ nguyên số y.

Schema ứng dụng dự kiến D5:

```json
{
  "schema_version": "drawai-strokes-v1",
  "coordinate_system": "canvas-y-down",
  "canvas": {"width": 400, "height": 400},
  "strokes": [[{"x": 100, "y": 80, "t": 0}, {"x": 140, "y": 220, "t": 120}]],
  "reference_label": null
}
```

x/y là số theo đơn vị CSS pixel của vùng vẽ gốc, chiều x sang phải, y xuống dưới. Không nhân tọa độ dữ liệu với devicePixelRatio; DPR chỉ dùng render. Timestamp `t`, khi có, là millisecond tương đối từ khi bắt đầu bản vẽ, không âm, không giảm trong một nét. Mẫu dataset không có thời gian điểm thì bỏ `t`; mô hình v1 không dùng timestamp hoặc pressure. Mỗi lần nhấc bút kết thúc một stroke. `reference_label` chỉ là nhãn nguồn/người viết cung cấp, không tự lấy dự đoán làm ground truth.

Parser nguồn giữ x/y nguyên bản, segment range, nhãn, source file/line/index và ba số trong `.COMMENT`; `writer_id=null` vì ý nghĩa các số comment chưa được tài liệu xác nhận. `.INCLUDE dene.doc` là metadata, không đọc/thực thi file ngoài archive. `.DT 100` không được biến thành timestamp mỗi điểm.

## Tiền xử lý

1. Kiểm tra danh sách stroke không rỗng; tối đa 16 nét và 8.192 điểm trước khi loại trùng. Mỗi nét không rỗng; x/y phải là số hữu hạn trong ±1.000.000, không nhận chuỗi/bool.
2. Bỏ điểm trùng liên tiếp trong từng nét; không thay đổi input. Không gộp hai stroke dù chúng chạm nhau.
3. PenDigits có chiều y tăng lên, được xác minh bằng hình mẫu 10 lớp; chuyển `y=-y`. Canvas y-down giữ nguyên.
4. Tính bounding box toàn chữ số. Tâm là `(min+max)/2`; scale là `max(width,height)`. Chia cả hai trục cho cùng scale để giữ tỷ lệ. Nếu scale=0 hoặc toàn bộ stroke không có chuyển động, trả lỗi `TOO_SHORT`.
5. Tính độ dài cung bằng khoảng cách Euclidean trong từng nét sau chuẩn hóa. Nét di chuyển được cấp ít nhất 2 điểm để giữ đầu/cuối; nét một điểm được giữ đúng 1 điểm.
6. Số token còn lại được chia theo tỷ lệ độ dài cung: lấy phần nguyên, chia phần dư cho fractional remainder lớn nhất; hòa điểm ưu tiên nét xuất hiện trước. Không cấp thêm token cho nét một điểm không có độ dài.
7. Nội suy tuyến tính ở các mốc cách đều theo cung của từng nét, gồm điểm đầu/cuối. Không có nội suy từ cuối nét trước sang đầu nét sau.
8. Nối token theo thứ tự nét; `stroke_start=1` ở token đầu mỗi nét, 0 ở các token còn lại. Tính hình học float64 rồi cast float32 ở đầu ra; không padding, không mean/std học từ dataset.

Giới hạn 16 nét bảo đảm ngân sách tối thiểu ≤32 token, phù hợp cả L=64 và L=128. Tất cả mẫu nguồn đã qua kiểm tra giới hạn; không cắt/bỏ mẫu. Đây là kiểm tra hình học, không phải bộ phát hiện mọi hình vẽ ngoài 10 chữ số.

## Tensor và kết quả

| Trường | Giá trị |
|---|---|
| Layout | NCT, channel-major |
| Tensor train | float32 `[N,3,L]` |
| Tensor ONNX | float32 `[1,3,L]`, fixed shape |
| Chiều dài | 64 hoặc 128 theo config của model; không tự đổi khi inference |
| Kênh | 0=x, 1=y, 2=stroke_start |
| Miền | x/y trong [-0,5;0,5]; stroke_start thuộc {0,1} |
| Nhãn | int64 0–9 |
| Output | 10 logits theo nhãn 0–9; softmax ngoài model |
| Tiền xử lý | Áp dụng đúng một lần trước model |

Top-3 sắp theo softmax giảm dần; hòa điểm theo ID tăng dần. Điểm softmax không phải xác suất đúng đã hiệu chuẩn. Mọi result của ứng dụng D5 phải gắn model version, request ID và drawing revision.

## Lỗi

`EMPTY`, `EMPTY_STROKE`, `NUMBER`, `STROKES`, `POINTS`, `TOO_SHORT`, `LENGTH`, `AXIS`, `BUDGET`, `INTERNAL`. Tên mã ổn định giữa Python/TS; thông báo giao diện được dịch sang tiếng Việt ở D5. Thay đổi thuật toán/giới hạn/hệ tọa độ phải đổi version và kiểm chứng lại tensor/model.

## Split và tái lập

- Development pool nguồn: 7.494 mẫu; train 5.999, validation 1.495, phân tầng 10 lớp; seed 42, val từng lớp `floor(n*0.2)`.
- Test chính thức: 3.498 mẫu, không thay đổi. Theo tài liệu UCI, test thuộc 14 người khác development pool.
- Train/validation có thể chung người viết vì writer ID chưa xác minh. Không công bố validation writer-disjoint.
- L=64/128 dùng cùng sample ID và split. Manifest chứa hash nguồn, entries và hash nội dung X/y/sample IDs. Các thông số cần chọn đều dùng validation; augmentation chỉ train.

Mỗi manifest entry giữ `source`, `source_index`, `source_line`, `segment_range`, nhãn, comment gốc, `writer_id=null` và SHA-256 chuỗi nét. Verifier đối chiếu từng entry với nguồn giải nén, nhãn/ID trong NPZ và metadata tiền xử lý; đồng thời kiểm tra L64/L128 có cùng provenance. Hash duplicate chỉ đo chuỗi gốc trùng hoàn toàn, không đo mọi hình học gần giống nhau.

Preprocessing Python nhận tọa độ real scalar (không bool/chuỗi/container), không dùng timestamp. Schema/timestamp của JSON ứng dụng được validator D5 kiểm tra ở lớp trước preprocessing; việc hoàn thành D1 không đồng nghĩa đã triển khai validator ứng dụng.
