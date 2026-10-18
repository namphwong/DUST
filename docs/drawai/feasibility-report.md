# DrawAI — Báo cáo D2

Kiểm chứng lại local ngày 11/10/2026 sau đợt kiểm toán D1. **D2 đạt đủ D2-01–D2-07** trên Python/ONNX CPU và Chromium WASM trong Web Worker. Artifact kỹ thuật là `trained=false`, chưa dùng nhận dạng chữ số. Đợt này chạy lại import runtime, model unit tests, forward/backward/tiny train/export/parity, production build và browser tests; không lấy report cũ làm kết quả mới.

## Mô hình và gradient

Dùng backbone thật `SwinTransformer_1D`, 3 kênh, 10 logits, patch 2, window 8, embed 32, depths [2,2,2], heads [2,4,8], cho L64 và L128. Batch 1 và batch 4 có output đúng; loss và gradient hữu hạn. Attention và patch embedding có gradient khác 0 sau optimizer steps với cách khởi tạo nguyên bản. Một sample train mỗi lớp được học đạt 10/10 trên tập con sau 60 bước; đây chỉ là kiểm tra khả năng học, không phải accuracy test.

Cả hai cấu hình có 545.038 tham số. Report ghi L1 gradient của patch embedding, shifted attention, MLP và head từng optimizer step; tất cả hữu hạn và khác 0 ở bước cuối. Backbone khởi tạo residual post-norm bằng 0 nên attention có thể chưa có gradient ngay bước đầu; kiểm tra không sửa khởi tạo này khi học. Unit tests còn kiểm tra eval deterministic, batch independence, input sai kênh/chiều dài và config JSON khớp factory.

Tiny train chỉ dùng 10 mẫu đầu tiên theo source order trong tập train, mỗi lớp một mẫu; sample ID và train tensor hash được lưu. Loss giảm từ 2,3033 ở bước 1 xuống 0,3992 ở bước 60; không đọc validation/test cho phép thử học. Không lưu checkpoint sản phẩm từ tập con này.

Export fixture dùng seed 42 và đặt residual post-norm weights thành 1 để phép thử thực sự chạy attention/MLP. Các weights này không phải checkpoint train; không thay backbone hoặc cách khởi tạo cho training. ONNX fixed shape, float32, opset 17; checker đạt.

## Kết quả

| Đo | L64 | L128 |
|---|---|---|
| Model bytes | 2.359.213 | 2.368.204 |
| Fixture mỗi cấu hình | 20 | 20 |
| Sai số logits tối đa PyTorch–ORT CPU | 3,58e-7 | 2,98e-7 |
| ORT CPU p95, 4 threads | 0,64 ms | 1,02 ms |
| Browser WASM median, 1 thread | 1,10 ms | 1,60 ms |
| Browser WASM p95, 1 thread | 1,80 ms | 2,40 ms |
| Browser load artifact | 103,90 ms | 105,00 ms |
| Browser init WASM/session | 536,10 ms | 393,70 ms |
| Sai số logits tối đa PyTorch–WASM | 3,28e-7 | 3,41e-7 |

20 fixture gồm 7 hình học, 10 mẫu train có nguồn, 3 tensor kỹ thuật; 17 fixture có nét được xử lý bằng TypeScript trước ONNX. Cả hai cấu hình đạt tensor tolerance 2e-7 và logits tolerance atol=1e-5/rtol=1e-4. Sai số tensor tối đa L64 là 2,22e-16, L128 là 0.

Mỗi cấu hình kiểm tra 14 input không hợp lệ: canvas trống, nét rỗng, một điểm, các điểm rời, NaN, quá nhiều nét, quá nhiều điểm, Infinity, tọa độ chuỗi/bool, tọa độ vượt giới hạn, tổng số điểm nhiều nét vượt giới hạn, chiều dài và trục y sai. Tất cả trả đúng mã lỗi.

Benchmark dùng cùng fixture tensor random cho Python và browser, 5 warm-up + 100 inference. Python dùng percentile nội suy tuyến tính của NumPy; browser dùng nearest rank. Đây là hai runtime/thread count khác nhau; báo riêng, không coi như phép so tốc độ công bằng giữa backend. Dung lượng bảng chỉ là ONNX, không bao gồm WASM runtime. Cả hai artifact đạt mục tiêu sơ bộ ≤10 MiB và browser p95 ≤100 ms trên máy local.

Browser thực: HeadlessChrome 153.0.8010.12 trên Windows, backend CPU WASM 1 thread trong worker; máy Ryzen 7 H 255, 16 logical cores, RAM 16.438.382.592 bytes. Đây là đo laptop local, chưa kiểm tra điện thoại/Edge hoặc chứng minh latency production. Load/init và logits đầy đủ có trong report JSON; các lần tải sau có thể chịu ảnh hưởng cache runtime của trình duyệt.

## Tests và artifact

- 4 model unit tests đạt: config/factory, eval/batch independence, shape sai và gradient sau optimizer với khởi tạo gốc.
- 6 Playwright tests đạt: parity cả L64/L128 và benchmark; model thiếu rồi retry thành công; hash sai bị từ chối; release trained bị từ chối ở trang kỹ thuật; preprocessing layout sai dù cập nhật hash đúng; fixture chứa null dù cập nhật hash đúng. Các trường hợp lỗi không sinh PASS.
- Production build TypeScript/Vite đạt.
- Python report: `artifacts/drawai/feasibility/python-report.json`.
- Browser report: `browser-report.json`; suite report `playwright-report.json`; ảnh `desktop.png` cùng thư mục.
- Tổng hợp task và hash code/artifact/bằng chứng: `d2-review.json` cùng thư mục.
- Artifact thử: `web/public/models/drawai-feasibility-l64/`, `drawai-feasibility-l128/`; có release/hash/preprocessing/report.
- Trang kỹ thuật: `/drawai/check`. [Lệnh chạy lại](environment.md).

Lỗi ONNX DLL đã giải quyết bằng runtime search path trong tiến trình. Tracer warnings về shape là giới hạn exporter fixed shape, không áp dụng cho dynamic batch/length. Không có model nhận dạng đã train hoặc UI canvas sản phẩm; hai phần này ở D3–D5.

## Nghiệm thu theo task

| Task | Kết quả |
|---|---|
| D2-01 | Import thật Torch/NumPy/ONNX/ORT đạt; runtime/DLL directory/providers lưu trong report |
| D2-02 | Factory/backbone thật, config khớp JSON, L64/L128 đúng và tham số được thống kê |
| D2-03 | Batch 1/4, loss/logits/gradient hữu hạn; attention/MLP/embedding/head có gradient sau optimizer |
| D2-04 | Tập con 10 train samples có provenance học đạt 10/10, không báo như metric test |
| D2-05 | ONNX fixed shape float32 opset 17, checker và CPU parity đạt 20 fixture mỗi cấu hình |
| D2-06 | WASM worker parity đạt; metadata/hash/input lỗi và retry có tests thực |
| D2-07 | 5 warm-up + 100 inference; artifact và độ trễ đạt mục tiêu sơ bộ trên laptop |

**Bàn giao D3:** hai cấu hình khả thi, chưa chọn L tốt nhất; chọn bằng validation khi huấn luyện. Không dùng weights kích hoạt riêng của export fixture cho training hoặc xem logits trang kiểm chứng là dự đoán chữ số.
