# Phase 5 — MotionSense, giao diện MVP

Đây là báo cáo phạm vi lịch sử Phase 5. CSV/full pipeline hiện đã hoàn thành ở [Phase 6](phase6-pipeline-report.md); các đoạn “chưa triển khai” bên dưới mô tả trạng thái khi kết thúc Phase 5.

Nghiệm thu 03/10/2026: hoàn thành **giao diện Phase 5**, không coi toàn bộ sản phẩm hoặc Phase 6 đã hoàn tất. Trang mặc định là MotionSense; trang kỹ thuật giữ tại `/?phase=4` và fixture untrained tại `/?phase=1`.

## Đã triển khai

- Giao diện tiếng Việt responsive: chọn dữ liệu, preview/biểu đồ XYZ, kết quả cửa sổ, hướng dẫn CSV và model card tóm tắt. Không tài khoản/backend/database hoặc API trả phí.
- Sáu CSV demo Phase 2: cửa sổ train đầu tiên mỗi lớp, không chọn theo dự đoán đúng. Hiển thị split/subject/dòng nguồn và nhãn tham chiếu riêng. Kiểm SHA-256 CSV, đúng 128 điểm/50 Hz, không nối timeline giả.
- Chọn mẫu → Phân tích mẫu: hai nút thao tác; ONNX chỉ tải khi phân tích. Model float32 thật của release har-swin-kaggle-v1, kiểm manifest/model/metadata hash. Không dùng prediction hard-code hay nhãn nguồn làm prediction.
- Biểu đồ raw XYZ, CSV m/s² hiển thị sang g khi chọn đơn vị; bảng giữ giá trị nguồn. Chart rút gọn ~600 điểm, không nối qua khoảng >100 ms, tô nền tối đa 200 khoảng. Đây không phải resample cho model.
- Argmax và sáu điểm softmax, nhãn nguồn và trường hợp không khớp. Cảnh báo test F1 87,05% chưa đạt 90%, domain shift, nhầm ngồi–đứng, điểm model không phải độ chính xác bảo đảm.
- Tải CSV mẫu và CSV kết quả thật của một cửa sổ: đủ các cột data-contract, model/contract version/source_unit; tạo file cục bộ.
- Nhập UTF-8/BOM, quoting/escaped quotes/CRLF, header đổi thứ tự, bỏ qua cột thêm có thông báo. Kiểm 10 MiB/100.000 dòng, thiếu/trùng cột, số không hữu hạn/ô trống, timestamp âm/trùng/giảm và malformed CSV; lỗi có dòng/cột liên quan khi xác định được.
- Preview số điểm/thời gian/tần số ước lượng/gaps/năm dòng đầu; chọn g/m/s² và xác nhận trọng lực. Đổi nguồn/file xóa result cũ; không hiển thị prediction cũ cho dữ liệu mới.
- Trạng thái đọc/tải/khởi tạo/suy luận, thử lại khi model lỗi, khóa nút trong lúc demo chạy. Keyboard, skip link, status/alert và viewport desktop/mobile đã test.

## Ranh giới Phase 6 — chưa triển khai

**CSV người dùng chỉ preview**: nút inference bị khóa, ghi rõ chờ Phase 6. Không giả vờ phân tích CSV hoặc lấy prediction mẫu cho file người dùng. Chưa resample 20 ms, kiểm tần số từng đoạn, chia đoạn/cửa sổ/đuôi, chuyển đơn vị cho toàn pipeline inference, Worker, tiến trình nhiều cửa sổ, hủy tác vụ và xuất nhiều cửa sổ. Parsing preview hiện trên main thread; chưa nghiệm thu responsiveness 100.000 dòng. Chưa cache release/mobile device benchmark/deploy Vercel.

WEB-01, WEB-09, phần keyboard/layout WEB-10 được nghiệm thu ở Phase 5. WEB-02–03 mới có phần preview/định dạng/UI, điều kiện sau resample còn thiếu. WEB-04–08 toàn pipeline/worker thuộc Phase 6. No-network test hiện tại chỉ bao phủ demo và đọc CSV preview, không thay thế test Phase 6.

## Kiểm chứng

**19/19 browser tests PASS**: 3 Phase 1, 5 Phase 4, 11 Phase 5. **14/14 Python unit tests PASS**; production build TypeScript/Vite, compileall/diff check PASS. Chromium headless 153.0.8010.12, desktop 1366×768 và viewport mobile 390×844 không tràn ngang. Không tuyên bố benchmark điện thoại hoặc tuân thủ WCAG đầy đủ.

`training/verify_phase5_demos.py` tạo golden logits/scores từ checkpoint PyTorch đã xác minh. Browser phân tích từng CSV và xuất kết quả: ID khớp PyTorch, mỗi score lệch ≤1e-5. Golden chỉ được test dùng, không phải prediction frontend dùng. Đây là sáu cửa sổ 50 Hz, **chưa phải full CSV/resample parity**.

Test gồm sáu mẫu/provenance/scores/download schema, nguồn mới xóa result, model 404/retry, quoting/BOM/reorder/cột thêm, missing/numeric/timestamp errors, file >10 MiB, nhận 100.000 dòng + blank cuối/từ chối 100.001, keyboard, không có upload/request khi preview CSV. Số thao tác đo là hai nút chọn mẫu/phân tích, không đếm scroll.

Bằng chứng: [browser](../artifacts/phase5/browser-verification.json), [golden](../artifacts/phase5/demo-reference.json), [desktop](../artifacts/phase5/desktop-proof.png), [mobile](../artifacts/phase5/mobile-proof.png), [tests](../web/tests/phase5.spec.ts), [checklist](../artifacts/phase5/verification.json).

## Chạy lại và handoff

```powershell
# Từ repo, cần checkpoint Kaggle đã import:
.venv-phase1/Scripts/python.exe training/verify_phase5_demos.py
cd web
npm run build
npm test
npm run preview -- --port 4173
```

Tests tự bật/tắt preview **4174**, không đóng phiên preview người dùng 4173. Mở/tải lại http://127.0.0.1:4173/, chọn mẫu → Phân tích mẫu. CSV chỉ kiểm tra/preview, chưa nhận prediction.

Phase 6: đưa parse/preprocessing/inference vào worker có cancellation/job ID; resample/segment/window đúng data-contract; Python ↔ browser CSV parity; 127/128/191/192 điểm, m/s²/g/gaps; CSV nhiều cửa sổ, responsiveness/no upload. Không đổi model/seed theo test đã mở.
