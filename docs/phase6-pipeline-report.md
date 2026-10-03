# Phase 6 — CSV người dùng, worker và toàn pipeline

Triển khai và nghiệm thu local ngày 03/10/2026. Model giữ nguyên release `har-swin-kaggle-v1` từ Kaggle; không train lại, không chọn lại seed hoặc điều chỉnh bằng test. Phase 6 làm tích hợp sản phẩm, không cải thiện macro-F1.

## Người dùng đã làm được gì?

Chọn CSV → chọn g hoặc m/s² → xác nhận gia tốc bao gồm trọng lực → **Kiểm tra resample** → đọc thống kê và xác nhận → **Suy luận CSV** → xem từng cửa sổ/sáu điểm → **Tải CSV kết quả**. Mẫu UCI vẫn chỉ cần hai nút chọn mẫu/phân tích; không resample cửa sổ UCI đã chuẩn bị.

CSV phải tuân thủ [hợp đồng](data-contract.md). Giới hạn 10 MiB, 100.000 dòng nguồn và 100.000 điểm sau resample. Không điền số 0, tự sort, tự đoán đơn vị hoặc xoay trục. Đoạn không đủ 128 điểm sau resample được bỏ có thống kê; không có cửa sổ nào thì báo TOO_SHORT. Body acceleration không có trọng lực không tương thích.

## Cách xử lý

- Worker đọc File/UTF-8 và parse CSV quoting/BOM/header đổi thứ tự; dữ liệu đầy đủ ở worker. Main thread chỉ nhận ~600 điểm biểu đồ, năm dòng đầu và thống kê, không nhận mảng 100.000 điểm để render.
- Trừ timestamp đầu phiên trước khi xử lý; m/s² chia 9.80665. Tách đoạn tại khoảng **>100 ms**, không vượt gap. Tần số median kiểm riêng từng đoạn đủ dài trong 10–25 ms; ngoài khoảng báo UNSUPPORTED_FREQUENCY.
- Nội suy tuyến tính trên lưới 20 ms bắt đầu tại đầu từng đoạn, không ngoại suy; nguồn đều 20 ms giữ giá trị nguyên. Cửa sổ 128, stride 64, không pad/không nối đoạn; báo điểm đuôi và đoạn ngắn bị bỏ.
- Chuẩn hóa channel-major bằng mean/std train-only, float32 và epsilon 1e-8; suy luận WASM một thread bằng session đã xác minh SHA-256 và metadata. Session dùng lại khi phân tích lại cùng nguồn; khi đổi nguồn, worker cũ bị hủy. HTTP cache phụ thuộc browser/server; cấu hình cache hosting thuộc Phase 7.
- Tiến trình cập nhật mỗi 16 cửa sổ, cùng cửa sổ đầu/cuối. Hủy bằng terminate worker, không chờ inference/mạng trả lời; source/request ID ngăn callback cũ thay state. Đổi file luôn có thể thực hiện khi đang bận và hủy worker cũ.
- Không xuất kết quả dở dang; sau hoàn tất có bảng 20 cửa sổ/trang, xem sáu điểm của cửa sổ được chọn. CSV xuất toàn bộ cửa sổ, segment/window ID, timestamp thật tương đối, model/contract version và đơn vị nguồn. Khoảng đầu–cuối mỗi cửa sổ là **2540 ms**, không cộng 20 ms giả.
- Đổi đơn vị, loại gia tốc hoặc bỏ xác nhận resample xóa kết quả và yêu cầu xác nhận lại. Hủy rồi có thể kiểm tra/chạy lại; lỗi tải model có thể thử lại.

Code: [pipeline](../web/src/csv-pipeline.ts), [worker](../web/src/pipeline-worker.ts), [client](../web/src/pipeline-client.ts), [giao diện](../web/src/motion-app.tsx), [đối chiếu Python](../training/csv_pipeline.py).

## Bằng chứng nghiệm thu

- 17 fixture CSV độc lập: 127/128/191/192, g/m/s², timestamp tuyệt đối, BOM/quoting/reorder, jitter, 100 Hz/40 Hz, nhiều đoạn, bỏ đoạn ngắn, ranh giới gap 100/101 ms, tần số không hỗ trợ, vượt giới hạn sau resample và thiếu gravity.
- 18 cửa sổ hợp lệ đối chiếu NumPy preprocessing và **checkpoint PyTorch thật**: tensor chuẩn hóa sai lệch tối đa **0** trên các fixture; logits sai lệch tối đa khoảng **2,86e-6**, trong atol 1e-5 / rtol 1e-4; mọi ID dự đoán khớp, từng score sai lệch ≤1e-5. Đây là fixture tổng hợp kiểm tra pipeline, **không phải dữ liệu hoạt động có nhãn hoặc phép đo accuracy mới**. Golden chỉ dùng trong test, không hard-code prediction frontend.
- 100.000 dòng → 1.561 cửa sổ, chạy đủ và xuất 1.561 dòng kết quả. Đo thời gian hủy ≤1 giây, heartbeat main thread 20 ms vẫn chạy xuyên suốt; kiểm giới hạn khoảng heartbeat lớn nhất <500 ms trên máy tham chiếu. Số đo thực từng lần ở [performance.json](../artifacts/phase6/performance.json), không coi là SLA cho mọi máy.
- Nhập file mới giữa inference không nhận result cũ; hủy không có nút xuất kết quả dở dang. Unit change làm mất xác nhận/result. Lỗi EMPTY_FILE, FILE_TOO_LARGE, TOO_MANY_ROWS, MISSING_COLUMN, DUPLICATE_COLUMN, INVALID_NUMBER (dòng/cột), INVALID_TIMESTAMP, TOO_SHORT, INCOMPATIBLE_SIGNAL, UTF-8/quoting và frequency được kiểm.
- Browser không có request ghi dữ liệu; request cần thiết chỉ tải tài nguyên static (worker, model, WASM, metadata/mẫu). CSV cục bộ không được đưa vào URL/body mạng, localStorage hoặc backend.
- Desktop 1366×768 và mobile **viewport** 390×844 không tràn ngang, xem ảnh. Checkbox/prepare/confirm/analyze CSV thực hiện được bằng bàn phím; không tuyên bố chứng nhận WCAG hoặc benchmark điện thoại thật.
- **29/29 browser tests và 19/19 Python unit tests**, production TypeScript/Vite build. Regression gồm Phase 1/4/5, sáu demo, artifact sai hash/version/trained và model 404/retry.

Tổng hợp: [verification.json](../artifacts/phase6/verification.json). Chi tiết: [Playwright](../artifacts/phase6/playwright-results.json), [Python golden](../artifacts/phase6/python-reference.json), [browser parity](../artifacts/phase6/browser-parity.json), [UI/privacy](../artifacts/phase6/ui-verification.json), [desktop](../artifacts/phase6/desktop-proof.png), [mobile](../artifacts/phase6/mobile-proof.png), [tests](../web/tests/phase6.spec.ts).

## Chạy lại

```powershell
# Tại repo, cần checkpoint Kaggle đã import và môi trường PyTorch của Phase 3:
.venv-phase1/Scripts/python.exe training/verify_phase6_pipeline.py
cd web
npm ci
npm run build
npm test
cd ..
.venv-phase1/Scripts/python.exe training/verify_phase6_results.py
```

Tests tự mở/tắt server 4174, không đóng preview người dùng 4173. Tải lại [web local](http://127.0.0.1:4173/); có thể dùng [CSV tổng hợp 192 điểm](../examples/phase0/valid-session-g.csv) để kiểm tra thao tác, không gắn nhãn hoạt động thật cho CSV này.

## Nghiệm thu và phần chưa làm

WEB-01–WEB-10 đã có bằng chứng local tương ứng ở checklist. F1 vẫn **87,05%**, chưa đạt mục tiêu 90%, dữ liệu ngoài UCI chưa được đánh giá. Chỉ là demo nghiên cứu, không dùng cho y tế/an toàn.

Phase 7 tiếp theo: deploy Vercel Hobby, kiểm quota/tài khoản, URL public tải model/WASM/worker, headers/cache theo version, mạng chậm/tải lỗi và browser/device thực tế. Phase 8: nghiệm thu phát hành, rollback và tài liệu vận hành. **Chưa deploy public; không phát sinh dịch vụ/backend trả phí trong Phase 6.**
