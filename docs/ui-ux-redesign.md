# Thiết kế lại UI/UX trước Phase 7

Ngày 03/10/2026. Đã triển khai trên MotionSense local; chưa deploy Vercel. Giữ nguyên checkpoint, artifact và thuật toán CSV/worker đã kiểm chứng ở Phase 6.

## Mục tiêu và thay đổi

Trang là công cụ nhập dữ liệu/nhận kết quả, không còn đặt hero giới thiệu model lớn trước khu vực sử dụng. Bốn bước thể hiện trạng thái thực: nhập dữ liệu → kiểm tra đầu vào → phân tích → nhận kết quả.

- Đầu vào: file CSV là lựa chọn chính, bấm chọn hoặc kéo thả. Có tab dán CSV, textarea và nút áp dụng rõ ràng; sửa bản nháp không tự thay nguồn đang phân tích. Cả ba cách dùng cùng parser/worker, giới hạn và báo lỗi.
- CSV ví dụ có thể tải ngay: 128 điểm tổng hợp trong g, không gán nhãn hoạt động thật. Sáu mẫu UCI vẫn có nguồn và nhãn tham chiếu riêng, thử bằng hai nút chọn/phân tích.
- Sau nhập: hiển thị tên nguồn/số điểm; đơn vị và xác nhận gravity; kiểm tra resample, thống kê bỏ đoạn/đuôi và xác nhận trước khi chạy. Có hướng dẫn hành động tiếp theo khi nút phân tích chưa bật.
- Đầu ra có khu vực riêng, ban đầu hiển thị những gì người dùng sẽ nhận nhưng không tạo số liệu/nhãn giả. Khi đang đọc/chuẩn bị dùng progress không xác định; khi suy luận hiện số cửa sổ thật. Có hủy, thử lại và thay CSV khi bận.
- Sau hoàn tất: tổng cửa sổ, đoạn có kết quả, đơn vị nguồn; phân bố số cửa sổ theo sáu nhãn. Đây **không phải thời lượng hoạt động** vì các cửa sổ chồng lấn, cũng không phải accuracy hay tỷ lệ nhãn thật.
- Chi tiết cửa sổ đang chọn hiển thị ID, đoạn, khoảng thời gian, nhãn và sáu điểm softmax. Chọn hàng trong bảng cập nhật đúng chi tiết; hàng hiện tại có màu nền/aria-pressed. Bảng phân trang 20 hàng, xuất toàn bộ kết quả chứ không chỉ trang đang xem.
- Nút tải kết quả được nhấn mạnh và có thông tin model/contract. Biểu đồ nguồn/năm dòng CSV nằm trong khu vực kiểm tra riêng bên dưới; thông tin Kaggle/UCI/metric chuyển xuống cuối trang.
- Desktop có hai cột đầu vào/đầu ra; mobile một cột, khi hoàn tất focus tiêu đề kết quả và cuộn đến đó. Có đường dẫn Xem kết quả. Tab hỗ trợ phím mũi tên; checkbox/nút, bảng cuộn, focus/skip link và status/alert hỗ trợ bàn phím.
- Đổi nguồn/đơn vị/gravity/xác nhận không giữ output cũ cho cấu hình mới. Không upload CSV, không backend trả phí; cảnh báo F1 87,05%, domain shift và giới hạn nghiên cứu luôn còn.

## Kiểm chứng

Production TypeScript/Vite build; **32 browser tests**, gồm 29 test hồi quy Phase 1/4/5/6 và ba test UI/UX mới. Python unit tests 19; parity 17 fixture CSV/18 cửa sổ giữ nguyên. Các số đo ở báo cáo runtime là phép đo local, không phải SLA.

Test mới kiểm empty output, tab/phím mũi tên, dán CSV → resample → suy luận thật → sáu điểm/metadata cửa sổ khớp golden → xuất cả file; phân bố cộng đúng tổng cửa sổ; đổi đơn vị xóa output; kéo thả dùng đúng validator; CSV ví dụ import hợp lệ; desktop 1366×768/mobile viewport 390×844 không tràn ngang. Mobile viewport không phải test điện thoại thật hay chứng nhận WCAG.

Bằng chứng: [UI verification](../artifacts/ui-redesign/verification.json), [Playwright results](../artifacts/phase6/playwright-results.json), [tests](../web/tests/ux-redesign.spec.ts), [Phase 6 audit](../artifacts/phase6/verification.json).

Ảnh đã kiểm tra trực quan:

- [Desktop đầu vào](../artifacts/ui-redesign/desktop-input.png), [desktop đầu ra](../artifacts/ui-redesign/desktop-output.png).
- [Mobile đầu vào](../artifacts/ui-redesign/mobile-input.png), [mobile đầu ra](../artifacts/ui-redesign/mobile-output.png).

## Sử dụng / chạy lại

Mở hoặc tải lại http://127.0.0.1:4173/. Nhập CSV/dán CSV → chọn đơn vị → xác nhận gravity → Kiểm tra resample → xác nhận dữ liệu → Suy luận CSV → chọn cửa sổ / Tải CSV kết quả.

```powershell
cd web
npm run build
npm test
cd ..
.venv-phase1/Scripts/python.exe training/verify_phase6_results.py
```

Tests dùng server riêng 4174, không đóng preview 4173. Chỉ UI và các đường nhập mới thay đổi; worker bundle/model hash giữ nguyên. Phase 7 vẫn chờ triển khai Vercel và kiểm tra public runtime/cache/quota.
