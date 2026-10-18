# DrawAI — Protocol thu và đánh giá nét canvas D6

Chốt ngày 11/10/2026 trước khi thu. **Chưa có người tham gia/thiết bị; số mẫu người thật hiện tại là 0.** Fixture, demo PenDigits và nét do Playwright sinh không được tính vào tập này.

## Thiết kế

- Ít nhất 5 người, ID ẩn danh P001…P005; không ghi tên, số điện thoại, email hoặc tài khoản. Người tham gia đồng ý dùng nét cho đánh giá đề tài và có thể dừng trước khi bàn giao.
- Mỗi người viết 0–9 năm lần: 50 mẫu/người, tổng ít nhất 250. Năm vòng, mỗi vòng viết 0→9; protocol cố định `drawai-canvas-eval-v1`. Mỗi người dùng một loại thiết bị nhập đã ghi (mouse/touch/pen).
- Dùng `/drawai`, vẽ tự do một chữ số đầy đủ, không nạp mẫu. Gắn nhãn người viết theo chữ số được yêu cầu và xuất JSON **trước khi bấm Nhận dạng**; không cho người viết xem model prediction để sửa nét theo model.
- Có thể undo/clear nét khi viết; không loại mẫu vì model sai. Ghi lại lần lỗi thu dữ liệu trước khi đánh giá, không thay sample đã thu bằng mẫu được model nhận đúng. Mỗi mẫu có participant ID, label, repetition, input device và file gốc.
- Toàn bộ tập này là held-out evaluation cho release `drawai-swin-v1`; không dùng fine-tune hoặc chọn seed/config/threshold. Nếu muốn nghiên cứu mới, chia development/evaluation theo người **trước** khi fine-tune, tạo release mới và protocol mới. Không gọi lại tập đã xem là test mù.

## Thao tác và bàn giao

1. Chọn ID ẩn danh và ghi loại máy/browser, input device. Người phụ trách xác nhận consent và nguồn người thật; công cụ không tự chứng minh danh tính hoặc tính xác thực của lời xác nhận.
2. Thu theo năm vòng; nhãn đúng do người viết/điều phối cung cấp, không lấy nhãn model. Xuất JSON vào `data/canvas/drawai-v1/P001/digit-0-repeat-1.json`…
3. Điền manifest từ [template](../../examples/drawai/canvas-manifest.template.json). Mỗi participant có consent=true, device metadata; mỗi sample trỏ đến file tương đối trong cùng dataset root. Không đổi schema nét để nhét tên người vào JSON.
4. Người phụ trách xác nhận `human_source_attested=true` sau khi kiểm tra nguồn. Trường này là xác nhận của người cung cấp, không phải bằng chứng tự động. Giữ các file JSON gốc, manifest/hash và protocol cùng report.
5. Chạy validator/evaluator khi đã đủ mẫu. Script từ chối tập thiếu người/lớp/lần lặp, sample trùng, nhãn thiếu/sai, nguồn dataset/synthetic, timestamp sai, file ngoài dataset root hoặc release sai.

Cấu trúc bản ghi (minh họa cách điền, không phải người/mẫu đã thu):

```json
{
  "participant": {"id":"P001","consent":true,"input_device":"mouse","device_description":"Laptop Windows / chuột","browser":"Chrome phiên bản thực tế"},
  "sample": {"participant_id":"P001","label":0,"repetition":1,"file":"P001/digit-0-repeat-1.json"}
}
```

Đưa object participant vào mảng `participants`, object sample vào `samples`; không thêm hai key minh họa vào root manifest. Các field manifest/participant/sample được kiểm tra chính xác để tránh ghi metadata nhận diện cá nhân ngoài protocol.

```powershell
$env:DUST_DLL_DIRECTORY = 'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
./.venv-phase1/Scripts/python.exe -m training.drawai.evaluate_canvas --manifest data/canvas/drawai-v1/manifest.json --output-dir artifacts/drawai/canvas-eval-v1
```

## Báo cáo

Accuracy/macro-F1, confusion matrix, precision/recall/F1/support theo chữ số; metric theo participant và input device; số mẫu và hash của manifest/file/model. Báo mọi sample sai theo ID, không chỉ chọn lỗi dễ giải thích. Đây là một tập nhỏ, cần đọc support từng nhóm; không suy rộng thành độ chính xác bảo đảm cho mọi người/thiết bị.

D6-06/D6-07 chỉ được đánh dấu đã chạy khi có dữ liệu nguồn người thật đã xác nhận và report. Chuẩn bị protocol, template, validator hoặc synthetic tests không thay thế việc thu nét. Điện thoại thật cũng cần thử riêng gesture/pointer/viewport/performance; giả lập mobile/DPR không tính là thử cảm ứng thực.
