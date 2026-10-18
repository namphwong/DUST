# DrawAI — Sổ tay điều phối P3

## Mục tiêu và trạng thái

Chuẩn bị250 mẫu pilot từ5 người thật; P3 phần công cụ đã sẵn sàng, chưa có mẫu người thật được bàn giao. Không dùng fixture test/chuột Playwright thay pilot. Quy mô đợt chính6.000 mẫu, split20/5/5 người theo plan đã khóa. [Protocol](protocol.md), [phiếu đồng ý](consent-template.md), [dataset card dự thảo](dataset-card.md).

## Chuẩn bị

1. Đọc protocol, ghi kênh liên hệ vào phiếu consent; cấp P001–P005 cho5 người khác nhau. Giữ mapping người thật↔mã riêng, dùng registry-template.csv trong kho riêng. Dùng cùng token điều phối cho cùng người để phát hiện đăng ký lại; mã khác nhau không tự chứng minh người khác nhau.
2. Hoàn tất ba lựa chọn consent và mã hồ sơ C-P001... thực tế. Không đánh dấu đồng ý thay người khác. Không lưu phiếu/mapping vào public/Git.
3. Chạy `npm run dev -- --port 4183 --strictPort` trong web; mở http://127.0.0.1:4183/collect. Không cần tải model/Python cho thao tác viết.
4. Phiên này phục vụ local127.0.0.1. Thử cảm ứng/bút thật cần mở công cụ trên thiết bị đó bằng môi trường HTTPS/localhost thích hợp; chưa triển khai hosting/LAN ở P3. Giả lập mobile không phải nghiệm thu cảm ứng thật.

## Bàn giao file private

Thư mục đã tạo và bị Git ignore:

```text
data/private/
  plans/private-v1/plan.json
  plans/private-v1/participants-reserved.csv
  inbox/pilot/                 chỉ JSON phiên mới nhất của mỗi người
  inbox/main/                  dành P4, không trộn pilot
  quarantine/                 bản lỗi/cũ giữ riêng, tự tạo khi cần
  reviews/                    mỗi lượt audit một thư mục mới
  builds/                     staging, chưa phải Kaggle package
```

Plan có35 mã dự kiến, không có nghĩa35 người đã tham gia. Bản policy public ở docs/drawai/collection/split-policy.json có cùng hash. Nếu cần tạo lại plan khi máy mới:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.private_data plan
```

Lệnh từ chối ghi đè thư mục không rỗng. Không sửa plan để hợp thức hóa thiếu quota/split sau thu. Không đặt private files vào web/public/dist.

## Audit và build sơ bộ

Cần env Python đã có NumPy/Pillow/jsonschema; `.venv-dinov3` hiện tại đã kiểm chứng. Chạy từ root, thay tên thư mục output ở mỗi lần:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.private_data audit --input data/private/inbox/pilot --output data/private/reviews/pilot-001
./.venv-dinov3/Scripts/python.exe -m training.drawai.private_data build --input data/private/inbox/pilot --output data/private/builds/pilot-001
```

Audit trống trả0 mẫu/quota_complete=false. Build trống hoặc có schema/label/device/time/duplicate errors bị từ chối. Build không đủ quota được phép chỉ làm staging; muốn bắt buộc đủ5×50:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.private_data build --input data/private/inbox/pilot --output data/private/builds/pilot-complete-001 --require-complete
```

Đợt chính P4 dùng `--stage main --input data/private/inbox/main`, chỉ sau khi xem báo cáo pilot và giữ/phiên bản hóa protocol. Mỗi mã có đúng một phiên canonical; resume JSON cũ để tiếp tục. Nếu phát hiện file hỏng/sai protocol đã ghi: cách ly phiên đầy đủ, ghi rejection-log.csv, điều phối thu lại theo cùng mã, giữ lịch sử ngoài inbox. Không xóa mẫu lẻ hoặc sửa nhãn trong JSON.

Audit kiểm tra schema strict, consent đầy đủ, số thực hợp lệ, nhãn/prompt, lịch vòng, bounds/tổng điểm, pointer thực, timestamp có timezone/thứ tự, mã người có trong plan, lặp ID/session/geometry, quota và thống kê người/lớp/thiết bị/split. Đúng hình học nhưng hình khác gần trùng được cảnh báo bằng dHash224→9×8/Hamming≤2 và raster hash. Không tự loại nét theo model. Báo cáo lưu tối đa5.000 cặp gần trùng để giới hạn RAM, kèm tổng số/truncated; P4 phải giải quyết toàn bộ, không bỏ qua phần bị giới hạn. Gần trùng chỉ là gợi ý rà soát, không phải bằng chứng cùng người hay sai nhãn.

Build tạo raw JSON/mỗi mẫu, PNG224, NumPy float32[3,128], manifest.csv mapping chung sample_id/split và SHA-256 của từng file; plan và manifest có hash. Không augmentation; pilot luôn pilot-dev. `dataset_locked=false`, `release_ready=false` kể cả đủ quota. Không tự công bố hoặc mở test để tune. Timestamp thu và session ID còn trong raw staging; cần rà soát/bỏ thông tin thời gian chi tiết và trường không cần thiết trước gói public P8. Mã hồ sơ consent không đưa vào raw per-sample/manifest.

## Đóng pilot P3

- Đủ5 người khác nhau P001–P005, mỗi người đủ5×10=50 mẫu; tổng250, đủ từng lớp/vòng.
- Điều phối xác nhận nguồn người thật/consent/dedup danh tính riêng; phần mềm chỉ kiểm tra lời khai trong JSON.
- Audit không có errors; rà soát và ghi kết luận duplicate/near-duplicate, lỗi thu, tỷ lệ loại theo lý do protocol, thiết bị và khả năng tiếp tục/tải lại.
- Điền pilot-report-template.md bằng số liệu thực, không điền số lượng test tự động vào mục người tham gia. Hết pilot mới quyết định protocol cần v2 hay giữ v1 trước P4.
- Chưa đủ người/mẫu hoặc chưa có bằng chứng consent thì P3 chưa hoàn thành toàn bộ. Giữ trạng thái TECHNICAL_READY_WAITING_FOR_HUMAN_PILOT.

## Tái kiểm chứng phần mềm

```powershell
./.venv-dinov3/Scripts/python.exe -m unittest discover -s training/tests -p "test_drawai*.py" -v
cd web
npm run build
npx playwright test --config playwright.drawai-collection.config.ts
npx playwright test --config playwright.drawai-p3-regression.config.ts
cd ..
./.venv-dinov3/Scripts/python.exe tools/verify_phase3.py
```

Logs Python cần ghi artifacts/drawai/phase3/python-tests.log nếu chạy auditor. Browser fixture tại artifacts/drawai/phase3/technical-only được sinh tự động, gắn nhãn synthetic; không copy vào inbox. Không chạy P2 documentation generator để cập nhật tiến độ P3 vì đó là generator tài liệu lịch sử P2.
