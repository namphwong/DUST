# DrawAI P4 — Rà soát nguồn dữ liệu trước khóa

P4 không tự xác nhận người thật hoặc đồng ý. Điều phối phải thực sự kiểm tra hồ sơ riêng và điền review; tuyệt đối không tích tất cả chỉ để lệnh chạy qua. Hiện chưa có250 mẫu pilot/6.000 mẫu chính.

## Điều kiện đi trước

Hoàn tất pilot P3:5 người khác nhau P001–P005,50 mẫu/người; đủ từng nhãn/vòng, không dùng test fixture. Điền báo cáo pilot, xử lý sự cố và quyết định giữ protocol-v1 hay đổi phiên bản trước khi thu chính. Đợt chính cần30 người mới M001–M030,200 mẫu/người, theo split20/5/5 đã khóa từ P3; không đổi mã/split sau khi xem nét/model.

Dùng `/collect`, [protocol](../collection/protocol.md) và [sổ tay điều phối](../collection/operator-guide.md). Mỗi inbox chỉ có bản JSON mới nhất của một phiên canonical cho mỗi người. Bản cũ/lỗi giữ trong quarantine ngoài inbox, có log/hash và lý do protocol. Không loại vì dự đoán sai.

## Sinh draft gắn với chính dữ liệu đang có

Từ root, mỗi lượt output dùng thư mục mới:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.private_dataset review --stage pilot --input data/private/inbox/pilot --output data/private/reviews/pilot-real-001
```

Output: review.json chưa ký (mọi xác nhận=false), source-inventory.json, audit.json và readiness.json. Inbox trống vẫn sinh draft để xem các blocker; không phê duyệt/khóa dữ liệu. Các draft `p4-pilot-draft-v1` và `p4-main-draft-v1` hiện có được sinh khi inbox trống, sẽ **không dùng lại để ký sau khi thêm dữ liệu**.

Review gắn SHA-256 với plan, danh sách/hash từng file JSON nguồn và toàn bộ kết quả audit. Thay file/nhãn/mã hoặc thêm snapshot làm review cũ không còn hiệu lực; phải sinh lượt review mới. Không sửa trường hash để lách lỗi.

## Người điều phối điền review.json

- `reviewer_ref`: mã người rà soát như R-001, không nhập email/tên vào bản dùng chia sẻ. `reviewed_at`: giờ ISO thực tế có timezone, sau lúc thu mẫu.
- `confirmations`: chỉ chuyển từng mục thành true sau khi đã xác minh nguồn người thật/consent, người không trùng, split cấp trước thu, không loại theo model, các phiên lỗi được log/cách ly và thu lại đúng protocol.
- Mỗi người có `person_token=person_<32 ký tự hex>` do điều phối cấp riêng và dùng ổn định cho **cùng một người** qua pilot/main. Không đổi token để né kiểm tra trùng người. Không đưa token/mapping vào Git/Kaggle; lưu mapping nhận diện bên ngoài công cụ.
- `human_origin_verified`, `consent_record_verified` chỉ true sau kiểm tra hồ sơ thật; `withdrawn` giữ false khi người còn đồng ý. Nếu đã rút, cách ly nguồn và thu lại theo kế hoạch được duyệt; không đổi cờ để bỏ qua yêu cầu rút.
- `near_groups`: xem toàn bộ sample_id trong từng nhóm (nhóm là connected component của các cặp dHash gần nhau). Chỉ chọn `keep_distinct_verified` với lý do10–1.000 ký tự khi đã xem nét gốc và xác minh là các lần viết độc lập hợp lệ. Nếu copy/lỗi protocol: cách ly/thu lại, sinh review mới. Không đơn thuần kết luận “model đoán đúng”.
- `exclusion_log`: điền mọi phiên bị cách ly/thay thế thực tế: session_id gốc, SHA-256 file gốc, lý do và replacement_session_id. Lý do hợp lệ: corrupt_file, empty_drawing, label_error_verified, protocol_violation, exact_duplicate, consent_withdrawn. Không có reason model_predicted_wrong/accuracy. Giữ file gốc/hồ sơ liên quan ngoài inbox để kiểm toán; script không xác minh được hồ sơ ngoài này.

Audit P3 vẫn chỉ lưu tối đa5.000 cặp gần trùng để giới hạn report, nhưng P4 bổ sung **nhóm đầy đủ của tất cả các cặp** kể cả khi danh sách cặp bị cắt. Review phải có quyết định cho mỗi nhóm; không bỏ qua các nhóm lớn.

## Phê duyệt pilot và rà soát đợt chính

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.private_dataset approve-pilot --input data/private/inbox/pilot --review data/private/reviews/pilot-real-001/review.json --output data/private/approvals/pilot-v1
```

Chỉ chạy được khi đủ quota, audit không lỗi và review đầy đủ. `pilot-approval.json` ghi hash nguồn/review, hash token người, geometry hashes và sample IDs để ngăn tái sử dụng người/mẫu pilot trong main; không phải chứng thư xác thực danh tính hay chữ ký số. Điều phối giữ approval cùng hồ sơ ngoài Git.

Khi đã thu và kiểm tra đủ main:

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.private_dataset review --stage main --input data/private/inbox/main --pilot-approval data/private/approvals/pilot-v1/pilot-approval.json --output data/private/reviews/main-real-001
```

Hoàn tất review main theo cùng quy trình. Mã M khác P không tự chứng minh người khác nhau; công cụ so token ổn định và geometry/sample IDs, người điều phối vẫn phải kiểm tra thực tế. Nguồn public metadata không chứa mapping danh tính. Các xác nhận thủ công/checksum chỉ ghi nhận lời xác nhận và chống sai lệch file; không thể chống một người cố ý cung cấp hồ sơ giả.
