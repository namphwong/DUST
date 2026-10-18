# DrawAI — Báo cáo P3: chuẩn bị thu dữ liệu riêng

Ngày18/10/2026. **Hoàn thành phần kỹ thuật P3; chưa hoàn thành toàn bộ P3 vì pilot người thật chưa thực hiện.** Trạng thái máy: `TECHNICAL_READY_WAITING_FOR_HUMAN_PILOT`, `technical_complete=true`, `phase_complete=false`. Inbox pilot có **0 mẫu/0 phiên người thật**, mục tiêu250 mẫu từ5 người.

## 1. Phạm vi đã triển khai

P3 chuẩn bị protocol, trang thu, schema và pipeline audit/build cho hai mô hình. Giữ nguyên DrawAI/Swin trên `/`, `/drawai`, `/drawai/`; không full-train, không khóa dataset và chưa công bố Kaggle. Không dùng demo PenDigits, dữ liệu sinh hoặc chuột tự động để báo đã thu pilot.

## 2. Quy mô và split đã khóa trước thu

| Đợt | Mã người | Mẫu/người | Số người | Tổng | Vai trò |
|---|---|---:|---:|---:|---|
| Pilot | P001–P005 | 10 chữ số ×5 vòng=50 |5|250|pilot-dev, kiểm tra công cụ/protocol |
| Chính/train | M001–M020 |10×20=200|20|4.000|huấn luyện |
| Chính/validation | M021–M025 |200|5|1.000|chọn cấu hình/checkpoint |
| Chính/test | M026–M030 |200|5|1.000|đánh giá cuối sau khóa lựa chọn |

Đã chọn phương án6.000 mẫu trong roadmap. Phương án2.000 mẫu chỉ là phương án dự phòng, chưa kích hoạt. Pilot và đợt chính phải là người khác nhau. Mỗi mã có một người, không đổi split theo nét hoặc kết quả model; điều phối cân đối thiết bị khi cấp mã trước thu. Phần mềm kiểm tra mã/plan, không xác minh được danh tính thật.

Policy public: `docs/drawai/collection/split-policy.json`; plan local: `data/private/plans/private-v1/plan.json`. SHA-256 nội dung plan: `1931d5d095a82295d5d3d9271fd0f1ffe46b08d0404620b4c92c3bd427bccd78`.35 mã trong plan là dự kiến, không phải35 người đã tham gia.

## 3. Trang thu `/collect`

Code: `web/src/drawai/collect-app.tsx`, `collection.ts`, `collect.css`; thêm route riêng trong `web/src/main.tsx`.

- Đăng ký pilot/main, mã người, loại thiết bị, mã hồ sơ consent. Chặn mã không thuộc đợt và thiếu một trong ba xác nhận.
- Ba phạm vi đồng ý hiển thị riêng, không tích sẵn: training; evaluation/reporting; Kaggle public. Người điều phối phải có hồ sơ thật bên ngoài; checkbox/JSON không tự chứng minh đồng ý.
- Lịch mỗi vòng là hoán vị0–9, sinh xác định theo participant/stage/repetition; Python và TypeScript dùng cùng thuật toán. Không lấy nhãn từ model, không hiển thị prediction hoặc mẫu để tô lại.
- Dùng lại DrawingCanvas hiện tại: tọa độ canonical400×400, nhiều nét, undo/xóa bản nháp. Kiểm tra nét di chuyển, tối đa16 nét/8.192 điểm; kiểm tra pointer thực so với mouse/touch/pen đã đăng ký.
- Mỗi mẫu có UUID riêng, nhãn/vòng, mã người/phiên, thiết bị, pointer type thực và thời gian thu thực. Không tạo timestamp cho từng điểm khi không có.
- Tải JSON toàn phiên, tên có đợt/mã người/session/đếm mẫu. Upload file để tiếp tục đúng vị trí; chặn metadata/nhãn/PII bổ sung/ID trùng hoặc lệch lịch. Bản nháp không thành mẫu khi chỉ chạm một điểm.
- Nét ở bộ nhớ browser, không upload server, không localStorage, không tải model inference. Nhắc tải sau mỗi10 mẫu/trước nghỉ; cảnh báo rời trang khi có mẫu chưa tải. Không có autosave bền vững, nên vẫn có rủi ro mất dữ liệu chưa tải.

Không thêm nút/form/bộ chọn model vào trang DrawAI chính. CSS của trang thu được scope `.drawai-collect`; ảnh root/alias desktop/mobile giữ nguyên từng pixel.

## 4. Schema, validator và builder

Code CLI: `training/drawai/private_data.py`; schema strict Draft2020-12 tại `docs/drawai/collection/session.schema.json`.

Audit kiểm tra:

- Trường/schema/protocol, UUID, nguồn khai báo, consent, mã người trong plan; không chấp nhận trường PII phát sinh.
- Nhãn integer0–9, label trong drawing và prompt/vòng khớp lịch, độ dài không vượt quota.
- Metadata phiên/mẫu và device/pointer khớp; tọa độ0..400 hữu hạn, shape/geometry hợp lệ. Time có timezone, consent≤start≤sample, mẫu theo thời gian, không ở tương lai.
- Không trộn pilot/main; không nhiều phiên cho một mã, không nhiều snapshot cùng phiên, không lặp sample ID hoặc exact strokes kể cả giữa các split.
- Exact raster hash và dHash/Hamming≤2 gắn cờ cần rà soát; không tự loại mẫu bằng prediction. Cặp gần trùng lưu tối đa5.000 trong report cùng tổng số/truncated để giới hạn bộ nhớ; P4 phải rà toàn bộ nếu vượt giới hạn.
- Quota người×lớp; thống kê người/lớp/thiết bị/split, số còn thiếu và các lỗi/cảnh báo. Inbox trống báo0 và quota_complete=false.

Builder từ chối input có errors, dataset trống hoặc thiếu quota khi bật `--require-complete`. Tạo staging gồm:

```text
raw/sample_<uuid>.json
images/sample_<uuid>.png            RGB224 cho DINOv3
swin-l128/sample_<uuid>.npy         float32[3,128] cho Swin1D
manifest.csv                       cùng sample_id, label, participant/split; file paths và SHA-256
plan.json                          policy/hash đã khóa
audit.json                        kết quả kiểm tra, không phải khóa dataset
dataset.json                      manifest hash; preliminary=true, release_ready=false
```

Hai representation lấy từ cùng JSON, không augmentation, không fusion, pilot luôn pilot-dev. Không ghi đè output cũ/không xuất vào public/dist. Mã hồ sơ consent không vào raw per-sample/manifest; raw staging còn thời gian thu và mã phiên, nên **không phải gói đã rà soát privacy để public**. P8 cần bỏ trường không cần thiết, kiểm tra consent/license và dataset card trước Kaggle. Mọi build P3 luôn `dataset_locked=false`, kể cả đủ quota; P4 mới khóa dataset sau rà soát.

## 5. Tài liệu bàn giao

| File tại `docs/drawai/collection/` | Nội dung |
|---|---|
| `protocol.md` | Hướng dẫn viết, nghỉ/tải/tiếp tục, thiết bị thật, quota/split và xử lý sự cố |
| `consent-template.md` | Phiếu đồng ý ba mục, liên hệ điều phối và giữ hồ sơ riêng |
| `operator-guide.md` | Quy trình cấp mã, inbox/quarantine, lệnh CLI và điều kiện đóng pilot |
| `session.schema.json` | Schema strict của JSON phiên và mẫu |
| `split-policy.json` |35 mã dự kiến, split/quotas và hash |
| `manifest-template.csv` | Header mapping chung JSON/PNG/NPY/hash |
| `registry-template.csv` | Mẫu sổ private cho điều phối chống trùng người/consent |
| `rejection-log-template.csv` | Mẫu ghi cách ly/thu lại và hash nguồn; không loại bằng model |
| `pilot-report-template.md` | Báo cáo thực thu250 mẫu, sự cố, duplicate review, quyết định protocol |
| `dataset-card.md` | Dự thảo nguồn/metadata/split/giới hạn; chưa có dữ liệu/license/link Kaggle cuối |

Protocol cho người viết được copy sang `web/public/collection-protocol.md`, có link trên trang thu. Không public registry đã điền/consent/JSON phiên. README/status/roadmap chỉ rõ phần đã làm và phần pilot còn chờ.

## 6. Kiểm chứng thực chạy

| Hạng mục | Kết quả |
|---|---|
| TypeScript/Vite build | PASS |
| Python DrawAI toàn bộ | **56/56 PASS**, gồm10 tests private-data mới |
| Browser collection | **5/5 PASS** |
| Browser sản phẩm/root/alias | **12/12 PASS** |
| Route thu không inference/upload và không localStorage | PASS |
| Xuất JSON canvas rồi Python schema/prompt validation | PASS |
| Resume đúng lịch, reject tamper/PII/thiết bị sai/tap | PASS |
| CLI preliminary build/hashes, duplicate/cross-split/quota guards | PASS |
| Hash code/model/data được bảo vệ | **58 files giữ nguyên** |
| Ảnh desktop/mobile root và alias | Pixel-identical với P0 |
| Pilot người thật / thiết bị touch/pen thật | **NOT RUN** |

Lần chạy regression đầu dùng cổng mới trong khi test cũ hard-code4186 nên2 ca thất bại; đã sửa config P3 dùng4186, rerun toàn suite12/12 đạt. Không thay assertions để bỏ qua lỗi; log đầu giữ tại `initial-port-test-failure.log`.

Bằng chứng: `artifacts/drawai/phase3/verification.json`, `python-tests.log`, `browser-tests.json`, `product-regression.json`, `build.log`, `current-pilot-audit.json`, ảnh `ui/` và `collector-desktop/mobile.png`. Fixture browser nằm riêng `technical-only/`, có README gắn nhãn synthetic; không được tính vào quota. Test Python dùng temporary directories và fixture lập trình, không đưa dữ liệu đó vào inbox.

## 7. Cách bắt đầu thu pilot ngay

1. Trong web chạy `npm run dev -- --port 4183 --strictPort`; mở `http://127.0.0.1:4183/collect`.
2. Điều phối mời5 người thật, đọc/giải thích protocol và phiếu đồng ý, cấp P001–P005 và mã hồ sơ riêng, kiểm tra không trùng người.
3. Mỗi người viết50 mẫu; tải JSON mỗi vòng, tiếp tục bằng file mới nhất. Bàn giao vào `data/private/inbox/pilot/`, chỉ giữ snapshot mới nhất/phiên.
4. Chạy audit/build theo [operator-guide.md](collection/operator-guide.md). Rà soát lỗi/duplicate/consent và điền báo cáo pilot bằng số liệu thực.
5. Đủ250 mẫu đúng quota cùng xác nhận người/consent và báo cáo pilot mới có thể đóng P3, xem xét giữ protocol-v1 hay tạo v2 trước P4. Chưa full-train hoặc mở test ở bước này.

Trạng thái kết thúc lượt triển khai: **TECHNICAL_READY_WAITING_FOR_HUMAN_PILOT**,0/250 mẫu người thật. Không đánh dấu hoàn thành toàn bộ P3 dựa trên kiểm thử tự động.
