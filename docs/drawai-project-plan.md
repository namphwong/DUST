# DrawAI — Kế hoạch triển khai

> Kế hoạch gốc của đợt PenDigits, giữ để truy nguyên. Kế hoạch hiện hành: [hai mô hình Swin/DINOv3](drawai/two-model-roadmap.md). P0–P1 đã hoàn thành; xem [báo cáo P1](drawai/phase1-report.md).

Ngày cập nhật: 11/10/2026. D0–D5 đã triển khai local; D6 đạt kiểm chứng kỹ thuật Chrome/Edge, còn chờ thu/đánh giá nét người và thiết bị thật; D7 chưa triển khai. D3 đạt mục tiêu PenDigits, D4 đã đóng gói model thật, D5 có canvas/worker tại `/drawai`. Kaggle GPU và chất lượng nét người tham gia chưa được kiểm chứng. Xem [tiến độ](drawai/status.md), [D3](drawai/experiment-report.md), [D4](drawai/artifact-report.md), [D5](drawai/ui-report.md), [nghiệm thu D6](drawai/acceptance-report.md). Các mục tiêu trong kế hoạch không thay thế kết quả đã đo.

Checklist thực thi, phụ thuộc, đầu ra và điều kiện qua từng phase: [Các phase DrawAI chi tiết](drawai/phases.md).

## 1. Mục tiêu và phạm vi

**DrawAI — Online Handwritten Digit Recognition Using 1D Swin Transformer** nhận dạng một chữ số 0–9 từ chuỗi nét bút người dùng vẽ bằng chuột, cảm ứng hoặc bút trên web.

Trong MVP, “online handwriting” có nghĩa là đầu vào lưu tọa độ theo thứ tự viết và ranh giới nét. Người dùng hoàn tất một chữ số rồi bấm **Nhận dạng**. Dự đoán liên tục khi chữ số chưa viết xong là phần mở rộng, cần đánh giá riêng trên nét chưa hoàn chỉnh.

Luồng chính: mở trang → vẽ một chữ số → nhận dạng → xem chữ số dự đoán và top-3 điểm → xóa hoặc hoàn tác để thử lại. Có mẫu dữ liệu thật để người dùng thử ngay.

MVP bao gồm:

- Canvas hỗ trợ nhiều nét, chuột/cảm ứng/bút; nhận dạng, xóa, hoàn tác nét cuối.
- Tiền xử lý chuỗi giống nhau giữa Python và trình duyệt.
- Mô hình Swin 1D đã huấn luyện, suy luận thật bằng ONNX Runtime Web trong Web Worker.
- Kết quả top-3, thời gian suy luận, trạng thái tải và lỗi có thể thử lại.
- Mười mẫu tham chiếu, mỗi lớp một mẫu, có nhãn và nguồn rõ ràng.
- Xuất nét vẽ dạng JSON để kiểm tra hoặc xây tập đánh giá canvas có gán nhãn.
- Giao diện tiếng Việt, responsive, hướng dẫn thao tác và thông tin mô hình.

Ngoài MVP: nhận dạng nhiều chữ số trong một lần vẽ, chữ cái/từ/câu, OCR ảnh tải lên, đăng nhập, lưu lịch sử trên server, train trong trình duyệt, tự học từ nét người dùng, dự đoán liên tục trên nét chưa hoàn chỉnh.

## 2. Dataset đã chọn

**UCI Pen-Based Recognition of Handwritten Digits — bản gốc `pendigits-orig`**, định dạng UNIPEN.

| Thuộc tính | Theo tài liệu nguồn |
|---|---|
| Nhãn | 10 lớp, 0–9 |
| Quy mô | 10.992 mẫu, 44 người viết |
| Development pool | `pendigits-orig.tra.Z`: 7.494 mẫu từ 30 người |
| Test chính thức | `pendigits-orig.tes.Z`: 3.498 mẫu từ 14 người khác |
| Dữ liệu dùng | Tọa độ x/y có thứ tự; độ dài mẫu thay đổi |
| Giấy phép trên UCI | CC BY 4.0; ghi nguồn khi sử dụng/phân phối |

File nguồn đã được tải và giải nén vào `data/raw/pendigits/`; D1 đã kiểm đếm độc lập đủ mẫu và kiểm tra cấu trúc nét. Writer ID chưa xác minh, vì trường comment chưa được tài liệu nguồn giải thích; đã áp dụng fallback train/validation phân tầng theo lớp. Xem [báo cáo dữ liệu](drawai/data-report.md).

Bản `pendigits.tra/.tes` đã resample mỗi mẫu còn 8 điểm; chỉ dùng làm tham chiếu tùy chọn. Pipeline chính đọc bản gốc. Không xem nội suy 8 điểm lên chuỗi dài hơn là khôi phục nét gốc.

Quy tắc chia dữ liệu:

1. Giữ nguyên test chính thức, không dùng để chọn cấu hình, checkpoint hoặc ngưỡng.
2. Kiểm tra writer ID trong bản gốc. Nếu xác minh được, tách khoảng 24/6 người của development pool thành train/validation bằng seed cố định; lưu danh sách ID thực tế và kiểm tra đủ lớp.
3. Nếu không khôi phục được writer ID đáng tin cậy, dùng train/validation phân tầng theo lớp với seed cố định; ghi rõ validation có thể chung người viết. Không suy đoán writer ID từ thứ tự mẫu.
4. Lưu source hash, sample ID và split manifest. Augmentation chỉ áp dụng sau khi tách train.
5. Dùng tập canvas riêng để kiểm tra khác biệt giữa Wacom và chuột/cảm ứng. Nếu dùng một phần để điều chỉnh mô hình, giữ một phần khác chưa xem để đánh giá cuối.

## 3. Hợp đồng dữ liệu dự kiến

Dữ liệu ứng dụng lưu danh sách `strokes`; mỗi stroke gồm các điểm `x`, `y`, `t`. Thời gian được lưu để có thể phát lại/kiểm tra, chưa dùng làm đặc trưng bắt buộc vì phải xác minh thông tin thời gian thực có trong dataset. Không yêu cầu pressure.

Pipeline đề xuất:

1. Đọc nhãn và nét; kiểm tra số hữu hạn, chuỗi rỗng, điểm lặp, số nét và độ dài.
2. Đồng nhất chiều trục y giữa nguồn và canvas; kiểm tra trực quan mẫu của cả 10 lớp.
3. Chuẩn hóa vị trí và kích thước theo bounding box của toàn chữ số, giữ tỷ lệ x/y; xử lý trường hợp một chiều có biên độ bằng 0.
4. Resample theo độ dài cung trong từng nét, giữ thứ tự và ranh giới nét. Phân bổ ngân sách điểm theo độ dài nét, giữ điểm đầu/cuối khi có thể; không nội suy đường nối qua khoảng nhấc bút.
5. Sinh tensor float32 theo layout `[batch, channels, length]`.

Cấu hình khởi điểm: **3 kênh `[x, y, stroke_start]`, chiều dài 64, 10 logits**. `stroke_start=1` tại điểm đầu mỗi nét, 0 ở các điểm còn lại. Kiểm tra parser giữ được ranh giới nét trước khi chốt hợp đồng. Thử chiều dài 128 nếu validation cho thấy có lợi. Thí nghiệm thêm `dx, dy` là tùy chọn; sai phân không được nối qua hai nét khác nhau.

Phase 1 phải chốt quy tắc cho nét một điểm, nhiều nét hơn ngân sách điểm, nét quá ngắn, giới hạn số điểm đầu vào và thông báo lỗi. Không âm thầm cắt mất nét. Nếu dùng padding, phải xác định cách tránh để padding tác động sai đến attention/pooling; ưu tiên resample chiều dài cố định cho MVP.

## 4. Kiến trúc và phần tái sử dụng

```text
Canvas → strokes → kiểm tra/chuẩn hóa/resample → tensor [1,3,L]
       → Web Worker / ONNX Swin 1D → 10 logits → softmax → top-3

PenDigits gốc → parser → split → cùng tiền xử lý → train/validation
             → checkpoint chọn bằng validation → test → export ONNX
```

- Backbone: `models/SwinTransformer_cls.py::SwinTransformer_1D`.
- Cấu hình ứng viên: `patch_size=2`, `window_size=8`, `embed_dim=32`, `depths=[2,2,2]`, `num_heads=[2,4,8]`, `num_classes=10`. Với L=64, số token các stage là 32 → 16 → 8. Đây là cấu hình cần kiểm chứng, chưa phải cấu hình tốt nhất.
- Baseline: CNN 1D dùng cùng dữ liệu, split, đặc trưng và quy tắc chọn checkpoint.
- Huấn luyện: PyTorch, ưu tiên notebook Kaggle GPU; CPU dùng cho kiểm tra nhanh. Phải có preflight môi trường và notebook tái lập được.
- Frontend: React + TypeScript + Vite hiện có; ONNX Runtime Web/WASM và Web Worker.
- Tái sử dụng cách quản lý artifact, hash, benchmark và kiểm tra parity của dự án HAR. Checkpoint HAR, chuẩn hóa gia tốc, nhãn hoạt động và metric HAR không áp dụng cho DrawAI.
- Thêm DrawAI ở route `/drawai`; quyết định trang mặc định khi nghiệm thu bản mới. Các tài liệu HAR hiện có vẫn là lịch sử của sản phẩm trước.
- Hosting dự kiến: static hosting, phù hợp với Vercel; kiểm tra điều kiện/quota thực tế khi phát hành. Nét vẽ được xử lý tại thiết bị.

## 5. Các giai đoạn và điều kiện hoàn thành

| Giai đoạn | Công việc | Đầu ra và điều kiện hoàn thành |
|---|---|---|
| D0 — Phạm vi | Chốt MVP, ý nghĩa “online”, dataset, luồng dùng, kiến trúc | Kế hoạch này; các lựa chọn còn mở có giai đoạn giải quyết rõ ràng |
| D1 — Kiểm toán dữ liệu | Giải nén `.Z`, parser UNIPEN, kiểm đếm nhãn/nét/độ dài, kiểm tra writer ID, trực quan hóa, split | Báo cáo dữ liệu, hash nguồn, manifest split, hợp đồng tiền xử lý; mọi mẫu lỗi đều được ghi nhận |
| D2 — Khả thi kỹ thuật | Forward/backward cấu hình mới, export thử ONNX, chạy WASM, đo dung lượng/độ trễ | Tensor → 10 logits hợp lệ, gradient hữu hạn; PyTorch/ONNX/browser khớp fixture; artifact thử ghi `trained=false` |
| D3 — Huấn luyện | CNN baseline và Swin, augmentation nhẹ, early stopping, thử L=64/128 theo validation, ba seed 42/43/44 | Checkpoint/log/config đầy đủ; lựa chọn cấu hình bằng validation; báo accuracy, macro-F1, recall từng lớp, confusion matrix và biến động giữa seed |
| D4 — Đóng gói mô hình | Export checkpoint đã chọn; đóng gói model, preprocessing, labels, manifest; kiểm tra toàn test | Report parity, model card và artifact có version/hash đồng bộ; chỉ artifact đã train mới dùng nhận dạng |
| D5 — Giao diện DrawAI | Canvas, thao tác nét, mẫu tham chiếu, top-3, trạng thái tải/lỗi, xuất JSON, responsive | Luồng vẽ → nhận dạng thật hoạt động; xóa/hoàn tác không để lại kết quả cũ; có thể thử lại khi tải lỗi |
| D6 — Tích hợp và đánh giá thực tế | Python/TS parity, kiểm tra worker/race, desktop/mobile, thu tập canvas có nhãn và người viết đồng ý | Báo cáo test tự động, kiểm tra thủ công thiết bị thật, kết quả canvas tách biệt với PenDigits; ghi rõ mục nào chưa có dữ liệu |
| D7 — Phát hành | Production build, deploy, kiểm tra URL mới, hoàn thiện README/model card/báo cáo | Web tải đúng model/WASM, hướng dẫn chạy lại và rollback; công bố đúng metric đã đo |

Thứ tự phụ thuộc: **D0 → D1 → D2 → D3 → D4 → D6 → D7**. D5 có thể thực hiện khi hợp đồng D1 đã chốt và D2 đã chứng minh khả thi; trong thời gian chưa có model thật, UI thể hiện trạng thái chưa sẵn sàng, không sinh dự đoán giả.

## 6. Kế hoạch thực nghiệm và tiêu chí nghiệm thu

### Chất lượng mô hình

- Cố định split trước khi huấn luyện. Chọn siêu tham số, early stopping và checkpoint bằng validation.
- Chạy CNN/Swin với cùng split và ba seed; ghi ngân sách huấn luyện từng mô hình. Báo kết quả trung bình/độ lệch chuẩn và kết quả checkpoint phát hành.
- Mục tiêu tham khảo trên test chính thức: accuracy ≥98%, macro-F1 ≥97%. Đây là mục tiêu dự án, chưa có số đo DrawAI.
- Không yêu cầu Swin phải thắng CNN mới được báo cáo. Nếu Swin kém hơn, phân tích độ dài chuỗi, số tham số, overfitting và lỗi từng lớp; không sửa nhãn hoặc chọn mẫu test thuận lợi.
- Augmentation train: thay đổi hình học nhỏ có kiểm soát, tránh biến đổi làm đổi nhãn; quy mô chọn bằng validation. Không lật ngang/dọc chữ số tùy tiện.
- Nghiên cứu thêm nếu còn thời gian: ảnh hưởng shifted windows, L=64/128, kênh tọa độ so với tọa độ + sai phân. Không mở rộng toàn bộ tổ hợp trước khi baseline chạy tốt.
- Sau khi cấu hình được khóa mới đánh giá test cuối. Nếu cần vòng cải tiến sau khi xem test, ghi rõ test đã được xem và dùng tập độc lập cho xác nhận mới.

### Tính đúng của pipeline

- Python và TypeScript tạo tensor tương đương trên fixture có một nét, nhiều nét, nét dọc/ngang, điểm lặp và đầu vào không hợp lệ.
- Kiểm tra chiều trục y, thứ tự kênh, stroke marker và không tạo đường nối qua khoảng nhấc bút.
- Đối chiếu logits PyTorch/ONNX Python/browser với ngưỡng ban đầu `atol=1e-5, rtol=1e-4`; nếu phải đổi ngưỡng, lưu lý do và sai số đo được.
- Đánh giá ONNX trên toàn bộ test; mục tiêu giảm macro-F1 không quá 0,001 tuyệt đối so với PyTorch. Báo số mẫu đổi argmax dù metric không đổi.
- Kiểm tra `request_id` hoặc cơ chế tương đương: kết quả của nét cũ không được ghi đè sau xóa, hoàn tác, nạp mẫu hoặc vẽ mới.
- Blank canvas/nét không đủ thông tin phải có hướng dẫn, không trình bày một chữ số ngẫu nhiên như kết quả hợp lệ.

### Trải nghiệm và hiệu năng

- Suy luận trên worker để thao tác vẽ vẫn phản hồi.
- Mục tiêu model ONNX ≤10 MiB; p95 suy luận sau warm-up ≤100 ms trên laptop tham chiếu; đo riêng tải model, khởi tạo, tiền xử lý và inference.
- Đo ít nhất 5 warm-up + 100 inference; ghi cấu hình máy/browser/runtime. Điện thoại thật cần số đo riêng.
- Kiểm tra viewport 390×844 và 1366×768 không tràn ngang; nút thao tác và mẫu dùng được bằng bàn phím; kết quả có thông báo cho công nghệ hỗ trợ.
- Kiểm tra pointer cancel, rời canvas, nhiều lần nhấc bút, undo/clear khi inference đang chạy, mạng lỗi và retry.
- Top-3 là điểm softmax, không gọi là “độ chính xác của nét này”.
- Tập canvas mục tiêu ban đầu: ít nhất 5 người × 10 chữ số × 5 lần = 250 mẫu; ghi loại thiết bị nhập. Đây là kiểm tra thăm dò, không đủ để suy rộng cho mọi người dùng.
- Chọn mẫu demo bằng quy tắc cố định từ train, mỗi lớp một mẫu; giữ cả mẫu dự đoán sai. Ghi rõ demo đã thuộc tập train và không dùng làm bằng chứng chất lượng test.

## 7. Cấu trúc đầu ra dự kiến

```text
docs/drawai-project-plan.md
docs/drawai/data-contract.md
docs/drawai/data-report.md
docs/drawai/experiment-report.md
docs/drawai/model-card.md
docs/drawai/acceptance-report.md
training/drawai/prepare_data.py
training/drawai/preprocessing.py
training/drawai/train.py
training/drawai/evaluate.py
training/drawai/export_onnx.py
training/drawai/configs/
training/tests/test_drawai_*.py
notebooks/kaggle_train_drawai.ipynb
data/processed/pendigits-drawai-v1/
artifacts/drawai/
web/public/models/drawai-swin-v1/
web/public/samples/drawai/
web/src/drawai/
web/tests/drawai.spec.ts
```

Các đường dẫn trên là cấu trúc đầu ra dự kiến của toàn project. Phần đã triển khai được ghi riêng trong [tiến độ](drawai/status.md); các file D3–D7 chưa có.

## 8. Ước lượng và bước tiếp theo

Ước lượng cho một người, với môi trường hoạt động và GPU sẵn dùng: **8–13 ngày làm việc**, chưa gồm thời gian chờ quota GPU, người tham gia vẽ hoặc vòng cải tiến ngoài MVP.

| Nhóm việc | Ước lượng |
|---|---|
| D1–D2: dữ liệu + kiểm chứng kỹ thuật | 2–3 ngày |
| D3–D4: huấn luyện + đánh giá + artifact | 2–4 ngày |
| D5: ứng dụng web | 2–3 ngày |
| D6–D7: tích hợp + nghiệm thu + phát hành | 2–3 ngày |

Bước triển khai tiếp theo là **D3: training pipeline và notebook Kaggle**. D1 đã chốt hợp đồng, D2 đã xử lý DLL và kiểm chứng cả hai chiều dài trên Python/ONNX/browser. Chưa có metric test classifier DrawAI. Xem [báo cáo khả thi](drawai/feasibility-report.md).

## Nguồn

- [UCI Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits).
- Tài liệu gốc đã tải: `data/raw/pendigits/pendigits-orig.names` và `pendigits.names`.
- Backbone trong repo: `models/SwinTransformer_cls.py`.
- Các report HAR hiện có chỉ làm mẫu quy trình kiểm chứng, không phải bằng chứng chất lượng của DrawAI.
