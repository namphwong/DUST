# DrawAI — Kế hoạch chuyển sang hai mô hình

Ngày lập: 17/10/2026; cập nhật 18/10/2026. **P0–P2 hoàn thành local**, P3–P8 chưa bắt đầu. P2 đã load pretrained DINOv3 từ publisher timm, export toàn bộ graph và chốt backend CPU theo số đo; giữ nguyên DrawAI/Swin. Xem [P0](phase0-report.md), [P1](phase1-report.md), [P2](phase2-report.md).

## Phạm vi chốt

- Chỉ giữ sản phẩm DrawAI: nhận dạng một chữ số 0–9 từ nét vẽ trên canvas.
- Hai mô hình chính: Swin Transformer 1D làm đối chứng; DINOv3 làm backbone ảnh, kèm head phân loại 10 lớp.
- Giữ bố cục, màu, typography, canvas, các nút và luồng nhận dạng hiện có. Không tự thêm dashboard, tab hoặc bộ chọn model trên giao diện chính.
- Cùng một nét vẽ tạo hai biểu diễn: chuỗi x/y/stroke_start cho Swin; ảnh raster RGB cho DINOv3. Không thêm chức năng OCR ảnh tải lên trong phạm vi này.
- Không fusion hai mô hình. So sánh bằng script/báo cáo và trang kiểm chứng riêng. Chọn model mặc định bằng validation và khả năng chạy trên thiết bị mục tiêu trước khi đánh giá test.
- Dữ liệu “private” được hiểu là tự thu thập, chưa công khai lúc thu; sau đó công bố bản đã ẩn danh lên Kaggle theo yêu cầu đầu bài.

## Mốc trước P1 (lịch sử)

- DrawAI đã có Swin 1D, preprocessing, training, export ONNX, inference worker, canvas và bộ kiểm chứng.
- Model Swin đã train ở web/public/models/drawai-swin-v1/model.onnx. Đây là bản PenDigits cũ, không phải model train bằng dữ liệu tự thu.
- Báo cáo cũ ghi test PenDigits accuracy 98,77%, macro-F1 98,79%; không dùng các con số này làm kết quả dữ liệu riêng.
- Chưa có DINOv3 trong pipeline. Trang gốc vẫn mở MotionSense. Nhiều file DrawAI chưa được Git theo dõi tại thời điểm rà soát.
- Protocol cũ dành 250 mẫu từ 5 người cho đánh giá held-out, không phải dataset để train. Protocol mới bên dưới là nghiên cứu mới, cần version riêng.

## Phase 0 — Khóa phạm vi và lưu mốc phục hồi

### Công việc

1. Kiểm kê file tracked/untracked/ignored, import, cấu hình test, data, checkpoint và notebook; xác định tài nguyên dùng chung.
2. Lưu mốc code hiện tại và sao lưu file cần giữ nhưng chưa có trong Git, đặc biệt checkpoint/dữ liệu bị ignore. Không dùng git reset --hard hoặc git clean để dọn.
3. Chụp giao diện DrawAI desktop/mobile, lưu build và inference mẫu làm mốc so sánh.
4. Chốt bảng hai model, hai input contract, 10 lớp và tiêu chí chọn model mặc định.

### Bàn giao / nghiệm thu

- Danh mục giữ/xóa/di chuyển với đường dẫn cụ thể và mốc phục hồi có thể dùng được.
- Giao diện hiện tại có ảnh tham chiếu; checkpoint Swin và provenance được lưu đầy đủ.
- Chưa xóa cho đến khi hoàn tất kiểm kê phụ thuộc và bảo toàn file DrawAI.

## Phase 1 — Chỉ giữ DrawAI và chuẩn hóa project

### Công việc

1. Đổi entrypoint để / mở DrawAI; giữ /drawai làm đường dẫn tương thích. Chỉ giữ trang kỹ thuật DrawAI còn cần.
2. Xóa MotionSense UI, CSS, CSV pipeline, signal chart, worker/client riêng; gỡ import và route tương ứng.
3. Xóa training/evaluation/export UCI HAR, notebook HAR, test HAR, docs và mẫu CSV HAR, artifact và model HAR sau kiểm tra phụ thuộc.
4. Xóa code video/fusion DuST gốc, toy data generator và dữ liệu/checkpoint video không phục vụ DrawAI sau kiểm kê. Giữ license, ghi công và SwinTransformer_cls.py vì DrawAI đang import trực tiếp.
5. Đưa CNN training/baseline khỏi pipeline chính hai model. Nếu giữ kết quả cũ để truy nguyên Swin, ghi rõ là lịch sử; không cần train lại chỉ để đổi tên file. Kiểm tra verifier cũ trước khi gỡ code CNN mà chúng còn phụ thuộc.
6. Dọn scripts/dependencies/config/.gitignore theo import thực tế; không xóa cả thư mục data, artifacts hoặc models theo tên chung.
7. Chuẩn hóa tên package thành DrawAI và cập nhật README sơ bộ.

### Bàn giao / nghiệm thu

- / và /drawai đều chạy DrawAI, không còn MotionSense trong runtime/build/test đang sử dụng.
- Build, test DrawAI hiện có và inference Swin đạt; giao diện khớp ảnh mốc.
- Có danh sách file đã xóa/di chuyển và lý do; giữ nguyên ghi công nguồn Swin.

## Phase 2 — DINOv3 chạy thử và chốt phương án xuất/chạy

### Công việc

1. Tích hợp DINOv3 ViT-S/16 pretrained từ nguồn Meta hoặc publisher timm đã xác minh; lưu tên checkpoint, revision/hash và dependency đã kiểm chứng. Xác minh quyền truy cập weights và điều kiện phân phối.
2. Thiết kế wrapper backbone + feature pooling theo API model + head 10 lớp. Ban đầu đóng băng backbone và train head; không gọi trọng số khởi tạo ngẫu nhiên là pretrained.
3. Chốt renderer drawai-raster-v1: ảnh RGB 224×224 làm cấu hình thử ban đầu, giữ tỷ lệ, padding, nền/màu nét/độ dày nét cố định; không vẽ đường nối khi nhấc bút. Resize/normalize theo checkpoint chính thức và khóa trong metadata.
4. Tạo ảnh từ cùng JSON gốc; xử lý rõ y-down, multi-stroke và pixel-center. Không lấy screenshot UI làm preprocessing huấn luyện.
5. Chạy forward/backward và tiny overfit để kiểm tra pipeline; artifact thử ghi trained=false hoặc smoke-only, không đưa vào release thật.
6. Thử export toàn bộ backbone + head sang ONNX và kiểm chứng PyTorch/ONNX Python/browser. Đo bytes, tải model, bộ nhớ, cold start, latency trên máy mục tiêu trước khi train đầy đủ.
7. Ưu tiên inference browser nếu đạt yêu cầu. Nếu không đạt, ghi bằng chứng và chốt backend inference trước khi triển khai tích hợp; giữ UI nhưng phải cập nhật mô tả nơi xử lý dữ liệu. Không âm thầm thay DINOv3 bằng model khác.

### Bàn giao / nghiệm thu

- Hai model forward ra [B,10], contract và renderer có version, báo cáo feasibility và runtime được chọn.
- Tiêu chí parity và ngân sách hiệu năng/dung lượng được ghi trước khi đánh giá release. Không cam kết DINOv3 chạy nhẹ như Swin.

## Phase 3 — Chuẩn bị thu dữ liệu riêng

### Quy mô đề xuất

- Pilot: 5 người × 10 chữ số × 5 lần = 250 mẫu; dùng kiểm tra công cụ và protocol, không gọi là test cuối.
- Đợt chính: 30 người mới × 10 chữ số × 20 lần = 6.000 mẫu. Đây là mục tiêu đồ án đề xuất, không bảo đảm độ chính xác.
- Tách theo người trước khi train: 20 người train / 5 validation / 5 test, tương ứng 4.000 / 1.000 / 1.000 mẫu nếu mỗi người đủ quota.
- Nếu nguồn lực hạn chế: 20 người × 10 × 10 = 2.000 mẫu, split 12/4/4 người; ghi giới hạn về độ đa dạng và số người test.
- Nếu dùng pilot cho development, người pilot không xuất hiện trong test chính; mọi điều chỉnh quota/split phải chốt trước mở test.

### Công cụ và schema

- Giữ nguyên design trang chính. Thu bằng xuất JSON hiện có kết hợp công cụ riêng tại /collect hoặc script; không chèn form thu dữ liệu vào màn hình nhận dạng chính.
- Sample ID duy nhất, participant_id ẩn danh, nhãn do người viết/điều phối cung cấp, session_id, repetition, input_device, phiên bản protocol, file JSON gốc và hash.
- Manifest giữ mapping JSON–PNG–nhãn–split; cùng sample_id luôn chung split cho cả hai mô hình. Timestamp chỉ lưu nếu có thật và cần dùng.
- Có hướng dẫn đồng ý riêng cho huấn luyện, đánh giá và công bố dữ liệu ẩn danh lên Kaggle. Tài liệu định danh/consent cá nhân để ngoài gói công khai; metadata public chỉ giữ thông tin không nhận diện cần thiết.
- Chuẩn bị script kiểm tra schema, nhãn, duplicate, participant overlap, thống kê lớp/người/thiết bị và tạo split manifest có hash.

### Bàn giao / nghiệm thu

- Protocol, hướng dẫn người thu/người viết, schema, mẫu manifest, công cụ thu, validator, split policy và dự thảo dataset card.
- Pilot đủ 250 mẫu nguồn người thật và báo cáo lỗi thu; chưa có người tham gia thì ghi chưa thực hiện, không thay bằng dữ liệu sinh.

## Phase 4 — Thu chính thức, kiểm tra và khóa dataset

### Công việc

1. Người dùng mời người tham gia và lấy đồng ý theo protocol; công cụ hỗ trợ lưu/kiểm tra, không tự xác nhận nguồn người thật.
2. Mỗi mẫu là một chữ số hoàn chỉnh 0–9, viết tự nhiên, không tô lại mẫu. Xen kẽ thứ tự chữ số giữa các vòng theo protocol; nghỉ để giảm mỏi.
3. Đa dạng chuột/cảm ứng/bút theo nguồn lực, cân bằng nhóm thiết bị giữa các split khi có thể. Ghi thiết bị thực; giả lập mobile không thay cho dữ liệu cảm ứng thật.
4. Không hiển thị prediction khi thu. Không loại mẫu vì model dự đoán sai. Chỉ loại lỗi protocol như file hỏng, trống, sai nhãn đã xác minh; ghi log lý do và số lượng.
5. Rà duplicate, gần trùng và overlap người; tạo hai representation từ một nguồn JSON. Chỉ augmentation trên train.
6. Khóa drawai-private-v1, split manifest, hash và dataset card trước training; giữ test khỏi quy trình chọn cấu hình.

### Bàn giao / nghiệm thu

- Dataset đạt quota đã chốt, 10 lớp, mỗi split đủ người/lớp, không overlap người hoặc sample; thiếu quota phải báo trước khi khóa.
- Gói JSON + PNG + CSV/manifest truy nguyên được; test được khóa. Tên private-v1 thể hiện nguồn tự thu, không hàm ý vẫn bí mật sau phát hành Kaggle.

## Phase 5 — Huấn luyện và so sánh hai mô hình

### Công việc

1. Dùng đúng split người và sample_id chung. Swin dùng chuỗi L128; DINOv3 dùng ảnh từ cùng nét.
2. PenDigits dùng làm nguồn bổ trợ/pretrain nếu cần và ghi rõ. Giữ kết quả PenDigits tách biệt kết quả private-v1; kiểm tra hai model được dùng nguồn bổ trợ nào.
3. Swin: train hoặc fine-tune theo protocol; phân biệt initialization từ checkpoint cũ với train from scratch.
4. DINOv3: frozen backbone + linear/MLP head trước; chỉ fine-tune một phần nếu validation và tài nguyên cho phép. Thay đổi phải ghi vào config/run.
5. Chốt budget tuning trước; chạy seeds 42/43/44 nếu đủ tài nguyên. Cùng dữ liệu/split/cách chọn checkpoint; báo chênh lệch pretraining, số tham số và compute, không gọi là so sánh thuần kiến trúc.
6. Chọn cấu hình/checkpoint/model mặc định bằng validation và feasibility; không chọn bằng test. Chốt release rồi đánh giá test trong đợt cuối có kiểm soát; không tune lại trên test.
7. Báo accuracy, macro-F1, confusion matrix, per-class/per-writer/per-device, mean/std qua seed, thời gian train, latency, bộ nhớ và kích thước model. Không bắt buộc DINOv3 thắng Swin.

### Bàn giao / nghiệm thu

- Checkpoint thật của hai model, config/dependency/data hash, training logs, selection record và báo cáo so sánh trên cùng test.
- Metric tính từ predictions lưu theo sample_id; hạn chế số người test và domain gap được ghi rõ.

## Phase 6 — Xuất model với tên rõ ràng

### Quy ước

```text
drawai_<architecture>_<dataset>_<input>_<precision>_v<release>.<extension>

drawai_swin1d_pendigits_l128_fp32_v1.onnx
drawai_swin1d_private-v1_l128_fp32_v1.onnx
drawai_dinov3-vits16_private-v1_rgb224_fp32_v1.onnx
```

- File .onnx phải chứa full pipeline neural network backbone + classifier, không phải chỉ head DINOv3. Preprocessing ngoài graph phải đi kèm contract.
- Checkpoint dùng stem tương ứng, thêm seed/best khi cần: drawai_swin1d_private-v1_l128_seed44_best.pth. Seed ví dụ, không chốt seed44 trước validation.
- Nếu release cuối đổi input size/precision thì đổi tên đúng thực tế. Không gọi artifact chưa train là release model.
- Mỗi model có labels.json, preprocessing.json, model-card.md và release.json ghi filename, hash, bytes, trained, checkpoint nguồn, dataset/split version, input/output và metric.

### Công việc / nghiệm thu

1. Đổi tên model Swin cũ và cập nhật exporter, loader, manifest, verifier, docs, tests theo cùng thay đổi; không chỉ rename file.
2. Xuất model đã chọn, kiểm tra graph/shape/finite output; đối chiếu checkpoint với artifact trên toàn test theo tiêu chí khóa ở Phase 2.
3. Đo lại runtime thực dùng. Nếu DINOv3 cần backend, bàn giao checkpoint/full model deploy được và hướng dẫn server; báo rõ định dạng/runtime, không tuyên bố ONNX browser khi chưa đạt.
4. Có hai model được đặt tên rõ và load được, không giảm chất lượng vượt ngưỡng đã chốt sau export.

## Phase 7 — Tích hợp, giữ nguyên design và nghiệm thu

### Công việc

- Model registry nhận model_id và tải đúng artifact/preprocessing/contract. Hai đường suy luận dùng chung nguồn nét vẽ và cùng cấu trúc kết quả UI.
- Trang chính chạy model mặc định được chọn trước test; trang kiểm chứng riêng chạy cả hai để so sánh. Không thêm bộ chọn model vào design hiện tại nếu chưa có yêu cầu mới.
- Giữ canvas, undo/clear, mẫu thử, top-3, lỗi/retry và chống kết quả cũ sau khi nét thay đổi. Không hiển thị kết quả giả khi model lỗi.
- Mẫu PenDigits và mẫu tự thu ghi đúng nguồn; không dùng test làm mẫu demo được xem trong development.
- Kiểm tra desktop/mobile, pointer/DPR, worker hoặc API, tải model, model corrupt/missing, build và parity preprocessing. Thử thiết bị cảm ứng thật riêng.

### Bàn giao / nghiệm thu

- DrawAI chạy end-to-end với model mặc định; cả hai model chạy được qua trang kiểm chứng/script.
- Giao diện khớp mốc Phase 0; báo cáo runtime/thiết bị và giới hạn có bằng chứng.

## Phase 8 — README, Kaggle và gói nộp

### README bắt buộc

1. Mục tiêu DrawAI, ảnh giao diện, hai sơ đồ pipeline và vai trò Swin đối chứng/DINOv3 backbone.
2. Bảng “Đã cập nhật ở đâu”: file, chức năng cũ, thay đổi mới; chỉ ghi file/chức năng đã thực sự hoàn tất.
3. Hướng dẫn cài, thu/chuẩn bị data, train từng model, evaluate, export, chạy web/backend nếu có.
4. Bảng model filename, input/output, dataset, metric, link tải và checksum.
5. Split theo người, nguồn/pretraining, augmentation, môi trường, cách tái lập, giới hạn và ghi công/license.
6. Link Kaggle public thật, không dùng placeholder làm bằng chứng hoàn thành; hướng dẫn tải dữ liệu/model lớn nếu không lưu trực tiếp Git.

### Kaggle và bàn giao

- Public package gồm strokes/, images/, metadata.csv, splits/, dataset card và license phù hợp với quyền công bố đã có. Không đưa định danh/consent cá nhân hoặc pretrained weights vào dataset nét bút.
- Người dùng cung cấp người tham gia, đồng ý công bố, tài khoản/quyền truy cập cần thiết; hỗ trợ kỹ thuật không thay cho các đầu vào thực này.
- Tạo/cập nhật README xuyên suốt; phase cuối rà tính chính xác và thử từ checkout sạch, không chỉ trên máy có data/cache sẵn.
- Bàn giao URL GitHub, URL Kaggle public, hai model export, báo cáo so sánh, hướng dẫn chạy và danh sách update.

## Thứ tự thực hiện và báo cáo tiến độ

```text
P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8
```

P2/P3 có thể chuẩn bị đan xen sau P1, nhưng không train đầy đủ trước khi DINOv3 khả thi và split đã khóa. Kaggle draft/README được chuẩn bị sớm; public release sau kiểm tra dữ liệu.

Mỗi phase kết thúc bằng: file thay đổi, lệnh đã chạy/kết quả, artifact bàn giao, việc chưa chạy và điều kiện chuyển phase. Trạng thái dùng TODO / IN PROGRESS / DONE / BLOCKED; không đánh DONE khi chỉ mới có script hoặc kế hoạch.

## Mục tiêu chất lượng

- Không đặt kết quả đoán trước cho private-v1. Sau pilot, chốt ngưỡng chất lượng dùng validation và yêu cầu demo; ghi trước khi mở test.
- Thành công học thuật: hai model đúng kiến trúc, dữ liệu tự thu có nguồn và split hợp lệ, đánh giá trung thực, artifact chạy được, README truy nguyên được.
- Nếu không đạt mục tiêu accuracy hoặc browser performance, vẫn báo kết quả thật và giới hạn, không đổi/loại mẫu test để cải thiện điểm.
