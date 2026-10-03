# Model card — har-swin-kaggle-v1

Model: Swin Transformer 1D small từ `models/SwinTransformer_cls.py`, 2.191.594 tham số. Phát hành float32 ONNX opset 17; nguồn Kaggle `swin-seed44`, checkpoint được chọn theo validation macro-F1, không chọn theo test. Trọng số đã train, không fine-tune khi export. Version/hash: [manifest](../web/public/models/har-swin-kaggle-v1/release.json).

## Mục đích và dữ liệu

Demo nghiên cứu phân loại sáu hoạt động trên [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). Dữ liệu public, ghi bằng smartphone tại eo, 30 người; project chỉ dùng total acceleration x/y/z (g, có gravity), cửa sổ 128 điểm ở 50 Hz. Không dùng video, không dùng 561 feature vector, không dùng gyroscope. Xem [data report](data-report.md) về nguồn, license và split. Train 5392/validation 1960/test 2947, không trùng subject; normalization fit train-only. Lớp theo thứ tự: đi bộ, lên cầu thang, xuống cầu thang, ngồi, đứng, nằm.

## Đánh giá và giới hạn

ONNX official test accuracy 87,17%, macro-F1 87,05%, giống Kaggle checkpoint; chưa đạt mục tiêu 90%. Ba seed Swin có macro-F1 mean 86,81% ± 0,80 điểm phần trăm (sample std, không phải confidence interval). Test đã xem từ lượt CPU trước, không gọi là benchmark mù mới. [Training report](kaggle-training-report.md) và [export report](phase4-artifact-report.md) có protocol/metric đầy đủ.

Nhầm ngồi–đứng đáng kể: recall ngồi 70,88%, đứng 84,02%. Điểm softmax có thể cao ngay cả khi dự đoán sai; chưa calibration, chưa có OOD detector hoặc lớp unknown. Không bảo đảm với điện thoại cầm tay, vị trí khác, người/thiết bị/quy trình mới hoặc động tác không thuộc sáu lớp. Không dùng để chẩn đoán, giám sát an toàn, ra quyết định y tế hoặc quyết định quan trọng về một người.

## Tích hợp

Input float32 NCT [1,3,128] **normalized**, tên signals; output logits [1,6], tên logits. Caller phải cung cấp total acceleration trong g và 50 Hz, đúng channel-major, rồi chuẩn hóa đúng một lần bằng preprocessing.json. Không dùng m/s² trực tiếp, không normalize lần hai. Batch/sequence động không hỗ trợ. Phase 6 đã có CSV người dùng trong worker: chọn đơn vị/gravity, đổi về g, tách gap >100 ms, nội suy nguồn 40–100 Hz về 50 Hz, xác nhận rồi chia cửa sổ/suy luận/xuất CSV. Xem [pipeline report](phase6-pipeline-report.md). Đây không phải bảo đảm chất lượng cho dữ liệu ngoài UCI.

Inference chạy client-side bằng WASM, không cần backend/GPU/server trả phí. Trang Phase 4 tải static artifact khi người dùng bấm chạy, không upload tín hiệu để dự đoán. Chưa deploy public; không tuyên bố đã nghiệm thu Vercel/cache/mobile runtime.
