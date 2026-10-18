# DrawAI — Model card drawai-swin-v1

Ngày 11/10/2026. Swin Transformer 1D cho một chữ số 0–9 từ chuỗi tọa độ nét bút; phát hành artifact local, chưa deploy canvas.

- Backbone `models/SwinTransformer_cls.py::SwinTransformer_1D`, 545.038 tham số, embed32, depths [2,2,2], heads [2,4,8], patch2/window8; 3 input channels x/y/stroke_start, 10 logits.
- PenDigits gốc: train 5.999, validation 1.495, official test 3.498. Validation phân tầng seed42; writer ID chưa xác minh, train/val không bảo đảm khác người. Test chính thức giữ nguyên theo D1.
- `drawai-geometry-v1`: tâm bounding box/max-span giữ tỷ lệ, resample theo từng nét thành L128; tensor float32 NCT `[1,3,128]`, hệ y xuống. Không dùng timestamp, padding hoặc learned normalization. Nhãn có thứ tự 0–9.
- Training local CPU Torch 2.5.1+cpu: AdamW lr .001, batch128, weight decay .01; augmentation train nhẹ; seeds42/43/44 và ngân sách pilot CNN/Swin ngang nhau. Run `official-swin-l128-aug1-seed44`, best epoch 49, chọn bằng validation trước test.
- Checkpoint SHA-256 `a5ccb65dd4699b6451bf03ff23b6071382c29443213263f1fbc9ef6089fcf324`; ONNX SHA-256 `28100de69dec82303a954af5c18936788e7c3dee623d70ec7a89ad9f7ef02366`. Release manifest ghi hash/bytes từng file.
- Official test accuracy **98.77%**, macro-F1 **98.79%**; D4 full-test PyTorch/ORT có 0 argmax khác, macro-F1 không giảm. CNN baseline trung bình tốt hơn Swin; xem [D3](experiment-report.md) để so đủ ba seed.
- ONNX float32 opset17 fixed batch1/L128, 2.27 MiB. Chromium WASM 1 thread: 20 fixture đạt, p95 2.20 ms trên laptop local. Chưa quantize hoặc benchmark điện thoại/Edge.
- Demo là 10 mẫu train có nhãn nguồn/provenance, chọn độc lập dự đoán. Softmax sau này là điểm model, không phải độ chính xác bảo đảm.
- Giới hạn: chỉ một chữ số hoàn chỉnh, không OCR ảnh/nhiều chữ số; PenDigits không đại diện đầy đủ chuột/cảm ứng/thiết bị/người mới. Nét canvas thực chưa được thu/đánh giá; Kaggle GPU chưa chạy. Không tuyên bố metric này là chất lượng ứng dụng ngoài dataset.

[Dataset/preprocessing](data-report.md) · [Protocol training](experiment-protocol.md) · [Export, parity và lệnh tái lập](artifact-report.md). Nguồn dataset, archive hashes và điều kiện nguồn nằm trong báo cáo D1; không bổ sung tuyên bố giấy phép khi chưa có xác minh.
