# DrawAI — Protocol D3 trước khi mở test

> Tài liệu lịch sử D0–D6 (11/10/2026). CNN và route cũ nếu được nhắc đến không thuộc pipeline P1 hiện hành. Xem [báo cáo P1](phase1-report.md) và [training guide hiện tại](training-guide.md).

Ngày: 11/10/2026. Experiment ID: `d3-local-v1`. Thiết bị thực chạy: CPU local; Kaggle GPU có notebook chuẩn bị riêng.

## Thiết kế cố định

- Development pool và split D1 giữ nguyên: train 5.999, validation 1.495. Test 3.498 mẫu chỉ được module evaluate mở sau khi khóa đủ 6 checkpoint chính thức.
- Factory CNN/Swin hiện có; cùng 3 kênh, split, metric 10 lớp và tiêu chí checkpoint. CNN channels 32/64/128; Swin embed 32, depths 2/2/2, heads 2/4/8, patch 2, window 8.
- Pilot: 2 model × L64/L128 × augmentation bật/tắt = 8 run, seed 42, 12 epoch mỗi run, patience 12, warm-up 1. Pilot không thuộc 6 run chính thức.
- Sau pilot, chọn L/augmentation riêng cho mỗi model bằng validation. Dùng cùng ngân sách tìm cấu hình, không bắt hai model phải có cùng L được chọn.
- Official: CNN và Swin, mỗi họ seeds 42/43/44, tối đa 50 epoch, patience 10, warm-up 3. CNN/Swin cùng batch 128, AdamW lr 0,001, weight decay 0,01, clip gradient norm 1,0. Cosine decay đến 5% LR, sau warm-up.
- Augmentation khi bật chỉ trên train: rotation ±8°, uniform scale 0,9–1,1, translation ±0,03 tọa độ chuẩn hóa. Không reflection/lật/đổi nhãn; stroke marker không đổi. Validation và test không augmentation. Affine train có thể ra ngoài bbox [-0,5;0,5]; đây là augmentation, không đổi hợp đồng tiền xử lý inference.

## Quy tắc lựa chọn

Checkpoint trong mỗi run: validation macro-F1 cao nhất; hòa điểm lấy validation loss thấp hơn; nếu vẫn hòa giữ epoch sớm hơn.

Pilot/config và checkpoint phát hành Swin: validation macro-F1 cao nhất, validation loss thấp nhất, L ngắn hơn, augmentation tắt, seed nhỏ hơn, run ID tăng dần. L/augmentation chỉ có tác dụng khi so config; checkpoint phát hành nằm trong họ Swin, không dùng metric test để chọn seed.

Lưu `experiment-plan.json` trước run; `configuration-lock.json` trước official; `selection.json` chứa 6 checkpoint hash trước test. `test-access.json` ghi hash selection tại lần mở test. Không đổi config/selection sau test để cải thiện con số đã báo.

## Tái lập và resume

Training đọc train/validation, đối chiếu NPZ với manifest/hash/preprocessing/labels. Torch deterministic algorithms; pilot CPU 4 threads; official CPU chạy tối đa 3 run độc lập đồng thời, mỗi run 2 threads. Cùng thiết lập cho CNN/Swin official; lưu thực tế trong summary. Seed theo run. DataLoader shuffle dùng generator seed+epoch; dropout/augmentation RNG được lưu trong last checkpoint. Resume cùng config/data/device/threads/Torch version; lưu model, optimizer, epoch, best, patience và RNG bằng checkpoint atomic. Schedule tính từ config và epoch, không reset khi resume.

Unit test so resume bị ngắt sau epoch 1 với run liên tục 3 epoch: toàn bộ model state và validation metrics phải khớp chính xác; test còn chặn đọc `test.npz`, kiểm tra dữ liệu tamper và marker augmentation.

## Báo cáo cuối

Accuracy, macro-F1, precision/recall/F1 từng lớp, confusion matrix từng run; mean và sample std (ddof=1) của ba seed mỗi họ. Báo lỗi test của checkpoint Swin được chọn bằng validation, learning curves, số tham số, thời gian và môi trường. Mục tiêu tham khảo: accuracy ≥98%, macro-F1 ≥97%; chưa đạt thì công bố giới hạn và phân tích lỗi, không dùng test chọn vòng chỉnh cấu hình trong experiment này.

Kết quả CPU không được ghi như GPU/Kaggle đã chạy. Validation có thể chung người viết với train vì writer ID chưa xác minh; quality trên canvas là D6, không suy từ demo train hoặc PenDigits test.
