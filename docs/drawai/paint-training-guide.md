# P5 — huấn luyện và so sánh Paint35

Code mới: `training/drawai/paint_train.py`. Thí nghiệm dùng snapshot P4 `paint35-main-v1`, không dùng pipeline PenDigits/JSON nét bút cũ. Hai họ dùng cùng sample_id và nhãn: Swin1D ảnh đọc float32[64,64], DINOv3 đọc float32[3,224,224]. Label order 0–9. Giao diện/model browser cũ giữ nguyên.

## Budget chốt trước khi chạy

Plan ghi trước training ở `artifacts/drawai/phase5-image64/experiment-plan.json`: mỗi họ đúng một cấu hình, seeds42/43/44, CPU4 threads, batch128, không augmentation/nguồn chữ số bổ trợ. Swin train từ đầu tối đa30 epoch, patience8; DINOv3 pretrained timm pin SHA giữ backbone cố định, CLS+mean-patches, linear head768→10 tối đa100 epoch, patience15. Cả hai AdamW, warmup2 epoch rồi cosine decay, gradient clip1.0. Learning rate Swin0.001/DINO head0.003; weight decay0.01. Không fine-tune backbone trong budget CPU này.

DINO trích đặc trưng train/validation5.000 mẫu một lần bằng full backbone pretrained thật. Cache float32[5000,768] ở `data/private/features/`, gắn lock hash, preprocessing/code/backbone hash, ordered IDs/labels/pixel hashes và cache SHA. Việc dùng lại đặc trưng tương đương train linear head trên backbone eval cố định, không áp augmentation. Không coi tốc độ train head là tốc độ train toàn bộ backbone.

Checkpoint mỗi run chọn validation macro-F1 cao nhất, tie bằng loss thấp hơn rồi epoch sớm hơn. Chọn seed/candidate bằng F1, loss, seed thấp hơn, tên họ. Khóa cả sáu run và selection trước final test. Đây là candidate nghiên cứu; default của sản phẩm vẫn dùng chính sách export/runtime ở P6/P7, chưa thay trong P5. Test không được dùng chọn cấu hình, seed hay đổi candidate.

## Chạy trên snapshot hiện tại

Từ repository root, dùng `.venv-dinov3` đã kiểm chứng ở P2–P4. Dependencies tại `training/drawai/requirements-paint-training.txt`. Không cần mạng nếu pretrained weights đã có; không gửi token vào source/chat.

```powershell
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_train plan
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_train train
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_train verify-selection
# Chỉ chạy sau khi toàn bộ training hoàn tất và selection đã khóa:
./.venv-dinov3/Scripts/python.exe -m training.drawai.paint_train evaluate
./.venv-dinov3/Scripts/python.exe tools/benchmark_phase5.py --family swin1d-image
./.venv-dinov3/Scripts/python.exe tools/benchmark_phase5.py --family dinov3-vits16
./.venv-dinov3/Scripts/python.exe tools/verify_phase5.py
```

Train chạy lại trước selection sẽ tiếp tục từ `last.pth` (optimizer/RNG/history), kiểm tra các run đã hoàn tất và bỏ qua run có summary/checksum hợp lệ. Không ghi đè thí nghiệm đã có selection; evaluate từ chối khi đã có final-test.json. Nếu final evaluation bị gián đoạn, chỉ tiếp tục cùng selection và receipt, không đổi model sau khi thấy test. Dùng output mới cho một thí nghiệm mới, không xóa kết quả test để tuning lại.

CLI nhận `--dataset`, `--output`, `--weights`, `--cache`. Kế hoạch v1 vẫn pin đúng lock SHA và pretrained SHA; đổi đường dẫn không đổi policy. Snapshot rebuild có timestamp/hash khác không được tự coi là cùng lock: cần verify nguồn/plan/manifest và đăng ký anchor/version thí nghiệm mới rõ ràng. Repo không chứa private snapshot, pretrained weights hay checkpoint lớn; cần các artifact local tương ứng để replay bản này. Chưa tuyên bố clean-install/Kaggle GPU đã chạy.

## Bàn giao

Trong `artifacts/drawai/phase5-image64/`:

- `experiment-plan.json`, `selection.json`, `feature-extraction-summary.json`.
- `runs/<family>-seed<seed>/`: history/summary, validation predictions, checkpoint đầy đủ và optimizer state để resume.
- `evaluation/`: test-access receipt, sáu file prediction và final-test.json. Predictions .npz bị Git ignore; summaries/metric không chứa định danh thực.
- `benchmark-*.json`: mỗi họ một fresh process, full model PyTorch CPU trên10 ảnh validation lặp lại,5 warmup+50 lần đo; không tính đọc PNG/HTTP/browser, có tính preprocessing từ uint8. RSS peak là toàn process gồm import/verify/load/inference, không phải VRAM hay riêng tham số.
- `verification.json`: replay logits validation của sáu checkpoint, đối chiếu metrics từ predictions, kiểm tra backbone không đổi và UI/model cũ.

Checkpoint tên rõ họ, dữ liệu, input, precision, seed:

```text
drawai_swin1d-image_paint35-main-v1_rows64_fp32_seed42_best.pth
drawai_dinov3-vits16_paint35-main-v1_rgb224_fp32_seed42_best.pth
```

Seed43/44 tương tự. DINO checkpoint chứa full backbone+head, không phải file head-only. `trained=true`, `release_ready=false`; ONNX/parity/runtime release thuộc P6, tích hợp canvas đúng input ảnh thuộc P7.

## Giới hạn bắt buộc khi đọc metric

37 nhóm gần giống qua split main (34 train–test,3 train–validation) vẫn được giữ/công khai từ P4. Kết quả không chứng minh benchmark sạch rò rỉ hay tổng quát hóa tới người mới hoàn toàn. Nguồn con người/35 mã người riêng biệt/quyền dùng dựa trên xác nhận chủ dữ liệu; metadata “Synthetic” gốc được giữ và đính chính riêng. Split hồi cứu theo ID, mỗi val/test chỉ5 người; ba seeds không thay thế nhiều bộ dữ liệu độc lập. Không có metadata thiết bị nên per-device là unavailable, không tự suy ra mouse/touch từ Paint.

So sánh có chênh lệch pretraining/compute: Swin scratch, DINO backbone pretrained tập ảnh lớn+head mới. Báo mean/std population(ddof0), confusion matrix/per-class/per-writer, không tuyên bố ưu thế thuần kiến trúc. Nguồn NPZ đã public, test ZIP chỉ cách ly theo quy trình, không bí mật. Canvas domain gap chưa đo bằng input ảnh mới; giữ release nét bút cũ cho tới P7.
