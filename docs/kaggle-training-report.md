# Phase 3 — Kết quả Kaggle GPU đã kiểm chứng

Cập nhật 03/10/2026. Đã chạy notebook riêng tư trên tài khoản Kaggle đã đăng nhập, không dùng GPU local hoặc dịch vụ trả phí. [Notebook và output](https://www.kaggle.com/code/namphwong/kaggle-train-har-gpu), version 1, run `354801477`, trạng thái successful, runtime toàn notebook **9 phút 47 giây**. Kaggle cấp T4 x2; mã train sử dụng một Tesla T4, không chạy distributed.

## Thí nghiệm

CNN chỉ là baseline đối chứng; model triển khai vẫn là Swin Transformer 1D của repo. Hai family × ba seed đã chốt 42/43/44, giữ nguyên hyperparameters của lượt CPU. Không thêm seed, đổi cấu hình hay chọn lại theo test. Train/validation/test: 5392/1960/2947 cửa sổ, subject-disjoint; total acceleration x/y/z, 128 điểm, 50 Hz; normalization fit train-only. Kiến trúc Swin small: embed 32, depths [2,2,2,2], heads [2,4,8,16], patch 2, window 8; 2.191.594 tham số. AdamW, lr 0,0003, batch 64, tối đa 60 epochs, patience 12.

Môi trường thực tế: Python 3.13.15, PyTorch 2.11.0+cu128, NumPy 2.1.3, CUDA. Khác môi trường CPU cũ; không thể quy mọi khác biệt điểm số chỉ cho GPU. Test chính thức đã xem trong lần CPU trước: đây là replication và kiểm tra độ ổn định đa seed, **không phải benchmark mù mới**.

## Metric tính lại từ prediction tải về

| Run | Best epoch | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| CNN seed 42 | 30 | 0,933829 | 0,916865 | 0,917791 |
| CNN seed 43 — chọn theo val | 21 | 0,954258 | 0,868341 | 0,869842 |
| CNN seed 44 | 14 | 0,948587 | 0,912793 | 0,914142 |
| Swin seed 42 | 17 | 0,906424 | 0,874449 | 0,874725 |
| Swin seed 43 | 7 | 0,879741 | 0,859518 | 0,859216 |
| Swin seed 44 — triển khai | 31 | 0,911692 | 0,871734 | 0,870489 |

| Family, 3 seeds | Test accuracy mean ± sample std | Test macro-F1 mean ± sample std |
|---|---:|---:|
| CNN | 0,899333 ± 0,026917 | 0,900591 ± 0,026692 |
| Swin | 0,868567 ± 0,007953 | 0,868143 ± 0,008016 |

Std dùng ddof=1; đây không phải khoảng tin cậy. Seed triển khai là **44**, dù seed 42 có test cao hơn: giữ quy tắc chọn bằng validation, không cherry-pick test. Swin seed 44 chạy 43 epochs, best epoch 31, train 191,68 giây (không gồm data preparation/evaluation).

## Chất lượng và quyết định

Swin triển khai đạt accuracy **87,17%**, macro-F1 **87,05%**, **chưa đạt mục tiêu 90% macro-F1**. GPU không tự làm model chính xác hơn. CNN trung bình tốt hơn Swin trong cấu hình 3 kênh này; không kết luận Swin tốt hơn baseline.

Ở seed 44, recall ngồi chỉ 70,88%, đứng 84,02%; 119/491 mẫu ngồi bị dự đoán đứng, 84/532 mẫu đứng bị dự đoán ngồi. Hai nhầm lẫn này chiếm 203/378 lỗi. Recall nằm 100% chỉ có nghĩa trên 537 cửa sổ của test này, không bảo đảm trên cảm biến thực tế.

Tiếp tục dưới dạng **demo nghiên cứu có giới hạn**, theo nhánh phân tích lỗi của P3-04. Phase 3 hoàn tất huấn luyện/nghiệm thu artifact; mục tiêu chất lượng vẫn không đạt. Không tune theo test đã mở để tuyên bố benchmark mới. Muốn cải thiện cần protocol mới chốt trước, lựa chọn bằng validation, và đánh giá độc lập chưa dùng để ra quyết định. Web cần cảnh báo domain shift, vị trí điện thoại/đơn vị/tần số và khả năng nhầm ngồi–đứng.

## Artifact và kiểm chứng

Output đã tải vào máy và import tách biệt với CPU tại `artifacts/phase3/kaggle/`. Không ghi đè checkpoint cũ. ZIP SHA-256:

`60c509d54bb1ca1dec80d64c757e30c4294097414267a88cd6854e91d3a0797c`

Checkpoint triển khai: `artifacts/phase3/kaggle/artifacts/phase3/runs/swin-seed44/best.pth`, SHA-256:

`f45501770c7a66b0cb5e29a5b3f7a84bd5ad204eb63dab80fdcad865fe9bc749`

Verifier `training/import_kaggle_results.py` kiểm tra ZIP path/loại file, source hash của đúng notebook đã upload, sáu cấu hình/seed CUDA, train/val data hash, labels/preprocessing, checkpoint hash, prediction với 2947 nhãn/subject/source index chính thức, logits hữu hạn/argmax, tính lại accuracy/F1/confusion/per-class, chọn theo val và mean/std. Không unpickle checkpoint lúc import. Kết quả: [verification.json](../artifacts/phase3/kaggle/verification.json). Các metric đầy đủ: [evaluation.json](../artifacts/phase3/kaggle/artifacts/phase3/evaluation/evaluation.json).

```powershell
.venv-phase1/Scripts/python.exe training/import_kaggle_results.py --archive 'C:\Users\admin\Downloads\phase3-results.zip'
```

Lệnh từ chối ghi đè thư mục output đã có file; dùng `--output-dir` mới nếu kiểm lại. Checkpoint/prediction binary được gitignore; giữ bản tải/Kaggle output để phục hồi. ZIP best checkpoint không có optimizer/last.pth nên không dùng resume.

Kiểm tra bổ sung: checkpoint seed 44 load strict trên CPU với `weights_only=True` (chỉ allowlist kiểu TorchVersion của PyTorch đã cài), forward 8 cửa sổ đầu test; logits lệch tối đa 3,8147e-6 so với CUDA, argmax giống 8/8. Xem [checkpoint smoke test](../artifacts/phase3/kaggle-checkpoint-smoke.json). Đây chưa phải ONNX/browser parity. Bộ unit test training/data/import: **11/11 pass**; compileall và git diff --check pass. [Ảnh xác nhận runtime Kaggle](../artifacts/phase3/kaggle-run-proof.jpg).

Cập nhật sau nghiệm thu Phase 4: checkpoint seed 44 đã export ONNX, full test/browser parity và benchmark PASS; trang mặc định dùng model đã train. Xem [báo cáo artifact](phase4-artifact-report.md). Chưa deploy Vercel, chưa có CSV người dùng/worker.
