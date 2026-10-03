# Phase 2 — Báo cáo chuẩn bị UCI HAR

Ngày nghiệm thu: 02/10/2026. Dataset version: bản UCI HAR gốc; artifact version: uci-har-3ch-v1; contract version: 1.

## Dữ liệu đã chuẩn bị

Dùng total_acc_x/y/z trong Inertial Signals, không dùng ma trận 561 đặc trưng X_train.txt. Ba kênh là gia tốc đã qua xử lý của UCI, bao gồm trọng lực, đơn vị g; đây không phải bản ghi liên tục chưa lọc. Mỗi dòng nguồn là cửa sổ 128 điểm ở 50 Hz. Giữ nguyên ranh giới cửa sổ và thứ tự dòng.

| Tập | Cửa sổ | Người | Shape X | Dtype |
|---|---:|---:|---|---|
| Train | 5.392 | 16 | [5392,3,128] | float32 |
| Validation | 1.960 | 5 | [1960,3,128] | float32 |
| Test chính thức | 2.947 | 9 | [2947,3,128] | float32 |
| Tổng | 10.299 | 30 | | |

Nhãn int64 0–5 theo hợp đồng Phase 0. Subject ID giữ nguyên 1–30.

## Chia theo người

Trong 21 subject của train chính thức, chọn 5 subject validation bằng NumPy default_rng(seed=42). Không stratify hoặc đổi split theo kết quả model.

- Train: 1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 19, 22, 23, 27, 28, 29.
- Validation: 17, 21, 25, 26, 30.
- Test chính thức: 2, 4, 9, 10, 12, 13, 18, 20, 24.

Cả ba tập không trùng subject. Train + validation chứa đủ 7.352 dòng train gốc, không trùng index; test giữ đủ 2.947 dòng gốc. Split manifest lưu từng source index zero-based và hash file nguồn để tái lập, không chỉ lưu seed.

## Phân bố lớp

| Lớp | Train | Validation | Test |
|---|---:|---:|---:|
| WALKING | 915 | 311 | 496 |
| WALKING_UPSTAIRS | 793 | 280 | 471 |
| WALKING_DOWNSTAIRS | 725 | 261 | 420 |
| SITTING | 932 | 354 | 491 |
| STANDING | 1.000 | 374 | 532 |
| LAYING | 1.027 | 380 | 537 |

Không có NaN/Infinity trong tín hiệu. Tất cả tập có đủ sáu lớp. Không cắt/clamp giá trị hoặc cân bằng lại lớp ở Phase 2.

## Chuẩn hóa train-only

Mean/std tính trên toàn bộ timestep trong các cửa sổ thuộc 16 subject train. Dùng population std (ddof=0), tích lũy float64 rồi lưu hệ số float32 để Python/web dùng đúng cùng giá trị. Cửa sổ UCI có overlap gốc; thống kê coi mỗi cửa sổ là một mẫu, không cố dựng lại dòng thời gian.

| Kênh | Mean (g) | Std (g) |
|---|---:|---:|
| X | 0.8044244051 | 0.4175981283 |
| Y | 0.0242555067 | 0.3909381628 |
| Z | 0.0713086128 | 0.3627556562 |

Áp dụng (x−mean)/max(std,1e-8) cho cả ba tập. Train sau chuẩn hóa có mean gần 0 (sai lệch lớn nhất khoảng 3,61e-8) và std gần 1. Validation/test không được chuẩn hóa bằng thống kê riêng; mean/std khác 0/1 ở hai tập này là bình thường.

## File đầu ra

Các file NPZ ở data/processed/uci-har-3ch-v1, được ignore bởi data/.gitignore vì có thể tái tạo:

- train.npz, val.npz, test.npz.
- Mỗi file có X đã chuẩn hóa, X_raw là total acceleration nguồn trước chuẩn hóa, y, subject_id và source_index.
- preprocessing.json, labels.json, split-manifest.json được đặt cùng dữ liệu để đưa lên Kaggle.

Metadata có thể version trong Git ở artifacts/phase2:

- preprocessing.json: hệ số thật và quy tắc chuẩn hóa.
- labels.json: ID/code/tên tiếng Việt.
- split-manifest.json: split và hash file nguồn.
- data-report.json: shape, class counts, thống kê và content hashes.
- verification.json: nghiệm thu dữ liệu thực và lần chạy tái lập.

Hash nội dung mảng được dùng để đối chiếu hai lần chạy; không yêu cầu hash container NPZ giống nhau vì ZIP timestamp có thể khác.

## Sáu mẫu demo

web/public/samples/uci-har-3ch-v1 có sáu CSV, mỗi file 128 điểm. Quy tắc chọn là dòng đầu tiên của mỗi lớp trong train chính thức, hoàn toàn không dùng model/dự đoán. Cả sáu mẫu tình cờ thuộc subject 1 và tập train đã chia.

| File | Nhãn tham chiếu | Dòng nguồn (one-based) |
|---|---|---:|
| walking.csv | WALKING | 79 |
| walking_upstairs.csv | WALKING_UPSTAIRS | 151 |
| walking_downstairs.csv | WALKING_DOWNSTAIRS | 126 |
| sitting.csv | SITTING | 28 |
| standing.csv | STANDING | 1 |
| laying.csv | LAYING | 52 |

index.json ghi split, subject, source index, reference label và SHA-256 CSV. Timestamp 0..2540 ms là tương đối để hiển thị một cửa sổ độc lập, không phải trial liên tục. CSV roundtrip đúng từng giá trị nguồn float32.

Đây là mẫu để dùng giao diện, không dùng sáu mẫu này báo độ chính xác. Metric Phase 3 phải tính trên toàn bộ tập test chính thức. Chưa nối demo vào giao diện smoke test Phase 1 vì model hiện chưa huấn luyện.

## Kiểm tra nghiệm thu

| Tiêu chí | Kiểm tra | Kết quả |
|---|---|---|
| P2-01 | So từng mảng tín hiệu/nhãn/subject với dòng nguồn; kiểm tra shape/dtype/phân bố | PASS |
| P2-02 | Giao subject của từng cặp train/val/test rỗng; index bao phủ đủ | PASS |
| P2-03 | Fit độc lập từ train, đối chiếu phép biến đổi; rebuild toàn pipeline và so mảng/metadata | PASS |
| P2-04 | Sáu CSV roundtrip, nhãn và provenance khớp; chọn mẫu độc lập model | PASS |

Có bốn unit test cho split tái lập, tránh rò rỉ thống kê, std bằng zero/non-finite và ID không nguyên/số dòng không khớp. Script verify_phase2.py kiểm tra dữ liệu UCI thực và chạy pipeline lần hai trong thư mục tạm; không dùng GPU.

## Chạy lại local

Từ repo, môi trường Phase 1 đã có NumPy phù hợp:

```powershell
.venv-phase1/Scripts/python.exe training/prepare_uci_har.py --download
.venv-phase1/Scripts/python.exe -m unittest discover -s training/tests -v
.venv-phase1/Scripts/python.exe training/verify_phase2.py
```

--download dùng lại ZIP local nếu đã có. Có thể chỉ định --archive PATH hoặc --raw-dir PATH thay thế. Raw directory phải chứa Inertial Signals và subject IDs, không phải bản Kaggle chỉ có train.csv 561 đặc trưng. Với môi trường mới, chỉ cần cài training/requirements-data.txt.

## Dùng trên Kaggle ở Phase 3

Có hai cách chuẩn bị dữ liệu:

1. Đưa thư mục data/processed/uci-har-3ch-v1 lên Kaggle Dataset riêng và đọc NPZ. Giữ kèm preprocessing/labels/manifest.
2. Chạy script trên Kaggle với ZIP/folder UCI ở /kaggle/input; đặt cả --output-dir, --metadata-dir và --demo-dir dưới /kaggle/working vì /kaggle/input chỉ đọc.

Ví dụ sau khi có source/script trong notebook:

```python
!python training/prepare_uci_har.py --raw-dir "/kaggle/input/your-dataset/UCI HAR Dataset" --output-dir /kaggle/working/uci-har --metadata-dir /kaggle/working/metadata --demo-dir /kaggle/working/demos
```

Đọc dữ liệu để train:

```python
import numpy as np
import torch

with np.load("/kaggle/input/your-prepared-dataset/train.npz", allow_pickle=False) as data:
    X_train = torch.from_numpy(data["X"].copy())
    y_train = torch.from_numpy(data["y"].copy())
# X đã chuẩn hóa, y đã 0–5: không normalize hoặc trừ 1 thêm lần nữa.
```

Phase 3 sẽ dùng train.npz để tối ưu, val.npz để chọn checkpoint; test.npz chỉ đánh giá cuối.

## Nguồn và trích dẫn

[UCI HAR chính thức](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). ZIP tải từ UCI có SHA-256 c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031, được ghi trong report.

Trích dẫn dataset theo README nguồn: Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra và Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN 2013.

README trong ZIP gốc ghi hạn chế sử dụng thương mại. Dự án hiện phục vụ học tập/nghiên cứu phi thương mại; giữ thông tin nguồn khi phát hành mẫu demo.
