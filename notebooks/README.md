# Phase 3 trên Kaggle

Các notebook tự chứa source/config; không cần GitHub, token hay backend trả phí. Bản GPU đã chạy thành công và artifact tải về đã kiểm chứng ngày 03/10/2026: [Kaggle notebook riêng tư](https://www.kaggle.com/code/namphwong/kaggle-train-har-gpu), version 1/run 354801477. Xem `docs/kaggle-training-report.md` và `artifacts/phase3/kaggle/verification.json`. Swin seed 44 được chọn bằng validation, test F1 0,870489, chưa đạt 0,90. Bản `kaggle_train_har.ipynb` một seed vẫn dành cho tham chiếu CPU.

## Chạy

Lượt nghiệm thu GPU ngày 03/10/2026 dùng **`kaggle_train_har_gpu.ipynb`**: CUDA bắt buộc, seeds 42/43/44 đã chốt, giữ nguyên hyperparameters local và báo mean/sample std của mọi seed. Đây là replication/kiểm tra độ ổn định, không phải test mù mới (test chính thức đã xem ở lượt CPU). Kế hoạch được lưu trong `artifacts/phase3/kaggle-run-plan.json`. Tạo lại bản này bằng `python training/build_kaggle_notebook.py --gpu`; không đổi candidates dựa vào test.

1. Tạo notebook trên [Kaggle Code](https://www.kaggle.com/code), import `kaggle_train_har.ipynb` từ máy.
2. Trong Settings chọn Accelerator GPU nếu tài khoản cho phép; bật Internet để tải UCI ZIP. Không cần multi-GPU. Không giả định số giờ quota cố định; kiểm tra trong tài khoản của bạn. Xem [tài liệu notebooks](https://www.kaggle.com/docs/notebooks) và [sử dụng GPU hiệu quả](https://www.kaggle.com/docs/efficient-gpu-usage).
3. Mặc định `SEEDS=[42]`. Nếu đủ quota và muốn đánh giá ổn định hơn, đổi thành `[42,43,44]` **trước khi chạy**, không thêm seed sau khi đã đọc test để tìm kết quả đẹp hơn.
4. Run All. Nếu không có Internet, Add Input chứa [ZIP UCI gốc](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip), sửa `DATA_SOURCE` thành đường dẫn ZIP hoặc folder `UCI HAR Dataset`. Không dùng CSV/features 561 chiều thay cho inertial signals.
5. Save Version để giữ output. Tải `phase3-results.zip` chứa checkpoint tốt nhất, metric, hình và preprocessing. Đừng tải last.pth không rõ nguồn hoặc gửi API token vào chat.

## Gián đoạn và resume

`last.pth` lưu model/optimizer/scheduler/RNG ở cuối mỗi epoch. Chạy lại các cell trong cùng thư mục sẽ resume với config và dữ liệu không đổi. Khi mở phiên Kaggle mới, cần khôi phục thư mục project từ output đã lưu hoặc bản sao có **last.pth**; notebook không tự dò output phiên trước. ZIP tải nhanh ở cell cuối chỉ có **best.pth**, không chứa optimizer, nên không dùng để resume training.

Nếu đổi kiến trúc/config/split, dùng thư mục `WORK_ROOT` khác và chốt thí nghiệm trước evaluation. `selection.json` sẽ từ chối thay candidates đã đóng băng.

## Kết quả và giới hạn

Chọn checkpoint và seed theo macro-F1 validation, riêng CNN và Swin. CNN chỉ là đối chứng; web vẫn dùng Swin 1D theo phạm vi project. Test chính thức được mở bởi `evaluate_har.py` sau khi ghi selection. Nhãn, mean/std thật và `trained=true` nằm trong checkpoint. Phase 4 đã export ONNX và nghiệm thu parity/benchmark, trang mặc định dùng model đã train; xem `docs/phase4-artifact-report.md`.

GPU/PyTorch phiên bản khác có thể cho kết quả khác CPU dù cùng seed. Bản tham chiếu chỉ chạy một seed; bản GPU chạy ba seed và báo sample std. Không coi metric là độ chính xác bảo đảm trên cảm biến điện thoại ngoài dataset. Không tái tạo bản GPU đã upload sau khi sửa source nếu còn cần đối chiếu source hash của run này.

Import output tải về bằng `python training/import_kaggle_results.py --archive PATH_TO_ZIP`. Thư mục mặc định `artifacts/phase3/kaggle` phải chưa có file để bảo vệ artifact đã lưu; muốn kiểm lại dùng `--output-dir` mới.

## Đồng bộ notebook khi sửa source

```powershell
.venv-phase1/Scripts/python.exe training/build_kaggle_notebook.py
uv pip install --python .venv-phase1/Scripts/python.exe -r training/requirements-notebook-check.txt
.venv-phase1/Scripts/python.exe training/verify_kaggle_notebook.py --reuse-runs
```

Verifier chạy toàn bộ code cells trong kernel Jupyter cục bộ, tái chuẩn bị/kiểm chứng dữ liệu, resume các run đã hoàn tất, so sánh metric CLI và tạo ZIP. Đây không phải bằng chứng notebook đã chạy trên máy chủ Kaggle. Bỏ `--reuse-runs` để train lại từ đầu trong thư mục tạm.
