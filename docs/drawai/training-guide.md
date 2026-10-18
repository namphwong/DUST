# DrawAI — Training guide sau P1

Cập nhật 17/10/2026. P1 chỉ có Swin baseline; DINOv3 thuộc P2. Lệnh dưới chạy từ root, trừ npm/npx. Dùng `.venv-drawai` thay `.venv-phase1` nếu đã cài môi trường riêng.

## Smoke Swin, không mở test

```powershell
./.venv-phase1/Scripts/python.exe -m training.drawai.train --config training/drawai/configs/smoke-swin.json --output-dir artifacts/drawai/swin-smoke-v2 --device cpu --threads 4
```

Chỉ thêm `--resume` khi output đã có last.pth với cùng config/môi trường. Smoke không phải model release hoặc metric chất lượng.

## Thí nghiệm Swin-only mới

Plan `drawai-swin-experiment-v2`: 4 pilot (L64/L128 × augmentation), chọn bằng validation; 3 official run seeds42/43/44. Không ghi đè thư mục `d3-local-v1` đã mở test.

```powershell
./.venv-phase1/Scripts/python.exe -m training.drawai.experiment --stage pilots --output-dir artifacts/drawai/swin-pendigits-v2 --device cpu --threads 4
./.venv-phase1/Scripts/python.exe -m training.drawai.experiment --stage official --output-dir artifacts/drawai/swin-pendigits-v2 --device cpu --threads 4
./.venv-phase1/Scripts/python.exe -m training.drawai.evaluate --output-dir artifacts/drawai/swin-pendigits-v2 --threads 4
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_training --output-dir artifacts/drawai/swin-pendigits-v2
```

Full experiment mới chưa được train ở P1. Không dùng test để sửa config. Plan dữ liệu riêng/split theo người thuộc P3–P5; các lệnh này đang dùng PenDigits.

## Audit và export release PenDigits hiện có

```powershell
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_training --output-dir artifacts/drawai/d3-local-v1
./.venv-phase1/Scripts/python.exe -m training.drawai.report_experiment --output-dir artifacts/drawai/d3-local-v1 --destination artifacts/drawai/swin-report.md
# Windows local: chỉ dùng khi cần workaround DLL theo environment.md.
$env:DUST_DLL_DIRECTORY = 'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
./.venv-phase1/Scripts/python.exe -m training.drawai.export_onnx
```

Audit lịch sử mặc định chỉ kiểm tra Swin được chọn: chain plan/config/selection/evaluation, hash checkpoint, strict state, validation history, dữ liệu và tái tạo 1.495 validation predictions; không yêu cầu CNN checkpoint. Các kết quả CNN lịch sử không được tuyên bố audit lại. Export chạy full-test parity và cần checkpoint local cùng dữ liệu processed; không thực hiện training mới.

Exporter vẫn xuất vào `web/public/models/drawai-swin-v1/` và ghi artifact export. P6 sẽ chuẩn hóa filename cùng loader/manifest/tests; không tự đổi riêng tên file.

## Notebook/Kaggle

```powershell
./.venv-phase1/Scripts/python.exe -m training.drawai.build_notebook
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_notebook
```

Builder/runner cần `requirements-notebook.txt`. Notebook hiện tại chỉ train Swin; smoke 2 epoch train/validation, test unopened. Full mode: 4 pilot + 3 official run. GPU/full branch chưa chạy lại ở P1.

Bundle `artifacts/drawai/kaggle/drawai-source.zip` chứa code/config/tensor DrawAI hiện tại. Gắn bundle làm Kaggle input và mở `notebooks/kaggle_train_drawai.ipynb`; đây không phải dataset private đã thu hoặc link Kaggle public đã hoàn thành.

## Kiểm chứng

```powershell
./.venv-phase1/Scripts/python.exe -m unittest discover -s training/tests -p "test_drawai*.py" -v
```

Xem [báo cáo P1](phase1-report.md) cho phạm vi đã chạy. Các báo cáo D3/D4 cũ là lịch sử checkpoint PenDigits, không mô tả pipeline DINOv3 chưa có.
