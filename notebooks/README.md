# DrawAI notebooks

Notebook [kaggle_train_drawai.ipynb](kaggle_train_drawai.ipynb) thuộc pipeline **Swin/PenDigits lịch sử**, được sinh từ `training/drawai/build_notebook.py`. Notebook này không train hai model ảnh Paint hiện hành. Dùng [paint-training-guide](../docs/drawai/paint-training-guide.md) cho Swin ảnh và DINOv3; xem [hướng dẫn chạy ứng dụng](../docs/drawai/paint-app-guide.md).

Sau P1 chỉ giữ Swin Transformer 1D: smoke 2 epoch không mở test; full 4 pilot + 3 official run trên PenDigits. DINOv3 được bổ sung ở P2. Không còn notebook HAR/MotionSense trong project.

Từ root chạy `python -m training.drawai.build_notebook` để tạo notebook và `artifacts/drawai/kaggle/drawai-source.zip`. Cần dependencies trong `training/drawai/requirements-notebook.txt`. Gắn bundle vào Kaggle input, bật GPU cho full mode. Xem [training guide](../docs/drawai/training-guide.md).

Chạy `python -m training.drawai.verify_notebook` để thực thi CPU smoke. Kết quả local không chứng minh đã chạy Kaggle GPU; full/GPU chưa thực thi trong P1. Không dùng notebook cũ từ bản phục hồi cùng source mới.
