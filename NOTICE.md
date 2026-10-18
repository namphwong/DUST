# DrawAI — nguồn gốc và ghi công

DrawAI được phát triển trong checkout DuST. Giữ nguyên LICENSE đi kèm repository; xem lịch sử Git và bản phục hồi P0/P1 để truy nguyên code đã dọn.

- Backbone Swin 1D tại `models/SwinTransformer_cls.py` giữ nguyên từ checkout trước P1, dựa trên [Swin-Transformer-1D](https://github.com/meraks/Swin-Transformer-1D). Header tác giả và code nguồn được giữ nguyên.
- Dự án gốc: [DuST: Dual Swin Transformer for Multi-modal Video and Time-Series Modeling](https://openaccess.thecvf.com/content/CVPR2024W/WAD/html/Shi_DuST_Dual_Swin_Transformer_for_Multi-modal_Video_and_Time-Series_Modeling_CVPRW_2024_paper.html), Shi và cộng sự, CVPR Workshops 2024. Code video/fusion đã chuyển ra ngoài project ở P1 vì không dùng trong DrawAI.
- Dữ liệu PenDigits: [UCI Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits); nguồn và hash trong `docs/drawai/data-report.md`.
- DINOv3 P2 dùng [timm/PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) 1.0.22 và [weights công khai do timm phát hành](https://huggingface.co/timm/vit_small_patch16_dinov3_qkvb.lvd1689m), đã tải và strict-load. Code timm có license riêng; weights theo DINOv3 custom license. License, revision, hashes và khác biệt native implementation được lưu trong báo cáo P2; không tuyên bố trùng bit với snapshot Meta gated.

Ghi công không thay thế license từng nguồn hoặc quyền công bố dataset; không bổ sung quyền phân phối pretrained weights chưa được xác minh.
