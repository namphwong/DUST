# Tài liệu DrawAI

## Pipeline hiện hành: ảnh Paint → Swin1D / DINOv3

- [Trạng thái và phần chưa hoàn thành](status.md), [kế hoạch P0–P8](two-model-roadmap.md).
- [Kế hoạch ưu tiên: Vercel frontend + Render backend VR0–VR6](vercel-render-plan.md) — chưa triển khai; kiểm tra tài nguyên Free trước khi public.
- [Phương án tham khảo GitHub Pages W0–W6](github-pages-plan.md), đã được thay hướng ưu tiên theo lựa chọn mới.
- [Chạy ứng dụng DINO mặc định / so sánh hai model](paint-app-guide.md).
- [Nguồn/chuẩn bị ảnh NPZ](collection/paint-image64-guide.md), [provenance theo xác nhận chủ dữ liệu](collection/paint-image64-provenance.json).
- [P3 dữ liệu](phase3-report.md), [P4 khóa split/review](phase4-report.md).
- [Training/validation selection](paint-training-guide.md), [kết quả P5](phase5-report.md).
- [Export/runtime/hash](paint-export-guide.md), [báo cáo P6](phase6-report.md).
- [Tích hợp, kiểm tra UX và khoảng cách Paint/canvas P7](phase7-report.md).
- [Contract hai model](two-model-contract.json), [rà soát và dọn repo](repo-review-report.md).

`trained`/`artifact_verified` không đồng nghĩa `release_ready`: hiện chưa nghiệm thu chất lượng canvas người viết mới/cảm ứng thật; model lớn chưa phân phối cho clean checkout. Giữ cảnh báo37 nhóm tương đồng qua split và kết quả5/10 demo PenDigits.

## Các tài liệu theo mốc cũ

[P0](phase0-report.md), [P1](phase1-report.md), [P2](phase2-report.md) là bằng chứng tại thời điểm phase đó. [DINO guide](dinov3-guide.md) mô tả nguồn pretrained/feasibility P2, không phải trạng thái head hiện nay.

Các tên D0–D7 tại [phases.md](phases.md), [training-guide](training-guide.md), [model-card](model-card.md), [ui-report](ui-report.md), [acceptance-report](acceptance-report.md) thuộc release **Swin PenDigits lịch sử**. Giữ để truy nguyên `/drawai/legacy`, nguồn kiến trúc và quá trình kiểm chứng; không lấy metrics đó làm kết quả Paint. [Notebook](../../notebooks/README.md) cũng thuộc pipeline PenDigits.

[Công cụ thu nét mới](collection/operator-guide.md) và [protocol canvas](canvas-protocol.md) phục vụ thu bổ sung; NPZ Paint chỉ có ảnh, không có trajectory. Chưa thực hiện thu/đánh giá canvas người viết mới theo protocol này.
