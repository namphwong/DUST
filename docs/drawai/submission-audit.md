# Đối chiếu 6 yêu cầu nộp bài - 31/10/2026

**Đã hoàn thành cả sáu hạng mục kỹ thuật và có link/bằng chứng tái kiểm tra.** Deploy là demo nghiên cứu `technical_only`; chưa thay thế nghiệm thu chất lượng trên người viết mới hoặc thiết bị cảm ứng thật.

| Yêu cầu | Trạng thái | Bằng chứng nộp |
|---|---|---|
| 1. DINOv3 backbone + GitHub/README | ĐÃ LÀM | [GitHub](https://github.com/namphwong/DUST), [README](https://github.com/namphwong/DUST/blob/main/README.md), [DINO source](https://github.com/namphwong/DUST/blob/main/training/drawai/dinov3_model.py), [hướng dẫn DINO](dinov3-guide.md) |
| 2. Data tự thu, công bố Kaggle | ĐÃ LÀM theo xác nhận chủ dữ liệu | [Kaggle public](https://www.kaggle.com/datasets/namphwong/drawai-handwritten-digits-placeholder); có `So_viet_tay_35_nguoi.npz` và README |
| 3. Design DrawAI + model xuất | ĐÃ LÀM kỹ thuật | [Code design](https://github.com/namphwong/DUST/tree/main/web), [bundle model public](https://github.com/namphwong/DUST/releases/tag/drawai-models-paint35-v1); Swin ONNX và DINO TorchScript |
| 4. Deploy | ĐÃ LÀM | [Frontend Vercel](https://drawai-pied.vercel.app), [backend health Render](https://drawai-dinov3-api.onrender.com/health);3 browser smoke tests cloud PASS |
| 5. CI/CD AI | ĐÃ LÀM | [Publish v1.0.1 SUCCESS](https://github.com/namphwong/DUST/actions/runs/41166886046), [CD Render SUCCESS](https://github.com/namphwong/DUST/actions/runs/41183564442), [báo cáo](cicd-docker-report.md) |
| 6. Build Docker và public Docker Hub | ĐÃ LÀM | [Docker Hub](https://hub.docker.com/r/namphwong/drawai-dinov3-api), tag `v1.0.1`, digest `sha256:bf43b3f…aede4c`; pull ẩn danh/resource gate PASS trước deploy |

PDF step by step cho mục5+6 được xuất riêng ngoài repo tại `C:\Users\admin\Downloads\DrawAI_Submission\output\pdf\DrawAI_CICD_Docker_Implementation_2026-10-31.pdf`.

## Chỗ thay DINOv3

`training/drawai/dinov3_model.py`: `TimmDinoBackbone` dùng pretrained DINOv3; `DinoDigitClassifier` nhận CLS + mean patch tokens (bỏ register tokens) rồi head10 lớp. Backbone đóng băng theo experiment Paint. Swin1D là đối chứng riêng nhận ảnh hàng pixel `[1,64,64]`, không ghép hai model và không giả lập trajectory từ NPZ.

Model mặc định DINO seed42 được chọn bằng validation P5; Swin seed43 là đối chứng. DINO primary là TorchScript; DINO ONNX chỉ diagnostic vì parity strict cũ không đạt2/1.000 mẫu.

## Dữ liệu và giới hạn chất lượng

Kaggle public chứa7.000 ảnh64×64/35 mã người theo hồ sơ dự án. Chủ dữ liệu xác nhận ảnh được người thật viết bằng Paint, mỗi `person_id` là một người và đồng ý dùng cho train/evaluate/công bố. Đây là xác nhận của chủ dữ liệu, chưa phải kiểm chứng độc lập; dòng `Synthetic` sai trong NPZ được giải thích trong provenance, không sửa bytes nguồn.

Giữ nguyên giới hạn37 nhóm tương đồng qua split main. Metric Paint không bảo đảm chất lượng nét canvas người mới; DINO chỉ đúng5/10 demo PenDigits khác miền. `release_ready=false`, `technical_only=true` không đổi.

## Bằng chứng deploy/CD cuối

- Frontend giữ design DrawAI, public profile chỉ mở `/`, `/drawai`, `/drawai/compare`; Swin chạy trong browser, DINO gọi Render HTTPS.
- Render Free Singapore chạy image bất biến `namphwong/drawai-dinov3-api@sha256:bf43b3f739ca7ea250eaa217d1b0ef922f879745a2f5aa206bc5494922aede4c`, source revision `c9593cac…c629`.
- CD run41183564442: fresh anonymous pull gate SUCCESS rồi mới gọi Render; cloud smoke8 fixture +30 request PASS, p95 `1299,825ms`, readiness `63790,040ms`, peak memory `435.429.376` bytes/512MiB, OOM kill0.
-3 browser cloud tests PASS: DINO inference thực, so sánh Swin-browser/DINO-Render và route/mobile layout. Bằng chứng JSON tại [public-deployment](../../artifacts/drawai/public-deployment/).
- CORS chỉ cho origin production `https://drawai-pied.vercel.app`; hook Render lưu trong GitHub Secret, không commit. Hook từng lộ trong log công cụ đã được regenerate ngay và GitHub Secret được cập nhật bằng hook mới.

Nghiệm thu cảm ứng vật lý/người viết mới vẫn NOT RUN; đây không phải điều kiện để chứng minh sáu hạng mục triển khai kỹ thuật nhưng là phần còn thiếu nếu muốn gọi model sẵn sàng sản phẩm.
