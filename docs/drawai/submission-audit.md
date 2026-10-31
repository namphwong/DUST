# Đối chiếu 6 yêu cầu nộp bài - 31/10/2026

**Chưa hoàn thành toàn bộ:** phần deploy cloud cần tài khoản Vercel/Render được đăng nhập hoặc kết nối. Code phục vụ deploy đã được bổ sung và kiểm tra; không coi file cấu hình hoặc test local là link deploy thực.

| Yêu cầu | Trạng thái | Bằng chứng nộp |
|---|---|---|
| 1. DINOv3 backbone + GitHub/README | ĐÃ LÀM | [GitHub](https://github.com/namphwong/DUST), [README](https://github.com/namphwong/DUST/blob/main/README.md), [DINO source](https://github.com/namphwong/DUST/blob/main/training/drawai/dinov3_model.py), [hướng dẫn DINO](dinov3-guide.md) |
| 2. Data tự thu, công bố Kaggle | ĐÃ LÀM theo xác nhận chủ dữ liệu | [Kaggle public](https://www.kaggle.com/datasets/namphwong/drawai-handwritten-digits-placeholder); API public xác nhận có file NPZ, không còn folder trống |
| 3. Design DrawAI + model xuất | ĐÃ LÀM kỹ thuật | [Code design](https://github.com/namphwong/DUST/tree/main/web), [bundle model public](https://github.com/namphwong/DUST/releases/tag/drawai-models-paint35-v1); Swin ONNX và DINO TorchScript |
| 4. Deploy | CHƯA HOÀN THÀNH cloud | [Hướng dẫn/cấu hình sẵn](public-deploy-guide.md), `web/vercel.json`, `render.yaml`; chưa có URL cloud đã kiểm chứng |
| 5. CI/CD AI | CI + DELIVERY ĐÃ CHẠY; cloud CD còn chờ | [Run CI đã PASS](https://github.com/namphwong/DUST/actions/runs/41164031905), [publish PASS](https://github.com/namphwong/DUST/actions/runs/41163501943), [báo cáo](cicd-docker-report.md). Nếu rubric đòi continuous deployment cloud thì mục này còn thiếu phần đó |
| 6. Build Docker và public Docker Hub | ĐÃ LÀM | [Docker Hub](https://hub.docker.com/r/namphwong/drawai-dinov3-api), tag `v1.0.0`; [pull ẩn danh + inference runner mới PASS](https://github.com/namphwong/DUST/actions/runs/41164033474) |

PDF step by step cho mục5+6 đã xuất riêng ngoài repo. Bản cập nhật ghi rõ actual run, digest, model release, hướng dẫn tái thực hiện và trạng thái cloud chưa chạy.

## Cụ thể chỗ DINOv3 đã thay

`training/drawai/dinov3_model.py`: `TimmDinoBackbone` dùng pretrained DINOv3, `DinoDigitClassifier` nhận CLS + mean patch tokens (bỏ register tokens) rồi head10lớp. Backbone đóng băng theo experiment Paint. Swin1D là đối chứng riêng nhận ảnh hàng pixel `[1,64,64]`, không ghép chung hai model và không giả lập trajectory từ NPZ.

Model mặc định seed42 theo validation P5; Swin seed43 đối chứng. Không huấn luyện lại hoặc chọn checkpoint theo test trong đợt deploy. DINO primary là TorchScript; DINO ONNX vẫn diagnostic do kết quả parity cũ không đạt ở2/1.000 mẫu.

## Dữ liệu thực có gì

Kaggle API public ngày31/10/2026 xác nhận dataset `isPrivate=false` và chứa `So_viet_tay_35_nguoi.npz` (2.828.016 bytes) cùng README. NPZ có7.000 ảnh64×64/35 mã người theo hồ sơ dự án. Chủ dữ liệu đã xác nhận viết bằng Paint, mỗi person_id là người khác nhau và đồng ý train/evaluate/công bố. Đây là xác nhận của chủ dữ liệu, không phải kiểm chứng độc lập. [Provenance](collection/paint-image64-provenance.json) giữ nguyên; dòng Synthetic sai trong NPZ được giải thích, không sửa bytes nguồn.

Giữ giới hạn37 nhóm tương đồng qua split main, không lấy accuracy Paint làm bảo đảm chất lượng nét canvas người mới.

## Phần bổ sung trong lần rà soát này

- Public frontend: DINO gọi origin HTTPS qua `VITE_API_BASE_URL`; Swin ảnh chạy WASM một thread trong worker, hash/bytes/shape cố định. So cùng pixelhash/logits với fixture P7 đã khóa.
- Chờ backend ngủ/thức có deadline, không tự gửi lặp POST;429 cho retry, giữ nét; clear/đổi nét hủy worker đang chờ, loại response cũ. Nhãn quyền riêng tư đúng với việc gửi ảnh ra server; không đổi CSS/canvas/layout.
- Swin ONNX2,39MB được thêm vào release public cho build Vercel tải trực tiếp và verify; không commit binary mới vào Git. Bundle P6/modelhash gốc giữ nguyên.
- `render.yaml` pin digest image public, planFree, Singapore, port10000, `/health`, CORS exact origins. `web/vercel.json` cấu hình Vite/build và SPA rewrites cụ thể, không rewrite assets thành HTML.
- CD smoke chờ **đúng revision mới** sau khi gọi hook, tối đa600giây; không chấm nhầm service cũ đang healthy. Container gate90giây/p955giây/memory450MB không bị nới.
- CI bổ sung job browser public với API/model thật và3 tests readiness; báo cáo P7 test mới chuyển sang thư mục riêng, không ghi đè snapshot cũ.

Kiểm chứng local: **44 Python tests PASS;7 public browser tests PASS;13 local regression browser tests PASS**. Hai cấu hình được instance-validate với schema chính thức Render/Vercel. Local initial desktop/mobile main và legacy pixel-identical; public giữ bố cục và đổi câu chữ trạng thái/quyền riêng tư. Bằng chứng: [public-deployment](../../artifacts/drawai/public-deployment/). Cloud/touch vật lý/người mới vẫn NOT RUN. **CI cho commit `8edc2c1` SUCCESS**, cả3job code/public_frontend/image PASS: [run41165630051](https://github.com/namphwong/DUST/actions/runs/41165630051), metadata lưu tại `ci-run.json` trong evidence.

## Điều kiện còn thiếu để hoàn tất

Chủ tài khoản cần đăng nhập/kết nối Vercel và Render. Khi có quyền: tạo serviceFree từ digest đã PASS; import frontend root`web`, đặt2 env; lấy hostname thực để cấu hình CORS; kiểm tra public HTTPS/inference; nối deploy hook và chạy CD cloud; cập nhật URL vào README/PDF. Không tạo service trả phí, không yêu cầu gửi mật khẩu/token vào chat.

Ngày rà soát, Render trình duyệt đang ở trang login, GitHub chưa có Render key/hook/API URL; kết nối ứng dụng đang chờ chủ tài khoản hoàn tất. Vì vậy chưa thể nghiệm thu mục4 và phần cloud của mục5.
