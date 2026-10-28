# Kế hoạch tối ưu DrawAI để công khai trên GitHub Pages

Ngày28/10/2026. Đây là kế hoạch mới, chưa triển khai. Giữ design DrawAI, hai model Swin1D ảnh/DINOv3, dữ liệu và checkpoint đã khóa. Không đổi model mặc định ngầm để làm một bản deploy trông như đã chạy DINO.

**Đã thay hướng ưu tiên:** theo lựa chọn mới của người dùng, dùng [Vercel frontend + Render backend](vercel-render-plan.md). Tài liệu Pages này giữ làm phương án tham khảo, không phải kế hoạch triển khai đang ưu tiên.

## Kết luận lựa chọn kiến trúc

GitHub Pages phù hợp phục vụ HTML/CSS/JavaScript, workers, WASM và model ONNX. Pages không chạy backend Python/TorchScript hiện tại. Với repo DUST, URL dự kiến là `https://namphwong.github.io/DUST/`, chưa phải URL đã deploy.

**Lộ trình khuyến nghị:** GitHub Pages phục vụ frontend; Swin1D ảnh chạy ONNX/WASM trên thiết bị; DINOv3 tiếp tục là mặc định qua backend HTTPS riêng. Khi DINO ONNX/Web đạt đầy đủ gate, chuyển DINO sang browser để loại dependency backend. Bản đầu dùng Render Free là ứng viên backend, phải vượt kiểm tra memory/timeouts trước; không bảo đảm tài nguyên miễn phí sẽ đủ.

Nếu yêu cầu chỉ GitHub Pages, không có backend ngay từ đầu, có thể công khai một bản Swin riêng. Đây là phiên bản phục vụ khác với DINO mặc định đã chọn bằng validation; phải ghi rõ model, không gọi bản Swin là bản DINO. Chưa triển khai lựa chọn đó trong kế hoạch này.

| Phương án | Hiện trạng / điều kiện |
|---|---|
| Pages + cả hai model trong browser | Đích tối ưu chi phí máy chủ; DINO ONNX hiện chưa được duyệt, cần parity/performance/preprocessing web |
| Pages + Swin browser + DINO API | Khuyến nghị cho lần deploy đầu để giữ DINO mặc định; backend có sleep/quota và cần kiểm tra RAM |
| Pages + chỉ Swin browser | Có thể làm nhanh cho demo riêng; chưa thay thế bản đầy đủ hai model/DINO mặc định |

## W0 — Chốt release web và tiêu chí nghiệm thu

- Ghi hash checkpoint, bundle P6, selection P5 và phiên bản canvas adapter. Không thay split/model/adapter theo kết quả demo/test.
- Khóa vai trò: Swin ảnh `[1,64,64]` là đối chứng; DINO RGB224 là mặc định. Không đưa nhầm model PenDigits `[1,3,128]` lên làm bản Paint.
- Giữ layout/màu/font/canvas/undo/clear/top3. Nội dung phải đúng runtime: DINO qua server không được ghi “hoàn toàn chạy trên thiết bị”; cập nhật câu chữ trong cùng vị trí/style.
- Chốt thiết bị mục tiêu, mẫu procedural/validation để debug, tolerance P6 và giới hạn bộ nhớ/latency trước khi đo.

**Bàn giao:** manifest release web, danh sách route public, checklist desktop/mobile; chưa publish trang có nút inference trỏ localhost.

## W1 — Làm frontend tương thích project Pages

- Vite build profile Pages đặt `base: '/DUST/'`; profile local hiện có vẫn dùng `/`.
- Thay router/link hardcode `/drawai`, `/collect` bằng helper theo base. Dùng hash routes để refresh/deep link không404, ví dụ `/DUST/#/drawai/compare`.
- Rà toàn bộ đường dẫn model, worker, WASM, sample, image, JSON và download; dùng base/URL helper, không đường dẫn ổ đĩa hoặc localhost trong bản public.
- Tách lazy loading các trang kỹ thuật. Chỉ deploy frontend cần thiết và assets public đã duyệt, không deploy raw NPZ/checkpoint/cache/venv/log/ảnh người viết.
- API base URL cấu hình riêng ở build; đây là URL công khai, không phải nơi cất secret/token.

**File dự kiến:** `web/vite.config.ts`, `web/src/main.tsx`, helper route/URL, các link/loaders liên quan.

**Nghiệm thu:** chạy build dưới prefix `/DUST/`, mở trực tiếp/refresh route, tải worker/WASM/mẫu không404, không phát request localhost.

## W2 — Chạy Swin1D ảnh trực tiếp trong browser

- Phát hành artifact ONNX ảnh đã duyệt (~2,39MB), cùng labels/preprocessing/fixtures/hash/license cần thiết; giữ gói P6 gốc nguyên vẹn.
- Worker dùng canvas adapter64 đã khóa → invert/float32/transpose → `[1,64,64]` → ONNX Runtime Web → logits/softmax/top3.
- WASM một thread làm baseline tương thích; không bắt buộc WebGPU hoặc cross-origin isolation để dùng được chức năng cơ bản.
- Tải model khi cần, tái sử dụng session, không tạo session mỗi nét. Cache theo model version/hash trong IndexedDB khi khả dụng; cache lỗi hoặc thiếu quota phải cho tải lại.
- Giữ kiểm tra hash/shape/finite logits, retry và revision; không đưa dữ liệu thật vào fixture public.

**File dự kiến:** browser registry/worker/client riêng, release web Swin, tests pixel/tensor/logit parity.

**Nghiệm thu:** browser nhận dạng không gọi API; cùng input cho kết quả trong tolerance P6; clear/retry/worker lỗi và mobile hoạt động. Không lấy quan sát10 demo làm accuracy benchmark.

## W3 — Public DINO API cho lần deploy đầu

- Tạo entrypoint production HTTPS, thay cấu hình backend chỉ-local hiện tại; load TorchScript DINO đã duyệt, không dùng ONNX diagnostic làm primary.
- Backend chỉ load model cần cho DINO; Swin đã chạy browser, tránh giữ thêm session CPU không cần thiết. Vẫn kiểm tra bundle/inventory/hash và giữ license/provenance.
- Một process/model instance, giới hạn một inference đồng thời, request/body limits, timeout/retry hợp lý và health check. Không lưu ảnh/nét mặc định.
- CORS chỉ cho origin `https://namphwong.github.io` và các origin local cần kiểm thử; origin không chứa `/DUST/`. Kiểm tra HTTPS/mixed content; không dùng wildcard tùy tiện.
- Frontend hiển thị trạng thái khởi động/lỗi backend thật. Khi backend không sẵn sàng, giữ nét và cho retry; không trả dự đoán giả hoặc chuyển sang Swin mà không thông báo model.
- Đóng gói Docker CPU, loại môi trường/data khỏi build context, kiểm tra Linux với giới hạn512MB. Nếu không đạt, chưa công bố Render Free là phương án chạy được; dùng bản Swin riêng hoặc tìm backend khác theo tài nguyên thực.
- Công khai model bundle đã duyệt kèm checksum qua release/model storage để build từ checkout sạch; không đưa token vào frontend.

**Nghiệm thu:** Pages gọi được DINO thật; hash/model ID/preprocessing khớp; cold start, memory và lỗi429/timeout được đo lại trên host. Không lấy p95 Windows local làm số đo cloud.

## W4 — Kiểm chứng DINO chạy hoàn toàn trên browser

- Đánh giá ONNX export/browser runtime từ checkpoint đã chọn. Bản ONNX hiện có fail strict per-logit tolerance ở2/1.000 mẫu native, nên chưa được duyệt cho runtime public.
- Debug bằng procedural/validation; giữ tolerance P6, không nới ngưỡng để “pass”. Export/graph/runtime mới có version và provenance rõ.
- Port preprocessing gray64→RGB224 đúng Pillow bilinear/ImageNet contract; không mặc định cho rằng `canvas.drawImage` giống Pillow từng byte.
- Kiểm tra tensor/logits/argmax và numerical parity trên tập kiểm chứng đã khóa. Không train/tune hoặc đổi checkpoint theo final test.
- WASM là baseline; thử WebGPU sau khi có parity riêng và fallback hợp lệ. DINO ~86,6MB ONNX cần đo download/init/peak memory/latency trên thiết bị mục tiêu; model bytes không bằng runtime RAM.
- Chỉ tải DINO khi cần, có tiến trình tải/cache theo hash. Quantization chỉ là nhánh tối ưu mới, phải đánh giá riêng; không ép INT8 ngay trước khi có baseline FP32 đạt.

**Gate:** chỉ chuyển DINO mặc định sang browser khi đạt đúng/supported/đủ tài nguyên. Nếu không đạt, giữ kiến trúc W3 và báo giới hạn; không hứa trước một bản Pages-only đầy đủ.

## W5 — Nghiệm thu bản public

- Desktop Chrome/Edge và các browser mobile mục tiêu; cảm ứng vật lý cần thử riêng, không gọi giả lập là thiết bị thật.
- Mạng chậm, tải model lỗi, cache hết quota, worker reset, API ngủ/timeout, clear/undo/sample và late response.
- Kiểm tra `/DUST/`/hash deep links, privacy text đúng từng model, không console error/404 hoặc localhost request.
- So design với baseline; kiểm tra canvas preprocessing/parity độc lập với chất lượng classifier.
- Chuẩn bị đánh giá canvas người viết mới độc lập. Giữ cảnh báo DINO đúng5/10 demo PenDigits và37 nhóm tương đồng qua split; deploy không tự giải quyết vấn đề chất lượng.

**Bàn giao:** báo cáo browser/device/model version/latency/memory/download thực đo; danh sách phần NOT RUN.

## W6 — GitHub Actions, Pages và gói nộp

- Thêm `.github/workflows/pages.yml`: checkout → Node đúng version → `npm ci` trong `web` → tests/build Pages → upload `web/dist` → deploy Pages; quyền tối thiểu cho Pages.
- Repository → Settings → Pages → Source: GitHub Actions. Dùng repo DUST hiện tại, không cần tạo repo mới để có URL project Pages.
- Khởi chạy workflow, chờ deploy thành công, thử URL bằng máy/browser khác. Chỉ ghi URL “đang chạy” vào README khi đã kiểm tra thực.
- README có GitHub/code design/Kaggle/link model/checksum/link web, cách tái tạo và model nào chạy tại đâu. Có mốc rollback về deployment đã kiểm tra.
- Kiểm soát kích thước và băng thông; Pages có giới hạn site1GB và soft bandwidth100GB/tháng. Không diễn đạt “mọi người dùng” là không giới hạn tài nguyên/lượt truy cập.

## Thứ tự và lựa chọn cuối

`W0 → W1 → W2 → W3 → W5 → W6` là đường public đầy đủ lần đầu, giữ DINO mặc định. W4 là nhánh loại backend sau khi numerical/browser gate đạt; chạy W5/W6 lại khi thay runtime.

Nếu ưu tiên tuyệt đối “chỉ github.io, không backend”, làm `W0 → W1 → W2 → W5 → W6` cho **bản Swin riêng**; bản DINO public chờ W4. Hai lựa chọn phải được ghi rõ trước khi triển khai, không thay đổi ý nghĩa model mặc định ngầm.

## Tài liệu nền tảng đã đối chiếu

- [GitHub Pages là static hosting và URL project site](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).
- [Vite: base path và GitHub Actions cho Pages](https://vite.dev/guide/static-deploy.html#github-pages).
- [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) và [deployment/cache](https://onnxruntime.ai/docs/tutorials/web/deploy.html).
- [Render Free: sleep/quota](https://render.com/docs/free), [CPU/RAM](https://render.com/docs/compute-plans).
