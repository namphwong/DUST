# Kế hoạch DrawAI: Vercel frontend + Render backend

Ngày28/10/2026. **Đây là hướng triển khai ưu tiên mới**, thay kế hoạch GitHub Pages làm hướng chính. Chưa thay code runtime hoặc deploy tài khoản nào trong bước lập kế hoạch.

**Cập nhật31/10/2026:** VR0 giữ snapshot; VR1 bundle/Swin release public, VR2 container API và VR3 frontend integration đã thực hiện. VR4/VR5 cloud còn chờ đăng nhập/kết nối tài khoản; VR6 bàn giao một phần. [Trạng thái hiện tại](status.md), [hướng dẫn deploy](public-deploy-guide.md), [đối chiếu6 yêu cầu](submission-audit.md). Phần dưới giữ nội dung kế hoạch ban đầu.

## 1. Kiến trúc được chọn

```text
GitHub namphwong/DUST
  ├─ Vercel Hobby: React/Vite + static assets + Swin ONNX/WASM worker
  ├─ Render Free: Python API + DINOv3 TorchScript CPU
  └─ GitHub Releases: bundle model đã khóa + checksum + license

Người dùng → https://<frontend>.vercel.app
  ├─ DINO mặc định → HTTPS API Render → logits → top3 UI
  └─ Trang so sánh → Swin chạy trên máy người dùng + DINO qua API
```

Giữ hai checkpoint P5, input contract/canvas adapter64 và design DrawAI. Không train lại hoặc đổi model theo demo/test. Không dùng DINO ONNX diagnostic làm runtime. Swin là model **ảnh** `[1,64,64]`, không nhầm release PenDigits `[1,3,128]`.

### Tại sao chọn cách này

- Vercel phù hợp build/host frontend Vite từ GitHub, có HTTPS và preview deployments. Dùng Hobby cho đồ án cá nhân phi thương mại, trong hạn mức gói.
- Render có Free Web Service chạy Python/Docker. Dùng một backend chỉ DINO để giảm RAM so với runtime local đang giữ cả hai model.
- Swin ONNX nhỏ (~2,39MB), đã có kiểm chứng WASM P6; chuyển suy luận Swin ra browser giúp giảm request/tài nguyên backend. Cần nối đầy đủ preprocessing/registry web, không coi P6 smoke là nghiệm thu public hoàn tất.
- Không đưa inference DINO vào Vercel Functions trong phương án này; frontend gọi Render trực tiếp bằng URL HTTPS, không thêm lớp proxy phụ thuộc timeout của Vercel.
- Giữ layout/màu/font/canvas/các nút. Chỉ sửa câu chữ trạng thái/quyền riêng tư cho đúng: DINO gửi ảnh64 tới backend, Swin chạy trong browser.

## 2. Điều kiện miễn phí và giới hạn thực tế

- Vercel Hobby là gói miễn phí dành cho cá nhân/phi thương mại. Mọi triển khai vẫn chịu hạn mức nền tảng; không hiểu “public” là phục vụ lượng người không giới hạn.
- Render Free hiện có512MB RAM,0,1 CPU; workspace có750 giờ Free mỗi tháng. Service ngủ khi15 phút không có truy cập; đánh thức thường khoảng một phút, có thể khác theo tình trạng host.
- Backend local hai model đã đo khoảng400MiB RSS. Số này trên Windows/CPU local **không chứng minh** Linux container Render đủ RAM hoặc có cùng tốc độ.
- Điều kiện quyết định dùng Render: container DINO-only phải qua kiểm tra512MB, cold start/inference/repeated requests không OOM và có dư địa tài nguyên. Nếu không đạt, ghi rõ backend free chưa đủ; không tự nâng gói trả phí hay tuyên bố đã deploy thành công.
- Không dùng cron/uptime bot để cố giữ service free luôn thức. Thiết kế UX chấp nhận sleep/cold start.

## VR0 — Khóa phạm vi và nghiệm thu

**Việc làm**

1. Ghi version/hash của checkpoint, selection P5, bundle P6, canvas adapter và preprocessing.
2. Khóa DINO mặc định, Swin đối chứng, design giữ nguyên; không fallback model âm thầm.
3. Chốt route public, browser/thiết bị mục tiêu, tolerances P6, ngân sách RAM/latency và checklist public.
4. Tách build local/research khỏi cấu hình public; giữ bằng chứng P0–P7, không sửa metadata gốc để che thay đổi deployment.

**Bàn giao:** public deployment contract và checklist có điều kiện PASS/NOT RUN rõ.

## VR1 — Đóng gói và phân phối model

1. Tạo Release có tag cố định cho bundle P6 đầy đủ và SHA-256 ZIP; giữ license/provenance. Không nhét NPZ/checkpoint train/venv/token vào gói.
2. Chuẩn bị gói Swin web riêng gồm ONNX ảnh, labels, preprocessing và manifest/hash/version. Gói này là derivative web release có provenance, không thay bundle gốc.
3. Backend build tải ZIP từ release cố định, kiểm tra checksum rồi kiểm tra bundle anchor/inventory. Đưa artifact vào image build để tránh tải lại weights trong mỗi request hoặc mỗi lần cold start.
4. Frontend chỉ phục vụ Swin/assets cần cho browser; DINO TorchScript không bị tải vào trình duyệt.

**Bàn giao:** link model public thực, checksum, manifest và script chuẩn bị model từ checkout sạch.

## VR2 — Backend DINO-only production, kiểm tra RAM trước

1. Tạo entrypoint API production riêng, tái dùng schema/preprocessing/strict hash của P6/P7; không đưa HTTP server chỉ-local hiện tại lên internet nguyên trạng.
2. Load DINO TorchScript một lần, CPU-only, một process/worker; dùng `inference_mode` và `optimized_execution(False)` như artifact đã kiểm chứng. Không import/load Swin ORT session hoặc factory training nếu không cần.
3. Thread budget phù hợp CPU nhỏ; bắt đầu một thread và đo lại. Giới hạn một inference đồng thời, trả429 rõ khi bận; không tạo queue không giới hạn.
4. Có `/health` hoặc `/api/drawai/health` báo readiness/version/hash và `/predict` hoặc `/api/drawai/predict` theo contract public đã khóa. Chốt một bộ endpoint, không để frontend gọi nhầm đường local.
5. Bind `0.0.0.0`, nhận `PORT`; body/pixel limits, duplicate/nonfinite JSON, model ID, error code và request/revision validation giữ chặt.
6. CORS allowlist các origin frontend thực; không cho toàn bộ `*.vercel.app`. Không coi CORS là authentication/rate limiting. Dùng giới hạn request/concurrency riêng; không đặt token quản trị trong client.
7. Docker CPU, `.dockerignore` loại data/cache/venv/node_modules/weights không phục vụ deploy. Chạy Linux container giới hạn512MB; đo peak/cold/warm/repeated request, verify logits/preprocessing với artifact gốc.

**Gate:** không OOM ở giới hạn host, kết quả đúng tolerance, request/error/retry hoạt động. Không vượt gate thì chưa chuyển sang deploy Render Free.

## VR3 — Frontend Vercel + Swin browser

1. Giữ Vite `base: '/'`; không dùng `/DUST/` của project Pages. Thêm SPA rewrites cho route ứng dụng; asset/model/WASM không bị rewrite nhầm thành HTML.
2. Thêm `VITE_API_BASE_URL` cho URL Render public. URL này là thông tin công khai; không chứa secret. Bản public không trỏ127.0.0.1 và không phụ thuộc Vite dev proxy.
3. Swin worker: canvas adapter64 → invert/float32/transpose → `[1,64,64]` → WASM → logits/softmax. Giữ hash/shape/metadata/revision validation, một session dùng lại, baseline một WASM thread.
4. Lazy-load model, cache theo version/hash khi khả dụng; cache hỏng/hết quota phải cho tải lại. Lazy-load các trang kỹ thuật không cần ở trang chính.
5. DINO client gọi API HTTPS trực tiếp. Khi người dùng yêu cầu nhận dạng, kiểm tra readiness/đánh thức backend với retry hữu hạn; hiển thị trạng thái trong vùng status hiện có. Timeout cold start tách khỏi timeout inference; tránh tự gửi lặp inference không kiểm soát.
6. Clear/đổi nét/unmount hủy chờ hoặc loại kết quả cũ; khi sleep/offline/429 vẫn giữ nét và cho retry. Không đưa dự đoán giả hoặc đổi sang Swin ngầm.
7. Nội dung hiển thị đúng model và việc gửi ảnh64 ra server. Không ghi “chạy hoàn toàn trên thiết bị” khi DINO đang chạy Render. Không lưu ảnh trên backend mặc định.

**Bàn giao:** frontend build được, browser Swin không cần API, DINO gọi được backend staging, design không đổi cấu trúc.

## VR4 — Thiết lập Vercel/Render và nối hai bên

### Vercel

- New Project → import `namphwong/DUST`.
- Plan: Hobby; framework Vite; Root Directory `web`; Install `npm ci`; Build `npm run build`; Output `dist`.
- Nhận hostname production thực, cấu hình API URL cho đúng Production environment. Đổi build-time env phải redeploy.
- Preview deployment chỉ được gọi backend nếu origin đó được allowlist rõ, hoặc dùng backend staging riêng; không mở wildcard vì preview.

### Render

- New Web Service → cùng repo GitHub, nhánh deployment đã chốt; Runtime Docker; Dockerfile theo đường dẫn đã tạo.
- Instance Type **Free**; bind port nền tảng; health check theo endpoint VR2.
- Env: origin Vercel production chính xác, bundle URL/tag/hash nếu không bake bằng build args; secrets chỉ đặt server-side.
- Không dùng nhiều worker, không load cả hai model, không bật database/disk trả phí không cần thiết.

### Nối và thử

- CORS bao gồm scheme+hostname frontend, không bao gồm path `/drawai`. Test preflight `OPTIONS`, POST JSON, HTTPS/mixed content và response headers trên URL thực.
- Frontend gọi thẳng Render; không cấu hình SPA rewrite nuốt `/predict` hay biến API error thành `index.html`.
- Giữ main deployment cuối đã test làm mốc rollback. Trong khi một nhánh chưa đủ cấu hình, không bật auto-deploy để public bản lỗi.

**Vai trò người dùng:** tạo/đăng nhập tài khoản Vercel và Render, kết nối repo, chọn gói Free/Hobby và cung cấp hostname thực. Không cần gửi token/mật khẩu qua chat. Các file, container và smoke test phải được chuẩn bị trước bước tài khoản.

## VR5 — Nghiệm thu link public và chi phí

1. Thử root/deep links, canvas/undo/clear/sample/top3/export JSON, so sánh cùng ảnh qua hai model và lỗi/retry.
2. Thử desktop/mobile, browser cache hết quota, mạng chậm, backend ngủ rồi thức,429/timeout, clear khi đang chờ. Cảm ứng vật lý ghi riêng, không gọi giả lập là thiết bị thật.
3. Xác minh model ID/hash/version, tensor/logit parity và design baseline; không dùng test/demos để đổi model hoặc tune adapter.
4. Đo download/init/warm latency/cold latency/peak memory trên môi trường thực; không lấy p95 Windows local làm cloud benchmark.
5. Kiểm tra service không mất weights khi restart, không ghi ảnh/PII, không lộ token; xem usage dashboard cả hai nền tảng và giữ trong free limits.
6. Báo đúng chất lượng: DINO đúng5/10 demo PenDigits;37 nhóm tương đồng qua split; chưa có benchmark canvas người viết mới. Deploy là khả năng truy cập, không tự nâng chất lượng classifier.

**Gate:** link hoạt động từ máy khác khi máy local tắt; cả hai model đúng runtime; phần NOT RUN được ghi rõ. Không hứa phục vụ số người đồng thời không giới hạn.

## VR6 — README, gói nộp và bàn giao

- README nêu link frontend/backend health/code/design/Kaggle/model release/checksum, môi trường, giới hạn free tier, rollback và cách tái tạo từ checkout sạch.
- Thêm hình kiến trúc và bảng “model chạy ở đâu”; giữ bằng chứng cũ, ghi test/deployment mới vào thư mục riêng.
- Công bố đây là demo nghiên cứu public nếu chưa nghiệm thu canvas thật; không tự đổi `release_ready` của gói P6.
- Bàn giao URL đã kiểm tra, tag/code revision, checklist và các giới hạn còn lại. Các hostname minh họa trong kế hoạch không được coi là link đã deploy.

## Thứ tự thực hiện

`VR0 → VR1 → VR2 → VR3 → VR4 → VR5 → VR6`.

VR2 là bước quyết định ngân sách: phải chứng minh DINO-only chạy ổn dưới512MB trước khi chọn Render Free làm host cuối cùng. DINO browser/quantization/đổi kiến trúc không nằm trên đường triển khai đầu tiên; chỉ làm nhánh tối ưu sau này và phải kiểm chứng riêng.

## Nguồn đối chiếu ngày28/10/2026

- [Vercel Hobby](https://vercel.com/docs/plans/hobby), [Vite trên Vercel](https://vercel.com/docs/frameworks/frontend/vite).
- [Render Free: sleep/quota](https://render.com/docs/free), [Compute plans](https://render.com/docs/compute-plans), [Web service host/port](https://render.com/docs/web-services).
