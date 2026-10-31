# Deploy DrawAI: Vercel Hobby + Render Free

Ngày31/10/2026. Deployment cloud đã nghiệm thu kỹ thuật: [frontend Vercel](https://drawai-pied.vercel.app), [backend Render](https://drawai-dinov3-api.onrender.com/health), [CD SUCCESS](https://github.com/namphwong/DUST/actions/runs/41183564442). Phần dưới vừa ghi cấu hình thực tế vừa là hướng dẫn tái tạo/rollback.

## 1. Render: dùng image AI đã public

Đăng nhập [Render](https://dashboard.render.com/). Có thể dùng [Deploy Blueprint của repo](https://render.com/deploy?repo=https://github.com/namphwong/DUST) với `render.yaml`, hoặc New > Web Service > Existing Image:

```text
docker.io/namphwong/drawai-dinov3-api@sha256:a4b131893b49a200de1440831806d3d80cd2eed414712ac620001e9c0d30fe52
```

- Name `drawai-dinov3-api`; region Singapore; **plan Free**. Không tạo database/disk hoặc nâng gói trả phí.
- Port `10000`; Health Check Path `/health`; giữ CMD của image, một worker CPU.
- `ALLOWED_ORIGINS`: origin Vercel thực, ví dụ scheme+hostname không có path hoặc dấu `/` cuối. Không dùng wildcard. Nếu tạo backend trước frontend, tạm đặt origin đã dự định rồi cập nhật thành hostname thực trước nghiệm thu.
- Deployment hiện hành dùng digest `sha256:bf43b3f…aede4c`, revision `c9593cac…c629`. Mở `/health` để kiểm tra model/bundle/model hash/revision; không chỉ dựa vào trạng thái UI Render.

## 2. Vercel: import frontend từ GitHub

Đăng nhập [Vercel](https://vercel.com/new), import `namphwong/DUST` vào tài khoản cá nhân Hobby.

| Thiết lập | Giá trị |
|---|---|
| Root Directory | `web` |
| Framework / Node | Vite / Node24 |
| Install | `npm ci` |
| Build | `npm run build:public` (đã khai báo trong `web/vercel.json`) |
| Output | `dist` |
| Production env | `VITE_DRAWAI_PUBLIC=true` |
| Production env | `VITE_API_BASE_URL=<origin HTTPS Render thực>` |

Không đặt key/token trong biến `VITE_*`. Build từ chối thiếu API URL, URL không phải HTTPS hoặc loopback. Script Node tải asset Swin2,39MB từ release cố định và verify SHA256. DINO weights không tải vào browser.

Sau khi nhận hostname Vercel, cập nhật `ALLOWED_ORIGINS` Render bằng origin đó. Đổi API URL/env build-time phải redeploy Vercel. Public chỉ mở `/`, `/drawai`, `/drawai/compare`; route thu dữ liệu/chẩn đoán/legacy chỉ dùng profile local.

## 3. Kiểm chứng link public

1. Từ trình duyệt/máy khác: root và deep link `/drawai/compare` mở được; assets/model/WASM trả đúng loại nội dung, không bị SPA rewrite thành HTML.
2. DINO mặc định gửi ảnh64 tới API HTTPS; không lưu ảnh. Swin đối chứng chạy WASM trong browser, chỉ DINO gửi request server. Hai model phải nhận cùng hash pixel.
3. Thử vẽ/undo/clear/sample/export, chờ backend ngủ/thức,429, tắt mạng, retry, clear lúc đang chờ. Không đổi model âm thầm. Giữ bố cục/màu/font/canvas hiện tại; chỉ nội dung trạng thái/quyền riêng tư đổi cho đúng môi trường.
4. Chạy `tools/smoke_public_api.py --url <Render origin> --expected-revision <source SHA của image> --bundle models/exported/paint35-v1 --readiness-timeout 600 --output artifacts/drawai/public-deployment/cloud-smoke.json` với dependency numpy/Pillow/jsonschema. Script chờ đúng revision rồi kiểm tra logits trên fixture công khai; không dùng private test.
5. Cloud run41183564442 đo p95 khoảng1,30s, readiness63,79s, peak435.429.376bytes, OOM0. Không dùng bot giữ service luôn thức. Nghiệm thu cảm ứng vật lý/người viết mới còn riêng.

## 4. Nối CD backend

Trong Settings của service Render, lấy Deploy Hook; lưu vào **GitHub Secret** `RENDER_DEPLOY_HOOK_URL`. Lưu origin Render vào **GitHub Variable** `RENDER_API_URL`. Không commit hook.

Actions > **AI CD - publish verified Docker Hub image** > Run workflow trên main, version mới (ví dụ `v1.0.1`) và `deploy_render=true`. CI + image gate + public pull trên runner mới phải PASS trước job Render. Hook truyền image **digest**, không dùng `latest`; smoke đợi đúng source revision tối đa600giây, sau đó kiểm tra inference. Job nhận request deploy không đồng nghĩa đã deploy thành công.

Nếu dùng Blueprint, khi đổi digest bằng CD cũng cập nhật `render.yaml` theo digest đã kiểm chứng để lần sync Blueprint sau không kéo lại image cũ. Rollback cũng dùng digest cũ đã biết; không ghi đè tag version.

Vercel Git integration tự build/deploy frontend khi main thay đổi. Backend phát hành qua workflow có gate và bước kích hoạt thủ công để tránh vô tình đẩy image chưa chọn; workflow `deploy-render.yml` cũng cho phép deploy lại một digest đã kiểm chứng.

`RENDER_API_KEY` chỉ cần khi muốn tự tạo/quản lý service bằng API; không bắt buộc cho cách Deploy Blueprint hoặc Existing Image. Có thể kết nối ứng dụng Vercel/Render trong Codex để tiếp tục triển khai mà không gửi key qua chat.

## 5. Chạy/kiểm tra profile public local

```powershell
cd D:\ChuyenDe\DUST\web
$env:VITE_DRAWAI_PUBLIC='true'
$env:VITE_API_BASE_URL='https://drawai-public-test.invalid'
npm ci
npm run build:public
npx playwright test --config playwright.drawai-public.config.ts
```

Hostname `.invalid` chỉ được intercept trong bài test tới production API local4198. Đây không phải backend đã deploy. Khi chạy ứng dụng với backend thật phải dùng HTTPS origin thật. Xóa hai env khỏi phiên shell để build P7 local mặc định; `.env.local` không commit.

Tài liệu nền tảng: [Vercel config](https://vercel.com/docs/project-configuration), [Render Blueprint](https://render.com/docs/blueprint-spec), [image-backed service](https://render.com/docs/deploying-an-image), [deploy hooks](https://render.com/docs/deploy-hooks).
