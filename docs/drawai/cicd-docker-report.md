# CI/CD và Docker AI - thực thi 31/10/2026

CI, phát hành Docker Hub và continuous deployment lên Render đã chạy thật. Image DINOv3 public được pull ẩn danh và kiểm tra model/resource trên runner mới trước khi deploy. Frontend Vercel và backend Render đã được kiểm thử cloud; design DrawAI và checkpoint P5/P6 giữ nguyên.

## Kết quả và bằng chứng

| Hạng mục | Kết quả / link |
|---|---|
| CI frontend/image hiện hành | [41165630051 - SUCCESS](https://github.com/namphwong/DUST/actions/runs/41165630051):44 Python +7 public browser, image gate |
| Publish v1.0.1 | [41166886046 - SUCCESS](https://github.com/namphwong/DUST/actions/runs/41166886046), fresh anonymous pull và resource gate |
| CD Render | [41183564442 - SUCCESS](https://github.com/namphwong/DUST/actions/runs/41183564442), verify public image -> deploy hook -> cloud smoke |
| Public app | [Vercel frontend](https://drawai-pied.vercel.app) · [Render health](https://drawai-dinov3-api.onrender.com/health) |
| Docker Hub | [namphwong/drawai-dinov3-api](https://hub.docker.com/r/namphwong/drawai-dinov3-api), public, tag `v1.0.1` |
| Model download | [GitHub Release](https://github.com/namphwong/DUST/releases/tag/drawai-models-paint35-v1), ZIP162.990.474bytes + Swin ONNX + checksum |

Digest phát hành bất biến:

```text
namphwong/drawai-dinov3-api@sha256:bf43b3f739ca7ea250eaa217d1b0ef922f879745a2f5aa206bc5494922aede4c
```

Source revision: `c9593cac3fe0a23a6f997596edec1e9bd0a0c629`. `v1.0.0`/digest cũ vẫn giữ làm mốc rollback, không bị ghi đè.

## Từng bước đã thực hiện

1. Workflow tải model ZIP từ GitHub Release theo `deployment/model-release.json`, kiểm tra bytes/SHA256, giải nén an toàn và xác minh full bundle; raw NPZ/checkpoint training không vào image.
2. `.github/workflows/ci.yml` chạy unit/API tests, build frontend public, build Docker Linux amd64 và chạy API thật với `--memory=512m --cpus=0.1`.
3. AI gate so logits/argmax trên8 fixture P6 công khai, không dùng private test;5 warmup +30 request. Ngưỡng: p95<=5s, readiness<=90s, peak<=450.000.000bytes, OOM/OOM kill0.
4. `.github/workflows/publish.yml` dùng `DOCKERHUB_USERNAME` Variable và `DOCKERHUB_TOKEN` Secret, build/test/push cùng image; tag version là immutable, không dùng `latest`.
5. Job `public_check` chạy trên runner mới với Docker auth trống, pull image theo digest và chạy lại inference/resource gate. Run41166886046 PASS cho `v1.0.1`.
6. Render Free Singapore được tạo từ public image; `/health` làm health check. `ALLOWED_ORIGINS` chỉ chứa `https://drawai-pied.vercel.app`.
7. `RENDER_DEPLOY_HOOK_URL` lưu trong GitHub Secret, `RENDER_API_URL` lưu trong Variable. Hook từng bị lộ trong ảnh công cụ đã được regenerate; hook cũ vô hiệu và secret GitHub đã cập nhật.
8. `.github/workflows/deploy-render.yml` kiểm tra lại public image rồi truyền `imgURL` theo digest vào hook. Smoke chờ đúng revision tối đa600s, sau đó chạy8 fixture +30 request trên endpoint Render thật. Run41183564442 PASS.
9. Vercel project `drawai`, root `web`, Hobby, Vite; production env `VITE_DRAWAI_PUBLIC=true` và `VITE_API_BASE_URL=https://drawai-dinov3-api.onrender.com`.
10.3 browser tests cloud trực tiếp trên Vercel/Render PASS: DINO inference, Swin-browser/DINO-Render comparison, route/mobile layout.

## Chạy image đã public

```powershell
docker pull namphwong/drawai-dinov3-api@sha256:bf43b3f739ca7ea250eaa217d1b0ef922f879745a2f5aa206bc5494922aede4c
docker run --rm --name drawai-api --memory=512m --cpus=0.1 -p 10000:10000 namphwong/drawai-dinov3-api:v1.0.1
```

Terminal khác:

```powershell
Invoke-RestMethod http://localhost:10000/health
python tools/smoke_public_api.py --url http://localhost:10000 --expected-revision c9593cac3fe0a23a6f997596edec1e9bd0a0c629 --enforce-container-budget --output artifacts/drawai/cicd/local-smoke.json
```

## Số đo thực tế

| Lần chạy | p95 HTTP | Readiness | Memory peak | OOM |
|---|---:|---:|---:|---:|
| CI 41163484236 |2.302ms |35.036ms |426.053.632bytes |0 |
| Fresh public v1.0.0 |2.501ms |35.035ms |425.177.088bytes |0 |
| Fresh public v1.0.1 |1.598ms |26.034ms |422.359.040bytes |0 |
| Render cloud CD41183564442 |1.299,825ms |63.790,040ms |435.429.376bytes |0 |

Ba dòng đầu là runner GitHub; dòng Render là cloud thật. `container_budget_enforced=false` trong cloud JSON vì smoke client không áp cgroup lên dịch vụ ngoài runner, nhưng số đo quan sát vẫn dưới mục tiêu peak450MB/p955s/readiness90s và OOM0. Image v1.0.1 compressed461.462.119bytes, khác RAM runtime. JSON gốc lưu tại [artifacts/drawai/cicd](../../artifacts/drawai/cicd/) và [public-deployment](../../artifacts/drawai/public-deployment/).

## Chỗ đã cập nhật

- `.github/workflows/{ci,publish,verify-public,deploy-render}.yml`: CI, publish, anonymous verification và cloud CD.
- `training/drawai/public_api.py`: DINO-only API, verify bundle, validation, CORS, concurrency, health tài nguyên.
- `deployment/docker/{Dockerfile,requirements.txt}`, `.dockerignore`: CPU image non-root, allowlist context, không chứa raw data/secrets.
- `tools/{prepare_public_models,smoke_public_api}.py`: download/checksum và inference/resource smoke.
- `web/playwright.drawai-cloud.config.ts`, `web/tests/drawai-cloud.spec.ts`: kiểm thử production URL tái lập.
- `artifacts/drawai/public-deployment/`: cloud URLs, CD smoke và browser evidence.

DINO TorchScript vẫn là primary; DINO ONNX diagnostic. Swin chỉ là baseline trong browser, không load vào API DINO. Không train lại, không đọc private test trong deploy. `release_ready=false`, `technical_only=true`;37 nhóm tương đồng qua split và domain gap Paint/canvas vẫn là giới hạn.
