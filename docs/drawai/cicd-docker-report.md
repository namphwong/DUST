# CI/CD và Docker AI - thực thi 31/10/2026

CI và continuous delivery đã chạy thật trên GitHub Actions. Image DINOv3 đã public trên Docker Hub, được kéo ẩn danh và kiểm tra inference trên runner mới. **Render chưa triển khai**: chưa có thông tin xác thực/hook; không có URL backend cloud đã nghiệm thu. Frontend/design và checkpoint P5/P6 được giữ nguyên.

## Kết quả và bằng chứng

| Hạng mục | Kết quả / link |
|---|---|
| CI code hiện hành | [41164031905 - SUCCESS](https://github.com/namphwong/DUST/actions/runs/41164031905): 41 Python tests, build frontend, build image, parity và resource gate |
| CI revision trong image | [41163484236 - SUCCESS](https://github.com/namphwong/DUST/actions/runs/41163484236), revision `26011c9dd74f26fc83879509cf7e497b40b9e340` |
| Phát hành image | [41163501943 - SUCCESS](https://github.com/namphwong/DUST/actions/runs/41163501943); job Render SKIPPED |
| Pull trên runner mới không login | [41164033474 - SUCCESS](https://github.com/namphwong/DUST/actions/runs/41164033474), cùng digest và revision image |
| Docker Hub | [namphwong/drawai-dinov3-api](https://hub.docker.com/r/namphwong/drawai-dinov3-api), public, tag `v1.0.0` |
| Model download | [GitHub Release](https://github.com/namphwong/DUST/releases/tag/drawai-models-paint35-v1), ZIP 162.990.474 bytes và checksum |

Digest phát hành bất biến:

```text
namphwong/drawai-dinov3-api@sha256:a4b131893b49a200de1440831806d3d80cd2eed414712ac620001e9c0d30fe52
```

Tag source: `sha-26011c9dd74f26fc83879509cf7e497b40b9e340`. Workflow bổ sung bảo vệ tag và kiểm tra runner mới nằm ở commit `c8cea8f8c1bbeca9590cfad8655856a6d25debd8`; không đánh đồng revision workflow với revision image đã phát hành.

## Từng bước tái thực hiện

1. Clone repo; đọc `deployment/model-release.json`. Script `tools/prepare_public_models.py fetch` tải ZIP public, kiểm tra bytes/SHA256, giải nén an toàn và xác minh full bundle trước khi dùng. Không cần raw NPZ hoặc checkpoint training trong image.
2. Push code/PR: `.github/workflows/ci.yml` chạy 41 tests được chọn, `npm ci`/build, tải bundle verified, build Docker Linux amd64 và chạy API thật với `--memory=512m --cpus=0.1`.
3. Gate AI: so logits/argmax trên 8 fixture P6 công khai, không dùng private test; 5 warmup + 30 request tính p95. Gate p95 <=5 giây, readiness <=90 giây, peak <=450.000.000 bytes, OOM/OOM kill bằng 0. Đây là parity/resource smoke test, không phải benchmark accuracy.
4. GitHub Variable `DOCKERHUB_USERNAME=namphwong`; Secret `DOCKERHUB_TOKEN` đã cấu hình. Mở Actions > Publish DrawAI AI image > Run workflow, chọn main và **version mới** (ví dụ `v1.0.1`). Không ghi đè `v1.0.0`. Mặc định `deploy_render=false`.
5. Publish workflow gọi lại CI, build/test, login qua stdin, push tag version và source SHA, ghi digest. Job `public_check` trên runner mới kéo bằng digest với Docker auth config trống và chạy lại inference/resource gates. Một lần kiểm tra độc lập cho v1.0.0 đã PASS ở run 41164033474.
6. Chỉ bật Render khi đã cấu hình backend: Secret `RENDER_DEPLOY_HOOK_URL`, Variable `RENDER_API_URL`, origin frontend trong `ALLOWED_ORIGINS`. Workflow truyền `imgURL` theo digest, đợi đúng revision và kiểm tra inference. Hiện **chưa chạy** bước này. `RENDER_API_KEY` được yêu cầu để có thể tự tạo service bằng API ở bước tiếp theo; workflow hiện tại chỉ dùng deploy hook, chưa dùng API key.

## Chạy image đã public

Cần Docker Engine đang chạy. Local Windows hiện chưa có Engine hoạt động; bằng chứng build/run phía trên được thực hiện trên GitHub Ubuntu runner.

```powershell
docker pull namphwong/drawai-dinov3-api@sha256:a4b131893b49a200de1440831806d3d80cd2eed414712ac620001e9c0d30fe52
docker run --rm --name drawai-api --memory=512m --cpus=0.1 -p 10000:10000 namphwong/drawai-dinov3-api:v1.0.0
```

Terminal khác:

```powershell
Invoke-RestMethod http://localhost:10000/health
python tools/smoke_public_api.py --url http://localhost:10000 --expected-revision 26011c9dd74f26fc83879509cf7e497b40b9e340 --enforce-container-budget --output artifacts/drawai/cicd/local-smoke.json
```

Smoke script cần numpy/Pillow/jsonschema và fixture metadata P6 trong checkout. `/predict` nhận protocol `drawai-paint-api-v1`, ảnh raw uint8 64x64 qua base64; tham khảo script để tạo request hợp lệ. API không lưu ảnh, một request inference tại một thời điểm; request đồng thời nhận 429.

## Số đo thực tế

| Lần chạy | p95 HTTP | Readiness | Memory peak | OOM |
|---|---:|---:|---:|---:|
| CI 41163484236 | 2.302 ms | 35.036 ms | 426.053.632 bytes | 0 |
| Publish 41163501943 | 1.705 ms | xem JSON | 422.641.664 bytes | 0 |
| Fresh public pull 41164033474 | 2.501 ms | 35.035 ms | 425.177.088 bytes | 0 |

Các số ms trong bảng dùng dấu chấm phân cách hàng nghìn. Số đo readiness từ smoke script đến ready trên GitHub runner, **không phải cold start của Render**. Image compressed size 461.460.145 bytes, khác RAM runtime. JSON gốc tại [artifacts/drawai/cicd](../../artifacts/drawai/cicd/) được commit để giữ bằng chứng sau khi Actions artifact hết hạn.

## Chỗ đã cập nhật

- `.github/workflows/{ci,publish,verify-public}.yml`: CI, phát hành có gate, kiểm chứng public image.
- `training/drawai/public_api.py`: DINO-only API production, verify bundle, validation, CORS, concurrency và health tài nguyên.
- `training/tests/test_drawai_public_api.py`: kiểm thử contract/API.
- `deployment/docker/{Dockerfile,requirements.txt}`, `.dockerignore`: CPU image chạy non-root, allowlist build context, không chứa raw data/secrets.
- `tools/{prepare_public_models,smoke_public_api}.py`: phân phối bundle có checksum và kiểm chứng inference/resource.
- `deployment/model-release.json`: URL public, hash ZIP và bundle; README/status/doc này cập nhật trạng thái thực tế.

Giữ DINO TorchScript làm primary; DINO ONNX vẫn diagnostic do parity cũ chưa đạt. Swin là baseline trong bundle, không load vào API DINO này. Chưa train lại, chưa đọc lại private test, chưa nghiệm thu người viết mới/thiết bị thật; `release_ready=false`, `technical_only=true`. 37 nhóm tương đồng qua split và khác miền Paint/canvas vẫn là giới hạn của dự án.

## Tham chiếu vận hành

[GitHub workflow triggers](https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows), [Docker Hub repositories](https://docs.docker.com/docker-hub/repos/), [Render deploy hooks](https://render.com/docs/deploy-hooks), [Render API key](https://render.com/docs/api). CD hiện có là **continuous delivery tới registry**; continuous deployment lên cloud chưa nghiệm thu.
