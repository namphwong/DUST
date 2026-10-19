# Rà soát, dọn repo và tổng hợp công việc DrawAI

Ngày19/10/2026; mốc trước rà soát `8673f72`. Đã rà565 file được Git theo dõi ở mốc ban đầu, gồm code, tài liệu, model metadata và bằng chứng. Không còn file code MotionSense/HAR/video-fusion trong inventory. Không train lại model hoặc mở lại private test trong lần rà soát này.

## Những phần đã hoàn thành

| Giai đoạn | Kết quả đã có |
|---|---|
| P0 | Inventory, mốc phục hồi, hash dữ liệu/model và ảnh design ban đầu |
| P1 | Chỉ giữ DrawAI; MotionSense/HAR/video-fusion đã lưu ngoài project; entrypoint DrawAI |
| P2 | DINOv3 pretrained công khai từ timm, strict-load, kiểm tra kiến trúc/gradient/raster và khả năng chạy CPU |
| P3 | Chuẩn bị7.000 ảnh64×64; provenance theo xác nhận chủ dữ liệu; không tạo trajectory giả; nguồn NPZ đã public Kaggle |
| P4 | Khóa split theo person_id: train4.000/val1.000/test1.000; pilot/reserved1.000 tách riêng; giữ cảnh báo37 nhóm tương đồng qua split main |
| P5 | Train Swin1D ảnh và DINOv3 backbone/head, ba seed mỗi họ; chọn checkpoint bằng validation trước final test |
| P6 | Swin ONNX và DINO TorchScript đã export/verify; filenames rõ; DINO ONNX diagnostic chưa đạt strict tolerance, không dùng làm runtime |
| P7 | DINO mặc định, Swin đối chứng ở trang riêng; cùng ảnh canvas64; giữ design; retry/undo/clear/top3/export JSON và kiểm tra revision |

Kết quả test Paint đã khóa ở P5: **Swin seed43 accuracy84,40%, macro-F184,28%; DINO seed42 accuracy98,80%, macro-F198,80%**. Đây là số đo lịch sử có giới hạn tương đồng dữ liệu, không phải accuracy canvas người viết mới.

## Các thay đổi trong lần dọn repo

1. **Sửa lỗi biên ở inference client/worker.** Response `null` hoặc logits/scores dạng object không còn gây exception hoặc đợi timeout. Gửi `postMessage` lỗi sẽ reset phiên, cho retry ngay. Worker nhận request `null` trả lỗi contract thay vì crash.
2. **Dừng tác vụ so sánh đã cũ.** Khi clear/đổi nét trong lúc model đầu chạy, không tiếp tục chạy model thứ hai cho snapshot cũ. Không cập nhật state sau unmount. Khởi tạo các client một lần thay vì tạo object mới mỗi render.
3. **Bỏ import không dùng.** Python canvas reference không kéo theo app runtime/ONNX khi chỉ cần renderer.
4. **Dọn tài liệu.** README yêu cầu rõ bundle P6 đầy đủ; guide export không còn nói P7 chưa tích hợp; DINO guide/notebook ghi rõ phạm vi P2/PenDigits. Thêm mục lục tài liệu và README cho bằng chứng lịch sử.
5. **Tách test output.** `npm run test:review` ghi riêng vào `artifacts/drawai/repo-review/`. Các snapshot/metrics P0–P7 không bị ghi đè. Test hỗ trợ `DRAWAI_EVIDENCE_DIR` để chạy đợt kiểm tra riêng.
6. **Dọn cache an toàn.** Xóa5 thư mục cache/generated test output, đợt đầu60 file/1.013.423byte (~0,97MiB). Có script allowlist, kiểm tra đường dẫn nằm trong workspace, không có reparse point hoặc file tracked trước khi xóa. Bổ sung ignore Playwright reports/blob reports/TypeScript build info.
7. **Thêm audit tái chạy được.** `tools/audit_repo.py` kiểm kê Git, parse Python/JSON, kiểm tra link Markdown nội bộ, nhóm trùng hash; `tools/verify_repo_review.py` kiểm tra evidence/tests/bundle/design hiện hành.

Không xóa model, checkpoint, NPZ, môi trường Python, node_modules hoặc các bản sao nằm trong manifest/hash của verifier. Alias Swin PenDigits và fixture/report trùng bytes có vai trò đối chiếu; xóa chúng sẽ làm sai inventory/khả năng truy nguyên. Các bản tài liệu snapshot giữ nguyên theo vị trí gốc; link tương đối không mở từ thư mục snapshot được liệt kê riêng, không báo nhầm là link tài liệu hiện hành.

## Kiểm tra đã thực hiện

| Kiểm tra | Kết quả |
|---|---|
| Inventory sau cập nhật |601 file tracked;71 Python/271 JSON/66 Markdown được kiểm tra; số tăng gồm tài liệu/công cụ/bằng chứng của lần rà soát |
| Python toàn bộ DrawAI | **133/133 đạt** |
| Browser review | **33/33 đạt**, không skip/flaky:10 P7+3 route+9 legacy+5 collection+6 regression mới |
| TypeScript/Vite build | Đạt |
| npm audit |0 advisory ở thời điểm kiểm tra; không cập nhật dependency không cần thiết |
| Model bundle | Inventory và toàn bộ SHA-256 P6 khớp anchor đã khóa; model/preprocessing không thay |
| Design | CSS/canvas khớp mốc P0; empty desktop/mobile khớp pixel legacy; không thêm selector trang chính |
| Docs/source | Python/JSON parse đạt;0 link nội bộ hỏng trong tài liệu hiện hành; snapshot link và log local được phân loại riêng |

Các JSON kết quả, screenshots và inventory nằm tại [repo-review](../../artifacts/drawai/repo-review/verification.json). Audit bao gồm toàn bộ file tracked; không duyệt/xóa dữ liệu riêng hoặc môi trường dependency theo tiêu chí “file thừa”. Một audit/test pass không phải chứng minh không còn mọi lỗi có thể có.

## Phần còn chờ, không đánh dấu đã hoàn thành

- **P8:** phân phối bundle weights lớn kèm license/checksum, kiểm tra clean checkout/máy khác, hoàn thiện dataset card/license/link và gói nộp. GitHub hiện có code/metadata; clone chưa có đủ weights DINO để chạy default.
- **Chất lượng canvas:** DINO đúng5/10 demo PenDigits hiện có; khác miền Paint. Không chỉnh adapter/mẫu/checkpoint theo kết quả này. Chưa có benchmark canvas người viết mới.
- **Thiết bị thực:** chưa nghiệm thu cảm ứng/bút vật lý; các browser tests là giả lập kỹ thuật.
- **Dữ liệu:**37 nhóm tương đồng qua split vẫn chưa giải quyết; nguồn/người/đồng ý theo xác nhận chủ dữ liệu, không có xác minh danh tính độc lập.
- **Triển khai:** chưa production/static hosting có backend. `release_ready=false` tiếp tục được giữ.

## Lệnh kiểm tra/dọn thường dùng

```powershell
./.venv-dinov3/Scripts/python.exe -B tools/audit_repo.py
powershell -ExecutionPolicy Bypass -File tools/clean_repo_cache.ps1
# Thêm -Apply để thực sự xóa các cache trong allowlist.
cd web
npm run build
npm run test:review
```

Chạy ứng dụng theo [paint-app-guide](paint-app-guide.md); trạng thái phase theo [status](status.md). Báo cáo phase cũ vẫn giữ nguyên số test/metrics tại thời điểm hoàn thành phase, không bị thay bằng số liệu của đợt rà soát.
