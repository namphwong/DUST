# DrawAI — Tiến độ

## Kế hoạch hai mô hình (cập nhật 19/10/2026)

| Phase | Trạng thái | Bằng chứng |
|---|---|---|
| P0 — kiểm kê, phục hồi, baseline | DONE local | [Báo cáo P0](phase0-report.md): 551 file phục hồi khớp hash, build, 32 Python tests, 9 browser tests và 10 demo PyTorch/ONNX/WASM đạt |
| P1 — chỉ giữ DrawAI | DONE local | [Báo cáo P1](phase1-report.md): 309 file lưu ngoài project; 34 Python + 59 browser tests; design pixel-identical; export 3.498 mẫu không khác dự đoán |
| P2 — DINOv3/raster/feasibility | DONE local | [Báo cáo P2](phase2-report.md): pretrained timm thật; 46 Python + 21 browser tests; 19 fixture parity; backend CPU đạt budget; design giữ nguyên |
| P3 — chuẩn bị dữ liệu ảnh Paint | DONE local · protocol ảnh | [P3 hiện tại](phase3-report.md):7.000 ảnh/35 mã người theo xác nhận chủ dữ liệu;250 pilot QC+750 reserved+6.000 main;78 Python+12 browser tests; input Swin ảnh mới, giữ design/release cũ |
| P4 — review/khóa ảnh | DONE local, có giới hạn tương đồng | [Báo cáo P4](phase4-report.md): khóa6.000 ảnh,95 Python+12 browser tests; giữ534 cờ tương đồng,37 nhóm qua split main; chưa chứng minh độc lập hình dạng |
| P5 — train/so sánh ảnh | DONE local CPU | [P5](phase5-report.md): sáu run/ba seed,113 Python+12 browser tests; checkpoint chọn Swin43 test84,40%, DINO42 test98,80%; selection bằng validation, giữ giới hạn tương đồng |
| P6 — export/runtime | DONE local, mixed formats | [P6](phase6-report.md): Swin ONNX + DINO TorchScript đạt1.000 test parity,128 Python+36 browser tests. DINO ONNX diagnostic chưa đạt strict logits ở2 mẫu; không coi là hai ONNX approved |
| P7 — tích hợp/UX | DONE local kỹ thuật; nghiệm thu người thật còn chờ | [P7](phase7-report.md): DINO mặc định, Swin đối chứng, cùng ảnh gray64;133 Python+22 browser tests; giữ design. DINO đúng5/10 demo PenDigits; chưa có benchmark canvas người mới |
| P8 — gói nộp | TODO | Clean checkout, phân phối weights lớn, dataset card/license và gói nộp; NPZ nguồn đã public Kaggle |

P0–P8 là kế hoạch mới Swin 1D/DINOv3, khác các phase D0–D7 lịch sử bên dưới.

## Lịch sử DrawAI PenDigits

Cập nhật 11/10/2026. **D0–D5 đã hoàn thành local; D6 đạt kiểm chứng kỹ thuật local, còn chờ người/thiết bị thật; D7 chưa triển khai.** D3 chạy CPU, Kaggle GPU chưa chạy.

| Phase | Trạng thái | Bằng chứng / phần tiếp theo |
|---|---|---|
| D0 | Hoàn thành, đã rà soát lại đặc tả/schema | scope.md, data-contract.md, architecture.md, strokes.schema.json, environment.md, acceptance-criteria.md; d0-report.md và artifacts/drawai/d0/verification.json |
| D1 | Hoàn thành, đã dựng lại và tăng kiểm toán | 17 unit tests; 10.992 mẫu; train/val/test 5.999/1.495/3.498; L64/L128; 21.984 tensor rebuild khớp; archive/provenance/metadata/fixture được đối chiếu. Writer ID chưa xác minh, val không bảo đảm writer-disjoint. data-report.md và data/d1-review.json |
| D2 | Hoàn thành, đã chạy lại sau D1 | 20 fixture + 14 input lỗi mỗi L64/L128; 4 model unit tests, 6 browser tests và build đạt; gradient, tiny train, ONNX/WASM parity, hash/metadata/retry và benchmark có bằng chứng. feasibility-report.md và feasibility/d2-review.json |
| D3 | Hoàn thành local CPU | 8 pilot + 6 official run (seeds 42/43/44); 7 tests và notebook CPU smoke đạt; kiểm toán 20.988 dự đoán test. Swin L128/augmentation/seed44 chọn bằng validation: accuracy 98,77%, macro-F1 98,79%, đạt mục tiêu. experiment-report.md và d3-local-v1/d3-review.json. Kaggle GPU chưa chạy. |
| D4 | Hoàn thành local CPU/Chromium | Release drawai-swin-v1 trained=true, ONNX 2,27 MiB; 3.498 test khớp argmax/metric, 20 fixture WASM và 13 browser tests đạt; 10 demo train có schema/provenance. artifact-report.md, model-card.md và export/d4-review.json. |
| D5 | Hoàn thành local Chromium | /drawai: canvas, undo/clear, 10 demo, export JSON, inference worker/session thật, top-3 và revision. Build + 9 browser tests đạt; 10 demo logits khớp D4, schema/softmax được kiểm toán. Desktop/mobile viewport và DPR2 đã kiểm tra; ui-report.md và ui/d5-review.json. |
| D6 | Kỹ thuật local đạt; nghiệm thu người thật còn chờ | 34 tests Chrome/Edge, 17 JSON parity + 7 lỗi mỗi browser, benchmark 5 warm-up/100 inference và 4 canvas-gate tests đạt. p95 Chrome 3,90 ms / Edge 3,60 ms, ONNX 2,27 MiB. Protocol/template/evaluator đã có; 0 mẫu người thật, cảm ứng thật chưa thử. acceptance-report.md và verification/d6-review.json; phase_complete=false. |
| D7 | Chưa bắt đầu | Deploy và bàn giao |

Trang `/drawai/check` chỉ là kiểm chứng D2. Tiny train đạt 10/10 không phải metric classifier/test. Không lấy kết quả HAR hoặc logits untrained để nhận dạng chữ số.

Bước còn lại của **D6**: thử thiết bị cảm ứng/bút thật, thu ít nhất 250 nét từ 5 người và đánh giá bằng model đã khóa theo [protocol](canvas-protocol.md). Theo lựa chọn hiện tại, hoàn tất phần local trước, giữ phần người/thiết bị là NOT RUN. [Báo cáo nghiệm thu](acceptance-report.md) lưu kết quả/giới hạn. Có thể chuẩn bị D7 như demo nghiên cứu, chưa gọi D6 hoàn thành toàn bộ.
