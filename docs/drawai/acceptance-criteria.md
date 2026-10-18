# DrawAI — Ma trận nghiệm thu

Các mục nghiên cứu/hiệu năng là mục tiêu; task chỉ hoàn thành theo bằng chứng. Checklist chi tiết: [phases.md](phases.md).

| Nhóm task | Kiểm tra bắt buộc | Bằng chứng |
|---|---|---|
| D0-01–06 | Phạm vi, schema, module, môi trường và tiêu chí rõ | scope.md, data-contract.md, environment.md, tài liệu này |
| D1-01–03 | Đúng hash, parser không bỏ lỗi, đúng nhãn/số mẫu/nét, audit trùng | artifacts/drawai/data/l64-report.json và l128-report.json |
| D1-04 | Hình mẫu 10 lớp xác nhận chiều y/ranh giới nét | artifacts/drawai/data/l64-samples.png |
| D1-05–08 | Split phủ đủ dữ liệu, không overlap ID, resample đúng tính tay, rebuild tensor chính xác | unit tests; artifacts/drawai/data/verification.json; split manifest |
| D2-01–04 | Import runtime; forward batch 1/nhiều mẫu, gradient attention/embedding hữu hạn, học tập con train | artifacts/drawai/feasibility/python-report.json |
| D2-05–07 | ONNX checker, logits PyTorch/ORT/WASM khớp; worker thật; tải lỗi/retry/hash lỗi; benchmark | python-report.json, browser-report.json, playwright-report.json |
| D3-01–08 | CNN/Swin cùng input/split, ba seed mỗi họ, checkpoint chọn bằng validation, test metric cuối | run configs/log/checkpoint hashes và experiment-report.md |
| D4-01–07 | Artifact đã train đúng config/nhãn; full-test parity; version/hash; demo có nguồn | export report, release manifest, model-card.md |
| D5-01–08 | Canvas/nét/undo/clear, TypeScript preprocessing, model thật, top-3, JSON, responsive | browser tests DrawAI sản phẩm và ảnh giao diện |
| D6-01–03 | Toàn tuyến và request revision; input sai, artifact sai, mạng/retry; không gửi nét qua mạng | verification report và network assertions |
| D6-04–05 | Benchmark và kiểm tra browser/thiết bị thực, phân biệt viewport giả lập | benchmark report ghi browser/máy; acceptance-report.md |
| D6-06–08 | Nét người thật có nhãn độc lập, protocol/split/metric, các mục chưa chạy rõ | canvas manifest/evaluation và acceptance-report.md |
| D7-01–07 | Build, URL thật/deep link, model/WASM, tài liệu tái lập và rollback | release-report.md, user-guide.md, reproduction.md |

## Nghiệm thu D0 theo đầu việc

| ID | Điều kiện PASS | Bằng chứng D0 |
|---|---|---|
| D0-01 | Dataset, MVP và phụ thuộc phase được định nghĩa | project-plan và phases |
| D0-02 | Có luồng, state, hành vi lỗi/sửa nét và phạm vi online | scope.md |
| D0-03 | Schema v1 hợp lệ; ví dụ có/không timestamp; quy tắc kiểm tra toàn bản vẽ rõ | strokes.schema.json, data-contract.md, examples/drawai |
| D0-04 | Module và worker request/response, revision, version và stale result rõ | architecture.md |
| D0-05 | Runtime được kiểm kê, requirements và lệnh bootstrap/tái lập rõ; phân biệt import metadata với chạy runtime | environment.md, requirements.txt |
| D0-06 | Task có kiểm tra/báo cáo; ngưỡng nghiên cứu và phụ thuộc ngoài máy được phân biệt | tài liệu này, phases.md |

Report D0: `artifacts/drawai/d0/verification.json` và [d0-report.md](d0-report.md). PASS D0 là nghiệm thu đặc tả/schema, không phải nghiệm thu ứng dụng hoặc model đã train. Với D1–D7, chỉ ghi PASS khi bằng chứng thực thi tương ứng có thật; kiểm tra chưa chạy ghi NOT RUN.

## Các ngưỡng nghiệm thu triển khai

- Tensor Python/TypeScript: sai số tuyệt đối tối đa 2e-7 trên fixture chuẩn hóa.
- Logits PyTorch/ONNX/browser: `atol=1e-5`, `rtol=1e-4`. Thay ngưỡng phải ghi lý do/sai số.
- ONNX đã train: macro-F1 giảm không quá 0,001 tuyệt đối; báo tất cả argmax khác trên test.
- Mục tiêu PenDigits: accuracy ≥98%, macro-F1 ≥97%; không được chọn siêu tham số bằng test.
- Mục tiêu artifact ONNX ≤10 MiB; p95 inference ≤100 ms trên laptop tham chiếu, 5 warm-up + 100 lần đo. WASM tải riêng, không gộp vào dung lượng model.
- Tập canvas mục tiêu ≥250 mẫu từ ≥5 người; đánh giá riêng, không lấy mẫu train demo thay thế.

Lỗi chức năng, tensor hoặc artifact phải sửa trước phát hành. Nếu metric nghiên cứu chưa đạt, có thể ghi rõ phạm vi demo và phân tích lỗi. Kiểm tra cần người/thiết bị nhưng chưa chạy phải ghi NOT RUN; không đánh dấu đầy đủ chỉ từ test tự động.
