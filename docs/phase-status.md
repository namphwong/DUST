# Trạng thái các phase

Cập nhật 03/10/2026: Phase 0–6 hoàn thành local; CSV người dùng đã có pipeline/worker, resample/chia đoạn/cửa sổ, suy luận thật và xuất kết quả. Swin F1 vẫn 0,870489, chưa đạt 0,90: demo nghiên cứu có giới hạn. Chưa deploy Vercel/ nghiệm thu phát hành.

| Phase | Trạng thái | Đầu ra/điều kiện tiếp theo |
|---|---|---|
| 0 | Hoàn thành đặc tả | project-scope.md, data-contract.md, acceptance-criteria.md và examples/phase0 |
| 1 | Hoàn thành nghiệm thu kỹ thuật | PyTorch/ONNX Python/WASM parity 8/8 mẫu, browser tests 3/3 pass; model 8,58 MiB, median browser khoảng 3,60 ms. Báo cáo: browser-feasibility.md |
| 2 | Hoàn thành nghiệm thu dữ liệu | Train 5392 / val 1960 / test 2947; subject-disjoint, train-only normalization, 6 CSV demo thật; rebuild mảng/metadata giống nhau. Báo cáo: data-report.md |
| 3 | Hoàn thành Kaggle GPU và kiểm chứng artifact | 6 run, seeds 42/43/44; Swin seed 44 chọn bằng val, test F1 0,870489; mean F1 0,868143 ± 0,008016. P3-04 theo nhánh phân tích lỗi/demo nghiên cứu, mục tiêu 0,90 chưa đạt. Báo cáo: kaggle-training-report.md |
| 4 | Hoàn thành P4-01–P4-04 | ONNX trained=true 8,62 MiB; full test 2947/2947 prediction khớp, F1 không giảm; 14/14 fixture browser + benchmark; release hash/labels/mean/std đồng bộ. Báo cáo: phase4-artifact-report.md |
| 5 | Hoàn thành giao diện MVP, không phải toàn pipeline | MotionSense, 6 UCI train demo/provenance/hash, chart/scores/xuất CSV mẫu; CSV người dùng chỉ preview; 19 browser tests pass. Báo cáo: phase5-ui-report.md |
| 6 | Hoàn thành WEB-01–WEB-10 trên môi trường local | Worker/CSV/g–m/s²/resample/segment/window, progress/cancel/source isolation, export; 17 fixture/18 cửa sổ Python–web parity, 100.000 dòng/1.561 cửa sổ; 29 browser + 19 Python tests. Báo cáo: phase6-pipeline-report.md |
| 7 | Chưa bắt đầu | Deploy Vercel |
| 8 | Chưa bắt đầu | Tài liệu và nghiệm thu phát hành |

Tiếp theo: Phase 7 → 8. Thu cảm biến điện thoại chỉ là giai đoạn mở rộng. Báo cáo Phase 5 giữ phạm vi lịch sử; pipeline hiện tại được nghiệm thu riêng ở Phase 6. Mobile viewport không phải benchmark điện thoại thật.

Trước Phase 7, đã thiết kế lại UI/UX theo hướng nhập dữ liệu/nhận output: file/kéo thả/dán CSV, hướng dẫn bước tiếp theo, phân bố cửa sổ, chi tiết timestamp/sáu điểm và tải toàn bộ kết quả. Model/worker không đổi; suite mở rộng 32 browser tests + 19 Python tests. Xem [báo cáo UI/UX](ui-ux-redesign.md).
