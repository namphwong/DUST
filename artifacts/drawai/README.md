# Bằng chứng kiểm chứng DrawAI

Các thư mục phase là **snapshot theo thời điểm**, không phải tất cả đều mô tả ứng dụng hiện hành. Xem [trạng thái hiện tại](../../docs/drawai/status.md) và [tài liệu](../../docs/drawai/README.md).

| Thư mục | Vai trò |
|---|---|
| `phase0/`, `phase1/` | Mốc phục hồi, inventory, dọn MotionSense và design gốc |
| `phase2/` | Pretrained DINOv3/feasibility, không phải classifier Paint đã train |
| `phase3-image64/`, `phase4-image64/` | Dữ liệu ảnh Paint, review, split và hash khóa |
| `phase5-image64/` | Training ba seed/họ, selection bằng validation và đánh giá đã khóa |
| `phase6-image64/` | Swin ONNX + DINO TorchScript, numerical parity và bundle anchor |
| `phase7-image64/` | Tích hợp canvas/UX, snapshots, quan sát demo và benchmark tại P7 |
| `repo-review/` | Rà soát/dọn repo hiện tại; test output tách khỏi bằng chứng phase cũ |
| `phase3/`, `phase4/`, `d3-local-v1/`, `export/`, `ui/`, `verification/` | Protocol nét bút/đợt kiểm chứng cũ; giữ cho truy nguyên, không dùng thay kết quả ảnh Paint |

Các bản `pre-rerun/`, `pre-p3/`, `pre-p4/` giữ nguyên bytes/hashes của tài liệu trước đó. Link tương đối bên trong chúng được viết theo vị trí gốc, nên có thể không mở từ thư mục snapshot. Không tự sửa chúng rồi coi hash snapshot vẫn hợp lệ; dùng các tài liệu hiện hành ở `docs/drawai/`.

Một số report/fixture/ảnh có cùng bytes ở nhiều vị trí để đối chiếu nguồn và gói public. Alias `model.onnx` và tên canonical của Swin PenDigits cũng có chủ đích, đã được manifest/loader/verifier tham chiếu. Giữ các file này; không xóa theo tiêu chí trùng hash đơn thuần.

Checkpoint/raw data/model mới lớn bị Git ignore. Các đường dẫn `.log` trong báo cáo lịch sử chỉ là log local, không phải file được phân phối qua GitHub. Các hash/evidence JSON được theo dõi trong Git mới là nguồn đối chiếu có thể tải cùng code.
