# DrawAI — Kiến trúc và hợp đồng worker D0

Ngày rà soát: 11/10/2026. Đây là thiết kế ứng dụng D5; worker kỹ thuật D2 chưa triển khai giao thức sản phẩm dưới đây.

## Ranh giới module

| Module | Trách nhiệm | Đầu vào → đầu ra |
|---|---|---|
| `training/drawai/prepare_data.py` | Parser nguồn, kiểm toán và split | UNIPEN → mẫu có provenance, manifest và tensor |
| `training/drawai/preprocessing.py` | Thuật toán hình học tham chiếu | Nét và cấu hình → float32 `[3,L]` |
| `training/drawai/models.py` | Factory CNN/Swin theo config | Tensor `[B,3,L]` → logits `[B,10]` |
| Train/evaluate/export, D3–D4 | Huấn luyện, chọn validation, test, ONNX | Dataset/config → checkpoint/report/release |
| Canvas, D5 | Thu một pointer, quản lý stroke, DPR/render | Pointer Events → nét theo CSS pixel |
| App/client, D5 | State, revision, snapshot và giao tiếp worker | Tác vụ người dùng → request; response hợp lệ → UI |
| `web/src/drawai/preprocessing.ts` | Đối chiếu thuật toán Python | Nét → Float32Array channel-major |
| Worker/inference, D5 | Tải và xác minh artifact, tiền xử lý, inference | Snapshot → logits, scores, latency |
| Export JSON, D5 | Lưu nét gốc và nguồn/nhãn độc lập | Bản vẽ → `drawai-strokes-v1` |

Các file D3–D5 chưa có là đầu ra dự kiến; không dùng bảng này như danh sách code đã hoàn thành. Module feasibility D2 chỉ dùng kiểm chứng `trained=false`.

## Luồng sản phẩm

```text
Pointer → canvas → strokes + drawing_revision
                       ↓ bấm Nhận dạng, chụp snapshot
Client → worker → load/hash/contract → preprocess → ONNX → softmax
   ↑                                                        ↓
   └──── request_id + revision + model_version + result ─────┘
```

Tách hai nhóm state: `input` (empty/drawing/valid/invalid) và `model` (unloaded/loading/ready/error). Ready về nét không đồng nghĩa model đã tải. Nút Nhận dạng có thể tải model lần đầu; vô hiệu khi đang vẽ, nét không hợp lệ hoặc có request đang chạy. Worker kiểm tra lại input trước inference.

## Giao thức message dự kiến v1

Mọi message có `protocol_version: "drawai-worker-v1"` và `request_id` số nguyên dương tăng trong mỗi vòng đời worker. Request dự đoán có `drawing_revision` nguyên không âm. Request/response là dữ liệu structured-clone, không chứa callback hoặc DOM.

| Chiều | `type` | Payload bắt buộc |
|---|---|---|
| Client → worker | `load` | `release_version` do ứng dụng chọn từ danh sách release hỗ trợ |
| Client → worker | `predict` | `drawing_revision`, `release_version`, `drawing` là snapshot schema v1 |
| Worker → client | `progress` | `stage`: fetching/verifying/initializing/preprocessing/inference; revision nếu thuộc predict |
| Worker → client | `ready` | `model_version`, `preprocessing_version`, `input_shape`, `labels` theo thứ tự 0–9 |
| Worker → client | `result` | revision, model/preprocessing version, 10 logits, 10 scores, `predicted_id`, timing |
| Worker → client | `error` | `code`, `message`, `retryable`, revision nếu thuộc predict |

Worker có thể tải model trong request predict đầu tiên, trả progress rồi result. Không có inference song song: client chỉ có một request hoạt động; worker từ chối request khác bằng `BUSY` để bảo vệ khi client lỗi. Chỉ trả một terminal message ready/result/error cho mỗi request.

Ví dụ response thiết kế (số liệu minh họa cấu trúc, không phải dự đoán đã chạy):

```typescript
type PredictionResult = {
  protocol_version: 'drawai-worker-v1';
  type: 'result';
  request_id: number;
  drawing_revision: number;
  model_version: string;
  preprocessing_version: string;
  logits: number[]; // đúng 10 số hữu hạn, thứ tự 0–9
  scores: number[]; // đúng 10 số hữu hạn trong [0,1], tổng xấp xỉ 1
  predicted_id: number; // argmax, hòa điểm lấy ID nhỏ nhất
  timing_ms: { preprocessing: number; inference: number; total: number };
};
```

`inference` chỉ đo `session.run`; `total` đo thời gian xử lý request trong worker, có thể gồm tải model ở lần đầu. UI đo thêm thời gian từ click đến nhận response nếu cần. Không dùng total của lần tải đầu so với ngưỡng p95 inference sau warm-up.

## Tính nhất quán của kết quả

1. Tăng revision ở mọi thay đổi dữ liệu: bắt đầu nét mới, thêm điểm, kết thúc/cancel nét có tác động dữ liệu, clear, undo, nạp mẫu. Thay đổi kích thước hiển thị không làm đổi tọa độ dữ liệu gốc.
2. Khi bấm Nhận dạng, clone dữ liệu và giữ request ID/revision. Worker không đọc lại state canvas sống.
3. Chỉ hiện result/error của predict khi request ID, revision và release version khớp request hiện hành, đồng thời revision vẫn khớp bản vẽ hiện tại.
4. Sửa nét xóa kết quả cũ ngay; request đang chạy được coi là hết hiệu lực. Cho phép hoàn tất tính toán nhưng bỏ response. Không hứa hủy `session.run` tức thì.
5. Client không mở predict mới trước terminal response hoặc reset worker. Nếu worker crash/không phản hồi, kết thúc promise, dọn worker rồi cho retry; response của vòng đời cũ bị bỏ.
6. Load error vẫn thông báo ở trạng thái model; stale input error không ghi đè hướng dẫn của bản vẽ mới. Retry giữ nguyên nét, tạo request ID mới.

## Release và lỗi

Release có model, labels, preprocessing và report đồng bộ qua manifest hash/bytes. Từ chối `trained=false` ở trang sản phẩm; nhãn phải là 0–9, input fixed `[1,3,L]`, output `[1,10]`, preprocessing version được client hỗ trợ. Chiều dài lấy từ release, không có nút tùy ý đổi L cho cùng model.

Lỗi phân tầng: mã hình học trong data-contract; `SCHEMA`, `TIMESTAMP`, `PROTOCOL`, `BUSY`; lỗi model `FETCH`, `HASH`, `CONTRACT`, `INIT`, `INFERENCE`, `WORKER`. Fetch/init/worker có thể thử lại; hash/contract phải tải bộ release nhất quán hoặc sửa triển khai, không bỏ xác minh để tiếp tục.

Không gọi `reference_label` là kết quả model. Nhãn dự đoán và 10 scores thuộc result; export nét chỉ thêm prediction metadata nếu D5 có tùy chọn riêng và gắn revision/version. D0 không đặt prediction vào nhãn tham chiếu.

## Điểm kiểm chứng theo phase

- D1: parser, split, hình học và semantics ranh giới nét.
- D2: forward/backward, ONNX, runtime và feasibility worker.
- D4: checkpoint đã train, release thật và full-test parity.
- D5: canvas/schema/worker protocol và UI thực.
- D6: toàn tuyến, race/reset/retry, network và thiết bị/người thật.
