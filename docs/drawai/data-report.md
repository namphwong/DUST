# DrawAI — Báo cáo D1

Kiểm chứng và dựng lại local ngày 11/10/2026. **D1 đạt** theo hợp đồng `drawai-geometry-v1`, với giới hạn writer ID nêu dưới đây. Đợt rà soát D1 hiện tại đã chạy lại parser/preparation cho L64/L128, 17 unit tests và full-data verification; không chỉ dựa vào report có sẵn.

## Dữ liệu và split

Đọc `pendigits-orig.tra.Z` và `pendigits-orig.tes.Z` từ archive [UCI PenDigits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits), giải nén bằng unlzw3. Parser đối chiếu từng `.SEGMENT` với range stroke toàn file, nhãn trong `.COMMENT` và các cặp `.PEN_DOWN/.PEN_UP`. Không đọc file `.INCLUDE`, không bỏ sample lỗi.

| Split | Mẫu | Điểm/mẫu min–max | Median điểm | Số nét/mẫu |
|---|---|---|---|---|
| Train | 5.999 | 9–121 | 39 | 1–3 |
| Validation | 1.495 | 10–185 | 40 | 1–3 |
| Test | 3.498 | 9–231 | 36 | 1–4 |

Development pool 7.494 mẫu và test 3.498 mẫu khớp tài liệu nguồn. Cả ba split đủ 10 lớp; không có duplicate trajectory trong hoặc giữa split. Phát hiện 3.525 điểm trùng liên tiếp (train 1.538, val 412, test 1.575); tiền xử lý bỏ trùng trong từng nét. Có 4 nét một điểm (train 1, test 3), giữ đúng một token; không bỏ toàn chữ số.

Duplicate ở đây là SHA-256 của toàn chuỗi nét gốc giữ thứ tự điểm và ranh giới nét, không phải kiểm tra mọi biến thể hình học tương tự. Không có sample lỗi bị loại; số điểm cao nhất 231, số nét cao nhất 4, đều trong giới hạn hợp đồng.

| Chữ số | Train | Validation | Test |
|---|---:|---:|---:|
| 0 | 624 | 156 | 363 |
| 1 | 624 | 155 | 364 |
| 2 | 624 | 156 | 364 |
| 3 | 576 | 143 | 336 |
| 4 | 624 | 156 | 364 |
| 5 | 576 | 144 | 335 |
| 6 | 576 | 144 | 336 |
| 7 | 623 | 155 | 364 |
| 8 | 576 | 143 | 336 |
| 9 | 576 | 143 | 336 |

`.COMMENT` có ba số, trường thứ hai có 30 giá trị trong development và 14 giá trị trong test. Đây là bằng chứng gợi ý nhưng không đủ xác nhận ý nghĩa writer ID vì tài liệu archive không mô tả các trường và không có `dene.doc`. Parser lưu nguyên comment, writer ID để null. Train/validation dùng phương án fallback đã định trước: phân tầng theo lớp, seed 42, lấy floor(20%) mỗi lớp cho validation. **Validation có thể chung người viết với train.** Test chính thức giữ nguyên; mô tả writer-independent của test dựa trên nguồn UCI, không dựa vào ID do dự án suy ra.

## Tiền xử lý và kiểm chứng

- Hình 10 lớp cho thấy source y-up; chuyển sang canvas y-down trước chuẩn hóa. Đã xem hình trực quan, các chữ số có hướng đúng.
- Chuẩn hóa bounding box giữ tỷ lệ; resample theo từng nét; marker đầu nét không bị mất. Sinh cả L64 và L128 bằng cùng split, float32 `[N,3,L]`.
- 17 unit tests đạt: đường thẳng tính tay, chiều y, translation/scale invariance, hai nét không nối, phân bổ remainder, singleton, điểm trùng, input lỗi, parser lỗi và split phủ đủ mẫu. Bổ sung giữ tỷ lệ/đầu-cuối nét, không thay đổi input, giới hạn đúng 16 nét/8.192 điểm, tổng điểm nhiều nét, bool/chuỗi/container/Infinity và cấu hình split không hợp lệ.
- `verify_data` tái tạo chính xác **21.984 tensor** của 6 split (3 split × 2 độ dài). Đọc lại compressed source, kiểm tra bản raw bằng chính nội dung giải nén, so các file với archive, kiểm tra hash nguồn/tài liệu.
- Split được tái dựng bằng đường kiểm tra độc lập với hàm split production. Kiểm tra từng manifest entry (nhãn/source/line/index/range/comment/trajectory hash), metadata labels/preprocessing, dtype/miền tọa độ, stroke marker và 10 fixture mỗi chiều dài. L64/L128 có cùng source provenance và split.
- Hash nội dung cả 6 tensor tập dữ liệu khớp các hash trước đợt bổ sung kiểm toán; preprocessing version không đổi. Giữ nguyên fixture/input hợp lệ nên không thay đổi dữ liệu dùng bởi feasibility D2.
- Manifest lưu hash compressed/uncompressed/archive/tài liệu; NPZ và source được ignore trong Git, có lệnh tải/tái tạo. Hash tensor không phụ thuộc timestamp của ZIP container.

## Bằng chứng và lệnh

| Task | Kết quả / bằng chứng |
|---|---|
| D1-01 | Archive, compressed/raw/tài liệu có hash; verify đối chiếu nội dung archive và giải nén |
| D1-02 | Parser strict, giữ sample ID/source line/index/segment/comment; test directive sai và nhãn/range lỗi |
| D1-03 | 10.992 mẫu, đủ 10 lớp, thống kê điểm/nét/trùng; writer ID giữ null |
| D1-04 | Đã xem hình nét gốc có marker/hướng viết và hình resample đủ 10 lớp |
| D1-05 | Train/val phân tầng seed 42, test giữ nguyên; coverage/ID overlap được verify |
| D1-06 | Bbox normalization giữ tỷ lệ, float32 hữu hạn; unit tests tính tay |
| D1-07 | Per-stroke L64/L128, giữ nét một điểm, không nối khoảng nhấc bút; tests ca biên |
| D1-08 | 6 split/21.984 tensor rebuild chính xác, provenance/metadata/fixture được verify |

Chạy từ root repo:

```powershell
./.venv-phase1/Scripts/python.exe -m unittest training.tests.test_drawai_data -v
./.venv-phase1/Scripts/python.exe -m training.drawai.prepare_data --length 64
./.venv-phase1/Scripts/python.exe -m training.drawai.prepare_data --length 128
./.venv-phase1/Scripts/python.exe -m training.drawai.verify_data
```

Thêm `--download` vào prepare_data khi chưa có nguồn. Đợt này dùng nguồn đã tải và kiểm tra archive thật; không cần tải lại cùng dataset. Lệnh download là cơ chế tái tạo cho máy mới.

- `artifacts/drawai/data/l64-report.json`, `l128-report.json`: thống kê/hash.
- `artifacts/drawai/data/l64-samples.png`, `l128-samples.png`: hình đủ 10 lớp.
- `artifacts/drawai/data/l64-samples-resampled.png`, `l128-samples-resampled.png`: tọa độ chuẩn hóa/resample, marker đầu nét.
- `artifacts/drawai/data/verification.json`: xác minh từng tensor.
- `artifacts/drawai/data/d1-review.json`: tổng hợp task/tests, hash code/bằng chứng của đợt rà soát.
- `data/processed/pendigits-drawai-v1/l64/` và `l128/`: NPZ, manifest, preprocessing, labels, fixtures.
- [Hợp đồng dữ liệu](data-contract.md), [môi trường và lệnh tái lập](environment.md).

Chưa chọn L tốt nhất; D3 dùng validation. Chưa train classifier và chưa có đánh giá nét người dùng canvas.
