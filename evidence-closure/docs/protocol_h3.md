# Protocol H3 — Learned action utility cho repair trong DocVQA (khóa ngày 2026-11-05, trước mọi output E10)

Bối cảnh: E8 cho thấy lỗi chủ yếu là tính nhầm, sau đó là scale, đọc sai ô và sai phép toán ([e8_error_taxonomy.md](e8_error_taxonomy.md)). E9 (dev, thăm dò) đo recovery rate của từng hành động. H3 hỏi: **một bộ dự đoán lợi ích kỳ vọng của từng hành động có chọn hành động sửa lỗi tốt hơn các chính sách cố định hoặc dựa trên confidence, ở cùng chi phí, hay không?**

## Hành động (cố định; không thêm hành động sau khi xem test)

| mã | hành động | chi phí (lượt gọi VLM) |
|---|---|---|
| `direct` | trả lời trực tiếp; luôn chạy, cho log-prob | 1 |
| `program` | biểu thức số học → Python | 1 |
| `scale_ocr` | scale từ closure quan hệ trên OCR, theo các số trong biểu thức | 0 |
| `scale_vlm` | hỏi lại VLM về scale | 1 |
| `zoom` | crop vùng bảng khớp câu hỏi, ×2, chạy lại program | 1 |
| `verify` | model tự kiểm tra và sửa biểu thức | 1 |

Câu trả lời cuối = giá trị từ một nguồn {direct, program, zoom, verify} + scale từ một nguồn {direct, scale_ocr, scale_vlm}. Chấm theo **metric chính thức kiểu TAT-DQA**: đúng khi giá trị đúng (hàm `value_correct`, chế độ strict, không ×100) **và** scale trùng scale gold, kể cả trường hợp rỗng.

## Dữ liệu

- **train**: 800 câu từ TAT-DQA train, cùng bộ lọc như E1 (seed 1). Chỉ dùng để học policy.
- **dev**: tập E1 (600 câu). Dùng để chọn siêu tham số và λ chi phí.
- **test**: tập E3 (600 câu TAT-DQA test). **Chạy đánh giá một lần.**
- Model: Qwen2.5-VL-7B (600 câu test) và Qwen3-VL-8B (300 câu test đầu, cùng lý do ở deviation #11). Mọi hành động chạy trên mọi câu (`scripts/run_h3.py`), nên các policy chỉ khác nhau ở *lựa chọn*, không ở dữ liệu.

## Policy so sánh (cùng ngân sách trung bình B lượt gọi VLM mỗi câu, B ∈ {2, 3})

1. **Fixed-2**: direct + program, scale từ direct. **Fixed-2+OCR**: như trên nhưng scale từ OCR.
2. **Fixed-3**: direct + program + một hành động thêm, chọn hành động tốt nhất trên dev (verify, zoom hoặc scale_vlm).
3. **Confidence-triggered**: luôn direct + program. Nếu log-prob của `direct` dưới ngưỡng τ (chọn trên dev để chi phí trung bình bằng B) thì gọi thêm hành động tốt nhất của Fixed-3.
4. **Error-type → hành động cố định**: một classifier dự đoán loại lỗi E8 từ đặc trưng; mỗi loại gán một hành động cố định (read → zoom, reason → verify, calc → program, scale → scale_vlm).
5. **Learned utility (phương pháp)**: với mỗi nguồn giá trị và scale, dự đoán P(đúng | đặc trưng). Chọn tập hành động tối đa hóa P(đúng) − λ·chi phí, chọn λ trên dev để chi phí trung bình ≈ B, rồi lấy cặp (giá trị, scale) có P cao nhất.

Đặc trưng: chỉ dùng những gì có trước khi trả lời, như output của các hành động đã chạy, log-prob, đồng thuận giữa các nguồn, grounding của số trên OCR, loại câu hỏi theo regex và số chữ số. Không dùng gold.

## Tiêu chí thành công (cả ba)

1. Trên **test**, theo metric chính thức, *learned utility* − (policy baseline tốt nhất ở cùng B) > 0, với CI 95 % bootstrap theo tài liệu không chứa 0, ở **ít nhất một B** và **cả hai model**.
2. **Recovery theo loại lỗi:** trên các câu mà Fixed-2 sai, policy học được phải chọn hành động sửa đúng loại lỗi nhiều hơn baseline confidence-triggered (báo cáo bảng recovery theo loại lỗi E8).
3. **Đặt cạnh số liệu đã công bố** trên TAT-DQA, ghi rõ rằng tập con và metric có khác biệt.

Nếu (1) không đạt: báo cáo H3 không được ủng hộ. Kết quả chính khi đó là bài phân tích lỗi (E8/E9) cùng mức đóng góp của program-of-thought và scale-OCR.

## Không được làm

- Không thêm hành động, đặc trưng hay policy mới sau khi đã chạy test.
- Không chọn B hoặc λ dựa trên test.
- Không đổi hàm chấm.

## Sửa đổi trước khi có output (2026-11-05)

- Vì chi phí (~30 s/câu cho 6 lượt gọi trên GPU dùng chung): **train = 600 câu đầu** của `h3_train_set.json`. **Bỏ chạy lại dev**: siêu tham số, τ và λ được chọn bằng **5-fold cross-validation theo tài liệu trên train**. Test giữ nguyên: 7B 600 câu, Qwen3 300 câu đầu.
- Ghi chú về quy tắc chấm của TAT-DQA: câu span có ký hiệu `%` trong đáp án có scale gold rỗng. Mọi policy chịu cùng quy tắc này.
- (trước khi có output train/test) Các baseline Fixed-3, Confidence và Error-type lấy scale từ **nguồn tốt hơn trên train**, giữa `direct` và `scale_ocr`, thay vì cố định OCR. Lý do: thử nghiệm trên dữ liệu giả lập E8+E9 cho thấy OCR-scale có thể kém hơn direct-scale (OCR luôn trả về một scale khi gold rỗng), nên cố định OCR sẽ làm baseline yếu đi một cách nhân tạo.
- (sau CV trên train, trước khi có output test) Bảng recovery được tính trên **các câu Fixed-2 sai** (scale từ direct), đúng như tiêu chí 2. Bản trước dùng nhầm Fixed-2+OCR làm mốc. Chỉ thay đổi phần báo cáo; nhãn của policy error-type giữ nguyên.
