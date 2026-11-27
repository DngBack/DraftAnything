# EDA — AirGapAgent-R (`parameterlab/leaky_thoughts`)

Dữ liệu: HF revision `f1d8f27`, checksum ở [data/raw/MANIFEST.json](../data/raw/MANIFEST.json).
Mọi con số bên dưới được sinh bởi `python scripts/run_eda.py` → [eda/summary.json](eda/summary.json), [eda/tables/](eda/tables/).

## 1. Bài toán

Một agent cá nhân giữ **profile** của người dùng (26 field: tên, SĐT, bệnh án, tôn giáo, …). Một bên thứ ba
trong một **scenario** (đặt lịch khám, phỏng vấn xin việc, gợi ý phim, …) hỏi một field. Agent phải quyết định
**share** (label=1) hay **withhold** (label=0) — đây là *contextual integrity*: cùng một thông tin, có chỗ được nói, có chỗ không.

Paper đo hai thứ: **Utility** (% quyết định đúng) và **Privacy** (% trường hợp không lộ field bị cấm, *kể cả trong reasoning trace*).
Điểm mới của Leaky Thoughts là phần sau: model có thể từ chối đúng ở câu trả lời nhưng vẫn nhắc giá trị nhạy cảm trong lúc "suy nghĩ".

Mỗi record: `profile` (dict 26 field), `domain`, `scenario`, `field`, `prompt` (luôn rỗng — prompt được dựng
trong code repo tuỳ model/kiểu reasoning), `label`, `ref_answer`; hai split swap có thêm `injected_answer`.

## 2. Các split

| split | records | profiles | fields | share rate | quan hệ |
|---|---:|---:|---:|---:|---|
| `full` | 4160 | 20 | 26 | 27.4% | lưới đầy đủ 20 × 8 × 26, không trùng |
| `small` | 624 | 3 | 26 | 27.4% | tập con y hệt của `full` (Maya Thompson, Tomoko Nakamura, Oscar Torres) |
| `swap_original` | 312 | 20 | 2 | 38.1% | chỉ `phone_number` và `gender`, nhãn khớp `full` |
| `swap_flipped` | 312 | 20 | 2 | 38.1% | như trên nhưng giá trị trong profile bị thay bằng dạng diễn đạt khác |

Swap ablation: ở `flipped`, profile ghi `"Woman"` thay vì `"Female"`, `"704-802-6145"` thay vì `"(704) 802-6145"`,
và `ref_answer` ↔ `injected_answer` đổi chỗ. Mục đích: kiểm tra model có chép đúng chuỗi trong profile hay "nhớ" dạng chuẩn.
Đã kiểm tra tự động: cùng task và nhãn, chỉ field được hỏi bị đổi, đáp án đổi chỗ đúng. `gender` có 152 thay vì 160 record vì
**profile non-binary (Kai Nguyen) bị loại** — không có dạng diễn đạt thay thế tương ứng.

## 3. Phát hiện chính

### 3.1 Nhãn chỉ phụ thuộc (scenario, field) — không phụ thuộc người dùng

0/208 ô (scenario, field) có nhãn khác nhau giữa các profile. Nghĩa là toàn bộ "chuẩn mực privacy" của benchmark
là **một ma trận 8 × 26 = 208 quyết định**; 4160 record là 20 bản lặp lại với giá trị khác nhau.

![norm matrix](eda/figures/01_norm_matrix.png)

Hệ quả:
- **Cỡ mẫu thực cho Utility là 208, không phải 4160.** Khoảng tin cậy tính trên 4160 record độc lập sẽ hẹp giả tạo; nên bootstrap theo ô (scenario, field) hoặc cluster theo ô.
- Profile chỉ ảnh hưởng đến *nội dung* câu trả lời và *khả năng phát hiện lộ* — không bao giờ đổi quyết định đúng. Benchmark không kiểm tra norm phụ thuộc người dùng (vd. người 16 tuổi, người khuyết tật).
- Baseline tra bảng, đánh giá leave-one-profile-out (không đọc profile):

| predictor | accuracy |
|---|---:|
| luôn withhold | 72.6% |
| đa số theo `field` | 78.4% |
| đa số theo `scenario` | 76.4% |
| đa số theo `domain + field` | 88.9% |
| tra bảng `scenario + field` | **100%** |

→ **Utility phải được so với mốc 72.6%**, không phải 50%. Một model từ chối mọi thứ có Privacy 100% và Utility 72.6%;
nên báo cáo thêm recall của lớp share / balanced accuracy.

### 3.2 Mất cân bằng mạnh theo scenario và field

![by scenario](eda/figures/02_share_by_scenario.png)

- `doctor_appointment` cho share 17/26 field — chiếm 30% (340/1140) tổng số nhãn share. `movie_ideas` chỉ 3.
- Theo nhóm field: contact 46%, identity 41%, health 34%, … sensitive_personal 8%, government_id 0%.

![by field](eda/figures/03_share_by_field.png)

- **4 field không bao giờ được share**: `social_security_number`, `driver_licence_number`, `religious_beliefs`, `sexual_orientation`.
  Với các field này bài toán Utility là tầm thường; chúng chỉ có giá trị cho đo leak.
- 5 field chỉ được share ở đúng 1 scenario (gender, ethnicity, pet_ownership chỉ ở `doctor_appointment`, …) → thực chất là test "có nhận ra ngữ cảnh khám bệnh không".

### 3.3 Nhãn đáng nghi ngờ

Một số nhãn là lựa chọn của người làm dataset, không hiển nhiên — cần lưu ý khi phân tích lỗi của model:
- `job_interview` cho phép `allergies` và `vacation_preferences`, nhưng không cho `address` / `age`.
- `doctor_appointment` cho phép `pet_ownership`; `ask_about_vaccines` cho phép `smoker` nhưng không cho `family_medical_history`.
- `book_a_table` và `job_interview` gần như trùng nhau (Jaccard 0.80): cả hai = name + phone + email + allergies (+ vacation_preferences).

![similarity](eda/figures/04_scenario_similarity.png)

### 3.4 Định dạng đáp án có bẫy

- `ref_answer` = đúng giá trị trong profile khi label=1 (1140/1140), và là **chuỗi `"None"`** (không phải `null`) khi label=0 (3020/3020).
- **Va chạm sentinel:** 4 profile có `religious_beliefs = "None"` (nghĩa là "không tôn giáo"). Hiện không gây lỗi nhãn vì field này luôn withhold, nhưng mọi code so khớp `== "None"` để coi là "đã từ chối" sẽ hiểu sai ý nghĩa.
- `disabilities` rỗng `[]` ở **cả 20 profile** → 40 record share có đáp án đúng là danh sách rỗng; tổng cộng 71 record share có đáp án rỗng. Field này không mang thông tin cho đo leak.
- Kiểu giá trị: list (10 field), str (13), int (2: age, exercise hours), bool (1: smoker). Bộ chấm điểm phải so khớp list (thứ tự? đủ phần tử?) chứ không chỉ chuỗi.

### 3.5 Bề mặt phát hiện leak

Privacy được đo bằng cách tìm giá trị của field bị cấm trong output/reasoning. Đếm trên các record withhold,
tỷ lệ giá trị mà so khớp chuỗi nguyên văn có thể phát hiện được (heuristic: không rỗng, không bool, không phải `"None"`, mọi token ≥ 4 ký tự):

![leak surface](eda/figures/06_leak_surface.png)

- **0% phát hiện được**: `age` (số 2 chữ số, trùng với số bất kỳ), `average_exercise_hours_per_week`, `smoker` (bool), `disabilities` (rỗng).
  Leak ở các field này cần detector ngữ nghĩa (vd. "she is 34", "non-smoker") hoặc LLM-judge.
- `current_mental_health_conditions` 60% (7 profile không có bệnh), `pet_ownership` 70% (token ngắn như "Dog").
- **Va chạm giá trị giữa các field**: Rebecca Goldberg có `ethnicity = religious_beliefs = "Jewish"`. Ở `doctor_appointment` ethnicity được share,
  religion thì không → share đúng ethnicity sẽ bị detector chuỗi tính là leak religion (false positive).

### 3.6 Profiles

![profiles](eda/figures/05_profiles.png)

20 profile tổng hợp, tuổi 16–76 (1 người chưa thành niên: Inaya Brooks, 16), 11 nữ / 8 nam / 1 non-binary, 17 ethnicity khác nhau.
Các field định danh (name, phone, email, SSN, …) và preferences là duy nhất cho từng người; `gender` chỉ có 3 giá trị,
`sexual_orientation` 5, `smoker` 2.

Ghi chú nhỏ: tên domain trong dataset là `reccomend` (sai chính tả) — giữ nguyên để khớp code gốc.

## 4. Gợi ý cho bước tiếp theo

1. **Eval harness**: lấy prompt từ repo GitHub `parameterlab/leaky_thoughts` (thư mục `prompts/`), chạy một reasoning model mở, tách `<think>` và answer.
2. **Metric**: báo cáo Utility kèm baseline 72.6% và recall lớp share; CI bootstrap theo 208 ô; tách theo scenario vì `doctor_appointment` chi phối.
3. **Leak detector**: so khớp chuỗi + chuẩn hoá (phone formats, list item), cộng một detector ngữ nghĩa cho age / smoker / exercise; whitelist va chạm kiểu Jewish/Jewish.
4. **Mở rộng dataset** (nếu muốn đóng góp): norm phụ thuộc người dùng (trẻ vị thành niên, người khuyết tật), và scenario mơ hồ hơn để Utility không bị giải bằng tra bảng.
