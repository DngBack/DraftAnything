# Giải thích chi tiết thuật toán bổ sung và các ý tưởng nghiên cứu VATr++ tiếng Việt

**Mã report:** 261114_report_07. **Ngày:** 14/11/2026.

**Mục đích:** giải thích từng thay đổi làm gì, làm ở đâu trong hệ thống, tại sao có thể giúp chữ hiếm, đã triển khai tới mức nào và phải kiểm chứng bằng cách nào. Đây là tài liệu thuật toán; số liệu training dài hạn được cập nhật riêng trong [report 05](261114_report_05_Continuation_component_den_250k.md).

## 1. Đọc phần này trước: hiện có những thuật toán nào?

Có bốn vị trí khác nhau để cải thiện bài toán. Chúng không phải bốn tên gọi cho cùng một thay đổi.

| Vị trí thay đổi | Câu hỏi dễ hiểu | Thuật toán/module | Trạng thái |
|---|---|---|---|
| Chọn dữ liệu | Mỗi lượt nên yêu cầu model viết văn bản nào? | Sampling theo thành phần và hoa/thường | **Đang dùng trong run 250k** |
| Biểu diễn nội dung | Khi yêu cầu viết Ễ, model được cung cấp thông tin gì? | Query tách base/modifier/tone/case | Có prototype; **chưa dùng trong run 250k** |
| Dựng ảnh | Model vẽ nguyên chữ hay vẽ thân và dấu riêng? | Renderer lớp mực thân–modifier–tone | Có prototype; **chưa dùng trong run 250k** |
| Giám sát nội dung | Làm sao biết ảnh đã có đúng thân và dấu? | OCR/CTC theo thành phần | Có prototype; **chưa thay OCR của run 250k** |

Ngoài bốn thuật toán trên còn có các **ý tưởng chưa triển khai**: memory thành phần theo writer; học hình chữ đơn từ nhiều ngữ cảnh; và bổ sung dữ liệu theo độ thiếu tổ hợp. Những ý tưởng này được mô tả ở phần 9–11.

**Câu quan trọng nhất:** model đang chạy hiện nay là VATr++ pretrained được fine-tune bằng cách chọn transcript mới. Nó chưa được thay bằng một generator bắt buộc ghép `E + mũ + ngã` để tạo `Ễ`.

## 2. Bài toán thực sự khó ở đâu?

### 2.1. Phân biệt thân chữ, modifier, thanh và hoa/thường

Trong tài liệu này:

- **Base:** thân Latin chưa có modifier, như e hoặc a.
- **Modifier:** phần tạo biến thể chữ cái, như mũ trong ê, trăng trong ă, móc trong ơ/ư.
- **Tone:** dấu thanh, như sắc, huyền, hỏi, ngã, nặng; thanh ngang tương ứng không có dấu thanh.
- **Case:** hoa/thường; với số và dấu câu có giá trị literal.

Ví dụ:

| Chữ | Base | Modifier | Tone | Case |
|---|---|---|---|---|
| e | e | không | không | thường |
| ê | e | mũ | không | thường |
| ễ | e | mũ | ngã | thường |
| E | e | không | không | hoa |
| Ê | e | mũ | không | hoa |
| Ễ | e | mũ | ngã | hoa |

`Ê` có thể được xem là một bước trung gian về hình dạng, còn trong parser thì `Ễ` được phân tích trực tiếp thành bốn thuộc tính. Việc phân tích Unicode là quy tắc xác định, không phải một mạng đã học được hình chữ. `Đ/đ` được xử lý ngoại lệ thành thân d + crossbar vì NFD không tự tách thành D/d và một dấu gạch.

### 2.2. Bốn loại thiếu dữ liệu

1. **Thiếu nguyên chữ:** không có ảnh Ễ của writer thật trong train.
2. **Thiếu tương tác:** có mũ và ngã riêng lẻ nhưng ít ví dụ đặt chúng cùng nhau trên chữ hoa.
3. **Thiếu phong cách cá nhân:** support của writer A không cho biết A viết ngã như thế nào.
4. **Thiếu ngữ cảnh:** có Ễ trong NGUYỄN nhưng không có ảnh/transcript Ễ đứng riêng.

Những thiếu hụt này cần các giải pháp khác nhau. Lặp lại một từ nhiều lần không tạo ra bằng chứng mới về phong cách; thêm một embedding ngã cũng không tự xác định vị trí ngã trên Ê.

Audit hiện có: writer thật train có 0 ảnh chứa Ễ, 0 ảnh Ê, nhưng có 166 ảnh chứa ễ. HAT có 100 ảnh chứa Ễ và 961 ảnh chứa Ê; HAT không có nhóm writer thật đáng tin cậy. Trong 100 ảnh chứa Ễ, không có transcript đúng một ký tự Ễ. [Số liệu chi tiết](character_context_audit.json).

Đây là **số ảnh chứa ký tự**, không phải số lần ký tự xuất hiện. Các số liệu về writer diversity và ngữ cảnh phải được giữ riêng.

## 3. VATr++ hiện tại làm gì trước khi bổ sung thuật toán?

```text
Ảnh support cùng writer → encoder phong cách → style memory
                                                ↓
Văn bản → hình glyph Unifont từng ký tự → Transformer → CNN → ảnh chữ viết tay
```

“Style memory” là các vector đặc trưng từ support để decoder tham khảo; nó chưa phải kho dấu ngã, dấu mũ có định vị.

Ví dụ `NGUYỄN` có sáu vị trí NFC: N, G, U, Y, Ễ, N. Native generator nhận sáu query từ sáu hình glyph và dựng ảnh cao 32px, rộng 6×16 = 96px. Đây là quy ước tổng chiều rộng; nét có thể nối qua vị trí ký tự.

VATr đã dùng hình glyph nên các chữ có thể chia sẻ bằng chứng thị giác. Không nên mô tả baseline là các class one-hot hoàn toàn không liên quan. Điều còn thiếu là cơ chế chia sẻ **vai trò** và/hoặc cơ chế dựng dấu được kiểm soát tường minh.

Bốn thay đổi dưới đây lần lượt can thiệp vào các vị trí khác nhau của đường đi trên.

## 4. Thuật toán A — chọn transcript theo thành phần/case

**Trạng thái: đã triển khai và đang chạy trong mainline.**

### 4.1. Trước đây chọn như thế nào?

Sampler cũ ưu tiên transcript chứa ký tự hiếm. Nó đếm nguyên ký tự: Ễ là một mã ký tự, ễ là mã khác, Ê là mã khác.

Với N phần tử corpus, U(t) là tập ký tự khác nhau trong transcript t và f(c) là số lần c xuất hiện trong toàn corpus:

```text
w_char(t) = min(20, trung bình sqrt(N / f(c)) trên c thuộc U(t))
```

Corpus có thể chứa nhiều phần tử mang cùng transcript vì các ảnh khác nhau có cùng nhãn. Vì thế “lấy đều theo corpus” không có nghĩa lấy đều theo từ điển các từ khác nhau.

### 4.2. Bổ sung gì?

Từ `Ễ`, sampler tạo bốn nhóm:

```text
base:e|upper
shape:circumflex|upper
tone:tilde|upper
pair:circumflex|tilde|upper
```

- Nhóm base e/upper chia sẻ giữa E, Ê, É, Ễ…
- Nhóm circumflex/upper chia sẻ giữa các biến thể hoa có mũ.
- Nhóm tilde/upper chia sẻ giữa các chữ hoa có ngã.
- Nhóm circumflex/tilde/upper biểu diễn việc mũ và ngã cùng xuất hiện trên chữ hoa.

Với một transcript, lấy hợp các nhóm của toàn bộ ký tự. Nhóm lặp chỉ tính một lần trong transcript đó. `NGUYỄN` có N hai lần, nhưng nhóm base n/upper không được đếm hai lần trong cùng phần tử corpus.

Gọi G(t) là tập nhóm, n(g) là số phần tử corpus chứa nhóm g:

```text
w_component(t) = min(20, trung bình sqrt(N / n(g)) trên g thuộc G(t))
w_new(t)       = 0,5 × w_char(t) + 0,5 × w_component(t)
```

### 4.3. Một lượt chọn văn bản thực hiện ra sao?

```text
1. Chọn corpus nguồn: kỳ vọng 65% real-writer, 35% HAT.
2. Trong corpus vừa chọn:
   - 70% lượt: chọn phần tử theo phân bố empirical.
   - 30% lượt: chọn phần tử với xác suất tỷ lệ w_new.
3. Lấy nguyên transcript của phần tử đó, ví dụ NGUYỄN.
4. Chuyển transcript sang query glyph và đưa qua generator native như cũ.
```

Các tỷ lệ là kỳ vọng xác suất, không phải ràng buộc cứng cho từng batch. Với một phần tử corpus:

```text
P(t | corpus) = 0,7/N + 0,3 × w_new(t)/tổng trọng số
```

**50/50 ở đây là trộn trọng số**, không phải thêm một vòng lấy mẫu có xác suất 50% độc lập.

### 4.4. Nó giúp bằng cách nào?

Giả thuyết là model được phân bổ thêm cơ hội tiếp xúc với những cấu trúc thành phần còn thiếu trong phân bố học. Các tham số G dùng chung có thể học tốt hơn từ những ví dụ liên quan.

Nhưng sampler không gửi các thành phần đó vào forward của G. Nếu lượt được chọn là NGUYỄN thì G vẫn nhận hình glyph nguyên chữ Ễ; không nhận ba lệnh “vẽ E, vẽ mũ, vẽ ngã”.

Nó cũng không bảo đảm tăng tần suất mọi chữ hiếm. Trong corpus HAT thật, xác suất transcript draw chứa Ễ đổi từ 0,13669% sang 0,13337%. Do đó thuật toán này là **cân bằng theo cấu trúc**, không phải “tăng lấy Ễ bằng mọi giá”. Phụ lục A có ví dụ số để thấy rõ.

### 4.5. Đã triển khai ở đâu và kết quả tới đâu?

- [ComponentCorpusSampler](../../util/vietnamese_longtail.py): tính nhóm, trọng số và phân bố lấy mẫu.
- [groups](../../research/longtail/protocol.py): định nghĩa các nhóm thành phần/case.
- [CorpusSampler gốc](../../data/vietnamese.py): phân bố empirical/rare.
- [Trainer native](../../train_vietnamese.py): cờ `--text-sampler component` và chọn corpus cho từng pha.

Pilot pretrained 175.500→176.500 cho CER 12,69% ở nhánh mới, 17,95% ở đối chứng và 13,63% ở nguồn. Đây là tín hiệu một-seed trong 1.000 outer steps, không chứng minh chất lượng thị giác, lợi ích dài hạn hay giải quyết chữ hoa hiếm. [Report pilot](261114_report_04_Finetune_pretrained_175500_den_176500.md).

## 5. Thuật toán B — query tách thành phần cho generator

**Trạng thái: đã có prototype và pilot riêng; chưa tích hợp vào run 250k.**

### 5.1. Ý tưởng

“Query” là vector mô tả nội dung mà generator cần viết. Baseline tạo query từ hình nguyên chữ. Prototype giữ query đó và bổ sung một nhánh mô tả cấu trúc:

```text
q_old = biểu diễn hình glyph Ễ

thân E + thông tin hoa ─┐
embedding mũ ───────────┼→ tương tác giữa các vai trò
embedding ngã ──────────┘              ↓
                             tham khảo style memory
                                       ↓
                                residual Δq

q_new = q_old + Δq
```

Embedding là vector học được dùng chung cho một thuộc tính. Ngã trong Ễ và Ỗ sử dụng cùng bảng embedding ngã; chữ hoa/thường và tương tác với thân vẫn có thể làm kết quả khác nhau.

### 5.2. Prototype hiện làm gì trong code?

1. Phân tích mỗi ký tự thành base/modifier/tone/case.
2. Lấy glyph của base theo case, ví dụ E, chiếu từ 512 xuống 128 chiều và cộng embedding case.
3. Tạo token modifier và tone, cộng gợi ý tọa độ vai trò.
4. Ba role token qua hai lớp Transformer tương tác, hidden 128, bốn attention heads.
5. Cross-attention tham khảo style memory của native G đã chiếu xuống 128 chiều.
6. Gộp ba token, chiếu residual trở lại 512 chiều và cộng vào query nguyên glyph.

Gợi ý tọa độ ở đây là feature cho token, **không phải vị trí pixel đã được dự đoán**. Padding không nhận residual.

### 5.3. Vì sao khởi tạo residual bằng zero?

Lớp chiếu cuối có weight và bias bằng zero. Khi mới lắp module:

```text
Δq = 0 → q_new = q_old
```

Mục tiêu là bảo toàn đầu vào nội dung đã học của checkpoint trước khi fine-tune, rồi học điều chỉnh dần. Zero-init không bảo đảm quá trình tối ưu sau đó sẽ tốt hơn; nó chỉ cung cấp điểm khởi đầu có kiểm soát. Pilot đã kiểm tra giữ nguyên 600 dự đoán OCR ở bước cài adapter ban đầu.

### 5.4. Nó khác sampler ở đâu?

Sampler đổi **văn bản nào được gặp**. Query adapter đổi **cách biểu diễn cùng văn bản đó**. Cả hai cùng xét thành phần nhưng có cơ chế tác động khác nhau.

Đây là bước có thể tạo sự chia sẻ tường minh để học Ễ từ các tổ hợp liên quan. Nó vẫn không bắt buộc một vùng pixel cụ thể phải là ngã hoặc mũ.

### 5.5. Đã đạt gì và chưa đạt gì?

[ComponentQuery và install_composition](../../research/vietcompose/model.py) đã có code. Pilot trước đóng băng G nguồn, chỉ học adapter 1.000 cập nhật, cho CER: nguồn 13,63%, factorized 15,38%, glyph adapter control 16,32%. Như vậy prototype chưa thắng nguồn; không được mô tả là cải thiện đã đạt.

Hướng tiếp theo là thử adaptation trên pretrained có kiểm soát: học adapter và một phần decoder, thay vì mặc định lặp lại cấu hình adapter-only đã không thành công. Đây là thiết kế thí nghiệm tiếp theo, chưa chạy trong mainline.

## 6. Thuật toán C — dựng ảnh bằng các lớp mực thân/modifier/tone

**Trạng thái: có module, kiểm thử và pilot exploratory; chưa thay CNN decoder native.**

### 6.1. Ý tưởng

Query tách thành phần vẫn để CNN dựng toàn bộ ảnh. Renderer này tiến thêm một bước: tạo riêng các lớp mực và ghép chúng.

```text
Body branch       → ảnh mực thân E ────────────────┐
Modifier branch   → patch mũ → đặt vị trí/kích thước ┼→ ghép mực → tinh chỉnh ở mức từ
Tone branch       → patch ngã → đặt vị trí/kích thước┘
```

Tách **hình dạng dấu** khỏi **hình học đặt dấu** giúp chia sẻ template ngã giữa nhiều chữ, trong khi cách đặt vẫn thay đổi theo thân, case và modifier.

### 6.2. Ràng buộc đang có trong code

- Body branch chỉ nhận base, case, vị trí và style; không nhận ID tone/modifier.
- Modifier/tone branch dùng các embedding chia sẻ và đặc trưng body.
- Không có modifier hoặc tone thì lớp mực tương ứng bị gate về zero.
- Patch dấu được đặt bằng phép biến đổi affine khả vi: có prior vị trí/kích thước và offset/scale học được, có giới hạn.
- Dấu nặng có prior dưới thân; mũ/trăng và tone phía trên có prior khác nhau.
- Các patch được nối thành từ rồi qua refiner với residual bị chặn biên độ 0,05 trong miền ảnh [-1,1].

Nếu B, M, T là độ phủ mực trong [0,1], phép ghép hiện dùng:

```text
Ink = 1 − (1−B)(1−M)(1−T)
Ảnh = 1 − 2×Ink
```

### 6.3. Những gì ràng buộc này chưa bảo đảm

Body không nhận tone có nghĩa đổi tone không trực tiếp đổi đầu vào body; không có nghĩa body đã chắc chắn không vẽ nét thừa. Tương tự, gate vắng dấu chưa đủ để đảm bảo dấu có mặt được vẽ đúng hình.

Geometry head hiện nhận đặc trưng có case; **chưa có bộ đo bounding box thân chữ từ ground truth**. Prior trên font không phải vị trí chuẩn cho mọi handwriting. Refiner cũng cần được kiểm tra xem có làm sai dấu hay che lấp lợi ích tách lớp không.

Các ô ký tự 16px và refiner nhỏ còn hạn chế spacing, kerning và nét nối. Một hệ thống chỉ dán các ô rời nhau không đủ cho mục tiêu chữ viết tay tự nhiên.

### 6.4. Cần giám sát gì để nó học đúng?

Các tín hiệu khả thi, phải kiểm tra riêng:

- Ảnh từ thật cùng transcript/writer, support khác target.
- Mask/box/anchor thân và dấu được gán nhãn, hoặc layer font tổng hợp có nguồn gốc rõ ràng.
- Loss có xét vùng dấu để dấu nhỏ không bị nền trắng lấn át.
- Loss nội dung và tín hiệu tự nhiên/phong cách giữ được chất lượng mức từ.

Global pixel loss giảm chưa chứng minh dấu đúng. Pixel pairing không khớp nét hoàn hảo giữa hai lần viết có thể làm mờ; label mask giả từ attention không được coi là ground truth.

Code: [LayeredInkComposer](../../research/vietcompose/renderer.py). Đây là renderer riêng; thay nó vào native VATr++ sẽ là thay đổi kiến trúc lớn, không phải load thêm một checkpoint tương thích sẵn.

## 7. Thuật toán D — OCR và CTC theo thành phần

**Trạng thái: có hai hướng code khác nhau. Cả hai đều không thay OCR của run 250k.**

### 7.1. Tại sao cần quan tâm OCR?

Generator nhận gradient nội dung từ OCR. Nếu OCR đọc “NGUYỄN” đúng nhờ ngữ cảnh nhưng không nhận biết cụm mũ–ngã khi đứng riêng, điểm OCR không đủ để chứng minh ảnh Ễ tốt.

Một ý tưởng là để tín hiệu giám sát chia sẻ giữa các chữ có cùng base/modifier/tone/case. CTC cho phép học transcript khi chưa biết vị trí chính xác của từng ký tự theo chiều ngang.

### 7.2. Hướng cũ: gộp xác suất từ OCR nguyên ký tự

```text
OCR gốc → xác suất các chữ nguyên vẹn
       → gộp theo base / modifier / tone / case
       → tính các CTC loss cho từng vai trò
```

Ví dụ xác suất ngã được cộng từ xác suất những ký tự có ngã. Cách này tận dụng OCR hiện có, nhưng bốn CTC có thể chọn những alignment khác nhau. Một vai trò có thể “đặt” bằng chứng ở frame này, vai trò khác ở frame khác; không chắc chúng cùng mô tả một chữ tại một vị trí.

Code: [ComponentCTC cũ](../../research/vietcompose/losses.py). Nó là một loss phụ từ OCR cũ, không phải một recognizer độc lập mới.

### 7.3. Hướng mới: ghép điểm vai trò trước một CTC chung

```text
Ảnh → CNN đặc trưng theo chiều ngang
          ├→ điểm base
          ├→ điểm modifier
          ├→ điểm tone
          └→ điểm case
                      ↓
          ghép thành điểm các ký tự hợp lệ
                      ↓
                 một CTC chung
```

Với một frame j, ký tự c có các thuộc tính b,m,t,k:

```text
score_j(c) = score_base_j(b)
           + score_modifier_j(m)
           + score_tone_j(t)
           + score_case_j(k)
```

Thêm một điểm blank riêng, normalize trên blank và các ký tự trong alphabet, rồi tính một CTC duy nhất. Như vậy mọi vai trò dùng chung alignment của nguyên ký tự.

Giá trị semantic `none` của dấu là một class thật trong factor head; nó không phải blank của CTC. Head base e, mũ, ngã và upper có thể được học qua các chữ khác rồi cung cấp điểm cho Ễ, kể cả khi nguyên tổ hợp có ít ví dụ.

### 7.4. Đối chứng và giới hạn

Có ba recognizer control:

| Head | Biểu diễn ký tự | Điểm cần lưu ý |
|---|---|---|
| Whole-character | Một hàng output cho mỗi ký tự | Hàng output của lớp giữ lại không có ví dụ dương; zero-shot bằng head này có giới hạn hiển nhiên |
| Factorized | Ghép điểm bốn vai trò | Chia sẻ thành phần, nhưng không mặc nhiên nhận dạng tốt hoặc định vị dấu đúng |
| Glyph prototype | So feature handwriting với embedding hình glyph chuẩn | Có đường nhận biết nhãn giữ lại nhờ canonical glyph; là semantic baseline mạnh hơn |

Output inventory hiện vẫn là alphabet đã biết, không phải nhận mọi Unicode mới ngoài charset. CNN của recognizer thử nghiệm khác OCR native, nên không thể thay trọng số vào OCR native mà bỏ qua tương thích kiến trúc.

Code: [FactorizedCTC / WholeCharacterCTC / GlyphPrototypeCTC](../../research/longtail/recognizer.py). Một diagnostic seed 742 cho tail F1 26,50% so whole-head 0%, nhưng đó là nhận dạng trên panel riêng, không phải bằng chứng generator/style tốt hơn. Không gộp với kết quả run 250k.

## 8. Điều kiện thực sự để học Ễ từ những chữ khác

“Có tách thành phần” mới là một điều kiện về thiết kế. Khả năng chuyển giao còn cần:

1. **Bằng chứng đủ về thành phần:** hình ngã, mũ và thân hoa có xuất hiện đủ trong dữ liệu liên quan không?
2. **Bằng chứng đủ về tương tác:** có ví dụ dấu ngã đi cùng mũ, đặc biệt trên chữ hoa không?
3. **Cách dùng phong cách:** có biết writer mục tiêu viết dấu thế nào hay đang dùng nét ngã trung bình toàn dataset?
4. **Giám sát đáng tin:** model có được thưởng cho dấu thật sự đúng, thay vì hoa văn mà OCR đọc được?
5. **Ngữ cảnh và hình học phù hợp:** bố trí dấu khi sinh chữ đơn có khác khi sinh cả từ không?

Nếu thiếu cả ảnh ngã của writer A và các writer có phong cách tương tự, không có cơ sở bảo đảm suy ra chính xác nét ngã riêng của A. Khi ấy hệ thống chỉ đang dự đoán từ prior và những đặc trưng phong cách đã quan sát.

## 9. Ý tưởng tiếp theo E — memory thành phần theo writer

**Trạng thái: đề xuất, chưa có component-local memory được triển khai.**

### 9.1. Vấn đề muốn xử lý

Style memory hiện tại chứa feature chung của support. Nếu support có chữ mang ngã, mô hình cần biết phần nào là ngã và nên tái sử dụng nét nào cho Ễ, thay vì phải tự tìm trong một chuỗi feature không gắn nhãn.

### 9.2. Thiết kế đề xuất

Mỗi entry memory cần chứa:

```text
(writer, vai trò, case, feature vùng mực,
 nguồn ảnh, ID ảnh, độ tin cậy định vị)
```

Quy trình:

1. Định vị thân và dấu trong support bằng annotation hoặc bộ định vị đã kiểm định.
2. Gom bằng chứng thành phần theo writer/case; không lấy ID ảnh HAT làm writer.
3. Query Ễ hỏi memory về thân hoa, mũ và ngã có liên quan.
4. Khi có bằng chứng phù hợp, attention dùng feature cục bộ.
5. Khi thiếu, trộn prior quần thể với style toàn cục bằng một gate; kiểm định gate trên episode có che thành phần.

Không được gọi một attention map chưa kiểm tra là crop dấu đúng. Điểm confidence của gate cũng cần calibration trước khi diễn giải như xác suất đúng.

### 9.3. Cách kiểm tra có thật sự dùng memory

So cùng một target trong ba điều kiện support: có đủ thành phần; có mũ nhưng thiếu ngã; thiếu cả dấu. Thêm control tráo feature ngã giữa writer hoặc dùng memory toàn cục. Nếu model không thay đổi hợp lý theo nguồn bằng chứng, chưa thể nói memory đã chuyển phong cách dấu.

## 10. Ý tưởng tiếp theo F — học chữ đơn từ nhiều ngữ cảnh

**Trạng thái: giả thuyết và thiết kế mới, chưa triển khai crop/alignment/distillation hay áp dụng vào training.**

### 10.1. Động cơ

Dữ liệu có Ễ trong từ nhưng không có transcript Ễ đơn. Generator và OCR đều có thể gặp lệch phân phối khi chuyển từ từ dài sang một ký tự. Tăng sampling các từ đang có chưa tạo ra trường hợp chữ đơn.

### 10.2. Hướng dùng ảnh thật HAT

```text
Ảnh thật của từ chứa Ễ + transcript
    → alignment ký tự có kiểm tra
    → crop giữ thân và toàn bộ dấu, có margin
    → kiểm tra nhãn/crop
    → dữ liệu bổ sung cho OCR chữ đơn
```

CTC forced alignment có thể cung cấp vị trí khởi đầu, nhưng alignment frame không phải bounding box ground truth. Receptive field của OCR, dấu chồng và nét chữ lân cận có thể làm crop sai. Cần xem panel crop, kiểm tra retention của mũ/ngã, không cắt chữ cạnh bên, và ghi mức bất định.

HAT crop có thể cung cấp tín hiệu **nội dung**. Nó không tạo ra nhóm writer đáng tin hoặc ảnh Ễ trong phong cách writer thật A.

### 10.3. Hướng context-to-grapheme self-distillation trên pretrained G

Một giả thuyết khác: dùng G nguồn đã sinh được từ để tạo tín hiệu cho dạng chữ đơn.

```text
G nguồn cố định + support A → NGUYỄN / DIỄN / LỄ / DỄ
    → định vị/crop chữ Ễ từ nhiều ngữ cảnh
    → lọc bằng chứng có kiểm tra hình dấu
    → hướng dẫn G học sinh Ễ đơn với cùng support A
```

Ở đây, ảnh teacher là **ảnh giả**, không phải ground truth mới. Không mặc định OCR đọc đúng cả từ thì crop Ễ đúng. Cần kiểm tra dấu thực sự hiện diện, so nhiều ngữ cảnh, và đánh giá độc lập. Nếu teacher sai hoặc crop lẫn chữ, distillation sẽ truyền sai đó sang student.

Thí nghiệm phải có teacher bất biến, nhánh không distill, ngân sách G update bằng nhau, log provenance của crop, và đánh giá không dùng riêng OCR đã chọn crop. Không tự ghép ý tưởng này vào run 250k đang chạy.

### 10.4. Khi nào mới nên bổ sung query chữ đơn?

Sau khi xác nhận bộ giám sát chấm chữ đơn đủ tin cậy, có thể thử một tỷ lệ query ký tự đơn bên cạnh transcript tự nhiên. Cần control tăng query chữ đơn thông thường, so với query chữ hiếm, để biết lợi ích đến từ độ dài/ngữ cảnh hay từ cấu trúc dấu. Đây sẽ là thay đổi sampling khác với thuật toán A hiện tại.

## 11. Ý tưởng tiếp theo G — bổ sung dữ liệu theo coverage

**Trạng thái:** bộ audit và ngân sách ảnh lồng nhau đã có; acquisition tối ưu coverage và dữ liệu uppercase mới chưa có.

Không chỉ đếm f(Ễ). Cần giữ riêng:

- Số lần ký tự xuất hiện.
- Số ảnh độc lập có chứa.
- Số writer thật cung cấp ví dụ.
- Số transcript/ngữ cảnh khác nhau.
- Độ phủ component×case và modifier×tone×case.

Chọn prompt thu thập theo phần thiếu trong ma trận coverage; giới hạn số ảnh mỗi writer để không nhầm nhiều ảnh một writer với phong cách đa dạng. So với cùng ngân sách chọn ngẫu nhiên. Đo các mốc 0/1/5/20 ảnh thật, dùng tập train lồng nhau và writer evaluation riêng.

Chữ đơn phục vụ chẩn đoán hình dạng; từ/ngữ cảnh phục vụ spacing và nét nối. Ký tự hợp lệ không có nghĩa mọi từ thay dấu tùy ý đều là âm tiết hợp lệ. Phải giữ hai loại bài kiểm tra riêng.

[Protocol và budget code](../../research/longtail/protocol.py) ghi ngân sách thiếu thay vì tự tạo mẫu hoặc xem dữ liệu đánh giá là dữ liệu train.

## 12. Cách bổ sung kiến trúc vào checkpoint pretrained ở thí nghiệm tiếp theo

**Đây là kế hoạch tích hợp, chưa phải code path của run 250k.** Ưu tiên giữ mô hình đã học thay vì xây lại generator.

Với query adapter nhỏ, trình tự thiết kế là:

```text
1. Chọn một checkpoint native cố định và ghi hash.
2. Khôi phục model và Adam native đúng cấu hình.
3. Lắp residual query adapter, giữ các Parameter gốc.
4. Chỉ thêm Parameter mới vào optimizer group mới.
5. Kiểm tra bước 0: query, ảnh/prediction và state tương thích.
6. Fine-tune theo lịch đã định, lưu architecture tag cho checkpoint mới.
```

Khi bọc `query_embed` bằng adapter, tên key state_dict có thể đổi. Không dùng `strict=False` tùy tiện rồi cho rằng đã nạp đủ pretrained. Cần loader/migration nhận biết architecture, kiểm tra key/shape và giữ trạng thái Adam cho các Parameter cũ. Param mới có Adam state mới là bình thường.

Một lịch ứng viên cần thử riêng là mở adapter và các block decoder cuối trước, rồi mới cân nhắc mở thêm backbone. LR và số bước phải chọn trên development; không có cơ sở nói lịch này đã tốt hơn. Style layout và positional encoding giữ nguyên trong phép thử đầu để tránh thay nhiều yếu tố cùng lúc.

Với renderer lớp mực, thay đổi cấu trúc lớn hơn nên không có cam kết tương thích toàn bộ decoder native. Cần một nghiên cứu migration/distillation riêng, không gọi việc khởi tạo một phần encoder là resume toàn model.

## 13. Thí nghiệm nào chứng minh được từng ý tưởng?

| So sánh | Giữ giống nhau | Câu hỏi được trả lời |
|---|---|---|
| Sampling cũ vs component sampling | Native checkpoint, loss, batch, LR, seed, updates | Phân bổ transcript có ích không? |
| Native vs glyph adapter vs factorized adapter | Cùng sampler, source, losses và ngân sách | Biểu diễn có ích hay chỉ do thêm module/compute? |
| Geometry có/không, chia sẻ case có/không | Cùng renderer và annotation budget | Hình học/case có cải thiện dấu không? |
| Memory cục bộ vs global, support đủ/thiếu dấu | Cùng target và số support | Có chuyển bằng chứng phong cách thành phần không? |
| Whole-head vs factorized vs glyph prototype OCR | Cùng ảnh và backbone, công bố số param | Giám sát có khái quát được tổ hợp không? |
| Có/không chữ đơn hoặc distillation | Khớp số cập nhật G và source | Lệch ngữ cảnh có phải nút thắt không? |
| Coverage acquisition vs random | Cùng số ảnh và số writer | Bổ sung dữ liệu có hiệu quả hơn không? |

Tách hai track:

- **Warm-start thực tế:** tiếp tục từ pretrained có exposure, như mainline hiện tại. Đánh giá cải thiện sau thích nghi.
- **Clean composition:** không có target handwriting combination trong image/OCR/corpus/support/memory/pretraining. Chạy sạch mới và khai báo quyền xem canonical glyph; không dùng checkpoint legacy đã thấy target để gọi zero-shot.

Mỗi kết quả quan trọng cần nhiều seed, công bố tham số/compute và khoảng tin cậy theo writer. Nhiều prompt của một writer hoặc nhiều mốc của một run không trở thành các lần chạy độc lập.

## 14. Đánh giá đúng và điều kiện dừng

Đánh giá ít nhất bốn mặt:

1. **Nội dung:** CER, exact, macro-F1 glyph hiếm, tách base/modifier/tone/case.
2. **Hình dấu:** có đúng mũ/ngã không, vị trí/scale có hợp lý không; dùng annotation/human chứ không chỉ OCR.
3. **Phong cách:** có giống writer và giữ khác biệt giữa writer không; kiểm tra cả nền giấy để tránh học shortcut.
4. **Chất lượng ảnh:** nét có tự nhiên, nền có sạch và khoảng cách có hợp lý không.

Thấy OCR tốt hơn nhưng ảnh nhiễu, nhòe hoặc mất style thì chưa đạt mục tiêu. Thấy L1 giảm do ảnh trắng cũng là failure. Nếu tín hiệu giám sát/crop không đáng tin, sửa hoặc giới hạn kết luận trước khi mở rộng ngân sách.

Prototype phân rã thành phần và local-style có liên hệ với prior art font generation đã nêu trong [đề xuất nghiên cứu](../vietcompose/RESEARCH_PROPOSAL.md). Không gọi thêm NFD embedding là một đóng góp mới toàn cục nếu chưa có đối chứng và rà soát đầy đủ.

## 15. Người đọc nên hiểu trạng thái hiện tại thế nào?

- **Đã làm và đang áp dụng:** component-aware transcript sampling trên native pretrained VATr++, cùng cơ chế resume/theo dõi/kiểm chứng.
- **Đã viết và thử riêng:** query adapter, renderer lớp mực, hai cách CTC thành phần và các semantic control; chưa có kết quả đủ để thay mainline.
- **Ý tưởng cho nghiên cứu tiếp:** local component memory, geometry có nhãn thật, chuyển từ ngữ cảnh sang chữ đơn, và acquisition theo coverage.

Mục tiêu cuối vẫn là học được tổ hợp hiếm như Ễ trong phong cách writer, nhưng không mô tả những ý tưởng này như thể model 250k hiện đã thực hiện tất cả.

## Phụ lục A — ví dụ số nhỏ của sampler

Ví dụ dưới đây chỉ minh họa code trên corpus giả lập gồm tám phần tử:

```text
E, Ê, Ễ, ễ, Ấ, Ỗ, ma, ma
```

Đây không phải số liệu corpus thật và các phần tử đơn không được coi là từ tự nhiên. Giữ rare fraction 0,3 và component_mix 0,5. Trọng số và xác suất tính trực tiếp bằng các class sampler:

| Transcript | Trọng số cũ | Trọng số mới | Xác suất chọn chuỗi cũ | Xác suất chọn chuỗi mới |
|---|---:|---:|---:|---:|
| E | 2.8284 | 2.5754 | 12.80% | 12.78% |
| Ê | 2.8284 | 2.3987 | 12.80% | 12.50% |
| Ễ | 2.8284 | 2.2951 | 12.80% | 12.34% |
| ễ | 2.8284 | 2.8284 | 12.80% | 13.17% |
| Ấ | 2.8284 | 2.6517 | 12.80% | 12.89% |
| Ỗ | 2.8284 | 2.4445 | 12.80% | 12.57% |
| ma (2 phần tử) | 2.0000 | 2.0000 | 23.22% | 23.75% |

Ví dụ cho thấy Ễ không nhất thiết được chọn nhiều hơn: một số thành phần của nó đã được chia sẻ với Ê/Ỗ. Ngược lại, ễ trong corpus giả lập này có các nhóm case/thành phần khác. Cơ chế này cần thí nghiệm để xác định phân bổ mới có ích hay không; tên “component-aware” không tự bảo đảm kết quả tốt.

## Phụ lục B — đường dẫn code và tài liệu

| Phần | Code/tài liệu |
|---|---|
| Parser bốn thuộc tính | [orthography.py](../../research/vietcompose/orthography.py) |
| Sampler đang chạy | [vietnamese_longtail.py](../../util/vietnamese_longtail.py) |
| Trainer pretrained | [train_vietnamese.py](../../train_vietnamese.py) |
| Query adapter | [model.py](../../research/vietcompose/model.py) |
| Renderer các lớp mực | [renderer.py](../../research/vietcompose/renderer.py) |
| Component-marginal CTC cũ | [losses.py](../../research/vietcompose/losses.py) |
| Joint factorized CTC và controls | [recognizer.py](../../research/longtail/recognizer.py) |
| Exposure/budget/grouping | [protocol.py](../../research/longtail/protocol.py) |
| Kiểm thử sampling | [test_pretrained_longtail.py](../../tests/test_pretrained_longtail.py) |
| Kiểm thử adapter/renderer | [test_vietcompose.py](../../tests/test_vietcompose.py) |
| Kiểm thử exposure/OCR/controls | [test_longtail.py](../../tests/test_longtail.py) |
| Báo cáo trạng thái chi tiết | [report 06](261114_report_06_Cai_tien_da_trien_khai_va_training_dang_chay.md) |
| Báo cáo training tự cập nhật | [report 05](261114_report_05_Continuation_component_den_250k.md) |

**Kiểm tra trạng thái khi viết tài liệu:** 14/11/2026 13:12:19 UTC; supervisor `running`, log gần nhất 186,600, mục tiêu 250,000. Đây là snapshot, không phải kết quả đã hoàn thành 250k.
