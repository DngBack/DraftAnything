# Giai đoạn 0 của PartScribe: tổng quan tài liệu và protocol giữ lại tổ hợp

- Ngày: 18/11/2026. Kế thừa [261118_plan_03](261118_plan_03_Kien_truc_moi_bieu_dien_thanh_phan_va_tron_style.md), mục 6, giai đoạn 0.
- Điều kiện đi tiếp của giai đoạn 0: chưa có công trình nào làm chuyển giao thành phần cho HTG offline, và protocol giữ lại tổ hợp đã được chốt trước khi viết code mô hình.
- Kết luận: **đạt có điều kiện.** Không tìm thấy công trình nào chiếm trước claim. Còn hai lỗ hổng về nguồn đọc (mục 1.4).
- Artifact:
  - [partscribe/holdout_protocol_v1.json](partscribe/holdout_protocol_v1.json), SHA-256 `e4729308e8c068751db64bb9c19da6a70dc7a69d8dc3d9173a7c2144cbe3b12b`;
  - script [scripts/holdout_protocol_stats.py](../scripts/holdout_protocol_stats.py).

## 1. Tổng quan tài liệu

### 1.1. Ba công trình được yêu cầu đọc

| Công trình | Bài toán | Nội dung (ký tự) | Style và cách trộn | Có kiểm tra ký tự/thành phần chưa thấy? | Ảnh hưởng tới PartScribe |
|---|---|---|---|---|---|
| **InkSpire** (Wang, Zhang, Yao, Yin, Liu; ICLR 2026) | Ảnh offline, dòng nhiều dòng, Anh và Trung | Ghép latent nội dung `Xc` với latent ảnh `X`. Nguồn đọc không nói rõ `Xc` là gì | Không dùng encoder riêng. Style, nội dung và nhiễu chung một latent, học bằng masked infilling. Có positional encoding sửa đổi cho phép sửa từng ký tự | **Không** thấy trong tóm tắt và trang review | Baseline mạnh nhất trên IAM: FID 7,92, HWD 0,62, so với DiffusionPen FID 12,89, HWD 2,13. PartScribe không thể thắng chỉ nhờ kiến trúc trộn style. Đóng góp phải là phép thử chuyển giao |
| **CASHG** (Shin, Hong, Bak; arXiv 2604.02103, 4/2026) | Online (quỹ đạo bút), câu, Anh và Trung | Embedding CANINE theo codepoint, mỗi ký tự độc lập. Thêm bộ nhớ ngữ cảnh câu và cửa sổ bigram | Hai bộ nhớ style: writer và glyph (theo ký tự). Trộn bằng cross-attention cộng cổng theo token | **Không.** Bài nói thẳng là không có chia sẻ theo bộ phận/thành phần. Độ bền với tổ hợp hiếm chỉ là lập luận, không có test riêng | Ý tưởng tách writer/glyph style đã có. Style theo **thành phần** dùng chung giữa các ký tự thì chưa. Nghiên cứu H1 của ta cũng cho thấy độ phủ bigram trong support không ảnh hưởng |
| **MX-Font++** (Wang, Sun, Zhang, Gao; ICASSP 2025) | Font, không phải chữ viết tay | Expert trích feature cục bộ, **có giám sát bằng nhãn thành phần** (content classifier so với `Comp_gt`) | Tách content/style bằng expert dị thể và loss đồng nhất content-style | Có ký tự chưa thấy (UFUC) và chuyển ngữ tiếng Uyghur/Kazakh, nhưng chỉ cho glyph font cô lập | Ý tưởng style cục bộ theo thành phần **đã có ở font**. Khác biệt của ta: chữ viết tay liền nét theo từ/dòng, writer thật, không nhãn thành phần |

### 1.2. Các công trình gần khác đã rà

| Nhóm | Công trình | Ghi chú |
|---|---|---|
| HTG offline | VATr/VATr++ (CVPR 2023, 2024) | Archetype Unifont cho phép sinh **một số** ký tự chưa thấy, chủ yếu đánh giá định tính. Đây là đối chứng trực tiếp: report 261118_02 cho thấy VATr++ thay dấu sai ở chữ hoa hiếm |
| HTG offline | HWT, Emuru, DiffusionPen, DiffBrush, One-DM | Chỉ kiểm tra OOV **từ**, không kiểm tra ký tự/tổ hợp giữ lại có đối chứng |
| HTG online | SDT, DiffInk (ICLR 2026) | Style writer + ký tự, OCR/style loss. Không có test thành phần giữ lại |
| Chữ Hán theo bộ thủ | RD-GAN (ECCV 2020), Skeleton & Font Generation (2025), DDPM sinh dữ liệu chữ chưa thấy | Phân rã bộ thủ có nhãn, chữ cô lập, chủ yếu font |
| Tiếng Việt | WriteViT, FW-GAN, SpiS-GAN, CONSTANT | Nêu độ khó của dấu chồng. Tóm tắt không thấy test tổ hợp giữ lại. **Chưa đọc toàn văn** |

### 1.3. Khoảng trống còn lại (claim của paper)

Chưa có công trình HTG offline nào:
1. giữ lại có kiểm soát **toàn bộ ảnh** của một số tổ hợp ký tự mà mọi thành phần vẫn còn trong train;
2. đo độ đúng và độ giống writer của chính các tổ hợp đó;
3. chứng minh lợi ích đến từ biểu diễn thành phần **học không nhãn**, so với archetype nguyên ký tự (VATr) và encoder ảnh glyph ngầm (CNN).

Bài đề xuất **về phương pháp** gần nhất là MX-Font++, nhưng đó là font và dùng nhãn thành phần. Bài gần nhất **về bài toán** là VATr, nhưng đó là archetype nguyên ký tự và đánh giá định tính.

### 1.4. Giới hạn của lần rà này

- **Chưa đọc được toàn văn InkSpire.** OpenReview chặn truy cập tự động, và bài không có bản arXiv hay code công khai. Thông tin trên lấy từ abstract, trang ICLR và trang review. Trước khi nộp bài cần đọc PDF bằng tay, nhất là phần chỉnh sửa từng ký tự và RQ6 (điều khiển style/nội dung độc lập).
- Các bài tiếng Việt và DiffBrush mới chỉ đọc tóm tắt.
- Công trình xuất hiện sau 11/2026 cần rà lại trước khi nộp.

## 2. Protocol giữ lại tổ hợp, phiên bản 1 (đã chốt)

### 2.1. Tập giữ lại tiếng Việt

Chọn chữ thường có ảnh thật ở val/test, để đo được cả độ giống writer. Mọi dấu thành phần vẫn còn trong train qua ký tự khác. Số liệu dưới đây đếm trực tiếp từ HDF5; `train.h5` và `test.h5` khớp SHA-256 trong [repro/data_fingerprints.json](../repro/data_fingerprints.json).

| Loại | Ký tự | Ý nghĩa | Ảnh train bị loại | Val / test còn để đánh giá |
|---|---|---|---:|---:|
| **C1:** dấu thanh đã thấy, thân chữ mới | è é ẻ ẹ ỷ | Dấu có trên nguyên âm khác, chưa từng có trên thân này | 762 | 98 / 155 |
| **C2:** cặp dấu chồng đã thấy ở thân khác | ễ ỡ ử | ^+~ có ở ỗ/ẫ (657 ảnh), móc+~ có ở ữ (759), móc+hỏi có ở ở (893) | 490 | 66 / 147 |
| **C3:** cặp dấu chồng chưa từng thấy | ẳ | Trăng+hỏi không xuất hiện ở ký tự nào khác (0 ảnh). Trăng (2.358) và hỏi (8.216) có riêng lẻ | 108 | 19 / 31 |

- Tổng ảnh có chứa ký tự giữ lại: train 1.360/97.734 (1,39%, 228 writer), val 183 (27 writer), test 333.
- HAT 2.695 ảnh và mọi dòng trong corpus prompt chứa ký tự giữ lại cũng bị loại khỏi mọi thứ mà mô hình sinh được thấy.
- ẽ, õ và các dấu khác của e/o vẫn ở lại train. Mục đích là để dấu ngã trên e/o có nguồn học.
- C3 chỉ có một ký tự, nên số liệu của nó là thăm dò, không dùng cho kết luận chính.

**C4, chữ hoa có dấu (42 ký tự, 0 ảnh thật tự nhiên):**
- Chỉ đo được độ đọc đúng, bằng evaluator độc lập và người đọc.
- Không đo được độ giống writer, cho đến khi có ảnh thu theo [kế hoạch bổ sung ảnh](261118_report_Do_phu_ky_tu_tieng_Viet_tap_huan_luyen.md#kế-hoạch-bổ-sung-ảnh-thật).
- Ảnh thu mới cho nhóm này phải để riêng cho **đánh giá**, không đưa vào train. Nếu không, C4 mất giá trị zero-image.

### 2.2. Tập giữ lại IAM

`j q J Y`:
- j = chấm của i + nét móc dưới của g/y;
- q = vòng của a/d + đuôi của p/g;
- J, Y là chữ hoa hiếm, kiểm tra cùng hiện tượng exposure như tiếng Việt.

Loại 435/44.421 ảnh train_val (192 writer). Test còn 250 ảnh (91 writer) để đánh giá. Với tiếng Anh, lập luận "thành phần" yếu hơn, nên IAM là phép thử **tính tổng quát**, không phải phép thử chính.

### 2.3. Phạm vi loại bỏ (bản nghiêm ngặt, là bản chính)

| Thành phần | Được thấy ký tự giữ lại? |
|---|---|
| Ảnh thật dùng cho G, D, writer head | Không |
| Prompt văn bản khi train (corpus, transcript HAT) | Không |
| OCR giám sát dùng trong train | Không. Huấn luyện lại trên dữ liệu đã lọc |
| Glyph font/Unifont của ký tự (tokenizer giai đoạn 1, archetype VATr) | **Có.** Biết hình dạng chuẩn là điều kiện bài toán, và mọi phương pháp đều được thấy như nhau |
| Evaluator OCR, writer-ID, HWD | Có. Huấn luyện trên train + HAT **đầy đủ**, không bao giờ trên val/test, và tách khỏi mọi mô hình sinh |

Mọi mô hình so sánh đều được **train lại** dưới cùng phạm vi loại bỏ:
- VATr++ (archetype nguyên ký tự);
- VATr++ với GlyphCNN (encoder ảnh glyph ngầm, theo plan 261117_03);
- baseline nội bộ DiT với token ký tự;
- PartScribe.

IAM parity dùng các bản train trên dữ liệu đầy đủ, so với số công bố.

### 2.4. Đánh giá

- **Prompt:**
  - (a) Transcript thật của ảnh val/test có ký tự giữ lại, dùng để đo độ đúng và HWD cùng writer;
  - (b) Danh sách từ hợp lệ có ký tự giữ lại, lấy từ từ điển, dùng để đo độ đúng.
- **Support:** 15 ảnh của **cùng writer**, không chứa ký tự giữ lại, chọn bằng seed cố định. Writer nào không đủ 15 ảnh sạch thì ghi vào log và loại khỏi đánh giá, không lấp bằng writer khác.
- **Chỉ số chính (P1):** tỉ lệ ký tự đúng tại vị trí ký tự giữ lại. Evaluator OCR đọc; so sánh theo vị trí sau khi căn chỉnh chuỗi.
- **Chỉ số phụ:**
  - tỉ lệ thay bằng anh em cùng thân (thước đo từ report 261118_02);
  - HWD theo từng loại C1–C3, so với ảnh thật cùng writer;
  - writer-ID top-1 trên ảnh sinh;
  - người đọc chấm mù trên một mẫu con.
- **Thống kê:** khoảng tin cậy 95% bằng bootstrap theo writer, với 3 seed huấn luyện cho mỗi phương pháp. Mọi chỉ số được báo cáo **riêng cho từng loại** C1/C2/C3/C4 và IAM, không gộp.
- **Chia val/test:** val dùng cho phát triển và chọn checkpoint. Test chỉ mở **một lần**, khi đã chốt mô hình cuối.

### 2.5. Tiêu chí quyết định (định trước, chỉ sửa trước khi chạy giai đoạn 3)

1. **Giả thuyết chính được ủng hộ** nếu PartScribe hơn baseline DiT token ký tự **và** hơn VATr++ GlyphCNN ở P1 trên C1 ∪ C2:
   - chênh ≥ 10 điểm phần trăm;
   - CI 95% không chứa 0;
   - HWD C1 ∪ C2 không tệ hơn.
2. **IAM parity:** trên split chuẩn, HWD không tệ hơn baseline nội bộ quá 5% tương đối, và CER evaluator không tệ hơn quá 1 điểm phần trăm.
3. **Nếu (1) không đạt:** ghi kết quả null. Không đổi tập giữ lại, ngưỡng hay chỉ số sau khi đã thấy kết quả val.

## 3. Việc tiếp theo

1. Đọc toàn văn InkSpire bằng tay để đóng lỗ hổng ở mục 1.4.
2. Viết bộ lọc dữ liệu đọc `holdout_protocol_v1.json` cho loader VN/IAM và corpus prompt, kèm test kiểm tra không lọt ký tự giữ lại.
3. Train evaluator OCR độc lập trên dữ liệu đầy đủ, đo CER trên ảnh thật của tập giữ lại làm chặn trên.
4. Giai đoạn 1: part tokenizer trên font.

## Nguồn

- InkSpire: [ICLR 2026 poster](https://www.iclr.cc/virtual/2026/poster/10010623), [ML Anthology](https://mlanthology.org/iclr/2026/wang2026iclr-learning-a/), [review](https://liner.com/review/learning-to-generate-stylized-handwritten-text-via-a-unified-representation)
- CASHG: [arXiv 2604.02103](https://arxiv.org/html/2604.02103v1)
- MX-Font++: [arXiv 2503.02799](https://arxiv.org/html/2503.02799). MX-Font: [ICCV 2021](https://ar5iv.arxiv.org/html/2104.00887). LF-Font: [arXiv 2112.11895](https://arxiv.org/pdf/2112.11895)
- VATr: [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Pippi_Handwritten_Text_Generation_From_Visual_Archetypes_CVPR_2023_paper.pdf). VATr++: [arXiv 2402.10798](https://arxiv.org/html/2402.10798v1). Emuru: [arXiv 2503.17074](https://arxiv.org/pdf/2503.17074). HWT: [arXiv 2104.03964](https://arxiv.org/pdf/2104.03964v1)
- DiffInk: [arXiv 2509.23624](https://arxiv.org/abs/2509.23624v3). DiffBrush: [arXiv 2508.03256](https://arxiv.org/abs/2508.03256). SDT: [arXiv 2303.14736](https://arxiv.org/abs/2303.14736)
- RD-GAN: [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510154.pdf). Skeleton & Font Generation: [arXiv 2501.08062](https://ar5iv.labs.arxiv.org/html/2501.08062)
- Tiếng Việt: [WriteViT](https://arxiv.org/html/2505.13235v1), [FW-GAN](https://arxiv.org/html/2508.21040v1), [CONSTANT](https://arxiv.org/html/2603.07543), [SpiS-GAN](https://huggingface.co/papers/2607.06949.md)
