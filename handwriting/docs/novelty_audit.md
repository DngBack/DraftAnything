# Novelty audit: within-line dynamics cho sinh dòng chữ viết tay (IAM lines)

Ngày kiểm tra: 2026-10-31. Phương pháp: web search và đọc nguồn gốc (arXiv, CVF, GitHub API, Hugging Face API).
Ký hiệu: **[V]** là đã kiểm tra trên nguồn gốc. **[P]** là kiểm tra một phần (chỉ abstract/README hoặc nguồn thứ cấp). **[U]** là chưa kiểm tra được, dựa trên trí nhớ hoặc snippet, cần đọc lại trước khi trích dẫn.

Bối cảnh (không suy lại ở đây): residual hình học theo từ (baseline, inter-word gap, slant, stroke width) có tương quan lag-1 dương trong dòng (excess corr 0.05–0.26) sau khi kiểm soát content và hằng số dòng/writer. Phần lớn là drift tần số thấp (xu hướng tuyến tính theo x). Chỉ baseline còn thành phần cục bộ sau khi detrend tuyến tính. x-height thì không có.

---

## 1. Các công trình gần nhất

### 1a. Sinh dòng/đoạn chữ viết tay có style (offline)

| Công trình | Venue/năm | Mô hình gì | Có đo/mô hình hóa dependence/drift trong dòng? | Code + checkpoint dòng IAM |
|---|---|---|---|---|
| **TS-GAN**: Davis et al., *Text and Style Conditioned GAN for Generation of Offline Handwriting Lines* | BMVC 2020 | GAN sinh cả dòng. Style là 1 vector toàn cục (concat ở mọi vị trí + AdaIN). Có **spacing network** 1-D dự đoán số blank trước mỗi ký tự. Có "additive noise" trong mỗi block (kiểu StyleGAN, per-pixel iid). Bài viết nói rõ noise là "natural variation of individual handwriting". [V] | Không. Có mô hình spacing (giá trị marginal, có điều kiện theo style + text) nhưng không có trạng thái tương quan dọc dòng. Không nói gì về baseline. Đánh giá: FID, GS, human study (52.2% accuracy). [V] | Code: https://github.com/herobd/handwriting_line_generation . Release `w1.0` có `IAM_weights.tar.gz` (~545 MB) và `RIMES_weights.tar.gz` [V, GitHub API]. **Checkpoint dòng IAM: có.** |
| **CSA-GAN (?)**: Kang et al., *Content and Style Aware Generation of Text-line Images for Handwriting Recognition* | TPAMI 2022 (arXiv 2204.05539) | GAN sinh dòng dài, mục tiêu là augmentation cho HTR. [P] | Theo abstract thì không. [P] | Chưa tìm được checkpoint công khai [U]. |
| **DiffBrush**: Dai et al., *Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation* | ICCV 2025 | Latent diffusion sinh dòng. "Content-decoupled style learning" dùng **column-wise masking (vertical enhancing head)** cho "vertical alignment across words" và **row-wise masking (horizontal head)** cho word/char spacing. Discriminator mức dòng (3D CNN trên các segment) và mức từ. [V] | **Gần nhất về động cơ**: nói rõ rằng alignment dọc và spacing là thuộc tính style mức dòng. Nhưng đây là kỹ thuật học đặc trưng style (masking). Bài **không đo** autocorrelation hay drift, và không có biến trạng thái dọc dòng. Metrics: HWD, FID, IS, GS, DCER/DWER, user preference/plausibility. [V] | Code (MIT): https://github.com/dailenson/DiffBrush . Checkpoint "Pretrained DiffBrush" trên Google Drive/Baidu. README không ghi dataset, nhưng lệnh test dùng `configs/IAM.yml`, nên **nhiều khả năng là IAM lines** [P]. |
| **Emuru**: Pippi et al., *Zero-Shot Styled Text Image Generation, but Make It Autoregressive* | CVPR 2025 | VAE + AR Transformer. Train trên dữ liệu font tổng hợp (>100k font), sinh dòng độ dài tùy ý, zero-shot sang chữ viết tay. [V] | Không đo. Bài nói rằng FID/KID "not representative of aspects like word alignment", rằng ghép từ (word stitching) gây "misaligned baselines", và coi sinh mức dòng là cách giữ "word baseline alignment". [V] Tức là cộng đồng đã coi baseline coherence là chất lượng. | Code+weights: https://huggingface.co/blowing-up-groundhogs/emuru (`model.safetensors`) [V]. Không train trên IAM nhưng có đánh giá trên IAM lines (Table 2). Bài so sánh TS-GAN, HiGAN+, HWT, VATr, VATr++, One-DM, DiffPen **trên dòng** (với các model mức từ thì có lẽ bằng cách ghép từ [U]). |
| **Eruku**: *Autoregressive Styled Text Image Generation, but Make it Reliable* (arXiv 2510.23240) | WACV 2026 | Kế thừa Emuru: content adherence tốt hơn, có stop mechanism, guidance kiểu CFG. [P] | Không thấy đề cập [P]. | Code: https://github.com/Blowing-Up-Groundhogs/Eruku . HF: https://huggingface.co/blowing-up-groundhogs/eruku (Apache-2.0) [V tồn tại]. |
| **InkSpire**: Wang, Zhang, Yao, Yin, Liu, *Learning to Generate Stylized Handwritten Text via a Unified Representation of Style, Content, and Noise* | ICLR 2026 | DiT hợp nhất style/content/noise trong một latent. **Multi-line masked infilling** train trực tiếp trên ảnh nhiều dòng thô. Positional encoding sửa lại để sinh nhiều dòng độ dài tùy ý. Song ngữ EN/ZH. Đánh giá IAM + ICDAR2013, có user study. [P, abstract] | Không rõ. Không truy cập được PDF trên OpenReview (bị chặn bởi bot check) [U]. Cần đọc xem có metric về layout hay spacing không. | Không tìm thấy code/checkpoint công khai [U]. |
| **Paragraph-level LDM**: Mayr et al., *Zero-Shot Paragraph-level Handwriting Imitation with Latent Diffusion Models* (arXiv 2409.00786) | IJCV 2025 | LDM sinh đoạn văn. [P] | Không thấy [P]. | Code: https://github.com/M4rt1nM4yr/paragraph_handwriting_imitation_ldm . README có link Google Drive cho checkpoint (mức paragraph, train trên IAM + dữ liệu tổng hợp) [P]. |
| **Spatio-Temporal Handwriting Imitation**: Mayr et al. (arXiv 2003.10593) | ECCVW 2020 [U venue] | Offline → online → offline, mức dòng. Có Turing-style study. [P] | Không. | [U] |
| **VATr** (CVPR 2023) / **VATr++** (arXiv 2402.10798) | CVPR 2023 / preprint | Transformer mức từ. VATr++ đề xuất chuẩn hóa protocol đánh giá HTG. [P] | Không. | VATr: https://github.com/aimagelab/VATr . VATr++: https://github.com/EDM-Research/VATr-pp và https://huggingface.co/blowing-up-groundhogs/vatrpp (có `model.safetensors`, IAM-32) [V]. **Mức từ**, chỉ có dòng nếu ghép. |
| **HiGAN+** (Gan et al.) | ACM TOG 2022 | GAN mức từ, text tùy ý. | Không. | https://github.com/ganji15/HiGANplus . README nhắc `deploy_HiGAN+.pth` nhưng không thấy link tải [P]. |
| **DiffusionPen** (Nikolaidou et al.) | ECCV 2024 | LDM mức từ, few-shot style. Có chế độ "paragraph sampling" (ghép). | Không. | https://huggingface.co/konnik/DiffusionPen (`ema_ckpt.pt`, `train_word_IAM.pt`) [V]. **Mức từ.** |
| **One-DM** (Dai et al.) | ECCV 2024 | Diffusion one-shot mức từ. | Không. | https://github.com/dailenson/One-DM , checkpoint trên Google Drive/Baidu/wisemodel [V README]. **Mức từ.** |
| **SpiS-GAN** (Nguyen et al., arXiv 2607.06949) | preprint 07/2026 | Dựa trên **HiGAN** (BigGAN-style G) + Frequency Distribution Loss của FW-GAN, Star-Spiral blocks, Sobel edge loss. Ảnh **từ** 32×128 / 64×256, split IAM 339/161 writer theo HWT/VATr. Metrics: FID, KID, CER/NED/WER. [V, arXiv HTML] | Không. | https://github.com/DAIR-Group/SpiS-GAN và https://huggingface.co/DAIR-Group/SpiS-GAN (có `data/bestIAM.pth`, config iam_32/iam_64) [V]. **Mức từ.** |
| HCT-GAN, ScriptViT (2511.18307), WriteViT (2505.13235), FW-GAN (2508.21040), Dual Orthogonal Guidance (2508.17017), CA-DIFF (IJDAR 2026) | nhiều | Chủ yếu mức từ hoặc tiếng Trung. | Không thấy [U]. | không kiểm tra |

**Hệ quả quan trọng cho kế hoạch:** SpiS-GAN là **mô hình mức từ**, ảnh có kích thước cố định. Muốn "sinh dòng dựa trên SpiS-GAN" thì phải tự mở rộng sang dòng (ghép từ có spacing/baseline, hoặc chuyển backbone sang ảnh rộng). Reviewer sẽ hỏi tại sao không dùng backbone vốn đã sinh dòng (DiffBrush, Emuru, TS-GAN). Đây là rủi ro lớn về baseline mạnh (xem mục 3).

### 1b. Mô hình hóa tường minh biến thiên hình học trong dòng / "writing dynamics"

| Công trình | Venue/năm | Liên quan |
|---|---|---|
| **Graves**, *Generating Sequences with RNNs* (arXiv 1308.0850) | 2013 | Online, RNN+MDN trên offset bút. Trạng thái ẩn mang dynamics, nên dependence dọc dòng (kể cả drift baseline) **được mô hình hóa ngầm**. Không đo tường minh. [U chi tiết] |
| **DeepWriting**, Aksan et al. (arXiv 1801.08379) | CHI 2018 | VRNN tách content/style cho online, latent theo thời gian. [U] |
| **Decoupled Style Descriptors**, Kotani et al. (arXiv 2008.11354) | ECCV 2020 | Style descriptor **theo ký tự**, phụ thuộc ngữ cảnh. Đây là style biến thiên dọc chuỗi (online). [U chi tiết] |
| **Decoupling Layout from Glyph in Online Chinese Handwriting Generation** (arXiv 2410.02309) | ICLR 2025 | **Layout generator autoregressive (LSTM)** dự đoán bbox mỗi ký tự (cao, rộng, tâm dọc, offset ngang so với ký tự trước), có điều kiện trên ký tự trước. Đo layout bằng 8 đặc trưng hình học cho **cặp ký tự kề nhau** (∇1..∇8). Code: https://github.com/singularityrms/OLHWG [V]. **Rất gần ý tưởng "trạng thái hình học dọc dòng"**, tuy là online/tiếng Trung và không phân tích autocorrelation hay drift. |
| **BoundInk: Boundary-Aware Online Handwriting Generation**, Shin, Hong, Bak (arXiv 2604.02103) | preprint 2026 | Online EN (IAM-OnDB + BRUSH) và ZH. **Vertical Drift Loss** phạt sai lệch giữa offset dọc dự đoán và offset dọc thật của các ký tự kề nhau (centroid, top band, bottom band). Có metric spacing riêng (SSS cho inter-word, KGS cho gap giữa ký tự, dạng symmetric log-ratio) và blind human study. [V] Drift được coi là **lỗi cần giảm**. Metric là trung bình theo cặp kề, không phải phân tích dependence. |
| **Template-Guided Cascaded Diffusion for Stylized Handwritten Chinese Text-Line Generation** (Layout-Diff + Imitating-Diff) | Springer LNCS 2025 (có lẽ ICDAR 2025) [U] | Tách style layout mức dòng khỏi glyph. [P snippet] |
| **Haines, Mac Aodha, Brostow, *My Text in Your Handwriting*** | ACM TOG 2016 | Glyph-centric, "learned parameters for spacing, line thickness, and pressure" [P abstract]. Theo trí nhớ có mô hình hóa vertical offset/spacing giữa glyph [U]. Project: visual.cs.ucl.ac.uk/pubs/handwriting |
| **Varga & Bunke**, *Generation of synthetic training data for an HMM-based handwriting recognition system* (ICDAR 2003). *Perturbation models for generating synthetic training data in handwriting recognition* (2008, book chapter) | 2003–2008 | **Perturbation model cho cả dòng**: biến đổi hình học (shear, scaling, baseline bending, thinning/thickening). Theo trí nhớ, chúng được điều khiển bởi **các hàm biến thiên mượt theo x (dạng cosine)** [U, cần đọc bản gốc]. Nếu đúng thì đây chính là tiền thân của **B3 (fixed-smooth correlated field)** và là augmentation, không phải generation học được. **Bắt buộc trích dẫn.** |
| **Wigington et al.**, *Data Augmentation for Recognition of Handwritten Words and Lines Using a CNN-LSTM Network* | ICDAR 2017 | Random grid distortion (trường dịch chuyển cục bộ mượt) + profile normalization (chuẩn hóa slant/slope) [P]. |
| **Luo et al.**, *Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition* | CVPR 2020 | Augmentation hình học **học được** (điều khiển bởi các fiducial point). Là precedent cho "learned geometric field" phía HTR [U chi tiết]. |
| **ALIS**: Skorokhodov et al., *Aligning Latent and Image Spaces to Connect the Unconnectable* | ICCV 2021 | GAN với **latent anchor đặt dọc trục ngang rồi nội suy**, tức là trường latent liên tục theo x. Về kỹ thuật gần **B3/M1**. Ngoài ra InfinityGAN (ICLR 2022) dùng latent local + global [U chi tiết]. |
| StyleGAN noise inputs / SPADE / FiLM | 2018–2019 | Noise iid per-pixel (= B2) và modulation biến thiên theo không gian là kỹ thuật chuẩn. M1 "FiLM theo 1-D field" **không mới về kiến trúc**. |
| Marti, Messerli, Bunke, writer ID dùng text-line features trên IAM (ICDAR 2001) [U] | 2001 | Đặc trưng mức dòng (slant, khoảng cách từ, các vùng chiều cao). Chỉ là thống kê marginal. |
| Văn liệu motor control / micrographia (vd. progressive micrographia: chữ nhỏ dần dọc dòng, ngưỡng >10%) | nhiều | Drift kích thước dọc dòng đã được đo trong lâm sàng (Parkinson), và người khỏe cũng có biến thiên. **Nên trích dẫn** để định vị "drift" như một hiện tượng vận động đã biết. Phát hiện x-height **không** có drift trên IAM đáng được thảo luận so với văn liệu này. [P] |

**Kết luận 1b:** Phía **online** đã có mô hình tường minh cho quan hệ hình học giữa các ký tự kề nhau (OLHWG ICLR 2025, BoundInk 2026). Phía **augmentation** đã có trường biến dạng mượt dọc dòng từ 2003. Chưa tìm thấy công trình **offline image generation** nào (i) **đo** dependence hay drift của residual hình học theo từ trong dữ liệu thật sau khi kiểm soát content và writer, hoặc (ii) kiểm tra xem generator có tái tạo được nó không.

### 1c. Metric đánh giá ngoài FID/HWD/CER

| Metric / công trình | Ghi chú |
|---|---|
| **HWD**, Pippi et al., BMVC 2023 (arXiv 2310.20316) | Perceptual distance trên đặc trưng style, dùng được ở mức từ và mức dòng. Là metric chuẩn hiện nay. [P] |
| **Rethinking HTG Evaluation** (HTG_HTR, HTG_style, HTG_OOV), Nikolaidou et al. (arXiv 2409.02683) | Metric dựa trên mức hữu ích cho HTR và style. [P] |
| **GS (Geometry Score)** | Dùng trong TS-GAN và DiffBrush. Đây là topological, không liên quan đến hình học chữ. |
| **DCER/DWER**, ΔCER, ΔI-LPIPS (Emuru, style consistency trên ảnh dài) | ΔI-LPIPS gần với "consistency dọc dòng" nhất trong các metric offline, nhưng nó đo độ giống nhau chứ không đo dynamics. [V] |
| **BoundInk SSS/KGS** | Metric spacing tường minh, nhưng cho online, và là trung bình theo cặp. [V] |
| **OLHWG ∇1..∇8** | Metric hình học cho ký tự kề nhau, online. [V] |
| **Human studies** | TS-GAN (phân biệt thật/giả), DiffBrush (preference/plausibility), InkSpire (plausibility), Mayr 2020 (Turing-style). |

**Kết luận 1c:** Chưa thấy protocol offline nào đo **thống kê bậc hai dọc dòng** (autocorrelation hoặc spectrum của residual baseline/gap/slant/width) của ảnh sinh so với ảnh thật. Đây là khoảng trống rõ nhất.

---

## 2. Checkpoint mức dòng công khai cho H2 (generator có tái tạo dynamics thật không)

Sẵn sàng dùng (đã xác minh file tồn tại):
1. **TS-GAN (Davis 2020)**: `IAM_weights.tar.gz` tại https://github.com/herobd/handwriting_line_generation/releases/tag/w1.0 . Train trên dòng IAM. Nên kiểm tra split writer (có thể không phải LWI).
2. **DiffBrush (ICCV 2025)**: Google Drive https://drive.google.com/file/d/1EWzBmLtnQ42cTf3k_CYQ-nF3RXCb35I6 . Nhiều khả năng là IAM lines (config `IAM.yml`). Cần xác minh.
3. **Emuru (CVPR 2025)** và **Eruku (WACV 2026)**: HF `blowing-up-groundhogs/emuru`, `blowing-up-groundhogs/eruku`. Zero-shot, không train trên IAM. Đây là ca thú vị vì dynamics chỉ có thể đến từ prompt style.
4. **Paragraph LDM (Mayr, IJCV 2025)**: checkpoint Google Drive (README). Mức đoạn văn, có thể cắt thành dòng.

Baseline "ghép từ" (đối chứng âm tự nhiên cho H2): VATr/VATr++ (HF vatrpp), DiffusionPen (HF konnik), One-DM, SpiS-GAN (HF DAIR-Group). Ghép từ iid thì về cấu trúc không có dependence giữa các từ, trừ khi quy tắc ghép đưa vào.

Chưa có: InkSpire (không thấy code), Kang TPAMI 2022 [U], HiGAN+ (không thấy link checkpoint).

**Lưu ý thiết kế H2:** các checkpoint này dùng split khác nhau (nhiều model dùng split HWT/VATr 339/161 writer, không phải LWI). Khi đo trên writer test cần ghi rõ ảnh style reference đến từ đâu, để tránh rò rỉ train/test.

---

## 3. Đánh giá novelty thẳng thắn

### Có vẻ mới (chưa tìm thấy tiền lệ)
1. **Phép đo thực nghiệm**: dependence lag-1 của residual hình học theo từ trong dòng offline thật, sau khi kiểm soát content và hằng số line/writer, có replication, và phân rã thành drift tuyến tính và phần cục bộ (baseline), với x-height âm tính. Không thấy bài HTG/HTR nào làm điều này.
2. **Kiểm toán các generator hiện có (H2)**: SOTA (TS-GAN, DiffBrush, Emuru/Eruku, các baseline ghép từ) có tái tạo được cấu trúc bậc hai này không. Chưa ai làm.
3. **Metric/protocol dynamics** cho offline lines (so autocorrelation/spectrum hoặc độ dốc drift của residual giữa ảnh thật và ảnh sinh), tách riêng khỏi content (CER) và writer similarity (HWD). Phía online có metric theo cặp (BoundInk, OLHWG) nhưng đó là thống kê bậc một, không phải dependence.

### Đã bị cover phần lớn
- **"Spatially varying modulation / noise field"** về kiến trúc: StyleGAN noise (B2), ALIS/InfinityGAN (trường latent liên tục theo x, gần B3/M1), SPADE/FiLM. M1 không thể là đóng góp kiến trúc chính.
- **"Trường biến dạng hình học mượt dọc dòng"**: Varga & Bunke 2003 (perturbation dòng), grid distortion của Wigington 2017, Learn-to-Augment CVPR 2020. B3 gần như chính là các phương pháp này.
- **"Line-level style gồm spacing và vertical alignment"**: DiffBrush (ICCV 2025) có head riêng cho vertical alignment và spacing. TS-GAN có spacing network. Emuru nêu baseline alignment là động cơ cho sinh mức dòng. OLHWG và BoundInk mô hình hóa offset kề nhau tường minh (online).
- **Mô hình dynamics ngầm**: mọi mô hình AR (Graves 2013, Emuru/Eruku, OLHWG) về nguyên tắc có thể học dependence. Reviewer sẽ hỏi "vậy AR có làm được sẵn không?". Câu này phải được trả lời bằng số đo H2.

### Rủi ro chính
1. **"Hiệu ứng chỉ là baseline skew/drift tuyến tính, augmentation đã làm từ 2003."** Kết quả nội bộ đã cho thấy phần lớn là drift tuyến tính theo x. Một biến đổi skew/rotation ngẫu nhiên theo dòng (hoặc warp cosine kiểu Varga–Bunke) có thể tái tạo phần lớn hiệu ứng. **Biện pháp:** đưa "random per-line linear drift" (và warp mượt kiểu Varga–Bunke) làm baseline **B3'** tường minh. Chỉ claim phần vượt quá drift tuyến tính (baseline cục bộ, gap, slant) và sự **liên kết giữa các kênh** (cross-descriptor co-drift, nếu có). Nếu M1 không thắng B3' một cách đáng kể trên metric dynamics thì claim mô hình sụp.
2. **Kích thước hiệu ứng nhỏ** (excess corr 0.05–0.26). Reviewer sẽ hỏi "con người có nhìn ra không?". Cần human study có kiểm soát (thật vs B1 vs B3' vs M1), hoặc chứng minh hiệu ứng có ý nghĩa cho downstream (writer ID, verification, HTR robustness). Nếu không thì chỉ là thống kê.
3. **Backbone yếu/không phù hợp**: SpiS-GAN là mô hình mức từ (32/64 px, ảnh kích thước cố định), là preprint, dựa trên HiGAN. Một paper CVPR về dòng mà backbone không phải SOTA mức dòng (DiffBrush, Emuru, InkSpire) sẽ bị chê. **Biện pháp:** để module field là plug-in, và chứng minh trên ít nhất một backbone mức dòng mạnh (vd. DiffBrush có code+checkpoint) ngoài SpiS-GAN. Hoặc làm phân tích hậu kiểm (post-hoc) trên các generator SOTA.
4. **Diễn giải sai nguyên nhân**: dependence có thể đến từ pipeline segmentation/normalization của IAM (deslant/baseline estimate theo dòng, lỗi segmentation từ) chứ không phải từ hành vi viết. Cần control: đo trên CVL hoặc RIMES lines, đo bằng hai bộ ước lượng hình học độc lập, và kiểm tra trên ảnh được render lại bằng font (phải ra ≈0).
5. **Split mismatch**: kết quả dùng LWI (train+val1), còn hầu hết checkpoint dùng split HWT/VATr. Phải chuẩn hóa khi so sánh H2.
6. **InkSpire chưa đọc được bản đầy đủ**: nếu nó có metric layout/spacing hoặc phân tích biến thiên dọc dòng thì novelty của mục 1c giảm. Cần đọc PDF ICLR 2026 trước khi chốt.
7. **BoundInk Vertical Drift Loss** (2026): reviewer có thể coi M2 (relational objective trên chuỗi descriptor) là phiên bản offline của VDL. Cần phân biệt: VDL khớp offset kề nhau với GT (cần paired GT, có tính chất supervised), còn M2 khớp **thống kê phân phối bậc hai** (unpaired).

### Framing có khả năng sống sót ở CVPR/ICCV/ECCV
**Khuyến nghị: "measurement study + benchmark + model đơn giản"**, không phải "pure model".

> *"Handwritten lines are not i.i.d. words: measuring, auditing and generating within-line writing dynamics."*

1. **Phân tích (đóng góp chính)**: định nghĩa residual hình học theo từ, có kiểm soát content và writer. Phân rã drift/local. Replication (val1, và lý tưởng là CVL/RIMES). Có negative control (x-height, render từ font).
2. **Benchmark/metric (đóng góp chính)**: "dynamics score" (khớp autocorrelation/spectrum, khớp phân phối độ dốc drift) để bổ sung cho FID/HWD/CER. **Kiểm toán SOTA** (TS-GAN, DiffBrush, Emuru, Eruku, các baseline ghép từ). Kết quả kỳ vọng: model mức từ ghép lại thì ≈0 dependence, model mức dòng thì ở một mức nào đó. Kết quả này có giá trị dù kết quả nào xảy ra (positive và negative đều publish được).
3. **Model (đóng góp phụ, phải tối giản)**: variation field như plug-in, so với ladder B1/B2/B3/B3'(random linear drift)/M1/M2, trên ≥2 backbone, trong đó có ≥1 backbone mức dòng.
4. Human study + một downstream (writer verification hoặc HTR robustness) để chứng minh hiệu ứng có ý nghĩa.

Framing "pure model" (FiLM field + relational loss trên SpiS-GAN) gần như chắc chắn bị đánh giá là incremental: kiến trúc đã có (ALIS/SPADE/StyleGAN noise), augmentation đã có (Varga–Bunke), line-level style đã có (DiffBrush). Venue thực tế hơn cho framing đó là ICDAR/BMVC/WACV.

---

## 4. Việc cần làm tiếp
- Đọc bản đầy đủ: InkSpire (ICLR 2026), Varga & Bunke 2003/2008 (xác minh dạng hàm cosine), Haines TOG 2016 (mô hình spacing/baseline), Kang TPAMI 2022, phụ lục F của BoundInk.
- Tải `IAM_weights.tar.gz` (TS-GAN) và checkpoint DiffBrush, kiểm tra split, rồi chạy pilot H2 trên ~200 dòng.
- Thêm baseline B3' = random per-line linear drift (+ warp Varga–Bunke) vào ladder.

## Nguồn chính
- TS-GAN: https://arxiv.org/abs/2009.00678 , https://github.com/herobd/handwriting_line_generation
- DiffBrush: https://arxiv.org/abs/2508.03256 , https://openaccess.thecvf.com/content/ICCV2025/html/Dai_Beyond_Isolated_Words_Diffusion_Brush_for_Handwritten_Text-Line_Generation_ICCV_2025_paper.html , https://github.com/dailenson/DiffBrush
- Emuru: https://arxiv.org/abs/2503.17074 , https://huggingface.co/blowing-up-groundhogs/emuru
- Eruku: https://arxiv.org/abs/2510.23240 , https://aimagelab.github.io/eruku-website/
- InkSpire: https://iclr.cc/virtual/2026/poster/10010623 , https://mlanthology.org/iclr/2026/wang2026iclr-learning-a/
- DiffInk (online lines, ICLR 2026): https://arxiv.org/abs/2509.23624 , https://github.com/awei669/DiffInk
- OLHWG: https://arxiv.org/abs/2410.02309 , https://github.com/singularityrms/OLHWG
- BoundInk: https://arxiv.org/abs/2604.02103
- Paragraph LDM: https://arxiv.org/abs/2409.00786 , https://github.com/M4rt1nM4yr/paragraph_handwriting_imitation_ldm
- Kang et al. TPAMI: https://arxiv.org/abs/2204.05539
- VATr++: https://arxiv.org/abs/2402.10798 , https://huggingface.co/blowing-up-groundhogs/vatrpp
- DiffusionPen: https://github.com/koninik/DiffusionPen , https://huggingface.co/konnik/DiffusionPen
- One-DM: https://github.com/dailenson/One-DM
- HiGAN+: https://github.com/ganji15/HiGANplus
- SpiS-GAN: https://arxiv.org/abs/2607.06949 , https://github.com/DAIR-Group/SpiS-GAN , https://huggingface.co/DAIR-Group/SpiS-GAN
- HWD: https://arxiv.org/abs/2310.20316 . Rethinking HTG Evaluation: https://arxiv.org/abs/2409.02683
- Varga & Bunke (perturbation): https://worldscientific.com/doi/abs/10.1142/S0218001404003666 (bản journal liên quan) [U chi tiết]
- Wigington et al. ICDAR 2017: https://www.aminer.cn/pub/5a9cb5f517c44a376ffb32fd
