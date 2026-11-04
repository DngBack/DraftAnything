# Phương pháp — PopEvidence (bản rõ ràng, 2026-10-30)

Tài liệu này mô tả chính xác những gì code thực hiện, công thức của từng thành phần, chỗ cài đặt, và trạng thái bằng chứng của từng phương pháp. Số liệu chi tiết nằm trong `docs/findings.md` (F1–F15); phần rà soát code ở `docs/code_review.md`.

---

## 1. Bài toán

- Một **mẫu** c có nhãn tập y_c ∈ {0,1}^K: lớp k có mặt hay không. Không có nhãn cho từng tế bào hay từng tile.
- Mỗi ảnh được cắt thành T tile. Mỗi tile được mã hóa bằng **DINOv2-S/14 đông cứng** thành z ∈ R^384 (L2-normalised).
- Một **bag** gồm n ảnh của cùng một mẫu, tức n·T tile. Đầu ra là điểm s_k ∈ [0,1] cho từng lớp.

| Dataset | Lớp | Tile mỗi ảnh | Đơn vị thống kê | Loại dịch chuyển khi test |
|---|---|---|---|---|
| PHOEBI (phoebi6) | 6 loài vi khuẩn | 16 (lưới 4×4, 224 px, hiệu chỉnh chiếu sáng của PHOEBI) | combination (1 video) | Tổ hợp loài chưa thấy (LCO, 3 partition) |
| phoebi4 | 4 loài | 16 | video | Batch nuôi cấy và phiên chụp mới |
| BBBC041 | 5 loại tế bào máu | 35 (lưới 5×7) | ảnh | Thiết bị chụp khác |
| PanNuke | 5 loại nhân | 16 (64 px phóng lên 224) | ảnh | Cùng phân phối (fold 3) |

Bounding box (BBBC041) và mask (PanNuke) **chỉ dùng để đánh giá định vị**, không dùng khi train.

Code: `popev/features.py`, `scripts/extract.py`, `scripts/{bbbc041,pannuke,phoebi4}_prepare.py`.

---

## 2. Decoder (baseline và kiến trúc)

Mọi decoder nhận bag x ∈ R^{B×T×D} và trả về logit e ∈ R^{B×K}. Code: `popev/models.py`, `popev/pooling.py`.

| Tên | Công thức | Ghi chú |
|---|---|---|
| `mean_lin` | e = W·mean_t(z_t) + b | Head tuyến tính trên embedding trung bình; đầu vào chuẩn hóa theo train |
| `meanstd_lin` | e = W·[mean_t z; std_t z] + b | Thêm độ lệch chuẩn |
| `meanquant_lin`, `kme_lin` | quantile trên PCA-32; kernel mean embedding (RFF 2048, bandwidth = median heuristic) | Probe phân phối |
| `deepsets` | e = ρ(mean_t φ(z_t)), φ và ρ là MLP 256 | Baseline set |
| `abmil` | Gated attention theo từng lớp: ℓ_{t,k} = w_kᵀ(tanh(V h_t) ⊙ σ(U h_t)); α = softmax_t(ℓ); e_k = c_kᵀ Σ_t α_{t,k} h_t + b_k | Ilse et al. 2018, một đầu attention cho mỗi lớp. **Tile evidence = ℓ_{t,k}** |
| `popevidence` | a_{t,k} = evidence head(MLP(z_t)) ∈ R^m; e_k = w_kᵀ pool_t(a_{t,k}) + b_k | Evidence riêng cho từng lớp; pool = mean / std / quantile / max; tile evidence = a_{t,k,0} |
| `protomatch` | s_k = mean_t cos(z_t, p_k); p_k = mean tile của pure culture k (hoặc hybrid nếu lớp không có pure culture) | Tái cài ProtoMatch của PHOEBI; không học gì từ mixture |

**Huấn luyện:**
- Set model: AdamW, lr 1e-3, weight decay 1e-2, 2000 bước (1500 cho BBBC041), batch 64 bag, cosine schedule.
- Sampler: lấy combination đều nhau, rồi n ảnh không lặp.
- Linear head: AdamW full-batch, lr 1e-2, 500 bước, tối đa 600 bag mỗi combination.
- Threshold: argmax-F1 theo từng lớp trên tập calibration in-distribution.

Code: `popev/train.py`.

---

## 3. Ràng buộc tile (tile constraints)

**Ý tưởng:** chỉ đưa vào giám sát ở cấp tile những điều mà nhãn tập cho phép kết luận, độc lập với việc lớp nào khác có mặt.

Ký hiệu t_{b,i,k} là tile evidence (kênh 0 của PopEvidence, hoặc attention logit của ABMIL).

1. **Âm chính xác (luôn hợp lệ):** y_{b,k} = 0 ⇒ không tile nào có lớp k.
   L⁻ = mean_i softplus(t_{b,i,k}).
2. **Dương MIL:** y_{b,k} = 1 ⇒ ít nhất một phần tile có lớp k.
   L⁺ = softplus(−quantile_q(t_{b,·,k})).
   - **q = 1.0 (max): hợp lệ về nhãn** ("ít nhất một tile").
   - q = 0.9: giả định khoảng 10 % số tile có lớp k. Đây là một giả định, không suy ra được từ nhãn.
3. **Dương "pure" (tùy chọn `tile_pure`):** nếu bag chỉ có một nhãn, mọi tile đều dương.
   L^pure = mean_i softplus(−t_{b,i,k}).
   - Hợp lệ với **pure culture thật** (PHOEBI).
   - Với BBBC041 và PanNuke đây là giả định **dương có nhiễu**: ảnh một nhãn vẫn có tile chỉ chứa hồng cầu hoặc nền.

**Loss tổng:** L = λ_bce · BCE(e, y) + λ_tile · mean_{b,k}[y_{b,k} · L⁺ (hoặc L^pure) + (1 − y_{b,k}) · L⁻].

Mặc định λ_tile = 1. Code: `models.tile_evidence_loss`; tham số `lambda_tile`, `tile_q`, `tile_pure`.

**Hai biến thể và kết quả của chúng:**

| Biến thể | Thành phần | Trạng thái |
|---|---|---|
| **V-hợp lệ** | âm chính xác + dương max (q = 1), không "pure" | PanNuke (dev): khôi phục định vị 0.77–0.89, tốn 0.015 AUROC. **BBBC041 (dev): không đủ** (0.35–0.38) |
| **V-nhiễu** (đã dùng trong test đã khóa) | âm chính xác + "pure" + q = 0.9 | Test BBBC041: định vị 0.86–0.98 (trừ ring). Test PanNuke: 0.76–0.92. Tốn ≤ 0.03 AUROC |

**Cơ chế (giả thuyết) vì sao V-nhiễu hoạt động với lớp cực thưa:**
- Tile chung (hồng cầu) bị ép âm chính xác ở mọi ảnh không có k, và bị ép dương có nhiễu ở ảnh chỉ có k.
- Hai lực triệt tiêu nhau ở tile chung, nên chỉ các tile đặc trưng của k giữ được evidence dương.
- Ràng buộc "ít nhất một tile" quá yếu: model có thể thỏa mãn nó bằng một tile ngữ cảnh.

---

## 4. Contextual Logit Adjustment (CLA)

**Vấn đề:** BCE trên nhãn tập có tối ưu tại σ(e_k) = p(y_k = 1 | x). Ảnh x chứa cả bằng chứng own o_k lẫn nội dung của các lớp khác, vốn tiết lộ y₋ₖ. Khi ngữ cảnh chỉ mang thông tin qua y₋ₖ:

  logit p(y_k = 1 | x) = **LLR_k(o_k)** + **logit p(y_k = 1 | y₋ₖ)**.

Số hạng thứ hai là prior có điều kiện theo nhãn. Nó học được từ các combination train và sai khi combination thay đổi.

**Phương pháp:**
- **Train:** ℒ = BCE(e_k + τ·b_k(y₋ₖ), y_k), với b_k(y₋ₖ) = logit p̂(y_k = 1 | y₋ₖ).
  - p̂ là hồi quy logistic có regularization (C = 1) của y_k theo **nhãn thật** của các lớp khác.
  - p̂ chỉ được fit trên nhãn của tập train, mỗi đơn vị (combination hoặc ảnh) một hàng.
  - τ = 1.
- **Test:** điểm dùng e_k, không có offset.
- Tại tối ưu (τ = 1, prior đúng): e_k = LLR_k(o_k), bất biến với phân phối combination.

**Quan hệ với prior work:**
- Tổng quát hóa logit adjustment (Menon 2021) từ prior biên sang prior có điều kiện theo nhãn.
- Gần với Product-of-Experts (Clark 2019), nhưng "mô hình bias" ở đây chỉ dùng **nhãn thật của các lớp khác**.

**Biến thể:**
- `mode: marginal`: chỉ dùng intercept, tức logit adjustment nhị phân.
- `covariate`: thêm biến kiểm soát vào prior, ví dụ loại mô trong PanNuke.

Code: `popev/cla.py`, phần `adj()` trong `popev/train.py`; tham số `cla: {mode, tau, C, covariate}`.

**Kết quả:**

| Thiết lập | Kết quả |
|---|---|
| Test PHOEBI (tổ hợp mới, cùng batch) | ABMIL + CLA so với ABMIL: **+0.039 [0.012, 0.066]** (n=1), **+0.041 [0.015, 0.066]** (n=4) |
| phoebi4 (batch mới) | −0.002 / +0.008: không giúp |
| PanNuke (đồng xuất hiện ổn định, mang thông tin sinh học) | −0.016; thêm biến loại mô cũng không giúp |
| Linear head trên PHOEBI | ≈ 0, vì prior trơn theo nhãn quá nhỏ so với bảng tra theo combination |

**Phạm vi đúng của CLA:** có ích khi tổ hợp nhãn dịch chuyển trong cùng một batch dữ liệu, với decoder attention.

---

## 5. Các phương pháp đã thử và bị bác bỏ

| Phương pháp | Ý tưởng | Kết quả |
|---|---|---|
| Pooling phân phối (std / quantile / KME) | Thông tin ngoài mean | Chỉ đúng trên PHOEBI dev (+0.02 đến +0.04 AP); BBBC041 −0.038, PanNuke −0.030 |
| PopEvidence + BCE | Evidence riêng theo lớp | Không hơn baseline |
| CompMix | Ghép tile của hai combination, nhãn hợp | Không giúp (≤ +0.011, không có ý nghĩa) |
| **Tile-only** (λ_bce = 0, điểm = mean của tile evidence) | Bỏ loss cấp tập để không học bảng tra combination | **Tốt nhất trên dev (0.635 / 0.648), kém nhất trên test PHOEBI (0.561)**; đoán quá nhiều nhãn |
| Context consistency loss | Ép điểm của k giống nhau giữa các combination | Không giúp |
| Chuẩn hóa feature theo batch | Bù dịch chuyển batch | ±0.02, không có ý nghĩa (thăm dò, post-hoc) |

---

## 6. Công cụ chẩn đoán và đánh giá

| Công cụ | Định nghĩa | Code |
|---|---|---|
| AUROC / AP theo lớp | Trung bình qua các lớp có cả dương và âm | `metrics._Prepared` |
| Bootstrap theo đơn vị | Resample combination (PHOEBI), `partition:combination` (gộp partition) hoặc ảnh; cài bằng trọng số, O(M) mỗi resample | `metrics.report` |
| So sánh cặp | Hai method trên cùng bag, cùng resample | `metrics.paired_delta` |
| **Định vị tile** | AUROC của tile evidence so với "tâm instance của lớp k nằm trong tile"; < 0.5 là định vị ngược | `scripts/analyze_instances.py`, `compare_inst.py` |
| **Phân rã own/context** | Hồi quy logit(score_k) lên y_k, log(1 + số tế bào k) và y_j (j ≠ k); hệ số của y_j là rò rỉ ngữ cảnh | `metrics.context_attribution`, `analyze_instances.fit` |
| **AUROC khử tương quan** | Trọng số w(y) = Π_k p(y_k) / p̂(y): mô phỏng tập test không có đồng xuất hiện | `metrics.decorrelation_weights` |
| Chuẩn hóa thứ hạng theo split | Đổi điểm thành hạng trong từng fold hoặc partition (không dùng nhãn) trước khi gộp | `metrics.rank_within` |

**Protocol:**
- Lựa chọn chỉ làm trên dev fold.
- Config test được khóa bằng hash (`scripts/lock_config.py`).
- Mỗi tập test chỉ đọc một lần (`scripts/run_test.py`).
- Mọi lần đọc được ghi vào `reports/experiment_log.md`.
- Test ngoài phoebi4 được pre-register trong `docs/protocol_external.md`, kèm Amendment 1.

---

## 7. Tóm tắt trạng thái

| Thành phần | Trạng thái | Phạm vi đúng |
|---|---|---|
| Chẩn đoán định vị ngược | **Xác nhận trên test** (BBBC041, PanNuke) | Decoder set và attention train trên nhãn tập |
| Ràng buộc tile V-nhiễu | **Xác nhận trên test**: khôi phục định vị | Có ảnh một nhãn; lớp thưa |
| Ràng buộc tile V-hợp lệ | Dev: đủ trên PanNuke, không đủ trên BBBC041 | Lớp chiếm nhiều tile |
| CLA | **Xác nhận trên test PHOEBI** (+0.04 so với ABMIL) | Dịch chuyển tổ hợp nhãn, cùng batch; decoder attention |
| Vượt mean_lin | **Chưa đạt ở bất kỳ tập test nào** | — |
| Nguyên nhân gốc | Embedding đông cứng mã hóa batch mạnh hơn loài (F12) | Cần can thiệp ở cấp biểu diễn và dữ liệu nhiều batch |
