# SLEEP-KIE — ý tưởng gốc

Bản lưu của research proposal (25/10/2026). Đã bỏ phần trích dẫn bị vỡ và biểu đồ minh họa; nội dung kỹ thuật giữ nguyên.

**Tên tạm:** SLEEP-KIE: Evidence-Preserving Recurrent Memory for Single-Read Long-Document Information Extraction.

**Mục tiêu:** đọc tài liệu 50–500 trang đúng một lượt theo thứ tự, trích JSON có cấu trúc, suy luận xuyên trang và chỉ ra bằng chứng (page, bbox), với latency và GPU memory thấp hơn các hệ thống long-document KIE mạnh. Lấy cảm hứng từ *Do Language Models Need Sleep?* nhưng không sao chép cơ chế lặp nguyên bản.

## 1. Bài toán

Tài liệu $D=\{P_1,\dots,P_T\}$, schema $S$ cho trước khi đọc. Dự đoán $(\hat Y,\hat E)=F_\theta(D,S)$: $\hat Y$ là JSON field→value, $\hat E$ là bằng chứng (page, bbox, quan hệ). **One-pass** = mỗi trang được encode đúng một lần; recurrence chỉ lặp trên trạng thái ẩn đã mã hóa, không đọc lại ảnh/OCR trang cũ.

Schema biết trước là khác biệt có chủ đích so với query-agnostic multi-query KIE: mô hình quyết định giữ gì.

- **RQ1 — Selective memory:** lưu thông tin quan trọng cho schema trong bộ nhớ cố định thay vì giữ toàn bộ visual tokens?
- **RQ2 — Offline reasoning:** vài bước recurrent consolidation sau mỗi trang có cải thiện suy luận xuyên trang mà không tăng số lần đọc?
- **RQ3 — Accuracy–latency:** có đạt Pareto frontier tốt hơn baseline mạnh (accuracy, latency end-to-end, peak memory, grounding)?

## 2. Đóng góp đề xuất

1. **Schema-Conditioned Evidence Memory (SCEM):** bộ nhớ giới hạn, slot theo field, slot quan hệ, tham chiếu vùng bằng chứng.
2. **Adaptive Latent Consolidation (ALC):** quyết định có cần thêm computation để hợp nhất trang mới với memory.
3. **Single-Read KIE Protocol:** kiểm soát số lần đọc trang, memory budget, tổng chi phí; quan hệ cách hàng trăm trang.

Đóng góp 2 chỉ có giá trị nếu recurrence tốt hơn memory updater sâu/lớn hơn **ở cùng compute**.

## 3. Kiến trúc

- **Page encoder** $H_t=E_\theta(P_t)$: VLM/encoder đông cứng + OCR/bbox, projection về $d=512/768$; module memory trainable vài chục triệu tham số.
- **Schema-guided token selection:** $a_{f,j}=q_f^\top W_k h_{t,j}/\sqrt d$, top-K (soft/straight-through khi train, hard khi infer) + token bảng + token coverage.
- **SCEM:** $M_t=\{M^{field},M^{relation},M^{evidence}\}$; cấu hình gốc 32/32/64 slot, d=512, 128–256 token/trang, 1–4 vòng sleep. Trạng thái persistent ~128 KiB FP16 và không tăng theo số trang (không phải tổng VRAM).
- **Recurrent consolidation:** $U^{(0)}=\mathrm{CrossAttn}(M_{t-1},Z_t)+M_{t-1}$; $U^{(k+1)}=U^{(k)}+G_\phi(U^{(k)},Z_t,S)$ với $G_\phi$ chia sẻ trọng số; gate $M_t=g_t\odot M_{t-1}+(1-g_t)\odot\tilde M_t$.
- **Adaptive stopping:** halting head + $\mathcal L_{compute}$. Chưa làm ở MVP.
- **Structured decoder:** $\hat Y=D_\psi(M_T,S)$, constrained JSON, hỗ trợ `null`/`not_found`.

## 4. Loss

$\mathcal L=\mathcal L_{KIE}+\lambda_e\mathcal L_{evidence}+\lambda_m\mathcal L_{cons}+\lambda_c\mathcal L_{compute}$; evidence dùng multi-positive $-\log\sum_{e\in E^*_f}p(e)$; consistency là distillation từ teacher thấy đủ evidence (rủi ro shortcut); tùy chọn $\mathcal L_{prefix}$ trên nhãn trung gian.

## 5. Benchmark

- **Tier 1 (KIE thật):** Kleister-NDA, Kleister-Charity, DocILE, UNIKIE-BENCH. Không dùng FUNSD/SROIE làm benchmark chính.
- **Tier 2:** MMLongBench-Doc (khóa phiên bản), chuyển câu hỏi thành schema 1 field.
- **Tier 3 — Controlled Long-KIE:** số trang 10–200, khoảng cách bằng chứng 1–150, 4–32 field, 0–5 lần cập nhật, tỉ lệ null, loại bằng chứng, loại suy luận (lookup, comparison, aggregation, revision, cross-page relation). Phụ thuộc ngữ nghĩa thật, chia theo template/entity/source.

## 6. Thí nghiệm

- **A — Sleep có giúp không:** A0 N=1/128 slot; A1 N=2; A2 N=4; A3 N=4/64 slot; A4 N=1/256 slot; **A5 N=1 updater sâu hơn, compute-matched (bắt buộc)**.
- **B — Long-range retention** theo khoảng cách bằng chứng.
- **C — Reasoning complexity:** lookup, linking, revision, aggregation, contradiction, missing-field.
- **D — Efficiency:** full-context VLM, chunked VLM, OCR+retrieval, streaming không sleep; cùng GPU/precision/batch; báo cáo $T_{total}$ đủ thành phần, cold/warm, throughput.

## 7–9. Metric, kế hoạch, go/no-go

Metric: field micro/macro F1, doc exact match, cross-page acc, evidence page recall, bbox IoU, null P/R, latency, peak GPU memory, pages/s, accuracy/GFLOP; mean ± CI nhiều seed.

- **Go 1:** N=4 cải thiện rõ cross-page F1 so với N=1 ở cùng memory, **và** vẫn hơn deeper updater compute-matched. Thất bại → bỏ adaptive sleep, chuyển sang evidence-aware streaming memory / hierarchical extraction.
- **Go 2:** nén memory không làm mất grounding.
- **Go 3:** có vùng vận hành nằm trên Pareto frontier so với baseline có caching/retrieval. Không đạt → bỏ chữ “Fast”.
- **Go 4:** lợi ích có trên ≥ 2 domain thật, không chỉ synthetic.

Lộ trình: 10–50 trang trước, rồi 100–200; 500 trang là stress test cuối.
