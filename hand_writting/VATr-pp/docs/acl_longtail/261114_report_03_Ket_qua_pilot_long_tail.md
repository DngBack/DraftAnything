# Measured long-tail research pilot

Updated UTC: 2026-11-14T11:51:41.299594+00:00

These are actual completed measurements. Missing entries remain pending. No final-test scoring or blinded human quality evaluation was performed.

## Clean recognition — fixed 5,000 updates

Same clean data and CNN backbone; heads and parameter counts differ. Evaluation is a fixed 187-image, 30-writer development panel, not full development. Three supported holdout glyphs are ễ/ỡ/ẵ; uppercase targets have no paired real images.

| Head | Seed | CER ↓ | Word exact ↑ | Holdout glyph macro-F1 ↑ |
|---|---:|---:|---:|---:|
| whole | 742 | 35.16% | 28.34% | 0.00% |
| whole | 744 | 34.06% | 32.09% | 0.00% |
| factorized | 742 | 34.84% | 29.41% | 26.50% |

Across completed seeds (sample SD; do not treat a one-seed row as replicated):

| Head | Seeds | CER mean ± SD | Tail macro-F1 mean ± SD |
|---|---:|---:|---:|
| whole | 2 | 34.61% ± 0.77% | 0.00% ± 0.00% |
| factorized | 1 | 34.84% ± not replicated | 26.50% ± not replicated |

A conventional unseen output row has no learned semantic route to the heldout class. Its zero F1 is an expected structural limitation, not by itself a novel result. The canonical-glyph prototype head is a stronger semantic control, with full target printed glyph access. These recognition results do not prove generation/style quality.

## Generation — separate legacy OCR evaluation

97 fixed paired development targets. Same immutable clean CTC guidance checkpoint (selected at step 4,500 by seen CER), reconstruction recipe and seed. GAN stage continues each 1,000-update run for another 1,000 updates with newly initialized D; it is a changed objective and larger compute budget.

| Stage | Renderer | Total G updates | Status | External CER ↓ | Word exact ↑ | Tail macro-F1 ↑ |
|---|---|---:|---|---:|---:|---:|
| guided | glyph | 1000 | complete | 58.41% | 2.06% | 0.00% |
| guided | factorized | 1000 | complete | 44.25% | 9.28% | 28.29% |
| gan | glyph | 2000 | complete | 52.51% | 12.37% | 0.00% |
| gan | factorized | 2000 | pending | not measured | not measured | not measured |

The external OCR is separate from new G training but was selected using legacy development. It is a screening proxy. Architecture capacity is not matched. Single-seed generator differences are exploratory; inspect real/generated/support images before any quality conclusion.

## Uppercase prompt-only diagnostics

| Stage | Renderer | Prompt | OCR exact / 30 writers |
|---|---|---|---:|
| guided | factorized | Ễ | 0/30 |
| guided | factorized | Ỡ | 0/30 |
| guided | factorized | Ẵ | 0/30 |
| guided | factorized | NGUYỄN | 0/30 |

These 300-prompt panels use known input transcripts and clean writer supports, without paired real target images. They test readability proxies; they do not establish exact writer-specific target-glyph fidelity. Standalone OCR itself may be unreliable.

## Failed pilot and remaining gaps

- Plain L1 collapsed toward white at 100 real-data updates. It was retained as a failure, not promoted for low loss.
- Guided generation produced visible gray/noisy backgrounds at 250 and 1,000 updates. Better OCR scores alone do not fix that.
- No component-local writer memory or annotated mark geometry is trained. No advantage over native VATr++, WriteViT or CONSTANT is established.
- Uppercase genuine-writer data collection, a stronger baseline suite, independent mark/style/human evaluation, and three-seed generation confirmation remain required.
- The current study is not an ACL-ready paper. See STUDY_PROTOCOL.md, PAPER_DRAFT.md and STATUS.md.
