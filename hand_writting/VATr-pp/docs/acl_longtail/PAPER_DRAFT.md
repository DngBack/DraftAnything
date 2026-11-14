# Compositional Orthographic Transfer for Long-Tailed Vietnamese Handwriting

**Working manuscript structure, not a submission-ready paper.** No new positive
performance claim is made. The current real-data pilots are exploratory;
`261114_report_03_Ket_qua_pilot_long_tail.md` records measurements separately from planned results.

## Abstract — provisional, no performance claims

Vietnamese handwriting synthesis must represent vowel modifiers, lexical tones,
and letter case under sparse combinations of characters and writers. Character
frequency alone obscures a critical failure regime: an uppercase marked vowel
can have no genuine-writer training images even when its lowercase form and
individual marks are observed. We formulate an exposure-controlled study of
orthographic combination transfer. We implement metadata audits that separate
character occurrences, independent images and writers; a shared policy governing
training images, text corpora, OCR and style support; and matched-data scratch
controls for glyph conditioning and explicit ink composition. A compositional
recognizer combines factor scores before a single CTC alignment to examine
whether component sharing can supply clean content supervision. Real-data
experiments and independent writer/mark evaluation are required before drawing
conclusions about generation quality or data efficiency.

## 1. Introduction

A name such as NGUYỄN contains Ễ, which combines an uppercase E body, circumflex
and tilde tone. In the audited source, genuine-writer train contains E and ễ but
no Ê, Ẽ or Ễ. HAT contributes Ễ images without reliable same-writer grouping.
Thus image scarcity, source shift and absent writer-component evidence occur
simultaneously. Repeating target text cannot manufacture missing visual evidence.

The question is how orthographic structure transfers across these missing
combinations, rather than whether an OCR can read a single generated word.
Vowel-modifier errors, tone errors and case errors carry distinct linguistic
roles. Isolated legal glyphs and contextual Vietnamese syllables must be evaluated
separately; arbitrary tone substitution is not a guarantee of syllable validity.

Candidate contributions, contingent on final evidence: (i) an audited benchmark
for combination×case×writer scarcity, (ii) evidence about the relative effects of
exposure, balancing and compositional conditioning, and (iii) a measured data
acquisition curve. Component decomposition itself is not new.

## 2. Related work

VATr supplies visual glyph archetypes, so it must not be characterized as merely
one-hot character conditioning. VATr++ changes data preparation and regularization.
DM-Font, LF-Font and MX-Font already use compositional or local-style mechanisms
for fonts; their insights motivate controls rather than establish novelty here.
WriteViT and CONSTANT directly cover Vietnamese handwritten generation and must
be compared under compatible exposure and support budgets. Class-balanced
reweighting addresses observed imbalanced evidence; no reweighting creates an
unobserved glyph's real-writer supervision.

Sources to cite, verified primary pages: VATr https://arxiv.org/abs/2303.15269 ;
VATr++ https://arxiv.org/abs/2402.10798 ; DM-Font https://arxiv.org/abs/2005.10510 ;
LF-Font https://arxiv.org/abs/2009.11042 ; MX-Font https://arxiv.org/abs/2104.00887 ;
WriteViT https://arxiv.org/abs/2505.13235 ; CONSTANT https://arxiv.org/abs/2603.07543 ;
Class-balanced loss https://arxiv.org/abs/1901.05555 . Existing project bibliography
is in `../vietcompose/references.bib`; publication metadata must be rechecked for
the final bibliography. No claim that this targeted review exhausts related work.

## 3. Task and exposure

Let c=(b,m,t,k) denote base, vowel modifier, tone and case. Given support S_w,
synthesize G(S_w,text). Count independent images n(c), writers u(c), and
case-conditioned component/pair coverage separately per source. An unseen full
combination is different from an unseen primitive. A writer without a mark in
support creates uncertainty about that writer's particular realization.

Our zero-shot track excludes complete holdout combinations from every training
consumer and starts from scratch. It permits canonical printed glyph access for
the glyph baseline and knowledge of the full legal inventory. A separate few-shot
track admits nested real-image budgets; it does not relabel exposed legacy runs
as zero-shot. The new exporter hashes images and rejects exact duplicate/split
violations, while inherited writer semantics and near duplicates remain limits.

## 4. Implemented models

Paired generation controls share a CNN support encoder. One renderer uses full
canonical glyphs and contextual decoding; the other reuses the explicit
body/modifier/tone ink renderer. Mark absence is gated, geometry is differentiable,
and refinement is bounded. Architectures and parameter counts differ; this is an
exploratory representation contrast, not a matched-capacity superiority test.
No component-local writer retrieval is implemented yet.

For content guidance, complete-character logits are formed by summing four
factor-head scores over legal tuples, with an independent blank score. The
normalization and CTC operate on whole characters, yielding a single alignment.
A conventional full-character head on the same backbone provides a control.
Both are trained on exposure-admitted images. Frozen clean OCR can guide G,
while a separate legacy OCR may only evaluate new G, not supervise it.

## 5. Experimental design

Use `STUDY_PROTOCOL.md` as the operational specification. Separate representation,
sampling, exposure budget, source and writer coverage. Report all changed losses
and failed pilots. Test data remain untouched until final decisions are locked.

Required result tables (not filled with invented values):

1. Actual image/writer and component-pair coverage by source and case.
2. Clean recognition: whole-character versus factorized CTC on seen/unseen tuples.
3. Generation: head/tail content, mark placement and writer/style fidelity.
4. Ablations: sampling, case-aware transfer, geometry, local memory if implemented.
5. Data efficiency: 0/1/5/20 real images, random versus coverage-directed acquisition.
6. Compute and parameter costs across seeds and external baseline families.

Only archived facts and completed new measurements belong in `261114_report_03_Ket_qua_pilot_long_tail.md`.

## 6. Results and analysis — pending

The prior warm-start adapter pilot did not improve the source: CER 13.63% source,
15.38% factorized adapter, 16.32% glyph adapter. Those are legacy supervisor metrics,
not clean generalization results. The new 100-update real-data L1 profile produced
near-white images despite lower reconstruction error. This motivates a controlled
shared-recipe change, not a quality claim. New measured tables will be generated
from immutable per-run artifacts and annotated with their evaluation limitations.

## 7. Limitations and responsible research

Uppercase target images are absent from current genuine-writer development/test
for the main holdout families. Prompt transcription can be tested but exact
writer-specific paired fidelity cannot. Collect independent uppercase samples,
retain original resolution, document permission/license and assign splits by
writer. Blank collection forms are not consent or a dataset. HAT writer overlap
and physical writer identity across unknown test IDs remain unresolved.

OCR fidelity is not visual naturalness. The legacy evaluator was selected on
legacy development, mark annotations are unavailable, parameters are not matched,
and most new conditions initially use one seed. Current engineering results
cannot establish a general method advantage. Full independent evaluation,
external baselines, statistical replication and verified resource rights are
required before submission.

Generative handwriting can enable impersonation. Release decisions should cover
writer consent, provenance and permitted use; do not include signatures or infer
permission from public availability. Disclose AI assistance accurately under the
venue's policy; authors remain responsible for all claims and citations.

## Submission readiness

Not ready. Resolve the measured gaps in `STATUS.md`, complete results and the
ARR Responsible NLP checklist with evidence rather than automatic yes responses:
https://github.com/acl-org/aclrollingreview/blob/main/authorchecklist.md .
The final target ACL-family venue/year and formatting are not yet selected.
