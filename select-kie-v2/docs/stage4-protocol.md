# Stage 4: stronger extractor and length control

Frozen before predictions and gold inspection, 2026-09-28. Select 100 Short-Form documents by SHA256 filename after excluding all 600 stage2 and 100 stage3 documents. No gold filtering; same pinned public source. Document-independent, not person-independent.

Image extraction: Qwen3.5-2B and Qwen3.5-9B, bf16, greedy, thinking disabled, all pages at maximum long side1280, max320 output tokens. Three conditions: original baseline, clarified semantic schema from stage3, baseline plus repeated generic “Copy faithfully.” guidance truncated to exactly the clarified instruction token count. The neutral repetition controls length/token budget; it does not perfectly control prose quality or attention effects. Do not claim this alone isolates every prompt confound.

Primary official VRDU field accuracy; secondary existing stage3 normalized matcher. Paired document bootstrap2000 for clarification vs length control and9B vs2B. Report per-field, present/absent, output caps, missing keys, high-confidence (>0.99 mean value-token probability) errors and timing. No tuning on this holdout, no replacing primary labels. Auto on-page error is not verified binding.

Prepare200 decision audit with100 uniformly sampled decisions from9B-clarified and100 targeted official errors (up to16 per condition, remaining slots sampled from pooled errors), fixed random seed20260928, blind model/confidence, blank human labels,20 overlapping decisions for independent reviewer. Human taxonomy: correct, binding, miss, hallucination, partial-value, transcription, gold/schema-ambiguity, other. Human review is pending; automated labels do not establish true binding.

No relational-head training before this gate: inspect remaining errors after clear schema and larger model; binding prevalence must be established by audit. This stage tests scale within one model family, not independent architecture generalization.

Pre-analysis amendment (before opening gold or metrics; 9B weights still downloading): random audit population restricted to9B-clarified because the research gate concerns binding after stronger extraction and clear schema. Targeted errors balanced across conditions. No outcome-driven sampling.

Runtime amendment:9B initial batches used2 then4. Batch4 hit CUDA OOM on multi-page prefill; resume disjoint document-hash shards at batch2 without omitting any page or changing1280px resolution. An attempted expandable allocator failed during loading and was reverted. Keep raw completed records, log failures and exclude failed attempts from reported successful generate-only timing. This is a resource adjustment, not outcome-based data selection.

Post-result decision amendment: after targeted AI PDF checks supported role errors, stage5 runs a correctness-only relational pilot although independent human prevalence audit is still pending. This revises the original wait-for-audit training gate; it does not satisfy that gate or establish binding prevalence. No auxiliary binding labels/loss or binding-specific claim is introduced. Stage4 evaluation is explicitly exploratory for the new method; stage6 provides a fresh-document check.
