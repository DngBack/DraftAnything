# Study protocol: Vietnamese orthographic long-tail transfer

Version 1, 2026-11-14. Exploratory engineering and development study, not a
retrospectively preregistered confirmatory experiment. Some legacy results and
this development distribution were already inspected. Final-test rules must be
locked before examining final test predictions.

## Question and ACL positioning

Can sharing vowel-body, vowel-modifier, tone and case structure transfer visual
orthography to unseen combinations, while preserving writer variation? The
linguistic distinction is between vowel identity (e/ê, o/ơ, a/ă/â), lexical tone,
and case; these must not be collapsed into generic accent noise. Frame the work
as multimodal orthographic generalization and resource efficiency, not simply an
image-aesthetics improvement. A positive ACL contribution needs language-grounded
experiments and strong comparisons, not merely an implementation.

## Exposure and data

Primary development holdouts: Ễ/ễ, Ỡ/ỡ, Ẵ/ẵ. Uppercase Ắ is an additional
prospective diagnostic; it is NOT filtered in the present default protocol and
must not be described as strict unseen-combination in this run.

The fresh pipeline removes target-containing images from real/HAT generator
training, OCR supervision, corpora, support and any future memory. Existing
pretrained handwriting checkpoints are not used. Canonical target glyphs are
available to the glyph-conditioned baseline, and the full legal character
inventory is known to recognizers. This is unseen target HANDWRITING under
known orthography, not unseen Unicode or unseen visual glyph from every source.

The source manifest has 97,734 genuine-writer train images, 102,533 HAT OCR
images, 11,786 development and 27,256 final test after filtering. Exact pixel
hashes are newly verified by export; no new image-dedup claims are based solely
on character-count artifacts. The zero-shot composition filter admits 97,497
real and 101,866 HAT images. Generator paired targets/support use real writers
only. The OCR learns from clean images of both sources, with equal source mass.
HAT IDs are never writer classes.

Metadata keys, not physical handwriting identity, underpin writer separation.
Unknown test source IDs cannot prove physical writer disjointness; exact pixel
hashes cannot exclude near duplicates or HAT writer overlap. Resolve these for
strong final claims or explicitly narrow the claim.

## Development and evaluation regimes

- Recognition diagnostics: fixed five ordinary words per development writer plus
  every target-containing development word (duplicates removed). These images
  do not enter training loss.
- Generation paired panel: fixed two ordinary words per development writer plus
  every eligible target-containing word; support is clean and different-text.
- Prompt-only uppercase evaluation: generated Ễ/Ỡ/Ẵ/NGUYỄN can be judged for
  transcription without a paired real image. This does not establish precise
  writer-specific glyph fidelity or paired mark placement in the absence of
  real uppercase examples.
- Final test: no selection or scoring in the present exploratory pilot.

Development real-image counts for ễ/ỡ/ẵ are 25/10/2 occurrences (37 words total
in the archived split). Uppercase Ễ/Ỡ/Ẵ each have zero. Rare-tail confidence
intervals may be very wide and two occurrences of ẵ cannot support a strong
population claim. Acquire additional independent writer examples for the final
uppercase study; templates exist but no samples were collected by this work.

## Experiments implemented

R1: clean factorized CTC versus conventional full-character CTC, same visual
backbone, seed, source sampling, batch and update budget. Factor scores recombine
into complete legal-character scores BEFORE a single shared CTC alignment.
There is one blank output, separate from the semantic absence of marks. Parameter
counts differ and must be reported. This is an auxiliary recognition experiment,
not a replacement for the generation study.

G0: 100-update paired L1 engineering profile. It collapsed toward blank images;
lower L1 was not treated as success. No test data informed this diagnosis.

G1: scratch canonical-glyph renderer versus explicit ink-layer renderer, same
support encoder, source protocol, optimization seed, balanced BCE and frozen
clean-CTC guidance when enabled. The clean recognizer's immutable checkpoint
must be the same in all arms. No full-data evaluator gradients may be used.

G2: natural/glyph-balanced/interaction-balanced sampling, crossed with renderer.
Start with a single-seed pilot and expand promising comparisons to seeds
742/743/744. These seed choices must not be changed to select a winner.

G3 (not implemented): local component memory and explicit geometry labels. Global
support pooling is not a component memory. Do not list this as an achieved method.

G4 (data missing for uppercase): nested 0/1/5/20 genuine-image budgets and
coverage-directed versus random acquisition. Unfilled budgets are explicit.

All pilot coefficients and decoder choices are exploratory. The code records
updates, runtime, eligible/excluded samples, pixel and CTC terms, optimizer states
and fingerprints. Same updates do not imply same FLOPs or parameter count.

## Selection, outcomes and comparisons

Clean recognizer selection: seen-combination development CER only. Holdout scores
are diagnostic. The recognizer becomes a TRAINING supervisor if its gradients
are used for G and cannot then serve as independent content evaluation.

G checkpoints: retain last and best pixel proxy; no automatic winner promotion.
An independent-of-new-G legacy OCR may screen transcriptions and knows heldout
letters; it was selected using legacy development and is not a blind evaluator.
All final claims need a documented independent OCR/human check. Report micro CER,
macro target-glyph F1, case/vowel-modifier/tone errors, writer-cluster bootstrap
intervals, and supported/unsupported target inventories. F1 includes insertion
penalties. Comparisons need paired writer-cluster differences and multiple seeds,
not just separate overlapping confidence intervals.

For visual fidelity, collect mark/anchor annotations and a blinded paired human
comparison. Pixel L1 and source-D distance cannot establish naturalness or
writer-style fidelity. Track the common-character head to detect regressions.
Final non-inferiority margins and sample size must be set with human supervision
before final-test access, not invented after seeing scores.

External generation baselines to reproduce: VATr/VATr++, WriteViT, and CONSTANT
where source/data/license allow. Component font methods DM-Font/LF-Font/MX-Font
bound the representation novelty. The small whole-glyph control is not a substitute
for those model families. Current tools do not justify a novelty or SOTA claim.

## Stop/continue rules

Do not launch long G runs solely because loss decreases. Inspect actual images
and transcription against a fixed independent-of-training recognizer. If OCR
supervision remains poor or images collapse, repair the shared training recipe
before spending the full ablation budget. Record failed pilots and changed
configurations rather than merging their curves. Stop and report an engineering
failure when data/gradients are invalid; do not silently fill missing classes.
