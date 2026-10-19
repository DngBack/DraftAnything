# G5 protocol: native-pathway bracketing, causal mediation, shortcut-controlled illusions, end-to-end routing

Written 2026-10-18, after an external-style review of paper ver31 (concerns W1-W5, "repair matrix is partly true by
definition") and before any G5 run or analysis. Already seen and NOT counted as confirmation: everything in G4 and its
exploratory addenda (incl. projector-output L0 probes, value added over SigLIP2, label cost, ICL on 2 models, channel
switching on Qwen3-VL-8B/4B).

Models (G5 set, 8): Qwen3-VL-4B/8B, Qwen3.5-4B/9B, Gemma-3-4B, InternVL3.5-8B, LLaVA-1.5-7B, Qwen2-VL-7B.
Readouts: as G4 (multinomial logistic, 5 folds grouped by base image) unless stated. A cell needs >= 30 errors for
type-share tests. "2/3 of models" = at least 6 of 8 (or 2/3 of the evaluable ones, stated).

## N: native-pathway input control (answers W1)

Features of each model's own visual pathway, question-blind, mean-pooled over image positions:
pre-projector vision-encoder output (input of the projector module), projector output (= LLM layer-0 input at image
positions, "L0"), image tokens at 75% depth (exists). All 13 G4 models where extraction works; tests on the G5 set.
- E_native := L0 probe. Types are recomputed with E_native in place of SigLIP2 (same calibrator and decision-state probe).
- N1: illusions, L0 probe >= decision-state probe (75%) - 5 points in >= 2/3 models.
- N2: CountBench, decision-state probe >= L0 probe + 20 points in >= 2/3 models.
- N3: illusions, with E_native, INHERITED > UNSPOKEN errors in >= 2/3 models.
- N4: value added by the LLM over the native input (probe on [L0 mean; decision state] minus L0 mean): <= +2 points on
  illusions and >= +15 points on CountBench, each in >= 2/3 models.
- N5 (capacity): N1/N2 re-evaluated with every feature set reduced to 256 PCA dims fitted inside the training folds;
  the direction of both comparisons must hold in >= 2/3 models.

## M: causal mediation through the native pathway (answers W3)

Pairs on the illusions: same illusion type and strength parameter, same wording and resolution, one item "equal"
(diff 0) and one "unequal" (diff != 0); both directions. The base item is run with its image-token states at the input
of LLM layer 0 replaced by those of the partner (prefill only; pairs with unequal token counts are skipped), generation
to the G3 decision step, decision state at 75% depth read. Probe: decision-state probe trained on stored G3 states,
5 folds grouped by (type, strength), applied to pairs of the held-out fold. Control: the same swap with a partner of
the SAME label (another diff-0 image of the same type and wording).
- M1: the probe's prediction after the swap equals the partner's label in >= 70% of pairs (median over models).
- M2: the model's answer after the swap equals the partner's label at least 30 points less often than the probe does,
  in >= 2/3 models.
- M3: same-label control: the probe's prediction changes in <= 15% of pairs (median).

## S: shortcut-controlled synthetic illusions (answers W2)

"illusyn": three families rendered procedurally -- Mueller-Lyer (two shafts with fins), Ponzo (two bars between
converging rails), vertical-horizontal (inverted T) -- with the question "Are the two lines equal in length?" style
prompts in VLMBias format (answer in braces). Label equal (diff 0) vs unequal (|diff| in {0.10, 0.15, 0.20, 0.30},
longer line random). Total line length is held constant across labels (ink matched); fin direction is independent of
which line is longer. Two nuisance regimes: A (black on white, width 2-4 px) and B (coloured lines on grey or noise
background, width 5-8 px, rotation +-8 deg); scale and position jitter in both. 200 equal + 200 unequal per family,
half per regime (1200 items), one wording per family, seeded.
- Pixel baseline: logistic regression on 64 hand-made pixel statistics (ink count, bounding box, row/column ink
  projections in 16 bins each, intensity moments).
- S1: cross-regime transfer (train A test B and vice versa, mean): SigLIP2 probe >= 70% and pixel baseline <= 60%.
- S2: cross-regime transfer of the native L0 probe >= 65% in >= 2/3 models.
- S3: model accuracy <= native L0 probe (in-regime, grouped CV) - 15 points in >= 2/3 models.
- S4: with E_native, INHERITED > UNSPOKEN in >= 2/3 of the cells with >= 30 errors.
Reported, not tested: per-family results, leave-one-family-out transfer, transfer from synthetic to VLMBias.

## R: end-to-end routed repair on held-out images (answers "true by definition")

Offline, on the G4 repair-matrix items (5 models x illusions, CountBench, MMVP). At test time a router sees only
label-free signals: the model's answer and the cross-fitted predictions of the calibrator, the decision-state probe
and the SigLIP2 probe (agreement pattern), never the label. Policies: (a) RULE, fixed now: if the calibrator disagrees
with the model -> R1; else if the SigLIP2 probe disagrees with the model -> R3x; else keep the answer (illusions and
CountBench; MMVP: R1 rule only); (b) LEARNED: for each agreement pattern, the repair with the highest accuracy on the
training half of the base images; evaluated on the other half (2-fold by base image, 20 repetitions).
Baselines: raw, each single repair applied to all items, random routing (same repair frequencies as LEARNED), oracle.
- R1: LEARNED >= best single repair + 1 point, pooled over cells (mean over repetitions).
- R2: LEARNED > raw in >= 2/3 of the 15 cells.
- R3: RULE >= best single repair in >= 1/2 of the illusion and CountBench cells.
Label cost of the router is reported (it uses the labels of the training half).

## C: label budget robustness (answers W5)

- ICL with k in {1, 2, 4, 8} on Qwen3-VL-4B, Gemma-3-4B, InternVL3.5-8B, Qwen3.5-4B (the protocol of explore_icl.py),
  and three format variants at k = 4 and 8 on Qwen3-VL-8B: (i) an instruction that the examples show the same question
  and that the answer depends on an exact measurement, (ii) alternating Yes/No order, (iii) all examples in one user turn
  as numbered images with their answers.
- C1: ICL gain over free generation at k = 8 is <= +5 points in >= 2/3 of the model x format cells.
- Reported: generated tokens per item for thinking (compute cost), labels per item of every repair.
- Wording change regardless of outcome: "label cost" -> "supervised label budget".

## W4 (writing, no test): "steering is a probe in disguise" is restated as "steering success is insufficient evidence
of usable latent knowledge"; reported: steering accuracy minus the accuracy of directly outputting the target class.

Deviations will be logged below with dates before the corresponding results are read.

## Addendum 1 (2026-10-18, before any G5 result was read)
- R, operational details: the repair set per cell is every column of g4_matrix_items.csv among R1, R2, R3, R3x, R4, R5
  that covers >= 98% of the items of the cell (missing items keep the raw answer; R6x excluded: exploratory in G4). "Best single repair" = the column with the
  highest accuracy in that cell on all items (chosen post hoc; this favours the baseline). Agreement pattern = whether
  the calibrator, the decision-state probe (75%) and the SigLIP2 probe each agree with the model's answer (missing
  probe = its own state). LEARNED ties go to "keep", then to the cheaper repair (R2, R4, R5, R1, R3, R3x order).
  Random routing draws, for each test item, a repair from the LEARNED policy's choice frequencies on that test half.
- N: InternVL has no module named multi_modal_projector / visual.merger; its projector is `mlp1` (added to the finder).

## Addendum 2 (2026-10-18, before any M result was read)
- The pre-registered M swap (all image-token states at the input of layer 0, identical text) is the same computation as
  running the partner item itself, so M1-M3 reduce to the probe's and the model's accuracy on the partner items. They are
  computed from the stored G3 states and reported as registered, but they do not localise anything.
- Added (exploratory, declared before results): the same swap at the input of layer l in {12.5, 25, 50, 75}% of depth,
  image positions only; text positions keep the states computed from the base image below l. If the decision-state probe
  still follows the partner when the swap is at depth l, the label information at the decision position is read from the
  image positions above l (carried, not computed earlier in the text stream). Reported per l: probe -> partner label,
  answer -> partner label, and the same-label control (equal base: another diff-0 image; unequal base: another unequal
  image of the same type) at 25% and 75%. Pairs: one partner per base item, seeded, same type, strength token, wording
  and number of image tokens and patch grid (image sizes differ by a few pixels; amended before results); all three resolutions. Models: as many of the G5 set as time allows,
  Qwen3-VL-8B, Qwen3.5-9B, Gemma-3-4B, LLaVA-1.5-7B first.

## Addendum 3 (2026-10-18, after S1 was read: SigLIP2 cross-regime 0.52, within 0.57; pixel baseline 0.51; Qwen3-VL-8B 0.51)
S1 fails because the input probe itself is at chance. Because the longer line is random, "equal vs unequal" is
|L1 - L2| < eps, which a linear readout of length features cannot compute even when lengths are encoded. Exploratory
diagnostics, declared before they are run: (a) signed task on unequal items: which line is longer (Mueller-Lyer: top;
Ponzo: upper; vertical-horizontal: horizontal), linear probes, within and cross regime, all readouts; (b) a two-layer
MLP probe (256 hidden, same folds) for equal vs unequal; (c) accuracy by |diff|. On VLMBias the edit direction is tied
to the illusion's sign (strneg <-> diffneg), so a signed comparison there is partly linear; this is reported as a
possible reason for the 95% probe and is not a test.
Read (a)-(c), Qwen3-VL-8B + model-free: signed "which is longer" is linearly readable within regime (SigLIP2 0.80,
DINOv2 0.83, l0 0.74, decision state 0.86; cross regime 0.58 / 0.73 / 0.58 / 0.70); equal vs unequal is not, even
with an MLP (<= 0.63) and at |diff| = 0.30.

## Addendum 4 (2026-10-18, before running it)
Exploratory "illusynL": the same generator with |diff| in {0.30, 0.40, 0.50} (the VLMBias range for Mueller-Lyer and
vertical-horizontal; our registered set is mostly below it), new seed, same size and regimes. Run: SigLIP2, DINOv2,
native features of the 8 G5 models, G3 states for Qwen3-VL-8B, Qwen3.5-9B, Gemma-3-4B, LLaVA-1.5-7B. Same readouts and
S1-S4 criteria, reported as exploratory.
