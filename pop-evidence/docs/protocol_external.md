# Pre-registration: external test on the PHOEBI four-species subset (phoebi4)

Written 2026-10-29, **before** the phoebi4 images were downloaded or any phoebi4 feature or prediction was seen.

## Why this set

- The phoebi6 test partitions, the BBBC041 test and PanNuke fold 3 have each been read once. No untouched evaluation set remains for methods developed afterwards.
- phoebi4 is a separately released subset: 4 species (b, f, k, p), 14 label sets, about 1,000 images per video. It has never been used in this project.
- Most of its label sets also occur in phoebi6, but **as new videos**, i.e. new acquisition sessions. In phoebi6 a combination is a single video, so a decoder can memorise "session ↔ label set" (findings F1, F10). phoebi4 tests whether what was learnt is the species or the session.
- `b_f` and `f_b` are two different videos with the same label set: a within-label-set session contrast.

## Design (fixed now)

- **Training data:** all of phoebi6 (all 40 combinations), the random split's `train` part. Calibration (thresholds): phoebi6 random-split `val`. No phoebi4 data is used for training, calibration or selection.
- **Species mapping:** b→bs, f→fj, k→ka, p→pf. To be verified on data before features are scored (§Checks).
  - bt and mx are absent from every phoebi4 video. They are scored only for false positives (all-negative).
- **Evaluation:** every phoebi4 image (all of its train/val/test parts). Bags of n ∈ {1, 4} images from one video; cap 300 bags per video.
  - Unit of replication = video (14 units).
  - Metrics: mean AUROC and macro AP over the 4 shared species (primary: mean AUROC); FPR on bt and mx; per-species AUROC.
  - Intervals: bootstrap over videos (2,000).
- **Methods:** exactly those in `configs/locked_phoebi4.yaml`, locked by hash before scoring. Its development basis is the PHOEBI dev folds and the locked phoebi6 test. No new tuning.

## Hypotheses (directional, stated before data)

- **E1 (CLA replication):** ABMIL + CLA > ABMIL on mean AUROC (paired over videos). Expected from the locked phoebi6 test (+0.04).
- **E2 (session memorisation):** decoders with more capacity to memorise sessions (Deep Sets, ABMIL) lose more from the phoebi6 test to phoebi4 than the linear mean head. Their phoebi4 AUROC is not higher than the linear mean's.
- **E3 (instance-only):** the instance-only objective (tileonly_mean) is not better than the linear mean on phoebi4. Prediction: worse, given its test failure.
- **E4 (prototype):** ProtoMatch (re-implementation, pure-culture prototypes), which learns nothing from mixtures, is compared descriptively. No directional claim.

## Checks before scoring (allowed to look at images and labels, not at predictions)

1. **Species correspondence:** the phoebi4 pure-culture videos (b, f, k, p) must sit nearest, in mean tile embedding, to phoebi6 pure cultures bs, fj, ka, pf respectively. If not, stop and report.
2. **Independence:** no phoebi4 image is a duplicate of a phoebi6 image (nearest-neighbour cosine of mean embeddings < 0.999 for all, checked on a sample). If duplicates exist, phoebi4 is not an independent set: report and stop.

## Amendment 1 (2026-10-29, after the checks, before any prediction was computed)

- **Check 1 failed:** only k→ka is the nearest phoebi6 pure culture. b→bt 0.921 (bs 0.883); f→mx 0.904 (fj 0.762); p→fj 0.951 (pf 0.919). See `results/analysis/phoebi4_checks.md`.
- **Check 2 passed:** no duplicates; max cosine 0.976.
- **The mapping is confirmed by the official dataset card:** "Its codes b, f, k and p are *Bacillus subtilis*, *Flavobacterium johnsoniae*, *Klebsiella aerogenes* and *Pseudomonas fluorescens*, the species coded bs, fj, ka and pf in phoebi6". phoebi4 is "an earlier collection, cultured in a separate batch and imaged in a separate session on the same instrument".
- Check 1 existed to catch a wrong mapping. Its failure is therefore not a mapping error. It shows that the **batch and session shift between phoebi6 and phoebi4 dominates the frozen embedding over species identity** for 3 of 4 species.
- **Decision:** proceed with the locked config unchanged. The external test now measures generalisation across culture batch and session, harder than the session-only shift anticipated.
- Hypotheses E1–E4 are unchanged.
- **Additional expectation, stated before predictions:** absolute AUROC will be low for all methods except on ka.
