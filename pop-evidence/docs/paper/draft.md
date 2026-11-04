# [Draft v0] When Set Labels Teach Context: Salience, Shortcuts and Label-Valid Tile Evidence for Compositional Recognition in Microscopy

> Status 2026-10-29. Sections 1–4 are drafted. Sections 5–6 are placeholders: **no result in them is final**. Numbers marked (dev) are development-fold estimates; the test sets are unopened. Claims are written at the strength the evidence allows today and must be revised after the go/no-go decision (docs/paper_plan.md).

## Abstract (v1, 2026-10-29; numbers are development estimates)

Microscopy datasets often label a sample only with the set of components it contains (species in a culture, cell types in a smear or tissue patch). Decoders trained on such set labels over frozen tile embeddings can reach good image-level accuracy, but it is unclear what they are looking at.

On BBBC041 blood smears, where instance boxes are available for evaluation only, we find that set-trained decoders are right for the wrong reasons:
- They separate images with and without trophozoites or schizonts (image AUROC 0.82).
- Their per-tile evidence for these stages is anti-localised: tile AUROC 0.08 and 0.09 against box ground truth. The evidence sits on tiles that do *not* contain the cells.
- Holding a class's presence and cell count fixed, its score still moves with which other classes are present (11–17 of 20 context coefficients resolved).
- The share of score variation explained by context grows with decoder flexibility, from 0.21 for linear heads to 0.33–0.39 for set networks.

We propose label-valid tile constraints, three conditions implied by set labels without reference to other classes: an absent class has no evidence in any tile; a present class has evidence in its top-quantile tiles; a pure sample has evidence everywhere. With these constraints, localisation is restored (trophozoite 0.88, schizont 0.84, gametocyte 0.87) at unchanged image accuracy. Image-level context reliance, however, is not reduced, which locates part of it in the frozen representation or in the biology of co-occurring classes rather than in the decoder.

On PHOEBI, a bacterial benchmark that holds out whole species combinations, the same decoders score near chance per species. Point estimates follow the same pattern, but with 26–27 combinations they are not resolvable, a statistical limit we quantify. [PanNuke replication pending.]

## 1. Introduction (v1 outline)

1. Set labels are the scalable label in microscopy; accuracy alone does not tell whether a decoder recognises components or their company.
2. Three diagnostics, each with the right unit of replication:
   - an own/context decomposition at fixed own evidence;
   - tile localisation against instance ground truth used only for evaluation;
   - a context share normalised by score scale.
3. Finding: set-trained decoders are accurate and anti-localised; flexibility buys accuracy with context.
4. Remedy: label-valid tile constraints restore localisation at no accuracy cost. They do not remove image-level context reliance, which is reported as a negative result with its implications.
5. Statistical honesty: on combination-level benchmarks (PHOEBI), per-species claims need far more combinations than exist. We show what can and cannot be resolved.

Contributions:
- (i) the diagnostics and protocol;
- (ii) the anti-localisation finding on two datasets with instance ground truth [BBBC041 done, PanNuke pending];
- (iii) label-valid tile constraints and their measured effect;
- (iv) negative results: no accuracy gain on unseen PHOEBI combinations, distribution pooling not general, context reliance not removed.

## 2. Related work

- **Microscopy benchmarks and baselines.**
  - PHOEBI introduces the leave-combinations-out bacterial benchmark and reports that mean, 90th-percentile and max aggregation, finer units and learned subspaces do not move held-out AUROC beyond about 0.63.
  - Svatko et al. (ICML 2026) show that simple baselines often match representation learning in bioimaging.
  - Borowa et al. (JBHI 2023) apply multiple-instance learning to polyculture images of four species. [Verify whether combinations were held out.]
- **Contextual bias in multi-label recognition.**
  - Singh et al. (CVPR 2020) decorrelate category features from co-occurring context.
  - Liu et al. (CVPR 2022) and Chen et al. (NeurIPS 2024) use causal formulations.
  - These works use natural images and object-level evaluation out of context. We study set-only supervision and evaluate on unseen combinations, where context is not a nuisance correlate but the only signal the loss rewards beyond the dominant species.
- **Multiple-instance and distributional pooling.**
  - Attention MIL (Ilse et al. 2018), class-wise pooling (WILDCAT, Durand et al. 2017) and quantile-function pooling (Couture et al. 2018) are direct antecedents of our pooling. We claim no novelty for pooling itself.
  - Kernel mean embeddings and distribution regression (Muandet et al. 2017; Szabó et al. 2016) and Deep Sets (Zaheer et al. 2017) are our distribution baselines.
- **Weak supervision from proportions and causal MIL.** Easy LLP (Busa-Fekete et al. 2023) needs label proportions, which set labels do not provide. CausalMIL (Zhang et al.) targets binary MIL. Set-DINO (Yao et al.) uses set-level consistency for representation learning rather than multi-label decoding.
- **Compositional generalisation of label sets** has been studied for text classification (Chai et al., AAAI 2024). We found no image benchmark besides PHOEBI that holds out label combinations.

## 3. Problem and protocol

- **Data.**
  - A sample c has fields x_{c,1..n}, each cut into T tiles with frozen embeddings z ∈ R^D, and a label y_c ∈ {0,1}^K.
  - In PHOEBI a sample is one video of one combination: 3,000 crops of 1024 px, T = 16 tiles of 224 px, DINOv2-S/14 CLS features with PHOEBI's illumination correction.
  - In BBBC041 a sample is one smear image (T = 35), y lists the infected-cell stages and leukocytes present, and boxes give tile-level ground truth used only for analysis.
- **Splits.**
  - PHOEBI: the leave-combinations-out partitions with seeds 1337, 1338 and 1339 (9 held-out combinations each, 27 pooled) and the five development folds over the 26 trained combinations of order 2–4 of partition 1337. All selection uses pooled development folds only.
  - BBBC041: 5-fold image-level cross-validation on the official training set, and the official test set (another acquisition) as a shift test.
- **Endpoints.**
  - Primary endpoint: macro average precision.
  - Co-primary for comparability with PHOEBI: mean per-species AUROC.
  - Secondary endpoints: per-sample F1 and its margin over the all-present predictor; macro F1; FPR; number of predicted labels.
  - Intervals: bootstrap over the resampling unit (combination, partition:combination, or image). Comparisons are paired on identical bags and identical resamples.
- **Own/context decomposition.** For decoder scores s_k, regress logit(mean_c s_k) on the presence vector y_c over evaluation units. B_kk is own evidence and B_kj (j ≠ k) is context. We report bootstrap intervals and count species whose own evidence is resolved above zero.

## 4. Method

- **Per-species tile evidence.**
  - A shared projection φ maps each tile embedding to h. Per-species heads give an evidence vector a_{t,k} ∈ R^m for every tile t and species k.
  - Pooling is per species: mean, standard deviation and quantiles of a_{·,k} over the tiles of the bag.
  - The score e_k reads only species k's pooled statistics. This class-specific read-out is ablated against a shared head that sees every species' statistics.
- **Label-valid tile constraints** on the first evidence channel a_{t,k,0}:
  - absent (y_k = 0): softplus(a_{t,k,0}) averaged over tiles;
  - present in a mixture: softplus(−q_{0.9}(a_{·,k,0}));
  - pure culture: softplus(−a_{t,k,0}) averaged over tiles.
  - None of these targets depends on which other species are present. The loss is BCE on e_k plus λ_tile times the constraint loss.
- **Optional cross-combination consistency:** variance of e_k across positive combinations within a batch, with a ranking term against negatives to exclude constant solutions.
- **Baselines on identical inputs:** linear heads on mean, mean+std, mean+quantile and kernel mean embeddings; Deep Sets; PHOEBI's ProtoMatch, SimplexUnmix and ChannelGroup on the same folds.

## 5. Results (v1 skeleton with current development numbers)

- 5.1 Set-trained decoders are right for the wrong reasons (BBBC041; 866 images, out-of-fold, 3 seeds).
  - Image AUROC: mean_lin 0.774, Deep Sets 0.819, PopEvidence 0.818.
  - Tile localisation without constraints [95 % CI over images]:

    | Class | Tile AUROC |
    |---|---|
    | trophozoite | 0.076 [0.068, 0.083] |
    | schizont | 0.090 [0.072, 0.110] |
    | ring | 0.385 [0.358, 0.413] |
    | leukocyte | 0.600 |
    | gametocyte | 0.844 |

- 5.2 Context at fixed own evidence.
  - Resolved context coefficients: 11/20 (mean_lin), 15/20 (mean+std), 17/20 (Deep Sets).
  - Context share normalised by logit spread: 0.21 for linear heads, 0.39 for Deep Sets, 0.35 for PopEvidence.
- 5.3 Label-valid tile constraints restore localisation.
  - Tile AUROC with λ = 1: trophozoite 0.878 [0.868, 0.889], schizont 0.837 [0.810, 0.863], ring 0.691 [0.658, 0.718], gametocyte 0.873.
  - Image AUROC 0.816 vs 0.818.
- 5.4 What constraints do not fix (negative).
  - Context share is unchanged: 0.35 → 0.34 (m = 1), 0.30 with max pooling.
  - Max pooling of localised evidence still leaves 13/20 context coefficients resolved.
- 5.5 PHOEBI (unseen combinations).
  - All decoders score per-species AUROC 0.54–0.60 (dev), below ProtoMatch's published 0.63.
  - Tile constraints do not change accuracy (0.558–0.565).
  - mean+std over mean: +0.021/+0.037 AP (resolved), but the effect reverses on BBBC041 (−0.038).
  - Own/context point estimates mirror BBBC041 (dense species carry own evidence; sparse species do not), but intervals over 26 combinations contain zero.
- 5.6 PanNuke replication (dev; 7,901 H&E patches, 5 nucleus types).
  - Unconstrained PopEvidence: image AUROC 0.88, but tile localisation is anti-aligned for 3/5 types (dead 0.115, epithelial 0.194, connective 0.228).
  - Constrained: all types 0.77–0.90, at −0.006 image AUROC.
- 5.7 Held-out test, BBBC041 (official test set: another acquisition, 1944×1383; locked config `bbbc041_v1`, read once).
  - Image AUROC is low for every method (0.39–0.58): acquisition shift dominates. The best method is ABMIL + tile constraints + CLA at 0.577 [0.498, 0.657], with the lowest FPR (0.12).
  - **Anti-localisation replicates on test:** ABMIL gametocyte 0.08, ring 0.19; PopEvidence schizont 0.04, trophozoite 0.07.
  - **Constraints restore it on test:** ABMIL + tile (+ CLA) gametocyte 0.97/0.98, schizont 0.95/0.95, trophozoite 0.86/0.93.
  - Exception: ring stays anti-localised (0.19–0.24) after the acquisition shift.
- 5.8 Instance-level supervision derived from set labels (PHOEBI dev, rank-normalised per fold).
  - tile-only (mean of tile evidence; no bag-level loss): 0.635 (n = 1), 0.648 (n = 4).
  - Baselines: mean+std 0.600 / 0.615; mean 0.582 / 0.586. Paired vs mean+std +0.036 / +0.033 (P = 0.87 / 0.84); vs mean +0.053 / +0.062 (P = 0.94 / 0.96).
  - ProtoMatch on the same five development folds (PHOEBI's own nested CV, pooled raw): 0.545 [0.478, 0.614]. Our raw pooled tile-only q50 is 0.609 (n = 1).
- 5.9 Held-out test, PanNuke fold 3 (2,722 patches, locked `pannuke_v1`).
  - Anti-localisation replicates: unconstrained PopEvidence dead 0.08, epithelial 0.18, connective 0.24.
  - Constraints restore it (ABMIL + tile: 0.77–0.92; tile-only 0.76–0.92).
  - Accuracy: Deep Sets 0.893 and mean 0.889 are best. Constraints cost 0.01–0.03 AUROC; CLA costs 0.026 (decorrelated 0.841 vs 0.843).
- 5.10 PHOEBI locked test (3 partitions, 27 combinations; rank-normalised per partition / raw):
  - ABMIL + CLA 0.679 / 0.671 (n = 1) and 0.683 / 0.673 (n = 4), the highest. Mean 0.667 / 0.654; mean+std 0.664 / 0.654; ABMIL 0.652 / 0.632.
  - **CLA vs its ablation (ABMIL):** +0.027 [−0.000, +0.055] (n = 1), +0.028 [+0.005, +0.050] (n = 4).
  - vs mean / mean+std: +0.01, n.s.
  - **The development-selected primary method (tile-only) failed on test:** 0.600 vs 0.664 (mean+std). It depends on pure cultures, and collapses when the held-out singleton removes one (partitions 1338: bs; 1339: mx). The development folds (partition 1337 only, all five singletons kept) could not reveal this.

## 6. Limitations (draft)

- PHOEBI has 40 combinations; per-species statements are rarely resolvable at this size. We restrict claims to pooled statistics and use a second dataset with image-level units.
- Species and acquisition session coincide in PHOEBI (one video per combination), so second-order statistics may encode session signatures.
- Development folds exist only for partition 1337. Some development combinations are test combinations of partitions 1338 and 1339, so the pooled three-partition test is not fully independent of selection, as in PHOEBI.
- bt has no pure culture, so the pure-culture constraint and probe do not apply to it.
- Frozen encoder only; fine-tuning is out of scope.
