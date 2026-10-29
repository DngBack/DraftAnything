# Right for the Wrong Tiles: Set-Labelled Microscopy Decoders Learn Context, and How to Make Them Look at the Cells

*Manuscript v2 (2026-10-29).* Every test number below comes from locked configurations evaluated once (`configs/locked_*.yaml`, hashes in `reports/experiment_log.md`). Development numbers are labelled as such.

## Abstract

Microscopy datasets often label a sample only with the set of components it contains: the species in a culture, the cell types in a smear or tissue patch. Decoders trained on such set labels over frozen tile embeddings reach useful image-level accuracy, but what they look at has not been measured.

On two datasets with instance annotations, used for evaluation only, we find that set-trained decoders are *right for the wrong tiles*. On held-out test sets their per-tile evidence is anti-localised for several classes: AUROC against instance ground truth is 0.04–0.19 on BBBC041 blood smears from a different acquisition, and 0.08–0.24 on PanNuke tissue. This holds for set networks and for attention-based multiple-instance learning, whose attention maps are commonly read as localisation.

We introduce two corrections, each derived from what set labels do and do not identify:

1. **Label-valid tile constraints.** An absent class has no evidence in any tile; a present class has evidence in its upper-quantile tiles; a pure sample has evidence everywhere. On test they restore localisation to 0.76–0.98 at a cost of at most 0.03 image AUROC.
2. **Contextual logit adjustment (CLA).** CLA generalises logit adjustment from the marginal label prior to the label-conditional prior p(y_k | y_{−k}), added during training only. On PHOEBI, a bacterial benchmark that holds out whole species combinations, CLA improves attention-MIL by +0.039 [0.012, 0.066] and +0.041 [0.015, 0.066] mean AUROC on 27 unseen combinations. It does not help where co-occurrence is stable and informative (PanNuke), as its derivation predicts.

We also report a negative result: a purely instance-level objective that led development by +0.03 to +0.06 AUROC was the worst method on test. Development folds with 26 combinations did not rank methods reliably.

## 1. Introduction

The label available at scale in microscopy is often a set: the species confirmed in a culture, the cell types present in a smear or a tissue patch. Multiple-instance and set decoders over frozen foundation-model features are the standard way to learn from such labels. They are usually judged by image-level accuracy, and their attention or tile scores are often shown as evidence of where the components are.

We ask what these decoders use. Image-level accuracy cannot answer this, because a decoder can score a class correctly from the cells of *other* classes that co-occur with it. That is harmless while co-occurrence is stable and harmful when it shifts, for example on species combinations never seen in training.

**Contributions.**

1. **Diagnostics with the right unit of replication:**
   - tile localisation against instance ground truth used only for evaluation;
   - an own/context decomposition of scores at fixed own evidence;
   - a decorrelated AUROC that reweights evaluation images so class presences are independent;
   - combination-level (or partition:combination-level) intervals for compositional benchmarks.
2. **Finding:** set-trained decoders are accurate yet anti-localised for several classes. This replicates on locked test sets of two datasets and across three decoder families (§5.1).
3. **Label-valid tile constraints**, which restore localisation at small cost (§5.2).
4. **Contextual logit adjustment**, with its derivation and its failure condition, validated on held-out combinations (§5.3).
5. **Negative results reported in full** (§5.4).

## 2. Related work

- **Benchmarks and baselines.**
  - PHOEBI introduces leave-combinations-out evaluation for bacterial mixtures. It reports that a trained aggregator "fits P(y | combination)", and that mean, quantile and max pooling, finer units and learned subspaces stay near 0.6 AUROC.
  - Borowa et al. (JBHI 2023) apply MIL to four-species polycultures.
  - Svatko et al. (ICML 2026) show that simple baselines rival representation learning in bioimaging.
- **Contextual bias in multi-label recognition.** Singh et al. (CVPR 2020), CCD (CVPR 2022) and Chen et al. (NeurIPS 2024) de-bias features or label correlations and evaluate on out-of-context images. We study set-only supervision, evaluate on unseen combinations and measure localisation directly.
- **MIL and pooling.**
  - Attention MIL (Ilse et al. 2018), class-wise pooling (Durand et al. 2017) and quantile pooling (Couture et al. 2018) are antecedents of our decoders. We claim no novelty for pooling.
  - Our tile constraints combine the standard MIL rules with exact pure-sample positives.
- **Prior correction and bias products.**
  - Logit adjustment (Menon et al., ICLR 2021) and its binary form (CUE) correct the marginal prior. Distribution-Balanced loss (Wu et al., ECCV 2020) uses co-occurrence only as sample weights.
  - Product-of-experts debiasing (Clark et al., EMNLP 2019; RUBi, NeurIPS 2019) adds a bias model's log-probabilities during training and removes it at test. CLA is closest to this, with the bias model replaced by a label-only model of the *other ground-truth labels*.
  - BCP (ICML 2026) does the mirror operation: it injects a label-conditional prior at test time.

## 3. Setting and protocol

**Data.**

| Dataset | Unit and size | Representation | Split |
|---|---|---|---|
| PHOEBI | 6 bacterial species, 40 combinations, one video per combination; bags of 1 or 4 images | 16 tiles of 224 px each, frozen DINOv2-S/14 with PHOEBI's illumination correction | Three leave-combinations-out partitions (9 held-out combinations each, 27 pooled); 5 development folds over 26 trained combinations |
| BBBC041 | *P. vivax* smears, 5 classes, 981 images | 35 tiles each | Official test set from another acquisition |
| PanNuke | H&E patches, 5 nucleus types, 7,901 patches | 16 tiles each | Official fold 3 as test |

Instance boxes and masks are used only for evaluation.

**Protocol.**

- Methods, hyper-parameters and the primary method were chosen on development data, locked by hash, and each test set was read once.
- Intervals are 95 % bootstraps over combinations (PHOEBI; resampling unit partition:combination) or images.
- Comparisons are paired on identical bags.
- Because the three PHOEBI partitions are three separately trained models, we also report AUROC after per-partition rank normalisation. It uses no labels. This was decided after development and is reported alongside the raw value.

## 4. Methods

**Decoders.**
- Linear heads on mean and mean+std tile embeddings.
- Deep Sets.
- Class-wise gated attention MIL (ABMIL).
- PopEvidence: per-class tile evidence, pooled per class.

**Label-valid tile constraints**, on the per-tile evidence a_{t,k} (for ABMIL, the attention logit):
- Absent class: softplus(a_{t,k}) averaged over tiles.
- Present class in a mixture: softplus(−q_{0.9}(a_{·,k})).
- Pure sample: softplus(−a_{t,k}) averaged over tiles.

None of these terms depends on which other classes are present.

**Contextual logit adjustment.** Decompose the class-k logit of the BCE-optimal posterior:

  logit p(y_k=1 | x) = LLR_k(o_k) + logit p(y_k=1 | y_{−k}).

This holds when x carries own evidence o_k and the other classes' content informs y_k only through y_{−k}. CLA:
- trains on e_k + τ·b_k(y_{−k}), where b_k is a regularised logistic model of y_k given the other ground-truth labels, fitted on training labels only;
- scores with e_k alone.

At the optimum with τ = 1, e_k is the own-evidence likelihood ratio, which is invariant to the combination distribution. The derivation predicts CLA helps when co-occurrence shifts between training and test, and costs accuracy when co-occurrence is stable and carries real information.

**Instance-level supervision.** Our development-selected primary method trains only the tile constraints, with no bag-level loss, and scores a sample by the mean (or median) of its tile evidence.

## 5. Results

### 5.1 Set-trained decoders are right for the wrong tiles

Tile localisation AUROC on locked test sets; 0.5 = chance, < 0.5 = anti-localised.

| Test set | Decoder (unconstrained) | Image AUROC | Anti-localised classes |
|---|---|---|---|
| BBBC041 (other acquisition) | ABMIL | 0.545 | gametocyte 0.08, ring 0.19 |
| BBBC041 | PopEvidence | 0.439 | schizont 0.04, trophozoite 0.07 |
| PanNuke fold 3 | PopEvidence | 0.880 | dead 0.08, epithelial 0.18, connective 0.24 |

On development data the same pattern holds: BBBC041 trophozoite 0.076 [0.068, 0.083] at image AUROC 0.82; PanNuke dead 0.115. At fixed own presence and count, 11–17 of 20 context coefficients are resolved on BBBC041.

### 5.2 Label-valid tile constraints restore localisation

| Test set | Constrained decoder | Localisation | Image AUROC (Δ vs unconstrained) |
|---|---|---|---|
| BBBC041 | ABMIL + tiles | gametocyte 0.97, schizont 0.95, trophozoite 0.86; ring 0.24 | 0.548 (+0.003) |
| BBBC041 | ABMIL + tiles + CLA | 0.98 / 0.95 / 0.93; ring 0.19 | 0.577 |
| PanNuke | ABMIL + tiles | 0.77–0.92 for all types | 0.874 (−0.010) |
| PanNuke | instance-only | 0.76–0.92 | 0.867 |

Ring stages stay anti-localised after the acquisition shift. This is a limitation.

### 5.3 Contextual logit adjustment under combination shift

PHOEBI locked test, 27 held-out combinations.

| Method | AUROC n=1 | AUROC n=4 | Rank-normalised n=1 / n=4 |
|---|---|---|---|
| **ABMIL + CLA** | **0.671** [0.602, 0.743] | **0.673** | **0.679 / 0.683** |
| mean (linear) | 0.654 | 0.659 | 0.667 / 0.677 |
| mean+std (linear) | 0.654 | 0.657 | 0.664 / 0.670 |
| ABMIL | 0.632 | 0.632 | 0.652 / 0.656 |
| Deep Sets | 0.634 | 0.618 | 0.644 / 0.630 |
| instance-only (dev-selected primary) | 0.561 | 0.551 | 0.600 / 0.600 |
| ProtoMatch (published) | 0.63 [0.55, 0.71] | | |

Paired comparisons on the same bags:

| Comparison | n=1 | n=4 |
|---|---|---|
| CLA vs ABMIL | **+0.039 [0.012, 0.066]** | **+0.041 [0.015, 0.066]** |
| CLA vs linear mean | +0.017 [−0.012, 0.048] | +0.013 [−0.025, 0.048] |

The development estimate of the ABMIL gain was +0.037.

The failure condition behaves as predicted. On PanNuke, where co-occurrence of nucleus types is stable:
- CLA lowers test AUROC (0.858 vs 0.874 with tile constraints).
- Adding tissue type as a covariate to the prior does not recover it (development decorrelated AUROC 0.839 vs 0.846).

### 5.4 External test: a new culture batch (PHOEBI four-species subset)

Locked before download (`docs/protocol_external.md`). Trained on all of phoebi6, tested on 14 phoebi4 videos cultured in a separate batch.

A pre-registered check found that the frozen embedding of 3 of the 4 phoebi4 pure cultures is closer to a *different* phoebi6 species than to the same species: batch shift dominates species identity (F12).

| Method | AUROC n=1 | AUROC n=4 |
|---|---|---|
| linear mean | **0.682** | **0.680** |
| ABMIL + tiles | 0.664 | 0.669 |
| mean+std | 0.659 | 0.644 |
| ProtoMatch | 0.642 | 0.654 |
| instance-only | 0.638 | 0.629 |
| Deep Sets | 0.629 | 0.605 |
| ABMIL | 0.622 | 0.582 |
| ABMIL + CLA | 0.620 | 0.590 |

- **CLA's gain does not transfer to batch shift:** −0.002 / +0.008.
- **Flexible decoders lose most:** linear mean − ABMIL = +0.06 / +0.10.
- **Tile constraints reduce ABMIL's loss:** +0.042 / +0.087 [0.010, 0.169]. This is exploratory: the comparison was in the locked set but was not a stated hypothesis.

### 5.5 Negative results

- **Instance-level supervision without a bag loss** led development: 0.635 / 0.648 against 0.600 / 0.615 for mean+std, and 0.545 for ProtoMatch on the same folds. It was the worst method on test (−0.09 vs the linear mean, P(Δ>0) = 0.04), over-predicting labels (4.1 per sample vs 3.11 true).
- **Distribution statistics beyond the mean** helped only on PHOEBI development (+0.021 / +0.037 AP) and hurt on BBBC041 (−0.038) and PanNuke (−0.030).
- **Combination mixing** (label-union augmentation) did not help.
- **Tile constraints did not reduce image-level context reliance.** The normalised context share stayed at 0.35 → 0.34.

## 6. Discussion and limitations

**What generalises.** Localisation failure and its repair generalise across datasets, acquisitions and decoder families. They are the main message: image-level accuracy of set-trained decoders does not indicate that they look at the right cells.

**What depends on the shift.**
- CLA helps when the label-combination distribution shifts within one culture batch (phoebi6 test). It does not help under a batch or session shift (phoebi4), nor when co-occurrence is stable and informative (PanNuke).
- Under batch shift, the most robust decoder is the simplest: a linear head on mean embeddings. Flexible set decoders memorise sessions. Tile constraints partly protect attention-MIL.

**Limitations.**
- **Small test sets.** PHOEBI has 27 test combinations; per-class statements are rarely resolvable.
- **Selection was unreliable.** Development selection with 26 combinations mis-ranked our primary method. We report it rather than re-select after seeing the test.
- **BBBC041 acquisition shift** depresses every method.
- **Frozen encoder.** One encoder (DINOv2-S/14) and tile-level, not cell-level, units.
- **Post-hoc metric.** Rank normalisation across partitions was decided after development.
- **No evaluation set remains unused** for further method development.

## Reproducibility

- Code: `popev/`, `scripts/`.
- Configurations: `configs/`.
- Locked test configurations: `configs/locked_{phoebi,bbbc041,pannuke}.yaml`.
- Logs: `reports/experiment_log.md`.
- Findings with all intermediate analyses: `docs/findings.md` (F1–F11).
