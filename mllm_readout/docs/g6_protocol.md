# G6 protocol: confirmatory shortcut test, label-specific interchange, price of repair

Written 2026-10-18 ~17:20, before any G6 run. Purpose: turn the exploratory G5 findings that carry the paper
(G5 Addendum 4, the swap depth sweep, the supervised label budget) into tests on data not yet seen.

**Already seen, not counted as confirmation:**
- illusynL (seed 2027) for gemma3_4b, llava15_7b, qwen35_9b and qwen3vl8b, plus the model-free readouts on it.
- Swap results at 0, 12.5, 25, 50 and 75% depth, with same-label controls at 0, 25 and 75%, for gemma3_4b, llava15_7b, qwen35_9b, qwen3vl4b and qwen3vl8b.
- All label-cost numbers on illusions and CountBench.

**Not yet seen:**
- Swap for qwen35_4b, internvl35_8b and qwen2vl_7b (running).
- illusynL readouts for the other four models: the features exist, but G3 states do not, and the features were never analysed.
- Same-label controls at 12.5 and 50%.
- Label cost on dice, tally and carlogos.

Model set: the eight G5 models (qwen3vl4b, qwen3vl8b, qwen35_4b, qwen35_9b, gemma3_4b, internvl35_8b, llava15_7b,
qwen2vl_7b). "In ≥2/3" means at least 6 of the 8, or 2/3 of those evaluated if a model fails to run.

## S' — shortcut test on fresh stimuli (confirmatory)

**Stimuli.** illusynC is illusyn.items(diffs=(0.30, 0.40, 0.50), seed=2028, tag="C_"): 1200 new images rendered the same way as illusynL.

**Readouts.**
- Model-free: pixel statistics, SigLIP2, DINOv2.
- Per model: own LLM input (l0) and the decision state (75% depth) from G3 states. These come from diag_gen with G3_DIR=results/g6_syn.

**Analysis.** Same script as G5 S (g5_syn.py with set illusynC). The linear probe is the primary readout. "within" is 5-fold CV inside a regime, averaged over regimes. "across" is train on one regime, test on the other, averaged. "signed" is which line is longer, on unequal items.

| Test | Criterion |
|---|---|
| S5 | l0 within ≥ 0.65 and l0 across ≤ l0 within − 0.10, in ≥2/3 of models |
| S6 | model accuracy ≤ l0 within − 0.10, in ≥2/3 |
| S7 | l0 signed across ≥ l0 across (equal/unequal) + 0.05, in ≥2/3 |
| S8 | model-free: SigLIP2 within ≥ 0.70, SigLIP2 across ≤ 0.65, pixel within ≤ 0.62 |

- S5: separable within a domain, does not transfer.
- S6: the model trails its own input within the domain.
- S7: relative length transfers better than equality.
- S8: the separability is not a pixel shortcut.

Also reported, not tested: the decision-state rows, the MLP probe and the type shares with E = l0.

## M' — label-specific interchange effect

The G5 M3 control mixed two effects. A same-label partner is a different image, so a probe that is right 78% of the time
disagrees across two same-label images about a third of the time even if it reads only the label. We therefore measure
the label-specific effect.

- Δ_probe(ℓ) = P(probe outputs the partner's label | other-label swap at ℓ) − P(probe outputs the other label | same-label swap at ℓ)
- Δ_answer(ℓ): the same, with the model's answer in place of the probe.

At depth 0 both terms come from stored G3 states (cross-fitted probe and answer on the partner item). The same-label swap is added at 12.5 and 50% (new run, g6_swapctrl.py), so every depth in {0, 12.5, 25, 50, 75}% has both terms.

| Test | Criterion |
|---|---|
| M4 | median over models of Δ_probe ≥ 0.30 at each of 0, 12.5 and 25% |
| M5 | median Δ_probe at 75% ≤ 0.10 |
| M6 | Δ_answer ≤ Δ_probe − 0.15 at 25% depth, in ≥2/3 of models |

- M4: image positions carry the label up to a quarter of the depth.
- M5: by 75% they no longer carry it.
- M6: the answer follows the image less than the probe.

All three are reported on the eight models and, separately, on the three unseen models (descriptive, since three is too few for a 2/3 rule).

## P — the price of a repair, predicted by the diagnostic

**Price.** For readout r (ridge on standardised features, as in explore_labelcost.py), price = the smallest k ∈ {1, 2, 4, 8, 16, 32} labelled items per class (and per wording on illusions) at which mean accuracy over 30 draws is ≥ the raw model + 5 points. If no k qualifies, the price is ∞.

Readouts:
- cal: candidate logits
- inp: own LLM input l0
- dec: decision state at 75% depth
- enc: SigLIP2

**Cells.** Model × set with ≥ 30 errors.
- illusions and countbench: 13 models.
- dice, tally and carlogos: 9 models, after running explore_native for them.
- illusynC: 8 models.

**Held-out prediction.**
- Base images are split in half by a fixed seed.
- Error types are computed on half A, with probes cross-fitted within A and E = l0.
- Prices are measured on half B: few-label draws within B, scored within B.
- The type shares on A predict the prices on B.

| Test | Criterion |
|---|---|
| P7 | cells with BIAS ≥ 1/3 of errors (on A): price(cal) ≤ 8, in ≥2/3 of those cells |
| P8 | cells with BIAS < 0.10 of errors: price(cal) = ∞, in ≥2/3 |
| P9 | cells with INHERITED + DROPPED ≥ 1/2 of errors: price(inp) finite and ≤ price(dec), in ≥2/3 |
| P10 | cells with UNSEEN ≥ 1/2 of errors: price(inp), price(dec) and price(cal) all ∞, in ≥2/3 (vacuous if no such cell) |

**Price table (descriptive).** Rows are sets; columns are repairs.
- The readout repairs above: median price and accuracy at k = 8.
- Steering: the labels it consumes, i.e. all labels outside the test fold, and its gain.
- In-context: k = 8 gain, where available.
- Thinking: 0 labels, and its gain, where available.

MMVP and POPE are excluded, because the question changes per item and the input readout is undefined.

## Deviations
Any change after first results is added below with a timestamp, before the affected result is read.
