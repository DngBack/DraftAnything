# Pre-analysis plan: linguistic structure in handwritten inter-word gaps

Written 2026-11-09, before any gap had been linked to a linguistic variable. Changes after this date go in the "Deviations" section at the end, with the reason.

## Question
When people copy English text by hand, do the spaces they leave between words reflect the text's linguistic structure, beyond letter shapes, word length and layout?

## Data and data discipline
IAM offline, `lines`, `xml`, `ascii/sentences.txt`.

| Role | Writers | Use |
|---|---|---|
| Discovery | train writers (train lines + their unassigned lines) | Build and debug everything |
| Replication | val1 writers | Run once the discovery pipeline is frozen |
| Confirmation | val2 + test writers | Sealed. No gap is computed until this plan's final specification is frozen, then run once |

A text prompt is the form id without its writer suffix (`a01-000u` -> `a01-000`). Prompts can be shared across splits; writers cannot.

## Unit and outcome
A boundary is two adjacent word tokens on one handwritten line, both containing a letter or digit, in a `segmentation=ok` line.

Excluded: the right token starts with an apostrophe (clitics such as `'s`, `n't`); either token has no ink.

Outcome: `gap = (ink_x0[right] - ink_x1[left]) / median core height of the line`, centred within line.

## Controls (orthography, length, layout)
- Left and right token: character count, uppercase, digit, ascender and descender counts.
- Identity of the left word's last character and right word's first character (one-hot).
- Log word frequency of both tokens (from the IAM development text).
- Position: relative position in line, first and last boundary flags, token count, normalised x position, remaining line width.
- Line fixed effects.

## Tests
**T1. Between-writer agreement (no language model).** Residualise gap on controls (cross-fitted by prompt). For each boundary written by at least two writers, correlate a writer's residual with the mean residual of other writers at the same token position in the same prompt.
Reference conditions:
- Same word pair, different prompt: this keeps pair-level orthography and word identity but changes the sentence.
- Label permutation within prompt.

**T2. Linguistic predictors.** OLS on residualised gap with line fixed effects, standard errors clustered by writer. Primary family of three, Holm-corrected, two-sided:
1. `surprisal`: GPT-2-large surprisal of the right word given preceding text in the form, in bits.
2. `tree_dist`: path length between the two words in the constituency parse (boundary strength).
3. `clause_start`: the right word begins a clause (S, SBAR or SBARQ in the parse).

Coefficients are per one SD of the predictor. Predicted direction is positive.

## Decision rules
- **Stop the linguistic claim** if no T2 predictor has a Holm-significant effect in discovery with the same sign and a 95% CI excluding zero in replication.
- **Too small to matter:** if the largest replicated effect is under 0.02 core heights per SD, report as null-in-practice. For scale, the within-line SD of gap in real IAM lines is about 0.39 core heights.
- **T1 is interpreted only against its reference conditions.** Agreement that also appears for the same word pair in a different prompt is attributed to the word pair, not the sentence.

## Known threats
- IAM is copied text, so results speak to copying, not composition.
- A spatial gap is not a pause. Without IAM-OnDB timing the link to processing time is an inference.
- Parser and language model were trained on modern text, while IAM's text comes from the 1961 LOB corpus.
- `segmentation=ok` selects cleaner lines.
- Surprisal is confounded with word length and frequency, which is why these are controlled.

## Frozen specification (added 2026-11-09 after Discovery, before Replication was analysed)
Primary: ridge controls, joint three-predictor model, writer-clustered SE, as above. Replication is judged by the decision rules; Confirmation (val2 + test writers, treated as one set) is run once on this exact code.

Robustness, all pre-stated here and reported whether or not they help:
- `lower`: both tokens fully lowercase letters (removes names, sentence-initial capitals, digits).
- `pos`: adds one-hot LOB tags of both tokens (top 30 per side). Tests whether structure survives word class.
- `clip`: gap clipped to the Discovery 1st to 99th percentile.
- `gbm`: gradient-boosted controls.
- `prompt` clustering instead of writer clustering.
- `pythia`: surprisal from Pythia-1B-deduped, a model with different training data, replacing GPT-2-large.
- Placebo outcome: mean background intensity of the two words. No linguistic predictor should explain it.

## Outcome against the plan (run 2026-11-09)
Confirmation was opened once. No predictor passed every stage: clause onset met the Replication rule but failed Confirmation, surprisal failed the Replication CI (same sign and size, 46 writers) and passed Confirmation. T1 could not be run on Confirmation because no prompt there has two writers or is shared with another set.

Added after Confirmation, exploratory: inverse-variance pooling, the printed-spacing control on forms A-H, and the prompt-level diagnosis of T1.

## Follow-up analyses (written 2026-11-09, before either was computed; exploratory)
All writer sets have been seen, so neither analysis is a confirmation. Both use the frozen T2 code (ridge, writer clustering, joint model) on Discovery, Replication and Confirmation, and pool the three by inverse variance. Replication + Confirmation is also pooled alone, because Discovery chose the specification.

**A. Clause label set.** Rebuild `clause_start` with the labels this plan names (S, SBAR, SBARQ), since the code also used SINV and SQ (Deviation 3). If the pooled coefficient moves by more than 0.005, the paper reports the plan's definition as primary.

**B. Is the surprisal effect contextual?** `surprisal_nc`: GPT-2-large surprisal of the right word alone, following only the start-of-text token. It is added to the joint model. The `surprisal` coefficient then measures predictability from context with the word's context-free probability held fixed.
- If the pooled CI of `surprisal` excludes zero and the estimate is at least 0.02, the claim "predictability in context" stands.
- If the CI excludes zero but the estimate is under 0.02, the paper says the contextual part is detectable but below the plan's size threshold.
- If the CI includes zero, the paper withdraws the claim about context and describes the effect as lexical.
The same model is reported for lowercase-only boundaries.

**Results (run 2026-11-09, `reports/acl/followups.{csv,json}`).** Pooled over three sets, with Replication + Confirmation alone in parentheses:
- A: clause onset moves from 0.0158 to 0.0154, under the 0.005 limit, so the five-label definition stays primary and the difference is reported. Under the plan's labels Confirmation stays non-significant (0.0071 [-0.0062, 0.0203]).
- B, all boundaries: `surprisal` 0.0310 [0.0204, 0.0416] (0.0344 [0.0185, 0.0502]); `surprisal_nc` -0.0080 [-0.0228, 0.0068]. The first rule applies: the claim about context stands.
- B, lowercase-only: `surprisal` 0.0129 [0.0029, 0.0229] (0.0121 [-0.0039, 0.0282]). Over the three sets this falls under the second rule (detectable, below 0.02). Without Discovery the interval includes zero.

**BRUSH (Kotani et al. 2020) was examined and not used.** Its prompts are strings of 2-4 unrelated words (median 2), and the 10 ms resampling keeps no pen-up interval. It can test neither sentence-level effects nor inter-word pauses.

## Deviations
0. The plan quoted 0.39 core heights as the within-line SD of gap. That value used a different normalisation. Under this plan's outcome the residual within-line SD is about 0.67 core heights. The 0.02 threshold is unchanged.
2. T1 caps any boundary or word-pair group at 60 rows (random sample) because the first run ran out of memory on very common pairs. No result from that run was seen.
1. 2026-11-09, before any gap was computed: lines by writers who appear in no official split (2,970 lines) are added to Discovery. Reason: Discovery alone had only 44 prompts written by two or more writers, too few for T1. These writers are disjoint from every official split, so the sealed sets stay sealed.
3. Found 2026-11-09 after Confirmation: `text_features.py` labels a clause onset with S, SBAR, SBARQ, SINV or SQ, while this plan names only the first three. All reported `clause_start` results used the five-label set. Follow-up A measures the effect of the difference.
