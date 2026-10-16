# What makes MLLM/VLM analysis and mechanism papers succeed or fail at top venues (for a CVPR 2027 "late-layer readout, not perception" paper with a training-free fix)

Scope note: background on the CVPR process (Findings track, highlight rates, reviewer guidelines) and on general "simple insight" award papers (Registers, Eyes Wide Shut, DVT) is in `../Ý tưởng nghiên cứu CVPR 2027/cvpr_process_and_spotlight_criteria.md` and `simple_insight_paper_patterns.md`. This file does not repeat it.

Method note: OpenReview forum pages and the `/notes?forum=` API returned a "ChallengeRequiredError" (bot check) on 2026-10-13. The `/notes/search?source=reply` endpoint still worked, so the review text below comes from that endpoint. Some threads came back partial (e.g., only 1 of 3–4 reviews; no rebuttal text for some). Partial coverage is flagged where it applies. Status tags: **[V]** = verified this session on an official page or OpenReview venue field; **[U]** = unverified (prior knowledge or secondary source only).

---

## Q1. Which MLLM/VLM analysis or mechanism papers reached oral, highlight, spotlight or award status (2024–2026), and what status did the named papers actually get?

### Takeaway
Pure MLLM analysis papers do reach oral or highlight, but rarely. The confirmed top-tier cases are Eyes Wide Shut (CVPR'24 oral), Visual Symbolic Mechanisms (ICLR'26 **oral**, with no fix), and two CVPR'26 highlights: "Same Attention, Different Truths" (analysis plus training-free fix) and "What Do Visual Tokens Really Encode?" (analysis). Most of the closest precedents for this paper's claim ("the model sees, but the late readout fails") were accepted as **posters**: "MLLMs Know Where to Look" (ICLR'25), "See What You Are Told" (ICLR'25), "Seeing but Not Believing" (ICLR'26), and "Towards Interpreting Visual Information Processing" (ICLR'25). "Hidden in Plain Sight" went to COLM 2025, not a vision venue.

### Cited Findings

**Oral / highlight / spotlight, verified**
- *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs*: CVPR 2024 **Oral** [V] — [CVPR oral page](https://cvpr.thecvf.com/virtual/2024/oral/32016)
- *Visual symbolic mechanisms: Emergent symbol processing in Vision Language Models*: ICLR 2026 **Oral** [V] (OpenReview venue field "ICLR 2026 Oral", forum 3RQ863cRbx) — [OpenReview](https://openreview.net/forum?id=3RQ863cRbx)
  - Pure mechanism paper with no fix. It identifies content-independent spatial "position IDs" for binding and shows that "binding errors… can be traced directly to failures in these mechanisms."
  - Review scores were only **6/6/6/8**, yet it became an oral (see Q2).
- CVPR 2026 **Oral** list [V], analysis-flavoured MLLM titles — [CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral):
  - *Understanding Task Transfer in Vision-Language Models*: a systematic study of 3 open VLMs × 13 perception tasks with a new metric (PGF) (abstract retrieved via the OpenReview search API, where the venue field reads "CVPR 2026"; oral status from the CVPR orals page above)
  - *FINER: MLLMs Hallucinate under Fine-grained Negative Queries*: benchmark plus tuning, with a claim-style subtitle
- CVPR 2026 **Highlights** [V] — [CVPR 2026 Highlights](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026):
  - *Same Attention, Different Truths: Put Logit-Lens over Visual Attention to Detect and Mitigate LVLM Object Hallucination*. The abstract says real and hallucinated objects get "equally strong visual attention in the model's mid-to-late layers", so "the key issue may not be how much the model attends, but what it attends to and why". Logit-lens decoding of the attended regions separates the two cases, and a mitigation follows. **This is the closest accepted CVPR precedent to the proposed paper.**
  - *What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in MLLMs*. It introduces a probing tool (EmbedLens), finds sink/dead/alive visual tokens (~60% alive), and reports that "internal visual computations… are redundant for most" tasks.
  - *One Layer's Trash is Another Layer's Treasure: Adaptive Layer-wise Visual Token Selection in LVLMs* (layer-wise analysis plus method)
- CVPR 2025 **Oral** [V], MLLM titles with an analysis component — [CVPR 2025 Orals](https://cvpr.thecvf.com/virtual/2025/events/oral):
  - *Thinking in Space: How MLLMs See, Remember, and Recall Spaces* (benchmark plus probing)
  - *Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding* (FarSight: training-free, plug-and-play decoding)
  - *Mitigating Hallucinations in LVLMs via DPO: On-Policy Data Hold the Key* (claim-style subtitle)
- ICLR 2025 **Spotlight** [V]: *Reducing Hallucinations in LVLMs via Latent Space Steering* (VTI; training-free steering) — [OpenReview LBl7Hez0fF](https://openreview.net/forum?id=LBl7Hez0fF)
- NeurIPS 2025 **Oral** [V]: *Boosting Knowledge Utilization in MLLMs via Adaptive Logits Fusion and Attention Reallocation* — [OpenReview qYkhCah8OZ](https://openreview.net/forum?id=qYkhCah8OZ)
- NeurIPS 2025 **Spotlight** [V]: *VLMs have Tunnel Vision: Evaluating Nonlocal Visual Reasoning in Leading VLMs* (evaluation/analysis) — [OpenReview bRWkBD2BfK](https://openreview.net/forum?id=bRWkBD2BfK)
- ICML 2026 **Spotlight** [V]: *Mitigating Hallucinations in LVLMs via Causal Route Gating* — [OpenReview LIcj73RLX6](https://openreview.net/forum?id=LIcj73RLX6)

**Named precedents that were posters or non-vision venues, verified via OpenReview venue fields**
- *MLLMs Know Where to Look: Training-free Perception of Small Visual Details*: ICLR 2025 **Poster** [V]. Finding: "MLLMs know where to look even when they fail to answer", with a training-free crop fix, 2 MLLMs and 7 benchmarks — [OpenReview DgaY5mDdmT](https://openreview.net/forum?id=DgaY5mDdmT); [arXiv 2502.17422](https://arxiv.org/abs/2502.17422v1)
- *See What You Are Told: Visual Attention Sink in LMMs*: ICLR 2025 **Poster** [V] — [OpenReview 7uDI7w5RQA](https://openreview.net/forum?id=7uDI7w5RQA); [ICLR poster page](https://iclr.cc/virtual/2025/poster/30795)
- *Towards Interpreting Visual Information Processing in VLMs* (LLaVA logit lens; tokens refine toward interpretable words in late layers): ICLR 2025 **Poster** [V] — [OpenReview chanJGoa7f](https://openreview.net/forum?id=chanJGoa7f)
- *Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations* (logit lens plus ProjectAway, training-free): ICLR 2025 **Poster** [V] — [OpenReview 94kQgWXojH](https://openreview.net/forum?id=94kQgWXojH)
- *Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Answer Correctness in VLMs*: ICLR 2026 **Poster** [V]. Training-free VEA fix, 8 VLMs — [OpenReview JAI7afWA9e](https://openreview.net/forum?id=JAI7afWA9e)
- *Hidden in plain sight: VLMs overlook their visual representations* (Fu, Bonnen, Guillory, Darrell): **COLM 2025** [V]. Claim: VLMs fall to near chance on vision-centric tasks that their own encoders solve, and the bottleneck is the LLM's use of the features — [OpenReview qQb1JLrwol](https://openreview.net/forum?id=qQb1JLrwol); [arXiv 2506.08008](https://arxiv.org/abs/2506.08008v1)
- *Cross-modal Information Flow in MLLMs* (attention knockout, LLaVA series; early layers carry global context, middle layers carry object-specific info, late layers route to the final position): CVPR 2025 **Poster** [V] — [CVPR poster page](https://cvpr.thecvf.com/virtual/2025/poster/33685); [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Cross-modal_Information_Flow_in_Multimodal_Large_Language_Models_CVPR_2025_paper.pdf)
- *Unveiling Visual Perception in Language Models: An Attention Head Analysis Approach*: CVPR 2025 Poster [V] — [CVPR poster page](https://cvpr.thecvf.com/virtual/2025/poster/32684)
- *Lost in Embeddings: Information Loss in VLMs*: appears on OpenReview only as ACL ARR 2025 Feb/May submissions. Final venue not verified [U] — [OpenReview search result 91IjzhYxWP](https://openreview.net/forum?id=91IjzhYxWP)
- *Large Vision-Language Models Get Lost in Attention* (attention is subspace-preserving, FFN subspace-expanding): **ICML 2026** regular [V] — [arXiv 2605.05668](https://arxiv.org/pdf/2605.05668)
- *Do Audio-Visual LLMs Really See and Hear?* (mechanistic analysis of modality conflict; deeper fusion layers privilege vision): hosted under the **CVPR 2026 Findings** proceedings path (`CVPR2026F/…_CVPRF_2026_paper.pdf`). The PDF itself returned 403, so this is inferred from the URL only — [CVF URL](https://openaccess.thecvf.com/content/CVPR2026F/papers/Selvakumar_Do_Audio-Visual_Large_Language_Models_Really_See_and_Hear_CVPRF_2026_paper.pdf)

**Unverified this session [U]**
- Cambrian-1 as a NeurIPS 2024 Oral (prior knowledge only).
- FastV ("An Image is Worth 1/2 Tokens After Layer 2") as an ECCV 2024 Oral (prior knowledge only).
- The CVPR 2025 highlight list could not be parsed (that page returned a 30 KB shell). Highlight status for "Your LVLM Only Needs a Few Attention Heads for Visual Grounding", "Devils in Middle Layers" and "What's in the Image?" (all CVPR 2025 papers) is therefore unknown — [CVPR 2025 OA](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_CVPR_2025_paper.html); [Devils poster](https://cvpr.thecvf.com/virtual/2025/poster/35035)

**Direct concurrent competition: ICLR 2027 submissions (anonymous, under review, visible on OpenReview as of 2026-10-13) [V that they exist]**
- *Causal Readout Dependence in Multimodal Language Models*. Transplanting image-conditioned states recovers 89.1%/82.8%/48.0% of the image effect under forced choice in Qwen/Mistral/Llama VLMs, but only 14.8%/0.3%/7.4% at generation. The framing is "readout-dependent" causal effects. — [OpenReview dORR3BQrmq](https://openreview.net/forum?id=dORR3BQrmq)
- *Many Paths, One Coordinate: Semantic Mediation and Readout Dissociation in VL Hallucinations*. Across Qwen3-VL and InternVL2, many upstream failures converge on a semantic "object-presence" coordinate that is "distinct from the model's direct output readout". — [OpenReview HBp4uTW5b0](https://openreview.net/forum?id=HBp4uTW5b0)
- *Object Hallucinations Are Written by Late Feed-Forward Layers*. In the last layers, attention favours present objects while the FFN favours hallucinated ones ("FFN overcommitment"). Proposes training-free Component-wise Contrastive Decoding, tested on 4 MLLMs. — [OpenReview tunVuLrXNu](https://openreview.net/forum?id=tunVuLrXNu)
- *Vision-Language Models Do Not Keep Looking*. Visual influence concentrates in intermediate layers and is "very weak in the final layers". — [OpenReview DnUl5Cio8T](https://openreview.net/forum?id=DnUl5Cio8T)
- *TRACE: Causal Tracing and Editing Visual Evidence Retrieval…*. "Visual Retrieval Heads", with gains of up to 11.7%. — [OpenReview KS9coFbr9Q](https://openreview.net/forum?id=KS9coFbr9Q)
- *Now You See It, Now You Don't*. VLMs "can often localize relevant regions… yet still fail". The relevant information sits in early-to-mid layers. — [OpenReview q92eqsV8QI](https://openreview.net/forum?id=q92eqsV8QI)
- *Beyond Feature Fidelity: Recovery and Readout Adaptation in MLLMs* — [OpenReview 1ki2Qs08aF](https://openreview.net/forum?id=1ki2Qs08aF)

### Inferences
- At CVPR, the "MLLM sees but fails to answer" story has precedent at **highlight** level ("Same Attention, Different Truths", CVPR'26), but at ICLR the same story mostly landed as **posters**: Know Where to Look, Seeing but Not Believing, attention sink, logit-lens editing. The ingredients are not enough on their own. A paper reaches oral when it either (a) names a new, general mechanism with strong causal evidence across many models (Visual Symbolic Mechanisms), or (b) exposes a broad, surprising failure with a clean benchmark (Eyes Wide Shut).
- **Novelty risk is very high.** At least 4–5 ICLR 2027 submissions posted this cycle make near-identical claims ("readout dissociation", "late-layer FFN writes hallucinations", "visual influence weak in final layers"). By the CVPR 2027 deadline (Nov 16, 2026) they will be public on OpenReview, and CVPR reviewers can cite them. Under CVPR policy, concurrent work posted within the preceding months is not grounds for rejection, but the paper must still cite them, differentiate from them, and preferably compare against them.
- Mechanistic analysis without a strong hook can be routed to the **CVPR Findings** track ("Do Audio-Visual LLMs Really See and Hear?" appears under CVPR2026F).

### Gaps
- Oral/highlight status for CVPR 2025 highlights, ICCV 2025 and ECCV 2024 MLLM analysis papers could not be enumerated: the CVPR 2025 highlight page did not parse, and ICCV/ECCV pages were not fetched within budget.
- Cambrian-1 (NeurIPS'24 oral) and FastV (ECCV'24 oral) were not re-verified.
- The final venue of "Lost in Embeddings" is unknown.
- I found no "VLMs know but don't tell" paper under that exact title.

---

## Q2. What did reviewers praise and criticise in public reviews, and how did authors rebut?

### Takeaway
Across six VLM mechanism papers with public reviews, the same objections recur:
1. Limited model coverage (one family, or LLaVA only).
2. "Already known in LLMs / prior work" (OPERA, LLM binding IDs, StreamingLLM, Registers, "Unveiling the Ignorance of MLLMs").
3. Correlation is not causation.
4. Synthetic or controlled data only, or only COCO/POPE.
5. Hyperparameters tuned on the test set, layer-specific and benchmark-dependent.
6. The fix "doesn't really fix the mechanism" or hurts other abilities.
7. Cherry-picked qualitative logit-lens examples.

Rebuttals that worked added models from new families (Qwen2-VL, InternVL2), released an interactive demo with random samples, and quantified previously qualitative claims.

### Cited Findings

**Visual Symbolic Mechanisms (ICLR'26 Oral; ratings 6, 6, 6, 8)** — [OpenReview 3RQ863cRbx](https://openreview.net/forum?id=3RQ863cRbx)
- Praise:
  - "novel mechanistic finding in multimodal interpretability"
  - "progression from RSA to CMA to direct interventions provides converging evidence"
  - "7 VLMs show a consistent pattern"
- Weaknesses:
  - "Limited exploration of real-world consequences… do not connect the mechanisms to the downstream behavioral consequences"
  - "Unclear how to improve binding… do not attempt to modify / strengthen the binding mechanisms. This would… enable direct causal claims"
  - "seems incremental since comparable mechanisms are already well-established in language models… what is genuinely new… in the visual domain"
  - "datasets are highly controlled and simplified… two clearly separated… objects"
  - "could the observed differences… reflect their positional encoding priors (RoPE vs sinusoidal)"
  - "The correlation between VLMs failures and position ID mechanism failures is just correlation, not causation"
  - "7 VLMs… from 2 model families"
- The paper became an oral despite all of these (no fix, synthetic data), presumably because of methodological rigour (three converging methods) and a fundamental question (binding). The meta-review text was not retrieved.

**See What You Are Told: Visual Attention Sink (ICLR'25 Poster; ratings 6, 3, 6, 8)** — [OpenReview 7uDI7w5RQA](https://openreview.net/forum?id=7uDI7w5RQA)
- "similar to the summary tokens proposed in OPERA"
- "What's the relationship between sink tokens in different kinds of models? Are they just similar phenomenons, or… special properties of multimodal models?" (with Registers/StreamingLLM cited as prior work)
- "Heavy hyper-parameter tuning for specific tasks… are these hyper-parameters tuned according to the test set?"
- "biggest concern is about the determination of hyperparameter ρ… benchmark-dependent… a poorly chosen ρ can lead to performance worse than the baseline… against the principle of machine learning"
- The reviewer who gave a 3 tried to replicate the token-masking experiment and got a different number ("I randomly selected 25% of the visual tokens… POPE test score was 82.4"). The authors defended the distinction "pruning… vs masking".
- The AC accepted on the strength of "consistent improvements across recent models (Qwen2-VL, InternVL2)" added in the rebuttal and "quantitative validation on… Pascal-VOC, MS-COCO showing sink tokens appear in background regions".
- The meta-review listed "Hyperparameter sensitivity" and "Reproducibility concerns" as the remaining weaknesses.

**Towards Interpreting Visual Information Processing in VLMs (ICLR'25 Poster; visible ratings 8, 6)** — [OpenReview chanJGoa7f](https://openreview.net/forum?id=chanJGoa7f)
- "analysis is performed on only two VLMs from the same (LLaVA) family… it is hard to take any findings at face value when they have only been performed on one model"
- Logit-lens results "hard to tell if these are meaningful correspondences or… cherry-picking/confirmation bias". The reviewer suggested annotating what % of patches map to a matching token.
- "contributions on how we can adopt such information processing pattern on better prompting or training… are limited". In other words, a reviewer asked for a fix.
- Successful rebuttal moves:
  - ran an adapted logit lens on **Qwen2-VL**
  - published an **interactive demo** with "100 randomly picked images"
  - added a quantitative patch–token match rate

**Interpreting and Editing VL Representations to Mitigate Hallucinations (ICLR'25 Poster; ratings 5, 5, 8)** — [OpenReview 94kQgWXojH](https://openreview.net/forum?id=94kQgWXojH)
- "reliance on LLaVA and InstructBLIP… does not provide a comprehensive comparison with the latest state-of-the-art models"
- "requires specifying weight factors and selecting specific layers… vary between models… cumbersome ablation"
- "only… object hallucinations in image captioning… unclear… on VQA"
- "only test… on COCO2014"
- "impact… on overall caption quality is not thoroughly evaluated quantitatively"
- "randomly sample… objects not present… making the comparison of the internal confidence too easy". This asks for co-occurring hard negatives.
- "surprised that the logit lens can be directly used in image representations" given the modality gap. The reviewer asked for logit-lens classification accuracy.

**Seeing but Not Believing (ICLR'26 Poster; ratings 6, 6, 6)** — [OpenReview JAI7afWA9e](https://openreview.net/forum?id=JAI7afWA9e)
- Praise: "simple yet effective… zero training cost… consistent improvements across eight different VLMs"
- Criticism:
  - "does not 'fix the VLM's attention behavior' (as the behavior of the VLM or attention is not changed)". The mechanism–fix link was judged weak.
  - "introduce extra cost… multi-turn/multi-image"
  - "evaluate cases where the model does not attend to the correct regions… Does performance degrade when highlighting regions based on incorrect attention?"
  - "motivation is somewhat similar to [Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly]… reduces the novelty"

**MLLMs Know Where to Look (ICLR'25 Poster; one visible review, rating 8)** — [OpenReview DgaY5mDdmT](https://openreview.net/forum?id=DgaY5mDdmT)
- Praise: "insightful findings… training-free method… significant improvement in 7 benchmarks and on two MLLMs"
- Weakness: "improvement on general VQA benchmarks… not as significant as on small visual concepts"; latency overhead

**VTI / Latent Space Steering (ICLR'25 Spotlight; ratings 8, 6; scores raised after rebuttal)** — [OpenReview LBl7Hez0fF](https://openreview.net/forum?id=LBl7Hez0fF)
- "How many LVLMs are tested here?… Are the observations generalized to different LVLMs?"
- "VCD… also adds Gaussian noise… explain the differences"
- "Only hallucination benchmarks are covered… will VTI damage model performance" on general tasks
- "Using variance changes to explain… stability is somewhat one-sided"
- Meta-review: "both Cwnc and eTNv raising their scores after discussion" after the authors added mask-ratio and feature-stability experiments.

### Inferences
Objection → pre-emptive answer for a "late-layer readout, not perception" paper:
1. **"Only one model / one family"** → ≥3 families (e.g., LLaVA-1.5/OneVision, Qwen2.5/3-VL, InternVL3, plus one non-RoPE or cross-attention model such as Llama-3.2-V or Idefics) at ≥2 scales. "7 VLMs, 2 families" still drew this objection.
2. **"Probe ≠ use / correlational"** → causal interventions are required. Use activation patching or transplant (the ICLR'27 "Causal Readout Dependence" already reports recovery %), ablation of the readout component, and a *dose–response* showing the fix's gain tracks the size of the measured readout gap per item. The ICLR'26 oral reviewer said a fix "would… enable direct causal claims". That makes the training-free fix itself causal evidence, if it targets exactly the identified component.
3. **"Known already"** → the differentiation table must cover: Hidden in Plain Sight (COLM'25), Know Where to Look (ICLR'25), Seeing but Not Believing (ICLR'26), Same Attention Different Truths (CVPR'26 highlight), logit-lens editing (ICLR'25), Cross-modal Info Flow (CVPR'25), and the ICLR'27 readout submissions. State the new *testable prediction* this account makes that the others do not.
4. **"Hyperparameters tuned on test / layer choice per model"** → choose the layer and strength on a held-out split, or set them automatically from an internal statistic. Report one fixed config across all models and benchmarks, plus a sensitivity plot. This was the single biggest complaint against VAR.
5. **"Synthetic only / COCO-POPE only"** → controlled diagnostic set for the mechanism, plus ≥5 real benchmarks spanning perception (MMVP, CV-Bench, V*), hallucination (POPE/CHAIR/AMBER) and general (MMBench/MME/SEED). Also report that general-ability scores do not drop.
6. **"Fix doesn't touch the mechanism"** → the fix must act on the identified readout stage, not on the input (cropping or masking drew this criticism).
7. **"Cherry-picked logit lens"** → quantitative match rates over random samples, plus a public demo.

### Gaps
- Full rebuttal texts and meta-reviews for Visual Symbolic Mechanisms (oral), Seeing but Not Believing, and the NeurIPS'25 oral (Adaptive Logits Fusion) were not retrieved. The bot challenge limited per-forum access, so the reasons the ICLR'26 oral was promoted despite 6/6/6/8 scores are inferred, not documented.
- CVPR reviews are not public, so no CVPR-side objections could be mined directly.

---

## Q3. What structure do successful papers of this type follow (teaser, number of models, fix, real-benchmark gains, page budget)?

### Takeaway
Successful mechanism papers follow **phenomenon (teaser) → localisation and causal tests → named mechanism → simple fix → broad benchmarks**. In the poster-to-oral range, the median model count is roughly 4–8 models across 2–4 families. A training-free fix appears in most ICLR/CVPR acceptances of this type, but the ICLR'26 oral had none. At CVPR, a fix with real-benchmark gains appears to be the de facto requirement (every CVPR'25/'26 oral or highlight found above includes a method, benchmark or tool).

### Cited Findings
- Reviewers repeatedly praised one presentation pattern. Attention-sink reviewers wrote: "first analyzing the property and effects of visual sink tokens, then providing a solution based on these investigations", and "Fig 1 and Fig 2 provide a clear and strong motivation. Fig 3-5… step-by-step motivation and the design choices" — [OpenReview 7uDI7w5RQA](https://openreview.net/forum?id=7uDI7w5RQA)
- Model counts in accepted papers:

  | Paper | Models | Status |
  |---|---|---|
  | Visual Symbolic Mechanisms | 7 VLMs, 2 families | oral |
  | Seeing but Not Believing | 8 VLMs | poster |
  | Know Where to Look | 2 MLLMs, 7 benchmarks | poster |
  | Towards Interpreting | 2 LLaVA models, Qwen2-VL added in rebuttal | poster |
  | "Object Hallucinations… Late FFN" | 4 MLLMs | ICLR'27 submission |

  Sources: [3RQ863cRbx](https://openreview.net/forum?id=3RQ863cRbx), [JAI7afWA9e](https://openreview.net/forum?id=JAI7afWA9e), [DgaY5mDdmT](https://openreview.net/forum?id=DgaY5mDdmT), [chanJGoa7f](https://openreview.net/forum?id=chanJGoa7f), [tunVuLrXNu](https://openreview.net/forum?id=tunVuLrXNu)
- Methods triangulation praised in the oral: "RSA → causal mediation analysis → direct interventions" — [3RQ863cRbx](https://openreview.net/forum?id=3RQ863cRbx)
- The CVPR'26 highlight "Same Attention, Different Truths" has a three-part arc: a negative finding about the prior explanation ("prior work largely attributes this to insufficient visual attention… we find… equally strong visual attention"), then a new diagnostic (logit lens on attended regions), then two named mechanisms with a mitigation — [OpenReview abstract via search](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026)
- Eyes Wide Shut has a similar arc: CLIP-blind pairs → MMVP benchmark → root cause in the encoder → Mixture-of-Features fix — [CVPR oral page](https://cvpr.thecvf.com/virtual/2024/oral/32016)

### Inferences
Proposed CVPR 8-page budget, inferred from the patterns above:
- **p1:** teaser figure. Same image, correct internal evidence (probe or logit lens decodes the right answer at layer L), wrong final output, fixed output after the training-free fix.
- **~0.75p:** intro with 3 numbered findings.
- **~0.5p:** related work as a differentiation table vs the 6–8 closest papers.
- **~3p:** analysis. (i) Perception is intact: encoder and mid-layer linear probes match encoder-only accuracy. (ii) Localisation: layer-wise decodability vs output accuracy curve, with the "readout gap". (iii) Causal: patch or ablate the late readout, restoring accuracy ≥X%. (iv) Generality across ≥3 families and scales.
- **~1.5p:** fix with a single fixed configuration.
- **~1.25p:** real-benchmark results and ablations, including "doesn't hurt general ability" and cost.
- **Supplement:** per-model plots, random-sample galleries, demo link.

A fix without real-benchmark gains, or gains only on the authors' own diagnostic set, is the main CVPR risk.

### Gaps
- No source gives exact page-allocation statistics. The budget above is inferred from paper structure, not measured.

---

## Q4. How do CVPR reviewers treat pure analysis without SOTA, compared with ICLR, and what are the CVPR 2026 Findings signals?

### Takeaway
ICLR has promoted a fix-less mechanism paper to oral (Visual Symbolic Mechanisms, 2026). No CVPR'25/'26 oral or highlight found in this session was a pure mechanism study without a method, benchmark or tool. CVPR's analysis-type orals and highlights all ship an artifact: a metric (PGF), a probing tool (EmbedLens), a benchmark (FINER, Thinking in Space), or a training-free fix (FarSight, logit-lens mitigation). At least one pure MLLM mechanistic analysis appears in CVPR 2026 **Findings**.

### Cited Findings
- ICLR 2026 Oral with no fix and controlled synthetic data — [OpenReview 3RQ863cRbx](https://openreview.net/forum?id=3RQ863cRbx)
- CVPR 2026 Oral "Understanding Task Transfer in VLMs" is an analysis paper, but it introduces a metric (Perfection Gap Factor) and a "task transfer graph" — [CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral)
- CVPR 2026 Highlight "What Do Visual Tokens Really Encode?" is analysis packaged with a named tool, EmbedLens — [CVPR 2026 Highlights](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026)
- CVPR 2026 Highlight "Same Attention, Different Truths" pairs analysis with detection and mitigation — [CVPR 2026 Highlights](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026)
- CVPR 2026 Findings includes "Do Audio-Visual LLMs Really See and Hear?", a mechanistic interpretability study of modality conflict (inferred from the `CVPR2026F` proceedings path; PDF access returned 403) — [CVF URL](https://openaccess.thecvf.com/content/CVPR2026F/papers/Selvakumar_Do_Audio-Visual_Large_Language_Models_Really_See_and_Hear_CVPRF_2026_paper.pdf)
- (Background, already in the sibling notes:) the CVPR 2026 Findings track targets "technically sound papers with more incremental novelty", and 1,717 rejected papers were recommended to it — see `cvpr_process_and_spotlight_criteria.md`.

### Inferences
- For CVPR, keep the training-free fix and show **real-benchmark gains on ≥3 model families with one config**. The fix acts as both the "artifact" CVPR expects and causal evidence for the mechanism claim.
- Packaging the diagnostic as a **named, released tool or metric** (e.g., a "readout gap" score) matches the CVPR'26 highlight pattern (EmbedLens, PGF).
- A pure analysis version without the fix would fit ICLR better than CVPR. At CVPR it risks being routed to Findings as "sound but incremental", especially given the ICLR'27 concurrent submissions.

### Gaps
- CVPR reviews and meta-reviews are not public, so the CVPR-vs-ICLR contrast rests on accept/oral/highlight outcomes, not reviewer text.
- No official CVPR 2027 statement on Findings continuation or analysis-paper policy was found (consistent with the sibling note).

---

## Q5. What title and abstract patterns do successful mechanism papers use?

### Takeaway
Titles are claim- or metaphor-style, often with a colon and a mechanism subtitle: "X Know Where to Look", "See What You Are Told", "Seeing but Not Believing", "Same Attention, Different Truths", "Eyes Wide Shut?", "Hidden in Plain Sight", "What Do Visual Tokens Really Encode?". Abstracts open with the accepted belief, state the surprising contrary finding, name the mechanism, then report the fix and its breadth.

### Cited Findings
- Claim and metaphor titles (all verified above):
  - "MLLMs Know Where to Look: Training-free Perception of Small Visual Details"
  - "See What You Are Told: Visual Attention Sink in LMMs"
  - "Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Answer Correctness"
  - "Same Attention, Different Truths: Put Logit-Lens over Visual Attention…"
  - "Hidden in plain sight: VLMs overlook their visual representations"
  - "FINER: MLLMs Hallucinate under Fine-grained Negative Queries"
  - "VLMs have Tunnel Vision"
  - ICLR'27 submissions in the same claim style: "Object Hallucinations Are Written by Late Feed-Forward Layers", "Vision-Language Models Do Not Keep Looking"
  
  Sources: [DgaY5mDdmT](https://openreview.net/forum?id=DgaY5mDdmT), [7uDI7w5RQA](https://openreview.net/forum?id=7uDI7w5RQA), [JAI7afWA9e](https://openreview.net/forum?id=JAI7afWA9e), [CVPR'26 Highlights](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026), [qQb1JLrwol](https://openreview.net/forum?id=qQb1JLrwol), [CVPR'26 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral), [bRWkBD2BfK](https://openreview.net/forum?id=bRWkBD2BfK), [tunVuLrXNu](https://openreview.net/forum?id=tunVuLrXNu), [DnUl5Cio8T](https://openreview.net/forum?id=DnUl5Cio8T)
- Abstract template in the CVPR'26 highlight: "Prior work largely attributes this to insufficient visual attention. However… we are surprised to find that both real and hallucinated objects receive equally strong visual attention… the key issue may not be how much the model attends, but what it attends to and why… we identify two distinct hallucination mechanisms" — [CVPR 2026 Highlights](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026)
- Know Where to Look abstract: "MLLMs know where to look, even when they provide the wrong answer", followed by a training-free method with breadth ("two MLLMs and seven… benchmarks") — [arXiv 2502.17422](https://arxiv.org/abs/2502.17422v1)
- Hidden in Plain Sight states the bottleneck as a contrast: VLMs "perform substantially worse than their visual encoders, dropping to near-chance" — [arXiv 2506.08008](https://arxiv.org/abs/2506.08008v1)

### Inferences
- The "X know but Y" / "seeing but not" title space is crowded: Know Where to Look, Seeing but Not Believing, Hidden in Plain Sight, Now You See It Now You Don't, and Causal Readout Dependence. A distinct claim-title should name the **stage** and the **direction**, for example a claim that "MLLMs fail at readout, not perception". Before finalising any title, check it against the ICLR'27 submission titles above.
- Recommended abstract skeleton:
  1. Common belief (failures = perception/encoder; cite Eyes Wide Shut, Cambrian).
  2. Counter-finding with a number (the answer is linearly decodable at layer L in X% of failures).
  3. Causal localisation (patching late layers restores Y%).
  4. Named mechanism.
  5. Training-free fix with one config, across N models and M benchmarks.
  6. Implication (where future training effort should go).

### Gaps
- No systematic dataset links title style to acceptance or oral outcome. The pattern rests on exemplars only.
