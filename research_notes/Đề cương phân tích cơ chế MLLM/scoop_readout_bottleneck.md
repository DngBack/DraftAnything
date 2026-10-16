# Scoop check: "MLLMs encode the right visual information but fail to use it" (late-layer readout/arbitration bottleneck across failure families)

Compiled 2026-10-13 for a CVPR 2027 submission (deadline Nov 16, 2026). Work posted after Sep 15, 2026 counts as contemporaneous. Full text was read (HTML or PDF) for 2604.09364, 2607.09544, 2605.30170, 2606.06890, 2610.00024, 2608.22916, 2510.17771 and 2506.08008. Every other entry comes from its arXiv abstract page or search snippets and is marked "(abstract only)".

**Corrections to `open_problems_vlm_representation.md`:**
- (a) "The Count Is There" is now arXiv **2607.09544** (Jul 10, 2026). Its +15.6 is a *re-prompting* gain for one model on one synthetic split; the average gain is +5.3.
- (b) The "fractured number space" result in 2605.30170 (ICML 2026) comes mostly from a **2-layer toy VLM**. It is validated only on Qwen3-VL-32B with CoT prompting.
- (c) In 2604.09364 the "arbitration" (MAC) layers sit at **36–71% depth**, i.e. mid-depth rather than late. The causal locus is on **image-token positions**; last-token patching does nothing.
- (d) Two newer papers directly complicate a "single shared late bottleneck" claim: 2606.06890 (v2, Sep 27) and 2610.00024 (Sep 1). See Q2.

---

## Q1. What the closest works show precisely (method, models, layers, numbers, families, cross-family tests)

### Takeaway
"Encoded but not used" is now well established *within* single failure families: attribute/colour priors, counting, ordinal labels, spatial transport, text-rich VQA, harmful memes and object hallucination. Each paper uses its own models, metric and locus. The papers **disagree on where the failure sits**: mid-depth image tokens (2604.09364), the late band at the final prompt token (2606.06890), misaligned probe directions (2607.09544), symbolic mapping (2605.30170), or the unembedding (2608.20999). None tests whether one mechanism is shared across families.

### Cited Findings

**Attribute/colour vs. language prior: 2604.09364, "Arbitration Failure, Not Perceptual Blindness" (Nooralahzadeh, Rohanian, Zhang, Fürst, Stockinger; ZHAW/Oxford; v2, Apr 13, 2026)** — [arXiv](https://arxiv.org/html/2604.09364)
- **Models:** 10 VLMs. LLaVA-1.5-7B/13B, LLaVA-Next, LLaVA-OneVision, Qwen2-VL-7B/72B, InternVL2-8B/26B, BLIP-2 and DeepSeek-Janus-Pro.
- **Data:** Visual-Counterfact (Golovanevsky et al. 2025) only. Colour has 493 items and Size has 727. Counting, spatial and OCR are **not tested**.
- **Method:** a logit lens with 6 surface-form variants defines the "Multimodal Arbitration Crossover" (MAC), the first stable layer where the visual logit exceeds the prior logit. It also uses linear probes at 10 layers, L2 distance between counterfactual and standard hidden states, and activation patching at MAC (full-sequence vs. last-token vs. image-only vs. text-only, 100 samples per model).
- **Numbers, encoding:**
  - Probe AUC is > 0.86 at about 10% depth and > 0.99 by mid-depth.
  - Encoding strength vs. success: ρ = 0.198 (p = 0.67). Final-layer logit gap vs. success: ρ = 0.847 (p = 0.016). Sample-level encoding AUC is 0.528.
- **Numbers, patching:**
  - Full-sequence patching flips 60–84% of outputs. Last-token patching flips 0–1%, and text-only patching 0–2%.
  - Image-only patching keeps 97–101% of the effect for LLaVA and Janus but only 70% for Qwen2-VL-7B.
  - There were 0 reverse flips in 900 samples.
- **MAC depth (Colour):** Janus 36%, LLaVA-Next 40%, LLaVA-1.5-7B 42%, LLaVA-OV 60%, Qwen2-VL-7B 71%, BLIP-2 62%, InternVL2-8B 66%, InternVL2-26B 37%, LLaVA-1.5-13B 52%, Qwen2-VL-72B 57%. These are **mid-depth and model-dependent, not a fixed late layer.**
- **Fix:** early-layer linear or SAE steering on 3 models gives **+1.4 to +3.8%**, with some degradation. The best layer for intervention is earlier than MAC.
- **Stated limits:** synthetic images only, Size results that are paraphrase-sensitive, and n = 100 for patching.

**Counting (a): 2607.09544, "The Count Is There, but Misaligned" (El-Shangiti, Nurgazy, AlQuabeh, Rozanov, Inui; MBZUAI/Imperial; Jul 10, 2026)** — [arXiv PDF](https://arxiv.org/pdf/2607.09544)
- **Models:** InternVL2-1B, InternVL2-4B, Qwen3-VL-2B and Qwen3-VL-8B. All are ≤ 8B and come from 2 families.
- **Data:** 4 synthetic sets (1,170 images each, counts 1–9) plus CountBench.
- **Method:** probes on the final text token at every layer: linear, logistic, circular and MLP. Three probe targets are used: GT count, the model's own output, and an error flag.
- **Probe results:**
  - All probes except the scalar linear one exceed 80% F1, and are near-perfect on Qwen3-VL-8B.
  - GT probes plateau *earlier*; output probes peak near the *final* layers.
  - Weight-space SVCCA between GT and output probes is Linear 0.623, Logistic 0.320, MLP 0.100 and Circular 0.098. This is the "misaligned readout directions" result, and it holds even at small layer gaps (r = −0.45 vs. layer gap).
- **Steering (InternVL2-1B only):** adds a class-difference direction at the best layer. The gain is up to about +19 pp on one synthetic set and +6–8 pp on others. Random directions degrade accuracy.
- **Fix:** a detector-guided **re-prompt**, not steering. The headline **+15.60 is InternVL2-4B on the diff-colour/diff-shape synthetic set**. The mean gain is +5.3. CountBench gains are +3.8 to +8.7.
  - Always-reprompt often matches or beats it, e.g. Qwen3-VL-8B CountBench 89.02 vs. 88.59.
  - Gain correlates with detector F1 (ρ = 0.803).
- **Scope:** no cross-family claim.

**Counting (b): 2605.30170, "Unveiling the Visual Counting Bottleneck" (Pang, Hou, Wang, Sachan; ETH; ICML 2026, v2 Jul 8, 2026)** — [arXiv PDF](https://arxiv.org/pdf/2605.30170)
- **Study 1** is a *toy* VLM (2-layer ViT + 2-layer GPT-2) trained on Go boards: visual counts 0–49, text counts 0–99.
- **Study 2** is Qwen3-VL-32B-Instruct with CoT on a 6×6 Go grid (counts 0–20). It is the only model analysed, because "other open-source models failed … even in N<10."
- **Method:** per-patch binary linear probes summed into a "hidden number". The Vision Gap is |N_H − N_G| and the Language Gap is |N_H − N_P|. There is **no layer-wise LLM patching.**
- **Claim:** the "fractured magnitude hypothesis", i.e. failure at the symbolic-mapping stage.
- **Relevance to the readout story:** the failure is an *extrapolation* failure (unseen counts). It is a different regime from 2607.09544's in-distribution misalignment, so the two papers do not strictly conflict.

**Counting (c): 2604.10039, "Counting to Four is still a Chore for VLMs" (Apr 11, 2026), abstract only** — [arXiv](https://arxiv.org/abs/2604.10039)
- Count evidence is "strongest at the modality projection stage" and degrades in later language layers under text priors.
- Fix: Modality Attention Share, a minimum visual-attention budget. Gains are not stated in the abstract.

**Counting (d): 2606.06890, "Diagnosing Visual Ignorance in VLMs" (Zhou, Zhang, Wang; PKU; v2 Sep 27, 2026)** — [arXiv HTML](https://arxiv.org/html/2606.06890)
- **Models:** Qwen2.5-VL-3B/7B and LLaVA-1.6-Mistral-7B.
- **Mechanistic part:** counting only, using VLMBias counterfactuals.
- **Probes:** MLP probes on the final prompt token. Among prior-following errors, the GT gets > 50% probe probability at some intermediate layer in **49.8% (3B) and 62.5% (7B)** of cases.
- **Coupling:** the coupling slope of image update → output margin is β = 0.123 for layers 12–23 and **0.876 for layers 24–35**. The "late band" starts at normalised depth 0.71, 0.59 and 0.77 for the three models.
- **Patching (Qwen-3B only):** residual interchange at the final prompt token has its largest effect at L27. The L27 attention gain is +0.160 and the MLP gain is −0.003.
  - This **contrasts with 2604.09364**, where last-token patching does nothing.
- **Fix:** none training-free. GRPO layer replacement is used as a diagnostic.
- **Behavioural part:** blur invariance across 12 benchmarks.

**Spatial and colour transport: 2608.22916, "Knowing Isn't Always Saying" (Wang, Xu; PKU/Tsinghua; Aug 24, 2026)** — [arXiv HTML](https://arxiv.org/html/2608.22916v1)
- **Models:** 10 VLMs from 5 families: Qwen2.5-VL-7B/32B, LLaVA-OV-7B, InternVL2.5-8B, InternVL3-8B/14B, Gemma-3-4B/12B/27B and Janus-Pro-7B.
- **Tasks:** 4-way quadrant and colour multiple choice from RefCOCO/+ and GQA.
- **Method:** class-centroid direction patching at the object-word vs. last-prefix token.
- **Where the signal lives:**
  - Spatial directions are readable at about L4–8, but **causal transport occurs only at L12–32**.
  - Under text CoT, 8 of 10 models fall to Δargmax ≤ 0.006 at the object word. InternVL3-8B shows Δargmax = 0 but a logit gain of +0.421, i.e. the signal is suppressed at the argmax rather than erased.
  - Visual-grounding prompts restore transport in all 10 models.
  - One head, InternVL3-8B L27.H21, gives +0.063 when knocked out.
- **Scope:** spatial and colour only. Counting and OCR are not tested, and **there is no cross-family mechanism claim**. It cites GroundingME (2512.17495), Kamath et al. 2023, 2604.09364, 2510.17771 and MIRAGE.

**Text-rich and fine-detail VQA: 2510.17771, "Seeing but Not Believing" (Liu et al.; ICLR 2026)** — [arXiv HTML](https://arxiv.org/html/2510.17771v1), [ICLR](https://iclr.cc/virtual/2026/poster/10010230)
- **Models:** 8 VLMs. LLaVA-NeXT-7B/13B, Qwen2.5-VL-7B/32B, Gemma3-4B/27B and InternVL3.5-8B/14B.
- **Data:** InfoVQA, DocVQA, SROIE and TextVQA from VisualCoT. All are text-centric.
- **Finding:** deep layers (e.g. L16–26 in LLaVA-1.5-7B) attend to the evidence region even when the answer is wrong.
- **Fix:** Vea, which darkens low-attention regions using the top-10%-AUROC layers. EM improves by **+3.6 to +11.1** (mean +5.67 EM, +6.83 F1).
- **Scope:** generality is shown across *models*, not task families.

**Vision-centric tasks in general: 2506.08008, "Hidden in plain sight" (Fu, Bonnen, Guillory, Darrell; Jun 2025)** — [arXiv HTML](https://arxiv.org/html/2506.08008v1)
- **Models:** Prismatic VLMs (DINOv2/IN-1k/CLIP/SigLIP + Vicuna-1.5), validated on InternVL, Phi-3-V and Qwen-VL.
- **Tasks:** depth, semantic correspondence, low-level matching, affordance, art style and 3D awareness.
- **Visual probe vs. VLM (DINOv2):**

  | Task | Probe | VLM | Blind baseline |
  |---|---|---|---|
  | Low-level matching | 0.819 | 0.170 | 0.251 |
  | Correspondence | 0.536 | 0.225 | 0.298 |
  | Depth | 0.887 | 0.653 | 0.597 |

- **Where the loss happens:** features stay decodable through the projector and the LLM. The largest change happens in the **final LLM layer**.
- **Fine-tuning:** LoRA on the LLM beats the projector or ViT, e.g. correspondence total-variation distance 0.078 vs. 0.391.
- **Fix:** no training-free fix.

**Early probing-gap evidence: 2407.06581, "VLMs are blind" (Rahmanzadehgervi et al.; v6 Mar 2025)** — [arXiv](https://arxiv.org/abs/2407.06581)
- BlindTest has 7 tasks. Four state-of-the-art VLMs average **58.07%**; Claude 3.5 Sonnet scores 77.84%.
- The latest version adds linear probing showing that "vision encoders retain enough information … but the language models fail to decode it".

**Projector or connector as the loss point (counter-narrative for dense/spatial detail)**
- **2603.17228, "From Drop-off to Recovery" (Wu, Kim, Akata; Mar 2026), abstract only.** Segmentation probes drop at the adapter, and LLM-layer attention then *recovers* much of the loss. Causal attention limits recovery for early image tokens — [arXiv](https://arxiv.org/abs/2603.17228)
- **2510.21740, FUGU data-visualisation (Tartaglini, Grant, Wurgaft, Potts, Fan; Oct 2025), abstract only.** Using patching and probes on 3 VLMs, correct coordinates are often decodable from the *vision encoder* while the model outputs wrong ones. Errors arise "in the handoff between vision and language", and fine-tuning does not reach ceiling — [arXiv](https://arxiv.org/abs/2510.21740)
- **2509.11986, "Lost in Embeddings" (Li, Tang, Li, Zhang, Vulić, Søgaard; Sep 2025), abstract only.** The connector changes **40–60% of k-NN neighbours**. Patch-level reconstruction loss predicts where VQA fails — [arXiv](https://arxiv.org/abs/2509.11986)

**Fine-grained recognition / nameability**
- **2604.02486, "VLMs Need Words" (Shahgir et al.; COLM 2026).** Probes show sufficient information for both nameable and unnameable entities, but performance is much better for nameable ones. A logit lens recovers labels, and teaching arbitrary names helps — [arXiv](https://arxiv.org/abs/2604.02486)
- **2501.15140 (Finedefics), snippet only.** Idefics2 extracts object information adequately, but object representations are misaligned with category-name representations — [arXiv](https://arxiv.org/html/2501.15140v1)

**OCR routing: 2602.22918, "Where Vision Becomes Text" (Steinberg, Gal; v3 May 2026), abstract only**
- **Models:** Qwen3-VL, Phi-4 and InternVL3.5.
- **Where the OCR signal peaks:** at about 50% depth in DeepStack models and at about 6–25% depth in single-projection models. PC1 explains up to 72.9% of its variance.
- **Side result:** OCR removal improves counting by +6.9 pp in Qwen3-VL-4B, which is a cross-family *interaction*.
- Source: [arXiv](https://arxiv.org/abs/2602.22918)

**Classic interpretability background**
- **Neo et al., ICLR 2025 (2410.07149).** On LLaVA-1.5 the logit lens on visual tokens decodes to vocabulary concepts in late layers. Ablating object tokens plus buffer cuts object identification by 71.79%, and object information is extracted from object tokens in mid-to-late layers — [arXiv](https://arxiv.org/pdf/2410.07149)
- **Zhang, Yadav, Han, Shutova, CVPR 2025 (2411.18620).** Attention knockout shows that early layers move whole-image information into question tokens, middle layers move question-relevant objects, and higher layers carry the result to the last position — [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Cross-modal_Information_Flow_in_Multimodal_Large_Language_Models_CVPR_2025_paper.pdf)
- **Kaduri, Bagon, Dekel (2411.17491).** About the middle 25% of layers carry image→text flow, and early and late layers contribute marginally — [arXiv](https://arxiv.org/abs/2411.17491)
- **Eyes Wide Shut / MMVP (Tong et al., CVPR 2024, 2401.06209).** Attributes failures to CLIP-blind pairs, i.e. *encoder* failure. This is the main "perception IS the problem" foil. I did not re-fetch its numbers in this session — [arXiv](https://arxiv.org/abs/2401.06209)

### Inferences
- The table below summarises the within-family claims and where each paper locates the failure.

  | Family | Paper(s) | Locus claimed | Cross-family? |
  |---|---|---|---|
  | Colour/size prior | 2604.09364 | mid-depth (36–71%), image tokens | No |
  | Counting | 2607.09544; 2606.06890; 2605.30170; 2604.10039 | misaligned readout direction; late band at the final token (L24+); symbolic mapping; post-projector decay | No |
  | Spatial | 2608.22916; 2610.13910 | transport L12–32, gated by prompt; lost at pooling | Colour only |
  | Text-rich / small detail | 2510.17771 | deep-layer attention correct, generation wrong | No |
  | Fine-grained | 2604.02486; 2501.15140 | name/label alignment | No |
  | Ordinal | 2608.20999 | unembedding filters the direction | No |
  | Vision-centric | 2506.08008; 2407.06581 | final LLM layer / LLM decoding | Multi-task, but no shared-mechanism test |

- The loci disagree on two axes: **token position** (image tokens vs. final prompt token) and **depth** (mid vs. late). A paper that resolves this disagreement, e.g. "evidence is *routed* at mid-depth via image tokens and *read out* in a late band at the answer token, and failures split by which stage breaks", has a clear contribution margin.
- The counting conflict is partly a regime difference: in-distribution counts (2607.09544) vs. extrapolation (2605.30170). A careful paper should say so instead of claiming to "resolve" it.

### Gaps
- I did not read the full texts of 2510.21740 (exact models and layer numbers), 2603.17228 (models) or 2604.10039 (models, MAS gains).
- I did not re-fetch Eyes Wide Shut or MMVP numbers.
- I could not verify GroundingME (2512.17495) beyond its citation in 2608.22916.

---

## Q2. Aug–Oct 2026 scoop scan: does any paper already make a cross-family "shared readout bottleneck" claim?

### Takeaway
**No paper I found claims one shared late-layer mechanism across attribute, counting, spatial, fine-grained and OCR failures.** The closest threat is **2610.00024 (Sep 1, 2026)**. It calls the "encoded-but-disconnected" *phenomenology* universal, then reports a residual/head **patching null** and states that the mechanism **differs by architecture**. That is close to the opposite of the hypothesis. Several Aug–Sep 2026 papers generalise the gap to new domains (ordinal labels, memes, omnimodal premises, blur insensitivity), so the *phenomenon* is no longer novel. Only a *shared-mechanism* claim plus a fix remains open.

### Cited Findings

**Highest scoop risk: 2610.00024, "Encoded but Disconnected: Decomposing VLM Failures under a Patching Null" (Genpei Zhang, UW–Madison; v1 Sep 1, 2026; 13 pp)** — [arXiv HTML](https://arxiv.org/html/2610.00024), [abs](https://arxiv.org/abs/2610.00024)
- **Models:** LLaVA-1.5-7B, Qwen2.5-VL-7B and InternVL3-8B.
- **Data:** POPE-adversarial (first 200 errors per model), MMVP (LLaVA only) and HallusionBench. These are mainly object-presence tasks; counting, spatial and OCR are not separate families.
- **Taxonomy:** Prior-Override (PO), Encoded-but-Disconnected (EBD; a raw logit lens shows a mid-layer top-1 hit on the GT at the last text position) and Perception Failure (PF).
- **Share of POPE errors in each mode (PF / EBD / PO):**

  | Model | PF | EBD | PO |
  |---|---|---|---|
  | LLaVA | 0% | 74% | 26% |
  | Qwen | 12% | 9% | 79% |
  | InternVL3 | 50% | 8% | 42% |

- **Patching null:** cross-instance residual patching gives 0/50 flips on LLaVA, 0/44 on Qwen and ≤ 1/50 on InternVL3. Per-head patching on Qwen gives 0 flips over 12,600 runs. The exception is InternVL3 L20.H2 at 13/50, which the author calls "causal but not vocab".
- **Quotes on universality:** "Architecture differs in *where* the encoded answer crystallizes, not whether it does", and the paper reports "a universal negative finding for mid-layer interpretability".
- **Fixes:** under oracle labels, a prompt-debias suffix flips 19.6–21.3% of PO errors in Qwen and InternVL3. Mid-layer "Re-Promote" (DoLa-like blending at L12) flips 85.9% of EBD errors in LLaVA but also 6 of 7 PF errors. The paper makes **no net-accuracy claim**.
- **Concurrent taxonomies it cites:** Xiong et al. 2026 (a 3-class hallucination decomposition) and Hong & Quan 2026 (a tri-layer framework). I could not locate either.
- **Implication:** a "shared bottleneck" paper must explain or overturn this null, e.g. with position-correct patching (image-token or full-sequence, as in 2604.09364) instead of last-token residual swaps.

**Same author: 2609.00868, "The Visual Insensitivity Gap" (Sep 1, 2026), abstract only** — [arXiv](https://arxiv.org/abs/2609.00868)
- 6 VLMs and 3 benchmarks. Blurring the relevant region leaves 40–97% of samples' outputs unchanged.
- A vision-encoder probe detects the perturbation (0.72–0.79 accuracy), but the argmax changes on only 2–11% of those samples. That is an "encoder–LLM gap > 0.65 on every model".
- Insensitivity is sample-level and shared across models (Spearman ρ = 0.40).

**Domain extensions of the same phenomenon, Aug–Sep 2026**
- **2608.20999, "Latent Ordinal Evidence, Misaligned Outputs" (Li et al.; EMNLP 2026; Aug 21).** 4 MLLMs × 4 ordinal tasks (age, image quality, disease grade).
  - Hidden-state Spearman correlation is up to 0.938. The digit-token unembedding row space keeps **< 1.15%** of the ordinal direction, and probes beat native output by **16–77 pts**.
  - Fix: OLA, frozen-backbone lenses on mid-to-deep layers that correct digit logits. It beats LoRA OrderChain in most settings.
  - **This is the strongest prior for an "unembedding filters the evidence" mechanism**, but it covers one family only.
  - Source: [arXiv](https://arxiv.org/abs/2608.20999)
- **2609.18860, "Decodable but Misrouted" (Koushik, Kanojia, Treharne; v1 Sep 16, v2 Sep 29, 2026, so contemporaneous).** Gemma-3 and Qwen3.5 on 6 harmful-meme tasks.
  - Sparse readout gives 0.740 vs. 0.432 native macro-F1 (Qwen) and 0.714 vs. 0.532 (Gemma).
  - Native threshold calibration recovers 69.8% of the gap.
  - It concludes that "routing [is] a common bottleneck", but only within harmful-content tasks.
  - Source: [arXiv](https://arxiv.org/abs/2609.18860)
- **2605.13737, "Senses Wide Shut" (v2 Oct 1, 2026).** 8 omnimodal LLMs plus Gemini 3.1 Pro on premise-conflict detection (IMAVB, 500 clips). Hidden states encode the mismatch, but outputs rarely reject the false premise ("Representation–Action Gap"). Probe-guided logit adjustment (PGLA) adds about 10–15 balanced-accuracy points per a search snippet — [arXiv](https://arxiv.org/abs/2605.13737)
- **2610.13910, "When VLMs Destroy What They Know" (Garg, Shani; Oct 9, 2026, contemporaneous), abstract only.** Spatial relations are identifiable at predicate tokens but lost at pooled EOS or mean representations — [arXiv](https://arxiv.org/abs/2610.13910)

**Multi-mode failure taxonomies (not one shared mechanism)**
- **2607.16094, "How Do VLMs Fail? Vision-Operation Misalignment" (ACM MM 2026), abstract only.** Four failure modes: grounding, reasoning, attribute extraction and language-prior dominance. **Different loci per operation:** object selection is mainly FFN, multi-step relations use late-layer attention, and attribute extraction uses the FFN at the answer position — [arXiv](https://arxiv.org/abs/2607.16094)
- **2607.04683, "Does It Fail to See or Fail to Know?" (Vo et al.; Jul 6, 2026), snippet only.** Splits errors into perception, entity recognition and knowledge retrieval. These are predictable pre-generation from visual tokens vs. prompt-conditioned states — [arXiv](https://arxiv.org/abs/2607.04683)
- **2605.08200, "Where Reliability Lives in VLMs" (ICLR 2026 workshop).**
  - Hidden-state probe AUROC > 0.95 on POPE for 2 of 3 families. Attention–correctness correlation is about 0.
  - Reliability sits in a **fragile late bottleneck in late-fusion LLaVA** but is distributed in PaliGemma and Qwen2-VL, which tolerate destroying about 50% of peak-layer dimensions.
  - This is architecture dependence again.
  - Source: [arXiv](https://arxiv.org/abs/2605.08200)
- **2603.18523, "Counting Circuits", snippet only.** Visual Activation Patching and HeadLens find a counting circuit whose transfer gives only **+1.54%** average on OOD visual tasks. This is weak evidence for a shared circuit — [arXiv](https://arxiv.org/html/2603.18523v1)

**Search coverage.** Queries covered "shared bottleneck", "arbitration layer", "readout failure/gap", "encoded but not used", "probe vs output gap", "linear probe outperforms answer", "patching counting spatial", "late-layer suppression" and "probe-guided decoding". None surfaced a cross-family shared-mechanism paper.

### Inferences
- **Scoop status:** the *phenomenon* ("encoded but not used") is crowded, with 15+ papers across 2025–26. A **single mechanism shared across families and models** is unclaimed.
- **Two papers argue against such a mechanism:** 2610.00024 (architecture-specific, patching null) and 2607.16094 (different loci per operation). The CVPR paper must engage both head-on. Turning "they found a null" into "the null comes from patching the wrong position or basis" would be a strong hook.
- **Biggest contemporaneous risk:** an ICLR 2027 submission (deadline about late Sep/early Oct 2026) making exactly this claim. OpenReview submissions were not searchable in this session, so re-check OpenReview ICLR 2027 when they become public (mid-Nov) and cite them as concurrent.

### Gaps
- OpenReview ICLR 2027 submissions were not visible to me, so this risk is unassessed.
- Semantic Scholar citation lists for 2604.09364 and 2506.08008 were not retrieved; I relied on web search.
- "Xiong et al. 2026" and "Hong & Quan 2026" (cited in 2610.00024) could not be found.
- I found no Sep 15–Oct 13 arXiv paper with a cross-family claim, but arXiv search via a web engine is incomplete.

---

## Q3. Training-free fixes that exploit encoded-but-unused information: which exist and which are saturated

### Takeaway
The fix space is large but fragmented. Family-specific fixes give small-to-moderate gains (+1.4 to +3.8 steering; +5.3 average re-prompting; +5.7 EM for Vea), often with degradation. **Hallucination-oriented layer-contrast and steering decoders are saturated, and their evaluation is compromised** (Rethinking-CD). A **single family-agnostic readout repair, validated across ≥ 4 families with matched decoding and strong trivial baselines**, does not exist.

### Cited Findings

**1. Probe/detector-based answer correction**
- 2607.09544 gates a re-prompt on a probe-based error detector: +5.3 average, up to +15.6 — [arXiv](https://arxiv.org/pdf/2607.09544)
- 2608.20999 OLA trains lenses that correct only the digit logits — [arXiv](https://arxiv.org/abs/2608.20999)
- 2605.13737 PGLA is a probe-guided logit adjustment — [arXiv](https://arxiv.org/abs/2605.13737)
- 2609.18860 routes SAE features directly, adding +0.094 macro-F1 beyond calibration — [arXiv](https://arxiv.org/abs/2609.18860)
- ProbeAct (2606.09740) is a probe-guided recovery loop for VLAs: OpenVLA-OFT on LIBERO-plus goes from 69.6% to 74.1% — [arXiv](https://arxiv.org/html/2606.09740v1)
- **Status:** emerging and not saturated, but each method is single-family. All need labelled probe training, so they are "training-free" only with respect to backbone weights.

**2. Activation steering**
- VTI (Liu, Ye, Xing, Zou; ICLR 2025, 2410.15778) uses latent directions precomputed from paired captions — [arXiv](https://arxiv.org/pdf/2410.15778)
- 2604.09364 linear/SAE steering gives +1.4–3.8% and sometimes hurts — [arXiv](https://arxiv.org/html/2604.09364)
- 2607.09544 probe-direction steering: up to about +19 pp on one model, destructive at large α — [arXiv](https://arxiv.org/pdf/2607.09544)
- 2608.22916 steering *toward wrong* quadrants drops accuracy by 8–21 pp, showing causal relevance but no fix — [arXiv](https://arxiv.org/html/2608.22916v1)
- **Status:** crowded for hallucination; weak for perception families.

**3. Early/intermediate-layer readout and DoLa-style layer contrast**
- DeCo (Wang et al. 2025) blends earlier-layer predictions, per ReWEIGH's description — [ReWEIGH 2608.19075](https://arxiv.org/pdf/2608.19075)
- VISTA (2502.03628) uses early-layer logits, with about 40% less hallucination across 4 architectures per a snippet — [search via ReWEIGH/SAVER](https://arxiv.org/pdf/2608.19075)
- SAVER (AAAI 2026) uses earlier layers with concentrated visual activation and is motivated by "LVLMs … progressively suppress visual information in later layers" — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40873/44834)
- VaLiD (2411.15839) is visual-layer-fusion contrastive decoding — [arXiv](https://arxiv.org/html/2411.15839v1)
- FADE (2606.29431) targets language-prior dominance — [arXiv](https://arxiv.org/pdf/2606.29431)
- ILVAD (2605.20965) uses inter-layer attention discrepancy — [arXiv](https://arxiv.org/html/2605.20965)
- 2610.00024 "Re-Promote" (mid-layer logit blending) is not error-mode specific — [arXiv](https://arxiv.org/html/2610.00024)
- **Status:** saturated for object hallucination (POPE/CHAIR).

**4. Visual re-injection, re-attention and evidence highlighting**
- Vea (2510.17771): +3.6 to +11.1 EM on text-rich VQA — [arXiv](https://arxiv.org/html/2510.17771v1)
- MAS (2604.10039) sets a minimum visual-attention budget for counting — [arXiv](https://arxiv.org/abs/2604.10039)
- An intermediate-layer attention re-alignment method (Sci. Rep. 2026) observes that the "visual signal is often diluted by linguistic priors in deeper layers" — [Nature Sci Rep](https://www.nature.com/articles/s41598-026-44935-1)
- 2607.27830 does intermediate-layer evidence routing for high-resolution VQA — [arXiv](https://arxiv.org/pdf/2607.27830)
- AdaptVis (2503.01773, ICML 2025) uses attention temperature for spatial reasoning — [arXiv](https://arxiv.org/html/2503.01773v3)
- **Status:** crowded, especially attention reweighting.

**5. Prompt-level gating**
- Visual-grounding prompts reopen spatial transport (2608.22916).
- Prompt-debias flips 19.6–21.3% of prior-override errors (2610.00024).

**6. Layer-skipping**
- "Skip-It?" (2509.25584) gives theoretical conditions for skipping layers in VLMs — [arXiv](https://arxiv.org/pdf/2509.25584)
- I found no paper using layer skipping to *recover* suppressed perception.

**Evaluation-hygiene threat (must cite)**
- Contrastive-decoding gains largely come from greedy decoding plus output bias. A no-image constant bias (b* = +0.84) reaches 86.86% on POPE vs. 85.89% for the best CD method — [Rethinking-CD 2607.25196](https://github.com/Aurnawr/Rethinking-CD)
- 2610.00024 shows a prediction-direction bit alone reaches 93–94% in error-mode classification on Qwen and InternVL3. Trivial baselines are strong — [arXiv](https://arxiv.org/html/2610.00024)
- 2609.18860: native threshold calibration recovers 69.8% of the probe gap — [arXiv](https://arxiv.org/abs/2609.18860)

### Inferences
- **Saturated:** contrastive and layer-contrast decoding for object hallucination; attention reweighting; generic steering for hallucination.
- **Open:** a *family-agnostic* readout repair, i.e. one lens, subspace or projection fit once that transfers across colour, counting, spatial, fine-grained and OCR. It should be reported against (i) a calibration/bias-only baseline (since 2609.18860 found calibration explains about 70%), (ii) always-reprompt, (iii) DeCo/VISTA-style early readout and (iv) Vea.
- Any fix must report **net accuracy with matched greedy decoding**, not flip rates. 2610.00024 explicitly avoided net-accuracy claims, which leaves room.

### Gaps
- I did not verify DeCo's arXiv ID or exact numbers, or MemVR ("Look Twice Before You Answer", ICML 2025). Both come from memory and secondary citations, so check them before citing.
- I found no head-to-head benchmark of all fix families on non-hallucination perception tasks. That absence is itself an opportunity.

---

## Q4. What would be genuinely new: candidate core claims, novelty margin and must-cite lists

### Takeaway
Do **not** claim the phenomenon ("VLMs see but don't say"); it is taken. The defensible novelty is **cross-family mechanistic identity plus a single transferable fix**, framed to engage the architecture-specific null (2610.00024) and the locus conflicts (2604.09364 vs. 2606.06890). Note one caution: the evidence places arbitration at **mid-depth image tokens (36–71%)** in one paper and at a **late band at the answer token (≥ 0.59 depth)** in another. "Late-layer" should therefore be a tested hypothesis, not the title.

### Cited Findings (basis for the claims)
- Per-family "encoded-not-used" evidence exists for:
  - colour/size — [2604.09364](https://arxiv.org/html/2604.09364)
  - counting — [2607.09544](https://arxiv.org/pdf/2607.09544); [2606.06890](https://arxiv.org/html/2606.06890)
  - spatial — [2608.22916](https://arxiv.org/html/2608.22916v1)
  - text-rich VQA — [2510.17771](https://arxiv.org/html/2510.17771v1)
  - fine-grained recognition — [2604.02486](https://arxiv.org/abs/2604.02486)
  - ordinal labels — [2608.20999](https://arxiv.org/abs/2608.20999)
  - vision-centric tasks — [2506.08008](https://arxiv.org/html/2506.08008v1)
- Evidence against a shared mechanism: [2610.00024](https://arxiv.org/html/2610.00024); [2607.16094](https://arxiv.org/abs/2607.16094); [2605.08200](https://arxiv.org/abs/2605.08200)
- An unembedding-level filtering mechanism has been shown for one family — [2608.20999](https://arxiv.org/abs/2608.20999)

### Inferences: candidate claims

**Claim A (strongest margin): "One readout subspace across failure families."**
- **Claim:** across ≥ 4 families (colour vs. prior, counting, spatial relation, fine-grained identity, with OCR/small-object as a fifth) and ≥ 4 model families, wrong answers share **the same low-rank evidence-to-vocabulary bottleneck**. The test is that a subspace estimated on one family predicts and repairs failures on the others.
- **Measurements:**
  - (i) a per-layer probe–output gap
  - (ii) GT-vs-output probe SVCCA, generalising 2607.09544
  - (iii) the fraction of the evidence direction kept by the answer-token unembedding rows, generalising 2608.20999's < 1.15% from ordinal labels to all families
  - (iv) **cross-family transfer** of the patch or steer direction, which no paper reports
- **Engaging the null:** show that the 2610.00024 patching null disappears under full-sequence or image-token patching (2604.09364 style) or under subspace patching. The null would then be a methodological artefact.
- **Margin:** high if transfer holds. If transfer fails, the result is still publishable as a "no shared bottleneck" analysis paper with a rigorous null.
- **Must cite:** 2604.09364, 2607.09544, 2605.30170, 2606.06890, 2610.00024, 2608.22916, 2608.20999, 2609.18860, 2605.13737, 2506.08008, 2407.06581, 2510.17771, 2604.02486, 2607.16094, plus the classics 2410.07149, 2411.18620 and 2411.17491.

**Claim B: "Two-stage account, routing vs. readout, with a per-sample diagnostic that predicts which fix works."**
- **Claim:** every family's errors decompose into (1) *routing* failures, where the evidence never reaches the answer position (mid-depth image→text transport; 2608.22916, 2411.18620), and (2) *readout* failures, where the evidence reaches the answer token but is filtered by the late band or unembedding (2606.06890, 2608.20999).
- **Prediction:** the family mix explains *which* training-free fix helps. Re-injection or attention fixes (Vea/MAS) work on routing errors; readout or lens fixes (OLA/PGLA) work on readout errors.
- **What this reconciles:** 2604.09364 (image tokens matter), 2606.06890 (final token at L24+ matters) and the counting conflict (2607.09544 vs. 2605.30170, which differ in regime).
- **Margin:** medium-high. It overlaps the 3-way taxonomies (2610.00024 PF/EBD/PO; 2607.04683; 2607.16094), so the novelty must come from *cross-family* coverage and *fix-selection prediction*, not from the taxonomy itself.
- **Must cite:** as for A, plus 2607.04683, 2603.17228, 2510.21740, 2509.11986, 2604.10039 and 2602.22918.

**Claim C (method-forward): "A family-agnostic training-free readout repair."**
- **Method:** one frozen-backbone readout correction, e.g. an evidence-subspace re-projection into answer-token logits or a mid-layer lens fused only on candidate-answer logits, estimated from a small unlabelled or self-consistency set, applied unchanged across families and models.
- **Evaluation protocol:** matched greedy decoding plus a no-image bias baseline (Rethinking-CD), native calibration (2609.18860), always-reprompt (2607.09544), DeCo/VISTA, Vea and steering (2604.09364).
- **Margin:** medium. Each component has a single-family precedent (OLA, PGLA, detector re-prompt, Vea), so novelty depends on demonstrating **transfer across families** and beating the calibration baseline. Gains must clearly exceed the +3.8 (steering) and about +5 (re-prompt/Vea) reference points.
- **Must cite:** 2608.20999, 2605.13737, 2607.09544, 2510.17771, 2604.09364, 2610.00024, 2607.25196, 2410.15778 (VTI), DeCo/VISTA/SAVER/VaLiD, 2609.18860 and 2606.09740.

**Recommended combination:** use **A as the main claim, with C as the payoff**, and keep B's routing/readout split as an analysis section that explains where the fix fails.
- Run models from ≥ 4 families (Qwen2.5-VL or Qwen3-VL, InternVL3, LLaVA-OV, Gemma-3) at 2 sizes.
- Add a 72B/32B spot check, since reviewers will ask after 2605.30170 (Qwen3-VL-32B) and 2604.09364 (Qwen2-VL-72B).

**Risk notes**
- **Benchmark validity:** reviewers will expect blind and no-image controls (Cambrian-S style) and counterfactual images.
- **Concurrent work:** ICLR 2027 submissions on OpenReview become visible around mid-November 2026, right at the CVPR deadline. Check them then and cite them as concurrent.

### Gaps
- Whether a shared subspace actually exists is an empirical unknown. No source tests cross-family transfer, so neither success nor failure is sourced.
- I did not establish CVPR 2027 reviewer norms for "concurrent work" beyond the user's Sep 15 cutoff.
