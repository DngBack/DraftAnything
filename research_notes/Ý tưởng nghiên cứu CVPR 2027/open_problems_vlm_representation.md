# Open Problems in MLLMs / VLMs, Visual Representation Learning, Vision Interpretability and Visual Reasoning, Scoped for One H200 (CVPR 2027)

Notes compiled 2026-10-13. Source details come from arXiv abstract pages and search snippets; only one full abstract was fetched and read (2604.09364). Read the full text of any paper before citing numbers in a submission. arXiv IDs 26xx.xxxxx are 2026 preprints. Compute estimates are my own inferences (see "Inferences" in each section), not sourced figures.

## Q0 (Objective): 10–15 concrete open research questions feasible on 1 H200

### Takeaway
Across all the evidence, the richest open seam for a 1-GPU CVPR paper is the **"encoded-but-not-used" gap**: across colour conflicts, counting, coordinates and spatial cues, 2025–2026 probing papers show that the visual information is present inside the MLLM but the output does not reflect it. The second-richest is **rigorous re-evaluation of saturated method families** (contrastive decoding, attention-based pruning, Platonic convergence, registers). Both rest on analysis, probing and LoRA, not pretraining.

### Cited Findings (evidence base for the list below)
- In a 10-VLM study, models that answer wrongly still encode the visual attribute about as strongly as models that answer correctly. Attributes are linearly decodable in early layers (AUC > 0.86). The final-layer logit gap predicts grounding outcomes (ρ = 0.847), and encoding strength does not. Full-sequence patching at "arbitration" layers flips 60–84% of outputs, while last-token patching does nothing. SAE- and linear-steering gives at most +3.8% and sometimes hurts — [Arbitration Failure, Not Perceptual Blindness (arXiv 2604.09364, Apr 2026)](https://arxiv.org/abs/2604.09364)
- VLMs do substantially worse than their own vision encoders on vision-centric tasks. Visual features stay intact through the projector and LLM, so the loss lies in how the LLM *uses* them (language priors) — [Hidden in plain sight (arXiv 2506.08008)](https://arxiv.org/html/2506.08008v1)
- Mechanistic probing of segmentation in MLLMs finds a representation drop-off at the adapter: projection trades fine-grained spatial fidelity for cross-modal alignment — [From Drop-off to Recovery (arXiv 2603.17228, Mar 2026)](https://arxiv.org/html/2603.17228)
- Linear probes decode object coordinates from the encoder, but LLM-layer probes never recover that accuracy — [Diagnosing Bottlenecks in Data Visualization Understanding (arXiv 2510.21740)](https://arxiv.org/pdf/2510.21740)
- Counting: one paper says the count is encoded, but readout directions are misaligned. Detector-guided self-correction gains up to +15.6 points with no parameter updates — ["The Count Is There, but Misaligned" (summary page)](https://djaym7-arxivmedia.hf.space/post/306931). A competing account places the bottleneck at the symbolic mapping from latent magnitude to number token ("fractured number space") — [Unveiling the Visual Counting Bottleneck (arXiv 2605.30170)](https://arxiv.org/pdf/2605.30170). An earlier account says counts build up across layers and use separator-token shortcuts — [CountScope (arXiv 2511.17699)](https://web3.arxiv.org/pdf/2511.17699). These accounts **conflict**.
- Plain accuracy-reward RLVR: a McNemar test shows no statistically significant perception gain over base models. The authors add a visual-perception reward to fix this — [Perception-R1 (arXiv 2506.07218, v3 Mar 2026)](https://arxiv.org/abs/2506.07218)
- Contrastive decoding (VCD/ICD/SID) tested on LLaVA-1.5-7B, Qwen2.5-VL-7B and InternVL3-8B: the shift is not selective to hallucinated objects, and greedy decoding explains the CHAIR gains. A single constant bias with no image (b* = +0.84) reaches 86.86% on POPE, beating the best CD method (85.89%) — [Rethinking-CD (arXiv 2607.25196; NeurIPS 2026 VLM4RWD workshop per README)](https://github.com/Aurnawr/Rethinking-CD)
- Gains from attention-based pruning are confounded by position bias. Random selection or pooling often beats complex methods — [Token Pruning in MLLMs: Are We Solving the Right Problem? (ACL Findings 2025)](https://arxiv.org/html/2502.11501v1)
- Platonic convergence: with permutation calibration, CKA/SVCCA/Procrustes scaling trends largely disappear. Only neighbourhood metrics (mKNN, CKNNA) keep the trends — [Revisiting the PRH: An Aristotelian View (arXiv 2602.14486)](https://arxiv.org/pdf/2602.14486)

### Inferences — the candidate list (each item: gap / why it matters / simple angle / H200 compute)

**1. Where does "arbitration" happen, and is it the same circuit across failure types?**
- *Gap:* 2604.09364 shows the encoding-grounding dissociation for attribute–prior conflicts. Similar "encoded but not output" results exist separately for counting (2605.30170, "Count Is There"), coordinates (2510.21740), and vision-centric tasks in general (2506.08008). Nobody has shown whether one shared mechanism (layers, heads, image-token positions) explains all of them. The counting papers even disagree about where the failure sits.
- *Why it matters:* A unified account would replace dozens of task-specific hallucination, counting and spatial fixes.
- *Angle:* Run the same pipeline (linear probe per layer, logit-lens crossover, full-sequence activation patching on image tokens) on Qwen2.5-VL-7B, Qwen3-VL-8B, InternVL3-8B and LLaVA-OneVision-7B across four failure families (attribute conflict, counting, relative position, fine-grained identity). Test whether the "arbitration layers" and the patch-transferable directions coincide.
- *Compute:* Inference, caching and patching on 7–8B models in bf16 (~16–18 GB each). About 50–150 H200-hours total. Feasible in 2–3 weeks.

**2. Does RL (GRPO/R1-style) change the vision pathway at all, or only language-side arbitration?**
- *Gap:* Perception-R1 finds no significant perception gain from accuracy-only RLVR. The thinking-with-images literature argues gains come from perception learned during training, not from the visual actions themselves ("Zooming without Zooming", cited in [Perceive-to-Reason (arXiv 2607.01191)](https://arxiv.org/html/2607.01191v1)). No paper does a layer-wise representational diff between base and RL checkpoints separating encoder, projector and LLM-arbitration changes; the [ACL 2026 image-grounded CoT survey](https://aclanthology.org/2026.acl-long.2087.pdf) frames its claims as synthesis, not controlled ablation.
- *Angle:* Use public base/RL pairs (e.g. Qwen2.5-VL-7B vs. open R1-style RL checkpoints). Measure CKA/mKNN drift per layer, probe accuracy for visual attributes, and run patching that swaps RL-model layers into the base model. Prediction to test: RL moves only late-layer arbitration directions. Optional: your own small GRPO run with LoRA on 7B.
- *Compute:* Analysis takes about 50 GPU-hours. A small GRPO + LoRA run on 7B is about 1–3 days on one H200 (rollout-bound; vLLM in colocate mode).

**3. Is attention-based visual-token importance causal or merely positional/sink-driven?**
- *Gap:* Pruning methods keep trading position bias against diversity ([D2Pruner (2512.19443)](https://arxiv.org/pdf/2512.19443), [PosPrune (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/38020), [RoRA (2608.07088)](https://arxiv.org/pdf/2608.07088), [layer-aware PE (2609.23715)](https://arxiv.org/pdf/2609.23715), [SinkPruner (2609.01004)](https://arxiv.org/pdf/2609.01004)). Random or pooling baselines remain competitive (2502.11501). No work measures the *true* causal importance of tokens (ablation and patching ground truth) and asks how far each heuristic is from it.
- *Angle:* Build an "oracle importance" dataset by leave-k-out token ablation on 5–10k samples, and report each pruner's regret against the oracle by task type (fine-grained, spatial, general). This is an analysis paper, not another pruner.
- *Compute:* The oracle is expensive but can be batched with KV caching. Roughly 200–400 H200-hours on 7B with sampled ablations. Feasible.

**4. Visual attention sinks and massive activations: cause or symptom of hallucination?**
- *Gap:* Sinks reuse the BOS massive-activation dimensions ([See What You Are Told (2503.03321)](https://arxiv.org/html/2503.03321)). One claim says RoPE's 1-D decay on linearised image tokens produces them ([ScienceDirect study](https://sciencedirect.com/science/article/abs/pii/S0306457325003723)). Another finds sinks carry useful global information ([To Sink or Not to Sink (2510.08510)](https://arxiv.org/pdf/2510.08510)). The causal role in hallucination is still unsettled: the evidence is correlation plus mitigation.
- *Angle:* Make controlled interventions (zeroing sink dimensions, re-ordering image tokens, swapping 1-D RoPE for M-RoPE models such as Qwen2.5-VL) and measure a hallucination metric that does not suffer the greedy-decoding confound (see #7).
- *Compute:* Under 100 GPU-hours.

**5. High-norm artifacts in the post-register era (DINOv3, SigLIP2, WebSSL, MLLM encoders)**
- *Gap:* DINOv3 keeps 4 registers. An attention-bias/value-gating alternative scores about the same (86.6 vs 86.5 on one benchmark) — [DINOv3 (2508.10104)](https://arxiv.org/pdf/2508.10104). Test-time register neurons remove outliers in CLIP and DINOv2 without retraining — [ViTs Don't Need Trained Registers (2506.08010)](https://arxiv.org/html/2506.08010v2). A 2026 cross-architecture reassessment questions whether all ViTs need registers — [Do All ViTs Need Registers? (2603.25803)](https://arxiv.org/pdf/2603.25803). I found nothing specific to SigLIP2, or to what artifacts do downstream inside an MLLM (e.g. whether the Qwen-VL or InternViT encoder's high-norm tokens become the LLM's visual sinks).
- *Angle:* Trace artifact tokens from encoder to projector to LLM-sink status, then apply test-time register removal and measure MLLM effects on dense, grounding and counting tasks.
- *Compute:* Under 50 GPU-hours.

**6. What exactly converges (and what doesn't) between vision SSL, CLIP-type and LLM representations?**
- *Gap:* Calibration erases most PRH scaling trends (2602.14486). The vision–language evidence is weaker than claimed — [Back into Plato's Cave (2604.18572)](https://www.alphaxiv.org/abs/2604.18572). "Structure over geometry" — [2609.27252](https://arxiv.org/abs/2609.27252). Alignment rests on linear structure — [2605.28870](https://arxiv.org/pdf/2605.28870). Open: does *dense/patch-level* structure converge (DINOv3 vs SigLIP2 vs WebSSL vs MAE vs V-JEPA 2), or only global semantics? Does language-free SSL ([Web-SSL (2504.01017)](https://arxiv.org/pdf/2504.01017)) converge toward LLMs as it scales?
- *Angle:* Use calibrated neighbourhood metrics at patch level and correspondence level across open checkpoint families and sizes, and relate them to downstream MLLM-connector performance.
- *Compute:* Feature extraction only, about 20–80 GPU-hours. Very cheap, and strong for an analysis paper.

**7. Fixing hallucination evaluation itself: a confound-free protocol**
- *Gap:* CD gains are a "mirage" from one-directional adjustment and sampling-to-greedy conversion — [Mirage of Performance Gains (2504.10020)](https://www.alphaxiv.org/overview/2504.10020v3); [Rethinking-CD (2607.25196)](https://arxiv.org/html/2607.25196). Yet new CD variants keep appearing in 2026 (SDCD, ACG, MACD, OAVCD — [Awesome-LVLM-Hallucination](https://github.com/NishilBalar/Awesome-LVLM-Hallucination)).
- *Angle:* Re-evaluate the 2025–26 training-free hallucination methods (CD, attention reallocation, steering) under matched decoding, a no-image constant-bias baseline and per-object selectivity. Release a protocol. The risk: workshop-level unless paired with a mechanistic finding (combine with #1/#4).
- *Compute:* Inference only, about 100 GPU-hours.

**8. Counting: reconcile "misaligned readout" vs "fractured number space"**
- *Gap:* Two 2026 accounts conflict (see Cited Findings). Holistic benchmarks still show failures — [HoloCount (2607.06420)](https://arxiv.org/pdf/2607.06420).
- *Angle:* Use probes and patching across count ranges (subitizing ≤4 vs larger), check whether a visual-magnitude direction maps to the numeral-token direction, and apply a rank-1 readout repair (training-free or a tiny LoRA).
- *Compute:* 50–150 GPU-hours.

**9. SAE features as an *intervention* space for vision and MLLMs: when do they work?**
- *Gap:* Whether SAE features can be actionable rather than post-hoc is open, and reconstruction saliency and task saliency are mismatched — [Visual Sparse Steering (2506.01247)](https://arxiv.org/pdf/2506.01247). Feature reliability problems: a DINOv2 "crash" feature does not reliably fire — [SAEs for Scientifically Rigorous Interpretation (2502.06755)](https://arxiv.org/html/2502.06755v1). SAE-guided steering in VLMs gives at most +3.8% and sometimes degrades (2604.09364). Cross-modal consistency is only now being studied — [Structured SAEs across modalities (2607.08605)](https://arxiv.org/pdf/2607.08605); [Information scope in CLIP SAEs (2604.05724)](https://arxiv.org/pdf/2604.05724); [Minkowski geometry in DINO (2510.08638)](https://arxiv.org/pdf/2510.08638).
- *Angle:* Train matched SAEs on vision-encoder output vs. LLM-layer image tokens of the same MLLM. Test whether steering succeeds only at layers before the arbitration crossover. Benchmark SAE steering against linear-probe steering and difference-of-means baselines (a likely negative result, but an informative one).
- *Compute:* An SAE on ~100–500M patch tokens of a 1–4k-dim stream is about 1–3 days of H200 time per SAE; 3–5 SAEs fit in about 2 weeks.

**10. The adapter/projector as the loss point for dense and spatial information**
- *Gap:* Drop-off at the adapter (2603.17228). Coordinates are never recovered in the LLM (2510.21740). Visual-cue forgetting — [Vision Remember (2506.03928)](https://arxiv.org/html/2506.03928).
- *Angle:* Use an information-theoretic or probe audit of projector choices (MLP vs pixel-shuffle vs Q-former) on matched data with a frozen LLM. Alternatively, cheap LoRA re-injection of encoder features and a test of whether spatial probe accuracy recovers in the LLM.
- *Compute:* Projector-only retraining for a 7B on ~1M samples takes about 1–2 days. Feasible.

**11. Spatial reasoning shortcuts in video and spatial benchmarks**
- *Gap:* Cambrian-S reaches 67.5% on VSI-Bench, but VSR degrades from 38.3% to 6.0% over 10→60-minute video — [Cambrian-S (2511.04670)](https://arxiv.org/pdf/2511.04670). A simple baseline collapses Cambrian-S on VSC-Repeat (42% → 0%), implying benchmark shortcuts — [Solving Spatial Supersensing Without Spatial Supersensing (2511.16655)](https://arxiv.org/html/2511.16655v1). VSI-SUPER is mostly household scenes — [Towards Spatial Supersensing in the Wild (2607.13681)](https://arxiv.org/pdf/2607.13681).
- *Angle:* A shortcut audit of spatial benchmarks (VSI-Bench, CV-Bench-3D, BLINK spatial) with blind, frame-shuffled and repeat controls, plus probing of whether 3-D quantities (depth, distance) are linearly present in the LLM hidden states.
- *Compute:* Inference on video MLLMs (7B, long context) takes about 100–300 GPU-hours.

**12. Language priors vs. visual fidelity: a causal, not benchmark, measure**
- *Gap:* A wave of 2026 benchmarks (V-FAT [2601.04897](https://arxiv.org/pdf/2601.04897), Seeing or Knowing? [2607.26326](https://arxiv.org/abs/2607.26326), BabyVision [2601.06521](https://arxiv.org/html/2601.06521v1), VIA-Bench [2602.01816](https://arxiv.org/html/2602.01816)) separates "perception failure" from "utilization failure". The older vision-centric sets (MMVP, MMStar, CV-Bench) are now "largely saturated" — [An Exam for Active Observers (2607.16165)](https://arxiv.org/pdf/2607.16165). LLM-pretraining origins of visual priors — [Demystifying LLM Visual Priors (2509.26625)](https://arxiv.org/pdf/2509.26625).
- *Angle:* A per-sample "visual reliance" score from counterfactual image patching (rather than image removal) that predicts error across benchmarks. Use it to show which benchmark items are genuinely visual.
- *Compute:* About 50–100 GPU-hours.

**13. Linguistic context reshaping visual representations inside the LLM**
- *Gap:* Linguistic context recodes visual representations — [2608.00035](https://arxiv.org/pdf/2608.00035). Visual representations in the LM — [2510.04819](https://arxiv.org/pdf/2510.04819). OCR heads — [2609.18823](https://arxiv.org/pdf/2609.18823). It is unclear whether prompt-dependent recoding helps or hurts perception, and whether it explains prompt sensitivity on perception benchmarks.
- *Compute:* Under 100 GPU-hours.

**14. Training-free dense features from frozen SSL backbones (DINOv3) for MLLM grounding**
- *Gap and evidence:* CVPR 2026 oral INSID3 segments concepts with frozen DINOv3 features, training-free (+6.1 mIoU, 3× fewer parameters) — [CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral). Combined with the adapter drop-off (#10), an open question is whether frozen dense SSL features injected at test time can fix MLLM grounding.
- *Compute:* LoRA or projector training on 7B takes about 1–3 days.

Rough priority for a single-GPU CVPR 2027 paper (my judgement): #1 (unified arbitration), #2 (RL vs. perception, mechanistically), #6 (dense convergence), #3 (oracle pruning regret), #9 (SAE intervention limits). These combine analysis novelty with low compute and are hard to "scoop" with a method trick.

### Gaps
- I did not verify the exact open checkpoints of R1-style RL VLMs (e.g. names of base/RL pairs) available as of Oct 2026. Check Hugging Face before committing to #2.
- Compute numbers are my estimates from model sizes (7–8B bf16 ≈ 16–18 GB; H200 = 141 GB), not from sources.
- I did not check the CVPR 2027 deadline and policies.

## Known, unexplained MLLM failure modes and the benchmarks that expose them

### Takeaway
Hallucination, counting, spatial reasoning, fine-grained perception and language-prior reliance all show the same pattern in 2026 probing work: the information is often decodable internally but not acted upon. The mechanism (arbitration vs. readout vs. adapter loss) is contested. First-generation vision-centric benchmarks are saturating, and new ones (BabyVision, V-FAT, VIA-Bench, HoloCount, VSI-SUPER) keep showing large human–model gaps.

### Cited Findings
- MMVP, MMStar and CV-Bench are now "largely saturated", which motivates new perception benchmarks — [An Exam for Active Observers (2607.16165)](https://arxiv.org/pdf/2607.16165)
- BabyVision: state-of-the-art MLLMs consistently fail basic visual tasks; models are compared against children aged 3–12 — [BabyVision (2601.06521)](https://arxiv.org/html/2601.06521v1)
- VIA-Bench (1,004 QA on illusions and anomalies): top models lag substantially behind humans — [2602.01816](https://arxiv.org/html/2602.01816)
- V-FAT deliberately pits linguistic priors against visual evidence — [2601.04897](https://arxiv.org/pdf/2601.04897). "Seeing or Knowing?" separates perception failure from utilization failure — [2607.26326](https://arxiv.org/abs/2607.26326)
- VisFactor digitises 20 vision-centric subtests of a cognitive psychology battery (v4 May 2026) — [2502.16435](https://arxiv.org/html/2502.16435)
- Encoding-grounding dissociation: correct and incorrect samples encode the visual attribute equally; the failure is in arbitration — [2604.09364](https://arxiv.org/abs/2604.09364)
- Visual attention sinks come from massive activations in the same dimensions as the BOS token — [2503.03321](https://arxiv.org/html/2503.03321)
- Only VSI-Bench and Tomato show significant drops when visual input is removed, i.e. many video benchmarks are answerable blind — [Cambrian-S (2511.04670)](https://arxiv.org/pdf/2511.04670)
- Thinking-with-images tool use is brittle under simple orientation changes or corruptions (reported via survey/snippet) — [Thinking with Programming Vision (2512.03746)](https://arxiv.org/html/2512.03746v1)
- Occlusion perception benchmark — [2508.04059](https://arxiv.org/pdf/2508.04059)

### Inferences
- The modality gap (CLIP embedding-space separation) did not surface as an active 2026 thread in my searches. It looks less central than the arbitration and readout story inside MLLMs.
- A "blind-baseline + counterfactual patch" control should be standard in any new benchmark or analysis paper. Reviewers in 2026–27 will expect it, given the Cambrian-S and Rethinking-CD findings.

### Gaps
- I did not fetch the original MMVP, BLINK, CV-Bench or VSI-Bench papers in this session. Known references: [MMVP / Eyes Wide Shut (2401.06209)](https://arxiv.org/abs/2401.06209), [BLINK (2404.12390)](https://arxiv.org/abs/2404.12390), [Cambrian-1 / CV-Bench (2406.16860)](https://arxiv.org/abs/2406.16860), [VSI-Bench / Thinking in Space (2412.14171)](https://arxiv.org/abs/2412.14171). Current leaderboard numbers are not verified.
- I found no 2026 source specifically on the CLIP/SigLIP2 modality gap.

## Open representation phenomena (registers/high-norm tokens, Platonic convergence, dense features, SSL scaling)

### Takeaway
Registers are no longer the only fix (attention bias and test-time neuron editing work about as well). How artifacts propagate into MLLMs is unstudied. Platonic convergence has been substantially deflated by calibration in 2026, leaving "what converges, at what granularity" open. Dense-feature degradation at scale is real (DINOv3's Gram anchoring), but whether it affects language-free WebSSL-type models is untested.

### Cited Findings
- DINOv3 keeps 4 registers. An attention-bias/value-gating alternative is near-equal (86.6 vs 86.5). DINOv3 scales to 7B parameters and 1.7B images with a focus on dense features — [DINOv3 (2508.10104)](https://arxiv.org/pdf/2508.10104); [Meta AI](https://ai.meta.com/research/dinov3/)
- DINOv3 introduces Gram anchoring to counter dense-feature quality degradation during long training — [2601.08078 (citing DINOv3)](https://arxiv.org/pdf/2601.08078)
- A sparse set of neurons drives the high-norm outliers in CLIP and DINOv2; editing them at test time removes the artifacts without retraining — [2506.08010](https://arxiv.org/html/2506.08010v2)
- 2026 cross-architecture reassessment of register necessity — [2603.25803](https://arxiv.org/pdf/2603.25803) (abstract not read)
- Web-SSL (DINOv2 architecture, 1–7B): visual SSL keeps improving with scale while CLIP saturates. Web-DINO-7B matches SigLIP/SigLIP2 on VQA with 5× less data — [2504.01017](https://arxiv.org/pdf/2504.01017)
- PRH similarity metrics are inflated by width and depth. After calibration only neighbourhood measures keep scaling trends — [2602.14486](https://arxiv.org/pdf/2602.14486); [EPFL project page](https://brbiclab.epfl.ch/projects/aristotelian/)
- Vision-language PRH evidence is weaker than claimed: the original used mKNN on 1,024 pairs, which captures only coarse structure — [Back into Plato's Cave (2604.18572)](https://www.alphaxiv.org/abs/2604.18572)
- "Structure over geometry" — [2609.27252](https://arxiv.org/abs/2609.27252). Alignment rests on linear structure — [2605.28870](https://arxiv.org/pdf/2605.28870)
- Frozen DINOv3 features alone give a CVPR 2026 oral (INSID3, training-free segmentation, +6.1 mIoU) — [CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral)

### Inferences
- Patch-level (dense) convergence across SSL, CLIP-type and MAE/JEPA models is a cheap, unclaimed analysis axis (#6 above).
- The artifact-to-sink pipeline (encoder high-norm token → LLM visual sink) links two literatures that currently cite each other little (#4 and #5).

### Gaps
- No SigLIP2-specific artifact study was found.
- I found no 2026 source on V-JEPA 2 or I-JEPA representation analysis; that thread is unverified.
- Whether WebSSL shows dense degradation without Gram anchoring is unknown.

## Does RL / "reasoning" (GRPO, R1-style) improve visual perception or just language?

### Takeaway
The evidence leans toward "mostly not perception, unless the reward targets perception". No controlled mechanistic study has localised what RL changes inside the model. That gap is the opportunity.

### Cited Findings
- Accuracy-only RLVR gives no statistically significant perception gain (McNemar test). A perception reward is needed — [Perception-R1 (2506.07218)](https://arxiv.org/abs/2506.07218)
- Region-level RL (regions as actions) improves fine-grained perception at every token budget on 6 benchmarks and 4 backbones, beating the largest-budget baseline with about 4× fewer tokens — [Vision-RL2 / Region-Level Policy Optimization (2609.19745)](https://arxiv.org/abs/2609.19745)
- GRPO-learned active region selection on Qwen2.5-VL-7B — [ACTIVE-o3 (2505.21457)](https://arxiv.org/html/2505.21457v2)
- Tool-use RL (ReVPT) reports up to +9.44% on CV-Bench (secondary summary; unverified) — [emergentmind](https://www.emergentmind.com/papers/2509.01656)
- "Zooming without Zooming" (2026) questions whether explicit tool use is needed at inference and distills region inspection into training (cited in [Perceive-to-Reason (2607.01191)](https://arxiv.org/html/2607.01191v1))
- Interleaved thinking-with-images produces long, noisy contexts. Decoupling localisation from answering helps — [2607.01191](https://arxiv.org/html/2607.01191v1)
- Reasoning in latent visual embedding space gives substantial gains on perception-heavy VQA — [Latent Visual Reasoning (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/efa0939caa8a906fd0a61a9c60e46e8b-Paper-Conference.pdf)
- DeepEyes shows end-to-end RL can induce visual thinking behaviours (survey synthesis, not a controlled ablation) — [ACL 2026 survey](https://aclanthology.org/2026.acl-long.2087.pdf)

### Inferences
- Testable hypothesis for a paper: R1-style RL leaves encoder and early-layer visual codes unchanged and shifts only late arbitration/readout. This links directly to the 2604.09364 framework. If true, "perception rewards" work by changing the readout, not the seeing.

### Gaps
- I found no paper that does a layer-wise base-vs-RL representational comparison for VLMs.

## Visual token compression/pruning: what remains unexplained?

### Takeaway
The method space is saturated: ZOO-Prune, D2Pruner, PosPrune, RoRA, SinkPruner and layer-aware PE all appeared in 2025–26. *Why* simple uniform or random baselines stay competitive, and how far heuristics sit from the causal importance of tokens, remain unexplained.

### Cited Findings
- Last-token attention importance is position-biased (FastV keeps late-sequence tokens far more often) — [Position-reweighted pruning overview](https://www.emergentmind.com/topics/position-reweighted-visual-token-pruning)
- Methods catastrophically fail on fine-grained and localisation tasks; LLaVA shows a content-agnostic bottom-of-image emphasis — [D2Pruner (2512.19443)](https://arxiv.org/pdf/2512.19443)
- Random selection or pooling often beats complex methods — [2502.11501](https://arxiv.org/html/2502.11501v1)
- Shallow-layer positional bias is driven by architecture, not semantics — [PosPrune (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/38020)
- Diversity methods over-allocate to cluttered backgrounds (HoloV); methods are agnostic to token role — [RoRA (2608.07088)](https://arxiv.org/pdf/2608.07088)
- Position encodings of retained tokens hurt grounding — [2609.23715](https://arxiv.org/pdf/2609.23715). Sink-free pruning — [SinkPruner (2609.01004)](https://arxiv.org/pdf/2609.01004). Training-free CVPR 2026 pruning — [ZOO-Prune](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_ZOO-Prune_Training-Free_Token_Pruning_via_Zeroth-Order_Gradient_Estimation_in_Vision-Language_CVPR_2026_paper.html)
- Token compression discards detail and causes visual-cue forgetting — [Vision Remember (2506.03928)](https://arxiv.org/html/2506.03928)

### Inferences
- "Yet another pruner" is saturated. An oracle-regret or causal-importance analysis (#3) is the defensible angle.

### Gaps
- I found no study measuring causal (ablation-based) token importance as ground truth.

## Sparse autoencoders / mechanistic interpretability for vision: open questions

### Takeaway
SAEs on CLIP and DINO produce monosemantic-looking features, but their reliability, task relevance and usefulness as steering handles are unresolved. Steering results inside VLMs are small and inconsistent.

### Cited Findings
- Whether SAE features can serve as an actionable intervention space is open; reconstruction saliency and task saliency are mismatched — [Visual Sparse Steering (2506.01247)](https://arxiv.org/pdf/2506.01247)
- A DINOv2 "crash" feature does not fire reliably. CLIP SAEs capture country-level cultural abstractions; DINOv2 SAEs do not — [2502.06755](https://arxiv.org/html/2502.06755v1)
- SAEs learn monosemantic features in VLMs (NeurIPS 2025) — [2504.02821](https://arxiv.org/pdf/2504.02821)
- DINOv2 does best in SAE probing among the foundation models tested — [ICCV 2025W](https://openaccess.thecvf.com/content/ICCV2025W/Findings/papers/Olson_Probing_the_Representational_Power_of_Sparse_Autoencoders_in_Vision_Models_ICCVW_2025_paper.pdf)
- Concept-bottleneck SAEs — [2512.10805](https://arxiv.org/pdf/2512.10805). Cross-modal structured SAEs — [2607.08605](https://arxiv.org/pdf/2607.08605). Information-scope disentangling in CLIP SAEs — [2604.05724](https://arxiv.org/pdf/2604.05724). Minkowski geometry of DINO concepts — [2510.08638](https://arxiv.org/pdf/2510.08638). LMs explaining visual features via steering — [2603.22593](https://arxiv.org/pdf/2603.22593)
- SAE-guided steering in VLMs: at most +3.8%, with degradation in some setups — [2604.09364](https://arxiv.org/abs/2604.09364)
- VLA mechanistic study (ICLR 2026): discrete action tokens block SAE intervention, continuous ones allow it — [ICLR 2026](https://iclr.cc/virtual/2026/10013199)

### Inferences
- The opportunity is a "where to steer" result: SAE features before the arbitration layer vs. after. It also needs strong non-SAE baselines (difference of means, linear probe), which reviewers increasingly demand.

### Gaps
- I could not confirm which SAE-for-vision papers were orals.

## Recognised training-free or analysis papers (orals/highlights) at CVPR 2025/2026, ICCV 2025, NeurIPS 2025, ICLR 2026

### Takeaway
I could confirm only a small sample. Training-free work on frozen strong backbones (INSID3 on DINOv3) does get CVPR 2026 orals. Venue-level lists must be checked manually.

### Cited Findings
- INSID3: a training-free segmentation oral using only frozen DINOv3 features (CVPR 2026, Oral Session 4D) — [CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral)
- MODIX (training-free positional granularity for VLMs) is on the CVPR 2026 highlights page; oral status not confirmed — [CVPR 2026 Highlights](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026)
- ZOO-Prune, ViTPrompt and LLMind are training-free CVPR 2026 papers; oral status not confirmed — [ZOO-Prune](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_ZOO-Prune_Training-Free_Token_Pruning_via_Zeroth-Order_Gradient_Estimation_in_Vision-Language_CVPR_2026_paper.html); [ViTPrompt](https://openaccess.thecvf.com/content/CVPR2026/html/Qin_ViTPrompt_Training-Free_Prompt_Refinement_with_Visual_Tokens_for_Open-Vocabulary_Detection_CVPR_2026_paper.html)
- Community lists of CVPR 2026 orals — [GitHub list](https://github.com/ooonesevennn/CVPR_2026_Oral_Papers); [SkalskiP top CVPR 2026](https://github.com/SkalskiP/top-cvpr-2026-papers)
- The ICLR 2026 orals page lists LLM interpretability orals (persistent-homology latent geometry; temporal SAEs). No clear VLM-interpretability oral was found — [ICLR 2026 Orals](https://iclr.cc/virtual/2026/events/oral)
- Latent Visual Reasoning is published at ICLR 2026 (presentation type not verified) — [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/efa0939caa8a906fd0a61a9c60e46e8b-Paper-Conference.pdf)

### Inferences
- Oral-tier analysis papers in this area usually pair a mechanistic claim with a cheap fix that works. Pure "benchmark plus failures" papers rarely reach oral in 2026.

### Gaps
- I did not verify ICCV 2025 or NeurIPS 2025 oral lists for VLM analysis papers. Known 2023–24 precedents (e.g. ViT registers, ICLR 2024 oral; Eyes Wide Shut, CVPR 2024) were not re-verified.

## Saturated directions to avoid

### Takeaway
Avoid the following unless they come with a new causal finding: contrastive/attention-reweighting hallucination decoding, attention-score token pruners, new "MLLMs fail at X" benchmarks without blind or counterfactual controls, and PRH claims based on uncalibrated CKA.

### Cited Findings
- CD hallucination gains are explained by greedy decoding and output bias; a no-image constant bias beats all CD methods on POPE — [2607.25196](https://arxiv.org/html/2607.25196); [Mirage (2504.10020)](https://www.alphaxiv.org/overview/2504.10020v3). New CD variants nonetheless keep appearing (SDCD, ACG, MACD, OAVCD in 2026) — [Awesome-LVLM-Hallucination](https://github.com/NishilBalar/Awesome-LVLM-Hallucination)
- Pruning: at least 6 new 2025–26 pruners in my searches alone; random/pooling baselines are competitive — [2502.11501](https://arxiv.org/html/2502.11501v1)
- Benchmarks: MMVP, MMStar and CV-Bench are saturated — [2607.16165](https://arxiv.org/pdf/2607.16165). VSI-SUPER has shortcut problems — [2511.16655](https://arxiv.org/html/2511.16655v1)
- PRH: uncalibrated metrics are inflated by scale — [2602.14486](https://arxiv.org/pdf/2602.14486)

### Inferences
- Thinking-with-images RL methods (DeepEyes-style) also look crowded, judging by the size of the awesome-list — [Awesome_Think_With_Images](https://github.com/zhaochen0110/awesome_think_with_images). A new *method* there needs multi-node rollouts to compete. An *analysis* fits 1 GPU.

### Gaps
- I did not count papers per topic systematically. The saturation judgements are qualitative.
