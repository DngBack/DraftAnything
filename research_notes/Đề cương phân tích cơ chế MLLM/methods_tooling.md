# Methods and Tooling for Mechanistic Analysis of 7–8B Open MLLMs on One H200 (claim: failures come from late-layer readout, not perception)

Note on sourcing: links marked **[web-verified]** were fetched or surfaced by search in this session (2026-10-13). Links marked **[prior-knowledge]** are standard references whose arXiv IDs I am confident of but did not re-fetch in this session. The report writer should spot-check them before final citation. Every compute or memory number is **[OWN ESTIMATE]** unless a source is given.

---

## Q1. Which techniques to use, and how to apply them correctly to VLMs

### Takeaway
To show that the failure is in readout and not perception, you need three converging lines of evidence. (a) **Decodability**: linear probes and logit/tuned lens on image-token and last-token states show the answer-relevant visual information is present at mid layers. (b) **Causal use**: activation patching, attention knockout and interchange/DAS interventions show where that information is moved into the answer position, and that the late layers fail to use it. (c) **Repair**: steering, back-patching or SAE-feature interventions at late layers recover accuracy. Probes alone are not enough. The paper must pair every probe result with a causal intervention.

### Cited Findings
- **Direct precedent for the "perception vs. readout" claim.** Fu, Bonnen, Guillory, Darrell (UC Berkeley, arXiv 2506.08008, Jun 2025) compared VLM VQA-style answers with a *direct readout (probe) of the VLM's own vision encoder*. On vision-centric tasks (depth, correspondence, low-level matching, affordance, art style, 3D awareness), VLMs do substantially worse than their encoders, sometimes near chance. The authors attribute this to the LLM failing to use the visual features, studying degradation through the model, prompt sensitivity and the LLM's role. Models: Prismatic VLMs (Vicuna-1.5 with frozen DINOv2/CLIP/SigLIP/ViT-IN1k) — [arXiv 2506.08008](https://arxiv.org/abs/2506.08008) **[web-verified]**. *This is the closest prior work. The new paper must go beyond it with causal, layer-resolved evidence and modern models (Qwen2.5/3-VL, InternVL3, LLaVA-OV, Gemma 3).*
- **Logit lens on visual tokens, token ablation and attention knockout (LLaVA-1.5).** Neo et al. (ICLR 2025) report three things. Visual-token hidden states in the LM are refined toward interpretable vocabulary embeddings. Ablating object-specific image tokens drops object-identification accuracy by >70%. Blocking attention shows the model extracts object information from object tokens in middle-to-late layers into the last position. Code: github.com/clemneo/llava-interp — [arXiv 2410.07149](https://arxiv.org/abs/2410.07149), [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/900fb3439e4968df79a6f2bfedec49cd-Abstract-Conference.html) **[web-verified]**
- **Attention knockout for image→question→last-token flow.** Zhang, Yadav, Han, Shutova (CVPR 2025) block attention edges between token groups at chosen layers (adapting Geva et al.) and measure the drop in answer probability. They find general image features moving into question tokens in lower layers, question-relevant object information in middle layers, and the combined representation reaching the last position in higher layers (LLaVA series, GQA). Code and data are released — [arXiv 2411.18620](https://arxiv.org/abs/2411.18620), [CVPR OA PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Cross-modal_Information_Flow_in_Multimodal_Large_Language_Models_CVPR_2025_paper.pdf) **[web-verified]**
- **A conflicting flow pattern.** Yin et al., "Lifting the Veil on Visual Information Flow in MLLMs" (CVPR 2025), report that shallow layers mix image and instruction tokens while deeper layers mostly aggregate visual information within image tokens — [CVPR OA](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_Lifting_the_Veil_on_Visual_Information_Flow_in_MLLMs_Unlocking_CVPR_2025_paper.pdf) **[web-verified, via search snippet]**. *Flow conclusions are model- and task-dependent, so test several models.*
- **Two readout pathways, with fallback.** Salazar, Frank, Oneata, Elliott, Fierro (arXiv 2607.03358, Jul 2026) use causal patching to identify a *direct* pathway (visual information stays in image tokens and is read by the final token at later layers) and a *text-mediated* pathway (visual information goes to query tokens first, then to the final token). Pathway choice depends on the task, data distribution and prompt. Under attention knockout or corrupted-input patching, models **fall back** to the other pathway — [arXiv 2607.03358](https://arxiv.org/abs/2607.03358) **[web-verified, abstract only]**. *Implication: single-path knockouts can understate a component's importance (redundancy). Knock out both paths jointly.*
- **Back-patching as a "repair" intervention.** "Same Task, Different Circuits" (arXiv 2506.09047, NeurIPS 2025) patches visual-token activations from later layers back into earlier layers. This closes about a third of the image-vs-text performance gap on average — [arXiv 2506.09047](https://arxiv.org/abs/2506.09047) **[web-verified, via search snippet]**. This is a good template for a causal "readout-fix" experiment.
- **Patching with noise-corrupted visual tokens, per family.** "Dual-Pathway Circuits of Object Hallucination" (arXiv 2605.13156) corrupts visual tokens at layer 0 with noise and restores individual components. It notes that visual information enters through architecture-specific mechanisms, so the patching interface differs per model family — [arXiv 2605.13156](https://arxiv.org/pdf/2605.13156) **[web-verified, via search snippet]**
- **Last-token patching for perception-vs-prior conflict.** "Vision-Default, Prior-Override" (arXiv 2606.28273) patches the last-token position between visual and prior-knowledge prompts on the same image to find the deciding components — [arXiv 2606.28273](https://arxiv.org/pdf/2606.28273) **[web-verified, via search snippet]**. This is directly relevant to a "late readout" story.
- **Linear probes with control tasks.** Hewitt & Liang (2019) define *selectivity* as probe accuracy on the real task minus accuracy on a control task (random labels assigned per word type). Report it and prefer low-capacity linear probes — [arXiv 1909.03368](https://arxiv.org/abs/1909.03368) **[prior-knowledge]**. Related: Belinkov "Probing Classifiers: Promises, Shortcomings, and Advances" — [arXiv 2102.12452](https://arxiv.org/abs/2102.12452); amnesic probing (Elazar et al.) tests *use* by removing the probed direction (INLP) and measuring the behavioral change — [arXiv 2006.00995](https://arxiv.org/abs/2006.00995) **[prior-knowledge]**
- **Tuned lens** (Belrose et al.) trains an affine translator per layer, which is more faithful than the logit lens on models whose early layers are not aligned to the unembedding. Library: `tuned-lens` — [arXiv 2303.08112](https://arxiv.org/abs/2303.08112) **[prior-knowledge]**
- **Activation-patching best practice.** Zhang & Nanda (ICLR 2024) recommend Symmetric Token Replacement (STR) or counterfactual corruptions over Gaussian Noise (GN), because GN produces OOD activations and can give misleading localization. They also recommend logit difference over probability as the metric, and caution about sliding-window vs. single-layer patching — [arXiv 2309.16042](https://arxiv.org/abs/2309.16042) **[prior-knowledge]**. Heimersheim & Nanda, "How to use and interpret activation patching": denoising (clean→corrupt) finds *sufficient* components and noising (corrupt→clean) finds *necessary* ones. Interpret AND/OR circuits accordingly — [arXiv 2404.15255](https://arxiv.org/abs/2404.15255) **[prior-knowledge]**. Causal tracing originated in ROME (Meng et al.), with Gaussian noise on subject embeddings — [arXiv 2202.05262](https://arxiv.org/abs/2202.05262) **[prior-knowledge]**
- **Attribution patching / AtP\*.** Kramár et al. (DeepMind 2024) use a gradient linear approximation to patching. Two fixes address false negatives: QK-fix for attention saturation and GradDrop for cancellation. They recommend verifying top candidates with real patching — [arXiv 2403.00745](https://arxiv.org/abs/2403.00745) **[prior-knowledge]**
- **Path patching** (Wang et al., IOI) isolates the effect of a sender→receiver edge — [arXiv 2211.00593](https://arxiv.org/abs/2211.00593). **Attention knockout** (Geva et al., "Dissecting Recall of Factual Associations") — [arXiv 2304.14767](https://arxiv.org/abs/2304.14767) **[prior-knowledge]**
- **DAS / interchange interventions.** Geiger et al. find causal-variable subspaces with a learned rotation — [arXiv 2303.02536](https://arxiv.org/abs/2303.02536). This is supported in pyvene **[prior-knowledge]**.
- **Semantic image corruption for VLM patching.** NOTICE (Golovanevsky et al., NAACL 2025) uses "Semantic Image Pairs" (SIP): a minimally edited image changes one object or attribute instead of adding Gaussian noise. They also use symmetric token replacement on text — [arXiv 2406.16320](https://arxiv.org/abs/2406.16320) **[prior-knowledge]**
- **Information storage and transfer in MLLMs** (Basu et al., NeurIPS 2024): causal tracing on constraint tokens with the MultimodalCausalTrace method; early-MLP and attention storage — [arXiv 2406.04236](https://arxiv.org/abs/2406.04236) **[prior-knowledge]**. **Kaduri et al. ("What's in the image? A deep-dive into the vision of VLMs", CVPR 2025)** use attention knockout to show that query tokens carry global image information and that mid layers dominate cross-modal flow — [alphaxiv 2411.17491](https://alphaxiv.org/abs/2411.17491) **[web-verified as existing; content prior-knowledge]**
- **Logit lens on image tokens for hallucination editing** (Jiang et al., ICLR 2025, "Interpreting and Editing VLM Representations to Mitigate Hallucinations") — [arXiv 2410.02762](https://arxiv.org/abs/2410.02762) **[prior-knowledge]**

### Inferences (recommended protocol for the CVPR paper)
1. **Probe stage (perception present?).** Fit a linear probe (logistic regression, L2, cross-validated) per layer on (i) mean-pooled or object-region image tokens and (ii) the last prompt token, predicting the ground-truth answer attribute. Also probe the *vision-encoder output* (pre-projector), as Fu et al. did. Report Hewitt–Liang selectivity with a control task, and a random-init-model or shuffled-image baseline. The key plot is probe accuracy versus layer next to model output accuracy. The "readout gap" is probe accuracy at the best layer minus generation accuracy.
2. **Lens stage.** Logit lens is cheap for the last token. Tuned lens is more faithful but needs training per model (a few GPU-hours, **[OWN ESTIMATE]**). Use the lens to show *when* the correct answer becomes top-k at the last position and whether it is overwritten in the final layers ("late override").
3. **Causal stage.**
   - (a) Attention knockout, image→last and image→query→last, in sliding layer windows of about 4–6 layers, done on both pathways (Salazar et al. fallback).
   - (b) Activation patching on residual stream / attention-out / MLP-out at the last position, using *counterfactual image pairs* (NOTICE-style SIP, or a dataset where the answer flips). Report both denoising and noising.
   - (c) AtP\* for full layer×position maps, then verify the top-k with real patching.
   - (d) DAS or interchange interventions with pyvene to find a low-dimensional answer subspace at mid layers and test whether late layers read it.
4. **Repair stage (the strongest evidence for "readout").** Inject the mid-layer correct-answer direction or the probe-derived steering vector at late layers. Alternatively, back-patch (Same-Task-Different-Circuits style) or fine-tune *only* late layers or the LM head (a "readout-only fine-tune" control), and show that accuracy recovers to the probe ceiling while the vision encoder and early layers stay frozen.

### Gaps
- I did not fetch full method sections of 2607.03358 / 2605.13156 / 2606.28273. Their models, corruptions and layer indices are unknown from this session.
- I found no peer-reviewed study comparing image-corruption types (GN on pixels vs. GN on visual embeddings vs. counterfactual images vs. token mean-ablation) in VLMs specifically. The recommendation to prefer counterfactual images is an extrapolation of Zhang & Nanda's text results plus NOTICE.

---

## Q2. Pitfalls documented in 2024–2026 and how to validate

### Takeaway
The main threats to a "readout, not perception" claim:
- Probes decode information the model never uses.
- Patching with OOD (Gaussian-noise) corruptions.
- Subspace-patching illusions.
- High-norm / attention-sink tokens dominating knockout results.
- Redundant pathways masking effects.
- Prompt-format and MC-letter biases being mistaken for "readout failure".

Validate across ≥3 model families, ≥3 seeds or data resamples, with bootstrap CIs and effect sizes.

### Cited Findings
- **Probes vs. use.** High probe accuracy shows decodability, not use. Selectivity and control tasks (Hewitt & Liang) — [arXiv 1909.03368](https://arxiv.org/abs/1909.03368). Ravichander et al., "Probing the Probing Paradigm": probes can find information that the task doesn't require — [arXiv 2005.00719](https://arxiv.org/abs/2005.00719). Amnesic probing (Elazar et al.) tests causal use — [arXiv 2006.00995](https://arxiv.org/abs/2006.00995) **[prior-knowledge]**. *For this paper the pitfall is welcome: "decodable but unused" is the readout claim. It must still be shown causally (e.g., injecting the decoded direction fixes the output).*
- **Subspace activation patching illusion.** Makelov, Lange, Nanda (ICLR 2024) show that patching along a DAS-found subspace can change behavior by activating a *dormant parallel pathway* rather than the true mechanism. Validate with the subspace's components in the row space vs. null space of downstream weights, and check generalization to held-out data — [arXiv 2311.17030](https://arxiv.org/abs/2311.17030) **[prior-knowledge]**
- **Gaussian noise yields OOD activations** and different localization than STR or counterfactuals. The logit-difference metric is preferred — [arXiv 2309.16042](https://arxiv.org/abs/2309.16042) **[prior-knowledge]**
- **Attention sinks / high-norm tokens.** Xiao et al. (StreamingLLM) show the first token absorbs attention — [arXiv 2309.17453](https://arxiv.org/abs/2309.17453). Sun et al., "Massive Activations in LLMs", find a few huge-magnitude activations acting as biases — [arXiv 2402.17762](https://arxiv.org/abs/2402.17762). Darcet et al., "Vision Transformers Need Registers", find high-norm artifact tokens in ViTs (CLIP/DINOv2) that carry global information — [arXiv 2309.16588](https://arxiv.org/abs/2309.16588) **[prior-knowledge]**. *Pitfall: knocking out attention to "all image tokens" also removes sink or high-norm visual tokens, which can break the model for non-semantic reasons. Control: knock out a random equal-size set of image tokens, or only the non-sink tokens, and report norm statistics.*
- **Visual information entry differs per architecture**, so the patching interface must be adapted per family — [arXiv 2605.13156](https://arxiv.org/pdf/2605.13156) **[web-verified, snippet]**
- **Pathway redundancy and fallback** after ablation — [arXiv 2607.03358](https://arxiv.org/abs/2607.03358) **[web-verified]**
- **Prompt sensitivity.** Fu et al. explicitly analyze task-prompt sensitivity as a factor in VLM underperformance — [arXiv 2506.08008](https://arxiv.org/abs/2506.08008) **[web-verified]**. Sclar et al. report up to tens of points of accuracy variance across semantically equivalent prompt formats (FormatSpread) — [arXiv 2310.11324](https://arxiv.org/abs/2310.11324) **[prior-knowledge]**
- **Multiple-choice letter bias.** Zheng et al. (ICLR 2024), "LLMs Are Not Robust Multiple Choice Selectors": token bias toward specific option IDs; PriDe debiasing — [arXiv 2309.03882](https://arxiv.org/abs/2309.03882) **[prior-knowledge]**. *Control: permute option order (cyclic over all positions) and report the permutation-averaged accuracy. A "readout failure" that disappears under permutation is letter bias, not a readout failure.*
- **Image-token position effects.** In LLaVA, object information is localized to spatially matching tokens (Neo et al.) — [arXiv 2410.07149](https://arxiv.org/abs/2410.07149) **[web-verified]**. Placing the image after the text vs. before it, and RoPE distance (Qwen M-RoPE), change information flow. *Control: fixed template; ablate image position; report the number of image tokens per sample.*

### Inferences (validation checklist)
- **Models.** At least 3 families with different connectors: Qwen2.5-VL-7B (or Qwen3-VL-8B), InternVL3-8B, LLaVA-OneVision-7B, plus Gemma-3-4B/12B for SAE analysis. Optionally LLaVA-1.5-7B for comparison with Neo/Zhang.
- **Stats.** Bootstrap 95% CIs over examples (1,000–10,000 resamples). Report paired effect sizes (Δ logit-diff, normalized patching effect ∈ [0,1]) and multiple-comparison control (Holm or BH) across the layer grid. Use ≥3 probe seeds and CV folds, and ≥2 prompt templates. Report permutation-averaged MC accuracy.
- **Corruptions.** Main: counterfactual images (answer-flipping minimal edits). Secondary: visual-token mean-ablation with dataset-mean image tokens. Report GN only as a robustness check.
- **Dissociation.** Show that at the same examples probe accuracy is high at layer ℓ\* while lens/patching shows the answer is lost or overwritten after ℓ\*. A double dissociation (perception-corrupted examples fail at early layers; readout failures fail late) is the strongest framing.

### Gaps
- I found no 2025–2026 paper that systematically quantifies the attention-sink confound *in VLM image-token knockouts*. This is an inference from the LLM and ViT literature.
- OpenReview reviews of VLM-interp papers were not retrieved, so common reviewer complaints are not sourced.

---

## Q3. Tools, and which models they support

### Takeaway
Use **nnsight** (or plain HF hooks) as the primary engine for all HF VLMs: Qwen2.5-VL, Qwen3-VL, InternVL3, LLaVA-OV, Gemma 3, Molmo. Use **pyvene** for DAS and interchange interventions. Use **SAELens + Gemma Scope 2** for SAE/transcoder work on Gemma 3. Use **ViT-Prisma** for the vision-encoder side. Use **clemneo/llava-interp** as a LLaVA-1.5 reference implementation. TransformerLens is not the right tool for VLMs: I found no confirmation of VLM support.

### Cited Findings
- **nnsight**: "supports local execution on any PyTorch model". `remote=True` runs the same code on NDIF-hosted models. There is a dedicated vLLM integration (activation capture, steering, patching, tensor parallelism). The docs navigation includes a "vlm" section, a "VLM Logit Lens" mini-paper and a vision cross-attention ablation tutorial — [nnsight.net](https://nnsight.net/) **[web-verified]**. Paper: NNsight and NDIF — [arXiv 2407.14561](https://arxiv.org/abs/2407.14561) **[prior-knowledge]**.
- **TransformerLens**: v3 "TransformerBridge" claims to load >9,000 HF models via architecture adapters, plus experimental Mamba adapters — [TransformerLens docs](https://docsearch.algolia.com/mcp/docs/repo/transformerlensorg/transformerlens) **[web-verified]**. TL3 is in beta, with ~5k models of "hypothetical support" vs. ~200 in v2 — [Manifund update](https://manifund.org/Bryce) **[web-verified]**. **I found no source confirming multimodal/VLM support.**
- **ViT-Prisma**: TransformerLens-style interface to 75+ vision/video transformers (timm, OpenCLIP, HF). Supports SAE, transcoder and crosscoder training. Releases 80+ pretrained SAEs for CLIP and DINO ViTs. Reports that effective vision SAEs can be much less sparse than language SAEs — [arXiv 2504.19475](https://arxiv.org/abs/2504.19475) **[web-verified]**
- **pyvene**: intervention library that wraps arbitrary PyTorch modules via config; supports trainable interventions (DAS/rotated-subspace, Boundless DAS) and interchange interventions — [arXiv 2403.07809](https://arxiv.org/abs/2403.07809) **[prior-knowledge]**
- **llava-interp** (Neo et al.): logit lens, token ablation and attention knockout code for LLaVA-1.5 — github.com/clemneo/llava-interp, per [arXiv 2410.07149](https://arxiv.org/abs/2410.07149) **[web-verified]**
- **Cross-modal information flow code** (Zhang et al., attention knockout on LLaVA) is released — [arXiv 2411.18620](https://arxiv.org/abs/2411.18620) **[web-verified]**
- **Public SAEs**:
  - **Gemma Scope 2** has SAEs on three sites plus transcoders for every layer of Gemma 3 270M/1B/4B/12B/27B (pt and it). It also includes residual-stream crosscoders for every base model and cross-layer transcoders for 270M and 1B. Loading is via SAELens `SAE.from_pretrained`, and there is a Neuronpedia demo (27B-IT safety features; uploads were planned to finish by Jan 26, 2026) — [HF google/gemma-scope-2](https://huggingface.co/google/gemma-scope-2), [Gemma Scope docs](https://ai.google.dev/gemma/docs/gemma_scope?hl=en), [Neuronpedia](https://www.neuronpedia.org/gemma-scope-2) **[web-verified]**. **Whether they were trained on, or validated for, image-token activations is NOT stated on the pages I checked.** The technical-report PDF is linked from the HF card.
  - **Qwen-Scope** (arXiv 2605.11887): SAEs for the *text* Qwen3 / Qwen3.5 backbones (7 backbones, 14 SAE groups, all layers). It does not appear to include VL variants — [arXiv 2605.11887](https://arxiv.org/pdf/2605.11887) **[web-verified, snippet]**
  - An S2AE paper trains an SAE on Qwen2.5-VL-7B-Instruct layer 5, but I found no public weights. Work on SAEs over the CLIP vision encoder shows steering LLaVA outputs via encoder-side SAE features — [arXiv 2504.02821](https://arxiv.org/abs/2504.02821v3), [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/119210) **[web-verified, snippet]**
  - **No official public SAE for Qwen2.5-VL, LLaVA-OV or InternVL3 was found.**

### Inferences (concrete tool→model recommendation)
| Model (7–8B class) | Primary engine | DAS / interchange | SAEs | Notes **[OWN ESTIMATE / prior-knowledge on architecture]** |
|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | nnsight `NNsight(model)` over HF `Qwen2_5_VLForConditionalGeneration`, or raw `register_forward_hook` | pyvene on `model.model.layers[i]` (module path varies by transformers version) | none public; train your own (see costs) | 28 LM layers, d=3584. Dynamic resolution, so the number of image tokens varies per image. M-RoPE, so fix the resolution (e.g., max_pixels) to make patching positions align across clean/corrupt pairs |
| Qwen3-VL-8B | nnsight / HF hooks | pyvene | none public (Qwen-Scope is text-only) | DeepStack injects ViT features into several early LM layers, so "corrupt at layer-0 input" is not the only entry point. Patch the DeepStack inputs too |
| InternVL3-8B | nnsight (custom `trust_remote_code` modules; hooks work on any nn.Module) | pyvene | none | 448px tiles of 256 tokens each. Fix the tile count for pairs |
| LLaVA-OneVision-7B | nnsight / HF `LlavaOnevisionForConditionalGeneration` | pyvene | none | AnyRes gives up to several thousand image tokens. Use a single-crop setting for patching sweeps |
| Gemma-3-4B / 12B-it | nnsight + **SAELens with Gemma Scope 2** | pyvene | **Gemma Scope 2** (text-trained; check fidelity on image tokens by reconstruction loss / ΔCE before trusting) | 256 image tokens per image (fixed), so position alignment is easy. There is no 7–8B Gemma 3, so use 4B and 12B |
| LLaVA-1.5-7B (baseline) | clemneo/llava-interp, nnsight | pyvene | CLIP-side SAEs via ViT-Prisma | 576 fixed tokens. Easiest for replicating Neo/Zhang |
| Molmo-7B | nnsight / HF hooks (`trust_remote_code`) | pyvene | none | Unverified; check the hook paths manually |
- Use the **vision encoder** (perception-side probes, SAEs) via ViT-Prisma or HF hooks on the vision tower. This shows perception is intact independently of the LLM.
- **baukit** (`TraceDict`, `Trace`) is a lightweight alternative for read/write hooks on any module **[prior-knowledge, not re-verified]**. Fine for probing feature extraction.
- **Tuned lens**: the EleutherAI `tuned-lens` package targets HF causal LMs. For a VLM, train the lenses on the LM backbone with mixed text and image-conditioned activations **[OWN INFERENCE]**.

### Gaps
- Not verified: whether nnsight's "vlm" docs page covers Qwen-VL, LLaVA or Gemma 3 specifically (the mini-paper URL 404'd).
- TransformerLens VLM support (e.g., a HookedLlava) is unconfirmed.
- Gemma Scope 2 image-token coverage is unconfirmed (read the technical report).
- I did not verify LVLM-Interpret (Intel) or other GUI tools in this session.

---

## Q4. Compute and memory costs on one H200 (141 GB HBM3e)

### Takeaway
**[OWN ESTIMATE]** A 7–8B VLM in bf16 uses about 16–18 GB of weights, so the H200 has room for large batches and even backward passes over 1–3k-token sequences. Probing and logit lens over 10–50k images take a few GPU-hours. Exhaustive layer × image-token patching is the cost driver and is infeasible per example at full resolution. Use AtP\* plus patching over token groups. The whole program (5 models) fits comfortably in about 10 GPU-days.

### Cited Findings
- No source in this session gives measured VLM patching throughput. All numbers below are my estimates.

### Inferences (all **[OWN ESTIMATE]**, assumptions stated)
- **Memory.**
  - Qwen2.5-VL-7B is ~8.3B params, about 16.6 GB in bf16. Gemma-3-12B is about 24 GB.
  - Residual stream size: 28 layers × 3584 × 2 B ≈ 200 KB per token. For one image with ~1.3k tokens, caching all layers and positions takes ≈ 260 MB, which is too much to store for many images. Store only (a) the last token, (b) mean-pooled image tokens and (c) object-region pools. That is ≈ 0.6 MB per image, so 50k images ≈ 30 GB on disk (fp16).
  - KV cache is negligible for single-forward scoring.
  - A backward pass for AtP over 1.5k tokens fits with batch ≈ 4–8 (activations ~10–20 GB per sequence without checkpointing).
- **Forward cost.** ≈ 2 × 7.6e9 × 1.5k tokens ≈ 23 TFLOP per example. At ~35–45% MFU of ~990 TFLOPS dense bf16 that is ≈ 0.05–0.07 s per example, or about 15–20 examples/s with batching (SDPA/FlashAttention-2, padded batches of similar token count). Image preprocessing and the ViT often become the bottleneck, so pre-tokenize and cache `pixel_values`.
- **Probing.** 10k images × 1 forward ≈ 10–20 min. Logistic-regression probes on 28 layers × 3 pools on CPU/GPU (cuML or torch) take minutes. Allow **≈1–3 GPU-h per model per dataset** including I/O.
- **Logit lens.** Free with the probing pass, using an unembedding matmul at the last position. Tuned lens training is ≈ 2–6 GPU-h per model.
- **Attention knockout.** About 6–8 layer windows × 3–4 edge types gives ~30 forwards per example. For 1k examples that is 30k forwards, about 0.5–1 GPU-h per model. Implement it by editing the 4D attention mask (SDPA with a custom mask; FlashAttention-2 does not accept arbitrary masks, so use the eager or SDPA path, which is slower and uses more memory).
- **Full residual patching sweep.**
  - L=28 × P≈1.5k positions ≈ 42k forwards per example. That is about 40–60 min per example, so ~200 examples ≈ 150–200 GPU-h. **Not worth it.**
  - Alternatives:
    - (a) Patch **token groups** (object region / background / question / last token, or a 4×4 spatial grid of image tokens): L × ~20 groups = 560 forwards per example, ≈ 30–40 s. 500 examples ≈ 5 GPU-h per model.
    - (b) **AtP\***: 2 forwards + 1 backward per example gives all layer × position × component estimates, ≈ 0.3 s per example, so 1k examples ≈ 5 min. Verify the top 1–5% of cells with real patching.
    - (c) Start the forward from the cached layer-ℓ state of the corrupted run. This roughly halves the cost of layer sweeps.
- **DAS / pyvene.** Training a rotation per layer × variable is ≈ 0.5–2 GPU-h per (layer, variable). For ~10 layers × 3 variables × 3 models that is ≈ 50 GPU-h.
- **SAE on a VLM layer** (if training your own for Qwen2.5-VL): 100–300M tokens of activations, generated on the fly, ≈ 10–20 GPU-h per layer for a 16k–64k width TopK SAE. Doing 2–3 layers ≈ 2 GPU-days. This is optional; use Gemma Scope 2 on Gemma 3 instead.
- **Suggested 10-GPU-day budget (≈240 GPU-h).**

  | Item | GPU-h |
  |---|---|
  | Probing + lens, 5 models × 3 datasets | ~30 |
  | Attention knockout | ~10 |
  | Group patching + AtP\* + verification, 5 models | ~40 |
  | DAS | ~50 |
  | Repair / steering / late-layer readout fine-tunes | ~40 |
  | Gemma Scope 2 analysis | ~15 |
  | Reserve for reruns / seeds | ~55 |

- **Batching tips.**
  - Sort by image-token count, or fix the resolution so that clean and counterfactual pairs have identical token layouts. This is required for position-aligned patching in Qwen and InternVL.
  - Run clean and corrupt in the same batch.
  - Use `torch.inference_mode()` and save only hooked slices to CPU in fp16.
  - For long sweeps run in the background with checkpointed results per example (parquet).

### Gaps
- No published wall-clock benchmarks for VLM patching on an H200 were found. All figures need a quick calibration run (e.g., 50 examples) before committing the budget.

---

## Q5. Exemplar VLM interpretability papers to imitate (2024–2026)

### Takeaway
Imitate four papers:
- **Neo et al. (ICLR 2025)** for combining logit lens, ablation and knockout, with released code.
- **Zhang et al. (CVPR 2025)** for clean attention-knockout methodology on VQA.
- **Fu et al. (2025)** for the encoder-probe vs. VLM-output gap framing.
- **Salazar et al. (2026)** for pathway redundancy.

Use Zhang & Nanda and Heimersheim & Nanda as the methodological standard for patching, and Makelov et al. for subspace validation.

### Cited Findings
- Neo et al., ICLR 2025: LLaVA-1.5; logit lens, token ablation and attention knockout; object information is read out at the last position in mid–late layers — [arXiv 2410.07149](https://arxiv.org/abs/2410.07149) **[web-verified]**
- Zhang et al., CVPR 2025: attention knockout reveals a two-stage image→question→last flow — [arXiv 2411.18620](https://arxiv.org/abs/2411.18620) **[web-verified]**
- Fu et al., 2025: VLMs underperform their own vision encoders; the LLM fails to use visual features — [arXiv 2506.08008](https://arxiv.org/abs/2506.08008) **[web-verified]**
- Salazar et al., 2026: direct vs. text-mediated readout pathways, with fallback under ablation — [arXiv 2607.03358](https://arxiv.org/abs/2607.03358) **[web-verified]**
- "Same Task, Different Circuits" (NeurIPS 2025): visual back-patching closes about 1/3 of the modality gap — [arXiv 2506.09047](https://arxiv.org/abs/2506.09047) **[web-verified, snippet]**
- Yin et al., CVPR 2025: an alternative flow account — [CVPR OA](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_Lifting_the_Veil_on_Visual_Information_Flow_in_MLLMs_Unlocking_CVPR_2025_paper.pdf) **[web-verified, snippet]**
- Kaduri et al., CVPR 2025, "What's in the image?" — [alphaxiv 2411.17491](https://alphaxiv.org/abs/2411.17491) **[web-verified as existing]**
- Golovanevsky et al., NOTICE (NAACL 2025): semantic image pairs for VLM patching — [arXiv 2406.16320](https://arxiv.org/abs/2406.16320) **[prior-knowledge]**
- Basu et al., NeurIPS 2024: causal tracing in MLLMs — [arXiv 2406.04236](https://arxiv.org/abs/2406.04236) **[prior-knowledge]**
- Prisma toolkit paper (vision-side SAEs) — [arXiv 2504.19475](https://arxiv.org/abs/2504.19475) **[web-verified]**
- 2026 causal VLM papers to check and cite as concurrent work: [2605.13156](https://arxiv.org/pdf/2605.13156) (hallucination circuits) and [2606.28273](https://arxiv.org/pdf/2606.28273) (perception–prior conflict, last-token patching) **[web-verified, snippet]**

### Inferences
- **Novelty risk.** Fu et al. (2025) and Salazar et al. (2026) already argue that the LLM, not the encoder, is the bottleneck, and that readout happens late. The CVPR 2027 paper should be positioned as:
  - (i) layer-resolved *causal* localization of the failure (not only an encoder-probe gap);
  - (ii) on modern dynamic-resolution VLMs (Qwen2.5/3-VL, InternVL3, LLaVA-OV, Gemma 3);
  - (iii) with a repair intervention that closes the gap;
  - (iv) with rigorous controls (MC permutation, prompt templates, sink-aware knockout, counterfactual-image patching, bootstrap CIs).
- Read 2605.13156 and 2606.28273 in full before finalizing claims. They may overlap with "late-layer override".

### Gaps
- I could not retrieve OpenReview reviews for these papers in the tool budget, so reviewer-flagged weaknesses are not sourced.
- I did not verify whether any 2026 paper already makes the exact "late-layer readout, not perception" claim on Qwen2.5/3-VL with causal evidence. A targeted literature search is recommended.
