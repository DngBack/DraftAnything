# Open Problems in Generative Models and 3D/4D Vision Feasible on 1x H200 (for CVPR 2027)

Notes compiled 2026-10-13. Method: about 16 web searches (abstracts, listings, and summaries; most full papers were NOT opened). A citation marked "(bg)" is a well-known paper cited from background knowledge and was not re-checked in this session. Every GPU-day estimate is my own inference, not a sourced number. Treat it as an order-of-magnitude figure for 1x H200 141GB with bf16, precomputed latents, and no unusual engineering.

---

## Q0. Shortlist: 14 concrete open research questions (gap, why it matters, angle, compute)

### Takeaway
The best openings for a small lab in late 2026 share a pattern. The phenomenon is documented in toy settings or by many patchwork fixes, but nobody has a unifying diagnostic or explanation tested on open-weight models. Examples: memorization, why REPA works, tokenizer "diffusability", CFG failure at low noise, few-step diversity collapse, the gap between what video models know about physics and what they render, and confidence calibration in feed-forward 3D. Analysis-plus-simple-fix papers on SiT-B-scale ImageNet-256 runs or on frozen Wan/FLUX/VGGT/π3 fit in 5–25 GPU-days.

### Cited Findings (organized per problem)

**P1. Where in the noise schedule, and at what data scale, do large diffusion/flow models memorize?**
- Gap: Most 2026 theory runs on toy models. ICLR 2026 analyzes a two-layer ReLU denoising autoencoder: memorization corresponds to "spiky", localized representations and generalization to balanced ones. The paper also proposes a detector and a training-free edit — [arXiv 2512.20963](https://arxiv.org/pdf/2512.20963.pdf), [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a5e3cf29c269b041ccd644b6beaf5c42-Abstract-Conference.html).
- Kaiser & Kollmann (Mar 2026) find overfitting concentrates at intermediate noise levels, which overlap little with the trajectories used at inference — [arXiv 2603.13419](https://arxiv.org/html/2603.13419v1). "Two Calm Ends and the Wild Middle" gives a geometric picture of the medium-noise regime — [arXiv 2602.17846](https://arxiv.org/abs/2602.17846v1).
- Ye et al. (ICLR 2026): implementing the empirical score needs network size that scales with sample size, and pruning reduces memorization in DiTs — [mlanthology](https://mlanthology.org/iclr/2026/ye2026iclr-provable/).
- Consistency distillation substantially reduces memorization inherited from the teacher — [Jiang & Zou, arXiv 2604.23552](https://www.alphaxiv.org/abs/2604.23552.md).
- Marion & Wu (May 2026) argue understanding diffusion "requires rethinking (again) generalization". I only found a secondary summary — [arXiv 2605.06077 summary](https://fugumt.com/fugumt/paper_check/2605.06077v1).
- Angle: produce a "memorization phase diagram" (training-set size N × noise level t × model size) for SiT on ImageNet subsets, then check whether the toy-model signatures (spiky representations, mid-noise overfitting) predict memorization in SD3.5/FLUX on known duplicated LAION prompts.
- Compute (inference): SiT-S/B on 5–6 subset sizes at roughly 1–2 GPU-days each, plus probing open T2I models. Total about 10–15 GPU-days.

**P2. Why does REPA work, and when does it hurt?**
- REPA aligns DiT hidden states to a frozen encoder. It narrows the representation gap across noise levels (better linear probes, higher CKNNA), and works with MAE/MoCo targets too — [REPA arXiv 2410.06940](https://arxiv.org/pdf/2410.06940).
- HASTE: REPA "works until it doesn't". Late in training, a capacity mismatch turns it from accelerator into brake (measured by gradient-direction similarity). Early-stopped holistic alignment claims a 28× speed-up over vanilla SiT-XL/2 — [arXiv 2505.16792](https://arxiv.org/pdf/2505.16792).
- iREPA (ICLR 2026): spatial structure, not global semantics, seems to drive the gains. A conv projector plus spatial normalization of the targets improves convergence across encoders — [arXiv 2512.10794](https://arxiv.org/pdf/2512.10794).
- REPA gives worse FID for pixel-space JiT as training proceeds (an "information asymmetry"), which PixelREPA fixes — [arXiv 2603.14366](https://arxiv.org/abs/2603.14366).
- Angle: build a predictive theory or metric that, from encoder/layer statistics measured before training (spatial self-similarity, effective rank, gradient conflict with the denoising loss at each t), predicts REPA's speed-up and the step where it turns into a brake. Validate across about 8 encoders (DINOv2/v3, SigLIP2, MAE, JEPA, etc.).
- Compute (inference): SiT-B/2 at 100–200K iterations on precomputed SD-VAE latents is about 1–1.5 GPU-days per run. 12–16 runs come to about 15–25 GPU-days. Avoid SiT-XL full runs (weeks).

**P3. Tokenizer–generator mismatch: is there a cheap "diffusability" metric?**
- Reconstruction vs. generation dilemma: higher-dimensional latents reconstruct better but need much larger DiTs and longer training. VA-VAE aligns latents to a vision foundation model (1.35 FID with LightningDiT) — [arXiv 2501.01423](https://arxiv.org/html/2501.01423v3).
- RecTok (CVPR 2026) says high-dimensional tokenizers still underperform low-dimensional ones, and moves semantics into the flow trajectory instead — [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/36717), [arXiv 2512.13421](https://arxiv.org/abs/2512.13421v1). VTP reports 1.11 gFID after scaling — [arXiv 2512.13687](https://arxiv.org/pdf/2512.13687v2). DINO-Tok addresses the texture–semantic trade-off — [arXiv 2511.20565](https://arxiv.org/pdf/2511.20565v2).
- RAE (ICLR 2026) replaces the VAE with frozen DINO/SigLIP/MAE encoders plus trained decoders, giving faster convergence — [arXiv 2510.11690](https://arxiv.org/abs/2510.11690v1). It has been scaled to T2I — [arXiv 2601.16208](https://arxiv.org/html/2601.16208v1). Pixel-space DiTs are a competing direction — [arXiv 2607.17585](https://arxiv.org/pdf/2607.17585).
- Angle: the field disagrees on where semantics should live (in the latent, in the trajectory, or by dropping the latent for pixels). Benchmark about 8 open tokenizers (SD-VAE, FLUX-VAE, VA-VAE, RAE-DINO, DC-AE, VTP, etc.) with candidate latent metrics (spectral decay, local smoothness or Lipschitzness of the decoder, linear-probe accuracy, noise-robustness of semantics). Correlate the metrics with the gFID of a fixed short SiT-B run.
- Compute (inference): 8 × roughly 1.5 GPU-days plus latent extraction comes to about 12–15 GPU-days. Training-free metric evaluation is cheap.

**P4. Guidance failure modes and principled guidance schedules for flow T2I/video models**
- "Guidance Breaks the Fitted Operator" (Jul 2026): at high CFG, guided updates violate DDIM's assumptions in the final low-noise steps. A no-cost repair wins 9/9 FID comparisons on CIFAR-10 but is not a universal improvement (KID sometimes prefers CFG) — [arXiv 2607.07665](https://arxiv.org/abs/2607.07665).
- Masked discrete diffusion (ICLR 2026): high early guidance hurts and CFG unmasks too fast early on. Fixed with a one-line change — [Sony AI](https://ai.sony/publications/Theory-Informed-Improvements-to-Classifier-Free-Guidance-for-Discrete-Diffusion-Models).
- Limited-interval guidance proposes exposing the guidance interval as a hyperparameter in all models — [review of arXiv 2404.07724](https://liner.com/review/applying-guidance-in-limited-interval-improves-sample-and-distribution-quality).
- A CVPR 2026 paper guides a DiT with its own internal dynamics, giving LightningDiT + IG FID 1.24 without CFG — [CVPR 2026 supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhou_Guiding_a_Diffusion_CVPR_2026_supplemental.pdf).
- Angle (training-free): measure the per-step conditional–unconditional divergence and off-manifold drift (for example, decoder-reconstruction error of x̂0) in FLUX/SD3.5/Wan2.1. Derive a per-sample adaptive guidance schedule from it. Test whether the low-noise failure seen on CIFAR transfers to rectified-flow samplers at scale.
- Compute (inference): inference only, about 3–8 GPU-days for FLUX/SD3.5 sweeps plus a Wan-1.3B subset.

**P5. Diversity/mode collapse in few-step generators (DMD, MeanFlow), and how to measure it**
- DMD's reverse-KL objective is mode-seeking. DP-DMD adds a flow-matching diversity loss on the first step — [arXiv 2602.03139](https://arxiv.org/pdf/2602.03139). 1.x-Distill: DMD loses diversity and fidelity at 2 steps or fewer — [arXiv 2604.04018](https://www.alphaxiv.org/overview/2604.04018).
- SubFlow names MeanFlow among the few-step families with "severe diversity degradation" and blames averaging distortion in the MSE objective, not distillation — [arXiv 2604.12273](https://www.alphaxiv.org/abs/2604.12273).
- In autoregressive video, collapse starts in the first chunk and spreads through the KV cache — [Uncertainty DMD, arXiv 2609.11265](https://arxiv.org/pdf/2609.11265). CrossDistill splits the trajectory by noise level — [arXiv 2609.14725](https://www.alphaxiv.org/abs/2609.14725). Data-Forcing Distillation (ICML 2026) — [ICML](https://icml.cc/virtual/2026/70547).
- My search found no paper that directly measures MeanFlow's mode coverage (see Gaps).
- Angle: a controlled study on ImageNet-256 that measures per-class recall/coverage/Vendi for MeanFlow, consistency, and DMD students distilled from the same SiT teacher. Locate which noise interval or which component causes the collapse.
- Compute (inference): MeanFlow-B/2 or consistency students on ImageNet-256 at about 2–4 GPU-days each. About 10–15 GPU-days in total.

**P6. Video diffusion models "know more physics than they show": can internal probes steer generation?**
- "The Invisible Hand of Physics" (Jun 2026) inverts real videos and linearly probes DiT states, getting about 81% average accuracy on IntPhys/InfLevel and beating V-JEPA/VideoMAE in this setup — [arXiv 2606.05328](https://arxiv.org/pdf/2606.05328).
- "Seeking Physics in Diffusion Noise": in CogVideoX-2B, plausible and implausible videos are partly separable in mid-layer features even at high noise. A light verifier reaches AUC about 0.68 — [arXiv 2603.14294](https://arxiv.org/html/2603.14294v2).
- A layerwise study finds V-JEPA leads, with LTX-Video weaker but non-trivial — [arXiv 2606.09646](https://arxiv.org/pdf/2606.09646). Physics representations are distributed and task-specific — [arXiv 2602.07050](https://arxiv.org/pdf/2602.07050).
- Angle: close the "know vs. show" gap. Use the probe as (a) an early-step verifier for best-of-N or noise re-selection (cheap, because the signal exists at high noise), and (b) a gradient or activation-steering direction. Evaluate on Physics-IQ (bg: [arXiv 2501.09038](https://arxiv.org/abs/2501.09038)) and VideoPhy-2 (bg: [arXiv 2503.06800](https://arxiv.org/abs/2503.06800)).
- Compute (inference): Wan2.1-1.3B or CogVideoX-2B inference plus linear probes. About 5–15 GPU-days. Wan2.1-14B is possible on 141GB, but generation is slow, so limit it to the final evaluation.

**P7. Object permanence and occlusion memory in video generators**
- MemoBench covers ten world-generation models and finds that none reliably keeps object memory across occlusion — [HF 2606.27537](https://huggingface.co/papers/2606.27537.md).
- WorldBench: all tested models lack physical consistency, including object permanence and scale/perspective — [arXiv 2601.21282](https://arxiv.org/html/2601.21282v1).
- "Current World Models Lack a Persistent State Core" — [alphaXiv 2606.20545](https://dev.alphaxiv.org/audio/2606.20545). What-If World lists morphing, teleportation, and impossible kinematics — [arXiv 2605.27589](https://arxiv.org/pdf/2605.27589).
- Angle (mechanistic, training-free): trace where an occluded object's identity disappears in the DiT's 3D attention (which layers and timesteps attend to the pre-occlusion frames). Then test a training-free fix such as attention or KV re-injection of the object's pre-occlusion tokens. This is a diagnosis-plus-fix paper rather than yet another benchmark.
- Compute (inference): about 5–10 GPU-days on Wan-1.3B/CogVideoX-2B. Attention-map extraction across 30–50 steps is memory-heavy but fits in 141GB.

**P8. A trustworthy, attributable physics-violation metric (VLM judges fail)**
- PhysicalRealismBench: even state-of-the-art VLMs miss basic violations such as objects vanishing or moving without cause. It cites Physion-Eval (Mar 2026), which found that most generated videos contain a human-identifiable glitch — [Reka blog](https://reka.ai/rekalabs/blogs/physicalrealismbench-attributable-physical-realism-evaluation-for-video-world-models).
- Angle: a cross-area metric that runs feed-forward 4D reconstruction (VGGT/π3 plus a point tracker) on generated videos and flags geometric violations: object teleport, metric scale change, non-rigid drift of rigid objects, and static-background inconsistency. Validate against human labels from MemoBench/Physics-IQ. This combines the 3D and generative halves of the brief.
- Compute (inference): inference only, about 3–6 GPU-days.

**P9. Inference-time search for diffusion: verifier hacking and compute allocation**
- Random search can overfit to verifier biases (verifier hacking) — [arXiv 2511.07362](https://arxiv.org/pdf/2511.07362). A single verifier can improve its own metric while degrading held-out ones (Ma et al., as cited in CachedSearch) — [arXiv 2607.23159](https://arxiv.org/pdf/2607.23159). Ma et al. original (bg): [arXiv 2501.09732](https://arxiv.org/abs/2501.09732).
- Optimizing one metric can hurt others; ImageReward and LLaVA-OneVision hack the least on compositional tasks — [arXiv 2510.16751](https://arxiv.org/pdf/2510.16751). Search samples can act as adversarial examples against verifiers — [arXiv 2505.23614](https://arxiv.org/pdf/2505.23614).
- EvoSearch makes Wan-1.3B competitive with Wan-14B, and S3 allocates compute with look-ahead — [CachedSearch related work](https://arxiv.org/pdf/2607.23159).
- Angle: a scaling-law-style study of the "hacking onset", meaning the point where held-out metrics peak as search compute grows, together with a cheap early-stopping or ensemble-disagreement criterion. Caution: the methods space is crowded (see Q4).
- Compute (inference): FLUX-schnell/SD3.5-medium search sweeps, about 5–10 GPU-days.

**P10. Feed-forward 3D on dynamic scenes: why does VGGT ignore dynamic content?**
- PAGE-4D: VGGT's depth error is 94% higher in dynamic regions than static ones on Odyssey, and it shows weaker activations there — [arXiv 2510.17568](https://arxiv.org/pdf/2510.17568).
- VGGT4D: motion is entangled with camera pose and corrupts background geometry. A training-free decoupling mines motion cues from VGGT attention — [arXiv 2511.19971](https://arxiv.org/pdf/2511.19971).
- VGGT-Ω (CVPR 2026) adds register attention and dynamic support — [CVPR 2026 supp](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_VGGT-ohm_CVPR_2026_supplemental.pdf). VGGT base (bg): [arXiv 2503.11651](https://arxiv.org/abs/2503.11651). π3 (bg): [arXiv 2507.13347](https://arxiv.org/abs/2507.13347).
- Angle: fix-papers are crowding in, so the open part is understanding. Is the dynamic failure caused by the training data (mostly static), by the first-frame reference design (VGGT) vs. permutation-equivariance (π3), or by the global attention averaging? A controlled probing and ablation study across VGGT, π3, and MapAnything.
- Compute (inference): inference plus light fine-tunes, about 5–10 GPU-days.

**P11. Calibrated uncertainty and metric-scale reliability in feed-forward 3D**
- Trust3R (May 2026): feed-forward confidence scores are "heuristic, lack probabilistic interpretation". An evidential Normal-Inverse-Wishart head gives a Student-t per point and better risk–coverage on ScanNet++/KITTI — [emergentmind 2605.19539](https://www.emergentmind.com/papers/2605.19539).
- "A Calibration Audit of Confidence in Feed-Forward 3D Reconstruction" (Deng, Cheng, Zeller, Cremers; ~Aug/Sep 2026) covers 7 backbones on 13 datasets. Confidence ranks error well, but predicted uncertainty is too low off-distribution. Only seen via a third-party index — [roboticscenter](https://www.roboticscenter.ai/research/papers/a-calibration-audit-of-confidence-in-feed-forward-3d-reconstruction-2608).
- MapAnything (3DV 2026) predicts metric reconstructions with optional poses, intrinsics, or depth inputs — [arXiv 2509.13414](https://arxiv.org/abs/2509.13414v3).
- Angle: given the audit's finding (good ranking, bad dispersion), test a post-hoc conformal or temperature-scaled calibration that transfers across datasets. Show it pays off downstream: confidence-gated 3DGS initialization, view selection, or metric-scale failure detection. No retraining of the backbone.
- Compute (inference): about 3–8 GPU-days.

**P12. Pose-free feed-forward 3DGS: pose/geometry entanglement and view-count scaling**
- YoNoSplat: joint pose and Gaussian learning is "highly entangled", self-forcing with predicted poses trains unstably, and earlier pose-free methods work with 2–4 views but do not scale to more — [arXiv 2511.07321](https://arxiv.org/html/2511.07321v1).
- Unfavorable or off-center views break models — [UFV-Splatter arXiv 2507.22342](https://arxiv.org/html/2507.22342v2). Static primitives blur view-dependent detail — [ViewSplat arXiv 2603.25265](https://www.alphaxiv.org/abs/2603.25265.md).
- Angle: a simple decoupling that freezes a π3/VGGT backbone for poses, trains only a Gaussian head, and uses an explicit pose-noise curriculum. Quantify how much NVS error comes from pose error vs. Gaussian prediction error as the view count grows from 2 to 32.
- Compute (inference): Gaussian-head training on RealEstate10K/DL3DV at 256px is about 7–15 GPU-days.

**P13. Long-horizon drift and memory in interactive video world models**
- AlayaWorld (15B, Jul 2026) fights drift with training on corrupted histories, an error bank, a sink frame, and geometry-aligned spatial memory — [arXiv 2607.18367](https://arxiv.org/pdf/2607.18367). Infinite-World claims 1000+ frame coherence via a recursive memory compressor — [ICML 2026](https://icml.cc/virtual/2026/poster/66617). WorldPlay uses Context Forcing — [gamedev.net](https://gamedev.net/news/3858/).
- PlayWorld argues fixed action-conditioned tests do not support fair cross-model comparison — [emergentmind 2608.13552](https://www.emergentmind.com/papers/2608.13552). WorldScore is an earlier unified benchmark (bg: [arXiv 2504.00983](https://arxiv.org/abs/2504.00983)).
- Angle: measure drift as 3D inconsistency on revisits (reproject with VGGT/π3 when the camera returns to a pose; this is a loop-closure error). Analyze which memory designs (sink, compressed, or spatial) actually reduce it. Training-free retrieval of revisited-pose frames is a cheap baseline.
- Compute (inference): inference on open world models (Wan-1.3B-based ones, or AlayaWorld if it runs on one GPU), about 8–15 GPU-days. Training a world model is out of scope.

**P14. Exposure bias in scale-wise autoregressive (VAR-style) generation**
- Scale-wise AR suffers exposure bias from a train–test mismatch and uneven per-scale difficulty. Self-Autoregressive Refinement cuts FID by 5.2% on FlexVAR-d16 within 10 epochs of post-training — [arXiv 2512.06421](https://arxiv.org/html/2512.06421v1).
- xAR: the optimal token definition for 2D image structure "remains an open question" — [arXiv 2502.20388](https://arxiv.org/pdf/2502.20388v1).
- VAR models beat diffusion on inference-time scaling, which makes them a good place to test search — [arXiv 2510.16751](https://arxiv.org/pdf/2510.16751).
- Angle: per-scale measurement of train–test distribution shift and error amplification, and which scales matter, followed by a scale-targeted fix. My search found no 2026 paper (see Gaps).
- Compute (inference): VAR/FlexVAR-d16 post-training at about 2–5 GPU-days per run.

### Inferences
- The highest-value, lowest-risk picks for 1 H200 are P2, P3, P5 (ImageNet-256 controlled studies), P4, P6, P8, P11 (training-free or inference-only), and P7 (mechanistic plus training-free fix). P8, which uses 4D reconstruction to judge generated video, is the most distinctive because it bridges both halves of the brief.
- Each pick should be framed as "explain plus predictive metric plus one simple fix". The density of 2025–2026 fix-papers (REPA variants, DMD diversity fixes, VGGT-4D variants, search methods) shows that a bare fix is no longer novel. A falsifiable explanation that unifies them is.
- Budget heuristic: SiT-B/2 at 400K iterations takes about 2–4 H200-days. SiT-XL/2 at 4M iterations is out of reach. 1 H200 gives about 30 GPU-days per month, so a CVPR 2027 paper (deadline likely Nov 2026, i.e. about 4–6 weeks) can afford roughly 25–40 GPU-days in total. Pick a direction whose core experiment fits in about 15 GPU-days.

### Gaps
- Exact single-GPU wall-clock figures for SiT-B/MeanFlow/VAR on H200 were not found. All GPU-day numbers are my estimates. The closest sourced small baseline is SR-DiT, a 140M model at FID 3.49 without CFG after 400K iterations, with no GPU-time stated — [arXiv 2512.12386](https://arxiv.org/abs/2512.12386v1).
- I found no paper measuring MeanFlow's mode coverage (P5) and no 2026 paper on VAR exposure bias (P14). These may be open, or my searches may have missed them.
- The Calibration Audit (P11) and Marion & Wu (P1) were seen only via third-party indexes. Verify on arXiv before citing.

---

## Q1. What diffusion/flow phenomena remain poorly understood?

### Takeaway
Memorization vs. generalization (noise-level-resolved), the mechanism behind REPA and when it hurts, tokenizer "diffusability", CFG failure at low noise, and diversity loss in few-step models are all actively debated in 2026, with competing explanations.

### Cited Findings
- Memorization: spiky vs. balanced representations — [arXiv 2512.20963](https://arxiv.org/pdf/2512.20963.pdf). Mid-noise overfitting — [arXiv 2603.13419](https://arxiv.org/html/2603.13419v1). Network size must scale with N to memorize — [Ye et al. ICLR 2026](https://mlanthology.org/iclr/2026/ye2026iclr-provable/). Associative-memory view — [arXiv 2505.21777](https://arxiv.org/abs/2505.21777v3).
- REPA: an early accelerator that later becomes a brake — [HASTE](https://arxiv.org/pdf/2505.16792). Spatial structure is what matters — [iREPA](https://arxiv.org/pdf/2512.10794). Hurts in pixel space — [PixelREPA](https://arxiv.org/abs/2603.14366).
- Tokenizers: unresolved where semantics should live — [VA-VAE](https://arxiv.org/html/2501.01423v3), [RecTok](https://cvpr.thecvf.com/virtual/2026/poster/36717), [RAE](https://arxiv.org/abs/2510.11690v1), [Pixel-space DiTs](https://arxiv.org/pdf/2607.17585).
- Guidance: CFG breaks the sampler's fitted operator at low noise — [arXiv 2607.07665](https://arxiv.org/abs/2607.07665).
- Few-step: diversity collapse in DMD and MeanFlow — [DP-DMD](https://arxiv.org/pdf/2602.03139), [SubFlow](https://www.alphaxiv.org/abs/2604.12273).
- Video physics: a gap between internal knowledge and output — [arXiv 2606.05328](https://arxiv.org/pdf/2606.05328), [arXiv 2603.14294](https://arxiv.org/html/2603.14294v2).

### Inferences
- A common thread: many phenomena are noise-level dependent (memorization in the mid noise range, REPA's benefit early in training, CFG failure at low noise, physics signal at high noise). A "per-timestep" analysis lens is a reusable methodology across several papers.

### Gaps
- No single 2026 paper unifies these noise-level findings. That is an opportunity, but I could not confirm nobody has done it.

---

## Q2. Known failure modes of video generators and feed-forward 3D models, and the benchmarks that expose them

### Takeaway
Video models fail at object permanence under occlusion, causal/interventional consistency, and basic physics, and VLM judges cannot reliably detect these failures. Feed-forward 3D models fail on dynamic content (pose–motion entanglement), give under-dispersed confidence off-distribution, and struggle with pose-free many-view scaling.

### Cited Findings
- Video: MemoBench (no model keeps object memory across occlusion) — [HF](https://huggingface.co/papers/2606.27537.md). WorldBench — [arXiv 2601.21282](https://arxiv.org/html/2601.21282v1). What-If World (2,871 instances; morphing, teleportation) — [arXiv 2605.27589](https://arxiv.org/pdf/2605.27589). PhysicalRealismBench and Physion-Eval (VLM judges miss violations) — [Reka](https://reka.ai/rekalabs/blogs/physicalrealismbench-attributable-physical-realism-evaluation-for-video-world-models). Persistent-state critique — [alphaXiv 2606.20545](https://dev.alphaxiv.org/audio/2606.20545). Older (bg): Physics-IQ [arXiv 2501.09038](https://arxiv.org/abs/2501.09038), VideoPhy-2 [arXiv 2503.06800](https://arxiv.org/abs/2503.06800), WorldScore [arXiv 2504.00983](https://arxiv.org/abs/2504.00983), "How far is video generation from world model" [arXiv 2411.02385](https://arxiv.org/abs/2411.02385).
- Few-step and AR video: collapse propagates from the first chunk through the KV cache — [arXiv 2609.11265](https://arxiv.org/pdf/2609.11265).
- 3D: 94% higher depth error in dynamic regions — [PAGE-4D](https://arxiv.org/pdf/2510.17568). Motion–pose entanglement — [VGGT4D](https://arxiv.org/pdf/2511.19971). Object duplication and hallucination when VGGT feeds spatial reasoning — [ESI-Bench arXiv 2605.18746](https://arxiv.org/pdf/2605.18746). Confidence under-dispersed OOD — [Calibration Audit](https://www.roboticscenter.ai/research/papers/a-calibration-audit-of-confidence-in-feed-forward-3d-reconstruction-2608), [Trust3R](https://www.emergentmind.com/papers/2605.19539). Pose-free 3DGS fails to scale past 2–4 views — [YoNoSplat](https://arxiv.org/html/2511.07321v1).
- World models: long-horizon drift is the central problem being engineered around — [AlayaWorld](https://arxiv.org/pdf/2607.18367), [Infinite-World](https://icml.cc/virtual/2026/poster/66617). Evaluation is contested — [PlayWorld](https://www.emergentmind.com/papers/2608.13552).

### Inferences
- There are now so many physics/world-model benchmarks (at least 6 in 2026 alone) that a new benchmark paper is a weak bet. A metric validated against these benchmarks' human labels (P8) or a mechanistic diagnosis (P6/P7) has more room.

### Gaps
- Per-model numbers for MemoBench/WorldBench were not retrieved.

---

## Q3. Training-free or analysis-driven work recognized recently (2025–2026)

### Takeaway
Top awards in 2026 went to large-scale 3D generation systems, not training-free work. Analysis-driven papers do appear at ICLR 2026 and CVPR 2026 (iREPA, memorization theory, RecTok, RAE), and training-free 4D extensions of VGGT are common. I could not retrieve the oral/highlight lists.

### Cited Findings
- CVPR 2026 Best Paper: TRELLIS.2, "Native and Compact Structured Latents for 3D Generation". Sources conflict on whether it was Best Paper or Best Student Paper — [Newswise press release](https://www.newswise.com/articles/cvpr-2026-honors-the-year-s-most-innovative-computer-vision-and-ai-research), [BasicAI](https://www.basic.ai/blog-post/cvpr-2026-top-papers-award-winners-and-notable-works).
- CVPR 2026 Honorable Mention: SAM 3D (Meta) — [Newswise](https://www.newswise.com/articles/cvpr-2026-honors-the-year-s-most-innovative-computer-vision-and-ai-research).
- ICLR 2026 analysis papers: iREPA — [ICLR poster](https://iclr.cc/virtual/2026/poster/10006607). Balanced-representation memorization — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a5e3cf29c269b041ccd644b6beaf5c42-Abstract-Conference.html). Provable separations — [mlanthology](https://mlanthology.org/iclr/2026/ye2026iclr-provable/). RAE — [arXiv 2510.11690](https://arxiv.org/abs/2510.11690v1).
- Training-free 3D/4D: VGGT4D — [arXiv 2511.19971](https://arxiv.org/pdf/2511.19971).
- Training-free video: CachedSearch (test-time search for video diffusion) — [arXiv 2607.23159](https://arxiv.org/pdf/2607.23159).
- Background (bg): VGGT received the CVPR 2025 Best Paper — [arXiv 2503.11651](https://arxiv.org/abs/2503.11651).

### Inferences
- For a 1-GPU lab, the realistic target is a strong poster or highlight with a crisp analysis finding, not an award. Analysis papers that land well give a falsifiable claim plus a simple fix that transfers across several open models.

### Gaps
- I did not retrieve the CVPR 2026 award-candidate or oral list (official page: cvpr.thecvf.com/virtual/2026/events/AwardCandidates2026), the ICCV 2025 orals, or the NeurIPS 2025 orals. Whether specific training-free papers got orals is unverified.

---

## Q4. Small-lab-friendly directions vs. saturated ones

### Takeaway
Small-lab friendly: controlled ImageNet-256 SiT-B studies (REPA, tokenizers, memorization, few-step diversity), probing or steering frozen open models (Wan-1.3B, CogVideoX-2B, FLUX, VGGT/π3), and post-hoc calibration. Saturated: new physics/world-model benchmarks, yet another DMD diversity fix, yet another training-free VGGT-to-4D extension, and generic verifier-search methods.

### Cited Findings
- Small-lab enablers: SR-DiT (140M, FID 3.49 without CFG at 400K iterations) is billed as a "computationally accessible baseline" — [arXiv 2512.12386](https://arxiv.org/abs/2512.12386v1). HASTE gives a 28× speed-up vs. SiT-XL — [arXiv 2505.16792](https://arxiv.org/pdf/2505.16792). Self-AR refinement shows gains in 10 epochs of post-training — [arXiv 2512.06421](https://arxiv.org/html/2512.06421v1). Data-Forcing restores diversity with 100–300 fine-tuning steps — [ICML 2026](https://icml.cc/virtual/2026/70547). Wan2.1 ships a 1.3B T2V model (bg: [arXiv 2503.20314](https://arxiv.org/abs/2503.20314)).
- Crowding evidence:
  - Physics/world benchmarks: WorldBench, MemoBench, What-If World, PhysicalRealismBench, Physion-Eval, and PlayWorld all appeared in 2026 — [WorldBench](https://arxiv.org/html/2601.21282v1), [MemoBench](https://huggingface.co/papers/2606.27537.md), [What-If World](https://arxiv.org/pdf/2605.27589), [Reka](https://reka.ai/rekalabs/blogs/physicalrealismbench-attributable-physical-realism-evaluation-for-video-world-models), [PlayWorld](https://www.emergentmind.com/papers/2608.13552).
  - DMD diversity fixes: DP-DMD, 1.x-Distill, Uncertainty DMD, CrossDistill, Data-Forcing, and SubFlow all date from 2026 — see P5 links.
  - VGGT-to-dynamic: PAGE-4D, VGGT4D, VGGT-Ω, and others — see P10 links.
  - Diffusion search: EvoSearch, S3, CachedSearch, classical search — [arXiv 2607.23159](https://arxiv.org/pdf/2607.23159), [arXiv 2505.23614](https://arxiv.org/pdf/2505.23614).
- Infeasible on 1 GPU: training world models at the 15B scale — [AlayaWorld](https://arxiv.org/pdf/2607.18367). Large structured-latent 3D generators — [TRELLIS.2 award](https://www.newswise.com/articles/cvpr-2026-honors-the-year-s-most-innovative-computer-vision-and-ai-research).

### Inferences
- Avoid the following, based on publication density and compute, not on a source that labels them saturated:
  - Standalone benchmarks with a VLM judge.
  - Prompt-to-prompt-style attention-editing variants.
  - SDS text-to-3D.
  - Per-scene 3DGS micro-improvements (compression, anti-aliasing).
  - VBench leaderboard chasing.
  - Training any video or world model from scratch.
  - Full-scale SiT-XL or ImageNet-512 SOTA chasing.
- Feasible on 1 H200 (141GB): Wan2.1-14B and HunyuanVideo-scale inference fit in memory (slow but workable for evaluation). Analysis work should use the 1.3B–2B video models. VGGT/π3/MapAnything inference and light head fine-tunes are cheap.
- Suggested first pick, by risk-adjusted value: P8 (4D-reconstruction-based physics-violation metric) or P6 (probe-steered physics sampling) for a bridge between generation and 3D. P2 or P3 for a clean ImageNet-256 analysis paper. P11 as a low-risk, training-free 3D paper.

### Gaps
- No source explicitly labels these areas "saturated". That judgment is my inference from how many 2026 papers I found. Check arXiv volume for each before committing.
- The CVPR 2027 deadline was not verified (I assume about mid-November 2026, following past cycles).
