# Simple-but-breakthrough CV/ML papers (2020–2026): exemplars, compute, and reusable patterns

Legend for award status: **[V]** = verified this session against an official or primary page (linked). **[U]** = from prior knowledge, NOT verified this session; treat as unconfirmed (listed under Gaps where relevant).

## Q1. Which concrete papers won oral / award status for a simple idea or new perspective rather than scale?

### Takeaway
Award committees reward one of three things: a *phenomenon found and explained* (ViT registers, DVT, Eyes Wide Shut, memorization in diffusion), a *one-line change that fixes or reframes a standard pipeline* (Autoguidance, REPA, MeanFlow, DyT, SimSiam, RAFT), or a *training-free use of a pretrained model* (VisProg, FlowEdit, ChordEdit). Recent **best paper** wins at CVPR 2025/2026 (VGGT, D4RT) were large-compute 3D foundation models. Cheap insight papers win more often at the **oral / honorable mention / best student paper / ICLR-outstanding** tier.

### Cited Findings

**Catalog of exemplars (verified status)**

| # | Paper | Venue + status | One-sentence insight | Compute (if known) |
|---|---|---|---|---|
| 1 | Vision Transformers Need Registers (Darcet et al.) | ICLR 2024 **Outstanding Paper** [V] | Large ViTs repurpose low-information background patches as high-norm "scratchpad" tokens. Adding a few extra learnable [REG] tokens removes the artifacts. | Analysis is cheap. The fix needs re-training (DINOv2/CLIP/DeiT scale, so not 1 GPU) |
| 2 | Denoising Vision Transformers (Yang, Luo et al.) | ECCV 2024 **Oral** [V] | Grid-like artifacts in ViT features come from positional embeddings. They can be removed without re-training the ViT. | Training-free/light per-model denoiser |
| 3 | Eyes Wide Shut? Visual Shortcomings of MLLMs (Tong, Liu, Zhai, Ma, LeCun, Xie) | CVPR 2024 **Oral** [V] | "CLIP-blind pairs" (images CLIP embeds as similar but that look clearly different) give a 9-pattern benchmark (MMVP) on which GPT-4V fails. The fix is to mix in SSL features. | Benchmark construction is cheap. The MoF fix needs LLaVA-scale fine-tuning |
| 4 | Guiding a Diffusion Model with a Bad Version of Itself, "Autoguidance" (Karras et al.) | NeurIPS 2024 **Best Paper Runner-up** + Oral [V] | Guide with a smaller, less-trained copy of the model itself instead of the unconditional model. FID 1.01 (64²) and 1.25 (512²) using publicly available networks. | Not reported in sources found. The guide model is deliberately small/under-trained |
| 5 | REPA: Representation Alignment for Generation (Yu et al.) | ICLR 2025 **Oral** [V] | Add one regularizer that aligns DiT/SiT hidden states with frozen DINOv2 features. Result: >17.5x faster convergence (matches 7M-step SiT-XL in <400K steps), FID 1.42. | Multi-GPU ImageNet DiT training. The idea itself is a ~1-line loss |
| 6 | Mean Flows for One-step Generative Modeling (Geng, Deng, Bai, Kolter, He) | NeurIPS 2025 **Oral** [V] | Model *average* velocity over an interval instead of instantaneous velocity. An identity links the two. 1-NFE FID 3.43 on ImageNet-256 from scratch, with no distillation or curriculum. | ImageNet-scale training (multi-GPU) |
| 7 | Transformers without Normalization / DyT (Zhu, Chen, He, LeCun, Liu) | CVPR 2025, **accepted. Oral/highlight NOT confirmed**: the virtual site shows a poster page [V] | LayerNorm behaves like an S-shaped squash, so `tanh(αx)` can replace it as a drop-in. | Many experiments at scale. A single-model check is cheap |
| 8 | FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models (Kulikov et al.) | ICCV 2025 **Best Student Paper** [V] | Edit by building a direct ODE path between source and target distributions, with no inversion. | Training-free; 1 GPU |
| 9 | ChordEdit: One-Step Low-Energy Transport for Image Editing (Lu et al.) | CVPR 2026 **Oral + Best Paper Honorable Mention** [V] | Treat editing as dynamic optimal transport. A low-energy, variance-reduced editing field allows one-step, training-free, inversion-free editing. Reported 19x faster than FlowEdit, <½ the VRAM of SwiftEdit. | Training-free; 1 GPU |
| 10 | Visual Programming: Compositional Visual Reasoning Without Training (Gupta & Kembhavi) | CVPR 2023 **Best Paper** [V] | An LLM writes a modular program that calls off-the-shelf vision models. Zero training. | Training-free (LLM API + pretrained models) |
| 11 | Generative Image Dynamics (Z. Li et al.) | CVPR 2024 **Best Paper** [V] | Model scene motion as a spectral volume (Fourier-domain per-pixel trajectories) predicted by diffusion, a representation change. | Not found. Diffusion training, likely multi-GPU |
| 12 | Rich Human Feedback for Text-to-Image Generation (Liang et al.) | CVPR 2024 **Best Paper** [V] | Replace scalar preference scores with dense feedback: implausibility/misalignment heatmaps and keywords. Mainly a data/annotation-perspective contribution. | Data collection (~18K images, per prior knowledge [U]) plus a moderate model |
| 13 | Generalization in diffusion models arises from geometry-adaptive harmonic representations (Kadkhodaie et al.) | ICLR 2024 **Outstanding Paper** [V] | Shows when and why diffusion models switch from memorizing to generalizing, using inductive bias and harmonic analysis. | Small denoisers on face/image datasets (cheap) |
| 14 | Never Train from Scratch (Amos et al.) | ICLR 2024 **Outstanding Paper**. Committee praised its "systematic simplicity" [V] | Long-range benchmarks underestimate Transformers. Self-pretraining on the task data closes the gap. A *fair-comparison* insight. | LRA-scale, cheap |
| 15 | Is ImageNet worth 1 video? (Venkataramanan et al.) | ICLR 2024 **Honorable Mention** [V] | One long unlabeled walking video can pretrain a strong image encoder (new data type plus a method). | Moderate |
| 16 | Model Tells You What to Discard (FastGen, adaptive KV cache) | ICLR 2024 **Honorable Mention**. Committee: "simple method… without fine-tuning" [V] | Profile attention heads and compress the KV cache adaptively. Training-free. | Inference only |
| 17 | Why Diffusion Models Don't Memorize (Bonnaire et al.) | NeurIPS 2025 **Best Paper** [V] | Two training timescales. The memorization time grows with dataset size, so early stopping in a window generalizes. Supported by random-matrix theory. | Small-scale experiments plus theory |
| 18 | Does RL Really Incentivize Reasoning Beyond the Base Model? (Yue et al.) | NeurIPS 2025 **Runner-up**: "masterfully executed… negative finding" [V] | pass@k at large k shows that RLVR improves sampling efficiency but does not expand the reasoning boundary. | Evaluation-heavy, little training |
| 19 | Superposition Yields Robust Neural Scaling (Liu, Liu, Gore) | NeurIPS 2025 **Runner-up** [V] | A toy model shows representation superposition as the mechanism behind scaling laws, checked on open LLMs. | Toy model (cheap) |
| 20 | LLMs Get Lost in Multi-Turn Conversation (Laban et al.) | ICLR 2026 **Outstanding Paper** [V] | "Sharding" single-turn tasks into underspecified multi-turn conversations exposes a large reliability drop. Committee: "viable and scalable diagnosis". | Evaluation only (API calls) |
| 21 | RAFT: Recurrent All-Pairs Field Transforms (Teed & Deng) | ECCV 2020 **Best Paper** [V] | Build an all-pairs 4D correlation volume once and iteratively refine a single high-res flow field with a GRU, instead of coarse-to-fine. | **Trained on two 2080Ti GPUs**; 10 fps at 1088×436 on a 1080Ti [V] |
| 22 | NeRF (Mildenhall et al.) | ECCV 2020 **Best Paper Honorable Mention** [V] | Represent a scene as an MLP of (x, y, z, θ, φ) → (σ, rgb) with volume rendering and positional encoding. | Per-scene, single GPU [U: ~1–2 days on one V100] |
| 23 | Exploring Simple Siamese Representation Learning, SimSiam (Chen & He) | CVPR 2021 **Best Paper Honorable Mention** [V] | No negatives, momentum encoder or large batch needed. Stop-gradient alone prevents collapse. | [U: 8 GPUs, batch 256] |
| 24 | Unsupervised Learning of Probably Symmetric Deformable 3D Objects (Wu, Rupprecht, Vedaldi) | CVPR 2020 **Best Paper** [V] | Use a "probable symmetry" prior plus a confidence map to learn 3D from single-view images without supervision. | Modest [U] |
| 25 | On the Versatile Uses of Partial Distance Correlation in Deep Learning (Zhen et al.) | ECCV 2022 **Best Paper** [V] | Revive a classical statistic (partial distance correlation) to compare and disentangle network representations. | Cheap (analysis on pretrained nets) |
| 26 | Minimalist Vision with Freeform Pixels (Klotz & Nayar) | ECCV 2024 **Best Paper** [V] | Very few freeform (learned-shape) pixels are enough for a task. Rethinks the camera rather than the network. | Small (hardware plus tiny nets) |
| 27 | Concept Arithmetics for Circumventing Concept Inhibition in Diffusion Models (Petsiuk & Saenko) | ECCV 2024 **Honorable Mention** [V] | Concept-erased diffusion models can be bypassed with compositional prompt arithmetic. A negative/attack result. | Inference only |
| 28 | Comparing Decision-Making Mechanisms of Transformers and CNNs via Explanation Methods (M. Jiang et al.) | CVPR 2024 **Best Student Paper Honorable Mention** [V] | An analysis paper with no new model. | Cheap (analysis) |
| 29 | Neural Inverse Rendering from Propagating Light (Malik et al.) | CVPR 2025 **Best Student Paper** [V] | Uses time-resolved (transient) light, a new signal/perspective. | Modest [U] |
| 30 | Massive Activations in LLMs (Sun, Chen, Kolter, Liu) | COLM 2024 (also ICLR'24 ME-FoMo workshop) [V]. No award known | A few fixed, huge activations act as implicit bias terms and concentrate attention. Also shown in ViTs. Pure analysis. | Inference-only analysis |
| 31 | Position: The Platonic Representation Hypothesis (Huh, Cheung, Wang, Isola) | ICML 2024 **Oral** (position paper) [V]. **No best-paper award found** in search | Representations of different models and modalities converge toward a shared statistical model of reality. | Cheap (kernel alignment across existing models) |

Sources for the table, in row order:
- ICLR 2024 awards (rows 1, 13–16): [ICLR 2024 Outstanding Paper Awards blog](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/)
- DVT (row 2): [ECCV 2024 oral page](https://eccv2024.ecva.net/virtual/2024/oral/1634), [Cornell project page](https://www.cs.cornell.edu/~katieluo/publication/denoisingvit)
- Eyes Wide Shut (row 3): [CVPR 2024 oral page](https://cvpr.thecvf.com/virtual/2024/oral/32016), [arXiv 2401.06209](https://arxiv.org/html/2401.06209v2)
- Autoguidance (row 4): [NeurIPS 2024 awards blog](https://blog.neurips.cc/2024/12/10/announcing-the-neurips-2024-best-paper-awards/), [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5ee7ed60a7e8169012224dec5fe0d27f-Abstract.html)
- REPA (row 5): [ICLR 2025 oral page](https://iclr.cc/virtual/2025/oral/31896), [arXiv 2410.06940](https://export.arxiv.org/pdf/2410.06940)
- MeanFlow (row 6): [NeurIPS 2025 oral page](https://neurips.cc/virtual/2025/oral/130275), [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html)
- DyT (row 7): [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Transformers_without_Normalization_CVPR_2025_paper.html), [CVPR 2025 poster page](https://cvpr.thecvf.com/virtual/2025/poster/32739)
- CVPR/ICCV/ECCV awards (rows 8, 10–12, 21–29): [CVF Computer Vision Awards list](https://www.thecvf.com/?page_id=413), [PAMI-TC ICCV awards](https://tc.computer.org/tcpami/awards/iccv-paper-awards/)
- ChordEdit (row 9): [CVPR 2026 oral page](https://cvpr.thecvf.com/virtual/2026/oral/40352), [arXiv 2602.19083](https://arxiv.org/pdf/2602.19083), [CVF awards list](https://www.thecvf.com/?page_id=413)
- NeurIPS 2025 awards (rows 17–19): [NeurIPS 2025 awards blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)
- ICLR 2026 awards (row 20): [ICLR 2026 Outstanding Papers blog](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/)
- RAFT compute (row 21): [RAFT arXiv 2003.12039](https://arxiv.org/abs/2003.12039), where the text says "We train RAFT using two 2080Ti GPUs"
- Massive Activations (row 30): [arXiv 2402.17762](https://arxiv.org/html/2402.17762v2)
- Platonic Representation Hypothesis (row 31): [ICML 2024 oral page](https://icml.cc/virtual/2024/oral/35559)

**Committee language that signals what is rewarded**
- ViT Registers was praised as a model of research: "identify the problem, understand its cause, then solve it" (paraphrase of the committee text) — [ICLR 2024 blog](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/)
- Never Train from Scratch: cited for "execution and systematic simplicity" — [ICLR 2024 blog](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/)
- "Proving Test Set Contamination": a "simple, elegant method" — [ICLR 2024 blog](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/)
- Gated Attention (NeurIPS 2025 Best Paper): "main recommendation… easily implemented". The committee also states the work was "possible only with access to industrial scale computing resources." Simple idea, but **not** cheap — [NeurIPS 2025 blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)
- The RL-reasoning runner-up was called a "masterfully executed and critically important negative finding" — [NeurIPS 2025 blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)
- ICLR 2026's "LLMs Get Lost": the committee kept its conclusions relevant "despite concerns about dated models", which suggests a strong diagnostic can survive weaker models — [ICLR 2026 blog](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/)
- Polar Express (ICLR 2026 HM): "empirical improvements were sometimes modest" but the principled methodology was valued — [ICLR 2026 blog](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/)

**Counter-examples: top CV best papers that were *not* cheap**
- CVPR 2025 Best Paper went to VGGT. CVPR 2026 Best Paper went to D4RT (Google DeepMind/UCL/Oxford), with TRELLIS.2 as best student paper. SAM 3D and NitroGen (foundation models) received HMs — [CVF awards](https://www.thecvf.com/?page_id=413), [CVPR 2026 news](https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers)
- CVPR 2026 accepted 4,089 of 16,092 submissions — [CVPR 2026 news](https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers)
- ICCV 2025 Marr Prize: BrickGPT (text→physically stable brick structures). HMs: Spatially-Varying Autofocus, and RayZer. RayZer's category is listed inconsistently across sources — [CVF awards](https://www.thecvf.com/?page_id=413); [CMU news](https://icontrol.ri.cmu.edu/news/iccv25.html)

### Inferences
- At CVPR/ICCV, the **top** prize has drifted toward large 3D/4D foundation models (VGGT 2025, D4RT 2026). For a ≤8-GPU lab, the realistic targets are **oral, best student paper, and honorable mention** (FlowEdit, ChordEdit, DVT, Eyes Wide Shut) and the **ICLR outstanding** tier, where analysis papers win repeatedly (Registers 2024, Kadkhodaie 2024, Never-Train-from-Scratch 2024, LLMs-Get-Lost 2026).
- Training-free editing/guidance on top of pretrained flow/diffusion models has won CV awards two years in a row: FlowEdit (ICCV 2025 BSP) and then ChordEdit (CVPR 2026 HM, explicitly benchmarked against FlowEdit). This is a lineage where a better theoretical lens (optimal transport) on the same training-free setting got rewarded.

### Gaps
- Not verified this session (prior knowledge only, treat with caution):
  - "Are Emergent Abilities of LLMs a Mirage?": NeurIPS 2023 Outstanding [U]
  - EDM, "Elucidating the Design Space of Diffusion Models": NeurIPS 2022 Outstanding [U]
  - EDM2, "Analyzing and Improving the Training Dynamics of Diffusion Models": CVPR 2024 Oral [U]
  - Token Merging (ToMe): ICLR 2023 notable-top-5% [U]
  - SigLIP: ICCV 2023 Oral [U]
  - "Interpreting CLIP's Image Representation via Text-Based Decomposition": ICLR 2024 Oral [U]
  - "Vision Transformers Don't Need Trained Registers" (training-free follow-up): status not verified
  - DIFT ("Emergent Correspondence from Image Diffusion") and Diffusion Hyperfeatures: both NeurIPS 2023, oral/spotlight status not verified
  - "Classifier-Free Diffusion Guidance" (Ho & Salimans): believed to have appeared only as a NeurIPS 2021 *workshop* paper, which would make a famous simple idea that never had a main-venue award [U]
  - "Applying Guidance in a Limited Interval" (Kynkäänniemi et al.): NeurIPS 2024 [U, status not verified]
- ICLR 2025 outstanding-paper blog URL returned 404, so the ICLR 2025 award list was not verified.
- DyT presentation tier (poster vs highlight) is unconfirmed. Only a poster page was found.
- Compute for Generative Image Dynamics, Rich Human Feedback, and Autoguidance (guide-model training) was not found.

## Q2. What compute did these use, and which are reproducible on a single H100/H200?

### Takeaway
About half of the exemplars need **no training at all**: they analyse or reuse pretrained models (DVT, Eyes Wide Shut benchmark, FlowEdit, ChordEdit, VisProg, Massive Activations, Platonic, LLMs-Get-Lost, RL-reasoning pass@k). Those are reproducible on one H100. Methods that change the *training recipe* of generative models (REPA, MeanFlow, Autoguidance, Registers) used multi-GPU ImageNet-scale runs. Their core idea can still be shown on one GPU at reduced scale (CIFAR/ImageNet-64, small DiT).

### Cited Findings
- RAFT (ECCV 2020 Best Paper) was trained on two 2080Ti GPUs, about 1/10 of an H100 by modern standards — [RAFT arXiv](https://arxiv.org/abs/2003.12039)
- FlowEdit is training-free and uses pre-trained flow models (stated in the title) — [CVF awards](https://www.thecvf.com/?page_id=413)
- ChordEdit is "model agnostic, training-free, and inversion-free", needs <½ the VRAM of SwiftEdit, and is reported 19x faster than FlowEdit — [arXiv 2602.19083](https://arxiv.org/pdf/2602.19083)
- DVT "does not require re-training existing pre-trained ViTs" — [Cornell page](https://www.cs.cornell.edu/~katieluo/publication/denoisingvit)
- VisProg: "Compositional Visual Reasoning Without Training" (title) — [CVF awards](https://www.thecvf.com/?page_id=413)
- REPA reaches 7M-step SiT-XL quality in <400K steps. The speedup is in iterations, not a small absolute budget — [arXiv 2410.06940](https://export.arxiv.org/pdf/2410.06940)
- MeanFlow is trained from scratch with no pre-training or distillation, 1-NFE FID 3.43 — [NeurIPS 2025 oral](https://neurips.cc/virtual/2025/oral/130275)
- Autoguidance results use "publicly available networks", so the main contribution is applied to existing EDM2 checkpoints — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5ee7ed60a7e8169012224dec5fe0d27f-Abstract.html)
- Gated Attention is the counter-example: the committee explicitly notes industrial-scale compute — [NeurIPS 2025 blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)

### Inferences
Single-H100 feasibility, my assessment:

| Paper | Feasible on 1×H100? |
|---|---|
| DVT, Massive Activations, Platonic, Eyes Wide Shut (benchmark half), FlowEdit, ChordEdit, VisProg, Concept Arithmetics, Partial Distance Correlation, Kadkhodaie et al., RL-reasoning pass@k (small models) | **Yes** (inference/analysis) |
| RAFT, NeRF, SimSiam (ResNet-50/100ep is tight but feasible), Autoguidance (train a small guide on top of public EDM2), Never Train from Scratch | **Yes** (light training) |
| REPA, MeanFlow, DyT ablations (at small DiT/ImageNet-64 scale), ViT registers (fine-tune or small ViT), Why-Diffusion-Doesn't-Memorize (small data) | **Partially**: the idea can be demonstrated, but not at paper scale |
| VGGT, D4RT, TRELLIS.2, SAM 3D, Gated Attention, Rich Human Feedback (data collection) | **No** |

The most compute-efficient award formula is **analysis of public pretrained checkpoints, then a fix that is training-free or a small add-on.**

### Gaps
- GPU-hour figures for REPA, MeanFlow, Autoguidance, Registers, Generative Image Dynamics, and Eyes Wide Shut were not extracted from the papers this session.
- No single source tabulates compute for award papers. The feasibility column is my own judgement.

## Q3. What do these papers have in common in framing and writing?

### Takeaway
They state a **claim in the title**, show the phenomenon in **one teaser figure** that is visible without reading, give it a **memorable name**, offer a **fix that is a few lines**, and validate it **broadly across many models** rather than deeply on one.

### Cited Findings
- **Title as claim or provocation:** "Vision Transformers Need Registers", "Training Diffusion Transformers Is Easier Than You Think" (REPA subtitle), "Guiding a Diffusion Model with a Bad Version of Itself", "LLMs Get Lost in Multi-Turn Conversation", "Transformers are Inherently Succinct", "Why Diffusion Models Don't Memorize", "Does RL Really Incentivize Reasoning…?", "Never Train from Scratch", "Eyes Wide Shut?" — titles from [ICLR 2024 blog](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/), [ICLR 2026 blog](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/), [NeurIPS 2025 blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/), [REPA](https://iclr.cc/virtual/2025/oral/31896)
- **A named artifact or concept:** "CLIP-blind pairs" / MMVP (Eyes Wide Shut) — [arXiv](https://arxiv.org/html/2401.06209v2); "massive activations" — [arXiv](https://arxiv.org/html/2402.17762v2); "high-norm artifact tokens" — [ICLR 2024 blog](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/); "Artificial Hivemind" — [NeurIPS 2025 blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)
- **Headline number in the abstract:** ">17.5x faster" (REPA), "FID 1.01" (Autoguidance), "1-NFE FID 3.43" (MeanFlow), "19x faster than FlowEdit" (ChordEdit) — sources as linked above
- **Story arc of problem → cause → fix**, explicitly praised by the ICLR 2024 committee for Registers — [ICLR 2024 blog](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/)
- **Committees accept modest gains when the lens is principled** (Polar Express), and **dated models when the diagnostic is scalable** (LLMs Get Lost) — [ICLR 2026 blog](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/)

### Inferences
- A writing template that recurs across the exemplars:
  1. A teaser figure showing the phenomenon (e.g., attention maps with artifacts, or a failing VQA pair).
  2. A hypothesis about the cause, tested with a minimal intervention.
  3. A one-line fix.
  4. "Works across N models/datasets" as the breadth evidence.
  5. Limitations framed as open problems (Eyes Wide Shut ends with "still open").
- Generality across many public checkpoints is the substitute for scale. The claim is about a *class* of models, not one SOTA number.

### Gaps
- OpenReview reviewer comments were not fetched this session. Only committee citations are used here.

## Q4. What reusable patterns can be distilled?

### Takeaway
Eight recurring patterns cover nearly all the exemplars. The cheapest and most frequently rewarded are **P1 (hidden artifact found and fixed)**, **P3 (training-free reuse of a pretrained model with a principled lens)** and **P5 (benchmark or diagnostic exposing a failure)**.

### Cited Findings (pattern → exemplars, all sourced in Q1)
- **P1. Discover a hidden artifact or phenomenon, explain its cause, fix it simply.** ViT Registers (ICLR'24 Outstanding), DVT (ECCV'24 Oral), Massive Activations (COLM'24), Gated Attention's attention-sink removal (NeurIPS'25 Best Paper, but high compute). Sources: [ICLR 2024](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/), [ECCV oral](https://eccv2024.ecva.net/virtual/2024/oral/1634), [arXiv 2402.17762](https://arxiv.org/html/2402.17762v2), [NeurIPS 2025](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)
- **P2. Reframe the quantity being modeled or represented.**
  - MeanFlow: average instead of instantaneous velocity
  - Generative Image Dynamics: spectral volume instead of pixels
  - RAFT: all-pairs plus iterative refinement instead of coarse-to-fine
  - ChordEdit: editing as optimal transport
  - Rich Human Feedback: dense instead of scalar preference

  Sources: [NeurIPS oral](https://neurips.cc/virtual/2025/oral/130275), [CVF awards](https://www.thecvf.com/?page_id=413), [ChordEdit](https://arxiv.org/pdf/2602.19083)
- **P3. Training-free exploitation of pretrained models.** VisProg (CVPR'23 Best), FlowEdit (ICCV'25 BSP), ChordEdit (CVPR'26 HM), DVT, FastGen KV-cache (ICLR'24 HM). Sources: [CVF awards](https://www.thecvf.com/?page_id=413), [ICLR 2024](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/)
- **P4. Borrow a pretrained representation or a "bad self" as the training/inference signal.** REPA (align with DINOv2, ICLR'25 Oral) and Autoguidance (a weak copy as guide, NeurIPS'24 runner-up). Sources: [REPA](https://iclr.cc/virtual/2025/oral/31896), [NeurIPS 2024](https://blog.neurips.cc/2024/12/10/announcing-the-neurips-2024-best-paper-awards/)
- **P5. Build a simple benchmark or diagnostic that exposes a failure.**
  - Eyes Wide Shut / MMVP (CVPR'24 Oral)
  - LLMs Get Lost (ICLR'26 Outstanding)
  - Artificial Hivemind (NeurIPS'25 Best, D&B track)
  - Concept Arithmetics (ECCV'24 HM)

  Sources: [CVPR oral](https://cvpr.thecvf.com/virtual/2024/oral/32016), [ICLR 2026](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/), [NeurIPS 2025](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/), [CVF awards](https://www.thecvf.com/?page_id=413)
- **P6. A counter-intuitive negative or fair-comparison result.**
  - RL does not expand reasoning (NeurIPS'25 runner-up)
  - Never Train from Scratch (ICLR'24 Outstanding)
  - SimSiam: negatives and momentum encoder are unnecessary (CVPR'21 HM)

  Sources: [NeurIPS 2025](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/), [ICLR 2024](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/), [CVF awards](https://www.thecvf.com/?page_id=413)
- **P7. A theoretical lens that explains an empirical mystery, validated on a toy model.**
  - Kadkhodaie et al. (ICLR'24 Outstanding)
  - Why Diffusion Doesn't Memorize (NeurIPS'25 Best)
  - Superposition → scaling (NeurIPS'25 runner-up)
  - Platonic Representation Hypothesis (ICML'24 Oral)
  - Transformers are Inherently Succinct (ICLR'26 Outstanding)

  Sources: as linked in Q1
- **P8. Subtractive simplification: remove a component and show it was unnecessary.** DyT removes normalization (CVPR'25), SimSiam removes negatives, VisProg removes training. Sources: [CVF OA DyT](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Transformers_without_Normalization_CVPR_2025_paper.html), [CVF awards](https://www.thecvf.com/?page_id=413)
- **(Bonus) P9. A new sensing modality or minimal hardware perspective.** Minimalist Vision with Freeform Pixels (ECCV'24 Best), Neural Inverse Rendering from Propagating Light (CVPR'25 BSP), Spatially-Varying Autofocus (ICCV'25 HM). This is a cheap-compute route, but it needs optics/hardware. Source: [CVF awards](https://www.thecvf.com/?page_id=413)

### Inferences
- Patterns compound well. The strongest papers chain P1 → P8 or P1 → P3: find an artifact, explain it, then fix it by removing something or without training (Registers, DVT). P5 + P4 is the other strong chain: a benchmark exposes a failure, then pretrained features fix it (Eyes Wide Shut + MoF).
- Pattern lineages are visible across venues: Registers (ICLR'24) → DVT (ECCV'24) → training-free registers (2025), and FlowEdit (ICCV'25) → ChordEdit (CVPR'26). A cheap way to find a topic is to take a recent award paper and ask "can this be done training-free / explained theoretically / shown to fail on a new benchmark?"
- For a single-H100 CVPR 2027 submission, the highest expected-value combination is P1 or P5 on public foundation-model checkpoints, plus a training-free (P3) or tiny add-on fix, demonstrated across ≥5 models. The title should be a claim and the teaser should make the phenomenon visible.

### Gaps
- No quantitative study was found on what fraction of orals or awards used ≤8 GPUs. The pattern frequencies above are qualitative, drawn from this catalog.
- CVPR 2025/2026 oral lists were not systematically mined for cheap papers. Only award lists and individually checked papers are covered.
