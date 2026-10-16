# Old idea, new framing: the 2024–2026 revival pattern in ML/vision, and old CV ideas ripe for CVPR 2027

Research date: 2026-10-13. CVPR 2027 deadline: 2026-11-16. OpenReview reviews were pulled through the public OpenReview API search endpoint (`api2.openreview.net/notes/search`), because the forum endpoint returned a challenge (403). Novelty checks used the arXiv API (`export.arxiv.org/api/query`, abstract field, run on 2026-10-13) plus web search.
Note on citations of the *original* old papers (e.g., "LSTM 1997", "Lucas–Kanade 1981"): these are standard bibliography from background knowledge. They were not re-fetched in this session and are marked [bg].

---

## Q1. Catalogue: 2024–2026 papers whose core idea was old but whose reframing was influential

### Takeaway
At least 22 high-visibility 2024–2026 papers revive a decades-old idea: energy-based models, x-prediction denoising, normalizing flows, LSTMs/RNNs, Kolmogorov–Arnold, cellular automata, coarse-to-fine pyramids, gating, unrolled iterative refinement, Hopfield memory, and classical multi-view geometry. The ones that landed (best paper or oral) changed one thing, showed it at a scale or on a benchmark people already cared about, and named the old idea in a provocative title. Several revivals were rejected or deflated: minLSTM, Equilibrium Matching, and HRM.

### Cited Findings

**Generative modeling**
1. **JiT, "Back to Basics: Let Denoising Generative Models Denoise"** (Li & He, MIT).
   - Old idea: predict the clean image, as classical denoising autoencoders did [bg: Vincent et al. 2008].
   - Framing: modern diffusion models "do not denoise" because they predict noise or noised quantities. By the manifold assumption, clean images live on a low-dimensional manifold, so x-prediction lets an under-capacity plain ViT work on raw pixel patches of size 16/32, with no tokenizer, no pre-training, and no extra loss.
   - Venue: CVPR 2026. Oral status was not found on the oral pages I could read. — [arXiv 2511.13720](https://arxiv.org/abs/2511.13720); [CVPR 2026 open access](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Back_to_Basics_Let_Denoising_Generative_Models_Denoise_CVPR_2026_paper.html)
2. **MeanFlow** (Geng, Deng, Bai, Kolter, He).
   - Idea: model the *average* velocity instead of the instantaneous velocity, using an identity that links the two.
   - Results: 1-NFE FID 3.43 on ImageNet 256 from scratch, with no distillation or curriculum.
   - Venue: **NeurIPS 2025 oral**. — [NeurIPS oral page](https://neurips.cc/virtual/2025/oral/115488); [proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html); [OpenReview uWj4s7rMnR, "NeurIPS 2025 oral"](https://openreview.net/forum?id=uWj4s7rMnR)
3. **Equilibrium Matching (EqM)** (Wang & Du).
   - Old idea: energy-based models.
   - Framing: replace time-conditioned flow with a single time-invariant equilibrium gradient field over an implicit energy landscape, so that sampling becomes gradient descent with adaptive step sizes and adaptive compute.
   - Result: FID 1.90 on ImageNet 256. — [arXiv 2510.02300](https://arxiv.org/abs/2510.02300)
   - **Not accepted at ICLR 2026**: OpenReview lists it as "Submitted to ICLR 2026", with reviewer ratings 4/8/4/2. — [OpenReview xqlxtkOhDX](https://openreview.net/forum?id=xqlxtkOhDX)
4. **Energy-Based Transformers (EBT)** (Gladstone et al.).
   - Old idea: EBMs [bg: LeCun et al. 2006].
   - Framing: "System 2 thinking from unsupervised learning". The model learns to *verify* (score input–prediction compatibility) and predicts by energy minimization.
   - Claims: up to 35% faster scaling than Transformer++, and 29% more gain from extra inference compute on language. On image denoising it beats DiT with 99% fewer forward passes. Thinking gains are larger out of distribution.
   - Venue: **ICLR 2026 oral**. — [ICLR oral page](https://iclr.cc/virtual/2026/oral/10008833); [OpenReview ZBj3Qp1bYg, "ICLR 2026 Oral"](https://openreview.net/forum?id=ZBj3Qp1bYg); [arXiv 2507.02092](https://arxiv.org/html/2507.02092v1)
5. **TarFlow, "Normalizing Flows are Capable Generative Models"** (Apple).
   - Old idea: normalizing flows / MAF [bg: Dinh 2014/2016; Papamakarios 2017].
   - Framing: a transformer autoregressive flow over patches, with alternating direction, plus noise augmentation, post-hoc denoising, and guidance.
   - Results: SOTA image likelihoods and the first standalone NF with sample quality comparable to diffusion.
   - Venue: **ICML 2025 oral**. Follow-up STARFlow was a NeurIPS 2025 spotlight. — [ICML oral page](https://icml.cc/virtual/2025/oral/47266); [arXiv 2412.06329](https://arxiv.org/abs/2412.06329v3)
6. **VAR, next-scale prediction**.
   - Old idea: coarse-to-fine image pyramids [bg: Burt & Adelson 1983].
   - Framing: autoregression over *scales* instead of raster order, with a multiscale VQ-VAE.
   - Venue: **NeurIPS 2024 Best Paper**. — [NeurIPS 2024 awards blog](https://blog.neurips.cc/2024/12/10/announcing-the-neurips-2024-best-paper-awards/); [award PDF](https://media.neurips.cc/Conferences/NeurIPS2024/NeurIPS2024_Best_Paper_Awards.pdf)
7. **MAR, "Autoregressive Image Generation without Vector Quantization"**.
   - Old idea: continuous-valued autoregression (pre-VQ era) [bg].
   - Framing: a per-token diffusion loss replaces the categorical head.
   - Venue: **NeurIPS 2024 spotlight**. — [OpenReview VNBIF0gmkb](https://openreview.net/forum?id=VNBIF0gmkb)
8. **Fractal Generative Models** (Li, Sun, Fan, He).
   - Old idea: fractals and self-similar recursion [bg: Mandelbrot].
   - Framing: generative modules recursively composed of generative modules.
   - Venue: **TMLR (12/2025)**, not a conference oral. — [arXiv 2502.17437](https://arxiv.org/abs/2502.17437v1); [TMLR/mlanthology](https://mlanthology.org/tmlr/2025/li2025tmlr-fractal)
9. **Drifting Models, "Generative Modeling via Drifting"** (Deng, Li, Li, Du, He; Feb 2026).
   - Idea: move the iteration into training. A "drifting field" goes to zero when the generated and data distributions match, and the loss is the field's squared norm. The field is computed in a pretrained feature space (MAE).
   - Results: 1-step FID 1.54 (latent) and 1.61 (pixel) on ImageNet 256.
   - Venue: OpenReview lists it as "Submitted to ICML 2026". Whether it was accepted is unverified. Many ICLR 2027 follow-ups already exist (DriftXpress, Gradient-Flow Drifting, Sliced-Wasserstein drifting).
   - Lineage: it looks like mean-shift/kernel two-sample ideas [bg: Fukunaga & Hostetler 1975]. That this is the paper's own framing was **not confirmed**.
   - Sources: [arXiv 2602.04770](https://arxiv.org/abs/2602.04770); [OpenReview kH8yxb9xCg](https://openreview.net/forum?id=kH8yxb9xCg)

**Architectures and sequence models**
10. **DyT, "Transformers without Normalization"** (Zhu, Chen, He, LeCun, Liu).
    - Old idea: saturating tanh nonlinearity, and normalization-free networks [bg: Fixup 2019, NFNets 2021].
    - Framing: LayerNorm already behaves like an S-shaped tanh, so replace it with the elementwise tanh(αx). It matches or beats normalized baselines across vision and language, supervised and self-supervised, mostly without retuning.
    - Venue: CVPR 2025. — [arXiv 2503.10622](https://arxiv.org/abs/2503.10622); [CVPR 2025 OA](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Transformers_without_Normalization_CVPR_2025_paper.html)
11. **Gated Attention for LLMs** (Qwen team).
    - Old idea: multiplicative gating [bg: LSTM 1997, highway nets 2015, GLU 2017].
    - Framing: a sigmoid gate after SDPA adds non-linearity and sparsity and removes attention sinks. It improves performance and stability across 30+ variants.
    - Award: **NeurIPS 2025 Best Paper**. — [NeurIPS 2025 awards blog](https://blog.neurips.cc/?p=1443)
12. **1000-Layer Networks for Self-Supervised RL** (Wang et al., Eysenbach).
    - Old idea: "just go deeper" (residual-depth scaling), applied to RL, where networks had stayed at 2–5 layers.
    - Award: **NeurIPS 2025 Best Paper**. — [NeurIPS 2025 awards blog](https://blog.neurips.cc/?p=1443)
13. **minLSTM/minGRU, "Were RNNs All We Needed?"**.
    - Old idea: LSTM/GRU, stripped so they can train with a parallel scan.
    - **Rejected at ICLR 2025**, with ratings 8/3/6. The meta-review cites novelty overlap with GILR (Martin & Cundy, ICLR 2018), xLSTM, HGRN, and MatMul-free, plus small-scale experiments (Shakespeare). — [OpenReview GrmFFxGnOR](https://openreview.net/forum?id=GrmFFxGnOR)
14. **xLSTM**.
    - Old idea: LSTM scaled to billions of parameters, with exponential gating and matrix memory.
    - Venue: NeurIPS 2024 main track. Spotlight status was **not confirmed**. — [NeurIPS 2024 poster page](https://neurips.cc/virtual/2024/poster/96260)
15. **KAN: Kolmogorov–Arnold Networks**.
    - Old idea: the Kolmogorov–Arnold representation theorem [bg: 1957], with learnable activations on edges.
    - Venue: **ICLR 2025 oral**. Later critiques question its practical claims. — [OpenReview Ozo7qJ5vZi, "ICLR 2025 Oral"](https://openreview.net/forum?id=Ozo7qJ5vZi); [critical assessment arXiv 2510.25781](https://arxiv.org/abs/2510.25781v3)
16. **Test-Time Training layers, "Learning to (Learn at Test Time)"**.
    - Old idea: TTT [bg: Sun et al. 2020], and fast weights [bg: Schmidhuber 1992].
    - Framing: the RNN hidden state *is* a model, and the update rule is an SSL gradient step.
    - Venue: **ICML 2025 spotlight**.
    - Vision descendant: **ViT^3: Unlocking Test-Time Training in Vision** was a **CVPR 2026 oral and award candidate**.
    - Sources: [ICML 2025 poster](https://icml.cc/virtual/2025/poster/43617); [PMLR](https://proceedings.mlr.press/v267/sun25h.html); [CVPR 2026 ViT^3](https://cvpr.thecvf.com/virtual/2026/poster/38326)
17. **Diffusion Forcing**.
    - Old idea: teacher forcing combined with per-token noise levels [bg: Williams & Zipser 1989].
    - Venue: **NeurIPS 2024 poster**. It became influential for streaming and world-model video. — [OpenReview yDo1ynArjj](https://openreview.net/forum?id=yDo1ynArjj)

**Recursion and test-time compute (reasoning)**
18. **HRM (Hierarchical Reasoning Model)**: 27M params, two recurrent modules at different timescales, with brain-inspired framing.
    - No peer-reviewed venue was found. OpenReview shows only a CoRR/arXiv record.
    - **ARC Prize's reproduction** got 32% on ARC-AGI-1 and 2% on ARC-AGI-2. Their ablations:
      - A same-size transformer came within about 5 points; the hierarchy mattered little.
      - The outer refinement loop drove the gains: going from 1 to 8 loops roughly doubled accuracy.
      - Training on the evaluation tasks drove performance.
      - About 300 augmentations were enough, versus the 1,000 used in the paper.
    - Cost: $148.50 for ARC-AGI-1 (9h16m).
    - Sources: [arXiv 2506.21734](https://arxiv.org/pdf/2506.21734); [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis)
19. **TRM, "Less is More: Recursive Reasoning with Tiny Networks"** (Jolicoeur-Martineau).
    - Design: a single 2-layer network (5–19M params) that recursively refines a latent state and an answer, with deep supervision. It drops HRM's biology and fixed-point theorem.
    - Results: 45% on ARC-AGI-1, 8% on ARC-AGI-2, and 87.4% on Sudoku-Extreme (HRM: 55%).
    - Compute: Sudoku on 1 L40S in under 36h; Maze on 4 L40S in under 24h; ARC in about 3 days on 4 H100. About 1K training examples, with about 1,000 augmentations each.
    - Award: **ARC Prize 2025 paper award, 1st place ($50k)**, out of 90 submissions. No main-conference venue was found.
    - Caveat: the paper itself cites evidence that deep supervision roughly doubled accuracy (19% to 39%), while recursion added little (35.7% to 39.0%).
    - Sources: [arXiv 2510.04871](https://arxiv.org/html/2510.04871v1); [ARC Prize 2025](https://arcprize.org/competitions/2025)
20. **VARC, "ARC Is a Vision Problem!"** (Hu, …, Andreas, He).
    - Old idea: treat a symbolic puzzle as image-to-image translation (per-pixel classification on a canvas) with a plain ViT/U-Net plus test-time training.
    - Result: 60.4% on ARC-1 from scratch.
    - Venue: **CVPR 2026**. Oral status was not found. — [CVPR 2026 OA](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_ARC_Is_a_Vision_Problem_CVPR_2026_paper.html)

**Other revivals**
21. **Neural cellular automata**: CAX ("Cellular Automata Accelerated in JAX") was an **ICLR 2025 oral**. — [OpenReview o2Igqm95SJ](https://openreview.net/forum?id=o2Igqm95SJ)
22. **Hopfield / associative memory**: Hopfield and Hinton received the **2024 Nobel Prize in Physics**, with the associative memory that "can store and reconstruct images" cited explicitly. Follow-on work in 2026 includes "Generative Associative Memory via Equilibrium Matching" (NFAM 2026 oral) and continual learning in modern Hopfield networks applied to diffusion. — [Nobel press release](https://www.nobelprize.org/prizes/physics/2024/press-release/); [OpenReview 4zjhUaeeVh](https://openreview.net/forum?id=4zjhUaeeVh); [arXiv 2605.27975](https://arxiv.org/abs/2605.27975)
23. **Classical numerical analysis and geometry**: **ICLR 2026 Honorable Mention, "The Polar Express"**, applies approximation theory (polar decomposition / Newton–Schulz-style iterations) to the Muon optimizer. — [X/DINQ summary of ICLR 2026 awards](https://x.com/dinq_me/status/2047733966191628636); [ICLR blog](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/)
    - **CVPR 2026 award candidates** that are classical geometry or classical optimization: "Linear Fundamental Matrix Estimation from 7 or 5 Points", "QuadSync: Quadrifocal Tensor Synchronization via Tucker Decomposition", "Efficient Unrolled Networks for Large-Scale 3D Inverse Problems" (algorithm unrolling), and "Global-Aware Edge Prioritization for Pose Graph Initialization". — [CVPR 2026 award candidates](https://cvpr.thecvf.com/virtual/2026/events/AwardCandidates2026)
    - **PixelDiT** (pixel-space diffusion, a return to pre-latent diffusion) was a CVPR 2026 **oral and award candidate**. — [CVPR 2026 orals](https://cvpr.thecvf.com/virtual/2026/events/Oral); [award candidates](https://cvpr.thecvf.com/virtual/2026/events/AwardCandidates2026)
    - Vision looped/recursive variants appeared in 2026:
      - ELT, a looped transformer for generation: FID 2.0 with 4× fewer params. — [HF 2604.09168](https://huggingface.co/papers/2604.09168)
      - LoopTrack (looped tracking). — [arXiv 2609.33306](https://arxiv.org/html/2609.33306)
      - ViTRM (TRM for CIFAR classification). — [arXiv 2603.19503](https://arxiv.org/abs/2603.19503)
      - Probabilistic TRM. — [arXiv 2605.19943](https://arxiv.org/abs/2605.19943)
      - Recursive VLMs for symbolic reasoning. — [arXiv 2608.01534](https://arxiv.org/abs/2608.01534)

### Inferences
- **Why the reframings resonated:**
  1. **They are one-line changes with a mechanistic story.** Examples: DyT ("LN is already a tanh"), JiT ("let denoisers denoise", via the manifold argument), and the gated-attention sigmoid gate, which also explains attention sinks.
  2. **They make the old idea win on today's flagship metric.** TarFlow, EqM, MeanFlow, and Drifting all report ImageNet-256 FID, and TRM and HRM report ARC.
  3. **They attach to the current "test-time compute" narrative.** EBT, TRM, EqM's adaptive compute, and TTT all do this. That is why EBMs and RNNs came back in 2025 and not in 2020.
  4. **The titles are provocative.** Examples: "Back to Basics", "Were RNNs All We Needed?", "ARC Is a Vision Problem!", "Less is More", "…are Capable Generative Models".
- **Concentration:** Kaiming He's group alone produced JiT, MeanFlow, DyT (co-author), Fractal, Drifting, VARC, and MAR in 2024–2026. A senior author's credibility clearly helps a "simple old idea" survive review. A CVPR 2027 student paper cannot rely on that and must supply stronger isolation ablations.
- **Compute:** the winning revivals ranged from "tiny" (TRM: about 12 H100-days for ARC; HRM: about $150 per run) to large (ImageNet-256 DiT-scale training on JiT, MeanFlow, and EqM; LLM-scale for gated attention). For vision on one H200, only the TRM/HRM-style and frozen-feature revivals are within budget.

### Gaps
- "Contrastive Language Models (Contrastive-LM/CLM)" is in the brief, but I could not identify a specific 2024–2026 paper with that name. It is not catalogued.
- Oral or highlight status could not be verified for JiT, VARC, or DyT; the CVPR oral pages were truncated.
- Whether xLSTM was a spotlight could not be verified.
- Drifting's ICML 2026 outcome could not be confirmed. I also could not confirm whether its authors explicitly invoke mean shift.
- Compute for JiT, MeanFlow, EqM, Fractal, and gated attention was not collected from primary sources.

---

## Q2. Which revivals got best paper or oral at NeurIPS 2025, ICLR 2026, CVPR 2026, ICCV 2025?

### Takeaway
Best papers at these four venues rewarded "simple old mechanism, rigorously shown at scale" twice: Gated Attention and 1000-layer RL, both at NeurIPS 2025. Revival papers were also well represented among orals and award candidates: MeanFlow (NeurIPS 2025 oral), EBT (ICLR 2026 oral), ViT^3/TTT and PixelDiT (CVPR 2026 orals and award candidates), and classical multi-view geometry (CVPR 2026 candidates). No pure "old idea revival" won the top prize at CVPR 2026 or ICCV 2025. Those went to 3D/4D reconstruction and generation, plus physics-constrained generation.

### Cited Findings
- **NeurIPS 2025**
  - Best Papers: "Artificial Hivemind", "**Gated Attention for LLMs**" (gating revival), "**1000 Layer Networks for Self-Supervised RL**" (depth-scaling revival), and "Why Diffusion Models Don't Memorize".
  - Runners-up: "Does RL Really Incentivize Reasoning…", "Optimal Mistake Bounds for Transductive Online Learning" (resolves a 30-year-old open problem), and "Superposition Yields Robust Neural Scaling".
  - The committee praised gated attention as "easy to adopt" and supported by 30+ experiments. — [NeurIPS blog](https://blog.neurips.cc/?p=1443)
  - Orals include **MeanFlow**. — [NeurIPS oral page](https://neurips.cc/virtual/2025/oral/115488)
- **ICLR 2026**
  - Outstanding Papers: "Transformers are Inherently Succinct" (theory comparing transformers with RNNs) and "LLMs Get Lost in Multi-Turn Conversation".
  - Honorable Mention: "**The Polar Express**" (classical approximation theory applied to Muon).
  - About 19k submissions. — [ICLR blog](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/); [MPI-SWS news](https://saarland-informatics-campus.de/en/piece-of-news/mpi-researcher-receives-outstanding-paper-award-at-iclr-2026/); [DINQ X post](https://x.com/dinq_me/status/2047733966191628636). The second title rests on secondary sources.
  - **EBT was an ICLR 2026 oral.** — [ICLR](https://iclr.cc/virtual/2026/oral/10008833)
  - **EqM was not accepted.** — [OpenReview](https://openreview.net/forum?id=xqlxtkOhDX)
- **CVPR 2026**
  - Best Paper: "Efficiently Reconstructing Dynamic Scenes One D4RT at a Time".
  - Best Student Paper: "Native and Compact Structured Latents for 3D Generation".
  - Honorable mentions: NitroGen, SAM 3D, and ChordEdit (student HM). — [CVPR 2026 closing news](https://cvpr.thecvf.com/Conferences/2026/News/Closing)
  - Revival-flavored **award candidates**: ViT^3 (TTT in vision; also an oral), PixelDiT (pixel-space diffusion; also an oral), "Linear Fundamental Matrix Estimation from 7 or 5 Points", QuadSync (quadrifocal tensors), and "Efficient Unrolled Networks for Large-Scale 3D Inverse Problems". The full list has 74 candidates; I could read 59. — [award candidates](https://cvpr.thecvf.com/virtual/2026/events/AwardCandidates2026); [orals](https://cvpr.thecvf.com/virtual/2026/events/Oral)
  - JiT and VARC were accepted at CVPR 2026. Oral status was not found.
- **ICCV 2025**
  - Best Paper (Marr Prize): **BrickGPT**, "Generating Physically Stable and Buildable Brick Structures from Text". It is a physics-stability prior inside autoregressive generation.
  - Honorable mentions: "Spatially-Varying Autofocus" (computational optics) and "RayZer" (self-supervised large view synthesis).
  - 13 award candidates were chosen from 2,701 accepted papers. — [CMU news](https://icontrol.ri.cmu.edu/news/iccv25.html); [HKUST portal](https://researchportal.hkust.edu.hk/en/prizes/iccv-2025-best-paper-award-candidate/)
- **Earlier anchors**
  - NeurIPS 2024 Best Paper: **VAR** (coarse-to-fine).
  - ICLR 2025 orals: **KAN** and **CAX/NCA**.
  - ICML 2025 oral: **TarFlow**.
  - ARC Prize 2025 1st paper award: **TRM**.
  - Sources: [NeurIPS 2024](https://blog.neurips.cc/2024/12/10/announcing-the-neurips-2024-best-paper-awards/); [KAN](https://openreview.net/forum?id=Ozo7qJ5vZi); [CAX](https://openreview.net/forum?id=o2Igqm95SJ); [TarFlow](https://icml.cc/virtual/2025/oral/47266); [ARC Prize](https://arcprize.org/competitions/2025)

### Inferences
- At CVPR, the closest route to awards for "old ideas" is **classical geometry or physics written as a clean, provable component**: 7/5-point linear F-matrix, quadrifocal synchronization, unrolled inverse problems, and BrickGPT's stability constraint. "Retro architecture" papers fit ML venues better. For CVPR 2027, a revival should be framed around a *vision* mechanism (correspondence, grouping, geometry), not as an architecture swap.
- Revivals that also deliver a **test-time-compute knob** (EBT, ViT^3/TTT, TRM) were in favor in 2025–2026 across venues.

### Gaps
- I could not read the full CVPR 2026 oral list or the last 15 award candidates (the pages were truncated), so revival orals may be under-counted.
- I did not find ICCV 2025 oral-level data on revivals.

---

## Q3. What do OpenReview reviewers say when a paper revives an old idea? What separates success from "not novel"?

### Takeaway
Reviewers do not punish oldness. They punish (a) an **unacknowledged near-equivalence** to a specific prior method, (b) a framing (e.g., "energy") that the experiments do not actually validate, and (c) toy-scale evidence. They reward the authors explaining *why it failed before and what changed*, a generalizable insight ("verification is easier than generation"), and broad scaling or cross-modality evidence. Accepted revivals still drew a 2/10 review (EBT), so one hostile reviewer is survivable if the AC sees a new insight.

### Cited Findings
- **minLSTM, rejected at ICLR 2025 (8/3/6).**
  - The rating-3 reviewer wrote that "minGRU architecture appears mathematically equivalent to their GILR architecture" (Martin & Cundy, ICLR 2018) and that "the parallel scan approach … is also very similar". They also said the LM evaluation "relies solely on the Shakespeare dataset" and that selective copying "seems carefully chosen".
  - The meta-review credits the "framing, and the attempt to spark a discussion … reassessing simpler RNNs", but cites novelty concerns (xLSTM, HGRN, MatMul-free, GILR raised in public comments) and scalability. The authors' defense, that minGRU's role is to "exemplify how architectures can be simplified", did not persuade.
  - Source: [OpenReview GrmFFxGnOR](https://openreview.net/forum?id=GrmFFxGnOR)
- **Equilibrium Matching, not accepted at ICLR 2026 (4/8/4/2).**
  - Reviewers called the change from unconditional FM "multiplying the target gradient by a scalar function c(γ)… a minor modification to claim a new 'framework'".
  - They found a "weak empirical link to EBMs": the explicit-energy variants got FID 73.40 and 75.53.
  - They asked for direct comparison with "Energy Matching [Balcerak 2025], VAPO, Action Matching".
  - They noted that the gains depend on the SiT backbone (a U-Net on CIFAR-10 failed) and that there were no wall-clock or FLOP comparisons.
  - A positive reviewer (8) still called the time-unconditional equilibrium view "a compelling conceptual shift".
  - Source: [OpenReview xqlxtkOhDX](https://openreview.net/forum?id=xqlxtkOhDX)
- **EBT, ICLR 2026 oral, despite reviews of 2/8/8 (among those retrieved).**
  - The rating-2 reviewer extrapolated the authors' scaling fits to a FLOP-efficiency breakeven "at roughly 10^52 total FLOPs… beyond any conceivable computational regime", and noted EBTs need about 10× more compute for equal perplexity.
  - The rating-8 reviewers praised the "genuinely novel and conceptually elegant" insight that "verification is easier than generation". They also praised the cross-modal validation (language, video, images), the larger thinking gains under distribution shift, and the honest limitations, including failure at text-to-image.
  - Source: [OpenReview ZBj3Qp1bYg](https://openreview.net/forum?id=ZBj3Qp1bYg)
- **KAN, ICLR 2025 oral (reviews 6/6/8/8/8).**
  - One reviewer explicitly framed it as a revival done right: "the connections between KAT and neural networks are quite old, dating back to the 1960s and 1970s… [the work] applies modern knowledge about neural networks to improve on previous failed attempts", and "helps spark discussion about an old topic using modern lenses".
  - Another praised that "why this theorem was not widely applied for machine learning previously is also discussed" and that the paper states KAN's "comfort zone" and limitations.
  - Source: [OpenReview Ozo7qJ5vZi](https://openreview.net/forum?id=Ozo7qJ5vZi)
- **Older precedent (NeurIPS 2019/2020 EBM reviews).** One 2019 reviewer said that "many, if not all, of the techniques have been previously proposed", but that combining them to scale EBMs to modern architectures "is a novel contribution". A 2020 meta-review accepted a paper whose "notions and leveraged algorithms are not novel" because it was the first to demonstrate the approach on real images. This attribution comes from a search-engine summary of those review pages. — [NeurIPS 2019 reviews](https://papers.nips.cc/paper_files/paper/2019/file/378a063b8fdb1db941e34f4bde584c7d-Reviews.html); [NeurIPS 2020 meta-review](https://papers.neurips.cc/paper_files/paper/2020/file/90525e70b7842930586545c6f1c9310c-MetaReview.html)
- **Post-publication deflation.**
  - ARC Prize showed that HRM's headline mechanism (the hierarchy) was not the driver; the outer refinement loop and training on the evaluation tasks were. — [ARC Prize](https://arcprize.org/blog/hrm-analysis)
  - TRM then turned this into a simpler, better paper that openly cites the deep-supervision finding (19% to 39%). — [TRM](https://arxiv.org/html/2510.04871v1)
  - KAN drew a "critical assessment" of its claims versus the evidence. — [arXiv 2510.25781](https://arxiv.org/abs/2510.25781v3)

### Inferences — a checklist for an "old idea" paper to survive CVPR 2027 review
1. **Name the old idea and its closest modern twin in the abstract**, and show a head-to-head against that twin (the GILR lesson). Do not let reviewers or public commenters discover the equivalence for you.
2. **State why it failed then and which specific ingredient changed now** (the KAN lesson): for example, DINOv3 features make local evidence reliable, or deep supervision makes recursion trainable.
3. **Validate the framing itself, not just the metric** (the EqM lesson). If the paper says "energy", the energy must rank errors or work as a verifier. If it says "voting", show that the vote maps explain the failures.
4. **Run the isolation ablation before reviewers or outsiders do** (the HRM-to-TRM lesson). Disentangle the old mechanism from confounds such as deep supervision, augmentation, extra test-time compute, and backbone.
5. **Give a minimal demonstration plus one scaling axis** (model size, iterations, data, or difficulty), on a benchmark people already report.
6. **Offer a generalizable one-sentence insight** ("verification is easier than generation"; "denoisers should denoise"; "LN is a tanh"). EBT's 2/10 review was outweighed by exactly this.
7. **Be honest about limits and compute.** Reviewers rewarded EBT's and KAN's limitation sections. Matched-compute tables pre-empt the "10× compute" attack.

### Gaps
- Public reviews for TRM and HRM were not found; they appear to have had no main-conference OpenReview record. DyT, JiT, and VARC (CVPR) have no public reviews.
- MeanFlow, MAR, and Diffusion Forcing reviews exist on OpenReview but were not retrieved because of the API challenge on forum queries.

---

## Q4. Old CV ideas ripe for revival with modern foundation models (CVPR 2027, one H200)

### Takeaway
Seven candidates passed a 2025–2026 novelty screen. Five returned **zero arXiv abstract hits** for the exact combination (old idea plus DINOv2/3 or a modern feed-forward model):
- Hough voting / implicit shape models on DINOv3 for counting and detection
- deformable part models / pictorial structures on DINOv3
- RANSAC-style consensus over feed-forward 3D (VGGT) subsets
- recursive "visual routines" (curve tracing) with TRM-style deep supervision
- Lucas–Kanade feature-constancy refinement on top of feed-forward point trackers

Energy-based dense refinement (SPEN-style) is thinly occupied. Several suggested candidates are **already crowded** and should be avoided as the main contribution:
- Kalman filtering on SAM 2
- active vision / saccades
- energy-based verification for robot actions
- test-time-trained 3D (analysis-by-synthesis style)
- TRM-for-ARC-in-vision
- MoE as modular vision

### Cited Findings (novelty screen, 2026-10-13)
- **Zero-hit arXiv abstract queries:**
  - `"Hough voting" AND (DINOv2 OR DINOv3 OR "foundation model")`: 0
  - `"Hough" AND counting AND "class-agnostic"`: 0
  - `"pictorial structure" AND (DINOv2|DINOv3|foundation)`: 0
  - `"deformable part" AND (DINOv2|DINOv3)`: 0
  - `RANSAC AND (VGGT|DUSt3R|MASt3R) AND dynamic`: 0
  - `pathfinder AND (recurrent|looped) AND extrapolat*`: 0
  - `"mean shift" AND (DINOv2|DINOv3)`: 0
  - `"multiple hypothesis tracking" AND (SAM|SAM2|"foundation model")`: 0
  - `"brightness constancy" AND "test-time" AND (tracking|flow)`: 1 hit, from 2018 (DF-Net)
  - `"Lucas-Kanade" AND (DINOv2|DINOv3|"foundation model")`: 1 hit (a SLAM front-end framework, 2607.01757)
  - `"structured prediction energy" AND (segmentation|depth)`: 1 hit, from 2018
  - `"energy-based" AND "test-time" AND (segmentation|depth|optical flow)`: 4 hits, the closest being ETA (energy-based TTA for depth completion, 2508.05989)
  - Source: [arXiv API](http://export.arxiv.org/api/query?search_query=abs:%22Hough%20voting%22)
- **Nearest neighbours found by web search:**
  - CountingDINO: training-free class-agnostic counting with DINOv2, 21.26 MAE on CARPK, no voting. — [arXiv 2504.16570](https://arxiv.org/html/2504.16570)
  - RobustVGGT: view-level outlier filtering from VGGT's internal attention (CVPR 2026); not subset consensus and not dynamic pixels. It notes that feed-forward models lack COLMAP's "geometric verification … RANSAC outlier rejection". — [arXiv 2512.04012](https://arxiv.org/abs/2512.04012)
  - Adaptive recurrent vision: zero-shot computation scaling on mazes (2023). — [arXiv 2311.06964](https://arxiv.org/pdf/2311.06964)
  - ViTRM: classification only. — [arXiv 2603.19503](https://arxiv.org/abs/2603.19503)
  - CoTracker3: notes that photometric consistency is a correspondence proxy, but that refinement with colour-constancy signals "quickly saturate[s]". DINO-Tracker pairs per-video test-time optimization with DINOv2. — [CoTracker3](https://arxiv.org/html/2410.11831v1)
  - EBT's vision evidence is image denoising and video only. — [ICLR 2026](https://iclr.cc/virtual/2026/oral/10008833)
- **Crowded (avoid as the main idea):**
  - **Kalman + SAM 2**: SAMURAI (Kalman motion model for mask selection), SAM2Plus, HiM2SAM, SatSAM2, a Sensors-2025 Kalman-SAM2 paper, and a 2025 memory-design study reusing SAMURAI's Kalman scoring. — [HiM2SAM](https://arxiv.org/pdf/2507.07603); [SatSAM2](https://arxiv.org/html/2511.18264); [Kalman-SAM2](https://pmc.ncbi.nlm.nih.gov/articles/PMC12252479/); [memory design](https://arxiv.org/html/2512.22624v1)
  - **Active vision / saccades**: DINO-guided saccades (2603.09613), Foveated Dynamic Transformer (2607.09480), FAVE (2609.04392), policy-based foveated imaging (2606.02565). This also overlaps the "zoom" idea in the earlier report. — [2603.09613](https://arxiv.org/pdf/2603.09613); [2606.02565](https://arxiv.org/pdf/2606.02565)
  - **Verifiers for robot actions**: CheckVLA, World Action Verifier, test-time scaling for world-action models, and τ0-WM's unified action evaluation. — [CheckVLA](https://arxiv.org/pdf/2607.26789); [WAV](https://arxiv.org/html/2604.01985v1); [TTS-WAM](https://arxiv.org/pdf/2607.17454); [τ0-WM](https://arxiv.org/html/2606.01027v1)
  - **Test-time 3D / analysis-by-synthesis**: Test3R (cross-pair consistency, prompt tuning), Free Geometry (self-supervised TTA from longer sequences, under 2 minutes per dataset), Scal3R, Info3R, VGG-T³. — [Test3R](https://arxiv.org/abs/2506.13750); [Free Geometry](https://arxiv.org/html/2604.14048v1); [Scal3R](https://arxiv.org/pdf/2604.08542); [Info3R](https://arxiv.org/pdf/2609.21938); [VGG-T³](https://arxiv.org/html/2602.23361v1)
  - **TRM/ARC in vision**: VARC (CVPR 2026, 60.4% ARC-1), Recursive VLMs (2608.01534), Probabilistic TRM (2605.19943). — [VARC](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_ARC_Is_a_Vision_Problem_CVPR_2026_paper.html)
  - **Looped vision backbones**: ELT, LoopTrack, Déjà View. — [ELT](https://huggingface.co/papers/2604.09168); [LoopTrack](https://arxiv.org/html/2609.33306)
- **Original DPM reference:** Felzenszwalb et al. — [lsvm-pami.pdf](https://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf). It notes that part locations are latent and inferred during learning.

### Inferences — candidate ideas (GPU-day figures are **my own estimates** for one H200)

Overlap check against the earlier reports (`reports/*.md`): none of these duplicate the 18 shortlisted ideas (decoder-swap detectors, GUI aliasing, null-space hallucination, zoom, VLA attribution, audio/subtitles, erasure, geometric forgetting, point-light V-JEPA, light linearity, register families, arbitration, RL readout, SAM 2 memory errors, 4D physics judge, 3D calibration, video-diffusion physics probe, REPA prediction). Idea 6 touches tracking, but at the level of sub-pixel refinement, not memory.

**1. "Counting is voting": the Implicit Shape Model / generalized Hough transform on frozen DINOv3** *(rank 1: cheapest and cleanest)*
- Old idea: the generalized Hough transform [bg: Ballard 1981] and the Implicit Shape Model [bg: Leibe, Leonardis & Schiele 2004/2008]. Local patches cast votes for the object centre via codebook offsets.
- Why it failed then: hand-crafted local descriptors (SIFT/patches) were weakly discriminative and viewpoint-fragile. The codebook offsets were noisy, so votes smeared. CNN detectors with learned regression replaced it.
- What changed: DINOv3 patch features are semantically consistent across instances and parts. A patch's nearest-neighbour exemplar patch carries a reliable offset to the exemplar's centre. The voting step needs no training, and the vote map explains each count.
- Evidence it has not been done: 0 arXiv hits for Hough/ISM with DINOv2/3/foundation models or class-agnostic counting. CountingDINO uses DINOv2 without voting.
- Falsifiable hypothesis: with 3 exemplar boxes, training-free DINOv3-ISM voting reduces MAE by at least 15% relative to CountingDINO and other training-free baselines on FSC-147 and CARPK, with the gain concentrated in dense or occluded scenes (more than 50 objects). If it does not beat density-similarity baselines at matched backbone, the hypothesis fails.
- Minimal experiment: DINOv3 ViT-L/16 frozen. Use exemplar-patch-to-centre offsets, kNN matching, scale-normalised votes, and peak finding. Evaluate on FSC-147, CARPK, and one microscopy cell-counting set, which gives a direct link to the lab's cell-tracking work. Ablations: no offsets (pure similarity) versus voting; DINOv2 versus v3 versus CLIP; vote-map entropy as an error predictor. **About 1–2 GPU-days.**
- Risk: Training-based CAC SOTA (CountGD, etc.) may still win. The paper must be framed as training-free, interpretable, and backbone-agnostic.

**2. Deformable Part Models revisited: latent-part star models on frozen DINOv3** *(rank 2)*
- Old idea: pictorial structures [bg: Fischler & Elschlager 1973] and DPM with latent SVM ([Felzenszwalb et al. PAMI 2010](https://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf)).
- Why it failed then: HOG part filters were weak, and CNNs subsumed parts implicitly.
- What changed: DINOv3 dense features already encode parts, so a K-part star model with quadratic deformation costs becomes a tiny, interpretable head. Inference uses a distance transform. A head like this could add occlusion robustness and few-shot sample efficiency that linear or attentive probes lack.
- Evidence it has not been done: 0 arXiv hits for DPM or pictorial structures with DINOv2/3. CompositionalNets (Kortylewski et al. 2020) [bg] covered occlusion-robust compositional models on CNN features, not foundation features. It must be cited as the closest twin.
- Falsifiable hypothesis: at equal parameter count, a DPM head on frozen DINOv3 beats linear, attentive-pooling, and MLP probes on occluded-object classification and detection (OccludedPASCAL3D+ at L1–L3 occlusion) by at least 5 points. The gap grows with occlusion level and shrinks to about 0 on unoccluded data. If the gap is flat across occlusion levels, the "parts help occlusion" story fails.
- Minimal experiment: frozen DINOv3 ViT-B/L, a K=8 part star head, and latent part placement learned by soft-argmax over displacements. Data: PASCAL3D+, OccludedPASCAL3D+, and a 10-shot COCO subset. Report part-consistency maps. **About 2–4 GPU-days.**
- Risk: "probe engineering" critique. Mitigate with the occlusion-scaling curve and part interpretability.

**3. RANSAC for feed-forward 3D: consensus over view subsets with VGGT/π3** *(rank 3)*
- Old idea: RANSAC hypothesise-and-verify [bg: Fischler & Bolles 1981].
- Why it failed then: nothing failed. RANSAC never left SfM, but feed-forward models dropped it.
- What changed: models like VGGT give a full pointmap plus cameras per forward pass. Random view subsets act as cheap "minimal samples", and per-pixel agreement across subset hypotheses (after Sim(3) alignment) marks static inliers versus dynamic or distractor outliers.
- Evidence it has not been done: 0 arXiv hits for RANSAC with VGGT/DUSt3R/MASt3R in dynamic scenes. RobustVGGT (CVPR 2026) filters whole views via attention, not pixel-level consensus.
- Falsifiable hypothesis: with N=16–32 subset hypotheses, consensus-weighted re-estimation improves camera ATE/RPE by at least 15% on dynamic sequences (Sintel, Bonn, TUM-dynamics) over single-pass VGGT and RobustVGGT, and yields motion masks with useful IoU at zero training cost. On static scenes (CO3D/ScanNet) the gain should be about 0. Gains on static scenes would mean averaging, not outlier rejection.
- Minimal experiment: VGGT-1B inference only. Use a subset-sampling schedule and per-point residual voting, then re-run on inliers. Compare against the D4RT/MonST3R-style dynamic-aware models as an upper reference. **About 2–4 GPU-days** (dominated by forward passes).
- Risk: dynamic-aware feed-forward models (the CVPR 2026 best paper D4RT is dynamic) may already solve it. Frame it as a *model-agnostic wrapper* and test on several backbones.

**4. Visual routines revived: tiny recursive networks with deep supervision for serial perception (curve tracing / connectivity)** *(rank 4: highest ceiling, highest risk)*
- Old idea: Ullman's visual routines (1984) and incremental grouping / curve tracing (Jolicoeur 1986; Roelfsema 2006) [bg]. Some relations ("are A and B on the same curve?") need serial, iterative computation. The Pathfinder benchmark (Linsley et al. 2018) [bg] showed that feed-forward CNNs fail at it.
- Why it failed then: recurrent vision nets were hard to train and did not scale. Fixed-depth ViTs became dominant.
- What changed: TRM showed that **deep supervision plus a tiny recursed network** trains stably and generalises from about 1K examples. ARC Prize showed the *outer refinement loop* is the active ingredient.
- Evidence it has not been done: 0 arXiv hits for Pathfinder plus recurrent/looped plus extrapolation. TRM in vision is limited to CIFAR (ViTRM) and ARC (VARC, Recursive VLMs). Adaptive recurrent vision (2023) did mazes without deep supervision and on synthetic data only.
- Falsifiable hypothesis: a 2–7M-param recursive network with TRM-style deep supervision, trained on short curves (length ≤ L), solves connectivity at 2–4× L by scaling test-time iterations. Fixed-depth ViTs, and DINOv3 with a probe, collapse to chance beyond L. The isolation ablation must show the gain comes from recursion and is not explained by deep supervision alone. If deep supervision on a fixed-depth net extrapolates equally well, recursion is not the cause.
- Minimal experiment: synthetic Pathfinder-style curves with controlled length and distractors, plus a real transfer set (neurite or vessel tracing; cell-lineage connectivity from microscopy). Ablations: {recursion} × {deep supervision} × {iterations at test}. **About 3–6 GPU-days.**
- Risk: "toy task" critique. A real-image transfer result is mandatory.

**5. Structured prediction energy networks for dense perception: EBT-style test-time compute for depth/flow** *(rank 5)*
- Old idea: structured prediction energy networks [bg: Belanger & McCallum 2016; end-to-end SPEN, [arXiv 1703.05667](https://arxiv.org/pdf/1703.05667)] and iterative reasoning as energy minimisation ([Du et al. ICML 2022](https://proceedings.mlr.press/v162/du22d/du22d.pdf)).
- Why it failed then: inexact inference, unstable training, and weak backbones.
- What changed: EBT (ICLR 2026 oral) showed stable EBM training recipes (replay, Langevin noise, path randomisation) and that thinking gains are larger OOD. Frozen DINOv3 features make a small energy head E(features, depth map) cheap.
- Evidence it has not been done: only ETA (energy-based TTA for depth *completion*) and one 2018 SPEN hit. EBT has no dense-geometry experiments.
- Falsifiable hypothesis: starting from Depth-Anything-v2/v3 predictions, k energy-descent steps reduce AbsRel monotonically, and more on OOD sets (e.g., ETH3D, DIODE outdoor, KITTI under fog) than in-distribution. Final energy ranks per-image error with AUROC ≥ 0.75, so it doubles as a verifier. If a learned iterative refiner (RAFT-style GRU) with equal parameters matches it, the "energy" framing is unnecessary; this is the EqM lesson.
- Minimal experiment: frozen DINOv3 features plus a 20–50M energy head, trained on synthetic depth (Hypersim) with noise-perturbed targets. Evaluate on NYU, KITTI, ETH3D, and DIODE. **About 4–8 GPU-days.**
- Risk: EBM instability, and "it's just a refiner". Pre-register the energy-as-verifier test.

**6. Lucas–Kanade on foundation features: feature-constancy test-time refinement for point tracking** *(rank 6: relevant to the lab's centre-error work)*
- Old idea: Lucas–Kanade (1981) and the inverse-compositional alignment of Baker & Matthews (2004) [bg].
- Why it failed then: brightness constancy breaks under lighting change, specularity, and large motion, and coarse-to-fine LK drifts.
- What changed: feed-forward trackers (CoTracker3, TAPNext) give a good initialisation, so only small residuals remain. DINOv3 or tracker-internal features satisfy *feature* constancy far better than raw intensity does. Gauss–Newton steps on a frozen feature map are cheap and differentiable.
- Evidence it has not been done: the arXiv screen returned only a 2018 hit for brightness constancy at test time and a SLAM front-end for LK with foundation models. CoTracker3 itself reports photometric-refinement gains "quickly saturate". Featuremetric refinement exists in SfM (PixLoc, Pixel-Perfect SfM, 2021 [bg]) and DINO-Tracker does per-video test-time optimization, so the paper must cite both and claim the *training-free, plug-in, sub-pixel* angle.
- Falsifiable hypothesis: 3–5 inverse-compositional LK iterations on DINOv3 features, gated by predicted visibility, improve δ<1px and δ<2px accuracy on TAP-Vid-DAVIS/RGB-Stacking (and sub-pixel centre error on microscopy cell tracks) by at least 2 points over the raw tracker, with no loss on occlusion accuracy. If gains vanish once the tracker's own iterative updates are run longer, the hypothesis fails.
- Minimal experiment: CoTracker3 and TAPNext outputs, frozen DINOv3 feature pyramid, a GN/LK solver, and visibility gating. **About 1–3 GPU-days.**
- Risk: incremental framing. Pitch it as "classical sub-pixel alignment is the missing last mile of foundation trackers".

**7. (Lower priority) Multiple Hypothesis Tracking over foundation segmenters**
- Old idea: Reid's MHT (1979) [bg].
- Evidence: 0 arXiv hits for MHT with SAM/SAM2, but SAM2Long's tree of memory pathways [bg, not verified this session] is effectively MHT. Combined with the crowded Kalman+SAM2 line, scoop risk is high.
- Use only as a component, e.g., for division hypotheses in cell lineage tracking; not as a headline. **About 3–5 GPU-days.**

**Explicitly not recommended as headline revivals**, given the cited crowding:
- Kalman/Bayesian filtering on SAM 2
- active vision / saccades
- energy-based verification of robot actions
- TTT / analysis-by-synthesis for feed-forward 3D
- TRM for ARC-style visual puzzles
- MoE as "modular vision" (V-MoE-era prior art [bg]; no novelty screen run)
- neural cellular automata for vision (an ICLR 2025 oral tooling paper exists; niche)

**Suggested priority for the November 16 deadline:**
- Ideas 1 and 3 are the safest: inference-only, about 2–4 GPU-days, and a clean "old idea, new evidence" story.
- Idea 4 has the most "paradigm" upside but needs a real-image result.
- Ideas 1 and 6 also reuse the lab's microscopy/cell-tracking assets.

### Gaps
- Novelty screens were arXiv-abstract and web searches only. Full-text, OpenReview ICLR 2027 submission, and Google Scholar checks were not done; the ICLR 2027 submissions have been public since about early October 2026 and should be screened next.
- CompositionalNets, PixLoc/Pixel-Perfect SfM, DINO-Tracker details, and SAM2Long's tree memory are cited from background knowledge or secondary mentions, not verified this session.
- No pilot numbers exist for any hypothesis. Every effect size above is a target, not a finding.
