# Agentic AI for Computer Vision: open problems a 1×H200 team can tackle for CVPR 2027

Scope note: researched 2026-10-13. CVPR 2027 deadlines are registration Nov 10 and paper Nov 16, 2026 (both fixed), with supplementary due Nov 23 ([CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)). These notes deliberately avoid the two ideas already in `reports/Ý tưởng nghiên cứu CVPR 2027.md`: MLLM "arbitration" layers (#2) and "RL changes readout, not perception" (#3). **All GPU-day figures below are my own estimates. They are not sourced.** Several sources were read only at abstract level, and those are flagged.

## Q0. What are 8–12 concrete open problems that fit 1×H200 before Nov 16, 2026?

### Takeaway
The open space is not building another agent. It is **controlled decomposition of where agentic visual gains and failures come from**: pixels vs. resolution vs. text scaffold, localization vs. relation, memory interference, and frame selection vs. agentic seeking. Most 2025–26 agent papers change base model, token budget, and tool loop at the same time, so a careful matched-compute study with a named phenomenon is the cheapest insight paper available. Ideas 1–3 and 6 are the strongest fits for CVPR, because their contribution is about pixels, patches, and resolution.

### Cited Findings (the gaps each idea builds on)
- **Thinking-with-images gains are largely not from returned pixels.** TextCall (Aug 10, 2026, HKUST/ZJU/CUHK/Ant; Qwen2.5-VL-7B) replaces returned crops with "[Image output skipped]" and matches or beats thinking-with-images.
  - 9.5K LoRA: V\* 76.96 vs 71.73. 65K full FT: 78.01 vs 78.53, and the 6-benchmark mean is +1.39 pp in TextCall's favour.
  - In a paired data audit, scaffold-only minus image-only = −0.40 pp (CI [−3.20, +2.40]).
  - Under RL, the thinking-with-images tool-call rate collapses to 0% by step 800.
  - Thinking-with-images drops V\* relative-position accuracy from 71.05 to 57.89.
  - The authors' own limitations: one base model, one seed, one corpus, and they "found no task where returned pixels are load-bearing" and call for such benchmarks.
  - [arXiv 2608.09682](https://arxiv.org/html/2608.09682v1)
- Prior evidence cited by TextCall. I read these only through TextCall, not the originals.
  - Replacing crops with noise costs only 0.52 pp on V\*, while corrupting the text costs 13 pp (arXiv 2510.23482).
  - Only 57% of returned crops contain the target (CodeV, arXiv 2511.19661).
  - Over 70% of post-RL gains come from intrinsic capability rather than tool use (arXiv 2602.01334).
  - Zoom can be distilled into a single pass (arXiv 2602.11858).
  - About 50% of pixel tokens can be skipped (VisionThink, arXiv 2507.13348).
  - Source: [arXiv 2608.09682](https://arxiv.org/html/2608.09682v1)
- "On the Faithfulness of Visual Thinking" (Oct 2025, preprint): intervening on the visual parts of a reasoning trace barely changes predictions, while intervening on the text parts changes them substantially. Visual reasoning traces are "unreliable and insufficient" — [arXiv 2510.23482](https://arxiv.org/abs/2510.23482)
- ViEBench (Jan 2026, under review; 200 high-res images with expert-annotated evidence): models are often right while grounding on irrelevant regions, or locate the right evidence and still answer wrong. The abstract gives no numbers — [arXiv 2601.11633](https://arxiv.org/abs/2601.11633)
- **GUI grounding is unstable under trivial perturbations.** MVP (Dec 2025) adds a 28-px border to ScreenSpot-Pro screenshots.
  - GTA1-7B's coordinates shift by 193 px on average. 7.3% of correct predictions flip to wrong and 7.8% of wrong ones flip to correct. Instability is worse at high resolution and for small elements.
  - Its training-free multi-view clustering lifts ScreenSpot-Pro: GTA1-7B 49.8→61.7, UI-TARS-1.5-7B 41.9→56.1, Qwen3-VL-8B 55.0→65.3, Qwen3-VL-32B 55.3→74.0.
  - [arXiv 2512.08529](https://arxiv.org/html/2512.08529)
- GUI-Actor (NeurIPS 2025) attributes coordinate-generation failures to weak spatial inductive bias and a "mismatch between visual feature resolution and action targets" — [NeurIPS 2025 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/16130af940e9dabb43c726119bd3b42e-Paper-Conference.pdf)
- A search summary says another line of work traces high-resolution failure to patch-to-pixel mapping breaking when models extrapolate to displays larger than those seen in training. I did not open that paper — [search hit: arXiv 2510.03230](https://arxiv.org/pdf/2510.03230)
- **Localization, not relational reasoning, dominates GUI grounding errors.** GUI-Primitives (EMNLP 2026; 994 contrastive pairs; 19 VLMs):
  - Strict accuracy is at most 32%, and 60–92% of predictions land outside *both* candidate elements.
  - When a prediction lands inside a candidate, selection accuracy is 0.82–0.90 for left/right, above/below, proximity, and ordinal relations, but at chance for containment and occlusion.
  - Marking the two candidates (an oracle) adds 35–57 points.
  - Accuracy correlates with ScreenSpot-Pro at ρ = +0.74 across 10 models.
  - [arXiv 2608.21832](https://arxiv.org/abs/2608.21832)
- ScreenSpot-Pro targets average about 0.07% of screen area, versus 2.01% in the original ScreenSpot, and about 35% of samples are ≥2560×1440 — [ScreenSpot-Pro (ICLR 2025)](https://iclr.cc/virtual/2025/32743), via [search summary](https://www.emergentmind.com/topics/screenspot-pro)
- **Planning vs. perception in computer-use agents is unresolved and shifting.**
  - Agent S2 reports that planning failures, not grounding, are now the most frequent — [arXiv 2504.00906](https://arxiv.org/pdf/2504.00906)
  - OSWorld-Human's GTA1 analysis (search snippet) finds the plan correct but the coordinates wrong in 23% of failures — [arXiv 2506.16042](https://arxiv.org/pdf/2506.16042)
  - OSWorld 2.0 (2026) says agents mostly fail on long-horizon work: dropped constraints, missed mid-task information, and skipped verification. Its five failure dimensions include perception–action timing — [arXiv 2606.29537](https://arxiv.org/html/2606.29537v1)
  - "How Benchmarks Mis-Score Computer-Use Agents" proposes a three-tier taxonomy that separates planning failures from perception–action failures (search snippet) — [arXiv 2607.28367](https://arxiv.org/pdf/2607.28367)
- **Visual memory can hurt GUI agents.** "Naive Visual Memory is Not Enough" (ICML 2026 workshop, KAIST/Shin group, OSWorld):
  - Adding full past screenshots reduces state-understanding errors but *increases* grounding errors and "hidden operation blindness."
  - Storing action-local crops (AGMem) improves success by 33.3% over full-image memory.
  - [arXiv 2606.14106](https://arxiv.org/abs/2606.14106)
- DMV-Bench (EMNLP 2026) builds a 1,000-product e-commerce environment. Each visited product image carries an incidental visual cue that exists only in pixels and must be recalled later. Conventional and caption-only memory struggle. DualMem (parallel visual and verbal codes) beats a caption-only baseline and three other memory systems — [arXiv 2606.27499](https://arxiv.org/abs/2606.27499)
- **Before/after screenshot verification is an active 2026 cluster.**
  - VisCritic (ECCV 2026) uses a Siamese ViT with a critic head over pre/post screenshots — [arXiv 2606.24525](https://arxiv.org/html/2606.24525)
  - Evidence-First Reflection extracts action-induced differences first, then judges — [arXiv 2608.24015](https://arxiv.org/html/2608.24015v1)
  - "Don't Act Blindly" predicts the expected screen change before acting — [arXiv 2604.05477](https://arxiv.org/pdf/2604.05477)
- **Long-video agents are compared under unmatched budgets.**
  - VideoSeek (CVPR 2026) compares against GPT-5 with 384 uniformly sampled frames and allows up to 20 turns — [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_VideoSeek_Long-Horizon_Video_Agent_with_Tool-Guided_Seeking_CVPR_2026_paper.pdf)
  - SVAgent: uniform sampling goes from 41.2% at 4 frames to 59.8% at 32 frames, while SVAgent reaches 60.7% with 8 frames — [arXiv 2604.05079](https://arxiv.org/pdf/2604.05079)
  - VideoBrain: +3.5 to +9.0% with 30–40% fewer frames — [arXiv 2602.04094](https://arxiv.org/abs/2602.04094)
  - None of the papers found isolates tool calls from frame selection, per my search ([search](https://arxiv.org/html/2603.20185v1)).
- **Visual programming's gains were already shown to be partly an artifact.** Khandelwal, Pavlick & Sun (EMNLP 2023) found ViperGPT's gains over BLIP-2 came from task-specific module selection and "go away" with a task-agnostic module set — [ACL Anthology](https://preview.aclanthology.org/setup/2023.emnlp-main.157). I found no 2025–26 re-analysis with modern VLMs. Meanwhile CodeV, a code-based visual agent with process-level RL, is listed among CVPR 2026 orals ([CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral)).

### Inferences: the list of open problems (ranked by fit for CVPR × feasibility × scoop risk)

**1. "Zoom is a resolution knob": a resolution-matched decomposition of thinking-with-images gains.**
- *Gap:* TextCall shows the scaffold carries the gain, but it uses one model and one seed, and it never matches the *visual-token budget*. The open question is whether a single pass at the same total token budget (higher `max_pixels`) recovers zoom's gain, and whether any task exists where pixels are load-bearing.
- *Angle:* Run a 2×2×k design on V\*, HR-Bench 4K/8K, and MME-RealWorld:
  - {returned crop, placeholder, random-region crop, oracle crop} × {native-resolution single pass at matched tokens} × {7B checkpoints: Qwen2.5-VL-7B, Qwen3-VL-8B, DeepEyes, Pixel-Reasoner, and TextCall if released}.
  - The key new piece is a **legibility-threshold benchmark**: render text or glyph targets whose size, after the model's resize, falls below or above the patch Nyquist limit. Below the threshold, the pixels *must* be load-bearing. This gives the first positive control that TextCall says is missing.
  - The deliverable is a "resolution-equivalent" curve: zoom gain ≈ the gain from raising input resolution to X.
- *Why CVPR:* It is about sampling, resolution, and the patch grid, which are classic vision concepts, not prompting.
- *Feasibility:* Inference only, about 2–4 GPU-days (estimate).
- *Risk:* TextCall partially scoops the negative result. The novelty must come from the matched-token control and the threshold benchmark.

**2. "Off-grid by 14 px": patch-grid aliasing as the cause of GUI grounding instability.**
- *Gap:* MVP shows a 28-px border moves predictions by 193 px, but nobody explains why.
- *Hypothesis:* Qwen-family models use 14-px patches merged 2×2 into 28-px tokens. Translations that are multiples of 28 px should be nearly equivariant, and sub-token shifts should not.
- *Angle:* Sweep translation, padding, and scale at sub- and super-patch offsets across UI-TARS-1.5, GTA1, OS-Atlas, ShowUI, and Qwen3-VL. Measure the equivariance error against element size in tokens.
- *Fix:* A training-free *patch-phase ensemble* (shift by 0, 7, 14, 21 px and un-shift). It is cheaper and more principled than MVP's crop clustering. Compare at matched forward passes.
- *Why CVPR:* Aliasing and equivariance are core vision concepts.
- *Feasibility:* About 2–3 GPU-days (estimate).
- *Risk:* If instability turns out not to be phase-dependent, the result is still a useful negative.

**3. "A size law for GUI grounding": accuracy as a function of patches per element.**
- *Gap:* My search found no paper that measures visual tokens per UI element as the explanatory variable, even though ScreenSpot-Pro targets average 0.07% of the screen.
- *Angle:* Re-plot ScreenSpot-Pro, ScreenSpot-v2, and UI-Vision accuracy against the element's footprint in post-resize tokens, across models and `max_pixels` settings. Test whether one curve collapses all models and resolutions.
- *Payoff:* If it does, the curve predicts *when* zoom helps, which unifies the RegionFocus, UI-Zoomer, and MVP literature, and tells practitioners the minimum resolution needed.
- *Feasibility:* About 1–3 GPU-days (estimate). This pairs naturally with #2 in one paper.

**4. "Find, then relate": a proposal-recall bottleneck in GUI grounding.**
- *Gap:* GUI-Primitives shows 60–92% of predictions land on neither candidate, and an oracle candidate mark adds 35–57 points. That points to an enumeration or detection bottleneck. But GUI-Primitives is EMNLP (cs.CL) and does not study it visually.
- *Angle:* Measure the implicit proposal recall of VLMs by size and by type (icon vs. text). Compare against OmniParser- or OCR-based proposal recall. Show that containment and occlusion are at chance because they need layered or occlusion reasoning, a true vision problem.
- *Feasibility:* About 2 GPU-days (estimate).
- *Risk:* Set-of-Marks style fixes are well known, so the contribution must be diagnostic.

**5. "Whose fault is it?": counterfactual oracle attribution for computer-use failures.**
- *Gap:* Agent S2 says planning dominates, OSWorld-Human says 23% of failures are grounding with a correct plan, and OSWorld 2.0 says long-horizon. Each uses a different taxonomy, and none uses counterfactual swaps.
- *Angle:* Replay OSWorld or AndroidWorld trajectories with an oracle grounder (from the accessibility tree), an oracle state description, or oracle verification. Report the success recovered by each.
- *Why CVPR:* It quantifies how much of agentic failure is *visual*.
- *Feasibility:* The OSWorld VM infrastructure is engineering-heavy. About 4–8 GPU-days (estimate) for 7B agents.
- *Risk:* It may read as an NLP or agents paper. Only the visual-attribution framing makes it a CVPR paper.

**6. "Memory interference": do past screenshots hijack grounding?**
- *Gap:* Full-screenshot memory *increases* grounding errors (arXiv 2606.14106), but no mechanism is given.
- *Hypothesis:* The policy clicks the *remembered* location of a similar element. This is cross-frame coordinate confusion.
- *Test:* Count the clicks that land on the location the target had in a memory frame. Add attention attribution to memory-frame tokens. Control the memory frame's similarity to the current screen.
- *Fixes:* Training-free options include frame-ID or positional separation, and masking the memory frames' coordinate pathway. Compare against AGMem crops at matched tokens. DMV-Bench-style incidental cues provide the opposite stress test, where memory *must* keep pixels.
- *Feasibility:* About 3–5 GPU-days (estimate).

**7. "GUI change blindness": can VLM verifiers see small state changes?**
- *Gap:* VisCritic, Evidence-First Reflection, and Don't Act Blindly all verify from pre/post pairs, but none measures detection as a function of change size or type.
- *Angle:* Build a synthetic controlled set (checkbox toggled, one character edited, a dropdown value, a 4-px focus ring). Plot a psychometric curve for VLM judges against a trivial pixel-diff+crop baseline. The likely finding is that VLMs are change-blind below some token footprint, which links to #3.
- *Feasibility:* About 1–2 GPU-days (estimate).
- *Risk:* Medium scoop risk from the verification cluster.

**8. "Seeking or selecting?": matched-budget controls for long-video agents.**
- *Gap:* Agentic long-video gains are reported against unmatched frame budgets and base models.
- *Angle:* Same base (Qwen2.5-VL-7B or Qwen3-VL-8B) and the same number of *frames seen*. Compare four arms: the agent, CLIP or SigLIP top-k retrieval, uniform sampling, and an oracle-frame upper bound, on LVBench, Video-MME-long, and LongVideoBench.
- *Payoff:* Decompose the gain into selection vs. iterative reasoning.
- *Feasibility:* About 4–6 GPU-days (estimate).
- *Risk:* The likely result ("mostly selection") is an incremental negative unless it is packaged with a named phenomenon.

**9. "Crop correctness as confidence": evidence-grounding signals for selective prediction.**
- *Gap:* Only 57% of crops contain the target (CodeV, via TextCall). Models are right with wrong evidence (ViEBench). Faithfulness is low (arXiv 2510.23482).
- *Angle:* Check whether the crop-target IoU, or the agreement between crop and attention, predicts answer correctness (AUROC). Then test whether a training-free abstain or re-zoom rule raises risk–coverage on V\* and HR-Bench.
- *Feasibility:* About 1–2 GPU-days (estimate).
- *Risk:* Modest headline, but it is cheap and could become a section of #1.

**10. "Visual programs, revisited at matched compute."**
- *Gap:* The 2023 finding that program gains come from module choice has not been re-tested with 2026 VLMs. Meanwhile, code-based visual agents (CodeV) reached a CVPR 2026 oral.
- *Angle:* Compare an end-to-end Qwen3-VL-8B against a VisProg/ViperGPT-style program using the *same* VLM as the only module, against a program with specialist modules (GroundingDINO, SAM, depth), at matched FLOPs. Break results down by task property: counting, multi-hop spatial, high-res, 3D.
- *Feasibility:* About 2–4 GPU-days (estimate).
- *Risk:* The result may be "programs only help via specialist modules," which repeats the 2023 finding unless the task-property breakdown is new.

**Recommended single pick for the Nov 16 deadline:** combine #2 and #3, optionally with #4, into one paper, *"GUI grounding is a sampling problem: patch-phase aliasing and a size law explain high-resolution failures."* It is inference-only, CV-native, about 4–6 GPU-days, and has a cheap training-free fix. The runner-up is #1, which is strong but needs clear separation from TextCall. Note that #6 and #9 lean toward the "encoded-but-not-used" theme of the existing shortlist only loosely, and do not duplicate it.

### Gaps
- I could not read the full text of GUI-Primitives (PDF extraction failed), CodeV, arXiv 2602.01334, or the arXiv 2510.03230 / I-MROPE line. Claims about them come from abstracts or from TextCall's citations.
- I did not check whether a patch-phase or aliasing analysis of GUI grounding already exists. A targeted arXiv search ("patch grid", "sub-patch shift", "translation equivariance GUI grounding") is needed before committing to #2.
- My search found no paper that varies `max_pixels` at a matched token budget against zoom. This is unconfirmed, not proven absent.

## Q1. Where do GUI agents fail visually (small elements, high-res) vs. in planning?

### Takeaway
For **single-step grounding**, failure is overwhelmingly visual and scale-related: small targets, high resolution, unstable coordinates, and localization rather than relation reasoning. For **multi-step agents on OSWorld**, recent analyses say planning, verification, and long-horizon tracking dominate. Nobody has reconciled the two with counterfactual attribution.

### Cited Findings
- ScreenSpot-Pro accuracy drops sharply as target size decreases. Targets average 0.07% of screen area. OS-Atlas-7B scored 18.9% at release — [ICLR 2025](https://iclr.cc/virtual/2025/32743), [summary](https://www.emergentmind.com/topics/screenspot-pro)
- Current open 7–8B models cluster around 50–55% on ScreenSpot-Pro without test-time tricks:
  - GTA1-7B 50.1, UI-Venus-7B 50.8, Qwen3-VL-8B 49.9 to 52.7 depending on protocol, MAI-UI-8B 65.8.
  - Scores vary with protocol and whether zoom is used. Aggregator entries above 70% are self-reported.
  - [POINTS-GUI-G](https://arxiv.org/pdf/2602.06391), [UI-Venus-1.5](https://arxiv.org/pdf/2602.09082), [benchmarklist](https://benchmarklist.com/benchmarks/screenspot_pro/)
- A 28-px padding causes a 193-px mean shift, and instability is worse for small elements and high resolution — [MVP](https://arxiv.org/html/2512.08529)
- Most relational-grounding failures are localization failures, and containment and occlusion are at chance — [GUI-Primitives](https://arxiv.org/abs/2608.21832)
- Agent S2 finds planning failures most frequent — [arXiv 2504.00906](https://arxiv.org/pdf/2504.00906)
- OSWorld-Human finds a correct plan with wrong grounding coordinates in 23% of GTA1 failures (search snippet) — [arXiv 2506.16042](https://arxiv.org/pdf/2506.16042)
- OSWorld 2.0 finds long-horizon failures dominate: dropped constraints, missed information, no verification — [arXiv 2606.29537](https://arxiv.org/html/2606.29537v1)
- CUADebug (2026) diagnoses and repairs computer-use agent failures. I saw the title only — [arXiv 2608.02643](https://arxiv.org/pdf/2608.02643)

### Inferences
- The disagreement ("grounding is the bottleneck" vs. "planning is") is partly an artifact of which agent was analyzed and which taxonomy was used. A counterfactual-oracle study (idea #5) would resolve it.
- Two vision explanations for small-element failure are under-tested: patch-grid aliasing and too few tokens per element. Ideas #2 and #3 test them.

### Gaps
- I did not obtain per-category error rates from OSWorld 2.0 or "How Benchmarks Mis-Score…".

## Q2. Do "thinking with images" / zoom-crop gains come from better perception or from resolution/compute? What are the fair controls?

### Takeaway
2025–26 evidence increasingly says the gains are **not** from the returned pixels. The text scaffold (coordinates, intent) and intrinsic capability carry most of it. No paper found has run the obvious control: a matched visual-token budget at higher native resolution. Tool loops also add latency and tokens, so compute-matched comparisons are missing.

### Cited Findings
- TextCall matches or beats pixel-returning thinking-with-images, cuts latency by 29–46% (V\* 2.91 s vs. 4.12 s), and its pixel-only vs. scaffold-only difference is not significant — [arXiv 2608.09682](https://arxiv.org/html/2608.09682v1)
- Crop-noise replacement costs 0.52 pp on V\*, while text corruption costs 13 pp. These are arXiv 2510.23482's numbers as cited in TextCall — [arXiv 2608.09682](https://arxiv.org/html/2608.09682v1); the abstract of [arXiv 2510.23482](https://arxiv.org/abs/2510.23482) says qualitatively that visual parts are largely ignored.
- Iterative zoom incurs high latency from repeated re-encoding. Region-to-Image Distillation moves zoom into training for single-pass inference — [arXiv 2602.11858](https://arxiv.org/pdf/2602.11858)
- One design argument holds that crop-and-zoom helps because it re-reads only the needed sub-region at high fidelity without growing the global token budget — [arXiv 2609.15800](https://arxiv.org/pdf/2609.15800)
- OpenThinkIMG (2025) offers RL with visual tools — [arXiv 2505.08617](https://arxiv.org/pdf/2505.08617)
- OpenAI's o3 "thinking with images" is the industrial origin of the term — [OpenAI](https://openai.com/index/thinking-with-images/)

### Inferences
Fair controls a CVPR reviewer should demand, which also define idea #1:
- matched total visual tokens, crop vs. a single higher-res pass;
- placeholder, random, and oracle crops;
- blind (question-only) baseline;
- matched wall-clock and FLOPs;
- a positive-control task where pixels are provably necessary, below the legibility threshold.

### Gaps
- I did not verify the exact numbers in arXiv 2510.23482, 2602.01334, or 2511.19661 from the primary papers.

## Q3. Visual memory and state tracking in multi-step agents: what is open?

### Takeaway
2026 benchmarks show visual memory is both necessary (DMV-Bench: pixel-only cues) and harmful when naive (full screenshots increase grounding errors). Verification via before/after screenshots is a fast-growing method cluster. Open questions are the *mechanism* of memory interference, the right *granularity and resolution* of stored visual state, and the *perceptual limits* of change detection.

### Cited Findings
- Full-screenshot memory reduces state errors but increases grounding errors and hidden-operation blindness. AGMem crops give +33.3% success on OSWorld — [arXiv 2606.14106](https://arxiv.org/abs/2606.14106)
- DMV-Bench and DualMem: caption-only memory loses pixel cues, and the visual and verbal codes contribute asymmetrically — [arXiv 2606.27499](https://arxiv.org/abs/2606.27499)
- Long-video memory agents: WorldMM ([arXiv 2512.02425](https://arxiv.org/pdf/2512.02425)), MemDreamer with hierarchical graph memory and agentic retrieval ([arXiv 2606.07512](https://arxiv.org/pdf/2606.07512)), M³Eval ([arXiv 2606.05008](https://arxiv.org/pdf/2606.05008)), and EmbodiedMemory-Bench ([arXiv 2609.28236](https://arxiv.org/pdf/2609.28236)). I saw titles and snippets only.
- Before/after verification: VisCritic (ECCV 2026), EFR, Don't Act Blindly — [arXiv 2606.24525](https://arxiv.org/html/2606.24525), [arXiv 2608.24015](https://arxiv.org/html/2608.24015v1), [arXiv 2604.05477](https://arxiv.org/pdf/2604.05477)

### Inferences
- Memory *architectures* are crowding fast (WorldMM, MemDreamer, DualMem, AGMem). The CVPR-native openings are diagnostic: cross-frame coordinate confusion (#6) and change blindness by size (#7).

### Gaps
- I did not read model lists or numbers for DMV-Bench, M³Eval, or EmbodiedMemory-Bench.

## Q4. Insight-driven agentic / visual-programming papers with awards, and 2025–26 successors

### Takeaway
The 2023–24 lineage (VisProg, ViperGPT, V\*) was rewarded for a *new paradigm*. Its 2025–26 successors are mostly RL recipes and agent frameworks. My search found **no 2026 CVPR award** going to a GUI or visual-agent paper, though agentic papers do appear among orals (CodeV) and highlights. That leaves room for an analysis paper in the style of the 2023 module-selection critique.

### Cited Findings
- VisProg (CVPR 2023 Best Paper), ViperGPT (ICCV 2023), and V\* (CVPR 2024) are named in the assignment as the lineage. I did not re-verify their award pages in this session. ViperGPT paper: [Columbia](https://viper.cs.columbia.edu/static/viper_paper.pdf)
- The 2023 critique attributes ViperGPT's gains to task-specific modules, and finds natural-language decomposition sometimes beats code — [EMNLP 2023](https://preview.aclanthology.org/setup/2023.emnlp-main.157)
- CVPR 2026:
  - CodeV, a code-based visual agent with process-level RL, is listed among orals — [CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral)
  - OralGPT-Plus, an agentic VLM for dental X-rays, is a Highlight, and SenseSearch (agentic search-reasoning) is a poster — [CVPR 2026 Awards page](https://cvpr.thecvf.com/virtual/2026/awards_detail)
  - VideoSeek (long-video agent) is a main-conference paper — [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_VideoSeek_Long-Horizon_Video_Agent_with_Tool-Guided_Seeking_CVPR_2026_paper.pdf)
  - UI-AGILE (GUI RL plus inference-time grounding) appeared in CVPR 2026 *Findings* — [CVF](https://openaccess.thecvf.com/content/CVPR2026F/html/Lian_UI-AGILE_Advancing_GUI_Agents_with_Effective_Reinforcement_Learning_and_Precise_CVPRF_2026_paper.html)
- Thinking-with-images successors: OpenThinkIMG ([arXiv 2505.08617](https://arxiv.org/pdf/2505.08617)), DeepEyes and Pixel-Reasoner (cited in [TextCall](https://arxiv.org/html/2608.09682v1)), and LongVT ([arXiv 2511.20785](https://arxiv.org/pdf/2511.20785))

### Inferences
- The award-winning pattern was "new paradigm, simple to state." In 2026–27 the paradigm exists, so the cheap high-impact move is the *"what actually causes the gain"* paper, analogous to the existing report's P1/P5 patterns. Idea #1 or #2+#3 fits this.

### Gaps
- I did not confirm CVPR 2026 oral or award status for specific GUI-agent papers beyond the listings above, and did not find a 2026 best-paper award for any agent paper.

## Q5. How do CVPR reviewers view agent papers? Is a vision contribution required?

### Takeaway
The CVPR 2027 CFP lists "Agentic AI" as a standalone topic with no special rules, and the conference scope remains "all aspects of computer vision and pattern recognition." Venue placement in 2026 suggests that **GUI and agent papers whose contribution is linguistic or planning go to NLP/ML venues**, while those with a visual mechanism (pixels, screenshots, video frames) land at CVPR/ECCV.

### Cited Findings
- The CVPR 2027 CFP lists "Agentic AI", plus "Physical AI: World Models, VLA, Planning, Simulation" and "Reasoning & Reinforcement Learning". It has no explicit agent or LLM rule, and says LLM use in reviewing is still being finalized — [CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)
- Venue placement:
  - EMNLP 2026: GUI-Primitives (cs.CL, [arXiv 2608.21832](https://arxiv.org/abs/2608.21832)) and DMV-Bench ([arXiv 2606.27499](https://arxiv.org/abs/2606.27499))
  - ICML 2026 workshop: visual-memory failure study (cs.MA, [arXiv 2606.14106](https://arxiv.org/abs/2606.14106))
  - ECCV 2026: VisCritic, a Siamese-ViT visual state comparison ([arXiv 2606.24525](https://arxiv.org/html/2606.24525))
  - CVPR 2026: VideoSeek and CodeV ([CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_VideoSeek_Long-Horizon_Video_Agent_with_Tool-Guided_Seeking_CVPR_2026_paper.pdf), [Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral))
  - CVPR Findings: UI-AGILE ([CVF](https://openaccess.thecvf.com/content/CVPR2026F/html/Lian_UI-AGILE_Advancing_GUI_Agents_with_Effective_Reinforcement_Learning_and_Precise_CVPRF_2026_paper.html))

### Inferences
- "Vision contribution required" is an unwritten norm, not a rule. To de-risk a submission:
  - frame the paper around a visual quantity (resolution, patch grid, element size in tokens, frames seen, visual change magnitude);
  - include a figure on pixels;
  - report blind and text-only baselines;
  - choose "Agentic AI" as the primary area and "Vision, language and reasoning" or "Explainable CV" as secondary.
- Planning-only or prompt-engineering agent papers risk "better suited to ACL/NeurIPS" reviews.

### Gaps
- I found no published CVPR reviewer guideline or AC statement specifically on agent papers. The venue-placement pattern above is my inference from a small sample.

## Q6. Saturated directions to avoid as the main contribution

### Takeaway
Avoid another method in any of these crowded clusters unless it is framed as a mechanism or diagnostic.

### Cited Findings
- **Training-free zoom/refocus for GUI grounding:**
  - RegionFocus ([arXiv 2505.00684](https://arxiv.org/abs/2505.00684)), DiMo-GUI ([arXiv 2507.00008](https://www.arxiv.org/pdf/2507.00008)), UI-Zoomer (+13.4% on ScreenSpot-Pro, [arXiv 2604.14113](https://www.emergentmind.com/papers/2604.14113)), MVP ([arXiv 2512.08529](https://arxiv.org/html/2512.08529))
  - GUI-Spotlight (52.8% from UI-TARS-1.5-7B, [arXiv 2510.04039](https://arxiv.org/pdf/2510.04039)), AutoFocus ([arXiv 2605.02630](https://arxiv.org/pdf/2605.02630)), Chain-of-Ground ([arXiv 2512.01979](https://arxiv.org/pdf/2512.01979)), GUIZoom-Bench ([arXiv 2512.05941](https://papers.cool/arxiv/2512.05941))
- **RL / test-time scaling recipes for GUI grounding and agents:**
  - UI-R1 ([arXiv 2503.21620](https://arxiv.org/pdf/2503.21620)), GTA1 ([arXiv 2507.05791](https://arxiv.org/pdf/2507.05791)), LPO ([arXiv 2506.09373](https://arxiv.org/pdf/2506.09373)), UI-AGILE (CVPR 2026 Findings)
  - Coordinate-free heads: GUI-Actor ([NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/16130af940e9dabb43c726119bd3b42e-Paper-Conference.pdf)), GUI-AIMA ([arXiv 2511.00810](https://arxiv.org/pdf/2511.00810))
- **Scaling GUI data and generalist agents** (needs industrial compute): UI-TARS ([arXiv 2501.12326](https://arxiv.org/pdf/2501.12326)), ScaleCUA ([arXiv 2509.15221](https://arxiv.org/pdf/2509.15221)), UI-Venus-1.5/2 ([arXiv 2602.09082](https://arxiv.org/pdf/2602.09082), [arXiv 2609.00028](https://arxiv.org/pdf/2609.00028)), OmegaUse ([arXiv 2601.20380](https://arxiv.org/pdf/2601.20380))
- **Thinking-with-images RL recipes:** DeepEyes, Pixel-Reasoner, OpenThinkIMG ([arXiv 2505.08617](https://arxiv.org/pdf/2505.08617)). The gains are now contested by [TextCall](https://arxiv.org/html/2608.09682v1).
- **Long-video agent frameworks:** VideoSeek, SVAgent, VideoBrain, VideoDeepResearch ([arXiv 2506.10821](https://arxiv.org/html/2506.10821v1)), LongVT, VideoZoomer, SCOUT ([arXiv 2608.07959](https://arxiv.org/pdf/2608.07959))
- **Multimodal agent memory architectures and benchmarks:** WorldMM, MemDreamer, DualMem, M³Eval, EmbodiedMemory-Bench, MemTools ([arXiv 2607.21404](https://arxiv.org/pdf/2607.21404)), plus surveys ([arXiv 2602.06052](https://arxiv.org/pdf/2602.06052), [arXiv 2606.30306](https://arxiv.org/pdf/2606.30306))
- **Pre/post-screenshot verifiers and critics:** VisCritic, EFR, Don't Act Blindly (above)

### Inferences
- Each of these clusters has 5–10 papers in 12 months, all reporting gains under different protocols. That inconsistency *is* the opening for the analysis ideas #1–#3 and #8.
- A new benchmark without blind and text-only controls is also low-value, consistent with the existing report's note that "MLLM fail at X" benchmarks are saturated.

### Gaps
- Saturation is judged from search-result density, not a systematic count.
