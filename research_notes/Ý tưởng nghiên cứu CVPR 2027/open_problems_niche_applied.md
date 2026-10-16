# Open Problems in Less-Crowded but CVPR-Relevant Vision Areas (for CVPR 2027, 1x H200)

Scope: scientific/biomedical vision (cell tracking/lineage, microscopy FMs), tracking & video (MOT, point tracking, SAM 2/3 video), computational photography/physics-based vision, robustness/calibration/TTA, evaluation & benchmark methodology, efficient vision. Research date: 2026-10-13. About 20 search/fetch calls. Every factual claim carries a source. Synthesis and opinion are kept under "Inferences".

---

## Which niche-area papers won awards, orals or highlights at CVPR/ICCV 2024–2026, and which areas get treated as "application papers"?

### Takeaway
Top CVPR/ICCV awards from 2024 to 2026 went mostly to 3D/4D reconstruction and generation (VGGT, D4RT, MegaSaM, SAM 3D). Physics-based and computational-photography work wins awards steadily: the CVPR 2025 Best Student Paper and an ICCV 2025 honorable mention both came from this area. Biology won once, but only as a broad foundation model with a new dataset (BioCLIP, CVPR 2024 Best Student Paper). No pure tracking, cell-tracking or metric-critique paper won a main award in 2024–2026. Metric critiques are published at CVPR (for example "Rethinking FID", CVPR 2024), and their influence comes through adoption rather than awards.

### Cited Findings
- CVPR 2025 Best Paper: *VGGT: Visual Geometry Grounded Transformer*. Best Student Paper: *Neural Inverse Rendering from Propagating Light* (models time-resolved LiDAR light measurements, so it is physics-based). Honorable mentions: MegaSaM, Navigation World Models, Molmo/PixMo, 3D Student Splatting and Scooping. Student HM: Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens — [CVPR 2025 Awards Press](https://cvpr.thecvf.com/Conferences/2025/News/Awards_Press)
- 15 papers (about 0.1%) were CVPR 2025 best-paper candidates — [Brown CS news](https://cs.brown.edu/news/2025/06/30/brown-visual-computing-is-in-the-top-01-at-cvpr-2025/)
- CVPR 2026 Best Paper: *Efficiently Reconstructing Dynamic Scenes One D4RT at a Time*. Best Student Paper: *Native and Compact Structured Latents for 3D Generation*. HMs: NitroGen and SAM 3D. Student HM: ChordEdit. Demo awards went to Computational Speckle Pattern Interferometry (CSPI), MIBURI and KV-Tracker. There were 16,092 submissions (+24% over 2025) and 4,089 accepted (about 25%) — [CVPR 2026 Closing](https://cvpr.thecvf.com/Conferences/2026/News/Closing)
- Newswise says the committee picked "two papers" for top honours but names only one. I found no second Best Paper — [Newswise](https://www.newswise.com/articles/cvpr-2026-honors-the-year-s-most-innovative-computer-vision-and-ai-research)
- ICCV 2025 Marr Prize: *Generating Physically Stable and Buildable Brick Structures from Text* (BrickGPT). HMs include *Spatially-Varying Autofocus* (Qin, Sankaranarayanan, O'Toole; a computational-photography camera that autofocuses each pixel to a different depth) and *RayZer*. There were 11,239 submissions, 2,701 accepted and 13 award candidates — [IEEE TCPAMI ICCV awards](https://tc.computer.org/tcpami/awards/iccv-paper-awards/), [CMU Graphics](https://graphics.cs.cmu.edu/), [aiwiki ICCV](https://www.aiwiki.ai/wiki/iccv)
- CVPR 2024 Best Student Papers: *Mip-Splatting* and *BioCLIP: A Vision Foundation Model for the Tree of Life*. BioCLIP introduced the TreeOfLife-10M dataset — [CVPR 2024 Awards](https://cvpr.thecvf.com/Conferences/2024/News/Awards), [OSU TDAI](https://tdai.osu.edu/news/bioclip-receives-top-honors-cvpr)
- "Rethinking FID" (Jayasumana et al., Google, CVPR 2024) showed that FID contradicts human raters, misses gradual improvements in iterative T2I models, and is inconsistent across sample sizes. It proposed CMMD (CLIP embeddings plus an MMD distance with an RBF kernel) — [CVF open access](https://openaccess.thecvf.com/content/CVPR2024/html/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.html)
- At NeurIPS 2025, the critique paper "The Illusion of Progress? A Critical Look at Test-Time Adaptation for VLMs" was accepted as a poster — [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/121767)
- Several relevant CVPR 2025 tracking/video papers exist (Track4Gen; "Exploring Temporally-Aware Features for Point Tracking"; M3-VOS). I could not confirm their oral status — [Adobe Research](https://research.adobe.com/publication/track4gen-teaching-video-diffusion-models-to-track-points-improves-video-generation), [ML Anthology](https://mlanthology.org/authors/c/cho-seokju/), [Labellerr roundup](https://labellerr.com/blog/cvpr-2025-part-1)

### Inferences
- Areas with an award record: (a) physics-based and computational imaging (Propagating Light 2025, Spatially-Varying Autofocus ICCV 2025, CSPI demo 2026); (b) 3D/4D geometry; (c) foundation-model and dataset papers that open a new domain (BioCLIP). The pattern for physics-based work: a new measurement model or new hardware plus a clear physical insight.
- Areas perceived as "application papers": cell tracking, medical segmentation and domain-specific MOT. None of these appeared among 2024–2026 main-award winners. Their accepted CVPR versions usually present biology as a testbed for a general problem (for example "tracking with discrete events / split-merge topology", or "long-horizon identity"), not as the goal itself.
- Metric and benchmark critiques get published and cited widely ("Rethinking FID" is the template), but in this window they won no awards. Their realistic upside is an oral or highlight plus adoption, not a best paper.
- Video tracking and point tracking: awards went to adjacent dynamic-reconstruction work (MegaSaM, D4RT), not to trackers. A tracking paper aiming for a highlight should link tracking to 4D or geometry, or provide a diagnosis that holds across many models.

### Gaps
- I did not find full lists of CVPR 2025/2026 orals or highlights filtered by tracking or biomedical topics. The open-access site would need manual filtering.
- I could not confirm the CVPR 2024 Best Papers from a primary source (I believe they were Generative Image Dynamics and Rich Human Feedback, but this is unverified), nor the full ICCV 2025 HM list.
- I found no quantitative data on how reviewers score "application" papers versus methods papers. The perception is inferred from award patterns only.

---

## What are the open problems in long-term and lineage/division-aware tracking, and how do foundation models (SAM 2/3, CoTracker3, DINOv3) fail there?

### Takeaway
Every current family of foundation trackers fails in documented ways over long horizons: SAM 2 accumulates memory errors, CoTracker3 is biased toward stale positions, and TAPNext drifts after about 150 frames and does not re-detect points. None models topology events such as division. Cell tracking still learns lineage from small labelled sets (Trackastra), division (DIV) scores stay consistently below linking (LNK) scores, and zero-shot SAM2 pipelines exist but lack any division mechanism.

### Cited Findings
- SAM 2's greedy memory selection lets one wrong or missed mask cascade into later frames. SAM2Long (ICCV 2025) keeps several pathways in a training-free memory tree and gains up to +5.3 J&F on SA-V/LVOS — [SAM2Long arXiv](https://arxiv.org/html/2410.16268v2), [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html)
- Lean-SAM2 (2026) reports that performance decays gradually from error accumulation in long videos (LVOSv2) — [arXiv 2607.19811](https://arxiv.org/pdf/2607.19811)
- In surgical video, SAM2 struggles because of the domain gap and limited long-term tracking (SA-SV/SAM2S) — [arXiv 2511.16618](https://arxiv.org/abs/2511.16618v1)
- In its own limitations section, SAM 3 reports weak zero-shot generalization to fine-grained niche concepts (thermal imagery is the example) and says concept generalization is harder than SAM/SAM 2's class-agnostic transfer. Its video tracker is SAM 2's memory-based tracker — [SAM 3 arXiv 2511.16719](https://arxiv.org/pdf/2511.16719), [Meta blog](https://ai.meta.com/blog/segment-anything-model-3)
- CoTracker3 is trained on synthetic data and is sensitive to discontinuous video. After a long static period it kept predicting points at their old location as visible after the object had moved — [Voxel51](https://voxel51.com/blog/cotracker3-enhanced-point-tracking-with-less-data), [Track-On2 arXiv 2509.19115](https://arxiv.org/pdf/2509.19115)
- TAPNext argues that windowed trackers fail during long occlusions mid-video. BootsTAPNext drifts and is predicted as occluded later in sequences — [TAPNext arXiv 2504.05579](https://arxiv.org/pdf/2504.05579), [Track-On2](https://arxiv.org/pdf/2509.19115)
- TAPNext++ (2026) calls re-detection after reappearance "a blind spot in current point tracking metrics and evaluations". It extends effective memory from about 150 to over 4,000 frames and lists hour-long tracking and occluder modelling as open problems — [TAPNext++ alphaXiv 2604.10582](https://www.alphaxiv.org/abs/2604.10582v1)
- Trackastra (ECCV 2024) learns pairwise associations, including divisions, with a transformer over a temporal window from annotated data — [arXiv 2405.15700](https://arxiv.org/html/2405.15700v2)
- On a large microbial benchmark, DIV is consistently lower than LNK, so division reconstruction is harder than linking — [arXiv 2411.00552](https://arxiv.org/pdf/2411.00552)
- A zero-shot SAM2-based cell-tracking pipeline for 2D/3D uses SAM-Med3D for large 3D volumes. I did not verify its division handling — [arXiv 2509.09943](https://arxiv.org/html/2509.09943v1)
- Cell as Point is a one-stage tracker that reports division and lineage errors (EdgeAdd) — [arXiv 2411.14833](https://arxiv.org/pdf/2411.14833)
- The 2017 Cell Tracking Challenge report found that no method gave fully correct solutions, and that methods using learned priors or global spatio-temporal analysis did better. The 2023 "10 years" update added silver-standard references — [PMC 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5777536), [CTC 10 years](https://dspace.vsb.cz/items/7a798c4f-eab9-4579-927f-b4327822a1ec)
- MOT: association remains the core difficulty (SportsMOT; DanceTrack has similar appearance and non-linear motion). McByte uses training-free association with propagated SAM-style masks — [SportsMOT](https://arxiv.org/pdf/2304.05170), [arXiv 2512.22105](https://arxiv.org/pdf/2512.22105), [MOT survey 2506.13457](https://arxiv.org/pdf/2506.13457)

### Inferences
- All memory-based foundation trackers share one structural assumption: one query gives one persistent identity. Division (1→2), fusion and fragmentation break this assumption, and this is where SAM 2/3 and point trackers have no mechanism at all. This is a general CV problem, "tracking with topology changes", for which cells are the cleanest and best-annotated testbed. The same framing covers splitting droplets, cell-like particles, breaking objects and crowd split/merge.
- Re-detection and identity recovery after long absence is acknowledged as unmeasured (TAPNext++). This is a gap in both metrics and methods that is cheap to attack.
- Note: the user's repo already has a LineageRAFT / genealogy-constrained decoding line of work (git status shows it). It sits directly on this gap.

### Gaps
- I found no 2025–2026 paper that evaluates SAM 3 / SAM 3.1 or DINOv3 trackers on time-lapse microscopy with divisions.
- I found no 2025–2026 Cell Tracking Challenge report with lineage-specific leaderboard analysis.
- AllTracker: the search returned no limitation analysis.

---

## Which metric/benchmark critique papers had big impact, and which current metrics are ripe for a critique?

### Takeaway
High-impact critiques follow a template: (1) show that the metric disagrees with humans or ground-truth intent, (2) show that it is unstable across sample size or settings, (3) propose a simple fix that drops in easily. Examples are "Rethinking FID"/CMMD (CVPR 2024), HOTA (IJCV 2020), CHOTA (ECCV-W 2024) and "Illusion of Progress" for TTA (NeurIPS 2025). Targets that look ripe now: re-detection in point tracking, lineage metrics in cell tracking, HOTA's per-frame assignment, J&F in long-video VOS, and contamination in VLM/medical benchmarks.

### Cited Findings
- CMMD/"Rethinking FID": Inception features represent modern T2I content poorly, the normality assumption is wrong, and sample complexity is poor. CMMD is unbiased and makes no distributional assumption — [arXiv 2401.09603](https://www.arxiv.org/abs/2401.09603)
- HOTA explicitly balances detection, association and localization — [HOTA arXiv 2009.07736](https://ar5iv.labs.arxiv.org/html/2009.07736)
- The TGOSPA paper (Dec 2024) argues that HOTA is not a mathematically well-defined metric (it fails identity and the triangle inequality), and that per-frame assignment is a heuristic compared with a multi-dimensional assignment — [arXiv 2412.08321](https://arxiv.org/pdf/2412.08321). An earlier analysis showed the same violations — [arXiv 2008.03533](https://arxiv.org/pdf/2008.03533)
- CHOTA extends HOTA so that a "trajectory" covers the whole lineage: ancestors share a trajectory, siblings do not. The authors say existing cell-tracking metrics mostly reward local correctness and only weakly reward global coherence. It is implemented in py-ctcmetrics and traccuracy — [CHOTA arXiv 2408.11571](https://arxiv.org/pdf/2408.11571), [traccuracy docs](https://traccuracy.readthedocs.io/en/latest/metrics/chota.html)
- The microbial benchmark proposes "experiment-aware robustness metrics" for single-cell tracking — [arXiv 2411.00552](https://arxiv.org/pdf/2411.00552)
- "Position: All Current Generative Fidelity and Diversity Metrics are Flawed" (ICML 2025) — [PMLR v267](https://proceedings.mlr.press/v267/raisa25a.html)
- VLM contamination: existing detectors fail or behave inconsistently when models are deliberately contaminated. A multimodal semantic-perturbation detector is proposed — [arXiv 2511.03774](https://arxiv.org/html/2511.03774v2)
- A 2026 audit of medical VQA benchmarks found image-side source overlap on SLAKE-En: 19.8% flagged with SigLIP-B-16 and 4.2% with SigLIP-SO400M (the two detectors disagree) — [Emergent Mind 2606.10066](https://www.emergentmind.com/papers/2606.10066)
- CLIP-era overlap: after re-deduplication, ImageNetV2 accuracy dropped while Stylized-ImageNet accuracy rose, which suggests pretraining overlap inflates "natural shift" robustness — [arXiv 2403.10499](https://arxiv.org/pdf/2403.10499)
- Dynamic or live benchmarks (MMBench-Live 2026) are a response to static-benchmark leakage — [arXiv 2607.01813](https://arxiv.org/pdf/2607.01813)
- "A Metric Learning Reality Check" (ECCV 2020) is an earlier template showing that claimed progress disappears under fair evaluation — [PwC](https://astro.paperswithcode.com/paper/a-metric-learning-reality-check)

### Inferences
- Ripe critique targets, with cheap experiments:
  1. **Point-tracking metrics (AJ, δ_avg, OA)** do not penalize missed re-detection or long-absence identity loss (TAPNext++ admits this blind spot). One could build a "re-detection-aware" variant and re-rank about 8 public trackers.
  2. **CTC TRA/DET and even CHOTA** may still be dominated by the easy links of non-dividing cells (DIV < LNK everywhere). One could quantify how much leaderboard rank depends on the roughly 1–5% of events that are divisions, and show rank flips.
  3. **J&F on long-video VOS (SA-V, LVOS, MOSE)**: frame-averaged scores hide catastrophic identity switches after reappearance. The SAM2Long-style gains of about 3–5 points could be decomposed by event type.
  4. **HOTA's per-frame assignment versus global-assignment metrics (TGOSPA)**: check whether MOT leaderboard rankings change.
  5. **Leakage of SA-1B/SA-V-scale pretraining into VOS and medical benchmarks** (overlap audits in the style of the SLAKE work).
- These are well matched to a single H200: running inference only on public checkpoints is cheap.

### Gaps
- I found no citation counts or adoption rates for CMMD and no evidence of CVPR awards for metric papers. Impact claims here are qualitative.
- I found no 2025–2026 paper critiquing AJ/δ_avg or J&F specifically. This absence is itself evidence that the gap is open, but it may reflect search coverage.

---

## Which test-time adaptation, calibration and uncertainty questions remain open for modern vision foundation models?

### Takeaway
2024–2025 evidence shows that TTA and adaptation of CLIP-like models often buy accuracy at the cost of calibration and trustworthiness. Gains over older baselines are small once evaluation is fair, and evaluation is still CLIP-centric. Calibration and uncertainty for dense or temporal foundation models (SAM 2/3 masks, point-track visibility, tracker confidence) are essentially unaddressed in what I found.

### Cited Findings
- TTA-VLM (NeurIPS 2025): current TTA methods give limited gains over earlier work, combine poorly with training-time fine-tuning (CoOp, MaPLe, TeCoA), and their accuracy gains often reduce trustworthiness (calibration, OOD detection, stability). The benchmark extends to SigLIP — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/121767), [alphaXiv 2503.12096](https://www.alphaxiv.org/overview/2503.12096)
- Adapters, prompt learning and TTA substantially degrade CLIP's zero-shot calibration under distribution shift. The authors attribute this to larger logit ranges and propose a model-agnostic fix — [arXiv 2407.13588](https://arxiv.org/html/2407.13588v1)
- C-TPT (ICLR 2024) calibrates test-time prompt tuning through text-feature dispersion, noting that prompts were tuned for accuracy and calibration was ignored — [arXiv 2403.14119](https://arxiv.org/html/2403.14119v3)
- O-TPT (2025) reports average ECE 4.23 on CLIP ViT-B/16, better than zero-shot CLIP — [OpenReview PDF](https://openreview.net/pdf/1a0c60310a2565e2703804be2d1bcea35527ae1e.pdf)
- SAM 3 says domain adaptation needs fine-tuning or synthetic data from its data engine, which shows that zero-shot is insufficient for niche domains — [SAM 3 arXiv](https://arxiv.org/pdf/2511.16719)
- CoTracker3 keeps predicting stale points as "visible", which is a visibility-calibration failure — [Track-On2](https://arxiv.org/pdf/2509.19115)

### Inferences
- Open questions: (a) whether SAM 2/3 predicted-IoU and object-presence scores are calibrated, and whether they can serve as reliable failure detectors for long-video tracking (useful for SAM2Long-style pathway selection and human-in-the-loop re-prompting); (b) calibration of point-track visibility and occlusion over long horizons; (c) TTA for trackers, meaning adaptation during a long video without accumulating errors, which shares a mechanism with the SAM 2 error cascade; (d) whether TTA benefits survive once results are corrected for pretraining overlap.
- "Illusion of progress" critiques are feasible and well received (NeurIPS 2025). A sequel for dense or video foundation models (SAM 2/3, CoTracker3, DINOv3 linear probes) is an open niche.

### Gaps
- I found no 2025–2026 paper on calibrating SAM 2/3 confidence scores or point-tracker visibility. The gap may be real or a search miss.
- I found nothing on TTA for DINOv3 specifically.

---

## How do CVPR reviewers treat cross-domain papers (for example biology), and what framing makes them generalize to the main CV audience?

### Takeaway
The only biology paper with a main CVPR award (BioCLIP, 2024) was a general-purpose foundation model plus a large new dataset. It was framed as a vision foundation-model contribution that happens to cover the tree of life. Cross-domain papers fare best when they (1) name a general CV failure mode that the domain shows cleanly, (2) evaluate at least one non-bio setting, and (3) give a transferable artifact (metric, dataset, mechanism).

### Cited Findings
- BioCLIP, CVPR 2024 Best Student Paper: a vision foundation model plus the TreeOfLife-10M dataset — [OSU TDAI](https://tdai.osu.edu/news/bioclip-receives-top-honors-cvpr), [CVPR 2024 Awards](https://cvpr.thecvf.com/Conferences/2024/News/Awards)
- Cell-tracking methods with CV-venue publications (Trackastra at ECCV 2024; CHOTA at an ECCV workshop; the ICCV 2025 BIC workshop) show that main-track bio work is presented as a general tracking or transformer method, while metric work often lands in workshops — [Trackastra](https://arxiv.org/html/2405.15700v2), [CHOTA](https://arxiv.org/pdf/2408.11571), [ICCVW 2025 BIC supp](https://openaccess.thecvf.com/content/ICCV2025W/BIC/supplemental/Lalit_An_Investigation_of_ICCVW_2025_supplemental.pdf)
- Microscopy FM evaluation is active: Teuber et al. 2026 compare SAM/SAM2/DINOv3 (v2 adds SAM3 and KRONOS) against µSAM and PathoSAM. In CellCLIP, OpenPhenom (pretrained on Cell Painting) did not beat natural-image DINO. MorphoHELM (2026) benchmarks representations for morphology assays, with DINOv2 and an untrained ResNet as controls — [arXiv 2603.19802](https://arxiv.org/abs/2603.19802), [CellCLIP 2506.06290](https://arxiv.org/pdf/2506.06290), [MorphoHELM 2605.15383](https://arxiv.org/pdf/2605.15383)
- SAM2-based surgical work frames itself as a domain gap plus a long-term tracking problem — [arXiv 2511.16618](https://arxiv.org/abs/2511.16618v1)

### Inferences
- Framing that works: "division / topology change is a failure mode of all memory-based foundation trackers. We characterize it, propose a fix, and show it on cells (primary) plus one non-bio split/merge setting." Framing that does not work: "a better cell tracker on CTC datasets", which is MICCAI or ISBI material.
- The finding that natural-image DINO matches microscopy-specific FMs (CellCLIP) is a general "domain-specific pretraining isn't worth it?" result, the kind of insight CVPR audiences like.

### Gaps
- I found no source documenting CVPR reviewer guidelines or statistics on application or cross-domain papers, and no explicit MICCAI-vs-CVPR acceptance comparison. The framing advice above is inference.

---

## Objective: 10–15 concrete open research questions (gap, why CVPR cares, simple angle, feasibility on 1x H200)

### Takeaway
Below are 13 questions, ranked roughly by fit to a small team with one H200. Questions 1–6 cluster around tracking with topology change and long-horizon identity, plus metrics for them. That is the least crowded intersection with documented foundation-model failures.

### Cited Findings
(Gap evidence for each item is cited in the sections above. Key anchors: SAM2Long/Lean-SAM2 on error accumulation; TAPNext++ on the re-detection blind spot; DIV < LNK; CHOTA; TGOSPA on HOTA; TTA-VLM; CLIP calibration under shift; SAM 3 niche-domain limits; CellCLIP DINO vs OpenPhenom; the VLM contamination detector and the SLAKE overlap audit; Spatially-Varying Autofocus and Propagating Light awards.)
- [SAM2Long](https://arxiv.org/html/2410.16268v2) · [Lean-SAM2](https://arxiv.org/pdf/2607.19811) · [TAPNext++](https://www.alphaxiv.org/abs/2604.10582v1) · [Track-On2](https://arxiv.org/pdf/2509.19115) · [Microbial DIV/LNK](https://arxiv.org/pdf/2411.00552) · [CHOTA](https://arxiv.org/pdf/2408.11571) · [TGOSPA](https://arxiv.org/pdf/2412.08321) · [TTA-VLM](https://neurips.cc/virtual/2025/poster/121767) · [CLIP calib](https://arxiv.org/html/2407.13588v1) · [SAM 3](https://arxiv.org/pdf/2511.16719) · [CellCLIP](https://arxiv.org/pdf/2506.06290) · [Teuber microscopy FM eval](https://arxiv.org/abs/2603.19802) · [VLM contamination](https://arxiv.org/html/2511.03774v2) · [Med-VQA overlap](https://www.emergentmind.com/papers/2606.10066) · [CVPR 2025 awards](https://cvpr.thecvf.com/Conferences/2025/News/Awards_Press) · [ICCV 2025 awards](https://tc.computer.org/tcpami/awards/iccv-paper-awards/)

### Inferences (the list. Gaps are sourced above. Angles and feasibility are my assessment.)

1. **Topology-aware memory: can SAM 2/3-style trackers handle one-to-many identity (division, splitting)?**
   - Gap: memory trackers assume one persistent identity per prompt. Divisions are the hardest events in cell tracking (DIV < LNK), and no 2025–2026 foundation-model tracker handles them.
   - Why CVPR cares: a general failure mode of foundation trackers that also applies to splitting objects, fluids and crowds.
   - Angle: a "spawn" token or memory-fork in SAM 2's memory bank, triggered by mask-area or shape-bimodality signals, trained with a lightweight adapter on CTC plus synthetic splits.
   - Feasibility: high. Fine-tuning a SAM2-S/B adapter fits in 141 GB with time to spare.

2. **Division-weighted evaluation: do CTC, HOTA and even CHOTA leaderboards rank methods mostly by easy links?**
   - Gap: CHOTA still pools all associations. Divisions are rare.
   - Why CVPR cares: a "metrics that mislead" story with rank flips.
   - Angle: stratify scores by event type (link, division, appearance, disappearance), bootstrap the rank stability, and propose an event-balanced score.
   - Feasibility: very high. It needs only CPU plus public predictions.

3. **Re-detection-aware point-tracking evaluation (AJ/δ_avg blind spot).**
   - Gap: TAPNext++ says re-detection is unmeasured.
   - Angle: a re-detection subset or metric built by mining TAP-Vid/RoboTAP for long occlusions, or by synthetically inserting long occlusions (Kubric). Re-rank CoTracker3, TAPNext/++, Track-On2, AllTracker and BootsTAPIR.
   - Feasibility: very high, since it is inference only.

4. **Why does memory error accumulate? A causal analysis of SAM 2/3 long-video failure.**
   - Gap: SAM2Long and Lean-SAM2 patch the symptom.
   - Angle: inject a controlled single-frame error, measure how far and for how long it propagates, and relate this to memory-attention entropy. Derive a training-free rule (for example, gating memory writes on calibrated confidence).
   - Feasibility: high.

5. **Are SAM 2/3 confidence scores (predicted IoU, object score) calibrated over time, and can they predict tracking failure?**
   - Gap: no calibration study found for these scores. The TTA literature shows adaptation hurts calibration.
   - Angle: reliability diagrams per frame and per event, and selective-prediction curves, then use the score as the trigger for re-prompting or adaptation.
   - Feasibility: high.

6. **Long-horizon test-time adaptation for trackers without error cascades.**
   - Gap: TTA-VLM shows TTA buys accuracy at the cost of trustworthiness, and the work is image-only and CLIP-centric.
   - Angle: TTA restricted to confident, cycle-consistent frames (forward-backward point tracks as self-supervision) in microscopy or surgical video. Compare with no-adapt and an oracle.
   - Feasibility: medium-high.

7. **Does domain-specific pretraining pay off for microscopy dynamics, or only for statics?**
   - Gap: natural-image DINO matches OpenPhenom on Cell Painting retrieval (CellCLIP), and microscopy FM benchmarks evaluate static tasks.
   - Angle: probe DINOv3, µSAM and OpenPhenom features for temporal association and division prediction (linear or attention probes on CTC). Ask whether the temporal signal is linearly decodable.
   - Feasibility: high, using frozen features.

8. **SAM 3 concept prompts in scientific domains: where does "segment anything with concepts" fail?**
   - Gap: SAM 3 admits weak generalization to niche concepts.
   - Angle: a small taxonomy of failures (morphology terms, mitotic phases, rare organelles) with a few-shot text-to-visual exemplar bridge, plus a measure of how much synthetic data closes the gap.
   - Feasibility: high, since it is inference plus light fine-tuning.

9. **Is HOTA's per-frame assignment changing MOT conclusions?**
   - Gap: TGOSPA shows HOTA violates metric axioms and uses heuristic per-frame matching.
   - Angle: recompute DanceTrack, SportsMOT and MOT17 leaderboards (public tracker outputs) under a global-assignment metric, and count rank inversions.
   - Feasibility: very high, since it is CPU work.

10. **Pretraining overlap inflating VOS and medical benchmark scores.**
    - Gap: VLM contamination detectors are unreliable, a medical VQA audit finds about 4–20% overlap depending on the embedder, and nobody has audited SA-V/SA-1B against DAVIS, MOSE, LVOS or medical sets.
    - Angle: near-duplicate retrieval with DINOv3/SigLIP, then report the performance split between overlapping and clean subsets.
    - Feasibility: medium. Embedding about 11M SA-1B images on one H200 takes days but is doable, and subsampling works too.

11. **Calibrated uncertainty for lineage reconstruction (global consistency, not per-link scores).**
    - Gap: CHOTA notes that metrics reward local correctness and weakly reward global coherence. Methods output per-link probabilities without lineage-level uncertainty.
    - Angle: sample lineages from the association posterior (constrained decoding, ILP perturbation), then report calibrated lineage-level confidence and a "where to proofread" ranking. Biologists care because it saves proofreading time.
    - Feasibility: high.

12. **Point tracking under physics-violating appearance (specular, transparent, refractive, fluorescence bleaching).**
    - Gap: CoTracker3 struggles with reflections and shadows because of its synthetic training data.
    - Angle: a physics-based synthetic stress suite (Kubric/Blender with controlled BRDF, refraction and bleaching) that isolates which physical effect breaks which tracker. It links to the physics-based vision area, which has an award record.
    - Feasibility: medium-high, since rendering is CPU/GPU-cheap at small scale.

13. **Small and efficient trackers: how much of CoTracker3/TAPNext performance survives distillation to under 20M params for on-microscope, real-time use?**
    - Gap: these models are trained on short 256x256 clips, and long-sequence generalization fails (TAPNext++).
    - Angle: distillation plus evaluation over long horizons, reporting the accuracy-memory Pareto curve per horizon length.
    - Feasibility: medium. Distillation runs fit on one H200 over days.

**Ranking for impact and highlight potential:**
- 1+2 (or 1+11) combined into a single "topology-change tracking" paper, with a method plus a metric, is the strongest.
- 3, 4 or 5 would be strong as a "diagnosis" paper on foundation trackers.
- 9 and 10 are low-cost critique papers, likely posters unless the rank flips are dramatic.
- 12 is the most likely to appeal to the physics-based community.

### Gaps
- I did not verify AllTracker, Track-On2 or TAPNext++ venue and acceptance status. TAPNext++ was seen through alphaXiv only.
- I did not find 2025–2026 computational-photography open problems beyond the award winners. Questions in that area (e.g., 12) are extrapolations.
- I did not survey the medical-imaging (non-microscopy) open problems in depth, beyond the SAM2 surgical domain gap and the medical VQA overlap.
