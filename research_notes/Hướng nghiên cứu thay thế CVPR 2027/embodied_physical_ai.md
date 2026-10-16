# Embodied AI / Physical AI from a CV perspective: open problems for a 1×H200 CVPR 2027 submission

Scope note: as of 2026-10-13, with 34 days to the deadline (Nov 16, 2026; registration Nov 10). All GPU-day figures are **my own estimates**, not measured or sourced. I checked the existing shortlist in `reports/Ý tưởng nghiên cứu CVPR 2027.md`, and it contains **no** VLA/robot/driving idea. The nearest overlaps are #5 ("4D reconstruction as physics judge" for *video generation*), #7 (physics probe in video diffusion), and the "encoded-but-not-used" framing of #2 (MLLM arbitration). The ideas below are positioned to avoid those. Under CVPR 2027 rules, work posted after 15/9/2026 is "contemporaneous", so it must be cited but need not be compared against (from the existing report). Several directly competing papers below (2610.10912, 2610.08183) fall in that window.

## Q1. What do VLAs actually use from vision? (vision vs language vs proprioception; viewpoint/lighting)

### Takeaway
Evidence from 2025–2026 says VLAs lean heavily on **visual context and proprioceptive state** and often ignore language. They are also extremely brittle to camera viewpoint and initial-state shifts, so high LIBERO scores reflect memorized scene→action mappings. The causal *mechanism* is still open: which tokens or layers carry the shortcut, and whether the failure sits in the encoder, in fusion, or in the action head. A rollout-free way to predict robustness is also still missing.

### Cited Findings
- LIBERO-Plus (CVPR 2026) defines 10,030 perturbed tasks over 7 dimensions (camera, robot initial state, language, lighting, background, sensor noise, layout). OpenVLA drops from 76.5% to **1.1%** under camera-viewpoint change and to 4.1% under robot-initial-state change. Models "tend to ignore language instructions completely, relying more on visual context" — [LIBERO-Plus arXiv 2510.13626](https://arxiv.org/pdf/2510.13626); [CVPR 2026 open access](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf)
- LIBERO-PRO: models scoring >90% on LIBERO "nearly collapse" when object positions change or tasks are slightly modified, which the authors read as "rote recall rather than robust task understanding" — [LIBERO-PRO arXiv 2510.03827](https://arxiv.org/html/2510.03827v2)
- "Scrubbing Visual Cues" (arXiv 2610.10912, submitted **7 Oct 2026**, contemporaneous) finds that VLM backbones differ substantially in susceptibility to visual shortcuts, that shortcuts enter action representations in **early layers** and are corrected to varying degrees by later layers via language, and that a rollout-free "action margin" metric correlates with behavior. Their fix is domain-adversarial "task scrubbing". The abstract names no specific models or benchmarks — [arXiv 2610.10912](https://arxiv.org/html/2610.10912)
- "How Should VLAs Use Proprioceptive State?" (arXiv 2608.03052) cites prior work showing that fused state lets a policy treat a task as finished from internal progress while missing visual failures, can suppress visual learning around motion-phase transitions, and acts as an action-correlation shortcut — [arXiv 2608.03052](https://arxiv.org/html/2608.03052)
- ThinkProprio (2602.06575): passive state conditioning changes little. Using state tokens to *select* visual patches helps and keeps ~12% of visual tokens on CALVIN ABC→D — [arXiv 2602.06575](https://arxiv.org/abs/2602.06575)
- 2512.02902 attributes viewpoint brittleness to "misalignment in Spatial Modeling, rather than Physical Modeling". A global affine transform on visual tokens (Feature Token Modulation, ~4K params) raises LIBERO viewpoint accuracy from 48.5% to 87.1%, and FLA (4.7M params) reaches 90.8% — [arXiv 2512.02902](https://arxiv.org/abs/2512.02902v1)
- E0: spherical warping augmentation raises average success from 66.5% to 83.9%, and π0 rises from 19.7% to 50.8% (viewpoint setting) — [arXiv 2511.21542](https://arxiv.org/pdf/2511.21542)
- GeoAware-VLA swaps in a frozen VGGT encoder, on the hypothesis that cross-view generalization depends on encoder geometric quality — [arXiv 2509.14117](https://arxiv.org/html/2509.14117v3)
- Camera-robustness work masks the wrist stream to avoid a confounding visual shortcut — [arXiv 2608.06965](https://www.alphaxiv.org/abs/2608.06965)
- Modality collapse: strong visual priors drown out language. The proposed fix is Residual Semantic Steering — [arXiv 2601.04052](https://arxiv.org/html/2601.04052v1)
- CoRP: language conditioning helps only when the goal is ambiguous (LIBERO-Goal 9.2%→95.8%), and removing it slightly *improves* RoboTwin 2.0 — [arXiv 2610.08183](https://arxiv.org/html/2610.08183)
- Driving: "Does Visual Information Play a Decisive Role in VLA Driving Behavior?" (2605.31041) uses channel-, information- and structure-level visual perturbations in open-loop and closed-loop settings. It finds "evaluation-dependent dependency patterns" and uneven visual grounding. The abstract gives no numbers or model names — [arXiv 2605.31041](https://arxiv.org/html/2605.31041v1)
- Mechanistic VLA work: Häon et al. (CoRL 2025) project FFN activations onto the token-embedding basis, find sparse semantic directions (speed, direction) that causally affect actions, and steer π0/OpenVLA in LIBERO and on a UR5 — [PMLR v305](https://proceedings.mlr.press/v305/haon25a.html); [GitHub](https://github.com/vla-mech-interp/mechanistic-steering-vlas). Follow-ups: SAEs on π0.5 — [arXiv 2603.19183](https://arxiv.org/html/2603.19183v1); "mech-interp of VLAs remains significantly underexplored" — [arXiv 2603.05487](https://arxiv.org/pdf/2603.05487); VLA-Trace — [arXiv 2605.30117](https://arxiv.org/pdf/2605.30117)

### Inferences
- The *phenomenon* (brittleness, ignoring language, state shortcuts) is well documented, so a new benchmark of perturbations would not be novel. The open part is a **unified causal attribution**: for a given failure, how much is due to vision features, proprio state, or language? Does it transfer across OpenVLA-OFT / π0 / π0.5 / SmolVLA / GR00T? 2610.10912 does part of this (visual shortcuts plus layers) and must be cited and differentiated.
- The 2512.02902 result (an affine map on tokens fixes most of the viewpoint loss) strongly hints at an "encoded but misaligned" story: the information survives the viewpoint shift but sits in a shifted subspace. This suggests a test-time / unsupervised fix is plausible. The paper's FTM appears to need training on the target view, which needs checking.

### Gaps
- I could not get the per-model numbers or the action-margin definition from 2610.10912 or 2608.03052, since only abstracts were read.
- I found no paper that directly tests proprio-shortcutting *under controlled visual perturbations* across several open VLAs. The search suggests this is open but did not confirm it.
- Lighting/background effects are reported as smaller than camera/state effects in LIBERO-Plus, but I did not extract exact numbers.

## Q2. Which vision-encoder properties predict policy success (DINOv2/v3 vs SigLIP vs video encoders)?

### Takeaway
The question is **open and the evidence is contradictory**. Rankings flip with the policy architecture, the benchmark (LIBERO-Long vs others), and frozen vs fine-tuned settings. No paper offers a cheap, pre-training *predictor* (a property measured on the encoder) that explains policy success across encoders. 2610.08183 points to "fine-grained, compressed" features, but only as an argument from one compact policy.

### Cited Findings
- Patch Policy: across five frozen encoders (incl. SigLIP 2, V-JEPA 2), "DINOv2 and WebSSL are the most effective vision backbones for robot learning" — [arXiv 2607.18236](https://arxiv.org/pdf/2607.18236)
- JEPA-VLA: on LIBERO-Long, encoder swap gives V-JEPA 2 41.0%, SigLIP 33.6%, DINOv2 ~26%; "adapting DINOv2 yields marginal or even negative gains" — [arXiv 2602.11832](https://arxiv.org/pdf/2602.11832). This **contradicts** Patch Policy's ranking.
- CoRP (2610.08183, 6 Oct 2026): a 48.9M-param policy reaches 97.0% on LIBERO. Random ViT-S drops to 78.1% and ImageNet ResNet-34 to 74.5%. Freezing the encoder costs 19.8 points. Resampling to 48 tokens/view gives 97.0% vs 83.2% with all patch tokens. The claim is that the "representation drives most performance; parameter scale and generative priors are largely incidental" — [arXiv 2610.08183](https://arxiv.org/html/2610.08183)
- Probing study: OpenVLA, π0.5, DINOv2 and CLIP features all linearly predict Monte-Carlo success outcomes on LIBERO-Goal. "Four families… surface a value-like signal at comparable strength" — [arXiv 2605.28527](https://arxiv.org/html/2605.28527v1)
- A world-model paper found that DINO features performed "significantly worse" than SigLIP because they are not aligned with the PaliGemma backbone, so alignment with the LLM matters in VLAs — (from search snippet; primary paper not identified, so treat as unverified)
- DynaFLIP (dynamics-guided encoder) is claimed to beat DINOv2/SigLIP by 22.5% under real OOD. Seen only via an aggregator — [Digg summary](https://digg.com/ai/3ko5q5o9) (unverified)
- LightVLA argues VLM token-pruning transfers poorly to VLAs because "robotic tasks depend more on local detail than global semantics" — [arXiv 2509.12594](https://arxiv.org/html/2509.12594v2)

### Inferences
- Plausible predictors to test: dense correspondence / keypoint accuracy (fine-grained), viewpoint-equivariance of patch features, temporal-difference sensitivity (video encoders), and LLM-alignment (CKA to the VLM's text space). Each comparison above varies a different one of these, which could reconcile the contradictory rankings. That reconciliation would be the "insight".
- The cheapest protocol is frozen encoder + small policy head (CoRP/Patch-Policy style) on LIBERO + LIBERO-Plus, not full VLA retraining.

### Gaps
- No primary source found for DINOv3 in robot policies. Whether DINOv3 has been benchmarked for VLAs is unknown from this search.
- I did not verify seed counts and variance in the encoder comparisons, so contradictions may partly be noise.

## Q3. Failure modes of action-conditioned / driving world models used as evaluators

### Takeaway
The failure modes are now named: action-ignoring on off-expert actions, "optimism bias" (predicting success under failing actions), object-permanence hallucination, and autoregressive drift. What is still missing is **quantifying how much each failure corrupts policy *rankings*** against a ground-truth oracle. A simulator such as LIBERO/RoboCasa, where the true success is free, is the natural oracle. On the driving side, no study tests whether generative world-model rollouts predict closed-loop scores. Pseudo-simulation (NAVSIM v2) is validated against nuPlan, not against generative WMs.

### Cited Findings
- WorldEcho / WorldSync (2608.24885, 25 Aug 2026, cs.RO+cs.CV): world models follow expert actions reasonably well but on off-expert trajectories "either ignore the commanded action or produce visually invalid rollouts". Metrics are visual integrity and SE(3) trajectory alignment. The fix trains on broader action coverage, an Action-Forcing Expert, and intervention-effect alignment — [arXiv 2608.24885](https://arxiv.org/html/2608.24885)
- MiraBench names **Optimism Bias**: the model predicts success under failure-inducing actions, which the authors attribute to success-dominated training data — [arXiv 2605.29360](https://arxiv.org/html/2605.29360)
- "Scalable Policy Evaluation with Video World Models" (Nov 2025) reports object-permanence and inconsistent-object-behavior artifacts that bias evaluation, plus error accumulation in AR rollouts — [arXiv 2511.11520](https://arxiv.org/html/2511.11520v3)
- Survey: predicted futures can be plausible but not causally tied to the action — [arXiv 2605.00080](https://arxiv.org/html/2605.00080v1)
- JEPA world models diagnosed via action-conditioned predictive consistency — [arXiv 2608.12939](https://arxiv.org/pdf/2608.12939)
- Pseudo-simulation (CoRL 2025): R²=0.8 vs nuPlan closed-loop across 83 planners (best open-loop metric R²=0.7) at 6× less compute. The authors list correlation with real-world deployment as a limitation — [Pseudo-Simulation paper](https://tisl.cs.utoronto.ca/publication/Pseudo-Simulation_for_Autonomous_Driving/Pseudo-Simulation_for_Autonomous_Driving.pdf); [NAVSIM GitHub](https://github.com/autonomousvision/navsim)
- DrivingGen (ICLR 2026) benchmarks 14 generative driving WMs: general models look better but break physics, while driving-specific ones get motion right but lag visually — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009746)
- Bench2Drive-R builds reactive closed-loop evaluation with a generative renderer and a separate behavior controller — [arXiv 2412.09647](https://arxiv.org/html/2412.09647v1)
- Sim-to-real validity of sim benchmarks: RobotArena∞ validated correlation on only one task with three policies — [arXiv 2510.23571](https://arxiv.org/pdf/2510.23571). A June 2026 study questions whether simulators preserve real-world rankings — [chatpaper summary](https://chatpaper.com/paper/297783) (findings unverified)

### Inferences
- A clean CV-flavoured contribution: use a simulator as the oracle to measure **ranking fidelity** (Kendall τ) of a world-model evaluator over many policy checkpoints. Then decompose the error into action-ignoring, optimism bias, and drift using counterfactual action perturbations. Finally, propose a training-free correction, e.g. a consistency check between rollouts under perturbed actions, or calibrating success with a vision success-detector. This avoids training a WM from scratch only if an open action-conditioned WM can be fine-tuned cheaply on LIBERO data. That is the main feasibility risk.
- This differs from existing shortlist #5, which judges physics in *text-to-video*. Here the question is whether an action-conditioned evaluator ranks *policies* correctly.

### Gaps
- I did not identify which open action-conditioned robot WM checkpoints (e.g. Cosmos-Predict variants, Ctrl-World, WorldGym) are runnable or fine-tunable on 1 H200, so this needs verifying before committing.
- No 2026 paper found that measures generative-driving-WM ranking fidelity against CARLA/Bench2Drive scores. That points to a gap, but my search was shallow ("WorldLens" was seen only via a summary).

## Q4. Insight-driven embodied papers recognized at CVPR/ICCV/CoRL/RSS 2025–2026

### Takeaway
The confirmed exemplar is **Navigation World Models (CVPR 2025 Best Paper Honorable Mention)**, a conceptually simple world model trained on existing video and used for planning, with no new robot hardware. Analysis/benchmark-style VLA work (LIBERO-Plus) is accepted at CVPR 2026 as a main-track paper. I could not verify CoRL 2025 / RSS 2025 main-award winners.

### Cited Findings
- Navigation World Models (Bar, Zhou, Tran, Darrell, LeCun) won the CVPR 2025 Best Paper Honorable Mention. VGGT won Best Paper, and MegaSaM was also an HM — [CVPR 2025 Best Papers](https://cvpr.thecvf.com/Conferences/2025/BestPapersDemos); [Amir Bar on X](https://x.com/_amirbar/status/1933639166652186727)
- LIBERO-Plus (a robustness *analysis* of VLAs on simulated perturbations) appears in CVPR 2026 open access — [CVF open access](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf)
- CoRL 2025 accepted "Mechanistic Interpretability for Steering VLAs" (analysis + training-free steering) — [PMLR v305](https://proceedings.mlr.press/v305/haon25a.html), and "Pseudo-Simulation for Autonomous Driving" (an evaluation-methodology paper) — [NAVSIM GitHub](https://github.com/autonomousvision/navsim)
- CVPR 2026 Best Paper went to D4RT (dynamic-scene reconstruction), not to an embodied paper (from the existing report's sourcing: [CVPR 2026 Closing](https://cvpr.thecvf.com/Conferences/2026/News/Closing))
- CoRL 2025 side awards found: VAMOS (Madrona Prize runner-up) — [UW](https://robotlearning.cs.washington.edu/awards/); DataMIL (workshop best paper) — [UT RobIn](https://robin-lab.cs.utexas.edu/news/announcement_44/). These are workshop/side awards, not main-track.
- SpecPrune-VLA (training-free token pruning) was accepted at ICML 2026, so efficiency papers do get into top venues — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/64525)

### Inferences
- The NWM recipe shows that a CV-native reframing ("navigation = conditional video prediction + planning by search") can win at CVPR without real-robot experiments. A 1-GPU lab can borrow the *style* (simple reframing plus clear evidence) but not the compute; NWM trained a ~1B CDiT model.
- For CVPR (not CoRL), analysis papers that cover many open checkpoints and come with a cheap fix seem to be the viable lane.

### Gaps
- CoRL 2025 and RSS 2025 main best-paper winners were not found in my searches, so this needs a direct check of corl.org / roboticsconference.org.
- ICCV 2025 embodied award/oral papers were not found. CVPR 2026 embodied orals/highlights were not enumerated.

## Q5. Is simulation-only evaluation accepted at CVPR?

### Takeaway
Yes, for analysis/benchmark/evaluation papers. LIBERO-Plus (CVPR 2026) is LIBERO-based simulation, NWM is offline video plus planning, and NAVSIM/pseudo-sim is data-driven. Method papers that claim robot capability more often add real-robot results (WorldSync, GeoAAC, 2603.05868 all include real robots), so a sim-only *method* paper is weaker. Framing as "analysis + diagnostic + training-free fix" reduces this risk.

### Cited Findings
- LIBERO-Plus evaluates zero-shot transfer from standard LIBERO training to simulated perturbations and was published at CVPR 2026 — [arXiv 2510.13626](https://arxiv.org/pdf/2510.13626); [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf)
- Method papers commonly add real-robot validation: WorldSync (RoboTwin + real robot) — [arXiv 2608.24885](https://arxiv.org/html/2608.24885); GeoAAC (real-world 53.3%→74.4%) — [alphaxiv 2609.20776](https://www.alphaxiv.org/abs/2609.20776); 3D-aware viewpoint method (LIBERO + real cameras) — [arXiv 2603.05868](https://arxiv.org/pdf/2603.05868)
- Sim-real correlation is itself contested. SimplerEnv's original validation shows strong correlation, but third-party summaries put it at ~0.7–0.8 with some rank flips — [claru summary](https://claru.ai/benchmarks/simpler-env) (secondary). RobotArena∞ validated only one task — [arXiv 2510.23571](https://arxiv.org/pdf/2510.23571)

### Inferences
- The defensible move is to make **every claim relative** ("model X's success depends on Y, measured in sim") rather than absolute capability claims. Include ≥4 open VLAs, and if possible one SimplerEnv (real-to-sim) check to pre-empt "LIBERO-only" criticism.

### Gaps
- No CVPR policy statement on sim-only robotics work exists. This is inferred from accepted papers only.

## Q6 (Objective). 10 concrete open research questions for 1×H200, CVPR 2027

Checkpoints assumed available, from my knowledge of the repos (not re-verified in this session): OpenVLA-7B and OpenVLA-OFT LIBERO checkpoints ([openvla-oft](https://github.com/moojink/openvla-oft)); π0 / π0-FAST / π0.5 with LIBERO/DROID fine-tunes ([openpi](https://github.com/Physical-Intelligence/openpi)); SmolVLA-450M ([LeRobot](https://github.com/huggingface/lerobot)); GR00T N1.5 ([Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)). GR00T N1.5 and π0.5 running on LIBERO is confirmed by [GeoAAC](https://www.alphaxiv.org/abs/2609.20776). LIBERO rollouts are largely CPU/MuJoCo-bound, so GPU-days overstate the bottleneck. Budget CPU cores too (my note).

| # | Claim-style title | Gap (evidence) | Simple angle | Setup | GPU-days (my est.) | Main risk |
|---|---|---|---|---|---|---|
| E1 | **"Who is driving the arm? Causal modality attribution in VLAs"** | Ignoring language (LIBERO-Plus), state shortcuts (2608.03052), visual shortcuts (2610.10912), but no cross-model attribution that predicts which perturbation hurts which model | Counterfactual input swaps: replace the image / state / instruction with one from another episode at time t, then measure the action delta (open-loop) and success (closed-loop). Attribution scores per model and phase. Show they predict LIBERO-Plus drop per model without rollouts. Training-free fix: state-dropout or classifier-free-style modality reweighting at inference | 4–5 open VLAs; LIBERO + LIBERO-Plus | 3–6 | 2610.10912 (contemporaneous) overlaps on visual shortcuts and rollout-free metric. Must differentiate on the 3-modality decomposition + cross-model breadth |
| E2 | **"Viewpoint failure is a coordinate problem, not a perception problem"** | Affine token modulation recovers 48.5→87.1% (2512.02902); VGGT encoder (GeoAware-VLA); spherical warping (E0). Nobody shows *where* the information is lost | Linear-probe object/EE 3D position and camera-frame direction from visual tokens and layers under view shift. If decodable but the action is wrong, the failure is in the frame mapping. Fix: **unsupervised test-time token alignment** (match feature statistics/CORAL on a few unlabeled frames from the new camera), no labels or demos | OpenVLA-OFT, π0.5, SmolVLA; LIBERO-Plus camera split | 2–5 | Framing resembles shortlist #2 ("encoded but not used"). Must check whether 2512.02902's FTM already works unsupervised |
| E3 | **"What makes a visual representation good for control? Reconciling contradictory encoder rankings"** | DINOv2 best (Patch Policy) vs V-JEPA 2 best on LIBERO-Long (JEPA-VLA) vs "comparable" (2605.28527) vs fine-grained matters (2610.08183) | Measure 4–5 cheap encoder properties (dense correspondence, view-equivariance, temporal sensitivity, effective rank, LLM-alignment) for ~8 frozen encoders (DINOv2/v3, SigLIP2, V-JEPA 2, CLIP, VGGT, RADIO, Theia). Train a small fixed policy head on each, regress success on the properties, then show the one predictor that explains the flips | Compact policy on LIBERO-4 suites + LIBERO-Plus; 3 seeds | 8–15 | Over budget if too many encoders. Kin of shortlist #8 (REPA predictor), different domain. Cut to 6 encoders × 2 suites to fit ≤10 |
| E4 | **"World-model evaluators rank policies wrong — and simulation tells you why"** | WorldEcho action-ignoring (2608.24885), optimism bias (MiraBench 2605.29360), hallucination (2511.11520); no ranking-fidelity audit with a ground-truth oracle | Use LIBERO/RoboCasa as the oracle. Generate N policy checkpoints of varied quality, score them in the WM and in sim, report Kendall τ. Counterfactual action perturbations decompose the error. Training-free fix: action-sensitivity consistency check / success-detector calibration | Requires an open action-conditioned WM fine-tunable on LIBERO | 8–15 | Feasibility of the WM (gap in Q3). Contemporaneous WM-eval papers |
| E5 | **"VLAs know they are failing under shift — but act anyway"** | Frozen VLAs linearly encode success (≈92% ordering for π0.5; 2605.28527); confidence calibration studied (2507.17383); not tested OOD (LIBERO-Plus shifts) | Train probes in-distribution and test whether the success signal survives camera/light/state shifts. Use it for training-free abstention/re-planning (selecting among sampled action chunks), and report risk–coverage curves | π0.5, OpenVLA-OFT, SmolVLA; LIBERO-Plus | 2–4 | If the probe also collapses OOD, the paper becomes a negative result (still publishable if framed as "internal value is as brittle as policy") |
| E6 | **"Is your VLA gain real? Statistical power of LIBERO-style evaluation"** | LIBERO saturated at 95–98% ([LIBERO-Plus](https://arxiv.org/pdf/2510.13626)); many 2026 papers report sub-2-point deltas (e.g. GeoAAC 95.5–98.0%) | Bootstrap/IRT over rollouts and published tables: per-task discriminability, minimum episodes for significance, and how many published claims fall within noise. Re-run 4 open checkpoints with 5 seeds | LIBERO, LIBERO-Plus, SimplerEnv | 1–3 (mostly CPU) | Evaluation-methodology papers can land as poster only. Needs a striking headline number |
| E7 | **"Proprioception tells the policy it is done"** | State lets policies declare completion while missing visual failure (cited in 2608.03052). ThinkProprio shows passive state is barely used | Phase-resolved analysis: inject visual failure (object slips/displaced mid-episode) and measure whether the policy reacts, with vs without state. Fix: test-time state dropout or vision-gated state | π0/π0.5 (state input), SmolVLA, GR00T | 2–4 | Strong overlap with 2608.03052. Best merged into E1 |
| E8 | **"Driving VLAs (and E2E planners) under visual counterfactuals on NAVSIM v2"** | 2605.31041 finds uneven visual grounding but no numbers. Ego-status shortcut in open-loop nuScenes ("Is Ego Status All You Need", [arXiv 2312.03031](https://arxiv.org/abs/2312.03031)). NAVSIM v2 pseudo-sim is validated (R²=0.8) | Blind/ego-only baselines + camera counterfactuals (mask lead vehicle, swap traffic light) on NAVSIM v2. Measure EPDMS sensitivity per scene category | Open NAVSIM planners (TransFuser etc.) + any open driving VLA | 4–8 | NAVSIM data size/setup time; driving VLAs open checkpoints uncertain |
| E9 | **"Spatial intelligence that helps manipulation: do 3D foundation features beat 2D under view shift, or is it just augmentation?"** | GeoAware-VLA (VGGT) vs E0 augmentation vs FTM: not compared under one protocol | Controlled 2×2: {2D encoder, VGGT/π3 features} × {no aug, view aug} on LIBERO-Plus camera. Isolate geometry vs invariance | Small policy head; LIBERO-Plus | 4–8 | Can collapse into a "method comparison" without insight. Pair with E2's probing |
| E10 | **"Visual token budgets reveal what VLAs need: task-relevant pixels are <10% "** (analysis, not a new pruner) | Pruning saturated (SpecPrune, VLA-Pruner, LightVLA, VLA-Cache). CoRP 48 tokens > all tokens; ThinkProprio ~12% tokens | Use pruning as a *probe*: the minimal token set that preserves actions, then compare it with object masks / gaze. The finding links to E1 (which pixels the policy uses) | Same VLAs as E1 | 1–3 | Reviewers may see it as another pruning paper. Keep it as an E1 sub-analysis |

Recommended ranking for Nov 16 (my judgment): **E1 (+E7 folded in) or E2** as the main bet (≤6 GPU-days, inference + probes, many open checkpoints, clear claim). **E5** as a cheap companion. **E6** as a CPU side-project. **E3/E4** are better suited to a later venue (ECCV 2027 / CoRL 2027) unless the WM feasibility check passes quickly.

## Q7. What is saturated or infeasible for a 1-GPU lab

### Takeaway
Avoid new VLA architectures/pretraining, LIBERO-SOTA chasing, token pruning/caching, action-chunking heads, yet another perturbation benchmark, and training video/driving world models.

### Cited Findings
- LIBERO in-distribution is saturated at ~95–98% — [LIBERO-Plus](https://arxiv.org/pdf/2510.13626). A 48.9M-param compact policy hits 97.0% — [arXiv 2610.08183](https://arxiv.org/html/2610.08183)
- Token pruning/caching is crowded: SpecPrune-VLA (ICML 2026), VLA-Cache, VLA-Pruner, LightVLA, EfficientVLA, SP-VLA — [SpecPrune](https://arxiv.org/pdf/2509.05614); [VLA-Pruner](https://arxiv.org/pdf/2511.16449v1); [LightVLA](https://arxiv.org/html/2509.12594v2)
- Action chunking / action-expert design is crowded: GeoAAC, AR-VLA (RSS 2026), StreamingVLA, Guided Action Flow, adaptive chunk length — [GeoAAC](https://www.alphaxiv.org/abs/2609.20776); [AR-VLA](https://arvla.insait.ai/); [StreamingVLA](https://arxiv.org/pdf/2603.28565v1); [2607.02092](https://arxiv.org/abs/2607.02092); [2604.04161](https://arxiv.org/html/2604.04161v1)
- Perturbation benchmarks already exist (LIBERO-Plus, LIBERO-PRO) and world-model benchmarks too (WorldEcho, MiraBench, DrivingGen with 14 models) — sources as in Q1/Q3
- VLA mech-interp is growing (Häon CoRL 2025, SAE 2603.19183, 2603.05487, VLA-Trace 2605.30117, 2605.00321) and is no longer empty, though still described as underexplored — [2605.00321](https://arxiv.org/pdf/2605.00321)

### Inferences
- Infeasible on 1 H200 in 34 days (my estimates): pretraining or full fine-tuning a 3–7B VLA on OXE/DROID-scale data; training an action-conditioned video WM (NWM-scale ~1B CDiT or Cosmos-scale); NAVSIM SOTA planner chasing with large backbones; any claim requiring real-robot experiments (no robot assumed).
- Feasible: inference, rollouts, probes and LoRA-scale fine-tunes of ≤3B models; small policy heads on frozen encoders; LIBERO/LIBERO-Plus/SimplerEnv rollouts (CPU-heavy). For an H200 (141 GB), memory is not the constraint. Rollout throughput and CPU cores are.

### Gaps
- I did not verify per-episode rollout time for each VLA on LIBERO, so GPU-day estimates could be off by 2×.
- I did not check whether driving VLAs (e.g. open checkpoints on NAVSIM) are publicly released, which affects E8.
