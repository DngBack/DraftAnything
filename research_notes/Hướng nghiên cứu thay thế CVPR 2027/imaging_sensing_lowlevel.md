# Physics-based vision, computational imaging, unconventional sensing, and low-level vision with generative priors: open problems for a 1×H200 CVPR 2027 paper

Scope note: the existing shortlist in `reports/Ý tưởng nghiên cứu CVPR 2027.md` covers LineageRAFT, MLLM "arbitration", RL post-training readout, SAM 2 memory error propagation, 4D-reconstruction physics judging, calibration of feed-forward 3D, video-diffusion physics probes, and REPA/diffusability. **None of them is in imaging or sensing**, so nothing below duplicates them. The one adjacent item is shortlist #6 (post-hoc calibration of feed-forward 3D). Idea Q3 below is calibration of *restoration* posteriors, a different object, so the two should not be pitched as the same contribution.

Timing: today is 2026-10-13 and the CVPR 2027 deadline is 2026-11-16, so about 34 days remain. **All GPU-day figures are my own estimates** for one H200 running inference or light fine-tuning on public checkpoints. None of them comes from a source.

---

## Key question 1: Which award or oral papers in this area (2024–2026) were low-compute, and what made them win?

### Takeaway
The imaging and sensing papers that won top awards from 2024 to 2025 were cheap in compute but expensive in **idea plus physical prototype**. Each one introduced a new image formation model (freeform pixels as an NN layer, per-pixel focus, time-resolved radiance caching) and validated it on real captures. The headline CVPR 2026 awards went to large 3D generation and reconstruction models, not imaging. Imaging still had a dedicated oral session in 2026, and those orals were mostly physics-model papers: a thermal emission/reflection model, a metasurface camera, a polarization-cue 3DGS method, ISP compensation, and optical watermarking.

### Cited Findings
- ECCV 2024 Best Paper was "Minimalist Vision with Freeform Pixels" (Klotz and Nayar, Columbia). The camera's freeform pixels (photodetector plus optical mask) are modeled as the first layer of a neural network and trained per task. Prototypes with **8 pixels** handled room monitoring, lighting estimation, and traffic flow, with performance comparable to conventional cameras that have far more pixels, plus a privacy benefit — [Columbia CS](https://www.cs.columbia.edu/2024/outstanding-research-and-best-paper-honors-at-eccv-2024); [ECCV 2024 awards](https://eccv2024.ecva.net/virtual/2024/awards_detail)
- ICCV 2025 Best Paper Honorable Mention went to "Spatially-Varying Autofocus" (Qin, Sankaranarayanan, O'Toole, CMU). It is a new camera that autofocuses different pixels to different depths, giving an arbitrary focal surface while keeping a large aperture and full resolution. The same year's Marr Prize went to "Generating Physically Stable and Buildable Brick Structures from Text" — [IEEE TCPAMI ICCV awards](https://tc.computer.org/tcpami/awards/iccv-paper-awards/); [CMU Graphics](https://graphics.cs.cmu.edu/)
- CVPR 2025 Best Student Paper was "Neural Inverse Rendering from Propagating Light" (Malik et al.), described as the first physically based neural inverse rendering system for multi-view videos of propagating light, built on a time-resolved extension of neural radiance caching — [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Malik_Neural_Inverse_Rendering_from_Propagating_Light_CVPR_2025_paper.html); [arXiv 2506.05347](https://arxiv.org/abs/2506.05347). I could not find GPU or training-time figures for it.
- CVPR 2026 headline awards: Best Paper was D4RT (dynamic scene reconstruction), Best Student Paper was TRELLIS.2 (structured 3D latents), and Best Student Paper HM was ChordEdit (training-free one-step editing). SAM 3D and NitroGen received HMs. No computational imaging paper appeared among the headline winners — [CVPR 2026 Best Papers](https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers); [CVF awards](https://www.thecvf.com/?page_id=413)
- CVPR 2026 orals in the imaging and physics area included:
  - "Dual Band Thermal Videography: Separating Time-Varying Reflection and Emission Near Ambient Conditions" (Narayanan, Ramanagopal, Narasimhan)
  - "MetaSpectra+" (metasurface snapshot hyperspectral camera)
  - "NOWA: Null-space Optical Watermark…"
  - "Towards Photorealistic and Efficient Bokeh Rendering via Diffusion Framework"
  - "UnReflectAnything: RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision"
  - "PhyGaP: Physically-Grounded Gaussians with Polarization Cues"
  - "PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction"
  - "GLINT" (scene-scale transparency)
  - SEM microstructure surface reconstruction
  - Relighting datasets OLATverse and POLAR
  - "Streaming Diffusion Model for Fast Infrared and Visible Video Fusion"

  Source: [CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral). I read about 200k of the page's 238k characters, so the list may be incomplete. The portion I read had **no event-camera, SPAD, transient, low-light, deblurring, or SR oral**.
- "Guided Lensless Polarization Imaging" (Kraicer, Yosef, Giryes) appears in the CVPR 2026 programme. One listing puts it in a Findings poster session, so its format is unclear — [arXiv 2603.27357](https://arxiv.org/pdf/2603.27357)
- gQIR (generative quanta image reconstruction with latent diffusion from binary SPAD frames) is a CVPR 2026 paper — [paper note](https://en.papernotes.org/CVPR2026/image_generation/gqir_generative_quanta_image_reconstruc_tion/)

### Inferences
- What these winners share is that **the sensor or forward model is the contribution**, explained by one picture (8 pixels; a focal surface that bends; light visibly propagating). Compute was modest, but every one of them had a lab-built prototype or capture rig.
- The 2026 imaging orals show a second winning pattern that needs **no new hardware**: a sharp physical model of an existing sensor. Examples are separating emission from reflection in LWIR, ISP photometric compensation in radiance fields, and a null-space optical watermark. This is the pattern a 1-GPU team can copy.
- Restoration and SR orals seem to have disappeared from the visible oral list. That fits saturation (see Key question 5).

### Gaps
- I found no ICCP 2025 or 2026 award list (searches returned unrelated conferences). The writer should not claim anything about ICCP awards.
- I did not find compute details (GPU-hours) for any of the award papers. "Low-compute" is my inference from their nature (per-scene optimization, small networks), not a sourced figure.
- I did not confirm whether CVPR 2026 had an imaging paper in the award *candidate* list.

---

## Key question 2: Open problems for event and SPAD cameras with foundation models (DINOv3, SAM, VGGT): can they be adapted, and what fails?

### Takeaway
2025–2026 work adapts VFMs to events in three ways: through an event-to-video reconstruction proxy, by feature distillation (DINOv2 alignment, VGGT distillation), or with SAM adapters. All three are **method papers that report gains. None explains which information VFMs are missing on events or on photon-starved data, or when they break.** For SPAD, the field moved quickly in 2026 (gQIR, QuantaFlow, probabilistic events, noise models, recovery bounds), yet nobody has a systematic "how many photons does a frozen VFM need" analysis. That is the clearest open niche, and it can be simulated exactly from public video.

### Cited Findings
- "Temporal-Guided Visual Foundation Models for Event-Based Vision" retrains event-to-video models on real data, feeds the reconstructions to transformer VFMs, and adds long-range temporal attention. It reports +16% segmentation, +21% depth, and +16% detection over existing methods. It states that the potential of image-pretrained VFMs "remains under-explored for event-based vision" — [arXiv 2511.06238](https://arxiv.org/abs/2511.06238)
- EventVGGT (ECCV 2026) distills VGGT's spatio-temporal and multi-view priors into an event depth model. It argues that earlier VFM-distillation methods treat event streams as independent frames, ignore temporal continuity, and therefore give temporally inconsistent depth. It names the RGB–event modality gap as the central challenge — [arXiv 2603.09385](https://arxiv.org/abs/2603.09385)
- Generative Event Pretraining aligns an event encoder to frozen DINOv2 (MSE plus InfoNCE) and then adds autoregressive modeling. It reports beating DINOv2 on the event benchmarks tested, with 24 training epochs — [arXiv 2603.23032](https://arxiv.org/html/2603.23032v1)
- EZSR (CVPR 2025) does CLIP-guided zero-shot event recognition and reaches 47.84% zero-shot on N-ImageNet with ViT-B/16 — [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_EZSR_Event-based_Zero-Shot_Recognition_CVPR_2025_paper.html)
- SAM-based event methods include SAM-Event-Adapter (ICRA 2024), EvSAM (TOMM 2025), and SEAL "Segment Any Events with Language" (listed as ICLR 2026). There is also a 2026 survey, "Event Camera Vision in the Era of Large Models". This is a secondary listing — [awesome-event-camera-vision](https://github.com/worldbench/awesome-event-camera-vision); survey of event representation learning: [arXiv 2606.23078](https://arxiv.org/pdf/2606.23078)
- Match-Any-Events does zero-shot, motion-robust wide-baseline feature matching for events — [arXiv 2604.18744](https://arxiv.org/pdf/2604.18744)
- SPAD/quanta work in 2026:
  - "Quanta Perception as Probabilistic Events" (Sundar, Thodima, Jungerman, Gupta; Aug 2026) proposes a recursive Bayesian representation of photon streams. It reports pose estimation of a running person at about 0.05 lux and more than 50k quanta frames per second "without retraining vision models", and says raw quanta streams exceed real-time compute and latency budgets "by orders of magnitude" — [arXiv 2608.27584](https://arxiv.org/abs/2608.27584)
  - QuantaFlow ("Optical Flow from Photons") claims to be the first to estimate dense optical flow directly from SPAD binary sub-streams — [arXiv 2608.00499](https://arxiv.org/html/2608.00499v1)
  - SPAD noise modeling and recovery bounds — [arXiv 2608.00489](https://arxiv.org/pdf/2608.00489); [arXiv 2601.07599](https://arxiv.org/pdf/2601.07599)
  - gQIR (CVPR 2026) applies a latent diffusion prior to sparse binary detections at 10k–100k fps — [paper note](https://en.papernotes.org/CVPR2026/image_generation/gqir_generative_quanta_image_reconstruc_tion/)
- Hardware momentum: Singular Photonics launched the Litavis SPAD sensor with in-pixel processing in September 2026 — [The Quantum Insider](https://thequantuminsider.com/2026/09/10/singular-photonics-launches-litavis-spad-image-sensor/)
- Event plus RAW low-light diffusion: NEC-Diff fuses events with RAW frames for extreme darkness — [arXiv 2603.20005](https://arxiv.org/html/2603.20005v1)

### Inferences
- **What likely fails (hypotheses to test, not established facts):**
  1. Events encode temporal log-intensity *changes*, so they carry little absolute intensity or low-frequency content. Semantic VFM features (DINO, CLIP) should therefore degrade more than geometric ones (VGGT, matching). The literature's reliance on E2VID proxies and temporal attention is consistent with this, but no paper isolates it.
  2. Event density depends on motion. A static camera on a static scene produces no input, so VFM performance should depend on speed. Nobody reports VFM accuracy as a function of the motion or contrast threshold.
  3. For SPAD, frozen VFMs face a photon-count/motion-blur trade-off, set by how many binary frames you sum. The optimum probably differs per VFM and per task, and generative reconstruction (gQIR-style) may *hurt* downstream tasks through hallucination. That would link SPAD work directly to the restoration-hallucination debate.
- Both event and SPAD data can be simulated **exactly** from public high-fps video: SPAD frames are Bernoulli draws with p = 1 − exp(−flux), and events can be produced with ESIM or v2e. Real public sets such as DSEC, MVSEC, and N-ImageNet exist for events. A purely analytic paper is therefore feasible on 1 GPU.
- Scoop risk is high for the SPAD side: the Gupta/Wision lab has published three 2026 quanta papers. The paper should be framed as an *analysis across many VFMs*, not as a new SPAD method.

### Gaps
- I found no paper that systematically measures DINOv3, SAM 2/3, or VGGT zero-shot on raw event or SPAD representations as a function of motion, contrast threshold, or photon count. This absence of evidence is from limited search and should be re-checked before committing.
- I could not confirm whether "Quanta Perception as Probabilistic Events" evaluates several downstream VFMs. Only the abstract was read.
- Public real SPAD datasets (sizes and licences) were not verified in this search.

---

## Key question 3: Diffusion-based restoration: measurable hallucination, the perception–distortion trade-off, and flawed metrics. What problems are open?

### Takeaway
By 2026 the community agrees that hallucination exists and that PSNR, LPIPS, and NR-IQA do not capture it. The proposed fixes are either an **MLLM judge** (Hallucination Score), VLM sample selection, **frequency-based** tests in medical imaging (sFRC), or **sampler tweaks** (RPU, uncertainty-guided SR). None of these uses the one object a physics-minded paper can exploit: **the null space of a known forward operator.** Also, the deterministic one-step restorers that now dominate challenges have no posterior to sample from, so they cannot express uncertainty at all.

### Cited Findings
- Blau and Michaeli proved that distortion and perceptual quality conflict, so a single PSNR number is fundamentally misleading — [alphaXiv 1711.06077](https://www.alphaxiv.org/abs/1711.06077)
- "Bridging the Perception Gap in Image SR Evaluation" (2025) found that full-reference metrics can rank methods **opposite to human preference** across datasets, because they reward blurry outputs near the posterior mean — [arXiv 2503.13074](https://arxiv.org/html/2503.13074v3)
- Hallucination Score (Ren et al., Samsung Toronto; v2 Jan 2026) uses an MLLM to score hallucinated elements in generative SR. The authors state that hallucinations are "orthogonal to both exact fidelity and no-reference quality". They also propose differentiable proxies and use them as rewards to fine-tune diffusion SR — [arXiv 2507.14367](https://arxiv.org/abs/2507.14367)
- VLM-based selection of trustworthy diffusion SR samples argues that PSNR, LPIPS, and DISTS do not capture information fidelity — [arXiv 2506.20832](https://arxiv.org/html/2506.20832)
- "Hallucination-Aware Diffusion Sampling for Inverse Problems via Robust Prior Updates" (Jin et al., Jun 2026) attributes hallucination to the *prior-update* step of DPS-type solvers and adds a stability check and re-anchoring step. It defines "measurement-conditioned hallucination" but **has no dedicated automatic hallucination metric**: it falls back on PSNR, LPIPS, and human studies — [arXiv 2606.02331](https://arxiv.org/abs/2606.02331)
- sFRC (Mar 2026) detects hallucinated features in CT super-resolution, sparse-view CT, and subsampled MRI, and measures hallucination rates under distribution shift — [arXiv 2603.04673](https://arxiv.org/html/2603.04673v1)
- Other uncertainty-guided faithful SR work: UGDiff (latent uncertainty) — [arXiv 2608.25998](https://arxiv.org/pdf/2608.25998); CVPR 2025 uncertainty-guided perturbation — [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Uncertainty-guided_Perturbation_for_Image_Super-Resolution_Diffusion_Model_CVPR_2025_paper.pdf); FaithDiff (CVPR 2025) — [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_FaithDiff_Unleashing_Diffusion_Priors_for_Faithful_Image_Super-resolution_CVPR_2025_paper.pdf); a single score-based model traversing the D–P curve (CVPR 2025) — [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Traversing_Distortion-Perception_Tradeoff_using_a_Single_Score-Based_Generative_Model_CVPR_2025_paper.pdf); and "Local Intrinsic Dimension Unveils Hallucinations in Diffusion Models" — [arXiv 2605.05026](https://arxiv.org/pdf/2605.05026)
- In NTIRE 2026, one-step Stable-Diffusion-based frameworks with LoRA "dominate the foundation" of mobile real-world SR, and the ×4 SR report frames SR as now "a trade-off between distortion and perception" — [NTIRE 2026 Mobile Real-World SR](https://arxiv.org/html/2604.17306v1); [NTIRE 2026 ×4 SR](https://arxiv.org/html/2604.14558v1)
- Hallucination is spreading to diffusion-assisted 3D: HAD (May 2026) — [arXiv 2605.16873](https://arxiv.org/abs/2605.16873)

### Inferences
- **Open problem A (metric):** an automatic, physics-grounded hallucination measure. For linear degradations with a known operator A (bicubic or known-kernel SR, inpainting, compressed sensing, and **lensless imaging with a measured PSF**), any output splits exactly into a range-space part (fixed by the measurement) and a null-space part (supplied by the prior). Hallucination is then null-space content that disagrees with ground truth while staying measurement-consistent. Measuring it needs no MLLM and gives a closed-form "hallucination budget" per image and per frequency band. I found no 2025–2026 paper that uses this as an evaluation protocol across many public restorers. Null-space ideas appear in DDNM-style solvers (pre-2025, not re-verified here) and in NOWA's optical watermark, but not as a metric.
- **Open problem B (benchmark):** "measurement metamers". These are pairs of different HR images (different digits, letters, or face identity) with *identical* LR measurements, built by adding null-space perturbations. A faithful restorer must be uncertain or refuse on them. Any confident and different answer is provably a hallucination, regardless of human preference.
- **Open problem C (uncertainty):** one-step distilled restorers (OSEDiff-style, dominant in NTIRE 2026) are deterministic, so they output a point estimate that looks like a posterior sample. Can calibrated uncertainty be recovered cheaply, for example through input-noise or LoRA-scale perturbation ensembles followed by conformal calibration?
- Medical imaging (sFRC) is ahead of natural-image SR on hallucination auditing. Transferring its protocols to natural-image SR is itself a defensible gap.

### Gaps
- I could not verify the content of "Evaluating and Preserving High-level Fidelity in Super-Resolution" (arXiv 2512.07037). It may overlap with open problem B (semantic fidelity) and **must be read before committing**.
- A conformal generative SR paper (arXiv 2502.09664) appeared only as a bibliography reference and was not verified. It may overlap with open problem C.
- I did not verify whether any 2026 paper already does null-space-decomposition *evaluation* across many SR models. This needs a targeted arXiv search ("range-null space hallucination evaluation").

---

## Key question 4: Can a purely simulated or public-data paper compete here without hardware prototypes?

### Takeaway
It can compete for **acceptance and orals**, but award-tier imaging papers have been prototype papers. The practical middle path is to (1) use **real captured public datasets** whose forward model is known (lensless DiffuserCam-style pairs, DSEC events, public RAW sets, public polarization and thermal sets), (2) simulate with **exact physics** (Bernoulli photon arrivals, ESIM events, linear light transport), and (3) make the contribution an *insight about a model class*, not a new sensor.

### Cited Findings
- All three imaging award papers above (Freeform Pixels, Spatially-Varying Autofocus, Propagating Light) introduced and demonstrated custom capture systems — [Columbia](https://www.cs.columbia.edu/2024/outstanding-research-and-best-paper-honors-at-eccv-2024); [CMU Graphics](https://graphics.cs.cmu.edu/); [arXiv 2506.05347](https://arxiv.org/abs/2506.05347)
- Several CVPR 2026 imaging orals rely on synthetic supervision or physical modeling of existing sensors and data: UnReflectAnything renders synthetic specular supervision, PPISP models ISP photometric variation in radiance fields, and the infrared-visible fusion paper is a diffusion model — [CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral)
- Generative priors now model the camera itself:
  - RawGen (Kim, …, Afifi, Levinshtein, Brown; Mar 2026) does text-to-raw generation for arbitrary target cameras and sRGB-to-raw inversion. It is trained on a many-to-one inverse-ISP dataset and claims synthetic raw helps downstream low-level tasks — [arXiv 2604.00093](https://arxiv.org/abs/2604.00093)
  - Metadata-conditioned diffusion sRGB noise generation — [arXiv 2603.04870](https://arxiv.org/html/2603.04870)
  - Autoregressive priors for RAW-to-sRGB ISP — [arXiv 2609.18302](https://arxiv.org/html/2609.18302)
  - Physics-guided rectified flow for low-light RAW — [arXiv 2509.08330](https://arxiv.org/pdf/2509.08330)
- SPAD and event methods are routinely trained with physics simulators and checked on a few real captures: QuantaFlow ([arXiv 2608.00499](https://arxiv.org/html/2608.00499v1)), and the SPAD noise-model paper, which targets the sim-to-real gap ([arXiv 2608.00489](https://arxiv.org/pdf/2608.00489)).

### Inferences
- A paper with no real data in an unconventional-sensor area will draw the reviewer objection "sim-to-real unverified". Mitigations: at least one real public dataset per claim; a sweep showing the conclusion is robust to simulator parameters; and choosing problems where the physics is *exact* (photon statistics, linear light transport, known PSF), so simulation is not an approximation.
- Low-level vision with generative priors (Key question 3) needs no hardware at all and has the lowest reviewer risk on this axis.
- Hardware-light options a small team could still try: a phone RAW capture of a dozen scenes, or a printed amplitude mask on a webcam for lensless imaging. These cost days, not months, but are not needed for the top ideas below.

### Gaps
- I found no explicit acceptance-rate statistics for simulation-only versus prototype papers in the CVPR computational imaging track.

---

## Key question 5: Which sub-areas are saturated, and how did generative priors change low-level vision?

### Takeaway
Generic restoration (SR, denoising, deblurring, all-in-one restoration, LOL-style low-light) is saturated. Challenge entries converge on one recipe: one-step SD plus LoRA plus IQA losses. Generative priors changed the field in three ways: the objective moved from distortion to perception, faithfulness and hallucination became the central unsolved issue, and the priors are now being pushed into the camera pipeline (RAW, ISP, noise, SPAD reconstruction). The open space is evaluation, uncertainty, and physics consistency, not new architectures.

### Cited Findings
- NTIRE 2026 mobile real-world SR: one-step SD frameworks with LoRA were "heavily favored". ×4 SR: teams build on one-step OMGSR and add no-reference IQA losses. Face restoration: one-step diffusion with test-time IQA-guided optimization — [Mobile RWSR](https://arxiv.org/html/2604.17306v1); [×4 SR](https://arxiv.org/html/2604.14558v1); [Face restoration](https://arxiv.org/html/2604.10532v2)
- Restoration and SR are absent from the portion of the CVPR 2026 oral list I read, while physics and sensor-model papers are present — [CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral)
- Metric crisis: FR metrics rank methods against human preference ([arXiv 2503.13074](https://arxiv.org/html/2503.13074v3)), and hallucination is orthogonal to both fidelity and NR quality ([arXiv 2507.14367](https://arxiv.org/abs/2507.14367))
- The prior-driven camera stack: RawGen ([arXiv 2604.00093](https://arxiv.org/abs/2604.00093)), gQIR on SPAD (CVPR 2026, [note](https://en.papernotes.org/CVPR2026/image_generation/gqir_generative_quanta_image_reconstruc_tion/)), NEC-Diff on event plus RAW ([arXiv 2603.20005](https://arxiv.org/html/2603.20005v1))
- The existing project report also advises against "pure SR/denoising" as a focus (internal: `reports/Ý tưởng nghiên cứu CVPR 2027.md`).

### Inferences
- Saturated: new SR or restoration architectures; all-in-one restoration with prompts; Retinex-style low-light on LOL; event-to-video reconstruction networks; per-scene NeRF/3DGS variants for a single sensor; "diffusion for X restoration" papers with no faithfulness analysis.
- Under-served: faithfulness metrics that are physics-grounded and need no MLLM; uncertainty for one-step restorers; the effect of generative reconstruction on *downstream* perception in sensor pipelines (SPAD, low light); physical-law consistency (linearity of light, energy conservation, reciprocity) of generative relighting and intrinsic models.

### Gaps
- No quantitative count of 2026 restoration submissions or acceptances was found. "Saturation" rests on challenge convergence and oral composition, not on a submission statistic.

---

## Objective deliverable: 10 open research questions (ranked for 1×H200 by the 2026-11-16 deadline)

The legend is **[PUBLIC]** for public data, simulators, or public checkpoints only, and **[HW]** for ideas that need custom hardware. All GPU-day numbers are my estimates.

### Takeaway
Six of the ten ideas are doable with public data alone within the 34 days left. The strongest are Q1 (null-space hallucination audit), Q2 (photon budgets for frozen VFMs), and Q5 (the superposition test for generative relighting), because each has a one-line physical insight, a teaser figure, and an evaluation across many models. The [HW] ideas suit a later venue.

### Cited Findings (each idea: gap, why it matters, angle, feasibility)

**Q1 [PUBLIC]: "Hallucination lives in the null space": a measurement-grounded hallucination audit for diffusion restorers.**
- *Gap:* Hallucination is acknowledged to be orthogonal to fidelity and NR metrics ([2507.14367](https://arxiv.org/abs/2507.14367)). Current fixes rely on MLLM judges ([2507.14367](https://arxiv.org/abs/2507.14367), [2506.20832](https://arxiv.org/html/2506.20832)) or are sampler fixes without a metric ([2606.02331](https://arxiv.org/abs/2606.02331)).
- *Why it matters:* A metric with no learned judge, which cannot be gamed by a reward-hacked MLLM proxy, would reset how generative restoration is evaluated.
- *Angle:* For known-A degradations (known-kernel SR, inpainting, CS, real lensless captures), decompose outputs into range and null parts and score null-space error per frequency band and per region. Plot all public restorers (multi-step, one-step, DPS-type) on a "hallucination vs. perception" plane. Show that current metrics rank them differently.
- *Feasibility:* Inference only. About 3–6 GPU-days for 10–15 models on a few thousand images.
- *Risk:* Real-world SR has an unknown A. Cover it with a known-A synthetic track plus a real lensless track (Q6).

**Q2 [PUBLIC]: "How many photons does a foundation model need?": photon-budget scaling of frozen VFMs on quanta data.**
- *Gap:* 2026 SPAD work is method-centric (gQIR, QuantaFlow, probabilistic events: [2608.27584](https://arxiv.org/abs/2608.27584), [2608.00499](https://arxiv.org/html/2608.00499v1)). I found no cross-VFM analysis.
- *Why it matters:* SPAD hardware is commercializing ([Litavis](https://thequantuminsider.com/2026/09/10/singular-photonics-launches-litavis-spad-image-sensor/)), and people need to know whether to reconstruct first or perceive directly.
- *Angle:* Simulate binary frames exactly from public high-fps video. Sweep photons per pixel, motion, and summation window. Measure DINOv3, SAM 2, Depth-Anything, VGGT, and a pose model. Compare (a) naive sums, (b) classical denoising, and (c) generative reconstruction. Hypothesis: generative reconstruction *hurts* downstream accuracy at low photon counts, which links to Q1.
- *Feasibility:* About 3–6 GPU-days.
- *Risk:* Scoop risk from Wision lab is high. Verify against one real public SPAD set.

**Q3 [PUBLIC]: "One-step restorers have no posterior": cheap calibrated uncertainty for distilled diffusion SR.**
- *Gap:* One-step SD restorers dominate NTIRE 2026 ([2604.17306](https://arxiv.org/html/2604.17306v1)) but are deterministic. Uncertainty work targets multi-step samplers ([UGDiff 2608.25998](https://arxiv.org/pdf/2608.25998), [CVPR25 UPSR](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Uncertainty-guided_Perturbation_for_Image_Super-Resolution_Diffusion_Model_CVPR_2025_paper.pdf)).
- *Angle:* Perturbation ensembles (input noise, LoRA scale, latent jitter) followed by conformal calibration, giving pixel or region risk maps. Show that the maps predict Q1's null-space error, and use them to abstain or fall back to a regression output.
- *Feasibility:* About 3–8 GPU-days.
- *Risk:* Must be distinguished from the unverified conformal-SR paper (2502.09664) and from internal shortlist #6.

**Q4 [PUBLIC]: "Events are gradients": what frozen VFMs lose on event streams, and a training-free remedy.**
- *Gap:* Event VFM papers adapt through E2VID proxies, distillation, or adapters ([2511.06238](https://arxiv.org/abs/2511.06238), [EventVGGT 2603.09385](https://arxiv.org/abs/2603.09385), [2603.23032](https://arxiv.org/html/2603.23032v1)) without diagnosing why plain transfer fails.
- *Angle:* Using ESIM/v2e on public video plus real DSEC/MVSEC, sweep contrast threshold and speed. Probe semantic versus geometric VFMs on several representations (voxel grid, time surface, E2VID, Poisson-integrated). Predicted finding: geometry transfers and semantics does not, and the gap closes with low-frequency recovery. Remedy: training-free integration with the right time constant.
- *Feasibility:* About 4–8 GPU-days.
- *Risk:* Moderate novelty. The framing as a "diagnosis across N VFMs" carries the paper.

**Q5 [PUBLIC]: "Light is linear; generative relighting isn't": a superposition and reciprocity test for diffusion relighting and intrinsic models.**
- *Gap:* CVPR 2026 has several relighting/OLAT orals (OLATverse, POLAR, Relightable Holoported Characters — [CVPR 2026 Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral)). In the sources read, I found no physics-consistency evaluation of generative relighters.
- *Angle:* Image(L1+L2) must equal Image(L1)+Image(L2), and scaling a light must scale radiance. Use OLAT data as ground truth. Measure violations across public relighting and intrinsic-decomposition models. Training-free fix: relight in an OLAT basis and recombine linearly.
- *Feasibility:* About 2–5 GPU-days.
- *Risk:* Depends on public checkpoint and dataset licences (OLATverse access not verified). Novelty must be checked against recent relighting evaluations.

**Q6 [PUBLIC, real data]: lensless imaging as the "honest" testbed for generative priors.**
- *Gap:* Lensless work in 2026 is method-centric ([Lensless Polarization Camera 2603.17156](https://arxiv.org/pdf/2603.17156), [Guided Lensless Polarization 2603.27357](https://arxiv.org/pdf/2603.27357)). Lensless forward models (a measured PSF) are known and real captures are public, which makes them ideal for Q1.
- *Angle:* Quantify how much diffusion-reconstructed lensless content comes from the PSF null space, and which PSF designs minimize hallucination. This is optics design guided by a hallucination metric, done in simulation on the measured PSF.
- *Feasibility:* About 3–6 GPU-days.
- *Note:* Public DiffuserCam-style datasets are known resources but were not re-verified here. Q6 could be merged into Q1 as its "real measurement" track.

**Q7 [PUBLIC]: "The ISP is the degradation": a controlled ISP-shift atlas for generative real-world SR.**
- *Gap:* Generative SR is trained with synthetic degradation pipelines, while real degradations come from the camera ISP (demosaic, denoise, sharpening, tone mapping, JPEG). Generative ISP and RAW modeling is now possible ([RawGen 2604.00093](https://arxiv.org/abs/2604.00093), [2603.04870](https://arxiv.org/html/2603.04870), [2609.18302](https://arxiv.org/html/2609.18302)).
- *Angle:* Render public RAW sets through parameterized ISPs. Measure per-stage sensitivity of public diffusion SR models for both fidelity and hallucination (using Q1). Find which ISP stage causes most failures.
- *Feasibility:* About 3–5 GPU-days.
- *Risk:* Could read as a benchmark only. It needs one actionable finding.

**Q8 [PUBLIC]: polarization as a cheap "hallucination detector" for generative normal and depth priors.**
- *Gap:* Polarization appears in 3DGS ([PhyGaP, CVPR 2026 oral](https://cvpr.thecvf.com/virtual/2026/events/Oral)) and thermal polarimetric MVS ([2510.20972](https://arxiv.org/pdf/2510.20972)). The complementarity between the π-ambiguous normals of shape-from-polarization and the failure modes of generative monocular normal predictors (on specular or transparent surfaces) is not quantified in the sources read.
- *Angle:* Training-free fusion. Polarization resolves orientation where the prior hallucinates; the prior resolves the π-ambiguity.
- *Feasibility:* About 2–4 GPU-days on public SfP datasets.
- *Risk:* Public polarization datasets are small, and their availability was not verified.

**Q9 [PUBLIC]: thermal reflections fool RGB-pretrained geometry models.**
- *Gap:* "Dual Band Thermal Videography" (CVPR 2026 oral) shows that time-varying reflection and emission must be separated in LWIR ([Orals](https://cvpr.thecvf.com/virtual/2026/events/Oral)), and IR–visible fusion is moving to diffusion.
- *Hypothesis (my inference):* Depth and 3D VFMs applied to thermal frames misread LWIR reflections (glass, polished floors, which are more reflective in LWIR) as geometry.
- *Angle:* A failure taxonomy on public thermal datasets, plus a reflection-aware masking fix.
- *Feasibility:* About 2–4 GPU-days.
- *Risk:* The physics claim needs ground truth, and public thermal datasets were not verified.

**Q10 [HW]: task-optimized minimalist sensing beyond freeform pixels.**
- *Gap and lineage:* [Freeform Pixels](https://www.cs.columbia.edu/2024/outstanding-research-and-best-paper-honors-at-eccv-2024) and [SV-Autofocus](https://graphics.cs.cmu.edu/).
- *Angle:* Learn the masks for a few-pixel sensor for a new task (e.g., fall detection, gesture) jointly with a tiny network, then build it with photodiodes and printed masks.
- *Feasibility:* Under 1 GPU-day of compute but weeks of hardware work. Not realistic before 2026-11-16.

**Q11 [HW]: SPAD and event hybrid capture for low-light 3D with foundation priors.**
- *Gap:* event plus RAW low-light diffusion ([NEC-Diff 2603.20005](https://arxiv.org/html/2603.20005v1)) and quanta perception ([2608.27584](https://arxiv.org/abs/2608.27584)).
- *Feasibility:* Needs SPAD hardware. The simulation half could be scoped as Q2.

**Q12 [HW-optional]: transient and time-of-flight inverse rendering with RGB foundation priors.**
- *Lineage:* [Propagating Light](https://arxiv.org/abs/2506.05347).
- *Angle:* Use monocular depth and normal priors to regularize multi-path-interference separation in simulated transients (rendered with a transient renderer), then validate on any released capture data.
- *Feasibility:* About 5–10 GPU-days. Real data availability is unverified.

### Inferences
- Recommended pairing for the 34 days: **Q1+Q6** as one paper (metric plus a real lensless track), or **Q2** alone. Q5 is the fastest backup. Each uses public checkpoints, has a physical one-liner as the title claim, and gets its breadth from evaluating many models. That matches the "insight, breadth over depth" framing in the existing report.
- Q1, Q2, and Q3 reinforce each other: generative reconstruction hallucinates (Q1), this harms downstream sensing (Q2), and uncertainty maps can flag it (Q3). That can form one coherent story if scope allows.

### Gaps
- Before committing to any idea, run a targeted novelty sweep. For Q1: "range-null space hallucination metric". For Q5: "relighting linearity evaluation diffusion". For Q2: "photon count foundation model SPAD benchmark". This search did not exhaustively rule out 2026 preprints.
- Licences, sizes, and availability of public datasets (DiffuserCam-style lensless, SPAD, polarization, thermal, OLATverse) were not verified in this session.
