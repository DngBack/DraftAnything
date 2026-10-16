# Data-centric vision and training dynamics: open problems for a 1×H200 CVPR 2027 paper

Scope note: this file deliberately avoids what `reports/Ý tưởng nghiên cứu CVPR 2027.md` already covers: the REPA mechanism, tokenizer diffusability, diffusion memorization phases, MLLM "arbitration" or readout bottlenecks, RL post-training readout, SAM2 memory, 4D physics judges, 3D calibration and video-physics probes. One closely related paper, "Hidden in plain sight" (2506.08008), is listed below only so it is not reused, because it overlaps the existing idea #2. All GPU-day figures are **my own estimates**. None comes from a source. Research date: 2026-10-13. I made about 20 search and fetch calls. Several OpenReview PDFs could not be fetched because of a browser check.

## Q1. When does training on synthetic images help vs hurt, and is there a predictive explanation?

### Takeaway
As of 2026 the evidence says synthetic images help when they are diverse and mixed with real data at a controlled ratio. They hurt when they dominate the data or are narrow. A CVPR 2026 paper shows that *newer, prettier* text-to-image models make *worse* training data. Nobody has yet published a quantitative predictor of synth→real utility that is measured before training, so that is the open gap.

### Cited Findings
- "When Pretty Isn't Useful" (Adamkiewicz et al., CVPR 2026) generated large synthetic datasets with state-of-the-art T2I models released 2022–2025, trained classifiers only on synthetic data and tested on real data. Accuracy on real test data "consistently declines with newer T2I models" even as fidelity and prompt adherence improve. The authors attribute this to collapse toward a "narrow, aesthetic-centric distribution" and test no mechanism beyond that. — [arXiv 2602.19946](https://arxiv.org/abs/2602.19946)
- Earlier ImageNet evidence: accuracy falls once synthetic data outnumbers real data, with a −2.69% drop at 9× synthetic. Synthetic data scales much worse than real data for supervised classifiers but close to real data for CLIP training. The authors blame concepts the T2I model cannot generate. — [Scaling Laws of Synthetic Images, arXiv 2312.04567](https://arxiv.org/pdf/2312.04567)
- Retrieving real images from the generator's own training set (LAION) beats generating synthetic images. Synthetic data can hurt even where real data helps, which the authors attribute to low-level generator artifacts and distorted class-specific details. — [The Unmet Promise of Synthetic Training Images, arXiv 2406.05184](https://arxiv.org/html/2406.05184v5)
- "Diversify, Don't Fine-Tune" (TMLR 2025): overfitting comes from fine-tuning the generator. Equal sampling weights for real and synthetic data plus a lower CFG scale give gains that keep growing with the number of unique synthetic samples. — [arXiv 2312.02253](https://arxiv.org/abs/2312.02253)
- Chamfer Guidance (2025), a diversity-oriented sampling method, reports up to 15% downstream accuracy gains over classifier-free guidance. Mixing real and synthetic data is best (about 63.8% with 32 real images per class). — [arXiv 2508.10631](https://arxiv.org/html/2508.10631v1)
- A synthetic-data curation paper from July 2026 (homogeneous–heterogeneous splitting of generated images) shows that filtering *after* generation is now an active sub-area. — [arXiv 2607.02637](https://arxiv.org/pdf/2607.02637)
- Model collapse: replacing data each generation causes collapse, while accumulating synthetic data alongside real data avoided collapse in every setting tested, including image diffusion models and VAEs. — [arXiv 2404.01413](https://arxiv.org/pdf/2404.01413v2). Under a *fixed compute budget*, mixing in synthetic data degrades performance more but still plateaus. — [arXiv 2410.16713](https://arxiv.org/html/2410.16713v2)
- NeurIPS 2025 recasts collapse as a generalization-to-memorization transition linked to training-set entropy, and proposes entropy-based data selection. — [arXiv 2509.16499](https://arxiv.org/html/2509.16499v1). Latent-space filtering to stabilize self-consuming diffusion (AAAI 2026). — [arXiv 2511.12742](https://arxiv.org/abs/2511.12742)
- CLIP-side evidence is more favorable: GIFT (CVPR 2025) reports that Stable Diffusion images closely approximate CLIP's pretraining distribution and uses them for continual VLM learning. — [CVPR 2025 GIFT](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Synthetic_Data_is_an_Elegant_GIFT_for_Continual_Vision-Language_Models_CVPR_2025_paper.pdf)
- A 2026 medical study (brain MRI, StyleGAN2-ADA) warns that selecting synthetic data by feature-space quality should be checked against the downstream classifier's decision surface. — [arXiv 2605.23094](https://arxiv.org/pdf/2605.23094)

### Inferences
- **Open problem A (predictive law):** the field has the phenomenon (newer generators are worse, CFG hurts, accumulation saves) but no *predictor* computed before training. A candidate: per-class precision/recall or coverage of synthetic data in a frozen DINOv2/CLIP feature space, or the class-conditional covariance rank, regressed against synth→real accuracy across about 10 generators × several CFG values. If one scalar such as "coverage of real-class support" explains most of the variance and also predicts *which classes* fail, that would be a CVPR-friendly insight paper ("Coverage, not fidelity, predicts synthetic data utility"). A small fix follows directly: pick the CFG value or sampler per class by maximizing predicted coverage.
- **Open problem B (compute-dependence of the optimal synthetic fraction):** Goyal et al. (CVPR 2024) showed that data curation is not compute-agnostic for real data ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Goyal_Scaling_Laws_for_Data_Filtering--_Data_Curation_cannot_be_Compute_CVPR_2024_paper.pdf)). An analogous "phase diagram" (optimal real:synthetic ratio vs epochs × model size) on ImageNet-100 is missing. The 2312.04567 drop at 9× synthetic may be a low-compute artifact, or it may get worse at high compute. Either outcome is reportable.
- Why reviewers care: synthetic data is now the default cheap fix in industry, and the CVPR 2026 "pretty isn't useful" result generates curiosity that is not yet resolved.
- Feasibility (my estimate): ImageNet-100, about 1.3k images/class × 100 classes ≈ 130k images per generator config. With few-step or distilled generators (SDXL-Turbo, SD3.5-Turbo, FLUX-schnell class) that is roughly 0.1–0.3 GPU-day per config on an H200. Training ResNet-50/ViT-S on 130k images for 100 epochs is roughly 0.2–0.4 GPU-day. About 10 generators × 3 CFG values ≈ 30 configs ≈ 10–20 GPU-days in total. Cut to 5 generators × 2 CFG for a November deadline.

### Gaps
- I could not see the exact generators, dataset or numbers in 2602.19946 (the abstract page has no figures). Read the full text before positioning against it. **Scoop risk is high**, since that group may already be doing the "why".
- I found no 2026 paper with a pre-training predictor of synthetic utility, but my search was not exhaustive.

## Q2. Open questions in model merging for vision (why it works, when it fails, LMC/permutation at scale)

### Takeaway
The CLIP ViT-B/32 / B/16 / L/14 benchmark with 8/14/20 tasks is saturated with SVD- and subspace-based merging methods (several at CVPR 2026). The open *science* questions are: (i) predicting mergeability before or early in fine-tuning, (ii) whether the task-vector subspace restriction is itself the bottleneck, and (iii) what merging does to *dense/spatial* features, since nearly every benchmark is classification.

### Cited Findings
- Task vectors approximate gradients of the task losses, so task arithmetic can be read as an approximation of multitask learning, and *early training dynamics* are the key factor (vision benchmarks included). — [On Task Vectors and Gradients, arXiv 2508.16082](https://arxiv.org/abs/2508.16082)
- "Demystifying Mergeability" (Zhou, Zhao, Yu, Rodolà; arXiv Jan 2026, v9 Jun 2026): fits L1-regularized linear models on interpretable pairwise metrics across 5 merging methods. **Gradient-alignment metrics are the most consistent predictors** of merge success, with 64.0% average top-5 metric overlap across architectures/methods and 79.3% sign agreement. TIES has a distinct "fingerprint". The authors point to merge-aware fine-tuning as future work. — [arXiv 2601.22285](https://arxiv.org/abs/2601.22285)
- "A Broader Look at Model Merging" (Oct 6, 2026): coefficient search confines merging to the task-vector subspace, an implicit regularization that often *limits* performance. Optimizing the weights directly with a small amount of extra data (even one instance per class) can beat existing merging methods. — [arXiv 2610.07990](https://arxiv.org/abs/2610.07990)
- Methods for the 8/14/20-task CLIP benchmark keep coming: TSV (CVPR 2025) — [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Gargiulo_Task_Singular_Vectors_Reducing_Task_Interference_in_Model_Merging_CVPR_2025_paper.pdf); ACE-Merging (CVPR 2026) — [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_ACE-Merging_Data-Free_Model_Merging_with_Adaptive_Covariance_Estimation_CVPR_2026_paper.pdf); DC-Merge (CVPR 2026) — [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_DC-Merge_Improving_Model_Merging_with_Directional_Consistency_CVPR_2026_paper.pdf); AlphaMerging (CVPR 2026) — [poster](https://cvpr.thecvf.com/virtual/2026/poster/40776); energy-proportional rank allocation (Sep 2026) — [arXiv 2609.24517](https://arxiv.org/pdf/2609.24517); LiNeS layer scaling — [arXiv 2410.17146](https://arxiv.org/pdf/2410.17146).
- LMC at scale: "dual learned matching" with attention-head permutations and residual rotations gives near-zero barriers for medium-size LMs and small barriers for billion-parameter LLMs. **ViT-L stays above 69% ImageNet top-1 along the whole interpolation path** (Li & Shen, Jun 2026). — [arXiv 2606.23607](https://arxiv.org/abs/2606.23607)
- ICLR 2026: widening models can place independently trained models in the same basin without permutation alignment. — [ICLR 2026 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/5eceaacf571380e70c1052ef6bd58cb3-Paper-Conference.pdf)
- A large LLM study found that only Task Arithmetic reliably gives constructive interference. Subspace methods rely on strong geometric assumptions. — [arXiv 2511.21437](https://arxiv.org/html/2511.21437v1)
- Merging has also been used as an *analysis tool*: perception and reasoning in VLMs studied by merging models. — [Bring Reason to Vision, arXiv 2505.05464](https://arxiv.org/html/2505.05464)

### Inferences
- **Open problem C ("Mergeability is decided in the first 1% of fine-tuning"):** combine 2508.16082 (early dynamics matter) with 2601.22285 (gradient alignment predicts success). Test whether pairwise gradient or task-vector cosine after k ≪ T steps predicts the final merged accuracy, then use it for *merge-aware early stopping or task grouping*. That gives a simple principled fix plus a clear plot. Feasibility: fine-tuning CLIP ViT-B/32 per task takes minutes, so 20 tasks × 3 seeds × checkpointing ≈ 2–4 GPU-days (my estimate).
- **Open problem D ("Merging breaks geometry before semantics"):** the benchmarks only measure classification accuracy. Measure what task arithmetic/TIES/TSV merges of CLIP/DINOv2 fine-tunes do to dense probes (depth, correspondence, segmentation) and to high-norm/register-like tokens (see Q6). If merging silently destroys dense features while keeping classification, that is a counter-intuitive finding reviewers would cite. Feasibility: about 3–6 GPU-days.
- Avoid: yet another SVD/subspace merging method on 8/14/20 tasks (saturated).

### Gaps
- I could not confirm whether 2601.22285 uses vision models; the abstract page does not say.
- I did not find a paper reporting a specific task-count failure threshold for ViT merging.

## Q3. Forgetting when fine-tuning vision foundation models / VLMs: open problems

### Takeaway
Most VLM-forgetting work measures *language/VQA* retention. The evidence that fine-tuning erodes **spatial/geometric** encoder ability (depth, correspondence) is thin and mostly from robotics. Meanwhile a principled forgetting predictor (KL to the base policy, "RL's Razor") has been shown for LLMs and robotics but not tested for vision encoders or VLMs. Both are open.

### Cited Findings
- OpenVLA fine-tuning badly degrades the visual encoders, especially DINOv2, which then fails at depth regression. The fix is post-hoc restoration toward pretrained weights (Dey et al. 2024, as described in a 2025 paper). — reported via [Actions as Language, ICLR 2026](https://iclr.cc/virtual/2026/poster/10007076) / [ResearchGate](https://www.researchgate.net/publication/395944766_Actions_as_Language_Fine-Tuning_VLMs_into_VLAs_Without_Catastrophic_Forgetting). The same ICLR 2026 paper keeps over 85% of base VQA performance by expressing actions as language.
- "Fine-tuning VLMs Without Forgetting Is Easier Than You Think" (OpenReview): a low learning rate and fewer trainable parameters prevent forgetting for OOD images. A second forgetting mode comes from task-specific overfitting, fixed by data-hybrid training. — [OpenReview PDF](https://openreview.net/pdf/278c682d3fdae858e47cc6e1111e5d282dd787fa.pdf) (claims taken from the search snippet; the PDF was blocked by a browser check)
- RL's Razor (Shenfeld, Pari, Agrawal; ICLR 2026): online RL forgets less than SFT at equal new-task performance. Forgetting tracks the **KL between fine-tuned and base policy on the new task**. An "oracle SFT" distribution that minimizes KL forgets even less than RL. Tested on LLMs and robotic foundation models, not VLMs. — [arXiv 2509.04259](https://arxiv.org/html/2509.04259v1), [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/618c95f4557c15b253fb0e6f548ea0c0-Abstract-Conference.html)
- LLaVA-OneVision: the original SigLIP outperforms the *fine-tuned* visual encoder on all 6 probing tasks. — [Visual Representations inside the Language Model, arXiv 2510.04819](https://arxiv.org/html/2510.04819v1)
- "Hidden in plain sight" (Fu, Bonnen, Guillory, Darrell, Jun 2025): VLMs do much worse than their own encoders on depth and correspondence, sometimes near chance. The bottleneck is the LLM's use of visual information, not encoder degradation. — [arXiv 2506.08008](https://arxiv.org/abs/2506.08008). **This overlaps the existing report's idea #2 (arbitration); do not reuse.**
- Visual representation drift inside LLM layers after instruction tuning, fixed by visual anchor learning (Aug 2026). — [arXiv 2608.01635](https://arxiv.org/pdf/2608.01635). Visual representation alignment for MLLMs. — [arXiv 2509.07979](https://arxiv.org/pdf/2509.07979)
- SpatioLM (Aug 2026): fine-tuning on 3D data disrupts pretrained features and causes forgetting of general semantics. — [arXiv 2608.01899](https://arxiv.org/html/2608.01899)
- Driving-model forgetting quantified (Apr 2026). — [arXiv 2604.04857](https://arxiv.org/pdf/2604.04857). Multimodal continual instruction tuning separates "superficial" from "essential" forgetting. — [SEFE, arXiv 2505.02486](https://arxiv.org/pdf/2505.02486). RL fine-tuning for visual continual learning. — [arXiv 2605.09640](https://arxiv.org/pdf/2605.09640)
- LiNeS (layer-scaling the update by depth) prevents forgetting and helps merging. — [arXiv 2410.17146](https://arxiv.org/pdf/2410.17146)

### Inferences
- **Open problem E ("Classification probes hide geometric forgetting"):** fine-tune DINOv2/CLIP ViT-B/L (full FT, LoRA, LP-FT) on classification or detection tasks. Track ImageNet k-NN (the usual metric) *and* frozen-feature depth (NYUv2 linear), keypoint correspondence (SPair) and segmentation probes. Hypothesis: dense or geometric abilities degrade earlier and more than semantic ones, so standard forgetting metrics underreport. Simple fix: layer-wise restoration (LiNeS-style), chosen by a dense-probe criterion, or KL/feature-distance regularization on *patch tokens* rather than the CLS token. Feasibility: about 4–8 GPU-days (my estimate) for ViT-B with a few runs at ViT-L.
- **Open problem F ("RL's Razor for vision encoders"):** does forward KL (or patch-feature distance) to the base model *on the new task's data* predict forgetting across fine-tuning recipes for vision classifiers or a 2–7B VLM with LoRA? If it does, a CVPR paper can claim "one scalar predicts forgetting across 30 recipes" and add a KL-minimal data or replay recipe. A natural link to Q1 is that *synthetic replay* (GIFT-style) may work *because* it keeps KL low. Feasibility: LoRA on Qwen2.5-VL-3B/7B fits on one H200. About 6–10 GPU-days for around 20 recipes (my estimate).
- Keep this distinct from the existing report's RL-post-training-readout idea: here the object is the **encoder and dense features**, and the predictor is KL, not readout geometry.

### Gaps
- I could not open Dey et al. (2024) directly. The DINOv2-depth claim is second-hand.
- I found no 2025–2026 paper that tests KL-predicts-forgetting on VLMs or ViTs.

## Q4. Data attribution for diffusion (copyright/memorization): practical open problems

### Takeaway
Methods exist (gradient/TRAK-style, unlearning-based, group-wise) but **evaluation is the bottleneck**. Counterfactual retraining is infeasible at text-to-image scale, leave-one-out ground truth is conceptually shaky, and cheap CLIP/DINO similarity baselines are competitive on some datasets and poor on others with no explanation of why. A paper published Oct 2026 just took the "attribution across timesteps" angle.

### Cited Findings
- Model-agnostic baselines (CLIP/DINO similarity) "can be very competitive in some datasets and metrics" but do relatively poorly on CIFAR-10 and ArtBench-10 (May 2026). — [arXiv 2605.17938](https://arxiv.org/html/2605.17938v1)
- Fast attribution for T2I models: full counterfactual retraining is infeasible at scale, so the authors "cannot definitively verify that these attributions reflect true influence" and fall back on a consistency check. — [arXiv 2511.10721](https://arxiv.org/html/2511.10721)
- Guda (NeurIPS 2025 Creative AI track): group-wise counterfactual attribution through unlearning, about 100× faster than leave-group-out retraining on CIFAR-10, plus SD style attribution. — [arXiv 2601.22651](https://arxiv.org/html/2601.22651)
- "Attributing HOW, Not Just WHICH" (Oct 2026): builds counterfactual trajectories at matched noisy states to show how influence unfolds over denoising steps. — [arXiv 2610.11238](https://arxiv.org/abs/2610.11238). **This takes the timestep-resolved angle; scoop risk.**
- Timestep-induced bias: some timesteps have high gradient norm, which biases influence estimates. — [arXiv 2401.09031](https://arxiv.org/html/2401.09031v2)
- Leave-one-out ground truth has computational and conceptual difficulties, and raw diffusion loss misrepresents contribution (Diffusion Attribution Score). — [arXiv 2410.18639](https://arxiv.org/abs/2410.18639)
- Attribution by unlearning synthesized images, evaluated with leave-K-out retraining on MSCOCO LDMs. Transfer from customized models to general models remains hard. — [arXiv 2406.09408](https://arxiv.org/html/2406.09408v1)
- Model-level (not data-level) attribution benchmark IABench covers 28 generators from 2022–2025. — [arXiv 2604.17736](https://arxiv.org/pdf/2604.17736)

### Inferences
- **Open problem G ("When is attribution just retrieval?"):** explain *when* cheap feature similarity agrees with counterfactual influence. Hypothesis: agreement is high in the memorization or low-data regime and low in the generalization regime, so dataset size per class or duplication rate predicts which attribution family to trust. Feasibility: needs ground-truth retraining. On an H200 a CIFAR-10 DDPM (35M params) retrain is roughly 0.3–0.6 GPU-day (my estimate), so LDS with about 20–30 subset retrains ≈ 8–15 GPU-days. That is borderline for November. Downscale to 32px subsets or use small EDM.
- This idea sits near the existing report's "memorization phases" idea. Pitch it as an attribution-evaluation paper, or skip it if the team is already pursuing memorization.

### Gaps
- I found no source giving a standard large-scale copyright attribution benchmark with ground truth as of Oct 2026.

## Q5. Insight-driven papers with orals/awards 2024–2026 in this space (models for framing)

### Takeaway
Recent recognized papers in this space are mostly diagnostic, with a claim as the title and a negative or counter-intuitive finding plus a small fix, and not method papers.

### Cited Findings
- CVPR 2026 **oral**: "Rethinking Dataset Distillation: Hard Truths about Soft Labels" (Dey et al.). With hard labels, only RDED reliably beats random on ImageNet-1K, and it still lags strong coresets. With soft labels plus KD, results are compute-dominated, largely invariant to coreset quality and size, and quickly saturate to the full-data baseline. They propose CAD-Prune/CA2D. — [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Dey_Rethinking_Dataset_Distillation_Hard_Truths_about_Soft_Labels_CVPR_2026_paper.pdf), [arXiv 2604.18811](https://arxiv.org/abs/2604.18811)
- CVPR 2026: "When Pretty Isn't Useful" (negative finding on T2I models as data generators). — [arXiv 2602.19946](https://arxiv.org/abs/2602.19946)
- NeurIPS 2025: "Vision Transformers Don't Need Trained Registers" (Jiang, Dravid, Efros, Gandelsman). A few neurons create high-norm outlier tokens, and moving their activation to an extra untrained token at test time matches trained registers. This is a training-free "free lunch". — [arXiv 2506.08010](https://arxiv.org/abs/2506.08010), [NeurIPS poster](https://neurips.cc/virtual/2025/poster/117199)
- NeurIPS 2025 best papers include "Gated Attention for LLMs: Non-linearity, Sparsity, and Attention-Sink-Free" and "Artificial Hivemind: The Open-Ended Homogeneity of Language Models". The committee also praised a "masterfully executed… negative finding" on RLVR. — [NeurIPS blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/), [HowAIWorks summary](https://howaiworks.ai/blog/neurips-2025-best-paper-awards-announcement)
- ICLR 2026 / NeurIPS 2025: RL's Razor (forgetting ∝ KL), an insight paper with a one-line principle. — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/618c95f4557c15b253fb0e6f548ea0c0-Abstract-Conference.html)
- CVPR 2024: "Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic" (title is a claim). — [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Goyal_Scaling_Laws_for_Data_Filtering--_Data_Curation_cannot_be_Compute_CVPR_2024_paper.pdf)

### Inferences
- The pattern that wins: a title that states a claim ("X is not Y", "Z doesn't need W"), a falsifiable finding shown across many models, and a fix of a few lines. Homogeneity and diversity collapse ("Artificial Hivemind") and aesthetic narrowing ("Pretty isn't useful") are the same story in two modalities, which supports a "coverage predicts utility" paper (Q1).

### Gaps
- I could not verify the official ICLR 2026 outstanding paper list. Third-party lists disagree ([GitHub list](https://github.com/FeijiangHan/Top-Conference-Best-Papers)).
- From memory, not verified in this session: "Vision Transformers Need Registers" received an ICLR 2024 Outstanding Paper award, and a NeurIPS 2025 best paper concerned why diffusion models don't memorize. Verify both before citing.
- I did not find CVPR 2025/ICCV 2025 award or oral lists specific to data-centric papers.

## Q6. Remaining areas: curation/distillation, small-scale proxies and scaling laws, grokking, "free lunch" phenomena

### Takeaway
Three gaps are cheap and well defined: (1) the tiny-learning-rate proxy fix for data-recipe ranking was shown **only for language**, so the vision transfer is open; (2) test-time registers and high-norm neurons have not been studied under fine-tuning or merging; (3) grokking in ViTs has almost no 2025–2026 evidence, which makes it a risky area for CVPR.

### Cited Findings
- "Can Small Training Runs Reliably Guide Data Curation?" (Wang, Wu, Lyu, Zou, Song, Jia, Mittal; Dec 2025, v2 Apr 2026): with a standard learning rate, small-proxy dataset rankings flip. Training proxies with a learning rate 1–2 orders of magnitude smaller raises proxy→target rank correlation from below 0.75 to above 0.92, with Spearman above 0.95 for GPT2-125M vs Pythia-1B over 253 pairs (23 recipes, 70M–1B models, more than 20k runs). **Language only. Vision, proxy architecture and size ratios are not studied.** — [arXiv 2512.24503](https://arxiv.org/html/2512.24503)
- DataDecide: 8 scaling-law methods are no more efficient than ranking single-scale experiments (language). — [arXiv 2504.11393](https://arxiv.org/pdf/2504.11393)
- CLIP filtering work is often limited to DataComp small/medium ("we only had the resources to experiment at the DataComp medium scale"). — [Filter Like You Test, arXiv 2503.08805](https://arxiv.org/pdf/2503.08805). Larger compute needs less aggressive filtering. — [Goyal et al. CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Goyal_Scaling_Laws_for_Data_Filtering--_Data_Curation_cannot_be_Compute_CVPR_2024_paper.pdf)
- Dataset distillation: the CVPR 2026 oral above shows soft-label KD makes coreset quality nearly irrelevant. "Hard Labels In!" reports local semantic drift under tight soft-label storage (49.5% ImageNet-1K at 50 IPC). — [arXiv 2512.15647](https://arxiv.org/html/2512.15647v3). Soft-label pruning and quantization. — [arXiv 2604.18135](https://arxiv.org/pdf/2604.18135)
- Free lunch: high-norm outlier tokens come from a small set of neurons in CLIP/DINOv2. Training-free test-time registers give cleaner maps and better downstream results, and also apply to off-the-shelf VLMs. — [arXiv 2506.08010](https://arxiv.org/abs/2506.08010). An attention-sink-free gated attention design won a NeurIPS 2025 best paper. — [NeurIPS blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)
- Grokking: it occurs in practical settings such as a CNN on CIFAR-10 and a ResNet on Imagenette. — [arXiv 2405.20233](https://arxiv.org/abs/2405.20233v2). The delay matches delayed changes in representational geometry (manifold capacity), not NTK change. — [Zheng et al. 2024](https://proceedings.mlr.press/v285/zheng24a.html). GrokTransfer removes the delay with embeddings from a weaker model. — [arXiv 2504.13292](https://arxiv.org/html/2504.13292v1). Grokking as learning under nonstationarity. — [arXiv 2507.20057](https://arxiv.org/pdf/2507.20057)

### Inferences
- **Open problem H ("Small proxies lie about vision data, unless…"):** test whether vision data-curation rankings (CLIP filtering scores on DataComp-small, or ImageNet-subset selection and pruning metrics) transfer from ViT-Ti/S proxies to ViT-B. Then test whether the tiny-learning-rate trick from 2512.24503 fixes it, or whether vision needs a different knob such as augmentation strength or epochs (per Goyal: compute dependence). Reviewers care because every curation paper uses small proxies. Feasibility: DataComp-small (12.8M samples) CLIP runs are roughly 0.2–0.5 GPU-day each on an H200 (my estimate), so about 10 recipes × 3 scales ≈ 8–15 GPU-days. The ImageNet-100 variant is cheaper at roughly 3–6 GPU-days.
- **Open problem I ("Fine-tuning moves the registers"):** check whether full FT, LoRA or merging changes which neurons create high-norm tokens, and whether test-time registers recover the dense features lost to forgetting (Q3-E) or merging (Q2-D). If artifact tokens explain a large share of "geometric forgetting", the training-free fix carries over. Feasibility: about 1–3 GPU-days (mostly inference), the cheapest idea here. Pair it with E or D for a CVPR-sized paper.
- **Open problem J (grokking/delayed generalization in ViTs on small data):** for example, ViT-S on ImageNet-100 subsets with strong weight decay. Does a geometry metric (manifold capacity or effective rank) predict the generalization jump early enough to serve as a stopping or LR-schedule signal? Low CVPR appeal unless tied to a practical recipe. Feasibility: about 3–6 GPU-days. Ranked last.

### Gaps
- I found no 2025–2026 paper measuring proxy→target rank transfer for vision or CLIP data curation. That makes it a likely open gap, but my search was not exhaustive.
- I found no 2025–2026 grokking-in-ViT paper.

## Q7. Ranked shortlist (10 questions) and saturated directions

### Takeaway
Best ratio of insight to compute for a Nov 16, 2026 deadline: **E+I (geometric forgetting hidden by classification probes, with a test-time-register/LiNeS fix)**, **C (early mergeability predictor)**, **H (proxy rank transfer for vision data)** and **A (coverage predicts synthetic utility)**. Avoid new method papers in dataset distillation and on the 8/14/20-task merging benchmark.

### Cited Findings (supporting each pick; details in Q1–Q6)
| # | Claim-style title | Gap evidence | 1×H200 estimate (mine) |
|---|---|---|---|
| 1 (E+I) | "Classification hides geometric forgetting" in DINOv2/CLIP fine-tuning, fixed by register/layer restoration | Dey et al. via [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007076); [2510.04819](https://arxiv.org/html/2510.04819v1); [2506.08010](https://arxiv.org/abs/2506.08010); [LiNeS](https://arxiv.org/pdf/2410.17146) | 5–10 GPU-days |
| 2 (C) | "Mergeability is decided early": early gradient alignment predicts merge interference; merge-aware stopping/grouping | [2508.16082](https://arxiv.org/abs/2508.16082); [2601.22285](https://arxiv.org/abs/2601.22285) | 2–4 |
| 3 (H) | "Small proxies mis-rank vision data": does the tiny-LR fix transfer to CLIP/ImageNet curation? | [2512.24503](https://arxiv.org/html/2512.24503) (language only); [Goyal CVPR'24](https://openaccess.thecvf.com/content/CVPR2024/papers/Goyal_Scaling_Laws_for_Data_Filtering--_Data_Curation_cannot_be_Compute_CVPR_2024_paper.pdf) | 4–15 |
| 4 (A) | "Coverage, not fidelity, predicts synthetic data utility" across around 10 generators; per-class CFG fix | [2602.19946](https://arxiv.org/abs/2602.19946); [2508.10631](https://arxiv.org/html/2508.10631v1); [2312.02253](https://arxiv.org/abs/2312.02253) | 8–20 |
| 5 (F) | "One KL predicts forgetting" for ViT/VLM fine-tuning recipes (RL's Razor for vision); synthetic replay as a KL-minimal fix | [2509.04259](https://arxiv.org/html/2509.04259v1); [GIFT](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Synthetic_Data_is_an_Elegant_GIFT_for_Continual_Vision-Language_Models_CVPR_2025_paper.pdf) | 6–10 |
| 6 (D) | "Merging breaks geometry before semantics" (dense probes on merged ViTs) | Benchmarks are classification-only ([TSV](https://openaccess.thecvf.com/content/CVPR2025/papers/Gargiulo_Task_Singular_Vectors_Reducing_Task_Interference_in_Model_Merging_CVPR_2025_paper.pdf), [ACE](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_ACE-Merging_Data-Free_Model_Merging_with_Adaptive_Covariance_Estimation_CVPR_2026_paper.pdf)) | 3–6 |
| 7 (B) | "Optimal synthetic fraction is compute-dependent": phase diagram on ImageNet-100 | [2312.04567](https://arxiv.org/pdf/2312.04567); [2410.16713](https://arxiv.org/html/2410.16713v2) | 6–12 |
| 8 | "Is the task-vector subspace the bottleneck?" Probe *why* direct few-shot optimization beats merging (which layers and directions sit outside the span) | [2610.07990](https://arxiv.org/abs/2610.07990) (Oct 2026; fresh, moderate scoop risk) | 2–5 |
| 9 (G) | "When is diffusion attribution just retrieval?" | [2605.17938](https://arxiv.org/html/2605.17938v1); [2511.10721](https://arxiv.org/html/2511.10721) | 8–15 (borderline) |
| 10 (J) | Geometry predicts delayed generalization in small-data ViTs | [Zheng 2024](https://proceedings.mlr.press/v285/zheng24a.html); [2405.20233](https://arxiv.org/abs/2405.20233v2) | 3–6 (low CVPR appeal) |

**Saturated or low-yield directions (avoid as method papers):**
- New dataset-distillation methods on CIFAR or ImageNet-1K with soft labels. The CVPR 2026 oral shows soft-label KD gains are compute-dominated and nearly independent of the coreset. — [2604.18811](https://arxiv.org/abs/2604.18811)
- New SVD/subspace/covariance merging methods on the CLIP 8/14/20-task benchmark: at least 5 such papers from CVPR 2025–2026 and Sep 2026 are cited in Q2.
- Self-consuming loop papers re-showing that "accumulate real data avoids collapse" on small image datasets. — [2404.01413](https://arxiv.org/pdf/2404.01413v2), [2511.12742](https://arxiv.org/abs/2511.12742)
- Feature-similarity attribution baselines, and timestep-resolved diffusion attribution (just taken by [2610.11238](https://arxiv.org/abs/2610.11238)).
- Grokking on modular arithmetic or toy tasks (not CVPR-relevant).
- "VLMs underuse their encoder" diagnostics: already published ([2506.08008](https://arxiv.org/abs/2506.08008)) and overlapping the existing report's idea #2.

### Inferences
- Combining picks 1 and 6 ("fine-tuning and merging both destroy geometric features first, and artifact tokens explain much of it") gives one coherent story covering forgetting, merging and free-lunch phenomena on public checkpoints, for about 8–14 GPU-days (my estimate). That fits about 34 days alongside LineageRAFT.
- Pick 2 is the cheapest stand-alone option with a clean claim. Pick 3 has the strongest "everyone relies on this practice" appeal for reviewers.

### Gaps
- The GPU-day figures are unbenchmarked estimates. Run a 1-config pilot (one generator, one fine-tune, one proxy run) in the first 2 days to calibrate them.
- I did not do a full scoop check on ideas 1, 2, 3 and 6 against Oct 2026 arXiv. Run a targeted arXiv search, e.g. "dense probe forgetting DINOv2", "early mergeability", "proxy model vision data selection learning rate", before committing.
