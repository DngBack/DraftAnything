# Trustworthy / Responsible Computer Vision: Open Problems for CVPR 2027 on 1 H200

Scope note: today is 2026-10-13, CVPR 2027 deadline is 2026-11-16 (~34 days). The existing shortlist in `reports/Ý tưởng nghiên cứu CVPR 2027.md` covers LineageRAFT, MLLM arbitration, RL-vs-readout, SAM2 memory error, 4D physics judge, 3D calibration, video-diffusion physics probe and REPA. It has **no trustworthy-vision item**, so none of the ideas below duplicate it. Every GPU-day figure is **my own estimate**, not taken from any source. Papers marked "(background, from prior knowledge)" are well-known 2023–2025 works whose arXiv IDs I did not re-open this session. Their links should be checked before citing.

## Why do AI-generated image detectors fail on new generators, and is there a principled explanation?

### Takeaway
The 2026 evidence converges on one mechanism. Detectors mostly learn **the generator's last stage**: the VAE decoder or upsampler leaves a global high-frequency/spectral fingerprint. They do not learn semantic or content-level fakeness. That explains why they generalize across generators that share a decoder family, why they collapse on generators with a different final stage, and why they flag VAE round-tripped real images as fake. Out-of-the-box zero-shot accuracy on recent commercial generators is very low (18–30% on Flux Dev, Firefly v4 and Midjourney v7).

### Cited Findings
- A zero-shot benchmark of 16 open detectors on 12 datasets found that Flux Dev, Firefly v4 and Midjourney v7 average only **18–30%** detection accuracy across methods. 64 generators (22%) defeat most detectors (<50% average accuracy). The best detector averages 75.0% and the worst 37.5%. Rankings are unstable across datasets (Spearman ρ 0.01–0.87). The authors conclude that **training-data alignment with the target generator matters more than architectural innovations** — [arXiv 2602.07814](https://arxiv.org/abs/2602.07814)
- On scientific figures, a detector trained on one generator often fails on another. Effort scores 97.57% on the "Banana" generator but 28.40% on "GPT" (the labels are assumed to mean Nano Banana and GPT-image; the source does not say so) — [SciFigDetect, arXiv 2604.08211](https://arxiv.org/pdf/2604.08211)
- **Inpainting Exchange (INP-X)**: detectors rely on the *global* spectral shift that the VAE encode/decode introduces across the whole image, not on the locally inpainted content. Restoring the original pixels outside the edit drops SOTA and commercial detectors from **91% to 55%**, near chance. The authors attribute this to high-frequency attenuation by the VAE bottleneck and call for "content-aware detection" (Nebioglu, Bilgiç, Popescu, Jan 2026) — [arXiv 2602.00192](https://arxiv.org/abs/2602.00192)
- "Final component" hypothesis: many diffusion and autoregressive generators share a last stage that maps latents to pixels. Passing real images through that component and training on the result (100 samples × 3 categories, DINOv3) gives **98.83%** average accuracy across 22 unseen-generator test sets, with a taxonomy of 21 generators by final component — [arXiv 2601.20461](https://arxiv.org/abs/2601.20461)
- Training with one reconstruction operator (VAE or DDIM) biases detectors toward that operator's artifacts, because the lossy bottleneck changes high-frequency detail, textures and edges. Adding SRGAN negatives is proposed as a fix — [arXiv 2605.14486](https://arxiv.org/abs/2605.14486)
- Dual Data Alignment: B-Free and AlignedForensics do not align JPEG format, which leaves JPEG shortcuts open. Aligning both pixel and frequency domains is proposed — [arXiv 2505.14359](https://arxiv.org/html/2505.14359v6)
- Detection via deviations from the real-image distribution, using masked fine-tuning that hides textured regions where generator-specific patterns concentrate — [arXiv 2601.03586](https://arxiv.org/abs/2601.03586)
- PROBE (ICML track): uses the detector as a critic to steer a generator toward hard samples. It argues training data covers only a small region of "generative space" — [arXiv 2605.24906](https://arxiv.org/abs/2605.24906)
- Benchmarks keep multiplying: NTIRE 2026 robust AIGI detection in the wild ([arXiv 2604.11487](https://arxiv.org/pdf/2604.11487)), AI-GenBench (chronological train/test splits) ([review](https://www.themoonlight.io/en/review/ai-genbench-a-new-ongoing-benchmark-for-ai-generated-image-detection)), GenSyn10 ([arXiv 2607.16283](https://arxiv.org/html/2607.16283v1)), DailyBench ([HF 2607.24016](https://huggingface.co/papers/2607.24016))
- Background, from prior knowledge: aligned real/fake pairs via VAE reconstruction (Rajan et al., ICLR 2025, [arXiv 2410.11835](https://arxiv.org/abs/2410.11835)); B-Free (Guillaro et al., CVPR 2025, [arXiv 2412.17671](https://arxiv.org/abs/2412.17671)); AEROBLADE, training-free detection via AE reconstruction error (Ricker et al., CVPR 2024, [arXiv 2401.17879](https://arxiv.org/abs/2401.17879)); UnivFD, CLIP features generalize (Ojha et al., CVPR 2023, [arXiv 2302.10174](https://arxiv.org/abs/2302.10174))

### Inferences
- The field has reached an implicit consensus ("detectors = decoder fingerprint detectors") but has never tested it **causally**. Nobody seems to have swapped *only* the decoder (same diffusion backbone, different VAE/upsampler; or a real image round-tripped through each decoder) and measured how much detector accuracy follows the decoder versus the generator. That would be a clean, cheap paper of the "one controlled intervention explains a whole literature" type.
- Corollary attack: if detectors read the decoder, then re-decoding a fake image through an *unseen* decoder (or a pixel-space/one-step decoder) should defeat every aligned-training detector. Conversely, re-decoding a real image through the known decoder should create false positives. The second is a fairness and liability issue: real photos edited with any VAE-based tool get flagged.
- Pixel-space and non-VAE generators (pixel diffusion, some AR models with different detokenizers) would be the natural stress test of the hypothesis.

### Gaps
- I found no paper that runs a decoder-swap causal ablation across many detectors. This is absence of evidence from roughly 3 searches, not proof that none exists. Before committing, search for "decoder swap", "VAE swap" and "re-decoding attack".
- I could not verify per-detector numbers for GPT-image or Nano Banana outside SciFigDetect.

## Do concept-erasure methods really erase? Latest "concepts come back" evidence

### Takeaway
No. Erased concepts come back through adversarial prompts or embeddings (2023–24), benign unrelated fine-tuning (concept resurgence), structural conditioning such as edge or depth maps (SGCR, Oct 2026), and stochastic trajectory expansion. The weight edits also leave a spectral footprint that reveals *which* concepts were erased (Tracer, Oct 2026). The open question is no longer "does it fail" but **why erasure is shallow**: where the concept still lives, and whether any erasure that keeps the model useful can be non-shallow.

### Cited Findings
- **Concept resurgence**: fine-tuning SD v1.4/v2.1 on *unrelated* images revives unlearned concepts without adversarial intent. Example: "automobile" erased with MACE reappears on the prompt "A photo of an airplane". About 16% resurgence after only 50 samples and 1000 LoRA steps (Suriyakumar, Alur, Sekhari, Raghavan, Wilson; ICLR 2025 DATA-FM workshop) — [arXiv 2410.08074](https://arxiv.org/abs/2410.08074v1); a follow-up proposes the PGU defense against task-agnostic fine-tuning revival — [emergentmind 2604.21041](https://www.emergentmind.com/papers/2604.21041) (secondary summary, not verified)
- **Structure-Guided Concept Reappearance (SGCR)**: a grey-box attack that uses edge or depth conditioning (ControlNet-style) to bring back erased content, bypassing text-focused defenses (Bao et al., *Cybersecurity* 9(1), Art. 227, Oct 2026). Seen only via a news write-up, so check the DOI — [bioengineer.org](https://bioengineer.org/edge-maps-and-depth-images-can-unlock-what-ai-safety-filters-erased/)
- **Tracer** (Deng et al., 4 Oct 2026): identifies *which* and *how many* concepts were erased, using spectral analysis of weight-change footprints. It needs no image generation and no knowledge of the unlearning algorithm. It is 150×–137,000× faster than MIA-based identification on image models and 133×–20,000× faster than brute force on video models — [arXiv 2610.05601](https://arxiv.org/abs/2610.05601)
- "When Are Concepts Erased From Diffusion Models?" (May 2025): some methods (UCE, ESD-X) keep latent traces that stochastic trajectory expansion can surface, while others show little reappearance — [arXiv 2505.17013](https://arxiv.org/html/2505.17013v1)
- AEGIS (ICLR 2026): mapping an erased concept to one generic target ("a photo") leaves class-level remnants that adversarial prompts exploit — [co-r-e summary](https://co-r-e.com/method/aegis-concept-erasure-diffusion)
- CVPR 2026 Findings: concept erasure by attention redirection with a trained eraser token. It reports better durability than prior methods "where erased concepts often reappear" — [CVF open access](https://openaccess.thecvf.com/content/CVPR2026F/papers/Schechter_Concept_Erasure_via_Attention_Redirection_CVPRF_2026_paper.pdf)
- Theory: robust concept erasure from a security-theoretic perspective — [arXiv 2509.12024](https://arxiv.org/html/2509.12024v1); scaling to thousands of concepts — [Pith 2604.16481](https://pith.science/paper/2604.16481); GRACE, geometry-guided retention — [arXiv 2609.12731](https://arxiv.org/pdf/2609.12731); survey — [arXiv 2502.14896](https://arxiv.org/pdf/2502.14896)
- Background, from prior knowledge: Circumventing Concept Erasure (Pham et al., ICLR 2024, [arXiv 2308.01508](https://arxiv.org/abs/2308.01508)); Ring-A-Bell (ICLR 2024, [arXiv 2310.10012](https://arxiv.org/abs/2310.10012)); UnlearnDiffAtk (ECCV 2024, [arXiv 2310.11868](https://arxiv.org/abs/2310.11868))

### Inferences
- Most published erasure methods edit **cross-attention K/V projections or the text pathway**. Attacks that bypass text (SGCR via structure, resurgence via unrelated fine-tuning, image-to-image) all suggest the concept still sits in the denoiser's visual or U-Net/DiT pathway. One unified explanation, "erasure cuts the *address* (text→concept binding), not the *content* (visual generator)", could be tested cheaply:
  1. Run linear probes or DAAM on the denoiser features of an erased model during img2img or ControlNet generation of the concept.
  2. Measure the minimal "re-addressing" needed to recover it (one token embedding, a 1-step LoRA, or a structural hint).
- Tracer shows erasure is *detectable from the weights*. The complementary open question is whether that spectral footprint *predicts* which erased concepts will resurge.
- Erasure in modern DiT/flow models (SD3.5, FLUX) and video models is much less studied than in SD1.4. Most attack papers still benchmark SD1.4.

### Gaps
- I did not open the SGCR primary paper, and the AEGIS details come from a secondary page.
- I found no paper that explains resurgence mechanistically, i.e. which weight subspace resurges and why benign fine-tuning restores it. This should be rechecked before committing.

## Watermark robustness against regeneration and editing: is invisible watermarking fundamentally broken?

### Takeaway
Post-hoc *pixel-level* invisible watermarks are, for practical purposes, broken against regeneration. There is a 2023 provable result, a 2026 information-theoretic argument, and a challenge-winning removal rate of about 96%. Ordinary, non-adversarial diffusion *editing* also erases them. In-generation *semantic* watermarks (Tree-Ring, Gaussian Shading) survive regeneration better but are **forgeable** with an unrelated proxy model (CVPR 2025 Oral). The open frontier is (a) the removal/forgery trade-off and (b) watermark survival under benign editing pipelines that users run daily.

### Cited Findings
- Regeneration attack (add noise, then reconstruct with a generative model) provably removes pixel-level invisible watermarks (Zhao et al., NeurIPS 2024) — [arXiv 2306.01953](https://arxiv.org/abs/2306.01953)
- "Vanishing Watermarks" (Feb 2026): diffusion editing erases robust watermarks. Regeneration cuts StegaStamp decoding accuracy to **7.4%** and TrustMark to **12.8%**. The paper argues the mutual information between image and payload goes to ~0 after enough diffusion transformations, and adds a guided attack. Author names unverified — [arXiv 2602.20680](https://arxiv.org/html/2602.20680v1)
- "When Denoising Becomes Unsigning" (Mar 2026): theoretical and empirical fragility under diffusion editing — [arXiv 2603.04696](https://arxiv.org/pdf/2603.04696)
- "Editing Away the Evidence" (Mar 2026): watermark loss from *ordinary* instruction, drag and composition editors, with no adversary — [arXiv 2603.12949](https://arxiv.org/pdf/2603.12949)
- "Watermarks Attack Watermarks" (May 2026): re-watermarking as a generic removal strategy (only the reference list was seen) — [arXiv 2605.16796](https://arxiv.org/pdf/2605.16796)
- The first-place NeurIPS 2024 watermark-removal challenge solution reported about **95.7%** removal with controlled-noise diffusion — [arXiv 2508.21072](https://arxiv.org/html/2508.21072v1)
- **Black-box forgery of semantic watermarks** (Müller et al., Ruhr-Bochum, **CVPR 2025 Oral**). Reprompting or inverting with an *unrelated* proxy model forges Tree-Ring and Gaussian Shading. With SD2.1 as proxy and SDXL as target, it reaches 0.995 detection/attribution on Gaussian Shading and 0.968 on Tree-Ring. A FLUX target resists more (65.6% GS, 12.8% TR) — [arXiv 2412.03283](https://arxiv.org/abs/2412.03283), [GitHub "CVPR2025 Oral"](https://github.com/and-mill/semantic-forgery)
- Geometric-distortion view of black-box forgery (June 2026) — [arXiv 2606.29807](https://arxiv.org/pdf/2606.29807); randomized key selection as a forgery mitigation — [arXiv 2507.07871](https://arxiv.org/pdf/2507.07871)
- Provenance: "Integrity Clash" is a valid C2PA manifest claiming human authorship while the watermark says AI, and each check passes alone. The paper also notes re-signing and soft-binding collision attacks — [arXiv 2603.02378](https://arxiv.org/html/2603.02378); a formal analysis of C2PA spec 2.2 finds several claimed security goals unmet — [IACR ePrint 2026/804](https://eprint.iacr.org/2026/804)
- Background, from prior knowledge: WAVES watermark-robustness benchmark (ICML 2024, [arXiv 2401.08573](https://arxiv.org/abs/2401.08573))

### Inferences
- The "removal" side is saturated: another regeneration attack on StegaStamp is not a CVPR contribution in 2027. The non-saturated insight is the **trade-off**. Semantic watermarks resist removal *because* they live in the initial-noise or semantic layout, and that is exactly why they can be forged by inversion. A paper that formalizes and empirically demonstrates an impossibility-style tension ("removal-robust ⇒ forgeable under inversion", across Tree-Ring, Gaussian Shading, PRC, WIND and similar, on DiT/flow models) would be insight-driven and cheap.
- The FLUX resistance (12.8% TR forgery) hints that **inversion error** of the proxy and target models governs forgery success. Measuring forgery success against proxy–target inversion mismatch could give a predictive law.
- Benign editing is the realistic threat for provenance. Nobody has checked how the C2PA soft binding plus watermark combination fails under Nano-Banana-style editing pipelines (not directly verified).

### Gaps
- I did not verify which 2025–2026 semantic watermarks (PRC watermark, WIND, etc.) have already been shown forgeable, so a literature check is needed before claiming the trade-off is new.

## Image-based jailbreaks of VLMs: what's open?

### Takeaway
Gradient-based adversarial images jailbreak white-box VLMs easily but **transfer poorly** across architectures, even when optimized against ensembles. 2026 work (UltraBreak, ICLR 2026) pushes universal and transferable images. Non-gradient "semantic" visual jailbreaks (typography, ciphers, riddles) need no gradients. Two things remain open: an *explanation* of when and why transfer happens (shared vision encoder? shared LLM base?), and defenses that are evaluated adaptively.

### Cited Findings
- UltraBreak (ICLR 2026): a single universal adversarial image optimized on an open surrogate with a semantic cosine loss, random transformations and TV regularization. It reports about 71% black-box average ASR on SafeBench (secondary review figure, not checked against the paper) — [arXiv 2602.01025](https://arxiv.org/pdf/2602.01025), [paper note](https://en.papernotes.org/ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models/)
- Simulated Ensemble Attack: PGD image jailbreaks rarely transfer across *fine-tuned* VLMs, but exploiting the shared base model gives >86.5% transfer ASR on Qwen2-VL variants. It cites Schaeffer et al.: gradient-based jailbreak images transfer poorly across architectures even when optimized against multiple VLMs — [arXiv 2508.01741](https://arxiv.org/html/2508.01741v1)
- Gradient-free visual jailbreaks via visual ciphers, object replacement and analogy riddles (May 2026) — [arXiv 2605.00583](https://arxiv.org/html/2605.00583v1)
- CVPR 2026 AdvML workshop challenge report exists (content not seen) — [arXiv 2607.11560](https://arxiv.org/pdf/2607.11560)
- Background, from prior knowledge: Visual adversarial examples jailbreak aligned LLMs (Qi et al., AAAI 2024, [arXiv 2306.13213](https://arxiv.org/abs/2306.13213)); FigStep typographic jailbreak ([arXiv 2311.05608](https://arxiv.org/abs/2311.05608)); Schaeffer et al., "Failures to Find Transferable Image Jailbreaks Between VLMs" ([arXiv 2407.15211](https://arxiv.org/abs/2407.15211), ID not re-verified this session)

### Inferences
- SEA's result (transfer works within a shared base, fails across architectures) is a partial explanation. The open, cheap, insight-style question is **what geometric quantity predicts transfer**. Candidates are the alignment between the source and target VLMs' projector outputs (CKA or mutual-kNN of image tokens), or whether the adversarial image's effect lives in the vision encoder or in the LLM's reading of visual tokens.
- This connects naturally to the existing shortlist's "arbitration" idea (#2) at the tooling level: probes and patching on the same 7–8B VLMs. That makes it a cheap *second* paper reusing infrastructure, but it is a different claim (safety, not perception), so it is not a duplicate.
- Universal-image jailbreak defenses (input purification, diffusion purification) probably fail against adaptive attacks, the classic Carlini/Tramèr pattern. An "adaptive evaluation of N VLM-jailbreak defenses" paper is a known-good genre, but scoop risk is high.

### Gaps
- I did not verify whether a 2026 paper already explains transferability via representation similarity.
- ASR numbers for UltraBreak come from a secondary review.

## Privacy (membership inference on diffusion/VLMs) and fairness: quick status

### Takeaway
MIA on foundation models has a known evaluation flaw: members and non-members are drawn from different distributions, so "blind" attacks that never query the model beat published MIAs. On realistic benchmarks, MIAs on diffusion models largely fail. The open question is whether *any* MIA on large T2I or VLMs works under properly distribution-matched splits. Fairness work is moving from output auditing to mechanism localization (attention heads, activation replacement), but most of it is in NLP/FAccT venues, not CVPR orals.

### Cited Findings
- Blind baselines that ignore the target model beat SOTA MIAs on foundation-model benchmarks, because members and non-members differ in distribution (Das, Zhang, Tramèr; SP Workshops 2025) — [arXiv 2406.16201](https://arxiv.org/abs/2406.16201)
- "Real-World Benchmarks Make Membership Inference Attacks Fail on Diffusion Models" — [arXiv 2410.03640](https://arxiv.org/pdf/2410.03640)
- The VLM MIA paper (USENIX Security 2025) assumes identical distributions for shadow/reference and target sets, and names distribution discrepancy as future work — [arXiv 2501.18624](https://arxiv.org/pdf/2501.18624)
- Single-sample black-box VLM MIA via cross-modal semantic alignment (May 2026) — [arXiv 2605.17341](https://arxiv.org/pdf/2605.17341)
- Fairness, mechanism level: controlled activation replacement to diagnose gender bias in VLMs — [arXiv 2609.16651](https://arxiv.org/pdf/2609.16651); it cites a 2026 preprint localizing demographic bias to CLIP vision-encoder heads (arXiv 2603.11793, seen only as a citation); T2I fairness survey — [arXiv 2604.16516](https://arxiv.org/pdf/2604.16516); INFELM finds most T2I models fail the four-fifths rule — [arXiv 2501.01973](https://arxiv.org/html/2501.01973)

### Inferences
- A cheap, insight-style CVPR paper: re-evaluate *2025–2026 VLM and T2I MIAs* (e.g. 2605.17341, 2501.18624) under blind baselines and distribution-matched splits (temporal or dedup-matched). Showing that a family of attacks reduces to a dataset-shift detector is the "Blind Baselines" recipe applied to vision. Risk: venue fit may be better at SaTML/S&P than CVPR.
- Fairness in generative or VLM models: the most CVPR-friendly angle is mechanistic. It also touches the detection topic: do AIGI detectors have demographic false-positive disparities, e.g. via skin-texture high frequencies? I found no source for this, so it is speculative.

### Gaps
- I did not find a 2026 paper that specifically re-audits *image* diffusion MIAs with blind baselines beyond 2410.03640, so this needs checking.
- I did not search fairness of AIGI detectors.

## Insight-driven papers with orals/awards/highlights (2024–2026)

### Takeaway
The pattern that wins in this area is "**a simple attack or test breaks a whole defense family**" or "**one controlled intervention explains a literature**". Confirmed examples from this session are the CVPR 2025 Oral on semantic-watermark forgery, NeurIPS 2024 provable watermark removal, ICLR 2026 UltraBreak, ICLR 2026 AEGIS, and the "blind baselines" MIA critique. I could not verify award or highlight status for most others in this session.

### Cited Findings
- Müller et al., Black-Box Forgery Attacks on Semantic Watermarks — **CVPR 2025 Oral**, a simple reprompting/inversion attack with an unrelated model — [GitHub](https://github.com/and-mill/semantic-forgery), [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Muller_Black-Box_Forgery_Attacks_on_Semantic_Watermarks_for_Diffusion_Models_CVPR_2025_paper.pdf)
- Zhao et al., Invisible image watermarks are provably removable — NeurIPS 2024 — [ACM DL](https://dl.acm.org/doi/10.5555/3737916.3738192)
- UltraBreak — ICLR 2026 — [arXiv 2602.01025](https://arxiv.org/pdf/2602.01025)
- AEGIS — ICLR 2026 — [summary](https://co-r-e.com/method/aegis-concept-erasure-diffusion)
- Das et al., Blind Baselines — ICLR 2025 DATA-FM workshop / SP Workshops 2025 — [arXiv 2406.16201](https://arxiv.org/abs/2406.16201)
- Concept resurgence — ICLR 2025 DATA-FM workshop — [ICLR page](https://iclr.cc/virtual/2025/34465)

### Inferences
- The cheapest route to a highlight in this area is a *negative-result-plus-explanation* paper with a memorable named phenomenon and a one-line fix or diagnostic. This matches the existing report's general strategy.

### Gaps
- Oral or highlight status for B-Free (CVPR 2025), the Rajan et al. aligned datasets paper (ICLR 2025) and any CVPR 2026 trustworthiness highlights was not verified. The CVPR 2026 highlight list should be checked directly.

## Which 8–12 open questions fit a small team with 1 H200 before 2026-11-16?

### Takeaway
Ten candidates are listed below, ranked by (insight × feasibility × low scoop risk). Top three for November 2026: **(1) Decoder-swap causal test of AIGI detectors**, **(2) "Erasure cuts the address, not the content"**, **(3) Removal–forgery trade-off for semantic watermarks**. Each core experiment is ≤5 GPU-days (my estimate) using public checkpoints only.

### Cited Findings (grounding per idea; see sections above for full citations)

| # | Claim-style title | Gap (sources) | Why CVPR reviewers care | Simple angle / minimal experiment | GPU-days (my estimate) | Main risk |
|---|---|---|---|---|---|---|
| 1 | **"Detectors detect decoders": a causal decoder-swap test of AIGI detector generalization** | Global VAE spectral shift drives detection ([2602.00192](https://arxiv.org/abs/2602.00192)); final-component training generalizes ([2601.20461](https://arxiv.org/abs/2601.20461)); zero-shot collapse on Flux/MJ7 ([2602.07814](https://arxiv.org/abs/2602.07814)) | Explains a whole literature with one intervention; actionable for benchmark design | Fix the content and swap only the decoder: (a) real images round-tripped through ~10 public VAEs/detokenizers (SD1.5, SDXL, SD3, FLUX, consistency decoders, AR tokenizers); (b) fakes re-decoded through unseen decoders. Run ~10 public detectors (UnivFD, AEROBLADE, B-Free, DDA, etc.) and measure the share of variance explained by decoder versus generator. Fix: decoder-randomized alignment | 1–3 (mostly inference) | Partially implied by INP-X and the final-component paper; must frame as *causal and quantitative* |
| 2 | **"Re-decoding attack": one VAE pass defeats aligned-training detectors** (could merge with #1 as its attack section) | Single-operator bias ([2605.14486](https://arxiv.org/abs/2605.14486)); JPEG shortcuts in B-Free/AlignedForensics ([2505.14359](https://arxiv.org/html/2505.14359v6)) | Simple attack, whole defense family fails; also false positives on real photos edited with VAE tools | Fake → encode/decode with a different public VAE, plus a real-photo VAE round trip to measure FPR. Report the rate of real images flagged after common editing tools | <1 | Low novelty alone; strongest merged with #1 |
| 3 | **"Erasure cuts the address, not the content"**: a mechanistic explanation of why erased concepts come back | Resurgence after benign fine-tuning ([2410.08074](https://arxiv.org/abs/2410.08074v1)); structure-guided reappearance (SGCR); latent traces ([2505.17013](https://arxiv.org/html/2505.17013v1)); weight footprints ([2610.05601](https://arxiv.org/abs/2610.05601)) | Unifies four attack families with one explanation; tells defense designers where to edit | On 6–8 public erased checkpoints (ESD, UCE, MACE, RECE, AdvUnlearn, etc.), probe the denoiser's mid-block for the concept under img2img, ControlNet and textual inversion. Measure the minimal re-addressing cost (1 token / 1-step LoRA / an edge map). Predict resurgence from the probe score | 3–6 | Erased checkpoints mostly SD1.4, so reviewers may want a DiT; SGCR primary not read |
| 4 | **Does Tracer's spectral footprint predict resurgence?** (erasure auditing) | Tracer identifies erased concepts from weights ([2610.05601](https://arxiv.org/abs/2610.05601)); resurgence ([2410.08074](https://arxiv.org/abs/2410.08074v1)) | A training-free auditing tool for regulators | Compute the footprint; correlate with resurgence after k LoRA steps across methods and concepts | 2–4 | Tracer is 9 days old, so follow-up race risk |
| 5 | **"Robust-to-removal ⇒ forgeable": an empirical trade-off for semantic watermarks on DiT/flow models** | Forgery via an unrelated proxy (CVPR'25 Oral, [2412.03283](https://arxiv.org/abs/2412.03283)); FLUX partially resists; geometric forgery ([2606.29807](https://arxiv.org/pdf/2606.29807)) | Turns attack-of-the-month into a law; informs EU-AI-Act-style policy | Sweep watermark strength and design (Tree-Ring, Gaussian Shading, PRC, WIND) on SD3.5/FLUX. Plot removal-robustness against forgery success; explain both with proxy–target **inversion error** | 4–8 | Must check that PRC/WIND are not already shown forgeable |
| 6 | **Benign-edit survival of provenance**: watermarks plus C2PA soft binding under everyday instruction editors | Ordinary editors erase watermarks ([2603.12949](https://arxiv.org/pdf/2603.12949)); C2PA integrity clash ([2603.02378](https://arxiv.org/html/2603.02378)) | A realistic, non-adversarial threat model | Run ~6 open instruction editors over N watermark schemes and measure bit accuracy against edit strength (LPIPS). Show a "survives edit ⇒ survives regeneration attack" equivalence | 2–4 | Overlaps 2603.12949; needs a sharper claim |
| 7 | **What predicts visual-jailbreak transfer?** Representation alignment of image tokens | Poor cross-architecture transfer; shared-base transfer works ([2508.01741](https://arxiv.org/html/2508.01741v1)); UltraBreak ([2602.01025](https://arxiv.org/pdf/2602.01025)) | Explains a confusing literature; tells defenders what to diversify | Make adversarial images on 6–8 open VLMs, then correlate the transfer matrix with CKA/mKNN of projector outputs and a shared-vision-encoder indicator | 3–5 | Scoop risk medium; reuses the shortlist's MLLM tooling |
| 8 | **Adaptive evaluation of VLM image-jailbreak defenses** | Defenses are rarely evaluated adaptively (inference, no direct source) | The classic "all defenses fail" genre | 5–6 public purification or detection defenses + an adaptive PGD/BPDA attack | 2–4 | Security-venue fit; highly competitive genre |
| 9 | **"Blind baselines for vision MIAs"**: do 2025–26 VLM/T2I membership attacks survive distribution-matched splits? | Blind baselines ([2406.16201](https://arxiv.org/abs/2406.16201)); real-world benchmarks fail ([2410.03640](https://arxiv.org/pdf/2410.03640)); VLM MIA assumes identical distributions ([2501.18624](https://arxiv.org/pdf/2501.18624)) | Calls the evaluation of a subfield into question, cheaply | Re-run 3–4 recent attacks; add a model-free classifier baseline and temporally or dedup-matched splits | 1–3 | May be seen as an "S&P paper"; CVPR fit moderate |
| 10 | **Do AIGI detectors have demographic false-positive gaps?** (fairness × detection) | No source found (gap) | Liability: real photos of some groups get flagged | Run public detectors on demographically labelled real-face sets (e.g. FairFace) and test whether the FPR gap tracks high-frequency/skin-texture statistics | 1–2 | Speculative; might be a null result |

### Saturated directions (avoid)
- Yet another deepfake/AIGI detector trained on FF++ or GenImage with an in-distribution SOTA claim. The 2026 benchmarks show rankings unstable across datasets ([2602.07814](https://arxiv.org/abs/2602.07814)) and a flood of new benchmarks (NTIRE 2026, AI-GenBench, GenSyn10, DailyBench).
- Yet another AIGI benchmark: at least four appeared in 2026 alone (see above).
- Another regeneration or removal attack on post-hoc pixel watermarks: already provably and empirically solved ([2306.01953](https://arxiv.org/abs/2306.01953), [2602.20680](https://arxiv.org/html/2602.20680v1), [2508.21072](https://arxiv.org/html/2508.21072v1)).
- Another concept-erasure *method* evaluated only on SD1.4 with I2P/UnlearnDiffAtk. The space is crowded (CVPR 2026 Findings, ICLR 2026 AEGIS, GRACE, scaling to thousands of concepts).
- Another prompt-level or typographic VLM jailbreak without a transfer or mechanism explanation.

### Inferences
- **Best fit with the team's current portfolio:** #1+#2 merged ("Detectors detect decoders") is the closest to "one controlled intervention explains a literature". It is almost pure inference on public detectors and VAEs (≤3 GPU-days, my estimate), so it can run in parallel with the LineageRAFT submission without taking the GPU from it. #3 is the best concept-erasure option. #7 reuses MLLM probe tooling from shortlist idea #2.
- Following the existing report's rule (LineageRAFT plus at most one new paper), pick **one** of #1/#3/#5 for November, not several.

### Gaps
- None of the ideas has had a dedicated scoop check (a 2026 arXiv search on each exact claim). This should be done before committing; it takes about 1 hour per idea.
- I have not confirmed the licenses or availability of public erased checkpoints for #3, or of semantic-watermark code for SD3.5/FLUX for #5.
