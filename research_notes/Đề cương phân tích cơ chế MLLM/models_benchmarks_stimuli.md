# Experimental Substrate for a Cross-Family "Late-Layer Readout Bottleneck" Study of MLLMs (CVPR 2027, 1× H200, deadline 2026-11-16)

Notes compiled 2026-10-13.

**How things were verified:**
- HF model and dataset availability, creation dates, gating, licence tags and `base_model` lineage come from live calls to the Hugging Face API (`huggingface.co/api/models/<id>`, `/api/datasets/<id>`) made today. Model links below point to the matching HF page.
- GitHub repo existence, stars, last push date and licence come from live calls to the GitHub API made today.
- arXiv titles and dates come from the arXiv export API.
- Encoder and token facts marked **[cfg]** were read from the live `config.json` or README. Facts marked **[prior knowledge, not re-fetched]** come from my memory of the model cards and papers, not from a page fetched today; check them before citing.
- "ERR Invalid username or password" from the HF API means the repo id does not exist or is private. It is listed as **not found**.

---

## Q1. Which open MLLM families (≥4, diverse encoders/architectures) to use, and their HF availability, licence, encoder, projector and visual-token budget

### Takeaway
A core panel of 5–6 models covers every major architecture axis, and each model fits on one H200 in bf16:
- Qwen3-VL-8B (SigLIP2, DeepStack multi-layer injection)
- InternVL3.5-8B (InternViT-300M, pixel-shuffle MLP)
- LLaVA-OneVision-7B or LLaVA-OV-1.5-8B (SigLIP or RICE-ViT, anyres MLP; open data)
- Molmo2-8B or Molmo2-O-7B (SigLIP2; fully open data, and with OLMo-3 a fully open LLM)
- Gemma-4-12B "Unified" (**encoder-free**, raw patches projected into the LLM), or Gemma-3-12B (SigLIP)
- Llama-3.2-11B-Vision (**cross-attention** fusion, not prefix tokens)

The encoder-free Gemma-4-12B and the cross-attention Llama-3.2-Vision are the two strongest "architecture diversity" controls for a claim that the bottleneck is late-layer readout. Avoid Qwen3.5 as a primary model: its 3:1 linear/full-attention hybrid complicates patching.

### Cited Findings
**Qwen family**
- **Qwen2.5-VL-7B-Instruct.** Exists, Apache-2.0, ungated, created 2025-01-26, about 5.2M downloads. There is no public separate "Qwen/Qwen2.5-VL-7B" base repo (API: not found) — [HF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **Qwen3-VL-8B-Instruct and Qwen3-VL-8B-Thinking.** Both exist, Apache-2.0, ungated, created 2025-10-11. There is no public `Qwen/Qwen3-VL-8B` base repo (not found) — [HF Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [HF Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
- **Qwen3-VL-8B config [cfg].**
  - Vision tower: depth 27, hidden 1152, patch 16, spatial_merge 2, so one LLM token covers 32×32 px.
  - `deepstack_visual_indexes` = [8, 16, 24].
  - The LLM has 36 layers.
  - Source: [config.json](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)
- **Qwen3-VL tech report.** The encoder is SigLIP2-SO-400M, continued with dynamic resolution. DeepStack adds features from three ViT levels into the first three LLM layers. A two-layer MLP merges each 2×2 group of patches into one token — [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/html/2511.21631)
- **Qwen3.5-9B** (created 2026-02-27, Apache-2.0).
  - "Early fusion training on multimodal tokens", with a vision encoder (27 layers, hidden 1152, patch 16, merge 2) [cfg].
  - The 32-layer text stack interleaves `linear_attention` ×3 with `full_attention` ×1 (full_attention_interval = 4) [cfg].
  - Source: [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)

**InternVL family**
- **InternVL3-8B and InternVL3.5-8B.** Apache-2.0, ungated.
  - InternVL3.5-8B-Pretrained lists `InternViT-300M-448px-V2_5` and `Qwen3-8B` as its merged bases.
  - Each 448px tile gives 1024 ViT tokens, which pixel shuffle compresses to 256 LLM tokens.
  - Source: [HF InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B), [HF Pretrained](https://huggingface.co/OpenGVLab/InternVL3_5-8B-Pretrained)
- **InternVL3.5-8B-HF** is the transformers-native format and exists (created 2025-08-29) — [HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF)

**LLaVA family**
- **LLaVA-OneVision-qwen2-7b-ov.** Apache-2.0. Exists in both the original lmms-lab format and the `llava-hf/...-ov-hf` transformers format — [HF lmms-lab](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov), [HF llava-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
  - **[prior knowledge, not re-fetched]:** SigLIP-so400m-384 + Qwen2-7B, 2-layer MLP, anyres-9 with up to roughly 7k visual tokens per image; 729 tokens per base crop.
- **LLaVA-OneVision-1.5-8B-Instruct** (2025-09-16, Apache-2.0).
  - Its base_model tag is `DeepGlint-AI/rice-vit-large-patch14-560`, a RICE-ViT encoder, which adds a fourth encoder family.
  - It is described as a "fully open-source family" with released pretraining and SFT data.
  - Source: [HF](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct)

**Gemma family**
- **Gemma-3-12B-it.** Gemma licence, **gated (manual)**, base is `gemma-3-12b-pt` — [HF](https://huggingface.co/google/gemma-3-12b-it)
  - **[prior knowledge, not re-fetched]:** SigLIP-400M encoder, 256 soft tokens per 896² image, Pan&Scan.
- **Gemma-4-12B-it** (created 2026-05-23).
  - **Apache-2.0** and **ungated**, with a matching `google/gemma-4-12B` base.
  - The card says: "Gemma 4 12B eliminates these encoders entirely, projecting raw image patches and audio waveforms directly into the LLM". The architecture class is `Gemma4UnifiedForConditionalGeneration`.
  - Config: `num_soft_tokens` = 280, `model_patch_size` = 48, `pooling_kernel_size` = 3 [cfg].
  - Other Gemma 4 sizes (E2B, E4B, 26B-A4B, 31B) do use encoders, of roughly 150M–550M parameters.
  - Source: [HF card](https://huggingface.co/google/gemma-4-12B-it)

**Molmo family**
- **Molmo-7B-D-0924** (base Qwen2-7B) and **Molmo-7B-O-0924** (base_model tag `openai/clip-vit-large-patch14-336`; LLM OLMo-7B per Molmo paper). Both Apache-2.0 — [HF D](https://huggingface.co/allenai/Molmo-7B-D-0924), [HF O](https://huggingface.co/allenai/Molmo-7B-O-0924)
- **Molmo2-8B** (Qwen3-8B + SigLIP 2, Apache-2.0, created 2025-12-14) and **Molmo2-4B** — [HF](https://huggingface.co/allenai/Molmo2-8B)
- **Molmo2-O-7B** exists, with base `allenai/Olmo-3-7B-Instruct` (fully open LLM) and Apache-2.0 — [HF](https://huggingface.co/allenai/Molmo2-O-7B)
- **Molmo2 paper.** Its data were "all collected without the use of closed VLMs". It reports "bi-directional attention on vision tokens", which matters for patching, because image tokens attend to each other non-causally — [Molmo2 (arXiv 2601.10611)](https://arxiv.org/abs/2601.10611)

**Others**
- **Llama-3.2-11B-Vision-Instruct.** Llama 3.2 licence, **gated (manual)** — [HF](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
  - **[prior knowledge, not re-fetched]:** image features enter through cross-attention layers interleaved into Llama-3.1-8B, not as prefix tokens.
- **Idefics3-8B-Llama3.** Apache-2.0, ungated — [HF](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
  - **[prior knowledge]:** SigLIP-so400m + Llama-3.1-8B, pixel-shuffle, 169 tokens per 364² tile.
- **PaliGemma2-10b-mix-448.** Gemma licence, gated, only about 442 downloads — [HF](https://huggingface.co/google/paligemma2-10b-mix-448)
  - **[prior knowledge]:** prefix-LM with full attention over image and prefix, SigLIP. Its "mix" checkpoint is task-prefix-tuned rather than chat-tuned, which makes it awkward for open-ended prompts.
- **Additional verified-available options:**
  - GLM-4.1V-9B-Base and -Thinking (MIT) — [HF](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking)
  - Kimi-VL-A3B-Instruct and -Thinking (MIT, MoE) — [HF](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)
  - MiniCPM-V-4.5 (Apache-2.0) — [HF](https://huggingface.co/openbmb/MiniCPM-V-4_5)
  - Pixtral-12B-2409 (Apache-2.0) — [HF](https://huggingface.co/mistralai/Pixtral-12B-2409)
  - Eagle2.5-8B (gated "auto", licence other) — [HF](https://huggingface.co/nvidia/Eagle2.5-8B)
  - FastVLM-7B (apple-amlr licence) — [HF](https://huggingface.co/apple/FastVLM-7B)

**Interpretability tooling**
- nnsight (MIT, pushed 2026-10-09) — [GitHub](https://github.com/ndif-team/nnsight)
- TransformerLens (MIT, pushed 2026-10-12) — [GitHub](https://github.com/TransformerLensOrg/TransformerLens)
- lmms-eval (pushed 2026-10-12) — [GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- VLMEvalKit (Apache-2.0, pushed 2026-10-13) — [GitHub](https://github.com/open-compass/VLMEvalKit)

### Inferences
**Recommended core panel** (each model is about 16–26 GB in bf16, so two or three can sit in H200 memory at once):

| # | Model | Vision path | Fusion / projector | LLM tokens per image | Why it's in the panel |
|---|---|---|---|---|---|
| 1 | Qwen3-VL-8B-Instruct (+ Thinking) | SigLIP2-so400m, patch16 | 2×2-merge MLP + **DeepStack** into LLM layers 0–2 | dynamic (one token per 32 px²) | strongest open 8B; DeepStack injects visual features below layer 3 only, so late-layer effects are not "late injection" |
| 2 | InternVL3.5-8B (+ Pretrained / Instruct / MPO ladder) | InternViT-300M-448 | pixel-shuffle + MLP | 256 per tile | different encoder lineage; gives a full training-stage ladder (see Q6) |
| 3 | LLaVA-OV-7B **or** LLaVA-OV-1.5-8B | SigLIP **or** RICE-ViT | 2-layer MLP, anyres | up to thousands | open training data (contamination check); the most "vanilla" LLaVA design |
| 4 | Molmo2-8B **or** Molmo2-O-7B | SigLIP2 | MLP; bidirectional attention over vision tokens | — | fully open data (O-7B also has an open LLM), so it rules out benchmark leakage; trained on pointing and counting data |
| 5 | Gemma-4-12B-it | **encoder-free** (raw 48 px patches, then pooling) | linear projection | 280 by default | tests whether a "bottleneck" survives with no pretrained vision encoder |
| 6 (optional) | Llama-3.2-11B-Vision-Instruct | ViT-H (prior knowledge) | **cross-attention** layers | — | the bottleneck must be defined differently when the image never enters the residual stream as tokens; a strong falsification test |

- Qwen2.5-VL-7B is worth keeping as a 7th model only because most R1-style RL checkpoints are built on it (Q6).
- Gated models (Gemma-3, Llama-3.2-V, PaliGemma2) need a manual licence click. Request access **now**.
- Qwen3.5-9B: linear-attention layers have no per-token attention matrices to inspect, so attention-based analyses fail. Residual-stream patching still works. Treat it as a "newest model" robustness check only.
- Molmo-7B-D was trained on PixMo-Count, so it has an in-distribution advantage on PixMo-Count and CountBench-like data (see the SITUATE note in Q2). Report this explicitly.

### Gaps
- Llama-3.2-Vision, LLaVA-OV, Idefics3, Gemma-3 and Molmo-v1 encoder and token counts were **not re-fetched today**. Verify them from each `config.json`.
- I could not tell from the README excerpt whether Gemma-4-12B "Unified" has any learned patch-embedding layer beyond linear projection. The card table showed "-" for encoder parameters in one column, and I could not map columns to sizes reliably. Read the full card.
- I did not check whether the Molmo2-8B HF repo includes the remote-code processor needed for transformers ≥4.5x hooks.

---

## Q2. Benchmarks per failure family, blind/text-only baselines, and saturation/contamination

### Takeaway
Every family has a usable benchmark on HF (all verified today). The main risks:
- Template-generated **CV-Bench is exploitable without the image**: test-set text-only training adds +13.1 points.
- **CountBenchQA is near-saturated** for Qwen3-VL-8B (about 88%).
- Molmo models are in-distribution on PixMo-Count.

So use real-image benchmarks only for external validity, and use **controlled stimuli (Q3) plus counterfactual pairs (Q4)** as the primary measurement substrate.

### Cited Findings
**Blind baselines and shortcuts**
- MMStar authors: "visual content is unnecessary for many samples". GeminiPro gets 42.9% on MMMU without images. Sphinx-X-MoE gets 43.6% on MMMU without images, versus 17.9% for its LLM backbone, which indicates leakage — [MMStar (arXiv 2403.20330)](https://arxiv.org/abs/2403.20330)
- Brown et al. trained a text-only Qwen2-7B by cross-validation on the test-set questions and options (Test-set Stress-Test, TsT).
  - It exceeds zero-shot by **+17.9 on VSI-Bench and +13.1 on CV-Bench**.
  - Iterative Bias Pruning and VSI-Bench-Debiased lower a fine-tuned model's blind score from 44.7 to 32.0, while its vision score falls only from 57.1 to 48.7.
  - v2 is at COLM 2026.
  - Source: [Benchmark Designers Should "Train on the Test Set" (arXiv 2511.04655)](https://arxiv.org/abs/2511.04655)
- A constant no-image bias gets 86.86% on POPE, versus 85.89% for the best contrastive-decoding method — [Rethinking-CD](https://github.com/Aurnawr/Rethinking-CD) (via background notes)

**Counting**
- **CountBenchQA:** `vikhyatk/CountBenchQA`, fewer than 1K items, available — [HF](https://huggingface.co/datasets/vikhyatk/CountBenchQA)
- **PixMo-Count:** `allenai/pixmo-count`, 10K–100K items, ODC-BY — [HF](https://huggingface.co/datasets/allenai/pixmo-count)
- **TallyQA test:** `vikhyatk/tallyqa-test` — [HF](https://huggingface.co/datasets/vikhyatk/tallyqa-test)
- **CV-Bench** (count, relation, depth, distance): `nyu-visionx/CV-Bench`, Apache-2.0 — [HF](https://huggingface.co/datasets/nyu-visionx/CV-Bench)
- **Saturation (search snippet only):** Qwen3-VL-8B scores **87.98%** on CountBenchQA, rising to 90.22% after counting fine-tuning — [Counting Circuits (arXiv 2603.18523)](https://arxiv.org/pdf/2603.18523)
- The Molmo card frames its own "Flickr Count" as "significantly harder than CountBenchQA" — [Molmo card (mirror)](https://huggingface.co/ctranslate2-4you/Molmo-7B-D-0924)
- A synthetic-counting paper notes that Molmo-7B-D is the only model known to have been trained on PixMo-Count — [SITUATE (arXiv 2602.00108)](https://arxiv.org/pdf/2602.00108)
- **HoloCount** (2026-07): a three-level taxonomy covering semantic, analytical and robustness counting, including "grounded counter-priors". Over 20 MLLMs degrade on the analytical and adverse levels. I did not find the HF dataset under `zli12321/HoloCount` — [arXiv 2607.06420](https://arxiv.org/abs/2607.06420)
- **Counting mechanism papers** (competing accounts the study must engage):
  - "Visual Activation Patching" and HeadLens find a counting circuit "largely shared across a variety of visual reasoning tasks" — [Counting Circuits (arXiv 2603.18523)](https://arxiv.org/abs/2603.18523). This is the **closest prior work to the unified-bottleneck claim**, so differentiation is needed.
  - The "fractured magnitude hypothesis" holds that the collapse is at the symbolic-mapping stage, while backbones keep linearly separable quantity representations. It is tested on synthetic Go boards — [arXiv 2605.30170](https://arxiv.org/abs/2605.30170)
  - Counting has also been studied with activation patching and the CountScope tool — [Hasani et al. (arXiv 2511.17699)](https://arxiv.org/abs/2511.17699)

**Spatial relations**
- **What'sUp:** 2023 paper — [arXiv 2310.19785](https://arxiv.org/abs/2310.19785). Data are on HF in community mirrors only: `amitakamath2/whatsup_vlms`, `Mayfull/whats_up_vlms`, `ServiceNow/whatsup_all`. Code is at [GitHub amitakamath/whatsup_vlms](https://github.com/amitakamath/whatsup_vlms) (MIT, last push 2024).
- **SpatialEval:** `MilaWang/SpatialEval` (10K–100K items) — [HF](https://huggingface.co/datasets/MilaWang/SpatialEval), [arXiv 2406.14852](https://arxiv.org/abs/2406.14852)
- **VSR:** `cambridgeltl/vsr_random`, CC-BY-4.0 — [HF](https://huggingface.co/datasets/cambridgeltl/vsr_random)
- **BLINK:** `BLINK-Benchmark/BLINK`, Apache-2.0 — [HF](https://huggingface.co/datasets/BLINK-Benchmark/BLINK), [arXiv 2404.12390](https://arxiv.org/abs/2404.12390)
- **VSI-Bench:** video spatial, with a debiased version — [HF](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)

**Attribute versus language prior**
- **Visual CounterFact** (`mgolov/Visual-Counterfact`, 1K–10K items) — [HF](https://huggingface.co/datasets/mgolov/Visual-Counterfact)
  - Realistic counterfactuals pit priors against pixels (e.g. a blue strawberry), for **color and size**.
  - Predictions "first reflect memorized priors, then shift toward visual evidence in mid-to-late layers".
  - PvP steering shifts 99.3% of color and 80.8% of size predictions.
  - Source: [Pixels Versus Priors (arXiv 2505.17127)](https://arxiv.org/abs/2505.17127)
- **VLMs are Biased / VLMBias** (`anvo25/vlms-are-biased`, MIT, 10K–100K items) — [HF](https://huggingface.co/datasets/anvo25/vlms-are-biased), [GitHub](https://github.com/anvo25/vlms-are-biased)
  - Counting on edited images of well-known subjects (e.g. a 4-stripe Adidas logo) across domains including animals, logos, chess, board games, illusions and patterned grids.
  - Average counting accuracy **17.05%**. Removing backgrounds adds +21.09 points.
  - Source: [arXiv 2505.23941](https://arxiv.org/abs/2505.23941)
- **ColorBench:** `umd-zhou-lab/ColorBench` — [HF](https://huggingface.co/datasets/umd-zhou-lab/ColorBench)
- **Arbitration Failure, Not Perceptual Blindness:** the attribute-conflict mechanistic precedent — [arXiv 2604.09364](https://arxiv.org/abs/2604.09364)

**Fine-grained recognition**
- **MMVP:** `MMVP/MMVP` (MIT, fewer than 1K items), plus `MMVP/MMVP_VLM` — [HF](https://huggingface.co/datasets/MMVP/MMVP), [Eyes Wide Shut (arXiv 2401.06209)](https://arxiv.org/abs/2401.06209)
- **NaturalBench:** `BaiqiL/NaturalBench`, paired natural adversarial samples — [HF](https://huggingface.co/datasets/BaiqiL/NaturalBench), [arXiv 2410.14669](https://arxiv.org/abs/2410.14669)
- **FGVC-Aircraft:** `Voxel51/FGVC-Aircraft`, `clip-benchmark/wds_fgvc_aircraft` — [HF](https://huggingface.co/datasets/Voxel51/FGVC-Aircraft)

**Small text and small objects**
- **V\*:** `craigwu/vstar_bench` — [HF](https://huggingface.co/datasets/craigwu/vstar_bench), [arXiv 2312.14135](https://arxiv.org/abs/2312.14135)
- **HR-Bench 4K/8K:** `DreamMr/HR-Bench`, licence "other" — [HF](https://huggingface.co/datasets/DreamMr/HR-Bench), [arXiv 2408.15556](https://arxiv.org/abs/2408.15556)
- **TextVQA:** `facebook/textvqa`, `lmms-lab-encoder/textvqa` — [HF](https://huggingface.co/datasets/facebook/textvqa)
- **MME-RealWorld(-Lite)** — [HF](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite)
- **Saturation (partial):** The Qwen3-VL report gives V\* 93.7 for the 235B flagship; I found no 8B number on the fetched part of the page — [arXiv 2511.21631](https://arxiv.org/html/2511.21631)

**Low-level perception**
- **BlindTest ("VLMs are blind"):** `XAI/vlmsareblind`, MIT — [HF](https://huggingface.co/datasets/XAI/vlmsareblind)
  - Four SOTA VLMs average 58.07% versus about 100% expected for humans.
  - "Linear probing experiments show that vision encoders contain sufficient visual information … language models fail to decode this information". This is direct support for the readout-bottleneck framing.
  - Source: [arXiv 2407.06581](https://arxiv.org/abs/2407.06581)

### Inferences
- **Primary set per family** (small, unsaturated, with paired or controlled structure):
  - Counting: VLMBias count subset, CounterCount, PixMo-Count (excluding Molmo-v1), CV-Bench-count after TsT filtering.
  - Spatial: What'sUp controlled subsets A and B, SpatialEval-real, CV-Bench-relation after TsT filtering.
  - Attribute versus prior: Visual CounterFact.
  - Fine-grained: MMVP, NaturalBench.
  - Small text and objects: V\*, HR-Bench-4K.
- Treat **CountBenchQA, TextVQA and VSR as saturated or legacy**. Report them only as sanity checks.
- **Run TsT (blind text-only model trained on the test set) on every MCQ benchmark you use.** Reviewers in 2026–27 will know 2511.04655.
- The **highest-prior-art risk is 2603.18523 ("counting circuit shared across visual reasoning tasks")**. The paper must show something beyond it: multiple architectures, including encoder-free and cross-attention; non-counting families; and a *readout* localisation as opposed to a circuit.

### Gaps
- I did not find exact Qwen3-VL-8B or InternVL3.5-8B scores on CV-Bench, BLINK, V\*, HR-Bench or MMVP, because the tech-report tables were truncated. Run your own evaluations (lmms-eval / VLMEvalKit) in week 1.
- The CounterCount dataset HF id was not found. Check the arXiv page for a link — [arXiv 2605.17826](https://arxiv.org/abs/2605.17826)
- No contamination audit was found for What'sUp or MMVP specifically.

---

## Q3. Controlled stimulus generators (vary difficulty along one axis; fit psychometric curves)

### Takeaway
Combine three sources:
1. **Your own PIL/matplotlib generator.** Dots for numerosity; two shapes for left/right, above/below and distance; color patches with controlled ΔE; text glyph size.
2. **The BlindTest code**, which is already parametric over line width, resolution and spacing.
3. **CLEVR / Super-CLEVR** for 3-D scenes with exact scene graphs.

A psychometric-curve template for MLLMs exists for contrast sensitivity (Weibull fits). I found **no** numerosity-psychometric study of MLLMs, so that is an open angle.

### Cited Findings
- **Vision LLMs are blind code:** [GitHub anguyen8/vision-llms-are-blind](https://github.com/anguyen8/vision-llms-are-blind) (MIT, 143 stars, pushed 2026-05-25).
  - The paper varies image resolution and line width.
  - It reports "near-100% accuracy when much more space is added to separate shapes", which is a natural one-axis difficulty knob.
  - Source: [arXiv 2407.06581](https://arxiv.org/abs/2407.06581)
- **VLMs-are-biased code** — [GitHub anvo25/vlms-are-biased](https://github.com/anvo25/vlms-are-biased) (MIT, pushed 2026-01-26). The paper calls it a "human-supervised automated framework" — [arXiv 2505.23941](https://arxiv.org/abs/2505.23941)
- **CLEVR generator:** [facebookresearch/clevr-dataset-gen](https://github.com/facebookresearch/clevr-dataset-gen) (654 stars, last push 2021, Blender-based).
- **Super-CLEVR:** [Lizw14/Super-CLEVR](https://github.com/Lizw14/Super-CLEVR) (pushed 2026-02-19), with data at `RyanWW/Super-CLEVR` — [HF](https://huggingface.co/datasets/RyanWW/Super-CLEVR)
- **BlenderProc:** [DLR-RM/BlenderProc](https://github.com/DLR-RM/BlenderProc) (GPL-3.0, 3.7k stars, pushed 2026-01-20).
- **Psychometric template:** MLLMs were shown gratings at varying contrast. A Weibull function was fitted to yes-proportions, the threshold was taken as the 50% point, and sensitivity as 1/threshold. Prompt variants were analysed separately — [Contrast Sensitivity in MLLMs (arXiv 2508.10367)](https://arxiv.org/pdf/2508.10367)
- **Prior counting studies with synthetic stimuli:**
  - LVLMs show "precise performance on small numerosities and noisy estimation for larger quantities" (human-like) — [arXiv 2603.18523](https://arxiv.org/abs/2603.18523)
  - Synthetic Go boards were used for the extrapolation probe — [arXiv 2605.30170](https://arxiv.org/abs/2605.30170)
- **CVPR 2026 precedent** for synthetic counting with patching (Hasani et al.) — [CVPR 2026 supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Hasani_Understanding_Counting_Mechanisms_CVPR_2026_supplemental.pdf)

### Inferences
- **Recommended generator: about 300 lines of PIL in one file, seeded and deterministic.** Each family gets one difficulty axis and fixed nuisance variables:
  - **Numerosity:** N ∈ 1..30 dots, with total area or convex hull controlled (the standard numerosity-psychophysics confound).
  - **Spatial:** offset Δx/Δy in pixels between two objects (left/right threshold), with distractor count as a second axis.
  - **Color versus prior:** hue rotation of a canonical object (e.g. banana hue sweep), giving a prior-conflict psychometric curve.
  - **Fine-grained:** morph between two shapes or classes, or a Vernier offset.
  - **Small text:** glyph height in px versus tokenizer patch size. Qwen3-VL uses 32 px per LLM token and Gemma-4 uses 48 px patches [cfg], so sweep glyph size across 0.25–2× the token footprint.
- Fit a logistic or Weibull per model and layer: the behavioural threshold versus the linear-probe threshold at each layer. **The probe-minus-behaviour threshold gap as a function of layer is the paper's core measurement.**
- Use Blender only if reviewers demand photorealism. Super-CLEVR already gives 3-D scenes with exact counts and relations.

### Gaps
- I found no published numerosity psychometric study of MLLMs. This is novelty for you, but it also means there is no reference curve to compare against.
- I did not check whether the BlindTest repo exposes a clean parametric API or only notebooks.

---

## Q4. Counterfactual minimal-pair images for activation patching

### Takeaway
Ready-made minimal pairs exist for counting and color/size:
- CounterCount (factual and counterfactual pairs with masks and boxes)
- Visual CounterFact (color and size)
- VLMBias (edited counts)
- MMVP and NaturalBench (paired images)
- What'sUp (same objects, different relation)
- EqBen and SugarCrepe

Synthetic rendering (Q3) gives the cleanest pairs at zero cost. For realistic pairs, **Qwen-Image-Edit-2509/2511 (Apache-2.0, ungated)** is the editing model to use. FLUX.1-Kontext-dev is gated with a non-commercial licence.

### Cited Findings
- **CounterCount:** paired factual and counterfactual images "with edited count-relevant attributes, verified answers, and localized evidence annotations". Models underweight attention on count-relevant visual tokens. Attention modulation gives up to +8% — [arXiv 2605.17826](https://arxiv.org/abs/2605.17826)
- **MVC dataset** (Symmetrical Visual Contrastive Optimization): minimal-contrast pairs covering object and attribute (color, shape, size) replacement, count modification and position flips, built by inpainting, editing and generation — [arXiv 2502.13928](https://arxiv.org/abs/2502.13928)
- **Hasani et al.** patched vision tokens between images of the same category but different counts to find the layer where patching stops mattering — [CVPR 2026 supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Hasani_Understanding_Counting_Mechanisms_CVPR_2026_supplemental.pdf)
- **What'sUp** controlled subsets on HF:
  - `ServiceNow/whatsup_controlled_images_a_inference_test`, `..._b_...`, `amitakamath2/whatsup_vlms` — [HF](https://huggingface.co/datasets/amitakamath2/whatsup_vlms)
  - Paper — [arXiv 2310.19785](https://arxiv.org/abs/2310.19785)
- **SugarCrepe:** HF splits `HuggingFaceM4/SugarCrepe_swap_att`, `_replace_rel`, `_swap_obj`, etc. These are text-side hard negatives on COCO images, not image pairs — [HF](https://huggingface.co/datasets/HuggingFaceM4/SugarCrepe_swap_att), [arXiv 2306.14610](https://arxiv.org/abs/2306.14610), [GitHub](https://github.com/RAIVNLab/sugar-crepe)
- **Winoground:** `facebook/winoground`, gated — [HF](https://huggingface.co/datasets/facebook/winoground)
- **EqBen:** images at `ytaek-oh/eqben-images` (community) — [HF](https://huggingface.co/datasets/ytaek-oh/eqben-images); code at [GitHub Wangt-CN/EqBen](https://github.com/Wangt-CN/EqBen) (Apache-2.0)
- **CounterCurate:** [GitHub HanSolo9682/CounterCurate](https://github.com/HanSolo9682/CounterCurate) (19 stars, no licence). I found no HF dataset.
- **Image editing models:**
  - Qwen-Image-Edit (2025-08-17), -2509, and **-2511 (2025-12-17)**: all Apache-2.0, ungated — [HF 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
  - FLUX.1-Kontext-dev and FLUX.2-dev: gated ("auto"), licence "other" — [HF](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)

### Inferences
- **Pairs to use, by family:**
  - Counting: synthetic dot pairs (N vs N+1, same layout except one dot) plus CounterCount and VLMBias.
  - Color: synthetic hue swap plus Visual CounterFact.
  - Spatial: synthetic mirror-swap plus What'sUp A and B.
  - Fine-grained: MMVP and NaturalBench pairs.
  - Small text: rendered word pairs that differ by one glyph at a given font size.
- Edited pairs need a check that the edit is minimal. Gate pairs by LPIPS or masked-pixel difference outside the edit region, and by an independent detector or OCR verifying the target attribute. Discard roughly 20–40% (my estimate, not sourced).
- **Patching protocol caveats:**
  - Qwen3-VL's DeepStack also writes visual features into LLM layers 0–2, so patching "image-token residuals at layer ℓ" must patch the DeepStack inputs too, or start at ℓ ≥ 3.
  - Molmo2 uses bidirectional vision attention.
  - Llama-3.2-V has no image tokens in the residual stream, so patch the cross-attention outputs instead.
- Qwen-Image-Edit is about 20B parameters (prior knowledge, not re-fetched) and runs on the H200 in bf16. Budget about 1–2 GPU-days for around 5k pairs (my estimate).

### Gaps
- I did not verify that the CounterCount and MVC datasets are publicly downloadable.
- No existing dataset of photorealistic minimal pairs for small-text or small-object identity was found.

---

## Q5. Controls reviewers expect

### Takeaway
The minimum control set:
- No-image (blind) evaluation and text-only test-set training (TsT)
- Shuffled or mismatched image
- A constant-bias or prior-only baseline
- MCQ option-order permutation with letter debiasing
- Open-ended versus MCQ formats
- At least 3 prompt paraphrases with per-prompt reporting
- Greedy decoding with fixed seeds
- A human or ideal-observer reference on synthetic stimuli

Several of these have direct 2024–2026 precedents that reviewers will cite.

### Cited Findings
- **Blind evaluation** reveals non-visual answerability and leakage (GeminiPro 42.9% on MMMU without images) — [MMStar (arXiv 2403.20330)](https://arxiv.org/abs/2403.20330)
- **TsT and IBP** are the 2025–26 standard for benchmark-level and item-level blind exploitability — [arXiv 2511.04655](https://arxiv.org/abs/2511.04655)
- **A no-image constant-bias baseline** beats contrastive-decoding methods on POPE, and decoding-mode confounds (sampling versus greedy) explain apparent gains — [Rethinking-CD](https://github.com/Aurnawr/Rethinking-CD)
- **Prompt-variant robustness** was analysed separately in the psychophysics MLLM study — [arXiv 2508.10367](https://arxiv.org/pdf/2508.10367)
- **Human reference:** BlindTest tasks have a stated human expectation of about 100% — [arXiv 2407.06581](https://arxiv.org/abs/2407.06581)
- **Causality versus decodability:** probe accuracy and causal importance dissociate in counting ViTs, so probes must be paired with interventions — [arXiv 2510.09794](https://arxiv.org/abs/2510.09794)
- **Encoding-versus-grounding precedent:** last-token patching does nothing, while full-sequence patching at arbitration layers flips 60–84% of outputs. Patch scope is therefore itself a control — [arXiv 2604.09364](https://arxiv.org/abs/2604.09364)

### Inferences
**Concrete control checklist (per model × family):**
1. **Blind:** no image, or a gray image with the same token count. The gray image keeps positions identical, which matters for patching baselines.
2. **Shuffled image:** an image from another item of the same family.
3. **Prior-only:** the most frequent answer per question template.
4. **TsT score** per benchmark.
5. **MCQ permutation:** all option orders, or a cyclic permutation, with accuracy averaged. Report the letter-bias entropy.
6. **Open-ended generation** with a regex or LLM judge, alongside **logit scoring** of the answer tokens. The readout claim needs logits anyway.
7. **Three or more prompt paraphrases.**
8. **Greedy decoding;** Thinking models evaluated both with and without thinking.
9. **Linear probe** on the vision encoder output, as a ceiling.
10. **Ideal observer:** a pixel-counting or analytic solver on the synthetic stimuli, giving a trivially perfect reference.
11. **Small human study:** about 10 participants on synthetic thresholds, for psychometric comparison.
12. **Random-direction and random-layer patching** baselines.
13. **Probe selectivity control:** control tasks or shuffled labels.

### Gaps
- I found no single 2026 paper that codifies MCQ letter-permutation debiasing for MLLMs. Cite LLM MCQ-bias literature (not searched here) or justify it empirically.
- I did not find published human psychometric thresholds on the exact stimuli you will generate, so a small human study is needed.

---

## Q6. Base vs RL-tuned checkpoint pairs on HF (verified)

### Takeaway
The cleanest options:
- **InternVL3.5-8B training-stage ladder:** Pretrained, then Instruct (SFT), then MPO (offline RL), then final (CascadeRL). All four are public and Apache-2.0.
- **GLM-4.1V-9B-Base versus -Thinking.**
- **Kimi-VL-A3B-Instruct versus -Thinking.**
- **Gemma-4-12B pt versus it.**
- For R1-style GRPO, **Qwen2.5-VL-7B-Instruct** has many verified RL descendants.

### Cited Findings
- **InternVL3.5-8B lineage** from HF `base_model` tags: `InternVL3_5-8B-Pretrained` → `-Instruct` (finetune of Pretrained) → `-MPO` (finetune of Instruct) → `InternVL3_5-8B` (finetune of MPO).
  - The card describes the stages as CPT → SFT → CascadeRL (offline MPO, then online RL).
  - Source: [HF InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B), [HF -MPO](https://huggingface.co/OpenGVLab/InternVL3_5-8B-MPO), [HF -Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-8B-Instruct)
- **InternVL3-8B** also has `-Pretrained` → `-Instruct` → `InternVL3-8B` (MPO) — [HF](https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct)
- **GLM-4.1V-9B-Base and -Thinking** (MIT, both finetunes of GLM-4-9B-0414) — [HF](https://huggingface.co/zai-org/GLM-4.1V-9B-Base), [arXiv 2507.01006](https://arxiv.org/abs/2507.01006)
- **Kimi-VL-A3B-Thinking and -Thinking-2506** are tagged as finetunes of Kimi-VL-A3B-Instruct (MIT) — [HF](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506)
- **Qwen3-VL-8B Instruct versus Thinking:** both exist. They are separate post-training branches, not a base→RL pair — [HF](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
- **Qwen2.5-VL-7B-Instruct RL descendants** that exist on HF today:

| Checkpoint | Notes | Link |
|---|---|---|
| VL-Rethinker-7B | Apache, base tag Qwen2.5-VL-7B-Instruct, arXiv 2504.08837 | [HF](https://huggingface.co/TIGER-Lab/VL-Rethinker-7B) |
| OpenVLThinker-7B | base tag Qwen2.5-VL-7B-Instruct, arXiv 2503.17352 | [HF](https://huggingface.co/ydeng9/OpenVLThinker-7B) |
| R1-Onevision-7B | base tag Qwen2.5-VL-7B-Instruct | [HF](https://huggingface.co/Fancy-MLLM/R1-Onevision-7B) |
| ViGoRL-7b-Spatial | base tag Qwen2.5-VL-7B-Instruct, arXiv 2505.23678 | [HF](https://huggingface.co/gsarch/ViGoRL-7b-Spatial) |
| ThinkLite-VL-7B | MIT, arXiv 2504.07934; base tag absent | [HF](https://huggingface.co/russwang/ThinkLite-VL-7B) |
| MM-Eureka-Qwen-7B | no tags | [HF](https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B) |
| Vision-R1-7B | Apache | [HF](https://huggingface.co/Osilly/Vision-R1-7B) |
| Perception-R1-7B | exists under `tongxiao2002`; link to the Perception-R1 paper (arXiv 2506.07218) not verified | [HF](https://huggingface.co/tongxiao2002/Perception-R1-7B) |

- **VLM-R1** (Qwen2.5-VL-**3B** OVD) — [HF](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321)
- **Not found on HF:** `lmms-lab/LLaVA-OneVision-1.5-8B-RL`, `Kangheng/PR1-Qwen2.5-VL-7B-Detection`, `Ahren09/ViGoRL-7B-Spatial`, and `xyliu6/NoisyRollout-Geo3K-7B` (the API call returned unparseable output).
- **Prior finding:** accuracy-only RLVR shows no significant perception gain over the base model (McNemar test) — [Perception-R1 (arXiv 2506.07218)](https://arxiv.org/abs/2506.07218) (via background notes)

### Inferences
- **Primary RL analysis:** the InternVL3.5-8B four-stage ladder. It holds data and architecture fixed and isolates the SFT → offline-RL → online-RL effects on the readout layers.
- **Secondary analysis:** Qwen2.5-VL-7B-Instruct versus {VL-Rethinker, ViGoRL-Spatial, OpenVLThinker}. These are external RL recipes, which tests whether any "late-layer only" change generalises.
- **Prediction to test:** RL changes the late-layer readout (logit-lens or tuned-lens crossover, patch-flip rates) but not the mid-layer probe accuracy.
- Reading out ViGoRL-Spatial on spatial stimuli is a natural matched-domain test.

### Gaps
- I did not verify that ThinkLite-VL-7B, MM-Eureka-Qwen-7B and Vision-R1-7B are from Qwen2.5-VL-7B-**Instruct**: their HF tags lack base_model. Check each card.
- Whether the HF Perception-R1-7B repo is the paper's official release is unverified.
- No Qwen3-VL-8B base (pre-post-training) checkpoint is public, so Qwen3-VL cannot be used for a clean base→RL diff.
