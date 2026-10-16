# Contrastive Language Models (CLM): what it is and how it could transfer to vision and robotics

Research date: 2026-10-13. Sources: the GitHub repo was fetched directly, including the raw README, examples/t_rex/README.md, docs/FINETUNING.md and the GitHub API metadata. I also read the HF model card, arXiv API abstracts for the related work, and secondary coverage. The Notion blog (contrastive-lm.notion.site) returned an empty JS shell, so **I could not read the blog's scaling-law figures, exponents or compute numbers**. **There is no arXiv paper.**

**The user's framing needs one correction up front.** CLM does **not** replace next-token prediction for generating text. It is a **discriminative dual-encoder scorer**, which the authors call a "System One" model. It ranks a *given* set of candidate actions or answers against a state. Its backbone (Qwen3-8B) is still a frozen next-token-pretrained LLM. In practice CLM *complements* generators, acting as a fast selector or verifier on top of them.

---

## Q1. What exactly is CLM? (paper, loss, architecture, inference, results, compute, claimed shift)

### Takeaway
CLM is a CLIP/DPR-style bi-encoder. It has two small (20M-parameter) projection heads, one for the state and one for the action, on top of a frozen Qwen3-8B, trained with symmetric InfoNCE plus hard negatives. The training runs in three LLM-style stages: about 60M Q&A pairs, then about 30M synthetic hard negatives, then about 1M agent trajectories. At inference it embeds the state once and takes a dot product with cached candidate embeddings, then a softmax over the candidate set. It does not generate. The headline claims are speed (up to 9x faster than TypeSafe's closed "Jev" decision model) and verifier SOTA on small held-out agentic-coding subsets.

### Cited Findings
**Identity and release**
- Title: "Contrastive Language Models", subtitle "A System One Model for Fast and Generalizable Decision-Making". Authors: Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré, Azalia Mirhoseini. The BibTeX is `@misc{kwok2026contrastivelanguagemodels, note={Notion Blog}}`, so the venue is a blog post: no arXiv ID and no peer review. — [GitHub README](https://github.com/Contrastive-LM/CLM)
- The repo was created 2026-09-23 and last pushed 2026-10-05. It had about 3,006 stars and 262 forks on 2026-10-13. Code and weights are Apache-2.0. — [GitHub API](https://api.github.com/repos/Contrastive-LM/CLM); [HF model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
- The T-Rex example dates the CLM run to 2026-09-23. Press coverage places the release on 23 Sept 2026. — [t_rex README](https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/README.md); [gradually.ai](https://www.gradually.ai/en/ai-models/clm-v0.1-8b/)
- Press attributes the work to "Stanford and NVIDIA" or to Hazy Research. The README and model card name no institution. — [pasqualepillitteri.it](https://pasqualepillitteri.it/en/news/19074/clm-8b-nvidia-stanford-jev-en); [HF card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)

**Loss**
- The loss is bidirectional InfoNCE over a B×B similarity matrix of matched (state, action) pairs:
  `L = -1/(2B) Σ_i [ log softmax_j(s_i·a_j/τ)[i] + log softmax_j(a_i·s_j/τ)[i] ]`.
  In mid-training, the s→a direction gets per-state hard negatives h_ik:
  `-1/B Σ_i log exp(s_i·a_i/τ) / (exp(s_i·a_i/τ) + Σ_k exp(s_i·h_ik/τ))`. — [README "Training Algorithm"](https://github.com/Contrastive-LM/CLM)
- In other words, "each state is pulled toward the ground-truth action that was taken and pushed away from all others". — [README "About"](https://github.com/Contrastive-LM/CLM)

**Architecture**
- Each encoder is a frozen LLM backbone plus a 20M-parameter trainable projection head. The reference model uses Qwen3-8B with last-token pooling. The checkpoint holds `state_head`, `action_head`, `logit_scale` and `cfg` (width, depth, projection_dim, activation, layernorm, residual). The head download is about 75 MB. — [README](https://github.com/Contrastive-LM/CLM)
- Serving runs a vLLM Qwen3-8B pooling server on a GPU, which produces embeddings, plus `clm-serve`, which can run on CPU. States are truncated at 2,048 tokens by default. — [README](https://github.com/Contrastive-LM/CLM)

**Inference, and why it "generates" nothing**
- "Given the current state and a set of candidate actions, CLM scores each action by how well its embedding aligns with the state embedding and selects the highest-scoring action." Typed questions (Noul = yes/no probability, Choice, Score) are a state plus a closed candidate set, and "a softmax over CLM's scores *is* the answer distribution". The same `rank` primitive handles best-of-N selection, tool routing and retrieval shortlisting. — [README](https://github.com/Contrastive-LM/CLM)
- The API is wire-compatible with TypeSafe's commercial "Jev" (`POST /v1/systemone`). Jev is TypeSafe's closed "System One" model that returns typed decisions instead of text. It launched in mid-September 2026, and its size and architecture are undisclosed. — [t_rex README](https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/README.md); [OpenRouter Jev](https://openrouter.ai/models/typesafe/jev-1.13); [ZenMux blog](https://zenmux.ai/blog/jev-vs-llms-choosing-between-typesafe-ais-system-one-and-traditional-large-language-models)

**Data recipe**
- Pre-training: about 60M Nemotron DQA question–answer pairs, with the question as state and the answer as action.
- Mid-training: about 30M synthetic hard negatives (plausible but wrong answers) generated by Gemini 2.5 Flash-Lite.
- Post-training: about 1M agent trajectories from the Agent Data Protocol plus Endless-Terminals and LiteCoder-Terminal-SFT. Each step is a (context, decision) pair. The mix is 40% Nemotron replay and 60% agentic data.
- Source: [README "Data Recipe"](https://github.com/Contrastive-LM/CLM)
- Ablation on about 100K held-out questions (1 gold answer, 10 hard negatives each). Pre-training alone reaches 52.1% top-1. Adding a short mid-training stage reaches 69.2%. Training with hard negatives from the start peaks at 62.4% and then overfits. — [README](https://github.com/Contrastive-LM/CLM)
- Replay matters. With replay, hard-negative top-1 goes from 69% to 68.5%. Training on agentic data alone drops it to 56.2%. — [README](https://github.com/Contrastive-LM/CLM)

**Scaling laws**
- Test InfoNCE loss follows L(X) ≈ (X_c/X)^α_X for X ∈ {compute, data, head size, encoder size}, following Kaplan et al. "Scaling the encoder size yields the strongest gains." — [README](https://github.com/Contrastive-LM/CLM)
- Iso-FLOP curves are parabolic in log head size. The optimal head size scales as N* ∝ D^1.02, at about 310 tokens per parameter. — [README](https://github.com/Contrastive-LM/CLM)

**Results (all self-reported)**
- Zero-shot, CLM is "on par with Jev" on T-Rex (Chrome dino), BFCL v4 tool calling, WikiRacing and Super Mario, with "up to 9× lower latency". The speedups are largest with large candidate sets (WikiRacing) or reused actions (T-Rex). The README gives the numbers only as figures. — [README](https://github.com/Contrastive-LM/CLM)
- As a fine-tuned best-of-N verifier, CLM scores 81.6% on 38 held-out DeepSWE tasks (candidates from Opus 5) and 87.6% on 30 held-out Terminal-Bench 2.1 tasks (candidates from Fable 5). It is 4.1–5.7x faster than Jev, and "Jev fails to serve as a verifier … scoring below pass@1". — [README](https://github.com/Contrastive-LM/CLM)
- With about 1k candidates, caching makes CLM about 13x faster than Jev. — [HF card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
- On revisited states, server-side p50 drops from 1.7 ms to 0.6 ms. — [README via fetch](https://github.com/Contrastive-LM/CLM)

**Compute**
- Training compute and GPU-hours are not stated anywhere I could read. Serving uses one RTX 4090; latency evals use an H100; fine-tuning uses batch 512. — [README](https://github.com/Contrastive-LM/CLM); [HF card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)

**What the authors say is the conceptual shift**
- CLMs are "a new class of System One model". States and actions are "disaggregated", so their embeddings are cached and reused independently, "which makes training and serving cheap and blazing fast". Every decision task (typed question, verification, routing) reduces to one primitive: score a candidate against a state. — [README](https://github.com/Contrastive-LM/CLM)
- The roadmap lists "Vision and multimodal support: images, video and other modalities for robotics and computer-use tasks". The model card announced a multimodal CLM-35B for "early October". — [README Roadmap](https://github.com/Contrastive-LM/CLM); [HF card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)

### Inferences
- In effect, CLM is DPR/CLIP retrieval applied to agent decisions. The contribution is the framing (decisions as retrieval), the LLM-style staged data recipe and its scaling laws, and the serving economics. The loss is not new.
- **The InfoNCE optimal critic is f(s,a) ≈ log p(a|s)/p(a) + c.** This follows from the InfoNCE theory in [CPC](https://arxiv.org/abs/1807.03748) and [Poole et al. 2019](https://arxiv.org/abs/1905.06922), which I did not re-fetch this session. So CLM ranks candidates by pointwise mutual information with the state, not by likelihood. That pushes against "popular" actions unless a prior correction is applied. I did not see the authors discuss this.
- **"System One" sets CLM against "System Two" generative or reasoning models.** The intended use is a CLM that selects among proposals from a slower generator: a generate-then-verify split. This is consistent with the authors' earlier papers: Large Language Monkeys (Ré), RoboMonkey (Kwok, Pavone, Mirhoseini) and Weaver (Saad-Falcon); see Q2.

### Gaps
- I could not access the Notion blog content (empty JS page). That leaves the scaling exponents, figure values, temperature τ, projection dimension and pre-training compute unknown.
- The zero-shot success rates are only in figures (assets/zero-shot.png). I saw no numeric table.
- Jev's size and architecture are unknown, so "on par with Jev" cannot be read as a comparison to any open model.
- I found no confirmation that CLM-35B had been released by 2026-10-13. [search result](https://pro.edgex.exchange/en-US/news/article/stanford-nvidia-clm-8b-9x-faster-ai-agents) only repeats the plan.

---

## Q2. Which old ideas does CLM revive, what is genuinely new, and how does it differ from CPC / NCE / EBT / JEPA / contrastive decoding?

### Takeaway
The objective is a decades-old idea. NCE dates to 2010, CPC/InfoNCE to 2018, CLIP/DPR to 2020–21, and implicit-BC EBMs to 2021. It is closest to CLIP or DPR bi-encoders and to Implicit BC and contrastive RL on the robotics side. Three things are new. The *object* being embedded is a (state, decision) pair from agent trajectories. The bi-encoder is used *as the policy/verifier itself* at LLM scale, with an LLM-style pre-, mid- and post-training pipeline and scaling laws. And state/action disaggregation is exploited for caching and latency. This is the "not new but impressive thinking" reading: it is a systems and framing move, not a new loss.

### Cited Findings
- **NCE** (Gutmann & Hyvärinen, AISTATS 2010) learns unnormalized densities by discriminating data from noise. It is the ancestor of all "pull positive / push negatives" losses. — [PMLR](https://proceedings.mlr.press/v9/gutmann10a.html) (bibliographic record, not re-fetched)
- **CPC / InfoNCE** (van den Oord et al. 2018) uses InfoNCE to *learn representations* by predicting future latents. The model is a representation learner and is not used as the decision-maker. CLM reuses the loss but makes the critic itself the product. — [arXiv 1807.03748](https://arxiv.org/abs/1807.03748)
- **CLIP / DPR** use the same symmetric InfoNCE bi-encoder. CLIP does image↔text and gets zero-shot classification by scoring prompts, which is architecturally what CLM's `Choice` is. DPR does question↔passage. — [CLIP 2103.00020](https://arxiv.org/abs/2103.00020); [DPR 2004.04906](https://arxiv.org/abs/2004.04906) (not re-fetched)
- **Energy-Based Transformers** (Gladstone et al., Jul 2025) also "learn to explicitly verify the compatibility between inputs and candidate-predictions". The difference is that EBTs *produce* predictions by gradient-descent energy minimization. That is System-2 thinking, with test-time optimization over a continuous candidate. CLM only ranks a discrete given set in one pass, which is System 1. — [arXiv 2507.02092](https://arxiv.org/abs/2507.02092)
- **JEPA / V-JEPA 2** (Assran et al., Jun 2025) predicts in embedding space by *regression* with no negatives. It is pre-trained on more than 1M hours of video, and V-JEPA 2-AC plans robot actions. JEPA is non-contrastive and its output is a predicted embedding. CLM is contrastive and its output is a score over candidates. — [arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
- **Contrastive decoding** (Li et al. 2022) is a *decoding-time* trick: it maximizes the log-probability difference between an expert and an amateur LM. It involves no contrastive training. Only the name is shared, so do not conflate it with CLM. — [arXiv 2210.15097](https://arxiv.org/abs/2210.15097)
- **Implicit BC** (Florence et al., CoRL 2021) trains EBM policies with sampled counter-examples (an InfoNCE-like loss) and does inference by argmin over sampled or optimized actions. It beat explicit MSE/MDN BC, especially for multimodal and discontinuous actions. This is the closest robotics ancestor of "policy = contrastive scorer over candidates". — [arXiv 2109.00137](https://arxiv.org/abs/2109.00137)
- **Contrastive RL** (Eysenbach et al., NeurIPS 2022) applies contrastive learning on action-labelled trajectories so that "the (inner product of) learned representations exactly corresponds to a goal-conditioned value function". — [arXiv 2206.07568](https://arxiv.org/abs/2206.07568)
- **1000-layer contrastive RL** (Wang, …, Eysenbach, 2025) scaled CRL depth to 1024 layers and got a 2x–50x improvement on goal-reaching. This shows contrastive critics scale. — [arXiv 2503.14858](https://arxiv.org/abs/2503.14858)
- **The authors' own lineage is generate-then-verify scaling.**
  - Large Language Monkeys (Ré) found coverage scales log-linearly with samples, from 15.9% to 56% on SWE-bench Lite with 250 samples. — [arXiv 2407.21787](https://arxiv.org/abs/2407.21787)
  - Weaver (Saad-Falcon, Ré, …) works on shrinking the generation–verification gap. — [arXiv 2506.18203](https://arxiv.org/abs/2506.18203)
  - RoboMonkey (Kwok, …, Mirhoseini, Pavone, CoRL 2025) samples VLA actions, perturbs them, and selects with a *VLM-based* verifier. It reports about +25 points out-of-distribution and about +8–9 points in-distribution, with 16 candidates in about 650 ms (about 1.5 Hz). — [arXiv 2506.17811](https://arxiv.org/abs/2506.17811); [PMLR v305](https://proceedings.mlr.press/v305/kwok25a.html)

### Inferences
- **What is genuinely new**, in my assessment:
  1. The verifier is a *bi-encoder* rather than a cross-encoder LLM judge or reward model. That makes actions cacheable and makes cost O(1 encoder pass + N dot products) instead of O(N LLM passes). It is the CLIP move applied to verification.
  2. An LLM-style curriculum: broad pre-training, then hard-negative mid-training, then domain post-training with replay, with ablations showing the ordering matters (+7 points).
  3. Scaling laws specifically for *InfoNCE verification loss*, including the head-size/data optimum.
  4. Framing all typed decisions as retrieval over a candidate set.
- **The likely root of the user's "not new but impressive" reaction:** in vision this exact pattern already succeeded (CLIP zero-shot classification = state↔label retrieval), and robotics tried it (Implicit BC, CRL). CLM's insight is that once a strong frozen LLM supplies the representation, the expensive part of a decision is *autoregressive generation*, and most decisions are really selections from a set. So you can drop generation for those decisions and recover CLIP-like economics.
- **A probable bridge to robotics:** RoboMonkey's 650 ms verifier bottleneck reads as the problem CLM solves. Since the CLM roadmap explicitly names robotics and video, **the same group is likely to publish a "contrastive VLA verifier"**. Scoop risk for H1 below is high.

### Gaps
- I could not read the authors' own related-work discussion (blog inaccessible), so I cannot say which of these priors they acknowledge.
- No formal comparison exists between CLM and a cross-encoder reward model trained on the same data. Whether the bi-encoder costs accuracy is untested in public material.

---

## Q3. Critiques, limitations, follow-ups

### Takeaway
The criticisms are mostly about evidence and scope. The work is a blog plus code with no paper or peer review, and the baseline is a closed commercial model. The headline SOTA comes from fine-tuned heads on tiny held-out subsets (38 and 30 tasks). CLM cannot propose actions outside the candidate set, and its probabilities are relative to that set and uncalibrated. Inference still needs the full 8B encoder. I found no independent replication and no arXiv follow-ups as of 2026-10-13.

### Cited Findings
- On DeepSWE, the per-head card shows pass@1 at 28/38, CLM selection at 31/38 and the oracle at 34/38. That is a gain of 3 tasks on a small sample. — [wavect.io write-up](https://wavect.io/zh/blog/clm-8b-self-hosting-action-cache-verifier.md)
- The agentic SOTA numbers come from *fine-tuned* heads, not the zero-shot checkpoint. — [HF card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B); [wavect.io](https://wavect.io/zh/blog/clm-8b-self-hosting-action-cache-verifier.md)
- CLM "cannot generate; it only scores the candidates given, and probabilities are relative to that set". The softmax assigns mass even when every option is wrong, a single candidate gets probability 1.0, and "confidence" is not calibrated accuracy. — [HF card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B); [wavect.io](https://wavect.io/zh/blog/clm-8b-self-hosting-action-cache-verifier.md)
- The cost of generating candidates is excluded from CLM's latency figures. — [wavect.io](https://wavect.io/zh/blog/clm-8b-self-hosting-action-cache-verifier.md)
- The 20M-parameter heads still need a full Qwen3-8B forward pass per fresh text. The head only works with the exact encoder and pooling it was trained against. — [README](https://github.com/Contrastive-LM/CLM); [wavect.io](https://wavect.io/zh/blog/clm-8b-self-hosting-action-cache-verifier.md)
- The T-Rex survival result measures a combined system: a physics planner labels actions safe or unsafe in the prompt, and a "shield" overrides unsafe answers. Only the agreement and intervention rows measure the model itself. — [t_rex README](https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/README.md)
- One independent write-up says Jev keeps an accuracy edge in some of its own tests, while CLM wins on speed. — [diariobitcoin](https://www.diariobitcoin.com/tecnologia/clm-8b-puntua-acciones-de-agentes-hasta-nueve-veces-mas-rapido-que-jev/)
- BenchLM finds no shared benchmark between CLM-8B and the models it is compared with, so it gives no quality ranking. — [benchlm.ai](https://benchlm.ai/compare/claude-opus-5-vs-clm-8b)
- The authors acknowledge "generalization is limited at this scale" and plan CLM-35B. — [HF card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
- Follow-ups: a mirror at [sekkit/CLM-v0.1-8B](https://huggingface.co/sekkit/CLM-v0.1-8B), and the repo ships an "autofinetune" agent loop for improving heads ([FINETUNING.md](https://github.com/Contrastive-LM/CLM/blob/main/docs/FINETUNING.md)). I found no follow-up papers.

### Inferences
- **Further critiques, my own analysis:**
  1. **Bi-encoder expressivity ceiling.** IR has long found that cross-encoders and late-interaction models such as ColBERT ([2004.12832](https://arxiv.org/abs/2004.12832), not re-fetched) beat single-vector bi-encoders on fine-grained distinctions. Long-horizon verification, like judging a whole SWE patch, compresses everything into one last-token vector.
  2. **PMI ranking bias** (see Q1).
  3. **Candidate-set dependence.** Success is bounded by the proposer's coverage (the oracle is 34/38), so "System One" quality depends on a System-Two generator.
  4. **No robustness study** under adversarial or near-duplicate candidates beyond the hard-negative top-1 numbers.
- For continuous control, the "cannot generate" limitation is the key obstacle. Actions are continuous, so a candidate set has to come from a sampler (as in RoboMonkey), a codebook, or retrieval.

### Gaps
- No Hacker News, Reddit or X critique threads turned up in my searches.
- No independent re-evaluation of the zero-shot claims is available.

---

## Q4. Existing vision/robotics analogues and the remaining gap

### Takeaway
The pieces already exist in vision and robotics: contrastive critics (CRL), EBM/implicit policies, contrastive *representation* regularizers for VLAs, contrastive latent-action alignment, InfoMax world models and JEPA latent planners. Nobody has yet shipped the CLM *package*. That package is a frozen VLM/video foundation backbone, a small bi-encoder head pre-, mid- and post-trained with hard negatives on (observation, action-chunk) pairs, used as a cached, real-time action selector or verifier, with scaling laws. Meanwhile, 2026 papers document the exact failure modes such a selector would target: latent world models rank actions poorly beyond short horizons, and min-cost selection degrades as the proposal pool grows.

### Cited Findings
- **RS-CL** (Oct 2025) adds a robot-state-aware contrastive loss as a *regularizer* next to the usual action-prediction objective in VLAs. Contrastive is auxiliary there, not the policy. — [arXiv 2510.01711](https://arxiv.org/abs/2510.01711)
- **CLAP** (Jan 2026) contrastively aligns human-video transitions to a robot action-token vocabulary (Act-VAE). It then deploys an autoregressive VLA plus a Rectified-Flow action head. Contrastive is used for pseudo-labelling, not for selecting actions. — [arXiv 2601.04061](https://arxiv.org/abs/2601.04061)
- **Quantile Head** (Sep 2026) observes that "standard flow-matching samplers require costly iterative sampling" and point regression gives only point estimates. This motivates cheaper, distribution-aware heads. — [arXiv 2609.34061](https://arxiv.org/abs/2609.34061)
- **Contrastive World Models** (Bonnie Li, Aug 2026) replaces Dreamer's pixel reconstruction with a Deep-InfoMax bound between state-action sequences and future patch features. It is "small-scale" and wins mainly under distractors. — [arXiv 2609.22175](https://arxiv.org/abs/2609.22175)
- **The Planning Limits of Latent World Models** (Sep 2026) used frozen V-JEPA 2, V-JEPA 2.1, VideoMAEv2, VideoPrism and DINOv2 predictors with 5-step rollouts. These rank actions reliably only for targets 5–10 steps ahead, while task goals are 16–53 steps away. Neither an 81x larger predictor nor longer-rollout training extends this range. — [arXiv 2609.39235](https://arxiv.org/abs/2609.39235)
- **ASAR, "proposal overgeneration"** (Jul 2026): growing the proposal pool from 72 to 288 *reduces* the feasibility of min-latent-cost selection, from .375 to .062 and from .344 to .031. This happens even though larger pools contain feasible sequences. — [arXiv 2607.23602](https://arxiv.org/abs/2607.23602)
- **IMWM** (Jun 2026) shows that even with a *perfect* world model, finite-budget planners fail. It adds an "intuition model" trained from demos, using a CPC/InfoNCE-style contrastive objective, to score promising actions. This is a CLM-like System One in miniature, on small pixel tasks (Two-Room, Reacher, Push-T, OGBench). — [arXiv 2606.01626](https://arxiv.org/abs/2606.01626)
- **LeWorldModel** (Mar 2026) is an end-to-end JEPA with about 15M parameters, trainable "on a single GPU in a few hours". It plans 48x faster than foundation-model world models and detects physically implausible events via surprise. — [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)
- **VLA-JEPA** (Feb 2026) and **JEPA-WAM** (Aug 2026) add JEPA latent prediction to VLAs; both are non-contrastive. JEPA-WAM reports 79.2% on LIBERO-Plus. — [arXiv 2602.10098](https://arxiv.org/abs/2602.10098); [arXiv 2608.09381](https://arxiv.org/abs/2608.09381)
- **RoboMonkey**'s verifier is a fine-tuned VLM, i.e. a cross-encoder scoring each candidate. It takes about 650 ms for 16 candidates. — [arXiv 2506.17811](https://arxiv.org/abs/2506.17811)

### Inferences
- **The gap**, as I read it:
  1. VLA verifiers are cross-encoders, which are slow and need few candidates.
  2. Contrastive objectives in VLAs are auxiliary.
  3. Contrastive critics (CRL, IMWM) have not been built on frozen *foundation* video/VLM features at scale, with hard-negative curricula and scaling laws.
  4. Latent world-model planning has a *selection* problem (overgeneration, short ranking horizon) that a discriminatively trained critic with hard negatives is designed to address.

### Gaps
- I did not verify whether any Sep–Oct 2026 arXiv paper already does "CLIP-style bi-encoder action verifier for VLAs". One extended search found none, but the coverage is not exhaustive.
- I did not check physical-plausibility benchmarks (IntPhys 2, Physics-IQ) for 2026 SOTA numbers.

---

## Q5. Transfer hypotheses for vision / robotics / video, scoped to one H200 before CVPR 2027 (deadline 2026-11-16, about 34 days away)

### Takeaway
I propose five hypotheses. All GPU-day figures are **my own estimates**: they assume cached frozen-backbone features and simulation benchmarks, and none comes from a source. With about 34 calendar days and one H200, about 25–30 usable GPU-days remain, which supports one main hypothesis plus one analysis. **The best ratio of novelty to risk is H1 combined with H5** (a contrastive action verifier plus scaling and curriculum analysis). H3 is the most "CVPR-vision" option. H1 has high scoop risk from the CLM/RoboMonkey authors, so speed matters.

### Cited Findings (grounding for the hypotheses)
- RoboMonkey verifier latency: about 650 ms for 16 candidates; gains of +25 OOD and +8–9 ID. — [arXiv 2506.17811](https://arxiv.org/abs/2506.17811)
- CLM's caching wins are largest with large or reused candidate sets: 9x zero-shot and about 13x at 1k candidates. — [README](https://github.com/Contrastive-LM/CLM); [HF card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
- In CLM, hard negatives as mid-training beat hard negatives from the start: 69.2 vs 62.4 top-1. — [README](https://github.com/Contrastive-LM/CLM)
- In CLM, the head optimum is N* ∝ D^1.02 at about 310 tokens/parameter, and encoder size matters most. — [README](https://github.com/Contrastive-LM/CLM)
- Latent world models rank actions only 5–10 steps ahead. — [2609.39235](https://arxiv.org/abs/2609.39235)
- Proposal overgeneration degrades min-cost selection. — [2607.23602](https://arxiv.org/abs/2607.23602)
- Implicit BC beats explicit BC on multimodal actions. — [2109.00137](https://arxiv.org/abs/2109.00137)
- Flow heads have iterative cost. — [2609.34061](https://arxiv.org/abs/2609.34061)
- CRL inner product equals a goal-conditioned value, and CRL scales with depth. — [2206.07568](https://arxiv.org/abs/2206.07568); [2503.14858](https://arxiv.org/abs/2503.14858)
- LeWM trains on one GPU in hours. — [2603.19312](https://arxiv.org/abs/2603.19312)
- V-JEPA 2 is a frozen video backbone. — [2506.09985](https://arxiv.org/abs/2506.09985)

### Inferences (the hypotheses)

**H1. A bi-encoder contrastive action verifier for VLAs, used for test-time selection ("CLM for robots").**
- *Idea:* a state tower (frozen VLM or SigLIP/V-JEPA features plus instruction plus proprio, with a small head) and an action tower (a small transformer over the action chunk), trained with symmetric InfoNCE. Hard negatives come from Gaussian-perturbed expert chunks and from VLA samples that diverge from the expert. Use it to select among N VLA samples.
- *Gap:* RoboMonkey uses a cross-encoder VLM verifier (650 ms for 16 candidates). Contrastive is only auxiliary in RS-CL and CLAP.
- *Falsifiable prediction:* at equal wall-clock verifier budget (≤100 ms per step), the bi-encoder with N=128 beats the cross-encoder with N=16 on LIBERO-Plus or SimplerEnv success. Alternatively, at equal N=16 it matches within 2 points at ≥5x lower latency. **It is falsified if** the gain over the no-verifier baseline is within seed noise (3 seeds), or if success saturates by N=16 (meaning the proposer, not the verifier, is the bottleneck).
- *Minimal experiment:* an open VLA (OpenVLA-OFT or π0-family weights) on LIBERO and LIBERO-Plus. Cache features for the demonstrations, train heads, sweep N ∈ {1, 4, 16, 64, 256}, and report the oracle@N upper bound as CLM did (34/38).
- *GPU-days (my estimate):* feature caching 0.5, negative generation by VLA sampling 1–2, head training 0.5, rollouts 3–4 seeds × N-sweep about 3. **Total about 5–7.**

**H2. A contrastive action head instead of a diffusion/flow head: Implicit BC revisited with a codebook plus caching.**
- *Idea:* replace the flow head with a dot product between the state embedding and a *cached codebook* of action-chunk embeddings. The codebook is built by k-means over training chunks (K = 4k–64k), optionally followed by one local refinement step.
- *Gap:* Implicit BC was unstable in high dimensions with sampled negatives (2021, pre-foundation-backbone). Flow heads cost iterative sampling (Quantile Head, 2026). CLAP uses contrastive only for labelling.
- *Falsifiable prediction:* on LIBERO, the contrastive head comes within 3 points of the flow head, with a single forward pass and better coverage of multimodal tasks (measurable as mode coverage on Push-T). **It is falsified if** the gap exceeds 5 points at K=64k, which would mean codebook quantization error dominates.
- *Minimal experiment:* a small VLA (about 0.5–3B, frozen VLM) with the head swapped, on LIBERO plus Push-T.
- *GPU-days (my estimate):* about 4–6.
- *Risk:* this is the "cannot generate" limitation from Q3. Continuous precision may require refinement, which erodes the novelty.

**H3. A contrastive (CRL-style) evaluator on frozen V-JEPA 2 features for latent planning.**
- *Idea:* train an InfoNCE critic between φ(o_t, a_{t:t+k}) and ψ(o_{t+Δ}) for Δ sampled geometrically, with hard negatives from the same episode. Use it as the planning cost or as a re-ranker over CEM proposals, instead of, or alongside, L1 distance between predicted latents.
- *Gap:* the Planning Limits paper reports a 5–10-step ranking horizon that model scaling does not fix. ASAR reports that selection *degrades* with pool size. Contrastive World Models and IMWM are small-scale and do not use foundation features.
- *Falsifiable prediction:* (a) the reliable ranking horizon extends to ≥2x the predictor's rollout length on Meta-World and BridgeData V2 offline ranking; (b) selection feasibility does *not* fall as the pool grows from 72 to 288, unlike min-latent-cost. **It is falsified if** the horizon gain is under 1.5x or the overgeneration curve keeps the same slope.
- *Minimal experiment:* reuse the Planning Limits protocol (5 frozen backbones), or just V-JEPA 2 ViT-L.
- *GPU-days (my estimate):* about 3–5.
- *Why it is my top CVPR pick:* it is vision-centric, cheap, directly answers two 2026 negative results, and gives a clean analysis story.

**H4. Contrastive next-clip selection as a video "System One" plausibility verifier.**
- *Idea:* a bi-encoder (context clip ↔ candidate continuation) trained with structured hard negatives: time-reversed, frame-shuffled, object-teleported, or generated continuations from a video model. Use it to re-rank best-of-N outputs of a video generator or world model, and as a physical-plausibility scorer.
- *Gap:* JEPA surprise (LeWM, V-JEPA) is non-contrastive and is not trained on hard negatives. Video generators lack cheap verifiers.
- *Falsifiable prediction:* at a matched frozen backbone, it beats V-JEPA-2 surprise on a possible-vs-impossible discrimination benchmark, and best-of-8 re-ranking improves a physics-adherence metric over random choice. **It is falsified if** the score just tracks low-level artifacts, which I would detect by checking that it fails on artifact-matched negatives.
- *GPU-days (my estimate):* about 5–8. Generating negatives with a video model is the expensive part, so I rate this one riskier.

**H5. Scaling laws and curriculum for contrastive action verification in robotics.** This is an analysis add-on to H1 or H3.
- *Idea:* test whether CLM's findings transfer to vision-action data.
  - (i) a power-law fit of test InfoNCE against head size, data and backbone size (e.g. SigLIP-B/L/So400m or V-JEPA 2 L/H/g);
  - (ii) whether N* ∝ D^~1;
  - (iii) whether hard negatives work better as mid-training than from the start;
  - (iv) whether lower test InfoNCE predicts downstream success (the bridge CLM claims but does not show for robots).
- *Falsifiable prediction:* the curriculum gives ≥3 points top-1 on held-out hard negatives, and the InfoNCE loss correlates with rollout success at Spearman ρ > 0.7 across checkpoints. **It is falsified if** ρ < 0.4, meaning the loss is not a usable proxy.
- *GPU-days (my estimate):* about 2–4 with cached features, because the heads are tiny.

**Planning note.**
- Scoop risk: CLM's roadmap names robotics and video, and the authors wrote RoboMonkey, so H1 is the most likely to be scooped. H3, H4 and H5 are less exposed.
- Positioning: frame the work as "System One selection for embodied models", citing CLM, RoboMonkey, Implicit BC and CRL. Explicitly distinguish it from JEPA (regression), EBT (test-time optimization) and contrastive decoding (decode-time trick).

### Gaps
- I did not check exact LIBERO-Plus, SimplerEnv or Meta-World SOTA numbers for the latest VLAs, or the availability and licensing of π0-family and OpenVLA-OFT weights in Oct 2026.
- The GPU-day estimates are unvalidated guesses. Simulator rollout throughput on a single H200 node (CPU-bound physics) could dominate.
- I did not verify whether the CLM authors or others released multimodal CLM-35B or a robot verifier after 2026-10-05.
