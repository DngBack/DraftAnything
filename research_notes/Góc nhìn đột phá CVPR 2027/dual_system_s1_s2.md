# Dual-system (System 1 / System 2) architectures in robotics and embodied AI: state of the art, open problems, and falsifiable hypotheses for a vision researcher

Scope note: written 2026-10-13. About 25 search and fetch calls. Several 2026 items were read only at abstract level, and this is flagged where it matters. All GPU-day numbers are **my own estimates** and have not been measured. I skimmed `reports/Hướng nghiên cứu thay thế CVPR 2027.md` first. Its VLA ideas (#5, "who is driving the arm?" modality attribution; E2 viewpoint-as-coordinate-problem; E5 "VLAs know they are failing") are **not** repeated here. Where a hypothesis below touches them, the overlap is stated explicitly.

## Q1. State of the art 2024–2026: what S1 and S2 are, their interface, their frequencies, and when S2 is invoked

### Takeaway
The field has converged on "big VLM (S2, roughly 2–10 Hz) conditions a small fast action module (S1, roughly 30–200 Hz)". The S2→S1 interface splits three ways:
- **(a) A single or few continuous latent tokens:** Helix, LCB, HiRT, RoboDual, OpenHelix, DualVLN.
- **(b) Shared backbone, where S1 attends to the VLM's layers/KV:** π0/π0.5 with knowledge insulation, GR00T N1.x, FiS-VLA.
- **(c) Natural-language subtasks:** Hi Robot, DexVLA, Gemini Robotics-ER → GR 1.5/2.

Almost all systems still invoke S2 on a **fixed clock**. Learned or uncertainty-triggered invocation (OneTwoVLA, TUD, "When to Plan") is the newest and most crowded sub-area in 2026.

### Cited Findings
**Industrial systems**
- **Figure Helix (Feb 2025):**
  - S2 is a 7B open-weight internet-pretrained VLM at 7–9 Hz. It takes monocular images, robot state projected into the VL embedding space, and language.
  - S1 is an 80M cross-attention encoder-decoder transformer with its **own** multi-scale conv vision backbone, at 200 Hz, with 35-DoF upper-body control.
  - Interface: S2 compresses task semantics into **a single continuous latent vector**, which is projected into S1's token space and concatenated with S1's visual features.
  - Trained end-to-end with one regression loss. Gradients flow from S1 into S2 through the latent.
  - **A temporal offset between S1 and S2 inputs is injected during training**, calibrated to the deployed latency gap.
  - S1 and S2 run on separate embedded GPUs, with S2 asynchronous. — [Figure, Helix](https://www.figure.ai/news/helix)
- **Helix 02 (Jan 2026)** extends Helix to the whole body. I found no architecture details. — [Wikipedia: Figure AI](https://en.wikipedia.org/wiki/Figure_AI)
- **NVIDIA GR00T N1** (VLM "System 2" + DiT "System 1"):
  - Frequencies: S2 about 10 Hz on an L40 and S1 120 Hz. These come from a secondary summary, not the paper. — [themoonlight review](https://www.themoonlight.io/tw/review/gr00t-n1-an-open-foundation-model-for-generalist-humanoid-robots)
  - N1.5: Eagle 2.5 VLM, improved MLP connector, co-trained flow matching + world-modeling objective, 16-layer DiT. — [LearnOpenCV](https://learnopencv.com/gr00t-n1_5-explained/)
  - N1.6: 32-layer DiT and a Cosmos-Reason-2B-based VLM. A single secondary source says the N1.5 adapter was removed and the top 4 VLM layers are unfrozen during pretraining. — [HF model card](https://huggingface.co/nvidia/GR00T-N1.6-3B), [CSDN](https://gitcode.csdn.net/69f1ffa10a2f6a37c5a6d97e.html)
  - N1.7 Early Access (Apr 2026), per a secondary roundup: 3B, Cosmos-Reason2-2B backbone, 32-layer DiT, described as an "Action Cascade" dual system. — [MarkTechPost](https://www.marktechpost.com/2026/04/28/top-10-physical-ai-models-powering-real-world-robots-in-2026/)
- **Physical Intelligence**
  - **π0.5 / Knowledge Insulation (KI, arXiv 2505.23705, NeurIPS 2025 poster):**
    - The VLM backbone is trained on FAST-tokenized discrete actions.
    - A flow-matching action expert produces continuous actions and attends to VLM layers, but its **gradients are stopped** from flowing into the backbone.
    - Motivation: action-expert gradients "lead to much slower learning" and make the backbone "lose some of the knowledge acquired during web-scale pretraining".
    - — [PI blog](https://pi.website/research/knowledge_insulation), [NeurIPS poster](https://neurips.cc/virtual/2025/poster/117079)
    - A community claim that stop-gradient alone recovers about 35 points of language following is **unverified**. — [HF blog](https://mcobzarenco-fontaine-blog.static.hf.space/papers/pi05-knowledge-insulation.html)
  - **Hi Robot (arXiv 2502.19417, ICML 2025 per one listing):**
    - A high-level VLM ("System 2") reads open-ended instructions plus base/wrist images and emits **language subtask commands**.
    - π0 acts as the reactive "System 1", consuming those commands, images and state, and outputs actions plus optional verbal replies.
    - I did not find frequencies. — [PI Hi Robot](https://pi.website/research/hirobot), [arXiv](https://arxiv.org/abs/2502.19417)
- **Google DeepMind Gemini Robotics 1.5 (arXiv 2510.03342):**
  - Gemini Robotics-ER 1.5 is the orchestrator. It decomposes tasks into steps for the VLA and does **success detection to decide when to switch steps**.
  - The GR 1.5 VLA itself "interleaves actions with a multi-level internal reasoning process in natural language" ("think before acting").
  - It adds Motion Transfer across embodiments.
  - — [arXiv 2510.03342](https://arxiv.org/pdf/2510.03342)
- **Gemini Robotics 2 (30 Jul 2026):**
  - A three-model family: ER 2 makes the high-level plan and the VLA turns it into low-level commands.
  - Whole-body humanoid control and On-Device 2 (based on GR 1.5 tech + Gemma).
  - — [SiliconANGLE](https://siliconangle.com/2026/07/30/google-deepmind-debuts-gemini-robotics-2-model-series-humanoid-robots/), [DeepMind On-Device 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/)

**Academic systems**
- **Taxonomy from OpenHelix (arXiv 2505.03912):**

  | Method | S2 | Interface | S2 training | S1 |
  |---|---|---|---|---|
  | LCB | LLaVA-7B | language `<ACT>` token | LoRA | 3D Diffuser Actor (pretrained) |
  | DP-VLA | OpenVLA-7B | vision + language | frozen | transformer |
  | HiRT | InstructBLIP-7B | max-pooled vision + language | LoRA | RT-1-style |
  | RoboDual | OpenVLA-7B | action + language tokens | LoRA | DiT |
  | DexVLA | Qwen2-VL-2B | language | LoRA | ScaledDP |

  OpenHelix **excludes π0 and GR00T N1 from "dual-system"** because their S1 has no real-time perception input of its own. — [OpenHelix](https://arxiv.org/html/2505.03912v1)
- **RoboDual** pairs an OpenVLA generalist with a DiT specialist. The real-world control frequency is about 15 Hz, and another paper benchmarks it at 14.79 Hz. — [RoboDual](https://opendrivelab.com/RoboDual/), [arXiv 2410.08001](https://arxiv.org/html/2410.08001v1)
- **HiRT** runs the VLM at low frequency and a lightweight visual policy at high frequency. — [VLA survey 2508.13073](https://arxiv.org/html/2508.13073v1)
- **Fast-in-Slow, FiS-VLA (NeurIPS 2025):**
  - S1 = the **final transformer blocks of the VLM itself**, repurposed, with asynchronous rates.
  - 117.7 Hz with action chunk 8.
  - +8% sim and +11% real average success over prior SOTA.
  - Dual-aware co-training keeps S1's latent conditioning stable.
  - — [arXiv 2506.01953](https://arxiv.org/html/2506.01953v1), [NeurIPS poster](https://neurips.cc/virtual/2025/poster/119931)
- **OneTwoVLA:** a single model learns a **decision token** to toggle between reasoning and acting, entering reasoning e.g. on detected errors. It needs manually annotated phase boundaries. — [arXiv 2505.11917](https://arxiv.org/pdf/2505.11917); critique in [TUD 2610.05025](https://arxiv.org/html/2610.05025)
- **DualVLN, "Ground Slow, Move Fast" (ICLR 2026, navigation):**
  - S2 is a 7B VLM at 2 Hz emitting a **pixel goal + latent goal**. S1 is a diffusion policy at 30 Hz.
  - The paper notes the latent goal at t is "outdated" at t+k, so S1 must account for distance travelled.
  - — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/14da7aea05debb963b3d8d46449d51a0-Paper-Conference.pdf), [arXiv 2512.08186](https://arxiv.org/html/2512.08186v1)
- **Libra-VLA (ACL 2026):**
  - A Semantic Planner predicts discrete coarse "macro-directional intent" tokens, and an Action Refiner produces continuous actions, run asynchronously.
  - Success is an **inverted-U against the granularity of the coarse/fine split**, peaking when learning difficulty is balanced across the two systems.
  - It criticises GR00T N1's "static latent embeddings" as lagging during async generation.
  - — [arXiv 2604.24921](https://arxiv.org/abs/2604.24921)
- **2026 asynchrony and efficiency work:**
  - An asynchronous fast-slow whole-body VLA notes that prior dual systems still run synchronously, so the control rate is bounded by the VLM. — [arXiv 2512.20188](https://arxiv.org/pdf/2512.20188)
  - **Latent Bridge** predicts S2 feature deltas between VLM calls and notes that flow heads are sensitive to the dense VLM conditioning. — [arXiv 2605.02739](https://arxiv.org/pdf/2605.02739)
  - **CloudEdgeVLA**: trained on randomly delayed cloud features, it keeps 63.8–78.0% LIBERO success under a 40-step delay window. VLASH reaches ≤6.4% and single-path baselines ≤3.0%. — [arXiv 2608.00569](https://arxiv.org/abs/2608.00569)
  - **VLA-Perf** notes that there is no widely adopted open-source diffusion-style dual-system VLA implementation. — [arXiv 2602.18397](https://arxiv.org/pdf/2602.18397)
  - Also: NebulaVLA, a dual-frequency model with a "guide action". — [arXiv 2608.16503](https://arxiv.org/pdf/2608.16503)
- **Pure vision / streaming video analogues:**
  - **Dispider (CVPR 2025 per an awesome-list)** disentangles a lightweight perception module (continuous monitoring), a decision module (when to trigger), and an asynchronous reaction LLM. — [arXiv 2501.03218](https://arxiv.org/html/2501.03218v1), [Awesome-Streaming list](https://github.com/sotayang/Awesome-Streaming-Video-Understanding)
  - **StreamBridge** uses a compact external MLLM as an "activation model" that decides when the main Video-LLM responds. — [arXiv 2505.05467](https://arxiv.org/pdf/2505.05467)
  - These papers do not use the "System 1/2" vocabulary.

### Inferences
- Three different things share the name "dual system":
  - **(i) Rate split.** The slow and fast parts run at different Hz: Helix, GR00T, FiS.
  - **(ii) Capacity split.** A big model and a small model: RoboDual, HiRT.
  - **(iii) Abstraction split.** Language plan vs motor command: Hi Robot, Gemini ER→VLA.
  
  Papers routinely conflate them. A paper that varies them independently would clarify the field.
- The **S1-has-its-own-eyes** distinction (Helix, HiRT, RoboDual, OpenHelix: yes; π0/π0.5, GR00T: no, since vision enters only through the VLM) should matter a lot for staleness and asynchrony. I found no paper that compares the two families under the same staleness protocol.
- Helix's "train with the deployment latency offset" and CloudEdgeVLA's "train with random delays" are the same idea. Robustness to S2 staleness has to be trained in. This makes **off-the-shelf staleness tolerance** a meaningful diagnostic of what S2 actually carries.

### Gaps
- No confirmed **GR00T N2** release was found. N1.7 EA (Apr 2026) is the latest I could see.
- Helix 02 architecture: not found.
- Hi Robot and π0.5 S2 frequencies: not found.
- GR00T frequencies come from secondary sources only.
- DexVLA and LCB details come only via the OpenHelix table.
- Gemini Robotics 2 interface specifics (language vs latent) come from press coverage only.

## Q2. Open conceptual questions in the S1/S2 split

### Takeaway
The open questions the user listed (interface, who triggers S2, does S2 help at matched compute, memory split, latency, transfer to vision) have shifted in status:
- **"When to think" is now crowded.** At least five 2026 papers cover it.
- **What S2 actually transmits and whether it is needed** has only one real empirical study (OpenHelix on CALVIN). That study found the S2 latent is largely **insensitive to visual changes and to a 10→60-step delay**.
- Nobody has measured **ground-truth per-timestep value of an S2 call**.
- Nobody has compared S1-with-vision vs S1-without-vision families under the same staleness protocol.

### Cited Findings
- **What the S2 latent carries (OpenHelix, CALVIN ABC→D):**
  - Probing the `<ACT>` latent shows it mostly encodes instruction semantics (target object, spatial relation, action words) and is "largely insensitive to changes in visual information".
  - Async delay of 10 vs 60 steps gave 3.45 vs 3.44 average length.
  - **Policy-only 3.27 vs prompt-tuned MLLM + policy 3.30** in the final 1000-sequence table. Only the auxiliary action loss lifts it to 3.45.
  - On enriched-language CALVIN-E, policy-only 1.42 vs full 2.26.
  - Fully fine-tuning the MLLM without CLIP loss *hurts* (3.13). Without projector pre-alignment every configuration collapsed to 0%.
  - The authors call it an initial technical report with no real robot.
  - — [OpenHelix](https://arxiv.org/html/2505.03912v1)
- **Interface form is fragile.**
  - TOWN-VLA finds that appending retrieved or "slow-path" text to a frozen π0.5 prompt cut success from **92.47% to 3.00%**. Both meaningful and length-matched meaningless appends failed on all 500 states ("prompt-form collapse").
  - Gating that only lets canonical compact instructions through raised LIBERO-Plus success from 69.5% to 73.1%, and a real PiPER arm from 52.7% to 78.7%.
  - — [arXiv 2608.23224](https://arxiv.org/abs/2608.23224)
- Steerable VLA policies report that with a "grasp the screw" subtask command, the low-level VLA sometimes grabs a distractor. They argue subtask language is too thin an interface. — [arXiv 2602.13193](https://arxiv.org/pdf/2602.13193)
- **When to invoke S2 (crowded):**
  - **TUD (KAIST, 4 Oct 2026):**
    - Re-invokes the PaliGemma generalist only when the dispersion of recent specialist re-predictions exceeds a percentile threshold.
    - Matches π0 on VLA-Arena Safety with 43% fewer generalist calls, and uses 58% fewer on Long-Horizon.
    - At about 17.8 calls/episode: 86.7% vs 83.7% for Diff-DAgger's signal.
    - Real SO-101: 0.833 success with 12.1 calls vs 0.467 with 48.5 calls for the best fixed interval.
    - A generalist call costs about 106 ms and a specialist forward about 13.5 ms.
    - **No random-trigger baseline.** The skip-safety bound assumes a Lipschitz specialist.
    - — [arXiv 2610.05025](https://arxiv.org/html/2610.05025)
  - **"When to Plan" (Jul 2026)** learns a meta-policy choosing reactive vs planning options, conditioned on reactive-policy uncertainty and high-level state. — [arXiv 2607.16421](https://arxiv.org/html/2607.16421v1)
  - **RARRL** treats when-to-reason as an MDP under budgets, on ALFRED. — [arXiv 2603.16673](https://arxiv.org/html/2603.16673v1)
  - Also OneTwoVLA's decision token. — [arXiv 2505.11917](https://arxiv.org/pdf/2505.11917)
- **Systematic hierarchy study.** "What Matters in Orchestrating Robot Policies" (Hu, Shridhar, …, Shah, Tan, Xie; Jun/Sep 2026) unifies Hi-VLA agents in an options framework. It ablates planner/controller choice, switching mechanism, and observation and memory representation. A tuned hierarchy is "substantially stronger" than flat VLA or a naive hierarchy. The abstract gives **no matched-compute control and no numbers**. — [arXiv 2606.10267](https://www.alphaxiv.org/abs/2606.10267)
- **Latency:**
  - Libra-VLA and DualVLN explicitly flag stale S2 latents. — [Libra-VLA](https://arxiv.org/pdf/2604.24921), [DualVLN](https://arxiv.org/html/2512.08186v1)
  - CloudEdgeVLA shows that training with random delays makes an "emergent" split: the slow path keeps stable task-level information and the fast path supplies state-sensitive corrections. — [arXiv 2608.00569](https://arxiv.org/abs/2608.00569)
- **Memory split:** EventVLA describes dual-system memory VLAs as decoupling cognition from control, but suffering "high latency and severe error propagation". — [arXiv 2606.20092](https://arxiv.org/html/2606.20092v2)

### Inferences
- **Interface:** the evidence points one way. OpenHelix's latent is instruction-dominated and vision-insensitive. CloudEdgeVLA's slow path specialises in task-level information. Language is fragile in form (TOWN-VLA). In current benchmarks, **S2 mainly acts as a (soft) task/subtask identifier**. If true, the S2→S1 channel would carry only a few bits on LIBERO/CALVIN. This would also explain why "S2 doesn't help much" on saturated benchmarks: the benchmarks don't require more bits. This is a measurable, vision-agnostic claim, and no source I found has measured it.
- **Metacognition:** every trigger paper evaluates its trigger by end-of-episode success vs call count. None measures the *ground-truth* value of an S2 call at each step. That gap is where an EVC-style framing (Q5) gives a new perspective rather than another trigger.
- **Vision transfer:** streaming Video-LLMs (Dispider, StreamBridge) and online trackers already have the S1/S2 structure. They face the same "when to call the slow model" problem but are not studied under the S1/S2 lens.

### Gaps
- No primary-source numbers for the Hi-VLA systematic study (2606.10267) were obtained.
- I found no paper with a strict **parameter/FLOP-matched single-vs-dual control**.
- I found no paper that varies S2 refresh rate on π0.5 or GR00T N1.5 off the shelf.

## Q3. Which dual-system papers got orals or awards (CVPR/ICCV/CoRL/RSS/NeurIPS 2025–2026)?

### Takeaway
Award-level recognition for dual-system VLAs is thin.
- The clearest case is **π0.5 as a CoRL 2025 Best Paper finalist**.
- Most dual-system papers (FiS-VLA, KI) appeared as **NeurIPS 2025 posters**, DualVLN at ICLR 2026, and Hi Robot at ICML 2025.
- I found no CVPR/ICCV oral or award for a dual-system paper.

### Cited Findings
- CoRL 2025 awards page (per search snippet): Best Student Paper went to "Visual Imitation Enables Contextual Humanoid Control". Finalists included "π0.5: a VLA with Open-World Generalization". — [CoRL 2025 Awards](https://2025.corl.org/program/awards)
- RSS 2025 Outstanding Paper went to FEAST (mealtime assistance). FAST action tokenization (PI) was a finalist. — [RSS Foundation](https://roboticsfoundation.org/awards/best-paper-award/)
- ICML 2025 outstanding papers contained no VLA work. — [ICML 2025 awards](https://icml.cc/virtual/2025/awards_detail)
- FiS-VLA was a NeurIPS 2025 poster. — [NeurIPS](https://neurips.cc/virtual/2025/poster/119931)
- KI was a NeurIPS 2025 poster. — [NeurIPS](https://neurips.cc/virtual/2025/poster/117079)
- DualVLN was in ICLR 2026. — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/14da7aea05debb963b3d8d46449d51a0-Paper-Conference.pdf)
- Libra-VLA was ACL 2026 Main. — [arXiv](https://arxiv.org/abs/2604.24921)
- From the sibling report (not re-verified here): the only confirmed "simple reframing" embodied award at CVPR was Navigation World Models (Best Paper HM, CVPR 2025), which is not a dual-system paper. — [CVPR 2025](https://cvpr.thecvf.com/Conferences/2025/BestPapersDemos)

### Inferences
- Dual-system method papers win at robotics venues (CoRL) when they come from big labs with real-robot scale.
- For CVPR, the realistic route is an **analysis/measurement paper** about what the S2→S1 channel carries, or a vision-native S1/S2 paper (streaming video, tracking). A new dual-system architecture is the wrong route.

### Gaps
- CoRL 2026 (Sept/Nov 2026), RSS 2026, CVPR 2026 and ICCV 2025 oral lists were not checked in detail, so dual-system orals there are **unknown**.
- The π0.5 finalist status comes from a search snippet of the CoRL page and was not fetched directly.

## Q4. Are there analysis papers showing S2 doesn't help, or that S1 ignores S2?

### Takeaway
Yes, there is partial evidence, though not from a dedicated "S2 doesn't help" paper:
- OpenHelix's own final table shows MLLM + policy ≈ policy alone (3.30 vs 3.27) unless an auxiliary loss is added.
- Its latent ignores visual change, and asynchrony up to 60 steps doesn't matter.
- For language/CoT-style S2, better reasoning does **not** translate into better LIBERO-Plus success (TRUST), and reasoning-action consistency is low in driving.
- No paper runs a clean parameter- or compute-matched control.

### Cited Findings
- OpenHelix:
  - Policy-only 3.27 vs prompt-tuned MLLM + policy 3.30 (CALVIN ABC→D avg length). The +aux loss reaches 3.45.
  - The MLLM matters more on enriched language: 1.42 vs 2.26.
  - Removing visual input from the MLLM gives 1.77 vs 3.45, so the MLLM is not purely language.
  - — [OpenHelix](https://arxiv.org/html/2505.03912v1)
- TRUST ("When Reasoning Helps Action", 30 Sep 2026): on DeepThinkVLA, steering raised grasp-state claim correctness from 69.3% to 90.2% and action-choice claims from 68.8% to 85.9%. Yet **closed-loop LIBERO-Plus performance "stayed largely unchanged"**. In driving (Alpamayo 1.5) it cut collisions by 30.4%. — [arXiv 2610.00601](https://arxiv.org/abs/2610.00601)
- Driving VLA faithfulness: Alpamayo-R1-10B had 42.5% reasoning fidelity, 48.3% reasoning-action consistency, and in 37.9% of "stop"-claimed cases it continues. — [arXiv 2605.17268](https://arxiv.org/abs/2605.17268)
- Pinocchio separates performance-helpful reasoning from reasoning faithful to the policy's decision process. — [arXiv 2607.04681](https://arxiv.org/html/2607.04681)
- "Do What You Say" frames a reasoning-action gap between textual plans and the low-level actions' physical outcomes. — [arXiv 2510.16281](https://arxiv.org/html/2510.16281v2)
- An action-models textbook chapter argues ECoT gains are evidence of functional, not faithful, reasoning. — [Action Models book §18.7](https://action-models-book.vercel.app/chapters/18/7/)
- Prompt-form collapse (92.47% → 3.00% when text is appended) shows the S1 language channel is brittle to S2-produced phrasing. — [arXiv 2608.23224](https://arxiv.org/abs/2608.23224)
- For π0/π0.5-style shared backbones, KI's motivation is itself evidence of S1/S2 interference: action gradients erode VLM knowledge. — [PI KI](https://pi.website/research/knowledge_insulation)

### Inferences
- The "S2 helps" evidence is confounded. Dual systems usually add a 2–7B VLM to a small policy, so gains may be a **bigger-model / better-pretraining effect**, not a two-system effect. The flat-baseline comparison in the Hi-VLA systematic study and FiS-VLA's +8% SOTA claim do not separate the two.
- "S1 ignores S2" is best stated quantitatively as the **mutual information between the S2 message and S1's action, conditioned on S1's own observation**. Off-the-shelf staleness/swap interventions can lower-bound this cheaply (see H1/H2).

### Gaps
- "Revisiting Embodied CoT" (ERVLA, arXiv 2606.03784) and "Unmasking the Illusion of Embodied Reasoning in VLA Models" (BeTTER) appeared only in reference lists. Their findings are unknown, and **they must be scoop-checked** before H1/H2/H6.
- No matched-compute dual-vs-single study was found.

## Q5. Cognitive-science critiques of the System 1/2 metaphor that could inspire a better design

### Takeaway
The strongest critiques attack two features robotics copies blindly:
- **Exclusivity:** two separate systems.
- **The switch:** who decides to deliberate, and how.

De Neys (BBS 2022) argues both are foundational problems. Meta-reasoning work frames the switch as **monitoring (confidence/conflict) plus control (rethink)**. The Expected Value of Control (EVC) model frames deliberation as a **cost-benefit allocation**: payoff of control minus effort cost.

Design implications:
- Treat S2 invocation as an *estimated value-of-computation* decision.
- Consider that "S1" may already carry the competing hypotheses whose *conflict* should trigger S2.

### Cited Findings
- De Neys, "Advancing theorizing about fast-and-slow thinking" (BBS, 2022). The critique "focuses on two interconnected foundational issues: the exclusivity and switch feature". The reply to 34 commentaries is "Further advancing fast-and-slow theorizing" (2023). — [Cambridge BBS](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/advancing-theorizing-about-fastandslow-thinking/316D2A018DFCB97676D3B2E8C6A1A0BA), [reply](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/further-advancing-fastandslow-theorizing/57847E353F098DC7325B3F7D6DFB3A1B)
- Ackerman & Morsanyi's commentary: the target article reviews meta-reasoning **monitoring (e.g., confidence) and control (e.g., rethinking)** processes that explain mental-effort regulation. — [Ackerman & Morsanyi PDF](https://ackerman.net.technion.ac.il/files/2023/06/Ackerman_Morsanyi_inpress_commentary.pdf)
- Low, Butterfill & Michael's counter-commentary: mindreading research shows abundant evidence for exclusivity. The debate is live, not settled. — [Butterfill page](https://www.butterfill.com/writing/low2023_view/)
- **EVC:** Shenhav, Botvinick & Cohen (Neuron 2013) frame control allocation as expected payoff minus effort cost, the Expected Value of Control. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7993114)
- Shenhav's 2017 Psychological Inquiry commentary says dual-process theories have come to encompass "an increasingly wide array of increasingly difficult to operationalize dichotomies", with self-control as a domain of "dual-process overreach". — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8528169)
- An Essex study finds people sometimes do not correct reasoning errors because doing so is not EV-maximising, i.e. rational non-deliberation. — [Essex repository](https://repository.essex.ac.uk/34555/1/2023-58265-001.pdf)
- Melnikoff & Bargh, "The mythical number two" (TiCS 2018), and a BBS target article "Hoist by its own petard: the ironic and fatal flaws of dual-process theory" were **seen only in reference lists/search titles**. Their content was not verified. — [Cambridge](https://resolve.cambridge.org/core/journals/behavioral-and-brain-sciences/article/hoist-by-its-own-petard-the-ironic-and-fatal-flaws-of-dualprocess-theory/8DEACBC7DF87E3FC378E7CCC48E6ACDF)

### Inferences (design translations; my own, not from sources)
- **Exclusivity critique → a continuum, not two boxes.** FiS-VLA (S1 = last blocks of S2) and KI (shared backbone) already blur exclusivity. The useful variable is *how often each depth of computation is refreshed*, not "two models". This motivates a "refresh-rate spectrum" analysis (H1).
- **Switch critique → conflict-triggered rather than uncertainty-triggered.** De Neys's account emphasises that intuitive processing can produce *competing* intuitions, and that their conflict prompts deliberation. In VLA terms: trigger S2 on **multimodality** of S1's action samples (two distinct modes), not on raw **variance**, which also fires on harmless noise. TUD's dispersion signal is variance-like.
- **EVC → value-of-computation labels.** The right target for any trigger is "how much would an S2 call *now* change success?" This can be measured counterfactually in simulation (H3).
- **Event segmentation** is my suggestion and was not sourced in this session. The cognitive-science idea that internal models update at event boundaries, driven by prediction error, maps onto "refresh S2 at visually detected event boundaries". A vision researcher can supply this with off-the-shelf temporal-boundary detectors.

### Gaps
- I did not retrieve Gigerenzer's or Kruglanski's specific critiques of dual-process theory.
- I did not retrieve the content of Melnikoff & Bargh.
- I did not retrieve a primary source for Event Segmentation Theory.
- De Neys's exact mechanism for the switch was not read in full text.

## Q6. Falsifiable hypotheses with minimal experiments on one H200 (GPU-day estimates are my own)

### Takeaway
The cheapest high-perspective bet is a **measurement paper on what the S2→S1 channel actually carries and when it is worth refreshing** (H1 + H2 + H3). It needs only inference on open checkpoints (π0.5 via openpi, GR00T N1.5, SmolVLA, plus OpenHelix if it runs) and is CVPR-2027 feasible. H7 is the most CVPR-native (pure vision, tracking/streaming). H5 (matched compute) is the most important scientifically but needs training, so it fits a later venue.

### Cited Findings (the evidence each hypothesis builds on)
- S2 latent is vision-insensitive and delay-insensitive on CALVIN. — [OpenHelix](https://arxiv.org/html/2505.03912v1)
- Training with random delays yields task-level slow features; untrained baselines collapse to ≤3–6% under a 40-step delay. — [CloudEdgeVLA](https://arxiv.org/abs/2608.00569)
- Uncertainty-triggered S2 cuts calls 43–75% with no random-trigger baseline. — [TUD](https://arxiv.org/html/2610.05025)
- π0.5 is brittle to the form of appended text (92.47% → 3.00%). — [TOWN-VLA](https://arxiv.org/abs/2608.23224)
- Better reasoning does not improve LIBERO-Plus success. — [TRUST](https://arxiv.org/abs/2610.00601)
- Streaming Video-LLMs already have a perception/decision/reaction split. — [Dispider](https://arxiv.org/html/2501.03218v1)

### Inferences — the hypotheses

**H1. "System 2 is a slow clock with a fast-decaying half-life only at event boundaries."** Staleness tolerance as a diagnostic.
- *Claim:*
  - In off-the-shelf VLAs, freezing the S2 output (π0.5: VLM prefix KV; GR00T N1.5: VLM tokens fed to the DiT) for k steps costs little success **except within a few steps of subtask boundaries** (gripper open/close, contact onset).
  - Damage is much larger for "blind-S1" architectures (π0.5, GR00T: vision enters only via S2) than for "S1-with-eyes" ones (OpenHelix/HiRT-style).
- *Falsified if:*
  - success decays uniformly with k regardless of phase, or
  - the blind-S1 and S1-with-eyes families decay equally.
- *Minimal experiment:*
  - Staleness k ∈ {0, 5, 10, 20, 40, ∞ = computed once at t = 0} on LIBERO's 4 suites.
  - Log per-step boundary distance and attribute failures to the first divergence step.
  - Add a "swap" control: S2 output from a *different episode of the same task* at the same phase.
- *Cost (my estimate):* 1–2 GPU-days for 2 models (plus about 1 more for OpenHelix on CALVIN). The bottleneck is CPU MuJoCo rollouts.
- *Venue:* **CVPR 2027 feasible.**
- *Overlap:* distinct from sibling idea #5 (modality attribution). This measures *temporal* information flow between systems, not image/state/language credit. They share LIBERO infrastructure and could be one paper's two chapters.

**H2. "The S2→S1 channel carries about log2(#subtasks) bits on today's benchmarks."** Channel-capacity curve.
- *Claim:* replacing the S2 message with (a) its per-task or per-subtask mean, or (b) a k-dim PCA/k-means-VQ compression (codebook fitted on rollout features; inference only), retains ≥90% of LIBERO success at a few bits. On LIBERO-Plus/LIBERO-PRO language and object perturbations, the bits needed grow sharply.
- *Reframing:* "S2 doesn't help" = "benchmarks don't demand bandwidth".
- *Falsified if:* success degrades smoothly with bits even on in-distribution LIBERO, i.e. S2 transmits rich per-step visual information.
- *Cost (my estimate):* 1–3 GPU-days (shares rollouts with H1).
- *Venue:* **CVPR 2027.**
- *Scoop risk:* medium. No source found measures this, but ERVLA/BeTTER must be checked.

**H3. "Value-of-computation (VoC) is sparse and is not what uncertainty measures."** EVC-inspired ground truth for triggers.
- *Claim:*
  - Using LIBERO state save/restore, branch at step t into "refresh S2 now" vs "keep stale for k" and roll out to the end.
  - Then VoC(t) = Δsuccess, and it is concentrated in <10–15% of steps.
  - Variance-type uncertainty (TUD-style dispersion, flow-sample variance) predicts VoC with AUROC ≤ about 0.7.
  - Visual event-boundary signals or action **multimodality** predict it better.
- *Falsified if:* dispersion has AUROC > 0.85, or VoC is spread uniformly.
- *Deliverable:* a VoC benchmark that any trigger paper (TUD, OneTwoVLA, When-to-Plan) can be scored on, plus a random-trigger baseline that TUD lacks.
- *Cost (my estimate):* 3–6 GPU-days, e.g. 200 episodes × 20–30 branch points × 2 arms × 3–5 seeds, mostly CPU. Trimming to one suite and one model gives about 2 GPU-days.
- *Venue:* CVPR 2027 if trimmed. Otherwise CoRL 2027.
- *Scoop risk:* medium-high in the trigger space (TUD 4 Oct 2026, "When to Plan", RARRL, TOWN-VLA). The *ground-truth VoC map* angle appears unclaimed.

**H4. "Conflict, not uncertainty, should wake System 2"** (De Neys-inspired; can be a section of H3).
- *Claim:* on LIBERO-Goal (same scene, multiple goals) and on perturbed scenes, a bimodality score over K action-chunk samples from the flow head (e.g. number of clusters, or a separation ratio) beats variance at matched S2-call budgets.
- *Falsified if:* variance ≥ multimodality at every budget.
- *Cost (my estimate):* +0.5–1 GPU-day on top of H3.
- *Venue:* CVPR 2027 as part of H3. Alone it is a workshop paper.

**H5. "Most dual-system gains are bigger-model gains."** Matched-compute control.
- *Claim:* on LIBERO and CALVIN in-distribution, a single-system policy given the same parameters/FLOPs-per-second as small-S1 + big-S2 (e.g. the big VLM running every step with action chunking, or S1 enlarged to equal total FLOPs) matches the dual system. A dual system whose S2 is replaced by a learned task-embedding lookup also matches it. Dual-system advantage appears only on language/novel-object generalization splits and under hard real-time constraints.
- *Falsified if:* the dual system beats both matched controls in-distribution.
- *Cost (my estimate):* 8–15 GPU-days (SmolVLA-scale training of 3–4 variants × 2–3 seeds).
- *Venue:* **later** (CoRL 2027 / ICLR 2028). It is too heavy for 34 days alongside other work.

**H6. "Language is the wrong wire: S2 phrasing is a covariate shift for S1."**
- *Claim:* for Hi-Robot-style language-subtask interfaces on π0.5/SmolVLA, S1 success degrades monotonically with the embedding distance of S2-generated subtask phrasings from the training phrasings (paraphrase sweep generated by an LLM). A spatial interface (point/box/trace overlay, à la Gemini-ER points or DualVLN pixel goals) is phrasing-invariant by construction.
- *Falsified if:* success is flat across paraphrase distance.
- *Cost (my estimate):* 1–2 GPU-days for the inference-only paraphrase study. A spatial-interface comparison requires fine-tuning S1 (+5–10 GPU-days, later venue).
- *Overlap:* partially overlaps sibling #5 (instruction attribution) and TOWN-VLA's prompt-form collapse. It is only worth doing as a chapter. **Not recommended standalone.**

**H7. Pure-vision S1/S2: "Call the slow model before the confusion, not after."** Tracking and streaming.
- *Claim:*
  - In online multi-object tracking (S1 = motion/IoU association; S2 = heavy appearance re-ID or a SAM2-style memory pass) and in streaming Video-LLMs (Dispider-style perception → reaction), invoking S2 on **predicted** future ambiguity is better than invoking it on current low confidence.
  - "Predicted future ambiguity" here means forecast track crossings or occlusions from S1 dynamics, or event boundaries.
  - The prospective trigger gives a strictly better ID-switch (or answer accuracy) vs S2-call-budget curve than the reactive one, at 5–10× fewer slow calls than per-frame S2.
- *Falsified if:* the reactive-confidence trigger dominates or equals the prospective trigger across budgets.
- *Cost (my estimate):* 1–3 GPU-days (MOT17/DanceTrack or cell-tracking data; inference only with off-the-shelf trackers/re-ID).
- *Venue:* **CVPR 2027**, the most CVPR-native.
- *Possible synergy* (my inference, unverified): cell-division events in the user's cell-tracking work are natural "event boundaries".
- *Scoop risk:* not checked. Adaptive re-ID invocation in MOT may exist under other names, so it needs about 1 hour of scoop search.

**Recommended packaging (my inference):**
- **CVPR 2027 paper:** "What does System 2 tell System 1? Staleness, bandwidth and value-of-computation in dual-system VLAs" (H1 + H2 + H3/H4).
  - Inference only, 4–8 GPU-days total.
  - Clear falsifiable plots: success-vs-staleness, success-vs-bits, VoC-vs-trigger AUROC.
  - One simple "fix" from the findings: boundary-triggered refresh at matched calls.
- **Alternative for a pure-vision submission:** H7.
- **Later venue:** H5 (matched compute) and the spatial-interface half of H6.

### Gaps
- I did not verify that openpi's π0.5 LIBERO checkpoint or GR00T N1.5's LIBERO fine-tune allow S2-output caching without code surgery. This is an implementation risk for H1–H4. Plan 1–2 days of engineering.
- I did not verify that OpenHelix code and checkpoints are public and runnable.
- LIBERO rollout throughput on the available CPUs is unknown. The sibling report notes GPU-day estimates for LIBERO can be off by 2×.
- Scoop checks still needed:
  - ERVLA (2606.03784), BeTTER, Latent Bridge (2605.02739), the full text of the Hi-VLA systematic study (2606.10267), and "Think Only When Needed" (2608.23224, trigger details) for H1–H4.
  - "adaptive re-identification invocation MOT" and "event-triggered streaming video LLM" for H7.
