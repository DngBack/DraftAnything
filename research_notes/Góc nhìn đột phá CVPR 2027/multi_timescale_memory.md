# Multi-timescale / multi-stage memory for robotics (VLA), video understanding and tracking

Scope note: these notes cover 2024 to Oct 2026. Every GPU-day number in the hypotheses section is **my own estimate**, not a measured figure. I deliberately avoided the "SAM 2 one-wrong-mask error cascade + confidence-gated write" idea (#4 in `reports/Ý tưởng nghiên cứu CVPR 2027.md`). Hypotheses that come close to it are flagged.

## Q1. Survey of mechanisms (2024–2026) and extracted design axes

### Takeaway
On the sequence-model side, memory has converged on a single frame: "associative memory updated by an online optimizer at some frequency" (Titans, MIRAS, ATLAS, Nested Learning/HOPE, LaCT). The robot and video side is still mostly hand-built keyframe and token banks. Since mid-2026 the frontier has moved to *event-triggered writes* (KEMO, WeaveLA, EventVLA, SelectStream) and *TTT fast weights as robot memory* (TTT-RM, Oct 2026). The open space is therefore no longer "use events" or "use TTT". It is matched-budget science on *when*, *where (space vs time)* and *how to consolidate*.

### Cited Findings
**Sequence-model memory (fast/slow weights, multi-frequency)**
- Nested Learning (NeurIPS 2025, poster) treats a model as nested, multi-level optimization problems. In this view architecture and optimizer differ mainly in *update rate and context flow*. The Continuum Memory System (CMS) turns the short/long-term split into a *spectrum of modules, each updated at its own frequency*. HOPE is a self-modifying Titans variant with CMS blocks. Titans has only two levels of parameter update, while HOPE has more. — [Google Research blog](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/116123); [arXiv 2512.24695](https://arxiv.org/abs/2512.24695v1) (the authors say the arXiv version is fuller than the condensed camera-ready).
- Titans (arXiv Dec 2024, NeurIPS 2025) pairs attention as short-term memory with a neural long-term memory that *learns at test time*. It scales to more than 2M context with better needle-in-haystack accuracy. — [mlanthology NeurIPS 2025](https://mlanthology.org/neurips/2025/behrouz2025neurips-titans); [arXiv 2501.00663](https://arxiv.org/pdf/2501.00663v1)
- MIRAS ("It's All Connected", ICLR 2026) recasts Transformers, Titans and linear RNNs as associative memories along four axes: memory architecture, attentional bias (the objective), retention gate (forgetting), and memory learning algorithm. ATLAS (May 2025) optimizes the memory update rule using the context window rather than single tokens. Secondary reports put it at about 80% accuracy at 10M tokens on BABILong; I did not verify this in the primary paper. — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008141); [course notes NL-1](https://barisdeniz.is-a.dev/courses/continual-learning/tracks/nested-learning/nl1-test-time-learning.html); [ATLAS summary](https://fugumt.com/fugumt/paper_check/2505.23735v1)
- LaCT "Test-Time Training Done Right" (ICLR 2026) updates fast weights on *very large chunks*, from 2K to 1M tokens, instead of every 16 to 64 tokens. Small-chunk TTT ran below 5% of GPU FLOPs. With large chunks the nonlinear state can reach about 40% of model size, and the method works with Muon in plain PyTorch. It is shown on novel view synthesis (more than 1M tokens), language modeling, and a 14B autoregressive video diffusion model (56K tokens). — [arXiv 2505.23884](https://arxiv.org/pdf/2505.23884); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009303)
- Follow-up "Fast Spatial Memory with Elastic TTT" (Apr 2026) notes that LaCT usually runs as a *single chunk over the whole input*. It adds an elastic prior so that multi-chunk updates stay stable, which shows that update frequency is still an unsolved knob. — [arXiv 2604.07350](https://arxiv.org/pdf/2604.07350)
- EM-LLM (ICLR 2025) segments the token stream into episodes where Bayesian *surprise* crosses an adaptive threshold (windowed mean and variance), then refines the boundaries with graph cohesion. Retrieval is two-stage: similarity, then temporally contiguous neighbors. It beats InfLLM and RAG on LongBench and ∞-Bench and retrieves over 10M tokens. Its boundaries correlate with human-perceived event boundaries. — [arXiv 2407.09450](https://arxiv.org/html/2407.09450v3); [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c05144b635df16ac9bbf8246bbbd55ca-Abstract-Conference.html)

**Robot / VLA memory**
- MemoryVLA (ICLR 2026) takes its design from cognitive science: working memory holds perceptual and cognitive tokens from the VLM, and a long-term Perceptual-Cognitive Memory Bank is retrieved, fused, and *merged when redundant* (a consolidation-like step). Reported results: SimplerEnv-Bridge 71.9, Fractal 72.7, LIBERO 96.5, MIKASA-Robo 41.2 (vs π0 29.4 and OpenVLA-OFT 28.4; CronusVLA reproduced at 18.0), and 84.0 on 12 real tasks. The v1 and v2 numbers differ. — [arXiv 2508.19236](https://arxiv.org/abs/2508.19236v2); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10011504)
- SAM2Act / SAM2Act+ (ICML 2025) builds a SAM2-style memory bank, encoder and memory-attention into the coarse branch of a coarse-to-fine 3D manipulation policy. It reaches 86.8% on RLBench (18 tasks), and its arXiv abstract reports 94.3% on its new MemoryBench. That number is not present in every version of the paper. — [arXiv 2501.18564](https://arxiv.org/html/2501.18564v4); [ICML 2025](https://icml.cc/virtual/2025/poster/44779)
- MemER (Stanford, Finn group) uses a hierarchy: a high-level VLM *selects keyframes* from experience, and an online consolidation step turns per-timestep candidates into a compact episodic memory. It helps on real tasks that need minutes of recall. Its stated limits are that keyframes cannot be deleted (no forgetting) and that memory is visual only. — [arXiv 2510.20328](https://arxiv.org/html/2510.20328v1)
- **Event-triggered writes for VLAs already exist (mid-2026).** KEMO detects events from robot kinematics plus visual filtering and stores keyframes as memory tokens. It reports +23.6% task success and +34.1% stage completion over memory-free π0.5 on real dual-arm tasks of 830 to 2,846 steps, and its ablation shows event keyframes beat uniform sampling and recent-frame retention. — [arXiv 2606.23589](https://arxiv.org/abs/2606.23589). WeaveLA writes 8 latent tokens at each sub-goal boundary into a frozen VLA, but the boundaries come from the simulator, so detection is not solved. — [arXiv 2606.17463](https://arxiv.org/abs/2606.17463). EventVLA uses NMS on keyframe scores to produce sparse write events. — [arXiv 2606.20092](https://arxiv.org/pdf/2606.20092). EvoMem-VLA keeps initial, recent and selected keyframes plus delta tokens. — [arXiv 2610.05418](https://arxiv.org/html/2610.05418v1)
- **TTT as robot memory already exists (3 Oct 2026).** TTT-RM stores in fast weights *only the residual history that the current observation and retrieved memory cannot explain*. The slow-weight objective is the reconstruction error of a history decoder. It reports gains on memory-intensive simulation and real tasks, including a 3-minute, 8-stage stowing task. — [arXiv 2610.04701](https://arxiv.org/abs/2610.04701). This counts as "contemporaneous" under the CVPR 2027 rule (published after 15 Sep 2026): it must be cited, but comparison is not required.
- μVLA adds recurrent memory to OpenVLA-OFT. On 5 trained MIKASA tasks, success rises from 0.42 to 0.84 at K=2, while K=1 and K=8 both reach 0.57. This non-monotonic result in K is direct evidence that the "memory timescale" knob matters. — [arXiv 2606.12497](https://arxiv.org/pdf/2606.12497) (via search summary)
- GaussMemory (IROS 2026) uses 3D Gaussian Splatting as a persistent spatial memory and *learns* which objects to track, how fast to update them and what to discard. It beats MemoryVLA on LIBERO-Goal and Long-10 and π0-FAST by 5.2 and 6.0 points on VLABench tracks. Absolute numbers were not in the abstract. — [arXiv 2608.14986](https://arxiv.org/abs/2608.14986)
- "Sleeping Robots" (CoRL 2026) alternates a wake phase (learn each skill) with an offline sleep phase that consolidates a shared policy. Memories are frozen skill surrogates, not trajectories, and per-skill gradients are combined by Nash bargaining. It is evaluated on Meta-World MT5 and SurgicAI. — [arXiv 2606.17493](https://arxiv.org/html/2606.17493v1)

**Video / world-model memory**
- WorldMem keeps a token memory bank with state-aware attention, retrieved by field-of-view overlap, to stop "amnesia" when a location is revisited (Minecraft). Context-as-Memory conditions on past frames selected by FOV overlap with camera pose. MemLearner (2606.31734) argues that these *rule-based (geometric) retrievals fail under occlusion and dynamic objects* and learns the query instead. — [survey 2603.28489](https://arxiv.org/pdf/2603.28489); [MemLearner](https://huggingface.co/papers/2606.31734.md); [Implicit retrieval 2606.23105](https://arxiv.org/pdf/2606.23105)
- SelectStream (arXiv 2606.16353, NeurIPS 2026) keeps the current frame visible to a frozen VLM. It writes history with *surprise-driven adaptive windows*, consolidates with priority under a fixed node budget, and reads through query-conditioned graph retrieval. Results: StreamingBench 82.67, OVO-Bench 67.03. It argues that "indiscriminate history injection can dilute current-scene perception". — [arXiv 2606.16353](https://arxiv.org/abs/2606.16353). Event-VStream also writes event memory only when a boundary probability crosses a threshold. — [arXiv 2601.15655](https://arxiv.org/html/2601.15655v1)
- Tracking: SAM2Long replaces SAM 2's single memory path with a training-free memory tree for +5.3 J&F. Its memory *write policy* is the lever. — [arXiv 2410.16268](https://arxiv.org/html/2410.16268v2)

### Design-axis table (synthesized from the sources above)
| System | What is stored | When written | Frequency | Read | Forgetting |
|---|---|---|---|---|---|
| Titans / ATLAS | weights of an MLP memory | every token (Titans), sliding window objective (ATLAS) | per token or chunk | forward pass of memory MLP | learned retention / weight decay gate |
| HOPE / CMS | weights at several levels | every level on its own clock | multi-frequency spectrum | stacked MLP blocks | slow levels compress fast ones |
| LaCT | fast-weight SwiGLU | per large chunk (2K–1M tokens) | very low | apply fast weights | overwritten per chunk |
| EM-LLM / SelectStream / Event-VStream | KV / latent events | surprise boundary | event-driven | similarity + contiguity / graph | budgeted consolidation |
| MemoryVLA | perceptual + cognitive tokens | every step, then merge | per step | cross-attention retrieval | merge redundant entries |
| MemER | keyframes (pixels) | VLM-selected, consolidated online | sparse | VLM context | none (stated limitation) |
| KEMO / WeaveLA / EventVLA | keyframe tokens / 8 latents | kinematic or subgoal events | event-driven | cross-attention + gated residual | bank cleared per episode |
| TTT-RM | fast weights (residual history) | every step | per step | query fast weights | implicit overwrite |
| SAM2Act+ / SAM 2 | spatial feature maps | each frame (FIFO) | per frame | memory attention | FIFO eviction |
| WorldMem / Context-as-Memory | frame tokens + pose | each generated frame | per frame | FOV-overlap (geometric) retrieval | none / retrieval-limited |
| GaussMemory | 3D Gaussians per object | learned update | learned per object | rendering / object queries | learned discard |
| Sleeping Robots | skill surrogates → policy weights | offline sleep phase | between sessions | policy forward | Nash-bargained consolidation |

### Inferences
- The field has two separate vocabularies. Sequence-model memory (MIRAS axes, CMS frequencies) is principled about *frequency and forgetting* but tested on text and NVS. Robot and video memory is principled about *what to store* (keyframes, 3D) but its forgetting is ad hoc: FIFO, none, or clearing per episode. A paper that measures robot or video memory using the MIRAS/CMS axes would be a recognizable bridge.
- Write *frequency* is the least studied axis on the robot side. μVLA's non-monotonic K and LaCT needing elastic priors for multi-chunk updates both point to a real optimum that nobody has characterized against task timescales.
- Forgetting is barely studied in robot memory: MemER cannot delete, and CheckVLA and others clear the bank per episode.

### Gaps
- I did not obtain the full RoboMME results table (14 memory variants on π0.5, ICML 2026). The only finding I have is "highly task-dependent". — [arXiv 2603.04639](https://arxiv.org/abs/2603.04639)
- RoboMemory and long-horizon VLA memory surveys were not fetched separately. Streaming video LLM memories (Flash-VStream, StreamMem, VideoLLaMB, etc.) were not individually verified; the [Awesome-Streaming list](https://github.com/sotayang/Awesome-Streaming-Video-Understanding) indexes them.
- SAM 2 memory internals (number of recent frames, prompted-frame retention) come from background knowledge, not from a source fetched here.

## Q2. Which of these memory papers got orals or awards in 2025–2026?

### Takeaway
None of the memory papers surveyed received a major best-paper or outstanding-paper award in 2025–2026. The award lists I could verify (ICLR 2026, NeurIPS 2025) contain no memory-architecture or robot-memory paper. I could not confirm the oral or spotlight status of the individual memory papers.

### Cited Findings
- ICLR 2026 Outstanding Papers: "Transformers are Inherently Succinct" and "LLMs Get Lost In Multi-Turn Conversation". Honorable mention: "The Polar Express" (Muon matrix-sign). — [ICLR blog](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/)
- NeurIPS 2025 best papers include Gated Attention (Qwen) and "Artificial Hivemind" (D&B track). A runner-up was "Does RL Really Incentivize Reasoning Capacity…". No memory paper appears in the confirmed items. — [NeurIPS blog](https://blog.neurips.cc/?p=1443); [search summary](https://www.theneuron.ai/explainer-articles/the-best-papers-at-neurips-2025-explained)
- Venue status found: Nested Learning was a NeurIPS 2025 *poster* page ([link](https://neurips.cc/virtual/2025/poster/116123)), but heavy Google blog promotion made it high-visibility. Titans was NeurIPS 2025 ([mlanthology](https://mlanthology.org/neurips/2025/behrouz2025neurips-titans)). MIRAS ([poster](https://iclr.cc/virtual/2026/poster/10008141)), LaCT ([poster](https://iclr.cc/virtual/2026/poster/10009303)) and MemoryVLA ([poster](https://iclr.cc/virtual/2026/poster/10011504)) were ICLR 2026 with poster pages. EM-LLM was ICLR 2025 ([poster](https://iclr.cc/virtual/2025/poster/30579)). SAM2Act was ICML 2025 ([poster](https://icml.cc/virtual/2025/poster/44779)). RoboMME is ICML 2026, and SelectStream and MIKASA-Robo-VLA are NeurIPS 2026.
- From the existing ideas file: CVPR 2025 and 2026 best papers went to 3D/4D foundation models (VGGT, D4RT). — [CVPR 2026 closing](https://cvpr.thecvf.com/Conferences/2026/News/Closing)

### Inferences
- A "virtual/poster/…" URL does not rule out an oral, since orals also get poster pages. Treat oral status as unknown, not as "no".
- Memory is a crowded, high-visibility area (Google's Nested Learning PR), but it has not won awards. An analysis paper that reframes it (e.g., "your memory benchmark doesn't need memory" or "write when surprised, not every frame, under matched budget") is likelier to stand out than another memory module.

### Gaps
- I did not find a reliable oral or spotlight list for NeurIPS 2025, ICLR 2026, ICML 2025/2026 or CoRL/RSS 2025–2026 covering these papers. Check Paper Copilot or OpenReview decision fields.
- CoRL 2025/2026 and RSS awards for robot-memory papers were not checked.

## Q3. Evidence that robot/VLA (and video) memory benchmarks are solvable without memory

### Takeaway
There is solid evidence that LIBERO(-Long) is essentially fully observable. VLAs encode history but barely *use* it, because history is "a redundant copy of the present". Even purpose-built memory suites leak, through action chunking and cue-independent guessing. Blind (no-memory) controls are reported inconsistently, and no cross-benchmark audit with matched controls exists that I could find.

### Cited Findings
- "Present but Not Remembered" (Jul 2026) applies layer-wise probes and causal interchange to 3 frozen VLAs from 2 families. Past frames are linearly decodable everywhere, but *information unique to history is nearly absent*. History is used causally only when the current frame is heavily degraded, and action readout progressively drops its dependence on history. Quote: "VLAs do not forget the past; they largely fail to represent it as information distinct from the present." The authors recommend injecting *past-unique* information rather than more history. — [arXiv 2607.03372](https://arxiv.org/abs/2607.03372)
- A search summary of related work says existing VLA sim benchmarks are "designed for memoryless tasks", so temporal-modeling gains may come "from other artifacts" ([arXiv 2602.03983](https://arxiv.org/pdf/2602.03983)). μVLA calls LIBERO a "fully observable control suite" in which the current observation exposes objects, layout and goals ([arXiv 2606.12497](https://arxiv.org/pdf/2606.12497)). Both come from search snippets and were not read in full.
- OpenVLA-OFT, which has no explicit memory, reaches a 97.1% average across the four LIBERO suites. — [arXiv 2502.19645](https://arxiv.org/abs/2502.19645v1). MemoryVLA reports 96.5 on LIBERO ([arXiv 2508.19236](https://arxiv.org/abs/2508.19236v2)). The ceiling leaves almost no room for memory to matter. I could not verify the exact single-frame LIBERO-Long figure (about 94–95%).
- In MIKASA-Robo (original RL suite), the authors show the tasks are solvable at 100% by a *state-based, memory-free oracle* when given full state. This validates that the tasks need memory only under partial observability. In real experiments, performance degraded sharply when an occluded interval required memory. — [arXiv 2502.10550](https://arxiv.org/html/2502.10550)
- MIKASA-Robo-VLA (90 tasks, 10 memory types, NeurIPS 2026) reports a memory-free π0.5 at 0.211±0.044 on 14 tasks. On 7 discrete-choice tasks it scored 33/140, against 35.3 expected from *cue-independent guessing*: no evidence of cue use. Open-loop chunking is a confound that shifts per-task success by up to 0.65 in μVLA's analysis. The authors say length alone does not imply memory demand and that their info-gap measure "does not rule out shortcuts". A search summary also says one task is solvable without explicit memory because a single action chunk keeps the initial observation. — [arXiv 2610.00604](https://arxiv.org/html/2610.00604)
- RoboMemArena reports that 68.9% of subtasks are history-dependent, and a reactive π0.5 reaches 21.5%. — [arXiv 2605.10921](https://arxiv.org/pdf/2605.10921). In the cross-session interference benchmark, no-history π0.5 scores 17.3%, the best memory reaches 45.3% with only the relevant demo, and accuracy falls as distractor sessions are added. — [arXiv 2606.22338](https://arxiv.org/html/2606.22338v1)
- Video: SelectStream argues that injecting history indiscriminately *dilutes* current-scene perception, i.e., more memory can hurt. — [arXiv 2606.16353](https://arxiv.org/abs/2606.16353). The existing ideas file also notes that VLM benchmarks often lack blind controls.

### Inferences
- The pieces exist (probe/interchange audit, info-gap measure, memory-free oracle), but they are scattered across papers, models and benchmarks. A single "memory necessity" audit (no-memory, shuffled-history, donor-history swap, and chunk-length control) across LIBERO-Long, MIKASA-Robo-VLA, RoboMME, MemoryBench and a streaming video QA set would be a clean CVPR-style analysis. The risk is overlap with 2607.03372, which audits encoding, while this audit would target *benchmarks*.
- Action chunking is an under-appreciated *implicit memory* (open-loop chunks carry the initial observation forward). Any memory claim must control chunk length.

### Gaps
- No public per-task "memory-free vs memory" table across *published memory VLAs* on the same benchmark exists; the comparisons use different subsets.
- SAM2Act MemoryBench was not checked for whether a memory-free policy (e.g., SAM2Act without the +) solves part of it. The paper's own SAM2Act vs SAM2Act+ gap is the natural evidence; the number is not verified here.
- Video VOS: I found no direct "no-memory-bank vs memory-bank" audit per benchmark (e.g., DAVIS vs MOSE vs LVOS) in this search.

## Q4. Neuroscience / cognitive-science results not yet operationalized in vision/robot memory

### Takeaway
*Surprise-triggered writing* (event segmentation) is now operationalized in LLMs (EM-LLM), streaming video (SelectStream, Event-VStream) and VLAs (KEMO, WeaveLA). It is no longer a novelty claim by itself. Less exploited ideas include: the *boundary advantage* (prioritizing boundary content at read time), *internal-state boundaries* (goal or subgoal changes, not visual change), *schema-dependent fast consolidation*, *prediction-error-gated reconsolidation (updating, not appending)*, and hippocampal *pattern separation* for perceptually aliased task stages.

### Cited Findings
- Event Segmentation Theory: boundaries arise when the active predictive model becomes inaccurate, and the prediction error triggers a transient attentional shift that updates the model. Memory is better for boundary moments, and hippocampal activity peaks are specific to subjective boundaries (Ben-Yakov & Henson 2018). — [PMC review 6788820](https://pmc.ncbi.nlm.nih.gov/articles/PMC6788820); [PMC 8674822](https://pmc.ncbi.nlm.nih.gov/articles/PMC8674822)
- Extension: *internal* state changes (goals, motivation, affect) also act as boundary triggers, not only external change. — [Psychon Bull Rev 2023](https://link.springer.com/article/10.3758/s13423-023-02375-2)
- Prediction error as the *sole* trigger is debated (2026). — [SFB1102 2026 paper](https://sfb1102.uni-saarland.de/sfbunisb/uploads/2026/03/s13415-026-01399-0.pdf)
- Spatial and nonspatial (event) segmentation may share neural machinery: boundaries shape representations of both spaces and events. — [Brunec, Moscovitch & Barense](https://neuropsychologylab.psych.utoronto.ca/files/Boundaries%20shape%20cognitive%20representations%20of%20spaces%20and%20events.pdf)
- Schema-dependent consolidation (Tse et al. 2007, Science): when a prior schema exists, new one-trial associations become hippocampus-independent *within ~48h*, i.e., consolidation can be fast if new info fits an existing structure. — [Edinburgh record](https://www.research.ed.ac.uk/en/publications/schemas-and-memory-consolidation/); [PMC 2271071](https://pmc.ncbi.nlm.nih.gov/articles/PMC2271071)
- Operationalized already: CLS-style dual stores (EVAF explicitly disclaims mapping to hippocampus or cortex, [arXiv 2606.29916](https://arxiv.org/pdf/2606.29916)), wake-sleep consolidation ([arXiv 2401.08623](https://arxiv.org/pdf/2401.08623)), sleep consolidation for robot skills (Sleeping Robots, [arXiv 2606.17493](https://arxiv.org/html/2606.17493v1)), and surprise boundaries (EM-LLM, KEMO, SelectStream as cited in Q1).

### Inferences
- **Not operationalized (as far as these searches found):**
  1. *Boundary advantage at read time*: weighting retrieval toward boundary-adjacent memories rather than only writing at boundaries.
  2. *Goal or subgoal (internal) boundaries in VLAs*: WeaveLA uses simulator subgoals, but nobody detects boundaries from the policy's *own* internal state, such as action-chunk prediction error or the language-plan step.
  3. *Schema-gated test-time consolidation*: consolidate an episodic item into fast weights only if it fits the pre-trained "schema", i.e., low loss under slow weights. TTT-RM does the opposite and stores only the unexplained residual. The two are directly contrastable.
  4. *Reconsolidation*: on retrieval with a prediction error, *edit* the retrieved memory entry rather than appending a new one. Current banks append or merge.
  5. *Pattern separation*: decorrelating keys for perceptually similar but stage-different states. KEMO's motivation ("similar observations at different task stages") is exactly this problem, but it is solved by keyframes, not by separated keys.
  6. *Event boundaries that fork memory*: in tracking with division, split or merge, the boundary should fork or merge identity memory, not just trigger a write. This links to the repo's LineageRAFT work.
- Reverse and compressed replay at reward (Foster & Wilson 2006) and dentate-gyrus pattern-separation reviews are from my background knowledge; I did not fetch or verify them in this session.

### Gaps
- No systematic review was found that checks which CLS or replay predictions have been tested in embodied AI. The "not operationalized" list above comes from absence in my searches, not from proof of absence.

## Q5. Falsifiable hypotheses and minimal experiments on one H200 (GPU-days = my estimates)

### Takeaway
The best CVPR 2027 bets (deadline 16 Nov 2026, about 24 usable experiment days) are *analysis-first* and need ≤10 GPU-days. They are the cross-benchmark memory-necessity audit (H1), and the matched-budget event-vs-fixed write study framed as a task-timescale law (H2). The spatial-vs-temporal crossover (H3) is feasible if it uses fine-tuned small policies. Consolidation (H4) and pattern separation (H5) are stronger but riskier, and fit ICCV/NeurIPS 2027 better.

### Cited Findings (anchors used by the hypotheses)
- Memory-free baselines are already low on memory suites: π0.5 scores 0.211 on MIKASA-Robo-VLA ([2610.00604](https://arxiv.org/html/2610.00604)), 21.5% on RoboMemArena ([2605.10921](https://arxiv.org/pdf/2605.10921)) and 17.3% on the interference benchmark ([2606.22338](https://arxiv.org/html/2606.22338v1)), but near ceiling on LIBERO ([2502.19645](https://arxiv.org/abs/2502.19645v1)).
- μVLA's non-monotonic memory size (K=2 best) ([2606.12497](https://arxiv.org/pdf/2606.12497)). KEMO shows event > uniform, but without a matched-budget regime analysis ([2606.23589](https://arxiv.org/abs/2606.23589)). RoboMME finds memory "highly task-dependent" ([2603.04639](https://arxiv.org/abs/2603.04639)).
- TTT-RM stores the residual ([2610.04701](https://arxiv.org/abs/2610.04701)). Tse et al. show schema-fitting info consolidates fast ([PMC 2271071](https://pmc.ncbi.nlm.nih.gov/articles/PMC2271071)).

### Inferences — hypotheses

**H1. "Most long-horizon memory benchmarks are Markov in disguise" (memory-necessity audit).**
- Claim: on LIBERO-Long, and on a measurable fraction of MemoryBench and MIKASA-Robo-VLA tasks, (a) a memory-free policy with the *same* chunk length is within 3 points of the memory policy, and (b) swapping in a *donor* history from a different episode does not change actions. On truly memory-dependent tasks, the donor swap flips the discrete choice above chance.
- Falsified if: the memory–no-memory gap on LIBERO-Long is ≥5 points with chunk length controlled, or if donor swaps change behavior on LIBERO-Long.
- Experiment: take public checkpoints (MemoryVLA, π0.5 / OpenVLA-OFT, μVLA if released). Run four controls (no memory, shuffled history, donor history, chunk=1 vs 8) and add a per-task "Memory Necessity Index" = (success_mem − success_blind)/(success_oracle − success_blind). Extend to one streaming-video QA set (OVO-Bench, blind-history control).
- Cost: 4–7 GPU-days (inference only, 20–50 episodes per task). **CVPR 2027: yes.** Risk: 2607.03372 audits models rather than benchmarks; frame the claim as being about benchmarks.

**H2. "Write when the world changes, at the rate the task changes" (event vs fixed frequency under matched budget).**
- Claim: at an *equal number of memory writes*, surprise- or kinematics-triggered writes beat uniform-rate writes *only* in proportion to cue sparsity (fraction of timesteps carrying task-relevant info). On dense-cue tasks the advantage disappears or reverses. A CMS-style 2-rate memory (fast FIFO + slow event store) dominates either single rate.
- Falsified if: the event-trigger gain is flat across cue sparsity, or if uniform writing matches event writing at equal budget, which would mean KEMO's gain was a budget artifact.
- Experiment: MIKASA-Robo-VLA tasks binned by memory type and delay. Fine-tune a small policy (e.g., a ≤1B VLA or diffusion policy from MIKASA demos) with a plug-in keyframe memory. Write triggers: uniform k, kinematic event, prediction-error (policy action-loss) surprise, random. Budgets: 4, 8, 16. Mirror the design training-free on streaming video QA with a frozen VLM to get a vision-side replication.
- Cost: 6–10 GPU-days. **CVPR 2027: feasible** if limited to 6–8 tasks. Novelty versus KEMO, SelectStream and Event-VStream lies in the *matched-budget regime law*, not in "events help".

**H3. "Manipulation memory is spatial, not temporal" (crossover).**
- Claim: for object or spatial memory tasks (where a hidden object is), a 3D-anchored memory (object-centric slots keyed by 3D position from depth) beats temporal keyframe memory at equal tokens. For temporal, counting or procedural tasks the reverse holds. The result is a 2×2 crossover, not a uniform winner.
- Falsified if: one representation wins on both categories, or the interaction is not significant.
- Experiment: RoboMME's or MIKASA-Robo-VLA's taxonomy gives the category labels. Use the same backbone and token budget; the memory is either keyframe tokens or depth-unprojected feature voxels/slots (GaussMemory-lite, no learning of discard). The camera-motion condition (moving wrist cam) should amplify the spatial advantage.
- Cost: 8–12 GPU-days. **CVPR 2027: borderline**; ICCV 2027 is safer. It explains RoboMME's "task-dependent" finding mechanistically.

**H4. "Schema-gated test-time consolidation" (CLS at test time, between episodes).**
- Claim: consolidating an episodic buffer into LoRA or fast weights between episodes, gated by *fit to slow weights* (low loss means schema-consistent, so consolidate quickly; high loss means keep it episodic), improves cross-episode and cross-session tasks and resists distractor sessions better than either retrieval-only or residual-only (TTT-RM-style) storage.
- Falsified if: ungated consolidation, or retrieval-only, matches the gated version on the interference benchmark as distractors increase.
- Experiment: the cross-session interference benchmark ([2606.22338](https://arxiv.org/html/2606.22338v1)) or a MIKASA cross-episode variant, small policy, a "sleep" step of N gradient updates between episodes.
- Cost: 10–15 GPU-days. **Later venue** (NeurIPS 2027 / CoRL 2027). It must cite TTT-RM and Sleeping Robots as contemporaneous work.

**H5. "Memory aliasing: VLA and VOS memory confuses look-alike states from different stages" (pattern separation).**
- Claim: retrieval errors concentrate on near-duplicate observations at different task stages (e.g., the same table before and after "put in drawer"). Adding a sparse random expansion plus an elapsed-event code to memory keys (a dentate-gyrus-style separator, training-free) reduces stage confusion and raises success specifically on aliasing tasks.
- Falsified if: the retrieval error rate is not elevated for high-similarity cross-stage pairs, or the key separator does not reduce it.
- Experiment: instrument MemoryVLA-style retrieval on MIKASA tasks or RoboMME procedural and counting tasks. Measure the attention mass that the retrieved memory puts on the wrong stage. Add the separator at inference.
- Cost: 3–5 GPU-days (largely inference). **CVPR 2027: possible** as part of H1 (a diagnostic section). It does not overlap the SAM 2 error-cascade idea, because it concerns *aliasing at read time*, not error propagation through writes.

**H6. "Boundaries should fork memory, not just write it" (tracking with topology change), a link to LineageRAFT.**
- Claim: for objects that split (cell division, object breaking, groups splitting), a memory-based tracker with one bank per identity fails at the boundary. Forking the bank at a detected event boundary (surprise in mask area or shape) restores daughter tracking without retraining.
- Falsified if: forking does not improve DIV/branching metrics over SAM2 baselines on cell time-lapse or a split/merge subset.
- Cost: 2–4 GPU-days. **CVPR 2027: as a section or teaser of LineageRAFT** (supports the "one query, one identity" critique). Overlap warning: this is adjacent to idea #4 in the existing file (SAM 2 memory), so keep the framing on *event-boundary forking*, not on error cascade.

**Ranking for CVPR 2027 (my judgement):** H1 (with H5 as a diagnostic) > H2 > H6 (inside LineageRAFT) > H3 > H4. A combined paper, "Do robot memory benchmarks need memory, and when should memory write?" (H1+H2), fits about 10–15 GPU-days and gives a single-sentence claim for an AC.

### Gaps
- No public weights or code were verified for MemoryVLA, μVLA, KEMO, WeaveLA or TTT-RM; H1 and H2 feasibility depends on release status. Check GitHub before committing.
- MIKASA-Robo-VLA and RoboMME simulator throughput on one H200 (episodes per hour) is unknown, so the GPU-day estimates could be off by 2×.
- Whether SAM2Act's MemoryBench is publicly runnable without RLBench/CoppeliaSim headaches was not checked.
