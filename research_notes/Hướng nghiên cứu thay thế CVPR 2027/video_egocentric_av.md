# Video understanding beyond tracking: open, insight-driven problems for CVPR 2027 on 1 H200

Scope note: this covers Video-LLMs (long and streaming video, temporal grounding), egocentric vision, audio-visual LLMs, action anticipation and video representation learning. It deliberately avoids the shortlist in `reports/Ý tưởng nghiên cứu CVPR 2027.md`: LineageRAFT, MLLM "arbitration" (image-level), RL-vs-base drift, SAM 2 memory error cascade / point tracking, 4D physics judge, 3D calibration, video-diffusion physics probes and REPA. Where an idea below touches the shortlist's #2 (arbitration), the overlap is flagged. Unless a source is cited, every GPU-day figure is my own rough estimate. I assumed bf16 7–8B Video-LLMs with 32–64 frames, served on vLLM or HF on one 141 GB H200. None of these figures has been measured.

## How much of Video-LLM benchmark performance is achievable from single frames / shuffled frames / text only (2025–2026 evidence)?

### Takeaway
A lot. Multiple-choice video benchmarks such as Video-MME and EgoSchema-style egocentric QA are largely solvable from a single frame or from priors. Shuffling or reversing frames barely moves scores on eight popular benchmarks. Even the "temporal" benchmarks show that the score mixes two things: whether a question needs several frames, and which input channel (pixels or positional encoding) supplies the order. The single-frame/shuffle audit by itself is now an established genre (TVBench, TimeBlind, TOC-Bench, EgoTempo, VBenchComp). A 2027 paper therefore needs a sharper second step, such as a mechanism, a long-video extension or a fix.

### Cited Findings
- Gemini-1.5-Pro exceeds 50% accuracy on Video-MME from a single random frame. On 21 video LMMs, accuracy drops by more than 25% when questions switch from MCQ to open-ended format. Adding frames often fails to improve scores on existing benchmarks — [VideoEval-Pro, arXiv 2505.14640](https://arxiv.org/html/2505.14640v1)
- On 8 popular video QA benchmarks, LMM performance "barely changes" when frames are shuffled or reversed. ArrowRL (an RL reverse-reward) gains more than 20% on AoTBench and more than 10% on standard VQA (NeurIPS 2025, Xue, Luo and Grauman) — [Seeing the Arrow of Time, arXiv 2506.03340](https://arxiv.org/pdf/2506.03340); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/118287)
- TVBench: shuffling barely changes scores on existing benchmarks. Text-only and single-frame models (including Gemini 1.5 Pro and GPT-4o) are at chance on TVBench, and most Video-LLMs are near chance. Only a few (Qwen2-VL, Tarsier) clearly beat it — [Lost in Time / TVBench, BMVC 2025](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_857/paper.pdf); [project page](https://daniel-cores.github.io/tvbench/)
- For several existing benchmarks, scores are essentially unchanged after shuffling, and shuffling sometimes improves them (e.g., Gemini-1.5-Pro on EgoSchema). The paper splits questions into LLM-Answerable / Semantic / Temporal / Others (VBenchComp, Apple authors, arXiv May 2025). I could not retrieve the per-benchmark proportions — [arXiv 2505.14321](https://arxiv.org/abs/2505.14321)
- EgoTempo (CVPR 2025): SOTA MLLMs reach "remarkably high performance" on existing egocentric QA benchmarks using only text or a single frame, yet fall short on EgoTempo — [arXiv 2503.13646](https://arxiv.org/abs/2503.13646)
- TimeBlind (2026): with GPT-5, the shuffled-frame condition scores 3.0% I-Acc and single-frame 4.5%, against 46.3% for full video. These controls show the benchmark needs sequence modelling — [arXiv 2602.00288](https://arxiv.org/html/2602.00288v4)
- TOC-Bench filters out items answerable from text alone, a single frame, or shuffled frames, and keeps about 17.9K of 45.5K generated items (roughly 61% removed) — [arXiv 2605.09904](https://arxiv.org/html/2605.09904v1)
- "Train on the test set" (NYU, Nov 2025): k-fold fine-tuning an LLM on text-only inputs of the test set yields a per-sample bias score. The authors report pervasive non-visual biases in VSI-Bench, CV-Bench, MMMU and **VideoMME**, and release VSI-Bench-Debiased — [arXiv 2511.04655](https://arxiv.org/html/2511.04655v1)
- Channel decomposition (single-author, 1× RTX 4090, Jul 2026), on TempCompass and TVBench across 7 models (Qwen3-VL 4B/8B, Molmo2 4B/8B/O-7B, PerceptionLM 1B/3B):
  - Overall multi-frame gain (full − single frame) is about +0.21–0.22, and order-gain (full − shuffled) about +0.17.
  - On the "action" aspect, order-gain is about 0.00–0.01 and blind (blank-frame) accuracy is about 0.50 against a chance of 0.29.
  - Disrupting RoPE alone costs Molmo 14–18 points but Qwen only 3–7. The authors read this as Molmo being "position-dominant" and Qwen "visual-sequence-dominant".
  - Stated limitations: only two short-video benchmarks, "longer-video datasets are left to future work", and mostly single-seed — [arXiv 2607.12304](https://arxiv.org/html/2607.12304v1)
- Video-HOCA: a single middle frame hurt reasoning-heavy tasks, while shuffling degraded only some tasks. The model seems to rely on multi-frame coverage more than on exact order — [arXiv 2602.19571](https://arxiv.org/pdf/2602.19571)
- EgoSchema already filtered out questions a blind LLM could answer when it was built. Original models scored under 33% against 20% chance and about 76% for humans — [EgoSchema, arXiv 2308.09126](https://arxiv.org/html/2308.09126v1)

### Inferences
- The behavioural audit (blind, single-frame, shuffle) is necessary but no longer novel. Reviewers in 2027 will expect it as a baseline, not as the contribution.
- Two clear openings remain. First, **long video**: 2607.12304 explicitly leaves long-video channel analysis open, and the long benchmarks (Video-MME-long, MLVU, LongVideoBench, LVBench) have not had a reversal/RoPE-conflict decomposition. Second, **internal versus behavioural**: is order information present in hidden states but unused? ArrowRL shows behaviour is insensitive to order, but nobody has shown whether the representation encodes order.
- A "temporal-necessity" per-item label for the major long benchmarks would likely be adopted widely, much as VSI-Bench-Debiased was.

### Gaps
- I found no single table giving single-frame / shuffle / text-only accuracy for **open** 7–8B models (Qwen2.5/3-VL, InternVL3, LLaVA-Video, VideoLLaMA3) across Video-MME, MLVU, LongVideoBench and EgoSchema together. That table itself is cheap to produce (my estimate: about 2–4 GPU-days).
- I could not retrieve VBenchComp's per-benchmark proportions (the PDF exceeded the fetch limit).
- I found no exact EgoSchema blind accuracy for 2025 models. The AVUT paper's Table 8 reportedly has it ([arXiv 2503.19951](https://arxiv.org/html/2503.19951)), but I did not verify this.

## Are video encoders like V-JEPA 2 learning dynamics/physics vs appearance? Open probes?

### Takeaway
The evidence is mixed. V-JEPA 2 leads on motion benchmarks and its physics signal is probe-dependent: weak with linear probes, strong with temporal attentive probes. It sits only slightly above chance on the realistic IntPhys 2. Prior work also finds that video models lean on static appearance and lose temporal-order information in deeper layers. No published work cleanly separates appearance from dynamics with stimuli where appearance carries no information (for example, point-light or random-dot motion). That is a clean, cheap opening.

### Cited Findings
- V-JEPA 2 outperforms SOTA encoders on motion tasks (SSv2, Diving-48, Jester) while staying competitive on appearance tasks (K400, COIN, ImageNet) — [V-JEPA 2, arXiv 2506.09985](https://arxiv.org/pdf/2506.09985)
- Probing study (Sep 2026, v2) of V-JEPA, VideoMAE and LTX-Video on IntPhys2 and MVP:
  - Physics information is most accessible at intermediate-to-late depth.
  - Disrupting frame order "substantially reduces performance, especially on MVP".
  - V-JEPA-family pair consistency rises from about 43–48% (linear probe) to 72–88% (MLP) to above 93% (temporal attentive probe).
  - Some temporal probes show a 0.00% relative drop under shuffling, which the authors read as probes collapsing onto non-temporal cues — [arXiv 2606.09646](https://arxiv.org/html/2606.09646v2)
- On IntPhys 2, V-JEPA 2 scores about 57.5% (main set) and 56.4% (held-out set), with failures in memory and occlusion. This is a secondary summary, not the primary paper — [Emergent Mind IntPhys 2](https://www.emergentmind.com/topics/intphys-2)
- A robustness study of V-JEPA 2/2.1, VideoPrism and VideoMAEv2 notes that prior work "collectively established that video models frequently rely on static appearance cues and that deeper layers progressively lose temporal order information" — [arXiv 2605.15618](https://arxiv.org/html/2605.15618v1)
- A V-JEPA 2 predictor-based violation-of-expectation study finds that middle encoder layers beat final layers — [Interpreting Physics in Video World Models, arXiv 2602.07050](https://arxiv.org/pdf/2602.07050)
- MotionJEPA (Sep 2026) names "temporal feature collapse" in JEPA video features and adds a latent visual-change objective — [arXiv 2609.23881](https://arxiv.org/pdf/2609.23881)
- Factorized latent dynamics for video JEPA, an empirical study of auxiliary objectives — [arXiv 2605.17165](https://arxiv.org/pdf/2605.17165)
- An independent GitHub replication finds that direction, speed and acceleration are linearly readable early in the V-JEPA 2 encoder. A mid-depth "emergence zone" appears only on a harder render. This is an unreviewed source — [jepa_physics repo](https://github.com/stevenybuilder/jepa_physics)

### Inferences
- The field has many intuitive-physics probe papers. What it lacks is an **appearance-nulled** test, a classic cognitive-science design: biological motion and point-light walkers, random-dot kinematograms, or the same motion rendered with re-textured appearance. With such stimuli, a frozen-feature probe shows whether a representation encodes dynamics or uses appearance as a proxy. It can be run on V-JEPA 2, VideoMAE v2, VideoPrism, SigLIP/DINOv2 per-frame plus temporal pooling (an appearance-only control), and the vision towers of Video-LLMs.
- The "temporal collapse" and "deep layers lose order" observations come from several papers, but each is a side note. A unified layer-wise "order-information curve" across about 8 public encoders is cheap and would be a citable reference figure.

### Gaps
- I found no published appearance-nulled (point-light or kinematogram) probe of V-JEPA 2 or VideoMAE. This is absence of evidence from about 3 searches, not proof that none exists. Run a quick arXiv check for "point-light", "biological motion" and "V-JEPA" before committing.
- I did not verify the exact IntPhys 2 numbers from the primary paper.

## Do audio-visual LLMs really fuse audio? Evidence.

### Takeaway
Mostly not. Several 2025–2026 papers converge on two findings. Omni and audio-visual LLMs show visual (and text) dominance and systematic audio neglect. Deep layers suppress audio, and blocking visual tokens at those layers restores audio reasoning. Existing AV benchmarks are visually biased, so audio adds little. The "do they use audio" question is now crowded at the diagnosis level. The remaining gap is **what kind of audio** matters (speech that a transcript could replace, versus non-speech sound) and a benchmark-wide audio-necessity audit.

### Cited Findings
- On ten omni LLMs, most show a pronounced visual preference with tri-modal input and "universally exhibit a systematic neglect of audio". Under tri-modal conflict, audio modality-selection rate stays below 21% for every model (Ming-Lite-Omni 1.5: 1%) — [Beyond Text-Dominance, arXiv 2604.16902](https://arxiv.org/pdf/2604.16902)
- CVPR 2026 Findings: attention knockout shows that deeper layers prioritise visual features and actively suppress audio. Blocking visual signals at those layers restores audio reasoning, and audio understanding degrades by up to 56% under AV conflict. This comes from a search snippet; the PDF fetch returned 403 — [Do Audio-Visual LLMs Really See and Hear?, CVPRF 2026](https://openaccess.thecvf.com/content/CVPR2026F/papers/Selvakumar_Do_Audio-Visual_Large_Language_Models_Really_See_and_Hear_CVPRF_2026_paper.pdf)
- DAVE (NeurIPS 2025 D&B): existing AV benchmarks suffer from "strong visual bias". On AVQA, video-only models score close to audio+video. DAVE is built so that both modalities are necessary — [arXiv 2503.09321](https://arxiv.org/pdf/2503.09321); [OpenReview](https://openreview.net/pdf/c7b7b8187b0925a4100a65f483ef3dff0215ab34.pdf)
- AVUT (EMNLP 2025): an audio-centric video benchmark designed without text shortcuts. It notes that audio content can often be inferred from visuals — [arXiv 2503.19951](https://arxiv.org/html/2503.19951); [ACL Anthology](https://aclanthology.org/2025.emnlp-main.333.pdf)
- Omni-R1: fine-tuning Qwen2.5-Omni on **text only** (audio dropped) still raises audio benchmark scores, so much of the "audio" gain is text reasoning — [arXiv 2505.09439](https://arxiv.org/pdf/2505.09439)
- Qwen2.5-Omni attends far more to text tokens than to audio tokens throughout generation — [Pay More Attention to Audio, arXiv 2509.18816](https://arxiv.org/pdf/2509.18816)
- Zero-shot Qwen2.5-Omni on Music-AVQA: 83.1% on Audio questions but only 52.0% on Audio-Visual and 52.1% on Visual — [Qwen-MusicAVQA-7B, arXiv 2608.11329](https://arxiv.org/html/2608.11329v1)
- Causal tracing of cross-modal "information hubs" in Qwen2.5-Omni and video-SALMONN — [arXiv 2605.10815](https://arxiv.org/html/2605.10815v1)
- MLLMs "rely heavily on the visual modality" for AV tasks (Findings of EMNLP 2025) — [arXiv 2504.16936](https://arxiv.org/pdf/2504.16936)
- Fix-type papers already exist: modality-specific CoT ([arXiv 2605.09906](https://arxiv.org/pdf/2605.09906)), AV contrastive decoding ([AVCD, arXiv 2505.20862](https://arxiv.org/pdf/2505.20862)), and modality-aware memory/token compression ([OmniMem, arXiv 2606.07577](https://arxiv.org/html/2606.07577); [OmniSIFT, arXiv 2602.04804](https://arxiv.org/pdf/2602.04804))
- New egocentric AV benchmarks at CVPR 2026: EgoSound (7,315 QA over 900 videos) and EgoAVU (3K evaluation and 3M training samples) — [EgoSound](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhu_EgoSound_Benchmarking_Sound_Understanding_in_Egocentric_Videos_CVPR_2026_paper.pdf); [EgoAVU](https://openaccess.thecvf.com/content/CVPR2026/html/Seth_EgoAVU_Egocentric_Audio-Visual_Understanding_CVPR_2026_paper.html)

### Inferences
- Another "AV-LLMs ignore audio" diagnosis would be scooped: at least 2604.16902, the CVPRF 2026 paper and 2605.10815 cover it. Contrastive-decoding fixes are also crowded.
- A sharper, untested question is whether the useful audio is just speech. Replace raw audio with a Whisper transcript, keep it, or replace it with audio from a mismatched video. If the transcript recovers nearly all of the "audio gain" on Video-MME-with-audio, WorldSense, DAVE and AVUT, then "audio-visual fusion" is mostly ASR plus text. That claim is nameable, and the experiment is cheap.

### Gaps
- I did not retrieve DAVE's numeric video-only versus AV gap or the CVPRF paper's model list (PDF 403).
- I found no paper that systematically compares raw audio with an ASR transcript across AV benchmarks. This is absence of evidence from 2 searches.

## Insight-driven video papers that got orals/highlights/awards at CVPR/ICCV/ECCV 2024–2026

### Takeaway
Top awards in 2025 went to 3D/4D and foundation-model work. The video-flavoured winners were MegaSaM (dynamic-video structure-from-motion) and Navigation World Models, both CVPR 2025 Best Paper Honorable Mentions. Pure Video-LLM analysis papers have not won top awards in the sources I found. Egocentric datasets (Ego4D, Ego-Exo4D, HD-EPIC) are well cited but need data collection. Insight papers in this space (TVBench, EgoTempo, Arrow-of-Time) landed as regular CVPR/NeurIPS/BMVC papers. For a 1-GPU team, oral or highlight is the realistic target, not an award.

### Cited Findings
- CVPR 2025: Best Paper went to VGGT. Honorable mentions went to **MegaSaM** (structure and motion from casual dynamic videos), **Navigation World Models**, Molmo & PixMo, and 3D Student Splatting and Scooping. Best Student Paper went to Neural Inverse Rendering from Propagating Light — [CVPR 2025 Awards](https://cvpr.thecvf.com/Conferences/2025/News/Awards_Press)
- ICCV 2025: the Marr Prize went to "Generating Physically Stable and Buildable Brick Structures from Text". Honorable mentions went to Spatially-Varying Autofocus and RayZer. None of these is a video-understanding paper — [IEEE TCPAMI ICCV awards](https://tc.computer.org/tcpami/awards/iccv-paper-awards/)
- Ego4D (CVPR 2022) was an Oral and Best Paper Finalist — [UT-Austin publications](https://www.cs.utexas.edu/~grauman/research/pubs-by-year.html)
- Ego-Exo4D (CVPR 2024), with an IJCV journal version published online on 24 Nov 2025 — [CVPR 2024 open access](https://openaccess.thecvf.com/content/CVPR2024/html/Grauman_Ego-Exo4D_Understanding_Skilled_Human_Activity_from_First-_and_Third-Person_Perspectives_CVPR_2024_paper.html); [Bristol research portal](https://research-information.bris.ac.uk/en/publications/ego-exo4d-understanding-skilled-human-activity-from-first-and-thi-2/)
- HD-EPIC (CVPR 2025): 41 h of video, 26K VQA questions including gaze, 3D and object motion. Gemini Pro scores only about 37–38.5% (the figure differs across paper versions) — [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Perrett_HD-EPIC_A_Highly-Detailed_Egocentric_Video_Dataset_CVPR_2025_paper.pdf); [arXiv 2502.04144](https://arxiv.org/html/2502.04144v1)
- CVPR 2026 oral (video/egocentric-adjacent): VGGT-Segmentor, cross-view segmentation on Ego-Exo4D (67.7/68.0 average IoU) — [CVPR 2026 oral](https://cvpr.thecvf.com/virtual/2026/oral/40296)
- Insight-type video papers with venues: EgoTempo at CVPR 2025 ([arXiv 2503.13646](https://arxiv.org/abs/2503.13646)), Arrow of Time as a NeurIPS 2025 poster ([NeurIPS](https://neurips.cc/virtual/2025/poster/118287)), TVBench at BMVC 2025 ([BMVC](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_857/paper.pdf)), and FOCUS frame selection at ICLR 2026 ([ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/file/a010577f93a8301dada61f0b97cda923-Paper-Conference.pdf))
- The sibling report gives CVPR 2026 Best Paper as D4RT (4D) ([CVPR 2026 Closing](https://cvpr.thecvf.com/Conferences/2026/News/Closing), cited there; not re-verified here)

### Inferences
- The awards pattern repeats here. Video work that wins either builds a geometry or world model (MegaSaM, NWM) or ships a large dataset (Ego4D). Diagnostic Video-LLM papers get accepted but rarely get top awards. A highlight needs a named phenomenon backed by evidence across many models, plus a cheap fix, as with Registers or Eyes Wide Shut.

### Gaps
- I could not find the official CVPR 2026 or ECCV 2026 oral/highlight lists filtered for video. I cannot confirm which 2026 Video-LLM analysis papers got orals or highlights. ECCV 2026 awards were not checked here.

## Egocentric open problems feasible without new data collection

### Takeaway
Existing egocentric data (Ego4D, EgoSchema, EgoTempo, HD-EPIC, Ego-Exo4D, EgoLife-style long logs) has enough annotation for analysis papers. The cleanest openings are three questions. Which egocentric cues (hands, gaze, head motion) do Video-LLMs actually use? What explains the jump from Gemini's 37% to 67% on HD-EPIC? Do the ego and exo benchmarks need the viewpoint they claim to need?

### Cited Findings
- HD-EPIC VQA: Gemini Pro scores about 37.6%. A CVPR 2026 challenge entry (EgoAdapt) reaches 67.22%, +23.01 pp over the previous best (DeepFrames). A fine-tuned Qwen2.5-VL entry reports 41.6% — [EgoAdapt, arXiv 2605.24500](https://arxiv.org/pdf/2605.24500); [HD-EPIC challenge solution, HF 2601.10228](https://huggingface.co/papers/2601.10228.md)
- Ego-Exo4D proficiency: ego views help close-up hand-object domains (cooking) and exo views help body-pose domains (bouldering). In a CVPR 2025 challenge entry:
  - Exo helped dance (59.3% vs 47.2% ego) and music (68.4% vs 57.9%).
  - Ego helped rock climbing (44.0% vs 34.6%).
  - Soccer was the same either way (62.5%).
  - The dataset skews toward intermediate and late experts. Health and bike repair are excluded because of label imbalance — [SkillFormer, arXiv 2505.08665](https://arxiv.org/pdf/2505.08665); [Ego-Exo4D, arXiv 2311.18259](https://arxiv.org/pdf/2311.18259)
- Egocentric QA benchmarks are solvable from text or a single frame (EgoTempo) — [arXiv 2503.13646](https://arxiv.org/abs/2503.13646)
- A wave of new egocentric benchmarks in 2025–2026: EgoIllusion (hallucination), EgoBlind, EgoIntent, BARISTA, Ego2Web, Minerva-Ego/EgoSTAR, EgoCross, EgoSound and EgoAVU — [EgoIllusion](https://arxiv.org/pdf/2508.12687); [EgoBlind](https://arxiv.org/html/2503.08221); [EgoIntent](https://arxiv.org/pdf/2603.12147); [BARISTA](https://arxiv.org/pdf/2605.12074); [Ego2Web](https://openaccess.thecvf.com/content/CVPR2026/papers/Yu_Ego2Web_A_Web_Agent_Benchmark_Grounded_in_Egocentric_Videos_CVPR_2026_paper.pdf); [Minerva-Ego](https://cvpr.thecvf.com/virtual/2026/poster/39579); [EgoCross](https://arxiv.org/pdf/2608.04589)
- Exo2Ego transfers exocentric knowledge to egocentric MLLMs — [arXiv 2503.09143](https://arxiv.org/pdf/2503.09143)
- Ego4D LTA 2025 winner: SlowFast-style encoder plus Transformer for verb and noun, then a fine-tuned Llama2-7B that predicts 20 future verb–noun pairs. It names intent recognition as future work — [arXiv 2506.02550](https://arxiv.org/abs/2506.02550)

### Inferences
- Another egocentric benchmark would be saturated (at least 9 in 2025–2026 above). Analyses of what drives egocentric performance are still scarce.
- HD-EPIC's 30-point spread between generic VLMs and adapted challenge systems is a ready-made object for an ablation-attribution study. It could separate frame rate, crop or zoom on hands, domain fine-tuning, prompt and answer-format effects, and text priors. No new data is needed, and HD-EPIC's own gaze and 3D annotations allow controlled cue-ablation.

### Gaps
- I did not verify which components of EgoAdapt drive its 67%. This needs the paper itself.
- No 2026 analysis of Ego-Exo4D proficiency appeared in my searches.

## What 8–12 specific open research questions fit a 1-H200 team for CVPR 2027 (Nov 16, 2026)?

### Takeaway
The ten questions below are ranked by my estimate of insight per GPU-day and scoop risk. All use open models (Qwen2.5-VL/Qwen3-VL, InternVL3, LLaVA-Video, VideoLLaMA3, Molmo2, Qwen2.5-Omni, video-SALMONN 2, V-JEPA 2, VideoMAE v2) and existing benchmarks. GPU-days are **my estimates**. The common template is to name a phenomenon, show it across 5 or more open models, explain it with a causal intervention, and fix it training-free.

| # | Open question (claim-style title) | Gap (cited above) | Simple insightful angle | Feasibility on 1 H200 (my estimate) | Main risk |
|---|---|---|---|---|---|
| Q1 | **"Long videos, short memories": how much of long-video benchmark accuracy uses more than one moment?** | Single-frame and shuffle audits exist for short or temporal benchmarks. 2607.12304 leaves long video to future work, and VideoEval-Pro shows >50% on Video-MME from 1 frame | Per-item temporal-necessity labels for Video-MME-long, MLVU, LongVideoBench and EgoSchema under blind, 1-frame, k-frame, shuffled, reversed and RoPE-conflict conditions, run on 6 open models. Publish a "Temporal-Necessary" subset and a frames-vs-accuracy dose-response curve | 3–5 GPU-days (inference only) | Audit genre is crowded (TVBench, TimeBlind, TOC-Bench, VBenchComp). The value rests on the long-video and channel angle plus a released subset |
| Q2 | **"Encoded but not used: order is in the hidden states, not in the answer"** | ArrowRL shows behaviour barely changes under reversal. No paper probes whether frame order or direction is linearly decodable inside Video-LLMs | Layer-wise linear probes for arrow-of-time, frame index and event order inside the vision tower and the LLM, compared against behavioural order-sensitivity. If decodable but unused, try a training-free fix (temporal-token re-weighting, RoPE time-scale, explicit timestamps) | 2–4 GPU-days | **Overlaps the shortlist's #2 "arbitration" idea.** It works best as its temporal extension or merged with it. ArrowRL has already handled the behavioural side |
| Q3 | **"Is audio just subtitles?" Raw audio vs ASR transcript in omni LLMs** | Visual dominance and audio neglect are established (2604.16902, CVPRF 2026). Omni-R1 shows text-only tuning raises audio scores. Nobody separates speech from non-speech audio | On Video-MME (with audio), WorldSense, DAVE, AVUT, Music-AVQA and EgoSound, compare video only, video + audio, video + Whisper transcript, video + mismatched audio, and non-speech-only audio (speech removed by source separation). Report "audio gain" split into speech and sound | 2–4 GPU-days (Whisper and Demucs-style separation are cheap) | Could reduce to a negative audit; needs one fix or a benchmark-subset release |
| Q4 | **"Point-light V-JEPA": do video encoders represent motion without appearance?** | Physics probes exist (2606.09646), but none is appearance-nulled. The "temporal collapse" claims are scattered | Synthetic point-light walkers, random-dot kinematograms and texture-swapped versions of SSv2 clips. Frozen probes for action, direction and causality across V-JEPA 2/2.1, VideoMAE v2, VideoPrism, the InternVideo2 vision tower and a per-frame DINOv2 control. Produce one "order-information-by-depth" curve per model | 2–4 GPU-days (feature extraction plus probes) | Needs a quick novelty check. Could be seen as cognitive science; frame it as an evaluation of world models |
| Q5 | **"Does frame selection matter? An oracle ceiling for long-video QA"** | Many training-free selectors (KTV, FOCUS, AKS, Q-Frame, MDP3, Q-Gate, QCA, MEDR). Nobody has reported the oracle-selection upper bound over uniform sampling for each benchmark | Estimate an oracle (answer-likelihood search over candidate subsets, or ground-truth evidence windows where available) and compare against uniform, random and the published selectors. If oracle minus uniform is small on most benchmarks, the selector line is aiming at the wrong bottleneck | 4–8 GPU-days (oracle search is the cost; use a subset) | Search is expensive on long benchmarks; sample about 500 items |
| Q6 | **"Grounding or guessing?" Prior-only baselines for LLM temporal grounding** | Charades-STA events cluster near the start and end of the video. A prior-only baseline beat SOTA models before the LLM era. Charades queries follow templates. No LLM-era blind or prior study exists | Run Video-LLM grounders (TimeLens, TimeSuite-style, Qwen-VL native timestamps) under (a) a training-set prior span, (b) blank video, (c) padded or shifted video with the event moved, and (d) the timestamp-format distribution (round-number bias). Fix with a training-free calibration of the timestamp head | 2–3 GPU-days | A narrower venue fit; needs evidence across Charades-STA, ActivityNet-Captions and an hour-long set |
| Q7 | **"Proactive or periodic?" Is proactive streaming better than polling an offline model?** | Many proactive benchmarks and models (IPIBench, StreamPro, Proact-VL, Em-Garde). Em-Garde notes that per-timestep accuracy is biased for evaluating triggering | Compare proactive models with trivial schedulers (fixed-interval polling, scene-change-triggered polling, an oracle-trigger upper bound) on OVO-Bench Forward Active Responding, StreamingBench PO and IPIBench. Decompose error into when-to-speak and what-to-say | 3–6 GPU-days | Streaming harnesses are fiddly. The benchmark-coverage claim needs care |
| Q8 | **"Length-blind counting": do long-video LLMs' answers respond to the evidence at all?** | Cambrian-S/VSI-SUPER: scale does not fix continual counting. A secondary review reports Gemini giving near-constant counts | Dose-response test: insert or remove N object occurrences (copy-paste, or reuse VSI-SUPER's editing) and fit how much the predicted count changes. A slope of about 0 means length-blindness. Pair with a surprise-based segmentation fix inspired by Cambrian-S, without training | 3–5 GPU-days | Synthetic edits may be out of distribution; Cambrian-S authors may follow up |
| Q9 | **"What closed HD-EPIC's 30-point gap?" Cue attribution for egocentric QA** | Gemini is at about 37% while adapted systems reach 67%; the drivers are unexplained. EgoTempo shows single-frame shortcuts | Controlled ablations on HD-EPIC and EgoTempo: hand-crop or zoom, gaze-centred crop (HD-EPIC has gaze), head-motion stabilisation, fps, answer-format priors, blind baseline. Find which egocentric cue carries the gain | 3–5 GPU-days | HD-EPIC VQA is large (26K); subsample. Ties to a challenge, so framing must be general |
| Q10 | **"Script, not sight": how much of long-term action anticipation is a text prior?** | 2025 LTA winners use an LLM on verb–noun text. Intent is named as future work | Compare a text-only LLM or n-gram model on the history of past actions, the same model with video, and a copy-last-action baseline on Ego4D LTA and EPIC-Kitchens anticipation. Measure the gain attributable to vision | 1–2 GPU-days (mostly CPU and LLM) | Unverified whether this has already been done; check the Ego4D LTA leaderboard reports |
| Q11 (optional) | **"Viewpoint necessity" in Ego-Exo4D proficiency** | Ego and exo gains vary by domain, and the labels skew to experts | Blind and identity-only baselines (scenario, participant, scene) against video. Test whether proficiency is predictable from non-skill cues | 1–2 GPU-days | Small community; might be workshop-level |

### Cited Findings
- All gaps in the table are cited in the sections above. Additional sources for Q6: Charades-STA start/end concentration ([UniTime, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/5d2e24df9cfaad3189833b819c40b392-Paper-Conference.pdf)); a bias-only baseline beating several SOTA models on Charades-STA ([ACM TOMM 2023](https://dl.acm.org/doi/10.1145/3565573)); template-bound Charades queries, with MATTR 0.54 for Charades against 0.78 for a new benchmark ([arXiv 2606.12300](https://arxiv.org/html/2606.12300v1)); TimeLens refines Charades ([arXiv 2512.14698](https://arxiv.org/pdf/2512.14698)).
- Q7: [IPIBench](https://arxiv.org/html/2605.27074), [StreamPro](https://arxiv.org/html/2605.16381), [Em-Garde](https://arxiv.org/pdf/2603.19054), [Proact-VL](https://arxiv.org/pdf/2603.03447), [Response-G1](https://arxiv.org/pdf/2605.07575), [Eyes Wide Open](https://arxiv.org/pdf/2510.14560)
- Q8: [Cambrian-S, arXiv 2511.04670](https://arxiv.org/html/2511.04670v1); near-constant counts per a secondary review ([themoonlight](https://www.themoonlight.io/review/cambrian-s-towards-spatial-supersensing-in-video))

### Inferences
- The best ratio of insight to cost, in my judgement: Q3 (audio vs transcript) for novelty and cheapness; Q1 merged with Q2 into one paper ("temporal necessity" plus "order encoded but unused") as the strongest Video-LLM story; Q4 as the cleanest representation-learning probe. Q2 should be merged with the existing shortlist #2 rather than run as a separate paper.
- With about 10–15 spare GPU-days (per the sibling report's budget), at most one of these fits alongside LineageRAFT. Q3 or Q4 has the lowest overlap with the sibling shortlist.

### Gaps
- Novelty for Q3, Q4, Q5 and Q10 rests on searches that did not find prior work. Each needs a targeted arXiv check (for example, "ASR transcript audio-visual LLM ablation", "point-light video foundation model", "oracle frame selection upper bound", "Ego4D LTA text-only baseline") before committing.
- The inference costs are estimates and assume no contention for the GPU. The sibling report notes the GPU was once shared with a 130 GB SFT job.

## Which directions are saturated (avoid as the main contribution)?

### Takeaway
Avoid these as the main contribution: another training-free frame or keyframe selector, another streaming KV-cache compressor, another proactive-streaming benchmark, another egocentric or "MLLMs fail at X" video benchmark, another blind-filtered temporal benchmark, another AV contrastive-decoding fix, and another intuitive-physics probe paper without a new control.

### Cited Findings
- Frame selection: KTV (AAAI 2026), FOCUS (ICLR 2026), AKS, Q-Frame, MDP3, adaptive key-clip selection, Q-Gate, QCA and MEDR (2026) — [KTV](https://www.alphaxiv.org/abs/2602.03615); [FOCUS](https://arxiv.org/pdf/2510.27280); [MDP3](https://arxiv.org/pdf/2501.02885); [Key clips](https://arxiv.org/html/2510.02262); [Q-Gate](https://arxiv.org/html/2604.17422v1); [QCA](https://arxiv.org/pdf/2607.00983); [MEDR](https://arxiv.org/pdf/2608.15058)
- Streaming KV and memory compression: StreamKV (AAAI 2026), InfiniPot-V (NeurIPS 2025), MuKV (CVPR 2026), HERMES (ACL 2026) and OmniMem — [StreamKV](https://arxiv.org/pdf/2511.07278); [InfiniPot-V](https://neurips.cc/virtual/2025/poster/116667); [MuKV](https://cvpr.thecvf.com/virtual/2026/poster/39020); [OmniMem](https://arxiv.org/html/2606.07577)
- Proactive or streaming benchmarks and models (at least 7 in 2025–2026): IPIBench, StreamPro, OmniInteract, Proact-VL, Em-Garde, Response-G1, "Harnessing Streaming Video in the Wild" — [OmniInteract](https://arxiv.org/pdf/2605.26485); [Streaming in the Wild](https://arxiv.org/pdf/2606.08615); others are cited above
- Temporal benchmarks with blind filtering: TVBench, TimeBlind, TOC-Bench, EgoTempo, Video-HOCA and AoTBench (all cited above)
- Egocentric benchmarks: at least 9 listed in the egocentric section
- AV fixes: AVCD contrastive decoding, modality-specific CoT, and "pay more attention to audio" re-weighting (cited above)
- The sibling report already flags token pruners and contrastive decoding as saturated for image MLLMs (random or pooling often wins) — [arXiv 2502.11501](https://arxiv.org/html/2502.11501v1), via the sibling report

### Inferences
- A paper in a saturated line can still work if it is the **audit of the line** rather than one more entry. Q5's oracle ceiling for frame selection is an example. "Uniform polling is as good as proactive" (Q7) is another.
- Reviewers in 2027 will likely require blind, single-frame and shuffle controls for any video benchmark or claim. Including them is table stakes, not a contribution.

### Gaps
- I did not count CVPR 2026 accepted papers per subtopic. "Saturated" here is based on counts of 2025–2026 papers found in searches, not on official statistics.
