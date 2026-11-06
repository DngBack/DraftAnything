# Verified references (checked 2026-11-06)

Companion to `refs.bib`. Metadata comes from the raw arXiv abstract pages, the ACL Anthology BibTeX, and Crossref. It was parsed by script, so titles and author lists are not retyped. No field was filled from memory.

## 1. Status summary

- **UNVERIFIED: none.** All 21 candidates and all 9 foundational references (counting Qwen3 and Qwen3-VL separately) opened at the source.
- Reading depth differs a lot, and a verified item is not a read item.

| depth | items |
|---|---|
| Full text, all pages (section 3) | 2611.01517, DocHallu, CADET, ConfBench |
| Most of the body read | 2610.02880 (through section 4; I did not read its section 5 or conclusion closely) |
| Abstract plus targeted passages found by term search | 2510.16926, 2610.32664, 2608.26949, 2609.25770, 2505.19091, 2603.09095, 2605.07019, 2609.27457, 2606.07861 |
| Term search of the full text only, nothing useful found | 2610.20964, 2603.26784 |
| Abstract only | every other arXiv item, including all foundational reports, MM-AQA, PriVE-Bench, TAT-QA and TAT-DQA |
| Metadata only (title, author, pages, DOI) | Shapley 1953 |

Limits of the full-text reads: I read text extracted from the PDFs, where tables come out scrambled and figures are absent. I checked table values against their captions as well as I could, but **recheck any number against the PDF before quoting it**. The only figure I viewed as an image is DocHallu page 6. The per-model heatmaps in CADET (pp. 32, 34-41) came through as bare number strings; I skimmed them and did not interpret them cell by cell.

### Metadata issues to handle when citing

- **DocHallu** is the benchmark name, not the title. Cite "Perceptual Hallucination in Vision-Language Models: Definition, Analysis and Verification" (Hwang et al., Findings of ACL 2026, pp. 24710-24725).
- **FaithC4 (2607.21617):** the arXiv page says "Submitted on 29 May 2026" although the id prefix is 2607. Reported as found; I did not resolve it. The year is 2026 either way.
- **Res-Bench (2510.16926):** history shows v2 "(withdrawn)" (1 KB) on 2 Nov 2025 and a full v3 on 14 Nov 2025. v3 is what I read. Cite v3 explicitly.
- **Venues taken only from arXiv comments**, not confirmed in proceedings: MathLens ("NeurIPS 2026"), LensVLM ("Accepted to NeurIPS 2026"). 2610.02880 says only "Submitted to ICASSP 2027". The 2611.01517 reference list gives venues for other papers; I did not use those.
- **Year in `refs.bib`** is the v1 year, so MathLens is 2025 although v5 is October 2026.
- **Shapley 1953:** Crossref gives pp. 307-318 and the chapter title "17. A Value for n-Person Games". I dropped the "17." and kept Crossref's page range; many citations use 307-317, which I did not check against the book.
- **Qwen3 vs Qwen3-VL:** these are two different reports (2505.09388 is the text LLM report, 2511.21631 is the VL report). Your model needs 2511.21631.
- **InternVL3 vs InternVL3.5:** your model is InternVL3-8B (2504.10479). 2611.01517, 2606.07861, 2603.09095 and 2609.27457 use InternVL3.5, a different report (2508.18265, seen only in a reference list and not opened). Do not transfer their InternVL statements to InternVL3-8B without saying so.

## 2. Bibliography

### 2.1 Candidates from related_work.md

| label | exact title | full author list | year | venue / id | BibTeX key |
|---|---|---|---|---|---|
| CADET | Where MLLMs Fail and Why: Causal Task Decomposition for Capability Failure Diagnosis | Hu, Xia; Potetz, Brian; Lu, Chun-Ta; Yao, Huanfen; Guibas, Leonidas; Wang, Zhicheng; Zhou, Howard; Xing, Pengfei; Gallagher, Andrew | 2026 | arXiv:2609.38851 [cs.CL], v1 30 Sep 2026 | `hu2026cadet` |
| MathLens | What MLLMs Learn about When they Learn about Multimodal Reasoning | Chung, Jiwan; Joshi, Neel; Sharma, Pratyusha; Yu, Youngjae; Vineet, Vibhav | 2025 | arXiv:2510.01719 [cs.CL], v1 2 Oct 2025; comment: "NeurIPS 2026. Project page: this https URL" | `chung2025mathlens` |
| Reading, Not Thinking | Reading, Not Thinking: Understanding and Bridging the Modality Gap When Text Becomes Pixels in Multimodal LLMs | Sun, Kaiser; Yuan, Xiaochuang; Liu, Hongjun; Zhao, Chen; Zhang, Cheng; Dredze, Mark; Bai, Fan | 2026 | arXiv:2603.09095 [cs.CL], v1 10 Mar 2026 | `sun2026readingnotthinking` |
| A Table Is Worth 64 Tokens | A Table Is Worth 64 Tokens: Pixel-level Compression for Multi-Table Document Question Answering | Alonso, Iñigo; Lapata, Mirella | 2026 | arXiv:2608.26949 [cs.AI], v1 27 Aug 2026 | `alonso2026table64` |
| Shapley precedent (agents, not VLMs) | More Is Not Always Better: Cross-Component Interference in LLM Agent Scaffolding | Liu, Ming | 2026 | arXiv:2605.05716 [cs.AI], v1 7 May 2026; comment: "10 pages, 5 tables; preprint, under review" | `liu2026cci` |
| Reading or Guessing? | Reading or Guessing? Visual Grounding Failures of Vision-Language Models for OCR in Ancient Greek Editions | Karamolegkou, Antonia; Angleraud, Nicolas; Sagot, Benoît; Clérice, Thibault | 2026 | arXiv:2605.27750 [cs.CL], v1 26 May 2026 | `karamolegkou2026readingorguessing` |
| VLMs Read Aloud | VLMs Read Aloud: The Gap Between Recognizing and Using Visual Text | Shi, Zhiyun | 2026 | arXiv:2611.01517 [cs.CV], v1 1 Nov 2026; comment: "23 pages, 7 figures, 21 tables" | `shi2026readaloud` |
| Last Visible Pixel / FineSightBench | The Last Visible Pixel: Probing Fine-Scale Perception in Vision-Language Models | Li, Lujun; Sleem, Lama; Gentile, Niccolo; Xu, Yangjie; Song, Yewei; Wu, Wenbo; State, Radu | 2026 | arXiv:2606.07861 [cs.CV], v1 5 Jun 2026; comment: "25 pages" | `li2026lastvisiblepixel` |
| FaithC4 | Do VLMs Read or Rewrite? On Transcription Faithfulness in Vision-Language Models | Lee, Gwang Gook; Ak, Kenan Emir; Mohta, Jay; Xu, Yan; Dimitriadis, Dimitrios | 2026 | arXiv:2607.21617 [cs.AI], v1 29 May 2026; comment: "15 pages, 6 figures" | `lee2026faithc4` |
| Inside VLM Chart Reading | Inside VLM Chart Reading: Tracing Value Reading from Vertical Bar Charts Across Space and Depth | Niu, Tianhao; Zhu, Qingfu; Che, Wanxiang | 2026 | arXiv:2609.13745 [cs.CL], v1 12 Sep 2026 | `niu2026chartreading` |
| ReadBench | ReadBench: Measuring the Dense Text Visual Reading Ability of Vision-Language Models | Clavié, Benjamin; Brand, Florian | 2025 | arXiv:2505.19091 [cs.CL], v1 25 May 2025 | `clavie2025readbench` |
| VELD-QA | VELD-QA: A Large-Scale Dataset for Visual Evidence Grounding and Question Answering in Long Documents | Sourati, Zhivar; Liu, Marianne Menglin; Wang, Rongguang; Wang, Zheng; Hu, Yazhe; Guo, Lan; Wang, Yuying; Guo, Mengqing; Gore, Bhavesh Dnyaneshwar; Pathak, Shubham; Ranganath, Ramapriya; Sharma, Sandeep; Sheng, Tao; Ravi, Sujith; Roth, Dan | 2026 | arXiv:2610.20964 [cs.CV], v1 14 Oct 2026 | `sourati2026veldqa` |
| HighlightBench | HighlightBench: Benchmarking Markup-Driven Table Reasoning in Scientific Documents | Wang, Lexin; Liu, Shenghua; Wang, Yiwei; Cai, Yujun; Ge, Yuyao; Yao, Jiayu; Guo, Jiafeng; Cheng, Xueqi | 2026 | arXiv:2603.26784 [cs.CV], v1 25 Mar 2026 | `wang2026highlightbench` |
| Small VLMs Know | Small Vision-Language Models Know When They Are Wrong But Cannot Say So: A Two-Model Study of Stated versus Internal Confidence Under Realistic Image Degradation | Ferdous, M M Asif | 2026 | arXiv:2607.22034 [cs.CV], v1 24 Jul 2026; comment: "15 pages, 4 figures" | `ferdous2026smallvlms` |
| Bigger or Cheaper? | Bigger or Cheaper? Scale and Quantization Effects on Uncertainty Signals in Vision-Language Models Under Image Degradation | Ferdous, M M Asif | 2026 | arXiv:2607.24440 [cs.CV], v1 27 Jul 2026; comment: "12 pages, 4 figures. Code and data: this https URL" | `ferdous2026biggerorcheaper` |
| ConfBench | Can You Trust the Confidence? ConfBench for Vision-Language Models on Document Extraction | Roy, Priyashree; Martin, Sujitha; Rostami, Mohammad; Romo, Spencer; Xue, Renhao; Strahan, Bob; Socolinsky, Diego A.; Xie, Boyi; Islam, Md Mofijul | 2026 | arXiv:2608.01792 [cs.AI], v1 3 Aug 2026 | `roy2026confbench` |
| Financial VLM confidence | Confidence Estimation for Financial Vision-Language Models in Chart and Document Understanding | Khanmohammadi, Reza; Kaur, Simerjot; Smiley, Charese H.; Brugere, Ivan; Ghassemi, Mohammad M. | 2026 | arXiv:2608.06532 [cs.CL], v1 6 Aug 2026 | `khanmohammadi2026financialconfidence` |
| How Murky | How Murky Before a VLM Knows It Cannot See? Accuracy, Confidence and Abstention on a Physical Visibility Axis | Fauzan, Mochamad Rizal; Surya, Irgi | 2026 | arXiv:2610.22981 [cs.CV], v1 16 Oct 2026; comment: "16 pages, 9 figures, 10 tables" | `fauzan2026murky` |
| VAD-R / Rep2Act | From Knowing to Abstaining: Bridging the Representation-Action Gap in Vision-Language Models | He, Jialuo; Chen, Huangxun | 2026 | arXiv:2609.32653 [cs.CL], v1 26 Sep 2026 | `he2026knowingabstaining` |
| Instinct vs. Reflection | Instinct vs. Reflection: Unifying Token and Verbalized Confidence in Multimodal Large Models | Dang, Yunkai; Jiang, Yifan; Jiang, Yizhu; Chen, Anqi; Li, Wenbin; Gao, Yang | 2026 | arXiv:2604.17274 [cs.CV], v1 19 Apr 2026 | `dang2026instinct` |
| DocHallu | Perceptual Hallucination in Vision-Language Models: Definition, Analysis and Verification | Taewook Hwang; Inbum Heo; Sung Jun Lee; Sangkeun Jung | 2026 | Findings of ACL 2026, pp. 24710-24725, San Diego; doi 10.18653/v1/2026.findings-acl.1237 | `hwang2026perceptual` |

### 2.2 Foundational references

| label | exact title | full author list | year | venue / id | BibTeX key |
|---|---|---|---|---|---|
| Qwen2.5-VL report | Qwen2.5-VL Technical Report | Bai, Shuai; Chen, Keqin; Liu, Xuejing; Wang, Jialin; Ge, Wenbin; Song, Sibo; Dang, Kai; Wang, Peng; Wang, Shijie; Tang, Jun; Zhong, Humen; Zhu, Yuanzhi; Yang, Mingkun; Li, Zhaohai; Wan, Jianqiang; Wang, Pengfei; Ding, Wei; Fu, Zheren; Xu, Yiheng; Ye, Jiabo; Zhang, Xi; Xie, Tianbao; Cheng, Zesen; Zhang, Hang; Yang, Zhibo; Xu, Haiyang; Lin, Junyang | 2025 | arXiv:2502.13923 [cs.CV], v1 19 Feb 2025 | `bai2025qwen25vl` |
| Qwen3 report | Qwen3 Technical Report | Yang, An; Li, Anfeng; Yang, Baosong; Zhang, Beichen; Hui, Binyuan; Zheng, Bo; Yu, Bowen; Gao, Chang; Huang, Chengen; Lv, Chenxu; Zheng, Chujie; Liu, Dayiheng; Zhou, Fan; Huang, Fei; Hu, Feng; Ge, Hao; Wei, Haoran; Lin, Huan; Tang, Jialong; Yang, Jian; Tu, Jianhong; Zhang, Jianwei; Yang, Jianxin; Yang, Jiaxi; Zhou, Jing; Zhou, Jingren; Lin, Junyang; Dang, Kai; Bao, Keqin; Yang, Kexin; Yu, Le; Deng, Lianghao; Li, Mei; Xue, Mingfeng; Li, Mingze; Zhang, Pei; Wang, Peng; Zhu, Qin; Men, Rui; Gao, Ruize; Liu, Shixuan; Luo, Shuang; Li, Tianhao; Tang, Tianyi; Yin, Wenbiao; Ren, Xingzhang; Wang, Xinyu; Zhang, Xinyu; Ren, Xuancheng; Fan, Yang; Su, Yang; Zhang, Yichang; Zhang, Yinger; Wan, Yu; Liu, Yuqiong; Wang, Zekun; Cui, Zeyu; Zhang, Zhenru; Zhou, Zhipeng; Qiu, Zihan | 2025 | arXiv:2505.09388 [cs.CL], v1 14 May 2025 | `yang2025qwen3` |
| Qwen3-VL report | Qwen3-VL Technical Report | Bai, Shuai; Cai, Yuxuan; Chen, Ruizhe; Chen, Keqin; Chen, Xionghui; Cheng, Zesen; Deng, Lianghao; Ding, Wei; Gao, Chang; Ge, Chunjiang; Ge, Wenbin; Guo, Zhifang; Huang, Qidong; Huang, Jie; Huang, Fei; Hui, Binyuan; Jiang, Shutong; Li, Zhaohai; Li, Mingsheng; Li, Mei; Li, Kaixin; Lin, Zicheng; Lin, Junyang; Liu, Xuejing; Liu, Jiawei; Liu, Chenglong; Liu, Yang; Liu, Dayiheng; Liu, Shixuan; Lu, Dunjie; Luo, Ruilin; Lv, Chenxu; Men, Rui; Meng, Lingchen; Ren, Xuancheng; Ren, Xingzhang; Song, Sibo; Sun, Yuchong; Tang, Jun; Tu, Jianhong; Wan, Jianqiang; Wang, Peng; Wang, Pengfei; Wang, Qiuyue; Wang, Yuxuan; Xie, Tianbao; Xu, Yiheng; Xu, Haiyang; Xu, Jin; Yang, Zhibo; Yang, Mingkun; Yang, Jianxin; Yang, An; Yu, Bowen; Zhang, Fei; Zhang, Hang; Zhang, Xi; Zheng, Bo; Zhong, Humen; Zhou, Jingren; Zhou, Fan; Zhou, Jing; Zhu, Yuanzhi; Zhu, Ke | 2025 | arXiv:2511.21631 [cs.CV], v1 26 Nov 2025; comment: "42 pages" | `bai2025qwen3vl` |
| InternVL3 report | InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models | Zhu, Jinguo; Wang, Weiyun; Chen, Zhe; Liu, Zhaoyang; Ye, Shenglong; Gu, Lixin; Tian, Hao; Duan, Yuchen; Su, Weijie; Shao, Jie; Gao, Zhangwei; Cui, Erfei; Wang, Xuehui; Cao, Yue; Liu, Yangzhou; Wei, Xingguang; Zhang, Hongjie; Wang, Haomin; Xu, Weiye; Li, Hao; Wang, Jiahao; Deng, Nianchen; Li, Songze; He, Yinan; Jiang, Tan; Luo, Jiapeng; Wang, Yi; He, Conghui; Shi, Botian; Zhang, Xingcheng; Shao, Wenqi; He, Junjun; Xiong, Yingtong; Qu, Wenwen; Sun, Peng; Jiao, Penglong; Lv, Han; Wu, Lijun; Zhang, Kaipeng; Deng, Huipeng; Ge, Jiaye; Chen, Kai; Wang, Limin; Dou, Min; Lu, Lewei; Zhu, Xizhou; Lu, Tong; Lin, Dahua; Qiao, Yu; Dai, Jifeng; Wang, Wenhai | 2025 | arXiv:2504.10479 [cs.CV], v1 14 Apr 2025; comment: "Technical Report" | `zhu2025internvl3` |
| MM-AQA | Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems | Madhusudhan, Nishanth; Yadav, Vikas; Lacoste, Alexandre | 2026 | arXiv:2604.14799 [cs.CL], v1 16 Apr 2026; comment: "10 pages and 4 figures (excluding appendix)" | `madhusudhan2026mmaqa` |
| PriVE-Bench | Seeing What Is Actually There: PriVE-Bench and PriVE-Tools for Counterfactual Evaluation of Agentic Visual Evidence in VLMs | Sun, Jingyu; Tu, Jiachen; Xue, Yuyang; Jiang, Yaoxin; Xu, Guoyi; Yao, Zhengtao; Qian, Rui; Sun, Yizheng; Zhou, Hongpeng; Sun, Jingyuan; Lin, Yan | 2026 | arXiv:2607.16311 [cs.CV], v1 14 Jul 2026 | `sun2026privebench` |
| TAT-QA | TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance | Fengbin Zhu; Wenqiang Lei; Youcheng Huang; Chao Wang; Shuo Zhang; Jiancheng Lv; Fuli Feng; Tat-Seng Chua | 2021 | ACL-IJCNLP 2021 (Long Papers), pp. 3277-3287; doi 10.18653/v1/2021.acl-long.254; arXiv:2105.07624 | `zhu2021tatqa` |
| TAT-DQA | Towards Complex Document Understanding By Discrete Reasoning | Fengbin Zhu; Wenqiang Lei; Fuli Feng; Chao Wang; Haozhou Zhang; Tat-Seng Chua | 2022 | ACM MM 2022, pp. 4857-4866; doi 10.1145/3503161.3548422; arXiv:2207.11871 | `zhu2022tatdqa` |
| Shapley 1953 | A Value for n-Person Games | L. S. Shapley | 1953 | Contributions to the Theory of Games (AM-28), Vol. II, Princeton University Press, pp. 307-318; doi 10.1515/9781400881970-018 | `shapley1953value` |

### 2.3 New items not on the candidate list (found in this review)

| label | exact title | full author list | year | venue / id | BibTeX key |
|---|---|---|---|---|---|
| Found but Not Read / FoveDoc-Bench | Found but Not Read: When Extracted Text Closes the Retrieval-Reading Gap in Document Vision-Language Models | Xia, Qingtao; Cheng, Siyao; Bao, Jiahua; Du, Jiaxing; Liu, Jie | 2026 | arXiv:2610.02880 [cs.CV], v1 2 Oct 2026; comment: "5 pages, 5 figures, 3 tables. Submitted to ICASSP 2027" | `xia2026foundnotread` |
| Res-Bench | Res-Bench: Benchmarking the Robustness of Multimodal Large Language Models to Dynamic Resolution Input | Li, Chenxu; Wang, Zhicai; Sheng, Yuan; Zhu, Xingyu; Hao, Yanbin; Wang, Xiang | 2025 | arXiv:2510.16926 [cs.CV], v1 19 Oct 2025; comment: "23 pages" | `li2025resbench` |
| The Answer Knows Where to Look / LookZoom | The Answer Knows Where to Look: Zooming before MLLMs Answer under Visual-Token Budgets | Hu, Zhensheng; Xu, Guanglong; Hu, Guanqiang | 2026 | arXiv:2610.32664 [cs.CV], v1 23 Oct 2026 | `hu2026lookzoom` |
| Reading Right, Answering Wrong | Reading Right, Answering Wrong: How Visual Configuration Changes Affect Evidence Use in VLMs | Lin, Dingyang; Luo, Yingfeng; Wang, Chenglong; Zhu, Chenwei; Ma, Anxiang; Zhu, Jingbo; Xiao, Tong | 2026 | arXiv:2609.25770 [cs.CV], v1 22 Sep 2026; comment: "6 pages, 3 figures, and 6 tables. Preprint" | `lin2026readingright` |
| LensVLM | LensVLM: Selective Context Expansion for Compressed Visual Representation of Text | Xie, Roy; Friedman, Dan; Yu, Donghan; Pan, Bowen; Fifty, Christopher; Kim, Jang-Hyun; Du, Xianzhi; Gan, Zhe; Rathod, Vivek; Dhingra, Bhuwan | 2026 | arXiv:2605.07019 [cs.CV], v1 7 May 2026; comment: "Accepted to NeurIPS 2026" | `xie2026lensvlm` |
| ElecVQA-Bench pixel-budget audit | Beyond Balanced Accuracy: A Resolution and Parity-Controlled Benchmark for Vision-Language and Vision-Only Defect Assessment in UAV Power-Line Inspection | Zhang, Linghao; Xiang, Siyu; Kuang, Junwei; Yi, Peiyu | 2026 | arXiv:2609.27457 [cs.CV], v1 23 Sep 2026; comment: "46 pages, 8 figures" | `zhang2026elecvqa` |

## 3. The four papers read in full

### 3.1 VLMs Read Aloud (Shi, arXiv:2611.01517v2)

Read: all 23 pages of extracted text, including the supplement. Figures seen only through captions. Note the supplement says code and registrations "will be released with the paper", so I could not check the registrations.

- **Method.** Renders only the gold evidence at 10x density (characters about 6 px), then enlarges the same image to 9, 15 and 24 px by bicubic interpolation. Each reader answers from text, from the image, and from its own transcript. Read-use gap: G = Acc(transcript) - Acc(image) on recognized questions (p. 3, Eq. 1). Then proposes ALOUD (low-resolution attention scan, enlarge found pages to 15 px, transcribe, answer from words).
- **Models.** Qwen3-VL 2B/4B/8B/32B, InternVL3.5-8B, Gemma-3-12B, GLM-4.1V-9B-Thinking, Qwen3-VL-8B-Thinking, Glyph, Elorian Glimpse, GPT-6.1, Gemini 4 (p. 3).
- **Data.** Gold evidence from HotpotQA, MuSiQue, NQ, 100 questions each, in the LensVLM setting; MMLongBench-Doc whole documents (745 answerable, 576 text-evidence) (pp. 2, 4).
- **Main numbers.**
  - Qwen3-VL-8B at 15 px: recognizes 84%; 66.4 from own transcript, 70.8 from text, 55.7 from image; gap +10.7, CI [+4.7, +16.6] (p. 3, Tab. 5 p. 15).
  - Recognition 49% to 86% between 9 and 15 px, then saturates; answering from image 64% at 15 px, 81% at 24 px at 2.6x tokens; gap +3.8 at 24 px (p. 3).
  - Evidence type on MMLongBench-Doc for Qwen3-VL-8B: table gap -3.0 / +4.9 / +12.0 at 144 / 72 / 48 DPI, n=117 at 48 DPI (Tab. 7 p. 16).
  - Rank test: Spearman 0.86, one-sided exact p=0.005 (p. 4, Tab. 17 p. 20).
  - ALOUD at 5x: 27.7 to 54.8, against 58.5 from original text (Tab. 18 p. 20).
- **Cautions when citing.**
  - "Preregistered" is the author's word. The sign rules failed ("Addendum 1: ... sign rule 3 of 7 (fail)", p. 20), one addendum came after seeing an outcome (p. 14), and the registrations are not yet public. Only the ranking tests passed.
  - related_work.md guesses that their "use" may mean answering over long documents. **That guess is wrong for the 15/24 px result.** The protocol isolates reading from search: "we take only its gold supporting evidence" (p. 2). The difference from your pilot therefore needs another explanation. Candidates I have not tested: their multi-hop prose QA vs copying a cell, and enlarging rendered text vs your downsampling.

### 3.2 DocHallu (Hwang et al., Findings of ACL 2026)

Read: all 16 pages of extracted text; page 6 also viewed as an image. Figure 6 (patching curves) came through garbled, so I rely on the prose about it.

- **Method.** Original-damaged image pairs where the answer-relevant pixels are removed. Three human annotators per task, label requires unanimity; Fleiss kappa 0.55-0.74 (p. 24715). Layer-wise activation patching, then LLM post-hoc verifier filtering.
- **Models.** GPT-5.2, Claude Sonnet 4.5, Gemini 2.5-Flash, Gemma3-4B, Qwen3-VL-4B; patching only on the last two (pp. 24713, 24715).
- **Data.** 2,671 pairs: MER-Hallu 1,469, KIE-Hallu 283, DVQA-Hallu 919 at char / word / value granularity (318 / 276 / 325) (Tab. 1 p. 24713).
- **Main numbers.**
  - Hallucination rate on damaged inputs: KIE 0.315-0.496, DVQA 0.472-0.719 (Tab. 2, p. 24714).
  - Filtering: 36% mean reduction, up to 88% (abstract); the 0.88 case has coverage 0.41 (Tab. 3 p. 24717).
  - DVQA by granularity: char 42.5%, value 39.9%, word 17.6% (p. 24715).
- **Cautions when citing.**
  - **Internal inconsistency:** the prose gives numeric-answer rates of "84.8% in DVQA-Hallu and 61.0% in KIE-Hallu" (p. 24714), but Figure 4 shows number-only 88.8 and 70.2, text-only 84.5 and 66.7. Do not quote 84.8 / 61.0. In the figure the number-vs-text gap is about 3.5-5 points, and text+number mixes are lowest.
  - Their own causal caveat: "our activation patching analysis does not establish a single causal origin" (p. 24717). related_work.md's "causal analysis" should be softened to "activation patching".
  - Table 5 repeats the Gemini 2.5-Flash MER values (0.969 / 0.758) in the GPT-5.2 column, which differ from GPT-5.2's 0.888 / 0.579 in Table 4. I would avoid those GPT-5.2 MER EM values.

### 3.3 CADET (Hu et al., arXiv:2609.38851v1)

Read: all 41 pages of extracted text, including proofs. Appendix heatmaps skimmed only.

- **Method.** Tasks decomposed into unit tasks with a declared SCM. A ternary intervention injects a correct answer, an incorrect answer, or nothing into the prompt, with visual annotations rendered for correct spatial answers (p. 6). Capability metrics NC, IC, RC; N-Score and S-Score are lower bounds on probabilities of necessity and sufficiency via Frechet bounds, without assuming monotonicity (pp. 4-5, 14-16).
- **Models.** 12: Gemini 3.1 Pro, 3.5 Flash, 3 Flash, 3.1 Flash-Lite; GPT-5.4, GPT-5.4 Nano; Grok 4.3; Gemma 4 31B, 26B; Qwen3.5-397B-A17B, Qwen3.6 27B; Kimi K2.5 (p. 23).
- **Data.** 10 tasks, 1,144 instances, 46 unit tasks, over 33,000 unit-task questions (pp. 5, 23). Maze, dice, music notes, street view, dashcam, traffic causation. No financial tables.
- **Main numbers.**
  - NC by category 0.59 / 0.56 / 0.46 / 0.41; cognitive IC 0.72; cascading errors are 54% of cognitive failures (p. 7, Tab. 6 p. 24).
  - Single best prerequisite recovers 84% of the full gain (p. 8).
  - RC < NC for all 12 models in perception and spatial; 10 of 12 positive in cognitive (p. 7, Tab. 9 discussion p. 25).
  - Text-only correct injection vs text plus annotation: overall 0.71 vs 0.74 on two models (p. 29-30, D.4.1).
- **Cautions when citing.** The scores act on prompt-level prerequisite answers, so they measure "how a model's output responds to prerequisite information provided in its input, rather than how the prerequisite is represented inside the model" (p. 32). The authors also state the decomposition is not claimed complete (p. 31).

### 3.4 ConfBench (Roy et al., arXiv:2608.01792v1)

Read: all 26 pages of extracted text. Reliability diagrams seen only through captions.

- **Method.** 20 Augraphy degradation pipelines over invoices. Verbalized confidence (1S-TopK, top-4 guesses with probabilities) vs log-prob (first-token, mean-token, margin), across OCR only / OCR+Image / Image only. Metrics: WOA, F1, AUROC, ECE, Brier, ECARB.
- **Models.** Opus 4.6, Sonnet 4.5, Haiku 4.5, Kimi K2.5, Qwen 3 VL-235B, Qwen 3.6-27B, Gemma 3-12B. Log-prob only on Qwen 3.6-27B and Gemma 3-12B (p. 5). Abstract says "four proprietary and three open-weight".
- **Data.** 75 FCC invoices from RealKIE-FCC-Verified, 1,346 variants, 70K+ entity evaluations (pp. 3-4).
- **Main numbers (Tab. 3 p. 9, OCR+Image).** Opus 4.6 AUROC 0.84, ECE 0.05, ECARB 2.43. Qwen 3.6-27B verbalized 0.72 vs first-token 0.62, mean-token 0.59. Gemma 3-12B verbalized 0.58 vs first-token 0.64, mean-token 0.60.
- **Cautions when citing.**
  - **It does not support "token signal beats stated confidence" as a general claim.** The two models disagree: "the two models that support both families disagree on which strategy wins" (p. 7). The authors call it "a two-point pattern" (p. 10). Contrast 2607.22034 and 2607.24440 (abstract-level only), which report the opposite direction on small Qwen2-VL models.
  - Supported: first-token beats margin and mean among log-prob variants, on two models.
  - Input sizes differ between arms: 1200x1000 for 1S-TopK, `max_pixels = 640,000` for 1S-Logprob (pp. 19-20, A.6). The claim that this is "without materially affecting extraction quality" is asserted, with no test shown. This is a possible confound in verbalized-vs-logprob comparisons.

## 4. Overlap with our claims

These labels follow your request and differ from the (a)-(d) in related_work.md.

Claims: (a) upsampling a low-resolution image as an information-preserving control, or token budget vs information loss; (b) 2x2 of localization x legibility; (c) Shapley across stages over an image quality axis; (d) digit edits in table cells.

| paper | (a) | (b) | (c) | (d) |
|---|---|---|---|---|
| VLMs Read Aloud | **YES, strong overlap** | Partial: not factorial | No | No |
| DocHallu | No | No | No | Adjacent: erases, does not edit |
| CADET | No | No | No: necessity/sufficiency, no image quality axis | No |
| ConfBench | No (resize asserted harmless) | No | No | No |

The words "Shapley", "upsampl", "upscal", "downsampl" and "super-resol" have zero hits in all four PDFs' text. "digit" has hits only in ConfBench (none about edits), and I read all four in full, so this is not resting on term search alone. Hyphenated line breaks could hide a term from search, which is why I do not rely on it alone.

### 4.1 Evidence for (a) in VLMs Read Aloud

This is the most important finding of the review.

- "enlarge the same image to 9, 15 and 24 px, so every size carries the same information and differs only in visual tokens per character" (p. 2; repeated in A.1, p. 12).
- Figure 2a is an upsampling recovery curve for Qwen3-VL-8B: recognition goes 49% to 86% between 9 and 15 px, and image answers reach 81% at 24 px (p. 3). All of this comes from bicubic enlargement, with no new pixels.
- It also has the model contrast: "InternVL tiles every image into fixed 448 px blocks and Gemma resizes it to a fixed input, so for them enlarging changes little of what they see" (A.1, p. 12).
- "pixels close only at 24 px with 2.6x the image tokens" (Fig. 1 caption, p. 1) frames extra resolution as a token cost.
- ALOUD itself enlarges "by bicubic interpolation of the compressed pages themselves" (p. 5).

What it does **not** do, as far as I read:

- It never compares enlarged pixels with the sharp originals at the same token count, so it does not split token budget from lost pixels. Its evidence begins at the 6 px render.
- It does not state the "token budget, not lost pixels" conclusion. Its framing is read-use, and it attributes the remaining difficulty to use rather than to encoding.
- The enlargement sweep uses rendered prose evidence, not cell-level numerics in financial tables. It does not test Qwen2.5-VL or InternVL3.
- It reports no measure of how much pixels were destroyed vs merely under-tokenized.

**Plain assessment:** "upsampling alone recovers much accuracy on Qwen models" is not new as a bare observation, and the preprocessing reason why InternVL may differ is already stated in print. That statement is an explanation about InternVL3.5 and Gemma, and I did not extract a per-model accuracy comparison beyond Fig. 7, which I could not view. A defensible novelty would be the split itself (restoring sharp pixels vs upsampling at matched token budget) on real table cells.

### 4.2 Evidence for (b)

- VLMs Read Aloud, Tab. 20a (p. 21) and text (p. 22): "Pointing is not enough: drawing a box around the gold lines gives +1.2/+0.0 at 10x/15x and copying them in pixels +1.7/+1.3, while enlarging them gives +25.6/+20.9." These are separate arms, not a factorial: there is no box-plus-enlarged cell, and "enlarged" also moves and isolates the evidence, so localization and legibility are confounded. Enlargement does not restore sharp pixels.
- CADET renders bounding boxes when injecting a correct localization answer (p. 6) and tests text-only injection (D.4.1), but has no legibility factor.
- **Found but Not Read (2610.02880, not on the candidate list) is closer than either:** "Red boxes around the same blocks on the page images gain +0.005 / -0.030", whereas crops "at the reader's own page pixel density" gain +0.123 / +0.152, 87% / 92% of the text gain (section 3). Uses Qwen3-VL-2B, with Qwen2.5-VL-7B among six readers for the text gain. Same intervention as your red box, opposite framing: "The reader can read the evidence but cannot find it". It holds resolution fixed, so it is not a localization x legibility design, and it sits at 16 dense pages rather than a single table.

**Plain assessment:** the 2x2 still looks open as a design, but the red-box null is published twice. Cite both and do not present it as a surprise. The two published nulls come from 16-page contexts and from compressed pages at 10x/15x, not a single table, so your result need not match them; compare explicitly.

### 4.3 Evidence for (c)

None in the four. CADET is the closest in spirit: "contribution metrics (N-Score, S-Score), adapted from probabilities of causation" (abstract), interventions in prompt space, with no image-quality variable. Exact Shapley appears in 2605.05716, on LLM agent components (abstract only), which is a method precedent and not a VLM one. Status "open" holds, within the limits of this review.

### 4.4 Evidence for (d)

None edit a digit into another digit in a table cell.

- DocHallu is closest: char-level damage removes a digit in pixels, for example "70%" to "7 %" and "480.940.2080" to "480. 40.2080" (Figs. 8-9, p. 24720), on DocVQA tables and lists among others (p. 24713). The evidence is removed rather than replaced, so the correct behaviour is to abstain. It is not a counterfactual readout, and not financial data.
- VLMs Read Aloud's "counterfactual prefill" edits the thinking text, not pixels (p. 13). It has synthetic records, not tables, for no-prior testing (p. 16-17).
- From abstracts only: 2605.27750 (character perturbations in Greek OCR), 2607.21617 (scramble and substitution of words), 2609.13745 (one-tick bar-height counterfactuals). None is about table cells, according to the abstracts.

## 5. Fresh searches on upsampling, effective resolution and patch budget

Five searches via WebSearch (one more than the 2-4 you asked for), covering upsampling and OCR, max_pixels and patch budgets, bicubic enlargement, resizing on financial tables, and Qwen resize behaviour. Search output was a tool summary, so I treated it as leads only and opened sources myself where listed. I did not use its general conclusion that upsampling adds no information and is unlikely to help; that is reasoning, not evidence. Absence of a hit does not show absence of prior work: papers in anonymous review or posted in the last days can be missed.

| closest hit | what I actually read | relevance |
|---|---|---|
| VLMs Read Aloud 2611.01517 | full | Section 4.1. Most direct prior work. |
| **Res-Bench 2510.16926 (v3)** | abstract, section 4.2 incl. Table 3, model list | **Closest to "token budget vs lost pixels".** "disentangle two factors: reduced information content and shorter visual token length", via white padding (more tokens, same information) and DiffIR super-resolution. Table 3 as extracted: 224 px base 0.445, padded to 448 0.448, SR to 448 0.473; 448 px base 0.532, padded to 896 0.553, SR to 896 0.558. "they cannot fully match the quality of original high-resolution images". OCR is its most resolution-sensitive task, with a threshold effect (section 4.1). Includes Qwen2.5-VL and InternVL-2.5. Differences from you: general benchmark of six capability dimensions, learned SR rather than plain upsampling, padding as the token control. I did not establish whether Table 3 is per model or averaged. |
| Found but Not Read 2610.02880 | sections 1-4 | Section 4.2. Crop vs box. No resolution axis. |
| The Answer Knows Where to Look 2610.32664 | abstract, intro, Observation 3, passages | "Under such a budget the image is downsampled, and the text ... becomes illegible." Qwen3-VL-8B averages 48.2 at 98 tokens vs 82.2 at 392 on five text-rich benchmarks (Tab. 1). Information gap rho = native over budget pixels; zoom gain rises monotonically with rho (Obs. 3). Zoom re-encodes real pixels, so it does not test upsampling. |
| A Table Is Worth 64 Tokens 2608.26949 | abstract, method passage | Downscaled tables with Qwen token budgets on data including FinLongDocQA: "models cannot decipher fine-grained cell values". Explicitly "images already below the budget are not upscaled". Closest in domain, and the upsampling case is excluded by design. |
| Reading Right, Answering Wrong 2609.25770 | abstract, intro, related work, table caption | Small resizes crossing tiling or token-grid boundaries flip answers, and models can still read the answer when told what to read. Includes Qwen3-VL-8B and InternVL3-8B, but InternVL n is small (for example 30 on DocVQA). Relevant to why Qwen and InternVL respond differently. |
| ElecVQA-Bench 2609.27457 | abstract, audit passages | Not documents. A logged `max_pixels` audit: Qwen3-VL-8B varies 10.78 points, InternVL3.5-8B 6.16 points. Useful as a method precedent. |
| Reading, Not Thinking 2603.09095 | abstract, resolution passage | "most models maintain accuracy at 0.50x-1.0x resolution but degrade sharply below"; "InternVL3.5 ... is uniquely resolution-invariant". |
| ReadBench 2505.19091 | abstract, section 2.2.1 | **Limit:** the "negligible" resolution effect comes from a sweep "with Gemini 2.0 Flash", 72-300 PPI, on multi-page text. Do not cite as a general claim about open VLMs. |
| Last Visible Pixel 2606.07861 | abstract, model list | Perception saturates near 12 px on synthetic small targets. No upsampling test found by term search. |
| LensVLM 2605.07019 | abstract, passages | "characters shrink below the vision encoder's effective resolution". Text expansion for rendered text, image expansion for native documents. |
| FocusVTC 2609.36651, Gao et al. 2608.01848 | abstract only (not added to `refs.bib`) | Adaptive resolution in text compression; decoupled evaluation. |

Seen only as search snippets, **not opened, not verified, not in `refs.bib`:** VisionThink 2507.13348, Idefics3 report 2408.12637, POINTS 2409.04828, TEVA (ICCV 2025), VILA 2312.07533, BlueLM-V-3B 2411.10640, Steal the Patch Size 2607.00174, VLM-RobustBench (via 2609.25770's text), classical OCR super-resolution such as 2005.03780. The last group is older, non-VLM work on upsampling before OCR, worth checking if you want historic context. Also opened but judged off-topic: 2510.13826 and 2504.05299, which I fetched by mistaken id guess.

Also worth opening, seen only in the 2611.01517 reference list: "VISTA-Bench" 2602.04802, "Encoded but Unused" 2610.29241, "Transcribe, Then Reason" 2609.18958, "Loud and Clear" 2610.21318.

### Bottom line for the main claim

- **Taken:** bicubic enlargement recovers text reading on Qwen3-VL-8B, and the same paper says enlarging changes little of what InternVL3.5 and Gemma see (2611.01517). Token length and information content have been separated before by padding and SR (Res-Bench).
- **Still looks open, in what I found:** enlarging vs sharp restoration at matched token budget on the same image, measured per cell on real financial tables with counterfactual digit edits, across Qwen2.5-VL-7B, Qwen3-VL-8B and InternVL3-8B.
- **Recommended before submission:** read Res-Bench in full, and rerun this search close to the deadline, since 2611.01517 and 2610.02880 are days to weeks old.

## 6. Corrections suggested for related_work.md (not edited)

1. Remove the guess that the 15/24 px gap reflects long-document answering (section 3.1).
2. Change DocHallu's "causal analysis" to "activation patching", with the authors' caveat.
3. Add Found but Not Read to related_work.md's claim (c) (its list misses the red-box result), and add Res-Bench as prior work for the upsampling claim.
4. Risk 1 says the token signal beats stated confidence: ConfBench gives a counterexample for Qwen 3.6-27B.
5. ReadBench's resolution finding is Gemini-only.
