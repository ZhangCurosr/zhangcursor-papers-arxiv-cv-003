# OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning

Zhongyu Yang<sup>1♥</sup> Jiale Tao<sup>1♥†</sup> Ruitao Chen<sup>1♥</sup> Zuhao Yang<sup>2</sup> Yingfang Yuan<sup>3</sup> Xueliang Zhao<sup>1</sup> Auden<sup>1</sup> Kai Wang<sup>1</sup> Shuai Shao<sup>1</sup> Biao Wang<sup>1✉</sup> Steve Yves<sup>1✉</sup> Qinglin Lu<sup>1</sup> <sup>1</sup>Hunyuan, Tencent <sup>2</sup>Nanyang Technological University <sup>3</sup>Northumbria University Project Website: https://01yzzyu.github.io/OmniCapBench/

![](images/57fa70069294b9a559638eb936b0ee0f1c5114ae1648f57d8a92eb10973602d1.jpg)  
Figure 1: Comparison of existing evaluation frameworks and the deep-structured evaluation framework. Both (A) Holistic Evaluation and (B) Probe-Based Evaluation derive scores from predicted unstructured, free-form text captions, but introduce distinct limitations. The former uses a global LLM-judge approach that treats the caption as a whole, resulting in an opaque score that masks local errors and lacks granularity, whereas the latter adopts a QA-probe approach to detect local errors but sacrifices global coverage and introduces instability across different LLMs. (C) OmniCapBench (Ours) proposes a deep-structured evaluation framework. The caption is first natively represented as atomic, verifiable audio-visual units. Deterministic rules then verify structural and temporal relationships and dispatch these aligned units to localized, reliable LLMs to perform fine-grained semantic checks. This design preserves comprehensive coverage and achieves fine-grained error localization while eliminating the uncertainty associated with global LLM-based judges.

## Abstract

Multimodal large language models (MLLMs) are rapidly evolving toward continuous audio–visual reasoning, creating an urgent need for evaluations that expose their capability limits. Audio–visual captioning is an ideal diagnostic task, yet current benchmarks face a coupled trade-off: whole-caption scores provide coverage without localization, local probes provide localization without coverage, and unconstrained LLM judges introduce instability. We introduce OmniCapBench (Omni-Video Caption Benchmark), a benchmark that reframes audio–visual caption evaluation as a deep-structured diagnostic framework. OmniCapBench shifts the prediction target from free-form text to sets of atomic, verifiable evaluation units across three tracks: entity references, visual shots, and audio events, enabling reliable scoring with deterministic constraint checks and localized LLM-based semantic comparisons. With 786 densely annotated videos, OmniCapBench effectively distinguishes MLLM perception errors, including temporal grounding failures, identity drift, cross-modal misalignment, and hallucinated descriptions. Evaluating frontier MLLMs reveals strong local perception but weak long-horizon audio–visual reasoning, particularly in identity drift and cross-modal misalignment, providing a fine-grained roadmap for omnimodal development.

## 1 Introduction

Multimodal large language models (MLLMs) are rapidly evolving to reason over continuous audio– visual streams [60, 51, 49, 11, 35]. This shift demands a corresponding evolution in evaluation. For representative tasks like audio–visual captioning [5, 62, 16], assessment must transcend coarse textquality rankings. In particular, explicitly diagnosing specific capabilities, such as identity tracking, temporal grounding, and audio-visual association, is essential to expose fine-grained weaknesses and provide actionable insights for model improvement.

Existing caption benchmarks rarely co-design two tightly coupled components that enable finegrained and reliable evaluation: the evaluation unit and the scoring operator. The unit determines the granularity of evaluation, whereas the scoring operator governs reliability over units.

For the evaluation unit, existing protocols are largely built around the predicted free-form caption, but differ in granularity. Whole-caption protocols treat the predicted–reference caption pair as a single unit (Figure 1 (A)) [62, 5], providing broad coverage but compressing fine-grained information into a global scalar that often fails to reveal local errors. Probe-based protocols instead define units as response–ground-truth pairs for question-answering (QA) probes or fact-related cloze tasks (Figure 1 (B)) [44, 50, 29, 16], thereby accessing localized information within the predicted caption. This enables more localized and specific scoring, but the sparse units lack coverage. For the scoring operator, most existing evaluation approaches employ large language models (LLMs) to assess unstructured text captions, yet systematic efforts to obtain stable and reliable scores remain limited. Specifically, LLMs are required to reason over LLM-generated dense, monolithic text under different evaluation protocols to compute global or local scores, introducing an inverse-engineering process that poses a substantial challenge for reliable judging. In detail, global scoring struggles to align fine-grained details within dense text, whereas local scoring is constrained by the need for accurate information extraction and matching. For instance, QA probes implicitly perform extraction and matching and are thus prone to distraction, which can lead to incorrect conclusions even when the caption is correct (Table 5), while event-list scoring explicitly requires post hoc extraction and matching, which can also introduce reliability issues [16].

Consequently, the free-form, monolithic representation of model-generated captions fundamentally obstructs the joint design of the evaluation unit and the scoring operator. First, evaluation units that simultaneously provide global coverage and fine-grained local detail are not explicitly encoded. Second, the scoring operator must perform post hoc extraction prior to scoring, which introduces instability. Together, these limitations lead to additional errors and unreliable evaluation.

To close this gap, we propose OmniCapBench (Omni-Video Caption Benchmark), a benchmark that redesigns fine-grained audio–visual captioning around deep-structured evaluation units (Figure 1 (C)). More specifically, rather than imposing a superficial format on global text, we decompose the evaluated object into native atomic units. This deep structure breaks the task into discrete, localizable targets, preserving comprehensive video coverage while eliminating the ambiguity of monolithic outputs. Crucially, it enables a reliable scoring operator: deterministic rules rigorously verify structural relationships, while LLMs are restricted to bounded semantic comparisons, avoiding the compounding uncertainty of unconstrained text-level judgments.

In practice, OmniCapBench operationalizes this structure via three native evaluation tracks. The Reference track catalogs persistent scenes and subjects, while the Event track captures timestamped audio occurrences. Crucially, the Shot track segments visual content, using explicit cross-links to temporally ground both references and events. This track design drives a two-stage scoring pipeline. First, deterministic rules and a bounded LLM matcher verify structural integrity, including valid

Table 1: Evaluation paradigm comparison. We compare whether existing protocols natively support the structural dimensions of omnimodal diagnosis. Identity Tracking explicitly evaluates entity permanence across discontinuous shots (Visual). Temporal Grounding enforces precise continuous time boundaries for actions and sounds (Audio/Visual). Audio-Visual Association evaluates whether audio events are correctly attached to concurrent visual shots (Audio-Visual). Diagnostic Traceability indicates whether the evaluation isolates explicit structural breakdowns from descriptive hallucinations. ✓: natively supported as a verifiable unit; △: implicitly judged; ✗: not supported.
<table><tr><td>Benchmark</td><td>Evaluated Unit</td><td>Scoring Operator</td><td>Identity Tracking (Visual)</td><td>Temporal Grounding (Audio/Visual)</td><td>Audio-Visual Association (Audio-Visual)</td><td>Diagnostic Traceability</td></tr><tr><td colspan="7">Whole-Caption Paradigm</td></tr><tr><td>AuroraCap [5]</td><td>Caption</td><td>Global LLM</td><td></td><td></td><td></td><td>x</td></tr><tr><td>VCapsBench [62]</td><td>Caption</td><td>Global LLM</td><td>××××</td><td></td><td></td><td></td></tr><tr><td>UGC-VideoCap [43]</td><td>Caption</td><td>Global LLM</td><td></td><td>×××</td><td>××××</td><td>×××</td></tr><tr><td>video-SALMONN 2 [34]</td><td>Caption</td><td>Global LLM</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7">Probe-Based Paradigm</td></tr><tr><td>Omni-Cloze [29]</td><td>Cloze QA</td><td>LLM Scorer</td><td>△</td><td>△</td><td>△</td><td>△</td></tr><tr><td></td><td></td><td></td><td></td><td>Parse-then-Score Paradigm</td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>LongVALE [16] TimeChat [59]</td><td>Event text Timed script</td><td>Parser + LLM Parser + LLM</td><td>x×</td><td></td><td>××</td><td>X</td></tr><tr><td>OmniScript [31]</td><td>Script text</td><td>Parser + LLM</td><td></td><td></td><td>△</td><td>△ △</td></tr><tr><td colspan="7"></td></tr><tr><td></td><td></td><td></td><td></td><td>Native Atomic Paradigm</td><td></td><td></td></tr><tr><td>OmniCapBench (Ours)</td><td>Atomic units</td><td>Rule + Local LLM</td><td>T</td><td></td><td>V</td><td>V</td></tr></table>

IDs, temporal bounds, and cross-links. Second, localized LLMs exclusively assess the semantic equivalence of these isolated fields. By enforcing this strict scoring contract, OmniCapBench elevates coarse text evaluation into a fine-grained diagnostic tool. Table 1 examines whether existing benchmarks natively support such verifiable atomic units.

To validate OmniCapBench, our experiments proceed in three parts. First, we establish a capability benchmark for state-of-the-art omnimodal models, decoupling evaluation into deterministic rules (Table 3) and bounded semantic checks (Table 4) to pinpoint localized bottlenecks. Second, we audit the holistic evaluation paradigm across models, metrics, and generation formats (Table B.7), demonstrating that global text-level scores mask local errors and reward structurally flawed outputs. Finally, by contrasting weak and strong LLM judges (Table 5), we show that global, unlocalized LLM scoring induces unstable conclusions, confirming the necessity of our decoupled framework.

Our contributions are three-fold: (i) We propose a Deep-Structured Evaluation Paradigm that decomposes audio–visual captioning into atomic, verifiable units, explicitly isolating semantic content, temporal grounding, identity tracking, and cross-modal association. (ii) We instantiate this paradigm into the OmniCapBench Benchmark, a rigorous testbed comprising 786 densely annotated videos. Its native Reference, Shot, and Event tracks yield 5,818 entities, 6,537 audio events, and 11,419 visual shots to evaluate fine-grained omnimodal comprehension. (iii) We conduct an Empirical Audit of Holistic Scoring, exposing its vulnerability to judge instability and tendency to mask localized errors, thereby demonstrating the necessity of deep-structured metrics.

## 2 Related Work

Omnimodal large language models. Omnimodal large language models are rapidly advancing toward reasoning over continuous audio–visual streams [51, 1, 48, 49, 32, 24, 28, 64, 52, 33, 13, 34, 6, 10, 15, 30, 53]. Accelerated by reinforcement learning and multi-agent systems [65, 63, 46, 9, 55, 57, 42, 61, 56, 58], audio–visual captioning has become a foundational task requiring the synthesis of visual, motion, and acoustic signals. While broad video benchmarks [12, 21, 40, 39, 66, 20, 17, 54, 47, 41, 19] track general progress, they often reduce complex behaviors to coarse text-level scores, lacking the diagnostic resolution needed to pinpoint localized audio–visual failures.

Audio–visual Captioning Evaluation. Moving beyond legacy metrics [37, 2, 26, 18], recent protocols explore different evaluation units. Holistic evaluation scores full captions across dimensions [62, 5, 27, 7, 43, 23], yet treating text as a monolithic unit masks specific temporal or relational errors. QA and temporal probes [44, 50, 29, 4, 25, 38] localize errors via explicit questions but suffer from sparse coverage. Other methods parse free-form text into script items [16, 59, 31, 8, 14], treating structure as a fragile post-hoc extraction layer. Crucially, when an LLM acts as a global judge—simultaneously parsing unstructured text, resolving references, and tracking time—these entangled tasks conflate factual hallucinations with stylistic variations, leading to unreliable scoring.

In short, existing methods face a strict tradeoff: they either sacrifice coverage for localization, or rely on opaque LLM judgments that obscure fine-grained errors. OmniCapBench resolves this tension through deep-structured evaluation. By mandating models to output native, atomic audio–visual units, our framework achieves both comprehensive coverage and exact localizability. Furthermore, we decouple the scoring process: deterministic probes verify structural relationships, while reliable localized LLMs are strictly bounded to local semantic checks. This eliminates the compounding uncertainty of global LLM judges, yielding a highly reliable diagnostic signal.

![](images/bbe5fbf3883160371d7a64914e914cf7e38d7f156438cf0b973b8a712323c81e.jpg)  
Figure 2: OmniCapBench data construction pipeline. The construction proceeds in three stages. Stage I filters raw videos to isolate inputs with rich audio–visual complexity. Stage II generates atomic units (References → Events → Shots) through an iterative multi-model loop, enforcing structural integrity before semantic refinement. Stage III applies traceable human audits to finalize the rigorously verified audio–visual reference system.

## 3 OmniCapBench Benchmark

To establish a fine-grained, reliable evaluation standard for omnimodal foundation models, we introduce the OmniCapBench benchmark. We first present the data construction pipeline (Section 3.1), which converts raw videos into well-defined, verifiable atomic units. We then outline the evaluation goals and diagnostic task suite built on this structured representation (Section 3.2).

## 3.1 Data Construction and Native Track Generation

From an evaluation standpoint, rigorously verifying an AI’s audio–visual understanding requires isolating what happened from when and where it occurred. To achieve this fine-grained diagnostic evaluation, we inherently require a structured representation that systematically decouples distinct video elements, such as entities, auditory events, and visual states. The Multi-Stream Scene Script (MTSS) [36] naturally provides this factorized foundation, making it an ideal fit for our evaluation objectives. Therefore, we build OmniCapBench upon MTSS, adopting its core decoupled design to independently track these information streams. To fully support precise benchmarking, we further optimize its coarse-grained limitations: specifically, we decompose visual shots into finer subshots and replace discrete point timestamps with continuous time ranges.

For a given video V, we formalize its content into a structured representation $S ^ { \star } ( V ) = ( \mathcal { R } , \mathcal { E } , \mathcal { H } )$ To guarantee structural integrity, this representation is built across three interdependent tracks:

• References (R): This track establishes persistent identities for key entities, including people, objects, and scenes, by assigning unique identifiers that enable consistent tracking of each entity throughout the video.

• Events (E): This track isolates auditory events and spoken dialogue together with their precise temporal boundaries, for example, extracting a continuous dog bark from 10.5s to 12.0s.

• Shots (H): This track segments the visual timeline into discrete camera shots and serves as the unifying structure by anchoring previously defined entities and audio events to specific frames, for example, linking the barking sound to the shot that shows the dog.

This atomic schema separates semantic content from the complex structure of audio–visual streams through explicit localization. In this way, the evaluation can precisely verify whether a predicted event occurs at the correct time and is linked to the appropriate visual entities, while enabling a localized LLM scorer to focus only on structurally aligned content for semantic verification.

The construction of this reference system proceeds in three stages (Figure 2).

<table><tr><td>Statistic</td><td>Avg./Pct.</td><td>Total</td></tr><tr><td>Videos (&lt; 1 min)</td><td>61.32%</td><td>482</td></tr><tr><td>Videos (1–3 min)</td><td>29.01%</td><td>228</td></tr><tr><td>Videos (3–5 min)</td><td>9.67%</td><td>76</td></tr><tr><td>Duration (hours)</td><td></td><td>12.8</td></tr><tr><td>Categories (Level-1) Categories (Level-2)</td><td></td><td>20 125</td></tr><tr><td></td><td></td><td></td></tr><tr><td>References Shots</td><td>7.40</td><td>5,818</td></tr><tr><td>Subshots</td><td>14.53</td><td>11,419</td></tr><tr><td>Events</td><td>49.82 8.32</td><td>39,160</td></tr><tr><td></td><td></td><td>6,537</td></tr><tr><td>– Dialogue</td><td>6.83</td><td>5,370</td></tr></table>

Table 2: Key statistics of Omni-CapBench, comprising 786 videos. $\bf \ddot { \tau } \mathrm { A v } \bar { g } . / \mathrm { P c t . } ^ { \bf , \vec { \tau } }$ denotes either the average or the percentage of total videos.

![](images/0e4781be12ee4c247c7f8762ce14845c7b9a5f1a3f9d6737b5151f401683f541.jpg)  
Figure 3: OmniCapBench category distribution. Inner and outer rings represent level-1 categories and level-2 subcategories, respectively. Segment area is proportional to video count, illustrating the benchmark coverage.

Stage I: Candidate Curation. We source candidate videos from diverse open datasets and public platforms. Following initial preprocessing (e.g., metadata extraction, shot detection, and categorization), we apply rigorous filtering based on audio presence, duration, quality, and bucket-specific density (Appendix A.1.1). This curation ensures all retained videos exhibit sufficient audio–visual complexity to warrant our multi-unit structural evaluation, directly supporting our goal of probing deep omnimodal reasoning rather than isolated single-event recognition.

Stage II: Native Track Construction. As defined previously, this stage iteratively builds the atomic units in the required dependency order $( \mathcal { R } \to \mathcal { E } \to \mathcal { H } )$ .

To autonomously guarantee structural integrity during data generation, we execute an iterative Local Refinement loop: Generate → Evaluate & Validate → Local Refine → Select. First, a generator proposes initial candidate tracks for the video. Next, an evaluator ensemble checks the factual accuracy of these candidates, while a programmatic validator rigorously enforces schema constraints, triggering automatic regeneration or discarding of invalid outputs. Crucially, once the foundational topology is validated, a local refinement step enriches the textual descriptions. During this process, the underlying structure, such as entity identifiers, temporal boundaries, and cross-links, remains strictly invariant. This constrained refinement explicitly prevents linguistic enhancements from introducing new hallucinations or corrupting the already verified topology. Finally, a selector model chooses the highest-quality refined version to form the final canonical tracks. Note that the frames used in this stage carry a red timestamp overlay purely as an annotation aid; evaluated models always receive the original frames without it (Appendix A.1.1).

Stage III: Quality Audits. Because the native reference system $S ^ { \star } ( V )$ directly dictates the evaluation scoring contract, its data integrity must be audited before scoring claims are reported. In this final stage, programmatic validators inspect schema validity, timestamp ranges, and cross-links, while targeted human audits inspect boundary and semantic disagreements. The finalized units provide each video with a reliable reference system $S ^ { \star } ( V ) = ( \mathcal { R } , \bar { \mathcal { E } } , \mathcal { H } )$ for model evaluation.

Dataset Statistics. Table 2 reports the verified dataset size and per-video averages, while Figure 3 summarizes the category coverage.

## 3.2 Evaluation Goals and Task Suite

The overarching goal of the OmniCapBench benchmark is to evaluate omnimodal foundation models with high granularity and reliability. Instead of using an unconstrained text-centric generation process that inherently conflates a model’s true audio–visual comprehension with its language style, OmniCapBench derives its diagnostic tasks directly from the rigorous native reference system $S ^ { \star } ( V )$ . Appendix A.1.1 details the exact input–output formats.

Task 1: Native Unit Generation. In this primary task, a model takes a video V and directly outputs the three prediction tracks (References, Shots, and Events) as a structured JSON object. Instead of writing a free-form paragraph, the model must explicitly construct the set of atomic units ${ \hat { S } } ( V )$ . This forces the model to expose its internal reasoning in a highly structured format, directly aligning with our goal of fine-grained evaluation and removing the ambiguity of free-form text. We use this task to assess state-of-the-art models and quantify their structural and semantic accuracy.

Table 3: Rule-based structural evaluation. Metrics quantify structural compliance across Visual, Audio, and Audio-Visual dimensions. SGC is an independent schema prerequisite. Scores are macro-averaged across videos. Best and second-best results are highlighted.
<table><tr><td rowspan="2">Model</td><td rowspan="2">SGC</td><td colspan="7"></td><td colspan="2">Audio</td><td colspan="2">Audio-Visual</td></tr><tr><td colspan="10">RefUse CCC Ref Subject F1 Ref Scene F1 Shot F1 Shot tIoU Subshot F1 Subshot tIoU Event F1 Event tIoU Speaker F1 EVSA F1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Proprietary Models</td><td>70.74</td><td></td><td>48.76</td><td>63.32</td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1-Pro [15]</td><td>97.36</td><td>70.39</td><td>34.43 37.81</td><td>78.60 80.70</td><td>82.68 83.65</td><td>76.24 70.93</td><td>65.27</td><td></td><td></td><td>71.92</td><td>91.18</td><td>51.46</td></tr><tr><td>Gemini 2.5-Pro [10] Qwen3.5-Omni-Plus [32]</td><td>96.73 96.09</td><td>70.75 66.92</td><td>29.55</td><td>73.57</td><td>79.97</td><td>72.58</td><td>66.69</td><td>49.18</td><td>58.52 53.92</td><td>69.35 67.77</td><td>89.72 91.20</td><td>43.74 40.20</td></tr><tr><td>Qwen3.5-Omni-Flash [32]</td><td>95.58</td><td>63.10</td><td>24.31</td><td></td><td>71.25 66.07 65.65</td><td>72.35 68.99</td><td>65.69 58.94</td><td>48.29 46.52</td><td>50.68</td><td>69.34</td><td>89.91</td><td>35.58</td></tr><tr><td>Seed2.0 [3]</td><td>87.68</td><td>64.69</td><td>31.88</td><td>72.27</td><td>75.13 72.62</td><td>72.56</td><td>68.82</td><td>47.94</td><td>4.54</td><td>62.78</td><td>91.80</td><td>3.52</td></tr><tr><td>MiMo-2.5 [45]</td><td>96.58</td><td>61.99</td><td>23.91</td><td>65.29</td><td>66.58</td><td>65.53</td><td>60.37</td><td>46.51</td><td>43.82</td><td>61.82</td><td>86.62</td><td>31.69</td></tr><tr><td></td><td></td><td></td><td></td><td>61.19</td><td>69.27</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Open-source Omnimodal Models</td><td colspan="3"></td></tr><tr><td>Qwen3-Omni-Instruct [49]</td><td>89.02</td><td>40.96</td><td>9.85 11.51</td><td>54.84 54.36</td><td>64.55 57.08</td><td>49.98 67.92 52.29</td><td>29.51</td><td>47.95</td><td>41.70</td><td>40.99</td><td>88.35</td><td>13.33</td></tr><tr><td>Qwen3-Omni-Captioner [49] 90.77</td><td>72.13</td><td>44.96 21.99</td><td></td><td></td><td></td><td>70.80 58.45</td><td>39.51 9.90</td><td>49.69 42.99</td><td>41.31 6.41</td><td>52.60 27.44</td><td>88.68 80.07</td><td>17.42</td></tr><tr><td>MiniCPM-o-2.6 [60]</td><td></td><td></td><td>4.29</td><td>24.36</td><td>3.38</td><td>17.28</td><td></td><td></td><td></td><td></td><td></td><td>2.46</td></tr></table>

Task 2: Structural Verification and Matching. Before semantic scoring, we first check whether each predicted unit is structurally valid. A valid unit must use defined identifiers, have a well-formed time span, and link only to existing and temporally compatible references, shots, or events. Predictions that violate these rules, such as undefined IDs, reversed start–end times, or impossible event–shot links, are removed by deterministic checks. We then match the remaining units to the ground truth. Shots and events are matched by temporal intersection-over-union (tIoU), while identity references and visual subshots are matched with a bounded LLM-assisted matcher that only compares local candidate pairs. This design separates structure from semantics: invalid or unmatched units are counted as structural errors, and the LLM scorer is used only on aligned unit pairs. Thus, errors such as invalid IDs, weak temporal overlap, or wrong event–shot links become traceable structural deficits rather than hidden components of a single caption-level score.

Task 3: Localized Semantic Scoring. In the final stage, we dispatch the matched unit pairs to an LLM judge. With structural complexities (e.g., timing and identity tracking) already resolved in Task 2, the LLM is strictly bounded to perform local semantic equivalence comparisons on isolated content fields. This “structure-first, semantics-second” approach isolates genuine descriptive fidelity from structural noise. Crucially, by decomposing the evaluation into deep-structured atomic units, the assessment becomes significantly simpler. This substantially reduces the dependency on advanced LLM reasoning capabilities, ensuring highly objective and reliable results.

Bidirectional Evaluation: Precision vs. Recall. Because open-ended video generation lacks a perfect one-to-one mapping between predictions and human annotations (for instance, a model might decompose a visual action into three dense subshots while the ground truth summarizes it in one), evaluating from a single direction is inherently biased. For open-vocabulary and continuous fields like References (Subject and Scene) and Subshots, OmniCapBench enforces a bidirectional matching and scoring paradigm. Specifically, we execute the matching and scoring processes from two distinct perspectives with different focuses: (1) The Recall Perspective (Ground Truth → Prediction) uses the ground truth as the subject to check if the prediction covers it, strictly penalizing omissions. (2) The Precision Perspective (Prediction → Ground Truth) uses the prediction as the subject to check if the ground truth supports it, strictly penalizing hallucinations. The final F1 scores for categories like Subshot F1 are directly derived by computing the harmonic mean of these two distinct directional metrics. In our reporting, Ref Subject encompasses both persons and objects, while Ref Scene corresponds to scene references. This prevents models from gaming the evaluation through overly terse responses (which would fail recall) or excessively verbose, speculative dumps (which would fail precision). Appendix A.2.2 provides extensive details on this mechanism.

Structured Diagnostic Metrics. OmniCapBench provides a suite of deterministic metrics designed to expose specific structural vulnerabilities, moving beyond holistic text similarity. Exact formulations are detailed in Appendix A.2.2. In the Visual domain, Reference Subject/Scene F1 evaluate basic entity detection, while Reference Use (RefUse) and Cross-Shot Coreference Consistency (CCC) check identity permanence across video cuts. For temporal grounding, Shot/Subshot F1 and tIoU measure the recovery and boundary alignment of visual states. In the Audio domain, Event F1 and tIoU verify the recovery and precise temporal alignment of sound events. In the Audio-Visual domain, Event-Shot Association F1 (EVSA) measures whether audio events are correctly attached to concurrent visual shots, while Speaker F1 verifies that dialogue is assigned to the correct visual speaker. Finally, Localized Semantic metrics (bounded Precision and Recall) quantify descriptive fidelity strictly on structurally valid units. This isolates factual comprehension from structural noise, preventing models from masking hallucinations or omissions behind fluent prose.

Table 4: Localized semantic evaluation. Semantic fidelity is assessed exclusively on structurally aligned units. Bidirectional scoring isolates distinct failure modes: Recall penalizes factual omissions, while Precision penalizes hallucinations. Best and second-best results are highlighted.
<table><tr><td rowspan="3">Model</td><td colspan="6">Visual</td><td colspan="2">Audio</td></tr><tr><td rowspan="2">Shot</td><td colspan="2">Subject</td><td colspan="2">Scene</td><td colspan="2">Subshot</td><td rowspan="2">Dialogue Non-dialogue</td></tr><tr><td>Recall</td><td>Precision</td><td>Recall Precision</td><td></td><td>Recall Precision</td><td></td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>Gemini 3.1-Pro [15]</td><td>59.05</td><td>43.78</td><td>67.07</td><td>51.78 56.80</td><td>75.62 53.29</td><td>70.92</td><td>91.18</td><td>41.49</td></tr><tr><td>Gemini 2.5-Pro [10]</td><td>55.58</td><td>51.53</td><td>57.08</td><td>65.12 69.80</td><td>53.05 50.22</td><td>71.21 70.66</td><td>89.72 91.20</td><td>37.38 36.82</td></tr><tr><td>Qwen3.5-Omni-Plus [32] Qwen3.5-Omni-Flash [32]</td><td>54.17</td><td>45.41</td><td>62.84</td><td>50.75</td><td>44.08</td><td>67.27</td><td>89.91</td><td>28.12</td></tr><tr><td>Seed2.0 [3]</td><td>47.66 54.77</td><td>43.87 42.05</td><td>65.59</td><td>44.41</td><td>71.26</td><td>72.88</td><td>91.80</td><td>30.20</td></tr><tr><td>MiMo-2.5 [45]</td><td>48.32</td><td>46.70</td><td>63.65 58.63</td><td>50.59 46.72</td><td>71.42 51.15 65.47</td><td>67.27</td><td>86.62</td><td>30.85</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>44.93</td><td></td><td></td><td></td></tr><tr><td colspan="9">Open-source Omnimodal Models</td></tr><tr><td>Qwen3-Omni-Instruct [49] Qwen3-Omni-Captioner [49]</td><td>34.63 36.08</td><td>40.31</td><td>62.02</td><td>36.07</td><td>70.73 54.53</td><td>29.42 34.01</td><td>57.84 88.35</td><td></td></tr><tr><td></td><td>28.53</td><td>42.00</td><td>50.96</td><td>38.07</td><td></td><td>46.05 55.88</td><td>88.68 80.07</td><td>19.65</td></tr><tr><td>MiniCPM-o-2.6 [60]</td><td></td><td>25.92</td><td>63.87</td><td>20.39</td><td>76.32</td><td>27.62</td><td></td><td>4.84</td></tr></table>

## 4 Experiments

## 4.1 Experimental Setup and Metrics

Model. We evaluate representative Omni MLLMs for OmniCapBench across three categories:

• Proprietary models: Gemini 3.1/2.5-Pro [15, 10], Qwen3.5-Omni-Plus/Flash [32], Seed2.0 [3], MiMo-2.5 [45]

• Open-source models: Qwen3-Omni-Instruct/Captioner [49], and MiniCPM-o-2.6 [60]).

• Specialized video-captioning models: ASID-Caption [25] and AVoCaDO [6].

However, we exclude specialized video-captioning models from the main evaluation, as their supervised fine-tuning toward free-form text leads to poor instruction following on structured generation tasks; see Appendix B.5.

Metrics and Results Format. Following Section 3.2, our deep-structured evaluation reports finegrained metrics across Visual (RefUse, CCC, Ref Subject/Scene F1, Shot/Subshot F1 and tIoU), Audio (Event F1 and tIoU), and Audio-Visual domains (Speaker F1 and EVSA F1). Table 3 presents the rule-based match scores, while Table 4 separately reports semantic equivalence scores computed only on structurally valid and aligned units.

## 4.2 Benchmarking Omnimodal Model Capabilities

Table 3 reports the performance of evaluated models across the rule-based metric families. By shifting from global scalar text scores to deep-deep-structured evaluation units, we can pinpoint where models fail in fine-grained audio–visual comprehension. Table 4 reports the corresponding bounded LLM-as-judge semantic scores on aligned units.

Diagnosing capabilities through deep structure. Our deep-structured evaluation exposes critical capability deficits that holistic text scores typically obscure. By decoupling evaluation into deterministic structural rules (Table 3) and bounded semantic checks (Table 4), we pinpoint three fundamental bottlenecks in current omnimodal models.

First, in the visual domain, OmniCapBench shifts the focus from isolated entity recognition to continuous identity tracking. While frontier models like Gemini 2.5-Pro exhibit strong basic perception (84.80% Ref Subject F1 on < 1 min videos, Appendix Table B.5), their ability to track these identities

Failure Mode Composition across Models  
![](images/0b960970460a116be4c508c3b963089b9bde13b856eb3f2b69c35626f724a456.jpg)

![](images/3ad155be44213f3091c1fd5d493d491968065c1d96293d415992b5467e0ddf16.jpg)  
(a) The illusion of global scores. Text met- (b) Failure composition. Distribution of structural errors, norrics reward fluent but flawed outputs; struc- malized by each model’s total failures. tured metrics expose them.

Figure 4: Evaluation paradigms and structural failures. (a) Global text metrics make distinct models appear similar, whereas structured constraints better match human audit. (b) Error composition shows different bottlenecks: frontier models concentrate failures in fine-grained hallucination and audio-visual alignment, while open-source models exhibit broader degradation.

across cuts degrades sharply. This is captured by the Cross-Shot Coreference Consistency (CCC) score, which drops to 37.81% for Gemini 2.5-Pro and collapses below 12% for open-source models. The insight is clear: current architectures struggle with long-term visual object permanence, a deficit hidden by traditional metrics that only check if an object was mentioned once.

Second, in the audio and cross-modal domains, our structural metrics reveal a persistent modality disconnect. While models excel at speech transcription (Dialogue semantics ≥ 80%), their auditory reasoning for environmental sounds remains weak. More critically, genuine omnimodal comprehension requires structurally binding these sounds to concurrent visual sources. However, Gemini 3.1-Pro’s performance drops from 63.32% (Event F1) to 51.46% on Event-Shot Association (EVSA F1), with open-source models failing to surpass 20%. Seed2.0 fails outright here, emitting almost no audio-event units (4.54% Event F1, 3.52% EVSA F1) despite top-ranked speaker attribution (91.80%). This suggests that models still largely process audio and visual streams as unaligned pathways rather than a unified structural graph.

Finally, by restricting semantic evaluation exclusively to structurally valid units, we isolate true descriptive fidelity from structural noise. This localized audit uncovers a stark precision-recall trade-off that holistic scoring masks behind fluent prose. For instance, Gemini 3.1-Pro adopts a conservative strategy (67.07% precision, 43.78% recall on subjects), omitting details to avoid errors, whereas other models inflate recall by hallucinating unverified attributes. MiniCPM-o-2.6 is the extreme case: it tops Scene precision (76.32%) while recalling only 20.39% of scene content. By enforcing strict structural prerequisites, OmniCapBench prevents models from gaming the evaluation, providing a clear roadmap for grounded omnimodal generation.

Error analysis. Figure 4b decomposes model failures into strict structural violations. We derive this composition by converting the performance on each rule-based structural metric into an absolute capability deficit (1 − score) and normalizing these deficits to represent 100% of each model’s failure distribution. This detailed breakdown reveals insights invisible to holistic text metrics. First, format errors are negligible (1.3% for Gemini 3.1-Pro), indicating that schema adherence is a solved prerequisite. Second, we observe distinct failure signatures across model capabilities. For frontier models like Gemini 3.1-Pro, errors are highly concentrated in deep cross-modal links and finegrained action details: action hallucination accounts for a massive 28.5% of its failures, followed by A-V misalignment (23.7%) and audio hallucination (17.9%). In contrast, open-source models suffer a systemic degradation, with errors distributed uniformly across both basic perception and complex audio-visual binding. This validates our core motivation: the true bottleneck in omnimodal video understanding lies not merely in describing isolated entities, but in avoiding deep-structured hallucinations like fabricating action details or assigning sounds to the wrong visual shot.

## 4.3 Auditing the Legacy Evaluation Paradigm

A fundamental motivation of OmniCapBench is that existing holistic evaluations conflate linguistic fluency with genuine multimodal comprehension. This section tests whether global text-centric evaluations provide a reliable diagnostic signal by auditing two core vulnerabilities: the tendency of monolithic scores to obscure localized structural errors, and the severe instability introduced by unconstrained LLM judges.

![](images/a230141c5f8f6b3613483358fcbdfa07023b1c7b2fa3c5c3ba0685fae263ecbf.jpg)  
Figure 5: Qualitative case study of deep-structured diagnostics. Traditional holistic evaluations often over-reward fluent prose, missing critical errors like hallucinated “protective eyewear” or temporally unsupported actions. By performing bounded semantic checks on structurally valid units and assigning localized Recall and Precision, OmniCapBench decouples genuine descriptive fidelity from structural noise, providing a high-resolution diagnostic trace.

Do global LLM judges introduce unstable conclusions? Beyond masking errors, holistic evaluations introduce severe unreliability by forcing LLM judges to perform unconstrained global reasoning. To demonstrate this, we score identical model predictions using weak (Qwen3.6-27B), mid (GPT-4o), and strong

Table 5: Judge sensitivity. Stability across judges.
<table><tr><td>Judge</td><td>Holistic</td><td>QA</td><td>OmniCapBench</td></tr><tr><td>Weak</td><td>71.24</td><td>61.17</td><td>56.82</td></tr><tr><td>Mid</td><td>83.08</td><td>53.49</td><td>58.24</td></tr><tr><td>Strong</td><td>93.92</td><td>56.57</td><td>57.60</td></tr></table>

(Gemini-2.5-Pro) judges across two baseline protocols: Holistic-style judging (UGC-VideoCap) and QA-style judging (Omni-Cloze). As Table 5 reports, both baselines exhibit extreme judge sensitivity—scores for the exact same outputs artificially inflate from 71.24 (weak judge) to 93.92 (strong judge). In contrast, OmniCapBench isolates the LLM’s role, restricting it exclusively to bounded semantic checks on structurally aligned units while leaving complex verification to deterministic rules. By eliminating implicit fact extraction and unconstrained reasoning, OmniCapBench establishes a stable, objective scoring framework resilient to judge idiosyncrasies.

Does global text-level scoring hide localized errors? To empirically investigate whether compressing omnimodal events into a global scalar masks capability deficits, we construct hard-to-distinguish subsets of 200 videos from the UGC-VideoCap and video-SALMONN 2 benchmarks. Specifically, we define two evaluation pairs representing model upgrades: Qwen3.5-Omni-Flash vs. Qwen3-Omni, and Qwen3.5-Omni-Plus vs. Qwen3.5-Omni-Flash (details in Appendix B.2). For each pair, we strictly sample videos where the weaker and stronger models exhibit near-identical performance (∼50% vs. ∼50%) under traditional text-centric metrics, creating a false illusion of parity. However, as Table B.7 and Figure 4a demonstrate, shifting the evaluation to native structural constraints completely breaks this illusion. The average score of the stronger models climbs to perfectly align with human audit (74.2%), whereas the weaker models collapse (25.8%) when their structural flaws are exposed. This confirms our core hypothesis: global metrics are easily manipulated by linguistic fluency, inadvertently rewarding structurally flawed outputs as long as the generated prose is coherent. Relying on a monolithic text presentation layer severely degrades diagnostic resolution, necessitating a shift towards deep-structured evaluation.

Qualitative validation of deep-structured diagnostics. We qualitatively validate this decoupled scoring approach in Figure 5. Traditional paradigms are easily confounded by fluent hallucinations; for instance, unconstrained metrics frequently reward Qwen3.5-Omni-Plus for hallucinating detailed but unsupported attributes (e.g., “protective eyewear”) because the overall prose reads well. Furthermore, models often generate plausible action descriptions that completely violate the continuous temporal boundaries of the visual shot. By strictly decomposing the evaluation into atomic units and enforcing structural alignment before any semantic checking, OmniCapBench explicitly penalizes these specific omissions and temporal violations through localized Recall and Precision scores. This traceable diagnostic process guarantees that the final score reflects precise spatial-temporal grounding rather than mere text-generation capability.

## 5 Conclusion

The reliable evaluation of audio–visual captioning must provide fine-grained, localizable diagnostic signals. To address this need, we propose OmniCapBench, which reframes the evaluation process through deep-structured evaluation units. The resulting atomic units across Reference, Shot, and Event tracks ensure that the evaluation achieves comprehensive video coverage while preserving fine-grained detail. Moreover, this atomic formulation enables a decoupled scoring process, in which deterministic rules rigorously verify structural relationships and LLMs are strictly restricted to local semantic checks. Beyond establishing a rigorous benchmark for current multimodal models, our empirical analysis reveals critical flaws in holistic paradigms. Global text scores often mask localized errors and reward structurally inconsistent outputs, while unconstrained LLM judges introduce severe ranking instability. Ultimately, OmniCapBench demonstrates that reliable evaluation requires moving from monolithic text generation to verifiable atomic units, transforming performance measurement into an actionable diagnostic tool for multimodal systems.

## References

[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025. doi: 10.48550/arXiv.2502.13923. URL https://arxiv.org/abs/2502.13923.

[2] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings ofthe acl workshop on intrinsic and extrinsic evaluation measuresfor machine translation and/or summarization, pages 65–72, 2005.

[3] ByteDance Seed Team. Seed2.0 model card: Towards intelligence frontier for real-world complexity, 2026. URL https://arxiv.org/abs/2607.00248.

[4] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, Yao Dou, Jaden Park, Jianfeng Gao, Yong Jae Lee, and Jianwei Yang. TemporalBench: Benchmarking fine-grained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818, 2024. doi: 10.48550/arXiv.2410.10818. URL https://arxiv.org/abs/2410.10818.

[5] Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jenq-Neng Hwang, Saining Xie, and Christopher D. Manning. Auroracap: Efficient, performant video detailed captioning and a new benchmark. arXiv preprint arXiv:2410.03051, 2025. doi: 10.48550/arXiv.2410.03051. URL https://arxiv.org/abs/2410.03051.

[6] Xinlong Chen, Yue Ding, Weihong Lin, Jingyun Hua, Linli Yao, Yang Shi, Bozhou Li, Yuanxing Zhang, Qiang Liu, Pengfei Wan, Liang Wang, and Tieniu Tan. Avocado: An audiovisual video captioner driven by temporal orchestration. arXiv preprint arXiv:2510.10395, 2025. doi: 10.48550/arXiv.2510.10395. URL https://arxiv.org/abs/2510.10395.

[7] Xinlong Chen, Yuanxing Zhang, Chongling Rao, Yushuo Guan, Jiaheng Liu, Fuzheng Zhang, Chengru Song, Qiang Liu, Di Zhang, and Tieniu Tan. Vidcapbench: A comprehensive benchmark of video captioning for controllable text-to-video generation. In Wanxiang Che, Joyce

Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Associationfor Computational Linguistics: ACL 2025, pages 8543–8563, Vienna, Austria, jul 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025. findings-acl.449. URL https://aclanthology.org/2025.findings-acl.449/.

[8] Xinlong Chen, Weihong Lin, Jingyun Hua, Linli Yao, Yue Ding, Bozhou Li, Bohan Zeng, Yang Shi, Qiang Liu, Yuanxing Zhang, Pengfei Wan, Liang Wang, and Tieniu Tan. Diadem: Advancing dialogue descriptions in audiovisual video captioning for multimodal large language models. arXiv preprint arXiv:2601.19267, 2026. doi: 10.48550/arXiv.2601.19267. URL https://arxiv.org/abs/2601.19267.

[9] Zhangquan Chen, Jiale Tao, Ruihuang Li, Yihao Hu, Ruitao Chen, Zhantao Yang, Xinlei Yu, Haodong Jing, Manyuan Zhang, Shuai Shao, Biao Wang, Qinglin Lu, and Ruqi Huang. Omnivideo-r1: Reinforcing audio-visual reasoning with query intention and modality attention. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=he06cvibXv.

[10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261.

[11] Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyue Sun, Yingjing Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, Jiancheng Gui, Luoyuan Zhang, Xian Sun, Fuwei Huang, Moye Chen, Zhuo Lin, Hanyu Liu, Qingxin Gui, Qingzhe Han, Yuyang Wen, Huiping Liu, Rongkang Wang, Yaqi Zhang, Hongliang Wei, Chi Chen, You Li, Kechen Fang, Jie Zhou, Yuxuan Li, Guoyang Zeng, Chaojun Xiao, Yankai Lin, Xu Han, Maosong Sun, Zhiyuan Liu, and Yuan Yao. Minicpm-o 4.5: Towards real-time full-duplex omni-modal interaction, 2026. URL https://arxiv.org/abs/2604.27393.

[12] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24108–24118, 2025.

[13] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, Long Ma, Xiawu Zheng, Rongrong Ji, Xing Sun, Caifeng Shan, and Ran He. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025. doi: 10.48550/arXiv.2501.01957. URL https://arxiv.org/abs/2501.01957.

[14] Soichiro Fujita, Satoshi Tsutsui, Yoshihiko Ejiri, Maris Shikida, and Toshihiko Yamasaki. SODA: Story oriented dense video captioning evaluation framework. In European Conference on Computer Vision, pages 517–531, 2020.

[15] Gemini Team. Gemini 3.1: Best for complex tasks and bringing creative concepts to life, 2026. URL https://deepmind.google/models/gemini/pro/.

[16] Tiantian Geng, Jinrui Zhang, Qingni Wang, Teng Wang, Jinming Duan, and Feng Zheng. Longvale: Vision-audio-language-event benchmark towards time-aware omni-modal perception of long videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18959–18969, June 2025.

[17] Arushi Goel, Sreyan Ghosh, Vatsal Agarwal, Nishit Anand, Kaousheik Jayakumar, Lasha Koroshinadze, Yao Xu, Katie Lyons, James Case, Karan Sapra, Kevin J. Shih, Siddharth Gururani, Abhinav Shrivastava, Ramani Duraiswami, Dinesh Manocha, Andrew Tao, Bryan Catanzaro, Mohammad Shoeybi, and Wei Ping. Mmou: A massive multi-task omni understanding and reasoning benchmark for long and complex real-world videos. arXiv preprint arXiv:2603.14145, 2026. doi: 10.48550/arXiv.2603.14145. URL https://arxiv.org/abs/2603.14145.

[18] Chaoqun He, Mingyang Xiang, Yingjing Xu, Bokai Xu, Junbo Cui, Jie Zhou, Yuan Yao, and Lijie Wen. Omni-duplexeval: Evaluating real-time duplex omni-modal interaction, 2026. URL https://arxiv.org/abs/2605.17360.

[19] Xuan He, Cong Wei, Yuhao Cheng, Linrui Ma, Yuxuan Zhang, Zuojun Li, Yuhao Wen, Zeyi Liu, Yuren Hao, Songcheng Cai, et al. Vgi-bench: Probing visual intelligence in video generation models. arXiv preprint arXiv:2608.19583, 2026.

[20] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. arXiv preprint arXiv:2502.04326, 2026. doi: 10.48550/arXiv.2502.04326. URL https://arxiv.org/abs/2502.04326.

[21] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-MMMU: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025. doi: 10.48550/arXiv.2501.13826. URL https://arxiv.org/abs/2501.13826.

[22] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP), pages 611–626, 2023.

[23] Caorui Li, Yu Chen, Yiyan Ji, Jin Xu, Zhenyu Cui, Shihao Li, Yuanxing Zhang, Wentao Wang, Zhenghao Song, Dingling Zhang, Ying He, Haoxiang Liu, Yuxuan Wang, Qiufeng Wang, Jiafu Tang, Zhenhe Wu, Jiehui Luo, Zhiyu Pan, Weihao Xie, Chenchen Zhang, Zhaohui Wang, Jiayi Tian, Yanghai Wang, Zhe Cao, Minxin Dai, Ke Wang, Runzhe Wen, Yinghao Ma, Yaning Pan, Sungkyun Chang, Termeh Taheri, Haiwen Xia, Christos Plachouras, Emmanouil Benetos, Yizhi Li, Ge Zhang, Jian Yang, Tianhao Peng, Zili Wang, Minghao Liu, Junran Peng, Zhaoxiang Zhang, and Jiaheng Liu. Omnivideobench: Towards audio-visual understanding evaluation for omni mllms. arXiv preprint arXiv:2510.10689, 2026. doi: 10.48550/arXiv.2510.10689. URL https://arxiv.org/abs/2510.10689.

[24] Yadong Li, Haoze Sun, Mingan Lin, Tianpeng Li, Guosheng Dong, Tao Zhang, Bowen Ding, Wei Song, Zhenglin Cheng, Yuqi Huo, Song Chen, Xu Li, Da Pan, Shusen Zhang, Xin Wu, Zheng Liang, Jun Liu, Tao Zhang, Keer Lu, Yaqi Zhao, Yanjun Shen, Fan Yang, Kaicheng Yu, Tao Lin, Jianhua Xu, Zenan Zhou, and Weipeng Chen. Baichuan-omni technical report. arXiv preprint arXiv:2410.08565, 2024. doi: 10.48550/arXiv.2410.08565. URL https: //arxiv.org/abs/2410.08565.

[25] Yunheng Li, Hengrui Zhang, Meng-Hao Guo, Wenzhao Gao, Shaoyong Jia, Shaohui Jiao, Qibin Hou, and Ming-Ming Cheng. Towards universal video mllms with attribute-structured and quality-verified instructions. arXiv preprint arXiv:2602.13013, 2026.

[26] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. Text summarization branches out, pages 74–81, 2004.

[27] Zhihang Liu, Chen-Wei Xie, Bin Wen, Feiwu Yu, Jixuan Chen, Pandeng Li, Boqiang Zhang, Nianzu Yang, Yinglu Li, Zuan Gao, Yun Zheng, and Hongtao Xie. Capability: A comprehensive visual caption benchmark for evaluating both correctness and thoroughness. arXiv preprint arXiv:2502.14914, 2025. doi: 10.48550/arXiv.2502.14914. URL https://arxiv.org/abs/ 2502.14914.

[28] Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model. arXiv preprint arXiv:2502.04328, 2025. doi: 10.48550/arXiv.2502.04328. URL https://arxiv.org/abs/2502.04328.

[29] Ziyang Ma, Ruiyang Xu, Zhenghao Xing, Yunfei Chu, Yuxuan Wang, Jinzheng He, Jin Xu, Pheng-Ann Heng, Kai Yu, Junyang Lin, Eng Siong Chng, and Xie Chen. Omni-captioner: Data pipeline, models, and benchmark for omni detailed perception. arXiv preprint arXiv:2510.12720, 2026. doi: 10.48550/arXiv.2510.12720. URL https://arxiv.org/abs/2510.12720.

[30] NVIDIA, Amala Sanjay Deshmukh, Kateryna Chumachenko, Tuomas Rintamaki, Matthieu Le, et al. Nemotron 3 nano omni: Efficient and open multimodal intelligence, 2026. URL https://arxiv.org/abs/2604.24954.

[31] Junfu Pu, Yuxin Chen, Teng Wang, and Ying Shan. Omniscript: Towards audio-visual script generation for long-form cinematic video. arXiv preprint arXiv:2604.11102, 2026. doi: 10.48550/arXiv.2604.11102. URL https://arxiv.org/abs/2604.11102.

[32] Qwen Team. Qwen3.5-omni technical report. arXiv preprint arXiv:2604.15804, 2026. doi: 10.48550/arXiv.2604.15804. URL https://arxiv.org/abs/2604.15804.

[33] Guangzhi Sun, Wenyi Yu, Changli Tang, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Yuxuan Wang, and Chao Zhang. video-salmonn: Speech-enhanced audio-visual large language models. arXiv preprint arXiv:2406.15704, 2024. doi: 10.48550/arXiv.2406.15704. URL https://arxiv.org/abs/2406.15704.

[34] Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zejun Ma, and Chao Zhang. video-salmonn 2: Caption-enhanced audio-visual large language models. arXiv preprint arXiv:2506.15220, 2025. doi: 10.48550/arXiv.2506.15220. URL https: //arxiv.org/abs/2506.15220.

[35] Qwen Team. Qwen3.5-omni technical report, 2026. URL https://arxiv.org/abs/2604. 15804.

[36] Tencent Hunyuan Team. Script-a-video: Deep structured audio-visual captions via factorized streams and relational grounding, 2026. URL https://arxiv.org/abs/2604.11244.

[37] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575, 2015.

[38] Jiahao Wang, An Ping, Yanghai Wang, Yuanxing Zhang, Shihao Li, Hanyan Bian, Yichi Ren, Yize Zhang, Han Wang, Haowen Chen, Junze Li, Jiaqi Wang, Yiyang Hu, Zhuze Xu, Zijie Zhang, and Jiaheng Liu. Omnicap-if: Benchmarking and improving instruction following abilities for omni-video captioning, 2026. URL https://arxiv.org/abs/2606.08572.

[39] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. LVBench: An extreme long video understanding benchmark. arXiv preprint arXiv:2406.08035, 2025. doi: 10.48550/arXiv.2406. 08035. URL https://arxiv.org/abs/2406.08035.

[40] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A benchmark for long-context interleaved video-language understanding. arXiv preprint arXiv:2407.15754, 2024. doi: 10.48550/arXiv.2407.15754. URL https://arxiv.org/abs/2407.15754.

[41] Keming Wu, Yijing Cui, Wenhan Xue, Qijie Wang, Xuan Luo, Zhiyuan Feng, Zuhao Yang, Sudong Wang, Sicong Jiang, Haowei Zhu, et al. Worldreasonbench: Human-aligned stress testing of video generators as future world-state predictors. arXiv preprint arXiv:2605.10434, 2026.

[42] Keming Wu, Baoyi Wang, Kaichen Zhang, Xiang An, Zuhao Yang, Sudong Wang, Haowei Zhu, Tingxuan Huang, Hongcheng Gao, and Bin Wang. Streamopd: A post-training recipe with spatio-temporal cue gating for streaming video understanding. arXiv preprint arXiv:2608.16320, 2026.

[43] Peiran Wu, Yunze Liu, Zhengdong Zhu, Enmin Zhou, and Junxiao Shen. Ugc-videocaptioner: An omni ugc video detail caption model and new benchmarks, 2025. URL https://arxiv. org/abs/2507.11336.

[44] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. NExT-QA: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9777–9786, June 2021.

[45] Xiaomi LLM-Core Team. MiMo-V2.5: A native omnimodal model with agentic capabilities. https://huggingface.co/XiaomiMiMo/MiMo-V2.5, 2026. Model card; language backbone described in the MiMo-V2-Flash technical report.

[46] Zhenghao Xing, Xiaowei Hu, Chi-Wing Fu, Wenhai Wang, Jifeng Dai, and Pheng-Ann Heng. Echoink-r1: Exploring audio-visual reasoning in multimodal llms via reinforcement learning. arXiv preprint arXiv:2505.04623, 2025. doi: 10.48550/arXiv.2505.04623. URL https: //arxiv.org/abs/2505.04623.

[47] Dannong Xu, Zhongyu Yang, Jun Chen, Yingfang Yuan, Ming Hu, Lei Sun, Luc Van Gool, Danda Pani Paudel, and Chun-Mei Feng. Multihaystack: Benchmarking multimodal retrieval and reasoning over 40k images, videos, and documents. In ECCV, 2026. URL https: //arxiv.org/abs/2603.05697.

[48] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025. doi: 10.48550/arXiv.2503.20215. URL https://arxiv.org/abs/2503.20215.

[49] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025. doi: 10.48550/arXiv.2509.17765. URL https://arxiv.org/abs/2509.17765.

[50] Yifan Xu, Xinhao Li, Yichun Yang, Desen Meng, Rui Huang, and Limin Wang. Carebench: A fine-grained benchmark for video captioning and retrieval. In ICLR, 2026.

[51] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. doi: 10.48550/arXiv.2505.09388. URL https://arxiv.org/abs/2505.09388.

[52] Qize Yang, Shimin Yao, Weixuan Chen, Shenghao Fu, Detao Bai, Jiaxing Zhao, Boyuan Sun, Bowen Yin, Xihan Wei, and Jingren Zhou. Humanomniv2: From understanding to omni-modal reasoning with context. arXiv preprint arXiv:2506.21277, 2025. doi: 10.48550/arXiv.2506. 21277. URL https://arxiv.org/abs/2506.21277.

[53] Zhongyu Yang, Ying-Fang Yuan, Xuanming Jiang, Baoyi An, and Wei Pang. Inex: Hallucination mitigation via introspection and cross-modal multi-agent collaboration. In AAAI Conference on Artificial Intelligence, 2025. URL https://api.semanticscholar.org/CorpusID: 283458455.

[54] Zhongyu Yang, Dannong Xu, Yonghan Zhang, Kefan Chen, Xinyi Wang, Yang Xu, Wei Pang, and Yingfang Yuan. Do vision and text cues exhibit evidential coupling? UFO: A benchmark for compositional multimodal reasoning in unified models. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=6UKaYYRM3h.

[55] Zhongyu Yang, Zuhao Yang, Shuo Zhan, Tan Yue, Wei Pang, and Yingfang Yuan. Svagent: Storyline-guided long video understanding via cross-modal multi-agent collaboration. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24062–24072, June 2026.

[56] Zuhao Yang, Yingchen Yu, Yunqing Zhao, Shijian Lu, and Song Bai. Timeexpert: an expertguided video llm for video temporal grounding. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 24286–24296, 2025. doi: 10.1109/ICCV51701.2025.02251.

[57] Zuhao Yang, Sudong Wang, Kaichen Zhang, Keming Wu, Sicong Leng, Yifan Zhang, Bo Li, Chengwei Qin, Shijian Lu, Xingxuan Li, and Lidong Bing. Longvt: Incentivizing “thinking with long videos” via native tool calling. In CVPR, 2026.

[58] Zuhao Yang, Kaichen Zhang, Sudong Wang, Keming Wu, Zhongyu Yang, Bo Li, Xiaojuan Qi, Shijian Lu, Xingxuan Li, and Lidong Bing. Paravt: Taming the tool prior paradox for parallel tool use in agentic video reinforcement learning. arXiv preprint arXiv:2605.20342, 2026.

[59] Linli Yao, Yuancheng Wei, Yaojie Zhang, Lei Li, Xinlong Chen, Feifan Song, Ziyue Wang, Kun Ouyang, Yuanxin Liu, Lingpeng Kong, Qi Liu, Pengfei Wan, Kun Gai, Yuanxing Zhang, and Xu Sun. Timechat-captioner: Scripting multi-scene videos with time-aware and structural audio-visual captions. arXiv preprint arXiv:2602.08711, 2026. doi: 10.48550/arXiv.2602.08711. URL https://arxiv.org/abs/2602.08711.

[60] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. MiniCPM-V: A GPT-4V level MLLM on your phone, 2024. URL https://arxiv.org/abs/2408.01800.

[61] Kaichen Zhang, Wei Huang, Keming Wu, Bo Li, and Xiaojuan Qi. Aero realtime: Fully aligned input-output streams for low-latency streaming multimodal generation. arXiv preprint arXiv:2608.08469, 2026.

[62] Shi-Xue Zhang, Hongfa Wang, Duojun Huang, Xin Li, Xiaobin Zhu, and Xu-Cheng Yin. Vcapsbench: A large-scale fine-grained benchmark for video caption quality evaluation. Proceedings of the AAAI Conference on Artificial Intelligence, 40(15):12726–12734, 2026. doi: 10.1609/aaai. v40i15.38269. URL https://ojs.aaai.org/index.php/AAAI/article/view/38269.

[63] Jiaxing Zhao, Xihan Wei, and Liefeng Bo. R1-omni: Explainable omni-multimodal emotion recognition with reinforcement learning. arXiv preprint arXiv:2503.05379, 2025. doi: 10. 48550/arXiv.2503.05379. URL https://arxiv.org/abs/2503.05379.

[64] Jiaxing Zhao, Qize Yang, Yixing Peng, Detao Bai, Shimin Yao, Boyuan Sun, Xiang Chen, Shenghao Fu, Weixuan Chen, Xihan Wei, and Liefeng Bo. Humanomni: A large vision-speech language model for human-centric video understanding. arXiv preprint arXiv:2501.15111, 2025. doi: 10.48550/arXiv.2501.15111. URL https://arxiv.org/abs/2501.15111.

[65] Hao Zhong, Muzhi Zhu, Zongze Du, Zheng Huang, Canyu Zhao, Mingyu Liu, Wen Wang, Hao Chen, and Chunhua Shen. Omni-r1: Reinforcement learning for omnimodal reasoning via twosystem collaboration. arXiv preprint arXiv:2505.20256, 2025. doi: 10.48550/arXiv.2505.20256. URL https://arxiv.org/abs/2505.20256.

[66] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. MLVU: Benchmarking multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2025. doi: 10.48550 arXiv.2406.04264. URL https://arxiv.org/abs/2406.04264.

## Appendix

A Implementation Details 17   
B More Experimental Results 28   
C Limitations and Broader Impact 37   
D Ethical Considerations 38   
E Dataset Licenses and Usage 38   
F Instruction Templates and Prompts 39

## Overview of Appendix

This appendix provides comprehensive supplementary materials to ensure the reproducibility and transparency of the OmniCapBench benchmark. The appendix is structured as follows:

• Implementation Details (Appendix A): Details the benchmark construction process, data statistics, and the full experimental setup used for evaluation.

• More Experimental Results (Appendix B): Provides additional quantitative results, detailed failure analyses, and qualitative Case Studies.

• Limitations and Broader Impact (Appendix C): Discusses the boundaries of our evaluation scope and the broader impacts of structured video benchmarking.

• Ethical Considerations (Appendix D): Addresses privacy, consent, and the ethical use of audiovisual datasets.

• Dataset Licenses and Usage (Appendix E): Documents the licenses and fair-use protocols for all underlying video sources and utilized models.

• Instruction Templates and Prompts (Appendix F): Provides the complete, executable prompts used for unit generation, local semantic judging, and structural parsing.

## A Implementation Details

## A.1 Benchmark Construction Details

## A.1.1 Benchmark Construction Details

Our construction pipeline combines data curation with a multi-model pipeline. Figure 2 illustrates this workflow.

1. Candidate Curation Thresholds: Before semantic annotation begins, candidate videos are filtered by technical quality, audio availability, and predefined duration buckets (< 1 min, 1–3 min, 3–5 min). We additionally enforce minimum structural density thresholds, such as minimum shots, events, and characters per minute, so that selected videos contain enough audio-visual structure for deep-structured evaluation. The exact thresholds are detailed in Table A.2.

2. Multi-Model Pipeline: We decompose the annotation into a multi-role pipeline:

• Generator: Initially drafts candidate JSON for the current track, for instance, all Reference entities.

• Format Verification: A deterministic script verifies required keys, ID uniqueness, temporal bounds, cross-links, and support fields. Malformed drafts are immediately rejected and returned with structured error traces for regeneration.

• Evaluators: Independent frontier models review the drafted facts for factual accuracy, temporal consistency, and structural coherence against the raw video.

• Refiners: Accepted drafts are passed to secondary models to locally refine and enrich descriptive fields, for instance, expanding the visual details of an appearance\_anchor. Crucially, this refinement occurs under the strict condition that the established structure remains invariant; a programmatic validator ensures all IDs, timestamps, and cross-links are unmodified. If a textual refinement breaks a structural link, the pipeline falls back to the original verified draft.

• Selector: Acts as the final judge, selecting the most accurate verified version without permission to rewrite any fields.

This separation of roles maintains structural consistency. Candidate states remain mere “drafts” until they pass all Format Verifications and consensus checks, culminating in human review.

3. Annotation-Only Timestamp Overlays: Throughout reference construction, the frames presented to the pipeline models and to human reviewers carry a red timestamp burned into the bottom-right corner. This overlay lets annotators and Evaluators anchor shot and event boundaries to an exact time index, which is what makes the audited time\_range fields frame-accurate. The overlay is strictly an annotation aid confined to the construction and review interfaces: all evaluation inputs are re-rendered from the original videos, so no evaluated model is ever shown a burned-in timestamp, and no ablation over overlay removal applies to the reported temporal-grounding numbers (Appendix A.1.3).

Table A.1: OmniCapBench Task Suite. All diagnostic tasks derive from the same frozen system of atomic units $S ^ { \star } ( V )$ , but each exposes a distinct model failure mode.
<table><tr><td>Task</td><td>Input</td><td>Output</td><td>Diagnostic Target</td></tr><tr><td>Generation</td><td>Raw Video V</td><td>Scored structured units</td><td>Native recovery and precision</td></tr><tr><td>Probing</td><td>Predicted JSON Ê(V)</td><td>Probe pass/fail</td><td>Internal structural validity</td></tr><tr><td>Verification</td><td>V + Candidate JSON unit</td><td>Accept/reject boolean</td><td>Local contradiction detection</td></tr></table>

Table A.2: Candidate curation thresholds. Video-level filters applied before reference-unit construction. Per-minute intervals are inclusive and follow [center − std, 3 × center]; absolute thresholds apply only to the listed duration buckets.
<table><tr><td>Bucket</td><td>Metric</td><td>Required Range</td></tr><tr><td>All</td><td>shots/min</td><td>[5, 30]</td></tr><tr><td>All</td><td>events/min</td><td>[2, 15]</td></tr><tr><td>&lt; 1 min</td><td>subjects/min</td><td>[3, 12]</td></tr><tr><td>&lt; 1 min</td><td>scenes/min</td><td>[3, 15]</td></tr><tr><td>&lt; 1 min</td><td>total shots</td><td>≥4</td></tr><tr><td>&lt; 1 min</td><td>total events</td><td>≥2</td></tr><tr><td>1-3 min</td><td>subjects/min</td><td>[2,9]</td></tr><tr><td>1-3 min</td><td>scenes/min</td><td>[2, 9]</td></tr><tr><td>3–5 min</td><td>main characters</td><td>≥3</td></tr><tr><td>3–5 min</td><td>main scenes</td><td>≥3</td></tr></table>

## A.1.2 Output Schema

To evaluate models natively, we instruct them to generate outputs strictly conforming to the OmniCapBench JSON schema. This schema explicitly defines the references, events, and shots domains, alongside their mandatory structural links and support fields. Table A.3 summarizes these required components. If a model fails to adhere to these structural instructions, such as outputting a continuous paragraph instead of a parseable JSON graph, it triggers a parse failure at the SGC gate and receives a score of 0 for all downstream metrics.

During inference, all generative models operate at a temperature of 0.0 (greedy decoding) to minimize structural hallucinations, with a maximum output limit of 4096 tokens. The detailed instruction templates used in our pipeline are provided at the end of this appendix.

Table A.3: Schema Field Definitions. Descriptions and typing rules for every structural component required from the evaluated model.
<table><tr><td>Component</td><td>Fields</td><td>Description</td></tr><tr><td>references</td><td>ref_id, type, semantic_description, appearance_anchor</td><td>Persistent entities or scenes with detailed visual descriptions under id_features Timed audio-centered units; spoken dialogue requires exact content. line</td></tr><tr><td>events</td><td>event_id, type, time_range, content</td><td></td></tr><tr><td>shots</td><td>shot_id, time_range, shot_transfer, visual_description, camera</td><td>Visual timeline segments containing fine-grained sub-shot units</td></tr><tr><td>Link fields</td><td>active_events, references_in_shot, reference IDs in content</td><td>Checkable structural links governing event-shot membership and reference context</td></tr><tr><td>Support fields</td><td>appearance_anchor, visual_description, time_range</td><td>Verifiable evidence fields utilized for local semantic diagnostics</td></tr></table>

## A.1.3 Generation and Evaluation Protocols

This section details the implementation mechanics of OmniCapBench, including how models are invoked for structured generation and how LLM-assisted matching and judging are executed in code.

API Invocation and Fallback Mechanics Because OmniCapBench strictly requires the generation of complex, interdependent JSON schemas, interacting with diverse model APIs requires robust programmatic handling. As implemented in our evaluation engine:

• Modality Inputs: Proprietary models, such as the Gemini family, receive the video via native API upload endpoints or HTTP URLs, while open-source checkpoints, including the Qwen-Omni series, receive interleaved Base64-encoded visual frames and raw audio waveforms alongside the task instruction.

• Retry and Repair Logic: The pipeline automatically strips hallucinated markdown formatting, for instance, “‘json ... “‘ delimiters. If the output remains structurally malformed, the system triggers up to 5 exponential-backoff retries. If the model consistently fails to produce a valid JSON schema, the output is permanently logged as a parse failure, and all downstream structural metrics for that sample evaluate to zero.

Native Track Generation Protocol To generate the required atomic units, models are guided by a comprehensive schema definition. Depending on the model’s context window and reasoning capacity, the inference engine employs one of two strategies:

• Single-Pass Generation: The model receives the full video alongside a unified system instruction and is required to output a single, complete JSON object containing the references, events, and shots arrays. This approach tests the model’s ability to globally reason over the entire narrative.

• Progressive Pipeline: For extremely long videos or models with limited context windows, generation is decoupled. First, the model processes the entire video to extract persistent entities. Then, it processes the video in sliding temporal segments to generate temporally grounded events and shots, actively referencing the global entities defined in the first step.

The generation instructions serve as the strict standard operating procedure for the models. Rather than allowing the model to freely interpret the task, we embed explicit annotation guidelines directly into the system instruction. To ensure models accurately ground their predictions, the protocol enforces several critical principles:

• Timestamp Convention: Models must report every temporal field as a [start, end] range in seconds relative to the beginning of the video, and are instructed to keep all boundaries inside the video duration; out-of-range predictions are clamped before scoring (Appendix A.2.5). Evaluated models receive the original, unaltered frames: no timestamp is burned into the pixels and no per-frame time index is injected into the prompt, so every temporal boundary must be inferred from the video stream itself (Listing B.1). The red timestamp overlays used during reference construction (Appendix A.1.1) are an annotation aid only and never appear in any model-facing input.

• Mute Video Handling: If a video contains no audio track, the model is strictly forbidden from inferring auditory events based on visual cues; for instance, a person moving their mouth must not trigger a hallucinated dialogue event unless audio is actually present.

• Minimum Granularity Principle (Visual Subshots): Models are instructed to decompose visual narratives into the smallest indivisible atomic actions. For instance, rather than generating a compound description like “picked up the cup, drank, and put it down,” the model is forced to split this into three distinct subshots.

• Temporal Overlap Principle: Crucially, models are explicitly encouraged to generate overlapping temporal boundaries for subshots. If Person A is walking while Person B is nodding, these are treated as concurrent independent streams rather than being artificially serialized.

• Strict Objectivity: Models must describe pure visual and auditory facts. We explicitly ban causal connective inferences, for instance, “The vase broke because the ball hit it.”, in favor of sequential factual statements.

## A.1.4 Dataset and Domain Statistics

OmniCapBench evaluates models across diverse content domains and temporal scales. We report detailed dataset statistics not to claim unrestricted universal coverage, but to explicitly define the boundaries of our evaluation scope. As detailed in Table A.4, the benchmark comprises 786 videos with verified structural averages. We break down this diversity across multiple structural and semantic dimensions to ensure our testbed evaluates joint audio-visual comprehension robustly.

Data Source and Category Distributions To ensure domain diversity, the videos in OmniCap-Bench are curated from multiple public datasets. Figure A.1(a) illustrates the distribution of source datasets. Beyond the source datasets, we analyze the semantic categories of the videos. Figure A.1(b) demonstrates the temporal footprint of these categories. Notably, categories such as "Entertainment" and "News & Politics" exhibit distinct duration distributions, highlighting the varying temporal demands placed on models when processing different content types.

Table A.4: Full Dataset Statistics. Comprehensive macro averages and distribution metrics supplementing Table 2. These statistics explicitly define the coverage and complexity of the OmniCapBench evaluation scope.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Global Counts and Averages</td><td></td></tr><tr><td>Total Videos</td><td>786</td></tr><tr><td>Mean Duration (seconds)</td><td>58.82</td></tr><tr><td>Median Duration (seconds)</td><td>28.32</td></tr><tr><td>Avg. References / Video</td><td>7.40</td></tr><tr><td>Avg. Events / Video Avg. Shots / Video</td><td>8.32</td></tr><tr><td></td><td>14.53</td></tr><tr><td>Video-Level Distributions</td><td></td></tr><tr><td>Train / Dev / Test Split Audio Presence Rate (Videos with active audio)</td><td>None (Test-only benchmark) 100.00%</td></tr></table>

![](images/d3a9c5585513f3e12be552ad6d4b0af5c164d2b620fe031961c82e2f9baa37b1.jpg)  
(a) Source Dataset Distribution. A lollipop chart detailing the origins of the videos in the OmniCap-Bench benchmark. The diverse sources ensure models are tested against varying production styles and visual characteristics.

![](images/da2198b05c8a056ce9efa4940e478c2e6f15909056a597fe1b6639a01f1b34dc.jpg)  
(b) Categories by Duration Bucket. A stacked bar chart showing the composition of video durations, including < 1 min, 1 − 3 min, and 3 − 5 min, across the top 10 semantic categories.  
Figure A.1: Data Source and Category Distributions. The left panel illustrates the distribution of source datasets, ensuring domain diversity. The right panel shows the composition of video durations across the top 10 semantic categories.

Temporal and Density Characteristics A core contribution of OmniCapBench is its fine-grained structural evaluation, which necessitates characterizing how densely events and shots are annotated across videos. To contextualize this annotation regime, Figure A.2(a) provides an exhaustive view of the benchmark’s temporal distribution, spanning short clips to videos extending up to 5 minutes. Figure A.4 then offers a complementary perspective by separating two related but distinct properties: the absolute relationship between video duration and event count, and the normalized per-minute density of shots and events.

Structural Complexity Metrics We measure the complexity of OmniCapBench by examining the rates of shots and events per minute, as well as the absolute counts of structural units per video. As shown in Figure A.3, the unit counts form stable distributions, demanding models to manage multiple entities and events concurrently. Figure A.4 jointly presents the relationship between video duration and event count and the per-minute density of shots and events. Figure A.2(b) further contextualizes this complexity by plotting the visual cuts density against action density for different categories.

![](images/30e30cb0efc58514808e7827e27da2f309c59df6f9e7d6982dee674cdf2a7ea1.jpg)  
(a) Duration Distribution (Seconds). The overall temporal footprint of the benchmark videos, bounded strictly within 5 minutes (300 seconds) to enable rigorous, exact analysis.

![](images/396f9e1e8de3f90eb069df2feb8506687051db3e3f519500535b324d37393a0d.jpg)  
(b) Category Complexity. A bubble chart correlating the visual cuts density (shots/min) with action density (events/min) across categories. Bubble sizes correspond to the number of videos.

Figure A.2: Temporal Footprint and Structural Complexity. The left panel provides an exhaustive view of the video duration distribution. The right panel contextualizes the structural complexity by plotting visual cuts density against action density for different categories.  
![](images/edaae081ecd46d0f39d48b9198cc99347ecf9f2d18f66e9a666d2b6aa04410a9.jpg)

![](images/ab5e4f26c54d951dea92c443eaa00d2cf85188e5093ddf68790597a6ae411b4d.jpg)

![](images/f7ce327c9b9f9d2649dbe23780c0a95ca9bb01ca6e115cb58ef100bb26b3327d.jpg)  
Figure A.3: Structured Unit Counts per Video. Individual histograms of References, Events, and Shots per video, illustrating the dense tracking requirements placed on the models.

Human Annotation and Quality Assurance Protocol To ensure the highest possible data quality for OmniCapBench, we implement a rigorous, iterative human annotation and refinement protocol. Rather than simply calculating post-hoc agreement scores, our pipeline is designed around continuous error discovery and correction. This multi-stage quality assurance process guarantees that the final benchmark data is structurally flawless, semantically rich, and strictly faithful to the original video evidence.

Professional Annotator Training. We engaged a team of professional annotators with extensive experience in dense video captioning and complex multi-modal data processing. Prior to the formal annotation phase, all annotators underwent comprehensive training on the OmniCapBench JSON schema and standardized corner cases. This training specifically addressed challenging scenarios such as heavily occluded characters, ambiguous shot boundaries, overlapping audio events, and fine-grained visual state transitions.

Iterative Refinement Pipeline. The core of our high-quality data generation is an iterative “reviewand-refine” mechanism. Initial drafts, whether generated by annotators or derived from the multimodel pipeline, are subjected to meticulous human review using a synchronized video-audio playback interface. Reviewers scrutinize the drafts for:

![](images/99244107d06f674abd8dab622fc7db1294529b0b1bbf8e0acee043dfe450f327.jpg)  
(a) Video Duration vs. Events Count. A hexagonal binning plot displaying the correlation between the length of the video and the number of annotated audio-visual events. Darker bins represent higher concentrations of videos.

![](images/a9d3a6a9dd45a5f1bd65f8e4893d5b0768e54d138dca37f74dbfbf85266e604d.jpg)  
(b) Per-Minute Density. Violin plots of the shots per minute and events per minute, visualizing the rapid pace of visual and narrative changes across the dataset.  
Figure A.4: Temporal Scale and Per-Minute Structural Density. The left panel relates video duration to event count; the right panel summarizes shots-per-minute and events-per-minute distributions. These two views separate absolute temporal scale from normalized structural density.

• Structural Integrity: Ensuring that all ID references (ref\_id, event\_id) are consistently maintained across temporal boundaries without hallucinated links.

• Temporal Precision: Fine-tuning the start and end timestamps for visual shots and audio events down to the frame level.

• Semantic Granularity: Expanding overly generic descriptions into highly specific factual statements, for instance, adding exact clothing details, specific character poses, or nuanced environmental audio cues.

Whenever an error, omission, or ambiguity is identified, the annotation is sent back for immediate refinement. This targeted correction loop is repeated until the JSON output perfectly reflects the video’s ground truth.

Multi-Stage Adjudication. To resolve complex edge cases, we employ a multi-stage adjudication process. If annotators disagree on the interpretation of a scene, for example, determining whether a background noise constitutes a distinct event or how to segment a continuous camera movement, the case is escalated to senior reviewers. These domain experts finalize the annotations based strictly on the raw video and audio evidence. This protocol ensures that our evaluation references are not only format-compliant but set a gold standard for comprehensive, verifiable audio-visual understanding.

## A.2 Full Experimental Details

## A.2.1 Experiments Compute Resources

All of our experiments are conducted on compute nodes equipped with 8× NVIDIA H20 GPUs. For local model deployment and batched inference, we employ vLLM [22] to optimize serving efficiency and memory usage.

Evaluation Matching Principles Before scoring, predicted units must be structurally aligned to ground-truth units (as formalized in Appendix A.2.5). The programmatic matching principles are implemented as follows:

• Reference Matching: Performed via an LLM-assisted bipartite matcher. The matcher is restricted to comparing only the appearance\_anchor.id\_features.detail\_description of the entities. It strictly enforces a one-to-one match based on semantic equivalence, rejecting generic overlaps, empty descriptions, or contradictory identity/clothing facts.

• Event Matching: Handled deterministically. Dialogue events are matched using normalized Word Error Rate (WER) against the transcribed text (threshold ≥ 0.50). Non-dialogue events, including sound effects and background music, are matched purely based on their temporal Intersection-over-Union (tIoU) (threshold ≥ 0.20).

• Shot and Subshot Matching: Visual shots are matched one-to-one via temporal IoU (threshold ≥ 0.30). Subshots are then matched locally within their aligned parent shots using an expanded time window (δ = 1.0s).

LLM-as-Judge Protocol As established in Section 3.2, OmniCapBench uses LLMs exclusively for bounded local semantic checks, avoiding opaque holistic evaluations. The following examples summarize the fixed local judging criteria used in our pipeline.

By injecting the exact matching scope into the instruction templates and decoupling the evaluation of different modalities, OmniCapBench mathematically guarantees that the final diagnostic scores reflect specific capability deficits rather than arbitrary stylistic penalties.

## A.2.2 Metric Definitions

This appendix formally defines the OmniCapBench evaluation suite. Our core evaluation philosophy shifts video benchmarking from weak, holistic definitions (where an entire paragraph is judged opaquely) to strong, structural definitions. We define individual information points, including characters, sound events, and visual actions, as verifiable atomic units, whose interactions are strictly verified by deterministic rules before localized LLM scoring.

Evaluation Protocol Why bidirectional evaluation is necessary. Unlike multiple-choice or short-answer tasks, audio-visual captioning is fundamentally open-ended: there is no single “correct” output string. When humans and models caption the exact same video, their structured representations rarely form a perfect one-to-one mapping. Three recurring sources of mismatch motivate our design:

• Granularity mismatch: A model might decompose a continuous 5-second visual action into three dense, fine-grained subshots, whereas the ground truth summarizes the same sequence into a single cohesive subshot. Both decompositions can be factually correct.

• Attribute selection divergence: A model might describe a character by clothing and posture, while the human annotator emphasizes hairstyle and facial features; both descriptions are valid, yet they share few lexical tokens.

• Structural topology variation: Models may group audio events under different shots than humans, or split a single reference entity into multiple aliases, creating non-trivial alignment challenges even when the underlying facts are equivalent.

The problem with unidirectional evaluation. Because of these mismatches, evaluating open-ended captioning from a single direction is intrinsically biased and gameable:

• Recall-only evaluation (checking whether the prediction covers the ground truth) rewards excessively verbose, speculative outputs that guarantee coverage at the cost of significant hallucinations.

• Precision-only evaluation (checking whether the ground truth supports the prediction) rewards overly terse, generic responses that are trivially correct but miss key video details.

Our solution: bidirectional matching and scoring. To address this, OmniCapBench applies a bidirectional matching and scoring paradigm specifically for open-vocabulary and continuous fields where the above mismatches are most pronounced, namely References (Subject and Scene appearances) and visual Subshots. We evaluate these elements from two complementary perspectives, both during the structural matching phase (determining which predicted units correspond to which ground-truth units) and the subsequent local semantic scoring phase (measuring how well a matched pair agrees on factual content).

Tie-Breaking Rules: For rule-based event and shot matching, candidate pairs are sorted by their matching score and consumed one-to-one. In instances where multiple predicted units achieve the exact same score against a ground-truth unit, we prioritize the prediction with the earlier temporal start time $( t _ { s } )$ . If a tie persists, we break it using the lexical order of the predicted ID string. Reference matching is parsed from the LLM-produced bipartite map and then defensively checked for valid IDs, type compatibility, and one-to-one consistency.

Table A.5 details the matching rules for each unit layer.

• Reference Matching: Typed one-to-one LLM-assisted matching over appearance\_anchor.id\_features.detail\_description. The matcher only considers person, scene, and object groups and does not use semantic summaries, attributes, timestamps, event links, shot links, or broader context.

• Event Matching: Typed one-to-one matching. Dialogue events are matched using normalized WER over content.line with threshold 0.50; non-dialogue events use temporal Intersection-over-Union (tIoU) with threshold 0.20.

• Shot and Subshot Matching: Shots are matched one-to-one using temporal IoU with threshold 0.30. Subshots are matched locally inside matched shot pairs: a ground-truth subshot creates a window expanded by δ = 1.0s, and overlapping predicted subshots within the matched shot form its candidate group.

Table A.5: Matching Rules Summary. Parameters defining structural alignment across the reference, event, and shot layers.
<table><tr><td>Layer</td><td>Matching Metric</td><td>Default threshold</td><td>Tie-breaking</td></tr><tr><td>Reference</td><td>LLM bipartite matching over detail_description</td><td>None (LLM output)</td><td>Defensive ID/type/one-to-one check</td></tr><tr><td>Dialogue event</td><td>1 – WER over content.line</td><td>≥ 0.50</td><td>Max line score, one-to-one</td></tr><tr><td>Non-dialogue event</td><td>Temporal IoU</td><td>≥ 0.20</td><td>Max tIoU, one-to-one</td></tr><tr><td>Shot</td><td>Temporal IoU</td><td>≥ 0.30</td><td>Max tIoU, one-to-one</td></tr></table>

Table A.6 formally defines how matched, unmatched, invalid, and out-of-scope predictions contribute to credit, penalties, or diagnostics. This scope-bounded scoring policy is critical: it prevents penalizing models for discovering true facts that fall outside our annotation budget, while still strictly penalizing structural hallucinations.

Table A.6: Open-World Scoring Policy. This table outlines exactly how we credit, penalize, or ignore different types of predictions within our bounded evaluation scope. Crucially, while OmniCapBench rewards models for recovering annotated truths, it does not unfairly penalize models for correctly identifying true facts that merely fell outside our annotation budget. Unsupported extras are logged for diagnostic analysis but do not directly dock the main precision metrics.
<table><tr><td>Prediction type</td><td>Matched</td><td>Valid structure</td><td>Supported</td><td>Credit / penalty</td><td>Denominator</td></tr><tr><td>Matched unit</td><td>yes</td><td>yes</td><td>n/a</td><td>Full credit; proceeds to structural/semantic checks.</td><td>Matched metric de- nominators</td></tr><tr><td>Unmatched reference</td><td>no</td><td>yes/no</td><td>no</td><td>No credit; retained in diagnostic logs.</td><td>Appendix audit</td></tr><tr><td>Unmatched event</td><td>no</td><td>yes/no</td><td>no</td><td>No credit; retained in diagnostic logs.</td><td>Appendix audit</td></tr><tr><td>Unmatched sub-shot</td><td>no</td><td>yes/no</td><td>no</td><td>No credit; retained in diagnostic logs.</td><td>Appendix audit</td></tr><tr><td>Invalid or dangling link</td><td>no</td><td>no</td><td>no</td><td>Zero credit; penalized as a structural error.</td><td>Appendix audit</td></tr><tr><td>Valid but out-of-scope fact</td><td>no</td><td>yes</td><td>unknown</td><td>No manual credit, but no strict penalty; logged for auditing.</td><td>Reported diagnos- tics</td></tr></table>

## A.2.3 Full Model and Evaluation Settings

To ensure fair and transparent comparisons, this section explicitly documents the technical parameters used to evaluate each model. In total, we assessed 12 distinct models. To clarify the division between the main results and the robustness analysis:

• Main Benchmark Table: Evaluates 7 native-capable omnimodal models (three proprietary models via API and four open-source checkpoints). These models are capable of processing unstructured audio-visual streams and following instructions to generate structured outputs.

Table A.7: Model inference and configuration settings. One row per evaluated model entry. All models use greedy decoding (temperature = 0, top\_p = 1) to minimize structural hallucination. Closedsource models are accessed via API with no frame cap; local Qwen3-Omni models (30B-A3B) are served via vLLM with TP = 4 and max 192 frames at 2 fps; videos exceeding 1 min use a 2-pass strategy. Qwen2.5-Omni-based models (7B and fine-tuned variants) only evaluate videos ≤ 60 s due to context limits.
<table><tr><td>Model</td><td>Checkpoint / Endpoint</td><td>Params</td><td>Backend</td><td>Audio</td><td>Frames / FPS</td><td>Max Tokens</td><td>Duration</td><td>Long-Video Strategy</td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>Gemini 3.1-Pro</td><td>gemini-3.1-pro-preview</td><td></td><td>API</td><td>√</td><td>no cap / 2 fps</td><td>65536</td><td>All</td><td>Single-pass</td></tr><tr><td>Gemini 2.5-Pro</td><td>gemini-2.5-pro</td><td></td><td>API</td><td>1</td><td>no cap / 2 fps</td><td>65536</td><td>All</td><td>Single-pass</td></tr><tr><td>Qwen3.5-Omni-Plus Qwen3.5-Omni-Flash</td><td>qwen3.5-omni-plus qwen3.5-omni-flash</td><td></td><td>API API</td><td></td><td>no cap / 2 fps</td><td>32768 32768</td><td>All All</td><td>Single-pass</td></tr><tr><td></td><td></td><td></td><td>Open-source Omnimodal Models</td><td>√</td><td>no cap / 2 fps</td><td></td><td></td><td>Single-pass</td></tr><tr><td colspan="9"></td></tr><tr><td>Qwen3-Omni-Instruct</td><td>Qwen3-Omni-30B-A3B- Instruct</td><td>30B-A3B</td><td>vLLM (TP=4)</td><td>√</td><td>192 / 2 fps</td><td>32768</td><td>All</td><td>2-pass (&gt;60 s)</td></tr><tr><td>Qwen3-Omni-Captioner</td><td>Qwen3-Omni-30B-A3B- Captioner</td><td>30B-A3B</td><td>vLLM (TP=4)</td><td>√</td><td>192 / 2 fps</td><td>32768</td><td>All</td><td>2-pass (&gt;60 s)</td></tr><tr><td colspan="9">Specialized Video-Captioning Models (Appendix B.5)</td></tr><tr><td>ASID-Caption-3B</td><td>ASID-Captioner-3B</td><td>3B</td><td>Transformers</td><td>√</td><td>192 / 2 fps</td><td>32768</td><td>≤60s</td><td></td></tr><tr><td>ASID-Caption-7B</td><td>ASID-Captioner-7B</td><td>7B</td><td>Transformers</td><td></td><td>192 / 2 fps</td><td>32768</td><td>≤60s</td><td></td></tr><tr><td>AVoCaDO TimeChat-Captioner</td><td>AVoCaDO TimeChat-Captioner-</td><td>7B</td><td>Transformers</td><td></td><td>192 / 2 fps 192 / 2 fps</td><td>32768 32768</td><td>≤60s ≤60s</td><td></td></tr><tr><td></td><td>GRPO-7B</td><td>7B</td><td>Transformers</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>UGC-VideoCaptioner</td><td>UGC-VideoCaptioner</td><td>7B</td><td>Transformers</td><td>√</td><td>192 / 2 fps</td><td>32768</td><td>≤60s</td><td></td></tr></table>

• Appendix Robustness Table (Appendix B.5): Evaluates 5 specialized video-captioning baselines. These models do not enter the main ranking because their SFT heavily degrades structured schema-following. We explicitly report these failures and their performance via the caption-posthoc route to demonstrate why free-form prose cannot substitute for native atomic unit prediction.

Because these models vary significantly in architecture, they operate under different hardware and context limitations. Table A.7 meticulously units the full configuration settings required to reproduce our runs, including parameter counts, exact checkpoint hashes, frame sampling policies, max token limits, and decoding temperatures. We also explicitly log our evaluation pipeline policies, including retry limits and parse repair strategies. For open-source reproducibility, our released code artifacts replace internal endpoints and filesystem paths with sanitized placeholders.

Error Analysis Details While our rule-based metrics quantify overall model performance, diagnosing specific failure modes requires a deeper analysis of the error distribution. Table A.8 outlines the precise audit contract behind the error-composition chart (Figure 4b) in the main text. It defines exactly what constitutes a failure under each metric family, ensuring every error is interpretable as a concrete, broken structural link rather than a subjective LLM penalty.

## How to read the diagnostic categories.

• Schema or canonicalization failure (Gate): The model completely failed instruction following. The output cannot be converted into the scored unit system, rendering downstream semantic checks mathematically impossible.

• Visual reasoning failures (Visual): The model misses persistent entities, hallucinates non-existent objects, uses the wrong mapped ID when referring to an object in a scene, fails to maintain the same identity for a character across disconnected scenes, or misaligns visual boundaries.

• Audio reasoning failures (Audio): Essential audio events are entirely missing, hallucinated, or aligned to the wrong time segment in the audio track.

• Cross-modal and dialogue failures (Audio-Visual): An event may be plausible in isolation but is attached to the wrong visual region (demonstrating a failure in cross-modal structural reasoning), or spoken dialogue is mis-transcribed or assigned to the wrong speaker.

Table A.8: Error analysis summary. This table defines exactly how we classify failures in structurally valid outputs. It explicitly links each failure mode to the corresponding rule-based metric. Counts and rates are derived from our evaluation logs over applicable samples for the Gemini 3.1-Pro model. This detailed breakdown ensures that every capability deficit is independently traceable rather than being obscured by a single aggregate score.
<table><tr><td>Error Family (Category)</td><td>Primary Metric What triggers this error?</td><td></td><td>Count</td><td>Rate</td></tr><tr><td>Schema Failure (Gate)</td><td>SGC</td><td>Output cannot be parsed into ref- erences, events, and shots, or is missing required JSON fields</td><td>31</td><td>3.8%</td></tr><tr><td>Reference Recovery (Visual)</td><td>Ref F1</td><td>Required ground-truth reference 3778 31.7% is missed, or a hallucinated predicted reference cannot be matched</td><td></td><td></td></tr><tr><td>Reference-Use (Visual)</td><td>RefUse</td><td>A matched shot or subshot cites the wrong persistent reference ID</td><td></td><td>8649 34.1%</td></tr><tr><td>Identity Drift (Visual)</td><td>CCC</td><td>A single persistent reference is split, merged, or inconsistently tracked across multiple shots</td><td></td><td>2656 65.2%</td></tr><tr><td>Shot Recovery/Alignment (Visual) Shot F1/tIoU</td><td></td><td>A visual micro-action or bound- 5156 26.2% ary is missed, hallucinated, or severely misaligned in time</td><td></td><td></td></tr><tr><td>Event Recovery/Alignment (Audio) Event F1/tIoU</td><td></td><td>A required audio event is missed, 4218 36.2% hallucinated, or aligned to a com- pletely wrong time segment</td><td></td><td></td></tr><tr><td>Event-Shot Link (Audio-Visual)</td><td>EVSA</td><td>A predicted audio event is at- 1654 44.5% tached to the completely wrong visual shot in the timeline</td><td></td><td></td></tr><tr><td>Dialogue Error (Audio-Visual)</td><td>Speaker F1</td><td>A matched dialogue event fails to accurately transcribe the spoken text or assign it to the correct vi- sual speaker</td><td>260</td><td>7.0%</td></tr></table>

These structured categories explain precisely why a fluent, holistic caption can appear superficially correct globally while failing rigorously on localized evaluations.

Analysis of the Caption-State Gap. As revealed in Table A.9, evaluating models purely via dense text generation obscures severe structural failures. The highest failure rates occur in Identity Persistence (65.2%) and Temporal Alignment (42.0%). These numbers prove that while models can fluently generate nouns and verbs (yielding high text-level scores), they fundamentally struggle to maintain object permanence across shots or anchor events to precise timeline segments. Relying solely on holistic LLM judging masks these precise failure modes, confirming our motivation that reliable evaluation inherently requires structured, atomic verification.

## A.2.4 Consistency Checks

As introduced in Task 2 (Section 3.2), Structural Consistency Probes are the deterministic rules for auditing the structural relationships of a model’s prediction before invoking any LLM-as-judge. To transition these from abstract evaluation concepts into an executable test suite, we formalize them as logical predicates. These probes automatically verify whether the predicted units utilize valid reference IDs, possess logically compatible time ranges, and maintain consistent cross-modal links.

Failure Conditions We define the following formal predicates, each tied to an exact, programmatic failure trigger:

• ValidRefUse(x, r): For any event or shot x, if x claims to feature reference r, then r must exist in the top-level references list. Failure trigger: A dangling ID is utilized.

Table A.9: Failure taxonomy for outputs exhibiting high text-level scores but low structural scores. This table dissects exactly why a seemingly fluent holistic prose caption can fail rigorous deep-structured evaluation. We map specific failure modes to the exact metric designed to penalize them. All rates are derived from our validation logs for the Gemini 3.1-Pro model.
<table><tr><td>Failure type</td><td>Category</td><td>Metric</td><td>What it looks like in practice</td><td>Rate</td></tr><tr><td>Broken identity persistence</td><td>Visual</td><td>CCC</td><td>The model correctly describes a character but assigns them a new ID, such as PERSON_2, when they reappear in a later shot, los- ing track of their persistent iden-</td><td>65.2%</td></tr><tr><td>Wrong event reference</td><td>Visual</td><td>RefUse</td><td>tity. An off-screen voice or a back- ground action is attributed to the wrong character ID in the struc-</td><td>34.1%</td></tr><tr><td>Event alignment error</td><td>Audio</td><td>Event tIoU</td><td>tured output. The model correctly identifies an event, like a dog barking, but aligns it to a completely wrong time segment where the dog is</td><td>42.0%</td></tr><tr><td>Missing structural field</td><td>Gate</td><td>SGC</td><td>silent. The JSON output is missing re- quired fields like time_range</td><td>3.8%</td></tr><tr><td>Unsupported extra event</td><td>Audio</td><td>Event (Precision)</td><td>or detail_description, ren- dering the unit untestable. The model hallucinates an ac- tion or event that never actually occurred in the source video.</td><td>36.2%</td></tr></table>

• ValidEventShotAssociation(e, h): If event e is linked to shot h, they must exhibit a strictly positive temporal overlap (IoU(time(e), time(h)) > 0). Failure trigger: An event is linked to a completely disjoint visual shot.

• TemporalCompatible(e, h): The duration of sub-elements, including subshots or active events, must be bounded by, or reasonably overlap with, their parent shot’s duration. Failure trigger: A subshot’s timestamp significantly exceeds the boundaries of its parent shot.

• ConsistentCoreference(r across H): An ID mapped to a persistent entity must map consistently across the video’s timeline without splitting into hallucinated variants. Failure trigger: A tracked character abruptly changes ID assignment across sequential cuts.

• SupportFieldPresent(r): Every reference r must possess its required evidence fields, most notably, the appearance\_anchor. Failure trigger: The prediction omits the appearance\_anchor dictionary entirely.

• NoDanglingID(S<sup>ˆ</sup>): No unreferenced or circular links exist in the final prediction graph. Failure trigger: A shot claims it contains an event ID that was never defined.

Table B.1 summarizes these probe families and their targeted failure conditions.

Isolating Structural Reasoning from Raw Detection. Table B.1 audits predictions strictly on their structural consistency. The high failure rates in Cross-reference integrity (65.2%) and Temporal compatibility (42.0%) highlight that the true bottleneck for current omnimodal architectures is not isolated frame perception, but rather building a coherent spatiotemporal graph. Models often hallucinate entity IDs or assign disjoint time boundaries even when the basic visual components are correctly detected. This validates our design of deterministic consistency probes: structural compliance must be verified programmatically before invoking LLMs for semantic validation, otherwise LLMs may be inadvertently prompted to score physically impossible or internally contradictory scenes.

## A.2.5 Alignment Protocol

Before calculating any structural metrics, we canonicalize raw model outputs into a standardized evaluation graph and then align prediction units to the ground-truth units. This pipeline explicitly tracks and preserves the structural intent behind raw JSON outputs. Canonicalization, temporal matching, and rule checks are rule-based; reference alignment is a bounded LLM-assisted one-to-one matching step over reference detail descriptions. All thresholds and tie-breaking rules for rule-based matching are fixed prior to computing model scores.

Table B.1: Structural Consistency Probe Types. Each row defines a deterministic probe family, its formal predicate, and the exact condition counted as a failure. Proportions are computed over the subset of structurally applicable predictions for the Gemini 3.1-Pro model to isolate structural reasoning from raw detection capability.
<table><tr><td>Probe type</td><td>Formal Predicate</td><td>Failure condition</td><td>Count / Prop.</td></tr><tr><td>Reference-use validity</td><td> $\mathrm { V a l i d R e f U s e } ( x , r )$ </td><td>Missing aligned region, dangling ID, or reference mismatch after</td><td>8649 / 34.1%</td></tr><tr><td>Temporal compatibility</td><td></td><td>alignment TemporalCompatible(e, h) Predicted interval is disjoint from the required shot/event support or violates ordering tolerance</td><td>12240 / 42.0%</td></tr><tr><td></td><td>Reference-support validity SupportFieldPresent(r)</td><td>Missing support field, mal- formed support field, or sup- port field incompatible with the</td><td>3778 / 31.7%</td></tr><tr><td>Cross-reference integrity</td><td>ConsistentCoreference(r)</td><td>matched reference Referenced ID is undefined, in- consistent across shots, or incom- patible with the matched persis- tent reference</td><td>2656 / 65.2%</td></tr></table>

Algorithm 1: Output Canonicalization and Structural Alignment   
Input: Predicted JSON V<sup>ˆ</sup> , Ground Truth $V ^ { \star }$   
Output: Matched units $M _ { R } , M _ { E } , M _ { H } , M _ { S }$   
1. V<sup>ˆ</sup> ← FixMalformedIDs(V<sup>ˆ</sup> ) // Resolve formatting errors   
2. V<sup>ˆ</sup> ← ClipTimeRanges $( \hat { V } , V _ { m a x \_ t i m e } ^ { \star } )$ // Clamp out-of-bounds timestamps   
3. R<sup>ˆ</sup>, E<sup>ˆ</sup>, H ←<sup>ˆ</sup> ParseAndFilterDanglingLinks(V<sup>ˆ</sup> )   
4. $M _ { R } \gets \mathrm { A l i g n R e f e r e n c e s } _ { L L M } ( \hat { \mathscr { R } } , \mathscr { R } ^ { \star } )$ // Bounded one-to-one over detail descriptions   
5. $M _ { E } \gets \mathrm { A l i g n E v e n t s } ( \hat { \mathcal { E } } , \mathcal { E } ^ { \star } )$ // WER for dialog, tIoU otherwise   
6. $M _ { H } \gets \mathrm { A l i g n S h o t s } ( \hat { \mathcal { H } } , \mathcal { H } ^ { \star } )$ // Temporal IoU   
7. M<sub>S</sub> ← LocalAlignSubshots(M<sub>H</sub>, δ = 1s)   
8. return ComputeMetricFamilies $( M _ { R } , M _ { E } , M _ { H } , M _ { S } )$

Cross-Track Linking Instruction (Excerpt)   
The most important step: Use the ‘active\_events‘ array to accurately link the current visual sho   
with the audio events happening simultaneously.   
- Every ‘event\_id‘ listed in a shot’s ‘active\_events‘ array MUST come from the previously defined   
‘events‘ list.   
- Fill this accurately based on temporal overlap: if any part of an event’s ‘time\_range‘ overlaps   
with the current shot’s ‘time\_range‘, that ‘event\_id‘ MUST be included in ‘active\_events‘.   
- CROSS-VALIDATION: Check that the union of ‘active\_events‘ across all shots covers every single   
‘event\_id‘ defined in the video. Do not leave any event dangling.

## B More Experimental Results

## B.1 Intra-Family Metric Correlation

A critical design goal of OmniCapBench is to provide fine-grained, decoupled diagnostic signals. To validate that our deep-structured atomic units measure distinct capabilities rather than redundant phenomena, we computed the instance-level (per-video) Pearson correlation among the metrics across all 786 test videos, using the predictions of Gemini 3.1-Pro. As illustrated in Figure B.1, we separate these into two distinct domains: Structural Metrics and Semantic Metrics.

The Structural Metrics Correlation heatmap reveals that fundamental spatial, temporal, and crossmodal capabilities are highly decoupled. For instance, the correlation between Event F1 (audio perception) and Shot F1 (visual temporal grounding) is remarkably low $( r = 0 . 2 5 )$ , indicating that models often succeed in one modality while failing in the other. While EVSA (audio-visual association) naturally correlates with both its audio $( r = 0 . 6 7 )$ and visual $( r = 0 . 6 2 )$ prerequisites, complex tracking metrics remain isolated. CCC (cross-shot tracking) shows almost no correlation with basic temporal grounding (Shot F1, $r \ : = \ : 0 . 1 7 )$ , proving that locating an action in time is fundamentally separate from persistently tracking the actor’s identity across discontinuous cuts.

The Semantic Metrics Correlation heatmap yields an equally crucial insight: descriptive fidelity is heavily fragmented. Strikingly, Subject Precision and Subject Recall exhibit zero correlation (r = −0.03). This proves that a model’s tendency to hallucinate vivid details (low precision) is completely decoupled from its ability to comprehensively cover all ground-truth facts (recall). Furthermore, crossdomain semantic scores, such as Audio descriptions versus Scene Recall $( r = - 0 . 0 2 )$ , are entirely uncorrelated. This quantitative analysis confirms that OmniCapBench’s suite of metrics effectively isolates distinct multimodal capabilities, exposing localized failures (e.g., severe hallucination despite high recall) that holistic scalars seamlessly average away.

![](images/4efb9577c95cc29e3be4f99823552bd38a036646e4ce99444f77b49740a846ae.jpg)

![](images/25fbf64f3bc754f1f89d1bbed6eb019f58bae86351d25394d8ed1b9cbd0a2206.jpg)  
Figure B.1: Instance-level correlation among diagnostic metrics. Pearson correlation coefficients computed across 786 videos (Gemini 3.1-Pro). Left: Structural correlations confirm that complex capabilities like identity tracking (CCC) are decoupled from basic temporal grounding (Shot F1, $r = 0 . 1 7 )$ . Right: Semantic LLM-as-judge scores exhibit near-zero correlation between Precision and Recall for Subject descriptions $( r = - 0 . 0 3 )$ , proving that hallucination tendency and factual coverage are independent failure modes.

## B.2 Detailed Setup

In Section 4, we argue that traditional dense caption evaluation using text-centric metrics can falsely equate distinct models. To empirically validate this claim, we conducted an experiment using cases from the UGC-VideoCap [43] and video-SALMONN 2 [34] benchmarks, both of which rely on whole-caption evaluation paradigms.

Specifically, we constructed a diagnostic subset of 200 videos to serve as a “blind spot” dataset for traditional metrics. The construction followed these steps:

1. Source Log Analysis: We collected the dense caption outputs and corresponding global text-level scores for distinct frontier and open-source models evaluated on UGC-VideoCap and video-SALMONN 2.

2. Filtering for Pseudo-Equivalence: We computed the variance in the text-level scores among the models for each individual video. We then specifically filtered for cases where the maximum score difference between models was strictly less than 10%.

3. Subset Construction: We sampled 200 such cases to form our final test set. By definition, traditional global text scoring evaluates these models as exhibiting nearly identical performance on this subset.

4. Structured Re-Evaluation: Finally, we evaluated the models’ structural comprehension on these exact 200 videos using OmniCapBench’s deep-structured atomic unit metrics.

As discussed in the main text and shown in Table B.7, while the models appeared identical under global text scoring, OmniCapBench’s localized metrics revealed significant underlying quality gaps. This confirms that global text metrics frequently reward linguistically fluent but structurally flawed outputs, degrading diagnostic resolution.

## B.3 Full Main Results

While the main text presents compact metric families to highlight macro-level capability gaps, rigorous diagnostic evaluation requires exposing the underlying structural submetrics. This section provides the fully expanded results for all evaluated models. Tables B.2 and B.3 separate the rule-based compact columns into their constituent Precision, Recall, F1, and tIoU components. For instance, expanding the Audio dimension reveals whether a low Event F1 score is driven by hallucination (low precision) or omission (low recall). Table B.4 reports the localized LLM-as-judge scores across specific descriptive fields. Finally, Tables B.5 and B.6 stratify performance by video duration (< 1 min, 1–3 min, and 3–5 min), testing whether models maintain identity persistence and temporal grounding over extended contexts. Numeric tables use the same column-wise bold/underline convention as the main paper (best and second-best, ties allowed).

Deep Analysis of Full Submetrics. By unpacking the compact F1 scores into their base Recall and Precision constituents, we uncover distinct behavioral paradigms across the models. In the Visual domain (Table B.2), Gemini 3.1-Pro and Gemini 2.5-Pro exhibit diverging strategies: Gemini 3.1-Pro dominates Reference Recall (77.02%) but lags slightly behind in Reference Precision (70.56%), suggesting an aggressive entity detection strategy. Conversely, Gemini 2.5-Pro achieves higher Precision (71.39%) but lower Recall. Furthermore, the structural parsing of shots reveals that while models can generally pinpoint when actions occur (Shot tIoU > 70%), their Subshot Precision plummets (e.g., Gemini 3.1-Pro drops from 88.77% for Shots to 60.29% for Subshots), proving that models hallucinate fine-grained temporal boundaries when forced to decompose continuous actions into structured sub-events.

In the Audio-Visual domain (Table B.3), the expanded EVSA (Event-Shot Association) metric isolates the cross-modal bottleneck. Gemini 3.1-Pro’s EVSA Precision is high (78.35%): when the model does link an audio event to a visual shot, the topological link is usually correct. Its EVSA Recall, however, is severely low (42.14%), confirming that the primary failure mode is omission—models perceive the audio and visual streams in isolation, miss their causal relationships, and drop the required cross-modal links entirely. This reinforces our core motivation: purely textbased evaluations cover up such missing links with generic paragraphs, whereas OmniCapBench structurally mandates them.

## B.4 Case Studies

Our core motivation is to evaluate video captions through explicit, verifiable units rather than through a single holistic text score. This section presents two qualitative case studies that illustrate what this structured representation makes observable. The first case focuses on identity persistence: whether a model uses the same entity identifier for the same visual subject across temporally separated shots. The second case focuses on event-shot association: whether a recognized audio event is linked to the visual shot that supports its source. Together, these examples show how OmniCapBench turns broad caption quality into localized diagnostic evidence.

Table B.2: Expanded Visual rule-based submetrics. Ref R/P are reference recall and precision. RefUse-shot and RefUse-sub report WHO\_shot and WHO\_sub. CCC is the macro average of cross-shot reference-trajectory IoU. Shot/Subshot R/P report alignment recall and precision, and tIoU reports temporal overlap (WHEN\_H, WHEN\_S). Within each column, bold is best and underline is second-best among models (ties allowed).
<table><tr><td>Model</td><td colspan="9">RefUse-shot RefUse-sub CCC Ref R Ref P Shot R Shot P</td><td>Subshot R Subshot P Subshot tIoU</td><td></td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Gemini 3.1-Pro [15]</td><td>68.49</td><td>68.77</td><td>34.43</td><td>77.02</td><td>70.56</td><td>71.73</td><td>88.77</td><td>70.74</td><td>75.40</td><td>60.29</td><td>48.76</td></tr><tr><td>Gemini 2.5-Pro [10]</td><td>67.01</td><td>71.42</td><td>37.81</td><td>76.96</td><td>71.39</td><td>66.29</td><td>86.27</td><td>72.58</td><td>70.26</td><td>66.89</td><td>49.18</td></tr><tr><td>Qwen3.5-Omni-Plus [32]</td><td>63.99</td><td>65.22</td><td>29.55</td><td>65.02</td><td>70.99</td><td>63.90</td><td>89.62</td><td>72.35</td><td>69.00</td><td>66.66</td><td>48.29</td></tr><tr><td>Qwen3.5-Omni-Flash [32]</td><td>59.39</td><td>61.07</td><td>24.31</td><td>59.50</td><td>69.26</td><td>55.77</td><td>89.71</td><td>68.99</td><td>61.93</td><td>60.45</td><td>46.52</td></tr><tr><td colspan="10">Open-source Omnimodal Models</td></tr><tr><td>Qwen3-Omni-Instruct [49]</td><td>37.97</td><td>50.37</td><td>9.85</td><td>50.63</td><td>53.53</td><td>45.98</td><td>64.36</td><td>67.92</td><td>35.80</td><td>28.79</td><td>47.95</td></tr><tr><td>Qwen3-Omni-Captioner [49]</td><td>40.88</td><td>50.80</td><td>11.51</td><td>49.44</td><td>46.97</td><td>47.79</td><td>71.63</td><td>70.80</td><td>44.62</td><td>42.29</td><td>49.69</td></tr></table>

Table B.3: Expanded Audio and Audio-Visual rule-based submetrics. Within each column, bold is best and underline is second-best among models (ties allowed).
<table><tr><td rowspan="2">Model</td><td colspan="3">Audio</td><td colspan="4">Audio-Visual</td></tr><tr><td>Event R</td><td>Event P</td><td>Event tIoU</td><td>Speaker</td><td>EVSA-P</td><td>EVSA-R</td><td>EVSA</td></tr><tr><td colspan="8">Proprietary Models</td></tr><tr><td>Gemini 3.1-Pro [15]</td><td>63.96</td><td>67.20</td><td>71.92</td><td>91.18</td><td>78.35</td><td>42.14</td><td>51.46</td></tr><tr><td>Gemini 2.5-Pro [10]</td><td>58.09</td><td>64.23</td><td>69.35</td><td>89.72</td><td>74.05</td><td>35.04</td><td>43.74</td></tr><tr><td>Qwen3.5-Omni-Plus [32] Qwen3.5-Omni-Flash [32]</td><td>51.22 48.01</td><td>62.53 58.91</td><td>67.77</td><td>91.20</td><td>71.11 66.28</td><td>31.82 27.69</td><td>40.20</td></tr><tr><td></td><td></td><td></td><td>69.34</td><td>89.91</td><td></td><td></td><td>35.58</td></tr><tr><td colspan="8">Open-source Omnimodal Models</td></tr><tr><td>Qwen3-Omni-Instruct [49]</td><td>39.73</td><td>49.67</td><td>40.99</td><td>88.35</td><td>28.26</td><td>9.92</td><td>13.33</td></tr><tr><td>Qwen3-Omni-Captioner [49]</td><td>39.64</td><td>50.07</td><td>52.60</td><td>88.68</td><td>36.70</td><td>13.00</td><td>17.42</td></tr></table>

Table B.4: Expanded LLM-as-judge submetrics. Dialogue is the rule-based dialogue line structural score. Within each column, bold is best and underline is second-best among models (ties allowed).
<table><tr><td rowspan="3">Model</td><td colspan="7">Visual</td><td colspan="2">Audio</td></tr><tr><td colspan="3">Subject Ref</td><td colspan="3">Scene Ref</td><td>Shot Fact</td><td>Dialogue</td><td>Non-dialogue</td></tr><tr><td>R</td><td>P</td><td>Mean</td><td>R</td><td>P</td><td>Mean</td><td>Mean</td><td>Mean</td><td>Mean</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Gemini 3.1-Pro [15]</td><td>43.78</td><td>67.07</td><td>55.43</td><td>51.78</td><td>75.62</td><td>63.70</td><td>59.05</td><td>91.18</td><td>41.49</td></tr><tr><td>Gemini 2.5-Pro [10]</td><td>51.53</td><td>57.08</td><td>54.30</td><td>56.80</td><td>65.12</td><td>60.96</td><td>55.58</td><td>89.72</td><td>37.38</td></tr><tr><td>Qwen3.5-Omni-Plus [32]</td><td>45.41</td><td>62.84</td><td>54.13</td><td>50.75</td><td>69.80</td><td>60.27</td><td>54.17</td><td>91.20</td><td>36.82</td></tr><tr><td>Qwen3.5-Omni-Flash [32]</td><td>43.87</td><td>65.59</td><td>54.73</td><td>44.41</td><td>71.26</td><td>57.84</td><td>47.66</td><td>89.91</td><td>28.12</td></tr><tr><td colspan="10">Open-source Omnimodal Models</td></tr><tr><td>Qwen3-Omni-Instruct [49]</td><td>40.31</td><td>62.02</td><td>51.16</td><td>36.07</td><td>70.73</td><td>53.40</td><td>34.63</td><td>88.35</td><td>20.51</td></tr><tr><td>Qwen3-Omni-Captioner [49]</td><td>42.00</td><td>50.96</td><td>46.48</td><td>38.07</td><td>54.53</td><td>46.30</td><td>36.08</td><td>88.68</td><td>19.65</td></tr></table>

## B.4.1 Identity Persistence: Cross-Shot Coreference Consistency

Cross-Shot Coreference Consistency (CCC) measures whether a model maintains a stable identity assignment for the same entity across multiple shots. This is different from detecting that a person or object appears somewhere in the video. A model may recognize the entity locally in each shot, but still fragment the entity into different identifiers after viewpoint changes, occlusions, or temporal gaps. CCC therefore evaluates whether the predicted reference graph is temporally consistent.

Table B.5: Per-duration rule-based structural results. Stratification follows the three duration buckets (< 1 min, 1–3 min, and 3–5 min); Ref Subj./Ref S. are recomputed per video in each bucket as in the main table. Within each column, bold is best and underline is second-best among models (ties allowed).
<table><tr><td rowspan="2">Model</td><td rowspan="2">SGC</td><td colspan="8"></td><td colspan="2">Audio</td><td colspan="2">Audio-Visual</td></tr><tr><td>RefUse CCC Ref Subj. F1 Ref S. F1 Shot F1 Shot tIoU Sub F1 Sub tIoU Evt F1 Evt tIoU Spk F1 EVSA F1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Duration: &lt;1 min</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1-Pro</td><td>97.46</td><td>74.13</td><td>38.41</td><td>83.09</td><td>86.54</td><td>77.90</td><td>69.76</td><td>67.55</td><td>47.73</td><td>63.09</td><td>70.65</td><td>91.15</td><td>53.30</td></tr><tr><td>Gemini 2.5-Pro</td><td>97.57</td><td>73.40</td><td>40.60</td><td>84.80</td><td>87.39</td><td>76.61</td><td>71.72</td><td>70.94</td><td>48.36</td><td>60.46</td><td>67.46</td><td>89.08</td><td>49.31</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>96.90</td><td>70.94</td><td>32.71</td><td>80.28</td><td>84.99</td><td>77.60</td><td>73.57</td><td>69.65</td><td>48.73</td><td>57.12</td><td>64.57</td><td>90.77</td><td>46.87</td></tr><tr><td>Qwen3.5-Omni-Flash Qwen3-Omni-Instruct</td><td>96.26</td><td>66.83</td><td>26.81</td><td>77.37</td><td>66.70</td><td>72.86</td><td>69.94</td><td>64.57</td><td>47.07</td><td>52.72</td><td>66.90</td><td>89.34</td><td>41.65</td></tr><tr><td>Qwen3-Omni-Captioner</td><td>93.91 94.15</td><td>50.85 54.17</td><td>13.42 16.45</td><td>68.94 69.03</td><td>76.29 76.25</td><td>40.06 51.50</td><td>52.58 55.64</td><td>27.29 42.08</td><td>38.83 41.09</td><td>44.41 46.57</td><td>47.30 55.78</td><td>88.32 88.54</td><td>17.09</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>24.51</td></tr><tr><td colspan="10">Duration: 1–3 min</td><td colspan="3"></td></tr><tr><td>Gemini 3.1-Pro</td><td>97.84</td><td>65.61</td><td>29.59</td><td>72.09</td><td>76.14</td><td>78.87</td><td>73.84</td><td>65.76</td><td>52.17</td><td>67.31</td><td>77.44</td><td>92.22</td><td>54.76</td></tr><tr><td>Gemini 2.5-Pro</td><td>96.47</td><td>66.99</td><td>35.43</td><td>72.70</td><td>77.32</td><td>69.25</td><td>75.17</td><td>66.68</td><td>52.01</td><td>57.77</td><td>73.12</td><td>91.06</td><td>39.80</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>95.52</td><td>60.98</td><td>25.64</td><td>64.09</td><td>70.40</td><td>68.70</td><td>71.65</td><td>65.75</td><td>49.32</td><td>53.40</td><td>72.80</td><td>92.07</td><td>35.03</td></tr><tr><td>Qwen3.5-Omni-Flash</td><td>94.99</td><td>56.99</td><td>21.17</td><td>63.76</td><td>63.72</td><td>61.31</td><td>68.57</td><td>55.76</td><td>47.21</td><td>49.38</td><td>75.16</td><td>90.83</td><td>29.70</td></tr><tr><td>Qwen3-Omni-Instruct</td><td>83.25</td><td>26.52</td><td>4.51</td><td>35.84</td><td>48.05</td><td>67.62</td><td>94.30</td><td>35.87</td><td>63.60</td><td>38.09</td><td>32.53</td><td>88.84</td><td>9.12</td></tr><tr><td>Qwen3-Omni-Captioner</td><td>86.87</td><td>33.13</td><td>4.82</td><td>36.29</td><td>32.86</td><td>55.32</td><td>93.30</td><td>37.98</td><td>63.48</td><td>34.25</td><td>49.30</td><td>88.72</td><td>8.98</td></tr><tr><td colspan="10">Duration: 3–5 min</td><td colspan="3"></td></tr><tr><td>Gemini 3.1-Pro</td><td>95.31</td><td>60.57</td><td>23.21</td><td>69.11</td><td>77.09</td><td>57.79</td><td>67.76</td><td>49.21</td><td>45.18</td><td>53.01</td><td>63.06</td><td>88.26</td><td>30.02</td></tr><tr><td>Gemini 2.5-Pro</td><td>92.12</td><td>64.89</td><td>27.02</td><td>78.16</td><td>78.35</td><td>39.43</td><td>70.38</td><td>39.42</td><td>46.03</td><td>48.22</td><td>69.53</td><td>89.24</td><td>19.83</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>92.55</td><td>58.82</td><td>20.96</td><td>58.84</td><td>76.43</td><td>38.10</td><td>66.63</td><td>40.08</td><td>42.37</td><td>34.95</td><td>73.16</td><td>90.95</td><td>13.14</td></tr><tr><td>Qwen3.5-Omni-Flash</td><td>93.04</td><td>57.36</td><td>17.62</td><td>64.84</td><td>69.00</td><td>32.29</td><td>64.08</td><td>32.30</td><td>40.74</td><td>41.44</td><td>66.60</td><td>89.87</td><td>14.36</td></tr><tr><td>Qwen3-Omni-Instruct</td><td>76.59</td><td>24.15</td><td>4.18</td><td>26.16</td><td>42.34</td><td>57.39</td><td>95.00</td><td>23.98</td><td>62.77</td><td>36.12</td><td>25.88</td><td>86.97</td><td>3.28</td></tr><tr><td>Qwen3-Omni-Captioner 83.97</td><td></td><td>29.92</td><td>4.44</td><td>28.07</td><td>23.09</td><td>47.58</td><td>93.51</td><td>29.99</td><td>63.43</td><td>33.67</td><td>45.74</td><td>89.16</td><td>4.16</td></tr></table>

Table B.6: Per-duration local semantic-equivalence score results. Stratification follows the three duration buckets (< 1 min, 1–3 min, and 3–5 min). Within each column, bold is best and underline is second-best among models.
<table><tr><td rowspan="3">Model</td><td colspan="7">Visual</td><td colspan="2">Audio</td></tr><tr><td rowspan="2">Shot</td><td colspan="2">Subject</td><td colspan="2">Scene</td><td colspan="2">Subshot</td><td rowspan="2"></td><td rowspan="2">Dialogue Non-dialogue</td></tr><tr><td>Recall</td><td>Precision</td><td>Recall</td><td>Precision</td><td>Recall</td><td>Precision</td></tr><tr><td colspan="10">Duration: &lt;1 min</td></tr><tr><td>Gemini 3.1-Pro</td><td>62.84</td><td>46.74</td><td>68.18</td><td>52.62</td><td>78.01</td><td>53.90</td><td>71.75</td><td>91.15</td><td>43.39</td></tr><tr><td>Gemini 2.5-Pro</td><td>59.70</td><td>54.97</td><td>57.11</td><td>57.66</td><td>67.97</td><td>54.19</td><td>72.40</td><td>89.08</td><td>40.03</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>58.92</td><td>48.27</td><td>62.41</td><td>51.83</td><td>71.96</td><td>51.88</td><td>70.08</td><td>90.77</td><td>38.23</td></tr><tr><td>Qwen3.5-Omni-Flash</td><td>51.38</td><td>44.16</td><td>62.10</td><td>46.15</td><td>72.85</td><td>45.58</td><td>66.84</td><td>89.34</td><td>28.94</td></tr><tr><td>Qwen3-Omni-Instruct</td><td>39.62</td><td>39.99</td><td>62.06</td><td>36.46</td><td>73.31</td><td>34.64</td><td>65.12</td><td>88.32</td><td>23.71</td></tr><tr><td>Qwen3-Omni-Captioner</td><td>42.82</td><td>41.44</td><td>51.46</td><td>38.73</td><td>56.76</td><td>36.28</td><td>49.78</td><td>88.54</td><td>23.21</td></tr><tr><td colspan="10">Duration: 1–3 min</td></tr><tr><td>Gemini 3.1-Pro</td><td>59.05</td><td>40.96</td><td>65.67</td><td>51.34</td><td>73.61</td><td>54.35</td><td>70.90</td><td>92.22</td><td>40.03</td></tr><tr><td>Gemini 2.5-Pro</td><td>54.75</td><td>49.53</td><td>57.48</td><td>56.33</td><td>62.88</td><td>52.95</td><td>70.14</td><td>91.06</td><td>33.44</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>53.57</td><td>42.86</td><td>63.97</td><td>50.39</td><td>68.11</td><td>50.20</td><td>71.75</td><td>92.07</td><td>34.91</td></tr><tr><td>Qwen3.5-Omni-Flash</td><td>47.16 34.11</td><td>43.29</td><td>68.36</td><td>42.76</td><td>69.67</td><td>44.45</td><td>67.48</td><td>90.83</td><td>24.12</td></tr><tr><td>Qwen3-Omni-Instruct Qwen3-Omni-Captioner</td><td>34.85</td><td>40.66</td><td>62.37</td><td>35.70</td><td>66.24</td><td>30.31</td><td>47.68</td><td>88.84</td><td>12.50</td></tr><tr><td></td><td></td><td>42.64</td><td>50.00</td><td>38.03</td><td>50.59</td><td>33.82</td><td>42.20</td><td>88.72</td><td>9.38</td></tr><tr><td colspan="10">Duration: 3–5 min</td></tr><tr><td>Gemini 3.1-Pro</td><td>54.15</td><td>43.41</td><td>68.01</td><td>49.32</td><td>71.23</td><td>49.68</td><td>68.88</td><td>88.26</td><td>33.85</td></tr><tr><td>Gemini 2.5-Pro</td><td>48.07</td><td>47.52</td><td>55.83</td><td>54.59</td><td>60.37</td><td>50.25</td><td>70.16</td><td>89.24</td><td>35.29</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>43.49</td><td>43.09</td><td>60.71</td><td>46.43</td><td>64.29</td><td>45.43</td><td>69.19</td><td>90.95</td><td>34.03</td></tr><tr><td>Qwen3.5-Omni-Flash</td><td>39.16</td><td>44.60</td><td>69.17</td><td>42.20</td><td>69.50</td><td>38.80</td><td>68.37</td><td>89.87</td><td>32.81</td></tr><tr><td>Qwen3-Omni-Instruct</td><td>33.04</td><td>41.13</td><td>60.22</td><td>34.43</td><td>68.85</td><td>23.54</td><td>63.59</td><td>86.97</td><td>14.71</td></tr><tr><td>Qwen3-Omni-Captioner</td><td>32.36</td><td>42.13</td><td>52.36</td><td>30.49</td><td>48.17</td><td>30.81</td><td>46.04</td><td>89.16</td><td>21.30</td></tr></table>

Figure B.2 shows that CCC provides a direct diagnostic signal for identity persistence. The key comparison is not whether a model can mention the visible subject, but whether it preserves the same identity assignment over time. In the displayed example, three models maintain relatively consistent identity traces, with CCC scores above 76, while Qwen3-Omni-Instruct obtains 30.77. This indicates

![](images/0f92900f04f917d86bf481040fff7535d43fed3d403f69a15643acc904aa1a57.jpg)  
Figure B.2: Case study of identity persistence measured by CCC. The timeline compares identity assignments across models over the same video interval. Colored spans show ground-truth and predicted identity coverage. Higher CCC indicates more consistent cross-shot identity reuse; lower CCC indicates identity fragmentation. In this case, Gemini 3.1 Pro, Gemini 2.5 Pro, and Qwen3.5- Omni-Plus score 76.92, 78.10, and 80.00, while Qwen3-Omni-Instruct scores 30.77, revealing a clear failure in preserving identity continuity.

that its prediction is more fragmented at the structural level. The implication is that CCC isolates a specific failure mode: identity drift across the video timeline. This failure would be difficult to identify from a fluent caption alone, because a paragraph can describe the visible subject in each shot without exposing whether the model has maintained a stable entity graph.

## B.4.2 Audio-Visual Association: Event-Shot Association

Event-Shot Association (EVSA) evaluates a different type of structure. It checks whether each audio event is linked to the visual shot or shots that support its source. This distinction matters because recognizing an audio category and grounding that audio category in the concurrent visual timeline are different capabilities. A model may detect that a dialogue or mechanical sound exists, but still omit the shot-level edge that makes the audio source verifiable.

Case study of event-shot association measured by EVSA   
Shots:   
• Shot 1 [0.0–0.6]: static outdoor medium shot.   
• Shot 2 [0.6–3.2]: close-up high-angle view into an open wooden hive.   
• Shot 3 [3.2–5.5]: indoor medium shot with PERSON\_1 speaking.   
• Shot 4 [5.5–8.1]: close-up of the top mechanism of the object.   
• Shot 5 [8.1–10.0]: extreme close-up of a white plastic gate valve.   
Audio events:   
• DIALOGUE\_1 [0.0–9.8]: PERSON\_1 speaks in an instructional tone.   
• SFX\_1 [5.5–8.0]: mechanical whirring and rhythmic geared turning.   
Ground-truth event-shot links: DIALOGUE\_1 → Shots 1–5; SFX\_1 → Shot 4.   
Model predictions:   
• Gemini 3.1-Pro (F1=1.000): all six links are correct, including SFX\_1 → Shot 4.   
• Gemini 2.5-Pro (F1=0.909): all dialogue links are correct, but SFX\_1 → Shot 4 is missed.   
• Qwen3.5-Omni-Plus (F1=0.909): all dialogue links are correct, but SFX\_1 → Shot 4 is missed.   
• Qwen3.5-Omni-Flash (F1=0.727): detects SFX\_1 → Shot 4, but adds the false link SFX\_1 → Shot 3 and misses DIALOGUE\_1 →   
Shots 4–5.   
• Qwen3-Omni-Instruct (F1=0.800): keeps DIALOGUE\_1 → Shots 1–4, but misses DIALOGUE\_1 → Shot 5 and SFX\_1 → Shot 4.   
• Qwen3-Omni-Captioner (F1=0.800): follows the same omission pattern as Qwen3-Omni-Instruct, missing the final dialogue shot   
and the mechanical SFX link.   
This example separates two EVSAfailure modes. Gemini 2.5-Pro and Qwen3.5-Omni-Plus preserve the long dialogue-to-shot coverage   
but omit the localized mechanical sound link. Qwen3.5-Omni-Flash instead detects the mechanical sound link but over-extends it to   
a neighboring shot and loses later dialogue coverage. These errors are small graph edits, but they change which visual evidence is   
considered to support each audio event.

This case study shows why EVSA is evaluated as explicit event-shot edges rather than as part of a holistic caption score. The dialogue event spans almost the full clip and should therefore be linked to all five shots. This is an easy relation to hide in prose, because a caption can simply state that a person is explaining the process throughout the video. EVSA makes the requirement explicit: each covered shot must be represented as a separate verifiable edge. The mechanical sound is more localized. It occurs from 5.5 to 8.0 seconds and is supported by Shot 4, so missing SFX\_1 → Shot 4 indicates an omission of a short cross-modal relation, while predicting SFX\_1 → Shot 3 indicates temporal over-extension. The six model traces therefore expose three distinct behaviors: Gemini 3.1-Pro recovers the full graph, Gemini 2.5-Pro and Qwen3.5-Omni-Plus miss the localized SFX edge, and the remaining Qwen variants either truncate the long dialogue coverage or add an unsupported neighboring SFX edge.

Taken together, CCC and EVSA reveal two complementary forms of structured video understanding. CCC tests whether a visual entity remains the same node across discontinuous shots; EVSA tests whether an audio event is attached to the correct visual evidence in the timeline. Both metrics convert a fluent caption into checkable graph constraints. The resulting insight is that current models can often describe local content, but still fail when the evaluation asks whether references and events are consistently bound across time and modality. This is precisely the failure mode that a global caption-quality score would tend to smooth over.

## B.5 Analysis of Specialized Captioning Models

As stated in Section 4.1, we purposefully excluded highly specialized video-captioning models (such as ASID-Caption [25], AVoCaDO [6], TimeChat-Captioner [59], and UGC-VideoCaptioner [43]) from the primary capability benchmarking. This exclusion is not because these models are incapable of perceiving videos, but because their rigorous supervised fine-tuning (SFT) heavily biases them toward generating free-form, holistic text paragraphs.

This intensive, domain-specific linguistic alignment degrades their general instruction-following capabilities in ways that are visible before any semantic scoring is applied. When required to output the deep-structured atomic units of OmniCapBench (i.e., a structured JSON with Reference, Event, and Shot arrays), representative outputs show four recurring failure modes, separated into instruction-following failures in Figure B.3 and structural-output failures in Figure B.4:

• Schema rejection: ASID-Caption models often ignore the JSON contract, emitting timestamped prose. In one 17.1-second clip, the raw response repeats its description template until At 1037s, while the parsed references, events, and shots arrays remain empty (Figure B.3(a)).

• Fixed caption schema: TimeChat-Captioner does produce machine-readable JSON, but it follows its own captioning template with fields such as timestamp, segment\_detail\_caption, camera\_state, storyline, and shooting\_style. These fields are fluent but do not instantiate the OmniCapBench evaluation graph: there are no persistent reference IDs, no event objects, and no shot-level cross-modal links (Figure B.3(b)).

• Incomplete graph construction: AVoCaDO superficially follows the requested key names, producing two references and two shots, but leaves events empty, assigns repeated zero-length sub\_shot\_time spans, and links shots to an undefined SCENE\_1. The resulting object is a captionlike shot summary rather than a connected Reference–Event–Shot graph (Figure B.4(a)).

• Degenerate structured output: UGC-VideoCaptioner also emits target keys, but the structure is not usable. It repeatedly assigns sub\_shot\_time=[0.0,0.0] and restates near-identical subject facts such as [PERSON\_1] is wearing a white shirt and blue jeans or [PERSON\_1] is holding a metal tool to remove honeycomb frames (Figure B.4(b)).

These failures occur at the interface between instruction following and structural generation, not at the final semantic-matching stage. The outputs either cannot be parsed into OmniCapBench units, follow a model-specific caption template, or collapse the temporal event graph into disconnected, repeated fragments. Reporting downstream structural scores would therefore conflate video perception with schema compliance; we instead treat these traces as evidence that caption-specialized SFT does no provide the controllable, graph-structured generation required by OmniCapBench.

![](images/ec741b35f8c6b08db07ac57228b51c46d2080c4d9321b640840dfdfcff0f553e.jpg)  
(a) ASID-Caption ignores the target schema and continues a repetitive timestamped prose loop far beyond the 17.1-second video.  
(b) TimeChat-Captioner returns its fixed dense-caption format instead of the requested OmniCapBench unit schema.  
Figure B.3: Instruction-following failures induced by caption-specialized SFT. The two models either reject the requested JSON contract outright or satisfy only the generic request for machinereadable captions while missing the required Reference–Event–Shot structure.

![](images/a4ca6f5fc8feadd00ce2a301173358172dcd86eaeedecd946669b1a8124d3d1c.jpg)  
(a) AVoCaDO uses several target field names, but leaves the event layer empty and produces disconnected shot summaries with undefined references and zero-length sub-shot spans.  
(b) UGC-VideoCaptioner produces target-looking JSON, but temporal grounding collapses and the visual descriptions repeat the same subject attributes and actions.  
Figure B.4: Structural-output failures after partial schema compliance. Even when captionspecialized models emit OmniCapBench-like keys, the generated objects do not form usable evaluation graphs: events are missing, references can be undefined, and local descriptions collapse into repeated zero-duration fragments.

Table B.7: The illusion of global text scores. We construct two hard-to-distinguish subsets where a weaker baseline and a stronger upgraded model exhibit near-identical text-centric scores. Across these pairs, we track performance through progressive strictness: (1) traditional text-centric metrics on dense captions, (2) strict structural verification using our native atomic units, and (3) human audit. Global text metrics falsely penalize the stronger models and overestimate the weaker ones, creating an illusion of parity. However, evaluating structural constraints directly resolves this gap, perfectly aligning with human judgment.
<table><tr><td>Model</td><td>Text Metric (Prose)</td><td>Native Constraints</td><td>Human Audit</td></tr><tr><td colspan="4">Pair 1: Qwen3.5-Omni-Flash vs. Qwen3-Omni</td></tr><tr><td>Weaker (Qwen3-Omni)</td><td>52.2</td><td>37.5</td><td>21.6</td></tr><tr><td>Stronger (Qwen3.5-Omni-Flash)</td><td>47.8</td><td>62.5</td><td>78.4</td></tr><tr><td colspan="4">Pair 2: Qwen3.5-Omni-Plus vs. Qwen3.5-Omni-Flash</td></tr><tr><td>Weaker (Qwen3.5-Omni-Flash)</td><td>41.7</td><td>21.4</td><td>29.9</td></tr><tr><td>Stronger (Qwen3.5-Omni-Plus)</td><td>58.3</td><td>78.6</td><td>70.1</td></tr><tr><td>Average (Stronger Models)</td><td>53.0</td><td>70.6</td><td>74.2</td></tr><tr><td>Average (Weaker Models)</td><td>47.0</td><td>29.4</td><td>25.8</td></tr></table>

Table B.8: Native generation vs. the post-hoc parsing track (rule-based metrics). Native asks the model for OmniCapBench units directly; Post-hoc lets the model write a free-form caption that a fixed, video-blind LLM parser converts into units. Post-hoc raises structural well-formedness (SGC, Shot F1, Shot tIoU) while every grounding-dependent metric (CCC, Event F1, EVSA) degrades, showing that format repair does not recover information the caption never contained.
<table><tr><td>Model</td><td>Track</td><td>SGC</td><td>RefUse</td><td>CCC</td><td>Ref Subj</td><td>Shot F1</td><td>Shot tIoU</td><td>Evt F1</td><td>Evt tIoU</td><td>Speaker</td><td>EVSA</td></tr><tr><td>Gemini 3.1-Pro [15]</td><td>Native Post-hoc</td><td>97.36 99.19</td><td>70.39 46.91</td><td>34.43 5.20</td><td>78.60 57.11</td><td>76.24 97.39</td><td>70.74 95.62</td><td>63.32 40.73</td><td>71.92 56.65</td><td>91.18 87.36</td><td>51.46 38.29</td></tr><tr><td>Qwen3-Omni-Instruct [49]</td><td>Native Post-hoc</td><td>89.02 96.61</td><td>40.96 40.93</td><td>9.85 9.62</td><td>54.84 49.21</td><td>49.98 96.44</td><td>67.92 95.59</td><td>41.70 12.82</td><td>40.99 54.40</td><td>88.35 81.65</td><td>13.33 12.23</td></tr><tr><td>ASID-Captioner-7B [25]</td><td>Post-hoc</td><td>96.17</td><td>39.58</td><td>9.65</td><td>41.89</td><td>96.51</td><td>95.73</td><td>11.09</td><td>54.83</td><td>87.85</td><td>10.87</td></tr></table>

## B.6 Post-hoc Parsing Baseline

Appendix B.5 shows that prose-optimized captioners cannot emit OmniCapBench units directly, which raises a fair question: do their low scores measure perception, or only instruction following? To separate the two, we add a fixed post-hoc track. A single LLM parser receives the model’s free-form caption and never the video, converts that caption into OmniCapBench units, and marks unsupported fields as unknown (Listing B.2). The parser is identical for every model, so it contributes no visual evidence and no per-model tuning.

Tables B.8 and B.9 show that this route buys format compliance, not information. Structural wellformedness rises: SGC improves from 97.36 to 99.19 for Gemini 3.1-Pro, and Shot F1 and Shot tIoU exceed 95 for every post-hoc entry, because a text-only parser emits clean, non-overlapping segments by construction. Every metric that requires grounding to the video degrades at the same time. Gemini’s CCC falls from 34.43 to 5.20 and its EVSA from 51.46 to 38.29, Qwen3-Omni-Instruct’s Event F1 falls from 41.70 to 12.82, and the localized semantic scores drop across the board, for instance Gemini’s Shot score from 59.05 to 35.84. This joint pattern is itself a diagnostic result: near-saturated shot metrics paired with single-digit CCC is precisely the profile that a global caption score hides and that our deep-structured units are built to expose.

The post-hoc track also makes specialized captioners evaluable instead of simply scoring them zero. ASID-Captioner-7B now produces valid units and reaches 96.17 SGC, yet stays at 9.65 CCC, 11.09 Event F1, and 10.87 EVSA. Format repair therefore widens the benchmark’s coverage, but the cross-shot identity and audio-visual relations that OmniCapBench targets are absent from the original dense captions and cannot be reconstructed after the fact.

Table B.9: Native generation vs. the post-hoc parsing track (localized LLM-judge metrics). Every local semantic score drops under post-hoc conversion, including for the caption-specialized model that the track is meant to accommodate. Precision rises on some fields only because the parser emits fewer, more conservative units.
<table><tr><td>Model</td><td>Track</td><td>Shot</td><td>Subj-R</td><td>Subj-P</td><td>Scene-R</td><td>Scene-P</td><td>Subsh-R</td><td>Subsh-P</td><td>Dialogue</td><td>Non-dial</td></tr><tr><td rowspan="2">Gemini 3.1-Pro [15]</td><td>Native</td><td>59.05</td><td>43.78</td><td>67.07</td><td>51.78</td><td>75.62</td><td>53.29</td><td>70.92</td><td>91.18</td><td>41.49</td></tr><tr><td>Post-hoc</td><td>35.84</td><td>24.75</td><td>75.90</td><td>24.72</td><td>82.59</td><td>36.15</td><td>62.94</td><td>87.36</td><td>21.42</td></tr><tr><td rowspan="2">Qwen3-Omni-Instruct [49]</td><td>Native</td><td>34.63</td><td>40.31</td><td>62.02</td><td>36.07</td><td>70.73</td><td>29.42</td><td>57.84</td><td>88.35</td><td>20.51</td></tr><tr><td>Post-hoc</td><td>28.67</td><td>28.49</td><td>71.82</td><td>29.32</td><td>77.37</td><td>33.77</td><td>50.75</td><td>81.65</td><td>13.15</td></tr><tr><td>ASID-Captioner-7B [25]</td><td>Post-hoc</td><td>22.87</td><td>25.24</td><td>65.66</td><td>32.96</td><td>64.35</td><td>33.25</td><td>40.17</td><td>87.85</td><td>10.37</td></tr></table>

Table B.10: Matching-threshold sweep. Each block varies one cutoff while predictions, references, and the one-to-one matching rule stay fixed; no inference is re-run. Default settings are shown in bold. Absolute scores shift with the cutoff, but the induced ranking does not: Spearman’s ρ against the default ranges from 0.914 to 1.000 (Appendix B.7).
<table><tr><td>Sweep</td><td>Metric</td><td>Setting</td><td>Gemini 3.1-Pro</td><td>Gemini 2.5-Pro</td><td>Qwen3.5- Omni-Plus</td><td>Qwen3.5- Omni-Flash</td><td>Qwen3-Omni- Instruct</td><td>Qwen3-Omni- Captioner</td></tr><tr><td rowspan="3">Non-dialogue tIoU</td><td rowspan="3">Event F1</td><td>0.10</td><td>64.02</td><td>59.54</td><td>54.63</td><td>51.43</td><td>42.29</td><td>41.88</td></tr><tr><td>0.20</td><td>63.32</td><td>58.52</td><td>53.92</td><td>50.68</td><td>41.70</td><td>41.31</td></tr><tr><td>0.40</td><td>61.86</td><td>56.34</td><td>52.38</td><td>48.95</td><td>40.71</td><td>40.07</td></tr><tr><td rowspan="3">Shot tIoU</td><td rowspan="3">Shot F1</td><td>0.20</td><td>77.84</td><td>72.32</td><td>72.96</td><td>67.81</td><td>54.22</td><td>55.34</td></tr><tr><td>0.30</td><td>76.24</td><td>70.93</td><td>71.25</td><td>65.65</td><td>49.98</td><td>52.29</td></tr><tr><td>0.50</td><td>64.07</td><td>60.87</td><td>60.37</td><td>52.79</td><td>37.41</td><td>38.82</td></tr><tr><td rowspan="3">Subshot window δ</td><td rowspan="3">Subshot F1</td><td>0.5s</td><td>64.64</td><td>66.07</td><td>64.79</td><td>57.91</td><td>29.18</td><td>38.89</td></tr><tr><td>1.0 s</td><td>65.27</td><td>66.69</td><td>65.69</td><td>58.94</td><td>29.51</td><td>39.51</td></tr><tr><td>2.0s</td><td>65.90</td><td>67.35</td><td>66.38</td><td>59.92</td><td>29.78</td><td>40.17</td></tr><tr><td rowspan="3">Dialogue 1 – WER</td><td rowspan="3">Speaker</td><td>0.40</td><td>89.92</td><td>88.10</td><td>89.16</td><td>87.54</td><td>85.82</td><td>86.64</td></tr><tr><td>0.50</td><td>91.18</td><td>89.72</td><td>91.20 93.88</td><td>89.91</td><td>88.35</td><td>88.68</td></tr><tr><td>0.60</td><td>94.08</td><td>92.57</td><td></td><td>92.84</td><td>91.46</td><td>91.62</td></tr></table>

## B.7 Threshold Sensitivity

OmniCapBench fixes four matching cutoffs (Table A.5): non-dialogue event tIoU $\mathrm { \Delta \ J \ge 0 . 2 0 }$ , shot tIoU $\geq 0 . 3 0$ , subshot window $\delta = 1 . 0 { \bar { \mathrm { s } } }$ , and dialogue $1 - \mathrm { W E R } \geq \bar { 0 } . 5 0$ . Since these values decide which predictions count as matched, we test whether our conclusions survive perturbation. We vary one cutoff at a time over the grid in Table B.10 for the six models listed there, holding predictions, references, and the one-to-one matching rule fixed, so no inference is re-run and only the alignment decision changes.

Absolute scores move as expected—tightening shot tIoU from 0.30 to 0.50 costs Gemini 3.1-Pro 12.2 Shot F1 points—but the induced ranking does not. Spearman’s ρ against the default, averaged over the two alternatives per sweep, is 1.000 for every metric except Shot F1 in the shot sweep (0.971) and Speaker in the dialogue sweep (0.914). No sweep produces a systematic open- versus closed-source reversal, and the only top-1 change within this set is a near tie on Speaker between Gemini 3.1-Pro (91.18) and Qwen3.5-Omni-Plus (91.20), two proprietary models 0.02 points apart. The structural bottlenecks we report, CCC and EVSA, are therefore properties of the evaluated models rather than artifacts of a particular cutoff.

## C Limitations and Broader Impact

Scoring atomic units rather than holistic text carries inherent limitations. First, OmniCapBench evaluates structural fidelity against our annotated reference system $S ^ { \star } ( V )$ , not every true fact in a video. A valid but peripheral detail outside our annotation scope therefore goes uncredited, a limitation shared by many generative benchmarks. Second, the dataset is bounded to videos under five minutes; far longer videos may require identity-tracking architectures beyond our scope. Third, the Stage II drafts seeding our reference system come from frontier models, some in the families we later evaluate, so an evaluator–model bias cannot be excluded: a reference inherits whatever blind spot survived review, and a model sharing the drafter’s stylistic priors may be scored slightly generously. We mitigate this by drafting with several model families, letting deterministic validators and human annotators arbitrate every unit (Appendix A.1.1), and restricting the scoring contract to locally verifiable fields, but we do not claim the bias is fully eliminated.

Finally, enforcing a strict, deep-structured JSON schema inherently disadvantages models heavily over-optimized for free-form prose (as discussed in Appendix B.5). This is a deliberate choice intended to encourage the research community to shift toward controllable, structured video generation rather than opaque text generation. OmniCapBench’s broader impact is providing developers with a highly auditable, traceable diagnostic tool, ensuring that future improvements in MLLMs represent genuine audio-visual reasoning rather than mere linguistic fluency.

## D Ethical Considerations

Constructing and releasing OmniCapBench raises ethical considerations around privacy, copyright, and the interpretation of diagnostic evaluations.

Privacy and Consent: The benchmark curates videos from publicly accessible media platforms and existing academic datasets. Since video intrinsically contains faces, voices, and potentially sensitive personal expressions, we filter out harmful, explicit, or highly sensitive private content during validation. Researchers using OmniCapBench must strictly adhere to the privacy guidelines of the original source platforms.

Copyright and Fair Use: OmniCapBench respects intellectual property rights by not distributing raw video files. Instead, we distribute the structured annotations alongside deterministic download scripts pointing to the original public URLs. This keeps the benchmark within academic fair use while respecting creators’ distribution rights.

Evaluation Bias and Capabilities: Our benchmark is explicitly designed to measure structured audio-visual understanding. Enforcing a strict, rule-based schema intrinsically disadvantages models optimized solely for free-form prose generation. It is ethically imperative to emphasize that Omni-CapBench measures a specific, rigorous form of grounded reasoning. A low score on OmniCapBench highlights localized structural deficits, but it does not necessarily imply that a model is entirely incapable of generating helpful video summaries in more relaxed, open-ended conversational settings.

## E Dataset Licenses and Usage

OmniCapBench is constructed under the MIT License, which permits free use, modification, and distribution for academic and commercial purposes. For the underlying video files, users must comply with the original licenses of the source datasets, which generally restrict usage to non-commercial research purposes. To respect copyright boundaries, we do not host or distribute the raw MP4 video files directly. Instead, we provide a deterministic script to download the publicly available videos using their original URLs. Table B.11 details the specific licenses for the underlying video data sources and the software models utilized during our evaluation.

Table B.11: License Information for Scientific Artifacts. This table details the licenses for the underlying video data sources and software models used in our evaluation.
<table><tr><td>Data Sources</td><td>URL</td><td>License</td></tr><tr><td>FunQA</td><td>https://github.com/Jingkang50/FunQA</td><td>MIT</td></tr><tr><td>LLaVA-Video-178K</td><td>https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K</td><td>Apache-2.0</td></tr><tr><td>Tarsier2-Recap</td><td>https://huggingface.co/datasets/omni-research/Tarsier2-Recap-585K</td><td>Research Only</td></tr><tr><td>Video-MME</td><td>https://github.com/MME-Benchmarks/Video-MME</td><td>Research Only</td></tr><tr><td>VisionRewardDB-Video</td><td>https://huggingface.co/datasets/zai-org/VisionRewardDB-Video</td><td>Apache-2.0</td></tr><tr><td>LongVideoBench</td><td>https://huggingface.co/datasets/longvideobench/LongVideoBench</td><td>CC BY-NC-SA 4.0</td></tr><tr><td>OmniVideoBench</td><td>https://huggingface.co/datasets/NJU-LINK/OmniVideoBench</td><td>CC BY-NC-SA 4.0</td></tr><tr><td>Software / Models</td><td>URL</td><td>License</td></tr><tr><td>Gemini Models</td><td>https://ai.google.dev/</td><td>Google ToS</td></tr><tr><td>Qwen Models</td><td>https://github.com/QwenLM/Qwen2.5-0mni</td><td>Apache-2.0</td></tr><tr><td>Seed Models</td><td>https://seed.bytedance.com/</td><td>ByteDance ToS</td></tr><tr><td>MiMo Models</td><td>https://huggingface.co/XiaomiMiMo/MiMo-V2.5</td><td>MIT</td></tr><tr><td>MiniCPM-o Models</td><td>https://huggingface.co/openbmb/MiniCPM-o-2_6</td><td>Apache-2.0</td></tr></table>

## F Instruction Templates and Prompts

## Listing B.1: Zero-shot Generation Instruction Example for predicting OmniCapBench units.

You are an expert audio - visual video analyzer . Watch the video carefully and   
decompose its contents into a strict JSON structure containing three tracks :   
‘ references ‘, ‘ events ‘, and ‘ shots ‘.   
1. ‘ references ‘: List all persistent entities ( PERSON , OBJECT , SCENE ) . Give each a   
unique ‘ref\_id ‘ (e.g., PERSON\_1 ). Include a ‘semantic\_description ‘ and an ‘   
appearance\_anchor ‘ detailing their visual features .   
2. ‘events ‘: List all audio events . Give each an ‘event\_id ‘, a ‘time\_range ‘ [start   
, end ] in seconds , and ‘ content ‘. If the event is spoken dialogue , include   
the exact transcribed ‘line ‘.   
3. ‘ shots ‘: Divide the video into non - overlapping temporal shots . For each shot ,   
provide its ‘ time\_range ‘, ‘ visual\_description ‘, and lists of ‘ active\_events ‘   
( referencing event\_ids ) and ‘ references\_in\_shot ‘ ( referencing ref\_ids ).   
Output ONLY valid JSON matching this schema . Do not include markdown formatting or   
conversational text .

## Listing B.2: Post-hoc Parser Instruction Example used to extract units from dense captions.

You are a structural parser . Given a free - form dense video caption , extract all   
implicit facts and format them into the explicit MTSS JSON schema .   
Identify all mentioned characters / objects and assign them consistent IDs in the   
‘ references ‘ list .   
Extract any mentioned sounds or dialogue into the ‘events ‘ list , attempting to   
infer timestamps if mentioned .   
Break the narrative into temporal ‘shots ‘, linking the extracted references and   
events . If a detail ( like an exact timestamp ) is missing from the text , omit   
it or use a best - guess estimate .   
Caption to parse : "{ caption\_text }"

## Listing B.3: Local Semantic Judge Instruction Examples (Subshot Matching and Quality).

\_SUBSHOT\_MATCH\_SYSTEM\_PROMPT : You are a strict binary matcher for video local   
visual states . Decide whether the TARGET item has at least one true   
counterpart in the opposite - side candidate list . Use only the provided   
sub\_shot\_content , times , ids , and reference mapping . A match must share the   
main visible subject (s), action / pose /state , and local scene facts . Accept   
paraphrases and small granularity differences ; reject generic overlap , nearby   
-but - different actions , or contradictory core facts . Do not evaluate   
hallucination or extra details separately ; decide only whether the target ’s   
core local facts are covered by at least one candidate . Output exactly <match   
>1 </ match > for a match or <match >0 </ match > for no match , followed by one   
short reason .   
SUBSHOT\_QUALITY\_SYSTEM\_PROMPT : You are an evidence - based evaluator for local   
video - state fact coverage . Compare the TARGET local - state facts against the   
provided matched context and score 1 -5. Do not evaluate hallucination or   
extra details as a separate criterion ; score only whether the requested   
TARGET facts are covered . Ignore minor wording differences and neighboring   
context that is not relevant to the target . Output format : first write < score   
>N </ score > (N=1 -5) , then 2-3 short evidence bullets naming the hit / missed   
target facts .

## Listing B.4: Generic Local 1–5 Judge System Instruction.

\_JUDGE\_SYSTEM\_PROMPT : You are a strict , evidence - based evaluator for video description quality . Your job is to compare a PREDICTION against a GROUND TRUTH and score 1 -5. Use only the fields shown in the instruction ; do not infer from outside knowledge or missing video context .

## RULES :

1. Judge factual consistency and coverage of the requested field only .

2. Contradictions to the GT are severe errors ; missing details are less severe than wrong details .

3. Ignore wording / order differences when the same fact is expressed . 4. Do not evaluate hallucination as a separate criterion . Ignore facts outside the requested coverage checklist ; score only factual coverage of the requested field . 5. Output format : first write < score >N </ score > (N =1 -5) , then 2 -3 concise evidence bullets naming the key matched / missed / wrong facts .

## System-Level Local Judging Instruction

\_JUDGE\_SYSTEM\_PROMPT : You are a strict , evidence - based evaluator for video description quality . Your job is to compare a PREDICTION against a GROUND TRUTH and score 1 -5. Use only the fields shown in the instruction ; do not infer from outside knowledge or missing video context .

## RULES :

1. Judge factual consistency and coverage of the requested field only .

2. Contradictions to the GT are severe errors ; missing details are less severe than wrong details .

3. Ignore wording / order differences when the same fact is expressed . 4. Do not evaluate hallucination as a separate criterion . Ignore facts outside the requested coverage checklist ; score only factual coverage of the requested field . 5. Output format : first write <score >N </ score > (N=1 -5) , then 2-3 concise evidence bullets naming the key matched / missed / wrong facts .

\_SUBSHOT\_MATCH\_SYSTEM\_PROMPT : You are a strict binary matcher for video local visual states . Decide whether the TARGET item has at least one true counterpart in the opposite - side candidate list . Use only the provided sub\_shot\_content , times , ids , and reference mapping . A match must share the main visible subject (s), action / pose /state , and local scene facts . Accept paraphrases and small granularity differences ; reject generic overlap , nearby - but - different actions , or contradictory core facts . Do not evaluate hallucination or extra details separately ; decide only whether the target ’s core local facts are covered by at least one candidate . Output exactly <match >1 </ match > for a match or < match >0 </ match > for no match , followed by one short reason .

\_SUBSHOT\_QUALITY\_SYSTEM\_PROMPT : You are an evidence - based evaluator for local video - state fact coverage . Compare the TARGET local - state facts against the provided matched context and score 1 -5. Do not evaluate hallucination or extra details as a separate criterion ; score only whether the requested TARGET facts are covered . Ignore minor wording differences and neighboring context that is not relevant to the target . Output format : first write < score >N </ score > (N=1 -5) , then 2-3 short evidence bullets naming the hit / missed target facts .

## Reference Recall Instruction Example

Entity type : { gt\_type }   
TASK : Evaluate RECALL — using the GROUND TRUTH as the subject , check whether the   
PREDICTION covers ALL factual content present in the GT detail\_description .   
Do not evaluate hallucination or extra prediction details ; only score   
coverage of GT facts .   
Compare only appearance\_anchor . id\_features . detail\_description . Ignore   
semantic\_description , attributes , timestamp , events , shots , and context .   
{ field\_guidance }   
Scoring criteria ( recall perspective — does prediction cover GT facts ?) :   
Count the total number of distinct factual elements in the GT description .   
For each GT fact , check if the prediction mentions it ( same meaning , wording   
differences OK).   
A GT fact is ’missed ’ if the prediction does not mention it at all .   
A GT fact is ’ contradicted ’ if the prediction states the opposite ( worse than   
missing ).   
=== GROUND TRUTH detail\_description ( SUBJECT — all facts here should be covered )   
=== { detail\_description ( gt\_ref )}   
=== PREDICTION detail\_description ( check coverage of GT facts ) === {   
detail\_description ( pred\_ref )}   
Score 1 -5:   
1 = Prediction covers almost none of the GT facts ( <20%) , or contradicts the   
core identity   
2 = Prediction covers a minority of GT facts (20 -40%) ; multiple key attributes   
are missing   
3 = Prediction covers roughly half of GT facts (40 -70%) ; some important details   
are missing   
4 = Prediction covers most GT facts (70 -90%) ; only minor details are missing   
5 = Prediction covers essentially all GT facts ( >90%) ; at most trivial omissions

## Reference Precision Instruction Example

Entity type : { gt\_type }   
TASK : Evaluate PRECISION — using the PREDICTION as the subject , check whether the   
GT covers ALL factual content present in the prediction detail\_description .   
Do not evaluate hallucination as a separate criterion ; only score coverage of   
prediction facts by the GT.   
Compare only appearance\_anchor . id\_features . detail\_description . Ignore   
semantic\_description , attributes , timestamp , events , shots , and context .   
{ field\_guidance }   
Scoring criteria ( precision perspective — are prediction facts supported by GT ?) :   
Count the total number of distinct factual elements in the PREDICTION   
description .   
For each prediction fact , check if the GT mentions or supports it.   
A prediction fact is ’not covered ’ if the GT does not mention the same factual   
content .   
A prediction fact is ’ contradicted ’ if the GT states the opposite ; count it as   
not covered .   
=== GROUND TRUTH detail\_description ( reference for verification ) === {   
detail\_description ( gt\_ref )}   
=== PREDICTION detail\_description ( SUBJECT — all facts here should be verified )   
=== { detail\_description ( pred\_ref )}   
Score 1 -5:   
1 = GT covers almost none of the prediction facts ( <20%)   
2 = GT covers a minority of prediction facts (20 -40%)   
3 = GT covers roughly half of prediction facts (40 -70%)   
4 = GT covers most prediction facts (70 -90%)   
5 = GT covers essentially all prediction facts ( >90%)

## Reference Recall field\_guidance: Subject Branch

For SUBJECT ( PERSON and OBJECT ) entities , the detail\_description typically includes :   
( for persons ) gender , age range , ethnicity , hairstyle , facial features , body build , clothing , held objects ;   
( for objects ) category , size , shape , material , color , texture , state ; and any other distinguishing visual traits . Each of these factual elements counts as a separate coverage point .

## Reference Recall field\_guidance: Scene Branch

For SCENE entities , the detail\_description typically includes : location type ( indoor / outdoor , specific venue ), spatial layout and dimensions , lighting conditions ( natural / artificial , brightness , color temperature ), dominant colors and textures of surfaces ( walls , floor , ceiling ) , key furniture / fixtures / objects present in the environment , atmosphere / weather conditions if outdoor , signage or text visible , and any other distinguishing environmental traits . Each of these factual elements counts as a separate coverage point .

## Reference Recall field\_guidance: Other Branch

Focus on all concrete visual / physical attributes mentioned in the detail\_description . Each distinct factual element counts as a separate coverage point .

## Reference Precision field\_guidance: Subject Branch

For SUBJECT ( PERSON and OBJECT ) entities , the detail\_description typically includes :

( for persons ) gender , age range , ethnicity , hairstyle , facial features , body build , clothing , held objects ;

( for objects ) category , size , shape , material , color , texture , state ; and any other distinguishing visual traits . Each of these factual elements counts as a separate verification point .

## Reference Precision field\_guidance: Scene Branch

For SCENE entities , the detail\_description typically includes : location type ( indoor / outdoor , specific venue ), spatial layout and dimensions , lighting conditions ( natural / artificial , brightness , color temperature ), dominant colors and textures of surfaces ( walls , floor , ceiling ) , key furniture / fixtures / objects present in the environment , atmosphere / weather conditions if outdoor , signage or text visible , and any other distinguishing environmental traits . Each of these factual elements counts as a separate verification point .

## Reference Precision field\_guidance: Other Branch

Focus on all concrete visual / physical attributes mentioned in the detail\_description . Each distinct factual element counts as a separate verification point .

## Reference Context Built by \_build\_ref\_context

Reference ID Mapping ( Pred → GT): { pred\_id } ({ detail\_description ( pred\_ref )}) → { gt\_id } ({ detail\_description ( gt\_ref ) })

## Event Quality Instruction Example

Event type : { gt\_type }   
{ ref\_context }   
Compare only content . description . Do not judge dialogue line , speaker field , event   
type , time\_range , shot placement , or reference ID correctness except as   
context for the description .   
An event description typically captures : the main action / activity happening , which   
entities ( persons / objects ) are involved and their roles , spatial   
relationships between entities , cause - effect or sequential sub - actions ,   
environmental / audio context relevant to the event , and the outcome or state   
change . Each of these factual elements is a separate coverage point .   
Scoring approach :   
Identify all distinct factual claims in the GT description .   
Check which are present in the prediction ( same meaning , wording differences OK )   
Contradictions ( stating the opposite ) are worse than omissions .   
Do not evaluate hallucination or extra prediction details ; ignore facts outside   
the GT checklist .   
=== GROUND TRUTH content . description === { gt\_content . description }   
=== PREDICTION content . description === { pred\_content . description }   
Score 1 -5:   
1 = Prediction contradicts or entirely misses the core event action / subject   
( <20% GT facts covered )   
2 = Prediction captures the broad category but misses or contradicts major   
factual elements (20 -40%)   
3 = Prediction captures the main gist but omits or gets wrong several important   
details (40 -70%)   
4 = Prediction covers most GT facts accurately ; only minor details are missing   
or imprecise (70 -90%)   
5 = Prediction faithfully covers essentially all GT factual content ( >90%)

## Subshot Formatting and task\_text Inserted by the Code

```hcl
mark = " [ TARGET ]" if focus_key and key == focus_key else "" lines . append ( f"- id
={ key }{ mark }; shot ={ sub . get (’ _shot_id ’, ’’)}; " f" time ={ sub .get (’
sub_shot_time ’, []) }: { sub . get (’ sub_shot_content ’, ’’)} " )
task_text = ( " Task : decide whether the TARGET PREDICTION subshot matches at least
one GT candidate in the context ." if direction == " precision " else " Task :
decide whether at least one prediction candidate matches the TARGET GT
subshot ." )
```

## Subshot Match Instruction Example

{ ref\_context }   
{ task\_text } A match should preserve the main entities , visible state , action / pose ,   
and local scene facts . Use [ TARGET ] to identify the item being evaluated .   
Ignore minor wording differences and slight granularity shifts . Reject   
candidates that only share a broad scene , are merely adjacent in time , or   
contradict the target ’s core subject / action / state . Do not evaluate   
hallucination or extra details ; this binary decision is only about whether   
the target facts are covered by at least one candidate .   
=== GROUND TRUTH local visual states === { gt\_text }   
=== PREDICTION local visual states === { pred\_text }   
Output exactly one decision : <match >1 </ match > if the target has at least one true   
counterpart in the opposite context <match >0 </ match > if no candidate matches ,   
or the best candidate is only generic / adjacent

## Subshot Quality Instruction Example

{ ref\_context }   
Compare only local visual state text : sub\_shot\_content . Use ids / times only to   
understand which item is the target and nearby context ; do not score timing   
accuracy here .   
{ scoring\_text }   
Scoring approach :   
Identify all distinct factual claims in the TARGET item .   
For each fact , check if the opposite - side context contains it ( same meaning ,   
wording OK ).   
Contradictions count as not covered .   
Nearby context items help disambiguate but are not themselves scored .   
Do not evaluate hallucination or extra details outside the target checklist .   
=== GROUND TRUTH sub\_shot\_content context === { gt\_text }   
=== PREDICTION sub\_shot\_content context === { pred\_text }   
Score 1 -5:   
1 = Target local state is essentially not hit ; core subject / action / state is   
absent or contradicted ( <20% facts covered )   
2 = A small fraction of target facts are hit (20 -40%) , but core subject / action /   
state is missing or wrong   
3 = Core state is partially hit (40 -70%) , with important target facts missing or   
inaccurate   
4 = Most target facts are hit (70 -90%) ; only minor misses or imprecision   
5 = Target facts are fully and accurately hit ( >90%) ; at most trivial omissions

## Subshot Recall scoring\_text Inserted by the Code

Score how well the prediction context covers the TARGET GT subshot facts ( RECALL   
direction ). Judge only the GT local - state facts that should be recovered . Do   
not evaluate hallucination or prediction - side extra details ; only score   
coverage of the target GT facts . A GT target fact that is omitted or   
contradicted by the prediction context is not covered .   
A sub\_shot\_content typically describes : which entities are visible ( persons /   
objects with ref IDs ) , their current pose / action / gesture / expression , spatial   
positions and relationships between entities , interaction with objects or   
environment , motion direction /speed , and any transient visual state , for   
instance , a lighting change or an object appearing / disappearing . Each of   
these factual elements is a separate coverage point .

## Subshot Precision scoring\_text Inserted by the Code

Score the TARGET PREDICTION subshot against the GT context ( PRECISION direction ).   
Judge only whether the target prediction ’s concrete local - state facts are   
supported by or accurately hit the GT context . Do not evaluate hallucination   
or extra target details separately ; only score coverage of the target   
prediction facts by the GT context . A target fact that conflicts with , or   
fails to match , the GT context is not covered .   
A sub\_shot\_content typically describes : which entities are visible ( persons /   
objects with ref IDs ), their current pose / action / gesture / expression , spatial   
positions and relationships between entities , interaction with objects or   
environment , motion direction / speed , and any transient visual state , for   
instance , a lighting change or an object appearing / disappearing . Each of   
these factual elements is a separate verification point .

## Shot Fact Instruction Example

Compare only camera and shot\_transfer fields . Do not judge visual\_description ,   
active\_events , references\_in\_shot , event content , or timing .   
Camera facts include : shot scale ( close - up / medium / wide / extreme ) , camera angle (   
high / low /eye - level / bird ’s-eye / dutch ), camera movement ( pan / tilt / dolly / zoom /   
tracking / crane / handheld / static ), framing composition (rule -of - thirds / centered   
/ off - center ) , depth of field ( shallow / deep ) , focus subject , and any special   
techniques ( rack focus , whip pan , etc .) . Shot\_transfer facts include :   
transition type ( cut / dissolve / fade / wipe / match - cut ) , transition speed , and any   
visual continuity or discontinuity noted .   
Scoring approach :   
Identify all distinct camera / transfer facts in the GT.   
Check which are present in the prediction ( same meaning , terminology differences   
OK ).   
Contradictions are worse than omissions .   
Do not evaluate hallucination or extra prediction details ; ignore facts outside   
the GT camera / transfer checklist .   
=== GROUND TRUTH === camera : { json . dumps ( gt\_camera , ensure\_ascii = False )}   
shot\_transfer : { gt\_transfer }   
=== PREDICTION === camera : { json . dumps ( pred\_camera , ensure\_ascii = False )}   
shot\_transfer : { pred\_transfer }   
Score 1 -5:   
1 = Camera and transfer facts are mostly missing or contradicted ( <20% GT facts   
covered )   
2 = Some overlap but major camera / transfer facts are wrong or missing (20 -40%)   
3 = Partly correct ; some key camera / transfer details are missing or inaccurate   
(40 -70%)   
4 = Mostly accurate with only minor omissions or imprecision (70 -90%)   
5 = Fully consistent with all GT camera and shot\_transfer facts ( >90%)