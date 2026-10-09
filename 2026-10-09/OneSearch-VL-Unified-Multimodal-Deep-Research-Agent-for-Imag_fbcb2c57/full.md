# OneSearch-VL: Unified Multimodal Deep Research Agent for Image and Video

Hongyu Li<sup>1</sup>, Manyuan Zhang†, Kaituo Feng<sup>2</sup>, Shu Chen<sup>2</sup>, Dian Zheng<sup>2</sup>, Hao Li<sup>3</sup>, Hao Yu<sup>4</sup>, Zhangquan Chen<sup>4</sup>, Zoey Guo<sup>2</sup>, Ray Zhang<sup>2</sup>, Shaofei Huang<sup>5</sup>, Tianrui Hui<sup>5</sup>, Linjiang Huang<sup>1</sup>, Si Liu<sup>1,</sup>‡

<sup>1</sup>BUAA <sup>2</sup>CUHK <sup>3</sup>NTU <sup>4</sup>THU <sup>5</sup>HFUT

†Project leader, ‡Corresponding author

## Abstract

Single-image, multi-image, and video deep research require diferent visual operations but share a workflow of visual grounding, external retrieval, and fact composition. A key challenge is to pre serve the dependencies linking localized visual anchors, entity relations, source-supported facts, and answer-producing operations. We introduce OneSearch-VL, a unified agent centered on the Visually Grounded Evidence Graph (VGEG), which encodes these dependencies as a shared task-level reference for data construction, process supervision, and operation-level evaluation. Our VGEG-based data engine constructs and verifies multi-image and video questions and filters expert trajectories. Using these data, we assemble OneSearch-VL-SFT-110K and OneSearch-VL-RL-10K for SFT and RL, respectively. We further derive the Evidence-aware Visual-Grounded Rubric reward (EVGR) from VGEG annotations to supervise evidence traceability and visual grounding during RL. For finegrained evaluation, we construct OneSearch-MI-Bench and OneSearch-Video-Bench, organizing questions by the research operations encoded in their VGEGs. Experiments show that OneSearch-VL-8B improves over Qwen3-VL-8B with tool access by 20.2 and 17.6 percentage points on the two new benchmarks, respectively, while also achieving substantial gains across 7 image benchmarks and VideoDR.

Home: https://github.com/appletea233/OneSearch-VL

Hugging Face: https://huggingface.co/OneSearch-VL

## 1 Introduction

Multimodal deep research enables vision-language models to move beyond direct answering by actively locating visual clues, retrieving external knowledge, and integrating evidence through multiturn tool use (Wu et al., 2025; Geng et al., 2025a; Huang et al., 2026; Chen et al., 2026). Recent work has extended this capability from images to videos by combining temporal localization, visual inspection, and open-web retrieval (Liu et al., 2026; Gao et al., 2026; Fang et al., 2026). Meanwhile, unified multimodal models extend spatial-temporal understanding and reasoning across images and videos (Li et al., 2025a; Feng et al., 2025b). These advances raise a natural question: can one policy jointly learn deep research over both images and videos? Although these settings require diferent visual operations, they share a workflow from visual grounding to external search and fact composition, motivating unified training across multiple visual input types.

Moving beyond prior work that primarily considers single-image or video inputs, we additionally study multi-image deep research and develop a unified framework spanning all three visual input types. Training data for this unified setting must link visual entities distributed across regions, images, or frames with relevant facts scattered across webpages. Prior work represents evidence through perception–knowledge chains, factual rubrics, and evidence graphs for task construction, supervision, and evaluation (Jiao et al., 2026; Zhang et al., 2026; Xu et al., 2026; Sun et al., 2026). However, these structures do not jointly preserve the dependencies from localized visual anchors, through entity relations and source-supported facts, to answer composition. Video-centric pipelines similarly use extracted entities or key-frame crops mainly as retrieval or question-generation seeds (Gao et al., 2026; Fang et al., 2026), without explicitly retaining this end-to-end provenance.

To address this gap, we introduce the Visually Grounded Evidence Graph (VGEG), a unified task-level reference structure for single-image, multi-image, and video research. VGEG links localized visual anchors to real-world entities, records multi-hop relational paths and source-supported facts, and specifies the operations that compose these facts into an answer. Based on VGEG, we develop a data engine for single-image, multi-image and video research that extracts visual entities, retrieves and verifies web facts, generates compositional questions, and synthesizes expert tool-use trajectories while retaining visual locations, source provenance, and evidence dependencies. These retained dependencies support question-answer construction and verification, expert-trajectory filtering, fine-grained trajectory rewards for RL, and operation-level evaluation. With this data engine, we construct OneSearch-VL-SFT-110K and OneSearch-VL-RL-10K for SFT and RL.

Using these data, we develop OneSearch-VL, a unified agent trained on data spanning singleimage, multi-image, and video inputs through SFT and RL. Existing multimodal agents supervise search through outcome rewards, query-quality signals, tool-use objectives, intermediate entity anchors, or evidence rubrics. We further introduce the Evidence-aware Visual-Grounded Rubric reward (EVGR), constructed from our VGEG annotations to evaluate complete trajectories along two complementary dimensions: evidence traceability, which verifies whether the required facts are supported by tool observations, and visual grounding, which verifies whether the relevant objects, regions, or frames are correctly identified and used in the solution.

For fine-grained evaluation, we also introduce OneSearch-MI-Bench and OneSearch-Video-Bench. Existing video deep-research benchmarks primarily evaluate answer accuracy, with analyses commonly organized around video content or source categories (Liu et al., 2026; Gao et al., 2026). Our benchmarks instead organize questions according to the research operations represented in their corresponding VGEGs, which capture the entity-relation paths and fact-composition dependencies required to derive the answer. This organization emphasizes how an agent retrieves and composes evidence rather than what the visual input contains.

We evaluate OneSearch-VL on established and two new benchmarks. OneSearch-VL-8B outperforms Qwen3-VL-8B by 20.2, 17.6, and 27.0 points on OneSearch-MI-Bench, OneSearch-Video-Bench, and VideoDR, respectively. Joint SFT on all three visual input types achieves a six-benchmark average of 55.8, versus 51.7-55.1 for single-type training. EVGR further raises the average from 57.3 to 61.1 over answer and query rewards.

Our contributions are summarized as follows:

• VGEG-centered data engine. We propose VGEG, a unified structure linking visual anchors, source-supported facts, and answer composition, together with a data engine for constructing and verifying multi-image and video questions and expert trajectories.

• Unified agent and evidence supervision. We develop OneSearch-VL for joint singleimage, multi-image, and video deep research, and use EVGR to incorporate evidence traceability and visual grounding into RL.

• Operation-oriented benchmark. We construct OneSearch-MI-Bench and OneSearch-Video-Bench benchmarks spanning research operations for fine-grained evaluation of fact retrieval and composition.

## 2 Unified Multimodal Deep Research

## 2.1 Problem Formulation

We study multimodal deep research over three types of visual input. Given a visual input X and a question $q ,$ an agent actively inspects the visual content, retrieves external information, and synthesizes a supported answer. The input is either a single image, an image collection, or a video:

$$
X \in \{ I , \mathcal { T } = \{ I _ { j } \} _ { j = 1 } ^ { K } , V \} .\tag{1}
$$

We learn a unified policy $\pi _ { \theta }$ that selects visual inspection, temporal localization, and web retrieval operations according to the input and question.

At step t, the policy conditions on the interaction history $h _ { t } = ( X , q , a _ { 1 } , o _ { 1 } , \dots , a _ { t - 1 } , o _ { t - 1 } )$ and generates an action $\boldsymbol { a } _ { t } = \left( z _ { t } , c _ { t } \right)$ , where $z _ { t }$ denotes the reasoning trace and $c _ { t }$ is either a tool call or the final response. A tool call produces an observation $o _ { t } .$ , such as a cropped region, a video frame, recognized text, or retrieved information. The complete trajectory is $\tau = ( X , q , a _ { 1 } , o _ { 1 } , \dots , a _ { T - 1 } , o _ { T - 1 } , a _ { T } )$ where $a _ { T }$ contains the final answer. Its likelihood is defined as:

$$
\pi _ { \boldsymbol { \theta } } ( \tau \mid X , q ) = \prod _ { t = 1 } ^ { T } \pi _ { \boldsymbol { \theta } } ( a _ { t } \mid h _ { t } ) .\tag{2}
$$

All three types of visual input share the same action–observation protocol while retaining inputspecific behaviors: single-image tasks inspect local regions, multi-image tasks aggregate information across indexed images, and video tasks localize relevant temporal segments and frames.

## 2.2 Tool Environment

OneSearch-VL exposes tools according to the visual input type, as detailed in Table 1:

$$
\mathcal { T } ( X ) = \left\{ \begin{array} { l l } { \mathcal { T } _ { \mathrm { v i s } } \cup \mathcal { T } _ { \mathrm { r e t } } , } & { X \in \{ I , \mathcal { T } \} , } \\ { \mathcal { T } _ { \mathrm { v i s } } \cup \mathcal { T } _ { \mathrm { r e t } } \cup \mathcal { T } _ { \mathrm { t e m p } } , } & { X = V . } \end{array} \right.\tag{3}
$$

All tools share a common dispatcher and observation format. This interface allows the policy to compose operations specific to each visual input type while using the same trajectory representation for expert synthesis, reinforcement learning, and inference. Detailed execution semantics are provided in Appendix C.

## 3 VGEG-Centered Multimodal Data Engine

Figure 1 presents the complete data engine. Starting from raw visual sources, the pipeline identifies localized visual anchors, connects them to source-supported web facts, constructs task-level VGEGs, and synthesizes verified question–answer pairs and expert trajectories. See Appendix A for details.

Table 1 Tool suite of OneSearch-VL.
<table><tr><td>Tool Set Tool</td><td></td><td>Description</td><td>Arguments</td><td>Visual Input Type</td></tr><tr><td rowspan="4"> $\mathcal { T } _ { \mathrm { v i s } }$ </td><td>CROP OCR</td><td>Crop an image region.</td><td>Image + Coordinates</td><td rowspan="4"></td></tr><tr><td></td><td>Extract visible text and layout.</td><td>Image</td></tr><tr><td></td><td>PERSPECTIVECoRRECT Correct perspective distortion.</td><td>Image</td></tr><tr><td>SUPERRESOLUTION SHARPEN</td><td>Upscale low-resolution images Reduce blur and enhance details.</td><td>Image + Scale Image + Amount</td></tr><tr><td rowspan="2"> $\mathcal { T } _ { \mathrm { r e t } }$ </td><td>IMAGESEARCH</td><td>Identify entities through image search. Image</td><td></td><td rowspan="2">I,I, V</td></tr><tr><td>TEXTSEARCH</td><td>Search, read, and summarize webpages. Query + Language + TopK</td><td></td></tr><tr><td rowspan="2"> $\mathcal { T } _ { \mathrm { t e m p } }$ </td><td>SELECTTIMESPAN</td><td>Select a contiguous video interval.</td><td>Video + Start + End</td><td rowspan="2">V</td></tr><tr><td>SELECTFRAME</td><td>Extract a frame by index.</td><td>Video + Frame Index</td></tr></table>

## 3.1 Visual Source Curation

As shown in Figure 1(a), we first curate source videos that contain identifiable entities, concrete event cues, and suficient potential for external knowledge retrieval. The pipeline combines metadata screening, task-suitability assessment, category balancing, and duration balancing, reducing the initial pool of 2.5M web videos to 70k candidates. See Appendix A.1 for details.

## 3.2 Dense Visual Anchor Discovery

Figure 1(b) converts each retained video into a hierarchical visual representation. We sample and group frames into local clips, generate clip- and frame-level descriptions, remove redundant key frames, aggregate temporally related clips into events, and localize searchable objects in representa tive frames. The resulting dense caption tree links each event to its temporal range, key frames, and localized object instances, which serve as visual anchors for subsequent retrieval. See Appendix A.2 for details.

## 3.3 Web Evidence Graph Construction

As illustrated in Figure 1(c), we connect visual anchors to external knowledge through image search, OCR, and text retrieval. Candidate identities are verified against the visual context and retrieved sources, after which source-supported relations and attributes are extracted and iteratively expanded. These results form an input-level evidence graph $\mathcal { G } _ { X }$ that preserves the correspondence among visual anchors, real-world entities, external facts, and supporting sources. See Appendix A.3 for details.

## 3.4 VGEG-Based Task Construction

Figure 1(d) constructs candidate research tasks and their task-level references from the input-level graph $\mathcal { G } _ { X }$ . For each candidate task $i ,$ its Visually Grounded Evidence Graph (VGEG) is derived through a task-conditioned projection $\Phi _ { i }$ that selects the required visual anchors and facts from $\mathcal { G } _ { X }$ and augments them with explicit answer-producing operations:

$$
\Gamma _ { i } = \Phi _ { i } ( \mathcal { G } _ { X } ) = ( A _ { i } , \mathcal { F } _ { i } , \mathcal { O } _ { i } , \mathcal { R } _ { i } ) .\tag{4}
$$

Here, $\mathbf { \mathcal { A } } _ { i }$ contains localized visual anchors and their associated entities, ${ \mathcal { F } } _ { i }$ contains source-supported external facts, $\mathcal { O } _ { i }$ specifies how the selected facts produce the answer, and $\mathcal { R } _ { i }$ combines grounding and support relations inherited from $\mathcal { G } _ { X }$ with task-specific dependencies among anchors, facts, and operations.

![](images/605505280dfc57af7c16c10664d891617ddd440af64f87e5a2d79f129f32a954.jpg)  
Figure 1 Overview of the VGEG-centered multimodal data engine. (a) Visual source curation retains videos suitable for external-knowledge research. (b) Dense visual anchor discovery organizes each video into events, key frames, and localized objects. (c) Web entity graph construction links these anchors to real world entities, source-supported facts, and webpages, forming the evidence graph $\mathcal { G } _ { X }$ . (d) VGEG-based task construction produces VGEG $\Gamma _ { i }$ for question generation, evidence verification, and visual-reference rewriting. (e) Expert trajectories in the tool environment are filtered by answer correctness and process quality to yield multi-turn training trajectories.

The generator jointly outputs $( q _ { i } , y _ { i } , \Gamma _ { i } )$ from $\mathcal { G } _ { X }$ . We verify the question and answer against the selected evidence and rewrite explicit entity mentions into visually grounded references. We then instantiate each task for its target visual input type. For a multi-image task, the deduplicated key frames containing the required anchors form an unordered image collection, and frame-level references are remapped to image indices. For a video task, the original video is retained together with its event, timestamp, frame, and region bindings. Additional filters remove answer leakage, ambiguous references, visually irrelevant questions, and instances that can be confidently answered without the visual input. See Appendix A.4 for details. For single-image task, we reuse existing data (Chen et al., 2026) and convert its Wikipedia entity-sampling paths and associated visual anchors into VGEGs.

![](images/7367a5c7c31f75db6d385c6d5097b9315e1d80c7cccd1ca0f4f51697c236681e.jpg)  
Figure 2 Operation-oriented design of the OneSearch benchmarks. Representative questions and reference VGEGs are shown for six principal research operations. Numbered markers bind visual evidence to graph nodes; green nodes denote retrieved entity/fact and purple nodes denote answer-producing operations. The center summarizes the operation distribution and the sizes of the multi-image and video benchmarks.

## 3.5 Expert Trajectory Synthesis

Finally, as shown in Figure 1(e), we synthesize expert trajectories for the verified tasks. An expert model receives only the visual input, question, and available tools and interacts with the environment until producing a final answer or reaching the interaction budget. We retain trajectories that pass both answer-correctness and process-quality evaluation and combine them with single-image trajectories in a unified multi-turn format to construct OneSearch-VL-SFT-110K. See Appendix A.5 for details. For RL, we construct OneSearch-VL-RL-10K from task pools spanning the same three visual input types, with construction details provided in Appendix D.

## 4 Onesearch-Bench

Existing multimodal search and video deep-research benchmarks primarily report aggregate answer accuracy, often with additional breakdowns by visual content or data source (Liu et al., 2026; Gao et al., 2026). These summaries measure overall performance but do not isolate failures in entity lookup, relation tracing, conditional filtering, or fact composition. We therefore construct OneSearch-MI-Bench and OneSearch-Video-Bench and organize their examples by the principal research operation required to answer each question. Both benchmarks use the same operation taxonomy, enabling a consistent analysis across multi-image and video inputs.

Figure 2 presents representative examples and their reference VGEGs, while Appendix Table 6 defines the six principal research operations and reports their distributions across the two benchmarks. Together, they support operation-level analysis of evidence retrieval and composition.

![](images/6a3986f039c880dee2ca497e4f3bb926ee86e0d8e40bd97de79ca654f436ef47.jpg)  
Figure 3 Overview of the Evidence-aware Visual-Grounded Rubric reward (EVGR). A task-level VGEG and evidence ledger align each multimodal rollout with the required visual anchors and sourcesupported facts. Claim-and-evidence and visual-grounding judges produce $r _ { \mathrm { t r a c e } }$ and $r _ { \mathrm { g r o u n d } }$ , whose combination forms $R _ { \mathrm { E V G R } }$ for policy optimization.

For OneSearch-MI-Bench, deduplicated evidence frames from the same visual source form an unordered image set $X = \{ I _ { 1 } , \ldots , I _ { K } \}$ . Each example depends on at least two images and retains image indices that bind visual anchors to their sources. We remove references to video playback, frame numbers, and unnecessary temporal order so that each question is self-contained over the image set, while preserving the relations needed for fact composition.

For OneSearch-Video-Bench, we preserve temporal structure and link visual anchors to their events, key frames, and web facts. Questions locate clues through visible objects, events, or temporal descriptions, requiring the agent to find relevant video content before retrieving and composing ex ternal knowledge. The two branches share candidate sources, VGEG representations, the operation taxonomy, and quality criteria, while producing separate multi-image and video sets.

Candidates pass checks for information masking, referential uniqueness, visual relevance, and non triviality. A text-only probe further removes questions that can be answered confidently without the visual input. We then jointly inspect the question, reference answer, and supporting facts for clarity, answer uniqueness, factual correctness, and visual grounding, followed by a strict review of the visual input.

## 5 OneSearch-VL Training

We train OneSearch-VL in two stages. Supervised fine-tuning learns a unified multimodal toolinteraction policy from expert trajectories, and reinforcement learning further optimizes answer quality, search behavior, and evidence use through online interaction with tool environment.

## 5.1 Agentic Supervised Fine-Tuning

We first perform SFT on OneSearch-VL-SFT-110K. Each trajectory interleaves reasoning, tool calls, tool observations, and a final answer. All three visual input types share the same action–observation schema, allowing one policy to select input-appropriate tools within a unified action space.

Tool observations are exogenous environment outputs and serve only as conditioning context. Let $y _ { t , k }$ denote the k-th policy-generated token at turn t, and let $M _ { t , k } ^ { \mathrm { p o l } }$ be one for reasoning, tool command, and final-response tokens and zero for serialized tool observations. The SFT objective

is

$$
{ \mathcal { L } } _ { \mathrm { S F T } } ( \theta ) = - \sum _ { \tau } \sum _ { t } \sum _ { k } M _ { t , k } ^ { \mathrm { p o l } } \log \pi _ { \theta } ( y _ { t , k } \mid h _ { t } , y _ { t , < k } ) .\tag{5}
$$

This stage yields the initial policy $\pi _ { \theta _ { \mathrm { S F T } } }$ for reinforcement learning.

## 5.2 Reinforcement Learning

Starting from $\pi _ { \theta _ { \mathrm { S F T } } }$ , we perform online rollouts on OneSearch-VL-RL-10K with the same dispatcher and tool environment used for expert trajectory synthesis and inference. Given visual input X, question $Q .$ , and reference answer $A ,$ the behavior policy samples a group of trajectories

$$
\tau _ { i } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid X , Q ; { \mathcal E } ) , \qquad i = 1 , \dots , G .\tag{6}
$$

Dual-dimensional EVGR. Terminal correctness does not directly measure whether a rollout obtains and uses the evidence required by the question. We therefore introduce the Evidence-aware Visual-Grounded Rubric reward (EVGR). As illustrated in Figure 3, EVGR aligns each executed tra jectory with an item-specific structured rubric derived from the data engine. The rubric records the answer core, required claim facts, supporting sources and snippets, and applicable frame or region annotations. Two separate judge calls evaluate complementary dimensions. Evidence Traceability $( r _ { \mathrm { t r a c e } } )$ measures whether tool observations establish the answer-core entities and required fact hops, and whether the reasoning follows those observations without unsupported substitutions. Visual Grounding $( r _ { \mathrm { g r o u n d } } )$ measures whether the correct visible entities, regions, or video frames are identified and used to drive subsequent evidence retrieval. For single- and multi-image inputs, the source images are already visible to the agent policy and no additional visual tool call is required; for video, the selected frames should cover the relevant moments and provide the visual identity used by later searches. The two dimensions are combined into the trajectory-level process reward $R _ { \mathrm { E V G R } }$ . Their scoring and combination coeficients are specified in the experimental setup.

Composite reward and policy optimization. Following OpenSearch-VL (Chen et al., 2026), we adopt its answer-correctness reward $R _ { \mathrm { a c c } }$ and query-quality reward $R _ { \mathrm { q u e r y } }$ , and augment them with our EVGR process reward. The resulting reward is gated by format validity:

$$
R ( \tau ) = R _ { \mathrm { f m t } } ( \tau ) \Bigl ( \lambda _ { \mathrm { a c c } } R _ { \mathrm { a c c } } ( \tau ) + \lambda _ { \mathrm { q u e r y } } R _ { \mathrm { q u e r y } } ( \tau ) + \lambda _ { \mathrm { E V G R } } R _ { \mathrm { E V G R } } ( \tau ) \Bigr ) .\tag{7}
$$

Here, $R _ { \mathrm { a c c } }$ evaluates the final answer, $R _ { \mathrm { q u e r y } }$ evaluates the relevance and progression of search queries, and $R _ { \mathrm { f m t } }$ enforces a valid reasoning, tool-call, and response structure. The reward coeficients and implementation details are provided in Appendix D.

We optimize the policy with Group Relative Policy Optimization (GRPO), normalizing rewards across trajectories sampled for the same question and applying the clipped objective only to policygenerated tokens. Tool observations remain conditioning context and are excluded from the policy loss. For trajectories terminated by tool-execution failures, we follow the fatal-aware mechanism of OpenSearch-VL (Chen et al., 2026) and retain only the valid pre-failure prefix for optimization.

## 6 Experiments

## 6.1 Experimental Setup

Model and training data. OneSearch-VL is initialized from Qwen3-VL-8B (Bai et al., 2025) and trained in two stages. Supervised fine-tuning uses OneSearch-VL-SFT-110K. Reinforcement learning starts from the joint SFT model and uses OneSearch-VL-RL-10K. Both datasets cover all three visual input types. All online rollouts use the same tool environment.

Benchmarks. For single-image evaluation, we follow OpenSearch-VL (Chen et al., 2026) and use seven knowledge-intensive benchmarks: SimpleVQA (Cheng et al., 2025), VDR (Zeng et al., 2026), MMSearch (Jiang et al., 2025), LiveVQA (Fu et al., 2025), BrowseComp-VL (Geng et al., 2025b), FVQA (Wang et al., 2017), and InfoSeek (Chen et al., 2023). For multi-image and video evaluation, we use our OneSearch-MI-Bench and OneSearch-Video-Bench, together with VideoDR (Liu et al., 2026). On the two OneSearch benchmarks, we additionally report performance by the six principal research operations defined in Section 4. Following OpenSearch-VL, we use GPT-4o (OpenAI Team, 2024) to compare each final response with the reference answer and return a binary correctness decision.

Implementation details. Expert trajectory synthesis, online RL rollouts, and inference share the same dispatcher and tool interfaces. All training variants are compared under the same tool en vironment and evaluation protocol. Full optimization, rollout, reward, and inference configurations are provided in the appendix.

## 6.2 Main Results

Single-image multimodal deep research. Table 2 summarizes the results on seven singleimage deep-research benchmarks. OneSearch-VL-8B obtains an average score of 58.3, outperforming the same-scale Qwen3-VL-8B Agent by 16.3 points and OpenSearch-VL-8B by 1.7 points. It improves over OpenSearch-VL on all seven benchmarks, with gains of 2.8, 2.1, and 2.0 points on VDR, InfoSeek, and MMSearch, respectively. It also achieves the best result among the listed agentic methods on six of the seven benchmarks.

The gains span benchmarks emphasizing visual entity identification, open-web retrieval, and multistep knowledge reasoning. Together with the data-mixture ablation below, these results show that joint training with multi-image and video trajectories does not trade of image research performance.

Multi-image and video deep research. As shown in Table 3, OneSearch-VL-8B improves over Qwen3-VL-8B by 20.2 points on OneSearch-MI-Bench, 17.6 points on OneSearch-Video-Bench, and 27.0 points on VideoDR. It improves all six research operations on both OneSearch benchmarks, showing that the gains extend beyond simple fact lookup to questions that combine evidence across images or video frames.

The operation-level breakdown further identifies where the gains arise. On multi-image tasks, knowledge-conditioned counting, multi-anchor arithmetic, and multi-anchor comparison improve by 29.8, 26.3, and 22.5 points, respectively. On video tasks, multi-hop retrieval, multi-anchor arithmetic, and multi-anchor joining improve by 25.5, 20.0, and 19.2 points. These categories require the model to filter, connect, or compute over facts associated with multiple visual anchors, matching the compositional operations covered by our data engine.

## 6.3 Ablation Studies

SFT data mixture. Table 4(a) compares SFT mixtures composed of single-image (I), multi-image (M), and video (V) trajectories using the same number of training iterations. Training on any one trajectory type raises the six-benchmark average from 40.7 to 51.7–55.1. The video-only model attains the strongest single-type average of 55.1 while also improving all three single-image benchmarks, indicating that research trajectories collected for one visual input type can provide useful supervision for others.

Table 2 Results on single-image deep-research benchmarks.
<table><tr><td>Model</td><td>SimpleVQA</td><td>VDR</td><td>MMSearch</td><td>LiveVQA</td><td>BrowseComp-VL</td><td>FVQA</td><td>InfoSeek</td><td>Avg.</td></tr><tr><td colspan="7">Direct Reasoning</td></tr><tr><td>GPT-4o (OpenAI Team, 2024)</td><td>51.7</td><td>1.7</td><td>18.7</td><td>28.1</td><td>5.5</td><td>48.0</td><td>52.9 29.5</td></tr><tr><td>GPT-5 (OpenAI, 2025b)</td><td>61.6</td><td>9.8</td><td>35.1</td><td>44.4</td><td>48.6</td><td>54.4</td><td>61.7 45.1</td></tr><tr><td>Gemini-2.5-Flash (Comanici et al., 2025)</td><td>57.9</td><td>6.2</td><td>30.4</td><td>51.0</td><td>37.1</td><td>47.7 44.1</td><td>39.2</td></tr><tr><td>Gemini-2.5-Pro (Comanici et al., 2025)</td><td>63.0</td><td>8.0</td><td>39.8</td><td>60.3</td><td>43.1</td><td>60.7 46.9</td><td>46.0</td></tr><tr><td>Claude-4-Sonnet (Anthropic Team, 2025b)</td><td>50.9</td><td>2.0</td><td>18.7</td><td>38.5</td><td>29.3</td><td>35.3 57.3</td><td>33.1</td></tr><tr><td>Claude-3.7-Sonnet (Anthropic Team, 2025a)</td><td>42.7</td><td>4.6</td><td>21.1</td><td>38.0</td><td>32.3</td><td>36.7 54.8</td><td>32.9</td></tr><tr><td>Qwen3-VL-8B (Bai et al., 2025)</td><td>47.1</td><td>2.8</td><td>11.7</td><td>23.1</td><td>24.1</td><td>24.2 23.1</td><td>22.3</td></tr><tr><td>Qwen3-VL-30B-A3B (Bai et al., 2025)</td><td>53.2</td><td>3.8</td><td>18.7</td><td>42.7</td><td>29.6</td><td>34.7 26.4</td><td>29.9</td></tr><tr><td>Qwen3-VL-32B (Bai et al., 2025)</td><td>58.0</td><td>4.1</td><td>19.8</td><td>45.5</td><td>30.8</td><td>34.1 28.8</td><td>31.6</td></tr><tr><td colspan="8">RAG Workflow</td></tr><tr><td>GPT-4o (OpenAI Team, 2024)</td><td>63.6</td><td>4.5</td><td>49.1</td><td>40.1</td><td>13.4</td><td>66.3</td><td>59.5 42.4</td></tr><tr><td>GPT-5 (OpenAI, 2025b)</td><td>55.9</td><td>22.3</td><td>52.6</td><td>56.0</td><td>54.9</td><td>62.6 70.6</td><td>53.6</td></tr><tr><td>Claude-3.7-Sonnet (Anthropic Team, 2025a)</td><td>59.3</td><td>11.3</td><td>32.7</td><td>30.3</td><td>10.0</td><td>59.1 60.2</td><td>37.6</td></tr><tr><td>Qwen3-VL-8B (Bai et al., 2025)</td><td>62.3</td><td>7.3</td><td>47.3</td><td>39.3</td><td>29.3</td><td>53.6 46.1</td><td>40.7</td></tr><tr><td colspan="8">Agentic Workflow</td></tr><tr><td>DeepMMSearch-R1-7B (Narayan et al., 2025)</td><td>55.8</td><td></td><td></td><td></td><td></td><td>47.5</td><td></td></tr><tr><td>Visual-ARFT-7B (Liu et al., 2025)</td><td>42.4</td><td>3.3</td><td>34.5</td><td>25.4</td><td>16.5</td><td>41.7 37.9</td><td>28.8</td></tr><tr><td>MMSearch-R1-7B (Wu et al., 2025)</td><td>57.4</td><td>2.9</td><td>53.8</td><td>48.4</td><td>20.9</td><td>58.4 55.1</td><td>42.4</td></tr><tr><td>DeepEyes-v2-7B (Hong et al., 2025)</td><td>59.4</td><td>7.8</td><td>63.7</td><td></td><td></td><td>60.6 51.1</td><td></td></tr><tr><td>WebWatcher-7B (Geng et al., 2025a)</td><td>54.3</td><td>10.3</td><td>49.1</td><td>51.2</td><td>21.2</td><td></td><td></td></tr><tr><td>Qwen3-VL-8B (Bai et al., 2025)</td><td>52.0</td><td>17.0</td><td>37.4</td><td>50.6</td><td>27.9</td><td>58.7 50.3</td><td>42.0</td></tr><tr><td>SenseNova-MARS-8B (Chng et al., 2025)</td><td>61.7</td><td>19.4</td><td>67.4</td><td>56.2</td><td>35.1</td><td>67.1 61.7</td><td>52.7</td></tr><tr><td>OpenSearch-VL-8B (Chen et al., 2026)</td><td>71.6</td><td>20.8</td><td>64.5</td><td>59.6</td><td>37.6</td><td>71.5 70.2</td><td>56.6</td></tr><tr><td>OneSearch-VL-8B (Ours)</td><td>72.4</td><td>23.6</td><td>66.5</td><td>61.2</td><td>39.0</td><td>73.4 72.3</td><td>58.3</td></tr></table>

Table 3 Results on multi-image and video deep-research benchmarks.
<table><tr><td></td><td colspan="2">VideoDR</td><td colspan="6">OneSearch-MI-Bench</td><td colspan="7">OneSearch-Video-Bench</td></tr><tr><td>Model</td><td>All</td><td>SA</td><td>MH</td><td>Count Join Arith.</td><td></td><td></td><td>Comp.</td><td>All</td><td>SA</td><td>MH</td><td>Count Join Arith.</td><td></td><td></td><td>Comp.</td><td>All</td></tr><tr><td colspan="10">Direct Reasoning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4o (OpenAI Team, 2024)</td><td>42.0</td><td></td><td>35.7 31.3</td><td>56.1</td><td>46.6</td><td>23.0</td><td>44.9</td><td>39.9</td><td>36.0</td><td>25.5</td><td>29.5</td><td>30.8</td><td>13.3</td><td>29.9</td><td>27.4</td></tr><tr><td>Gemini-2.5-Flash (Comanici et al., 2025)</td><td>39.0</td><td></td><td>32.1 33.3</td><td>61.4</td><td>46.6</td><td>21.3</td><td>42.9</td><td>40.2</td><td>44.0</td><td>27.7</td><td>32.8</td><td>23.1</td><td>20.0</td><td>33.8</td><td>29.6</td></tr><tr><td>Gemini-2.5-Pro (Comanici et al., 2025)</td><td>50.0</td><td>42.9</td><td>39.6</td><td>61.4</td><td>46.6</td><td>34.4</td><td>44.9</td><td>45.2</td><td>44.0</td><td>34.0</td><td>37.7</td><td>23.1</td><td>17.8</td><td>37.7</td><td>32.3</td></tr><tr><td>Qwen2.5-VL-3B (Bai et al., 2025)</td><td>9.0</td><td></td><td>10.7 12.5</td><td>35.1</td><td>13.8</td><td>1.6</td><td>14.3</td><td>15.0</td><td>12.0</td><td>6.4</td><td>18.0</td><td>5.8</td><td>4.4</td><td>3.9</td><td>8.1</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025)</td><td>5.0</td><td></td><td>17.9 12.5</td><td>22.8</td><td>17.2</td><td>4.9</td><td>14.3</td><td>14.6</td><td>4.0</td><td>4.3</td><td>16.4</td><td>5.8</td><td>2.2</td><td>6.6*</td><td>7.2*</td></tr><tr><td>Qwen2.5-VL-72B (Bai et al., 2025)</td><td>16.0</td><td></td><td>14.3 14.6</td><td>43.9</td><td>24.1</td><td>6.6</td><td>22.5</td><td>21.6</td><td>28.0</td><td>6.4</td><td>32.8</td><td>13.5</td><td>2.2</td><td>13.0</td><td>15.6</td></tr><tr><td>Qwen3-VL-8B (Bai et al., 2025)</td><td>11.0</td><td></td><td>14.3 10.4</td><td>26.3</td><td>22.4</td><td>3.3</td><td>18.4</td><td>16.0</td><td>12.0</td><td>2.1</td><td>23.0</td><td>11.5</td><td>2.2</td><td>13.0</td><td>11.4</td></tr><tr><td>Qwen3-VL-30B-A3B (Bai et al., 2025)</td><td>14.0</td><td></td><td>21.4 18.8</td><td>68.4</td><td>25.9</td><td>11.5</td><td>26.5</td><td>29.6</td><td>16.0</td><td>8.5</td><td>29.5</td><td>13.5</td><td>4.4</td><td>14.3</td><td>15.0</td></tr><tr><td>Qwen3-VL-32B (Bai et al., 2025)</td><td>17.0</td><td></td><td>25.027.1</td><td>42.1</td><td>37.9</td><td>18.0</td><td>30.6</td><td>30.6</td><td>20.0</td><td>6.4</td><td>37.7</td><td>9.6</td><td>2.2</td><td>20.8</td><td>17.3</td></tr><tr><td colspan="10">Agentic Workflow</td><td colspan="7"></td></tr><tr><td>Qwen3-VL-8B (Bai et al., 2025)</td><td>30.0</td><td></td><td>39.3 27.1</td><td>45.6</td><td>44.8</td><td>18.0</td><td>40.8</td><td>35.6</td><td>40.0 12.8</td><td></td><td>22.9</td><td>15.4</td><td>2.2</td><td>20.8</td><td>17.9</td></tr><tr><td>OpenSearch-VL-8B (Chen et al., 2026)</td><td>47.0</td><td>57.1</td><td>35.4</td><td>63.2</td><td>55.2</td><td>24.6</td><td>36.7</td><td>44.5</td><td>44.0</td><td>34.0</td><td>23.0</td><td>15.4</td><td>11.1</td><td>28.6</td><td>24.7</td></tr><tr><td>OneSearch-VL-8B (Ours)</td><td>57.0</td><td></td><td>46.4 43.8</td><td>75.4</td><td>56.9</td><td>44.3</td><td>63.3</td><td>55.8</td><td>44.0</td><td>38.3</td><td>36.1</td><td>34.6</td><td>22.2</td><td>39.0</td><td>35.5</td></tr></table>

Combining trajectory types further improves overall performance. Adding either multi-image or video trajectories to the single-image data raises the average score. Joint training on all three types reaches 55.8 and gives the best result in this ablation on SimpleVQA, OneSearch-MI-Bench, and OneSearch-Video-Bench. This result supports the complementarity of the three trajectory sources under our training setup.

RL reward. Table 4(b) analyzes answer correctness $( R _ { \mathrm { a c c } } )$ , query quality $( R _ { \mathrm { q u e r y } } )$ , and the Evidence Traceability $( r _ { \mathrm { t r a c e } } )$ and Visual Grounding $( r _ { \mathrm { g r o u n d } } )$ dimensions of EVGR. Starting from the joint SFT model, accuracy-only RL improves the six-benchmark average from 55.8 to 56.3, and adding query quality raises it to 57.3. Adding Trace or Ground separately further improves the average to 59.0 and 59.3, showing that each process dimension supplies useful supervision beyond answer and query rewards.

Table 4 Ablation studies on the SFT data mixture and RL reward.
<table><tr><td></td><td colspan="9">(a) SFT data mixture</td><td></td></tr><tr><td>Data Mixture</td><td></td><td>Image Multi-Image</td><td>Video</td><td>SimpleVQA InfoSeek FVQA</td><td></td><td></td><td>MI</td><td>OneSearch- OneSearch- Video</td><td>VideoDR</td><td>Avg.</td></tr><tr><td>Qwen3-VL-8B</td><td>X</td><td>X</td><td>X</td><td>52.0</td><td>50.3</td><td>58.7</td><td>35.5</td><td>17.9</td><td>30.0</td><td>40.7</td></tr><tr><td>I</td><td>√</td><td>X</td><td>X</td><td>66.1</td><td>62.4</td><td>65.3</td><td>44.5</td><td>24.8</td><td>47.0</td><td>51.7</td></tr><tr><td>M</td><td>X</td><td>√</td><td>X</td><td>67.0</td><td>62.0</td><td>67.1</td><td>48.5</td><td>25.0</td><td>49.0</td><td>53.1</td></tr><tr><td>V</td><td>X</td><td>X</td><td>√</td><td>68.6</td><td>62.5</td><td>68.5</td><td>51.2</td><td>28.0</td><td>52.0</td><td>55.1</td></tr><tr><td>I + M</td><td>√</td><td>√</td><td>X</td><td>68.4</td><td>64.5</td><td>67.1</td><td>50.8</td><td>28.6</td><td>50.0</td><td>54.9</td></tr><tr><td>I + V</td><td>√</td><td>X</td><td>√</td><td>68.1</td><td>63.8</td><td>68.2</td><td>49.1</td><td>29.3</td><td>51.0</td><td>54.9</td></tr><tr><td> $\mathrm { ~ I ~ } + \mathrm { ~ M ~ } + \mathrm { ~ V ~ }$ </td><td>√</td><td>√</td><td>√</td><td>68.7</td><td>64.4</td><td>67.8</td><td>51.5</td><td>31.2</td><td>51.0</td><td>55.8</td></tr><tr><td colspan="9">(b) RL reward</td><td></td></tr><tr><td>Method</td><td> $R _ { \mathrm { a c c } }$ </td><td> $R _ { \mathrm { q u e r y } }$ </td><td>Itrace</td><td> $r _ { \mathrm { g r o u n d } }$  SimpleVQA InfoSeek FVQA</td><td></td><td></td><td>MI</td><td>OneSearch- OneSearch- Video</td><td>VideoDR</td><td>Avg.</td></tr><tr><td>OneSearch-VL-8B-SFT</td><td>一</td><td>I</td><td>一</td><td>68.7</td><td>64.4</td><td>67.8</td><td>51.5</td><td>31.2</td><td>51.0</td><td>55.8</td></tr><tr><td>RL (Acc.)</td><td>√</td><td>X</td><td>X</td><td></td><td>64.5</td><td>69.3</td><td>50.8</td><td>32.8</td><td>51.0</td><td>56.3</td></tr><tr><td>RL  $( \mathrm { A c c . \ t + Q u e r y } )$ </td><td>√</td><td>√</td><td>×</td><td></td><td>65.8</td><td>70.2</td><td>52.5</td><td>33.2</td><td>52.0</td><td>57.3</td></tr><tr><td> $\mathrm { R L } \ ( \mathrm { A c c . \ + \ Q u e r y \ + \ T r a c e } )$ </td><td>√</td><td>√</td><td>√</td><td></td><td>68.0</td><td>71.5</td><td>54.0</td><td>35.5</td><td>54.0</td><td>59.0</td></tr><tr><td> $\mathrm { R L } \ ( \mathrm { A c c . \ + \ Q u e r y + G r o u n d } )$ </td><td>V</td><td>√</td><td>X</td><td></td><td>68.2</td><td>71.8</td><td>54.2</td><td>35.2</td><td>55.0</td><td>59.3</td></tr><tr><td>RL (Full reward)</td><td>√</td><td>√</td><td>√</td><td></td><td>72.3</td><td>73.4</td><td>55.8</td><td>35.5</td><td>57.0</td><td>61.1</td></tr></table>

Using both Trace and Ground yields the best average of 61.1, improving by 3.8 points over answer plus-query rewards and by 2.1 and 1.8 points over the two single-dimension variants. The full reward performs best on five benchmarks and ties for best on the remaining one. This result indicates that evidence traceability and visual grounding provide complementary feedback on factual support and the use of visual cues during research.

## 7 Related Work

## 7.1 Multimodal Search and Deep Research

Active search enables language models to supplement parametric knowledge with external evi dence acquired through multi-turn interaction (Jin et al., 2025). Multimodal research extends this paradigm to visual inputs by combining image and text retrieval with visual inspection, and training search policies through synthetic tool-use trajectories and reinforcement learning (Wu et al., 2025; Geng et al., 2025a; Huang et al., 2026; Chen et al., 2026; Yao et al., 2026; Narayan et al., 2025). Related work explores active visual perception, generalizable tool use, and joint optimization of reasoning and search, advancing research agents that integrate visual understanding with external evidence acquisition (Liu et al., 2025; Hong et al., 2025; Chng et al., 2025).

## 7.2 Video Deep Research

Video deep research additionally requires agents to locate relevant moments, identify objects across frames, and connect them to external knowledge. VideoDR formalizes this setting through an open-web benchmark (Liu et al., 2026). Recent methods develop video-oriented research pipelines by jointly training temporal localization, spatial inspection, and multimodal retrieval (Gao et al., 2026), or by staging tool use to guide visual grounding before web exploration (Fang et al., 2026). These studies highlight temporal and spatial grounding as essential links between video content and

external evidence.

## 7.3 Unified Visual Reasoning

Reinforcement learning has improved image question answering and detection (Huang et al., 2025; Shen et al., 2025), image segmentation (You & Wu, 2025), video question answering (Feng et al., 2025a), and temporal or spatio-temporal video understanding (Wang et al., 2025; Li et al., 2025b), with many methods designed for specific tasks. LLaVA-ST jointly addresses fine-grained spatial, temporal, and spatio-temporal understanding (Li et al., 2025a). OneThinker jointly learns question answering, captioning, grounding, tracking, and segmentation across images and videos, demonstrating the potential of training across tasks and visual inputs (Feng et al., 2025b). Our work focuses on unification at the level of deep research: a single policy combines input-specific visual operations with external retrieval and fact composition across single images, image collections, and videos.

## 7.4 Structural References and Process Supervision

Beyond answer correctness, research agents receive supervision through query-quality and tool-use rewards (Chen et al., 2026; Gao et al., 2026). Other approaches use intermediate entity anchors and citation-supported factual requirements as references for credit assignment (Jiao et al., 2026; Zhang et al., 2026), while process-oriented objectives evaluate search decisions or complete rollouts (Yan et al., 2026; Wang et al., 2026). These studies establish the value of structured evidence for agent supervision. We use VGEG to preserve visual provenance and answer-composition dependencies in a shared task-level reference, connecting task construction and verification with trajectory supervision. EVGR derives evidence-traceability and visual-grounding criteria from this reference to evaluate complete research trajectories.

## 8 Conclusion

We introduced OneSearch-VL, a unified agent for deep research over single-image, multi-image, and videos. VGEG links visual anchors, source-supported facts, and answer-producing operations, enabling verified task construction, expert-trajectory filtering, and process-level reward design. Building on this structure, EVGR evaluates both evidence traceability and visual grounding. Across seven established single-image benchmarks and two operation-oriented benchmarks, OneSearch-VL consistently improves over the evaluated open-source baselines; the ablations further demonstrate complementary supervision across visual input types and the benefit of EVGR.

## 9 Limitations and Future Work

OneSearch-VL depends on external search tools such as TextSearch and ImageSearch (Table 1) and changing webpages, which can afect evidence availability and exact reproducibility. Future work should preserve retrieval snapshots and evaluate robustness to tool failures. Automated VGEG construction and model-based rewards may also introduce annotation errors or judging biases. Human calibration and open multimodal process judges could improve the reliability of trajectory-level assessment. Multi-turn interaction incurs additional inference and tool costs, motivating adaptive budgets and cost-aware training.

## References

Anthropic Team. Claude 3.7 Sonnet system card. https://www-cdn.anthropic.com/9ff93dfa8f445c932 415d335c88852ef47f1201e.pdf, 2025a.

Anthropic Team. System card: Claude Opus 4 and Claude Sonnet 4. https://www-cdn.anthropic.com/6 d8a8055020700718b0c49369f60816ba2a7c285.pdf, 2025b.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

ByteDance Seed. Seed2.0 model card: Towards intelligence frontier for real-world complexity. arXiv preprint arXiv:2607.00248, 2026.

Shuang Chen, Kaituo Feng, Hangting Chen, Wenxuan Huang, Dasen Dai, Quanxin Shou, Yunlong Lin, Xiangyu Yue, Shenghua Gao, and Tianyu Pang. OpenSearch-VL: An open recipe for frontier multimodal search agents. arXiv preprint arXiv:2605.05185, 2026.

Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. Can pre-trained vision and language models answer visual information-seeking questions? arXiv preprint arXiv:2302.11713, 2023.

Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang, Jiaheng Liu, Yuying Mai, et al. SimpleVQA: Multimodal factuality evaluation for multimodal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4637–4646, 2025.

Yong Xien Chng, Tao Hu, Wenwen Tong, Xueheng Li, Jiandong Chen, Haojia Yu, Jiefan Lu, Hewei Guo, Hanming Deng, Chengjun Xie, et al. SenseNova-MARS: Empowering multimodal agentic reasoning and search via reinforcement learning. arXiv preprint arXiv:2512.24330, 2025.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Zhen Fang, Yu Zeng, Wenxuan Huang, Yiming Zhao, Shiting Huang, Tianfei Ren, Qi Lu, Qingnan Ren, Qisheng Su, Lionel Z. Wang, Qingyu Yin, Shuang Chen, Zehui Chen, Lin Chen, Zhenfei Yin, Yao Hu, Shaohui Lin, Wanli Ouyang, Shaosheng Cao, and Feng Zhao. Video-DeepResearch: Towards the next generation multimodal deepresearch agent. arXiv preprint arXiv:2608.03979, 2026.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-R1: Reinforcing video reasoning in multimodal large language models. arXiv preprint arXiv:2503.21776, 2025a.

Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, Yan Feng, Peng Pei, Xunliang Cai, and Xiangyu Yue. OneThinker: All-in-one reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025b.

Mingyang Fu, Yuyang Peng, Benlin Liu, Yao Wan, and Dongping Chen. LiveVQA: Live visual knowledge seeking. arXiv preprint arXiv:2504.05288, 2025.

Zhenkun Gao, Yicheng Bao, Jinlong Peng, Xueheng Li, Theo Huang, Bangwei Liu, Kunquan Li, Zhenye Gan, Tao Hu, Chengjun Xie, Mingqian Yang, Xuanhua He, Zhizhong Zhang, Xin Tan, Chengjie Wang, and Yuan Xie. VideoSearcher: Empowering video deep research with multi-tool agentic reasoning via reinforcement learning. arXiv preprint arXiv:2607.02927, 2026.

Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. WebWatcher: Breaking new frontier of vision-language deep research agent. arXiv preprint arXiv:2508.05748, 2025a.

Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. WebWatcher: Breaking new frontiers of vision-language deep research agent. arXiv preprint arXiv:2508.05748, 2025b.

Jack Hong, Chenxiao Zhao, ChengLin Zhu, Weiheng Lu, Guohai Xu, and Xing Yu. DeepEyesV2: Toward agentic multimodal model. arXiv preprint arXiv:2511.05271, 2025.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-R1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

Wenxuan Huang, Yu Zeng, Qiuchen Wang, Zhen Fang, Shaosheng Cao, Zheng Chu, Qingyu Yin, Shuang Chen, Zhenfei Yin, Lin Chen, Zehui Chen, Xu Tang, Yao Hu, Shaohui Lin, Philip Torr, Feng Zhao, and Wanli Ouyang. Vision-DeepResearch: Incentivizing deepresearch capability in multimodal large language models. arXiv preprint arXiv:2601.22060, 2026.

Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui Chen, Chaoyou Fu, Guanglu Song, Peng Gao, Yu Liu, Chunyuan Li, and Hongsheng Li. MMSearch: Benchmarking the potential of large models as multi-modal search engines. In International Conference on Learning Representations, 2025.

Zhengbo Jiao, Yiming Cheng, Yilei Jiang, Kaituo Feng, Rui Huang, Tianyi Jiang, Juanxi Tian, Jiapeng Li, Qunzhong Wang, Tailai Chen, Qianshan Wei, Chuan Xiao, Shanyu Rong, Yangfu Li, Yanhan Zhou, Yunpu Ma, Yifan Zhang, and Xiangyu Yue. SearchEyes: Towards frontier multimodal deep search intelligence via search world simulation. arXiv preprint arXiv:2607.05943, 2026.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Hongyu Li, Jinyu Chen, Ziyu Wei, Shaofei Huang, Tianrui Hui, Jialin Gao, Xiaoming Wei, and Si Liu. LLaVA-ST: A multimodal large language model for fine-grained spatial-temporal understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8592–8603, June 2025a.

Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Qwen3-VL-Embedding and Qwen3-VL-Reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.

Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. VideoChat-R1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025b.

Chengwen Liu, Xiaomin Yu, Zhuoyue Chang, Zhe Huang, Shuo Zhang, Heng Lian, Kunyi Wang, Rui Xu, Sen Hu, Jianheng Hou, Hao Peng, Chengwei Qin, Xiaobin Hu, Hong Peng, Ronghao Chen, and Huacan Wang. Watching, reasoning, and searching: A video deep research benchmark on open web for agentic video reasoning. arXiv preprint arXiv:2601.06943, 2026.

Ziyu Liu, Yuhang Zang, Yushan Zou, Zijian Liang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual agentic reinforcement fine-tuning. arXiv preprint arXiv:2505.14246, 2025.

Kartik Narayan, Yang Xu, Tian Cao, Kavya Nerella, Vishal M. Patel, Navid Shiee, Peter Grasch, Chao Jia, Yinfei Yang, and Zhe Gan. DeepMMSearch-R1: Empowering multimodal LLMs in multimodal web search. arXiv preprint arXiv:2510.12801, 2025.

OpenAI. Introducing GPT-4.1 in the API. https://openai.com/index/gpt-4-1/, 2025a.

OpenAI. Introducing GPT-5. https://openai.com/index/introducing-gpt-5/, 2025b.

OpenAI Team. GPT-4o system card, 2024.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. VLM-R1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

Yubo Sun, Chunyi Peng, Yukun Yan, Zhenghao Liu, Sen Mei, Bangrui Xu, Xuanhe Zhou, Chi Chen, and Maosong Sun. HiEviDR-Bench: A benchmark for hierarchical evidence aggregation in deep research. arXiv preprint arXiv:2607.25151, 2026.

Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton Van Den Hengel. FVQA: Fact-based visual question answering. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(10):2413–2427, 2017.

Shengqin Wang, Wentao Yan, Huichi Zhou, Yihang Chen, Kun Shao, Zhizhong Zhang, and Yuan Xie. DR-MMSearchAgent: Deepening reasoning in multimodal search agents. arXiv preprint arXiv:2604.19264, 2026.

Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-R1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025.

Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, and Ziwei Liu. MMSearch-R1: Incentivizing LMMs to search. arXiv preprint arXiv:2506.20670, 2025.

Ke Xu, Han Xu, Xinran Chen, Yuqian Wang, Zhixuan Li, Xiaojian Liu, Changwo Wu, Jianqiang Xia, and Yuchen Li. STAMP: Provenance-guided credit assignment for deep search agents. arXiv preprint arXiv:2607.11172, 2026.

Wentao Yan, Shengqin Wang, Huichi Zhou, Yihang Chen, Kun Shao, Yuan Xie, and Zhizhong Zhang. ProMM SearchAgent: A generalizable multimodal search agent trained with process-oriented rewards. arXiv preprint arXiv:2604.20486, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Huanjin Yao, Qixiang Yin, Min Yang, Ziwang Zhao, Yibo Wang, Haotian Luo, Jingyi Zhang, and Jiaxing Huang. MM-DeepResearch: A simple and efective multimodal agentic search baseline. arXiv preprint arXiv:2603.01050, 2026.

Zuyao You and Zuxuan Wu. Seg-R1: Segmentation can be surprisingly simple with reinforcement learning. arXiv preprint arXiv:2506.22624, 2025.

Yu Zeng, Wenxuan Huang, Zhen Fang, Shuang Chen, Yufan Shen, Yishuo Cai, Xiaoman Wang, Zhenfei Yin, Lin Chen, Zehui Chen, Shiting Huang, Yiming Zhao, Yao Hu, Philip Torr, Wanli Ouyang, and Shaosheng Cao. Vision-DeepResearch Benchmark: Rethinking visual and textual search for multimodal large language models. arXiv preprint arXiv:2602.02185, 2026.

Jiajie Zhang, Xin Lv, Ling Feng, Lei Hou, and Juanzi Li. Chaining the evidence: Robust reinforcement learning for deep search agents with citation-aware rubric rewards. arXiv preprint arXiv:2601.06021, 2026.

## Appendix Contents

A Details of the VGEG-Centered Multimodal Data Engine 17   
A.1 Stage 1: Visual Source Curation 17   
A.2 Stage 2: Dense Visual Anchor Discover 18   
A.3 Stage 3: Web Evidence Graph Construc 18   
A.4 Stage 4: VGEG-Based Task Construct 20   
A.5 Stage 5: Expert Trajectory Synthesis 21   
B Benchmark Construction and Evaluation 21   
B.1 Visual Input Construction . 21   
B.2 Research Operations and Reference Structure 22   
B.3 Dificulty and Visual-Domain Distributions 22   
B.4 Human Review and Annotation 23   
C Tool Interface and Trajectory Representation 23   
C.1 Resource Registry and Tool Availability 23   
C.2 Tool Execution Semantics . 24   
C.3 Observation and Trajectory Serialization 25   
D Training and Evidence Reward Details 25   
D.1 Supervised Trajectory Learning . 25   
D.2 Reinforcement Learning Data Constructio 25   
D.3 From VGEG Annotations to EVGR Rubric 26   
D.4 Reward Composition and Policy Optimizat 27   
E Qualitative Case Studies 29   
E.1 Multi-image Evidence Composition 29   
E.2 Cross-frame Evidence Compositi 31

## A Details of the VGEG-Centered Multimodal Data Engine

This appendix expands the five stages in Figure 1. The input-level graph G<sub>X</sub> collects candidate visual and web evidence for a visual source. A task-level VGEG $\Gamma _ { i }$ is derived from $\mathcal { G } _ { X }$ through a task-conditioned projection: it selects the anchors and facts needed by a particular question and augments them with explicit answer-producing operations and dependencies. We report the recorded sampling and execution settings for each stage; prompt templates are omitted.

## A.1 Stage 1: Visual Source Curation

Figure 1(a) contains six blocks: Web Video Source Pool, Upload Date Filter, Metadata Coarse Filter, Category Balance & Exclusion, Video Snippet LLM Filter, and Video Duration Balance. Table 5 gives the recorded size after each block.

Table 5 Source-video curation. Counts refer to candidate video records.
<table><tr><td>Stage</td><td>Retained records</td></tr><tr><td>Web video source pool</td><td>2,500,000</td></tr><tr><td>Upload date filter</td><td>1,295,220</td></tr><tr><td>Metadata coarse filter</td><td>377,955</td></tr><tr><td>Category balance &amp; exclusion</td><td>127,562</td></tr><tr><td>Video snippet LLM filter</td><td>72,403</td></tr><tr><td>Video duration balance</td><td>70,781</td></tr></table>

Step 1: Web Video Source Pool. The pipeline starts from 2.5M candidate YouTube video records. This pool provides the common input to all subsequent source-level filters.

Step 2: Upload Date Filter. We retain records with available metadata and an upload date on or after January 1, 2023. This stage reduces the pool to 1.3M records.

Step 3: Metadata Coarse Filter. A rule-based score combines entity and event cues in the title, description, and tags with duration and metadata richness. We retain the higher-scoring records, leaving 378k candidates.

Step 4: Category Balance & Exclusion. We retain Film & Animation, Autos & Vehicles, Pets & Animals, Sports, Travel & Events, Gaming, People & Blogs, Comedy, Entertainment, News & Politics, Howto & Style, and Science & Technology. Candidates are ranked within each category, with equal category quotas used to maintain content coverage and per-channel limits used to prevent concentration on a small number of sources. We exclude Music, Nonprofits & Activism, Education, and Unknown because these categories more often exhibit limited visual dynamics or contain fewer identifiable, searchable entities for evidence-grounded task construction. This stage retains 128k records.

Step 5: Video Snippet LLM Filter. We use GPT-4.1 (OpenAI, 2025a) to assess whether each video potentially contains multiple scenes, searchable entities, specific events, visual information, and available web evidence. We combine its assessment with the rule-based metadata score and retain 72k candidates. This step operates on textual metadata; visual inspection occurs in the subsequent anchor-discovery stage.

Step 6: Video Duration Balance. We group the remaining candidates by video duration and sample across duration ranges to reduce the dominance of any single interval. This step yields 70k records with a more balanced duration distribution.

## A.2 Stage 2: Dense Visual Anchor Discovery

Figure 1(b) contains five blocks: Video Clip & Caption, Key-Frame Selection & Caption, Cross-Clip Frame Deduplication, Event Aggregation & Caption, and Event Key-Frame Selection & Key Object Grounding. Their outputs form the Video Dense Caption Tree.

Step 1: Video Clip & Caption. We uniformly sample each video at a target rate of 2 FPS, and divide the frames into temporally ordered local clips of ten frames. A final group with fewer than five frames is merged into its predecessor. We use Seed 2.0 Pro (ByteDance Seed, 2026) to describe the scenes, actions, and identifiable entities in each clip.

Step 2: Key-Frame Selection & Caption. For each clip, we use Seed 2.0 Pro (ByteDance Seed, 2026) to select representative key frames and generate key-moment descriptions and candidate entity cues, while retaining the frame indices and timestamps.

Step 3: Cross-Clip Frame Deduplication. We use Qwen3-VL-Embedding-8B (Li et al., 2026) to encode each candidate frame and its description as normalized image and text embeddings. Candidates are compared in temporal order against the most recently retained frame. A candidate is removed when image similarity is at least 0.9 or text similarity is at least 0.8. The similarity thresholds are stored with the deduplication output.

Step 4: Event Aggregation & Caption. We use Seed 2.0 Pro (ByteDance Seed, 2026) to merge temporally adjacent, semantically related clips into coherent events based on the clip descriptions and deduplicated key frames. Each event records a title, temporal range, description, covered clip range, and searchable entity cues, forming the temporal hierarchy from the video to its events.

Step 5: Event Key-Frame Selection & Key Object Grounding. For each event, we select representative key frames and use Qwen3-VL-30B-A3B-Instruct (Bai et al., 2025) to localize visible, searchable objects. Each object records a name or description, a type, a normalized bounding box, and its relation to the event. Bounding boxes use coordinates in [0, 1000]<sup>4</sup>.

Video Dense Caption Tree. The resulting hierarchy preserves temporal and spatial references:

$$
{ \mathrm { v i d e o } }  { \mathrm { e v e n t } }  { \mathrm { k e y ~ f r a m e } }  { \mathrm { l o c a l i z e d ~ o b j e c t . } }\tag{8}
$$

These records supply the visual anchors used by later retrieval and question construction.

## A.3 Stage 3: Web Evidence Graph Construction

Figure 1(c) routes localized Objects through image search or OCR followed by text search, assembles a Candidate Entity Pool, expands related entities and attributes through the Expand Entity Pipeline, and produces a Web Entity Graph.

Objects and Retrieval Routing. Object grouping and representative-view selection use deterministic matching. Objects with matching types and similar entity or name cues are grouped across frames; the largest available bounding box is selected as the representative view, with alternate views retained from other frames. Crops with a short side below 256 pixels are enlarged to 256 pix els. We use GPT-4.1 (OpenAI, 2025a) to route each object to reverse image search, OCR followed by text search, or omission based on its crop, name, type, candidate entity cues, and event relation. For the OCR route, we also use the model to read visible text and propose a candidate entity; the image-search route uses an external reverse-image-search service to retrieve candidate entities and webpages.

Candidate Entity Pool. Because reverse image search is sensitive to viewpoint and cropping, we construct a retrieval pool for each object grouped across frames, containing its representative crop and alternate views from other sampled frames. If one crop yields no usable candidates, we retry the search with another view from the same object pool. We first use Seed 2.0 Pro (ByteDance Seed, 2026) to select visually grounded anchors with research potential from the event context and initial retrieval results. We then use the model to combine the object crop, local visual description, and image-search or OCR evidence to propose an entity identity, assess whether the retrieved entity matches the visual object, and formulate entity-specific text queries. We use Qwen3-32B (Yang et al., 2025) to read the retrieved pages, produce query-focused summaries, and verify consistency between the page evidence and the candidate identity; search snippets are used when the page body is unavailable. Duplicate pages and explicitly mismatched identities are removed. The remaining identities, source objects, and webpage evidence form the candidate entity pool.

Expand Entity Pipeline. We use Seed 2.0 Pro (ByteDance Seed, 2026) to extract sourcesupported relations from the verified page summaries. Each fact records its head entity, relation, tail entity or attribute value, supporting URL, and evidence quotation:

$$
f = ( u , r , z , p , \xi ) .\tag{9}
$$

Here, p is the source URL and $\xi$ is the retained supporting text. A deterministic parser checks required fields and matches normalized quotations against the supplied source material; a matched quotation must contain at least six normalized characters. Numeric facts can additionally retain units and value types. For entity-valued tails, we use Seed 2.0 Pro (ByteDance Seed, 2026) to decide whether to continue based on entity specificity, relation confidence, task relevance, and evidenceexpansion potential and to propose a follow-up query; we then use Qwen3-32B (Yang et al., 2025) to summarize and verify the retrieved expansion pages. For each expandable entity, the pipeline retains one query, retrieves five results, deduplicates pages by host, and reads the selected pages. Each parent selects high-confidence successors with scores of at least 0.5, and expansion stops when no suitable successor remains.

Web Entity Graph. Graph assembly uses no additional model and deterministically merges the preceding outputs. The graph contains events, frames, localized objects, entities, attributes, and source pages. Structural edges associate events with frames and frames with objects; grounding edges bind objects to candidate entities; factual edges connect entities to other entities or attributes while retaining source evidence. Entity and relation normalization merges compatible records, removes duplicate edges, and removes self-loops. The resulting input-level graph $\mathcal { G } _ { X }$ supplies a common evidence context for multiple candidate questions.

## A.4 Stage 4: VGEG-Based Task Construction

Figure 1(d) contains four blocks: LLM-Guided QA Generation, QA Verifier, Visual Entity Fuzzing Rewrite, and Quality Filter. VGEG is retained as the task-level reference throughout these blocks.

Step 1: LLM-Guided QA Generation. We organize clip and event descriptions from the dense video caption tree, visual-anchor records, source-supported facts from the input-level graph $\mathcal { G } _ { X }$ and candidate relation chains into a generation context. We use Seed 2.0 Pro (ByteDance Seed, 2026) to propose candidate questions spanning the six research operations. The model uses visual descriptions and anchor mappings rather than raw video frames and outputs a question, answer, supporting facts, operation type, visual references, and corresponding frame indices. Multi-anchor questions retain a separate visual description and frame mapping for each referenced object. Rela tion chains can occur within each branch before the branch results are joined, counted, compared, or used in arithmetic.

The same step assembles the task-level VGEG using the main-text definition $\Gamma _ { i } = \Phi _ { i } ( { \mathcal G } _ { X } ) =$ $( \mathcal { A } _ { i } , \mathcal { F } _ { i } , \mathcal { O } _ { i } , \mathcal { R } _ { i } )$ . The projection selects $A _ { i }$ from the visual anchors and grounded entities in $\mathcal { G } _ { X }$ retaining their image, frame, or region locations. It selects ${ \mathcal { F } } _ { i }$ from the graph relations and attributes that provide source support for the task. The operation set $\mathcal { O } _ { i }$ describes the answer-producing retrieval or composition, while $\mathcal { R } _ { i }$ retains the selected grounding and support edges and adds task specific dependencies connecting anchors, entities, facts, and operations. These components are assembled deterministically from the visual-reference fields, selected used\_facts, task-structure labels, source records, and their grounding and dependency links, without an additional model call.

Step 2: QA Verifier. The verifier first checks the evidence path and task structure against the candidate graph using deterministic rules without an additional model call. Relation-chain questions are checked for continuity between successive facts. Multi-anchor questions require distinct referenced objects and corresponding supporting facts. Comparisons use a common semantic attribute; arithmetic uses numeric operands with compatible meanings; knowledge-conditioned counting associates counted entities with facts supporting the condition. Frame references are checked against the available entity-to-frame mappings, and the facts required to derive the answer must be supported by the candidate graph.

For a candidate $( q _ { i } , y _ { i } , \Gamma _ { i } )$ , the QA Verifier then gathers graph relations, source quotations, page summaries, and visual descriptions associated with its supporting entities. We use Seed 2.0 Pro (ByteDance Seed, 2026) to re-answer the question using only this evidence and to check semantic agreement between the regenerated and proposed answers, allowing aliases and equivalent units or date formats. To prevent answer leakage, explicit occurrences of the candidate answer in the fact summary are masked when possible. This stage checks that the supplied evidence supports recovery of the answer.

Step 3: Visual Entity Fuzzing Rewrite. We use Seed 2.0 Pro (ByteDance Seed, 2026) to inspect the referenced key frames and rewrite explicit entity names into descriptions that locate the intended objects. Multi-hop questions also conceal intermediate entity names so that the final question starts from visible targets without exposing entities along the retrieval path.

The same step instantiates the task for its target visual input type. For a multi-image task, we extract the deduplicated key frames referenced by its VGEG, organize them as an unordered image collection, and remap frame- and region-level references to image indices. References to video playback, timestamps, and unnecessary temporal order are removed so that the question is selfcontained over the image collection. For a video task, we retain the original video and its event, timestamp, key-frame, and region bindings, preserving the need for temporal localization.

Step 4: Quality Filter. We use Seed 2.0 Pro (ByteDance Seed, 2026) to assess information masking, referential uniqueness, visual relevance, and whether external retrieval is required. A question-only probe rejects a candidate when the model can reliably recover the reference answer without the visual input. The retained record contains the visual input, final question, answer, VGEG annotations, operation label, and visual-reference mappings.

## A.5 Stage 5: Expert Trajectory Synthesis

Figure 1(e) contains two steps: multi-turn trajectory synthesis and rejection sampling.

Multi-turn trajectory synthesis. We use Seed 2.0 Pro (ByteDance Seed, 2026) as the expert model to solve each verified task in the real tool environment. The expert receives the visual input, question, and tool definitions, while the reference answer is withheld for subsequent evaluation. At each turn, the model produces reasoning followed by a tool call or a final response. The environment executes the tool and appends its observation, including returned visual resources, to the interaction history, producing an interleaved reasoning–action–observation trajectory.

Rejection sampling: answer-correctness judge. We use GPT-4o (OpenAI Team, 2024) to compare the final response of each raw expert rollout with the reference answer while allowing semantically equivalent formulations. Rollouts with an incorrect answer or without a valid terminal response are rejected.

Rejection sampling: process-level judge. The remaining rollouts undergo process-level evaluation, which checks that each trajectory contains at least one efective tool call, remains logically consistent with tool observations, and avoids inefective repetition. We retain only trajectories that pass both judges as high-quality expert demonstrations for SFT.

## B Benchmark Construction and Evaluation

This appendix details the construction protocols for OneSearch-MI-Bench and OneSearch-Video-Bench, including visual-input construction, VGEG-based operation assignment, dificulty and visualdomain annotation, and human review.

## B.1 Visual Input Construction

For OneSearch-MI-Bench, deduplicated evidence frames from the same source video form an unordered image collection. Questions are rewritten to refer to the images and visible objects, while image indices preserve the bindings to their visual anchors. References to video playback, frame numbers, and unnecessary temporal order are removed. Each retained item requires evidence from at least two images, preventing reduction to a single-image question.

For OneSearch-Video-Bench, we preserve the temporal structure of the source video and bind each visual anchor to its corresponding event and key frames. Questions refer to relevant content through visible objects, events, or temporal descriptions, requiring the model to localize visual evidence before retrieving and composing external facts.

## B.2 Research Operations and Reference Structures

Each benchmark item stores $\left( X _ { i } , q _ { i } , y _ { i } , \Gamma _ { i } , s _ { i } \right)$ , where $s _ { i }$ is determined by the final answer-producing operation in its task-level VGEG rather than by visual content or source category. Figure $2 ( \mathrm { a } ) { - } ( \mathrm { f } )$ shows representative examples of the six operations, and Table 6 provides their definitions and distributions.

When evidence branches contain intermediate relation chains, the category follows the operation that combines the branch results into the answer. For example, two anchors may each initiate multihop retrieval, but a question is labeled as multi-anchor comparison when its final answer compares the retrieved attributes. This rule assigns one principal operation to each item while preserving all intermediate dependencies in its VGEG.

The two benchmarks contain 608 questions. The five compositional categories beyond single-anchor lookup account for 555 questions, or 91.3% of the combined set. Multi-image questions contain 2–8 images, with a mean of 3.05 images per question. These statistics show that the benchmarks primarily test relation tracing, conditional filtering, and cross-anchor fact composition.

Table 6 Definitions and distribution of the six principal research operations.
<table><tr><td>Operation</td><td>Definition</td><td>Multi-image</td><td>Video</td><td>Total</td></tr><tr><td>Single-anchor lookup</td><td>Retrieve one external attribute for one anchor.</td><td>28</td><td>25</td><td>53</td></tr><tr><td>Multi-hop retrieval</td><td>Follow an external relation chain from one anchor.</td><td>48</td><td>47</td><td>95</td></tr><tr><td>Knowledge-conditioned count</td><td>Test a retrieved condition per anchor, then count.</td><td>57</td><td>61</td><td>118</td></tr><tr><td>Multi-anchor join</td><td>Combine retrieved facts across anchors.</td><td>58</td><td>52</td><td>110</td></tr><tr><td>Multi-anchor arithmetic</td><td>Compute over retrieved numeric facts.</td><td>61</td><td>45</td><td>106</td></tr><tr><td>Multi-anchor comparison</td><td>Compare the same retrieved attribute across anchors.</td><td>49</td><td>77</td><td>126</td></tr><tr><td>All questions</td><td></td><td>301</td><td>307</td><td>608</td></tr></table>

## B.3 Difficulty and Visual-Domain Distributions

We derive a structural dificulty score from the principal research operation, reasoning depth, and number of required facts. Let $h _ { i }$ denote the number of reasoning hops, $| \mathcal { F } _ { i } |$ the number of used facts in the VGEG, and $w ( s _ { i } )$ the operation weight, set to 0 for single-anchor lookup, 1 for multi-hop retrieval and knowledge-conditioned counting, and 2 for the three multi-anchor operations:

$$
d _ { i } = 1 0 w ( s _ { i } ) + h _ { i } + 0 . 1 | \mathcal { F } _ { i } | .\tag{10}
$$

Within each benchmark, examples are ranked by this score and partitioned into easy, medium, and hard groups. Both branches yield empirical boundary scores of 14.4 and 22.2; ties at a boundary are deterministically assigned across adjacent groups to keep their sizes balanced. The three groups contain 100, 100, and 101 multi-image questions and 102, 102, and 103 video questions, respectively. Figure 4(a) shows the full score distributions and their partition boundaries. These labels represent relative evidence complexity within each visual input type rather than independently calibrated human dificulty.

We divide the source videos into six broad visual domains: Daily Life, Entertainment, Geography, Technology, Sports, and News. As shown in Figure 4(b), every visual domain spans all six principal research operations, indicating that the operation taxonomy is not tied to a particular source domain. Diferences in their proportions reflect the benchmark’s overall emphasis on compositional research questions.

(a) Difficulty score distribution  
![](images/68d54983fa0cc7a50d827276de5f6bcb9900d2c3df937c705aa13094006404d6.jpg)

(b) Research operations by visual domain  
![](images/38ef04a6c5b0937b7ad4219d4251f9cfa0c1dc335b3c3bd755b290f8ca0dfa93.jpg)  
Figure 4 Benchmark distributions. (a) Smoothed structural-dificulty score distributions, with circles marking observed values. Dashed lines mark the empirical boundaries at 14.4 and 22.2; the easy, medium, and hard groups contain 100/100/101 multi-image and $1 0 2 / 1 0 2 / 1 0 3$ video questions. (b) Operation composition within six visual domains. Bars are normalized within each domain, and n denotes the number of questions. Every domain covers all six principal research operations.

## B.4 Human Review and Annotation

Candidates first pass the automatic information-masking, referential-uniqueness, visual-relevance, and non-triviality checks described in Appendix A.4. These checks prevent answer or intermediateentity leakage, ensure that each visual reference identifies its intended object, remove questions that do not require the visual input, and reject questions that can be answered reliably from text alone.

After automatic filtering, human annotators jointly inspect the visual input, question, reference answer, task-level VGEG, and supporting webpages. They verify that each visual reference identifies the intended object, each required fact is source-supported, the reference answer follows from the VGEG dependencies, and the principal operation matches the final answer-producing step. The reviewed record retains the confirmed visual-anchor bindings, supporting facts, reference answer, and operation annotation.

A final quality review further checks question clarity, the presence and distinguishability of the referenced visual objects, and consistency across the annotations. Candidates with ambiguous visual references, unsupported required facts, indeterminate answers, or unclear operation labels are rejected from the final benchmarks.

## C Tool Interface and Trajectory Representation

This appendix specifies the execution semantics of the tools in Table 1, including visual-resource registration, tool inputs and outputs, and the common interaction records used for expert synthesis, reinforcement learning, and inference.

## C.1 Resource Registry and Tool Availability

At the beginning of an episode, the environment registers the visual input in an episode-level resource table. A single image and each member of an image collection receive separate image handles, while a video handle refers to the sampled frame grid visible to the policy. Crops, enhanced images, extracted frames, and selected clips are registered as new image or clip handles and can be reused by later calls. This enables compositions such as selecting a video frame, cropping a local region, and then applying OCR or image search.

Single-image and multi-image inputs expose $\mathcal { T } _ { \mathrm { v i s } } \cup \mathcal { T } _ { \mathrm { r e t } }$ , while video inputs additionally expose T<sub>temp</sub>. All three visual input types use the same call protocol: one tool call is executed per interaction turn, and its observation is returned before the next action.

## C.2 Tool Execution Semantics

Crop. We implement spatial cropping with PIL. Given an image handle and a normalized box $( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } ) \in [ 0 , 1 0 0 0 ] ^ { 4 }$ , the executor maps the box to pixel coordinates, extracts the region, and registers it as a new image resource. The tool isolates local visual cues before OCR, enhancement, or image search.

OCR. We implement OCR with PaddleOCR. The backend detects text regions, recognizes their contents, and returns the text spans together with bounding boxes and confidence scores as a textual observation. The tool is typically applied to cropped or enhanced signs, documents, logos, and captions.

PerspectiveCorrect. We implement perspective correction with OpenCV. The executor applies grayscale conversion, Gaussian smoothing, and Canny edge detection, locates a quadrilateral among the external contours, orders its corners, and estimates a four-point perspective transform. The rectified fronto-parallel view is registered as a new image resource.

SuperResolution. We implement super-resolution with EDSR through the OpenCV dnn\_- superres interface. The network upsamples the input at the scale supplied by the call and registers the result as a new image resource for subsequent OCR, cropping, or image search.

Sharpen. We implement sharpening with OpenCV unsharp masking. Given amount α, the tool combines the input with its Gaussian-blurred version as $I _ { \mathrm { o u t } } = ( 1 + \alpha ) I - \alpha ( G _ { \sigma } * I )$ , enhancing edges and fine details before registering the result as a new image resource.

ImageSearch. The executor materializes a registered image as a reference accessible to an external image-to-image search service. It condenses the returned candidate identities, visual matches, source pages, and titles into a textual observation. This tool provides the bridge from a visual anchor to real-world entities and related webpages.

TextSearch. Text search calls an external search service that performs web retrieval, page reading, and query-focused summarization within one request. Each returned passage retains its title, URL, and summary, enabling identity verification and source-grounded fact acquisition.

SelectTimespan. This tool operates on the sampled frame grid of a video or an existing clip. For a resource with N frames, it selects the half-open interval [s, e), where $0 \leq s < e \leq N$ , slices the corresponding frame list, and uses FFmpeg to materialize the associated video subclip. The selected frames are re-indexed from zero and registered with the subclip as a new clip resource.

SelectFrame. Given an index $0 \leq i < N$ , this tool resolves the corresponding frame from the sampled grid and registers it as a new image resource. The returned handle can be passed directly to cropping, OCR, enhancement, and image search, connecting temporal localization to the com mon image-tool pipeline. Both temporal tools use sampled-grid indices rather than timestamps in seconds.

## C.3 Observation and Trajectory Serialization

Each tool execution produces a textual observation $O t$ and may additionally create a set of visual resources $\Delta \mathcal { R } _ { t }$ . We serialize the corresponding turn as $\boldsymbol { r } _ { t } = \left( a _ { t } , \hat { c } _ { t } , o _ { t } , \Delta \mathcal { R } _ { t } \right)$ , where $a _ { t }$ is the original policy output and $\hat { c } _ { t }$ is the parsed tool name and arguments. Returned media and text are appended to the next interaction context, allowing subsequent reasoning to reference the executed result. A final response closes the trajectory without another tool execution.

Malformed calls, invalid resource handles, and out-of-range indices are returned as tool-error observations, allowing the policy to correct them in a later turn. Execution status is retained so that data filtering and reward computation can distinguish successful calls, recoverable errors, and incomplete trajectories.

## D Training and Evidence Reward Details

This appendix complements Section 5 with implementation details on supervised trajectory learning, the conversion from task-level VGEGs to EVGR rubrics, and the use of process rewards in GRPO.

## D.1 Supervised Trajectory Learning

SFT uses OneSearch-VL-SFT-110K, comprising approximately 110k expert trajectories: 36k single-image, 37k multi-image, and 35k video trajectories. All three visual input types use the unified multi-turn format defined in Appendix C, where each turn records model reasoning, a tool call, the resulting environment observation, and the subsequent response. Images, video clips, and textual observations returned by tools remain available through resource handles, preserving executable dependencies across tool calls.

Visual inputs, user questions, tool definitions, and tool observations serve as conditioning context, while the supervised targets comprise assistant reasoning, tool commands, and final responses. The loss mask is therefore one for policy-generated tokens and zero for environment observations and other conditioning tokens. This prevents the model from being trained to reproduce tool outputs while preserving its ability to condition subsequent actions on executed observations.

Table 7 reports the complete SFT configuration for OneSearch-VL-8B. We perform full-parameter finetuning while freezing the vision encoder and multimodal projector. Videos are sampled at 2 fps with at most 100,352 pixels per frame and 128 frames. SFT uses 64 H800 GPUs across eight nodes and runs for approximately four days.

## D.2 Reinforcement Learning Data Construction

We construct OneSearch-VL-RL-10K, containing approximately 10k tasks after filtering and deduplication: 3.7k single-image, 2.8k multi-image, and 3.6k video tasks. Candidates are drawn from the corresponding pools for the three visual input types, exposing online training to spatial grounding, cross-image evidence composition, and temporal localization.

<table><tr><td>Category</td><td>Hyperparameter</td><td>Value / Setting</td></tr><tr><td rowspan="5">Model</td><td>Base Model</td><td>Qwen3-VL-8B-Instruct</td></tr><tr><td>Image Max Pixels</td><td>262,144 (≈ 512 × 512)</td></tr><tr><td>Video Max Pixels</td><td>100,352 per frame</td></tr><tr><td>Video Sampling Rate</td><td>2 fps</td></tr><tr><td>Maximum Video Frames</td><td>128 True</td></tr><tr><td rowspan="5"></td><td>Trust Remote Code</td><td>Full</td></tr><tr><td>Finetuning Type</td><td>True</td></tr><tr><td>Vision Tower Frozen MM Projector Frozen</td><td>True</td></tr><tr><td>DeepSpeed Stage</td><td>ZeRO-2</td></tr><tr><td>Mixed Precision</td><td>bfloat16</td></tr><tr><td rowspan="6">Dataset</td><td>Total Samples</td><td>~110k</td></tr><tr><td>Visual Input Types</td><td>Single-image, multi-image, video</td></tr><tr><td>Template</td><td>qwen3_vl</td></tr><tr><td>Cutoff Length</td><td>32,768 tokens</td></tr><tr><td>Preprocessing Workers</td><td>1</td></tr><tr><td>Preprocessing Batch Size</td><td>16</td></tr><tr><td rowspan="7">Training</td><td>Dataloader Workers Batch Size per Device</td><td>4 1</td></tr><tr><td>Gradient Accumulation Steps</td><td>2</td></tr><tr><td>Effective Batch Size</td><td>128  $( = 1 \times 2 \times 6 4 ~ \mathrm { G P U s } )$ </td></tr><tr><td>Gradient Checkpointing</td><td>True</td></tr><tr><td>Learning Rate</td><td> $2 . 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Epochs</td><td>8</td></tr><tr><td>LR Scheduler</td><td>cosine</td></tr><tr><td rowspan="3">Infrastructure</td><td>Warmup Ratio</td><td>0.1</td></tr><tr><td>Total GPUs Training Time</td><td>64 NVIDIA H800 GPUs (8 nodes × 8) 4 days</td></tr><tr><td>Distributed Backend</td><td>Torchrun + DeepSpeed ZeRO-2</td></tr><tr><td rowspan="5">Logging /10</td><td>Logging Steps</td><td>5</td></tr><tr><td>Checkpoint Save Steps</td><td>500</td></tr><tr><td>Maximum Saved Checkpoints</td><td></td></tr><tr><td>Plot Loss</td><td>5 True</td></tr><tr><td>Report Backend</td><td>TensorBoard</td></tr></table>

Table 7 Agentic SFT configuration for OneSearch-VL-8B.

We use Qwen3-VL-8B (Bai et al., 2025) to perform eight independent rollouts for each candidate in the unified tool environment and retain tasks satisfying $0 < n _ { \mathrm { c o r r e c t } } < 8 .$ , where $n _ { \mathrm { c o r r e c t } }$ denotes the number of rollouts producing a correct final answer. We deduplicate the retained tasks to form the final RL dataset for subsequent GRPO optimization.

## D.3 From VGEG Annotations to EVGR Rubrics

As shown in Figure 3, each training item is paired with a structured rubric derived from its task-level VGEG. The rubric retains answer-core entities, required fact hops, supporting sources and snippets, and applicable image, region, or frame locations. The evaluator receives this rubric together with the question, reference answer, visual input type, and serialized executed trajectory, directly aligning the agent’s observations with the evidence required by the task.

The executed trajectory is serialized in interaction order, including reasoning, tool calls and arguments, and returned observations. For visual-grounding evaluation, the judge additionally receives multimodal inputs alongside the rubric and trajectory records. We use Seed 2.0 Pro (ByteDance Seed, 2026) in two separate calls for evidence traceability and visual grounding, each returning a score and a brief evidence-based explanation.

Evidence traceability. The judge checks whether tool observations establish the answer-core entities and fact hops required by the rubric and whether subsequent reasoning follows those observations. Missing evidence, unsupported entity substitutions, broken fact chains, and claims that contradict the retrieved sources reduce the score. This dimension evaluates the evidence process independently of an incidental match with the reference answer.

Visual grounding. The judge checks whether the trajectory identifies the visible entities, regions, or video moments required by the question and uses them to drive subsequent retrieval. For singleimage and multi-image inputs, the source images are directly visible to the agent policy, so an additional visual-tool call is not required; the criterion instead checks whether the correct images and objects are used. For videos, the selected frames are compared with the reference moments, and the visual identities obtained from those frames must guide the subsequent evidence search.

Table 8 Anchored EVGR scoring criteria used by the trajectory judge.
<table><tr><td></td><td>Score Evidence traceability</td><td>Visual grounding</td></tr><tr><td></td><td>1.00 All required fact hops are directly supported.</td><td>All required visual anchors are correctly grounded and used.</td></tr><tr><td></td><td>0.75 Most facts are supported, with one minor gap.</td><td>Grounding is mostly correct, with one minor localization gap.</td></tr><tr><td></td><td>0.50 Useful evidence is present, but a major hop is indirect or missing.</td><td>Only part of the required visual evidence is grounded.</td></tr><tr><td></td><td>0.25 Evidence is mostly noisy, speculative, or contradicted.</td><td>Identification is weak or incorrect, with limited recovery.</td></tr><tr><td></td><td>0.00 No supporting evidence is established</td><td>No grounding is established, or the visual identity remains wrong.</td></tr></table>

Each judge call must return a valid anchored score and its explanation. The rubric explicitly avoids rewarding verbosity, fluent but unsupported reasoning, inefective tool calls, or final-answer matching alone. The resulting $r _ { \mathrm { t r a c e } }$ and r<sub>ground</sub> can be used separately for ablations or combined into the full process reward R<sub>EVGR</sub>.

## D.4 Reward Composition and Policy Optimization

Using the reward terms defined in the main text, the weights for $R _ { \mathrm { a c c } } , R _ { \mathrm { q u e r y } }$ , and R<sub>EVGR</sub> are 0.6, 0.2, and 0.2, respectively, and $R _ { \mathrm { f m t } }$ gates the resulting weighted reward.

Following the main-text setup, we use Group Relative Policy Optimization (GRPO). Given a group of trajectories sampled for the same question, GRPO derives relative advantages from their withingroup rewards and updates the policy with a clipped objective. The policy loss is applied only to model-generated reasoning, tool-command, and final-response tokens, while tool observations remain conditioning context. For trajectories truncated by a terminal tool error, only the valid interaction prefix preceding the failure participates in optimization (Chen et al., 2026).

Table 9 summarizes the RL configuration for OneSearch-VL-8B. Training uses asynchronous SGLang rollouts and follows the OpenSearch-VL 8B recipe for the Megatron-LM actor and learning rate. RL uses 32 H800 GPUs across four nodes and runs for approximately four days.
<table><tr><td>Category</td><td>Hyperparameter</td><td>Value / Setting</td></tr><tr><td>Model</td><td>Initial Policy Training Dtype</td><td>OneSearch-VL-8B-SFT bfloat16</td></tr><tr><td>Data</td><td>Train Batch Size (prompts) Val Batch Size Max Prompt Length Max Response Length Data Seed Image Max Pixels Video Max Pixels Video Sampling Rate</td><td>256 64 40,960 tokens 16,384 tokens 3407 262,144 100,352 per frame 2 fps 128</td></tr><tr><td>Rollout</td><td>Engine Rollout Tensor Parallel # Samples per Prompt (n) GPU Mem. Utilization Train / Val Temperature Train / Val Top-p Top-k Strategy</td><td>SGLang (async mode) 8 0.60 0.7 / 0.7 1.0 / 0.95 -1 (disabled) Megatron-LM</td></tr><tr><td>Policy (Actor)</td><td>Tensor Parallel (TP) Pipeline Parallel (PP) Context Parallel (CP) PPO Mini-batch Size PPO Max Token Len / GPU Micro-batch Size / GPU Dynamic Batch Size Param / Optim / Grad Offload Gradient Checkpointing Actor LR</td><td>4 2 4 64 74,576 1 True CPU Full recompute, uniform (1 layer)  $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Optim. / Loss</td><td>PPO Clip Ratio (high) Entropy Coefficient Use KL Loss KL Loss / Controller Coef Loss Aggregation Advantage Estimator</td><td>0.28 0.0 False  $1 { \times } 1 0 ^ { - 3 } \ / \ 1 { \times } 1 0 ^ { - 3 }$  seq-mean-token-sum RLOO (within GRPO objective)</td></tr><tr><td></td><td>KL Type Fatal-aware Masking # Parallel Tasks</td><td>low-variance KL True (unknown + error) 512</td></tr><tr><td></td><td># Parallel Tool Calls Stepwise Advantage Cluster</td><td>1,024 False 32 NVIDIA H800 GPUs (4 nodes × 8)</td></tr><tr><td>Trainer</td><td>Training Time Save / Test Freq (steps) Total Training Steps Total Epochs Critic Warmup</td><td>4 days 10 / disabled 200 100</td></tr></table>

Table 9 GRPO configuration for OneSearch-VL-8B.

## E Qualitative Case Studies

This appendix presents example outputs of OneSearch-VL-8B, illustrating how the agent localizes visual anchors, links them to external facts, and produces answers. Figures 5–8 show four trajectories covering multi-image and cross-frame evidence composition. Each figure shows the question, visual material, model reasoning, tool observations, and final answer. Model responses are excerpted and tool observations are condensed, with all tool calls and arguments preserved.

Labels on input images, videos, selected frames, and crops match the handles used in the tool calls.

## E.1 Multi-image Evidence Composition

Multi-anchor arithmetic. Figure 5 asks for the diference between two manufacturers’ founding years. The agent crops the rear of a blue sports car and identifies a Ford Mustang through ImageSearch, then crops a brake-caliper logo in another image to identify Brembo. TextSearch returns 1903 for Ford and 1961 for Brembo, and the final response computes 1961 − 1903 = 58. The two visual anchors lead to distinct entities and numerical facts, which a subtraction operation combines into the answer, illustrating the dependencies captured by VGEG.

![](images/eb3cb35a4e8ca70ff33eafb05928eb0c1e5624294a3a7b69dfed3e2cc2604b4c.jpg)  
Figure 5 Multi-image arithmetic. Separate crops and image searches link the car and brake caliper to Ford and Brembo. Subtracting their retrieved founding years yields 58 years.

Knowledge-conditioned counting. Figure 6 asks how many car models in three separate images are produced by a company owned by Volkswagen. The agent first crops a logo, identifies Bentley through image search, and retrieves its ownership relation with Volkswagen. It then crops and image-searches each car to identify the Continental-family models and variants, followed by a query about their distinctions. Under the sample’s model/variant counting convention, the final response counts the three pictured cars and returns 3. This case links three visual anchors to an external ownership condition before aggregating the qualifying targets, with all ten tool calls retained.

![](images/48a91c74a71312e9b0ac790886a7fe60f7dc5156787104cbfd0428e6c3640a22.jpg)  
Figure 6 Knowledge-conditioned counting. Three input images and four crops support brand and model identification, followed by counting under a retrieved ownership condition. Image handles match the tool calls.

## E.2 Cross-frame Evidence Composition

The following cases retrieve evidence for distinct objects in two and three diferent frames of the same video, then compare their attributes. Frame strips provide neighboring visual context; blue borders and Used labels mark frames selected by the agent, and ellipses indicate omitted intervals. F denotes a selectable frame index in the benchmark input, not a timestamp in seconds.

Comparing two objects across frames. Figure 7 shows a video-only SFT checkpoint comparing two toy blasters. The agent selects F9, crops and identifies the Commander RD-6 through image search, and retrieves its release year. It then selects F116 and repeats the crop, image-search, and text-search sequence for the yellow Recon CS-6. Each branch is thus tied to a distinct frame, object, and external attribute. The final response compares the retrieved years 2020 and 2008 and returns 2020, matching the reference answer. The conflicting year summaries in the original trajectory remain visible in the figure.

![](images/245df1e95b316c25b377217e318af494de4a16887396d4b2752a52ca096712bb.jpg)  
Figure 7 Cross-frame attribute comparison. Blue borders mark selected frames F9 and F116; adjacent frames provide context. Each object undergoes cropping, image retrieval, and text retrieval before releaseyear comparison. All eight calls and arguments are retained.

Comparing landmarks in three frames. Figure 8 extends cross-frame retrieval to three anchors. The agent selects F0, F12, and F15, independently crops and image-searches the buildings, and identifies the Chicago Board of Trade, Chicago Temple, and John Hancock Center. For the first building, it enlarges an initially small clock crop before searching. It then retrieves building heights and issues follow-up queries to clarify conflicting values, ultimately returning 1,128 feet, matching the reference. The saved observations retain inconsistent height definitions, and the final response adds an inaccurate antenna height. The trajectory makes the distinction between visual grounding and evidence traceability concrete: the three buildings are independently grounded, while their numerical evidence remains inconsistent.

![](images/b49525700f44b7db9aeb6579b7ea7056ec2d966ba384ccb4edf53f40098c1426.jpg)  
Figure 8 Three-anchor cross-frame retrieval. Blue borders mark selected frames F0, F12, and F15 amid neighboring views. The three buildings are independently image-searched before height comparison. All fourteen calls, crop refinement, follow-up searches, and conflicting observations are retained.