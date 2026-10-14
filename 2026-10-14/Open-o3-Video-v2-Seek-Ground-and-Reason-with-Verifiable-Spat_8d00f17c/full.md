# Open-o3-Video-v2: Seek, Ground, and Reason with Verifiable Spatio-Temporal Evidence

Open-o3-Video Team: PKU, FDU, NTU, UC Merced

## Abstract

Reliable video reasoning requires models to produce correct answers and expose supporting visual evidence. Open-o3-Video enables verifiable reasoning by embedding timestamps and object-level bounding boxes into reasoning traces, but is restricted to frames from an initial sparse sampler. We introduce Open-o3-Video-v2, which formulates video reasoning as a unified agentic evidencegrounded process. The model identifies missing information, acquires additional observations, grounds the moments and objects supporting its reasoning, and writes this evidence into the trajectory before continuing or answering. To support this behavior, we construct STGR-v2-CoT-135K, including 52K multi-turn agentic grounded reasoning trajectories, and STGR-v2-RL-13K for reinforcement learning. We further propose Clue-Seek GRPO (CS-GRPO), which tracks how tool calls change accumulated evidence, rewards clue coverage and step-wise gains, and penalizes excessive, repetitive, or uninformative retrieval. On V-STaR, Open-o3-Video-v2-4B achieves 40.4% mAM and 56.3% mLGM, outperforming its Qwen3-VL-4B backbone by 8.5% and 12.6%, respectively. It also delivers consistent gains across general video understanding, long-video reasoning, and temporal grounding benchmarks. These results show that active evidence acquisition and verifiable object-level grounding are complementary for reliable video reasoning. The code and models are released at https://msalab-pku.github.io/projects/Open-o3-Video-v2.

## 1 Introduction

Video understanding has advanced rapidly with multimodal large language models (MLLMs) [1, 2, 44, 69]. These models can summarize long videos, answer questions about dynamic events, and localize relevant segments. Post-training methods [18, 53] further strengthen temporally aware reasoning through chain-ofthought supervision and reinforcement learning. Yet reliable, fine-grained reasoning over long and cluttered scenes remains dificult. Rapid motion, frequent scene changes, and subjects repeatedly leaving and reentering the view make visual events dificult to track, while a decisive clue may occur briefly or occupy only a small region. These settings demand more than fluent language reasoning: a capable model must locate reliable visual evidence and compose it into a defensible inference. Recent work on Thinking with Images [40, 51] ofers a promising direction. By interleaving language with operations such as cropping and zooming, these methods let a model inspect details and feed new visual information back into reasoning. Extending this paradigm to video raises a central question: how can a model integrate visual observations into its reasoning so that both the intermediate process and final answer are reliable, accurate, and verifiable?

Achieving grounded and active video reasoning is dificult for three reasons. First, suitable supervision is scarce and fragmented: video question-answering datasets typically omit visual evidence, grounding datasets provide only partial signals, and few resources jointly capture temporal clues, object localization, multi-turn tool interactions, and aligned reasoning traces. Second, spatio-temporal grounding must align with the chain of thought: plausible timestamps, boxes, and rationales may still be mutually inconsistent, while final-answer supervision or independently optimized grounding signals do not ensure that each reasoning claim is supported by the correct moment and object. Reliable optimization therefore requires structured, evidence-aware supervision linking intermediate reasoning to visual support. Third, a single observation is often insuficient: sparse sampling may miss brief actions or small objects, whereas dense processing is costly and introduces irrelevant context. Agentic frame selection and temporal cropping [15, 67, 76] ofer an adaptive alternative, but retrieved clips are often weakly tied to fine-grained object evidence, and task-level rewards provide limited credit for whether each call contributes useful new information. Efective tool use should therefore account for how successive observations update the accumulated evidence for reasoning.

![](images/25d2ed37eeaafa91ae57c772aa622bfd1f4ed84860b73298852d3215c79b39b3.jpg)  
Figure 1 From single-pass spatio-temporal grounded reasoning to agentic evidence seeking and grounded reasoning. Left: Open-o3-Video-v1 reasons over fixed sampled frames and grounds visible evidence, yielding verifiable spatio-temporal support on short videos. Center: Open-o3-Video-v2 performs adaptive temporal zoom-in through iterative crop\_video calls, following a seek–ground–reason process to find evidence beyond the initial frames for long video understanding. Right: Comparison of five models on V-STaR and six general video understanding and reasoning benchmarks.

Table 1 Capability comparison with closely related video reasoning methods. “Adaptive temp. res.” denotes adaptive temporal resampling, “Temp. ev.” temporal evidence explicitly cited in reasoning, “Obj. ev.” object-level evidence explicitly cited in reasoning, and “Native agent” multi-turn interaction with interleaved observations.
<table><tr><td>Method</td><td>Adaptive temp. res. Temp. ev. Obj. ev. Native agent</td><td></td><td></td><td></td></tr><tr><td>Open-o3-Video [38]</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>VideoZoomer [15]</td><td>√</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Video-o3 [67]</td><td>√</td><td>x</td><td>x</td><td>√</td></tr><tr><td>VideoTemp-o3 [36]</td><td>√</td><td>x</td><td>x</td><td>√</td></tr><tr><td>VideoSeeker [76]</td><td>√</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Open-o3-Video-v2</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

To address these challenges, we develop two generations of Open-o3-Video for reasoning with verifiable visual evidence. The original Open-o3-Video embeds timestamps and object boxes directly into the chain of thought, making fine-grained spatio-temporal reasoning traceable [38]. It introduces STGR-CoT-30K and STGR-RL-36K to jointly supervise temporal grounding, spatial grounding, general video QA, and spatiotemporal grounded reasoning. Each grounded trace links semantic reasoning claims to timestamped frames and localized objects, making intermediate visual evidence readable, parseable, and directly optimizable beyond final-answer correctness. Full construction details are provided in Appendix A.

Open-o3-Video addresses the coupled optimization of temporal and spatial grounding. Since spatial rewards are meaningful only after the relevant moment is identified, inaccurate early timestamps can suppress useful spatial feedback. Adaptive temporal proximity gradually tightens the temporal reward neighborhood during training, while temporal gating activates spatial rewards only for suficiently accurate timestamps. Together, they alleviate early reward sparsity while preventing boxes at irrelevant moments from receiving credit. Openo3-Video consequently improves mAM and mLGM over Qwen2.5-VL by 14.4% and 24.2% on V-STaR [12], with consistent gains on multiple general video and temporal-grounding benchmarks. Detailed ablations are reported in Appendix D. Nevertheless, Open-o3-Video remains restricted to its initially sampled frames and cannot recover decisive evidence omitted by that observation.

Going beyond this fixed visual context, we develop Open-o3-Video-v2 by formulating video reasoning as a unified agentic evidence-grounded process. As summarized in Table 1, recent agentic video methods allow models to retrieve targeted temporal observations. However, acquiring a relevant interval mainly determines where to observe. Reliable evidence-grounded reasoning further requires the model to determine which moments and objects within the acquired observation support a particular inference. Open-o3-Video-v2 integrates these decisions into the same reasoning trajectory. The model first identifies unresolved information and decides where additional observation is needed. It then grounds the acquired observation to query-relevant moments and objects and records this evidence in the reasoning history. The newly acquired evidence becomes an explicit premise for subsequent inference, allowing the model to revise its interpretation, continue seeking, or answer from the accumulated evidence. As illustrated in Fig. 1 (middle), this seek–ground–reason process turns tool-acquired observations into explicit evidence that directly participates in the evolving reasoning process. The central contribution of v2 is therefore not temporal cropping or grounding in isolation, but their integration into an agentic reasoning formulation in which evidence is actively acquired, grounded, and reused. Our main extensions to Open-o3-Video and contributions are threefold:

First, we establish an agentic evidence-grounded reasoning formulation that extends grounded reasoning from fixed observations to an iterative process of seeking, grounding, and integrating missing evidence. The model jointly determines where additional observation is required and which moments and objects in the acquired content provide support for subsequent reasoning. Second, we construct STGR-v2-CoT-135K, extending grounded supervision from single responses to multi-turn agentic trajectories. Its core 52K examples connect sparse observation, targeted video cropping, returned frames, object-level grounding, and final answers in coherent reasoning traces. To encourage evidence-driven rather than ritualized tool use, we remove questions consistently solved from sparse observations, verify that each retained clue supports the answer, annotate timestamped object boxes with SAM3 [6], and filter inconsistent calls, boxes, and visual references. Together with grounding, dense-captioning, general video QA, and auxiliary tool-use data, the resulting corpus contains 135K SFT samples. We further construct STGR-v2-RL-13K with informative rollout samples for subsequent policy optimization. Third, we propose Clue-Seek GRPO (CS-GRPO) to optimize evidence acquisition over multi-turn trajectories. CS-GRPO tracks how the accumulated temporal clue state changes across successive interactions, rewards final clue coverage and useful evidence gains, and regularizes excessive, repetitive, fragmented, or uninformative retrieval. Together with answer and grounded-thinking supervision, the final objective jointly optimizes evidence acquisition and the grounding of acquired observations.

Open-o3-Video-v2-4B achieves 40.4% mAM and 56.3% mLGM on V-STaR, improving the Qwen3-VL-4B baseline by 8.5% and 12.6%, respectively. It also improves over the baseline across general video understanding and temporal-grounding benchmarks, including gains of 10.7% on Video-Holmes [11] and 7.2% on LVBench [55]. The 8B model further reaches 56.5% mLGM on V-STaR, 63.0% accuracy on VideoMMMU, and 51.1% mIoU on Charades-TimeLens [73].

## 2 Related Work

Video Spatial and Temporal Grounding. Video grounding identifies when an event occurs and where its visual referents appear. For temporal grounding, TimeChat [46] associates frames with timestamps and uses a sliding video query transformer, while LITA [27] combines relative time tokens with SlowFast temporal modeling. Grounded-VideoLLM [50] preserves inter-frame dynamics through an additional temporal stream and discrete time tokens, and UniTime [35] generates interleaved timestamps with coarse-to-fine localization. Recent methods directly optimize boundary quality with verifiable rewards: Time-R1 and TVG-R1 [9, 56] reward temporal overlap, OMTG [59] supports multiple disjoint segments, and TimeLens [73] combines re-annotated data, textual time encoding, and thinking-free reinforcement learning. Spatial grounding focuses on object localization and correspondence across frames. VideoRefer [66] combines region-level instructions with a spatio-temporal object encoder, Sa2VA [64] connects an MLLM with SAM-2 [45] for temporally consistent masks, and SpaceR [41] constructs explicit maps of object layouts and relations. Joint spatio-temporal methods include STCAT [31], which associates object-tube predictions with a consistency-aware transformer; LRR [4], which organizes grounded reasoning into look, remember, and reason stages; LLaVA-ST [32], which aligns coordinate tokens while decoupling spatial and temporal compression; and STVG-o1 [22], which inserts bounding-box reasoning before localization. Our previous Open-o3-Video [38] places timestamps, object references, and boxes directly inside a reasoning trace, but produces them in one round and cannot acquire evidence omitted by the initial observation. Open-o3-Video-v2 extends this formulation into an iterative evidence-grounded process, where missing observations are actively acquired, grounded to supporting moments and objects, and incorporated into subsequent reasoning.

Video Reasoning. Video reasoning methods can be broadly divided into text-only reasoning and reasoning with videos [39]. Text-only approaches generate a rationale after the video has been encoded or converted to textual memory. They learn this behavior through chain-of-thought supervision [21, 74], reinforcement learning [14, 18, 43, 53, 57], or preference optimization [28, 34]. Video-R1 encourages temporally aware reasoning, VideoRFT rewards semantic agreement between reasoning and video features, and DeepVideo-R1 stabilizes optimization through advantage regression and dificulty-aware data. Other systems organize captions, memories, or retrieval indexes for multi-stage inference and verification [16, 17, 30, 47, 58, 62, 77]. Although efective, their decisive intermediate states remain textual and do not expose precise visual support.

Thinking-with-video methods instead go beyond a fixed visual context by interleaving inference with explicit visual acquisition [40]. Agent-based methods difer in the granularity of their actions. VideoChat-R1.5 [60] scales iterative perception; VITAL [71] and FrameMind [20] request task-relevant clips or frames; and Pixel Reasoner [52] adds frame selection and spatial zoom. For long-video search, VideoExplorer [65] and Conan [42] formulate subquestions or identify evidence frames, while LongVT [63] and VideoZoomer [15] use temporal cropping and denser resampling in a global-to-local loop. At the policy level, Video-o3 [67] learns multi-hop clue seeking with adaptive termination, VideoTemp-o3 [36] couples tool use with temporal localization and correction, and VideoSeeker [76] extends native tool invocation to instance-level spatio-temporal perception. Evidence-based approaches require the reasoning trace to identify supporting observations through timestamps, dense captions, boxes, or other visual references [37, 54, 70, 72]. Open-o3-Video [38] directly generates timestamped object boxes but remains restricted to the initial sample. Most agent-based methods can retrieve additional frames or clips, but rarely link the retrieved observations to spatio-temporal evidence supporting reasoning steps. Open-o3-Video-v2 addresses this gap with an agentic evidence-grounded formulation. Temporal calls recover observations missing from the initial context, while timestamped object boxes link reasoning to identifiable visual referents and make the acquired evidence human-verifiable.

## 3 Open-o3-Video

Reliable video reasoning requires the model to identify the visual evidence that supports its answer. Open-o3- Video addresses this problem by embedding explicit temporal and object-level evidence into the reasoning trace. Given a video question, the model predicts the answer while identifying when the relevant event occurs and where the supporting object appears. In this section, we briefly review the STGR data, grounded-thinking objective, and remaining limitations of Open-o3-Video that motivate Open-o3-Video-v2.

## 3.1 STGR Data Construction

Open-o3-Video constructs STGR-CoT-30K for supervised fine-tuning (SFT) and STGR-RL-36K for reinforcement learning (RL), as summarized in Fig. 2(a). Both corpora mix temporal grounding, spatial grounding, spatio-temporal reasoning, and general video QA. Each grounded trace links a reasoning claim to a timestamped frame, an object name, and a bounding box in a parseable format. We filter visually inconsistent annotations so that intermediate evidence remains directly evaluable. The complete data composition, annotation format, and quality-control pipeline are reported in Appendix A.

![](images/d2eed4d9d850275e4a03761c36281c3ac96abe61af9ae4b79e529e992e5ed089.jpg)  
Figure 2 Overview of the data construction pipelines and STGR-v2 data statistics. (a) Open-o3-Video constructs single-turn spatio-temporal grounded reasoning data through initial annotation, bounding-box filtering, and selfconsistency checking. (b) Open-o3-Video-v2 extends this pipeline to agentic multi-turn reasoning through strict data filtering, spatio-temporal evidence annotation, CoT (chain-of-thought) rewriting with tool observations, and quality checking. (c) STGR-v2-CoT-135K contains 135K SFT samples across seven task groups, including 59K spatio-temporal grounded reasoning samples, while STGR-v2-RL-13K contains 13K RL samples selected through strict filtering.

## 3.2 Training with Grounded Thinking Rewards

We train Open-o3-Video in two stages. SFT teaches the evidence-grounded output pattern through responses containing <think> and <answer> blocks. In the subsequent RL stage, each response is scored by answer, thinking, and format rewards, as illustrated in Fig. 3(a):

$$
R ( x , y ) = r _ { \mathrm { a c c } } ( x , y ) + r _ { \mathrm { t h k } } ( x , y ) + r _ { \mathrm { f m t } } ( x , y ) .\tag{1}
$$

Here, $r _ { \mathrm { a c c } }$ evaluates final-answer correctness and $r _ { \mathrm { f m t } }$ checks the required output format. The grounded-thinking reward evaluates evidence cited in <think> and is decomposed as

$$
r _ { \mathrm { t h k } } ( x , y ) = r _ { \mathrm { t } } ( x , y ) + r _ { \mathrm { s } } ( x , y ) .\tag{2}
$$

The temporal term $r _ { \mathrm { t } }$ evaluates whether cited timestamps agree with interval or point supervision. For point supervision, Adaptive Temporal Proximity anneals the temporal neighborhood from broad to strict, providing smoother feedback early in training while preserving precise localization later. The spatial term $r _ { \mathrm { s } }$ evaluates the IoU between a predicted object box and the annotation at its matched timestamp. Temporal Gating activates this spatial reward only when the predicted timestamp is suficiently close to the annotated time, preventing boxes at irrelevant moments from receiving credit. The complete reward definitions are provided in Appendix A.

## 3.3 Remaining Challenges in Open-o3-Video

Open-o3-Video still has two limitations in complex video scenarios. First, it reasons within a fixed visual context and cannot request denser observations when sparse sampling misses a critical clue. Second, its grounded reasoning supervision remains limited in scale and diversity. Recent agentic video methods overcome fixed observations through temporal zooming [15, 67, 76], but retrieving an interval mainly determines where to observe and does not by itself identify which moment and visual entity support a reasoning step. These limitations motivate Open-o3-Video-v2, which extends the fixed-context formulation into an active process of acquiring, grounding, and integrating missing visual evidence.

![](images/cd99607d240517f63223b7625fdbb4a40c335e26dc738ef9d248df662441ba6b.jpg)  
Figure 3 Overview of the reinforcement learning designs in Open-o3-Video and Open-o3-Video-v2. (a) Open-o3-Video produces a single-turn spatio-temporal grounded reasoning trace and optimizes it with format, answer-accuracy, and grounded thinking rewards; the thinking reward jointly evaluates temporal proximity and spatial overlap with the target evidence. (b) Open-o3-Video-v2 extends this formulation to multi-turn agentic reasoning with CS-GRPO. The model iteratively calls crop\_video to acquire denser observations, while CS-GRPO tracks how the accumulated evidence evolves across calls and assigns credit according to whether each interaction contributes useful new evidence.

## 4 Open-o3-Video-v2

Building on the Open-o3-Video formulation reviewed in Sec. 3, Open-o3-Video-v2 extends evidence-grounded reasoning from a fixed observation setting to an active process of acquiring, grounding, and integrating missing visual evidence. The model can identify unresolved information during reasoning, selectively inspect a relevant temporal interval, and convert the returned observation into explicit spatio-temporal evidence for subsequent inference. This section presents the new $\mathrm { v 2 }$ formulation in detail, including the multi-turn reasoning process, STGR-v2 data pipeline, and Clue-Seek GRPO (CS-GRPO) for optimizing evidence acquisition over interaction trajectories.

## 4.1 Task Formulation

We represent a video of duration T as

$$
V = \{ ( t _ { i } , I _ { i } ) \} _ { i = 1 } ^ { N } , \qquad 0 \leq t _ { 1 } < \cdot \cdot \cdot < t _ { N } \leq T ,\tag{3}
$$

where N is the number of frames available to the observation system and $I _ { i }$ is the frame at timestamp $t _ { i }$ (in seconds). Given a question q, the default sparse sampler $ { \boldsymbol { S } } _ { 0 }$ produces the initial observation $O _ { 0 } = S _ { 0 } ( V )$ yielding the model input $x = ( q , O _ { 0 } )$ . The policy generates an interleaved trajectory containing $K \geq 0$ tool calls,

$$
\tau = ( z _ { 1 } , o _ { 1 } , z _ { 2 } , o _ { 2 } , \ldots , z _ { K } , o _ { K } , z _ { K + 1 } ) .\tag{4}
$$

For $k \le K , z _ { k }$ contains a <think> block followed by a <tool\_call> block, and $O _ { k }$ is the returned observation;   
the terminal response $z _ { K + 1 }$ instead ends with <answer>a</answer>, where a is the final answer.

Tool call. At non-final turn $k ,$ the model requests a segment using

$$
\boldsymbol { u } _ { k } = ( s _ { k } , e _ { k } , d _ { k } ) ,\tag{5}
$$

where $0 \leq s _ { k } < e _ { k } \leq T$ bounds the interval and $d _ { k } \in$ {coarse, medium, fine} controls its temporal sampling density. Similar to prior agentic video reasoning methods [67], the request is serialized as a <tool\_call>

to crop\_video with start\_time, end\_time, and sampling\_strategy. The tool returns $o _ { k } \ = \mathcal C ( V , u _ { k } )$ comprising frames sampled from $[ s _ { k } , e _ { k } ]$ according to $d _ { k } .$ , and appends it to the context. Each response is therefore conditioned on the question, initial observation, and preceding interaction history:

$$
z _ { k } \sim \pi _ { \theta } ( \cdot \mid q , { \cal O } _ { 0 } , z _ { < k } , o _ { < k } ) , \qquad k = 1 , \ldots , K + 1 ,\tag{6}
$$

where $z _ { < k } = ( z _ { 1 } , \dots , z _ { k - 1 } ) { \mathrm { ~ a n d ~ } } o _ { < k } = ( o _ { 1 } , \dots , o _ { k - 1 } ) .$

Embedded evidence. The <think> blocks may also contain explicit spatio-temporal evidence claims, each parsed as

$$
\varepsilon _ { m } = ( \hat { t } _ { m } , c _ { m } , b _ { m } ) ,\tag{7}
$$

where $\hat { t } _ { m } \in [ 0 , T ]$ is the claimed timestamp in seconds, $c _ { m }$ is the object name, and $b _ { m } = [ x _ { 1 m } , y _ { 1 m } , x _ { 2 m } , y _ { 2 m } ]$ is its bounding box. The two coordinate pairs give the top-left and bottom-right corners, respectively, in the model’s 1000 × 1000 image coordinate system. We consistently order evidence as timestamp, object name, and bounding box, and collect all M valid claims in $\mathcal { E } ( \tau ) = \{ \varepsilon _ { m } \} _ { m = 1 } ^ { M }$ . The resulting evidence claims remain in the interaction history and can serve as explicit premises for subsequent reasoning. Their structured form also makes the intermediate evidence directly measurable during optimization.

## 4.2 STGR-v2: Data Optimization

From grounded responses to agentic trajectories. Open-o3-Video-v2 represents samples in a conversation format that supports both single-turn responses and multi-turn interactions. For tool-use trajectories, this format interleaves reasoning, tool calls, returned frames, embedded object-level evidence, and a final answer. Compared with v1’s single response, these trajectories jointly teach when to inspect a clip, which evidence within the returned observation supports the reasoning, and how that evidence should be integrated into later inference. We additionally retain single-turn grounding, temporal localization, and dense-caption supervision to preserve complementary video capabilities.

Agentic data construction. We derive 52K spatio-temporal grounded tool-use QA trajectories from Seeker-173K [67] through the four stages in Fig. 2(b).

(1) Strict filtering. Qwen3-VL [2] performs five rollouts from eight sparse frames; we remove questions answered correctly in every rollout because they do not require another observation. We also discard samples that remain unanswerable when the annotated clue clip is supplied, ensuring that retained clues support the answer.

(2) Evidence annotation. Qwen3.5 [44] extracts question-relevant objects and their temporal spans; SAM3 [6] then produces boxes on sampled clue frames. We remove boxes covering more than 80% of a frame and retain the two highest-confidence boxes per object and target segment.

(3) Agentic CoT rewriting. Gemini-2.5-Pro [13] rewrites the original rationale using the verified clue and boxes. The resulting trajectory contains sparse-observation reasoning, tool calls, returned frames, grounded reasoning in the v1 evidence format, and a final answer.

(4) Quality checking and filtering. We apply four checks to the constructed trajectories. First, each bounding box should accurately localize the target object without being excessively large or imprecise. Second, the cited timestamp, object, and bounding box should be correctly matched. For these two checks, we first filter boxes exceeding an area threshold and then crop the corresponding region from the cited frame for Qwen3.5 [44] to verify whether it correctly localizes the referenced object instance. Third, we apply rule-based checks to ensure that each tool call specifies a valid temporal interval within the video range. Finally, we use Qwen3.5 to verify whether each reasoning description is semantically supported by its cited visual evidence. Samples that fail these checks are removed unless the inconsistency can be resolved unambiguously.

We also manually audit 500 randomly sampled examples covering diferent sources and task types after stages (2) and (3). The audits examine box quality, annotation consistency, tool-call validity, and the reasoning logic. More than 90% of the audited examples satisfy the corresponding quality criteria at each stage, indicating that the annotation and trajectory-construction pipeline is generally reliable. Recurring errors identified during the audits are incorporated into the checking and filtering criteria in stage (4). Detailed error categories and inter-annotator agreement are reported in Appendix B.

SFT corpus. STGR-v2-CoT-135K contains 52K multi-turn and 7K single-turn spatio-temporal grounded QA samples, 7K auxiliary tool-use samples, 37K temporal- and 4K spatial-grounding samples, 16K general video QA samples, and 12K dense captions (Fig. 2(c)). Sources include Seeker-173K, STGR-CoT-30K, CG-Bench, OMTG, TimeLens, OmniVideo-100K, and CapRL-Video-178K [5, 7, 38, 59, 61, 67, 73]. Thus, 59K samples (44%) provide direct spatio-temporal grounded QA supervision, while the remaining tasks preserve localization and general video perception.

RL corpus. STGR-v2-RL-13K combines 40% v1 data, 50% agentic tool-use data, and 10% TimeLens temporal grounding [73]. From five SFT-policy responses sampled at temperature 1.0, we retain questions with both correct and incorrect rollouts, removing already-solved and currently intractable prompts to provide informative reward contrast.

## 4.3 Clue-Seek GRPO

Multi-turn observation is useful only when the model learns how each newly acquired observation changes what is already known. An efective policy must identify missing evidence, inspect a relevant interval, integrate the returned observation with the accumulated context, and stop once suficient support has been obtained. Standard GRPO with task-level supervision can optimize the final outcome, but provides limited information about which intermediate observation actually improves the evidence available for reasoning.

We introduce Clue-Seek GRPO (CS-GRPO) to optimize this sequential evidence-acquisition process. As illustrated in Fig. 3(b), CS-GRPO maintains an evolving evidence state over the interaction history and uses its transitions to evaluate successive tool interactions. This makes the utility of a call history-dependent: a retrieved interval is valuable when it contributes new answer-relevant evidence, but less useful when it repeatedly covers information that has already been observed. The resulting optimization encourages the policy to seek informative intervals, accumulate complementary evidence, and terminate when further observation provides little additional value.

For a complete trajectory $\tau ,$ let $\{ p _ { k } = [ s _ { k } , e _ { k } ] \} _ { k = 1 } ^ { K }$ be the intervals from its K valid calls. Let $\mathcal { G } = \{ g _ { j } \} _ { j = 1 } ^ { J }$ denote the set of J annotated answer-relevant clue intervals. Let $N _ { \mathrm { c a l l } }$ count all complete <tool\_call> blocks, including malformed calls excluded from the K valid intervals. The evidence state after valid call k is the multi-segment temporal IoU between all intervals retrieved so far and the annotated clues,

$$
\gamma _ { k } = \frac { \left| \left( \cup _ { l = 1 } ^ { k } p _ { l } \right) \cap \left( \cup _ { j = 1 } ^ { J } g _ { j } \right) \right| } { \left| \left( \cup _ { l = 1 } ^ { k } p _ { l } \right) \cup \left( \cup _ { j = 1 } ^ { J } g _ { j } \right) \right| } ,\tag{8}
$$

where | · | measures temporal duration. We set $\gamma _ { 0 } = 0$ , so a trajectory without a valid call has zero clue coverage. The scalar $\gamma _ { k }$ jointly measures clue recall and irrelevant temporal coverage accumulated up to step k.

Rather than scoring each call in isolation, we use the sequence $\{ \gamma _ { k } \} _ { k = 0 } ^ { K }$ to characterize how the evidence available to the model evolves throughout the trajectory. CS-GRPO rewards improvements in this state while regularizing acquisition patterns that add little useful evidence.

Two terms characterize final evidence quality and step-wise progress. The terminal clue reward

$$
r _ { \mathrm { c l u e } } ( \tau ) = \gamma _ { K }\tag{9}
$$

evaluates final clue coverage while accounting for irrelevant temporal content. Because it does not identify which call improved the state, we also define

$$
r _ { \mathrm { g a i n } } ( \tau ) = \sum _ { k = 2 } ^ { K } \mathbb { I } \left[ \gamma _ { k } - \gamma _ { k - 1 } > 0 . 1 \right] ,\tag{10}
$$

where $\mathbb { I } [ \cdot ]$ is the indicator function. This rewards a subsequent call when it adds suficient clue-aligned evidence, and the first call remains evaluated by $r _ { \mathrm { c l u e } }$

Evidence-state improvement alone does not discourage ineficient acquisition patterns, so we additionally regularize excessive, repeated, fragmented, and degrading calls. Because annotated clues can typically be found within three calls, the budget penalty

$$
r _ { \mathrm { b u d g e t } } ( \tau ) = \mathrm { m a x } ( 0 , N _ { \mathrm { c a l l } } - 3 )\tag{11}
$$

counts calls beyond this allowance.

The repetition penalty discourages nearly identical intervals:

$$
r _ { \mathrm { r e p e a t } } ( \tau ) = \sum _ { 1 \le i < j \le K } \mathbb { I } \left[ \mathrm { t I o U } ( p _ { i } , p _ { j } ) > 0 . 7 \right] ,\tag{12}
$$

where overlapping pairs with temporal IoU above 0.7 are counted.

To discourage sequential scanning through adjacent fragments rather than selecting promising clues, we introduce a fragmentation penalty. For consecutive intervals $p _ { i } = [ s _ { i } , e _ { i } ]$ and $p _ { i + 1 } = [ s _ { i + 1 } , e _ { i + 1 } ] ,$ , we treat them as adjacent when $- 1 0 ^ { - 6 } \le s _ { i + 1 } - e _ { i } \le 1 0 ^ { - 6 }$ . Each maximal sequence of at least three consecutive adjacent intervals contributes one count to $r _ { \mathrm { f r a g } } ( \tau )$

Finally, the drop penalty discourages a subsequent call from diluting already located evidence:

$$
r _ { \mathrm { d r o p } } ( \tau ) = \sum _ { k = 2 } ^ { K } \mathbb { I } \left[ \gamma _ { k - 1 } - \gamma _ { k } > 0 . 1 \right] ,\tag{13}
$$

which counts state decreases larger than 0.1.

We summarize these evidence-state transitions into a trajectory-level tool-use score:

$$
\begin{array} { r } { r _ { \mathrm { t o o l } } ( x , \tau ) = \mathrm { c l i p } _ { [ 0 , 1 ] } \big ( r _ { \mathrm { c l u e } } ( \tau ) + 0 . 1 r _ { \mathrm { g a i n } } ( \tau ) - 0 . 1 r _ { \mathrm { b u d g e t } } ( \tau ) } \\ { - 0 . 1 r _ { \mathrm { r e p e a t } } ( \tau ) - 0 . 1 r _ { \mathrm { f r a g } } ( \tau ) - 0 . 0 5 r _ { \mathrm { d r o p } } ( \tau ) \big ) . } \end{array}\tag{14}
$$

This score provides trajectory-level supervision for the evolution of acquired evidence while retaining the standard GRPO estimator. CS-GRPO thereby makes three coupled decisions directly optimizable: where to search, whether a new observation improves the accumulated evidence, and when further observation is no longer beneficial.

## 4.4 Reward Design

The final objective jointly evaluates the quality of the answer, grounded reasoning, trajectory structure, and evidence acquisition. For model input x and trajectory τ, we first clip each component to [0, 1] and compute

$$
R ( x , \tau ) = w _ { \mathrm { a c c } } r _ { \mathrm { a c c } } ( x , \tau ) + w _ { \mathrm { f m t } } r _ { \mathrm { f m t } } ( x , \tau ) + w _ { \mathrm { t h k } } r _ { \mathrm { t h k } } ( x , \tau ) + w _ { \mathrm { t o o l } } r _ { \mathrm { t o o l } } ( x , \tau ) ,\tag{15}
$$

where $w _ { \mathrm { a c c } } , w _ { \mathrm { f m t } } , w _ { \mathrm { t h k } }$ , and $w _ { \mathrm { t o o l } }$ are nonnegative coeficients that control the contribution of each component. The answer reward $r _ { \mathrm { a c c } } ( x , \tau )$ follows the task-specific metrics used in Open-o3-Video, including option matching, text similarity, spatial IoU, and temporal IoU.

The binary format reward $r _ { \mathrm { f m t } } ( x , \tau )$ equals one when the complete trajectory satisfies all output requirements and otherwise equals zero. Its major tags must follow

$$
\begin{array} { r l } & { ( \mathrm { < t h i n k > < / t h i n k > < t o o l \_ c a l l > < / t o o l \_ c a l l > } ) * } \\ & { \mathrm { < t h i n k > < / t h i n k > < a n s w e r > < / a n s w e r > } } \end{array}
$$

The major tags must also be balanced and non-crossing. The minor tags <obj>, <box>, and <t> must be balanced. Each tool call must be valid JSON, invoke crop\_video, use a legal sampling strategy, and satisfy $e _ { k } > s _ { k }$ . For temporal-answer tasks, the answer must follow From <t>start</t>s to <t>end</t>s.

The thinking reward $r _ { \mathrm { t h k } } ( x , \tau )$ extends the grounded thinking reward in Sec. 3.2 from a single response to all <think> blocks in the trajectory. It evaluates the timestamps, objects, and boxes explicitly cited in reasoning using the available spatio-temporal annotations, and therefore provides the primary supervision for object-level grounding correctness.

The tool reward $r _ { \mathrm { t o o l } } ( x , \tau )$ follows the temporal evidence-acquisition score defined in Sec. 4.3 for clue-supervised samples. For samples without clue intervals, it provides lightweight regularization against excessive, repeated, or fragmented calls. The two terms optimize complementary aspects of agentic evidence-grounded reasoning: $r _ { \mathrm { t o o l } }$ mainly supervises where to observe, while r<sub>thk</sub> supervises how the acquired observation is grounded to fine-grained moments and objects. They are jointly optimized in Eq. 15.

## 4.5 Training Recipes

The v2 model is trained with SFT followed by CS-GRPO. SFT represents each sample in a conversation format that supports both single-turn responses and multi-turn interactions, with user video context, assistant reasoning, optional tool calls and returned frames, grounded evidence, and a final answer. The standard next-token loss is applied only to assistant-generated tokens; user prompts, videos, and tool observations are masked as context. CS-GRPO then optimizes complete trajectories from the agent loop using the objective in Sec. 4.4, where evidence-state transitions provide additional supervision for successive observation decisions.

## 4.6 Training Infrastructure

The static, task-specific v1 trainers do not naturally support long trajectories that alternate between model actions and returned video observations. We therefore perform SFT with ms-swift<sup>1</sup>, whose multimodal conversation format and distributed backends support scalable training on the mixed 135K-sample corpus, and implement crop\_video with the verl<sup>2</sup> AgentLoop for RL. The loop executes tools between model turns, masks returned frames as context, retains model tokens as policy actions, and passes each complete trajectory to the objective in Sec. 4.4. We have open-sourced this training stack.

## 5 Experiments

Benchmarks and evaluation metrics. We evaluate three video understanding tasks, including spatio-temporal grounded question answering, general video understanding, and temporal grounding. We use a diferent set of benchmarks for each task. For spatio-temporal grounded question answering, V-STaR [12] measures three complementary capabilities: answering the question (What), localizing the supporting moment (When), and grounding the relevant object (Where). It evaluates two reasoning orders, what–when–where (Chain1) and what–where–when (Chain2), and reports answer accuracy, temporal IoU, visual IoU, and the composite mAM and mLGM scores. For general video understanding, we use VideoMME [19], VideoMME-v2 [49], WorldSense [25], VideoMMMU [26], Video-Holmes [11], and LVBench [55]. Together, these benchmarks cover broad video comprehension, real-world multimodal perception, knowledge-intensive reasoning, multi-hop clue inference, and extremely long videos. For temporal grounding, we evaluate on Charades-TimeLens [73] and TVGBench [8], both of which require a model to localize the temporal interval described by a language query; we report mean temporal IoU (mIoU).

Implementation details. Open-o3-Video is instantiated with Qwen2.5-VL-7B [3] and, for controlled comparisons with v2, we also train it from Qwen3-VL-4B and Qwen3-VL-8B. Open-o3-Video-v2 uses the same two Qwen3- VL backbones [2]. Both generations are trained on NVIDIA H100 GPUs. For Open-o3-Video, we uniformly sample 16 frames from each video, cap the resolution of each frame at $1 2 8 \times 2 8 \times 2 8$ , and insert annotated key frames when they are available. Each frame is prepended with its absolute timestamp to strengthen temporal perception. For Open-o3-Video-v2, SFT uniformly samples the full video at a target rate of 1 FPS with up to 96 frames and uses coarse-to-fine cropped observations for multi-turn samples. During CS-GRPO, each rollout starts from at most 32 uniformly distributed frames, and the three crop strategies progressively increase the number of frames returned from the requested interval. At inference time, the initial observation is sampled at 1 FPS with at most 96 frames, and each trajectory is limited to at most five assistant turns. We perform full-parameter SFT and CS-GRPO for one epoch each, using learning rates of $5 \times 1 0 ^ { - 6 }$ and $1 \times 1 0 ^ { - 6 }$ respectively. CS-GRPO is trained on 13K samples with eight rollouts per prompt. More detailed training and evaluation configurations are provided in Appendix C.

Table 2 Performance on V-STaR (%). Chain1 denotes what–when–where, whereas Chain2 denotes what–where–when. mAM and mLGM jointly summarize answer accuracy and spatio-temporal alignment. Bold and underlined numbers denote the best and second-best results, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">What Acc</td><td colspan="2">When (Temporal loU)</td><td colspan="2">Where (Visual loU)</td><td colspan="2">Overall</td></tr><tr><td>Chain1</td><td>Chain2</td><td>Chain1</td><td>Chain2</td><td>mAM</td><td>mLGM</td></tr><tr><td>GPT-4o [29]</td><td>60.8</td><td>16.7</td><td>12.8</td><td>6.5</td><td>3.0</td><td>26.8</td><td>38.2</td></tr><tr><td>Gemini-2-Flash [13]</td><td>53.0</td><td>24.5</td><td>23.8</td><td>4.6</td><td>2.2</td><td>26.9</td><td>35.6</td></tr><tr><td>Gemini-3-Pro [13]</td><td>59.1</td><td>24.2</td><td>23.8</td><td>7.2</td><td>4.8</td><td>29.9</td><td>41.2</td></tr><tr><td>Video-LLaMA3-7B [69]</td><td>41.9</td><td>23.0</td><td>23.1</td><td>0.9</td><td>0.2</td><td>21.7</td><td>27.0</td></tr><tr><td>LLaVA-Video-7B [75]</td><td>49.5</td><td>10.5</td><td>12.2</td><td>1.9</td><td>1.3</td><td>20.8</td><td>27.3</td></tr><tr><td>VideoChat2-7B [33]</td><td>36.2</td><td>13.7</td><td>12.5</td><td>2.5</td><td>1.0</td><td>17.0</td><td>20.3</td></tr><tr><td>InternVL-2.5-8B [10]</td><td>44.2</td><td>8.7</td><td>7.8</td><td>0.7</td><td>0.1</td><td>17.6</td><td>24.9</td></tr><tr><td>Qwen2.5-VL-7B [3]</td><td>33.5</td><td>15.4</td><td>13.8</td><td>17.0</td><td>2.5</td><td>19.3</td><td>22.4</td></tr><tr><td>VideoZoomer-7B [15]</td><td>56.5</td><td>20.4</td><td>20.8</td><td>2.8</td><td>0.7</td><td>26.2</td><td>36.0</td></tr><tr><td>VideoTemp-o3-7B [36]</td><td>37.0</td><td>16.8</td><td>24.1</td><td>1.3</td><td>0.4</td><td>19.4</td><td>23.4</td></tr><tr><td>Video-o3-7B [67]</td><td>31.7</td><td>26.8</td><td>25.2</td><td>0.0</td><td>0.0</td><td>19.3</td><td>22.8</td></tr><tr><td>Qwen3-VL-4B [2]</td><td>59.6</td><td>18.6</td><td>19.2</td><td>26.8</td><td>7.7</td><td>31.9</td><td>43.7</td></tr><tr><td>Qwen3-VL-8B [2]</td><td>42.9</td><td>23.5</td><td>24.4</td><td>27.3</td><td>7.1</td><td>28.0</td><td>34.4</td></tr><tr><td>TRACE [23]</td><td>17.6</td><td>19.1</td><td>17.1</td><td>0.0</td><td>0.0</td><td>12.0</td><td>13.3</td></tr><tr><td>Sa2VA-8B [64]</td><td>16.4</td><td>0.1</td><td>0.0</td><td>32.3</td><td>37.5</td><td>17.1</td><td>20.3</td></tr><tr><td>Open-o3-Video-7B [38]</td><td>61.0</td><td>24.5</td><td>24.0</td><td>25.4</td><td>6.0</td><td>33.7 (+14.4%)</td><td>46.6 (+24.2%)</td></tr><tr><td>Open-o3-Video-4B</td><td>59.7</td><td>25.1</td><td>23.8</td><td>31.2</td><td>8.1</td><td>34.6 (+2.7%)</td><td>47.3 (+3.6%)</td></tr><tr><td>Open-o3-Video-8B</td><td>61.1</td><td>24.9</td><td>23.5</td><td>33.2</td><td>8.8</td><td>35.5 (+7.5%)</td><td>49.0 (+14.6%)</td></tr><tr><td>Open-o3-Video-v2-4B</td><td>62.8</td><td>28.8</td><td>30.0</td><td>38.8</td><td>19.1</td><td>40.4 (+8.5%)</td><td>56.3 (+12.6%)</td></tr><tr><td>Open-o3-Video-v2-8B</td><td>62.5</td><td>27.9</td><td>28.0</td><td>39.5</td><td>23.9</td><td>40.7 (+12.7%)</td><td>56.5 (+22.1%)</td></tr></table>

## 5.1 Main Results

Spatio-temporal grounded QA on V-STaR. Table 2 compares three complementary baseline families: frontier proprietary models (GPT-4o [29] and Gemini [13]), general-purpose open-source video MLLMs [2, 3, 67, 69, 75], and task-specialized grounding methods (TRACE [23] and Sa2VA [64]). The composite mAM and mLGM metrics test whether a model can answer correctly while maintaining temporally and spatially consistent evidence.

Open-o3-Video. Starting from Qwen2.5-VL-7B, Open-o3-Video-7B improves answer accuracy by 27.5%, temporal IoU by 9.1%/10.2% on Chain1/Chain2, and visual IoU by 8.4%/3.5%. Together, these gains increase mAM and mLGM by 14.4% and 24.2%, respectively, exceeding Gemini-3-Pro by 3.8% and 5.4% and achieving the strongest composite grounding performance among the compared contemporaneous baselines. The improvement therefore covers not only answer accuracy but also alignment with explicit spatio-temporal evidence. Even with limited dedicated temporal-localization supervision, Open-o3-Video-7B surpasses TRACE by 5.4%/6.9% temporal IoU on Chain1/Chain2, slightly exceeds Gemini-3-Pro by 0.3%/0.2%, and remains competitive with the agentic Video-o3 model. For spatial grounding, Sa2VA retains the highest Chain2 visual IoU, but its weak answer and temporal performance limit its composite scores. Open-o3-Video instead substantially improves spatial IoU over Qwen2.5-VL and Video-o3 while preserving strong QA and temporal localization. The Qwen3-VL experiments further confirm transfer across model scales: Open-o3-Video-4B improves its backbone by 2.7% mAM and 3.6% mLGM, while Open-o3-Video-8B gains 7.5% and 14.6%, respectively. These consistently stronger composite scores verify the value of jointly optimizing answer correctness, temporal localization, and spatial grounding.

Open-o3-Video-v2. The v2 models combine explicit evidence grounding with agentic, multi-turn clue acquisition. Relative to Qwen3-VL-4B, Open-o3-Video-v2-4B improves answer accuracy by 3.2%, temporal IoU by 10.2%/10.8%, and visual IoU by 12.0%/11.4%, resulting in gains of 8.5% mAM and 12.6% mLGM. Open-o3- Video-v2-8B reaches the best overall results, 40.7% mAM and 56.5% mLGM, improving over Qwen3-VL-8B by

Table 3 Performance on general video understanding and temporal grounding benchmarks (%). For WorldSense, we report overall and recognition accuracy. The last two columns report mIoU. All other columns report accuracy. Bold and underlined numbers denote the best and second-best results, respectively.
<table><tr><td rowspan="3">Model</td><td colspan="6">General Video Understanding and Reasoning</td><td colspan="2">Temporal Grounding</td></tr><tr><td>Video- Holmes</td><td>LV Bench MMMU</td><td>Video-</td><td>WorldSense</td><td>Video MME</td><td>Video MME-v2</td><td>Charades- TimeLens</td><td>TVG Bench</td></tr><tr><td>Acc</td><td>Acc</td><td>Acc</td><td>Overall Recog.</td><td></td><td>Acc Acc</td><td>mloU</td><td>mloU</td></tr><tr><td>Qwen2.5-VL-7B [3]</td><td>34.7</td><td>45.3</td><td>51.2</td><td>36.1</td><td>33.7</td><td>62.4 25.3</td><td>39.3</td><td>16.3</td></tr><tr><td>Qwen3-VL-4B [2]</td><td>37.3</td><td>46.0</td><td>54.4</td><td>41.7</td><td>42.0 65.3</td><td>26.2</td><td>45.6</td><td>27.1</td></tr><tr><td>Qwen3-VL-8B [2]</td><td>44.2</td><td>47.3</td><td>57.3</td><td>45.7</td><td>44.0 67.4</td><td>27.9</td><td>48.3</td><td>29.4</td></tr><tr><td>VideoRFT-7B [53]</td><td>40.3</td><td>40.4</td><td>51.1</td><td>38.2</td><td>36.6 59.8</td><td>22.4</td><td>17.0</td><td>14.3</td></tr><tr><td>VideoR1-7B [18]</td><td>42.6</td><td>39.8</td><td>52.4</td><td>35.5</td><td>32.8 61.4</td><td>22.7</td><td>20.8</td><td>9.6</td></tr><tr><td>VideoZoomer-7B [15]</td><td>45.7</td><td>41.5</td><td>52.2</td><td>42.4</td><td>43.1 65.2</td><td>23.6</td><td>33.5</td><td>20.0</td></tr><tr><td>VideoTemp-o3-7B [36]</td><td>40.3</td><td>43.0</td><td>53.2</td><td>40.6</td><td>40.6 64.5</td><td>23.3</td><td>38.9</td><td>30.4</td></tr><tr><td>Video-o3-7B [67]</td><td>46.5</td><td>47.6</td><td>51.7</td><td>45.4</td><td>48.1 66.5</td><td>24.1</td><td>36.9</td><td>27.6</td></tr><tr><td>Open-o3-Video-7B [38]</td><td>40.6</td><td>41.2</td><td>52.3</td><td>37.5</td><td>36.8</td><td>63.6 22.4</td><td>30.9</td><td>20.8</td></tr><tr><td>Open-o3-Video-v2-4B</td><td>48.0</td><td>53.2</td><td>59.8</td><td>46.3</td><td>48.3 66.6</td><td>27.0</td><td>49.8</td><td>31.3</td></tr><tr><td>Open-o3-Video-v2-8B</td><td>46.1</td><td>53.9</td><td>63.0</td><td>45.5</td><td>47.2 67.9</td><td>29.2</td><td>51.1</td><td>30.9</td></tr></table>

12.7% and 22.1%, respectively. Compared with Open-o3-Video-4B, the v2 4B model improves answer accuracy by 3.1%, temporal IoU by 3.7%/6.2%, visual IoU by 7.6%/11.0%, and mAM/mLGM by 5.8%/9.0%. The 8B model similarly exceeds its v1 counterpart by 5.2% mAM and 7.5% mLGM. These consistent gains show that combining evidence-grounded reasoning with agentic clue seeking enables more accurate spatio-temporal localization and stronger overall performance.

General video understanding and temporal grounding. Table 3 evaluates generalization across a broader range of video understanding and grounding benchmarks. We compare our model with Qwen baselines [2, 3], textual reasoning methods (Video-R1 [18], VideoRFT [53]) and agentic clue-seeking methods (VideoZoomer [15], VideoTemp-o3 [36], and Video-o3 [68]). Together, these benchmarks assess general video QA, long-video understanding, knowledge-intensive reasoning, multi-hop clue integration, and temporal grounding.

Open-o3-Video. Relative to Qwen2.5-VL-7B, Open-o3-Video-7B improves VideoMME by 1.2%, WorldSense overall/recognition by 1.4%/3.1%, VideoMMMU by 1.1%, and TVGBench mIoU by 4.5%. It also outperforms VideoRFT and VideoR1 on VideoMME and exceeds them on TVGBench by 6.5% and 11.2%, respectively, showing that explicit temporal evidence ofers an advantage over text-only reasoning for localization. Overall, Open-o3-Video-7B remains comparable to other video understanding and reasoning models while additionally providing intuitive spatio-temporal evidence for its predictions. However, the fixed-context v1 model remains limited in harder temporal regimes. Its performance decreases on LVBench, whose videos average over one hour [55], and on Charades-TimeLens [73]. This limitation in extreme long-video understanding and fine-grained temporal localization motivates the adaptive clue-seeking mechanism in v2.

Open-o3-Video-v2. Open-o3-Video-v2-4B consistently improves over Qwen3-VL-4B across all reported benchmarks. Its gains of 1.3%, 0.8%, and 4.6% on VideoMME, VideoMME-v2, and WorldSense overall demonstrate stronger general video comprehension, while the 6.3% improvement in WorldSense recognition indicates enhanced fine-grained visual recognition. The 5.4% gain on VideoMMMU further reflects stronger knowledgeintensive video reasoning. On the more challenging Video-Holmes and LVBench, it improves multi-hop clue integration and long-video understanding by 10.7% and 7.2%, respectively. The benefits also extend to temporal grounding: Open-o3-Video-v2-4B improves Charades-TimeLens mIoU by 4.2% and reaches 31.3% mIoU on TVGBench, exceeding Open-o3-Video-7B by 10.5%. Open-o3-Video-v2-8B achieves the strongest results among our variants on VideoMME, VideoMME-v2, VideoMMMU, LVBench, and Charades-TimeLens. Across both model scales, the broad gains show that combining evidence-grounded reasoning with agentic clue seeking strengthens fine-grained spatio-temporal localization while improving, rather than sacrificing, general video understanding and reasoning.

![](images/180b826c2e212e016bd0f2e0b3e94611aca6aaa015060195cc3e3bd324e4f65b.jpg)  
Figure 4 Qualitative comparison with Video-o3. Left: Video-o3 searches an incorrect segment, while Open-o3-Video-v2 inspects the relevant 1170–1340 s interval and grounds the pigs at 1310 s, providing direct evidence for answer D. Right: although the question explicitly specifies 07:48–07:50, Video-o3 queries 478–480 s and incorrectly predicts bowing. Open-o3-Video-v2 inspects the correct 468–470 s interval, grounds the three girls at 469 s, and identifies their group handshake. In both cases, the acquired observation is converted into explicit spatio-temporal evidence and used directly in the subsequent reasoning process.

Table 4 Stage-wise ablation of Open-o3-Video-v2-4B (%). V-STaR is evaluated with mAM and mLGM, while VideoMME is evaluated with accuracy.
<table><tr><td>Model</td><td colspan="2">V-STaR</td><td>VideoMME</td></tr><tr><td></td><td>mAM</td><td>mLGM</td><td>Acc</td></tr><tr><td>Qwen3-VL-4B (baseline)</td><td>31.9</td><td>43.7</td><td>65.3</td></tr><tr><td>Open-o3-Video-v2-SFT-4B (SFT)</td><td>38.2</td><td>53.4</td><td>63.7</td></tr><tr><td>Open-o3-Video-v2-4B (SFT+RL)</td><td>40.4</td><td>56.3</td><td>66.6</td></tr></table>

## 5.2 Ablation Studies and Analysis

We report the complete Open-o3-Video data, reward, training-stage, evidence-faithfulness, and test-time scaling ablations in Appendix D. We focus the remaining analysis on our v2 components.

Contribution of the two training stages. We conduct a stage-wise comparison to examine how supervised agentic trajectories and subsequent RL contribute to Open-o3-Video-v2. As shown in Table 4, SFT improves the Qwen3-VL-4B base model by 6.3% mAM and 9.7% mLGM, showing that multi-turn grounded trajectories efectively establish agentic evidence-seeking and spatio-temporal grounding behaviors. Meanwhile, accuracy on VideoMME decreases from 65.3% to 63.7%, suggesting a specialization trade-of: imitation of densely grounded multi-turn trajectories strengthens structured reasoning behavior but may temporarily afect general video QA capability. CS-GRPO subsequently improves the SFT model by 2.2% mAM, 2.9% mLGM, and 2.9% VideoMME accuracy. Unlike SFT, which mainly imitates annotated trajectories, RL evaluates sampled trajectories with answer, grounded-thinking, and tool-use rewards, allowing the model to explore more efective reasoning strategies beyond the supervised patterns. This preserves the learned seeking and grounding behaviors while recovering general video reasoning capability. The final model therefore exceeds the base by 8.5% mAM, 12.6% mLGM, and 1.3% VideoMME accuracy.

Agentic spatio-temporal grounded reasoning. To isolate the contribution of the reasoning mechanism, we construct three controlled variants: single-turn reasoning without explicit evidence, multi-turn clue seeking without explicit evidence, and our full multi-turn reasoning with grounded spatio-temporal evidence. All variants use the same Qwen3-VL-4B backbone, the same source training data, and the same evaluation setting. As shown in Table 5, introducing multi-turn clue seeking improves Video-Holmes from 45.0% to 47.1%, LVBench from 48.8% to 51.9%, and V-STaR mLGM from 47.8% to 52.3%, demonstrating the benefit of actively acquiring observations beyond the initial visual context. Under the same multi-turn setting, adding explicit grounded evidence further raises the three results to 48.0%, 53.2%, and 56.3%, respectively. The 4.0% gain in

Table 5 Comparison of reasoning mechanisms. All variants use the same backbone, source training data, and evaluation setting. Only the reasoning mechanism is changed.
<table><tr><td>Reasoning mechanism</td><td></td><td></td><td>Video-Holmes LVBench V-STaR mLGM</td></tr><tr><td>Qwen3-VL-4B (base) [2]</td><td>37.3</td><td>46.0</td><td>43.7</td></tr><tr><td>Single-turn, no evidence</td><td>45.0</td><td>48.8</td><td>47.8</td></tr><tr><td>Multi-turn, no evidence</td><td>47.1</td><td>51.9</td><td>52.3</td></tr><tr><td>Multi-turn with grounded evidence (ours)</td><td>48.0</td><td>53.2</td><td>56.3</td></tr></table>

Table 6 Comparison of GRPO and CS-GRPO variants across three video understanding benchmarks (%).
<table><tr><td>Method</td><td>Video-Holmes</td><td>LVBench</td><td>VideoMME</td></tr><tr><td>GRPO</td><td>46.7</td><td>52.0</td><td>65.1</td></tr><tr><td>CS-GRPO w/o gain bonus</td><td>47.0</td><td>52.2</td><td>66.2</td></tr><tr><td>CS-GRPO w/o penalties</td><td>46.9</td><td>50.4</td><td>65.5</td></tr><tr><td>CS-GRPO (ours)</td><td>48.0</td><td>53.2</td><td>66.6</td></tr></table>

V-STaR mLGM is particularly notable, showing that explicit grounding provides complementary value beyond temporal clue seeking alone. Together, these comparisons isolate two complementary efects: multi-turn seeking expands the available observations, while explicit grounding associates the acquired observations with localized evidence used in subsequent reasoning.

CS-GRPO reward components. To evaluate the efectiveness of CS-GRPO, we compare GRPO and CS-GRPO variants on Video-Holmes, LVBench, and VideoMME. GRPO uses the same SFT initialization, RL data, multi-turn rollouts, and optimization setup as CS-GRPO, but does not explicitly model how successive tool interactions change the accumulated evidence state. The other variants remove either the gain bonus for newly acquired clues or the penalties for unhelpful retrieval. As shown in Table 6, full CS-GRPO achieves the best performance on all three benchmarks, outperforming GRPO by 1.3, 1.2, and 1.5 points on Video-Holmes, LVBench, and VideoMME, respectively. Removing either the gain bonus or the retrieval penalties leads to consistent performance degradation, showing that rewarding informative evidence gains and regularizing ineficient tool use are both beneficial. These results demonstrate the efectiveness of CS-GRPO for optimizing multi-turn evidence acquisition. The reward-weight sensitivity analysis in Appendix E further shows that the improvement does not rely on a narrowly tuned reward configuration.

Figure 4 provides qualitative examples of Open-o3-Video-v2 correctly locating relevant temporal evidence and grounding the supporting objects. More quantitative analyses are provided in Appendix E, including reward-weight sensitivity, the joint efect of initial visual context and interaction budget, matched visualbudget comparisons, the faithfulness of adaptive evidence seeking, long-horizon evaluation analysis, and answer–grounding consistency. Appendix F further presents qualitative comparisons and failure analysis.

## 6 Conclusion

We introduce Open-o3-Video-v2, which formulates video reasoning as an agentic, evidence-grounded process. The model actively identifies missing information, acquires targeted observations, and integrates the decisive moments and grounded objects into its subsequent inference. We further develop STGR-v2-CoT-135K, STGR-v2-RL-13K, and CS-GRPO to make this behavior efective and eficient. Experiments show consistent improvements in grounded reasoning, general video understanding, and temporal localization. These results demonstrate the efectiveness of jointly seeking, grounding, and reasoning with explicit visual evidence. We hope our findings can provide the community with insights for building stronger spatio-temporal reasoning MLLMs.

Contributors: Jiahao Meng\*, Zhongyuan Bao\*, Yue Tan, Haiquan Wen, Renrui Zhang, Anna Wang, Jason Li, Yanwei Fu, Yunhai Tong, and Ming-Hsuan Yang. (First two authors share equal contributions.)

Acknowledgement. This work is supported by the National Key Research and Development Program of China (No. 2023YFC3807600).

## References

[1] Xiang An, Yin Xie, Feilong Tang, Yunyao Yan, Huajie Tan, Didi Zhu, Changrui Chen, Xiuwei Zhao, Bin Qin, Kaicheng Yang, et al. Llava-onevision-2: Towards next-generation perceptual intelligence. arXiv preprint arXiv:2605.25979, 2026.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, et al. Qwen3-vl technical report, November 2025.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[4] Apratim Bhattacharyya, Sunny Panchal, Mingu Lee, Reza Pourreza, Pulkit Madan, and Roland Memisevic. Look, remember and reason: Grounded reasoning in videos with language models. In International Conference on Learning Representations, 2024.

[5] Xinyue Cai, Chaoyou Fu, Yi-Fan Zhang, Ran He, and Caifeng Shan. Omnivideo-100k: A dataset for audio-visual reasoning through structured scripts and evidence chains. arXiv preprint arXiv:2606.14702, 2026.

[6] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

[7] Guo Chen, Yicheng Liu, Yifei Huang, Yuping He, Baoqi Pei, et al. Cg-bench: Clue-grounded question answering benchmark for long video understanding. arXiv preprint arXiv:2412.12075, 2024.

[8] Ruizhe Chen, Tianze Luo, Zhiting Fan, Heqing Zou, Zhaopeng Feng, Guiyang Xie, Hansheng Zhang, Zhuochen Wang, Zuozhu Liu, and Zhang Huaijian. Datasets and recipes for video temporal grounding via reinforcement learning. arXiv preprint arXiv:2507.18100, 2025.

[9] Ruizhe Chen, Tianze Luo, Zhiting Fan, Heqing Zou, Zhaopeng Feng, Guiyang Xie, Hansheng Zhang, Zhuochen Wang, Zuozhu Liu, and Zhang Huaijian. Datasets and recipes for video temporal grounding via reinforcement learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 983–992, Stroudsburg, PA, USA, 2025. Association for Computational Linguistics.

[10] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

[11] Jiale Cheng, Yuying Ge, Teng Wang, Yixiao Ge, Jingjing Liao, and Ying Shan. Video-holmes: Can mllm think like holmes for complex video reasoning? arXiv preprint arXiv:2505.21374, 2025.

[12] Zixu Cheng, Jian Hu, Ziquan Liu, Chenyang Si, Wei Li, et al. V-star: Benchmarking video-llms on video spatio-temporal reasoning. arXiv preprint arXiv:2503.11495, 2025.

[13] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[14] Jisheng Dang, Jingze Wu, Teng Wang, Xuanhui Lin, Nannan Zhu, Hongbo Chen, Wei-Shi Zheng, Meng Wang, and Tat-Seng Chua. Reinforcing video reasoning with focused thinking. arXiv preprint arXiv:2505.24718, 2025.

[15] Yang Ding, Yizhen Zhang, Xin Lai, Ruihang Chu, and Yujiu Yang. Videozoomer: Reinforcement-learned temporal focusing for long video reasoning. arXiv preprint arXiv:2512.22315, 2025.

[16] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. In European Conference on Computer Vision, pages 75–92, 2024.

[17] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Meishan Zhang, Mong-Li Lee, and Wynne Hsu. Video-of-thought: Step-by-step video reasoning from perception to cognition. arXiv preprint arXiv:2501.03230, 2024.

[18] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.

[19] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.

[20] Haonan Ge, Yiwei Wang, Kai-Wei Chang, Hang Wu, and Yujun Cai. Framemind: Frame-interleaved video reasoning via reinforcement learning. arXiv preprint arXiv:2509.24008, 2025.

[21] Sara Ghazanfari, Francesco Croce, Nicolas Flammarion, Prashanth Krishnamurthy, Farshad Khorrami, and Siddharth Garg. Chain-of-frames: Advancing video understanding in multimodal llms via frame-aware reasoning. arXiv preprint arXiv:2506.00318, 2025.

[22] Xin Gu, Haoji Zhang, Qihang Fan, Jingxuan Niu, Zhipeng Zhang, Libo Zhang, Guang Chen, Fan Chen, Longyin Wen, and Sijie Zhu. Thinking with bounding boxes: Enhancing spatio-temporal video grounding via reinforcement fine-tuning. arXiv preprint arXiv:2511.21375, 2025.

[23] Yongxin Guo, Jingyu Liu, Mingda Li, Qingbin Liu, Xi Chen, and Xiaoying Tang. Trace: Temporal grounding video llm via causal event modeling. In International Conference on Learning Representations (ICLR), Online, 2025. OpenReview.net. Afiliations: School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen; Tencent PCG; Shenzhen Institute of Artificial Intelligence and Robotics for Society (AIRS); Guangdong Provincial Key Laboratory of Future Networks of Intelligence.

[24] Songhao Han, Wei Huang, Hairong Shi, Le Zhuo, Xiu Su, Shifeng Zhang, Xu Zhou, Xiaojuan Qi, Yue Liao, and Si Liu. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26181–26191, Los Alamitos, CA, USA, 2025. IEEE Computer Society.

[25] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. arXiv preprint arXiv:2502.04326, 2025.

[26] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, et al. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025.

[27] De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. Lita: Language instructed temporal-localization assistant. In European Conference on Computer Vision (ECCV), Cham, Switzerland, 2024. Springer.

[28] Haojian Huang, Haodong Chen, Shengqiong Wu, Meng Luo, Jinlan Fu, Xinya Du, Hanwang Zhang, and Hao Fei. Vistadpo: Video hierarchical spatial-temporal direct preference optimization for large video models. arXiv preprint arXiv:2504.13122, 2025.

[29] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[30] Hongbo Jin, Ruyang Liu, Wenhao Zhang, Guibo Luo, and Ge Li. Cot-vid: Dynamic chain-of-thought routing with self verification for training-free video reasoning. arXiv preprint arXiv:2505.11830, 2025.

[31] Yang Jin, Yongzhi Li, Zehuan Yuan, and Yadong Mu. Embracing consistency: A one-stage approach for spatiotemporal video grounding. In Advances in Neural Information Processing Systems, volume 35, pages 29192–29204, 2022.

[32] Hongyu Li, Jinyu Chen, Ziyu Wei, Shaofei Huang, Tianrui Hui, Jialin Gao, Xiaoming Wei, and Si Liu. LLaVA-ST: A multimodal large language model for fine-grained spatial-temporal understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8592–8603, 2025.

[33] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024.

[34] Yunxin Li, Xinyu Chen, Zitao Li, Zhenyu Liu, Longyue Wang, Wenhan Luo, Baotian Hu, and Min Zhang. Veripo: Cultivating long reasoning in video-llms via verifier-gudied iterative policy optimization. arXiv preprint arXiv:2505.19000, 2025.

[35] Zeqian Li, Shangzhe Di, Zhonghua Zhai, Weilin Huang, Yanfeng Wang, and Weidi Xie. Universal video temporal grounding with generative multi-modal large language models. In Advances in Neural Information Processing Systems (NeurIPS), Red Hook, NY, USA, 2025. Curran Associates, Inc. Afiliations: Shanghai Jiao Tong University; ByteDance Seed.

[36] Wenqi Liu, Yunxiao Wang, Shijie Ma, Meng Liu, Qile Su, Tianke Zhang, Haonan Fan, Changyi Liu, Kaiyu Jiang, Jiankang Chen, et al. Videotemp-o3: Harmonizing temporal grounding and video understanding in agentic thinking-with-videos. arXiv preprint arXiv:2602.07801, 2026.

[37] Muhammad Maaz, Hanoona Rasheed, Fahad Shahbaz Khan, and Salman Khan. Video-r2: Reinforcing consistent and grounded reasoning in multimodal language models. arXiv preprint arXiv:2511.23478, 2025.

[38] Jiahao Meng, Xiangtai Li, Haochen Wang, Yue Tan, Tao Zhang, Lingdong Kong, Yunhai Tong, Anran Wang, Zhiyang Teng, Yujing Wang, et al. Open-o3 video: Grounded video reasoning with explicit spatio-temporal evidence. arXiv preprint arXiv:2510.20579, 2025.

[39] Jiahao Meng, Yue Tan, Qi Xu, Kuan Gao, Weisong Liu, Yanwei Li, Jason Li, Lingdong Kong, Haochen Wang, Qianyu Zhou, et al. Watch, remember, reason: Human-view video understanding with mllms. arXiv preprint arXiv:2606.07433, 2026.

[40] OpenAI. OpenAI-o3. https://openai.com/index/introducing-o3-and-o4-mini/, 2025.

[41] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.

[42] Kun Ouyang, Yuanxin Liu, Linli Yao, Yishuo Cai, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Conan: Progressive learning to reason like a detective over multi-scale visual evidence. arXiv preprint arXiv:2510.20470, 2025.

[43] Jinyoung Park, Jeehye Na, Jinyoung Kim, and Hyunwoo J Kim. Deepvideo-r1: Video reinforcement fine-tuning via dificulty-aware regressive grpo. arXiv preprint arXiv:2506.07464, 2025.

[44] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5.

[45] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In International Conference on Learning Representations, volume 2025, pages 28085–28128, 2025.

[46] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. arXiv preprint arXiv:2312.02051, 2023.

[47] Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Videorag: Retrieval-augmented generation with extreme long-context videos. arXiv preprint arXiv:2502.01549, 2025.

[48] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. In Advances in Neural Information Processing Systems, volume 37, pages 8612–8642, 2024.

[49] Video-MME Team. Video-mme v2: Evaluating multimodal llms with cohesive question groups on fresh videos. Preprint, 2025.

[50] Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan Wang, Weifeng Ge, and Lifu Huang. Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290, 2024.

[51] Haochen Wang, Xiangtai Li, Zilong Huang, Anran Wang, Jiacong Wang, Tao Zhang, Jiani Zheng, Sule Bai, Zijian Kang, Jiashi Feng, et al. Traceable evidence enhanced visual grounded reasoning: Evaluation and methodology. In International Conference on Learning Representations, 2026.

[52] Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025.

[53] Qi Wang, Yanrui Yu, Ye Yuan, Rui Mao, and Tianfei Zhou. Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning. arXiv preprint arXiv:2505.12434, 2025.

[54] Shijian Wang, Jiarui Jin, Xingjian Wang, Linxin Song, Runhao Fu, Hecheng Wang, Zongyuan Ge, Yuan Lu, and Xuelian Cheng. Video-thinker: Sparking" thinking with videos" via reinforcement learning. arXiv preprint arXiv:2510.23473, 2025.

[55] Weihan Wang, Zehai Yang, Bin Gu, Hongcheng Hu, Ke Xu, Jialv Zhang, Jinghui Xu, Jieyi Ma, Shiyu Zhang, Jinbao Xu, et al. Lvbench: An extreme long video understanding benchmark. arXiv preprint arXiv:2406.08035, 2024.

[56] Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025.

[57] Ziyang Wang, Jaehong Yoon, Shoubin Yu, Md Mohaiminul Islam, Gedas Bertasius, and Mohit Bansal. Video-rts: Rethinking reinforcement learning and test-time scaling for eficient and enhanced video reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 28114–28128, Stroudsburg, PA, USA, 2025. Association for Computational Linguistics.

[58] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3272–3283, Los Alamitos, CA, USA, 2025. IEEE Computer Society.

[59] Qi Xu, Tan Yue, Shihao Chen, Jiahao Meng, Anran Wang, Shunping Ji, Hao Fei, and Xiangtai Li. Towards one-to-many temporal grounding. In Proceedings of the 43rd International Conference on Machine Learning (ICML), Brookline, MA, USA, 2026. PMLR.

[60] Ziang Yan, Xinhao Li, Yinan He, Zhengrong Yue, Xiangyu Zeng, Yali Wang, Yu Qiao, Limin Wang, and Yi Wang. Videochat-r1. 5: Visual test-time scaling to reinforce multimodal reasoning by iterative perception. arXiv preprint arXiv:2509.21100, 2025

[61] Penghui Yang, Long Xing, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Yibin Wang, Yujie Zhou, Jiazi Bu, Jianze Liang, Qidong Huang, et al. Caprl++: Unified reinforcement learning with verifiable rewards for dense image and video captioning. arXiv preprint arXiv:2606.09393, 2026.

[62] Zongxin Yang, Guikun Chen, Xiaodi Li, Wenguan Wang, and Yi Yang. Doraemongpt: Toward understanding dynamic scenes with large language models (exemplified as a video agent). arXiv preprint arXiv:2401.08392, 2024.

[63] Zuhao Yang, Sudong Wang, Kaichen Zhang, Keming Wu, Sicong Leng, Yifan Zhang, Chengwei Qin, Shijian Lu, Xingxuan Li, and Lidong Bing. Longvt: Incentivizing" thinking with long videos" via native tool calling. arXiv preprint arXiv:2511.20785, 2025.

[64] Haobo Yuan, Xiangtai Li, Tao Zhang, Yueyi Sun, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, et al. Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv preprint arXiv:2501.04001, 2025.

[65] Huaying Yuan, Zheng Liu, Junjie Zhou, Hongjin Qian, Yan Shu, Nicu Sebe, Ji-Rong Wen, and Zhicheng Dou. Videoexplorer: Think with videos for agentic long-video understanding, 2025.

[66] Yuqian Yuan, Hang Zhang, Wentong Li, Zesen Cheng, Boqiang Zhang, Long Li, Xin Li, Deli Zhao, Wenqiao Zhang, Yueting Zhuang, Jianke Zhu, and Lidong Bing. VideoRefer suite: Advancing spatial-temporal object understanding with video LLM. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18970–18980, Los Alamitos, CA, USA, 2025. IEEE Computer Society.

[67] Xiangyu Zeng, Zhiqiu Zhang, Yuhan Zhu, Xinhao Li, Zikang Wang, Changlian Ma, Qingyu Zhang, Zizheng Huang, Kun Ouyang, Tianxiang Jiang, et al. Video-o3: Native interleaved clue seeking for long video multi-hop reasoning. arXiv preprint arXiv:2601.23224, 2026.

[68] Xiangyu Zeng, Zhiqiu Zhang, Yuhan Zhu, Xinhao Li, Zikang Wang, Changlian Ma, Qingyu Zhang, Zizheng Huang, Kun Ouyang, Tianxiang Jiang, et al. Video-o3: Native interleaved clue seeking for long video multi-hop reasoning. arXiv preprint arXiv:2601.23224, 2026.

[69] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.

[70] Congzhi Zhang, Zhibin Wang, Yinchao Ma, Jiawei Peng, Yihan Wang, Qiang Zhou, Jun Song, and Bo Zheng. Rewatch-r1: Boosting complex video reasoning in large vision-language models through agentic data synthesis. arXiv preprint arXiv:2509.23652, 2025.

[71] Haoji Zhang, Xin Gu, Jiawen Li, Chixiang Ma, Sule Bai, Chubin Zhang, Bowen Zhang, Zhichao Zhou, Dongliang He, and Yansong Tang. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. arXiv preprint arXiv:2508.04416, 2025.

[72] Jinglei Zhang, Yuanfan Guo, Rolandos Alexandros Potamias, Jiankang Deng, Hang Xu, and Chao Ma. Vtimecot: Thinking by drawing for video temporal grounding and reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24203–24213, Los Alamitos, CA, USA, 2025. IEEE Computer Society.

[73] Jun Zhang, Teng Wang, Yuying Ge, Yixiao Ge, Xinhao Li, and Limin Wang. Timelens: Rethinking video temporal grounding with multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10419–10429, 2026.

[74] Shuyi Zhang, Xiaoshuai Hao, Yingbo Tang, Lingfeng Zhang, Pengwei Wang, Zhongyuan Wang, Hongxuan Ma, and Shanghang Zhang. Video-cot: A comprehensive dataset for spatiotemporal understanding of videos based on chain-of-thought. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 12745–12752, New York, NY, USA, 2025. Association for Computing Machinery.

[75] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.

[76] Yiming Zhao, Yu Zeng, Wenxuan Huang, Zhen Fang, Qing Miao, Qisheng Su, Jiawei Zhao, Jiayin Cai, Lin Chen, Zehui Chen, et al. Videoseeker: Incentivizing instance-level video understanding via native agentic tool invocation. arXiv preprint arXiv:2605.16079, 2026.

[77] Zhuo Zhi, Qiangqiang Wu, Wenbo Li, Yinchuan Li, Kun Shao, Kaiwen Zhou, et al. Videoagent2: Enhancing the llmbased agent system for long-form video understanding by uncertainty-aware cot. arXiv preprint arXiv:2504.04471, 2025.

## Appendix

This appendix provides more supporting details and analyses. Appendix A describes the data sources, spatio-temporal annotation format, quality-control pipeline and details of the grounded thinking reward used to train Open-o3-Video. Appendix B provides additional details on STGR-v2 annotation quality and human verification. Appendix C summarizes the training hyperparameters, video observation settings, and inference configuration of Open-o3-Video-v2. Appendix D reports Open-o3-Video stage-wise, reward, data, evidence-faithfulness, and test-time-scaling studies. Appendix E further analyzes CS-GRPO reward sensitivity, frame and interaction budgets, matched visual-budget performance, adaptive evidence-seeking faithfulness, long-horizon evaluation analysis, and answer–grounding consistency. Finally, Appendix F presents analyses of successful and failed cases of multi-turn evidence acquisition and grounding.

## A Open-o3-Video Data Construction and Reward Details

## A.1 Data Sources and Composition

Open-o3-Video requires supervision that connects an answer to what visual evidence is used, when it appears, and where the supporting object is located. We therefore construct two corpora: STGR-CoT-30K for supervised fine-tuning (SFT) and STGR-RL-36K for reinforcement learning (RL). The SFT corpus teaches the model to interleave natural-language reasoning with explicit timestamps and object boxes, whereas the RL corpus exposes it to task-specific rewards for answer correctness and grounded thinking. Both corpora combine temporal grounding, spatial grounding, spatio-temporal grounding, and general video question answering. This mixture preserves broad video understanding while introducing explicit evidence supervision, rather than training a specialist that can localize evidence but cannot answer general video questions.

STGR-CoT-30K contains 4.1K TVG-Coldstart temporal-grounding samples [8], 5K TreeVGR-SFT spatialgrounding samples [51], 5.9K newly annotated spatio-temporal samples, and 15K Video-R1-CoT general video QA samples [18]. The first three components establish temporal, spatial, and joint evidence-grounding behaviors, while the general video QA component maintains diverse reasoning capability. STGR-RL-36K contains 5.2K temporal-grounding samples from Time-R1 [56] and TVG-RL [8], 5K spatial-grounding samples from VisCoT [48], 10.9K spatio-temporal samples, and 15K Video-R1 samples [18]. The spatio-temporal RL subset combines our 5.9K annotations with 5K filtered VideoEspresso samples [24], providing complementary supervision from newly annotated and existing grounded-video data.

## A.2 Spatio-Temporal Annotation

We annotate the new 5.9K samples with Gemini-2.5-Pro [13]. Each sample contains a question–answer pair, one to five key frames, one to three object boxes per key frame, and a reasoning trace that connects the evidence to the answer. The key frames identify when decisive evidence appears, and the boxes identify the query-relevant objects within those frames. The reasoning trace is required to cite these annotations when explaining the answer, thereby aligning the textual rationale with machine-verifiable temporal and spatia evidence. Evidence is serialized as

$$
\begin{array} { r l } & { \mathrm { < o b j > o b j e c t < / o b j > < b o x > [ x 1 , y 1 , x 2 , y 2 ] < / b o x > } } \\ & { \mathrm { a t < t > t i m e < / t > s , } } \end{array}
$$

where the object name identifies the referent, the box gives its spatial extent, and the timestamp gives its temporal position. This representation is readable as part of the reasoning trace and can also be parsed automatically. It therefore supports direct temporal- and spatial-overlap rewards without requiring a separate latent grounding head.

## A.3 Filtering and Consistency Checking

We apply three levels of filtering to reduce noisy or unverifiable supervision. First, we remove boxes covering more than 80% of the frame because they provide little localization information. For every remaining box, we crop the region and ask Qwen2.5-VL [3] whether it depicts the annotated object; only afirmative matches are retained. Second, we verify structural consistency: every temporal and spatial reference in the reasoning trace must be covered by the corresponding key\_frames and key\_items. Third, we crop each referenced region and use Qwen2.5-VL to check whether the associated reasoning sentence is semantically supported by its visual evidence. Samples with missing references, invalid or oversized boxes, or inconsistent visual–textual evidence are discarded. These checks ensure that the final traces are not merely well formatted, but also grounded in the annotated video content.

## A.4 Grounded Thinking Reward Details

For completeness, we provide the detailed grounded-thinking reward used in Open-o3-Video. Let $\{ \hat { t } _ { m } \} _ { m = 1 } ^ { M }$ denote the parsed evidence timestamps. For temporal supervision type $\tau _ { t } ~ \in$ {Interval, Point, ∅}, interval supervision gives $S ^ { \mathrm { g t } } = [ s ^ { \mathrm { g t } } , e ^ { \mathrm { g t } } ]$ , point supervision gives $\{ t _ { j } ^ { \mathrm { { \bar { g } t } } } \} _ { j = 1 } ^ { J }$ , and ∅ denotes samples without temporal supervision. With $\Delta t _ { m } = \operatorname* { m i n } _ { j } | \hat { t } _ { m } - t _ { j } ^ { \mathrm { g t } } |$ , the temporal reward is

$$
r _ { \mathrm { t } } ( x , y ) = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathbb { I } \left[ s ^ { \mathrm { g t } } \leq \hat { t } _ { m } \leq e ^ { \mathrm { g t } } \right] , } & { \tau _ { t } = \mathrm { I n t e r v a l } , } \\ { \displaystyle \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \exp \left( - \frac { \Delta t _ { m } ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , } & { \tau _ { t } = \mathrm { P o i n t } , } \\ { \displaystyle 0 , } & { \tau _ { t } = \emptyset . } \end{array} \right.\tag{16}
$$

Adaptive Temporal Proximity anneals σ from large to small to provide smooth early feedback and stricter late localization.

For a predicted time–object–box claim $( \hat { t } _ { m } , c _ { m } , b _ { m } )$ , let $j ^ { \star } ( m )$ = arg min<sub>j</sub> $| \hat { t } _ { m } - t _ { j } ^ { \mathrm { g t } } |$ be the nearest annotated time and $v _ { m } = \mathrm { I o U } ( b _ { m } , b _ { j ^ { \star } ( m ) } ^ { \mathrm { g t } } )$ the matched box IoU. Temporal Gating computes the spatial reward only when the predicted timestamp is suficiently close to the annotated time:

$$
r _ { \mathrm { s } } ( x , y ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathbb { I } \left[ | \hat { t } _ { m } - t _ { j ^ { \star } ( m ) } ^ { \mathrm { g t } } | \leq \delta \right] \cdot v _ { m } ,\tag{17}
$$

where δ is the temporal gating threshold.

## B Additional Details on STGR-v2

Human audit protocol. We randomly sample 500 examples covering diferent sources and task types, and two annotators independently audit all examples after stages (2) and (3). After stage (2), they mainly inspect the correctness of object references and bounding-box annotations. After stage (3), they further inspect tool-call validity, reasoning quality, and the consistency between reasoning descriptions and the associated spatio-temporal evidence. After resolving disagreements, 94.0% and 91.6% of the audited examples satisfy the corresponding quality criteria after stages (2) and (3), respectively, indicating that the annotation and trajectory-construction pipeline is generally reliable.

We categorize the observed errors into five groups: (1) object referent mismatches, where the localized instance does not correspond to the instance referred to by the question; (2) inaccurate object boxes, where the bounding box is excessively large or does not precisely localize the target object; (3) invalid tool-call intervals, where the requested temporal interval is invalid or inconsistent with the valid video range; (4) hallucinated reasoning descriptions, where the reasoning introduces events, actions, or visual details that are not supported by the video clip; and (5) reasoning-evidence mismatches, where the reasoning description is not aligned with the cited spatio-temporal evidence. Representative error patterns identified during the audits are incorporated into the consistency checking and filtering criteria in stage (4). A final manual review confirms that the resulting data quality meets our expectations.

Inter-annotator agreement. Inter-annotator agreement is calculated over all 500 audited examples before disagreement resolution. After stage (2), the two annotators achieve an agreement of 94.2% with a Cohen’s κ of 0.88. After stage (3), they achieve an agreement of 92.6% with a Cohen’s κ of 0.84. These results indicate strong inter-annotator consistency across both evidence annotation and trajectory-quality assessment. Disagreements are resolved through joint re-inspection of the source video, intermediate annotations, and rewritten reasoning trajectory.

Table 7 Training and optimization configurations of Open-o3-Video-v2-4B.
<table><tr><td>Setting</td><td>SFT</td><td>CS-GRPO</td></tr><tr><td>Training framework</td><td>ms-swift</td><td>verl AgentLoop</td></tr><tr><td>Model initialization</td><td>Qwen3-VL-4B-Instruct</td><td>SFT checkpoint</td></tr><tr><td>Training corpus</td><td>STGR-v2-CoT-135K</td><td>STGR-v2-RL-13K</td></tr><tr><td>Training epochs</td><td>1</td><td>1</td></tr><tr><td>Global batch size</td><td>16</td><td>32</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Minimum learning rate</td><td> $1 \times { 1 0 } ^ { - 6 }$ </td><td></td></tr><tr><td>Warmup ratio</td><td>0.05</td><td>0.03</td></tr><tr><td>Rollouts per prompt</td><td></td><td>8</td></tr><tr><td>Rollout temperature</td><td></td><td>1.0</td></tr><tr><td>Top-p / Top-k</td><td></td><td>1.0 / -1</td></tr><tr><td>KL regularization</td><td></td><td>0.001, actor loss</td></tr><tr><td>Reward weights  $( w _ { \mathrm { a c c } } , w _ { \mathrm { f m t } } , w _ { \mathrm { t h k } } , w _ { \mathrm { t o o l } } )$ </td><td></td><td>(1.0, 0.1, 0.1, 0.8)</td></tr><tr><td>Tool reward coefficients  $( \lambda _ { \mathrm { c l u e } } , \lambda _ { \mathrm { g a i n } } , \lambda _ { \mathrm { b u d g e t } } , \lambda _ { \mathrm { r e p e a t } } , \lambda _ { \mathrm { f r a g } } , \lambda _ { \mathrm { d r o p } } )$ </td><td></td><td>(1.0, 0.1, 0.1, 0.1, 0.1, 0.05)</td></tr></table>

Table 8 Video observation and interaction settings of Open-o3-Video-v2. SFT frame caps are applied to each observation. For CS-GRPO, crop strategies control the maximum number of uniformly sampled frames within the requested interval rather than fixed sampling rates.
<table><tr><td>Stage</td><td>Initial Observation</td><td>Coarse Crop</td><td>Medium Crop</td><td>Fine Crop</td><td>Interaction Limit</td></tr><tr><td>SFT</td><td>1 FPS, ≤ 96 frames</td><td>1 FPS, ≤ 96 frames 2 FPS, ≤ 96 frames 4 FPS, ≤ 96 frames</td><td></td><td></td><td>Given by annotated trajectory</td></tr><tr><td>Inference</td><td>CS-GRPO 1 FPS, ≤ 32 uniform frames 1 FPS, ≤ 96 frames</td><td>≤ 8 frames</td><td>≤ 16 frames Model-selected crop_video observations</td><td>≤ 32 frames</td><td>≤ 5 assistant turns ≤ 5 assistant turns</td></tr></table>

## C Additional Implementation Details

Training configuration. Open-o3-Video-v2 follows the two-stage training recipe in Sec. 4.5. For the 4B model, SFT starts from Qwen3-VL-4B-Instruct and performs full-parameter optimization on STGR-v2-CoT-135K for one epoch. We use a global batch size of 16, a maximum sequence length of 64,000, and a learning rate of $5 \times 1 0 ^ { - 6 }$ with a warmup ratio of 0.05. CS-GRPO is initialized from the SFT checkpoint. Each training batch contains 32 prompts, and eight trajectories are sampled for each prompt with temperature 1.0. We use a learning rate of $1 \times 1 0 ^ { - 6 }$ and a warmup ratio of 0.03. KL divergence is not included in the reward. The actor objective instead uses a low-variance KL loss with coeficient 0.001. The main optimization settings are summarized in Table 7.

Video observation and interaction. The SFT initial observation uniformly covers the complete video at a target rate of 1 FPS with at most 96 sampled frames. For multi-turn samples, coarse, medium, and fine crops target 1, 2, and 4 FPS, respectively, with at most 96 frames for each observation. For samples with annotated key frames, these frames can be additionally inserted into the initial observation. CS-GRPO uses a more compact initial observation with a target rate of 1 FPS and at most 32 uniformly distributed frames. Its crop\_video implementation does not assign fixed FPS values to coarse, medium, and fine retrieval. Instead, it uniformly samples at most 8, 16, and 32 frames from the requested interval, respectively. Each trajectory allows at most five assistant turns.

Inference configuration. At inference time, the model starts from a full-video observation sampled at a target rate of 1 FPS with at most 96 frames. The model can invoke crop\_video to acquire additional observations and is allowed at most five assistant turns. This interaction budget is an upper bound rather than a fixed number of calls, allowing the model to terminate once suficient evidence has been accumulated.

Table 9 Stage-wise ablation of Open-o3-Video-7B (%). V-STaR uses mAM and mLGM; VideoMME uses accuracy.
<table><tr><td>Model</td><td>mAM</td><td>mLGM</td><td>VideoMME</td></tr><tr><td>Qwen2.5-VL-7B</td><td>19.3</td><td>22.4</td><td>62.4</td></tr><tr><td>Open-o3-Video-SFT-7B</td><td>28.5</td><td>37.1</td><td>60.7</td></tr><tr><td>Open-o3-Video-7B (SFT+RL)</td><td>33.7</td><td>46.6</td><td>63.6</td></tr></table>

Table 10 Ablation of adaptive temporal proximity (Ada.) and temporal gating (Gat.) in the thinking reward.
<table><tr><td>Ada.</td><td>Gat.</td><td>mAM</td><td>mLGM</td></tr><tr><td>×</td><td>√</td><td>33.0</td><td>45.2</td></tr><tr><td>√</td><td>×</td><td>32.3</td><td>44.9</td></tr><tr><td>√</td><td>√</td><td>33.7</td><td>46.6</td></tr></table>

Table 12 Faithfulness test on a 150-question VideoMME subset. Both interventions remove the same number of frames.

Table 11 Efect of spatio-temporal training data on Openo3-Video, evaluated on V-STaR (%).
<table><tr><td>Training data</td><td>mAM</td><td>mLGM</td></tr><tr><td>w/o spatio-temporal data</td><td>28.3</td><td>36.2</td></tr><tr><td>+ VideoEspresso</td><td>31.1</td><td>43.6</td></tr><tr><td>+ Our annotations</td><td>33.7</td><td>46.6</td></tr></table>

<table><tr><td>Setting</td><td>Accuracy (%)</td></tr><tr><td>Uniform sampling (64 frames)</td><td>68.7</td></tr><tr><td>Remove predicted-evidence frames</td><td>66.0 (−2.7)</td></tr><tr><td>Remove random frames</td><td>68.0 (−0.7)</td></tr></table>

## D Open-o3-Video Ablation Studies

Contribution of the two training stages. We separate supervised fine-tuning from reward-based optimization to determine how each stage contributes to Open-o3-Video. As shown in Table 9, SFT improves the Qwen2.5- VL-7B backbone by 9.2% mAM and 14.7% mLGM, demonstrating that supervised reasoning traces establish the basic spatio-temporal grounding behavior. RL subsequently improves the SFT model by 5.2% mAM, 9.5% mLGM, and 2.9% VideoMME accuracy. The final model therefore exceeds the backbone by 14.4% mAM, 24.2% mLGM, and 1.2% VideoMME accuracy. Unlike SFT, which primarily teaches the model to imitate annotated traces, RL permits broader response exploration and rewards more precise evidence and reliable answers. The results show that the two stages are complementary: SFT establishes the grounded output pattern, while RL sharpens evidence localization and recovers general video understanding.

Grounded-thinking reward. We remove adaptive temporal proximity and temporal gating individually to test whether both mechanisms are necessary for grounded-thinking optimization. Table 10 shows that, on the V-STaR benchmark, removing adaptive temporal proximity reduces mAM/mLGM by 0.7%/1.4%, whereas removing temporal gating reduces them by 1.4%/1.7%. The degradation without temporal gating confirms that spatial overlap should receive credit only when the predicted moment is temporally valid; otherwise, a box at an irrelevant time can introduce a misleading spatial reward. The gains from adaptive temporal proximity validate the complementary role of gradually tightening the temporal criterion: a broader early reward neighborhood alleviates reward sparsity, while the stricter later criterion preserves temporal precision. Together, the two mechanisms provide accessible early supervision without weakening the final grounding objective.

Spatio-temporal supervision. This experiment evaluates whether explicit spatio-temporal training data is necessary and whether our annotation pipeline adds value beyond existing grounded-video supervision. Starting from the setting without spatio-temporal data, adding filtered and rewritten VideoEspresso [24] samples improves mAM/mLGM by 2.8%/7.4%, as shown in Table 11. Adding our annotations yields a further 2.6%/3.0%, producing total gains of 5.4% mAM and 10.4% mLGM. The first improvement demonstrates the importance of joint temporal and spatial supervision, while the additional gain verifies the benefit of our evidence-aligned annotation and filtering pipeline. These results establish high-quality spatio-temporal data as a central component of evidence-grounded video reasoning.

Faithfulness of explicit evidence. We evaluate whether the generated spatio-temporal evidence is genuinely related to the final prediction through a controlled frame-removal test. Open-o3-Video produces 1.15 boxes and 1.37 timestamps per instance on average. For every example in a 150-question VideoMME subset, we remove the two frames temporally closest to each predicted timestamp and measure the resulting answer accuracy. As a control, we remove the same number of randomly selected frames. Table 12 shows that removing evidence-related frames decreases accuracy by 2.7%, whereas random removal decreases it by only

Table 13 Test-time scaling of Open-o3-Video on WorldSense and VideoMMMU (%). Confidence-aware voting uses grounded evidence to weight eight sampled responses.
<table><tr><td>Setting</td><td>WorldSense</td><td>VideoMMMU</td></tr><tr><td>Single response (N = 1)</td><td>37.5</td><td>52.3</td></tr><tr><td>Majority voting (N = 8)</td><td>37.3</td><td>53.1</td></tr><tr><td>Confidence-aware voting (N = 8)</td><td>38.5</td><td>54.1</td></tr></table>

Table 14 Consistency between answer correctness and spatio-temporal localization on V-STaR Chain1 (%). Spatial correctness uses vIoU> 0.1, and temporal correctness uses tIoU> 0.3.
<table><tr><td>Model</td><td>QA</td><td>S</td><td>T</td><td>S&amp;T</td><td>QA&amp;S</td><td>QA&amp;T</td><td>QA&amp;S&amp;T</td></tr><tr><td>Qwen3-VL-4B</td><td>59.6</td><td>52.7</td><td>25.9</td><td>17.8</td><td>40.1</td><td>15.6</td><td>11.7</td></tr><tr><td>Open-o3-Video-v2-4B</td><td>62.8</td><td>68.0</td><td>41.5</td><td>32.8</td><td>50.1</td><td>26.4</td><td>21.7</td></tr></table>

0.7%. Because the intervention size is matched, the larger degradation from evidence-frame removal indicates that the cited timestamps capture visual information materially related to the model’s answer rather than arbitrary moments in the video.

Confidence-aware test-time scaling. We sample N = 8 independent responses in parallel with temperature 1.0. Each response contains object names, timestamps, and bounding boxes in its reasoning trace. We extract every predicted box, crop the corresponding region from the original frame, and pair the crop with the question. Open-o3-Video then assigns each crop an evidence score $s \in \{ 0 , 1 , 2 \}$ , where 0 denotes irrelevant evidence, 1 denotes partially useful evidence, and 2 denotes evidence that strongly supports an answer. We average the scores of all cited objects within a response and use the result as its voting weight. The answer with the highest total weight is selected. This procedure suppresses responses supported by irrelevant or hallucinated regions and favors answers backed by consistent visual evidence.

Table 13 shows that confidence-aware voting improves the single-response result by 1.0% on WorldSense and 1.8% on VideoMMMU. It also exceeds naive majority voting by 1.2% and 1.0%, respectively. These results show that explicit spatio-temporal evidence supports inference-time verification and self-correction in addition to making individual predictions interpretable.

## E More Analysis

Reward sensitivity. We study the sensitivity of CS-GRPO at two levels. The first controls the balance among answer correctness, format compliance, grounded thinking, and tool use in the final objective $R ,$ whose default configuration is $( w _ { \mathrm { a c c } } , w _ { \mathrm { f m t } } , w _ { \mathrm { t h k } } , w _ { \mathrm { t o o l } } ) = ( 1 . 0 , 0 . 1 , 0 . 1 , 0 . 8 )$ . The second controls the individual terms within $r _ { \mathrm { t o o l } } .$ terminal clue quality, step-wise evidence gain, excess calls, repeated intervals, sequential fragmentation, and evidence degradation. Their default coeficients are $( \lambda _ { \mathrm { c l u e } } , \lambda _ { \mathrm { g a i n } } , \lambda _ { \mathrm { b u d g e t } } , \lambda _ { \mathrm { r e p e a t } } , \lambda _ { \mathrm { f r a g } } , \lambda _ { \mathrm { d r o p } } ) =$ (1.0, 0.1, 0.1, 0.1, 0.1, 0.05).

For each row in Table 15, we vary only the indicated coeficient while keeping the remaining reward weights and all training settings unchanged. Overall, CS-GRPO remains relatively stable across the tested reward ranges, with only modest variations on both Video-Holmes and V-STaR. This suggests that the method does not rely on a narrowly tuned reward configuration. At the same time, several components benefit when their weights increase from very small values to the default setting, indicating that both evidence-gain rewards and penalties on ineficient tool use play meaningful roles in optimization. Overall, the results demonstrate that CS-GRPO is robust to moderate reward-weight changes without requiring precise tuning.

Input frames and interaction budget. We study how the initial observation budget interacts with the maximum number of reasoning turns. We randomly sample 30% of LVBench and Video-Holmes and evaluate all configurations on the same subsets. We vary the initial observation from 32 to 384 uniformly sampled frames and the maximum interaction budget from 2 to 8 turns, while keeping the model and decoding configuration fixed.

![](images/4c869e3ad8fef317c5634ea1359d2c50a78cce275a88b80314bce228dcbbf2f3.jpg)  
Figure 5 Efect of initial input-frame count and maximum interaction budget on LVBench and Video-Holmes. Results are evaluated on randomly sampled 30% subsets of each benchmark using the same model and decoding configuration.

Table 15 Sensitivity of CS-GRPO to reward weights. Each coeficient is varied independently while all others remain at their default values. Results follow the low/default/high order of the tested weights.
<table><tr><td>Reward level</td><td>Coefficient</td><td>Low / default / high</td><td>Video-Holmes</td><td></td><td>V-STaR mLGM</td><td></td></tr><tr><td rowspan="4">Final reward R</td><td> $w _ { \mathrm { a c c } }$ </td><td>0.5/1.0/1.5</td><td>47.6 48.0</td><td>48.0</td><td>55.8 56.3</td><td>56.2</td></tr><tr><td> $w _ { \mathrm { f m t } }$ </td><td>0.05/0.1/0.2</td><td>48.1 48.0</td><td>48.2</td><td>56.1 56.3</td><td>55.7</td></tr><tr><td> $w _ { \mathrm { t h k } }$ </td><td>0.05/0.1/0.2</td><td>48.4 48.0</td><td>47.6 55.9</td><td>56.3</td><td>56.6</td></tr><tr><td> $w _ { \mathrm { t o o l } }$ </td><td>0.5/0.8/1.0</td><td>47.3 48.0</td><td>47.9</td><td>56.0 56.3</td><td>56.3</td></tr><tr><td rowspan="6">Tool reward  $r _ { \mathrm { t o o l } }$ </td><td> $\lambda _ { \mathrm { c l u e } }$ </td><td>0.5/1.0/1.5</td><td>47.6 48.0</td><td>48.1</td><td>56.0</td><td>56.3 56.6</td></tr><tr><td> $\lambda _ { \mathrm { g a i n } }$ </td><td>0.05/0.1/0.2</td><td>47.3 48.0</td><td>47.8</td><td>55.4 56.3</td><td>57.1</td></tr><tr><td> $\lambda _ { \mathrm { b u d g e t } }$ </td><td>0.05/0.1/0.2</td><td>48.2 48.0</td><td>48.6</td><td>55.7 56.3</td><td>/ 56.0</td></tr><tr><td> $\lambda _ { \mathrm { { r e p e a t } } }$ </td><td>0.05/0.1/0.2</td><td>47.5 48.0</td><td>47.7</td><td>56.3 56.3</td><td>56.1</td></tr><tr><td> $\lambda _ { \mathrm { f r a g } }$ </td><td>0.05/0.1/0.2</td><td>47.6 48.0</td><td>47.5</td><td>56.7 56.3</td><td>56.6</td></tr><tr><td> $\lambda _ { \mathrm { d r o p } }$ </td><td>0.025/0.05/0.1</td><td>47.8 48.0</td><td>48.1</td><td>56.0 / 56.3</td><td>55.4</td></tr></table>

As shown in Fig. 5, increasing the initial frame budget from 32 to 96 yields the largest improvement on both benchmarks. With 5 turns, accuracy on LVBench increases from 46.2% to 52.7%, while accuracy on Video-Holmes improves from 44.5% to 47.0%. Beyond 96 frames, the gains become much smaller: accuracy on LVBench largely saturates around 53–54%, while accuracy on Video-Holmes changes little between 96 and 192 frames and slightly declines at 384 frames. This suggests that, with active evidence seeking, strong performance does not require an excessively dense initial observation. Even on LVBench, whose videos are typically very long, a moderate initial frame budget can provide suficient global context for subsequent targeted seeking and grounding.

Increasing the interaction budget is also beneficial, but the marginal gain diminishes as more turns are allowed. LVBench largely saturates by 5 turns, and Video-Holmes shows a similar pattern of progressively smaller improvements at larger budgets. This indicates that the model can often resolve missing evidence within a limited number of interactions, rather than relying on long trajectories with many potentially redundant tool calls. Overall, the results support a moderate initial observation budget together with a bounded interaction horizon, where active seeking complements sparse global observation without requiring either dense frame input or excessive tool use.

At inference time, Open-o3-Video-v2 uses only 2.2 assistant turns on average, including the final answer turn, and processes approximately 120 frames per question. To isolate the benefit of adaptive evidence seeking, we disable tool use and uniformly sample 120 frames over the full video, matching the average visual observation budget of Open-o3-Video-v2. As shown in Table 16, the full model improves VideoMME, Video-Holmes, and LVBench by 1.2, 0.6, and 7.7 points, respectively, under a comparable frame budget. The gap is largest on LVBench, where uniform sampling becomes increasingly sparse as video duration grows, while multi-turn retrieval can selectively zoom into informative temporal intervals with denser observations. These results indicate that, under comparable visual observation budgets, multi-turn adaptive seeking uses the available frames more efectively by zooming into task-relevant temporal regions rather than uniformly sampling the full video.

Faithfulness of adaptive evidence seeking. To examine whether adaptively retrieved observations provide efective evidence for subsequent reasoning and answer prediction, we replace each retrieved clip with a length-matched random interval from the same video for samples involving tool calls, while keeping the number of calls and returned frames unchanged. All other inference settings remain fixed. As shown in Table 16, replacing the retrieved observations reduces accuracy by 2.0% on VideoMME, 0.9% on Video-Holmes, and 7.0% on LVBench. Since the interaction count and visual observation budget remain unchanged, the performance degradation shows that the adaptively retrieved clips provide useful evidence for subsequent reasoning toward the final answer, rather than serving as incidental or superficial tool-use behavior. The larger drop on LVBench further highlights the importance of retrieving informative evidence for long-video reasoning.

Table 16 Analysis of adaptive evidence acquisition under comparable visual observation budgets (%). The uniform baseline uses the same model with tool use disabled and uniformly samples 120 frames. For random retrieval, each tool-returned clip is replaced by a length-matched random interval from the same video while preserving the number of tool calls and returned frames.
<table><tr><td>Setting</td><td>VideoMME</td><td>Video-Holmes</td><td>LVBench</td></tr><tr><td>Uniform sampling (120 frames, no tool)</td><td>65.4</td><td>47.4</td><td>45.5</td></tr><tr><td>Length-matched random intervals</td><td>64.6</td><td>47.1</td><td>46.2</td></tr><tr><td>Open-o3-Video-v2</td><td>66.6</td><td>48.0</td><td>53.2</td></tr></table>

Table 17 Accuracy on a 149-question long-horizon subset of LVBench, covering long videos with sparse evidence and questions involving event chains, ordinal localization, or repeated-instance tracking.
<table><tr><td>Model</td><td>Accuracy (%)</td></tr><tr><td>Qwen3-VL-4B</td><td>37.6</td></tr><tr><td>Video-o3</td><td>38.3</td></tr><tr><td>Open-o3-Video-v2</td><td>48.3</td></tr></table>

Long-horizon evidence-chain analysis. To further evaluate reasoning over sparse evidence in long videos, we construct a long-horizon subset of LVBench using only question patterns and dataset annotations, without using model predictions. The subset covers before/after event chains, ordinal localization such as first or last occurrences, and repeated-instance tracking. We further restrict the subset to videos longer than 45 minutes, resulting in 149 questions with a median video length of 79 minutes. As shown in Table 17, Open-o3-Video-v2 achieves 48.3% accuracy, outperforming Qwen3-VL-4B and Video-o3 by 10.7% and 10.0%, respectively. These results show that the proposed seek–ground–reason formulation is efective when relevant evidence is temporally sparse and must be located or distinguished across distant or repeated events.

Answer--grounding consistency. Table 14 examines whether improvements in answer accuracy and spatiotemporal grounding occur jointly rather than independently. Open-o3-Video-v2-4B increases the joint QA–spatial–temporal success rate from 11.7% to 21.7%, an absolute gain of 10.0%. More importantly, among correctly answered questions, the fraction that is also supported by both spatially and temporally correct evidence increases from 19.6% to 34.5%. The gains are also consistent across the pairwise metrics: QA–spatial correctness rises from 40.1% to 50.1%, while QA–temporal correctness increases from 15.6% to 26.4%. These results show that the improvements are not limited to answer accuracy or localization in isolation. Open-o3-Video-v2 increases the frequency with which a correct answer is accompanied by consistent spatio-temporal evidence, directly supporting our goal of reasoning with verifiable visual support.

## F Qualitative Results

## F.1 Successful Cases

Figs. 6 and 7 present representative successful cases of Open-o3-Video-v2. The first comparison focuses on fine-grained action and event understanding, where the decisive evidence is dificult to resolve from the initial sparse observation. Open-o3-Video-v2 selectively inspects the relevant temporal intervals and grounds the supporting objects within the acquired observations, exposing visual details that are missed or misinterpreted by the baselines. The second comparison focuses on fine-grained visual counting, where the target instances are spatially distinguishable once the relevant observation is acquired. These examples demonstrate how targeted observation and explicit spatio-temporal grounding provide more precise visual evidence for subsequent reasoning.

## F.2 Failure Analysis

Our quantitative analysis of the incorrect predictions on LVBench shows that precise temporal evidence retrieval and event selection constitute the dominant failure mode, accounting for approximately 43% of the errors. Object grounding and fine-grained perception account for another 27%, including errors in instance discrimination, counting, and visual attribute recognition. Knowledge-related errors are less frequent, accounting for approximately 6%, but remain challenging when correct prediction requires identity or external semantic knowledge. The remaining errors mainly involve reasoning inconsistencies or cases that cannot be attributed to a single factor.

Fig. 8 illustrates representative examples of these failure modes. In the first example, the model retrieves a visually relevant face but still fails to identify the public figure correctly, reflecting a remaining limitation in identity and world knowledge. In the second example, the camera pans across the ofstage area, requiring the model to distinguish and associate instances across changing views before determining the total count. The model instead grounds incorrect instances and produces an erroneous count, indicating that dynamic counting under camera motion remains challenging when visually similar or partially observed objects appear across multiple frames. In the third example, the model grounds a genuine scoring event but not the game-ending one, indicating dificulty in distinguishing the target event from visually similar events occurring nearby in time. These failures therefore expose directions for further improvement: better preservation and integration of world knowledge during post-training, and more precise spatio-temporal localization and instance association in complex scenes with confusingly similar events and objects. Explicit spatio-temporal traces make these weaknesses easier to localize and analyze, while further improvements will require stronger fine-grained perception and temporal event discrimination.

Future work. In future work, we will explore longer-horizon video agents that integrate visual evidence with world knowledge and perform counting and analysis on extremely long videos.

![](images/e239c0e77ab3cafda0f043c56817c58a4911e230b785aeae7587fa704126d883.jpg)  
Figure 6 Qualitative comparisons on fine-grained visual reasoning. Top: For action-change perception, Open-o3-Videov2 inspects the relevant 134–143 s interval and grounds the man’s hand, sea urchin, and fish at 138.5 s. The denser observation reveals the fish’s blue body and yellow fins after the interaction, enabling the correct answer. Bottom: For fine-grained event understanding, Open-o3-Video-v2 grounds the metal ring cutter and pineapple slice and further inspects the 33–46 s interval using denser temporal sampling. The resulting observation reveals the repeated removal of the fibrous core and provides the missing evidence for answer B. In both cases, targeted temporal observation together with explicit object-level grounding exposes the decisive visual evidence and yields a human-verifiable spatio-temporal reasoning trace.

![](images/0930243cec41e6e0244f079bc69e9493fd9cc411b01f6c8078a250bddb2f3ffd.jpg)  
Figure 7 Qualitative comparisons on fine-grained visual counting. Top: For counting food pictures on the menu at 17:49, Open-o3-Video-v2 inspects the relevant 1065–1075 s interval and grounds the menu at 1069.5 s, providing a clearer view of the individual food images and enabling the correct count of 11. In contrast, Video-o3 searches an incorrect temporal interval and produces an incorrect count. Bottom: For counting the people in the car, both methods inspect the relevant 10–12 s interval, but Open-o3-Video-v2 further grounds the car and its occupants, allowing it to distinguish three individuals rather than two. In both cases, explicit instance-level grounding complements temporal evidence acquisition by resolving the individual visual entities required for accurate counting.

![](images/728ac3ec52bc585e96c4425d0ee7d6b41252dd3f72b14e360b143a1aca608c1f.jpg)  
Question: Who are the two people who appear in the next frame after Evan Osnos finishes speaking? A. Trump and Biden B. Obama and Putin C. Obama and Trump D. Obama and Biden Ground Truth: C

![](images/f3dee28b6ad3d88973c50481f3c73756055d17ff7525effb4e18823cc1f3ab6b.jpg)

Question: During Putin's first speech on stage in the video,   
how many women appeared when the camera panned   
offstage?   
A. 2 B. 4 C. 1 D. 0   
Ground Truth: C Question: Who is the player that ends the game in the video?   
A. White number 6 B. Black number 5.   
C. Black number 4 D. White number 8.   
Ground Truth: C

![](images/bac55f01d8e2517bac1fe45ac37a95accb5a00989922bd44a234ec4dbd9e80f2.jpg)

## Open-o3-video v2:

<think> The video segment from <t>2510</t>s to <t>2540</t>s shows Evan Osnos speaking. Immediately after he finishes, at around <t>2515</t>s, the video cuts to a scene showing <obj>barack obama </obj><box>[232,196,556,1000]</box>at<t>2515.0</t>s and <obj>vladimir putin</obj><box>[232,196,556,1000]</box>at<t>2515.0</t>s shaking hands in front of their respective national flags. This visual evidence directly confirms that the two people who appear after Evan

## Open-o3-video v2:

Osnos finishes speaking are Barack Obama and Vladimir Putin. This matches option B. </think>

## <sub>Lack</sub> <sub>World</sub> <sub>Knowledge</sub>！

![](images/4ba5e2b1c16d7545a21065435f6d7fcb32c4e98c240292876ad39207d7d5edb5.jpg)

<think>The video segment from <t>288</t>s to <t>294</t>s shows Putin walking down a red carpet towards a stage. At <t>291.0</t>s, the camera pans to the right, revealing the audience. In this shot, there are several people in the audience. I can clearly identify two women. One is a <obj>woman in light-colored suit</obj><box>[568,400,598,580]</ box>at<t>291.0</t>s standing near the center of the frame, and another is a <obj>woman in dark suit</obj><box>[850,580,956,1000]</ box>at<t>291.0</t>s on the far right. No other women are visible in this specific pan. Therefore, the total number of women who appeared when th offstage is two. This directly supports option A.</think>

<think>At <t>0.0</t>s, the <obj>white number 8 player</obj><box>[274,508,310,738]</ box>at<t>0.0</t>s is in possession of the ball. He drives toward the basket and makes a layup. The <obj>basketball</obj> <box>[303,606,320,644]</box>at<t>0.0</t>s goes through the <obj>basketball hoop</obj> <box>[591,254,626,336]</box>at<t>0.0</t>s

## Open-o3-video v2:

## <sub>Wrong</sub> S<sub>patial</sub> <sub>Evidence</sub>！

![](images/2b3e6c47fbb36ab6ba8994fa467dfe592b93b0bd250b2b0b9aada94166bde631.jpg)

Immediately after, at <t>8.008</t>s, the <obj>scoreboard</obj><box>[702,766,980,978]</ box>at<t>8.008</t>s updates to show the score changing from 48-48 to 50-49, confirming that the white number 8 player scored the final basket. This action ends the game. Therefore, the player who ends the game is the white number 8 player. , </think>

## <sub>Wrong</sub> <sub>Temporal</sub> <sub>Evidence</sub>！

![](images/964299fcfbd984414e93c967212e666014aca04e0ba8ed5f1226a63abb064f6f.jpg)

Figure 8 Representative failure cases of Open-o3-Video-v2. Top: Insuficient identity knowledge leads to an incorrect answer even after a relevant face is retrieved. Middle: During a camera pan, inaccurate instance discrimination and association across changing views lead to an incorrect count of the women appearing ofstage. Bottom: The model anchors its reasoning to the wrong scoring event and misidentifies the player who ends the game. These examples expose remaining limitations in world knowledge, fine-grained spatio-temporal instance reasoning, and temporal event selection, respectively.