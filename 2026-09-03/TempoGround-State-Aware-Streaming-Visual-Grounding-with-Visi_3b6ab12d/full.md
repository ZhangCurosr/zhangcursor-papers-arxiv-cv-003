# TempoGround: State-Aware Streaming Visual Grounding with Vision-Language Models

Leqian Ding<sup>1∗</sup>, Junning Qiu<sup>2†</sup>, Manwen Yang<sup>1</sup>, Yu Guo<sup>1†</sup>, Fei Wang<sup>1</sup>

<sup>1</sup>Xi’an Jiaotong University, Xi’an, China <sup>2</sup>EngineAI, Shenzhen, China

## Abstract

Visual grounding maps language referents to spatial targets and is central to open-vocabulary perception with visionlanguage models. Existing methods have made substantial progress on single-frame and video-based visual grounding, yet under streaming inputs they still sufer from identity drift, cross-frame inconsistency, and fragile localization under partial occlusion. To address these issues, we present TempoGround, a VLM-native framework that detects crossframe object correspondence and explicitly models object presence states, thereby enabling accurate and consistent visual grounding under streaming inputs. The key is a curriculum prediction mechanism guided by state-aware cross-frame correspondence: TempoGround resolves 2D instance association, predicts whether each object newly enters, continues in, or leaves the view, decodes the 2D box, and then lifts it to a camera-frame 3D box. As token-level supervision alone cannot capture the geometric objectives of streaming grounding, we further introduce Streaming Grounding Reinforcement (SGR), which optimizes TempoGround with verifiable Grounding, Identity, and Consistency rewards, jointly reinforcing persistent localization and temporally consistent predictions. We carefully design a three-stage training strategy and train TempoGround on large-scale data. We evaluate visual grounding under causally streaming inputs on multiple challenging benchmarks: TempoGround improves F1<sub>2D</sub>@0.5 and $\Gamma 1 _ { 2 D } @ 0 . 9 5$ by 4.4 and 0.5 on average, and $\mathrm { F } 1 _ { 3 D } @ 0 . 2 5$ and $\mathrm { A P } _ { 3 D }$ by 6.2 and 7.5, respectively. These results demonstrate that TempoGround provides a practical foundation for visual grounding under streaming inputs.

## 1 Introduction

Vision-language models (VLMs) have made visual grounding a practical route to open-vocabulary perception: given a free-form language query, the model localizes the referred objects as spatially actionable targets, which is important for embodied perception and related downstream control. Most existing VLM grounders operate on single-frame or video-based inputs. In practical deployments, however, sensory observations are delivered as a causal stream. We define streaming visual grounding as the setting in which a language query is given, images arrive incrementally, and the model must continuously track the matching objects in view and produce their 2D/3D bounding boxes. As shown in Figure 1, streaming visual grounding commonly exhibits three characteristic failure modes. Identity drift confuses samecategory instances across time. Cross-frame inconsistency yields temporally unstable boxes across adjacent frames. Fragile localization under partial occlusion arises when intermittent visibility causes missed or degraded detections that cannot be corrected without causal cross-frame reasoning.Consequently, visual grounding under streaming inputs remains challenging.

Recent 3D scene understanding methods enhance objectlevel perception and grounding by injecting globally aligned geometric information. These approaches typically perform object grounding in a world or scene coordinate system and therefore introduce a global coordinate dependency. For object-level perception under ego-centric viewpoints, as in embodied manipulation and related scenarios, world-frame predictions are not directly applicable in the current camera view and require an additional alignment step that introduces additional alignment error. We therefore aim for a VLM-native method for streaming visual grounding that consumes causal image streams and predicts in the current camera frame.

To address this limitation, we present TempoGround with a state-aware streaming curriculum prediction mechanism that leverages cross-frame correspondence for object state prediction and spatial localization. At each timestep, under language guidance, TempoGround first associates instances across adjacent frames by matching a lightweight 2D cue from the previous prediction against corresponding regions in the current image, thereby maintaining consistent object identity. Based on this correspondence, TempoGround then estimates whether each object newly enters, persists in, or exits the field of view. Conditioned on these correspondence and presence cues, TempoGround better captures cross-frame object states and spatial localization. As reliable 3D predictions benefit from stable 2D evidence (Man et al. 2026), TempoGround first predicts a 2D box for each instance and then lifts it to a camera-frame 3D box. Taken together, TempoGround follows a curriculum prediction chain that resolves 2D correspondence, estimates object presence, decodes the 2D box, and lifts the prediction to 3D. This design allows TempoGround to exploit cross-frame instance correspondence and spatial position changes under streaming inputs, supporting more robust and accurate tracking.

As a VLM-native approach to streaming object grounding,

![](images/6f32f93dd083fe27bc9ba3a030e3ad652cf534e1b13507dbdf9b07a3d9bf6b9e.jpg)  
Figure 1: TempoGround for streaming visual grounding. Under streaming inputs, visual grounding commonly exhibits three characteristic failure modes (left). By exploiting cross-frame correspondence for state-aware curriculum prediction, TempoGround yields more stable camera-frame 2D/3D localization under streaming inputs (right) for both detection and referring grounding.

TempoGround is built on a standard VLM decoder, produces outputs in a standard autoregressive manner, and is trained with standard supervised fine-tuning on carefully constructed large-scale data. However, supervised fine-tuning alone remains insuficient, because token-level supervision treats bounding box prediction as an autoregressive token sequence and therefore does not directly model the geometric attributes of visual bounding boxes, nor does it provide direct box-level supervision under streaming inputs. To meet the consistency requirements of streaming grounding, we introduce Streaming Grounding Reinforcement (SGR), an RLVR stage that optimizes TempoGround with verifiable Grounding, Identity, and Consistency rewards. These rewards target localization quality along the stream, correct enter and exit decisions, and stable cross-frame predictions.

Building on the above considerations, we construct a threestage training recipe that progresses from single-frame geometric competence to streaming supervision and finally to sequence-level reinforcement. Stage 1 applies supervised fine-tuning on 26M samples to strengthen large-scale 2D detection and 2D-to-3D lifting, establishing the spatial foundation for subsequent learning. Building on this competence, Stage 2 continues with supervised fine-tuning on 5.1M samples to establish curriculum prediction, enabling cross-frame correspondence to support presence estimation and consistent localization. Stage 3 then refines these streaming capabilities with SGR under verifiable rewards, aligning training with the geometric objectives that token-level supervision alone cannot capture.

## Our contributions are as follows:

• We propose TempoGround, a VLM-native framework that enables accurate and consistent streaming visual grounding through curriculum prediction guided by state-aware cross-frame correspondence, from 2D association and presence estimation to camera-frame 3D grounding.

• We introduce Streaming Grounding Reinforcement (SGR), an RLVR method that optimizes TempoGround with verifiable Grounding, Identity, and Consistency rewards to align training with the geometric objectives of streaming grounding.

• Extensive experiments on multiple challenging benchmarks under causally streaming inputs show that TempoGround improves $\mathrm { \dot { F } } 1 _ { 2 D }$ at IoU thresholds 0.5 and 0.95 by 4.4 and 0.5 on average, and improves $\mathrm { F } 1 _ { 3 D }$ at IoU 0.25 and $\mathrm { A P } _ { 3 D }$ by 6.2 and 7.5, providing a practical foundation for streaming visual grounding.

## 2 Related Works

## 2.1 VLM-based Visual Perception

Vision-language models have been increasingly applied to spatial understanding in 3D environments. Reconstructed geometry or multi-view video can be encoded into LLMs for relational and layout-level dialogue (Hong et al. 2023), coupled with embodied reasoning and control (Huang et al. 2023), or unified with grounding, captioning, and question answering under shared instruction following (Huang et al. 2024; Zheng, Huang, and Wang 2025). Under camera motion, explicit geometry is further injected through positionaware visual tokens (Zhu et al. 2024), globally consistent geometric fusion (Zheng et al. 2026), or reconstructive 3D tokens for spatial reasoning (Fan et al. 2025), while firstperson 3D sensing resources continue to expand (Wang et al. 2024). These methods improve scene-level semantics and spatial reasoning over ofline reconstructions and short clips, yet their supervision and inference remain largely scan- or clip-centric, and object predictions are often formulated in a world or scene coordinate system. TempoGround instead targets object-level localization in the current camera frame under causally arriving image streams.

## 2.2 Visual Object Grounding

Visual object grounding associates language queries with spatial targets. Open-vocabulary detectors condition localization on category names or referring expressions (Liu et al. 2024; Chen, Chang, and Nießner 2020), while multimodal LLMs cast boxes as autoregressive tokens (Peng et al. 2024; Bai et al. 2025; Chen et al. 2023). Subsequent work strengthens this generative formulation through pixelaligned grounding (Rasheed et al. 2024), high-throughput generative detection (Jiang et al. 2025; Wang et al. 2026), and staged 2D-to-3D prediction in the camera frame (Man et al. 2026). Reinforcement with verifiable rewards further aligns generated geometry with evaluation metrics (Linghu et al. 2026). These advances mainly address single-frame or ofline clip settings, where each observation can be treated largely independently or with access to the full sequence. Streaming visual grounding additionally requires identitystable and cross-frame-consistent box updates as images arrive causally. TempoGround addresses this setting.

## 3 Method

We present TempoGround, a VLM-native framework for streaming visual grounding. Figure 2 overviews the stateaware curriculum prediction interface and the progressive three-stage training recipe. Section 3.1 formalizes the streaming visual grounding task and the geometry conventions used throughout. Section 3.2 describes curriculum prediction guided by state-aware cross-frame correspondence. Section 3.3 introduces Streaming Grounding Reinforcement (SGR) with verifiable Grounding, Identity, and Consistency rewards. Section 3.4 presents the progressive three-stage training recipe. Section 3.5 summarizes data curation for single-frame and streaming supervision. Additional implementation details are deferred to the appendix.

## 3.1 Problem Formulation

Streaming visual grounding. Let q denote a free-form language query, either a category name (detection) or a referring expression (refer). Let $I _ { 1 : T } = ( I _ { 1 } , \ldots , I _ { T } )$ be a causally arriving RGB stream, where $I _ { t } \in \dot { \mathrm { R } } ^ { H \times W \times 3 }$ is the observation at timestep t. At each timestep t, the model must localize all objects in view that match q and output their 2D boxes and camera-frame 3D boxes. Formally, the prediction at timestep t is a (possibly empty) set

$$
\mathcal { V } _ { t } = \left\{ y _ { t } ^ { ( k ) } \right\} _ { k = 1 } ^ { N _ { t } } , \qquad y _ { t } ^ { ( k ) } = \left( \mathbf { b } _ { t } ^ { \mathrm { 2 D } , ( k ) } , \mathbf { b } _ { t } ^ { \mathrm { 3 D } , ( k ) } \right) ,\tag{1}
$$

where $N _ { t }$ is the number of predicted instances, $\mathbf { b } _ { t } ^ { 2 \mathrm { D } , ( k ) } \in \mathrm { R } ^ { 4 }$ is an axis-aligned 2D box, and $\mathbf { b } _ { t } ^ { \mathrm { 3 D } , ( k ) } \in \mathbb { R } ^ { 9 }$ is a cameraframe 3D box. The streaming setting is causal: the prediction at timestep t may depend only on $q ,$ observations $I _ { 1 : t }$ , and previous predictions $\mathcal { \partial } _ { 1 : t - 1 }$

Geometry conventions. All 2D boxes are represented in normalized image coordinates $\mathbf { b } ^ { \mathrm { 2 D } } = \left( x _ { 1 } , y _ { 1 } , \dot { x } _ { 2 } , y _ { 2 } \right)$ with each coordinate quantized to integers in [0, 1000]. All 3D boxes are expressed in the current camera frame as

$$
{ \bf b } ^ { \mathrm { 3 D } } = ( c _ { x } , c _ { y } , c _ { z } , s _ { x } , s _ { y } , s _ { z } , r , p , y ) ,\tag{2}
$$

where $\left( c _ { x } , c _ { y } , c _ { z } \right)$ is the box center in meters, $\left( { { s _ { x } } , { s _ { y } } , { s _ { z } } } \right)$ is the size in meters, and $( r , p , y )$ denotes roll, pitch, and yaw in radians.

## 3.2 Curriculum Prediction with State-Aware Cross-Frame Correspondence

TempoGround is built on a standard autoregressive VLM decoder. At timestep t, under the guidance of $q ,$ the decoder emits a structured token sequence that realizes the following curriculum: 2D correspondence → presence estimation → 2D box decoding → camera-frame 3D lifting.

Cross-frame correspondence. Let $\hat { \mathcal { V } } _ { t - 1 }$ denote the model prediction at the previous timestep (empty when $t = 1 )$ . At each timestep $t > 1$ , TempoGround takes both the previous image $I _ { t - 1 }$ and the current image $I _ { t }$ as visual inputs. We construct a lightweight correspondence cue

$$
\begin{array} { r } { \mathbf { c } _ { t - 1 } = \left\{ \left( \boldsymbol { \ell } _ { t - 1 } ^ { ( k ) } , \mathbf { b } _ { t - 1 } ^ { \mathrm { 2 D } , ( k ) } \right) \right\} _ { k } , } \end{array}\tag{3}
$$

which retains only labels and 2D boxes from $\hat { \mathcal { V } } _ { t - 1 }$ and discards 3D boxes. Camera-frame 3D boxes are coupled to the viewpoint at the previous timestep; carrying them forward would inject stale geometric bias into the current frame. The 2D cue instead provides a view-stable association signal, after which TempoGround re-estimates 3D in the current camera frame. Conditioned on the adjacent image pair $\left( I _ { t - 1 } , I _ { t } \right)$ , language query $q ,$ and correspondence cue $\mathbf { c } _ { t - 1 } .$ , TempoGround associates instances across frames by matching $\mathbf { c } _ { t - 1 }$ against corresponding regions in $I _ { t }$

Presence state estimation. To strengthen cross-frame instance association, TempoGround predicts a discrete presence state for each associated instance. We use the presence vocabulary

$$
\begin{array} { r } { \mathcal { S } = \{ \mathrm { n e w } , \mathrm { t r a c k } , \mathrm { 1 } \mathrm { o s t } \} , } \end{array}\tag{4}
$$

where new, track, and $\bot _ { \mathsf { O S t } }$ respectively indicate that an object newly enters the field of view, persists in the view, or exits the view. A predicted state $s _ { t } \in S$ determines whether the corresponding instance contributes a box to $\mathcal { \partial } _ { t } .$ If a previously lost query-matching object re-enters the view, TempoGround treats it as new again: presence is defined with respect to contiguous visibility under language guidance, rather than as a persistent identity memory across exits. Supervising these states is critical for suppressing identity drift among currently visible instances and for preventing continued box predictions after objects leave the view.

From 2D boxes to camera-frame 3D. After correspondence and presence are resolved, TempoGround decodes a 2D box for each visible instance and then lifts it to a camera-frame 3D box of the form $\begin{array} { r l } { \mathbf { b } ^ { \mathrm { 3 D } } } & { { } = } \end{array}$ $( c _ { x } , c _ { y } , c _ { z } , s _ { x } , s _ { y } , s _ { z } , r , p , y )$ . Predicting 2D first and then lifting to 3D lets the 2D box serve as an intermediate spatial anchor that stabilizes monocular camera-frame 3D estimation. Multi-instance responses are serialized in a canonical order. The resulting autoregressive target interleaves presence, 2D, and 3D tokens in the same order used at inference.

![](images/ecff202de9811f32a54bfcbbd93bac52779be87e7e9a28243326416c2b289aff.jpg)  
Figure 2: TempoGround overview. Top: state-aware curriculum prediction guided by cross-frame correspondence, consuming adjacent dual-image frames with the language query and previous 2D cue, from presence-aware 2D localization to camera-frame 3D lifting. Bottom: progressive three-stage training from single-frame SFT to streaming curriculum prediction and Streaming Grounding Reinforcement (SGR).

## 3.3 Streaming Grounding Reinforcement

Token-level supervised fine-tuning trains the decoder to imitate discrete coordinate tokens. This objective cannot directly model continuous box geometry, nor does it provide trajectory-level supervision when each step depends on previous model outputs. We therefore introduce Streaming Grounding Reinforcement (SGR), an RLVR stage that optimizes TempoGround with verifiable rewards on streamed rollouts.

Objective. Given a streaming prompt $x = \left( q , I _ { 1 : T } \right)$ , the policy π<sub>θ</sub> samples a trajectory $\tau = \mathcal { V } _ { 1 : T } . \mathrm { ~ A ~ }$ deterministic verifier returns a scalar reward $R ( \tau )$ . SGR maximizes the expected reward

$$
J ( \theta ) = \mathrm { E } _ { x \sim \mathcal { D } , \tau \sim \pi _ { \theta } ( . | x ) } \left[ R ( \tau ) \right] .\tag{5}
$$

We optimize this objective with Group Relative Policy Optimization (GRPO) (Shao et al. 2024). For each prompt, GRPO samples a group $\{ \tau _ { i } \} _ { i = 1 } ^ { G }$ from the old policy and computes the group-normalized advantage

$$
A _ { i } = \frac { R ( \tau _ { i } ) - \mathrm { m e a n } ( \{ R ( \tau _ { j } ) \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R ( \tau _ { j } ) \} _ { j = 1 } ^ { G } ) + \epsilon } ,\tag{6}
$$

where $\epsilon > 0$ is a small constant for numerical stability. During SGR rollouts, the correspondence cue $\mathbf { c } _ { t - 1 }$ is taken from the model’s own previous prediction, so optimization reflects closed-loop streaming behavior.

![](images/d7204141ba1cef72c466e90eb0875ded1c8e9fa3860ff0187b865b136b429c87.jpg)  
Figure 3: Verifiable rewards in SGR.

Verifiable rewards. SGR assigns each closed-loop trajectory a scalar reward

$$
R ( \tau ) = R _ { \mathrm { g } } ( \tau ) + R _ { \mathrm { i d } } ( \tau ) + R _ { \mathrm { c o n } } ( \tau ) - f _ { \mathrm { f m t } } ( \tau ) ,\tag{7}
$$

where $f _ { \mathrm { f m t } }$ penalizes unparsable outputs. As illustrated in Figure 3, The three main terms are:

• Grounding $R _ { \mathrm { g } } { \mathrm { : } }$ encourages accurate and persistent 2D/3D boxes for query-matching objects throughout the stream;

• Identity $R _ { \mathrm { i d } } \colon$ encourages correct enter and exit behavior, so objects are detected when they appear and are not predicted after they leave;

• Consistency $R _ { \mathrm { c o n } } i$ : discourages long streaks of failures that propagate through self-generated correspondence cues.

Computation details are provided in the appendix.

## 3.4 Progressive Three-Stage Training

We train TempoGround with a progressive three-stage curriculum that moves from single-frame geometric competence to streaming supervision and finally to sequence-level reinforcement.

Stage 1: Single-frame 2D detection and 2D-to-3D lifting. Stage 1 applies supervised fine-tuning on about 26M singleframe samples covering large-scale 2D detection/referring and camera-frame 3D detection/referring. The model learns open-vocabulary localization and the 2D-then-3D decoding order that underlies later curriculum prediction.

Stage 2: Streaming curriculum prediction. Stage 2 continues supervised fine-tuning on about 5.1M samples. Each sample provides an adjacent image pair $\left( I _ { t - 1 } , I _ { t } \right)$ , a language query, and a correspondence cue derived from the previous frame, and supervises presence states together with 2D/3D boxes. This stage installs state-aware cross-frame correspondence so that presence estimation and consistent localization become learnable under temporal cues.

Stage 3: SGR. Stage 3 refines the Stage 2 policy with SGR on streamed trajectories under the objective and rewards introduced above. This stage aligns training with geometric and consistency objectives that token-level imitation cannot capture.

## 3.5 Data Curation

We curate a camera-centric corpus that supports both singleframe Stage 1 supervision and streaming Stages 2–3. Heterogeneous public sources are converted into a unified JSONL schema and packaged as VLM conversations whose response order matches the curriculum above. Detailed filtering thresholds, packaging templates, and per-source statistics are provided in the appendix.

Sources and scale. Table 1 summarizes training sources. Stage 1 mixes large-scale 2D corpora with camera-frame 2D+3D corpora, totaling about 26M samples. Stages 2–3 are built from multi-view RGB-D sequences with stable object identities. Stage 2 uses 5.1M streaming samples, from which Stage 3 draws a subset for SGR.

Unified geometry interface. Every sample is stored in a shared JSONL format with image pointers, language queries, instance lists, and camera intrinsics. Box normalization and geometry conventions follow Section 3.1. This unification removes dataset-specific heads and keeps Stage 1 3D supervision consistent with streaming camera-frame prediction.

Quality control. Aggregating heterogeneous public sources inevitably introduces inconsistent category granularity and unreliable geometry. We canonicalize ambiguous or overly fine-grained labels, remove geometrically degenerate classes, and discard low-quality boxes through frustum, visibility, and consistency checks. Calibrated negatives are further included so that absent queries teach rejection rather than hallucination. On streaming sources, presence-state labels follow per-frame visibility, and clips are sampled with randomized length and stride to diversify temporal contexts.

Concrete filtering thresholds and cleaning pipelines are provided in the appendix.

Correspondence-cue perturbation. A natural concern for cue-conditioned streaming is error amplification: once the previous prediction is wrong, under fast ego-motion the previous 2D box may misalign severely with the current image, and a model that over-relies on $\mathbf { c } _ { t - 1 }$ can drift across subsequent frames. To prevent such brittle dependence, Stage 2 constructs $\mathbf { c } _ { t - 1 }$ from a mixture of clean ground-truth cues, Stage 1 rollouts, and controlled synthetic perturbations, while keeping supervised targets unchanged. Table 2 lists the nonclean modes, whichjointly cover 80% of Stage 2 samples; the remaining 20% use clean previous-frame ground-truth 2D boxes. Rollout cues provide realistic closed-loop errors; synthetic modes further cover ego-motion misalignment, missed and spurious detections, stale cues, and stream restarts. Implementation details of each operator are deferred to the appendix.

View-grounded referring annotation. Public datasets such as ScanRefer provide scene-level referring expressions that are not suited to map-free streaming, so we discard them as primary supervision and regenerate view-grounded queries on ego imagery. Each expression avoids absolute directional phrases tied to the global scene layout and instead relies on target attributes and relations to co-visible anchors in the current view. Along a stream, the target and its anchors jointly determine the first resolvable timestep: the object is labeled new only when the query becomes uniquely decidable from the observed view, after which presence states follow the target with cross-frame correspondence. Annotation protocols and uniqueness checks are detailed in the appendix.

## 4 Experiments

## 4.1 Implementation Details

Training details. We initialize from Qwen3.5 and fully fine-tune all parameters with AdamW in ms-swift (Zhao et al. 2025). We use bfloat16, FlashAttention, and DeepSpeed ZeRO-3, with weight decay 0.05 and gradient clipping at 1.0. All training runs on 128 Alibaba Cloud PPUs and takes about 109 hours.

Stage 1 trains for 1 epoch with learning rate $4 \times 1 0 ^ { - 5 }$ vision learning rate $2 \times 1 0 ^ { - 5 }$ , sequence length 8192, packing length 8192, and global batch size 1024, taking about 69 hours.

Stage 2 uses the same schedule for 1 epoch with sequence length 16384, packing length 16384, and global batch size 1024, taking about 21 hours.

Stage 3 trains with GRPO for 2 epochs on about 4.1K streaming trajectories, with 32 prompts per step, learning rate $1 \times \mathrm { { 1 0 ^ { - 6 } } }$ , vision learning rate $5 \times 1 0 ^ { - 7 }$ , KL coeficient $\beta { = } 0 . 0 1$ , G=4 rollouts at temperature 1.0, and up to 15 streaming turns, taking about 19 hours.

Evaluation setup. To ensure a fair comparison, we evaluate all methods under the same causal streaming protocol. Single-frame methods take each frame independently; videobased methods take the clip observed so far at timestep t and report only the current-frame prediction. Each dataset contributes 24 test sequences with equal detection and referring samples. We report $\mathrm { F 1 _ { 2 D } @ 0 . 5 / @ 0 . 9 5 }$ for 2D and $\mathrm { F 1 _ { 3 D } \mathrm { \bar { @ } 0 . 2 5 / A P _ { 3 D } } }$ for 3D. These metrics are suficient for our setting because streaming failures ultimately surface as localization errors: incorrect enter/exit or identity decisions cause misses and spurious boxes, while cross-frame inconsistency yields unstable matches over time. F1 averages perframe precision and recall along the stream and therefore reflects whether presence and correspondence remain correct at typical timesteps. $\mathrm { A P _ { 3 D } }$ aggregates predictions over the full trajectory and is more sensitive to persistent false positives and fragmented tracks under closed-loop updates. Together, they evaluate both step-wise grounding quality and stream-level stability without requiring separate state-only scores as primary metrics.

Table 1: Training sources for TempoGround. Stage 1 uses 15.6M 2D samples and 10.5M camera-frame 2D+3D samples. Stage 2 uses 5.1M samples, from which Stage 3 draws a subset.
<table><tr><td colspan="3">Stage Modality</td><td>Sources</td><td>Samples</td></tr><tr><td rowspan="2">1</td><td>2D</td><td>2025), Flickr30k Entities (Plummer et al. 2015)</td><td>Objects365 (Shao et al. 2019), V3Det (Wang et al. 2023), COCO (Lin et al. 2014), LVIS (Gupta, Dollar, and Girshick 2019), EgoObjects (Zhu et al. 2023), BDD100K (Yu et al. 2020), nuImages (Caesar et al. 2020), PACO (Ramanathan et al. 2023), RefCOCO/+/g (Kazemzadeh et al. 2014; Yu et al. 2016), RefL4 (Chen et al.</td><td>15.6M</td></tr><tr><td>2D+3D</td><td></td><td>ScanNet++ (Yeshwanth et al. 2023), ScanNet (Dai et al. 2017), ADT (Pan et al. 2023), ASE (Avetisyan et al. 2024), ARKitScenes (Baruch et al. 2021), Hypersim (Roberts et al. 2021), Objectron (Ahmadyan et al. 2021), SUN-RGBD (Song, Lichtenberg, and Xiao 2015), KITTI (Geiger, Lenz, and Urtasun 2012), nuScenes</td><td>10.5M</td></tr><tr><td>2-3</td><td>Stream</td><td></td><td>ARKitScenes, ASE, ADT, ScanNet++, ScanNet</td><td>5.1M</td></tr></table>

Table 2: Non-clean correspondence-cue modes in Stage 2 (80% of samples; the remaining 20% use clean GT cues). Each mode corrupts only the input prior $\mathbf { c } _ { t - 1 } ;$ supervision targets remain unchanged.
<table><tr><td>Mode</td><td>Ratio</td><td>Role in training</td></tr><tr><td>Stage 1 rollout</td><td>25%</td><td>Uses frozen Stage 1 predictions as cues to expose realistic closed-loop errors.</td></tr><tr><td>Jitter</td><td>20%</td><td>Shifts prior 2D boxes to simulate fast ego-motion and cross-frame misalignment.</td></tr><tr><td>Dropout</td><td>12%</td><td>Removes prior instances to teach recovery from previous-step misses.</td></tr><tr><td>Phantom</td><td>12%</td><td>Inserts false boxes or wrong labels to discourage trusting spurious cues.</td></tr><tr><td>Stale</td><td>7%</td><td>Replaces the t—1 cue with an older cue to handle outdated association.</td></tr><tr><td>Empty</td><td>4%</td><td>Clears the cue to force re-initialization after a stream break.</td></tr></table>

## 4.2 Main Results

Table 3 and Table 4 report 3D and 2D results under causa streaming inputs. TempoGround consistently surpasses both general-purpose VLMs and specialist grounders, indicating that state-aware curriculum prediction and SGR are efective for online localization rather than only for ofline singleframe recognition. The 2D gains under the same protocol further show that the benefit is not confined to camera-frame 3D lifting, but also improves image-plane grounding when predictions must be refreshed causally.

## 4.3 Ablation Studies

All ablations keep the same TempoGround-9B backbone, data budget, and evaluation protocol on ScanNet under causal streaming with $\mathrm { A P _ { 3 D } }$ . Curriculum-prediction ablations use the Stage 2 supervised checkpoint for computational eficiency, while SGR ablations remove SGR or individual rewards from the full model.

Table 5 shows that both cross-frame correspondence and presence supervision improve $\mathrm { A P _ { 3 D } }$ , to diferent degrees. Carrying camera-frame 3D boxes in the prior underperforms the 2D-only cue, consistent with Section 3.2: previous-view 3D geometry can bias localization under the current camera frame. Table 6 shows that Stage 3 SGR brings a clear overall gain, and removing any of Grounding, Identity, or Consistency lowers accuracy, indicating that all three rewards contribute positively.

## 4.4 In-depth Analysis

Under closed-loop streaming, errors in $\mathbf { c } _ { t - 1 }$ can propagate across timesteps. We therefore corrupt the incoming cue at inference with spatial jitter, box dropout, and label mixing, sweeping the corruption intensity from 0% to 80%. Figure 4 compares four Stage 2 cue schedules on ScanNet $\mathrm { A P _ { 3 D } }$

Training with clean ground-truth cues alone yields a low closed-loop starting point and a steep decay, revealing an oracle-prior gap. Pure Stage 1 rollout improves the clean operating point and remains relatively stable under mild corruption, but degrades once the noise exceeds the Stage 1 error distribution. High cue perturbation flattens the curve yet suppresses peak accuracy, indicating that excessive prior noise weakens useful association signals. Ours, which mixes rollout with the synthetic modes in Table 2, attains the best accuracy across the full intensity range: rollout supplies realistic closed-loop residuals, while synthetic perturbations cover rarer failure modes that rollout alone does not.

Table 3: 3D grounding results. Best results are in bold; second best are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">ARKitScenes</td><td colspan="2">ScanNet</td><td colspan="2">ScanNet++</td><td colspan="2">Mean</td></tr><tr><td>F1@0.25</td><td> $\mathrm { A P _ { 3 D } }$ </td><td>F1@0.25</td><td> $\mathrm { A P _ { 3 D } }$ </td><td>F1@0.25</td><td> $\mathrm { A P _ { 3 D } }$ </td><td>F1@0.25</td><td> $\mathrm { A P _ { 3 D } }$ </td></tr><tr><td>Qwen3-VL-4B</td><td>14.3</td><td>8.4</td><td>19.2</td><td>9.3</td><td>14.5</td><td>8.1</td><td>16.0</td><td>8.6</td></tr><tr><td>Qwen3-VL-8B</td><td>19.4</td><td>11.2</td><td>21.2</td><td>11.2</td><td>14.4</td><td>12.4</td><td>18.3</td><td>11.6</td></tr><tr><td>Qwen3.5-4B</td><td>10.3</td><td>7.3</td><td>27.3</td><td>11.0</td><td>8.8</td><td>8.4</td><td>15.5</td><td>8.9</td></tr><tr><td>Qwen3.5-9B</td><td>14.8</td><td>9.9</td><td>22.2</td><td>11.6</td><td>11.9</td><td>9.2</td><td>16.3</td><td>10.2</td></tr><tr><td>VG-LLM</td><td>27.9</td><td>11.5</td><td>25.3</td><td>18.3</td><td>22.7</td><td>13.1</td><td>25.3</td><td>14.3</td></tr><tr><td>DetAny3D</td><td>21.0</td><td>14.4</td><td>51.3</td><td>30.6</td><td>31.3</td><td>18.5</td><td>34.5</td><td>21.2</td></tr><tr><td>OVMono3D</td><td>46.3</td><td>24.7</td><td>52.9</td><td>29.8</td><td>46.2</td><td>29.4</td><td>48.5</td><td>28.0</td></tr><tr><td>TempoGround-4B (ours)</td><td>47.6</td><td>25.5</td><td>56.1</td><td>38.6</td><td>40.7</td><td>27.5</td><td>48.1</td><td>30.5</td></tr><tr><td>TempoGround-9B (ours)</td><td>49.5</td><td>30.8</td><td>63.0</td><td>45.3</td><td>51.7</td><td>30.4</td><td>54.7</td><td>35.5</td></tr></table>

Table 4: 2D grounding results. Best results are in bold; second best are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">ASE</td><td colspan="2">ARKitScenes</td><td colspan="2">ScanNet++</td><td colspan="2">Mean</td></tr><tr><td>F1@0.5</td><td>F1@0.95</td><td>F1@0.5</td><td>F1@0.95</td><td>F1@0.5</td><td>F1@0.95</td><td>F1@0.5</td><td>F1@0.95</td></tr><tr><td>Qwen3-VL-4B</td><td>36.8</td><td>5.1</td><td>36.0</td><td>11.9</td><td>54.1</td><td>19.2</td><td>42.3</td><td>12.1</td></tr><tr><td>Qwen3-VL-8B</td><td>38.0</td><td>2.4</td><td>34.4</td><td>11.1</td><td>54.2</td><td>17.7</td><td>42.2</td><td>10.4</td></tr><tr><td>Qwen3.5-4B</td><td>16.4</td><td>1.5</td><td>31.0</td><td>6.3</td><td>52.9</td><td>13.5</td><td>33.4</td><td>7.1</td></tr><tr><td>Qwen3.5-9B</td><td>12.1</td><td>1.7</td><td>33.7</td><td>7.1</td><td>56.4</td><td>12.7</td><td>34.1</td><td>7.2</td></tr><tr><td>Cosmos-Reason2-8B</td><td>12.4</td><td>2.1</td><td>33.7</td><td>6.3</td><td>43.3</td><td>7.6</td><td>29.8</td><td>5.3</td></tr><tr><td>GroundingDINO-T</td><td>43.4</td><td>20.1</td><td>45.7</td><td>13.8</td><td>74.5</td><td>40.5</td><td>54.5</td><td>24.8</td></tr><tr><td>GLaMM</td><td>13.5</td><td>2.1</td><td>15.9</td><td>4.0</td><td>35.5</td><td>10.2</td><td>21.6</td><td>5.4</td></tr><tr><td>RexOmni-3B</td><td>43.3</td><td>19.4</td><td>42.9</td><td>17.6</td><td>65.8</td><td>38.3</td><td>50.7</td><td>25.1</td></tr><tr><td>LocateAnything</td><td>35.1</td><td>21.7</td><td>52.1</td><td>18.7</td><td>72.3</td><td>37.3</td><td>53.2</td><td>25.9</td></tr><tr><td>TempoGround-4B (ours)</td><td>38.2</td><td>19.4</td><td>49.8</td><td>21.0</td><td>62.3</td><td>30.5</td><td>50.1</td><td>23.6</td></tr><tr><td>TempoGround-9B (ours)</td><td>45.5</td><td>18.6</td><td>51.9</td><td>21.2</td><td>79.4</td><td>39.3</td><td>58.9</td><td>26.4</td></tr></table>

Table 5: Ablation of curriculum prediction.
<table><tr><td>Configuration</td><td> $\mathrm { A P 3 D }$ </td></tr><tr><td>w/o correspondence prior</td><td>37.7</td></tr><tr><td>2D+3D prior</td><td>41.8</td></tr><tr><td>w/o presence-state prediction</td><td>40.5</td></tr><tr><td>Ours</td><td>42.3</td></tr></table>

Table 6: Ablation of SGR rewards.
<table><tr><td>Configuration</td><td> $\mathrm { A P _ { 3 D } }$ </td></tr><tr><td>w/o SGR</td><td>42.3</td></tr><tr><td>w/o Grounding reward  $R _ { \mathrm { { g } } }$ </td><td>42.7</td></tr><tr><td>w/o Identity reward  $R _ { \mathrm { i d } }$ </td><td>44.1</td></tr><tr><td>w/o Consistency reward  $R _ { \mathrm { c o n } }$ </td><td>43.9</td></tr><tr><td>Ours</td><td>45.3</td></tr></table>

## 5 Conclusion

We presented TempoGround, a VLM-native framework for accurate and consistent visual grounding under causally arriving image streams. TempoGround detects cross-frame object correspondence, explicitly models object presence states, and follows a curriculum prediction chain from 2D association and presence estimation to camera-frame 3D grounding. We further introduced Streaming Grounding Reinforcement (SGR), which optimizes TempoGround with verifiable Grounding, Identity, and Consistency rewards beyond tokenlevel imitation. Extensive experiments demonstrate that TempoGround provides a practical foundation for streaming visual grounding.

![](images/19808c27d1612e907ae99211b8281367a074ae31f1c216cb75c623e1cb2ccc32.jpg)  
Figure 4: Closed-loop robustness to inference-time cue corruption on ScanNet $( \mathrm { \bar { A } P _ { 3 D } ) }$ .

## References

Ahmadyan, A.; Zhang, L.; Ablavatski, A.; Wei, J.; and Grundmann, M. 2021. Objectron: A large scale dataset of object-centric videos in the wild with pose annotations. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 7822–7831.

Avetisyan, A.; Xie, C.; Howard-Jenkins, H.; Yang, T.-Y.; Aroudj, S.; Patra, S.; Zhang, F.; Frost, D.; Holland, L.; Orme, C.; et al. 2024. Scenescript: Reconstructing scenes with an autoregressive structured language model. In European Conference on Computer Vision, 247–263. Springer.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Baruch, G.; Chen, Z.; Dehghan, A.; Dimry, T.; Feigin, Y.; Fu, P.; Gebauer, T.; Jofe, B.; Kurz, D.; Schwartz, A.; et al. 2021. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897.

Caesar, H.; Bankiti, V.; Lang, A. H.; Vora, S.; Liong, V. E.; Xu, Q.; Krishnan, A.; Pan, Y.; Baldan, G.; and Beijbom, O. 2020. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 11621–11631.

Chen, D. Z.; Chang, A. X.; and Nießner, M. 2020. Scanrefer: 3d object localization in rgb-d scans using natural language. In European conference on computer vision, 202–221. Springer.

Chen, J.; Wei, F.; Zhao, J.; Song, S.; Wu, B.; Peng, Z.; Chan, S.-H. G.; and Zhang, H. 2025. Revisiting referring expression comprehension evaluation in the era of large multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 513–524.

Chen, K.; Zhang, Z.; Zeng, W.; Zhang, R.; Zhu, F.; and Zhao, R. 2023. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195.

Dai, A.; Chang, A. X.; Savva, M.; Halber, M.; Funkhouser, T.; and Nießner, M. 2017. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, 5828–5839.

Fan, Z.; Zhang, J.; Li, R.; Zhang, J.; Chen, R.; Hu, H.; Wang, K.; Qu, H.; Zhou, S.; Wang, D.; et al. 2025. Vlm-3r: Visionlanguage models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279.

Geiger, A.; Lenz, P.; and Urtasun, R. 2012. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition, 3354–3361. IEEE.

Gupta, A.; Dollar, P.; and Girshick, R. 2019. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5356–5364.

Hong, Y.; Zhen, H.; Chen, P.; Zheng, S.; Du, Y.; Chen, Z.; and Gan, C. 2023. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36: 20482–20494.

Huang, H.; Chen, Y.; Wang, Z.; Huang, R.; Xu, R.; Wang, T.; Liu, L.; Cheng, X.; Zhao, Y.; Pang, J.; et al. 2024. Chatscene: Bridging 3d scene and large language models with object identifiers. Advances in Neural Information Processing Systems, 37: 113991–114017.

Huang, J.; Yong, S.; Ma, X.; Linghu, X.; Li, P.; Wang, Y.; Li, Q.; Zhu, S.-C.; Jia, B.; and Huang, S. 2023. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871.

Jiang, Q.; Huo, J.; Chen, X.; Xiong, Y.; Zeng, Z.; Chen, Y.; Ren, T.; Yu, J.; and Zhang, L. 2025. Detect anything via next point prediction. arXiv preprint arXiv:2510.12798.

Kazemzadeh, S.; Ordonez, V.; Matten, M.; and Berg, T. 2014. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings ofthe 2014 conference on empirical methods in natural languageprocessing (EMNLP), 787–798.

Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Dollár, P.; and Zitnick, C. L. 2014. Microsoft coco: Common objects in context. In European conference on computer vision, 740–755. Springer.

Linghu, X.; Huang, J.; Jia, B.; and Huang, S. 2026. 3D-RFT: Reinforcement Fine-Tuning for Video-based 3D Scene Understanding. arXiv preprint arXiv:2603.04976.

Liu, S.; Zeng, Z.; Ren, T.; Li, F.; Zhang, H.; Yang, J.; Jiang, Q.; Li, C.; Yang, J.; Su, H.; et al. 2024. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, 38– 55. Springer.

Man, Y.; Wang, S.; Zhang, G.; Bjorck, J.; Gui, L.-Y.; Fan, J.; Kautz, J.; Wang, Y.-X.; and Yu, Z. 2026. Locateanything3d: Vision-language 3d detection with chain-of-sight. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 31089–31102.

Pan, X.; Charron, N.; Yang, Y.; Peters, S.; Whelan, T.; Kong, C.; Parkhi, O.; Newcombe, R.; and Ren, Y. C. 2023. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 20133–20143.

Peng, Z.; Wang, W.; Dong, L.; Hao, Y.; Huang, S.; Ma, S.; Ye, Q.; and Wei, F. 2024. Grounding multimodal large language models to the world. In International Conference on Learning Representations, volume 2024, 51575–51598.

Plummer, B. A.; Wang, L.; Cervantes, C. M.; Caicedo, J. C.; Hockenmaier, J.; and Lazebnik, S. 2015. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, 2641–2649.

Ramanathan, V.; Kalia, A.; Petrovic, V.; Wen, Y.; Zheng, B.; Guo, B.; Wang, R.; Marquez, A.; Kovvuri, R.; Kadian, A.; et al. 2023. Paco: Parts and attributes of common objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7141–7151.

Rasheed, H.; Maaz, M.; Shaji, S.; Shaker, A.; Khan, S.; Cholakkal, H.; Anwer, R. M.; Xing, E.; Yang, M.-H.; and Khan, F. S. 2024. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13009–13018.

Roberts, M.; Ramapuram, J.; Ranjan, A.; Kumar, A.; Bautista, M. A.; Paczan, N.; Webb, R.; and Susskind, J. M. 2021. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, 10912–10922.

Shao, S.; Li, Z.; Zhang, T.; Peng, C.; Yu, G.; Zhang, X.; Li, J.; and Sun, J. 2019. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings ofthe IEEE/CVF international conference on computer vision, 8430–8439.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Song, S.; Lichtenberg, S. P.; and Xiao, J. 2015. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings of the IEEE conference on computer vision and pattern recognition, 567–576.

Wang, J.; Zhang, P.; Chu, T.; Cao, Y.; Zhou, Y.; Wu, T.; Wang, B.; He, C.; and Lin, D. 2023. V3det: Vast vocabulary visual detection dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19844–19854.

Wang, S.; Liu, S.; Kuang, Y.; Wei, X.; Liu, Y.; Li, Z.; Man, Y.; Chen, G.; Tao, A.; Liu, G.; et al. 2026. LocateAnything: Fast and high-quality vision-language grounding with parallel box decoding. arXiv preprint arXiv:2605.27365.

Wang, T.; Mao, X.; Zhu, C.; Xu, R.; Lyu, R.; Li, P.; Chen, X.; Zhang, W.; Chen, K.; Xue, T.; et al. 2024. Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition, 19757–19767.

Yeshwanth, C.; Liu, Y.-C.; Nießner, M.; and Dai, A. 2023. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 12–22.

Yu, F.; Chen, H.; Wang, X.; Xian, W.; Chen, Y.; Liu, F.; Madhavan, V.; and Darrell, T. 2020. Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2636–2645.

Yu, L.; Poirson, P.; Yang, S.; Berg, A. C.; and Berg, T. L. 2016. Modeling context in referring expressions. In European conference on computer vision, 69–85. Springer.

Zhao, Y.; Huang, J.; Hu, J.; Wang, X.; Mao, Y.; Zhang, D.; Jiang, Z.; Wu, Z.; Ai, B.; Wang, A.; et al. 2025. Swift: a scalable lightweight infrastructure for fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 29733–29735.

Zheng, D.; Huang, S.; and Wang, L. 2025. Video-3d llm: Learning position-aware video representation for 3d scene understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8995–9006.

Zheng, D.; Li, Y.; Wang, L.; et al. 2026. Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. Advances in neural information processing systems, 38: 20560–20586.

Zhu, C.; Wang, T.; Zhang, W.; Pang, J.; and Liu, X. 2024. Llava-3d: A simple yet efective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125.

Zhu, C.; Xiao, F.; Alvarado, A.; Babaei, Y.; Hu, J.; El-Mohri, H.; Culatana, S.; Sumbaly, R.; and Yan, Z. 2023. Egoobjects: A large-scale egocentric dataset for fine-grained object understanding. In Proceedings ofthe IEEE/CVF international conference on computer vision, 20110–20120.

# Supplementary Material

This supplementary material provides details omitted from the main paper. Appendix A specifies the output schema, the computation of localization metrics, and runtime measurements. Appendix B gives the Stage 3 reward formulas used by Streaming Grounding Reinforcement. Appendix C describes data curation: quality control, referring annotation, correspondence-cue perturbation, and prompt templates. Appendix D shows additional qualitative results.

## A Implementation Details

## A.1 Output Schema

TempoGround introduces special tokens that delimit the curriculum prediction fields. The token inventory is

$$
\begin{array} { r l r } & { } & { < \mathrm { r e } \mathrm { f } > / < / \mathrm { r e } \mathrm { f } > , < \mathrm { s t a t e } > / < / \mathrm { s t a t e } > , } \\ & { } & { < 2 \mathrm { d } \mathrm { \_ b b o x } > / < / 2 \mathrm { d } \mathrm { \_ b b o x } > , < 3 \mathrm { d } \mathrm { \_ b b o x } > / < / 3 \mathrm { d } \mathrm { \_ b b o x } > , } \\ & { } & { < \mathrm { n u } 1 1 > . } \end{array}
$$

A visible instance with state new or track is serialized as

<ref>label</ref><state>state</state>   
<2d\_bbox>x<sub>1</sub>, y<sub>1</sub>, x<sub>2</sub>, y<sub>2</sub></2d\_bbox>   
<3d\_bbox>[c<sub>x</sub>, c<sub>y</sub>, c<sub>z</sub>] [s<sub>x</sub>, s<sub>y</sub>, s<sub>z</sub>]   
[r, p, y]</3d\_bbox>

where the 2D coordinates are integers in [0, 1000] and the 3D quantities follow the camera-frame convention of the main paper. An exited instance is serialized without boxes as

$$
< \tt r e f > \tt l a b e l < / \tt r e f > < \tt s t a t e > \tt l o s t < / \tt s t a t e > .
$$

When no query-matching object is visible, the model emits <null>. Multiple instances are concatenated in a canonical order. At inference, thinking-mode generation is disabled and the model is served with vLLM.

## A.2 Evaluation Protocol

We report $\mathrm { F 1 _ { 2 D } @ 0 . 5 , F 1 _ { 2 D } @ 0 . 9 5 , F 1 _ { 3 D } @ 0 . 2 5 }$ , and $\mathrm { A P _ { 3 D } }$ Under the TempoGround protocol, the previous-frame correspondence cue is constructed from the model’s own prediction at the preceding timestep.

Scored instances. Before scoring, instances with state lost are discarded from both the ground truth and the prediction. Frames that contain no remaining visible groundtruth box are excluded from F1 and $\mathrm { A P _ { 3 D } }$

Matching. On each scored frame, correspondence between predictions and ground-truth boxes is established by greedy matching on axis-aligned 2D IoU. All candidate pairs are sorted in decreasing order of 2D IoU, and a pair is accepted only when neither side has already been matched. The resulting one-to-one assignment is shared by 2D and 3D F1: correspondence is determined by 2D overlap, after which oriented 3D IoU is evaluated on the matched pairs.

F1 score. Let $N _ { t } ^ { \mathrm { p r e d } }$ and $N _ { t } ^ { \mathrm { g t } }$ denote the numbers of predicted and visible ground-truth boxes on a scored frame t, and let $\mathrm { T P } _ { t } ( \delta )$ denote the number of matched pairs whose overlap is at least δ. For $\mathrm { F 1 _ { 2 D } @ } \delta$ the overlap is 2D IoU; for

F $\mathrm { 1 _ { 3 D } @ \delta }$ it is oriented 3D IoU on the same pairs. Precision and recall are

$$
P _ { t } ( \delta ) = \frac { \mathrm { T P } _ { t } ( \delta ) } { N _ { t } ^ { \mathrm { p r e d } } } , \qquad R _ { t } ( \delta ) = \frac { \mathrm { T P } _ { t } ( \delta ) } { N _ { t } ^ { \mathrm { g t } } } ,\tag{8}
$$

with the convention that a zero denominator yields zero. The per-frame score is

$$
\mathrm { F } 1 _ { t } ( \delta ) = \frac { 2 P _ { t } ( \delta ) R _ { t } ( \delta ) } { P _ { t } ( \delta ) + R _ { t } ( \delta ) } ,\tag{9}
$$

or 0 when $P _ { t } ( \delta ) + R _ { t } ( \delta ) = 0$ . Within each dataset we take the mean of F1 over scored frames; the reported number is then the unweighted mean over datasets.

Average precision. $\mathrm { A P _ { 3 D } }$ pools predictions and groundtruth boxes from all scored frames in a dataset. At each oriented 3D IoU threshold $\delta \in \Theta = \{ 0 . 1 5 , 0 . 2 5 , 0 . 5 0 \}$ , predictions are matched greedily to unused ground-truth boxes, and AP is computed with the Pascal VOC 11-point interpolation. Detection averages AP over categories; referring uses a single referred target. The dataset-level score is

$$
\mathrm { A P } _ { \mathrm { 3 D } } = \frac { 1 } { | \Theta | } \sum _ { \delta \in \Theta } \mathrm { A P } ( \delta ) ,\tag{10}
$$

and the reported number is the unweighted mean over datasets.

## A.3 Runtime

As shown in Table A.1, the average single-image inference latency of TempoGround on one NVIDIA H200 GPU is 0.65 s for the 9B model and 0.45 s for the 4B model, each averaged over 10 runs. The reported time covers the full forward path for one observation, including vision encoding of the input image, prefilling, and autoregressive generation of the grounding outputs. These results show that TempoGround enables a general-purpose VLM to perform continuous object grounding at sub-second speed, providing a practical foundation for general embodied perception and downstream tasks such as spatial reasoning and question answering.

<table><tr><td>Model</td><td>Mean Latency (s)</td></tr><tr><td>TempoGround-9B</td><td>0.65</td></tr><tr><td>TempoGround-4B</td><td>0.45</td></tr></table>

Table A.1: Average inference latency on one NVIDIA H200 GPU over 10 runs.

## B Streaming Grounding Reinforcement

The main paper defines the Stage 3 objective and the roles of $R _ { \mathrm { g } } , R _ { \mathrm { i d } }$ , and $R _ { \mathrm { c o n } }$ . This appendix specifies how each term is computed.

Shared geometric quality. After pairing a visible groundtruth instance with a prediction, we define

$$
q = 0 . 3 5 \mathrm { I o U _ { 2 D } } + 0 . 6 5 \mathrm { I o U _ { 3 D } } .\tag{11}
$$

If the instance has no match, $q { = } 0$ . This scalar is reused by $R _ { \mathrm { g } }$ and $R _ { \mathrm { i d } }$

## B.1 Grounding Reward $R _ { \mathrm { g } }$

$R _ { \mathrm { g } }$ measures trajectory-level localization quality while penalizing unmatched predictions. It is not a single global average of all per-frame q values. A global average would let long tracks dominate short ones and would ignore spurious boxes that never match any ground-truth instance. The implementation therefore separates localization credit from false-positive cost.

For each ground-truth track $i ,$ let $\mathcal { T } _ { i }$ be the set of frames on which its annotated presence is new or track. The track score is the mean quality on those frames,

$$
r _ { i } = \frac { 1 } { | T _ { i } | } \sum _ { t \in \mathcal { T } _ { i } } q _ { i , t } .\tag{12}
$$

Equal weight is then given to every track,

$$
R _ { \mathrm { t r a c k } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } r _ { i } ,\tag{13}
$$

where M is the number of tracks. Frames annotated as lost are excluded from $r _ { i }$ because no box is required there; presence errors on those frames are handled by $R _ { \mathrm { i d } }$

$R _ { \mathrm { t r a c k } }$ alone is insuficient: a policy can keep $R _ { \mathrm { t r a c k } }$ high while emitting many unmatched boxes. Let $N _ { \mathrm { f p } }$ be the number of unmatched predictions over the trajectory and $N _ { \mathrm { G T } }$ the number of ground-truth instances over the trajectory. The grounding reward subtracts a normalized false-positive penalty and clips to [0, 1]:

$$
R _ { \mathrm { g } } = \mathrm { c l i p } \bigg ( R _ { \mathrm { t r a c k } } - 0 . 3 \frac { N _ { \mathrm { f p } } } { \mathrm { m a x } ( 1 , N _ { \mathrm { G T } } ) } , 0 , 1 \bigg ) .\tag{14}
$$

Normalization by $N _ { \mathrm { G T } }$ keeps the penalty comparable across trajectories of diferent length and object count.

## B.2 Identity Reward $R _ { \mathrm { i d } }$

$R _ { \mathrm { i d } }$ is the mean of per-instance event scores over the trajectory, clipped to [−1, 1]. Each ground-truth instance on each frame contributes one event score according to its annotated presence:

• new: matched new gives +q; matched track gives −0.5; a miss or any other predicted state gives −1.

• track: matched track gives 0; a miss or predicted lost gives −α, where $\alpha \in [ 0 , 1 ]$ is the visible fraction of the object’s 2D box after clipping to the image boundary; any other incorrect predicted state gives −0.3.

• lost: a matched non-lost prediction gives −1; otherwise the event gives 0.

## B.3 Consistency Reward $R _ { \mathrm { c o n } }$

The correspondence cue at timestep $t { + } 1$ is constructed from the prediction at timestep t. A localization error is therefore not confined to the current frame: it can corrupt later association and produce a contiguous run of unstable up dates. Adjacent-frame flicker is a visible symptom of this process, but penalizing diferences between consecutive predictions is ill-posed under ego-motion, where boxes are expected to change. $R _ { \mathrm { c o n } }$ instead checks each frame against ground truth and penalizes failures that form long contiguous chains, namely the self-conditioned streaks that cue feedback amplifies.

Frame failure. After 2D greedy matching of visible ground truth, a frame is marked as a localization failure if either (i) at least one prediction remains unmatched, or (ii) the mean quality $\bar { q }$ over visible ground-truth instances falls below 0.25. A frame with no visible ground truth and no prediction is not a failure.

Phantom after exit. A frame is marked as a phantom frame when the ground truth contains at least one lost instance and unmatched predictions remain after the still-visible targets have been matched. If all query objects have left and predictions remain nonempty, the frame is also marked as a phantom frame. This criterion targets continued emission after exit without penalizing boxes on objects that remain visible.

Chain penalty. Let $\{ m _ { t } \} _ { t = 1 } ^ { T }$ and $\{ p _ { t } \} _ { t = 1 } ^ { T }$ be the binary failure and phantom indicators along the trajectory. For a binary sequence, let $\ell _ { 1 } , \ell _ { 2 } , \ldots$ . be the lengths of its maximal contiguous runs of ones. The chain cost is

$$
\mathrm { c h a i n } ( \{ z _ { t } \} ) = \frac { 1 } { T ^ { 2 } } \sum _ { k } \ell _ { k } ^ { 2 } .\tag{15}
$$

An isolated error contributes little; a long unbroken run grows with the square of its length. The consistency reward is

$$
R _ { \mathrm { c o n } } = - \frac { 1 } { 2 } \Big ( \mathrm { c h a i n } ( \{ m _ { t } \} ) + \mathrm { c h a i n } ( \{ p _ { t } \} ) \Big ) ,\tag{16}
$$

clipped to [−1, 0]. IoU decides whether the current step is wrong; $R _ { \mathrm { c o n } }$ decides whether those wrongs have turned into a self-reinforced streak under cue feedback.

In Stage 3 training, $R _ { \mathrm { g } } , R _ { \mathrm { i d } }$ , and $R _ { \mathrm { c o n } }$ are mixed with weights 0.70, 0.15, and 0.15. Let $N _ { \mathrm { e m p t y } }$ be the number of frames whose prediction text is empty while the ground truth is nonempty, and define

$$
f _ { \mathrm { f m t } } = \frac { N _ { \mathrm { e m p t y } } } { T } .\tag{17}
$$

The scalar trajectory reward subtracts $0 . 0 2 f _ { \mathrm { f m t } }$ . Empty frames still enter the geometric terms with no parsed boxes; the format term is an additional penalty on the empty-output rate.

## C Data Curation Details

## C.1 Quality Control

Beyond the high-level frustum and visibility filters in the main paper, the streaming builders use the following two geometric gates, transition rules, and sequence acceptance criteria.

In-view versus present. Let $A _ { \mathrm { a m o d a l } }$ be the area of the projected 2D box before image clipping, $A _ { \mathrm { c l i p } }$ the area after clipping to the image, and $A _ { \mathrm { i m g } }$ the image area. An instance is in view if its camera-frame depth is positive $( z > 0 . 1 )$ $A _ { \mathrm { c l i p } } \geq 6 4$ pixels, and the clipped fraction $A _ { \mathrm { c l i p } } / A _ { \mathrm { a m o d a l } }$ is at least 0.05. This gate only asks whether the projection still intersects the camera frustum with nontrivial mass.

Positive localization supervision further requires the stricterpresent gate. Present demands $A _ { \mathrm { c l i p } } / A _ { \mathrm { a m o d a l } } \geq 0 . 3 0$ and $A _ { \mathrm { c l i p } } / A _ { \mathrm { i m g } } \geq 2 \times 1 0 ^ { - 4 }$ . When pixel-level visibility is available, the visible-pixel fraction inside the box must be at least 0.15; if the clipped box covers at least 25% of the image, the ratio of visible pixels to box-image fraction must also be at least 0.20, so a large empty amodal box is rejected. When only mesh-extent visibility is available, we require extent fraction at least 0.35 (with a weak band down to 0.30). Boxes that are in view but fail present may still participate in enter/exit bookkeeping; they do not contribute positive 2D/3D regression targets.

Presence-state transitions. States are assigned from the in-view sequence, not from raw preprocessing files. For detection, an object is supervisable when it is in view with a box; referring adds a dependency check described in Appendix C.2. An object becomes new when it first becomes supervisable in a track, or when it re-enters after exit. While it remains in view after that enter event it is track. The lost label is emitted on exactly one frame: the transition from inview to out-of-view. That frame carries no 2D or 3D box and is never written into the next-step correspondence cue. Subsequent out-of-view frames are empty responses rather than repeated lost tokens with a stale box. Re-entry again uses new. Automated track assertions reject lost streaks longer than one frame, boxes on lost or empty frames, and illegal new/track transitions.

Sequence acceptance and source hygiene. A retained streaming window must contain an enter event and either an exit event or a leading empty prefix, so that the clip is not a static always-centered fragment. Along the window of length T, the number of present frames must be at least max(6, ⌊0.4T⌋), and the best present frame must reach visible-pixel fraction 0.20 (or extent 0.40 when pixels are unavailable). Training windows use lengths in [24, 64] and test windows in [32, 64], with dataset-dependent base strides, multiplicative jitter on ScanNet-family sources, and a smal integer ofset so neighboring clips are not identical. Windows that prefer enter/persist/exit transitions are favored over nearstatic runs. Category labels are canonicalized across sources; large structural or geometrically ill-posed classes (for example wall, floor, and ceiling) are removed or heavily downweighted. For ScanNet++, frames whose color field is nearly flat are rejected as appearance-unstable. Calibrated negatives are retained so that a query with no matching in-view object supervises abstention.

## C.2 Referring Expression Annotation

Public scene-level expressions are discarded as primary supervision, as stated in the main paper. We detail the annotation inputs, frame-selection thresholds, programmatic gates, and uniqueness checks used when regenerating viewgrounded queries.

Index-only use ofpublic refer corpora. ScanRefer, Nr3D, and related releases contribute only object indices for the referring domain. All pixels and camera-frame boxes come from the underlying RGB-D scans (for example ScanNet), not from the scene-level expression text or map-centric wording in those corpora.

Annotation-frame selection. A frame is eligible for writing an expression only if the target is suficiently complete and the view retains surrounding context. Concretely, the clipped-to-amodal completeness must be at least 0.90, the visible-pixel fraction at least 0.35 when available, and the target box must cover less than 40% of the image while leaving at least 25% of the image as non-target context. At least one co-visible neighbor that itself meets milder completeness thresholds must remain in the allowlist; otherwise the frame is discarded because relational language has no verified anchor. When multiple views of the same object are kept, each view is represented by a layout signature that quantizes the target box onto a 10 × 10 grid and records the coarse image regions of co-visible neighbors. Signatures that difer by at most one grid cell with an identical neighbor set are treated as near-duplicates and are not both sent to the annotator.

Proposal and programmatic gates. On the selected primary frame, the target is rendered with a salient green box; neighbors appear only as a text allowlist that lists oficial category names, object indices, and coarse image regions. A closed-source vision-language model (Qwen3.7-Plus) proposes three short candidate expressions per target, preferring intrinsic attributes and using allowlisted neighbors only when attributes cannot separate same-category distractors. Candidates are discarded if they mention doorway- or entrancecentric viewpoint language, cardinal directions, hedged uncertainty phrases, or any dependency outside the allowlist. When more than one same-category instance is co-visible, left/right or foreground/background alone is also forbidden as the sole discriminator, and at least one allowlisted neighbor dependency is required.

Uniqueness verification. If several same-category instances remain co-visible, two visual checks are applied before the expression is accepted. In the candidate-index check, every same-category box is numbered on the image and the model must return the target index given only the candidate sentence. In the contrastive check, the target is marked in green and one distractor in red; the model must select the green box for that sentence. Failure of either check discards the paraphrase. Accepted expressions are attached to the object track for streaming packaging.

Streaming resolve-then-track. Along a referring track, a frame is labeled new only when the target is in view and every linguistic dependency in the expression is co-visible (the query is uniquely decidable from the observed view). After that enter event, subsequent track labels follow in-view geometry under cross-frame correspondence even if some anchors leave the frame. Exit uses the same one-frame boxfree lost transition as detection; re-entry again requires the dependencies to be co-visible. Frames that are geometrically visible but not yet decidable remain empty rather than receiving an under-specified new.

## C.3 Correspondence-Cue Perturbation

Stage 2 constructs the correspondence cue $\mathbf { c } _ { t - 1 }$ while keeping the supervised target at timestep t unchanged. The cue contains only instance labels and previous-frame 2D boxes; camera-frame 3D is never carried forward. For each streaming sample a mode is sampled according to the mixture in the main paper, and the selected operator is applied to the previous-frame ground-truth cue (or replaces it entirely). We detail the operators below.

Clean. $\mathbf { c } _ { t - 1 }$ is the set of ground-truth labels and 2D boxes on frame t−1.

Stage 1 rollout. A frozen Stage 1 checkpoint is run on frame t−1 under the same query. Its predicted labels and 2D boxes replace the ground-truth cue, so $\mathbf { c } _ { t - 1 }$ follows the closed-loop residual distribution of the Stage 1 policy rather than an oracle prior.

Jitter. Starting from the clean previous-frame cue, each 2D box is translated independently. For every instance a magnitude m is drawn uniformly from the integer range [5, 30] in the normalized coordinate scale [0, 1000], after which horizontal and vertical ofsets are sampled uniformly from [−m, m] and mapped to pixels by the image width and height. All four corners of that box receive the same ofset, so shape is preserved and only the association location is displaced.

Dropout. Starting from the clean cue, a random nonempty subset of prior instances is removed. If the clean cue contains N instances, the number ofremovals is drawn uniformly from 1 to max $( 1 , \lfloor N / 2 \rfloor )$ , after which the remaining instances are kept as $\mathbf { c } _ { t - 1 }$

Phantom. One or two spurious instances are appended to the clean cue. Each phantom copies an existing prior box when available, perturbs every coordinate by an independent ofset up to 20% of the image width or height, and replaces the label by a category sampled from the dataset vocabulary when that vocabulary is available. If the clean cue is empty, phantoms are instead sampled as random boxes in the image with vocabulary labels.

Stale. Instead of the cue from frame $t - 1 , \mathbf { c } _ { t - 1 }$ is taken from an older frame in the same stream, either t−2 or t−3 when available. The older cue is otherwise left uncorrupted, so the association signal is temporally outdated rather than spatially jittered.

Empty. $\mathbf { c } _ { t - 1 }$ is set to the empty set and rendered as a null prior in the prompt. This matches a stream restart: the current frame must be resolved without cross-frame correspondence.

## C.4 Prompt Templates

Streaming prompts follow the curriculum prediction interface: presence, then 2D, then camera-frame 3D. First-frame prompts omit the previous image and cue; streaming prompts include both. Detection prompts request every instance of a category; referring prompts request a single expression. Each template family has a small set of paraphrases selected deterministically from the sample key, so surface wording varies without changing the output schema.

A first-frame detection prompt is

Current frame: <image>   
Detect every “category” in the current   
frame. For each visible instance,   
output a 2D box then a 3D bounding box   
in the camera frame.

A streaming detection prompt is

Previous frame: <image>   
Current frame: <image>   
Previous-frame detections for   
“category” (label with 2D box): prior   
Using the previous frame as reference,   
detect every “category” in the current   
frame. For each object, report its   
status (newly appeared, still tracked,   
or gone) and output a 2D box then a 3D   
box in the camera frame.

Referring prompts use the same first-frame and streaming split, with the category query replaced by the regenerated expression and a single referred target. Prompted status terms are aligned with the supervised presence vocabulary, and geometric fields follow Appendix A.1.

## D Additional Qualitative Results

We show additional qualitative visualizations for streaming detection (Figures D.1 and D.2) and referring (Figures D.3 and D.4).

![](images/c69b48820132686f2fbb82ecd2890789d6cd2e0b2dd4add3a8d779a57c6b8d2f.jpg)  
Figure D.1: While the bed remains in the field of view, TempoGround maintains correct 2D/3D tracking.

![](images/465ab581a0b23d2a71748d11545b902573aef8c7be5ec4bfb8f68aab5af8c394.jpg)  
Figure D.2: The scene contains both a large and a small trash can; TempoGround accurately detects the two instances.

![](images/35ea359fcfa70dcafe8f9c50763882ff0ccbd403d7fb946cb156a0730bce7fc7.jpg)  
Figure D.3: When the frame above the sofa enters the view, TempoGround tracks it accurately in 2D and 3D.

![](images/fa84f7723534b3f31ccc2e01fc6aefd6e49a2de0786176e65f2199c1990be927.jpg)  
Figure D.4: TempoGround interprets the query “loveseat” and correctly identifies the two-seat couch in the view, avoiding false detections of single-seat or three-seat ones.