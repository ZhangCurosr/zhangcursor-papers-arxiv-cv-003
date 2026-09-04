# Unfold The World: Factorize 4D Properties in Reinforcing Spatial Reasoning

Yijun Yang<sup>1,2,∗</sup>, Shenghe Zheng<sup>3,∗</sup>, Wenbo Li<sup>2,†</sup>, Jianhui Liu<sup>4</sup>, Haoze Sun<sup>2</sup>, Yanbing Zhang<sup>2</sup>, Jiaxiu Jiang<sup>2</sup>, Lin Song<sup>2</sup>, Haoyang Huang<sup>2</sup>, Nan Duan<sup>2</sup>, and Lei Zhu<sup>1,B</sup>

<sup>1</sup> The Hong Kong University of Science and Technology (Guangzhou) <sup>2</sup> Joy Future Academy <sup>3</sup> The Hong Kong University of Science and Technology <sup>4</sup> The University of Hong Kong

![](images/5a8172f88690dfeb4063fd2a8248eafebabe057468df028a21d6d32359e1fd53.jpg)  
Fig. 1: Motivation. (1) Single-view learning contradicts how the physical world is structured, as real-world perception is inherently multi-view, depth-aware, and temporally continuous. (2) Scaling SFT or adding spatial tokens does not yield a coherent latent world model, while directly optimizing a 4D objective over space and time is computationally and algorithmically intractable.

Abstract. Despite the remarkable prowess of Vision-Language Models (VLMs) in general multimodal tasks, they remain fundamentally “flat” when reasoning about the physical world. We argue that this spatial bottleneck stems from a profound dimensional mismatch: while VLMs are trained to interpret 2D projections, true spatial reasoning demands the recovery of latent 3D geometry and temporal continuity. To conquer this high-dimensional complexity, we advocate a shift from monolithic learning to a “divide and conquer” paradigm. We present FactoSR, a factorized reinforcement learning framework that explicitly interpret the

dimensions collapsed by visual projection. At its core, FactoSR decomposes the monolithic problem of world-consistent reasoning into three orthogonal, geometric sub-objectives: planar correspondence (XY), depth consistency (Z), and temporal reversibility (T). By optimizing these verifiable constraints within a unified policy learning mechanism, we efectively transform an ill-posed projection recovery problem into a series of tangible reasoning steps. Extensive evaluations on multi-view and video benchmarks demonstrate that this elegant decomposition yields substantial gains in 3D and 4D reasoning, achieving a 5.9% boost on VSI-Bench and 4.5% on All-Angles-Bench. Our findings suggest that reinforcing explicit, factorized 4D consistency is a critical step toward evolving VLMs into robust, world-aware reasoners.

Code: https://github.com/ZimaBlue-WAM/FactoSR

Keywords: Spatial Intelligence · Reinforcement Learning · VLMs

## 1 Introduction

Vision-Language Models (VLMs) [1, 2, 14, 16, 29, 40, 44] have achieved remarkable success in general visual tasks, yet a critical capability remains elusive, i.e., spatial reasoning. This spatial reasoning ability is a fundamental component for VLMs in approaching real-world artificial general intelligence. While humans efortlessly identify spatial relationships in sequential visual environments, current VLMs struggle with even basic spatial queries [30, 41, 51], which requires interpreting the dynamic world beyond 2D projections. This limitation severely constrains their deployment in applications requiring dynamic spatial intelligence, from autonomous driving, robotics navigation to world models.

To mitigate this issue, many studies have synthesized massive spatial questionanswering datasets for supervised fine-tuning (SFT) [9, 30, 52] or explored the incorporation of additional spatial tokens [7, 19, 47]. These approaches are often constrained by 3D explicit data hunger, and have poor transferable capability to 4D scenes. Even more critically, while they sample images from 4D scenes [4,10,17,57], they heavily rely on single-view-based question answer. Their static learning on 2D patterns would induce hallucination in dynamic spatial reasoning, due to the absence of true 4D physical-world perception. Consequently, they tend to infer spatial relationships heuristically, often making premature decisions without properly accounting for geometric correspondence, depth estimation, or temporal consistency, as illustrated in Fig. 1.

Recently, a few pioneers [31, 32, 43] alternatively investigate applying reinforcement learning with verifiable rewards (RLVR) to enhance the spatial reasoning abilities of VLMs. RLVR has demonstrated superior generalization over SFT by learning diverse reasoning strategies rather than static patterns [21,26,36,55]. However, existing RLVR suites for spatial reasoning [5,23,35,43,52] employ simple rewards inherited from general understanding, which focus only on final correctness. They also learn in the single-view spirit, which squeezes the dynamics of the latent world by depth distortion and temporal drift. Instead, spatial intelligence requires explicit reasoning across dimensions of the physical world that are collapsed by the camera projection from reality to observations.

Unfortunately, formulating a unified 4D objective over both space and time is computationally and algorithmically intractable. To divide and conquer [6], we present FactoSR, a method beyond pixel-level understanding that factorizes spatial reasoning into plane, depth, and time through online policy reinforcement learning, with three parallel rewards that guide models toward explicit 4D reasoning. The training uses a multi-objective reward framework: format rewards ensure structured outputs; accuracy rewards prioritize correctness; XY rewards constrain re-projection consistency and correspondences to geometrically valid regions across views; Z rewards promote precise 3D localization and relative depth ordering; and T rewards enforce temporal cycle consistency and reversible camera-motion reasoning. Together with supervised spatial fine-tuning, our suite moves beyond image understanding toward models that reason across the dimensions collapsed by projection, supporting a process of observing, localizing, thinking, and answering.

Building on our constructed datasets, our contributions are threefold.

– We present a learning suite that injects spatial knowledge into VLMs. By cascading supervised fine-tuning (FactoSR-SFT) with reinforcement learning (FactoSR-RL), we establish a transition from initial spatial perception to explicit reasoning about the latent world.

We introduce a novel factorized reward framework for 4D Reasoning that decomposes the complicated task into three complementary objectives: XY-plane, Z-depth, and T-time. This design recovers the dimensions typically collapsed by 2D projection, guiding the model to reason precisely across depth and temporal sequences.

– We translate this design into significant spatial gains. Our paradigm achieves state-of-the-art performance on multiple spatial benchmarks, particularly an improvement of 5.9% on VSI-Bench and 4.5% on All-Angles-Bench, while preserving decent general multi-modal capabilities.

## 2 Related Work

Spatial Intelligence in MLLMs. Recent large vision language models show strong perceptual abilities, yet their spatial intelligence remains limited. Spatial intelligence involves understanding geometric structure, relative positions, and viewpoint transformations, which requires consistent reasoning over spatial configurations rather than surface level recognition. Existing eforts improve spatial intelligence from three aspects. Architectural approaches such as Spatial-MLLM [46], VLM-3R [19], and 3DThinker [13] introduce geometric biases or 3D representations, while SpatialBot [8] and VILASR [48] leverage external percep tion tools. Data scaling works including SpatialVLM [11], SpatialRGPT [15], VST [52], and SenseNova-SI [9] expand spatial supervision. Reasoning oriented frameworks such as SpatialLadder [23], Cambrian-S [53], SpaceR [32], and Mind-Cube [58] enhance structured spatial inference. However, these approaches do not provide both depth and temporal analysis of feasible learning pathways for acquiring spatial intelligence. To address this gap, we propose a reinforcement learning framework with decoupled rewards to explicitly strengthen spatial reasoning and guide the model toward more efective behaviors.

Multimodal Reinforcement Learning. We focus on enhancing the general reasoning abilities of multimodal models via reinforcement learning. Open-VLThinker [18] leverages iterative SFT–RL cycles to facilitate reasoning emergence, while VL-Rethinker [42] and SRPO [61] incorporate self-reflection into RL to promote slow thinking. Open Vision Reasoner [45] studies cognitive behavior transfer from LLMs, and NoisyGRPO [33] and EvolvedGRPO [37] improve generalization and stability through noise modeling and progressive training. Generative RLHF-V [62] further advances multimodal reward modeling. Overall, prior work moves beyond simple outcome rewards toward more structured and robust RL paradigms for multimodal reasoning.

Reinforcement learning for spatial intelligence is a subfield of multimodal RL, mainly focusing on how to stably improve spatial reasoning through verifiable rewards. SVQA-R1 [43] and SpatialThinker [5] incorporate spatial relations into RL objectives via single-view-consistent or dense spatial rewards. Visionary-R1 [50] and SATORI-R1 [35] show that free-form reasoning sufers from shortcut learning, and introduce intermediate verifiable stages such as captioning or region localization. Perception-R1 [59] highlights the importance of perception-oriented rewards. Meanwhile, Visual Spatial Tuning [52] and Spatial-Ladder [23] adopt progressive training from perception to reasoning, while SpatialReasoner [31] explores explicit 3D representations for better generalization. Overall, prior work suggests that long CoT reasoning or sparse rewards alone is insuficient, motivating our structured solution with explicit representations and multiple verifiable rewards.

## 3 Preliminaries

Problem Formulation. In this work, we study 4D spatial-temporal intelligence in Vision-Language Models (VLMs), which requires joint reasoning over 3D geometric structures and temporal dynamics. Let $D = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { N } \}$ denote a spatial-temporal reasoning dataset, where each sample $x _ { i } = ( V _ { 1 : t } , Q _ { s t } , a )$ consists of a sequence of visual observations $V _ { 1 : t } ,$ a spatial-temporal query $Q _ { s t } ,$ and the corresponding ground-truth answer a. Here, $V _ { 1 : t } = \{ V _ { 1 } , V _ { 2 } , \ldots , V _ { t } \}$ represents a time-ordered visual sequence that encodes dynamic 3D scene evolution.

Unlike conventional multimodal reasoning, the 4D intelligence task requires constructing an implicit spatio-temporal representation $S ( V _ { 1 : t } )$ , which captures geometric attributes $( e . g .$ , distance, orientation, occlusion, topology) as well as temporal dynamics $( e . g .$ , motion, interaction, and state transitions). Formally, given $x _ { i } \in D$ , the VLM aims to generate a textual token sequence y by reasoning over $S ( V _ { 1 : t } )$ to resolve the spatial-temporal query $Q _ { s t }$ , where the generated sequence yˆ is expected to align with the ground-truth answer y.

Reinforcement Learning with Verifiable Rewards. RLVR departs from conventional RL by deriving rewards directly from ground-truth correctness rather than relying on a learned reward model. This eliminates the need for auxiliary reward estimation, simplifies the training pipeline, reduces computational cost, and mitigates reward hacking by grounding supervision in objectively verifiable outcomes.

Current implementations of RLVR generally follow a two-part structure: a reward computation scheme that evaluates answer validity, and a policy optimization procedure built upon Group Relative Policy Optimization (GRPO) [21,34], which performs stable updates through intra-group comparative advantage estimation. GRPO is a reinforcement learning algorithm derived from PPO that improves policy stability via group-based relative advantage estimation. Its main advantage is that it does not require a value model, reducing memory and computational cost. The training objective of GRPO is to maximize:

$$
J _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { q \sim P ( Q ) , \{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi \theta _ { \mathrm { o l d } } ( O | q ) } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | o _ { i } | } \sum _ { t = 1 } ^ { | o _ { i } | } \left( \operatorname* { m i n } \bigl ( r _ { i , t } ( \theta ) A _ { i } , \theta \bigr ) \right) \right] ,\tag{1}
$$

$$
\begin{array} { r } { \mathrm { c l i p } \big ( r _ { i , t } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon \big ) A _ { i } \big ) - \beta D _ { \mathrm { K L } } \big ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } \big ) \bigg ) \Bigg ] , } \end{array}\tag{2}
$$

where

$$
\boldsymbol { r } _ { i , t } ( \theta ) = \frac { \pi _ { \boldsymbol { \theta } } ( o _ { i , t } \mid \boldsymbol { q } , o _ { i , < t } ) } { \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( o _ { i , t } \mid \boldsymbol { q } , o _ { i , < t } ) } , \boldsymbol { A } _ { i } = \frac { \boldsymbol { r } _ { i } - \operatorname * { m e a n } ( \{ \boldsymbol { r } _ { 1 } , \boldsymbol { r } _ { 2 } , \dots , \boldsymbol { r } _ { G } \} ) } { \mathrm { s t d } ( \{ \boldsymbol { r } _ { 1 } , \boldsymbol { r } _ { 2 } , \dots , \boldsymbol { r } _ { G } \} ) } .\tag{3}
$$

Here, $\pi _ { \theta }$ and $\pi _ { \theta _ { \mathrm { o l d } } }$ denote the updated and previous policies. q denotes the input query, $\{ o _ { i } \} _ { i = 1 } ^ { G }$ are G sampled responses, |o<sub>i</sub>| is the length of the i-th response, and $r _ { i }$ is the reward assigned to response $o _ { i }$

## 4 Methodology

## 4.1 Overview

Our goal is to endow general vision-language models with explicit spatial reasoning capabilities beyond flat image understanding. Hence, we build our framework upon a strong general-purpose VLM backbone, Qwen3-VL [2], and train it through a two-stage pipeline, as shown in Fig. 2. The data statistics of two stages are introduced in Sec. 5.1.

Stage 1: Spatial Perception Fine-tuning. This stage aims to cultivate spatial grounding and foundational perception capabilities through supervised fine-tuning. Rather than introducing complex reasoning signals abruptly, we implement a short-to-long supervision curriculum to prepare the model for subsequent RL optimization.

We begin by jointly training the model on short-form paired data, where general multimodal understanding from LLaVA-OneVision [27] is synergized with specialized spatial perception tasks. These concise responses emphasize core grounding skills, such as localization, correspondence, and spatial relations, thereby enabling the model to align visual observations with spatial semantics.

![](images/c8341ddca5fcb2af7df468817b1951afb7c744e7c16530ab3c5e9df32e7112e6.jpg)  
Fig. 2: The progressive training framework of FactoSR. Stage 1 performs supervised fine-tuning on diverse spatial tasks to build foundational spatial perception. Stage 2 introduces factorized reinforcement learning with verifiable rewards (Accuracy, XY, Z, T), explicitly enhancing correspondence, depth, and temporal reasoning.

During this process, visual tokens extracted from images serve as the conditioning context, and the model is optimized via a standard autoregressive objective:

$$
\mathcal { L } _ { \theta } ( y \mid V _ { 1 : t } , Q _ { s t } ) = - \sum _ { i = 2 } ^ { L } w _ { i } \log p _ { \theta } \left( y _ { i } \mid V _ { 1 : t } , Q _ { s t } , y _ { 1 : i - 1 } \right) .\tag{4}
$$

This mixed-task training is performed for a single epoch to inject spatial priors without compromising generalist performance.

Building upon this spatial foundation, we then introduce long-form data featuring structured “Anchor-Transfer-Verify” reasoning trajectories for cross-frame alignment, as illustrated in Fig. 4. The model undergoes further refinement over several hundred iterations using these extended sequences. By progressively scaling from basic grounding to reasoning, Stage I equips the model with incipient spatial logic while ensuring training stability. Crucially, this phase significantly facilitates convergence during the subsequent RL stage.

Stage 2: Factorized Spatial Reinforcement Learning. To bridge the gap between supervised awareness and physically consistent reasoning, we employ Group Relative Policy Optimization (GRPO) to further refine the model’s reasoning trajectories. Building upon the “Anchor-Transfer-Verify” CoT framework from Stage I, this reinforcement stage iteratively optimizes the model’s thinking process against our curated verification dataset. Specifically, we transition from static supervision to a factorized rule-based reward that evaluates the consistency of the generated reasoning across three physical dimensions: XY (planar correspondence), Z (depth order), and $T$ (temporal cycle-consistency). This structured feedback steers the model’s latent thinking away from heuristic shortcuts toward a coherent, self-verifying 4D spatial logic. The detailed formulations of these rewards and the final optimization objective are introduced in the following subsections.

## 4.2 Factorized Spatial Reinforcement Learning

In this stage, we employ RL to further enhance the spatial reasoning capabilities of the stage-2 model. For this purpose, we utilize the revised GRPO algorithm [60], which bypasses the need for a value model by computing the relative advantage of each response within a group of responses to the same question. To facilitate this process, we curated a verification dataset comprising tasks related to spatial understanding, 3D object detection, and general multi-modal understanding. In the GRPO framework, we employ a mixed rule-based reward to evaluate the generated responses, including format, accuracy, XY, Z, and T rewards. For a given response $\hat { y }$ and its corresponding ground truth y, the basic accuracy reward function is used to ensure correctness and is defined as: $\mathcal { R } _ { a c c } ( y , \hat { y } ) = \mathbb { I } [ \hat { y } = y ]$

XY Reward: Point Correspondence. For multi-view spatial reasoning, a model should not “guess” correspondence from 2D appearance alone. Instead, a correct correspondence must be geometrically admissible: the predicted point in the target view should agree with the reprojection of the reference point under camera intrinsics, poses, and depth. Therefore, our $X Y$ reward directly supervises 2D correspondence by enforcing reprojection consistency and overlap validity, providing dense, physically grounded guidance during RL.

Given two views $V _ { 1 } , V _ { 2 }$ with depth maps $\mathcal { D } _ { 1 } , \mathcal { D } _ { 2 }$ , intrinsics ${ \bf K } _ { 1 } , { \bf K } _ { 2 }$ , and camera-to-world poses $\mathbf { T } _ { 1 } , \mathbf { T } _ { 2 }$ , we are provided a reference pixel $\mathbf { p } _ { 1 } = ( u _ { 1 } , v _ { 1 } )$ in $V _ { 1 }$ and a discrete candidate set $\{ \mathbf { p } _ { 2 } ^ { ( A ) } , \mathbf { p } _ { 2 } ^ { ( B ) } , \mathbf { p } _ { 2 } ^ { ( C ) } , \mathbf { p } _ { 2 } ^ { ( D ) } \}$ in $V _ { 1 }$ . Let $\hat { \bf p } _ { 2 } = ( \hat { u } _ { 2 } , \hat { v } _ { 2 } )$ be the model-selected candidate point in view 2. Specifically, we first compute the reprojection map from $V _ { 1 }$ to $V _ { 2 }$ . For each pixel ${ \bf p } _ { 1 } = ( u _ { 1 } , v _ { 1 } )$ in view 1 with depth $d _ { 1 } = \mathcal { D } _ { 1 } ( \mathbf { p } _ { 1 } )$ , we unproject to camera coordinates:

$$
\mathbf { x } _ { 1 } = d _ { 1 } \mathbf { K } _ { 1 } ^ { - 1 } \tilde { \mathbf { p } } _ { 1 } , \quad \tilde { \mathbf { p } } _ { 1 } = ( u _ { 1 } , v _ { 1 } , 1 ) ^ { \top } .\tag{5}
$$

We then transform it to world coordinates and reproject to view 2:

$$
{ \bf X } = { \bf T } _ { 1 } \tilde { \bf x } _ { 1 } , \quad { \bf x } _ { 2 } = { \bf T } _ { 2 } ^ { - 1 } { \bf X } , \quad \tilde { \bf p } _ { 2 } \sim { \bf K } _ { 2 } { \bf x } _ { 2 } ,\tag{6}
$$

where $\tilde { \mathbf { x } } _ { 1 } = ( \mathbf { x } _ { 1 } ^ { \top } , 1 ) ^ { \top }$ and $\sim$ denotes equality up to scale, and the resulting pixel coordinate is $\begin{array} { r } { \mathbf { p } _ { 2 } ^ { \star } = ( u _ { 2 } ^ { \star } , v _ { 2 } ^ { \star } ) = \left( \frac { \tilde { p } _ { 2 , x } } { \tilde { p } _ { 2 , z } } , \frac { \tilde { p } _ { 2 , y } } { \tilde { p } _ { 2 , z } } \right) } \end{array}$ . We also record the projected depth in view-2 camera coordinates $z _ { 2 } ^ { \dot { \star } }$ and a validity mask

$$
\mathbb { I } _ { \mathrm { v a l i d } } ( { \mathbf p } _ { 1 } ) = \mathbb { I } [ z _ { 2 } ^ { \star } > 0 ] \cdot \mathbb { I } [ { \mathbf p } _ { 2 } ^ { \star } \in V _ { 2 } ] ,\tag{7}
$$

If $\mathbb { I } _ { \mathrm { v a l i d } } ( \mathbf { p } _ { 1 } ) = 0$ , the correspondence is undefined, and we assign a zero reward. Otherwise, we compare the model prediction $\hat { { \bf p } } _ { 2 }$ with the projected target $\mathbf { p } _ { 2 } ^ { \star }$ in a normalized coordinate system:

$$
d = \left\| \left( \frac { u _ { 2 } ^ { \star } } { W _ { 2 } } , \frac { v _ { 2 } ^ { \star } } { H _ { 2 } } \right) - \left( \frac { \hat { u } _ { 2 } } { W _ { 2 } } , \frac { \hat { v } _ { 2 } } { H _ { 2 } } \right) \right\| _ { 2 } ,\tag{8}
$$

where $( H _ { 2 } , W _ { 2 } )$ is the size of $V _ { 2 }$ . We then assign a soft, distance-aware reward:

$$
r _ { \mathrm { r e p r o j } } = \exp \left( - \frac { d } { \sigma } \right) \cdot \mathbb { I } [ d \leq 3 \sigma ] ,\tag{9}
$$

where $\sigma$ controls tolerance. The hard cutof $\mathbb { I } [ d \le 3 \sigma ]$ prevents rewarding faraway guesses and stabilizes RL by suppressing spurious gradients from grossly incorrect correspondences.

Reprojection alone may still reward points that are geometrically close but not visible in view 2 due to occlusion. To enforce physical plausibility, we construct an overlap mask on view 2 by checking depth consistency. For each valid reprojection, we mark $\mathbf { p } _ { 2 }$ as visible if

$$
\left| \mathcal { D } _ { 2 } ( \mathbf { p } _ { 2 } ) - z _ { 2 } ^ { \star } \right| \leq \delta ,\tag{10}
$$

where $\delta$ is a depth threshold. Thus, we construct the overlap mask $\mathbf { M } _ { 2 } ( \cdot ) \in$ $\{ 0 , 1 \}$ , which identifies pixels in view 2 that are truly visible from view 1 under the given camera configuration. It is derived from the same reprojection process used to compute p<sup>⋆</sup>. We gate the $X Y$ reward using the mask:

$$
R _ { X Y } = r _ { \mathrm { r e p r o j } } \cdot \mathbf { M } _ { 2 } ( \hat { \mathbf { p } } _ { 2 } ) \ .\tag{11}
$$

This overlap gating turns the reward into a visibility-aware signal: even if $\hat { { \bf p } } _ { 2 }$ is close to $\mathbf { p } _ { 2 } ^ { \star } .$ , it receives zero reward if it falls outside the depth-consistent overlap region, discouraging correspondences on occluded or non-overlapping areas. Overall, the XY reward explicitly avoids appearance heuristics and promotes cross-view alignment that is both reprojection-consistent and visibility-valid.

Z Reward: Depth Order. After SFT establishes the model’s basic grounding ability, we observe that further optimizing IoU localization during RL yields no benefit for spatial visual question answering. The core dificulty is not object detection itself, but reasoning about relative depth between objects in 3D space from 2D observations. Rather than supervising metric depth values, we directly optimize the correctness of depth order.

The policy is required to output a set of structured 3D bounding boxes. All predicted boxes are matched with ground-truth boxes using Hungarian assignment [22] with a 3D GIoU-based cost. For each matched object, we extract the depth of its center in the camera coordinate system, producing the predicted and ground-truth depth sequences. Given the predicted and ground-truth depth sequences $\hat { \textbf { z } } = \{ \hat { z } ^ { ( 1 ) } , \dots , \hat { z } ^ { ( n ) } \}$ and $\textbf { z } = \{ z ^ { ( 1 ) } , \dots , z ^ { ( n ) } \}$ , we evaluate whether the predicted front–back relationships between objects are consistent with the ground truth using the Kendall-τ rank correlation.

Specifically, for every pair of objects $( i , j )$ with $i < j$ , we compare the ordering of their depths in the predicted and ground-truth sequences:

$$
( \hat { z } ^ { ( i ) } - \hat { z } ^ { ( j ) } ) ( z ^ { ( i ) } - z ^ { ( j ) } ) > 0 .\tag{12}
$$

A pair is considered concordant if the predicted and ground-truth orders agree, and discordant if the two orders disagree. Let $N _ { c }$ and $N _ { d }$ denote the numbers of concordant and discordant pairs, respectively. The Kendall-τ coeficient is then defined as:

$$
\tau = \frac { N _ { c } - N _ { d } } { \frac { n ( n - 1 ) } { 2 } } .\tag{13}
$$

The coeficient $\tau \in [ - 1 , 1 ]$ measures the consistency between predicted and ground-truth depth rankings. We further normalize it to [0, 1] to obtain the depth ordering reward: $\begin{array} { r } { R _ { Z } = \frac { \tau + { \bar { 1 } } } { 2 } } \end{array}$

Finally, this reward directly reinforce the model to recover the relative depth structure of the scene, which constitutes the core reasoning requirement for 3D spatial understanding.

T Reward: Temporal Cycle Consistency. While XY and Z rewards collaboratively reinforce 3D reasoning, they do not guarantee that a model truly understands motion over time. A model may answer a navigation question using static cues without reasoning about how the camera actually moves. To explicitly reinforce the temporal dimension, we introduce a T reward that enforces cycle consistency, i.e., reasoning from the start view to the end view must be logically reversible when the process is queried in the opposite direction.

Each training sample, therefore, contains a forward question-answer pair together with a constructed inverse question and its expected answer. For instance, if the forward solution of camera motion corresponds to $^ { 6 6 } \mathrm { T }$ urn right, Turn back”, the inverse problem, reasoning from the terminal state back to the start, should yield “Turn back, Turn left”, ensuring a physically consistent cycle. During RL, the model rolls out both the forward prediction $\hat { y }$ and the inverse prediction ${ \hat { y } } ^ { i n v }$ . The temporal cycle consistency reward evaluates whether both predictions match the expected answers:

$$
\mathcal { R } ( y , \hat { y } ) = \mathcal { R } _ { a c c } ( y , \hat { y } ) \cdot \mathcal { R } _ { a c c } ( y ^ { i n v } , \hat { y } ^ { i n v } ) .\tag{14}
$$

T reward complements the accuracy reward by requiring the model to maintain logical reversibility between forward and inverse motion sequences, fostering a robust understanding of ego-motion and temporal causality.

Final Objective. To synergistically integrate the semantic accuracy with granular geometric constraints, we define a factorized total reward function ${ \mathcal { R } } _ { \mathrm { t o t a l } }$ We stipulate that any reinforcement is strictly contingent upon the fulfillment of the format constraint $\mathbb { I } [ \mathcal { R } _ { \mathrm { f o r m a t } } = 1 ]$ , thereby ensuring the structure of the model output. The final objective is formulated as a weighted composition:

$$
\mathcal { R } _ { \mathrm { t o t a l } } ( y , \hat { y } ) = \mathbb { I } [ \mathcal { R } _ { f o r m a t } = 1 ] \cdot ( \lambda _ { 1 } \mathcal { R } _ { a c c } + \lambda _ { 2 } \mathcal { R } _ { X Y } + \lambda _ { 3 } \mathcal { R } _ { Z } + \lambda _ { 4 } \mathcal { R } _ { T } ) \ ,\tag{15}
$$

where $\lambda _ { \{ \cdot \} }$ denotes the task-specific importance of correspondence, depth reasoning, and temporal consistency.

![](images/9eba955c8261ad028ea19c9ce20f741fbab2f54f1dec1bc8346d5e8592cd2033.jpg)  
Fig. 3: Overview of (a) FactoSR-SFT and (b) FactoSR-RL datasets. Please zoom in.

By factorizing the reward space into explicit spatial, depth, and temporal dimensions, our RL framework efectively transitions from simple pattern matching to a deeper, physically-grounded understanding of 4D scenes.

## 5 Experiments

## 5.1 Data

Fig. 3 summarizes the training data used in our framework. The nested charts present hierarchical statistics, where each ring from the center outward corresponds to progressively finer-grained categories in the legend.

In the SFT stage, the dataset contains 8.2M samples, dominated by shortanswer tasks (7.4M, 90.3%), while long-answer reasoning data (795K, 9.7%) provides additional chain-of-thought supervision. The short-answer portion mainly consists of general instruction data LLaVA-OneVision (5.1M) [27], followed by spatial reasoning (1.6M) and mathematical reasoning (737K). We build a new spatial reasoning dataset using our designed data pipeline. This dataset organizes 1.2M samples across both long-answer and short-answer formats, covering diverse spatial tasks, such as single-view depth estimation, multi-view correspondence, multi-view camera-relative motion, and video spatial understanding [53].

Finally, the RL stage employs a smaller but targeted dataset of 32K samples, where spatial reasoning dominates (81.2%), complemented by general mathematical reasoning data (18.8%).

## 5.2 Implementation Details

Stage 1. This training stage aims to establish a strong foundation of spatial understanding capabilities. For this stage, we use a global batch size of 128, a sequence length of 8,192, and a dynamic data packing strategy to accelerate the training process. We employ the AdamW [28] optimizer, setting the base learning rate to $5 \times 1 0 ^ { - 5 }$ and the vision encoder’s learning rate to $5 \times 1 0 ^ { - 6 }$ . For the CoT cold-start, we continue training the model. The hyper-parameters are adjusted to a global batch size of 128, a base learning rate of $1 \times 1 0 ^ { - 5 }$ , a vision encoder learning rate of $1 \times 1 0 ^ { - 6 }$ , and a sequence length of 8,192.

Stage 2. In the RL stage, we refine the model from the first stage using the VeRL [38] framework. We adopt a revised version of the GRPO algorithm [60], using a rollout size of 8 samples per query and a sampling temperature of 1.0.

Table 1: Fine-grained comparison on 4D reasoning benchmarks. The best results among open-source VLMs are bold . “Base” denotes Qwen3-VL-8B-Instruct.
<table><tr><td rowspan="2">Methods</td><td colspan="2"></td><td colspan="6">All-Angles-Bench [56]</td><td colspan="10">VSI-Bench [51]</td></tr><tr><td>Avg. |Attr.</td><td></td><td>Pose</td><td></td><td></td><td></td><td>Cnt. Manip. Rel-Dir. Rel-Dist.Avg.</td><td></td><td>Cnt. Obj-Size Room-Size Abs-Dist. Dir-H</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Dir-M Dir-E Rel-Dist. Appr-Ord. Route</td><td></td></tr><tr><td></td><td></td><td>Open-Source General/</td></tr><tr><td>InternVL3-2B [63]</td><td>48.6</td><td>66.3</td><td>42.6</td><td>48.2</td><td>41.8</td><td>39.5</td><td>50.2</td><td>|30.4</td><td>|57.7</td><td>19.8</td><td>33.3</td><td>29.0</td><td>28.7</td><td>25.7</td><td>47.5</td><td>37.0</td><td></td><td>14.7 22.7</td></tr><tr><td>InternVL3-8B [63]</td><td>50.5</td><td>78.6</td><td>36.4</td><td>51.0</td><td>42.0</td><td>34.7</td><td>52.8</td><td>38.7</td><td>55.4</td><td>39.7</td><td>41.1</td><td>33.6</td><td>18.5</td><td>42.6</td><td>53.0</td><td>40.8</td><td>35.1</td><td>22.7</td></tr><tr><td>SpaceR-7B [32]</td><td>49.8</td><td>71.8</td><td>51.1</td><td>44.6</td><td>42.4</td><td>36.4</td><td>51.4</td><td>44.4</td><td>53.1</td><td>60.1</td><td>37.8</td><td>28.6</td><td>40.5</td><td>47.4</td><td>48.8</td><td>43.1</td><td>40.9</td><td>33.5</td></tr><tr><td>MiMo-VL-7B [40]</td><td>52.9</td><td>78.3</td><td>38.1</td><td>53.4</td><td>39.9</td><td>43.8</td><td>57.1</td><td>47.8</td><td>64.9</td><td>61.4</td><td>48.9</td><td>22.4</td><td>36.2</td><td>44.2</td><td>50.2</td><td>46.5</td><td>60.2</td><td>29.9</td></tr><tr><td>Qwen2.5-VL-7B-Instruct [3]</td><td>50.1</td><td>74.7</td><td>50.0</td><td>51.0</td><td>39.1</td><td>37.2</td><td>50.6</td><td>36.0</td><td>42.3</td><td>46.2</td><td>40.2</td><td>22.1</td><td>29.0</td><td>40.7</td><td>50.7</td><td>36.3</td><td>28.5</td><td>30.4</td></tr><tr><td>SpatialThinker-7B [5]</td><td>47.6</td><td>70.5</td><td>43.2</td><td>41.4</td><td>38.7</td><td>39.5</td><td>49.0</td><td>33.0</td><td>41.4</td><td>37.7</td><td>41.7</td><td>13.5</td><td>24.7</td><td>40.5</td><td>48.4</td><td>40.6</td><td>27.5</td><td>29.4</td></tr><tr><td>VST-7B-SFT [52]</td><td>49.5</td><td>76.5</td><td>35.8</td><td>45.8</td><td>42.0</td><td>41.8</td><td>48.2</td><td>55.3</td><td>68.0</td><td>73.2</td><td>57.6</td><td>39.8</td><td>44.8</td><td>54.0</td><td>51.6</td><td>51.5</td><td>52.8</td><td>43.3</td></tr><tr><td>VST-7B-Thinking [52]</td><td>49.0</td><td>75.2</td><td>39.2</td><td>47.4</td><td>42.4</td><td>36.9</td><td>48.0</td><td>52.6</td><td>61.9</td><td>73.0</td><td>58.3</td><td>33.2</td><td>40.5</td><td>46.8</td><td>50.2</td><td>50.3</td><td>53.9</td><td>41.2</td></tr><tr><td>Qwen3-VL-8B-Instruct [2] Qwen3-VL-8B-Thinking [2]</td><td>49.5 50.9</td><td>76.5 78.9</td><td>21.0 24.4</td><td>51.0 47.4</td><td>37.6 43.7</td><td>43.8 42.9</td><td>53.6 53.0</td><td>55.6 49.8</td><td>63.8 47.0</td><td>73.0 65.6</td><td>56.3 43.9</td><td>44.3 38.8</td><td>42.3 42.3</td><td>52.1</td><td>53.0 51.6</td><td>53.3</td><td>57.3</td><td>32.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>44.4</td><td></td><td>52.5</td><td>54.7</td><td>35.1</td></tr><tr><td></td><td></td><td colspan="10">Our Method</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FactoSR-8B-SFT</td><td>51.6 |</td><td>77.5</td><td>40.9</td><td>50.2</td><td>45.6</td><td>38.9</td><td>50.8</td><td>58.5</td><td>66.7 66.4</td><td>73.7 75.7</td><td>58.2</td><td>45.8</td><td>46.4</td><td>53.7</td><td>50.2</td><td>59.3</td><td>63.1</td><td>38.2</td></tr><tr><td>FactoSR-8B-RL ∆ (Ours vs. Base)</td><td>55.4 +5.9</td><td>79.1</td><td>44.3</td><td>51.0</td><td>45.2</td><td>44.6</td><td>61.1</td><td>61.5</td><td>+2.6</td><td></td><td>63.7</td><td>46.7</td><td>53.4 +11.1 +11.6 +8.8</td><td>63.7</td><td>61.8</td><td>60.3</td><td>65.7</td><td>46.9</td></tr><tr><td></td><td></td><td></td><td>+2.6+23.3+0.0</td><td></td><td>+7.6</td><td>+0.8</td><td>+7.5</td><td>+5.9</td><td></td><td>+2.7</td><td>+7.4</td><td>+2.4</td><td></td><td></td><td></td><td>+7.0</td><td>+8.4</td><td>+14.4</td></tr></table>

Table 2: Quantitative Comparison with state-of-the-art VLMs on 13 3D/4D spatial benchmarks and general benchmarks.

<table><tr><td rowspan="2">Benchmarks</td><td rowspan="2">3/4D-Avg.</td><td colspan="2">[4D Reasoning| VSI AllAngles</td><td colspan="6">3D Reasoning</td><td colspan="4">GeneralQA</td></tr><tr><td colspan="10">BLINK 3DSR C CV-2D CV-3D ERQA RealWorldQA MMSIMMB CN MMB EN MMStar OCRB</td></tr><tr><td></td><td colspan="10"></td></tr><tr><td>Gemini-2.5-Pro [16]</td><td>64.4</td><td>|48.4</td><td>61.3 70.6</td><td>Proprietary Models 57.6</td><td>80.4</td><td>91.3</td><td>55.8</td><td>77.3</td><td>36.9</td><td>90.2</td><td>89.2</td><td>79.1</td><td>86.6</td></tr><tr><td>GPT-4o [1]</td><td>57.7</td><td>34.0</td><td>52.4 65.9</td><td>44.3</td><td>75.8</td><td>83.0</td><td>57.0</td><td>76.2</td><td>30.3</td><td>83.9</td><td>84.8</td><td>65.1</td><td>80.6</td></tr><tr><td colspan="10">Open-Source General/Spatial VLMs</td><td colspan="3"></td></tr><tr><td>InternVL3-2B [63]</td><td>50.6</td><td>30.4</td><td>48.6</td><td>52.8</td><td>46.4</td><td>71.9 77.3</td><td>36.2</td><td>65.5</td><td>25.9</td><td>77.1</td><td>78.3</td><td>61.5</td><td>83.7</td></tr><tr><td>InternVL3-8B [63]</td><td>56.2</td><td>38.7</td><td>50.5</td><td>55.7</td><td>52.7 80.6</td><td>86.0</td><td>40.5</td><td>70.6</td><td>30.9</td><td>81.9</td><td>82.0</td><td>68.2</td><td>88.0</td></tr><tr><td>SpaceR-7B [32]</td><td>53.3</td><td>44.4</td><td>49.8</td><td>54.3</td><td>47.5</td><td>73.9 76.2</td><td>40.5</td><td>64.2</td><td>29.4</td><td>80.3</td><td>83.0</td><td>61.6</td><td>85.9</td></tr><tr><td>MiMo-VL-7B [40]</td><td>58.2</td><td>47.8</td><td>52.9</td><td>59.7</td><td>56.1</td><td>76.9 86.9</td><td>41.0</td><td>73.5</td><td>29.3</td><td>80.9</td><td>80.8</td><td>71.1</td><td>84.5</td></tr><tr><td>Qwen2.5-VL-7B-Instruct [3]</td><td>52.8</td><td>36.0</td><td>50.1</td><td>55.3</td><td>49.0</td><td>75.6 73.8</td><td>41.0</td><td>68.1</td><td>26.5</td><td>82.3</td><td>82.3</td><td>70.9</td><td>87.9</td></tr><tr><td>SpatialThinker-7B [5]</td><td>53.2</td><td>33.0</td><td>47.6</td><td>55.9</td><td>47.6</td><td>78.7</td><td>83.3 39.5</td><td>67.6</td><td>25.7</td><td>79.8</td><td>81.0</td><td>63.4</td><td>87.7</td></tr><tr><td>VST-7B-SFT [52]</td><td>60.2</td><td>55.3</td><td>49.5</td><td>62.1</td><td>53.3</td><td>77.9 94.8</td><td>43.8</td><td>71.5</td><td>33.3</td><td>80.4</td><td>81.1</td><td>63.1</td><td>86.3</td></tr><tr><td>VST-7B-Thinking [52]</td><td>60.5</td><td>52.2</td><td>49.0</td><td>62.5</td><td>57.6</td><td>78.6 95.5</td><td>44.5</td><td>69.3</td><td>34.9</td><td>76.5</td><td>77.2</td><td>63.9</td><td>86.9</td></tr><tr><td>Qwen3-VL-8B-Instruct [2]</td><td>59.1</td><td>55.6</td><td>49.5</td><td>66.1</td><td>52.8</td><td>78.6 90.8</td><td>40.1</td><td>70.7</td><td>28.1</td><td>83.3</td><td>84.2</td><td>70.1</td><td>90.3</td></tr><tr><td>Qwen3-VL-8B-Thinking [2]</td><td>59.9</td><td>49.8</td><td>50.9</td><td>62.3</td><td>54.8</td><td>78.8 93.1</td><td>42.8</td><td>73.3</td><td>30.7</td><td>82.2</td><td>82.7</td><td>74.3</td><td>85.3</td></tr><tr><td colspan="10">Our Method</td><td colspan="3"></td></tr><tr><td>FactoSR-8B-SFT</td><td>60.3</td><td>58.5</td><td>51.6</td><td>62.6</td><td>51.9</td><td>81.1</td><td>90.7 45.3</td><td>71.6</td><td>29.7</td><td>83.2</td><td>85.1</td><td>67.4</td><td>87.7</td></tr><tr><td>FactoSR-8B-RL ∆ (Ours vs. Base)</td><td>62.0</td><td>61.5</td><td>55.4</td><td>66.0</td><td>52.9 81.3</td><td>91.7</td><td>47.3</td><td>71.6</td><td>30.7</td><td>84.1</td><td>85.7</td><td>69.1</td><td>87.6</td></tr><tr><td></td><td>+2.9</td><td>+5.9</td><td>+5.9</td><td>-0.1</td><td>+0.1 +2.7</td><td>+0.9</td><td>+7.2</td><td>+0.9</td><td>+2.6</td><td>+0.8</td><td>+1.5</td><td>-1.0</td><td>-2.7</td></tr></table>

This stage utilizes the AdamW optimizer with a constant learning rate of $1 \times 1 0 ^ { - 6 }$ and a global batch size of 128.

Evaluation. We assess the abilities of state-of-the-art and our models across three distinct capabilities: (1) 4D reasoning ability is benchmarked with All-Angles-Bench [56] and VSI-Bench [51]. We evaluate the fine-grained performance on sub-categories of the two core benchmarks. (2) 3D reasoning ability is benchmarked with BLINK [20], 3DSRBench (3DSR\_C) [30], CVBench (CV-2D, CV-3D) [41], ERQA [39], RealWorldQA [49], and MMSI-Bench (MMSI) [54]. (3) General multi-modal understanding is evaluated across a suite of standard benchmarks: MMBench [24] (MMB\_CN, MMB\_EN), MMStar [12], and OCR-Bench [25]. All the models are evaluated using their native system prompt to ensure the fairness of comparisons. More details on prompts, SFT, and RL training setups, are provided in Appendices.

## 5.3 Quantitative Results

4D Reasoning Analysis. We compare FactoSR with a range of open-source spatial and general VLMs, including InternVL, Qwen-VL, MiMo-VL, SpaceR, SpatialThinker, and VST. As shown in Tab. 1, our method achieves the best performance on both All-Angles-Bench and VSI-Bench, two challenging 4D spatial reasoning benchmarks. Specifically, FactoSR-8B-RL achieves 61.5% on VSI and

![](images/6e64256b2c722b801e9556a9e0338ab5490e8e8001f703e6d2bc39aad4509e9f.jpg)  
Fig. 4: Qualitative comparison between Qwen3-VL-8B Instruct and FactoSR-8B-RL on spatial reasoning tasks. While Qwen3-VL-8B Instruct often relies on appearance cues and makes inconsistent judgments, FactoSR-RL produces 4D-consistent reasoning across views, depth, and motion.

55.4% on AllAngles, outperforming the strongest open-source baselines by +5.9 and +2.5 points, respectively. The improvements are consistent across diverse sub-tasks such as attribute identification, camera pose estimation, and relative spatial reasoning on AllAngles, as well as video-based tasks including relative direction, spatial distance reasoning, and route planning on VSI.

Overall Benchmark Comparison. Tab. 2 further compares our model with state-of-the-art proprietary and open-source VLMs across 13 benchmarks.

FactoSR-8B-RL achieves the best spatial reasoning performance with the average of 62.0 across all 3D/4D spatial benchmarks, improving over the base model by +2.9 points. Besides the strong gains on the 3/4D benchmarks, FactoSR also achieves competitive performance on general multimodal benchmarks, e.g., MMBench-CN, MMBench-EN. More results are provided in Appendices.

## 5.4 Ablation Study

To quantitatively evaluate the contributions of the factorized rewards, we construct three specialized evaluation groups to analyze spatial reasoning improvements: The correspondence group $\pmb { \Delta } _ { \mathrm { C o r } }$ includes All-Angles-Bench (Attribute Identification), BLINK (Functional Correspondence, Semantic Correspondence, Visual Correspondence). The depth $\mathrm { g r o u p } { \pmb \Delta } _ { \mathrm { D e p t h } }$ includes CV-Bench-3D (Depth), BLINK (Relative Depth, Spatial Relation). The camera motion group $\Delta _ { \mathrm { t } }$ includes RealWorldQA, MMSIBench (Motion-Cam), BLINK (Multi-view Reasoning), VSI-Bench (Route Planning). The 3D/4D Avg. Acc. is computed over 9 spatial benchmarks.

Tab. 3 shows that vanilla GRPO yields only marginal gains (+0.2), indicating that general RL signals alone are insuficient for spatial reasoning. Introducing factorized rewards leads to targeted improvements: XY mainly enhances correspondence reasoning $( \varDelta _ { \mathrm { C o r } } + 2 . 7 )$ , Z significantly improves depth understanding $\left( \Delta _ { \mathrm { D e p t h } } \ + 1 . 3 \right)$ , and

Table 3: Average accuracy across all 9 benchmarks and performance on three specialized tasks with relative improvements. FactoSR models consistently outperform SFT and vanilla GRPO.
<table><tr><td>Model</td><td>|3D/4D Avg. Acc. (9)|</td><td>∆Cor</td><td> $\Delta _ { \mathrm { D e p t h } }$ </td><td>∆t</td></tr><tr><td colspan="5">Supervised Fine-Tuning</td></tr><tr><td>FactoSR-SFT</td><td>60.3</td><td>69.9</td><td>86.3</td><td>45.7</td></tr><tr><td colspan="5">Reinforcement Learning</td></tr><tr><td> $+ \ \mathrm { { V a n i l l a } \ G R P O { } | }$ </td><td> $6 0 . 5 { \scriptstyle \pm 0 . 1 } \ ( + 0 . 2 )$ </td><td> $7 1 . 6 { \scriptstyle \pm 0 . 8 \ } ( + 1 . 7 )$ </td><td> $8 5 . 8 { \scriptstyle \pm 1 . 1 } \ \left( - 0 . 5 \right)$ </td><td> $4 7 . 1 \pm 2 . 2 \ ( + 1 . 4 )$ </td></tr><tr><td>+ XY Reward</td><td> $6 0 . 7 _ { \pm 0 . 1 } \ \dot { ( + 0 . 4 ) }$ </td><td> ${ \bf 7 2 . 6 { \scriptstyle \pm 0 . 7 } } \ ( + 2 . 7 )$ </td><td> $8 6 . 6 { \overset { - } { \pm } } 1 . 0 \ { \overset { ^ { } } { ( + 0 . 3 ) } }$ </td><td> $4 7 . 2 \overset { - } { \pm } 2 . 8 \ \overset { } { ( + 1 . 5 ) }$ </td></tr><tr><td>+ Z Reward</td><td> $6 0 . 6 _ { \pm 0 . 2 } \ ( + 0 . 3 )$ </td><td> $6 9 . 8 _ { \pm 0 . 6 } \ : \ : \ : ( - 0 . 1 )$ </td><td> ${ \bf 8 7 . 6 _ { \pm 0 . 4 } ^ { - } \left( + 1 . 3 \right) }$ </td><td> $4 7 . 4 _ { \pm 2 . 0 } ^ { - } \ ( + 1 . 7 )$ </td></tr><tr><td>+ T Reward</td><td> $6 0 . 9 _ { \pm 0 . 2 } ^ { - } \ ( + 0 . 6 )$ </td><td> $6 9 . 9 _ { \pm 0 . 3 } ~ ( + 0 . 0 )$ </td><td> $8 7 . 4 _ { \pm 0 . 8 } ~ ( + 1 . 1 )$ </td><td>53.6±1.1 (+7.9)</td></tr><tr><td>+ Full Rewards</td><td>62.0±0.2 (+1.7)</td><td>72.1±0.7 (+2.2) 87.1±0.6 (+0.8) 50.9±1.3 (+5.2)</td><td></td><td></td></tr></table>

T notably strengthens temporal reasoning $\left( \varDelta _ { t } + 7 . 9 \right)$ . Further analysis in Tab. 4 reveals that the vanilla grounding reward [52] is inefective, causing overall performance drops (-0.2) and degraded spatial relation reasoning (-0.7). In contrast, Z reward consistently improves all depth-related metrics, especially Relative Depth (+3.2), validating that RL should emphasize relative depth of objects rather than the simple localization. Combining all rewards achieves the best overall improvement (1.7%), demonstrating that $\mathrm { X Y , ~ Z , }$ and T provide complementary supervision.

Table 4: Ablation on depth supervision showing the efectiveness of the Z reward and vanilla grounding reward over SFT models.
<table><tr><td>Model</td><td>|3D/4D Avg. Acc. (9)</td><td> $\pmb { \Delta } _ { \mathrm { D e p t h } }$ </td><td>Depth Relative_Depth Spatial_Relation</td><td></td></tr><tr><td>+ Vanilla Grounding</td><td>-0.2</td><td>-0.1</td><td>-0.5</td><td>+0.8 -0.7</td></tr><tr><td>+ Z Reward</td><td>+0.3</td><td>+1.3</td><td> $+ 0 . 2$ </td><td> $+ 3 . 2$  +0.6</td></tr></table>

## 5.5 Visualization

Analysis of Factorized Rewards. Fig. 4 illustrates how factorized rewards improve structured spatial reasoning. In multi-view correspondence, SFT models rely on size heuristics and fail to preserve identity. After RL, the model follows a cross-view CoT (anchor→transfer→verify), correctly tracking the target person across viewpoints, driven by the XY reward. In single-view spatial relation, the model adopts a depth-aware CoT, reasoning via occlusion and layering rather than size of the potted plant, enabled by the Z reward. In video camera motion, the model develops a temporal CoT by analyzing coherent foreground–background shifts around the mug, guided by the T reward. Overall, the factorized rewards promote consistent single-view and multi-view reasoning across space and time.

![](images/8e6345434a87302a7e7b0691551d97ee820cae452953215f334f70d35d8ac097.jpg)  
Fig. 5: The example of video route plan. Given a sequence of egocentric observations, our model infers a consistent navigation plan by reconstructing the underlying spatial layout and camera motion, efectively acting as a latent world model.

Generalization to Video Route Plan. The visual evidence from the FactoSR-RL-8B model in Fig. 5 demonstrates that the integration of camera motion data as a structural prior, coupled with a temporal cycle consistency reward in RL framework, significantly generalizes well to video route planning. While standard VLMs like Qwen3-VL often fail to maintain spatial orientation, as seen in its false choice of “Turn Back”, the RL-tuned agent leverages motion parallax and yaw rotation (detailed in Steps 1 and 2) to construct a coherent topological map of the environment. Crucially, the model is encouraged to maintain a bijective mapping between temporal video observations and the robot’s latent spatial state. This ensures that the transition from the closet to the bed, and ultimately to the heater, is grounded in cross-frame verification rather than isolated object recognition. As a result, the agent efectively decodes the underlying scene dynamics (Step 4), correctly identifying that “Turn Left, Turn Right” is the only path consistent with the reconstructed 4D navigation manifold.

## 6 Conclusion

Spatial intelligence requires the ability to reason freely within the vast 4D world, yet most vision-language models are shaped under a single-view paradigm that collapses this rich structure into flat image observations. To move beyond this limitation, we introduce FactoSR, a reinforcement learning framework that factorizes spatial reasoning into complementary dimensions of plane, depth, and time, allowing models to reason over the latent spatial structure underlying visual observations. Results on diverse 3D and 4D spatial benchmarks show that FactoSR pushes forward spatial reasoning, encourage multi-modal models to observe, reason, and interact with the latent structure of the physical world.

## Acknowledgements

This work is supported by the Program of Guangdong Education Department. This work is also supported by Joy Future Academy and JD.com for research funding and computing resources.

## References

1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv:2303.08774 (2023)

2. Bai, S., Cai, Y., Chen, X., Huang, Q., Li, K., Lin, Z., Zhu, K., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025), https://arxiv.org/ abs/2511.21631

3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv:2502.13923 (2025)

4. Baruch, G., Chen, Z., Dehghan, A., Dimry, T., Feigin, Y., Fu, P., Gebauer, T., Jofe, B., Kurz, D., Schwartz, A., et al.: Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv:2111.08897 (2021)

5. Batra, H., Tu, H., Chen, H., Lin, Y., Xie, C., Clark, R.: Spatialthinker: Reinforcing 3d reasoning in multimodal llms via spatial rewards. arXiv preprint arXiv:2511.07403 (2025)

6. Bertasius, G., Wang, H., Torresani, L.: Is space-time attention all you need for video understanding? In: Icml. vol. 2, p. 4 (2021)

7. Bigverdi, M., Luo, Z., Hsieh, C.Y., Shen, E., Chen, D., Shapiro, L.G., Krishna, R.: Perception tokens enhance visual reasoning in multimodal language models. In: CVPR (2025)

8. Cai, W., Ponomarenko, I., Yuan, J., Li, X., Yang, W., Dong, H., Zhao, B.: Spatialbot: Precise spatial understanding with vision language models. arXiv:2406.13642 (2024)

9. Cai, Z., Wang, R., Gu, C., Pu, F., Xu, J., Wang, Y., Yin, W., Yang, Z., Wei, C., Sun, Q., et al.: Scaling spatial intelligence with multimodal foundation models. arXiv preprint arXiv:2511.13719 (2025)

10. Chang, A., Dai, A., Funkhouser, T., Halber, M., Niessner, M., Savva, M., Song, S., Zeng, A., Zhang, Y.: Matterport3d: Learning from rgb-d data in indoor environments. arXiv:1709.06158 (2017)

11. Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., Xia, F.: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In: CVPR (2024)

12. Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al.: Are we on the right way for evaluating large vision-language models? arXiv:2403.20330 (2024)

13. Chen, Z., Zhang, M., Yu, X., Luo, X., Sun, M., Pan, Z., Feng, Y., Pei, P., Cai, X., Huang, R.: Think with 3d: Geometric imagination grounded spatial reasoning from limited views. arXiv preprint arXiv:2510.18632 (2025)

14. Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al.: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv:2412.05271 (2024)

15. Cheng, A.C., Yin, H., Fu, Y., Guo, Q., Yang, R., Kautz, J., Wang, X., Liu, S.: Spatialrgpt: Grounded spatial reasoning in vision-language models. NeurIPS (2024)

16. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv:2507.06261 (2025)

17. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet: Richly-annotated 3d reconstructions of indoor scenes. In: CVPR (2017)

18. Deng, Y., Bansal, H., Yin, F., Peng, N., Wang, W., Chang, K.W.: Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. arXiv e-prints pp. arXiv–2503 (2025)

19. Fan, Z., Zhang, J., Li, R., Zhang, J., Chen, R., Hu, H., Wang, K., Qu, H., Wang, D., Yan, Z., et al.: Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv:2505.20279 (2025)

20. Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N.A., Ma, W.C., Krishna, R.: Blink: Multimodal large language models can see but not perceive. In: ECCV (2024)

21. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv:2501.12948 (2025)

22. Kuhn, H.W.: The hungarian method for the assignment problem. Naval research logistics quarterly 2(1-2), 83–97 (1955)

23. Li, H., Li, D., Wang, Z., Yan, Y., Wu, H., Zhang, W., Shen, Y., Lu, W., Xiao, J., Zhuang, Y.: Spatialladder: Progressive training for spatial reasoning in visionlanguage models. arXiv preprint arXiv:2510.08531 (2025)

24. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? In: ECCV (2024)

25. Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.C., Liu, C.L., Jin, L., Bai, X.: Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences 67(12), 220102 (2024)

26. Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., Wang, J.: Visualrft: Visual reinforcement fine-tuning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2034–2044 (2025)

27. LLaVA-Lab: Llava-onevision-data. https://huggingface.co/datasets/lmmslab/LLaVA-OneVision-Data (2024)

28. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

29. Ma, W., Wang, C., Yuan, R., Chen, H., Dai, N., Zhou, S.K., Yang, Y., Yuille, A., Chen, J.: Causalspatial: A benchmark for object-centric causal spatial reasoning. arXiv preprint arXiv:2601.13304 (2026)

30. Ma, W., Chen, H., Zhang, G., Chou, Y.C., de Melo, C.M., Yuille, A.: 3dsrbench: A comprehensive 3d spatial reasoning benchmark. arXiv:2412.07825 (2024)

31. Ma, W., Chou, Y.C., Liu, Q., Wang, X., de Melo, C., Xie, J., Yuille, A.: Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning. arXiv:2504.20024 (2025)

32. Ouyang, K., Liu, Y., Wu, H., Liu, Y., Zhou, H., Zhou, J., Meng, F., Sun, X.: Spacer: Reinforcing mllms in video spatial reasoning. arXiv:2504.01805 (2025)

33. Qiu, L., Ning, S., Sun, J., He, X.: Noisygrpo: Incentivizing multimodal cot reasoning via noise injection and bayesian estimation. arXiv preprint arXiv:2510.21122 (2025)

34. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300 (2024)

35. Shen, C., Wei, W., Qu, X., Cheng, Y.: Satori-r1: Incentivizing multimodal reasoning with spatial grounding and verifiable rewards. arXiv e-prints pp. arXiv–2505 (2025)

36. Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q., Zhang, Z., Zhao, K., Zhang, Q., et al.: Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615 (2025)

37. Shen, Z., Yu, Q., Li, J., Ji, W., Chen, Q., Tang, S., Zhuang, Y.: Evolvedgrpo: Unlocking reasoning in lvlms via progressive instruction evolution. In: The Thirtyninth Annual Conference on Neural Information Processing Systems

38. Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., Wu, C.: Hybridflow: A flexible and eficient rlhf framework. In: Proceedings of the Twentieth European Conference on Computer Systems. pp. 1279–1297 (2025)

39. Team, G.R., Abeyruwan, S., Ainslie, J., Alayrac, J.B., Arenas, M.G., Armstrong, T., Balakrishna, A., Baruch, R., Bauza, M., Blokzijl, M., et al.: Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020 (2025)

40. Team, M.V.: Mimo-vl technical report (2025), https://arxiv.org/abs/2506. 03569

41. Tong, P., Brown, E., Wu, P., Woo, S., IYER, A.J.V., Akula, S.C., Yang, S., Yang, J., Middepogu, M., Wang, Z., et al.: Cambrian-1: A fully open, vision-centric exploration of multimodal llms. NeurIPS (2024)

42. Wang, H., Qu, C., Huang, Z., Chu, W., Lin, F., Chen, W.: Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837 (2025)

43. Wang, P., Ling, H.: Svqa-r1: Reinforcing spatial reasoning in mllms via viewconsistent reward optimization. arXiv preprint arXiv:2506.01371 (2025)

44. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv:2409.12191 (2024)

45. Wei, Y., Zhao, L., Sun, J., Lin, K., Yin, J., Hu, J., Zhang, Y., Yu, E., Lv, H., Weng, Z., et al.: Open vision reasoner: Transferring linguistic cognitive behavior for visual reasoning. arXiv preprint arXiv:2507.05255 (2025)

46. Wu, D., Liu, F., Hung, Y.H., Duan, Y.: Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747 (2025)

47. Wu, H., Huang, X., Chen, Y., Zhang, Y., Wang, Y., Xie, W.: Spatialscore: Towards unified evaluation for multimodal spatial understanding. arXiv e-prints pp. arXiv– 2505 (2025)

48. Wu, J., Guan, J., Feng, K., Liu, Q., Wu, S., Wang, L., Wu, W., Tan, T.: Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv:2506.09965 (2025)

49. x.ai: Grok-1.5 vision preview (2024), https://x.ai/blog/grok-1.5v

50. Xia, J., Zang, Y., Gao, P., Li, S., Zhou, K.: Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning. arXiv preprint arXiv:2505.14677 (2025)

51. Yang, J., Yang, S., Gupta, A.W., Han, R., Fei-Fei, L., Xie, S.: Thinking in space: How multimodal large language models see, remember, and recall spaces. In: CVPR (2025)

52. Yang, R., Zhu, Z., Li, Y., Huang, J., Yan, S., Zhou, S., Liu, Z., Li, X., Li, S., Wang, W., et al.: Visual spatial tuning. arXiv preprint arXiv:2511.05491 (2025)

53. Yang, S., Yang, J., Huang, P., Brown, E., Yang, Z., Yu, Y., Tong, S., Zheng, Z., Xu, Y., Wang, M., et al.: Cambrian-s: Towards spatial supersensing in video. arXiv preprint arXiv:2511.04670 (2025)

54. Yang, S., Xu, R., Xie, Y., Yang, S., Li, M., Lin, J., Zhu, C., Chen, X., Duan, H., Yue, X., et al.: Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv:2505.23764 (2025)

55. Yang, Y., Wang, Z.Y., Liu, Q., Sun, S., Wang, K., Chellappa, R., Zhou, Z., Yuille, A., Zhu, L., Zhang, Y.D., et al.: Medical world model. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8319–8329 (2025)

56. Yeh, C.H., Wang, C., Tong, S., Cheng, T.Y., Wang, R., Chu, T., Zhai, Y., Chen, Y., Gao, S., Ma, Y.: Seeing from another perspective: Evaluating multi-view understanding in mllms. arXiv preprint arXiv:2504.15280 (2025)

57. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset of 3d indoor scenes. In: ICCV (2023)

58. Yin, B., Wang, Q., Zhang, P., Zhang, J., Wang, K., Wang, Z., Zhang, J., Chandrasegaran, K., Liu, H., Krishna, R., et al.: Spatial mental modeling from limited views. arXiv:2506.21458 (2025)

59. Yu, E., Lin, K., Zhao, L., Yin, J., Wei, Y., Peng, Y., Wei, H., Sun, J., Han, C., Ge, Z., et al.: Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954 (2025)

60. Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al.: Dapo: An open-source llm reinforcement learning system at scale. arXiv:2503.14476 (2025)

61. Zhang, X., Wang, J., Cheng, Z., Zhuang, W., Lin, Z., Zhang, M., Wang, S., Cui, Y., Wang, C., Peng, J., et al.: Srpo: A cross-domain implementation of large-scale reinforcement learning on llm. arXiv preprint arXiv:2504.14286 (2025)

62. Zhou, J., Ji, J., Chen, B., Sun, J., Chen, W., Hong, D., Han, S., Guo, Y., Yang, Y.: Generative rlhf-v: Learning principles from multi-modal human preference. arXiv preprint arXiv:2505.18531 (2025)

63. Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al.: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv:2504.10479 (2025)