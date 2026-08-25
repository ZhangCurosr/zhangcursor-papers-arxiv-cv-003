# UniMem: Unifying Multimodal Memory and Control for Vision-Language-Action Models

Lars Osterberg<sup>1</sup>, Maggie Wang<sup>1</sup>, and Mac Schwager<sup>1</sup>

Abstract— While Vision-Language-Action (VLA) models have leveraged internet-scale pretraining and task-focused finetuning to achieve strong performance on long-horizon tasks, they often struggle with non-Markovian tasks that require memory. Existing approaches to memory typically involve additional Vision-Language-Models (VLMs) for long-term memory management, introducing a memory bottleneck and a fractured training pipeline. Conditioning on multiple historical frames can provide the VLA with access to more descriptive features of past scenes, but can degrade performance if frames are chosen at arbitrary, fixed intervals. To address these limitations, we present UNIMEM, a framework that unifies high-level, multimodal memory and lowlevel control under one backbone. UNIMEM employs an event classifier for memory updates, a keyframe encoder for dense spatial memory, and a keyframe caching technique to minimize overhead during policy rollouts. We evaluate UNIMEM across five simulation and four hardware tasks targeting sequential and spatial memory, demonstrating that our unified, single-model system outperforms fixed-interval image sampling baselines (93.4% vs. 68.2%) in simulation and hierarchical baselines (80.0% vs. 43.5%) in hardware, while offering faster inference and a simple training pipeline for easy adoption. Project website: https://losterberg3.github.io/unimem-vla/

## I. INTRODUCTION

While vision-language-action (VLA) models [1]–[7] have achieved remarkable progress in robot manipulation, their success is largely confined to short-duration, Markovian tasks due to an absence of architectural memory. For instance, state-of-the-art policies [8] typically struggle on rudimentary tasks such as picking up an object and returning it to its starting position [9]–[11]. We characterize this failure mode as perceptual aliasing—when several moments in a training dataset share similar observations but demand entirely different actions depending on preceding events. At these functionally distinct but apparently similar phases, a VLA requires not only sequential memory to orient itself within the progress of a task, but spatial memory to retain pertinent information about the scene’s past. Motivated by human cognition and recent breakthroughs in vision-language models showing that temporal history is crucial for contextaware reasoning [12], we pose the central question: Can conditioning VLAs on multimodal memory directly unlock robust performance in non-Markovian tasks?

Existing methods [11], [13]–[15] incorporate memory in VLA systems via an additional Vision-Language Model (VLM) that maintains textual summaries of completed milestones and produces sub-task commands for the VLA to execute. Although effective in simpler tasks, this factorized approach creates artificial silos that isolate the VLA from the rich historical context required for fine-grained reasoning. Furthermore, these approaches discard essential spatiotemporal information inherently encoded in past video frames—data that VLAs are already structured to process and act on.

![](images/f7e5db984168d32a02cf9d6fc197d9f19970bedec052601bcfe96ada30e91ef5.jpg)  
Fig. 1: Overview of UNIMEM. An event classifier $( f _ { \phi } )$ detects sparse sub-task transition events from the backbone latent space, autoregressively updating textual memory and a cache of precomputed keyframe hidden states. This routes to our keyframe encoder and a tokenizer, providing event-driven, multimodal memory directly to the backbone and action expert while unifying memory and control in one self-sustaining, low-latency VLA.

To bridge this gap, we present UNIMEM, a unified, multimodal memory framework that equips vision-languageaction models with direct access to historical context across both language and vision. UNIMEM is unified in two key respects: it integrates historical textual events with visual context, and it unifies memory and control under a shared backbone, as shown in Figure 1. Crucially, UNIMEM positions the memory loop adjacent to the control loop to seamlessly integrate multimodal history, minimizing inference latency in non-Markovian tasks.

Indeed, a longer historical context can also introduce spurious correlations [16]–[18]: a model that sees past observations regardless of their relevance will learn to exploit coincidental structure in the training data rather than the desired relationship between memory and action. In contrast to prior work, UNIMEM introduces an event-driven memory architecture that retains task-relevant textual and visual context to guide future action prediction. By separating meaningful history from task-irrelevant observations, this design mitigates this causal confusion and extends the VLA’s temporal efficiency. Specifically, UNIMEM employs a lightweight event classifier to decode latent event signatures and save memories. Since memory grows incrementally, context can be efficiently cached—preserving real-time inference speeds by eliminating expensive attention-based computations over raw images. Furthermore, because frames are cached only when internal VLA features trigger the classifier head, our event detector serves a dual purpose as an in-distribution gating mechanism, ensuring the policy conditions exclusively on keyframes from high-confidence task milestones.

Our core contributions are as follows:

• Multi-Modal Memory Conditioning: We present a framework that equips VLAs with a unified multimodal memory across both textual and visual modalities for robust performance in challenging non-Markovian tasks. By integrating the VLA’s memory and control loops, UNIMEM enables a seamless training pipeline and realtime inference.

• Event-Driven Architecture: We introduce a lightweight event classifier that dynamically detects task-critical milestones within the VLA’s latent space and progressively constructs task-relevant memory from informative language and vision keyframes.

• Real-Time Inference via Keyframe Caching: By caching keyframes across inference steps, UNIMEM preserves single-frame inference speeds (∼90 ms) despite conditioning on rich visual histories, achieving a 6× speedup over hierarchical memory baselines.

Through extensive simulation and hardware experiments, we demonstrate the robust performance of UNIMEM compared to state-of-the-art baselines, e.g., MemER [14]. Across nine benchmark tasks, UNIMEM achieves approximately 93.4% and 80.0% average success rates in simulation and realworld hardware, respectively, compared to 72.6% and 43.5% success rates achieved by the best-performing baselines.

## II. RELATED WORK

## A. Recurrent & Sequence-Based Robotic Memory

To tackle partial observability and non-Markovian dynamics, early approaches incorporated temporal sequence models over continuous observation histories. Standard methods explored recurrent architectures like LSTMs and RNNs over temporal sequences [19], [20] or state-space belief encoders [21] to maintain implicit hidden states across time. Some methods maintain memory through proprioception [22], although this is limited to tasks where past scene states are not necessary for future task decisions. Other frameworks use transformer-based architectures to ingest temporal sequences of frames [23]–[25], although the length of these sequences are limited and do not necessarily capture all task-pertinent moments due to fixed-interval sampling. While effective in low-dimensional or short-horizon domains, blindly ingesting dense observation streams scales poorly with horizon length—leading to severe computational overhead, context degradation, and an inability to selectively retain task-critical events.

## B. Decoupled & Multimodal Memory in VLA Paradigms

VLA models generalize across diverse robotic tasks by scaling up internet-scale vision-language pretraining to low-level motor control. While foundational works display remarkable semantic reasoning and zero-shot generalization to novel objects and scenes across different embodiments [4], [6], [8], [26], they provide actions based solely on the immediate camera frame and task instruction. While highly effective for short-horizon tasks and fully observable decision processes, this reactive nature is fundamentally limiting in tasks that demand historical context. To equip these models with longhorizon memory, recent works vary in both the representation stored and how memory conditions the policy.

For navigation, hybrid systems leverage a high-level VLM and a low-level VLA to construct topological environment graphs [27]. For tabletop manipulation, frameworks like SAM2Act [28] maintain spatial memory of past scene states via a latent memory bank. MemER [14] saves historically relevant keyframes, but its decoupled architecture prevents the low-level actor from conditioning directly on memory. Some works have managed to unify the hierarchical approach in one VLA for complicated multi-stage manipulation tasks [29], with thinking and acting modes swapped intermittently; other recent methods condition the low-level policy on temporal image sequences [30], [31]. MEM [32] introduces a memory architecture for VLAs that aligns more closely with prevailing paradigms. Specifically, MEM leverages a VLM to generate textual summaries of past events and subsequently assigns sub-tasks to the VLA. However, this approach perpetuates the artificial silos separating VLAs from critical historical context. Furthermore, by combining sub-task commands with fixed-interval historical frames, MEM exposes the VLA to visual distractions.

While these methods touch on many aspects of memory individually, none condition the VLA directly on memory from multiple modalities, under one unified architecture. In contrast to MEM and prior methods, UNIMEM unifies memory retrieval and action generation under a single backbone, enabling event-driven visual and textual conditioning while maintaining low-latency control.

## III. UNIFIED MEMORY FOR VLAS

## A. Problem Formulation and Method Overview

We consider a long-horizon, non-Markovian robotic manipulation task framed as a Partially Observable Markov Decision Process (POMDP). At each timestep t, the agent receives an observation of its environment $o _ { t }$ and a high-level task instruction g. A standard reactive policy, which conditions exclusively on the current observation $\scriptstyle ( \pi ( a _ { t : t + H } | o _ { t } , g ) )$ , fails when tasks require historical context. We characterize this failure mode as perceptual aliasing. Formally, perceptual aliasing occurs when two global steps i and $j$ in a demonstration dataset yield near-identical observations $( o _ { i } \approx o _ { j } )$ , but require different actions $( a _ { i } \neq a _ { j } )$ depending on task history. Resolving this ambiguity requires conditioning on a history of past states, mapping $O t - T { : } t$ to the correct low-level motor action $a _ { t }$

UNIMEM addresses this problem by maintaining two complementary forms of history: textual memory $( \boldsymbol { \mathcal { M } } _ { t } )$ that records discrete task events and visual keyframe memory $( \mathcal { H } _ { t } )$ that retains key spatial information. Both forms are updated online by an event classifier integrated directly into the VLA backbone and jointly condition subsequent action prediction.

## B. Event-Driven Multimodal Memory

We implement UNIMEM on top of $\pi _ { 0 . 5 } \ [ 8 ]$ , an open source VLA consisting of a PaliGemma backbone and a Gemma action expert [33]. To compress execution history into discrete, task-relevant events, we define the event vocabulary $\mathcal { E } _ { \mathrm { a l l } } =$ $\mathcal { E } \cup \{ \mathrm { n u l l } \}$ , where E represents the subset of valid, task-critical milestones (e.g., "grabbed box"). As shown in Figure 1, we append an event classifier head $f _ { \phi } ,$ , implemented as an MLP, directly onto the final layer latent representation $z _ { t }$ extracted from the VLA backbone. The classifier produces a probability distribution over event classes, and the predicted event $e _ { t } \in \mathcal { E } _ { \mathrm { a l l } }$ at step t is chosen via the most likely class index:

$$
e _ { t } = \mathcal { E } _ { \mathrm { a l l } } \left[ \operatorname { a r g m a x } f _ { \phi } ( z _ { t } ) \right] .\tag{1}
$$

We use $z _ { t }$ for event prediction because it is the final-layer representation that would otherwise be used for autoregressive text generation in the underlying VLM. Rather than using this representation to predict a language token, we pass $z _ { t }$ through the MLP event classifier $f _ { \phi }$ to predict the event $e _ { t }$ . Because $f _ { \phi }$ is trained jointly with the VLA, the event classification loss in Eq. (2) also updates the shared backbone used for action generation. This couples memory learning with control—rather than separating them across hierarchical modules—and prioritizes latent features that support both objectives.

At each step t, if the predicted class $e _ { t } \in \mathcal { E } ( \mathrm { i . e . , } e _ { t } \neq$ null), the framework triggers a memory update and $e _ { t }$ is dynamically appended in the form of natural language to our textual history $\mathcal { M } _ { t }$ , modifying the language instruction for the subsequent step: $\boldsymbol { \mathcal { M } } _ { t + 1 } = \left[ \boldsymbol { \mathcal { M } } _ { t } \parallel e _ { t } \right]$ . At the start of every rollout, we initialize $\mathcal { M } _ { t } = \emptyset$ . This tightly integrated loop provides explicit textual context that cleanly breaks the visual ambiguity of downstream states and anchors the model to its own task progress.

The same detection of a task-critical event also triggers a visual memory update. When $e _ { t } \in \mathcal { E }$ , UNIMEM stores the corresponding multi-view visual observation $I = \{ I ^ { \mathrm { w r i s t } } , I ^ { \mathrm { e x t } } \}$ These observations are appended to our collection of keyframes, defined as $\mathcal { H } _ { t } = \{ I _ { k } \} _ { k \in \mathcal { K } }$ , where $\kappa$ represents the set of discrete, non-consecutive event timestamps along with the current time step t. At the start of every rollout, we initialize $\mathcal { H } _ { t }$ with only the initial scene image. Due to compute limitations during training, the size of this history is capped to three past milestones $( | \mathcal { K } | \leq 4 )$ , dropping the oldest frame when exceeding this limit during rollouts. Depending on event sparsity, this can equate to over a minute of visual memory.

Crucially, the event classifier head $f _ { \phi }$ operates on the unified latent representation from both textual memory $\mathcal { M } _ { t }$ and visual history $\mathcal { H } _ { t }$ . Because $z _ { t }$ is constructed via crossattention over both the tokens from textual memory $\mathcal { M } _ { t }$ and keyframes $\mathcal { H } _ { t }$ , the event classifier inherently conditions on the entire multimodal history. This creates a recursive memory loop where prior milestones directly inform the detection of subsequent events. The updated policy formulation is thus defined as $\pi ( a _ { t : t + H } , e _ { t } | \mathcal { H } _ { t } , \mathcal { M } _ { t } , q _ { t } , g )$ , where $q _ { t }$ is proprioception.

## C. Efficient Keyframe Encoding and Caching

To ingest this keyframe bank $\mathcal { H } _ { t }$ without degrading execution speeds, we modify $\pi _ { 0 . 5 } \mathrm { ^ { \circ } s }$ vision encoder (SigLIP [34]). Specifically, we interleave causal temporal self-attention every few layers of the standard spatial attention stack; at these layers, the temporal attention pass reuses that same layer’s own query/key/value and layer-norm weights, a configuration closely adapted from [32], [35]. Unlike prior approaches that sample frames at fixed temporal intervals and rely on separate pretraining, we train our keyframe encoder end-to-end during fine-tuning. Because keyframes explicitly isolate semantically meaningful events rather than potentially uninformative fixed interval frames, they provide a concentrated signal for taskrelevant memory representations without requiring a separate pretraining phase.

To avoid repeatedly re-encoding historical keyframes, we cache their intermediate representations in a feature cache $\mathcal { H } _ { \mathrm { c a c h e } }$ for future timesteps to attend to. Specifically, we cache their hidden states prior to the positional embeddings and layer-normalizations applied during each temporal attention step. When a new frame arrives, we simply shift the positional embedding applied to each cached entry without recomputing costly spatial attention at each layer; the current image alone forms the query for temporal attention. This allows the encoder to project dense spatiotemporal representations into the embedding of the VLA backbone while preserving a real-time control loop as temporal context scales.

## D. Data and Training

To train our policies end-to-end, we need a robot action dataset (containing $a _ { t : t + H } , o _ { t } , q _ { t }$ , and $g )$ labeled with $e _ { t }$ and $\mathcal { M } _ { t }$ . Once demonstrations have been collected for a specific task, we prompt an agent (Claude Sonnet 5.0 [36]) to generate a script that parses our dataset for action signatures with the corresponding $e _ { t } \in \mathcal { E } _ { \mathrm { a l l } }$ (i.e., a robot pitching upwards means scooping, the gripper closing means a bottle is being grasped, etc.). Frames outside these event windows get labeled with the null class by our script. For simplicity, we predefine our event vocabulary ${ \mathcal { E } } _ { \mathrm { a l l } }$ , although this could be constructed autonomously using methods like RoboInter [37] in future work. To prevent $\mathcal { M } _ { t }$ from leaking the target during training, we only update $\mathcal { M } _ { t }$ with an event once the entire window has passed. After the script is run on our dataset, a human verifies a subset of the demonstration videos has been labeled correctly and refines the script through more prompting if necessary. We build $\mathcal { H } _ { t }$ at training time, sampling a single frame uniformly from each past event window along with $I _ { t } .$ Refer to Figure 2 for a conceptual overview of this process.

![](images/cf3d3fff455e29eed998f8d4b56492c402b5b99be290fd130ab8c59429692778.jpg)  
Fig. 2: Overview of UNIMEM’s automated data labeling procedure. We prompt an agent to generate a script for the given task that labels demonstrations with $e _ { t }$ and $\mathcal { M } _ { t } .$ . Multiple frames in the window $W _ { i }$ are labeled with the corresponding event, and $\mathcal { M } _ { t }$ is only updated once this window has passed. At training time, a keyframe from each past $W _ { i }$ along with $I _ { t }$ is used to build $\mathcal { H } _ { t }$

With our dataset curated, we train the policy end-to-end with a two-term objective: the standard action-chunking loss and an auxiliary cross-entropy on the event classifier head. We keep the flow-matching objective of $\pi _ { 0 . 5 }$ [8] unchanged. The action expert integrates the chunk $a _ { t : t + H }$ along the denoising velocity field conditioned on $o _ { t } ;$ we denote this term $\mathcal { L } _ { \mathrm { a } } ( \theta )$ . We supervise the event classifier head $f _ { \phi }$ with a class-weighted cross-entropy loss against the automatically generated label $\textstyle e _ { t } \colon$

$$
\mathcal { L } _ { \mathrm { e } } ( \theta , \phi ) = - w ( e _ { t } ) \log p _ { \phi } ( e _ { t } \mid z _ { t } )\tag{2}
$$

where $w ( e _ { t } ) = 1 \mathrm { i f } e _ { t } \in \mathcal { E }$ and $w _ { \mathrm { n u l l } }$ if $e _ { t } ~ = ~ \mathrm { n u l l }$ . Null frames dominate the dataset (most timesteps are not events), so we downweight them by setting $w _ { \mathrm { n u l l } } = 0 . 0 2$ rather than masking them out of the loss entirely. The distinction matters at deployment: masking would leave the head unsupervised on precisely the frames where it must decline to fire. Training on null frames teaches the head to withhold a prediction, while the low weight keeps it from collapsing onto the majority class. The asymmetry is deliberate, since memory is appendonly: a missed event can still be caught at the next timestep, whereas a single false positive writes a wrong entry into $\mathcal { M } _ { t }$ and an incorrect keyframe into $\mathcal { H } _ { t }$ that persist for the remainder of the rollout. The two terms are optimized jointly,

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { a } } + \lambda \mathcal { L } _ { \mathrm { e } } , \qquad \lambda = 0 . 1 . } \end{array}\tag{3}
$$

Since $z _ { t }$ is a shared backbone representation rather than a detached feature, the classification gradient flows back through the language-model trunk, realizing the auxiliary supervision described in Sec. III-B.

## IV. EXPERIMENTS AND ANALYSES

## A. Experimental Setup

We evaluate UNIMEM on nine memory tasks (shown in Figure 3), split across simulation and hardware. We use the robosuite environment with 7DoF Franka Panda for simulation [38] and a UFactory xArm6 for hardware. In both domains, we equip our robot with wrist-mounted and external image streams. We use real time chunking for smooth inference on hardware [39], querying UNIMEM at ∼10 Hz. We retain up to three event keyframes $( | \mathcal { K } | = 3 )$ for simulation tasks and four $( | \mathcal { K } | = 4 )$ for hardware tasks. During experiments, a human observes each rollout, either from a saved video in simulation or in real time, and decides binary subtask success.

## B. Baselines and Ablations

We compare UNIMEM against VLA memory baselines and ablations on the memory input. In simulation, we compare against $\pi _ { 0 . 5 }$ augmented with a video encoder, with frames sampled at 6-second intervals and processed using our keyframe encoder architecture $( | \mathcal { K } | = 4 )$ . This baseline tests how arbitrary frame sampling performs relative to eventdriven memory.

On hardware, we compare against MemER with $\pi _ { 0 . 5 }$ as the primary baseline, characterizing a hierarchical memory system [14]. To isolate the contributions of textual and keyframe memory, we train ablations with no memory at all (vanilla $\pi _ { 0 . 5 } ) .$ , text memory only $( I _ { t }$ images only), and keyframe memory only (no $\boldsymbol { \mathcal { M } } _ { t } )$ . All ablations are trained with auxiliary classifier supervision.

![](images/e9acb41a2079f2150e06d615968f113130a38d8333359018a55dcc1372fda63f.jpg)  
Fig. 3: Overview of Tasks. (Top) Manipulation tasks in robosuite. (Bottom) Real-world tasks on the xArm6 setup.

## C. Simulation Experiments and Ablations

We evaluate UNIMEM on five simulation tasks designed to verify its ability to overcome perceptual aliasing and outperform fixed-interval visual history baselines across sequential and spatial memory settings:

• UpDown: Pick up a box and put it back down once.

• UpDown3Times: Pick up a box and put it back down three times.

• OccludedTap: Pick up a box, place it in one of two bins, retract until the contents of both bins are occluded, and then tap the bin containing the box.

• UpDownSpatial: Pick up a box from a table and then place it back in its original location. Performance is measured using a continuous score based on the final placement position error.

• PlateRecall: Pick up a box from one of the four plates, place it to the side, and then tap the plate that originally had the box.

As shown in Table I, UNIMEM achieves a 93.4% simulation average across non-Markovian evaluation environments, substantially outperforming the fixed-window video baseline (π<sub>0.5</sub>+V.E., 68.2%). Basic non-Markovian tasks like UpDown and OccludedTap require minimal historical context and verify that history conditioning does not impede low-level control.

Visual memory alone struggles in counting tasks; textual memory provides critical progress information by compressing history into discrete events. In UpDown3Times, keyframe-only conditioning fails (23%) due to limited temporal reach, while text-only conditioning reaches 93% success.

In spatial memory tasks, visual and textual memory is similarly shown to improve task performance. In OccludedTap, the keyframe ablation performs 12% better than the text ablation; we hypothesize that this is because keyframes provide many memory pathways for the policy to exploit while text is left to just one. In continuous spatial grounding (UpDownSpatial), text-only drops to 30% without geometric awareness of past scenes. While the video encoder and keyframes offer better spatial guidance (49% and 52% respectively), they still occasionally lose track of task progression; UNIMEM, which combines keyframes with textual memory, resolves both failure modes, raising success to 79%. Finally, in PlateRecall, text-only achieves 20% success— roughly matching the 1-in-4 chance of correct plate selection.

TABLE I: Simulation Results & Ablations. Success rates (%) (N = 25). (†) reports mean subtask success due to longhorizon complexity. (‡) reports spatial accuracy.
<table><tr><td>Task</td><td>π0.5 + V.E.</td><td>N.M.</td><td>T.O.</td><td>K.0.</td><td>Ours</td></tr><tr><td>UpDown</td><td>84</td><td>52</td><td>96</td><td>92</td><td>100</td></tr><tr><td>UpDown3Times†</td><td>16</td><td>22</td><td>93</td><td>23</td><td>96</td></tr><tr><td>OccludedTap</td><td>96</td><td>60</td><td>88</td><td>100</td><td>96</td></tr><tr><td>UpDownSpatial</td><td>49</td><td>6</td><td>30</td><td>52</td><td>79</td></tr><tr><td>PlateRecall</td><td>96</td><td>8</td><td>20</td><td>96</td><td>96</td></tr><tr><td>Avg. Performance</td><td>68.2</td><td>29.6</td><td>65.4</td><td>72.6</td><td>93.4</td></tr></table>

V.E. = Video Encoder, N.M. = No Memory, T.O. = Text Only, K.O. = Keyframe Only

TABLE II: Hardware Results & Ablations. Success rates (%) (N = 15).
<table><tr><td>Task</td><td>MemER</td><td>N.M.</td><td>T.O.</td><td>K.O.</td><td>Ours</td></tr><tr><td>HammerMeasure</td><td>87</td><td>13</td><td>53</td><td>53</td><td>87</td></tr><tr><td>BeanScoop</td><td>67</td><td>0</td><td>27</td><td>20</td><td>93</td></tr><tr><td>TableClean</td><td>13</td><td>0</td><td>0</td><td>47</td><td>80</td></tr><tr><td>TapScoopPour</td><td>7</td><td>7</td><td>7</td><td>27</td><td>60</td></tr><tr><td>Avg. Performance</td><td>43.5</td><td>5.0</td><td>21.8</td><td>36.8</td><td>80.0</td></tr></table>

N.M. = No Memory, T.O. = Text Only, K.O. = Keyframe Only

Keyframes and the video baseline match the full model (96%), demonstrating that visual memory cleanly resolves discrete spatial ambiguities once progress is more easily inferred.

## D. Real-world Experiments and Analyses

Our empirical evaluations of UNIMEM in real-world experiments explore the following three core research questions:

Q1) Does providing textual and visual memory to the VLA overcome the memory bottleneck of hierarchical systems that only condition the VLA on subtask commands?

Q2) How do textual and visual memories individually contribute to policy execution, and is joint conditioning necessary for task success?

Q3) Does keyframe caching enable UNIMEM to maintain near single-frame inference latencies and scalability?

We designed the following real-world tasks around these questions, testing both memory modalities and human interaction in long-horizon, non-Markovian settings:

• HammerMeasure: Measure the width of a hammer using a tape measure while controlling its retraction to prevent the hook from slipping from the hammer.

• BeanScoop: Pick up a spoon, put three scoops of beans into a bowl, and then place the spoon back.

• TableClean: Pick and place a bottle off a 60 cm by 80 cm table, pick up a sponge, wipe the bottle’s original location, and then place the sponge back.

• TapScoopPour: Wait for a human to tap one of the eight cups, pick up a spoon, scoop the beans, pour the beans into the cup that the human tapped, and then place the spoon back.

We also perform a simulated inference speed benchmark of UNIMEM to test Q3. Overall success on these tasks is detailed in Table II, while stage-by-stage analysis for select tasks is in Figure 4.

![](images/33c139548814c21d545e1f7f05524fcc638cdf19f23190df940c5981a1c1817f.jpg)  
Fig. 4: Cumulative Success Rates. Hardware tasks of BeanScoop, TableClean, and TapScoopPour.

1) Does UNIMEM overcome the memory bottleneck of hierarchical VLA systems?

We test UNIMEM and the hierarchical baseline in tasks of graduated difficulty, ranging in the extent direct memory access is needed. In HammerMeasure, both UNIMEM and MemER achieve matching success rates (87%), as high-level subtask commands are sufficient when task progression is simple and failure modes are purely mechanical due to the precision required to keep the hook on the hammer. However, as tasks grow in horizon and spatial complexity, the limits of hierarchical methods become evident.

In BeanScoop, MemER performs better than the ablations and proceeds to the third scoop in 67% of rollouts. However, after completing two scoop-pour cycles, MemER’s high-level planner may output an incorrect subtask command—instructing the robot to place the spoon when it should command a third scoop. By the time the high-level planner outputs a correction, the low-level policy is in an unrecoverable regime and ignores subsequent commands. In contrast, UNIMEM conditions the policy on memory directly at every control step. Continuous memory access prevents high-level misclassifications from happening in the first place, while near single-frame inference speeds ensure the robot continuously adjusts its trajectory before physical drift becomes uncorrectable, ultimately achieving 93.0% success.

In TableClean, we test continuous spatial memory. Although MemER conditions the high-level policy on keyframes and augments its commands to the low-level policy with cues like left, right, and center, it only achieves 13% success due to such a large area to select from. Only when keyframes and textual memory combine in full UNIMEM does the policy gain access to past visual states and wipe the bottle’s exact spot on the table 80% of the time, while missing by no more than 10 cm.

In TapScoopPour, MemER pours into the correct cup 7% of the time—even though MemER’s command vocabulary is augmented, this is still insufficient to disambiguate such a large number of cups. Only when we provide visual history directly to the low-level policy do we see large improvements, with full UNIMEM achieving 60% success. Unified, multimodal memory provides spatial and sequential awareness directly to our VLA, an ability that hierarchical policies clearly lack.

## 2) Is joint conditioning on textual and visual memory necessary for success?

We compare our ablations in tasks of varying lengths and memory requirements to evaluate contributions of both memory modalities. In HammerMeasure, we explicitly characterize perceptual aliasing. Without memory, the policy moves back and forth, struggling to distinguish task progress (13%). Text-only and keyframe-only ablations improve to 53%, but still struggle to distinguish and remember certain events. Only when both modalities combine in UNIMEM do we achieve robust memory (87%).

In BeanScoop, with no memory, the policy doesn’t stop scooping and pouring, collapsing on the most common signal from the dataset. With textual memory, long-horizon tracking improves to 27%; however, the policy often moves to the next scoop prematurely without adding the previous pour to its memory, leading to undercounting. The keyframe-only ablation ameliorates this, only proceeding to the next scoop once a pour has been committed to memory. However, it cannot count all scoop-pour cycles with a limited context window and achieves 20% success. When both modalities combine, we see robust long- and short-term memory, with UNIMEM achieving 93% success.

In TableClean, with no memory, the policy fails to even begin wiping as the grabbing and placing actions alias each other. With only text, the policy wipes in random locations and does not perform a full wiping motion. With keyframes, the policy is endowed with spatial memory, completing the correct wipe 53% of the time, but nonetheless suffers from infinite wipe sequences and fails to place the sponge. Only when keyframes and textual memory combine in full UNIMEM does the policy overcome these specific failure modes by gaining access to sequential and spatial memory.

In TapScoopPour, the policy without memory only proceeds to the grasp after the first tap 27% of the time, requiring multiple taps for the rest. This is because as soon as the human stops tapping, the robot loses its signal to start. In the text ablation, the policy manages to pour into the correct cup 7% of the time, essentially picking at random. Only with keyframes do we see an improvement, with that ablation achieving 27% success. Its errors are also qualitatively different: misses are to an adjacent cup rather than arbitrary, indicating that the keyframe encoder does supply the correct spatial content. Adding textual memory further refines performance, with full UNIMEM achieving exact cup selection 60% of the time. Across all tasks, neither the text- nor keyframe-only ablations matched the performance of full UNIMEM, demonstrating that textual and visual memory are complementary and essential for success on multimodal, non-Markovian tasks.

![](images/98c2804dbb2c64b100261a9ec31a37919739917b5e454897627a038160c7e84e.jpg)

![](images/ca465b49daad4347d0cdd9a8a4fbd9b277be66bfdd5c81e337dbc9636e8fe5fb.jpg)  
Fig. 5: Benchmarking inference latency on an RTX 4090. We compare inference speed when varying images per camera stream for 2 streams (left) and a simulated bimanual setup with 4 streams (right).

3) Does UNIMEM maintain near single-frame inference speeds?

To provide long-horizon memory with minimal computational overhead, UNIMEM leverages a hidden-state caching mechanism for keyframes. In a naive implementation, keyframes are retained as raw pixels and re-encoded through the vision backbone at every control step, leading to poor scaling as history grows. By instead caching the pre-computed keyframe representations prior to temporal self-attention, we eliminate redundant computations.

We benchmark the latency of our caching mechanism against an ablation without caching, single-frame $\pi _ { 0 . 5 }$ , and MemER’s dual-system VLM architecture across varying context window sizes for 2 and 4 camera streams to the left and right, respectively, of Figure 5. Even on a standard workstation GPU, maintaining a 16-keyframe memory context across four camera streams adds only ∼ 25 milliseconds of latency beyond the single-frame, 2-camera stream base policy. Although our experiments utilize contexts of up to 4 keyframes for 2-camera streams, these figures demonstrate that UNIMEM can scale to significantly longer context windows and different embodiments like a bimanual manipulator with 4-camera streams without sacrificing inference speed. Thus, in tasks significantly longer than the ones we evaluated UNIMEM on, the bottleneck would be dataset coverage and training compute, which we leave for future work.

## V. CONCLUSION

We present UNIMEM, a streamlined framework that unifies long-horizon spatial and sequential memory within a single VLA architecture. By using an event classifier to autoregressively update its multimodal history, UNIMEM avoids the memory bottleneck and high-latency of dualsystem architectures. Across nine tasks in simulation and hardware, our system demonstrates superior task success, simpler training pipelines, and low-latency real-world rollouts.

Despite its strong empirical performance, our framework has several limitations that highlight promising directions for future research. First, while UNIMEM demonstrates robust state tracking over multi-minute execution horizons, it has not yet been evaluated on extended tasks spanning tens of minutes or hours. Exciting future work includes extending this framework with memory editing mechanisms—such as pruning and consolidation—to maintain high-performance.

Second, our current keyframe extraction pipeline relies on automated offline labeling. While this eliminates some human labor when curating the dataset, future work could leverage unsupervised temporal clustering or reinforcement learning to let the policy autonomously determine which events warrant long-term retention.

Finally, UNIMEM maintains memory with only one temporal context that extends several minutes. Distinguishing between short and long-term memories, to not only remember past environmental states but also improve motor function and learn from mistakes, remains a promising direction for memory conditioning. Memory that lives through not only the input stream, but also in the weights of the policy itself, could make memory more natural and human-like.

## ACKNOWLEDGMENT

Toyota Research Institute provided funds to support this work. M. Wang is supported by the NASA NSTGRO Fellowship.

## REFERENCES

[1] A. Brohan et al., “RT-1: Robotics transformer for real-world control at scale,” in Robotics: Science and Systems (RSS), 2023.

[2] Octo Model Team et al., “Octo: An open-source generalist robot policy,” in Proceedings of Robotics: Science and Systems, Delft, Netherlands, 2024.

[3] A. Brohan et al., “RT-2: Vision-language-action models transfer web knowledge to robotic control,” in Conference on Robot Learning (CoRL), 2023.

[4] K. Black et al., “π<sub>0</sub>: A vision-language-action flow model for general robot control,” in International Conference on Learning Representations (ICLR), 2025.

[5] P. Intelligence et al., “π<sub>0.7</sub>: a steerable generalist robotic foundation model with emergent capabilities,” arXiv preprint arXiv:2604.15483, 2026. [Online]. Available: https://arxiv.org/abs/2604.15483

[6] NVIDIA GR00T Team et al., “GR00T N1: An open foundation model for generalist humanoid robots,” arXiv preprint arXiv:2503.14734, 2025. [Online]. Available: https://arxiv.org/abs/2503.14734

[7] M. J. Kim et al., “OpenVLA: An open-source vision-language-action model,” in Conference on Robot Learning (CoRL), 2024.

[8] Physical Intelligence et al., “π<sub>0.5</sub>: A vision-language-action model with open-world generalization,” in Proceedings of the Conference on Robot Learning (CoRL), 2025.

[9] N. Chung et al., “Rethinking progression of memory state in robotic manipulation: An object-centric perspective,” in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026.

[10] S. Lian et al., “Intentvla: Short-horizon intent modeling for aliased robot manipulation,” arXiv preprint arXiv:2605.14712, 2026. [Online]. Available: https://arxiv.org/abs/2605.14712

[11] H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, E. Zhou, H. Fan, X. Zhang, and G. Huang, “MemoryVLA: Perceptual-cognitive memory in vision-language-action models for robotic manipulation,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=54U3XHf7qq

[12] Y. Chen et al., “Longvila: Scaling long-context visual language models for long videos,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 18 227–18 246.

[13] L. X. Shi et al., “Hi robot: Open-ended instruction following with hierarchical vision-language-action models,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu, Eds., vol. 267. PMLR, 13–19 Jul 2025, pp. 54 919–54 933. [Online]. Available: https://proceedings.mlr.press/v267/shi25d.html

[14] A. Sridhar, J. Pan, S. Sharma, and C. Finn, “Memer: Scaling up memory for robot control via experience retrieval,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://arxiv.org/abs/2510.20328

[15] T. Chen et al., “Rmbench: Memory-dependent robotic manipulation benchmark with insights into policy design,” arXiv preprint arXiv:2603.01229, 2026. [Online]. Available: https://arxiv.org/abs/ 2603.01229

[16] M. T. Villasevil, A. Tang, Y. Liu, and C. Finn, “Learning long-context diffusion policies via past-token prediction,” in Proceedings of The 9th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Lim, S. Song, and H.-W. Park, Eds., vol. 305. PMLR, 27–30 Sep 2025, pp. 1744–1755. [Online]. Available: https://proceedings.mlr.press/v305/villasevil25a.html

[17] P. de Haan, D. Jayaraman, and S. Levine, “Causal confusion in imitation learning,” in Advances in Neural Information Processing Systems (NeurIPS), 2019.

[18] J. Spencer, S. Choudhury, A. Venkatraman, B. Ziebart, and J. A. Bagnell, “Feedback in imitation learning: The three regimes of covariate shift,” in Proceedings of the 38th International Conference on Machine Learning (ICML), 2021.

[19] R. Rahmatizadeh, P. Abolghasemi, L. Bol¨ oni, and S. Levine, “Vision-¨ based multi-task manipulation for inexpensive robots using end-to-end learning from demonstration,” in 2018 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018, pp. 3758–3765.

[20] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın, “What matters in learning from offline human demonstrations for robot manipulation,” in Proceedings of the 5th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, A. Faust, D. Hsu, and G. Neumann, Eds., vol. 164. PMLR, 08–11 Nov 2022, pp. 1678–1690. [Online]. Available: https://proceedings.mlr.press/v164/mandlekar22a.html

[21] Y. Liang and E. Noronha, “Membot: Memory-based robot in intermittent pomdp,” arXiv preprint arXiv:2509.11225, 2025. [Online]. Available: https://arxiv.org/abs/2509.11225

[22] Z. Zhang, H. Xu, Z. Yang, C. Yue, Z. Lin, H.-a. Gao, Z. Wang, and H. Zhao, “Ta-vla: Elucidating the design space of torque-aware vision-

language-action models,” in Proceedings of the 9th Conference on Robot Learning (CoRL), J. Lim, S. Song, and H.-W. Park, Eds., vol. 305. PMLR, 2025, pp. 4019–4037.

[23] K. Fang, A. Toshev, L. Fei-Fei, and S. Savarese, “Scene memory transformer for embodied agents in long-horizon tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 5380–5390.

[24] S. Lee, Y. Wang, H. Etukuru, H. J. Kim, N. M. M. Shafiullah, and L. Pinto, “Behavior generation with latent actions,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, Eds., vol. 235. PMLR, 21–27 Jul 2024, pp. 26 991–27 008. [Online]. Available: https://proceedings.mlr.press/v235/lee24y.html

[25] N. M. M. Shafiullah, Z. J. Cui, A. Altanzaya, and L. Pinto, “Behavior transformers: Cloning k modes with one stone,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[26] S. Wei et al., “ψ<sub>0</sub>: An open foundation model towards universal humanoid loco-manipulation,” in Proceedings of Robotics: Science and Systems (RSS), 2026.

[27] Z. Xu et al., “Mobility vla: Multimodal instruction navigation with long-context vlms and topological graphs,” in Proceedings of The 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, P. Agrawal, O. Kroemer, and W. Burgard, Eds., vol. 270. PMLR, 06–09 Nov 2025, pp. 3866–3887. [Online]. Available: https://proceedings.mlr.press/v270/xu25b.html

[28] H. Fang, M. Grotz, W. Pumacay, Y. R. Wang, D. Fox, R. Krishna, and J. Duan, “SAM2Act: Integrating visual foundation model with a memory architecture for robotic manipulation,” in Proceedings of the International Conference on Machine Learning (ICML), 2025.

[29] F. Lin, R. Nai, Y. Hu, J. You, J. Zhao, and Y. Gao, “OnetwoVLA: A unified vision-language-action model with adaptive reasoning,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=tWMfhoP3as

[30] M. S. Mark, J. Liang, M. Attarian, C. Fu, D. Dwibedi, D. Shah, and A. Kumar, “Bpp: Long-context robot imitation learning by focusing on key history frames,” in Conference on Robot Learning, vol. 297. PMLR, 2025, pp. 2679–2713.

[31] H. Li et al., “CronusVLA: Towards efficient and robust manipulation via multi-frame vision-language-action modeling,” in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026.

[32] M. Torne et al., “Mem: Multi-scale embodied memory for vision language action models,” arXiv preprint arXiv:2603.03596, 2026. [Online]. Available: https://arxiv.org/abs/2603.03596

[33] L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang et al., “Paligemma: A versatile 3b vlm for transfer,” arXiv preprint arXiv:2407.07726, 2024.

[34] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 11 975– 11 986.

[35] G. Bertasius, H. Wang, and L. Torresani, “Is space-time attention all you need for video understanding?” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 813–824. [Online]. Available: https://proceedings.mlr.press/v139/bertasius21a.html

[36] Anthropic, “Claude 5.0 sonnet,” 2026, large language model. [Online]. Available: https://claude.ai

[37] H. Li et al., “Robointer: A holistic intermediate representation suite towards robotic manipulation,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=PGUC3mmMoi

[38] Y. Zhu, J. Wong, A. Mandlekar, R. Mart´ın-Mart´ın, A. Joshi, K. Lin, S. Nasiriany, and Y. Zhu, “robosuite: A modular simulation framework and benchmark for robot learning,” in arXiv preprint arXiv:2009.12293, 2020.

[39] K. Black, M. Galliker, and S. Levine, “Real-time execution of action chunking flow policies,” in Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, Eds., vol. 38, Main Conference. Curran Associates, Inc., 2025, pp. 33 383–33 407. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2025/file/ 300ccb2187dedd4edcc07f7e76d8e553-Paper-Conference.pdf

[40] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in International Conference on Learning Representations (ICLR), 2022.

## APPENDIX I DATA CURATION

We collect human teleoperated robot demonstrations at 10 Hz in simulation and 20 Hz on hardware. All labeling is done via a VLM-generated script, except for the human tap in TapScoopPour which is flagged whilst collecting the demonstration. An example prompt provided to Claude for the generation of labeling script for TapScoopPour is shown below.

## Example Claude Sonnet 5.0 Prompt

Write a labeling function for our xArm demonstration dataset. The task being demonstrated is: a human taps one cup, and the robot then puts a single scoop of beans into that cup.

Input. One LeRobot parquet per episode, recorded at 20 fps. Per frame you have the end-effector pose (xyz in mm, roll/pitch/yaw), a gripper channel, and a human event column that is 1.0 on the single frame where the operator pressed the tap key during teleoperation.

Output. The script should label each frame in the dataset with both a discrete event id from a vocabulary set and a textual memory string.

Vocabulary. Five events, each occurring exactly once per episode: {0: "human tap", 1: "grabbed spoon", 2: "scooped beans", 3: "poured beans", 4: "placed spoon"}. Frames belonging to no event get the null target -1.

## Detection and labeling.

• 0 — human tap: first frame where human event == 1.0. Warn if there is more than one marker and use the first.

• 1 — grabbed spoon: the fully-closed gripper plateau of the first gripper close occurring after the tap.

• 2 — scooped beans: scan for the first 40-frame window satisfying all of: roll std $< 5 ^ { \circ }$ , mean roll within ±20<sup>◦</sup> of 180<sup>◦</sup> (either sign), yaw std $< 5 ^ { \circ }$ , mean z < 255 mm, and pitch increasing on at least 60% of frames (sustained straightening while low in the bowl). Label the end of the 40-frame window with this event.

• 3 — poured beans: the frame of minimum roll.

• 4 — placed spoon: first gripper open after the scoop.

Label the window of frames surrounding the event, starting 5 frames before and ending 20 frames after the detection.

Textual memory. An event’s phrase becomes visible in the memory string only once the frame is no longer labeled with that event. Render as "History: human tap, grabbed spoon, ...", and "History: none" before the first event is visible.

## APPENDIX II

## TRAINING DETAILS

We fine-tune VLA checkpoints (MemER low-level $\pi _ { 0 . 5 } ,$ $\pi _ { 0 . 5 } + \mathrm { V . E } .$ ., all ablations, and full UNIMEM) with LoRA [40]. For tasks such as BeanScoop and TapScoopPour, we note improved performance when upweighting the probability of sampling decisive moments by ∼ 4× in our training dataset. Conceptually, this focuses training effort on moments/frames where the robot’s path diverges, such as deciding whether to scoop again or which cup to pour into.