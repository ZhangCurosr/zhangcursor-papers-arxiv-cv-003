# InsightMap: Structured Spatial Modeling for Embodied Multimodal Reasoning

Hongpei Zheng, Hujun Yin

University of Manchester

Abstract— Language-guided navigation requires connecting partial observations to a persistent spatial reference and learning how actions change that representation. We introduce InsightMap, a framework that uses top-down maps as both explicit spatial memory and action-conditioned prediction targets. Historical views are linked to labeled map locations, and a shared multimodal backbone jointly learns navigation action prediction and post-action map generation. Map prediction provides auxiliary training supervision, while navigation inference decodes actions from the observed spatial context. An aligned RGB-D data pipeline supports a common interface for navigation, visual question answering, situated reasoning, and 3D grounding. On the validation-unseen splits of R2R-CE and RxR-CE, InsightMap achieves success rates (SR) of 56.9% and 54.9%, respectively. Adding map-prediction supervision improves R2R-CE SR by 4.3 and success weighted by path length (SPL) by 3.2 percentage points. On static spatial tasks, InsightMap achieves 103.7 CIDEr on ScanQA, 60.1% exactmatch accuracy on SQA3D, and 53.1% grounding accuracy at 0.5 IoU on ScanRefer with detected object proposals. On Unitree Go2, it outperforms NaVid and NaVILA in hallway, lab, and office environments.

## I. INTRODUCTION

Language-guided embodied tasks require robots to relate partial observations to a persistent spatial context. In visionand-language navigation (VLN), this means connecting current views with previously visited places to follow an instruction. The resulting action–observation sequences also provide spatial transitions that can supervise policy learning.

Maps provide an explicit reference for organizing observation history. Topological, grid, multi-granularity, and annotated semantic maps aggregate spatial evidence and support navigation decisions [1]–[4]. However, action-label supervision alone does not explicitly train a policy to predict how its map changes after an action. This motivates using post-action maps as additional prediction targets for navigation learning.

Visual world models predict future observations, and World Action Models couple future visual prediction with action prediction [5], [6]. We apply this predictive principle to top-down spatial maps. Even in a static environment, ego-motion changes the map’s local reference frame and new observations reveal additional geometry. Post-action map prediction therefore supplies a structured target for the evolution of accumulated spatial evidence.

We introduce InsightMap, a framework that uses maps as both explicit spatial memory and action-conditioned prediction targets (Fig. 1). Historical first-person views are linked to labeled map locations, allowing visual observations and trajectory geometry to be interpreted together. Built on BAGEL [7], the model jointly learns navigation action prediction and post-action map generation.

Question answering and 3D grounding also require associating objects and spatial relations across viewpoints. We incorporate these tasks into the same spatial interface to examine whether auxiliary spatial training benefits navigation. An aligned RGB-D data pipeline supplies navigation transitions, scene-level and situated question–answer pairs, and grounding examples with textual boxes and targetannotated maps.

We evaluate InsightMap on R2R-CE and RxR-CE navigation, ScanQA and SQA3D question answering, and Scan-Refer visual grounding. Adding map supervision improves R2R-CE success rate (SR) and success weighted by path length (SPL) by 4.3 and 3.2 percentage points, respectively. Adding historical location IDs to the map raises SR by 2.8 percentage points. The full task mixture achieves 56.9% SR, compared with 49.2% for navigation-only training. On Unitree Go2, InsightMap outperforms NaVid and NaVILA in all three evaluated environments, with an SR gain of 45 percentage points over NaVILA in the office.

Our contributions are:

• A map-based spatial interface that links historical views to labeled observation locations for navigation and static scene reasoning.

• A joint action-and-map training formulation that uses action-conditioned post-action map prediction as auxiliary spatial supervision for navigation.

• An aligned RGB-D data pipeline yielding 800K training examples, with benchmark and real-world evaluations that assess navigation, static spatial reasoning, and the effects of spatial supervision.

## II. RELATED WORK

## A. Embodied Navigation and Spatial Memory

VLN models organize observation history through recurrent states and spatial maps. Recurrent VLN-BERT [8] carries a cross-modal state, ETPNav [1] builds an evolving topological map, and WS-MGMap [3] and GridMM [2] aggregate spatial features in metric and grid representations.

Recent systems use vision-language models to connect visual context with action selection. NaVid [9] predicts actions from video history, NaVILA [10] couples a high-level VLA with locomotion control, and InstructNav [11] uses language reasoning and value maps. MapNav [4] annotates an online semantic map with object labels to reduce reliance on historical frames. InsightMap retains sampled historical views and explicitly links them to labeled map locations.

![](images/2e67df11011723a70fd6366c3f5d851f54b1307bdf09ff990a7eb904cd0f43e7.jpg)  
Fig. 1. Overview of InsightMap across vision-and-language navigation, visual question answering, and 3D visual grounding. An observed top-down map provides spatial context for all tasks. Navigation training pairs actions with post-action map targets. Question answering produces answer text, and grounding predicts a 3D box and a target-annotated map.

Spatial prediction also supports navigation learning and planning. BEVBert [12] predicts semantic labels of masked map regions during pre-training. HNR [13] predicts features of candidate locations for lookahead path evaluation. InsightMap uses reconstructed maps as spatial memory and action-conditioned post-action map prediction as auxiliary training supervision.

## B. 3D Question Answering and Grounding

ScanQA [14], SQA3D [15], and ScanRefer [16] evaluate scene-level question answering, situated reasoning, and language-guided 3D object localization, respectively.

Scene interfaces range from 3D features to object- and video-based representations. 3D-LLM [17] lifts multi-view features into 3D, while LL3DA [18] operates on point clouds with text and visual prompts. Chat-Scene [19] uses object identifiers and object-centric embeddings, and 3D-LLaVA [20] combines superpoint representations with mask prediction. Video-3D LLM [21] incorporates global 3D position encodings into video features and selects views by maximum coverage.

GPT4Scene [22] establishes global–local correspondence through a reconstructed BEV image and consistent object markers across views. InsightMap uses camera-location labels to associate selected views with the scene map. For grounding, it couples frame-indexed 3D box prediction with generation of a target-annotated map; QA uses the shared spatial context with answer-text supervision.

## C. Unified Understanding, Generation, and World Models

BAGEL [7] unifies understanding and generation through a Mixture-of-Transformers backbone, trained with next-token prediction for text and rectified flow for images. InsightMap adopts this architecture and these objectives, using spatial maps as visual targets aligned with action or grounding text.

Cosmos [5] formulates visual world modeling as predicting future observations conditioned on history and an action or instruction. DreamZero [6] jointly generates future video and actions. InsightMap applies this principle of coupling visual prediction with action learning to action-conditioned post-action maps as structured prediction targets.

## III. METHOD

InsightMap couples spatial memory with structured world imagination (Fig. 2): it anchors visual experience in an explicit map and learns to predict how this representation evolves after an action. Joint supervision of actions and their post-action maps connects embodied decision-making with prediction of spatial consequences. The same spatial interface supports question answering and grounding. Section IV describes the construction of aligned observations, maps, and targets.

![](images/bd41b4741af6885b8a0c24c218f34fa7b47bda94334ae55183eae16a9e489eda.jpg)  
Fig. 2. Architecture of InsightMap. Labeled historical views, the current observation and map, and the instruction condition a shared understanding and generation backbone. The training targets are action text followed by an action-conditioned post-action map.

## A. Spatial Memory

To retain the spatial context of past observations, InsightMap links historical views to labeled map locations. At navigation step t, let $\mathcal { H } _ { t } ~ = ~ ( ( o _ { i } , \ell _ { i } ) ) _ { i \in \mathcal { S } _ { \mathrm { ~ } } }$ denote the selected history in temporal order, where $S _ { t }$ indexes earlier macro states and $\ell _ { i }$ labels the location of observation $o _ { i } .$ The context is $h _ { t } ~ = ~ ( q , \mathcal { H } _ { t } , o _ { t } , m _ { t } ) \colon$ $q$ is the instruction, $o _ { t }$ is the current observation, and $m _ { t }$ is the reconstructed top-down map. The map contains the corresponding location labels, the connecting trajectory, and the current-pose marker. These view–location correspondences let visual history and trajectory geometry be interpreted together, preserving where evidence was acquired as the viewpoint changes.

## B. Structured World Imagination

The map also provides a structured space in which to learn the spatial consequences of navigation. We formulate structured world imagination as predicting the agent’s postaction spatial representation, conditioned on the current context and action. The navigation training targets are an action $a _ { t }$ and the map $m _ { t + 1 }$ constructed after executing it. Their joint prediction is factorized as

$$
p ( a _ { t } , m _ { t + 1 } \mid h _ { t } ) = p ( a _ { t } \mid h _ { t } ) p ( m _ { t + 1 } \mid h _ { t } , a _ { t } ) .\tag{1}
$$

The post-action map represents accumulated geometry in the agent’s new local reference frame. Predicting it requires accounting for both ego-motion and newly observed geometry. During training, the context precedes the macro-action text and future map, represented as a continuous VAE latent. Ground-truth action tokens condition map generation, and the reference future map supplies supervision for the predicted spatial transition.

Navigation inference. At each navigation step, InsightMap decodes only the action text conditioned on $h _ { t } .$ . After execution, new sensor observations update the observed map and visual history for the next decision. Future-map generation serves as auxiliary training supervision and is not performed during navigation evaluation.

## C. Unified Spatial Reasoning

Spatial memory provides a common interface for relating language to visual evidence across tasks. For static scene reasoning, context h contains the task text, selected views, and a scene map with corresponding observation-location labels. The text target $y$ is an answer or a frame-indexed 3D box representation. Grounding additionally supplies a targetannotated map $m ^ { * }$ . Examples with both targets follow

$$
p ( y , m ^ { * } \mid h ) = p ( y \mid h ) p ( m ^ { * } \mid h , y ) .\tag{2}
$$

Equation 1 is recovered with $h = h _ { t } , y = a _ { t } .$ , and $m ^ { * } =$ $m _ { t + 1 } .$ . The tasks share this interface while assigning different meanings to their outputs.

Visual grounding. ScanRefer [16] pairs the spatial context with a referring expression. The text target is $[ f , c _ { x } , c _ { y } , c _ { z } , s _ { x } , s _ { y } , s _ { z } ] $ , where $f$ identifies the first frame in which the referred object appears. The box center $\left( c _ { x } , c _ { y } , c _ { z } \right)$ and size $( s _ { x } , s _ { y } , s _ { z } )$ are expressed in that frame’s camera coordinate system. The visual target $m ^ { * }$ adds the object’s box to the map, providing a spatial annotation of the referred object. This target box is excluded from the input map.

Visual question answering. ScanQA [14] and SQA3D [15] pair the spatial context with a question; SQA3D additionally includes the situation description. These tasks train only the answer distribution $p ( y \mid h )$ , with no target map.

## D. Joint Multimodal Learning

The shared task interface uses a Mixture-of-Transformers (MoT) [23] backbone. First-person views and top-down maps are encoded through both ViT and VAE paths. The two visual paths provide complementary representations for action prediction and structured world imagination. ViT features capture high-level visual semantics that help relate observations to language instructions and support action prediction. VAE latents encode low-level visual detail in a representation that supports image reconstruction, providing the latent space for post-action map prediction. Text and ViT tokens follow the understanding branch, while VAE latents follow the generation branch. Shared self-attention allows both branches to integrate semantic context and visual detail for action prediction and map generation.

The understanding head predicts text tokens, and the generation head predicts rectified-flow velocities for targetmap latents conditioned on the preceding context and text. Conditioning-image latents use the same generation branch but are not map-generation targets.

Task prompts and observations form packed text–image sequences, with supervision selecting the target text tokens and any target-map latents. Using the common context h and text target $y ,$ the base text loss for one example at supervised positions $\tau$ is

$$
\mathcal { L } _ { \mathrm { C E } } = - \frac { 1 } { | T | } \sum _ { i \in \mathcal { T } } \log p _ { \theta } ( y _ { i } \mid y _ { < i } , h ) .\tag{3}
$$

For examples with a target map $m ^ { * }$ , let $x _ { 0 }$ be its clean VAE latent. We sample noise $x _ { 1 }$ and a flow time $\tau \in [ 0 , 1 ]$ , distinct from navigation time $t ,$ and construct

$$
x _ { \tau } = ( 1 - \tau ) x _ { 0 } + \tau x _ { 1 } , \qquad v = x _ { 1 } - x _ { 0 } .\tag{4}
$$

The map-generation loss, conditioned on the preceding text target, is

$$
\mathcal { L } _ { \mathrm { m a p } } = \frac { 1 } { \vert \mathcal { V } \vert } \sum _ { j \in \mathcal { V } } \Vert \hat { v } _ { \boldsymbol { \theta } } ( x _ { \tau } , \tau , h , y ) _ { j } - v _ { j } \Vert _ { 2 } ^ { 2 } ,\tag{5}
$$

where V indexes the supervised target-map latent positions. For navigation, $\begin{array} { r } { \boldsymbol { y } ~ = ~ a _ { t } } \end{array}$ is the ground-truth action text; for grounding, it contains the target frame ID and box parameters. Writing $\tilde { \mathcal { L } } _ { \mathrm { C E } }$ for the length-reweighted text loss used in training, the combined objective is

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { C E } } \widetilde { \mathcal { L } } _ { \mathrm { C E } } + \lambda _ { \mathrm { m a p } } \mathcal { L } _ { \mathrm { m a p } } . } \end{array}\tag{6}
$$

The MoT generation branch remains trainable during joint training, while the VAE is frozen. Navigation and grounding activate both losses. For QA, V is empty and we define $\mathcal { L } _ { \mathrm { m a p } } = 0$ . Joint training therefore couples textual task prediction with explicit spatial supervision through the shared backbone.

## IV. SPATIAL DATA CONSTRUCTION

We construct a training corpus of 800K examples that aligns visual observations, spatial context, and task supervision. Navigation contributes 65% of the corpus, providing

![](images/53f55d507b6cc5f0334ac1010eff7fc00240368f69ca72037738ec4c7aa13836.jpg)  
Fig. 3. Composition of the 800K training examples. Navigation accounts for 65%, and visual question answering and grounding for 35%. All percentages refer to the full corpus.

action-aligned map transitions for structured world imagination; visual question answering and grounding contribute the remaining 35% (Fig. 3).

## A. Navigation Spatial Supervision

Navigation data combine R2R-CE, RxR-CE, ScaleVLN [24], and DAgger trajectories. Consecutive actions of the same type are grouped into macro actions. We pair each pre-action context with the action and the map reconstructed after execution, yielding aligned examples $\left( h _ { t } , a _ { t } , m _ { t + 1 } \right)$

Trajectory replay in Habitat [25] supplies depth observations and poses for incremental truncated signed distance function (TSDF) reconstruction. Maps contain geometry observed up to each step, rendered over a fixed local region centered on the agent at its current floor, with image-up aligned to its heading. History views sampled at earlier macro states share location labels with the map, which also records the trajectory and current pose.

After one epoch of VLN-only training, we roll out the resulting policy and use expert guidance to construct corrective trajectories from erroneous navigation cases. These DAgger [26] trajectories are aggregated with the original navigation data and provide both action and map supervision.

## B. Static Scene Construction

For ScanQA [14], SQA3D [15], and ScanRefer [16], we construct full-scene maps from ScanNet geometry. Following Video-3D LLM [21], we greedily select RGB-D views that add the most scene-surface coverage and link them to indexed camera locations on the map.

We expand the VQA data through LLM-based question rewriting, conversion of object descriptions into question– answer pairs, and alternative situations for situated QA.

Visual grounding data are augmented by rewriting referring descriptions. The resulting examples use the task targets defined in Section III: answers for VQA, and boxes with target-annotated maps for grounding. Grounding input maps contain the scene and observation-location markers, excluding the target box.

## V. EXPERIMENTS

## A. Experimental Setup

Simulation Benchmark Setup. We evaluate InsightMap on the validation-unseen splits of R2R-CE [27] and RxR-CE [28] in continuous Matterport3D environments through the Habitat simulator. For R2R-CE, we report navigation error (NE, in meters), oracle success rate (OS), success rate (SR), and success weighted by path length (SPL). OS, SR, and SPL are reported as percentages. For RxR-CE, we report NE, SR, SPL, and normalized dynamic time warping (nDTW) to additionally assess trajectory fidelity. Lower NE and higher values of the other metrics indicate better performance.

Throughout the tables, bold and underlined scores denote the best and second-best listed values, respectively. Dashes in metric columns indicate unavailable comparable results.

Real-World Evaluation Setup. We deploy InsightMap on a Unitree Go2 quadruped, using its built-in camera for visual observations and LiDAR-reconstructed point clouds for map construction. Model inference runs remotely on a server with a single NVIDIA H100 GPU. We evaluate navigation in hallway (easy), lab (medium), and office (hard) settings. InsightMap, NaVid, and NaVILA are evaluated on the same robot platform, scenes, and navigation instructions.

## B. Implementation Details

Training. We build InsightMap on BAGEL [7]. After one epoch of VLN-only training, the resulting policy is used to collect expert-guided DAgger corrections. Joint training uses the 800K-example corpus spanning navigation, visual question answering, and visual grounding described in Section IV. We fine-tune all learnable parameters except the VAE, including both MoT branches. Training uses AdamW with a learning rate of $2 \times 1 0 ^ { - 5 }$ on eight NVIDIA H100 GPUs, accumulating gradients over four micro-steps for an effective global batch size of 64 examples per step. The text and map loss weights are $( \lambda _ { \mathrm { C E } } , \lambda _ { \mathrm { m a p } } ) = ( 3 , 1 )$

Training inputs and actions. Navigation examples contain the current RGB observation and up to ten history frames. ScanQA, SQA3D, and ScanRefer use 24, 24, and 16 views, respectively; all tasks also receive a map. The image transforms specify a maximum RGB size of 384 pixels and a target map resolution of 512 × 512, before encoderspecific alignment. Navigation maps cover approximately 16 m×16 m around the agent by default. Constructed macroaction targets span 25–75 cm or 15–45<sup>◦</sup>.

## C. Main navigation results

Table I compares InsightMap with the listed navigation baselines. On RxR-CE, InsightMap achieves NE 5.58, SR 54.9, SPL 46.1, and nDTW 62.8, the strongest values among the listed methods. Relative to NaVILA, SR and SPL improve by 5.6 and 2.1 percentage points, respectively, while nDTW increases by 4.0 points, indicating gains in task completion, path efficiency, and trajectory fidelity.

TABLE I  
NAVIGATION RESULTS ON R2R-CE AND RXR-CE VAL-UNSEEN.
<table><tr><td colspan="4">(a) R2R-CE</td></tr><tr><td>Method</td><td>NE↓</td><td>OS↑</td><td>SR↑</td><td>SPL↑</td></tr><tr><td>HPN+DN* [29]</td><td>6.31</td><td>40.0</td><td>36.0</td><td>34.0</td></tr><tr><td>CMA* [30]</td><td>6.20</td><td>52.0</td><td>41.0</td><td>36.0</td></tr><tr><td>Sim2Sim* [31]</td><td>6.07</td><td>52.0</td><td>43.0</td><td>36.0</td></tr><tr><td>GridMM* [2]</td><td>5.11</td><td>61.0</td><td>49.0</td><td>41.0</td></tr><tr><td>ScaleVLN* [24]</td><td>4.80</td><td></td><td>55.0</td><td>51.0</td></tr><tr><td>InstructNav [11]</td><td>6.89</td><td></td><td>31.0</td><td>24.0</td></tr><tr><td>WS-MGMap [3]</td><td>6.28</td><td>47.6</td><td>38.9</td><td>34.3</td></tr><tr><td>Seq2Seq [27]</td><td>7.77</td><td>37.0</td><td>25.0</td><td>22.0</td></tr><tr><td>NaVid [9]</td><td>5.47</td><td>49.1</td><td>37.4</td><td>35.9</td></tr><tr><td>MapNav [4]</td><td>4.93</td><td>53.0</td><td>39.7</td><td>37.2</td></tr><tr><td>NaVILA [10]</td><td>5.22</td><td>62.5</td><td>54.0</td><td>49.0</td></tr><tr><td>InsightMap</td><td>4.89</td><td>64.7</td><td>56.9</td><td>50.7</td></tr></table>

(b) RxR-CE
<table><tr><td>Method</td><td>NE↓</td><td>SR↑</td><td>SPL↑</td><td>nDTW↑</td></tr><tr><td>CMA*[30]</td><td>8.76</td><td>26.5</td><td>22.1</td><td>47.0</td></tr><tr><td>LAW [32]</td><td>10.90</td><td>8.0</td><td>8.0</td><td>38.0</td></tr><tr><td>Seq2Seq [27]</td><td>12.10</td><td>13.9</td><td>11.9</td><td>30.8</td></tr><tr><td>NaVILA [10]</td><td>6.77</td><td>49.3</td><td>44.0</td><td>58.8</td></tr><tr><td>UniNaVid [33]</td><td>6.24</td><td>48.7</td><td>40.9</td><td></td></tr><tr><td>InsightMap</td><td>5.58</td><td>54.9</td><td>46.1</td><td>62.8</td></tr></table>

On R2R-CE, InsightMap achieves the best listed OS and SR at 64.7 and 56.9, respectively. Its NE of 4.89 and SPL of 50.7 rank second, trailing ScaleVLN by 0.09 m and 0.3 points. The results support competitive navigation and path efficiency within this comparison.

The comparison includes methods with different sensor inputs, training data, and waypoint predictors. Asterisks in Table I denote source-designated simulator-trained waypoint predictors.

Figure 4 shows the agent moving around the central unit toward the doorway specified by the instruction. As the viewpoint changes, the map retains the visited locations and traversed path, while the history strip preserves earlier visual observations. The sequence combines small heading adjustments with forward movements and ends with STOP. This example illustrates how the spatial representation connects changing local views with accumulated trajectory information throughout instruction following.

## D. Ablation studies

We examine three factors on R2R-CE val-unseen: mapprediction supervision (Table II), spatial map input and location-ID linking (Table III), and auxiliary training tasks (Table IV).

Generation-module trainability and map supervision. Table II retains the full spatial input and conditioning-image VAE route in all variants. Unfreezing the generation module without map supervision (A1 to A2) improves SR and SPL by 2.3 and 2.4 percentage points. Adding map supervision at the same trainability (A2 to A3) yields further gains of 4.3 and 3.2 points, reaching 56.9 SR and 50.7 SPL. These results support map prediction as auxiliary supervision for navigation.

![](images/2b98d887cb641275b7f67c1fe5348f0160422470e3e5092763f96a112ab8e1ab.jpg)  
Instruction: Go around the right side of the center unit and stop by the right side doorway with the dining table and mirror in it.  
Fig. 4. Recorded navigation trajectory of InsightMap. The eight displayed steps pair top-down maps and current RGB views with history strips (past to current) and recorded actions. Green circles mark the agent, blue points indicate observation history, and yellow lines trace the path. The sequence ends with STOP.

TABLE II  
ABLATION OF GENERATION-MODULE TRAINABILITY AND MAP SUPERVISION ON R2R-CE VAL-UNSEEN.
<table><tr><td>Variant</td><td>Gen. module</td><td>Map loss</td><td>SR↑</td><td>SPL↑</td></tr><tr><td>A1</td><td>Frozen</td><td>Off</td><td>50.3</td><td>45.1</td></tr><tr><td>A2</td><td>Trainable</td><td>Off</td><td>52.6</td><td>47.5</td></tr><tr><td>A3 (Full)</td><td>Trainable</td><td>On</td><td>56.9</td><td>50.7</td></tr></table>

Spatial map input and location-ID linking. Table III keeps generation parameters trainable and future-map supervision unchanged. B1 uses a same-sized blank input map. B2 provides an unnumbered map, retaining geometry, location markers, trajectory, current pose, and RGB-frame labels. Relative to B1, B2 improves SR and SPL by 1.5 and 2.4 percentage points, showing the benefit of spatial context. Restoring historical location IDs on the map (B2 to B3) adds 2.8 and 0.8 points, supporting explicit links between historical views and map locations. Each input configuration applies during both training and evaluation, with the same map token budget and output resolution.

Task mixtures. Table IV uses ScanRefer for Visual Grounding (VG) and both ScanQA and SQA3D for Visual Question Answering (VQA). C0 is the one-epoch VLN-only policy used to collect DAgger trajectories. C1 is retrained in a separate run using VLN and DAgger data and serves as the navigation-only reference. C1–C4 all include DAgger data. Adding VG (C2) or VQA (C3) improves SR/SPL over C1 by 2.5/2.4 or 5.9/4.7 percentage points, respectively.

TABLE III  
ABLATION OF MAP INPUT AND LOCATION IDS ON R2R-CEVAL-UNSEEN.
<table><tr><td>Variant</td><td>Map input</td><td>Location IDs</td><td>SR↑</td><td>SPL↑</td></tr><tr><td>B1</td><td>Blank</td><td>No</td><td>52.6</td><td>47.5</td></tr><tr><td>B2</td><td>Unnumbered map</td><td>No</td><td>54.1</td><td>49.9</td></tr><tr><td>B3 (Full)</td><td>Full map</td><td>Yes</td><td>56.9</td><td>50.7</td></tr></table>

TABLE IV

TASK-MIXTURE ABLATIONS ON R2R-CE VAL-UNSEEN.
<table><tr><td rowspan="2">Variant</td><td colspan="3">Training tasks</td><td rowspan="2">DAgger</td><td colspan="4">R2R-CE</td></tr><tr><td>VLN</td><td>VG</td><td>VQA</td><td>NE↓</td><td>OS↑</td><td>SR↑</td><td>SPL↑</td></tr><tr><td>CO</td><td>√</td><td>一</td><td>一</td><td></td><td>6.02</td><td>49.8</td><td>46.1</td><td>41.7</td></tr><tr><td>C1</td><td>√</td><td>一</td><td></td><td>√</td><td>5.83</td><td>53.8</td><td>49.2</td><td>44.6</td></tr><tr><td>C2</td><td>√</td><td>V</td><td></td><td>√</td><td>5.72</td><td>58.2</td><td>51.7</td><td>47.0</td></tr><tr><td>C3</td><td>V</td><td>一</td><td>√</td><td>V</td><td>5.09</td><td>61.2</td><td>55.1</td><td>49.3</td></tr><tr><td>C4</td><td>V</td><td>V</td><td>V</td><td>V</td><td>4.89</td><td>64.7</td><td>56.9</td><td>50.7</td></tr></table>

Combining both tasks (C4) achieves the best results across all four metrics, reaching 56.9 SR and 50.7 SPL, gains of 7.7 and 6.1 points over C1. Adding VG to VQA (C3 to C4) further improves SR/SPL by 1.8/1.4 points, supporting complementary benefits from the two auxiliary tasks.

## E. Static scene reasoning

We evaluate static scene reasoning on ScanQA [14] validation, SQA3D [15] test, and ScanRefer [16] validation. Tables V and VI report task results alongside source-reported baselines.

For ScanQA, we report BLEU-4 (B-4), METEOR (M), ROUGE-L (R-L), and CIDEr (C). SQA3D uses exact-match accuracy (EM), reported as a percentage. Table V shows that InsightMap achieves the best listed ScanQA validation results across all four metrics. Its CIDEr score of 103.7 exceeds NaVILA (16 frames) by 3.9 points. On SQA3D test, InsightMap reaches 60.1% EM, 5.53 percentage points above the best listed baseline, Chat-Scene. LL3DA is evaluated without test-time visual prompts.

TABLE V  
QUESTION ANSWERING ON SCANQA VALIDATION AND SQA3D TEST.
<table><tr><td rowspan="2">Method</td><td colspan="4">ScanQA</td><td>SQA3D</td></tr><tr><td>B-4↑</td><td>M↑</td><td>R-L↑</td><td>C↑</td><td>EM↑</td></tr><tr><td>ScanQA [14] 3D-LLM [17] LL3DA [18] Chat-Scene [19]</td><td>10.08 12.0 13.53</td><td>13.14 14.5 15.88</td><td>33.33 35.7 37.31</td><td>64.86 69.4 76.79</td><td>一 54.57</td></tr><tr><td>3D-LLaVA [20] NaviLLM [34]</td><td>17.1 12.0</td><td>18.4 15.4</td><td>43.1 38.4</td><td>92.6 75.9</td><td>54.5</td></tr><tr><td>NaVILA [10] InsightMap (Ours)</td><td>15.2 17.9</td><td>19.6 19.7</td><td>48.3 49.2</td><td>99.8 103.7</td><td>1 60.1</td></tr></table>

TABLE VI

SCANREFER VALIDATION RESULTS. PARENTHESES DENOTE RAW BOX PREDICTIONS BEFORE PROPOSAL REFINEMENT.
<table><tr><td>Method</td><td>Acc@0.25↑</td><td>Acc@0.5↑</td></tr><tr><td>ScanRefer [16]</td><td>41.19</td><td>27.40</td></tr><tr><td>Chat-Scene [19]</td><td>55.52</td><td>50.23</td></tr><tr><td>3D-LLaVA [20]</td><td>51.2</td><td>40.6</td></tr><tr><td>Video-3D LLM [21]</td><td>57.87</td><td>51.18</td></tr><tr><td>GPT4Scene [22]</td><td>62.6</td><td>57.0</td></tr><tr><td>VG-LLM [35]</td><td>57.6 (41.6)</td><td>50.9 (14.9)</td></tr><tr><td>InsightMap (Ours)</td><td>59.7 (43.8)</td><td>53.1 (20.8)</td></tr></table>

Table VI reports overall ScanRefer box-localization accuracy (%) at 3D intersection over union (IoU) thresholds of 0.25 and 0.5, denoted Acc@0.25 and Acc@0.5. InsightMap directly generates the box center and size as text from the selected views and scene map, achieving raw accuracies of 43.8% and 20.8%. These exceed the raw results of VG-LLM [35] by 2.2 and 5.9 percentage points, respectively, showing stronger localization before proposal refinement within this comparison.

To refine these predictions, we match each predicted box to the candidate with the highest 3D IoU and use the matched proposal as the final box. The candidates are detected by Mask3D [36] and provided by LEO [37]. Proposal matching raises Acc@0.25 and Acc@0.5 to 59.7% and 53.1%, gains of 15.9 and 32.3 percentage points over raw predictions.

With proposal refinement, InsightMap ranks second on both metrics among the listed methods, exceeding VG-LLM by 2.1 and 2.2 percentage points and trailing GPT4Scene by 2.9 and 3.9 points. These results show competitive grounding performance from the shared spatial interface, with detected proposals contributing substantially to the final accuracy.

## F. Real-world evaluation

We conducted 20 trials per scenario, measuring success rate (SR) in hallway, lab, and office settings. A trial is successful if the robot reaches within 2 m of the goal within 50 steps. Figure 5 summarizes the results. InsightMap achieves 95%, 90%, and 65%, exceeding NaVILA by 20, 20, and 45 percentage points, respectively. The largest improvement occurs in the office setting.

![](images/0ea5a808dd3319ee5991e602a8a55fad16e741fd1fc3063f6211c358a822a073.jpg)  
Fig. 5. Real-world navigation success rates (%) in hallway (easy), lab (medium), and office (hard) settings. All methods use the same Unitree Go2 platform, scenes, and navigation instructions.

![](images/3388a8d48d186e1c4aaa6db7be37dd58ca38967eae3b5c603b8eea80773d701d.jpg)  
Instruction: Walk forward and turn left in front of the white lockers. Continue along the corridor and stop beside the black trash bin with a blue lid on your left  
Fig. 6. Real-world office navigation with InsightMap on Unitree Go2. Third-person frames are ordered left to right, top to bottom.

The office example illustrates the need to link object descriptions with spatial relations over successive observations. The instruction combines a landmark-dependent left turn with a stopping location beside a second object. In Fig. 6, the robot approaches the white lockers, changes direction, and proceeds toward the black trash bin with a blue lid. This sequence provides a concrete example of carrying out a multi-stage instruction on the physical platform. Together with the success-rate comparison, it supports the applicability of the shared spatial interface to real-world navigation.

## VI. CONCLUSION

We presented InsightMap, a framework that links historical views to map locations and jointly trains navigation action prediction and action-conditioned map generation. On R2R-CE val-unseen, adding map supervision improves SR and SPL by 4.3 and 3.2 percentage points. The navigation ablations also show higher SR with explicit view–location links and the full task mixture. Static scene evaluation yields 103.7 CIDEr on ScanQA, 60.1% exact-match accuracy on SQA3D, and 53.1% Acc@0.5 on ScanRefer with detected object proposals. Unitree Go2 trials show higher SR than NaVid and NaVILA in all three evaluated environments. These findings establish the practical value of organizing observation history and learning spatial transitions through a shared map representation.

## REFERENCES

[1] D. An et al., “Etpnav: Evolving topological planning for visionlanguage navigation in continuous environments,” 2024. [Online]. Available: https://arxiv.org/abs/2304.03047

[2] Z. Wang, X. Li, J. Yang, Y. Liu, and S. Jiang, “Gridmm: Grid memory map for vision-and-language navigation,” 2023. [Online]. Available: https://arxiv.org/abs/2307.12907

[3] P. Chen et al., “Weakly-supervised multi-granularity map learning for vision-and-language navigation,” 2022. [Online]. Available: https://arxiv.org/abs/2210.07506

[4] L. Zhang et al., “Mapnav: A novel memory representation via annotated semantic maps for vision-and-language navigation,” 2026. [Online]. Available: https://arxiv.org/abs/2502.13451

[5] NVIDIA et al., “Cosmos world foundation model platform for physical ai,” 2025. [Online]. Available: https://arxiv.org/abs/2501.03575

[6] S. Ye et al., “World action models are zero-shot policies,” 2026. [Online]. Available: https://arxiv.org/abs/2602.15922

[7] C. Deng et al., “Emerging properties in unified multimodal pretraining,” 2025. [Online]. Available: https://arxiv.org/abs/2505.14683

[8] Y. Hong, Q. Wu, Y. Qi, C. Rodriguez-Opazo, and S. Gould, “Vln bert: A recurrent vision-and-language bert for navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 1643–1653.

[9] J. Zhang et al., “Navid: Video-based vlm plans the next step for vision-and-language navigation,” 2024. [Online]. Available: https://arxiv.org/abs/2402.15852

[10] A.-C. Cheng et al., “Navila: Legged robot vision-languageaction model for navigation,” 2025. [Online]. Available: https://arxiv.org/abs/2412.04453

[11] Y. Long, W. Cai, H. Wang, G. Zhan, and H. Dong, “Instructnav: Zero-shot system for generic instruction navigation in unexplored environment,” 2024. [Online]. Available: https://arxiv.org/abs/2406.04882

[12] D. An et al., “Bevbert: Multimodal map pre-training for language-guided navigation,” 2023. [Online]. Available: https://arxiv.org/abs/2212.04385

[13] Z. Wang et al., “Lookahead exploration with neural radiance representation for continuous vision-language navigation,” 2024. [Online]. Available: https://arxiv.org/abs/2404.01943

[14] D. Azuma, T. Miyanishi, S. Kurita, and M. Kawanabe, “Scanqa: 3d question answering for spatial scene understanding,” 2022. [Online]. Available: https://arxiv.org/abs/2112.10482

[15] X. Ma et al., “Sqa3d: Situated question answering in 3d scenes,” 2023. [Online]. Available: https://arxiv.org/abs/2210.07474

[16] D. Z. Chen, A. X. Chang, and M. Nießner, “Scanrefer: 3d object localization in rgb-d scans using natural language,” 2020. [Online]. Available: https://arxiv.org/abs/1912.08830

[17] Y. Hong et al., “3d-llm: Injecting the 3d world into large language models,” 2023. [Online]. Available: https://arxiv.org/abs/2307.12981

[18] S. Chen et al., “Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 26 428–26 438.

[19] H. Huang et al., “Chat-scene: Bridging 3d scene and large language models with object identifiers,” 2024. [Online]. Available: https://arxiv.org/abs/2312.08168

[20] J. Deng, T. He, L. Jiang, T. Wang, F. Dayoub, and I. Reid, “3d-llava: Towards generalist 3d lmms with omni superpoint transformer,” 2025. [Online]. Available: https://arxiv.org/abs/2501.01163

[21] D. Zheng, S. Huang, and L. Wang, “Video-3d llm: Learning position-aware video representation for 3d scene understanding,” 2025. [Online]. Available: https://arxiv.org/abs/2412.00493

[22] Z. Qi, Z. Zhang, Y. Fang, J. Wang, and H. Zhao, “Gpt4scene: Understand 3d scenes from videos with vision-language models,” 2025. [Online]. Available: https://arxiv.org/abs/2501.01428

[23] W. Liang et al., “Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models,” 2025. [Online]. Available: https://arxiv.org/abs/2411.04996

[24] Z. Wang et al., “Scaling data generation in vision-and-language navigation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 12 009–12 020.

[25] M. Savva et al., “Habitat: A platform for embodied ai research,” 2019. [Online]. Available: https://arxiv.org/abs/1904.01201

[26] S. Ross, G. J. Gordon, and J. A. Bagnell, “A reduction of imitation learning and structured prediction to no-regret online learning,” 2011. [Online]. Available: https://arxiv.org/abs/1011.0686

[27] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the nav-graph: Vision-and-language navigation in continuous environments,” 2020. [Online]. Available: https://arxiv.org/abs/2004.02857

[28] A. Ku, P. Anderson, R. Patel, E. Ie, and J. Baldridge, “Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding,” 2020. [Online]. Available: https://arxiv.org/abs/2010.07954

[29] J. Krantz, A. Gokaslan, D. Batra, S. Lee, and O. Maksymets, “Waypoint models for instruction-guided navigation in continuous environments,” 2021. [Online]. Available: https://arxiv.org/abs/2110.02207

[30] Y. Hong, Z. Wang, Q. Wu, and S. Gould, “Bridging the gap between learning in discrete and continuous environments for vision-and-language navigation,” 2022. [Online]. Available: https://arxiv.org/abs/2203.02764

[31] J. Krantz and S. Lee, “Sim-2-sim transfer for vision-and-language navigation in continuous environments,” 2022. [Online]. Available: https://arxiv.org/abs/2204.09667

[32] S. Raychaudhuri, S. Wani, S. Patel, U. Jain, and A. Chang, “Language-aligned waypoint (LAW) supervision for vision-andlanguage navigation in continuous environments,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, Eds. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, Nov. 2021, pp. 4018–4028. [Online]. Available: https://aclanthology.org/2021.emnlp-main.328/

[33] J. Zhang et al., “Uni-navid: A video-based vision-language-action model for unifying embodied navigation tasks,” 2025. [Online]. Available: https://arxiv.org/abs/2412.06224

[34] D. Zheng, S. Huang, L. Zhao, Y. Zhong, and L. Wang, “Towards learning a generalist model for embodied navigation,” 2024. [Online]. Available: https://arxiv.org/abs/2312.02010

[35] D. Zheng, S. Huang, Y. Li, and L. Wang, “Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors,” arXiv preprint arXiv:2505.24625, 2025.

[36] J. Schult, F. Engelmann, A. Hermans, O. Litany, S. Tang, and B. Leibe, “Mask3d: Mask transformer for 3d semantic instance segmentation,” 2023. [Online]. Available: https://arxiv.org/abs/2210.03105

[37] J. Huang et al., “An embodied generalist agent in 3D world,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, R. Salakhutdinov et al., Eds., vol. 235. PMLR, 21–27 Jul 2024, pp. 20 413–20 451. [Online]. Available: https://proceedings.mlr.press/v235/huang24ae.html