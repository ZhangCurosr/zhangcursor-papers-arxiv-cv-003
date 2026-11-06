# Reinforcing End-to-End Autonomous Driving Safety via Negative-Aware Planning in a Unified World-Action Model

Zihao Zhang<sup>1,2</sup>, Xinyu Zhang<sup>1</sup>, Bing Zeng<sup>1</sup>, Xingyu Chen<sup>3</sup>, Zhenning Li<sup>4</sup>, Bin Yu<sup>5</sup>, Ziyuan Pu<sup>1‡</sup>

<sup>1</sup>Southeast University <sup>2</sup>Shenzhen Loop Area Institute <sup>3</sup>Guangxi University <sup>4</sup>University of Macau <sup>5</sup>Beihang University

Abstract—World-action models (WAMs) jointly model future traffic states and driving actions for end-to-end autonomous driving in intelligent transportation systems. However, their action learning remains dominated by expert-trajectory imitation, leaving unsafe alternatives close to expert trajectories insufficiently constrained. To improve WAM robustness in safetycritical scenarios, we propose NegWAM, a safety post-training framework for unified WAMs that learns explicit safety boundaries from near-expert failure trajectories. NegWAM constructs negative trajectories that are geometrically close to the expert trajectory yet carry safety risks, uses them as repulsive anchors to steer predictions away from dangerous behavior, and encodes them as scene-conditioned rider tokens to learn scene–trajectory compatibility. During training, NegWAM jointly optimizes the flow-matching and near-expert negative objectives, while the auxiliary rider branch is discarded at inference, incurring no additional overhead. On NAVSIM v2, NegWAM achieves 40.4 EPDMS on navhard, a 4.1 % relative improvement over the strongest compared scorer-free WAM baseline, and achieves 93.3 EPDMS on navtest. These results demonstrate that NegWAM achieves strong performance in safety-critical scenarios while preserving routine-driving performance. Zero-shot closed-loop evaluation on HUGSIM further demonstrates cross-benchmark generalization. The code is available at https://github.com/Zizizi hao/NegWAM.

Index Terms—Autonomous driving, contrastive learning, methods for safety, world-action models.

## I. INTRODUCTION

End-to-end autonomous driving systems typically learn driving policies from large-scale expert demonstrations and perform well in routine scenarios [1], [2]. However, hazardous behaviors and failure cases are scarce in real-world driving data, making it difficult for imitation learning to acquire explicit safety boundaries [3], [4]. World-action models (WAMs) introduce stronger priors over scene dynamics into end-toend autonomous driving by predicting future world states and leveraging future representations to guide planning [5]–[7]. Yet their action branches are still trained primarily through flow matching on expert trajectories. This supervision tells the model what to imitate, but not which near-expert alternatives should be rejected as unsafe. Consequently, stronger world modeling alone does not establish explicit safety boundaries, motivating safety-oriented post-training for WAMs.

Post-training offers a natural avenue beyond these limitations, but a central challenge is constructing learning signals that faithfully encode desired driving behavior [4], [8]. Existing methods primarily rely on two sources of supervision. One line of work (Figure 1a) explores policies in closedloop environments built with a simulator [9], [10] or a world model [11], [12], but is constrained by the sim-to-real gap, environment construction costs, and reconstruction artifacts [12], [13]. Another line (Figure 1b) scores candidate trajectories using rule-based evaluators or learned reward models, but the quality of its supervision is limited by handcrafted rules, privileged information, and evaluator bias [14]–[16]. These limitations underscore the need for additional sources of effective supervision for autonomous driving post-training.

![](images/3909ea68cba0b7c9149fe8304b37bc5f2dbea23cf86da8c6a05895d994a0c1d5.jpg)  
Fig. 1. Sources of supervision for post-training in end-to-end autonomous driving. (a) Environment-derived rewards from closed-loop interaction with a simulator or world model. (b) Trajectory scores from an external rule-based or learned scorer. (c) NegWAM uses offline-mined negative trajectories as repulsive anchors and scene-conditioned rider tokens to introduce negativeaware losses into a unified world–action model.

Unsafe driving behaviors provide critical cues for delineating safety boundaries and are often easier to characterize than the full range of desirable driving behavior [17], [18]. Building on this insight, we propose NegWAM, a negative-aware safety post-training method that incorporates offline negative supervision into a unified WAM (Figure 1c). Specifically, nearexpert negatives serve as repulsive anchors that mark locally unsafe regions in trajectory space and as scene-conditioned rider tokens that enable shared attention layers to learn scene– trajectory compatibility. Together with the original imitation objective, these mechanisms enable the WAM to learn not only how experts drive but also which behaviors should be avoided in the current scene.

Unlike prior hard-negative learning that primarily regularizes planner outputs [17], NegWAM post-trains both the single-rollout action flow and the shared world–action representation of a unified WAM. It mines near-expert trajectories that fail offline safety evaluation and jointly optimizes the original flow-matching objective with two hard-negative objectives. The evaluator is used only for offline mining, while the rider branch and compatibility head are removed at inference, adding no inference-time modules. On NAVSIM v2 [19], [20], NegWAM achieves 40.4 EPDMS on navhard, a 4.1 % relative improvement over the strongest compared scorerfree WAM baseline, and achieves 93.3 EPDMS on navtest, surpassing the strongest baseline score of 92.9 by 0.4 points and demonstrating stronger challenging-scenario performance without compromising routine driving. Zero-shot closed-loop evaluation on HUGSIM [21] further supports cross-benchmark transfer, with NegWAM achieving a higher HD-Score than DriveWAM (30.6 vs. 27.2).

In summary, our key contributions are as follows:

• We revisit unified world–action models through safety post-training, showing that augmenting flow-matching imitation with near-expert negative supervision improves challenging-scenario performance without compromising routine driving.

• We propose NegWAM, a near-expert negative safety posttraining method for WAMs that jointly optimizes the original flow-matching objective and two complementary near-expert negative objectives without introducing additional inference-time overhead.

• We introduce two complementary near-expert negative mechanisms. Repulsive anchors push predictions away from locally unsafe trajectories, while scene-conditioned rider tokens enable the unified WAM to determine whether a trajectory is compatible with the current scene.

The remainder of this paper is organized as follows. Section II reviews related work on end-to-end autonomous driving, world–action models, and policy post-training. Section III introduces NegWAM, including its core components and training objective. Section IV evaluates its planning performance and zero-shot closed-loop generalization, with ablation studies and supporting analyses. Section V summarizes the main findings and discusses limitations and future directions.

## II. LITERATURE REVIEW

## A. End-to-End Autonomous Driving

End-to-end autonomous driving maps sensory observations to planned trajectories through a learned policy [2], [22]. Joint optimization connects scene understanding with planning and reduces information loss across separately designed modules [1]. Scaling driving data further improves the coverage of learned behavior [23], while generative future prediction introduces temporal structure and motion priors into policy learning [24], [25]. These developments strengthen the policy’s ability to represent traffic scenes and reproduce expert decisions across diverse driving conditions.

Expert demonstrations nevertheless provide an uneven account of driving behavior: successful maneuvers are abundant, whereas safety-critical failures and recovery interactions are comparatively scarce [3], [4], [12]. Expert-trajectory supervision identifies a desirable action for the observed scene, but supplies few explicit comparisons with nearby unsafe alternatives. Such alternatives are informative because geometric similarity to an expert trajectory does not establish that an action is safe in the same traffic context [17], [26]. Learning to distinguish these alternatives introduces a specific supervision requirement beyond reproducing the demonstrated trajectory.

## B. World-Action Models for Driving

Pretrained video generative models provide temporal and motion priors that can be transferred to action learning [5], [7], [27]. Their role in driving has expanded from supporting scene synthesis and simulation [11], [28] to providing future representations for trajectory prediction [29]–[31]. Recent approaches further couple future visual modeling with action generation, bringing anticipated scene evolution into the policy’s prediction process [5], [6], [32]. This progression makes world modeling increasingly relevant to how driving actions are represented and generated.

Recent work [7], [33] realizes this coupling through shared video–action transformers trained using flow matching. Such architectures allow action supervision to shape the shared world–action representations. Their training objectives, however, focus on fitting demonstrated video–action sequences without explicitly contrasting expert trajectories with undesirable alternatives. This raises the question of how negative trajectories can provide supervision for both action generation and scene–trajectory compatibility during safety post-training.

## C. Post-Training for Driving

Driving policy post-training introduces feedback beyond expert matching through simulated rollouts, reward-based optimization, and trajectory preferences [3], [4]. Closedloop learning uses synthetic simulators such as CARLA [9], [34], reconstructed scenes, or learned environments to expose policies to the consequences of their actions [12], [13]. These mechanisms often overlap: DreamerAD [14] combines latent-world rollouts with learned reward feedback, while WorldRFT [15] uses collision-aware rewards for reinforcement fine-tuning. Complementary to policy post-training, TOAD [16] uses a learned trajectory scorer to guide test-time search without retraining the policy.

Preference-based methods express the additional supervision through comparisons between trajectories. TakeAD [35] derives preferences from expert takeover data. DriveDPO [26] combines imitation similarity with rule-based safety scores for policy learning, then applies Safety DPO using human-like but unsafe rejected trajectories. Its preference construction thus retains explicit safety evaluation without requiring a separately trained reward model. BeyondDrive [17] develops this failurebased perspective through a flow-matching generator, expertproximity and safety filtering, and a repulsive distance loss defined on predicted trajectories. These studies establish the value of near-expert failures as supervision, with distinct choices of policy objective and negative construction.

![](images/c54cba3ad583d95bda36df37500e38f3be606626b701c259a33443e17b6a63b8.jpg)

![](images/e0c6ec844966f4ab4180d71c68417165aaf047d57daa26aa0b31e161210539ba.jpg)  
Fig. 2. Architecture of our NegWAM. In Stage 1, we adapt a pretrained world-action DiT into a unified world–action model through flow matching. I Stage 2, scene-specific hard-negative trajectories are used as both repulsive anchors and trajectory riders for compatibility learning. The auxiliary rider branch is removed at inference, introducing no additional deployment cost.

For a unified WAM, an unsafe alternative is both a trajectory to avoid and an example of incompatibility between the scene and an action. NegWAM learns these two aspects from the same offline-mined failures: repulsive anchors supervise the generated action trajectory, while rider tokens expose the expert and negative trajectories to the observed scene through shared attention for compatibility learning. The original flowmatching objective remains active throughout post-training. Safety evaluation supplies the offline negative labels; subsequent optimization internalizes this supervision in the WAM, and the auxiliary rider branch is removed at inference.

## III. METHODS

NegWAM post-trains a unified world–action model using scene-specific failures close to expert trajectories. Each retained failure supplies two related learning signals: a motion to avoid during action generation and a scene–trajectory pair to distinguish from the expert. Figure 2 illustrates how repulsive anchors and scene-conditioned rider tokens introduce these signals into the same backbone while retaining its flowmatching objective.

## A. Preliminaries

1) Unified World–Action Model: To couple future scene modeling with action generation in one backbone, we adopt a unified world–action architecture [7], [27] built on a pretrained video backbone [36] as the base model. A training sample comprises a current camera image $\mathcal { T } ^ { \mathrm { o b s } }$ , future visual targets ${ \cal T } ^ { \mathrm { f u t } }$ , an expert trajectory $\tau ^ { + }$ , and navigation and egostate conditions including four historical ego poses. A frozen causal video VAE encodes the observed and future frames into the clean visual stream $\mathbf { x } _ { 0 } ^ { v }$ . The action tokenizer forms $\mathbf { x } _ { 0 } ^ { a } = E _ { a } ( \tau ^ { + } )$ by normalizing the expert’s eight future ∆-pose steps and prepending a zero conditioning chunk. Prediction masks exclude the observed visual prefix and the conditioning action chunk from the losses. We use $m \in \{ v , a \}$ for the two modalities.

For each future chunk and modality, we independently sample Gaussian noise $\epsilon ^ { m }$ and a scheduler index, whose shifted noise scale defines $t ^ { m } \in [ 0 , 1 ]$ . We suppress chunk indices below. On prediction elements, flow matching constructs

$$
\mathbf { x } _ { t } ^ { m } = ( 1 - t ^ { m } ) \mathbf { x } _ { 0 } ^ { m } + t ^ { m } \epsilon ^ { m } , \qquad \mathbf { v } ^ { m } : = \frac { \partial \mathbf { x } _ { t } ^ { m } } { \partial t ^ { m } } = \epsilon ^ { m } - \mathbf { x } _ { 0 } ^ { m } .\tag{1}
$$

Let $\mathbf { t } ~ = ~ \left( t ^ { v } , t ^ { a } \right)$ , the shared DiT receives a four-segment sequence

$$
S _ { \mathbf { t } } = \big [ \mathbf { x } _ { t } ^ { v } \ \big \| \mathbf { x } _ { c } ^ { v } \ \big \| \mathbf { x } _ { t } ^ { a } \ \big \| \mathbf { x } _ { c } ^ { a } \big ] ,\tag{2}
$$

where $\mathbf { x } _ { c } ^ { m }$ is the corresponding teacher-forcing conditioning stream. During training, conditioning chunks are exposed only according to their causal temporal positions; future targets are not available as observations at inference. The video conditioning stream may receive noise augmentation, while its observed prefix remains clean. The action conditioning stream uses the zero initial chunk and preceding expert-action chunks. All streams share DiT blocks, with separate modality input and output projections. Under the original block-causal mask $\boldsymbol { A } _ { \mathrm { b a s e } } .$ , noisy tokens can use preceding conditioning chunks and other noisy tokens in the same temporal block; conditioning tokens retain causal visibility. A finite temporal window limits their history. NegWAM preserves this mask for all original tokens.

2) Flow Matching: Flow matching [37]–[39] learns a velocity field along a continuous interpolation between the data distribution and Gaussian noise. The shared DiT jointly predicts the video and action velocity fields:

$$
( \hat { \mathbf { v } } _ { \theta } ^ { v } , \hat { \mathbf { v } } _ { \theta } ^ { a } ) = F _ { \theta } ( S _ { \mathbf { t } } ; \mathbf { c } ) ,\tag{3}
$$

where c contains command guidance and ego-state conditions. The original training objective is

$$
\mathcal { L } _ { \mathrm { F M } } = \sum _ { m \in \{ v , a \} } \mathbb { E } \left[ w _ { m } ( t ^ { m } ) \frac { \| \mathbf { M } ^ { m } \odot \big ( \hat { \mathbf { v } } _ { \theta } ^ { m } - \mathbf { v } ^ { m } \big ) \| _ { 2 } ^ { 2 } } { \| \mathbf { M } ^ { m } \| _ { 1 } } \right] ,\tag{4}
$$

where ${ \bf { M } } ^ { m }$ selects valid future prediction elements and $w _ { m } ( t ^ { m } )$ is the scheduler’s normalized flow-matching weight. The scheduler grid, sampling rule, and weight function are specified in the supplementary implementation details. Under this parameterization, the clean action estimate and its decoded trajectory are

$$
\hat { \mathbf { x } } _ { 0 } ^ { a } = \mathbf { x } _ { t } ^ { a } - t ^ { a } \hat { \mathbf { v } } _ { \theta } ^ { a } , \qquad \hat { \boldsymbol { \tau } } = D _ { a } ( \hat { \mathbf { x } } _ { 0 } ^ { a } ) ,\tag{5}
$$

where $D _ { a }$ selects the future action chunk, denormalizes its $\Delta \cdot$ -pose channels, and cumulatively integrates them into egoframe poses. Our method preserves this flow-matching pathway and appends trajectory riders as additional action-token groups processed by the shared DiT.

## B. Near-Expert Negative Safety Post-Training

1) Near-Expert Negative Trajectory Construction: Directly sampling failure trajectories from the current policy can be ineffective under strong scene conditioning, consistent with prior observations of limited sampling diversity [4]. In a diagnostic probe using the same sampling configuration as the main experiments, the average pairwise ADE among eight trajectories generated for the same scene is only 0.02 m, indicating limited diversity in this probe. We therefore mine negative trajectories from the 16K dense trajectory vocabulary of GTRS [40]. Each vocabulary trajectory $\bar { \tau } _ { j }$ contains 40 egoframe poses over a 4 s horizon. The subsampling operator $S$ takes every fifth pose, corresponding to 0.5 s intervals, to obtain the eight-step trajectory $\tau _ { i , j } ~ = ~ S ( \bar { \tau } _ { j } )$ aligned with the expert $\tau _ { i } ^ { + }$ . We measure their distance using the Average Displacement Error (ADE) over their planar positions $\bar { \mathbf { p } ^ { t } } = ( x ^ { \bar { t } } , y ^ { t } )$

$$
a _ { i , j } = \mathrm { A D E } ( \tau _ { i , j } , \tau _ { i } ^ { + } ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left. \mathbf { p } _ { i , j } ^ { t } - \mathbf { p } _ { i } ^ { + , t } \right. _ { 2 } .\tag{6}
$$

Algorithm 1 separates offline failure mining from the preparation of training negatives. For each training scene, we first retain vocabulary candidates with $a _ { i , j } < 1 . 0 \mathrm { m }$ . We evaluate their original 40-step trajectories using the scene’s NAVSIM v2 metric cache and log-replay traffic. All candidates with zero EPDMS [20] are cached as evaluator-defined failures, forming a scene-specific pool $\mathcal { H } _ { i }$

Algorithm 1 Near-Expert Negative Trajectory Construction   
Require: Training scenes s<sub>i</sub> with expert trajectories $\tau _ { i } ^ { + } ;$ ; 40-step   
vocabulary ${ \ v V } = \{ \bar { \tau } _ { j } \} ;$ offline EPDMS evaluator E   
Require: $d _ { \operatorname* { m i n } } \ = \ 0 . 3 \mathrm { { i m } } .$ , d<sub>max</sub> = 1.0 m, $K \ = \ 1 6 ;$ eight-step   
subsampling operator $S$   
Ensure: Cached failure pools $\mathcal { H } _ { i }$ and per-read negatives, expert   
ADEs, and validity masks   
1: Offline scene-conditioned failure mining   
2: for each training scene $s _ { i }$ do   
3: Compute expert ADEs $a _ { i , j }$ on $S ( \bar { \tau } _ { j } )$ for all candidates   
4: ${ \mathcal { C } } _ { i } \gets \{ \bar { \tau } _ { j } \in \stackrel { \cdot } { \gamma } : a _ { i , j } < d _ { \operatorname* { m a x } } \}$   
5: $\mathcal { H } _ { i }  \bar { \{ \bar { \tau } \in \mathcal { C } _ { i } : \mathcal { E } ( s _ { i } , \bar { \tau } ) = 0 \} }$   
6: Cache all trajectories in $\mathcal { H } _ { i }$ at their original 40 steps   
7: end for   
8: Preparing a training negative set   
9: for each training-sample read for scene i do   
10: Load $\mathcal { H } _ { i } ;$ subsample each trajectory to eight steps using S   
11: Compute expert $\mathrm { \bf A D E } ~ a _ { i , j }$ for each loaded trajectory   
12: $\mathcal { T } _ { i } \stackrel { \cdot } {  } \{ j : a _ { i , j } \geq d _ { \operatorname* { m i n } } \} ;$ q<sub>i</sub> ← min $\left( K , \left| { \mathcal { I } } _ { i } \right| \right)$   
13: Draw $q _ { i }$ indices from $\mathcal { J } _ { i }$ uniformly without replacement   
14: Initialize K trajectory, ADE, and validity entries to zero   
15: Fill the first $q _ { i }$ entries with selected trajectories and ADEs   
16: Set $M _ { i , k } ^ { - } = \mathrm { i }$ for $k \leq q _ { i } ;$ all padding remains masked   
17: Output $\ddot { \{ ( \tau _ { i , k } ^ { - } , a _ { i , k } , M _ { i , k } ^ { - } ) \} } _ { k = 1 } ^ { K }$ 1   
18: end for

At each training iteration, the cached pool $\mathcal { H } _ { i }$ of the corresponding scene is subsampled to eight steps per trajectory, and the expert-relative ADEs are recomputed on the subsampled poses. We retain candidates with ADE at least 0.3 m, giving the final range [0.3, 1.0) m. From these candidates, we randomly select at most $K \ : = \ : 1 6$ without replacement and zero-pad the trajectory and ADE tensors. A validity mask $M _ { i , k } ^ { - }$ distinguishes selected negatives from padding. The same selected subset supplies both hard-negative objectives for that update; different reads may select different subsets from the cached pool. Samples without valid negatives contribute only the flow-matching loss. EPDMS is queried only during offline mining.

This procedure yields at least one valid hard negative for 85.8 % of the training samples. Across all valid, unpadded negatives, the median ADE to the expert trajectory is 0.80 m, and approximately 9 % fall within [0.3, 0.5] m.

2) Noise-Aware Anchor Repulsion: We treat the selected negative trajectories as fixed repulsive anchors in trajectory space. Let $d _ { i , k }$ denote the ADE between the one-step decoded trajectory $\hat { \tau } _ { i }$ from Eq. (5) and the k-th negative $\tau _ { i , k } ^ { - }$ , and let $\boldsymbol { a } _ { i , k }$ denote the ADE between this negative and the expert trajectory. A fixed margin cannot accommodate negatives at different distances: a small margin leaves many negatives inactive, whereas a large one may conflict with the imitation objective. We therefore define the repulsion loss using a distance-adaptive margin:

$$
\ell _ { i , k } ^ { \mathrm { r e p } } = [ \alpha a _ { i , k } - d _ { i , k } ] _ { + } , \qquad 0 < \alpha \leq 1 .\tag{7}
$$

We use $\alpha ~ = ~ 0 . 9$ . The loss is active within the distancedependent neighborhood of a negative anchor. At the exact expert trajectory, $d _ { i , k } = a _ { i , k }$ , so this repulsion term is zero for $\alpha \leq 1$

Since one-step trajectory estimates are less reliable at high noise levels, we weight the repulsion term by $1 - t _ { i } ^ { a }$ and average it over valid negatives:

$$
\mathcal { L } _ { \mathrm { r e p } } = \mathbb { E } _ { i \sim \mathcal { B } } \left[ ( 1 - t _ { i } ^ { a } ) \frac { \sum _ { k } M _ { i , k } ^ { - } \left[ \alpha a _ { i , k } - d _ { i , k } \right] _ { + } } { \operatorname* { m a x } ( 1 , \sum _ { k } M _ { i , k } ^ { - } ) } \right] ,\tag{8}
$$

where B is the training minibatch. A sample without valid negatives contributes zero. Differentiating the reconstruction in Eq. (5) introduces a factor $t _ { i } ^ { a }$ ; together with the loss weight, this gives a $t _ { i } ^ { a } ( 1 - t _ { i } ^ { a } )$ prefactor that vanishes at the noise endpoints. The actual gradient also depends on the trajectory decoder and active hinge terms. Repulsion reuses the original action output without another backbone forward pass and defines its supervision through trajectory distances.

3) Scene-Conditioned Rider Tokens: The same failures also define scene–trajectory comparisons. We encode the expert and each negative as a separate auxiliary rider group $\mathcal { R } _ { g } ,$ using the shared action embedding and the same rotary positional encoding grid. Each group contains a zero conditioning chunk and an eight-step trajectory chunk, giving 16 action tokens. Riders are clean, with no diffusion noise and a flow-time input $t = 0 ,$ . The expert occupies group zero; the remaining groups represent the negatives. Figure 3 illustrates the shared attention and scoring pathway.

The riders access the scene through shared attention while remaining invisible to the original streams. Let $\mathbf { x } _ { c } ^ { v , \mathrm { o b s } }$ contain the clean visual conditioning tokens of the current observed frame. At every DiT block, the attention visibility is

$$
\begin{array} { r } { A ( p , q ) = \left\{ \begin{array} { l l } { A _ { \mathrm { b a s e } } ( p , q ) , } & { p , q \in S _ { \mathrm { t } } , } \\ { 1 , } & { p \in \mathcal { R } _ { g } , \ q \in \mathbf { x } _ { c } ^ { v , \mathrm { o b s } } \cup \mathcal { R } _ { g } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{9}
$$

Thus, each rider sees only the current observed visual tokens and its own trajectory group. Future target tokens are excluded from these keys, and the observed tokens retain their original causal visibility. The original sequence cannot attend to riders, so appending them preserves its forward attention path at fixed backbone weights. Matching the expert and negative groups tokenization, positions, and visibility reduces structural differences in their scoring inputs.

We average the output features over all 16 tokens in each group and apply a shared compatibility head $h _ { \phi } \colon$ a linear projection to 256 dimensions, SiLU, and a scalar linear output. Zero-initializing the final layer’s weights and bias gives identical initial scores and zero initial compatibility gradient to the backbone; this gradient path becomes active as the head’s output weights are updated. Let $s _ { i } ^ { + }$ and $s _ { i , k } ^ { - }$ be the expert and negative scores. For $\begin{array} { r } { B _ { + } = \{ i \in \mathcal { B } : \sum _ { k } M _ { i , k } ^ { - } > 0 \} } \end{array}$ , we optimize

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { c o m p a t } } = - \mathbb { E } _ { \mathcal { B } _ { + } } \log \frac { \exp ( s _ { i } ^ { + } / \tau _ { s } ) } { Z _ { i } } , } } \\ & { } & { Z _ { i } = \exp ( s _ { i } ^ { + } / \tau _ { s } ) + \sum _ { k } M _ { i , k } ^ { - } \exp ( s _ { i , k } ^ { - } / \tau _ { s } ) . } \end{array}\tag{10}
$$

![](images/d471f495015e4ff17318b9ffafa5254e68e64089e21613e7565192cce3b6d9de.jpg)  
Fig. 3. Scene-conditioned rider tokens for scene–trajectory compatibility learning. Expert and negative riders attend to observed scene tokens through shared DiT attention, and their compatibility scores define a ranking loss that updates the shared backbone.

where $\tau _ { s } ~ = ~ 0 . 1$ . The expectation denotes the mean over samples with valid negatives; we set $\mathcal { L } _ { \mathrm { c o m p a t } } = 0$ when $B _ { + }$ is empty. This objective ranks the expert above valid scenespecific negatives. Its gradients propagate through the shared attention projections, adding an explicit scene–trajectory ranking objective to the backbone. Rider groups and the compatibility head are used only during post-training.

## C. Training Objective

Our training procedure consists of two consecutive stages. In the first stage, we initialize the model from the pretrained video generative model Wan2.2-TI2V-5B [36] and adapt it to autonomous driving by following the unified world–action flow-matching paradigm of DriveWAM [7]. This stage optimizes only the flow-matching objective defined in Section III-A, enabling the model to jointly learn future visual dynamics and expert trajectory prediction. The resulting model serves as the base WAM checkpoint shared by all subsequent experiments.

In the second stage, we initialize from this shared checkpoint and perform hard-negative safety post-training. We retain the original flow-matching objective as supervision for visual dynamics and expert actions, while introducing repulsive trajectory supervision and scene-conditioned compatibility learning. The overall post-training objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p o s t } } = \mathcal { L } _ { \mathrm { F M } } + \lambda _ { \mathrm { r e p } } \mathcal { L } _ { \mathrm { r e p } } + \lambda _ { \mathrm { c o m p a t } } \mathcal { L } _ { \mathrm { c o m p a t } } , } \end{array}\tag{11}
$$

where $\mathcal { L } _ { \mathrm { r e p } }$ pushes predictions away from unsafe neighbors in trajectory space, and $\mathcal { L } _ { \mathrm { { c o m p a t } } }$ learns scene–trajectory compatibility through rider tokens. $\lambda _ { \mathrm { { r e p } } }$ and $\lambda _ { \mathrm { c o m p a t } }$ control the contributions of the two hard-negative objectives. All posttraining variants start from the same first-stage checkpoint, controlling for differences in base-model initialization.

TABLE I  
PERFORMANCE COMPARISON ON THE NAVSIM V2 navhard BENCHMARK. S1 DENOTES EVALUATION ON THE INITIAL REAL-WORLD SCENES, WHILE S2 DENOTES EVALUATION ON THE PRE-GENERATED REACTIVE FOLLOW-UP SCENES. ALL RESULTS ARE REPORTED USING THE CORRECTED OFFICIAL NAVSIM V2.2 EVALUATION PROTOCOL. MV AND SV DENOTE MULTI-VIEW AND SINGLE-VIEW CAMERA INPUTS, RESPECTIVELY. DRIVEFUTURE<sup>†</sup> DENOTES ITS VARIANT WITHOUT THE GTRS-DENSE SCORER; THE FULL SYSTEM REPORTS 55.5 WITH THE GTRS-DENSE SCORER [40].
<table><tr><td>Methods</td><td>Sensor</td><td>Stage</td><td>NC↑</td><td>DAC ↑</td><td>DDC ↑</td><td>TLC ↑</td><td>EP↑</td><td>TTC ↑</td><td>LK↑</td><td>HC ↑</td><td>EC ↑</td><td>EPDMS ↑</td></tr><tr><td>PDM-Closed</td><td></td><td>S1 S2</td><td>94.4 90.5</td><td>98.8 90.6</td><td>100 95.4</td><td>99.5 98.4</td><td>100 100</td><td>93.5 86.6</td><td>99.3 74.2</td><td>87.7 91.9</td><td>36.0 29.7</td><td>56.6</td></tr><tr><td colspan="9">E2E-based Methods</td><td></td><td></td><td></td></tr><tr><td>DiffusionDrive [41]</td><td>MV</td><td>S1</td><td>96.0</td><td>79.7</td><td>97.4</td><td>99.5</td><td>81.3</td><td>93.1</td><td>90.8</td><td>96.8</td><td>73.8</td><td>24.2</td></tr><tr><td>GuideFlow [42]</td><td>MV</td><td>S2 S1</td><td>82.1 96.6</td><td>72.2 80.5</td><td>88.5 96.3</td><td>98.7 99.3</td><td>85.1 82.3</td><td>78.8 94.9</td><td>49.2 91.5</td><td>89.3 97.7</td><td>71.2 67.8</td><td></td></tr><tr><td></td><td></td><td>S2</td><td>87.3</td><td>76.7</td><td>88.8</td><td>99.2</td><td>84.3</td><td>85.1</td><td>49.7</td><td>93.1</td><td>44.2</td><td>27.1</td></tr><tr><td colspan="9">VLA-based Methods</td><td></td><td></td><td></td></tr><tr><td>DriveVLA-W0 [43]</td><td>SV</td><td>S1 S2</td><td>96.8 76.8</td><td>83.3 64.3</td><td>99.0 79.9</td><td>99.6 98.3</td><td>84.6 89.2</td><td>95.3 75.0</td><td>96.4 46.8</td><td>97.6 95.8</td><td>78.2 53.1</td><td>24.4</td></tr><tr><td>SpanVLA [44]</td><td>MV</td><td>S1</td><td>98.4</td><td>94.3</td><td>97.8</td><td>99.9</td><td>85.7</td><td>97.2</td><td>94.2</td><td>97.6</td><td>72.1</td><td>40.1</td></tr><tr><td></td><td>World-Model-based Methods</td><td>S2</td><td>86.9</td><td>84.3</td><td>87.1</td><td>98.2</td><td>85.5</td><td>82.7</td><td>62.3</td><td>96.8</td><td>67.4</td><td></td></tr><tr><td colspan="9"></td><td></td><td></td><td></td></tr><tr><td>DriveLAW [5]</td><td>SV</td><td>S1 S2</td><td>97.1 82.3</td><td>88.7 67.9</td><td>99.2 83.7</td><td>99.6 98.1</td><td>84.2 84.7</td><td>97.1 78.5</td><td>96.2 46.7</td><td>97.8 96.3</td><td>69.8 60.6</td><td>30.4</td></tr><tr><td>DriveFuture† [45]</td><td>MV</td><td>S1 S2</td><td>96.3 82.3</td><td>87.8 78.8</td><td>98.2 88.1</td><td>99.6 98.4</td><td>83.1 83.6</td><td>96.9 79.6</td><td>94.9 47.6</td><td>97.6 97.0</td><td>76.9</td><td>34.6</td></tr><tr><td>EponaV2 [6]</td><td>SV</td><td>S1 S2</td><td>97.3</td><td>90.7</td><td>99.4</td><td>100</td><td>83.3</td><td>97.3</td><td>97.3</td><td>97.6</td><td>75.9 60.9</td><td>36.1</td></tr><tr><td>DriveWAM [7]</td><td>SV</td><td>S1</td><td>83.6 96.8</td><td>78.0 88.7</td><td>88.0 99.1</td><td>98.9 99.6</td><td>86.0 95.9</td><td>80.3 95.6</td><td>50.1 95.6</td><td>96.1 97.8</td><td>52.0 80.4</td><td></td></tr><tr><td>BeyondDrive [17]</td><td>SV</td><td>S2 S1</td><td>79.5 96.6</td><td>75.8 93.1</td><td>87.8 98.8</td><td>98.1 99.6</td><td>96.5 96.9</td><td>75.7 95.8</td><td>48.7 96.9</td><td>95.8 97.8</td><td>73.7 78.7</td><td>37.7</td></tr><tr><td></td><td></td><td>S2 S1</td><td>81.2 96.6</td><td>77.7 94.4</td><td>85.6 98.8</td><td>98.2 99.6</td><td>97.0 95.8</td><td>77.7 96.0</td><td>47.8 96.4</td><td>95.4 97.8</td><td>72.2 79.6</td><td>38.8</td></tr><tr><td>Ours</td><td>SV</td><td>S2</td><td>79.2</td><td>78.0</td><td>86.4</td><td>98.1</td><td>96.3</td><td>75.3</td><td>48.4</td><td>95.7</td><td>73.7</td><td>40.4</td></tr></table>

At inference, we discard the rider groups and compatibility head and use the original world–action generation pathway with the updated backbone. The model encodes the available observations, generates future visual latents with three Euler updates, and then generates the eight-step action chunk with ten Euler updates. These are integration-step counts rather than total backbone forward-call counts. The action decoder converts the generated tokens to the final trajectory; no hardnegative candidates or evaluator are used at this stage.

## IV. EXPERIMENTS AND RESULTS

In this section, we investigate the following questions:

• Does NegWAM improve planning performance on both challenging and routine benchmarks, and do these gains transfer to zero-shot closed-loop evaluation? (Section IV-B)

• What are the individual and combined contributions of repulsive anchors and scene-conditioned rider tokens, and how sensitive is performance to the post-training budget and hyperparameters? (Section IV-C)

• What qualitative driving behaviors are observed in trajectory comparisons and closed-loop rollouts? (Section IV-D)

• What inference cost accompanies the planning improvements? (Section IV-E)

## A. Experimental Setup

1) Datasets and Benchmarks: NAVSIM [19] is a real-world autonomous driving dataset built upon OpenScene [46] and nuPlan [47], comprising 103K training samples. We train all models on navtrain and evaluate them on navtest and navhard. To further assess zero-shot closed-loop generalization, we directly evaluate the same checkpoints on HUGSIM [21].

2) Evaluation Metrics: NAVSIM v2. Both navtest and navhard report the Extended Predictive Driver Model Score (EPDMS) [48], which combines multiplicative penalties with a weighted average of sub-metric scores:

$$
\mathrm { E P D M S } = \left( \prod _ { m \in \mathcal { M } _ { \mathrm { p e n } } } S _ { m } \right) \frac { \sum _ { m \in \mathcal { M } _ { \mathrm { a v g } } } w _ { m } S _ { m } } { \sum _ { m \in \mathcal { M } _ { \mathrm { a v g } } } w _ { m } } ,\tag{12}
$$

where $S _ { m }$ and $w _ { m }$ denote the score and weight of submetric m, respectively. The multiplicative penalty set $\mathcal { M } _ { \mathrm { p e n } }$ comprises No At-Fault Collisions (NC), Drivable Area Compliance (DAC), Driving Direction Compliance (DDC), and Traffic Light Compliance (TLC). The weighted-average set $\mathcal { M } _ { \mathrm { a v g } }$ includes Time-to-Collision (TTC), Ego Progress (EP), Lane Keeping (LK), History Comfort (HC), and Extended Comfort (EC). For navhard, scoring is further adapted to the two-stage setting through cross-stage aggregation, reactive traffic simulation, and the exemption of penalties when the corresponding expert rollout exhibits the same failure. All results use the corrected official NAVSIM v2.2 evaluation protocol, which fixes an inconsistency in the human-penalty filtering of the aggregated score [20]. All compared methods follow the same corrected evaluation protocol.

HUGSIM. We report per-level and overall Route Completion (RC) and HD-Score. For an episode of length T, HD-Score combines RC with No Collision (NC), Drivable Area Compliance (DAC), Time to Collision (TTC), and Comfort (COM):

$$
\begin{array} { l } { \displaystyle \mathrm { H D - S c o r e } = \frac { \mathrm { R C } } { T } \sum _ { t = 1 } ^ { T } \Biggl [ \mathrm { N C } _ { t } \mathrm { D A C } _ { t } } \\ { \displaystyle \mathrm { ~ \times ~ } \frac { 5 \mathrm { T T C } _ { t } + 2 \mathrm { C O M } _ { t } } { 7 } \Biggr ] , } \end{array}\tag{13}
$$

where t indexes the simulation time steps. Higher EPDMS, RC, and HD-Score indicate better planning performance.

3) Implementation Details: Our model uses one current camera image at a resolution of 256 × 448, together with four historical ego poses, while the action branch predicts a 4 s future trajectory. We train the model on navtrain in two stages. Following the unified world–action flow-matching paradigm of DriveWAM [7], we initialize the video backbone from Wan2.2-TI2V-5B [36] and perform 100K steps of WAM pretraining to jointly learn future visual dynamics and expert trajectory prediction. Starting from this checkpoint, we conduct 10K steps of hard-negative safety post-training. We retain the original flow-matching objective to preserve the learned world dynamics and expert-driving prior, while introducing the proposed repulsive-anchor and scene-conditioned ridertoken objectives. The auxiliary rider branch is removed at inference, and measured inference efficiency is reported in Section IV-E. Complete model configurations, optimization settings, and training costs are provided in the supplementary material.

For the BeyondDrive [17] baseline in Tables I and II, we port the complete method, rather than only its repulsive loss, to our WAM architecture, strictly following the authors’ original source code and using BeyondDrive’s original negative samples. The reported results are obtained from our own experiments with this implementation and are not taken from the original paper.

## B. Planning Performance and Generalization

1) NAVSIM v2: On navhard, NegWAM achieves 40.4 EPDMS, surpassing the BeyondDrive [17] baseline by 1.6 points, corresponding to a 4.1 % relative improvement, and establishing state-of-the-art performance among WAM without an external trajectory scorer (Table I). On navtest, NegWAM further achieves 93.3 EPDMS, outperforming DriveWAM [7] by 1.1 points and establishing a new state of the art among the compared learning-based methods (Table II). The consistent improvements across both benchmarks show that the proposed negative-aware post-training strategy enhances robustness in challenging scenarios while preserving, and even improving, the general driving capability acquired by the pretrained WAM.

TABLE II  
PERFORMANCE COMPARISON ON THE NAVSIM V2 navtest BENCHMARK. ALL RESULTS ARE REPORTED USING THE CORRECTED OFFICIAL NAVSIM V2.2 EVALUATION PROTOCOL. MV AND SV DENOTE MULTI-VIEW AND SINGLE-VIEW CAMERA INPUTS, RESPECTIVELY.
<table><tr><td>Methods</td><td>Sensor</td><td>NC ↑</td><td>DAC ↑</td><td>EPDMS ↑</td></tr><tr><td>Human agent</td><td></td><td>100.0</td><td>100.0</td><td>97.9</td></tr><tr><td colspan="5">E2E-based Methods</td></tr><tr><td>DiffusionDrive [41]</td><td>MV</td><td>98.2</td><td>95.9</td><td>84.5</td></tr><tr><td>DiffusionDriveV2 [49]</td><td>MV</td><td>97.7</td><td>96.6</td><td>87.5</td></tr><tr><td colspan="5">VLA-based Methods</td></tr><tr><td>Recogdrive [50]</td><td>MV</td><td>98.3</td><td>95.2</td><td>83.6</td></tr><tr><td>DriveVLA-W0 [43] DriveWorld-VLA [51]</td><td>SV MV</td><td>98.5 98.6</td><td>99.1 99.1</td><td>86.1 86.8</td></tr><tr><td>World-Model-based Methods</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="5"></td></tr><tr><td>Latent-WAM [31]</td><td>MV MV</td><td>98.1 98.8</td><td>97.3 99.1</td><td>89.3</td></tr><tr><td>DriveFuture [45]</td><td>SV</td><td></td><td>96.9</td><td>89.9</td></tr><tr><td>DriveLAW [5]</td><td>SV</td><td>98.8 98.5</td><td></td><td>88.9</td></tr><tr><td>Discrete-WAM [52]</td><td>SV</td><td></td><td>98.2</td><td>90.4</td></tr><tr><td>DriveWAM [7]</td><td>SV</td><td>98.4 98.2</td><td>96.7</td><td>92.2</td></tr><tr><td>BeyondDrive [17] Ours</td><td>SV</td><td>98.1</td><td>97.7 98.4</td><td>92.9 93.3</td></tr></table>

2) Zero-shot Generalization to HUGSIM: Under zero-shot evaluation on HUGSIM [21], NegWAM improves RC and HD-Score by 3.4 % and 12.5 % relative to DriveWAM [7], reaching 45.1 and 30.6, respectively (Table III). The gains are concentrated in the Easy settings, while performance on Hard remains nearly unchanged and both metrics decrease on Extreme. These results show improved aggregate closedloop performance without HUGSIM-specific fine-tuning, with transfer varying across scenario difficulty.

TABLE III  
ZERO-SHOT PERFORMANCE ON THE HUGSIM BENCHMARK. E/M/H/X DENOTE EASY, MEDIUM, HARD, AND EXTREME SCENARIOS. HIGHER IS BETTER FOR BOTH ROUTE COMPLETION (RC) AND HD-SCORE.
<table><tr><td></td><td colspan="5">RC↑</td><td colspan="5">HD-Score ↑</td></tr><tr><td>Method</td><td>E</td><td>M</td><td>H</td><td>X</td><td>Overall</td><td>E</td><td>M</td><td>H</td><td>X</td><td>Overall</td></tr><tr><td>VAD [53]</td><td>38.7</td><td>27.0</td><td>25.5</td><td>23.0</td><td>27.9</td><td>24.3</td><td>9.9</td><td>10.4</td><td>8.2</td><td>12.3</td></tr><tr><td>LTF [19]</td><td>68.4</td><td>40.7</td><td>36.9</td><td>25.5</td><td>41.4</td><td>52.8</td><td>24.6</td><td>19.8</td><td>8.1</td><td>24.8</td></tr><tr><td>DriveWAM [7]</td><td>76.6</td><td>33.8</td><td>33.9</td><td>30.2</td><td>43.6</td><td>64.0</td><td>16.1</td><td>16.1</td><td>12.6</td><td>27.2</td></tr><tr><td>UniAD [1]</td><td>58.6</td><td>41.2</td><td>40.4</td><td>26.0</td><td>40.6</td><td>48.7</td><td>29.5</td><td>27.3</td><td>14.3</td><td>28.9</td></tr><tr><td>NegWAM (Ours)</td><td>90.3</td><td>3 32.8</td><td>31.2 26.1</td><td></td><td>45.1</td><td>81.7</td><td>15.8</td><td>14.2</td><td>10.8</td><td>30.6</td></tr></table>

## C. Analysis of Near-Expert Negative Supervision

1) Effect of Repulsive Anchors: As shown in Table IV, continuing post-training without either proposed component yields 38.2 EPDMS, only 0.5 points above the pretrained baseline. Introducing repulsive anchors increases EPDMS to 39.1, outperforming the matched post-training control and pretrained baseline by 0.9 and 1.4 points, respectively. This result demonstrates that explicitly separating predictions from hard-negative trajectories provides effective safety supervision beyond additional training alone.

2) Effect of Scene-Conditioned Rider Tokens: Using sceneconditioned rider tokens alone achieves 38.8 EPDMS, outperforming the post-training control and pretrained baseline by 0.6 and 1.1 points, respectively. Combining rider tokens with repulsive anchors further increases EPDMS to 40.4, exceeding the anchor-only and rider-only variants by 1.3 and 1.6 points, respectively. The additional 1.3 EPDMS gain over the anchoronly variant further shows that NegWAM’s improvement cannot be explained by trajectory-space repulsion alone.

TABLE IV  
ABLATION STUDY OF THE PROPOSED POST-TRAINING COMPONENTS. A CHECKMARK INDICATES THAT THE CORRESPONDING COMPONENT IS ENABLED.
<table><tr><td>Post-training Repulsive anchors</td><td>Scene-conditioned rider tokens</td><td>EPDMS</td></tr><tr><td colspan="3">Pretrained Baseline 37.7</td></tr><tr><td>√</td><td></td><td>38.2</td></tr><tr><td>√</td><td>√</td><td>39.1</td></tr><tr><td>√</td><td>√</td><td>38.8</td></tr><tr><td>√</td><td>√ √</td><td>40.4</td></tr></table>

![](images/d0cf2fe437fea95bed2f603aa69c0db7a91675db1a1e74fb24f351886ad10b28.jpg)  
Fig. 4. Scene–trajectory compatibility scores for three example scenes. Rows correspond to the trajectories shown on the left, and columns correspond to the observed scenes shown above. Each cell reports the compatibility score of the corresponding scene–trajectory pair.

Figure 4 further illustrates the scene–trajectory compatibility learned by the rider branch. Pairing three trajectories with scenes from Boston, Pittsburgh, and one-north yields the highest score for each trajectory in its corresponding scene: the diagonal scores are 1.11, 0.32, and 1.58, respectively, and all mismatched pairs receive negative scores. This example complements the component ablation by illustrating the scene dependence of the learned compatibility scores.

3) Sensitivity to Post-Training Budget and Hyperparameters: Following the public NAVSIM v2 protocol, which does not define a separate validation split, we report the complete checkpoint and one-factor-at-a-time sensitivity results on navhard for transparency. As shown in Figure 5, NegWAM consistently outperforms the matched standard posttraining baseline from 10K to 30K steps, confirming that the improvement holds across the post-training budget rather than at a single checkpoint. The corresponding navtest results in Figure 6 also show consistent gains over the baseline from 5K to 30K steps. Table V varies the learning rate, $\lambda _ { \mathrm { { r e p } } } ,$ and $\lambda _ { \mathrm { c o m p a t } }$ around the default configuration used in the main experiments. All tested settings yield 39.0–40.4 EPDMS, showing robustness to moderate hyperparameter changes.

![](images/0cb610d446418d883991a0181c86bc63829331158e1dfb9a35cbb000c537b1d8.jpg)  
Fig. 5. Effect of Post-Training Steps for NAVSIM v2 navhard. NegWAM results are reported as mean ± one standard deviation over three random seeds, while the baseline is a single run.

![](images/3aaeb8eeb54c2b13d68f5342e7f0f60f5c469e9a1a30f7a924ea4e54a73ed004.jpg)  
Fig. 6. Effect of Post-Training Steps for NAVSIMv2 navtest. NegWAM results are reported as mean ± one standard deviation over three random seeds, while the baseline is a single run.

TABLE V  
SENSITIVITY TO THE LEARNING RATE, REPULSION WEIGHT, ANDCOMPATIBILITY WEIGHT.
<table><tr><td colspan="2">Learning rate</td><td colspan="2">Repulsion</td><td colspan="2">Compatibility</td></tr><tr><td>Learning rate</td><td>EPDMS</td><td> $\lambda _ { \mathrm { r e p } }$ </td><td>EPDMS</td><td> $\lambda _ { \mathrm { { c o m p a t } } }$ </td><td>EPDMS</td></tr><tr><td> $5 \times 1 0 ^ { - 7 }$ </td><td>39.5</td><td>50</td><td>39.0</td><td>0.05</td><td>39.3</td></tr><tr><td> $1 \times 1 0 ^ { - 6 }$ </td><td>40.4</td><td>100</td><td>40.4</td><td>0.1</td><td>40.4</td></tr><tr><td> $2 \times 1 0 ^ { - 6 }$ </td><td>39.1</td><td>150</td><td>39.8</td><td>0.2</td><td>39.1</td></tr></table>

## D. Qualitative Analysis

We examine NegWAM’s behavior through trajectory comparisons and a representative closed-loop rollout.

1) Trajectory Comparison: As shown in Figure 7, Neg-WAM maintains safer clearance than DriveWAM [7] and avoids potential collision regions. In the second scenario, an occupant of a roadside parked vehicle opens the door and begins to exit. DriveWAM [7] still produces a nominal close-passing trajectory, leaving insufficient clearance for this emerging hazard. After post-training, NegWAM instead shifts outward and maintains a safe distance from the parked vehicle. Although the baseline trajectory may be safe without the dooropening interaction, it entails substantial risk in this safetycritical context. This comparison suggests that NegWAM internalizes scene-dependent safety boundaries that are difficult to capture through expert imitation alone, enabling it to avoid trajectories that are geometrically close to expert behavior yet potentially unsafe.

Agent EgoPredicted TrajectoryHuman-Expert Trajectory  
![](images/c84aa8a6f8c7fdd26ecb06f2d33760d75801a79dce4b83177aef23c043a80814.jpg)  
Fig. 7. Qualitative comparison between DriveWAM [7] and our NegWAM in safety-critical scenarios. DriveWAM (top) predicts trajectories that approach nearby traffic participants, whereas NegWAM (bottom) maintains safer clearance while better following the human-expert trajectory.

TABLE VI  
PLANNING SCORE ON NAVHARD AND INFERENCE EFFICIENCY. HIGHER EPDMS AND FPS AND LOWER LATENCY, MODEL SIZE, AND PEAK GPU MEMORY ARE PREFERRED.
<table><tr><td>Method</td><td>Latency (ms) ↓</td><td>FPS ↑</td><td>Model size (MB) ↓</td><td>Peak GPU memory (MB) ↓</td><td>EPDMS ↑</td></tr><tr><td>DriveWAM [7]</td><td>871.35</td><td>1.15</td><td>6241.81</td><td>14029.08</td><td>37.7</td></tr><tr><td>NegWAM</td><td>877.86</td><td>1.14</td><td>6241.81</td><td>14029.08</td><td>40.4</td></tr></table>

2) Closed-Loop Rollout in HUGSIM: Figure 8 shows a representative closed-loop rollout in HUGSIM. Across the displayed time points, an oncoming vehicle approaches and passes on the left while the ego vehicle continues along the road. The rendered observations and trajectory overlays illustrate how the driving scene evolves during closed-loop execution.

## E. Inference Efficiency

Table VI compares planning performance on navhard and inference efficiency. NegWAM improves EPDMS from 37.7 to 40.4 while retaining the same model storage size and peak inference GPU memory usage. Measured latency increases from 871.35 to 877.86 ms, a difference of 6.51 ms (0.75 %), while throughput changes from 1.15 to 1.14 FPS. The auxiliary rider groups and compatibility head are removed at inference, allowing the planning gains to be obtained with nearly un changed deployment cost.

![](images/538fb343be9f95837a38c765c24aca8558e86e1edd7f9f40fdaa3887298f90c6.jpg)  
Fig. 8. Representative closed-loop rollout in HUGSIM. Rows are ordered chronologically from top to bottom, with three rendered camera views at each time point. Yellow dots mark the overlaid trajectory in the center view. An oncoming vehicle approaches and passes on the left as the ego vehicle proceeds along the road.

## V. CONCLUSION

We revisit unified world–action models from the perspective of safety post-training and introduce NegWAM to address their reliance on expert imitation and lack of explicit safety boundaries. NegWAM leverages offline-constructed, near-expert hard-negative trajectories as repulsive anchors and scene-conditioned rider tokens, internalizing scene-dependent safety boundaries at both the trajectory and representation levels. Experiments demonstrate that NegWAM improves robustness in challenging scenarios while preserving routine driving performance, without requiring online simulation during posttraining or introducing additional inference overhead. Zeroshot closed-loop evaluation further validates the transferability of these learned safety boundaries to unseen interactive environments.

The generalization of NegWAM to the diversity and unpredictability of real-world driving remains to be established. Learning safety boundaries that remain effective across evolving environments and complex traffic interactions is a broader challenge for world–action models. Future work will investigate more generalizable and adaptive safety learning paradigms, toward integrating predictive world understanding with reliable long-horizon decision-making for autonomous driving.

## REFERENCES

[1] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang et al., “Planning-oriented autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 17 853–17 862.

[2] L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li, “End-toend autonomous driving: Challenges and frontiers,” IEEE Trans. Pattern Anal. Mach. Intell., 2024.

[3] P. Karkus, M. Igl, Y. Chen, K. Chitta, J. Packer, B. Douillard, R. Tian, A. Naumann, G. Garcia-Cobo, S. Tan et al., “Beyond behavior cloning in autonomous driving: a survey of closed-loop training techniques,” Authorea Preprints, 2025.

[4] R. Yang, M. Wang, Y. Chen, T. Guo, Y. Xu, C. Cui, Z. Yang, Y. Zhang, Z. Wang, Y. Fu, and L. Su, “Post-training in end-to-end autonomous driving,” 2026. [Online]. Available: https://arxiv.org/abs/2607.08072

[5] T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, G. Chen, H. Ye et al., “Drivelaw: Unifying planning and video generation in a latent driving world,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026, pp. 39 701–39 712.

[6] J. Xu, Z. Zhong, Z. Shu, M. Jia, M. Li, J.-W. Bian, Q. Zhang, K. Zhang, J. Xie, J. Yang et al., “EponaV2: Driving world model with comprehensive future reasoning,” 2026. [Online]. Available: https://arxiv.org/abs/2605.14696

[7] C. Shi, J. Xu, S. Shi, K. Sheng, B. Zhang, and L. Jiang, “DriveWAM: Video generative priors enable scalable world-action modeling for autonomous driving,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2026.

[8] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023, pp. 53 728–53 741.

[9] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “Carla: An open urban driving simulator,” in Proc. Conf. Robot Learn. (CoRL). PMLR, 2017, pp. 1–16.

[10] B. Kerbl, G. Kopanas, T. Leimkuhler, G. Drettakis ¨ et al., “3d gaussian splatting for real-time radiance field rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[11] J. Yang, K. Chitta, S. Gao, L. Chen, Y. Shao, X. Jia, H. Li, A. Geiger, X. Yue, and L. Chen, “ReSim: Reliable world simulation for autonomous driving,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2025, pp. 167 710– 167 741.

[12] T. Li, L. Chen, C. Wang, H. Liu, K. Chitta, Z. Yang, Y. Lu, N. Ye, Y. Qiu, Y. Wang et al., “World engine: Towards the era of post-training for autonomous driving,” 2026. [Online]. Available: https://arxiv.org/abs/2606.19836

[13] H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, X. Li, W. Liu, Q. Zhang et al., “Rad: Training an end-to-end driving policy via largescale 3dgs-based reinforcement learning,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2025, pp. 32 551–32 576.

[14] P. Yang, Y. Zheng, D. Qian, Z. Xing, Q. Zhang, L. Wang, Y. Zhang, S. Guo, Z. Xia, Q. Chen et al., “DreamerAD: Efficient reinforcement learning via latent world model for autonomous driving,” 2026. [Online]. Available: https://arxiv.org/abs/2603.24587

[15] P. Yang, B. Lu, Z. Xia, C. Han, Y. Gao, T. Zhang, K. Zhan, X. Lang, Y. Zheng, and Q. Zhang, “Worldrft: Latent world model planning with reinforcement fine-tuning for autonomous driving,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 40, no. 14, 2026, pp. 11 649–11 657.

[16] Y. Xu, E. Zablocki, Y. Yin, E. Ramzi, E. Kirby, A. Boulch, and M. Cord, “Test-time trajectory optimization for autonomous driving,” 2026. [Online]. Available: https://arxiv.org/abs/2606.07170

[17] J. Wang, Z. Hua, X. Liu, Z. Xing, H. Tian, K. Ma, H. Ye, G. Chen, L. Chen, and Q. Zhang, “Beyond imitation: Learning safe end-to-end autonomous driving from hard negatives,” 2026. [Online]. Available: https://arxiv.org/abs/2605.19771

[18] Y. Gao, D. Liu, Q. Zhang, Y. Zheng, H. Tian, G. Li, H. Ye, L. Chen, D.-W. Ding, and D. Zhao, “Learning from mistakes: Post-training for driving VLA with takeover data,” 2026. [Online]. Available: https://arxiv.org/abs/2603.14972

[19] D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta, “Navsim: Data-driven nonreactive autonomous vehicle simulation and benchmarking,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2024.

[20] W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta, “Pseudosimulation for autonomous driving,” in Proc. Conf. Robot Learn. (CoRL), 2025.

[21] H. Zhou, L. Lin, J. Wang, Y. Lu, D. Bai, B. Liu, Y. Wang, A. Geiger, and Y. Liao, “Hugsim: A real-time, photo-realistic and closed-loop simulator for autonomous driving,” IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), 2025.

[22] Z. Zhang, H. Tian, T. Li, C. Jing, J. He, N. Ye, Z. Pu, and Z. Yang, “Do better visual representations always lead to better end-to-end autonomous driving?” 2026. [Online]. Available: https://arxiv.org/abs/2610.09695

[23] A. Naumann, X. Gu, T. Dimlioglu, M. Bojarski, A. Degirmenci, A. Popov, D. Bisla, M. Pavone, U. Muller, and B. Ivanovic, “Data scaling laws for end-to-end autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 2571–2582.

[24] X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu, “Drivedreamer: Towards real-world-drive world models for autonomous driving,” in Proc. Eur. Conf. Comput. Vis. (ECCV). Springer, 2024, pp. 55–72.

[25] K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X.-X. Long et al., “Epona: Autoregressive diffusion world model for autonomous driving,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 27 220–27 230.

[26] S. Shang, Y. Chen, Y. Wang, Y. Li, and Z.-X. ZHANG, “Drivedpo: Policy learning via safety dpo for end-to-end autonomous driving,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2025, pp. 81 565–81 585.

[27] L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu et al., “Causal world modeling for robot control,” 2026. [Online]. Available: https://arxiv.org/abs/2601.21998

[28] Z. Yang, Z. Liu, Y. Lu, L. Hou, C. Miao, S. Peng, B. Feng, X. Bai, and H. Zhao, “Geniedrive: Towards physics-aware driving world model with 4d occupancy guided video generation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026, pp. 35 680–35 690.

[29] G. Zhou, H. Pan, Y. LeCun, and L. Pinto, “DINO-WM: World models on pre-trained visual features enable zero-shot planning,” in Proc. 42nd Int. Conf. Mach. Learn. (ICML), 2025, pp. 79 115–79 135.

[30] L. Wang, Z. Yang, C. Bai, G. Zhang, X. Liu, X. Zheng, X.-X. Long, C.-T. Lu, and C. Lu, “Drive-JEPA: Video JEPA meets multimodal trajectory distillation for end-to-end driving,” 2026. [Online]. Available: https://arxiv.org/abs/2601.22032

[31] L. Wang, Y. Zheng, Q. Chen, S. Li, Y. Zhang, Z. Xing, Q. Zhang, X. Li, D. Qian, P. Yang et al., “Latent-WAM: Latent world action modeling for end-to-end autonomous driving,” 2026. [Online]. Available: https://arxiv.org/abs/2603.24581

[32] Y. Zhou, X. Wang, H. Shao, L. Wang, G. Zhao, J. Shao, J. Zhu, T. Yu, Z. Zhu, G. Huang et al., “DriveDreamer-Policy: A geometry-grounded world-action model for unified generation and planning,” 2026. [Online]. Available: https://arxiv.org/abs/2604.01765

[33] M. Liu, D. Zhang, J. Liu, J. Cui, H. Xie, G. Chen, H. Ye, M. Y. Yang, F. Nex, and H. Cheng, “DriveVA: Video action models are zero-shot drivers,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2026, pp. 315–335.

[34] Z. Zhang, A. Liniger, D. Dai, F. Yu, and L. Van Gool, “End-to-end urban driving by imitating a reinforcement learning coach,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 15 222–15 232.

[35] D. Liu, Y. Gao, D. Qian, Q. Zhang, X. Ye, J. Han, Y. Zheng, X. Liu, Z. Xia, D. Ding et al., “Takead: Preference-based post-optimization for end-to-end autonomous driving with expert takeover data,” IEEE Robot. Autom. Lett., vol. 11, no. 2, pp. 1738–1745, 2025.

[36] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al., “Wan: Open and advanced large-scale video generative models,” 2025. [Online]. Available: https://arxiv.org/abs/2503.20314

[37] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[38] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[39] A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio, “Improving and generalizing flowbased generative models with minibatch optimal transport,” Trans. Mach. Learn. Res., 2024.

[40] Z. Li, W. Yao, Z. Wang, X. Sun, J. Chen, N. Chang, M. Shen, Z. Wu, S. Lan, and J. M. Alvarez, “Generalized trajectory scoring for end-to-end multimodal planning,” 2025. [Online]. Available: https://arxiv.org/abs/2506.06664

[41] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang et al., “Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 12 037–12 047.

[42] L. Liu, C. Jia, G. Yu, Z. Song, J. Li, F. Jia, P. Wu, X. Hao, and Y. Luo, “Guideflow: Constraint-guided flow matching for planning in end-to-end autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026, pp. 3719–3728.

[43] Y. Li, S. Shang, W. Liu, B. Zhan, H. Wang, Y. Wang, Y. Chen, X. Wang, Y. An, C. Tang et al., “DriveVLA-W0: World models amplify data scaling law in autonomous driving,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026.

[44] Z. Zhou, R. Yang, Y. Guo, S. X. Chen, T. Feng, K. Pistunova, Y. Shen, L. Su, J. Ma et al., “SpanVLA: Efficient action bridging and learning from negative-recovery samples for vision-language-action model,” 2026. [Online]. Available: https://arxiv.org/abs/2604.19710

[45] Y. Hong, X. Zhou, Y. Li, X. Zhou, L. Liu, Y. Luo, S. Xu, L. Yang, and Z. Song, “DriveFuture: Future-aware latent world models for autonomous driving,” 2026. [Online]. Available: https: //arxiv.org/abs/2605.09701

[46] OpenScene Contributors, “Openscene: The largest up-to-date 3d occupancy prediction benchmark in autonomous driving,” https://github.com/ OpenDriveLab/OpenScene, 2023.

[47] N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo et al., “Towards learning-based planning: The nuplan benchmark for real-world autonomous driving,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024.

[48] K. Li, Z. Li, S. Lan, Y. Xie, Z. Zhang, J. Liu, Z. Wu, Z. Yu, and J. M. Alvarez, “Hydra-MDP++: Advancing end-to-end driving via expert-guided hydra-distillation,” 2025. [Online]. Available: https://arxiv.org/abs/2503.12820

[49] J. Zou, S. Chen, B. Liao, Z. Zheng, Y. Song, L. Zhang, Q. Zhang, W. Liu, and X. Wang, “DiffusionDriveV2: Reinforcement learningconstrained truncated diffusion modeling in end-to-end autonomous driving,” 2025. [Online]. Available: https://arxiv.org/abs/2512.07745

[50] Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang et al., “ReCogDrive: A reinforced cognitive framework for end-to-end autonomous driving,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026.

[51] L. Liu, Z. Song, C. Jia, H. Ye, X. Hao, L. Chen et al., “DriveWorld-VLA: Unified latent-space world modeling with visionlanguage-action for autonomous driving,” 2026. [Online]. Available: https://arxiv.org/abs/2602.06521

[52] Z. Yao, H. Liu, Y. Jiang, Z. Zhu, Z. Guo, J. Wang, T. Liu, J. Cui, K. Yang, H. Xie et al., “Discrete-wam: Unified discrete vision-action token editing for world-policy learning,” arXiv preprint arXiv:2606.05645, 2026.

[53] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang, “Vad: Vectorized scene representation for efficient autonomous driving,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 8340–8350.

[54] N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo et al., “Towards learning-based planning: The nuplan benchmark for real-world

autonomous driving,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024.

## APPENDIX A

This appendix provides temporal indexing and optimization details, additional NAVSIM component scores, and qualitative cases from both trajectory prediction and closed-loop simulation. It also documents the scope of the negative supervision and the resources used in our implementation.

## A. Implementation Details

1) Data Layout and Optimization: Table VII summarizes the model configuration, optimization settings, and resource requirements of the two training stages. The four future visual targets use zero-based indices {1, 3, 5, 7} within the futureframe sequence, corresponding to cache indices {2, 4, 6, 8} when the current image occupies index zero. Each action channel $( \Delta x , \Delta y , \Delta \theta )$ is normalized using its 1st and 99th percentiles.

2) Scheduler and Conditioning: For each predicted temporal chunk, we uniformly sample an index $j \in \{ 0 , \ldots , S - 1 \}$ with $S = 1 0 0 0$ and use the shifted noise scale

$$
u _ { j } = 1 - \frac { j } { S } , \qquad t _ { j } ^ { m } = \frac { \gamma _ { m } u _ { j } } { 1 + ( \gamma _ { m } - 1 ) u _ { j } } ,\tag{14}
$$

where $\gamma _ { v } ~ = ~ 5 . 0$ and $\gamma _ { a } ~ = ~ 1 . 0$ . Uniform index sampling therefore does not imply a uniform distribution over the shifted video noise scale. Let $g ( t ) = \exp [ - 2 ( t - 0 . 5 ) ^ { 2 } ]$ and $\begin{array} { r } { g _ { m , \mathrm { m i n } } = \operatorname* { m i n } _ { 0 \leq l < S } g ( t _ { l } ^ { m } ) } \end{array}$ . The flow-matching weight is

$$
w _ { m } ( t _ { j } ^ { m } ) = \frac { g ( t _ { j } ^ { m } ) - g _ { m , \mathrm { m i n } } } { S ^ { - 1 } \sum _ { l = 0 } ^ { S - 1 } [ g ( t _ { l } ^ { m } ) - g _ { m , \mathrm { m i n } } ] } ,\tag{15}
$$

which has mean one over the scheduler grid. The video conditioning stream receives noise augmentation with probability 0.5, while its observed prefix remains clean. The action conditioning stream is not augmented.

3) Negative-Cache Indexing: The offline negative pools preserve the original 40 ego-frame poses of each GTRS candidate [40], at 0.1 s intervals. Let $\bar { \mathbf p } _ { q }$ denote the pose at zero-based cache index $q .$ The eight-step training trajectory uses

$$
\mathbf { p } _ { j } = \bar { \mathbf { p } } _ { 5 j + 4 } , \qquad j = 0 , \ldots , 7 ,\tag{16}
$$

corresponding to indices $4 , 9 , \ldots , 3 9$ . This aligns negative trajectories with the expert’s 0.5 s prediction grid while retaining the denser representation for offline evaluation. Fixed-size sampling and padding are applied by the training loader rather than stored in the offline pools; the filtering and validity-mask rules are specified in the main algorithm.

## B. Additional Quantitative Results

Table VIII provides the full component-wise navtest comparison, complementing the condensed comparison in the main paper.

TABLE VII  
MODEL AND TRAINING CONFIGURATIONS.
<table><tr><td colspan="10">Model Configuration</td></tr><tr><td>Sensors</td><td>Observed Input</td><td colspan="2">Future Indices</td><td>Action Target</td><td>SNR Shift (V/A)</td><td>Denoising Steps (V/A)</td><td colspan="2">Model Size</td><td>Peak GPU Memory</td><td colspan="2"></td></tr><tr><td>1 × Cam. (SV)</td><td>1 × 256 × 448 4 historical ego poses</td><td>{1, 3, 5, 7}</td><td colspan="2"></td><td>8-step ∆-pose</td><td> $5 . 0 / 1 . 0$ </td><td>3/10</td><td>6,241.81 MB</td><td colspan="2">14,029.08 MB</td></tr><tr><td colspan="14">Training Configuration</td></tr><tr><td>Stage</td><td>Initialization</td><td>Steps</td><td></td><td>Initial LR</td><td>LR Schedule</td><td></td><td>Loss Weights</td><td>GPUs</td><td></td><td>Wall Time</td><td>GPU Hours</td></tr><tr><td>WAM Pretraining</td><td>Wan2.2-TI2V-5B</td><td></td><td>100K</td><td> $1 \times 1 0 ^ { - 5 }$ </td><td>×0.5 at 30K, 70K, 90K</td><td></td><td></td><td></td><td>4× H100</td><td>22 h</td><td>88</td></tr><tr><td>Safety Post-training</td><td>Pretrained WAM</td><td></td><td>10K</td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>×0.5 at 5K</td><td></td><td>λrep = 100, λcompat = 0.1</td><td></td><td>4× H100</td><td>2 h</td><td>8</td></tr></table>

Both stages are trained on navtrain using AdamW with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ and a weight decay of 0.1. The per-GPU and global batch sizes are 1 and 4, respectively. Hard-negative defaults are $\alpha = 0 . 9$ and $\tau _ { s } = 0 . 1 ;$ the compatibility head has a hidden width of 256.

TABLE VIII  
PERFORMANCE COMPARISON ON THE NAVSIM V2 navtest BENCHMARK. ALL RESULTS ARE REPORTED USING THE CORRECTED OFFICIAL NAVSIM V2.2 EVALUATION PROTOCOL. MV AND SV DENOTE MULTI-VIEW AND SINGLE-VIEW CAMERA INPUTS, RESPECTIVELY. ALL EVALUATED METHODS USE CAMERA-ONLY INPUTS WITHOUT LIDAR.
<table><tr><td>Methods</td><td>Sensor</td><td>NC↑</td><td>DAC ↑</td><td>DDC ↑</td><td>TLC ↑</td><td>EP↑</td><td>TTC ↑</td><td>LK↑</td><td>HC ↑</td><td>EC ↑</td><td>EPDMS ↑</td></tr><tr><td>Human agent</td><td></td><td>100.0</td><td>100.0</td><td>99.8</td><td>100.0</td><td>97.8</td><td>100.0</td><td>100.0</td><td>98.1</td><td>90.1</td><td>97.9</td></tr><tr><td>E2E-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DiffusionDrive [41]</td><td>MV</td><td>98.2</td><td>95.9</td><td>99.4</td><td>99.8</td><td>87.5</td><td>97.3</td><td>96.8</td><td>98.3</td><td>87.7</td><td>84.5</td></tr><tr><td>DiffusionDriveV2 [49]</td><td>MV</td><td>97.7</td><td>96.6</td><td>99.2</td><td>99.8</td><td>88.9</td><td>97.2</td><td>96.0</td><td>97.8</td><td>91.0</td><td>87.5</td></tr><tr><td>VLA-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Recogdrive [50]</td><td>MV</td><td>98.3</td><td>95.2</td><td>98.3</td><td>99.8</td><td>87.1</td><td>97.5</td><td>96.6</td><td>99.5</td><td>86.5</td><td>83.6</td></tr><tr><td>DriveVLA-W0 [43]</td><td>SV</td><td>98.5</td><td>99.1</td><td>98.0</td><td>99.7</td><td>86.4</td><td>98.1</td><td>93.2</td><td>97.9</td><td>58.9</td><td>86.1</td></tr><tr><td>DriveWorld-VLA [51]</td><td>MV</td><td>98.6</td><td>99.1</td><td>99.6</td><td>99.8</td><td>87.4</td><td>97.9</td><td>97.0</td><td>97.8</td><td>78.6</td><td>86.8</td></tr><tr><td>World-Model-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Latent-WAM [31]</td><td>MV</td><td>98.1</td><td>97.3</td><td>99.6</td><td>99.8</td><td>87.7</td><td>97.3</td><td>97.6</td><td>98.1</td><td>87.3</td><td>89.3</td></tr><tr><td>DriveFuture [45]</td><td>MV</td><td>98.8</td><td>99.1</td><td>99.6</td><td>99.9</td><td>86.6</td><td>98.4</td><td>96.4</td><td>98.3</td><td>74.8</td><td>89.9</td></tr><tr><td>Drivelaw [5]</td><td>SV</td><td>98.8</td><td>96.9</td><td>99.6</td><td>99.8</td><td>87.4</td><td>98.3</td><td>97.7</td><td>98.4</td><td>80.1</td><td>88.9</td></tr><tr><td>Discrete-WAM [52]</td><td>SV</td><td>98.5</td><td>98.2</td><td>99.7</td><td>99.8</td><td>90.5</td><td>97.9</td><td>97.2</td><td>98.3</td><td>78.1</td><td>90.4</td></tr><tr><td>DriveWAM [7]</td><td>SV</td><td>98.4</td><td>96.7</td><td>99.7</td><td>99.8</td><td>97.7</td><td>97.7</td><td>97.7</td><td>98.3</td><td>87.9</td><td>92.2</td></tr><tr><td>ByondDrive [17]</td><td>SV</td><td>98.2</td><td>97.7</td><td>99.4</td><td>99.8</td><td>97.9</td><td>97.8</td><td>97.7</td><td>98.3</td><td>87.6</td><td>92.9</td></tr><tr><td>Ours</td><td>SV</td><td>98.1</td><td>98.4</td><td>99.5</td><td>99.8</td><td>97.8</td><td>97.5</td><td>97.3</td><td>98.3</td><td>87.6</td><td>93.3</td></tr></table>

## C. Additional Qualitative Results

1) Junction Interaction in HUGSIM: Figure 9 shows an additional closed-loop interaction in HUGSIM [21]. A white vehicle appears near the junction in the first two snapshots and moves out of the displayed views as the ego vehicle advances along the road. The three rendered views at each time point show the surrounding traffic and road geometry, while the yellow trajectory overlay changes with the observations. This sequence complements the oncoming-vehicle example in the main paper by showing another traffic interaction during closed-loop execution.

2) Additional Trajectory Comparisons: Figure 10 compares DriveWAM with NegWAM in two scenarios beyond those shown in the main paper. In the intersection-straight case, NegWAM’s trajectory bypasses a stopped roadside vehicle, while DriveWAM predicts a path close to the obstacle. In the left-turn case, NegWAM’s trajectory remains within the drivable surface, whereas the baseline trajectory leaves it. The two examples illustrate differences in obstacle clearance and adherence to road geometry.

3) Failure Cases: Figure 11 documents two cases in which NegWAM performs worse than DriveWAM. In the first, the expert overtakes a stopped leading vehicle on the left, but

NegWAM continues straight and collides with it. DriveWAM slows behind the vehicle and avoids the collision. In the second, NegWAM turns with insufficient clearance from a vehicle in the target lane, while DriveWAM maintains a wider margin. These failures identify limitations in maneuver selection and interaction clearance that remain after post-training.

## D. Scope of the Negative Supervision

a) Hard-Negative Coverage.: The learned safety boundary depends on the near-expert failures covered by the fixed trajectory vocabulary. Rare maneuvers and interactions outside that candidate space receive no direct negative supervision. Vocabulary-based retrieval provides broad trajectory coverage but limited adaptation to individual scenes [40]; diversityaware failure generation is a complementary direction [17].

b) Evaluator and Geometric-Distance Dependence.: The zero-EPDMS criterion assigns a common failure label to different violation types and severities. Both hard-negative objectives therefore inherit the offline evaluator’s assumptions. ADE additionally measures geometric proximity rather than interaction risk: similar paths can have different outcomes depending on timing, relative velocity, and road geometry [19].

![](images/450987de76bcf1fe21fa31ec1b625df727ac929bac48f807ab3a70bf08d28dfb.jpg)  
Fig. 9. Additional closed-loop rollout near a junction in HUGSIM. The four rows are ordered chronologically from top to bottom; each row contains thre rendered camera views. Yellow dots show the overlaid trajectory in the center view. A nearby white vehicle is visible in the early snapshots, followed by the ego vehicle’s continued motion along the road.

The rider branch uses scene context, but its negative labels still come from this same construction process.

c) Broader Impact.: The additional failure cases show that benchmark gains can coexist with unsafe individual decisions. The evidence remains benchmark-based and provides no formal safety guarantee. Deployment requires independent closed-loop validation, evaluation across traffic conditions and road-user categories, and safeguards for hazardous or unfamiliar situations.

## E. License of Assets

research and in compliance with the licenses accompanying their original releases.

All training and evaluation data are obtained from publicly released datasets, including nuPlan [54], OpenScene [46], and NAVSIM v2 [20], and are used in accordance with their respective licenses and terms of use. Our implementation further builds upon publicly available model assets from LingBot-VA [27], Wan2.2-TI2V-5B [36], and DriveWAM [7]. All thirdparty assets are used solely for non-commercial academic

AgentEgoPredicted TrajectoryHuman-Expert Trajectory  
![](images/a54dc4faf24556c34ab6ad2598ecefafbc9a691c7e18f0daf2115a33bc2f2889.jpg)  
Fig. 10. Additional successful cases. Comparison between DriveWAM [7] (top) and NegWAM (bottom) in an intersection-straight scenario with a stopped roadside vehicle and an intersection left-turn scenario. NegWAM safely avoids the stopped vehicle and maintains its trajectory within the drivable area, while the baseline produces collision-prone or off-road trajectories

Agent EgoPredicted TrajectoryHuman-Expert Trajectory  
![](images/f3e9ca16b54094e7e10297704216355be5eead1ffeea5e2516677d1d374e111b.jpg)  
Fig. 11. Representative failure cases. Comparison between DriveWAM (top) and NegWAM (bottom). In the first case, NegWAM fails to follow the expert’ leftward overtaking maneuver and collides with the stopped leading vehicle, whereas DriveWAM avoids the collision through conservative deceleration. In the second case, NegWAM maintains insufficient clearance from a vehicle in the target lane while turning, resulting in a collision.