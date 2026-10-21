# SLOT-JEPA: OBJECT-STRUCTURED PREDICTION IN-TERFACES FOR LATENT WORLD MODELS

Peisong Wen<sup>1</sup> Qianqian Xu<sup>2,3,∗</sup> Minye Lei<sup>1,2</sup> Siran Dai<sup>4,5</sup> Yang Liu<sup>1</sup> Qingming Huang<sup>1,∗</sup>

<sup>1</sup>School of Computer Science and Technology, University of Chinese Academy of Sciences <sup>2</sup>State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences <sup>3</sup>Beijing Academy of Artificial Intelligence

<sup>4</sup>Institute of Information Engineering, Chinese Academy of Sciences <sup>5</sup>School of Cyber Security, University of Chinese Academy of Sciences

{wenpeisong,qmhuang}@ucas.ac.cn daisiran@iie.ac.cn

{xuqianqian,leiminye25z}@ict.ac.cn liuyang232@mails.ucas.ac.cn

Corresponding authors.

## ABSTRACT

As a prototypical latent world model, Joint-Embedding Predictive Architectures (JEPAs) provide strong semantic representations. However, dense patch representations expose a redundant interface to downstream predictors, which must implicitly infer spatial organization and temporal correspondence, presenting three challenges. First, object information is distributed across patches, forcing each downstream predictor to redundantly recover object structure that could otherwise be reused across tasks before modeling interactions. Second, the lack of explicit entity correspondence across time makes historical transitions expensive, restricting the predictor’s flexibility in utilizing context. Third, limited by the lack of explicit transport structure, existing JEPA models typically use pointwise $\ell _ { 1 } / \ell _ { 2 }$ distances, which are insensitive to spatial displacement when predicted and goal supports are largely disjoint. We propose the Slot-based Joint-Embedding Predictive Architecture (Slot-JEPA), which decomposes dense representations into stable semantic slots and time-varying spatial masks. Pretrained once and frozen downstream, this decomposition exposes a reusable object-structured interface. Building on it, our Transition Memory Predictor (TMP) stores visual transitions as compact masks and retrieves useful context using statistics of corresponding slots. Its two branches separate slot-specific dynamics from shared cross-slot dynamics. We further use Unbalanced Optimal Transport (UOT) to supervise mask prediction and evaluate candidate plans by transporting mass between aligned predicted and goal masks, providing a signal better aligned with spatial progress. Across seven visualunderstanding benchmarks, the slot-reconstructed features remain transferable. Slot-JEPA also improves goal-conditioned planning across six planar environments and four robotic arm planning tasks on the REALM-JEPA benchmark. Code is available at https://github.com/KID-7391/Slot-JEPA.git.

## 1 INTRODUCTION

Building agents that can anticipate the consequences of their actions is a long-standing goal of artificial intelligence. World models pursue this goal by predicting how an environment evolves, allowing agents to evaluate possible behaviors before acting (Ha & Schmidhuber, 2018). Early visual world models operate in pixel space (Finn et al., 2016; Finn & Levine, 2017), but reconstructing every pixel requires modeling details that may be unpredictable. Latent world models instead predict action-conditioned changes in learned representations (Hafner et al., 2019; 2020) without rendering pixels.

Joint-Embedding Predictive Architectures (JEPAs) provide a promising foundation for this approach. Typically, JEPAs learn to predict masked image/video representations from visible context (Assran et al., 2023; Bardes et al., 2024). DINO-WM predicts representations changing from offline interactions (Zhou et al., 2025), while V-JEPA 2-AC post-trains an action-conditioned predictor for planning (Assran et al., 2025). Recent work further improves representation density and temporal consistency (Mur-Labadia et al., 2026), or adapts the model online (Wang et al., 2026). Parallel studies diagnose structural degradation and over-dispersed patch semantics in dense self-supervised representations, and develop structure-aware selection or semantic concentration to improve dense transfer (Dai et al., 2025; Wen et al., 2026). For video representations, T-CoRe reduces temporal sampling uncertainty using two-sided temporal correspondence, while Co-Settle studies the trade-off between intra-video consistency and inter-video separability during image-to-video transfer (Liu et al., 2025; 2026).

Despite this progress, we argue that dense patch representations might not be ideal for downstream dynamics learning and planning for three reasons: 1) An object’s information is distributed across patches, leaving its structure implicit. Each task-specific predictor must recover this structure, model interactions, and decode the result back into patches. The object organization encoded during pretraining is not exposed as a reusable interface. 2) A predictor needs relevant and diverse context to balance local and global views of dynamics. Yet patch-based history is highly redundant, making transitions expensive to store and their relevance costly to compute. 3) Patch representations form dense spatial grids without explicit entity correspondence or transport structure. Existing methods therefore compare predicted and goal grids with pointwise $\ell _ { 1 }$ or $\ell _ { 2 }$ distance (Zhou et al., 2025; Assran et al., 2025). When their supports are largely disjoint, these losses can become insensitive to spatial displacement and rank progress poorly.

We therefore propose Slot-based Joint-Embedding Predictive Architecture (Slot-JEPA), which turns dense patches into a reusable object-structured interface. A task-agnostic decomposition module addresses problem 1) by factorizing the dense feature field into stable semantic slots (Locatello et al., 2020; Seitzer et al., 2023; Nam et al., 2026) and time-varying spatial masks. Pretrained once and frozen for downstream tasks, it preserves the original feature field through reconstruction while exposing its object organization.

For problem 2), the temporal stability of slots allows visual history to be stored and reused as compact masks rather than high-dimensional patch representations. Inspired by memory-based Video Object Segmentation (Oh et al., 2019; Cheng et al., 2024), our proposed Transition Memory Predictor (TMP) uses statistics of masks associated with the same slots to assess relevance to the current query and retrieve relevant and diverse, potentially non-contiguous transitions. We further design two branches that separate slot-specific dynamics from inter-slot interaction dynamics, reducing the burden on dynamics learning.

Finally, problem 3) motivates our Unbalanced Optimal Transport (UOT) loss for planning. Using the temporal correspondence of slots, it replaces pointwise comparison between predicted and goal patch representations with transport between their aligned masks. Moving mask mass across space provides a smooth planning signal that remains informative over long displacements.

Slot decomposition turns a perception-oriented patch field into a reusable interface for prediction, memory maintenance, and goal evaluation. Across seven visual-understanding benchmarks, slotreconstructed features retain most of the semantic information. Across six planar tasks and four Robotic Arm Planning tasks from the REALM-JEPA dataset, Slot-JEPA improves goal-conditioned planning. The contributions are:

• We introduce Slot-JEPA, which compresses patch representations into shared semantic slots and time-varying masks, separating reusable visual organization from task-specific dynamics and easing predictor learning.

• We develop TMP to retrieve relevant, non-contiguous transitions from memory instead of relying only on a continuous recent-history window.

• We predict action effects in mask space and use UOT for goal evaluation, providing a signal better aligned with spatial progress when $\ell _ { 1 } / \ell _ { 2 }$ losses become insensitive to displacement.

## 2 RELATED WORK

Slot-JEPA connects four lines of work that are often studied separately: joint-embedding representation prediction, object-centric decomposition and dynamics, memory-based video segmentation, and optimal transport. Together, they motivate its slot-and-mask state, transition memory, and spatial planning objective.

## 2.1 JOINT-EMBEDDING PREDICTIVE WORLD MODELS

Joint-embedding methods predict target representations rather than reconstructing raw observations, allowing the learned state to retain predictable semantic structure while discarding some pixellevel detail. I-JEPA introduces this principle for masked image regions, and V-JEPA extends it to spatiotemporal video representations (Assran et al., 2023; Bardes et al., 2024). Dense self-supervised features from DINO and DINOv2 provide another strong latent target for spatial prediction (Caron et al., 2021; Oquab et al., 2024). Recent work identifies structural degradation over the course of self-supervised training and over-dispersion of semantically related patches, then addresses these issues through structure-aware model selection or explicit semantic concentration (Dai et al., 2025; Wen et al., 2026). In video self-supervision, T-CoRe uses frames on both sides of the prediction target to reduce temporal sampling uncertainty, while Co-Settle studies image-to-video transfer as a balance between within-video temporal consistency and cross-video semantic separability (Liu et al., 2025; 2026). These methods primarily concern how a useful visual representation is learned.

A second line of work turns representation prediction into an action-conditioned world model. DINO-WM predicts future DINOv2 patch features from offline interaction trajectories and plans by comparing imagined features with goal features (Zhou et al., 2025). V-JEPA 2-AC post-trains a frame-causal, action-conditioned predictor for robot planning, and V-JEPA 2.1 improves the spatial density and temporal consistency of the underlying video features (Assran et al., 2025; Mur-Labadia et al., 2026). Subsequent studies isolate the effects of representation, predictive objective, and planner design (Terver et al., 2026); AdaJEPA adapts the encoder and predictor online from newly observed transitions (Wang et al., 2026); and LeWorldModel investigates stable end-to-end training of JEPA world models from pixels (Maes et al., 2026).

Slot-JEPA is complementary to these advances: rather than replacing joint-embedding pretraining, it introduces a structured slot-and-mask interface between a pretrained dense encoder and action dynamics, together with a spatial objective for prediction and planning.

## 2.2 OBJECT-CENTRIC REPRESENTATION AND DYNAMICS

Slot Attention maps a set of visual features to a small collection of exchangeable components through competition for input tokens (Locatello et al., 2020). Subsequent methods extend this idea from individual images to video. SIMONe separates temporally abstracted object latents from per-pixel mixture assignments (Kabra et al., 2021); SAVi and SAVi++ use temporally propagated slots to support object tracking in synthetic and real videos (Kipf et al., 2022; Elsayed et al., 2022); and SlotFormer learns dynamics over slot sequences for video prediction and planning (Wu et al., 2023). DINOSAUR shows that reconstructing pretrained self-supervised features, rather than RGB pixels, can induce object-centric components in real images (Seitzer et al., 2023). This last observation is especially relevant to our goal of reorganizing a pretrained feature field without discarding its semantic information.

Object-centric prediction commonly updates the full latent content of every slot at each time step. Causal-JEPA, the closest joint-embedding counterpart, applies masked and future prediction to object-centric latents (Nam et al., 2026); Better Slots, Better Worlds studies how the quality of these latents affects prediction, planning, and robustness (Nazirjonov et al., 2026). In contrast, Slot-JEPA holds a reference-derived semantic slot bank fixed within a prediction context and predicts its spatial masks. As in unsupervised slot learning generally, these slots are object-centric components rather than guaranteed physical-instance segmentations.

## 2.3 MEMORY-BASED VOS AND TRANSITION MEMORY

Semi-supervised Video Object Segmentation (VOS) must preserve the identity of a target while its appearance and location change. Early propagation methods predict the next mask from an initial annotation and recent frames or masks (Perazzi et al., 2017; Oh et al., 2018). Space-Time Memory Networks instead store past image–mask pairs and let the current query read from them (Oh et al., 2019). XMem separates working, long-term, and sensory memories for long videos, while adaptive memory methods retain frames according to their utility rather than keeping every observation (Cheng & Schwing, 2022; Pourganjalikhan & Poullis, 2022). Cutie introduces object-level memory reading, and SAM 2 develops a streaming memory architecture for promptable segmentation (Cheng et al.,

2024; Ravi et al., 2025). Collectively, these methods show that a query can benefit from selected non-contiguous history instead of only the immediately preceding frame.

TMP transfers this memory principle from perceptual propagation to controlled prediction. Unlike VOS memories that store evidence for locating a target, TMP stores observed source–action–target transitions and retrieves potentially non-contiguous examples to predict a new action outcome.

## 2.4 UNBALANCED OPTIMAL TRANSPORT FOR VISUAL CORRESPONDENCE AND CONTROL

Pointwise distances compare features or masks only at matching grid locations. They can therefore provide little information about howfar mass should move when predicted and target supports do not overlap. Optimal transport instead compares nonnegative spatial measures under a ground cost, explicitly charging for displacement between source and target locations. Entropic regularization makes the problem amenable to efficient Sinkhorn scaling (Cuturi, 2013). Unbalanced optimal transport (UOT) further relaxes exact marginal matching by assigning a finite penalty to the creation or removal of mass (Chizat et al., 2018b;a). This relaxation is appropriate for visual masks, whose total mass can change with occlusion, viewpoint, scale, or imperfect decomposition.

Slot-JEPA applies UOT between masks with the same slot index, using their shared correspondence to preserve semantic grouping. UOT complements pointwise and reconstruction losses during training and supplies a displacement-aware goal cost during planning; its ground cost can additionally incorporate obstacle geometry in planar environments.

## 3 METHODOLOGY

## 3.1 OVERVIEW

At each time step, the multimodal observation $o _ { t } = \left( v _ { t } , r _ { t } \right)$ contains a visual observation $v _ { t }$ and, when available, a low-dimensional state observation $r _ { t } ,$ such as the robot end-effector pose and gripper state; $a _ { t }$ denotes the action. Given previously observed sequences, a world model predicts how the world will evolve under candidate future action sequences. At test time, given a current observation $o _ { 0 }$ and a visual goal $v _ { g } ,$ , model-predictive control (MPC) (Chua et al., 2018) selects the action sequence whose predicted visual observations are closest to $v _ { g } ,$ as measured by a goal distance $d ( \hat { v } , v _ { g } )$

An overview of Slot-JEPA is shown in Fig. 1. Slot-JEPA inserts a reusable object-structured interface between a general visual encoder and task-specific predictors. We therefore initialize the visual encoder with V-JEPA 2.1 (Mur-Labadia et al., 2026), and jointly pretrain the slot decomposition module and Transition Memory Predictor (TMP). The module converts patch representations into stable slots and spatial masks, which serve as the basic TMP inputs and outputs. After pretraining, we freeze the visual encoder and slot decomposition module, and fine-tune the transferred TMP on each downstream control task. A UOT-based loss is applied during training and planning.

## 3.2 PRETRAINED SLOT DECOMPOSITION MODULE

Let $\mathcal { V } ^ { S }$ denote the reference frames for slot extraction, and let $Z ^ { S } = E ( \mathcal { V } ^ { S } )$ be their dense patch representations. The slot decomposition module extracts a shared set of K slots ${ \pmb S } = \{ \pmb s _ { k } \in \mathbb { R } ^ { \hat { D } } \} _ { k = 1 } ^ { K }$ from $Z ^ { S }$ . Holding S fixed, it maps any encoded sequence $\pmb { Z } = \pmb { E } ( \pmb { v } _ { 1 : T } ) \in \mathbb { R } ^ { T \times N \times D }$ from the same environment to a time-varying spatial assignment $M = [ m _ { t , k , n } ] \stackrel { \sim } { \in } [ 0 , 1 ] ^ { T \times K \times N }$ . The slots summarize visual content shared by the reference frames, while $m _ { t , k , : } \in [ 0 , 1 ] ^ { N }$ indicates where the content of slot k appears. Ideally, they reconstruct each patch representation through

$$
Z _ { t , n } \approx \bar { Z } _ { t , n } = \sum _ { k = 1 } ^ { K } m _ { t , k , n } \pmb { s } _ { k } .\tag{1}
$$

Whereas the dense representation Z contains TND scalars, the factorized representation $( S , M )$ contains $K D + T K { \dot { N } }$ scalars. The latter is therefore more compact since $K \ll D$ in practice. To obtain S, we use Slot Attention (Locatello et al., 2020), which iteratively aggregates the reference

![](images/7342271278720fb8f191eb2acfe70f0b095ea05b275c63e3dd3a264e6b3f8849.jpg)  
Figure 1: Overview of Slot-JEPA. The slot decomposition module factorizes patch representations into stable slots $\pmb { S }$ and time-varying masks M. TMP predicts $\hat { M } _ { t + 1 }$ with sampled transition memory. The prediction is supervised against the next-frame mask by $\mathcal { L } _ { \mathrm { p r e d } }$ , including UOT loss.

patches into competing slots. Let $\boldsymbol { s } _ { j , r } ^ { \mathrm { l o c } }$ denote local slot j at iteration r, the local update is

$$
\begin{array} { r l } & { \alpha _ { n , j , r } = \mathrm { s o f t m a x } _ { j } \Big ( k ( z _ { n } ) ^ { \top } q ( s _ { j , r } ^ { \mathrm { l o c } } ) / \big ( \tau \sqrt { D } \big ) \Big ) , \quad u _ { j , r } = \displaystyle \sum _ { n } \frac { \alpha _ { n , j , r } } { \sum _ { n ^ { \prime } } \alpha _ { n ^ { \prime } , j , r } } v ( z _ { n } ) , } \\ & { \tilde { s } _ { j , r + 1 } ^ { \mathrm { l o c } } = \mathrm { G R U } ( u _ { j , r } , s _ { j , r } ^ { \mathrm { l o c } } ) , \quad s _ { j , r + 1 } ^ { \mathrm { l o c } } = \tilde { s } _ { j , r + 1 } ^ { \mathrm { l o c } } + \mathrm { M L P } \big ( \mathrm { L N } ( \tilde { s } _ { j , r + 1 } ^ { \mathrm { l o c } } ) \big ) . } \end{array}\tag{2}
$$

Here, $\tau > 0$ controls assignment softness, and GRU, LN, and MLP denote the gated recurrent unit, layer normalization, and multilayer perceptron, respectively. For initialization, we partition the reference frames into non-overlapping windows, extract local slots from each window, and use farthest-point sampling to select diverse initial slots. Importantly, Slot-JEPA does not assume that slots correspond to predefined semantic categories. Instead, the shared reference-derived slots provide a fixed latent basis within each planning context, and masks encode temporal variation relative to this basis.

On top of the extracted slots S, we compute M in the pretrained latent space (Kabra et al., 2021). Patch features cross-attend to the slots, and an MLP g produces slot-assignment logits:

$$
\begin{array} { r } { \pmb { H } = \mathrm { C A } ( \pmb { Z } ; \pmb { S } ) , \qquad \pmb { A } _ { t , k , n } = g ( [ \pmb { H } _ { t , n } | \pmb { | s } _ { k } ] ) , \qquad m _ { t , k , n } = \mathrm { s o f t m a x } _ { k } ( A _ { t , k , n } ) . } \end{array}\tag{3}
$$

Here, CA denotes the cross-attention stack. Competition across slots encourages object-aligned grouping but does not require one slot per physical instance: a slot may cover multiple entities, a part, or background. Increasing K permits a finer decomposition but increases the prediction cost.

Having obtained S and M for frame t, we reconstruct the patch representation $\bar { Z } _ { t }$ with Eq. (1). Then we train the decomposition module by reconstructing the original dense representation $\scriptstyle { Z _ { t } }$ , while encouraging different slots to encode distinct content:

$$
\mathcal { L } _ { \mathrm { s l o t } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \lVert \bar { Z } _ { t } - Z _ { t } \rVert _ { 1 } + \frac { \lambda _ { \mathrm { o r t h } } } { K ( K - 1 ) } \sum _ { k \neq j } \Big ( \frac { s _ { k } ^ { \top } s _ { j } } { \lVert s _ { k } \rVert _ { 2 } \lVert s _ { j } \rVert _ { 2 } } \Big ) ^ { 2 } .\tag{4}
$$

## 3.3 TRANSITION MEMORY PREDICTOR

The temporally stable slots make compact, query-dependent reuse of the observed trajectory possible: visual states can be stored as masks and compared through statistics of corresponding slots. During training, the observed transitions in each sampled video clip provide the context directly. At inference, TMP samples relevant and diverse context transitions $\mathcal { C } _ { t }$ based on the query state and mask geometry. It then uses the same symmetric context–query structure to predict the next mask in both settings.

Transition memory bank. Each memory bank contains transitions from one environment that are decoded against the same slots S. Visual states can therefore be stored solely as masks $M _ { i } { \mathrm { : } }$

$$
\mathcal { B } _ { t } = \{ ( M _ { i } , r _ { i } , a _ { i } , M _ { i + 1 } , r _ { i + 1 } ) \} _ { i \in \mathcal { T } _ { t } ^ { \mathrm { m e m } } } ,\tag{5}
$$

where $r _ { i }$ is the optional low-dimensional state, $a _ { i }$ is the action, and $\mathcal { T } _ { t } ^ { \mathrm { m e m } }$ indexes the transitions available before the query. Let $\pmb { \rho } _ { i } \in \mathbb { R } ^ { K } , \pmb { \mu } _ { i } \in \mathbb { R } ^ { 2 K }$ collect the areas of the K masks and 2-D centroids in $M _ { i }$ respectively, and $\pmb { u } _ { i } = [ r _ { i } \| \sqrt { \lambda _ { v } } \pmb { \rho } _ { i } \| \sqrt { \lambda _ { v } } \pmb { \mu } _ { i } ]$ summarizes each state. At inference,

we sample transitions $\mathcal C _ { t } \subset \mathcal B _ { t }$ without replacement according to

$$
d _ { \mathrm { m e m } } ( t , i ) = \Vert u _ { t } - u _ { i } \Vert _ { 2 } ^ { 2 } , \qquad P _ { t } ( i ) = ( 1 - \gamma ) \mathrm { s o f t m a x } _ { i } ( - d _ { \mathrm { m e m } } ( t , i ) ) + \gamma / | \mathcal { B } _ { t } | .\tag{6}
$$

The mixing weight $\gamma \in \ [ 0 , 1 ]$ balances distance-weighted sampling for relevance with uniform sampling for diversity, which is essential for in-context examples (Huang et al., 2026).

Predictor with symmetric structure. TMP contains a slot-wise mask branch for spatial changes and a shared feature branch for interactions among slots. Both process observed transitions as context blocks and the transition to be predicted as a query block. For each context transition i, we encode its state and action as $\pmb { c } _ { i } = E _ { c } ( [ \hat { r } _ { i } \lVert \boldsymbol { a } _ { i } \rVert r _ { i + 1 } ] )$ . In the mask branch, slot k forms a spatial field $X _ { i , k , n } = m _ { i , k , n } \pmb { s } _ { k }$ . The current and next fields are paired to form a context block

$$
B _ { i , k } = \left[ P _ { c } ( c _ { i } ) , P _ { s } ( s _ { k } ) , P _ { m } ( X _ { i , k } \Vert X _ { i + 1 , k } ) \right] \in \mathbb { R } ^ { ( N ^ { \prime } + 2 ) \times d ^ { \prime } } .\tag{7}
$$

Here $P _ { c } , P _ { s }$ , and $P _ { m }$ project inputs to width $d ^ { \prime }$ . For the query, unknown future inputs are replaced by learnable tokens. Specifically, $\pmb q ^ { r }$ replaces $r _ { t + 1 }$ in the state–action encoding: $\boldsymbol { c } _ { t } = E _ { c } ( [ r _ { t } ] | \boldsymbol { a } _ { t } | | \boldsymbol { q } ^ { r } ] )$ and $\pmb q _ { k }$ replaces the future spatial field. Then we have

$$
B _ { t , k } = \big [ P _ { c } ( c _ { t } ) , P _ { s } ( s _ { k } ) , P _ { m } ( X _ { t , k } | | q _ { k } ) \big ] , \quad B _ { k } = \big [ B _ { i , k } \big ] _ { i \in \mathcal { I } _ { t } } , \quad B _ { k } ^ { + } = [ B _ { k } ; B _ { t , k } ] .\tag{8}
$$

where $\mathcal { T } _ { t }$ indexes the selected transitions. Then we apply multiple attention blocks:

$$
H _ { t , k } ^ { \operatorname * { m a x } { \bf k } } = \mathrm { A t t n } \big ( B _ { t , k } W _ { Q } , B _ { k } ^ { + } W _ { K } , B _ { k } ^ { + } W _ { V } \big ) , \quad \mathrm { A t t n } ( Q , K , V ) = \mathrm { s o f t m a x } ( Q K ^ { \top } / \sqrt { d } ) V .\tag{9}
$$

We select only the query positions $\mathcal { I }$ occupied by learnable tokens and concatenate $( H _ { t , k } ^ { \mathrm { m a s k } } ) _ { \mathcal { I } }$ into $\pmb { H } _ { t + 1 } ^ { \mathrm { m a s k } }$ . The feature branch follows the same process, but takes $[ \bar { Z } _ { i } \Vert M _ { i } ^ { \top } ]$ as its spatial input, yielding interaction features $\pmb { H } _ { t + 1 } ^ { \mathrm { f e a t } }$ . Finally, we broadcast $\pmb { H } _ { t + 1 } ^ { \mathrm { f e a t } }$ across slots and combine both branches with an MLP $f _ { \mathrm { o u t } }$ and slot-wise softmax to predict the future assignment map:

$$
\begin{array} { r } { \hat { M } _ { t + 1 } = \mathrm { s o f t m a x } _ { \mathrm { s l o t } } \left( f _ { \mathrm { o u t } } \left( [ { H } _ { t + 1 } ^ { \mathrm { m a s k } } \Vert { H } _ { t + 1 } ^ { \mathrm { f e a t } } ] \right) \right) . } \end{array}\tag{10}
$$

Action-free pretraining. To pretrain TMP on natural videos without action annotations, an inverse transition encoder $B _ { \psi }$ compresses each observed change into a low-dimensional pseudo-action:

$$
\tilde { a } _ { t } = B _ { \psi } ( S , M _ { t } , M _ { t + 1 } ) \in \mathbb { R } ^ { d _ { b } } , \qquad d _ { b } \ll K N .\tag{11}
$$

For the query transition, $M _ { t + 1 }$ reaches TMP only through the bottleneck $\tilde { a } _ { t }$ , limiting direct target leakage while preserving the transition information needed for prediction. After representation pretraining, we discard the inverse transition and state encoders, and replace them with task-specific encoders for $r _ { t }$ and $a _ { t }$ . The remaining TMP weights are fine-tuned on downstream interaction data.

## 3.4 TRANSPORT-BASED OBJECTIVE FOR MASK PREDICTION

We next specify how the mask predictions are trained and used for goal evaluation. During training, the fixed S decodes the held-out future representation $\boldsymbol { Z } _ { t + 1 }$ into the target mask $M _ { t + 1 }$ . Given action $a _ { t }$ and transition context $\mathcal { C } _ { t }$ , TMP produces $\hat { M } _ { t + 1 }$ . The target and predicted masks reconstruct $\begin{array} { r } { \bar { Z } _ { t + 1 , n } = \sum _ { k } m _ { t + 1 , k , n } \pmb { s } _ { k } } \end{array}$ and $\begin{array} { r } { \hat { Z } _ { t + 1 , n } = \sum _ { k } \hat { m } _ { t + 1 , k , n } { \mathbf { s } } _ { k } } \end{array}$ , respectively.

Conventional latent predictors directly predict $\hat { \boldsymbol Z } _ { t + 1 }$ and minimize its $\ell _ { 1 }$ distance to $\boldsymbol { Z } _ { t + 1 }$ (Bardes et al., 2024; Assran et al., 2025). With our factorization, the same objective naturally transfers to $\lVert \hat { M } _ { t + 1 } - M _ { t + 1 } \rVert _ { 1 }$ . However, in either representation or mask space, pointwise $\ell _ { 1 }$ can become insensitive to further spatial displacement when predicted and target supports are largely disjoint. We therefore use Unbalanced Optimal Transport $( U O T )$ to measure the cost of moving mask mass from a prediction to its target.

Consider masks ${ \boldsymbol { p } } , { \boldsymbol { q } } \in \mathbb { R } _ { + } ^ { N }$ , let $\xi _ { i } ~ \in ~ [ 0 , 1 ] ^ { 2 }$ be the normalized coordinate of patch $i ,$ and set $c _ { i j } = \| \pmb { \xi } _ { i } - \pmb { \xi } _ { j } \| _ { 2 } ^ { 2 }$ in the ground-cost matrix C. A transport plan $\pmb { \pi } \in \mathbb { R } _ { + } ^ { N \times N }$ moves $\pi _ { i j }$ units of mass from patch i to j. Because p and q may have different total mass, KL penalties on the plan’s marginals allow mass to be created or removed (Chizat et al., 2018b). We obtain an entropy-regularized plan and evaluate its transport and mass-change costs:

$$
\boldsymbol { \pi } _ { \varepsilon , \rho } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { \pi } > 0 } \langle \boldsymbol { \pi } , \boldsymbol { C } \rangle + \varepsilon \Omega ( \boldsymbol { \pi } ) + \rho \mathrm { K L } ( \boldsymbol { \pi } \mathbf { 1 } \| p ) + \rho \mathrm { K L } ( \boldsymbol { \pi } ^ { \top } \mathbf { 1 } \| q ) ,\tag{12}
$$

$$
\begin{array} { r } { \mathcal { T } _ { \varepsilon , \rho } ( p , q ) = \langle \pi _ { \varepsilon , \rho } ^ { * } , C \rangle + \rho \mathrm { K L } ( \pi _ { \varepsilon , \rho } ^ { * } \mathbf { 1 } \| p ) + \rho \mathrm { K L } ( ( \pi _ { \varepsilon , \rho } ^ { * } ) ^ { \top } \mathbf { 1 } \| q ) . } \end{array}
$$

Table 1: Visual-understanding results on all three probes.
<table><tr><td rowspan="2">Method</td><td colspan="3">Segmentation (mIoU↑)</td><td colspan="3">Depth (δ1 ↑)</td><td rowspan="2">VOS (J&amp;F ↑)</td></tr><tr><td>COCO</td><td>ADE</td><td>VOC</td><td>NYUv2</td><td>KITTI</td><td>SUNRGBD DAVIS</td></tr><tr><td>MoCo v3 (Chen et al., 2021)</td><td>43.4</td><td>24.4</td><td>61.7</td><td>69.8</td><td>77.2</td><td>50.6</td><td>61.7</td></tr><tr><td>iBOT (Zhou et al., 2022a)</td><td>52.2</td><td>34.3</td><td>73.3</td><td>77.7</td><td>88.3</td><td>69.1</td><td>64.3</td></tr><tr><td>Mugs (Zhou et al., 2022b)</td><td>51.4</td><td>34.6</td><td>72.7</td><td>74.9</td><td>87.7</td><td>66.6</td><td>62.1</td></tr><tr><td>MAE (He et al., 2022)</td><td>37.9</td><td>20.6</td><td>52.3</td><td>65.3</td><td>84.3</td><td>59.1</td><td>48.9</td></tr><tr><td>DINOv2† (Oquab et al., 2024)</td><td>53.0</td><td>35.5</td><td>74.9</td><td>78.1</td><td>88.0</td><td>68.0</td><td>64.2</td></tr><tr><td>V-JEPA 2.1 (Mur-Labadia et al., 2026)</td><td>52.8</td><td>35.5</td><td>72.1</td><td>80.5</td><td>89.2</td><td>70.0</td><td>65.9</td></tr><tr><td>Slot-JEPA</td><td>56.1</td><td>38.9</td><td>75.1</td><td>83.1</td><td>87.8</td><td>72.3</td><td>66.6</td></tr></table>

Here, Ω is the entropic regularizer, ε controls the smoothness of the computed plan, and $\rho$ controls the cost of changing mass. We solve for the plan with Sinkhorn iterations (Chizat et al., 2018a) (see Appendix A.3). Combining UOT, mask $\ell _ { 1 }$ , and the original representation reconstruction gives

$$
\mathcal { L } _ { \mathrm { p r e d } } = \lambda _ { \mathrm { u o t } } \sum _ { k } \mathcal { T } _ { \varepsilon , \rho } ( \hat { m } _ { t + 1 , k } , m _ { t + 1 , k } ) + \lambda _ { \mathrm { m a x k } } \| \hat { M } _ { t + 1 } - M _ { t + 1 } \| _ { 1 } + \lambda _ { \mathrm { f e a t } } \| \hat { Z } _ { t + 1 } - Z _ { t + 1 } \| _ { 1 } .\tag{13}
$$

At test time, the first two terms form the planning objective after replacing the target masks with M<sup>g</sup> extracted from the visual goal. Both the current observation and goal are decoded against the same S: The goal therefore does not receive an independent slot set, and slot index k remains aligned across current, predicted, and goal masks. We roll out candidate action sequences, optimize them with the cross-entropy method (CEM) (Chua et al., 2018), execute an action chunk, and replan. Appendix A.3 details generalized Sinkhorn scaling, change-aware weighting, and static-slot identification; Appendix B.3 gives the environment-specific obstacle-aware ground cost and MPC settings.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

We conduct experiments around four questions: 1) Does the slot decomposition preserve the semantic information (Sec. 4.2)? 2) How well does Slot-JEPA perform on visual planning in controlled planar environments (Sec. 4.3)? 3) Does the framework generalize to Robotic Arm Planning (Sec. 4.4)? 4) How does each component contribute to performance (Sec. 4.5)?

Representation pretraining. Starting from a V-JEPA 2.1 encoder (Mur-Labadia et al., 2026), we pretrain the slot decomposition module and TMP on Kinetics-400 videos (Kay et al., 2017) using inverse transition encoders. We freeze the pretrained visual encoder and slot decomposition module for downstream tasks. The full pretraining setup is given in Appendix B.1.

## 4.2 PRESERVATION OF SEMANTIC INFORMATION

For each image or video frame, the frozen visual encoder and the slot decomposition module jointly provide a patch representation for downstream visual understanding tasks. We evaluate linear semantic segmentation on COCOStuff-27 (Caesar et al., 2018), ADE20K (Zhou et al., 2017), and PASCAL VOC (Everingham et al., 2010) to measure category-level structure; monocular depth estimation on NYUv2 (Silberman et al., 2012) and KITTI (Geiger et al., 2012), plus NYUv2-to-SUNRGBD transfer (Song et al., 2015) without adaptation; and VOS on DAVIS-2017 (Pont-Tuset et al., 2017) using zero-shot label propagation. The evaluation settings are detailed in Appendix B.2.

Results in Tab. 1 support two conclusions: 1) Slot decomposition preserves the main task-relevant semantics under strong compression. With $K = 1 6$ , the factorized slot representation contains only about 2.5% as many scalars as the dense patch field, yet the slot-reconstructed representation retains the semantic information needed across segmentation, depth, and VOS. 2) Slot reconstruction makes the patch representation more object-aware. During representation pretraining, the slotreconstruction objective fine-tunes the visual encoder and propagates object-level grouping back into its patch features. This supplies explicit object-aware supervision that is absent from the original V-JEPA 2.1 training objective, helping explain the consistent downstream gains.

Table 2: Planar-planning success rates (%). Goal State indicates whether the planner also receives the low-dimensional goal state.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Goal State</td><td colspan="3">ID Environments</td><td colspan="3">OOD Environments</td></tr><tr><td>TwoRoom</td><td>Maze</td><td>PushT</td><td>TwoRoomR</td><td>MazeR</td><td>PushObj</td></tr><tr><td rowspan="2">DINO-CLS (Caron et al., 2021)</td><td>x</td><td>6</td><td>88</td><td>20</td><td>34</td><td>70</td><td>2</td></tr><tr><td>√</td><td>84</td><td></td><td>28</td><td>48</td><td></td><td>8</td></tr><tr><td rowspan="2">ResNet (He et al., 2016)</td><td>x</td><td>6</td><td>98</td><td>34</td><td>12</td><td>68</td><td>4</td></tr><tr><td>√</td><td>84</td><td></td><td>40</td><td>60</td><td></td><td>16</td></tr><tr><td rowspan="2">R3M (Nair et al., 2023)</td><td>×</td><td>38</td><td>100</td><td>16</td><td>8</td><td>32</td><td>2</td></tr><tr><td>√</td><td>90</td><td></td><td>42</td><td>46</td><td></td><td>36</td></tr><tr><td rowspan="2">DINO-WM (Zhou et al., 2025)</td><td>x</td><td>50</td><td>100</td><td>74</td><td>82</td><td>86</td><td>28</td></tr><tr><td>√</td><td>84</td><td></td><td>88</td><td>82</td><td></td><td>40</td></tr><tr><td>V-JEPA 2.1 (Mur-Labadia et al., 2026)</td><td>x</td><td>50</td><td>94</td><td>58</td><td>50</td><td>86</td><td>46</td></tr><tr><td>LeWM (Maes et al., 2026)</td><td>√ x</td><td>82</td><td>98</td><td>66</td><td>72</td><td></td><td>54</td></tr><tr><td></td><td></td><td>80</td><td></td><td>88</td><td>52</td><td>88</td><td>30</td></tr><tr><td>Slot-JEPA (Ours)</td><td>x</td><td>100</td><td>100</td><td>92</td><td>90</td><td>96</td><td>58</td></tr></table>

![](images/8924fa77215b704bb55d2430fb841b6438665f9553aa9b2d745b521a787e547f.jpg)  
Figure 2: Goal-loss landscapes in MazeR and TwoRoomR. Stars mark goals; lighter colors indicate lower loss.

![](images/cf019358d310c8ec0ac8fc8a648e6a97f28592b17666a346290bb6b9b9daa5c1.jpg)  
Figure 3: Three-view start and stage-wise goal images for a REALM-JEPA Place task.

## 4.3 PLANAR VISUAL PLANNING

Evaluation protocol. We evaluate three in-distribution (ID) tasks and their out-of-distribution (OOD) variants. The ID tasks use the same layouts and object shapes during training and testing, whereas the OOD variants use previously unseen layouts or object shapes. In TwoRoom (also called Wall in prior work) and Maze (Zhou et al., 2025), a point agent passes through a doorway and a ball navigates a U-shaped corridor, respectively. Both use position and velocity as $r _ { t } ,$ while $a _ { t }$ specifies velocity in TwoRoom and force in Maze. In PushT (Florence et al., 2022; Chi et al., 2025), a pusher moves a T-shaped object and itself to their target poses; $r _ { t }$ is the pusher state and $a _ { t }$ is a relative target tracked by a PD controller. The OOD variants TwoRoomR and MazeR randomize obstacle layouts, while PushObj changes the object shape. Each task provides an initial image $v _ { 0 }$ and visual goal $v _ { g }$ For comparison, we report DINO-WM baselines that additionally use the low-dimensional goal state for planning in TwoRoom and PushT. We evaluate 50 cases per environment using MPC (Rawlings et al., 2026) with CEM (de Boer et al., 2005). See Appendix B.3 for more details.

Main results. Tab. 2 reports success rates and indicates whether goal-state input is used. We draw two conclusions: 1) Slot-JEPA improves visual planning in both ID and OOD environments. Using only a visual goal, it succeeds in 100%, 100%, and 92% of ID cases and 90%, 96%, and 58% of OOD cases. It also outperforms every baseline that receives the low-dimensional goal state in all four environments where goal-state results are available. 2) Slot masks and UOT cost provide a planning signal aligned with progress. As shown in Fig. 2, the obstacle-aware UOT loss changes gradually even far from the goal, which provides a smoother landscape compared to $\ell _ { 1 }$ losses. Stable slots and object-level mask decomposition provide the foundation for applying UOT for planning. Additional loss landscapes and predicted-mask rollouts are collected in Appendix B.5.

Table 3: Robotic arm planning success rates on REALM-JEPA (count and %).
<table><tr><td>Method</td><td>Reach</td><td>Pick</td><td>Rotate</td><td>Place</td><td>Overall</td></tr><tr><td>V-JEPA 2.1</td><td>13/18 (72.2)</td><td>19/30 (63.3)</td><td>6/30 (20.0)</td><td>0/30 (0.0)</td><td>38/108 (35.2)</td></tr><tr><td>V-JEPA 2.1 + TMP</td><td>16/18 (88.9)</td><td>22/30 (73.3)</td><td>13/30 (43.3)</td><td>12/30 (40.0)</td><td>63/108 (58.3)</td></tr><tr><td>Slot-JEPA</td><td>18/18 (100.0)</td><td>27/30 (90.0)</td><td>25/30 (83.3)</td><td>23/30 (76.7)</td><td>93/108 (86.1)</td></tr></table>

Table 4: Spearman correlations among observed/predicted loss and EE–goal distance.
<table><tr><td rowspan="3">Method</td><td colspan="3">Obs. loss vs. EE-goal dist.</td><td colspan="3">Pred. loss vs. EE-goal dist.</td><td colspan="3">Pred. loss vs. Obs. loss</td></tr><tr><td>Ext. 1</td><td>Ext. 2</td><td>Wrist</td><td>Ext. 1</td><td>Ext. 2</td><td>Wrist</td><td>Ext. 1</td><td>Ext. 2</td><td>Wrist</td></tr><tr><td>V-JEPA 2.1</td><td>0.350</td><td>0.407</td><td>0.533</td><td>0.503</td><td>0.453</td><td>0.297</td><td>0.511</td><td>0.609</td><td>0.549</td></tr><tr><td>V-JEPA 2.1 + TMP</td><td>0.154</td><td>0.103</td><td>0.652</td><td>0.327</td><td>0.319</td><td>0.454</td><td>0.409</td><td>0.421</td><td>0.720</td></tr><tr><td>Slot-JEPA</td><td>0.792</td><td>0.891</td><td>0.807</td><td>0.677</td><td>0.788</td><td>0.814</td><td>0.668</td><td>0.825</td><td>0.865</td></tr></table>

![](images/e84232cb6b334ca33ab48c1fffcd937c100f7df1595ec341333d144ca23a7c00.jpg)  
Figure 4: Goal-loss trajectories for Pick and Place. Panels use independent scales.

## 4.4 ROBOTIC ARM PLANNING

Dataset & evaluation protocol. Based on the REALM manipulation environment and task configurations (Sedlacek et al., 2026), we construct the REALM-JEPA dataset for JEPA-based visual-goal planning. It provides start observations and stage-wise visual goals for 108 cases in four task families: 18 Reach cases and 30 each for Pick, Rotate, and Place. We fine-tune the robot dynamics predictor on DROID trajectories (Khazatsky et al., 2024). The training settings are detailed in Appendix B.4.

In each task, we control a 7-DoF robot arm: six dimensions describe the end-effector (EE) pose, and one specifies the gripper opening angle. At step t, the visual observation $v _ { t }$ contains images from two fixed external cameras and one wrist camera. Following V-JEPA 2 (Assran et al., 2025), we provide up to two stage-wise visual goals, each comprising images from the same three views, as illustrated in Fig. 3. The low-dimensional state $r _ { t }$ contains the current EE pose, gripper angle, and camera poses. The action $a _ { t }$ specifies changes to the EE pose and gripper angle. The three views share one predictor, but maintain separate slots, transition memories, and predictions. At each MPC step, the two fixed views first update the CEM distribution; the wrist view then starts from that distribution, continues the search, and selects the action for execution. An inverse-kinematics controller converts the selected absolute EE pose into joint targets, which the arm tracks using PD feedback. Each planning attempt is limited to 100 MPC steps. Since $\ell _ { 1 }$ losses provide weak guidance at large displacements, we use ground-truth target-position guidance for the patch-based baselines until the EE is within 30 cm of the target position. Slot-JEPA uses visual goal only throughout. Full multi-view and guidance details are provided in Appendix B.4.

Quantitative results. Tab. 3 reports success rates for the four task families. To assess whether the visual goal loss reflects physical progress, Tab. 4 reports its Spearman correlation with 7-DoF EE–goal distance, using both losses predicted by the predictor and losses computed from the observations after execution. We also examine how the observed goal loss changes over planning steps in Fig. 4. Appendix B.4 gives the complete correlation definitions and statistics. These results support three conclusions: 1) Slot-JEPA improves success across all four task families, especially Rotate and Place. Its overall success rate is 86.1%, compared with 35.2% for V-JEPA 2.1 and 58.3% for V-JEPA 2.1 + TMP. On Rotate and Place, Slot-JEPA reaches 83.3% and 76.7%, respectively, compared with 20.0% and 0.0% for V-JEPA 2.1. 2) Pointwise $\ell _ { 1 }$ losses give weak guidance far from the goal. Despite this ground-truth guidance, both patch-based baselines still fail often. Their losses computed from actual observations also correlate less strongly with EE–goal distance than Slot-JEPA’s loss, indicating that predictor error alone cannot explain these failures. 3) Replacing V-JEPA 2.1’s predictor with TMP substantially improves planning. The success rate rises from 35.2% to 58.3% while retaining the patch representation and goal loss. This gain is consistent with TMP retrieving useful information from the full trajectory history, whereas the original predictor relies on temporally contiguous frames and is restricted to nearby observations.

![](images/8474288f63d7befdd9521ab820d8484492174af61818c7ecff7517c879f06c8e.jpg)  
Figure 5: Slot-mask overlays from Kinetics-400, DROID, and REALM-JEPA (top to bottom). Colors track slots across each sequence.

![](images/fb5f08f6d60473ef9433c6a3c5f90ac70f3a0ce3133548b91b3c2fc2d20c7891.jpg)  
Figure 6: Wrist-view RGB, observed features, and reconstructed features from predicted masks without seeing the frame (top to bottom).

Table 5: Ablation success rates (%) across six planar environments and robot Reach. Bold marks the highest value in each column.
<table><tr><td>ID</td><td>Predictor</td><td>Planning loss</td><td>Context</td><td>TwoRoom</td><td>Maze</td><td>PushT</td><td>TwoRoomR</td><td>MazeR</td><td>PushObj</td><td>Reach</td></tr><tr><td>1</td><td>Sequential</td><td>UOT + mask l1</td><td>Recent transitions</td><td>100</td><td>100</td><td>76</td><td>94</td><td>98</td><td>40</td><td></td></tr><tr><td>2</td><td>TMP</td><td>Mask l1</td><td>Sampled via Eq. (6)</td><td>96</td><td>92</td><td>82</td><td>78</td><td>90</td><td>50</td><td>88.9</td></tr><tr><td>3</td><td>TMP</td><td> $\mathrm { U O T } + \mathrm { m a s k } \ : \ell _ { 1 }$ </td><td>Recent transitions</td><td>94</td><td>86</td><td>76</td><td>84</td><td>80</td><td>58</td><td>55.6</td></tr><tr><td>4</td><td>TMP</td><td> $\mathrm { U O T } + \mathrm { m a s k } \ : \ell _ { 1 }$ </td><td>Random transitions</td><td>96</td><td>88</td><td>80</td><td>40</td><td>72</td><td>48</td><td>94.4</td></tr><tr><td>5</td><td>TMP</td><td> $\mathrm { U O T } + \mathrm { m a s k } \ : \ell _ { 1 }$ </td><td>Sampled via Eq. (6)</td><td>100</td><td>100</td><td>92</td><td>90</td><td>96</td><td>58</td><td>100.0</td></tr></table>

Visualizations. Fig. 5 shows examples from Kinetics-400 pretraining, DROID downstream training, and REALM-JEPA planning inference, respectively. The selected slots remain temporally consistent despite object motion, camera viewpoint changes, and objects leaving the view. Although the slot decomposition is frozen after pretraining, it continues to produce coherent masks in the unseen downstream scenes. Predicted future representations also closely match those extracted from subsequent observations (Fig. 6), supporting TMP’s ability to model action-conditioned dynamics. Full-scale slot decompositions, additional UOT landscapes, and predicted-mask rollouts are provided in Appendix B.5.

## 4.5 ABLATION STUDIES

Tab. 5 compares five configurations. Line 1: replace TMP with a V-JEPA 2.1-style sequential predictor (Mur-Labadia et al., 2026) using recent transitions while retaining the mask interface and two-branch design. Line 2: remove UOT from planning. Line 3: give TMP only recent transitions. Line 4: use random transitions from the memory bank. Line 5: use the full Slot-JEPA model with transitions sampled according to Eq. (6).

These comparisons support three conclusions: 1) UOT is important for planning. Comparing lines 2 and 5, UOT improves success in every planar environment and robot Reach, even with goal-pose guidance for line 2. 2) Complex planar tasks benefit from TMP’s diverse transition context. Lines 1 and 5 perform comparably on TwoRoom, TwoRoomR, Maze, and MazeR, consistent with recent information being most useful for these simpler tasks. On PushT and PushObj, line 5 performs better, highlighting the value of TMP’s access to diverse historical transitions. 3) Context must

balance relevance and diversity. Lines 3 and 5 show the benefit of drawing from a broader history;   
lines 4 and 5 show that query-conditioned relevance matters, especially under layout shift.

## 5 LIMITATIONS

Slot decomposition is trained without instance or part annotations, using feature reconstruction rather than physical identity as its objective. Its masks support planning, but one slot may merge multiple objects or split one object, so consistent instance- or part-level decomposition is not guaranteed. This can limit reasoning about specific components and contact points. TMP builds memory from observed transitions and retrieves context using fixed state and mask cues. These cues may miss contact or appearance changes and select redundant examples. Better memory construction and retrieval criteria that balance relevance with diversity remain open problems.

## 6 CONCLUSION

We introduced Slot-JEPA, which factorizes dense visual representations into stable object-structured slots and time-varying spatial masks for action-conditioned prediction. This structure enables a UOT planning loss that is better aligned with spatial progress when pointwise losses provide weak guidance. TMP further draws on relevant transitions beyond a short contiguous history. Across seven visual-understanding benchmarks, mixtures of encoder and slot-reconstructed features remain transferable, while six planar environments and four Robotic Arm Planning tasks in the REALM-JEPA dataset show improved planning success, especially on the more complex manipulation tasks. These results show the value of combining structured mask prediction, spatially informed goal costs, and transition memory for visual planning.

## REFERENCES

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15619–15629, 2023.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. Transactions on Machine Learning Research, 2024.

Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. COCO-Stuff: Thing and stuff classes in context. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1209–1218, 2018.

Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In Advances in Neural Information Processing Systems, volume 33, pp. 9912–9924, 2020.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In IEEE/CVF International Conference on Computer Vision, pp. 9630–9640, 2021.

Xiaokang Chen, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo, Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, and Jingdong Wang. Context autoencoder for self-supervised representation learning. International Journal of Computer Vision, 132(1):208–223, 2024. doi: 10.1007/ s11263-023-01852-4.

Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020.

Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In IEEE/CVF International Conference on Computer Vision, pp. 9620–9629, 2021.

Ho Kei Cheng and Alexander G. Schwing. XMem: Long-term video object segmentation with an atkinson-shiffrin memory model. In European Conference on Computer Vision, pp. 640–658, 2022.

Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the object back into video object segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3151–3161, 2024.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10–11):1684–1704, 2025. doi: 10.1177/02783649241273668.

Lénaïc Chizat, Gabriel Peyré, Bernhard Schmitzer, and François-Xavier Vialard. Scaling algorithms for unbalanced transport problems. Mathematics ofComputation, 87(314):2563–2609, 2018a.

Lénaïc Chizat, Gabriel Peyré, Bernhard Schmitzer, and François-Xavier Vialard. Unbalanced optimal transport: Dynamic and kantorovich formulations. Journal of Functional Analysis, 274(11): 3090–3123, 2018b.

Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In Advances in Neural Information Processing Systems, volume 31, pp. 4754–4765, 2018.

Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems, volume 26, pp. 2292–2300, 2013.

Siran Dai, Qianqian Xu, Peisong Wen, Yang Liu, and Qingming Huang. Exploring structural degradation in dense representations for self-supervised learning. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-0565. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 187d94b3c93343f0e925b5cf729eadd5-Abstract-Conference.html.

Pieter-Tjerk de Boer, Dirk P. Kroese, Shie Mannor, and Reuven Y. Rubinstein. A tutorial on the cross-entropy method. Annals of Operations Research, 134(1):19–67, 2005. doi: 10.1007/ s10479-005-5724-z.

David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. In Advances in Neural Information Processing Systems, pp. 2366–2374, 2014.

Gamaleldin F. Elsayed, Aravindh Mahendran, Sjoerd van Steenkiste, Klaus Greff, Michael C. Mozer, and Thomas Kipf. SAVi++: Towards end-to-end object-centric learning from real-world videos. In Advances in Neural Information Processing Systems, pp. 28940–28954, 2022.

Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes (VOC) challenge. International Journal ofComputer Vision, 88:303–338, 2010. doi: 10.1007/s11263-009-0275-4.

Chelsea Finn and Sergey Levine. Deep visual foresight for planning robot motion. In IEEE International Conference on Robotics and Automation, pp. 2786–2793, 2017.

Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised learning for physical interaction through video prediction. In Advances in Neural Information Processing Systems, volume 29, pp. 64–72, 2016.

Pete Florence, Corey Lynch, Andy Zeng, Oscar A. Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson. Implicit behavioral cloning. In Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, pp. 158–168, 2022.

Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the KITTI vision benchmark suite. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 3354–3361, 2012.

David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In Advances in Neural Information Processing Systems, volume 31, pp. 2451–2463, 2018.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 2555–2565, 2019.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16000–16009, 2022.

Yihang Huang, Peisong Wen, Chenhao Zhou, Difu Feng, and Qianqian Xu. When 1+1 backfires: Rethinking relevance-diversity balancing for sample-aware visual token reduction. In ACM International Conference on Multimedia, 2026. Accepted.

Rishabh Kabra, Daniel Zoran, Goker Erdogan, Loic Matthey, Antonia Creswell, Matt Botvinick, Alexander Lerchner, and Chris Burgess. SIMONe: View-invariant, temporally-abstracted object representations via unsupervised video decomposition. In Advances in Neural Information Processing Systems, volume 34, pp. 20146–20159, 2021.

Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

Alexander Khazatsky et al. DROID: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems, 2024.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Thomas Kipf, Gamaleldin F. Elsayed, Aravindh Mahendran, Austin Stone, Sara Sabour, Georg Heigold, Rico Jonschkowski, Alexey Dosovitskiy, and Klaus Greff. Conditional object-centric learning from video. In International Conference on Learning Representations, 2022.

Yang Liu, Qianqian Xu, Peisong Wen, Siran Dai, and Qingming Huang. When the future becomes the past: Taming temporal correspondence for self-supervised video representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24033–24044, 2025. URL https://openaccess.thecvf. com/content/CVPR2025/html/Liu\_When\_the\_Future\_Becomes\_the\_Past\_ Taming\_Temporal\_Correspondence\_for\_CVPR\_2025\_paper.html.

Yang Liu, Qianqian Xu, Peisong Wen, Siran Dai, Xilin Zhao, and Qingming Huang. From static to dynamic: Exploring self-supervised image-to-video representation transfer learning. arXiv preprint arXiv:2603.26597, 2026. doi: 10.48550/arXiv.2603.26597. URL https://arxiv. org/abs/2603.26597. Accepted at CVPR 2026.

Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. In Advances in Neural Information Processing Systems, volume 33, pp. 11525–11538, 2020.

Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations, 2017.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. LeWorld-Model: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312, 2026.

Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-JEPA 2.1: Unlocking dense features in video self-supervised learning. In European Conference on Computer Vision, pp. 671–689, 2026. doi: 10.1007/978-3-032-37369-4\_36.

Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3M: A universal visual representation for robot manipulation. In Conference on Robot Learning, volume 205 of Proceedings ofMachine Learning Research, pp. 892–909, 2023.

Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, and Randall Balestriero. Causal-JEPA: Learning world models through object-level latent masking. In 2nd Workshop on Compositional Learning at ICML, 2026.

Shukrullo Nazirjonov, Sai Prasanna, Anna Manasyan, and Georg Martius. Better slots, better worlds: Representation quality and robustness in object-centric world models. In Model-Based Reinforcement Learning in the Era ofGenerative World Models Workshop at RLC, 2026.

Seoung Wug Oh, Joon-Young Lee, Kalyan Sunkavalli, and Seon Joo Kim. Fast video object segmentation by reference-guided mask propagation. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 7376–7385, 2018.

Seoung Wug Oh, Joon-Young Lee, Ning Xu, and Seon Joo Kim. Video object segmentation using space-time memory networks. In IEEE/CVF International Conference on Computer Vision, pp. 9226–9235, 2019.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

Federico Perazzi, Anna Khoreva, Rodrigo Benenson, Bernt Schiele, and Alexander Sorkine-Hornung. Learning video object segmentation from static images. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 2663–2672, 2017.

Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alexander Sorkine-Hornung, and Luc Van Gool. The 2017 DAVIS challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017.

Ali Pourganjalikhan and Charalambos Poullis. Adaptive memory management for video object segmentation. In Conference on Robots and Vision, pp. 75–82, 2022.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloé Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. In International Conference on Learning Representations, 2025.

James B. Rawlings, David Q. Mayne, and Moritz M. Diehl. Model Predictive Control: Theory, Computation, and Design. Nob Hill Publishing, 2 edition, 2026.

Martin Sedlacek, Pavlo Yefanov, Georgy Ponimatkin, Jai Bardhan, Simon Pilc, Mederic Fourmy, Evangelos Kazakos, Cees G. M. Snoek, Josef Sivic, and Vladimir Petrik. REALM: A real-to-sim validated benchmark for generalization in robotic manipulation. IEEE Robotics and Automation Letters, 11(7):8315–8322, 2026. doi: 10.1109/LRA.2026.3692088.

Maximilian Seitzer, Max Horn, Andrii Zadaianchuk, Dominik Zietlow, Tianjun Xiao, Carl-Johann Simon-Gabriel, Tong He, Zheng Zhang, Bernhard Schölkopf, Thomas Brox, and Francesco Locatello. Bridging the gap to real-world object-centric learning. In International Conference on Learning Representations, 2023.

Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from RGBD images. In European Conference on Computer Vision, pp. 746–760, 2012.

Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

Shuran Song, Samuel P. Lichtenberg, and Jianxiong Xiao. SUN RGB-D: A RGB-D scene understanding benchmark suite. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 567–576, 2015.

Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, and Yann LeCun. What drives success in physical planning with joint-embedding predictive world models? Transactions on Machine Learning Research, 2026.

Ying Wang, Oumayma Bounou, Yann LeCun, and Mengye Ren. AdaJEPA: An adaptive latent world model. arXiv preprint arXiv:2606.32026, 2026.

Peisong Wen, Qianqian Xu, Siran Dai, Runmin Cong, and Qingming Huang. Semantic concentration for self-supervised dense representations learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(2):1185–1201, 2026. doi: 10.1109/TPAMI.2025.3609758.

Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, and Animesh Garg. SlotFormer: Unsupervised visual dynamics simulation with object-centric models. In International Conference on Learning Representations, 2023.

Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ADE20K dataset. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 633–641, 2017.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World models on pre-trained visual features enable zero-shot planning. In International Conference on Learning Representations, 2025.

Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. iBOT: Image BERT pre-training with online tokenizer. In International Conference on Learning Representations, 2022a.

Pan Zhou, Yichen Zhou, Chenyang Si, Weihao Yu, Teck Khim Ng, and Shuicheng Yan. Mugs: A multi-granular self-supervised learning framework. arXiv preprint arXiv:2203.14415, 2022b.

Adrian Ziegler and Yuki M. Asano. Self-supervised learning of object parts for semantic segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14482–14491, 2022.

## APPENDIX CONTENTS

A Additional Method Details 17   
A.1 Complete Slot Decomposition . . 17   
A.2 Predictor Architecture, Pretraining, and Memory . .17   
A.3 UOT and Static-Slot Identification. .19   
B Experimental Details and Complete Results 20   
B.1 Pretraining . . 20   
B.2 Visual Understanding . 21   
B.3 Planar Planning . 25   
B.4 Robotic Arm Planning . 25   
B.5 Additional Visualizations . .28

## A ADDITIONAL METHOD DETAILS

This appendix gives the architecture and procedures shared by pretraining and all downstream tasks. Environment-specific slot and memory initialization, UOT settings, and planning procedures are reported in Appendix B.

## A.1 COMPLETE SLOT DECOMPOSITION

Reference features and initialization. During pretraining, the reference set $\mathcal { V } ^ { S }$ consists only of the observed/context frames in the sampled clip; none of the target frames to be predicted are used to construct the slots. We encode these frames once and partition their patch features into non-overlapping windows spanning 4 × 4 spatial patches. Within each window, four input patch tokens at evenly spaced valid positions initialize four local slots. This feature-dependent initialization avoids introducing a separate learned slot template while retaining spatial diversity.

Local-to-global Slot Attention. Each local window is processed by three Slot Attention iterations. In every iteration, layer-normalized slots form queries and layer-normalized patch tokens form key and values. Attention is normalized first across the competing slots and then across the input tokens assigned to each slot. The resulting weighted value is passed through a GRU update followed by a residual two-layer MLP, as formalized in Sec. 3.2. We pool the local slots from all windows and select $K = 1 6$ of them by farthest-point sampling (FPS) in feature space. FPS starts from a valid local slot and repeatedly chooses the slot with the largest distance to its nearest selected slot. The selected features initialize a second, global Slot Attention module, which applies the same three-step iterative update to the pooled local-slot candidates and produces the shared slot bank S. The complete local-to-global extraction and mask-decoding network is shown in Fig. 7.

Mask decoding and optimization. The mask decoder first projects patch features and slots to width 128. Two decoder blocks alternately update the patch queries by cross-attending to the fixed slot bank and by applying self-attention among patches. For every patch–slot pair, an MLP receives the concatenated decoded patch feature and slot feature and produces an assignment logit. A softmax across the K slots yields M, and the mask-weighted slot sum reconstructs the original dense representation as in Eq. (1). We jointly optimize the local and global Slot Attention modules and the mask decoder with the feature-reconstruction and slot-orthogonality terms in Sec. 3.2. The same extracted slot bank is held fixed when decoding the observed, predicted, and goal frames of one environment. The environment-specific choice of inference reference frames and the slot-refresh policy are given in Appendix B.

## A.2 PREDICTOR ARCHITECTURE, PRETRAINING, AND MEMORY

Predictor architecture. Fig. 8 shows the layer-level architecture of the two TMP branches. TMP uses branch-specific context–query stacks. The reconstructed-feature branch has 12 blocks of width 192 with three attention heads. The mask branch applies separate object-wise predictors at three spatial scales, each with four blocks. Their (spatial size, width, heads) configurations are (16, 48, 3), (12, 96, 3), and (6, 196, 4). The current and future fields are concatenated channel-wise at each scale; the unavailable future query field is replaced by learned tokens. After attention, we retain only the query positions associated with those future tokens, upsample the three outputs, and concatenate them to form $\pmb { H } _ { t + 1 } ^ { \mathrm { m a s k } }$ . The shared feature branch uses the same context–query organization on the reconstructed feature field and masks to model interactions across slots. Its output is fused with every object-wise mask feature before the slot-wise softmax. These dimensions are fixed in comparisons with the sequential-predictor ablation unless stated otherwise.

Bidirectional pretraining contexts. Each pretraining sample contains seven frames: four observed/context frames and three held-out target frames. We construct both directions from every clip. In the forward case, the first four frames predict the last three; in the backward case, the last four predict the first three. Thus the predictor learns a relation between observed transitions and a query transition without relying on a fixed chronological direction. The observed transitions from the same sampled clip directly form the training context, so pretraining neither retrieves from nor maintains a persistent memory bank.

![](images/b5d8fd9334ffe5a371312d37137e5a8750b7b32ef7f174f6f6b4a13fb1fa7914.jpg)  
Figure 7: Network architecture of the slot decomposition module.

Inverse condition encoders. Natural pretraining videos have no action labels. We therefore reconstruct each pair of source and target frames from their masks and shared slots, and feed the paired reconstructions and masks to an inverse transition encoder. A spatial projection, learned condition tokens, and an eight-layer, three-head Transformer of width 192 compress their change into the low-dimensional pseudo-action $\tilde { a } _ { t }$ in Sec. 3.3. A companion Transformer with the same dimensions maps each reconstructed frame and its masks to a pseudo-state. The target reaches TMP through these low-dimensional condition tokens rather than through the full target mask field. After pretraining, both inverse encoders are discarded and replaced by task-specific state and action encoders; the TMP body is transferred.

Memory construction and maintenance at inference. All entries in an inference memory bank are observed transitions decoded with one common slot bank. An entry stores $\left( M _ { i } , r _ { i } , a _ { i } , M _ { i + 1 } , r _ { i + 1 } \right)$ and each newly executed transition is appended after its next observation becomes available. For a query, TMP computes relevance from normalized low-dimensional state, per-slot area, and per-slot centroid, then samples five transitions without replacement using Eq. (6). One sampled context set is shared by all candidate actions in the same prediction request. When the slot bank is refreshed, retained frames are decoded again with the new slots and the associated memory masks are rebuilt, preserving slot-index consistency. The source of the initial nonempty bank and the refresh schedule depend on the environment and are specified in Appendix B.

![](images/71139d542fb05ecd203996df4e52a38f992f5cfe3b9d0f9d94d6788636022523.jpg)  
Figure 8: TMP architecture. Observed transitions form the context blocks, while inverse-encoded pseudo-action and learned future tokens specify the query transition. The reconstructed-feature tower and the three-scale object feature pyramid run in parallel and are fused to predict the future slot mask.

## A.3 UOT AND STATIC-SLOT IDENTIFICATION

UOT computation. The UOT term in Sec. 3.4 is evaluated independently for corresponding slot masks. We retain each mask’s original mass: the masks are neither divided by their areas nor constrained to have equal total mass. Instead, the relaxed marginal constraints allow mass to disappear or emerge at finite cost. For nonnegative vectors x and y, we use the generalized KL divergence

$$
\operatorname { K L } ( x \parallel y ) = \sum _ { i } \left[ x _ { i } \log { \frac { x _ { i } } { y _ { i } } } - x _ { i } + y _ { i } \right] .\tag{14}
$$

The default ground cost is $c _ { i j } = \| \pmb { \xi } _ { i } - \pmb { \xi } _ { j } \| _ { 2 } ^ { 2 }$ on normalized patch coordinates. Given ${ \boldsymbol { p } } , { \boldsymbol { q } } \in \mathbb { R } _ { + } ^ { N }$ , we form the Gibbs kernel and unbalanced exponent

$$
K _ { i j } = \exp ( - c _ { i j } / \varepsilon ) , \qquad \theta = \frac { \rho } { \rho + \varepsilon } .\tag{15}
$$

Algorithm 1 gives the implemented scaling procedure. All divisions and logarithms clamp their arguments below by $\delta = \bar { 1 0 } ^ { - 8 }$ for numerical stability. The entropy parameter ε enters through K and the scaling exponent; after the final iteration, we evaluate the spatial transport and relaxed-marginal terms of Eq. (12) at the resulting plan.

To emphasize slots that must change between the reference mask $p ^ { \mathrm { r e f } }$ and target mask $q ,$ we compute weights only from observed or target masks, never from the prediction:

$$
\begin{array} { l } { { \displaystyle d _ { k } = \frac { 1 } { N } \| q _ { k } - p _ { k } ^ { \mathrm { r e f } } \| _ { 1 } , \quad \bar { d } = \frac { 1 } { K } \sum _ { j } d _ { j } , } } \\ { { \displaystyle w _ { k } = \mathrm { c l i p } \bigg ( \frac { d _ { k } } { \bar { d } } , w _ { \mathrm { m i n } } , w _ { \mathrm { m a x } } \bigg ) , \quad \mathcal { L } _ { \mathrm { u o t } } = \frac { \sum _ { k } w _ { k } \mathcal { T } _ { \varepsilon , \rho } \big ( p _ { k } , q _ { k } \big ) } { \sum _ { k } w _ { k } } . } } \end{array}\tag{16}
$$

Algorithm 1 Generalized Sinkhorn scaling for one aligned slot-mask pair   
Input: masks $\boldsymbol { p } , \boldsymbol { q } \in \mathbb { R } _ { + } ^ { N }$ , ground cost $\mathbf { \Psi } _ { C } \in \mathbb { R } _ { + } ^ { N \times N }$ , entropy ε, mass penalty $\rho ,$ iteration count $L ,$ and   
numerical floor δ.   
1. Set a = max(p, δ), b = max(q, δ), K = max(exp(−C/ε), δ), and $\theta = \rho / ( \rho + \varepsilon )$ element  
wise.   
2. Initialize $u = { \bf 1 } _ { N }$ and $v = \mathbf { 1 } _ { N } .$   
3. For $\ell = 1 , \ldots , L ,$ update   
$u \gets \left( \frac { a } { \operatorname* { m a x } ( K v , \delta ) } \right) ^ { \theta } , \qquad v \gets \left( \frac { b } { \operatorname* { m a x } ( K ^ { \top } u , \delta ) } \right) ^ { \theta } .$   
4. Construct π = diag(u)K diag(v).   
Return: ⟨π, $C \rangle + \rho \mathrm { K L } ( \pi \mathbf { 1 } \| a ) + \rho \mathrm { K L } ( \pi ^ { \top } \mathbf { 1 } \| b ) .$

If all $d _ { k }$ vanish, we set all weights to one. We use $w _ { \mathrm { m i n } } = 0 . 0 5$ and $w _ { \mathrm { m a x } } = 5 ;$ ; the same weights are applied to the mask $\scriptstyle - \ell _ { 1 }$ term. This reduction prevents persistent background slots from dominating the objective.

Static-slot identification. We determine static slots from predicted mask changes rather than semantic labels. For each probe transition $p$ and slot k, we compare an aligned source mask $m _ { k }$ with its model-predicted next mask $\hat { m } _ { k } ^ { \left( p \right) }$ . We retain the L largest absolute patch-wise changes and average them. The model-level change score is

$$
\Delta _ { k } = \frac { 1 } { { \cal P } } \sum _ { p = 1 } ^ { { \cal P } } \frac { 1 } { { \cal L } } \sum _ { n \in \mathrm { T o p L } ( | \hat { m } _ { k } ^ { ( p ) } - m _ { k } | ) } | \hat { m } _ { k , n } ^ { ( p ) } - m _ { k , n } | .\tag{17}
$$

A slot is static when $\Delta _ { k } \ \le \ \tau _ { \mathrm { s t a t i c } }$ and moving when $\Delta _ { k } > \tau _ { \mathrm { m o v i n g } } ;$ slots in the gap are left unclassified. This two-threshold rule prevents borderline changes from contaminating either group. As illustrated in Fig. 9, moving slots shift with controlled entities, whereas static slots remain aligned with persistent scene geometry.

For obstacle-aware planning, we peak-normalize and optionally sharpen the selected static masks, then take their maximum to obtain an obstacle map. For a moving slot, this map modifies the ground cost as

$$
c _ { i j } ^ { \mathrm { o b s } } = c _ { i j } + \lambda _ { \mathrm { o b s } } r _ { i j } ,\tag{18}
$$

where $r _ { i j }$ measures obstacle occupancy along the path from patch i to patch j. The ground cost therefore encodes obstacle geometry directly. The probe transitions, thresholds, and ground-cost variant are environment-specific and are reported with the corresponding experiments.

## B EXPERIMENTAL DETAILS AND COMPLETE RESULTS

This appendix follows the four stages of the empirical study: representation pretraining, visualunderstanding evaluation, planar planning, and robotic arm planning. We collect qualitative analyses in a final visualization subsection so that the representation, planning objective, and predicted rollouts can be compared in one place.

## B.1 PRETRAINING

Initialization and data. We initialize the visual encoder from the released distilled V-JEPA 2.1 ViT-B/16 checkpoint (Mur-Labadia et al., 2026). This 80M-parameter encoder is distilled from the 2Bparameter V-JEPA 2.1 ViT-G teacher on VisionMix-163M, which combines LVD-142M images with videos from SSv2, Kinetics, HowTo100M, and YT-1B. We then pretrain Slot-JEPA on the training split of Kinetics-400 (Kay et al., 2017), which contains roughly 240,000 clips. The complete dataset contains 306,245 clips from 400 human-action classes. Each clip lasts approximately 10 seconds and comes from a distinct YouTube video. Its content spans individual actions such as drawing and drinking, person–person interactions such as hugging and shaking hands, and person–object interactions such as opening objects and playing instruments. We decode the videos at $3 8 4 \times 3 8 4$ resolution.

![](images/5129ed071cdac99091e0fac995eb06e41301a2a00e4cbeb01d1e8b0e30efa07a.jpg)  
Figure 9: Examples of slots identified from their predicted changes. Moving-slot masks follow the controlled entity across probe transitions, while static-slot masks remain aligned with the wall or maze geometry. The labels are produced by the change-score rule rather than ground-truth segmentation.

Optimization. We use $K = 1 6$ global slots and $4 \times 4$ spatial windows with four local slots per window. Each seven-frame clip supplies four observed frames and three target frames; the observed transitions provide context for predicting the targets. We jointly optimize the slot decomposition module, TMP, and the inverse transition encoders while fine-tuning the final six visual-encoder blocks; the remaining blocks are frozen. Pretraining runs for 10 epochs on eight RTX 4090 GPUs and takes about 30 hours. We use AdamW (Loshchilov & Hutter, 2019) with learning rate $1 0 ^ { - 4 }$ and weight decay $1 0 ^ { - 2 }$ . The learning rate follows cosine decay (Loshchilov & Hutter, $2 0 1 7 )$ to $1 0 ^ { - 5 }$ . We set $\lambda _ { \mathrm { o r t h } } = 0 . 0 0 1$ in $\mathcal { L } _ { \mathrm { s l o t } }$ and $( \lambda _ { \mathrm { u o t } } , \lambda _ { \mathrm { m a s k } } , \lambda _ { \mathrm { f e a t } } ) = ( \dot { 3 } . 0 , 3 . 0 , 0 . 1 )$ in $\mathcal { L } _ { \mathrm { p r e d } }$ . UOT uses five Sinkhorn iterations and $\varepsilon = 0 . 0 5$ . For all downstream tasks, the pretrained encoder and slot decomposition module are frozen.

## B.2 VISUAL UNDERSTANDING

Downstream protocols. The encoder and decomposition module remain frozen for all three probes. Segmentation trains a linear head for 2,000 steps using Adam (Kingma & Ba, 2015) with batch size 256, selecting the learning rate from $\{ 1 0 ^ { - 4 } , \dot { 3 } \times 1 0 ^ { - 4 } , 8 \times 1 0 ^ { - 4 } , 1 0 ^ { \smile 3 } , 2 \times 1 0 ^ { - 3 } , 3 \times \bar { 1 } 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ Depth trains a linear scale-invariant logarithmic loss (SigLoss) head (Eigen et al., 2014) for 38,400 steps (AdamW, learning rate $3 \times 1 0 ^ { - 4 }$ , batch size 16). VOS requires no training: nearest-neighbor propagation uses the latest 20 frames, top-5 neighbors, and 16 frames to initialize slots from first-frame mask centers.

Competitors. We compare against MoCo v2 (Chen et al., 2020), MoCo v3 (Chen et al., 2021), SwAV (Caron et al., 2020), DINO (Caron et al., 2021), iBOT (Zhou et al., 2022a), Mugs (Zhou et al., 2022b), Leopart (Ziegler & Asano, 2022), Context Autoencoder (CAE) (Chen et al., 2024), and Masked Autoencoders (MAE) (He et al., 2022). The locally evaluated V-JEPA 2.1 (Mur-Labadia et al., 2026) is our initialization baseline. In Tables C1–C2, only the two DINOv2 (Oquab et al., 2024) models pretrained on LVD-142M are treated as gray scale references; the reproduced DINOv2 models trained on ImageNet-1K remain in the ranked comparison. In Table C3, only the 7B-parameter DINOv3 (Siméoni et al., 2025) model is excluded as a scale reference.

Table C1: Complete linear semantic-segmentation results (%). “Data” and “Ep.” denote representation-pretraining data and epochs; IN1K, COCO, K400, and VM163M denote ImageNet-1K, COCO, Kinetics-400, and VisionMix-163M. † marks reproduced DINOv2.
<table><tr><td colspan="5"></td><td colspan="2">COCOStuff-27</td><td colspan="2">ADE20K</td><td colspan="2">PASCAL VOC</td></tr><tr><td>Method</td><td>Backbone</td><td>Data</td><td> $\mathrm { E p . }$ </td><td>mIoU↑</td><td>Acc.↑</td><td>mIoU↑</td><td>Acc.↑</td><td>mIoU↑</td><td></td><td>Acc.↑</td></tr><tr><td>DINOv2</td><td>ViT-S/14</td><td>LVD-142M</td><td></td><td>57.4</td><td>77.7</td><td>43.5</td><td>78.9</td><td></td><td>79.4</td><td>95.3</td></tr><tr><td>DINOv2</td><td>ViT-B/14</td><td>LVD-142M</td><td>一</td><td>59.1</td><td>78.4</td><td>46.3</td><td></td><td>79.8</td><td>81.3</td><td>95.7</td></tr><tr><td>MoCo v2</td><td>RN-50</td><td>IN1K</td><td>800</td><td>37.6</td><td>64.0</td><td></td><td>20.0</td><td>61.2</td><td>53.3</td><td>88.4</td></tr><tr><td>SwAV</td><td>RN-50</td><td>IN1K</td><td>800</td><td>42.2</td><td>68.0</td><td></td><td>23.3</td><td>65.8</td><td>55.4</td><td>89.0</td></tr><tr><td>DINO</td><td>RN-50</td><td>IN1K</td><td>800</td><td>40.7</td><td>66.6</td><td></td><td>22.4</td><td>64.5</td><td>53.1</td><td>88.4</td></tr><tr><td>MoCo v3</td><td>ViT-S/16</td><td>IN1K</td><td>300</td><td>43.4</td><td>70.6</td><td></td><td>24.4</td><td>70.5</td><td>61.7</td><td>91.4</td></tr><tr><td>DINO</td><td>ViT-S/16</td><td>IN1K</td><td>800</td><td>39.5</td><td>67.1</td><td></td><td>23.1</td><td>67.4</td><td>50.0</td><td>88.4</td></tr><tr><td>iBOT</td><td>ViT-S/16</td><td>IN1K</td><td>800</td><td>47.7</td><td></td><td>72.7</td><td>28.5</td><td>72.3</td><td>66.7</td><td>92.4</td></tr><tr><td>Mugs</td><td>ViT-S/16</td><td>IN1K</td><td>800</td><td>48.7</td><td>72.6</td><td></td><td>30.9</td><td>72.8</td><td>71.8</td><td>93.3</td></tr><tr><td>Leopart</td><td>ViT-S/16</td><td>IN1K+COCO</td><td>800+50</td><td>48.8</td><td>73.2</td><td></td><td>29.7</td><td>72.1</td><td>68.2</td><td>92.6</td></tr><tr><td>DINOv2†</td><td>ViT-S/14</td><td>IN1K</td><td>800</td><td>48.1</td><td>72.7</td><td></td><td>29.6</td><td>72.6</td><td>68.3</td><td>92.7</td></tr><tr><td>CAE</td><td>ViT-B/16</td><td>IN1K</td><td>800</td><td>45.5</td><td>71.0</td><td></td><td>27.0</td><td>71.9</td><td>64.9</td><td>92.2</td></tr><tr><td>CAE</td><td>ViT-B/16</td><td>IN1K</td><td>1600</td><td>46.2</td><td>71.6</td><td></td><td>27.5</td><td>72.2</td><td>65.6</td><td>92.3</td></tr><tr><td>MAE</td><td>ViT-B/16</td><td>IN1K</td><td>800</td><td>37.9</td><td>65.5</td><td></td><td>20.6</td><td>66.2</td><td>52.3</td><td>88.8</td></tr><tr><td>MoCo v3</td><td>ViT-B/16</td><td>IN1K</td><td>300</td><td>48.2</td><td>73.0</td><td></td><td>29.6</td><td>72.9</td><td>69.6</td><td>92.9</td></tr><tr><td>DINO</td><td>ViT-B/16</td><td>IN1K</td><td>800</td><td>45.7</td><td>71.5</td><td></td><td>28.8</td><td>72.0</td><td>63.6</td><td>91.6</td></tr><tr><td>DINO</td><td>ViT-B/8</td><td>IN1K</td><td>800</td><td>47.0</td><td>72.3</td><td></td><td>30.5</td><td>72.7</td><td>65.3</td><td>92.3</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>IN1K</td><td>400</td><td>52.2</td><td>75.3</td><td></td><td>34.3</td><td>75.2</td><td>73.3</td><td>93.9</td></tr><tr><td>Mugs</td><td>ViT-B/16</td><td>IN1K</td><td>400</td><td>51.4</td><td>74.1</td><td></td><td>34.6</td><td>74.5</td><td>72.7</td><td>93.5</td></tr><tr><td>DINOv2†</td><td>ViT-B/14</td><td>IN1K</td><td>400</td><td>53.0</td><td>75.6</td><td></td><td>35.5</td><td>75.8</td><td>74.9</td><td>93.9</td></tr><tr><td>V-JEPA 2.1</td><td>ViT-B/16</td><td>VM163M</td><td>一</td><td>52.8</td><td>75.8</td><td></td><td>35.5</td><td>76.0</td><td>72.1</td><td>93.8</td></tr><tr><td>Slot-JEPA</td><td>ViT-B/16</td><td>K400</td><td>10</td><td>56.1</td><td>77.8</td><td></td><td>38.9</td><td>77.8</td><td>75.1</td><td>94.5</td></tr></table>

Results. As shown in Tab. C1, Slot-JEPA consistently improves over its V-JEPA 2.1 initialization on all three segmentation datasets. The mIoU gains are 3.3% on COCOStuff-27, 3.4% on ADE20K, and 3.0% on PASCAL VOC, with corresponding improvements in pixel accuracy; it also achieves the strongest results across all six metrics among the comparison models, excluding the gray reference rows. Tab. C2 shows improvements on NYUv2 and on transfer from NYUv2 to SUNRGBD. On NYUv2, $\delta _ { 1 }$ increases from 80.5% to 83.1% while AbsRel and RMSE decrease from 0.139 and 0.486 to 0.130 and 0.456. All three transfer metrics also improve, although performance on KITTI decreases slightly. Finally, Tab. C3 reports an increase from 65.9% to 66.6% in DAVIS J&F, with gains in both region similarity and contour accuracy. This is the strongest result among the comparison entries, supporting that slot reconstruction preserves temporally consistent object information.

Table C2: Complete monocular-depth and NYUv2-to-SUNRGBD transfer results. $\dagger$ marks reproduced DINOv2.
<table><tr><td></td><td></td><td colspan="3">NYUv2</td><td colspan="3">KITTI</td><td colspan="3">NYU→SUNRGBD</td></tr><tr><td>Method</td><td>Backbone</td><td> $\delta _ { 1 }$  ↑</td><td>AbsRel↓</td><td>RMSE↓</td><td> $\delta _ { 1 }$  ↑</td><td>AbsRel↓</td><td>RMSE↓</td><td> $\delta _ { 1 }$ </td><td>↑AbsRel↓</td><td>RMSE↓</td></tr><tr><td>DINOv2 (LVD-142M)</td><td>ViT-S/14</td><td>84.8</td><td>0.134</td><td>0.437</td><td>91.0</td><td>0.096</td><td>3.119</td><td>77.4</td><td>0.269</td><td>0.473</td></tr><tr><td>DINOv2 (LVD-142M)</td><td>ViT-B/14</td><td>87.8</td><td>0.120</td><td>0.408</td><td>92.0</td><td>0.092</td><td>2.931</td><td>80.7</td><td>0.255</td><td>0.443</td></tr><tr><td>MoCo v2</td><td>RN-50</td><td>62.9</td><td>0.230</td><td>0.719</td><td>82.4</td><td>0.130</td><td>4.319</td><td>55.5</td><td>0.366</td><td>0.749</td></tr><tr><td>SwAV</td><td>RN-50</td><td>61.9</td><td>0.225</td><td>0.721</td><td>84.3</td><td>0.123</td><td>3.985</td><td>55.0</td><td>0.367</td><td>0.753</td></tr><tr><td>DINO</td><td>RN-50</td><td>61.6</td><td>0.228</td><td>0.726</td><td>83.7</td><td>0.127</td><td>4.068</td><td>54.7</td><td>0.370</td><td>0.755</td></tr><tr><td>MoCo v3</td><td>ViT-S/16</td><td>69.8</td><td>0.192</td><td>0.615</td><td>77.2</td><td>0.170</td><td>5.007</td><td>50.6</td><td>0.411</td><td>0.815</td></tr><tr><td>DINO</td><td>ViT-S/16</td><td>67.7</td><td>0.202</td><td>0.651</td><td>83.9</td><td>0.132</td><td>4.061</td><td>59.6</td><td>0.343</td><td>0.678</td></tr><tr><td>iBOT</td><td>ViT-S/16</td><td>72.1</td><td>0.183</td><td>0.584</td><td>85.3</td><td>0.125</td><td>3.907</td><td>63.5</td><td>0.330</td><td>0.634</td></tr><tr><td>Mugs</td><td>ViT-S/16</td><td>72.3</td><td>0.179</td><td>0.586</td><td>84.1</td><td>0.132</td><td>3.963</td><td>64.4</td><td>0.306</td><td>0.613</td></tr><tr><td> $\mathrm { D I N O v } 2 ^ { \dag }$ </td><td>ViT-S/14</td><td>72.7</td><td>0.184</td><td>0.583</td><td>86.6</td><td>0.112</td><td>3.757</td><td>63.2</td><td>0.332</td><td>0.631</td></tr><tr><td>MAE</td><td>ViT-B/16</td><td>65.3</td><td>0.224</td><td>0.67384.3</td><td></td><td>0.123</td><td>4.087 59.1</td><td></td><td>0.350</td><td>0.688</td></tr><tr><td>MoCo v3</td><td>ViT-B/16</td><td>72.5</td><td>0.181</td><td>0.585</td><td>85.3</td><td>0.122</td><td>3.954</td><td>65.9</td><td>0.308</td><td>0.606</td></tr><tr><td>DINO</td><td>ViT-B/16</td><td>71.7</td><td>0.182</td><td>0.593</td><td>86.5</td><td>0.117</td><td>3.771</td><td>62.8</td><td>0.336</td><td>0.644</td></tr><tr><td>DINO</td><td>ViT-B/8</td><td>72.8</td><td>0.178</td><td>0.579</td><td>86.9</td><td>0.115</td><td>3.697</td><td>65.1</td><td>0.314</td><td>0.613</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>77.7</td><td>0.160</td><td>0.523</td><td>88.3</td><td>0.110</td><td>3.514</td><td>69.1</td><td>0.299</td><td>0.565</td></tr><tr><td>Mugs</td><td>ViT-B/16</td><td>74.9</td><td>0.164</td><td>0.550</td><td>87.7</td><td>0.111</td><td>3.620</td><td>66.6</td><td>0.294</td><td>0.585</td></tr><tr><td> $\mathrm { D I N O v } 2 ^ { \dag }$ </td><td>ViT-B/14</td><td>78.1</td><td>0.160</td><td>0.519</td><td>88.0</td><td>0.112</td><td>3.502</td><td>68.0</td><td>0.310</td><td>0.570</td></tr><tr><td>V-JEPA 2.1</td><td>ViT-B/16</td><td>80.5</td><td>0.139</td><td>0.486</td><td>89.2</td><td>0.102</td><td>3.481</td><td>70.0</td><td>0.292</td><td>0.557</td></tr><tr><td>Slot-JEPA</td><td>ViT-B/16</td><td>83.1</td><td>0.130</td><td></td><td>0.45687.8</td><td>0.107</td><td>3.705</td><td>72.3</td><td>0.278</td><td>0.524</td></tr></table>

Table C3: Complete DAVIS video object segmentation results (%). MoCo v3, SwAV, and MAE report best-checkpoint J&F only.
<table><tr><td>Method</td><td>Backbone</td><td> $\mathcal { I } \uparrow$   $\mathcal { F } \uparrow$ </td><td> $\mathcal { T } \& \mathcal { F } \uparrow$ </td></tr><tr><td>DINOv3</td><td>ViT-7B/16</td><td></td><td>71.1</td></tr><tr><td>MoCo v3</td><td>ViT-S/16</td><td>一</td><td>61.7</td></tr><tr><td>SwAV</td><td>RN-50</td><td></td><td>41.3</td></tr><tr><td>DINO</td><td>ViT-B/16</td><td>61.5 64.9</td><td>63.2</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>63.2 65.5</td><td>64.3</td></tr><tr><td>MAE</td><td>ViT-S/16</td><td></td><td>48.9</td></tr><tr><td>CAE</td><td>ViT-L/16</td><td>62.8 67.5</td><td>65.1</td></tr><tr><td>iBOT</td><td>ViT-L/16</td><td>62.1 65.5</td><td>63.8</td></tr><tr><td>Mugs</td><td>ViT-L/16</td><td>60.3 63.0</td><td>62.1</td></tr><tr><td> $\mathrm { D I N O v } 2$ </td><td>ViT-L/14</td><td>62.2 66.3</td><td>64.2</td></tr><tr><td>V-JEPA 2.1</td><td>ViT-B/16</td><td>63.4 68.3</td><td>65.9</td></tr><tr><td>Slot-JEPA</td><td>ViT-B/16</td><td>63.5 69.7</td><td>66.6</td></tr></table>

![](images/460c3d2dcccb715cb91a1243f1e3bcddb5e53c5696d65e8984fceab03abf5839.jpg)  
Figure 10: The three planar ID/OOD environment pairs.

Table C4: Training settings for the planar environments.
<table><tr><td>Training environment</td><td>Resolution</td><td>Epochs</td><td>Batch size</td><td>Weight decay</td></tr><tr><td>TwoRoom / Maze</td><td>384</td><td>100</td><td>128</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>PushT</td><td>512</td><td>100</td><td>128</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>TwoRoomR / MazeR</td><td>384</td><td>50</td><td>128</td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>PushObj</td><td>512</td><td>500</td><td>64</td><td> $1 0 ^ { - 4 }$ </td></tr></table>

## B.3 PLANAR PLANNING

Environment examples. Fig. 10 organizes the six environments into three ID/OOD pairs. Each pair forms a $2 \times 4$ grid of equally sized images: the first column contains the ID start and goal, the middle two columns contain four OOD training instances, and the final column contains two held-out OOD test instances. For TwoRoomR and MazeR, these are representative subsets of the training and test layouts; PushObj shows its complete shape split.

PushObj is trained with turquoise I, aquamarine Z, violet small-T, and orange square objects, and evaluated on the held-out L and plus shapes. Object poses vary within both splits; shape identity defines the ID/OOD distinction.

Competitors. The planar baselines use LeWorldModel (LeWM) (Maes et al., 2026), DINO-WM (Zhou et al., 2025), V-JEPA 2.1 (Mur-Labadia et al., 2026), DINO class-token representations (Caron et al., 2021), ResNet (He et al., 2016), and R3M (Nair et al., 2023). We report each method with and without the low-dimensional goal state when those results are available.

Slot and memory initialization. All planar cases use one fixed slot bank extracted from a small subset of training trajectories. In-distribution cases initialize the transition memory with randomly sampled training transitions. For an OOD case, we first execute random actions in the new simulated layout or with the new object shape and use the resulting transitions to initialize its memory. These observations are collected before the evaluated goal-directed rollout and decoded with the same fixed slots. Each transition subsequently observed during planning is appended to the bank.

Downstream fine-tuning. We freeze the visual encoder and slot decomposition module and optimize TMP with task-specific state and action encoders using AdamW. The learning rate is $5 \times 1 0 ^ { - 4 }$ for both transferred and new predictor parameters, using the AdamW defaults $( \beta _ { 1 } , \beta _ { 2 } ) =$ (0.9, 0.999) and $\epsilon = 1 0 ^ { - 8 } ;$ no learning-rate schedule is applied. Training uses bfloat16, seed 0, and at most 1,000 iterations per epoch. Each training query predicts $M _ { t + 1 }$ from $( M _ { t } , r _ { t } , a _ { t } )$ using five observed context transitions from the same sampled training segment. During planning, five context transitions $( | \mathcal { C } _ { t } | = 5 )$ are sampled from the 50 nearest memory candidates. We set $( \lambda _ { \mathrm { u o t } } , \lambda _ { \mathrm { m a s k } } , \lambda _ { \mathrm { f e a t } } ) = ( 1 , 1 0 , 1 )$ in $\mathcal { L } _ { \mathrm { p r e d } }$ , except for Push-Object, where we use (0.1, 10, 1) together with a Sobel mask-boundary term of weight 10. UOT uses five Sinkhorn iterations, $\varepsilon = 0 . 0 1$ , and $\rho = 0 . 2 $

Obstacle-aware UOT and MPC. For TwoRoom, TwoRoomR, Maze, and MazeR, the static and moving slot sets are estimated once from 128 deterministic training-transition pairs using the change score in Appendix A.3; the resulting classification is then shared across test cases. We use the ten largest patch changes per transition, with $\tau _ { \mathrm { s t a t i c } } = 0 . 0 5$ and $\tau _ { \mathrm { m o v i n g } } = 0 . 1 5$ . Their UOT ground cost penalizes paths through the static-mask obstacle map. PushT and PushObj use the standard Euclidean ground cost. For each MPC step, the sampled memory contexts are fixed across CEM candidates. TMP rolls out every candidate action sequence, CEM ranks it with the UOT and mask-ℓ<sub>1</sub> goal loss, and the agent executes the first action before observing the next state and replanning.

## B.4 ROBOTIC ARM PLANNING

Environment and benchmark. We construct REALM-JEPA from the REALM simulator and task configurations (Sedlacek et al., 2026) to evaluate visual-goal planning in tabletop manipulation. The environment uses a mounted Franka Panda arm with a Robotiq gripper. At every step, it synchronously renders 1280×720 RGB observations from two fixed external cameras and one wrist-mounted camera. The robot state comprises a 6-D end-effector (EE) pose and the gripper opening, and the 7-D action specifies an absolute EE target pose and gripper command. REALM-JEPA contains 108 cases across four task families. Reach includes 18 targets corresponding to the six Pick objects, six Rotate objects, and six objects used in Place. Each of the other families contains six task definitions under five scene configurations—the nominal scene, two shifted target-object poses, a block-distractor scene, and a shifted-distractor scene—giving 30 Pick, 30 Rotate, and 30 Place cases. Pick uses a banana, bowl, soda can, lemon, marker, or spoon; Rotate uses a banana, soda can, marker, mug, toy die, or water bottle; and Place uses the six object–receptacle pairs banana–box, soda can–bowl, green block–bowl, lemon–tray, marker–bowl, and orange–bowl. In total, the benchmark covers 11 manipulated-object types and three receptacle types, with 18 + 3 × (6 × 5) = 108 cases.

![](images/0b41781943909aba1618bf62b09b174fa4590b96bf3dc22dcdb6e70bc1186a66.jpg)  
Figure 11: Representative REALM-JEPA start and goal images for all four Robotic Arm Planning categories. Each task panel shows the same scene and goal stages from the two external cameras and the wrist camera. Goal 1 and Goal 2 denote successive visual goals when a task has multiple stages.

Each benchmark case fixes the task configuration and initial simulator state, and provides synchronized visual goals from all three cameras. These goals are sampled from a scripted successful reference trajectory; the reference actions and intermediate states are used only to construct the benchmark and are unavailable to the planner. Fig. 11 shows representative start and goal observations for every task family. Reach moves the EE to the target object and uses one visual goal. Pick reaches and grasps the object before lifting it, Rotate grasps and changes its orientation, and Place grasps, transports, and releases it into the designated receptacle; these longer tasks use two successive visual goals. Every goal contains the same three synchronized views as the current observation.

Competitors. We compare with the standard V-JEPA 2.1 predictor (Mur-Labadia et al., 2026) and V-JEPA 2.1 + TMP, which keeps patch representations and the original goal loss but uses our transition-memory architecture. This isolates the effect of the predictor design from the slot-based state interface. On place and rotate tasks, both patch-based baselines use the ground-truth target EE position to bias CEM proposals while the 3-D positional distance exceeds 30 cm. Specifically, the proposal mean is initialized with 20% of the target displacement and clipped per axis to 20% of the action cap. At 30 cm or closer, the bias is disabled and planning uses the visual goal loss alone. Slot-JEPA does not use this guidance.

Slot and memory initialization. Each REALM-JEPA case begins with a short random-action warmup. The initial frame, provided goal images, and warmup observations form the reference set for each view’s slot extraction; the observed warmup transitions initialize its corresponding memory bank. Goal images contribute reference features but no memory transitions. Goal-directed planning starts only after this initialization, and each newly observed transition is appended to the appropriate per-view bank. After an extended rollout, slots may be refreshed from accumulated observed frames. All retained transitions are then decoded again so that their memory masks remain aligned with the refreshed slots. No future frame from the evaluated rollout is used for initialization or refresh.

Downstream fine-tuning. The robot predictor is trained on seven-frame DROID (Khazatsky et al., 2024) clips sampled at 1 fps and resized to $3 8 4 \times 3 8 4$ . Training uses eight GPUs, four clips per GPU, bfloat16 precision, seed 239, 100 epochs, and 300 iterations per epoch. The visual encoder and slot decomposition module remain frozen, while TMP and the task-specific state and action encoders are optimized with AdamW using $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 ) , \epsilon = 1 0 ^ { - \mathsf { \tilde { B } } }$ , and weight decay $1 0 ^ { - 2 }$ ; bias and normalization parameters are excluded from weight decay. The learning rate increases linearly from $1 0 ^ { - 4 }$ to $1 0 ^ { - 3 }$ over the first 15 epochs, remains at $1 0 ^ { - 3 }$ for 70 epochs, and decays linearly to zero over the final 15 epochs. We set $( \lambda _ { \mathrm { u o t } } , \lambda _ { \mathrm { m a s k } } , \lambda _ { \mathrm { f e a t } } ) = ( 3 . 0 , 3 . 0 , 0 . 1 )$ in $\mathcal { L } _ { \mathrm { p r e d } }$ , with five Sinkhorn iterations and $\varepsilon = 0 . 0 5$ . Each training sample uses four observed frames to predict three future target frames and additionally applies two autoregressive prediction steps.

Multi-view prediction and planning. Let $v \in \mathcal { V } =$ {external-1, external-2, wrist} index the three views. They share the same predictor parameters $\theta ,$ but each view independently maintains its slot set $S ^ { ( v ) }$ , mask history, and transition memory $\boldsymbol { B } _ { t } ^ { ( v ) }$ . Context is sampled within each memory, and prediction is performed separately as

$$
\begin{array} { r } { \hat { M } _ { t + 1 } ^ { ( v ) } = F _ { \theta } \Big ( S ^ { ( v ) } , M _ { t } ^ { ( v ) } , r _ { t } ^ { ( v ) } , a _ { t } ^ { ( v ) } ; \mathcal { C } _ { t } ^ { ( v ) } \Big ) . } \end{array}\tag{19}
$$

The same candidate robot action is transformed into each camera coordinate system using its camera pose before prediction; there is no cross-view attention or joint memory bank inside TMP. CEM then proceeds in two stages. First, both fixed views evaluate every pose candidate, and the mean of their goal losses updates the CEM mean and variance. Second, the wrist-view search is initialized with this updated distribution and continues optimizing the pose using the wrist-view goal loss. The best candidate from this refinement is executed; the gripper search is likewise evaluated from the wrist view. Thus the fixed cameras provide a stable global search direction, while the moving wrist camera refines the action from its local view. Robot planning uses the standard UOT ground cost. Within each CEM stage, all candidates use the same sampled context transitions for the corresponding view.

Correlation distance. For the correlations in Tab. 4, EE–goal distance at step t is

$$
d _ { t + 1 } ^ { \mathrm { E E } } = \lVert \boldsymbol { e } _ { t + 1 } - \boldsymbol { e } _ { g } \rVert _ { 2 } ,\tag{20}
$$

where $e _ { t + 1 }$ is the realized next 7-DoF state and $e _ { g }$ is the active visual-goal stage’s target state; each contains the 6-D EE pose and gripper opening. This evaluation metric is distinct from the 3-D positional threshold used only for baseline guidance.

Predicted loss is the selected candidate’s goal loss before executing action ${ { a } _ { t } } ,$ while observed loss is recomputed from the observation immediately after executing that action. Thus, predicted loss, observed loss, and $d _ { t + 1 } ^ { \mathrm { E E } }$ refer to the same transition. To prevent task scale and trajectory length from dominating the aggregate, we standardize each variable within every task, view, and method, divide the distance range into 12 equal-width bins, and evenly subsample overrepresented bins. We compute each statistic per task and report the macro average across tasks.

Tab. C5 reports all measured statistics for three comparisons: observed loss against physical distance, predicted loss against physical distance, and predicted loss against observed loss. Kendall’s τ and Spearman’s $\rho$ measure monotonic agreement. The standardized least-squares slope $\beta$ measures linear sensitivity, and $R ^ { 2 }$ measures linear goodness of fit. Slot-JEPA gives the strongest agreement for every comparison and camera view under all four statistics.

Table C5: Complete correlation analysis. Values are task-wise statistics macro-averaged after withintask standardization and distance-stratified sampling.  
(a) Kendall’s τ
<table><tr><td rowspan="2">Method</td><td colspan="2">Obs. loss vs. EE-goal dist. Pred. loss vs. EE-goal dist. Pred. loss vs. Obs. loss</td><td colspan="2"></td><td colspan="2"></td><td colspan="3"></td></tr><tr><td></td><td>Ext. 1 Ext. 2</td><td>Wrist</td><td>Ext. 1 Ext. 2</td><td></td><td>Wrist</td><td>Ext. 1 Ext. 2</td><td></td><td>Wrist</td></tr><tr><td>V-JEPA 2.1</td><td>0.299</td><td>0.326</td><td>0.397</td><td>0.386</td><td>0.345</td><td>0.248</td><td>0.391</td><td>0.481</td><td>0.419</td></tr><tr><td>V-JEPA 2.1 + TMP</td><td>0.167</td><td>0.075</td><td>0.538</td><td>0.277</td><td>0.266</td><td>0.356</td><td>0.322</td><td>0.304</td><td>0.599</td></tr><tr><td>Slot-JEPA</td><td>0.692</td><td>0.804</td><td>0.720</td><td>0.551</td><td>0.671</td><td>0.706</td><td>0.527</td><td>0.706</td><td>0.758</td></tr><tr><td colspan="10">(b) Spearman&#x27;s ρ</td></tr><tr><td rowspan="2">Method</td><td>Obs. loss vs. EE-goal dist. Pred. loss vs. EE-goal dist. Pred. loss vs. Obs. loss</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ext. 1 Ext. 2</td><td></td><td>Wrist</td><td></td><td>Ext. 1 Ext. 2</td><td>Wrist</td><td>Ext. 1 Ext. 2</td><td></td><td>Wrist</td></tr><tr><td>V-JEPA 2.1 V-JEPA 2.1 + TMP</td><td>0.350</td><td>0.407</td><td>0.533</td><td>0.503</td><td>0.453</td><td>0.297</td><td>0.511</td><td>0.609</td><td>0.549</td></tr><tr><td>Slot-JEPA</td><td>0.154</td><td>0.103</td><td>0.652</td><td>0.327</td><td>0.319</td><td>0.454</td><td>0.409</td><td>0.421</td><td>0.720</td></tr><tr><td></td><td>0.792</td><td>0.891</td><td>0.807</td><td>0.677</td><td>0.788</td><td>0.814</td><td>0.668</td><td>0.825</td><td>0.865</td></tr><tr><td colspan="10">(c) Standardized slope β</td></tr><tr><td rowspan="2">Method</td><td>Obs. loss vs. EE-goal dist. Pred. loss vs. EE-goal dist. Pred. loss vs. Obs. loss</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Ext. 1 Ext. 2</td><td>Wrist</td><td></td><td>Ext. 1 Ext. 2</td><td>Wrist</td><td>Ext. 1 Ext. 2</td><td></td><td>Wrist</td></tr><tr><td>V-JEPA 2.1 V-JEPA 2.1 + TMP</td><td>0.423</td><td>0.391</td><td>0.594</td><td>0.549</td><td>0.519</td><td>0.391</td><td>0.601</td><td>0.610</td><td>0.504</td></tr><tr><td></td><td>0.144</td><td>0.106</td><td>0.699</td><td>0.388</td><td>0.360</td><td>0.544</td><td>0.411</td><td>0.475</td><td>0.745</td></tr><tr><td>Slot-JEPA</td><td>0.865</td><td>0.940</td><td>0.874</td><td>0.706</td><td>0.848</td><td>0.882</td><td>0.730</td><td>0.887</td><td>0.922</td></tr><tr><td colspan="10">(d) Coefficient of determination R²</td></tr><tr><td rowspan="2">Method</td><td>Obs. loss vs. EE-goal dist. Pred. loss vs. EE-goal dist.</td><td></td><td></td><td></td><td></td><td></td><td>Pred. loss vs. Obs. loss</td><td></td><td></td></tr><tr><td>Ext. 1 Ext. 2</td><td></td><td>Wrist</td><td></td><td>Ext. 1 Ext. 2</td><td>Wrist</td><td>Ext. 1 Ext. 2</td><td></td><td>Wrist</td></tr><tr><td>V-JEPA 2.1</td><td>0.402</td><td>0.362</td><td>0.391</td><td>0.394</td><td>0.352</td><td>0.237</td><td>0.424</td><td>0.477</td><td>0.312</td></tr><tr><td>V-JEPA 2.1 + TMP</td><td>0.313</td><td>0.262</td><td>0.530</td><td>0.290</td><td>0.311</td><td>0.404</td><td>0.312</td><td>0.490</td><td>0.602</td></tr><tr><td>Slot-JEPA</td><td>0.792</td><td>0.890</td><td>0.793</td><td>0.531</td><td>0.740</td><td>0.793</td><td>0.561</td><td>0.797</td><td>0.859</td></tr></table>

Action execution. The 7-D planning command consists of a 6-D end-effector target pose and a scalar gripper-opening command. In absolute-pose mode, the arm controller computes Cartesian pose error, maps it through inverse kinematics to desired joint positions, and applies joint-space PD feedback with Jacobian-weighted Cartesian gains. A separate continuous gripper controller applies the opening command. We allow at most 100 MPC steps per planning attempt.

## B.5 ADDITIONAL VISUALIZATIONS

UOT objective. Fig. 12 shows how the goal loss varies over sampled states. Low-loss regions follow feasible progress toward the goal while the obstacle-aware ground cost assigns less favorable energy to visually nearby but obstructed states, yielding a planning signal that better reflects scene geometry.

Predicted masks. Fig. 13 compares action-conditioned slot-mask predictions with the RGB observations obtained after execution. The predicted masks track the task-relevant regions across environments and time. Because the future is represented as separate slot masks, the prediction is also more interpretable: one can inspect which object or region is expected to move and where, rather than only comparing an opaque dense latent.

Slot decomposition. Figures 14–17 visualize individual slots as colored masks over the input RGB frames. Across Kinetics-400 pretraining videos and DROID downstream-training clips, different slots emphasize distinct objects or spatial regions and remain recognizable over time. This decomposition exposes which parts of a scene contribute to the latent state instead of collapsing them into one dense feature map.

![](images/6c42998b292d235fe63dae1fdcdd14e6788a1c8bc955d2ef4a4fa1b5a21906ec.jpg)  
Figure 12: Additional goal-loss landscapes in MazeR and TwoRoomR. Each sampled state is rendered independently; the panels visualize energy geometry rather than executed rollouts. Stars mark goals, and lighter colors indicate lower loss.

Overall, these visualizations indicate that slot decomposition provides a spatially structured and inspectable state, while UOT and TMP turn that state into goal-directed costs and coherent futuremask predictions.

![](images/d66a19c3bef4ff90e00c7de3014f423caa4cbc46523374eb19da9c02528f836d.jpg)

![](images/9b5da189ca7658a766a7d3af9c332065e3cf0ffddad0a9e59ea5e225d169c070.jpg)

![](images/7791f16b2bd01ecad53d88f77f9c8487171b89cb5e59b5580daea8e744117dfe.jpg)

![](images/b140881e475bd1df1be95d42e80fba9920fe522a77401194b147875ca75b9493.jpg)  
Figure 13: Predicted slot masks along successful TwoRoomR, MazeR, PushObj, and Place planning rollouts (top left, top right, bottom left, and bottom right, respectively). Each column pairs the RGB observation obtained after action execution with masks predicted for the same time step. The planar panels show mask heatmaps, whereas the Place panel overlays sharpened predictions on the executed wrist-view RGB.

![](images/76c8b7e10e9c04491e1815cbe05d4246e4f4f65e6aae1d0004eedfaf2c27b600.jpg)  
Figure 14: Slot decomposition on Kinetics-400 pretraining example. The first row contains the RGB frames, and each subsequent row overlays one selected soft slot mask.

![](images/451840b903b6306490abad057678666b02485d5d6b75db2b2f1fbe28a32187b8.jpg)  
Figure 15: Slot decomposition on Kinetics-400 pretraining example. The first row contains the RGB frames, and each subsequent row overlays one selected soft slot mask.

![](images/50955bf82bcece23aac1d529777cd956285cecdc06ff1baa5d8234c6c94d1b48.jpg)  
Figure 16: Slot decomposition on DROID downstream-training example. The first row contains the RGB frames, and each subsequent row overlays one selected soft slot mask.

![](images/69cdd281da62dd2ca8ae9f7c367c4236cea449d778e8f5a1ba40c2a461718812.jpg)  
Figure 17: Slot decomposition on DROID downstream-training example. The first row contains the RGB frames, and each subsequent row overlays one selected soft slot mask.