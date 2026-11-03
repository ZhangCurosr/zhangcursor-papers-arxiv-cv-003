# TailorRig: Controllable and Structured Auxiliary Skeleton Generation for Garment-Aware Rigging

YAJING CHEN<sup>∗</sup>, LIGHTSPEED, Tencent, China XIN CHEN<sup>∗</sup>, Xi’an Jiaotong University, China SAIDAN LU<sup>∗</sup>, LIGHTSPEED, Tencent, China YEQI YUAN, Peking University, China WEIKAI CHEN, LIGHTSPEED, Tencent, United States CHANGYU HU, Tsinghua University, China CHENG ZENG, Tsinghua University, China XIN WANG, LIGHTSPEED, Tencent, China

![](images/102b4311caed021dad274b48db7abb14fa1156a8ec606da895a8646977871074.jpg)  
Fig. 1. TailorRig turns artist controls into animation-ready garment-aware rigs. TailorRig accepts any body template and any subset of auxiliary category labels (left). Diferent label choices yield distinct full rigs (middle, with chosen labels marked under each result), all driving stable animation (right). 3D characters courtesy of Adobe Mixamo [Adobe Inc. 2021].

Humanoid rigging for modern characters with complex garments remains challenging due to the need for dense auxiliary skeletons and precise skinning control. Existing automatic methods primarily recover rigs from geometry, providing limited artist control over auxiliary articulation and often lacking explicit semantic or kinematic-chain structure for dense garment bones, leading to instability and weight bleeding. We present TailorRig, a framework that reformulates garment-aware humanoid rigging as a structured, controllable full-rig generation problem. This formulation shifts the goal from automatic rig recovery to artist-driven rig authoring, where both body-template compatibility and auxiliary articulation are explicit design choices. TailorRig exposes two explicit axes of artist control within a uni fied framework: a user-specified body skeleton template that preserves

<sup>∗</sup>Yajing Chen, Xin Chen, and Saidan Lu contributed equally to this research.

body-rig compatibility, and artist-provided semantic labels that drive an autoregressive generator to synthesize requested category-specific auxiliary bone chains. To robustly assign skinning weights under dense skeletal configurations, we further introduce a hierarchical chain-to-bone formulation: vertices are first associated to a kinematic chain, and bone weights are then refined within that chain, reducing weight bleeding between spatially adjacent but semantically unrelated structures. We evaluate TailorRig on a diverse set of characters and introduce Human-Mix, a large-scale dataset of over 38,000 humanoid models curated for garment-aware rigging. Experiments show that TailorRig improves skeleton accuracy, deformation stability, controllability, and practical usability over existing methods.

Additional Key Words and Phrases: Humanoid Rigging, Auxiliary Bone Generation, Garment-Aware Rigging

## 1 Introduction

Humanoid rigging is essential for articulated deformation in animation pipelines, including games, films and virtual production [Parent 2012]. The recent surge in generative 3D assets [Hong et al. 2023; Lai et al. 2025; Li et al. 2023; Liu et al. 2023; Poole et al. 2022; Siddiqui et al. 2024; Zhang et al. 2024] requires scalable and automated rigging solutions. While significant progress has been made in automating body rigging, modern characters increasingly feature complex garments, accessories, and layered structures, such as skirts, capes, coats, and long hair, that require auxiliary skeletal structures to deform naturally. These auxiliary bones are no longer optional embellishments, but essential components of production-oriented character rigs.

A large body of prior work has sought to automate humanoid rigging. Template-based approaches [Baran and Popović 2007; Chu et al. 2025; Guo et al. 2025b; Ma and Zhang 2023] leverage predefined skeletons to ensure compatibility with existing animation libraries, while template-free methods [Lin et al. 2021; Song et al. 2025a,b; Xu et al. 2020; Zhang et al. 2025] infer skeletal structures directly from geometry to improve flexibility and generalization. However, when applied to characters with complex garments and dense auxiliary joints, these methods exhibit two recurring failures. Auxiliary bones are treated implicitly—either omitted altogether or generated indiscriminately—leaving artists little control over their presence, category, or placement. Moreover, dense auxiliary skeletons exacerbate skinning ambiguity, causing weight bleeding across unrelated body and garment parts (e.g., a skirt erroneously following arm motion). These failures share a common root: prior methods formulate rigging as geometry-to-rig recovery, while production demands artist-driven rig authoring. We therefore reformulate garment-aware humanoid rigging as a structured, controllable full-rig generation problem. By controllable, we mean that artists can specify both the body rig template and the auxiliary bone categories; by structured, we mean that rig generation and skinning are guided by semantic grounding and kinematic-chain hierarchy.

Given an input mesh, an arbitrary user-defined body skeleton template, and optional semantic labels specifying auxiliary categories, TailorRig produces an animation-ready rig that preserves body skeleton compatibility while supporting editable auxiliary structures. TailorRig consists of two core components (Fig. 2). (1) Controllable body-to-auxiliary skeleton generation. A semantically-guided alignment module first maps an arbitrary user-defined body skeleton to the input character while preserving its kinematic topology, enabling immediate reuse of existing animation assets. Conditioned on artist-provided semantic labels (e.g. skirt), a label-driven autoregressive generator then synthesizes category-specific auxiliary bone chains only when desired, providing the explicit control required in professional workflows. (2) Hierarchical chain-to-bone skinning. When dozens of garment bones from diferent chains overlap in close spatial proximity, a vertex cannot be correctly skinned by reasoning about individual bones in isolation. We therefore first predict coarse vertex-to-chain associations – identifying which kinematic branch (e.g., left-arm chain vs. skirt chain) each vertex belongs to – and then refine per-bone weights within the assigned branch. This two-level decomposition suppresses weight bleeding across unrelated body and garment parts. Both components use a hierarchical bone encoder with stage-specific parameters, providing consistent kinematic priors across the pipeline.

To support training and evaluation, we curate Human-Mix, a large-scale dataset comprising over 38, 000 humanoid models with semantic annotations for garment-aware rigging. We will release Human-Mix in a license-compliant form, sharing the in-house VRoid assets and annotations while providing annotations and source identifiers for retained third-party characters (Sec. 4.1).

In summary, our main contributions are:

• A controllable rigging formulation that supports explicit body template control and auxiliary label control, enabling artists to govern both body and garment skeletal structures.

• A structured full-rig generation framework that injects semantic grounding and kinematic-chain hierarchy into body alignment, auxiliary-bone generation, and skinning.

• A hierarchical chain-to-bone skinning method that reduces weight bleeding through hierarchical structural reasoning, where vertices are first localized to the correct kinematic branch before competing among nearby bones.

• Human-Mix, a large-scale dataset of 38K+ humanoid models with semantic annotations for garment-aware rigging.

We validate these claims through dedicated controllability experiments, structural ablations targeting both semantic and geometric priors, weight bleeding analyses, and production-oriented evaluations on complex real-world assets.

## 2 Related Work

Existing methods generally fall into two categories: template-based and template-free approaches.

## 2.1 Template-based Rigging.

Template-based methods leverage a predefined skeletal topology to provide strong structural priors. Pinocchio [Baran and Popović 2007] pioneered this line by formulating joint placement as discrete geometric embedding on a medial surface, while later optimization approaches extended it to inverse-kinematics fitting [Jacobson et al. 2011; James and Twigg 2005]; both families, however, typically require watertight meshes and ofer limited adaptability. Deeplearning methods then took over: TARig [Ma and Zhang 2023] introduces cross-attention maps that regress vertex-to-joint correspondences for arbitrary mesh resolution, and MIA [Guo et al. 2025b] proposes a coarse-to-fine estimation strategy that progressively refines joint positions through multi-stage mesh-aware prediction. Other concurrent works explore blendshape-coupled alignment [Li et al. 2021], image-lifted keypoints [Chu et al. 2025], and voxel-based prediction [Lefèvre et al. 2025]. Despite their robustness, these models are typically confined to a single skeletal standard, and their support for auxiliary bones [Guo et al. 2025b; Ma and Zhang 2023] remains limited by fixed topologies, failing to accommodate diverse clothing styles.

## 2.2 Template-free Rigging.

Template-free methods infer arbitrary topologies directly from geometry. Classical approaches extract skeletons via medial-axis analysis [Amenta and Bern 1998; Tagliasacchi et al. 2009], while RigNet [Xu et al. 2020] pioneered learning-based rigging by predicting joints and connectivity with a graph neural network, followed by MoRig [Xu et al. 2022] and HeterSkinNet [Pan et al. 2021] for motion-aware and heterogeneous settings; Anymate [Deng et al. 2025] scales this pre dictive paradigm with a large rigged-asset corpus and a three-stage joint–connectivity–skinning framework. Inspired by autoregressive 3D generation [Ibing et al. 2023; Mittal et al. 2022; Nash et al. 2020; Yan et al. 2022], recent methods reformulate skeleton synthesis as sequence modeling: RigAnything [Liu et al. 2025b] applies difusionguided BFS prediction, UniRig [Zhang et al. 2025] autoregressively decodes hierarchical skeletons from discretized spatial tokens, and MagicArticulate [Song et al. 2025b], Puppeteer [Song et al. 2025a], and Auto-Connect [Guo et al. 2025a] explore related grid-token formulations. Though flexible, these generative approaches introduce topological stochasticity that often violates studio standards, and their implicit auxiliary generation [Zhang et al. 2025] lacks explicit semantic conditioning for artist-driven control. More fundamentally, existing methods provide limited explicit modeling of garment semantics and kinematic-chain structure, which becomes critical when dense auxiliary bones must be generated and skinned under artist control, or for intersection-free garment retargeting [Huang et al. 2025].

## 2.3 Automatic Skinning.

Traditional skinning methods rely on geometric formulations such as bounded-biharmonic weights [Jacobson et al. 2011], geodesic voxel binding [Dionne and de Lasa 2013], and heat-difusion solvers [Baran and Popović 2007], which assume watertight meshes and degrade when bone influences densely overlap on garment-rich characters; example-based methods instead approximate LBS from animated mesh sequences [James and Twigg 2005; Kavan et al. 2010], which requires pre-existing animations. Learning-based approaches improve quality substantially. NeuroSkinning [Liu et al. 2019] trains a global PointNet-based model for fixed-bone weight inference, while other template-specific methods [Chu et al. 2025; Guo et al. 2025b; Ma and Zhang 2023] achieve high fidelity within their skeletal standards. For arbitrary skeletons, graph-based architectures [Pan et al. 2021; Xu et al. 2020] propagate bone influences along mesh edges, and transformer-based methods [Guo et al. 2025a; Song et al. 2025a,b; Zhang et al. 2025] model long-range vertex–bone interactions via attention; orthogonally, weight-transfer methods inpaint an existing weight set onto a new mesh [Abdrashitov et al. 2023]. Despite these advances, weight bleeding remains challenging for dense auxiliary bones, where closely packed bones create ambiguous influence boundaries. Our hierarchical chain-to-bone skinning addresses this by first localizing each vertex to the correct kinematic chain, then refining per-bone weights within that chain, efectively suppressing cross-chain bleeding.

## 3 Methods

## 3.1 Problem Formulation

We formulate garment-aware humanoid rigging as a controllable, structured full-rig generation problem. Given a mesh $\begin{array} { r l } { \boldsymbol { { M } } } & { = } \end{array}$ $( \mathcal { V } , \mathcal { E } _ { M } )$ with vertex positions V<sup>�</sup> and normals $\mathbf { V } ^ { n }$ , a user-specified body skeleton template $\tilde { \mathcal { B } } _ { \mathrm { b o d y } }$ , and a set of auxiliary category labels, TailorRig outputs a full skeleton $\boldsymbol { \mathcal { B } } = ( \mathrm { J } , \mathcal { E } _ { \mathrm { j o i n t } } )$ (body plus garmentaware auxiliary bones) and skinning weights W. The two control inputs $( \tilde { \mathcal { B } } _ { \mathrm { b o d y } } ,$ , labels) realize the controllable axis; the structured axis is carried by a shared 21-class semantic taxonomy and a hierarchical skeleton encoder (stage-specific parameters) that propagates joint–chain structure into every prediction stage (Sec. 3.2). Built on these ingredients, the pipeline addresses three production needs: template alignment for animation reuse (Sec. 3.3), label-conditioned generation for explicit auxiliary control (Sec. 3.4), and chain-to-bone skinning against cross-branch weight bleeding (Sec. 3.5).

3.2 Hierarchical Skeleton Encoder and Common Modules The following modules are used across skeleton and skinning prediction. Each is defined by a common architectural design but instantiated with stage-specific parameters: mesh, label, and bone encoders, plus a semantic classifier that supervises a shared pervertex taxonomy during training (see Fig. 2).

Mesh and label encoders. We employ a PartField-style encoder [Liu et al. 2025a] to build a continuous 3D feature field from the input mesh. Global mesh tokens $\mathbf { F } _ { \mathcal { M } } \in \mathbb { R } ^ { N _ { M } \times D }$ are obtained by querying this field at $N _ { \mathrm { \it M } } = 5 1 2$ farthest-point-sampled locations, providing a compact shape summary used as mesh conditioning for skeleton generation; per-vertex features $\mathbf { F } _ { \mathcal { V } } \in \mathbb { R } ^ { V \times D }$ are obtained by evaluating the field at every mesh vertex, providing dense local geometry for bone encoding, semantic classification, and skinning prediction. For each label, we map its category index to a learnable embedding vector $\mathbf { f } _ { \mathrm { l a b e l } } \in \mathbb { R } ^ { D }$ , which is later used to query meshand body-aware features for label-conditioned auxiliary skeleton generation.

Hierarchical skeleton encoder. This module produces the per-joint representation $\mathbf { F } _ { \mathcal { B } } \in \mathbb { R } ^ { J \times D }$ by encoding the skeleton at both joint and chain granularities. Each joint coordinate is embedded with Fourier features and refined with mesh context from $\mathbf { F } _ { M }$ to obtain mesh-aware joint tokens. Each maximal unbranched path in the skeleton tree defines a kinematic chain $( \mathrm { e . g . }$ , spine, left arm, skirtleft), whose constituent tokens are aggregated into a chain-level representation. Fusing the joint- and chain-level features yields $\mathbf { F } _ { \mathcal { B } }$ so each joint reflects both local geometry and chain-level structure. The same encoder architecture is instantiated at each stage—on the template body skeleton for alignment, on the aligned body skeleton for auxiliary generation, and on the complete skeleton for skinning— providing a consistent two-level structural prior throughout the pipeline.

Semantic classifier. A vertex-wise classifier predicts a distribution over 21 semantic classes (11 body kinematic-chain classes and 10 auxiliary classes) from $\mathbf { F } _ { \mathcal { V } }$ . It is instantiated at each stage with stage-specific parameters and label masks (11 body classes for alignment; all 21 classes for auxiliary generation and skinning), providing consistent semantic grounding across body alignment, auxiliary generation, and skinning.

## 3.3 Topology-Aware Body Skeleton Generation

The body skeleton template serves as TailorRig’s first explicit control signal. By accepting arbitrary artist-specified templates rather than fixed or stochastic skeletons, this module aligns each template to the target mesh while preserving kinematic topology and semantic consistency, enabling immediate reuse of existing animations.

Canonical Template Space. For an arbitrary input template, we first convert it to a standard T-pose and then normalize it in a unified geometric space to reduce diferences in pose, orientation, and global scale (see Fig. 2). After canonicalization, the template preserves its topology while being mapped into a consistent coordinate system shared with the target mesh.

![](images/3958e8432019c35c3bb3acf3857ea2148e8417c2d00463431bdb989bbc2085e3.jpg)  
Fig. 2. Overview of TailorRig. Given a mesh, a body skeleton template, and auxiliary category labels, TailorRig produces an animation-ready rig in two stages. (1) Controllable skeleton generation: a Topology-Aware body decoder (TAMB) aligns the template to the mesh, and a Label-Aware autoregressive decoder synthesizes category-specific auxiliary chains, both use the same hierarchical skeleton-encoder architecture with stage-specific parameters. (2) Hierarchical chain-to-bone skinning: vertices are first assigned to kinematic chains and then refined into per-bone weights, suppressing cross-chain weight bleeding.

Topology-Aware Body Skeleton Decoder. To generate body joint positions that are both geometrically accurate and topologically con sistent, we propose a Topology-Aware Mesh-Bone (TAMB) fusion decoder. We apply � layers of topology-aware mesh–bone fusion blocks with pre-norm residual connections. At each layer �, we first perform cross-attention from bones to mesh and then refine via EdgeConv [Wang et al. 2019] over the bone connectivity graph:

$$
\mathbf { F } _ { \mathrm { a t t n } } ^ { ( l ) } = \mathbf { F } _ { \mathcal { B } } ^ { ( l - 1 ) } + \mathrm { A t t n } \big ( \mathrm { L N } ( \mathbf { F } _ { \mathcal { B } } ^ { ( l - 1 ) } ) , \mathbf { F } _ { \mathcal { M } } , \mathbf { F } _ { \mathcal { M } } \big ) ,\tag{1}
$$

$$
\mathbf { F } _ { \mathcal { B } } ^ { ( l ) } = \mathbf { F } _ { \mathrm { a t t n } } ^ { ( l ) } + \mathrm { M L P } \big ( \mathrm { E d g e C o n v } \big ( \mathrm { L N } ( \mathbf { F } _ { \mathrm { a t t n } } ^ { ( l ) } ) , \mathcal { E } _ { \mathrm { b o d y } } \big ) \big ) .\tag{2}
$$

Here, Atn(Q, K, V) denotes multi-head cross attention and LN(·) denotes layer normalization. The decoder is initialized with the template body skeleton features, i.e., $\mathbf { F } _ { \mathcal { B } } ^ { ( 0 ) } = \mathbf { F } _ { \tilde { \mathcal { B } } _ { \mathrm { b o d y } } }$ . After � layers, the fused bone features are decoded into aligned joint positions via an MLP head.

Semantic and Geometric Supervision. The overall body alignment loss combines position, projection, and semantic terms: $\mathcal { L } _ { \mathrm { a l i g n } } =$ $\mathcal { L } _ { \mathrm { p o s } } + \mathcal { L } _ { \mathrm { p r o j } } + \mathcal { L } _ { \mathrm { s e m } } . \mathcal { L } _ { \mathrm { p o s } }$ is a standard $L _ { 1 }$ regression loss on joint positions. $\mathcal { L } _ { \mathrm { s e m } }$ is a vertex-wise classification loss (11 body kinematicchain classes) that enforces mesh–bone semantic correspondence, ensuring each predicted joint is grounded to its correct body region. The key geometric term is the bone projection loss $\mathcal { L } _ { \mathrm { p r o j } } ,$ designed to suppress anatomically implausible transverse drift. For joint � with parent $p ( i )$ , we project <sup>ˆ</sup>j<sub>�</sub> onto the ground-truth bone segment $[ \mathbf { j } _ { P ( i ) } , \mathbf { j } _ { i } ]$ and penalize the squared distance to the closest point: $\begin{array} { r } { \mathcal { L } _ { \mathrm { p r o j } } = \frac { 1 } { J } \sum _ { i = 1 } ^ { J } \left\| \hat { \bf j } _ { i } - { \bf p } _ { \mathrm { c l o s e } , i } \right\| _ { 2 } ^ { 2 } , } \end{array}$ where $\mathbf { p } _ { \mathrm { c l o s e } , i }$ is the closest point on the GT bone segment. This lets the predicted joint slide along the bone direction while penalizing lateral deviation, preventing drift of its kinematic axis.

## 3.4 Label-Aware Auxiliary Skeleton Generation

Auxiliary bone labels are the second explicit control signal in TailorRig. Given a category label (e.g., “skirt”), the model generates auxiliary bones for that category, anchored to the mesh region. We model this as an autoregressive generation problem conditioned on mesh and body features and a category label.

Label-to-Geometry Semantic Grounding. Rather than using the category label as a static global condition, we ground it in the current mesh and body-skeleton context through a multi-step attention mechanism. We first apply one round of reciprocal cross-attention with pre-norm residual connections between global mesh tokens $\mathbf { F } _ { M }$ and body-skeleton features $\mathbf { F } _ { \mathcal { B } _ { \mathrm { b o d y } } }$ to obtain mutually aligned features $\mathbf { F } _ { M } ^ { \prime }$ and $\mathbf { F } _ { \mathcal { B } _ { \mathrm { b o d v } } } ^ { \prime }$ . We then use the label embedding $\mathbf { f } _ { \mathrm { l a b e l } }$ as a query to probe both aligned feature sets via cross-attention, producing two geometry-aware label tokens—one encoding where the category manifests on the mesh surface, the other on the body skeleton. These tokens are concatenated with the aligned features to form the conditioning prefix $\mathbf { F } _ { \mathrm { p r e f i x } } \in \mathbb { R } ^ { ( N _ { M } + J _ { \mathrm { b o d y } } + 2 ) \times D }$ for the autoregressive decoder.

Structured Autoregressive Decoding. Following [Zhang et al. 2025], we serialize each auxiliary skeleton tree into a token sequence via depth-first traversal, with joint coordinates quantized into discrete bins. We adopt OPT-350M [Zhang et al. 2022] as the decoder backbone. The conditioning prefix $\mathbf { F } _ { \mathrm { p r e f i x } }$ is treated as token embeddings and prepended to the skeleton token sequence, allowing the decoder to attend to mesh and body context through its standard causal selfattention. The model then autoregressively predicts the auxiliary bone sequence.

Loss Functions. The auxiliary generation module is trained with a next-token prediction loss over the skeleton token sequence, supplemented by the same semantic classification loss $\mathcal { L } _ { \mathrm { s e m } }$ (all 21 classes) to enhance semantic consistency and label-aware grounding.

## 3.5 Hierarchical Chain-to-Bone Skinning Generation

We implement the skinning stage as a chain-to-bone decomposition: vertices are first softly assigned to kinematic chains, and per-bone weights are then refined under this chain assignment, restricting bone-level competition within each chain.

Chain-level representation. We define a kinematic chain as a maximal unbranched path of consecutive bones between two branching or terminal joints in the skeleton tree. The full skeleton (body + aux iliary) is automatically partitioned into � such chains (e.g., spine, left arm, skirt-left), where � varies per character depending on the skeleton topology. Each chain representation is obtained by meanpooling constituent bone features from the hierarchical skeleton encoder, yielding chain features $\mathbf { F } _ { C } \in \mathbb { R } ^ { K \times D }$

Hierarchical feature refinement. Feature interaction proceeds in a coarse-to-fine manner via reciprocal cross-attention. In the coarse stage, vertex features and chain features alternately serve as query and key/value for 4 rounds of cross-attention, yielding chain-aware vertex features $\mathbf { F } _ { \mathcal { V } } ^ { c }$ and vertex-aware chain features $\mathbf { F } _ { C } ^ { \prime } .$ In the fine stage, vertex features and bone features undergo the same reciprocal interaction for 8 rounds, producing refined bone-aware vertex features $\mathbf { F } _ { \mathcal { V } } ^ { b }$ and vertex-aware bone features $\mathbf { F } _ { \mathcal { B } } ^ { \prime } .$ . The coarse stage requires fewer rounds as chain-level associations are simpler than bone-level ones.

Chain-to-bone weight prediction. The chain weight decoder predicts coarse chain weights $\mathbf { W } _ { C } \in \mathbb { R } ^ { V \times K }$ by combining two cues for each vertex–chain pair: a learned similarity from cross-attention between vertex and chain features, and a geometric prior derived from the geodesic distance between the vertex and the nearest bone in each chain. The bone weight decoder follows a similar design but additionally conditions on the predicted chain weight: each bone’s parent-chain weight is concatenated into its feature as a structural prior, which is combined with the bone-level similarity and distance cues to predict the final per-bone weights ${ \bf W } \in \mathbb { R } ^ { V \times \dot { B } }$ . Because the parent-chain weight acts as a continuous structural prior, bonelevel competition is efectively concentrated within each vertex’s associated chains, suppressing cross-chain weight bleeding.

Loss Functions. The overall skinning objective combines four terms: $\begin{array} { r } { \mathcal { L } _ { \mathrm { s k i n } \_ { \mathrm { t o t a l } } } = \lambda _ { \mathrm { s k i n } } \mathcal { L } _ { \mathrm { s k i n } } + \lambda _ { \mathrm { d e f } } \mathcal { L } _ { \mathrm { d e f } } + \lambda _ { \mathrm { c h a i n } } \mathcal { L } _ { \mathrm { c h a i n } } + \lambda _ { \mathrm { s e m } } \mathcal { L } _ { \mathrm { s e m } } , } \end{array}$ where $\mathcal { L } _ { s \mathrm { k i n } }$ is the KL divergence between ground-truth and predicted per-bone weight distributions, $\mathcal { L } _ { \mathrm { d e f } }$ measures vertex-level deformation error under �=3 random training LBS poses, $\mathcal { L } _ { \mathrm { c h a i n } }$ supervises the coarse chain-level weight distribution (chain weights are obtained by summing per-bone weights within each chain), and $\mathcal { L } _ { \mathrm { s e m } }$ is the same vertex-wise semantic classification loss used in body alignment (Sec. 3.3), here applied with all 21 classes to suppress cross-part bleeding.

## 4 Experiments

We evaluate TailorRig on rigging quality (Sec. 4.2), controllability (Sec. 4.3), structural ablations (Sec. 4.4), and in-the-wild production readiness (Sec. 4.5).

## 4.1 Experimental Setup

Dataset. No existing dataset provides auxiliary bone annotations for training and evaluating TailorRig. We therefore construct Human-Mix, aggregating 38,088 high-quality rigged humanoid characters from three public sources and an in-house VRoid collection (Table 1) with diverse skeletal structures and semantic annotations, spanning a broad spectrum of auxiliary bone configurations. All characters undergo rigorous multi-stage curation and are annotated with our 21-class bone taxonomy (11 body, 10 auxiliary) co-designed with professional artists, of which 7,824 contain at least one auxiliary class (Fig. 3). All Mixamo-derived assets are excluded from training via source-independent geometric de-duplication (Sec. 4.1); they appear only in qualitative demonstrations.

Data provenance and release. We shortlist cross-source duplicate candidates by vertex and bone counts, then manually confirm matches with mean post-ICP vertex error below 0.001 using skeleton topology and joint distances as additional checks. The same source-independent procedure is applied against a Mixamo reference library; matched assets are excluded before training. The 3,404 VRoid characters were created in-house using VRoid Studio under its content license. Upon publication, we plan to release these rigged assets with our annotations. For retained characters from Articulation-XL-v2, HumanRig, and ModelResource, we plan to release only our annotations and original source identifiers, allowing users to obtain the assets from their original providers. Mixamo assets are used for qualitative figures and are not included in the training data or dataset release.

![](images/1fa4a8666e65927fff4c913514e2e913eba49013652b14566d6102fa2fb8fdb1.jpg)  
Fig. 3. Auxiliary-class distribution in Human-Mix. Bars report per-class counts across the 10 auxiliary classes among the 7,824 characters containing at least one auxiliary bone; the Body group (11 body kinematic-chain classes) covers all 38,088 characters and is reported in the legend.

Baselines. For their applicable tasks, we retrain UniRig, MIA, RigNet, and NBS on the same Human-Mix split using released pipelines, changing only the data loader. Methods without reproducible training pipelines use oficial checkpoints, while non-learning baselines require no retraining.

Metrics. Skeleton geometry. We measure skeletal fidelity via bidirectional Chamfer Distance at three granularities: CD-J2J (jointto-joint), CD-J2B (joint-to-bone segment), and CD-B2B (bone-tobone stick). For auxiliary bones, a predicted bone inherits the category of a GT bone when it lies inside that bone’s capsule—a cylinder of radius 0.15 around the joint-to-joint segment—and the three distances are computed within each category. Skinning. We report the $L _ { 1 }$ error of the skinning matrix and Deformation $L _ { 1 } ~ ( \mathcal { E } _ { \mathrm { d e f } } )$ , which evaluates vertex-level geometric distortion under LBS with $N { = } 5$ random test poses.

Table 1. Source composition and curation statistics of Human-Mix. #Raw: characters collected before curation; #Curated: characters retained after curation; #Aux.: characters containing at least one auxiliary bone category.
<table><tr><td>Source</td><td>#Raw</td><td>#Curated</td><td>#Aux.</td></tr><tr><td>Articulation-XL-v2 [Song et al. 2025b]</td><td>48,635</td><td>22,533</td><td>4,060</td></tr><tr><td>HumanRig [Chu et al. 2025]†</td><td>11,434</td><td>11,434</td><td>0</td></tr><tr><td>VRoid* [Isozaki et al. 2021]</td><td>3,404</td><td>3,404</td><td>3,337</td></tr><tr><td>ModelResource [Xu et al. 2020]</td><td>2,703</td><td>717</td><td>427</td></tr><tr><td>Total (Human-Mix)</td><td>_+</td><td>38,088</td><td>7,824</td></tr></table>

<sup>∗</sup>Created in-house using VRoid Studio; used under its content license.  
<sup>†</sup>HumanRig is a pre-curated dataset; we adopt it in full without additional filtering.  
<sup>‡</sup>Raw totals are omitted because cross-source overlap prevents simple summation.  
We will release the in-house VRoid subset as rigged FBX assets with annotations; for retained third-party characters we release our annotations with original source identifiers (Sec. 4.1).

Implementation details. Each mesh is resampled to 16,384 points and normalized to a unit ball; we use a 90%:5%:5% split with random augmentation. The feature dimension is �=512; the skeleton and skinning modules are trained independently with AdamW $( \mathrm { L } \mathrm { R } { = } 1 0 ^ { - 5 }$ weight decay 0.05).

## 4.2 Comparison Studies

Table 2 reports quantitative results and Figs. 7–9 provide qualitative comparisons. Our method achieves the best score on every metric.

Skeleton quality. For body skeletons, TailorRig achieves J2J 0.023, improving on the best template-based baseline (TARig, 0.061) and reducing UniRig’s 0.040 by 42.5%; UniRig is template-free and thus serves as a reference rather than a same-input baseline. For auxil iary bones, the gain is larger: J2J decreases from UniRig’s 0.191 to 0.068, indicating that label-conditioned generation better localizes garment-aware chains. Fig. 7 further illustrates that existing methods often confuse garment-related auxiliary bones with the body skeleton. For example, Pinocchio inserts body bones into the sleeve region in the left example and the hair region in the middle example, while learning-based baselines produce missing or incorrectly attached auxiliary chains. TailorRig avoids these failures by using the specified labels, yielding cleaner body–auxiliary separation and topology closer to the ground truth.

Skinning quality. For skinning, we evaluate all applicable methods in a controlled setting using the same ground-truth skeleton, thereby isolating skinning-weight prediction from skeleton errors. TailorRig achieves the best skinning scores in Table 2, with scaled $L _ { 1 }$ / Def- ${ \boldsymbol { \cdot } } L _ { 1 }$ errors of 7.70 / 0.60. As shown in Fig. 8, this advantage is most visible under auxiliary-bone rotations: baselines exhibit stronger body–auxiliary coupling and local artifacts, such as braid–body adhesion and skirt coupling in the highlighted regions, whereas TailorRig keeps the deformation closer to the ground truth.

Full-rig animation. We further evaluate the complete rigs under the same input motion. As shown in Fig. 9, prior methods exhibit garment–body coupling and interpenetration artifacts, while Tailor-Rig’s dedicated auxiliary structures and cleaner skinning produce more stable frame-wise deformation. The sampled frames in Fig. 9 illustrate the resulting garment motion.

## 4.3 Controllability Evaluation

Body-template control. Fig. 4 shows that diferent body templates produce distinct rigs and hand behaviors on the same mesh, confirming template-level structural control.

Auxiliary-label control. Fig. 5 shows that diferent labels activate only the corresponding auxiliary bones and produce distinct animated behaviors. Quantitatively, we request one GT auxiliary category per test character and report two metrics: requested-bone recall, the fraction of requested-category GT bones matched by same-category predictions under the capsule rule of Sec. 4.1; and oflabel rate, the fraction of predicted auxiliary bones whose category falls outside the request. Since UniRig emits no labels, each predicted bone inherits the category of its containing GT capsule (Sec. 4.1)—a mapping favorable to UniRig, which is judged on placement alone without risk of category mislabeling. TailorRig achieves 92.6% recall with only 7.9% of-label bones, versus UniRig’s 42.1%/25.1%, confirming that labels govern which structures are generated.

![](images/3eae56a0577652178bdf6fa11cc23c1aace3b734c3c09b270a7719c4d9f786ac.jpg)  
Fig. 4. Body-template control. Diferent body templates control diferent rigs: a template without finger joints (left) yields a finger-less hand rig, while one with articulated fingers (right) yields an articulated one—producing distinct hand behaviors under the same motion.

Table 2. Quantitative comparison across all rigging tasks in TailorRig. Values in Skinning Weight are scaled by $1 0 ^ { 3 } .$
<table><tr><td rowspan="2">Method</td><td colspan="3">Body Skeleton Alignment</td><td colspan="3">Auxiliary Bone Generation</td><td colspan="2">Skinning Weight</td></tr><tr><td>J2J↓</td><td>J2B↓</td><td>B2B↓</td><td>J2J↓</td><td>J2B↓</td><td>B2B↓</td><td> $L _ { 1 } \downarrow$ </td><td> $\mathrm { D e f } { - } L _ { 1 } \ \downarrow$ </td></tr><tr><td>MIA [Guo et al. 2025b]</td><td>0.065</td><td>0.053</td><td>0.031</td><td>1</td><td>一</td><td>一</td><td>14.43</td><td>1.56</td></tr><tr><td>TARig [Ma and Zhang 2023]</td><td>0.061</td><td>0.045</td><td>0.034</td><td>一</td><td>1</td><td>一</td><td>28.25</td><td>2.68</td></tr><tr><td>NBS [Li et al. 2021]</td><td>0.141</td><td>0.121</td><td>0.111</td><td></td><td>1</td><td></td><td>62.86</td><td>28.52</td></tr><tr><td>RigNet [Xu et al. 2020]</td><td>0.147</td><td>0.111</td><td>0.100</td><td>0.141</td><td>0.127</td><td>0.118</td><td>8.56</td><td>2.14</td></tr><tr><td>MagicArti. [Song et al. 2025b]</td><td>0.064</td><td>0.046</td><td>0.031</td><td>0.202</td><td>0.209</td><td>0.197</td><td>_Ⅱ</td><td>-Ⅱ</td></tr><tr><td>Puppeteer [Song et al. 2025a]</td><td>0.062</td><td>0.044</td><td>0.032</td><td>0.201</td><td>0.211</td><td>0.198</td><td>10.31</td><td>0.64</td></tr><tr><td>RigAnything [Liu et al. 2025b]</td><td>0.060</td><td>0.046</td><td>0.030</td><td>0.170</td><td>0.174</td><td>0.168</td><td>_§</td><td>_§</td></tr><tr><td>UniRig [Zhang et al. 2025]</td><td>0.040</td><td>0.031</td><td>0.020</td><td>0.191</td><td>0.209</td><td>0.219</td><td>11.55</td><td>0.91</td></tr><tr><td>GVB [Dionne and de Lasa 2013]</td><td></td><td></td><td></td><td></td><td>一</td><td>一</td><td>26.67</td><td>2.08</td></tr><tr><td>Pinocchio [Baran and Popović 2007]</td><td>0.101</td><td>0.072</td><td>0.045</td><td>一</td><td>-</td><td>-</td><td>48.77</td><td>4.42</td></tr><tr><td>Ours</td><td>0.023</td><td>0.013</td><td>0.011</td><td>0.068</td><td>0.096</td><td>0.098</td><td>7.70</td><td>0.60</td></tr></table>

<sup>§</sup>RigAnything jointly predicts skeleton and skinning and does not accept an external skeleton; controlled skinning is therefore inapplicable.  
<sup>∥</sup> MagicArticulate shares the same skinning module as Puppeteer; results are not duplicated.

Table 3. Ablation study on body skeleton alignment, auxiliary bone generation, and skinning quality. Values in Skinning Quality are scaled by $1 0 ^ { 3 }$
<table><tr><td rowspan="2">Variant</td><td colspan="3">Body Skeleton Alignment</td><td colspan="3">Auxiliary Bone Generation</td><td colspan="2">Skinning Weight</td></tr><tr><td>J2J↓</td><td>J2B↓</td><td>B2B↓</td><td>J2J↓</td><td>J2B↓</td><td>B2B↓</td><td> $L _ { 1 } \downarrow$ </td><td> $\mathrm { D e f } { - } L _ { 1 } \ \downarrow$ </td></tr><tr><td>Ours w/o  $\mathcal { L } _ { \mathrm { { s e m } } }$ </td><td>0.029</td><td>0.018</td><td>0.015</td><td>0.120</td><td>0.130</td><td>0.131</td><td>11.50</td><td>0.83</td></tr><tr><td>Ours w/o Hier. Encoder</td><td>0.026</td><td>0.015</td><td>0.013</td><td>0.096</td><td>0.106</td><td>0.102</td><td>18.60</td><td>1.88</td></tr><tr><td>Ours w/o TAMB Fusion</td><td>0.030</td><td>0.018</td><td>0.015</td><td>一</td><td>一</td><td>一</td><td></td><td>一</td></tr><tr><td>Ours w/o  $\mathcal { L } _ { \mathrm { p r o j } }$ </td><td>0.031</td><td>0.020</td><td>0.017</td><td>一</td><td>一</td><td>I</td><td>一</td><td>一</td></tr><tr><td>Ours w/o Label-Aware Attn</td><td>一</td><td>一</td><td></td><td>0.107</td><td>0.126</td><td>0.125</td><td></td><td></td></tr><tr><td>Ours w/o Chain-to-Bone</td><td>一</td><td>一</td><td>一</td><td>一</td><td>1</td><td></td><td>11.24</td><td>0.84</td></tr><tr><td>Ours</td><td>0.023</td><td>0.013</td><td>0.011</td><td>0.068</td><td>0.096</td><td>0.098</td><td>7.70</td><td>0.60</td></tr></table>

Each variant removes one component from the full model. “–” marks task groups unafected by the ablated component: e.g., TAMB Fusion and $\mathcal { L } _ { \mathrm { p r o j } }$ act only on body alignment; Label-Aware Attn acts only on auxiliary generation; Chain-to-Bone acts only on skinning.

![](images/1a3785395e393a9479ae338bd4d37d7370718cdf8ddd2f0990132ee67875b29a.jpg)  
Fig. 5. Auxiliary-label control. With body skeleton and mesh fixed, diferent auxiliary labels control the generated bones and, under the same driving motion, lead to diferent animations across the skirt, accessory, and skirt+accessory rigs. Blue/red: body/auxiliary bones.

## 4.4 Ablation Studies

We ablate key design choices to validate two forms of structure in our framework: semantic structure (Sec. 4.4.1) and geometrickinematic structure (Sec. 4.4.2). All variants share the same training data and schedule; each removes exactly one component. Results are summarized in Table 3.

4.4.1 Semantic Structure. We first ablate the semantic structure used to localize garment-related bones and skinning regions. Removing $\mathcal { L } _ { \mathrm { s e m } }$ substantially degrades auxiliary generation (J2J 0.068→0.120, B2B $0 . 0 9 8 { \scriptstyle \longrightarrow } 0 . 1 3 1 )$ and skinning quality $( L _ { 1 } ~ 7 . 7 0 {  } 1 1 . 5 0 _ { \ Z }$ , Def-�

0.60→0.83), indicating that vertex-level semantic supervision helps both auxiliary placement and cross-part weight separation. Removing Label-Aware Attention also hurts auxiliary generation, increasing J2J from 0.068 to 0.107 and B2B from 0.098 to 0.125, showing that category labels need to interact with mesh and body features rather than act as global tags. Fig. 6 shows the corresponding fail ures: skeleton joints drift from the intended regions, and skinning weights bleed across unrelated parts.

## 4.4.2 Geometric Structure.

![](images/16f55393ed5027a868be8b73ac40f6303270ab852af1cc5903d45a6524e92bb3.jpg)  
Fig. 6. Qualitative ablations on skeleton generation and skinning. Left: blue and green denote predicted and ground-truth skeletons, respectively. Removing semantic supervision causes both body and auxiliary bones to drift from their intended regions. Right: under articulated deformation, removing the chain-to-bone decomposition or semantic supervision introduces visible cross-part skinning bleeding and deformation artifacts.

Hierarchical Bone Encoder. The hierarchical bone encoder architecture is used in all three modules; replacing it with a bone encoder that omits chain information (w/o Hier. Encoder in Table 3) degrades all three tasks—body J2J rises from 0.023 to 0.026, auxiliary J2J from 0.068 to 0.096, and skinning Def- ${ - } L _ { 1 }$ from 0.60 to 1.88—confirming that chain-level hierarchy provides useful priors throughout the pipeline.

Topology-Aware Body Alignment. The Body Skeleton Alignment block in Table 3 isolates two geometric choices. Removing TAMB Fusion increases body J2J/B2B from 0.023/0.011 to 0.030/0.015, while removing $\mathcal { L } _ { \mathrm { p r o j } }$ increases body J2J/B2B from 0.023/0.011 to 0.031/0.017, confirming that topology-aware fusion and bone-axis projection provide complementary constraints for body skeleton fitting.

Structural Priors for Skinning Prediction. Removing the chain-tobone decomposition (w/o Chain-to-Bone) raises skinning $L _ { 1 }$ from 7.70 to 11.24 and Def-� from 0.60 to 0.84. Without the coarse chain assignment stage, weights leak across part boundaries, producing visible cross-part bleeding in Fig. 6 (Right).

## 4.5 In-the-Wild Production Evaluation

We further perform evaluation on 50 diverse in-the-wild assets covering diferent genders, body types, and clothing styles such as layered skirts, ribbons, and capes. Five professional technical artists independently assess each method’s output in a blind setting, rating overall rig quality (skeleton and skinning jointly) on a 5-point Likert scale (1: broken, 5: production-ready) and estimating the manual repair time needed to bring the rig to production standard. The 50 evaluation-only assets come from a held-out internal production pool disjoint from Human-Mix, with 52–128 total bones and 2–36 auxiliary bones per character. Artists specify the auxiliary categories they intend to author; only TailorRig consumes these labels, while all methods are judged against the same intended rig. Method names are hidden and presentation order is randomized per asset. Repair time is an expert estimate rather than a timed editing session. Ratings are highly consistent across artists (ICC(2,k)= 0.95).

On in-the-wild assets (Table 4), TailorRig attains the lowest errors, the highest artist rating (4.56/5), and the shortest repair time (0.80 h)—42% below UniRig (1.38 h). On the held-out Human-Mix split, an end-to-end test where each method uses its own predicted skeleton and skinning weights, with only bone rotations optimized to fit the same five target poses, shows the same ordering: Def- $\cdot L _ { 1 }$ is $1 . 1 6 \times 1 0 ^ { - 2 }$ for TailorRig vs. $1 . 5 0 \times 1 0 ^ { - 2 }$ for UniRig. This crosstopology test is distinct from the controlled-skeleton comparison in Table 2.

## 5 Conclusion and Future Work

We presented TailorRig, a controllable, hierarchical framework for garment-aware rigging that couples template-based body alignment with label-conditioned auxiliary generation and chain-to-bone skin ning. It improves template consistency, garment flexibility, and deformation stability for production assets. Current limitations include the fixed discretization used in auxiliary bone generation, which may introduce small ofsets in dense regions, and the reliance on a humanoid template with predefined semantic labels. Future work will explore adaptive or multi-scale representations and extend the taxonomy to broader character types and accessory structures.

Table 4. In-the-wild production evaluation on 50 assets rated by 5 artists. Qual.: 1–5 Likert; Repair Time: estimated hours to production quality.
<table><tr><td>Method</td><td>Body J2J</td><td>Aux J2J↓</td><td>Skin L1↓</td><td>Qual.↑</td><td>Repair Time↓</td></tr><tr><td>Pinocchio [Baran and Popović 2007]</td><td>0.142</td><td>0.274</td><td>0.010</td><td>2.23</td><td>3.48</td></tr><tr><td>Puppet. [Song et al. 2025a]</td><td>0.082</td><td>0.226</td><td>0.033</td><td>2.98</td><td>2.70</td></tr><tr><td>UniRig [Zhang et al. 2025]</td><td>0.025</td><td>0.188</td><td>0.010</td><td>3.37</td><td>1.38</td></tr><tr><td>Ours</td><td>0.024</td><td>0.160</td><td>0.003</td><td>4.56</td><td>0.80</td></tr></table>

![](images/c0ab0f988ebca12a60bf1b89123c8b825e04b136086de14654ff168050c657d8.jpg)

Fig. 7. Skeleton comparison on garment-rich characters. Blue denotes body bones, and red denotes auxiliary or garment-related bones (manually colored for baselines when no explicit auxiliary labels are provided). Baselines often omit, merge, or mis-parent garment-related auxiliary bones, whereas TailorRig produces cleaner body–auxiliary separation under the specified labels and beter matches the ground truth.  
![](images/1c72b364eef24dc872ba3963dba20ff45dacc0c20da651057f0c7a0fde16044f.jpg)  
Fig. 8. Skinning comparison under controlled auxiliary-bone rotations. All methods use the same ground-truth skeleton and are evaluated under the same auxiliary-bone rotations, isolating skinning-weight prediction from skeleton errors. Surface heat maps visualize skinning-weight error against the ground truth, with warmer colors indicating larger errors. TailorRig produces cleaner garment/accessory deformation, whereas baselines exhibit stronger body–auxiliary coupling and local artifacts.

![](images/8026f72d9c1527f2c126ff8af5aa90d86cc7c125a7e7143559311cce08266743.jpg)

Fig. 9. Frame-wise animation comparison after full rigging. After each method generates a complete skeleton and skinning weights, we drive all results with the same motion and render the same sampled frames. TailorRig beter preserves garment motion relative to the body, closely matching the ground truth. In contrast, Pinocchio and UniRig exhibit garment–body coupling, and UniRig further shows leg–cloth interpenetration in the highlighted regions.

## References

Rinat Abdrashitov, Kim Raichstat, Jared Monsen, and David Hill. 2023. Robust skin weights transfer via weight inpainting. In SIGGRAPH Asia 2023 Technical Communications. 1–4.

Adobe Inc. 2021. Mixamo.

Nina Amenta and Marshall Bern. 1998. Surface reconstruction by Voronoi filtering. In Proceedings ofthe fourteenth annual symposium on Computational geometry. 39–48.

Ilya Baran and Jovan Popović. 2007. Automatic rigging and animation of 3d characters. ACM Transactions on graphics (TOG) 26, 3 (2007), 72–es.

Zedong Chu, Feng Xiong, Meiduo Liu, Jinzhi Zhang, Mingqi Shao, Zhaoxu Sun, Di Wang, and Mu Xu. 2025. Humanrig: Learning automatic rigging for humanoid character in a large scale dataset. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 304–313.

Yufan Deng, Yuhao Zhang, Chen Geng, Shangzhe Wu, and Jiajun Wu. 2025. Anymate: A dataset and baselines for learning 3d object rigging. In Proceedings ofthe Special Interest Group on Computer Graphics and Interactive Techniques Conference Papers. 1–10.

Olivier Dionne and Martin de Lasa. 2013. Geodesic voxel binding for production char acter meshes. In Proceedings ofthe 12th ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 173–180.

Jingfeng Guo, Jian Liu, Jinnan Chen, Shiwei Mao, Changrong Hu, Puhua Jiang, Junlin Yu, Jing Xu, Qi Liu, Lixin Xu, et al. 2025a. Auto-Connect: Connectivity-Preserving RigFormer with Direct Preference Optimization. arXiv preprint arXiv:2506.11430 (2025).

Zhiyang Guo, Jinxu Xiang, Kai Ma, Wengang Zhou, Houqiang Li, and Ran Zhang. 2025b. Make-it-animatable: An eficient framework for authoring animation-ready 3d characters. In Proceedings of the Computer Vision and Pattern Recognition Conference. 10783–10792.

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023).

Zizhou Huang, Chrystiano Araújo, Andrew Kunz, Denis Zorin, Daniele Panozzo, and Victor Zordan. 2025. Intersection-free Garment Retargeting. In Proceedings ofthe Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–12.

Moritz Ibing, Gregor Kobsik, and Leif Kobbelt. 2023. Octree transformer: Autoregressive 3d shape generation on hierarchically structured sequences. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2698–2707.

Nozomi Isozaki, Shigeyoshi Ishima, Yusuke Yamada, Yutaka Obuchi, Rika Sato, and Norio Shimizu. 2021. VRoid studio: a tool for making anime-like 3D characters using your imagination. In SIGGRAPH Asia 2021 Real-Time Live! 1–1.

Alec Jacobson, Ilya Baran, Jovan Popovic, and Olga Sorkine. 2011. Bounded biharmonic weights for real-time deformation. ACM Trans. Graph. 30, 4 (2011), 78.

Doug L James and Christopher D Twigg. 2005. Skinning mesh animations. ACM Transactions on Graphics (TOG) 24, 3 (2005), 399–407.

Ladislav Kavan, P-P Sloan, and Carol O’Sullivan. 2010. Fast and eficient skinning of animated meshes. In Computer Graphics Forum, Vol. 29. Wiley Online Library, 327–336.

Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, et al. 2025. Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details. arXiv preprint arXiv:2506.16504 (2025).

Jeanne-Emma Lefèvre, Théo Cheynel, and Thomas Daniel. 2025. Comparing 3D Data Representations for Keypoint Estimation in Humanoid Characters. In Proceedings ofthe SIGGRAPH Asia 2025 Posters. 1–3.

Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. 2023. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214 (2023).

Peizhuo Li, Kfir Aberman, Rana Hanocka, Libin Liu, Olga Sorkine-Hornung, and Baoquan Chen. 2021. Learning skeletal articulations with neural blend shapes. ACM Transactions on Graphics (TOG) 40, 4 (2021), 1–15.

Cheng Lin, Changjian Li, Yuan Liu, Nenglun Chen, Yi-King Choi, and Wenping Wang. 2021. Point2skeleton: Learning skeletal representations from point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4277–4286.

Isabella Liu, Zhan Xu, Wang Yifan, Hao Tan, Zexiang Xu, Xiaolong Wang, Hao Su, and Zifan Shi. 2025b. Riganything: Template-free autoregressive rigging for diverse 3d assets. ACM Transactions on Graphics (TOG) 44, 4 (2025), 1–12.

Lijuan Liu, Youyi Zheng, Di Tang, Yi Yuan, Changjie Fan, and Kun Zhou. 2019. Neu roskinning: Automatic skin binding for production characters with deep graph networks. ACM Transactions on Graphics (ToG) 38, 4 (2019), 1–12.

Minghua Liu, Mikaela Angelina Uy, Donglai Xiang, Hao Su, Sanja Fidler, Nicholas Sharp, and Jun Gao. 2025a. Partfield: Learning 3d feature fields for part segmentation and beyond. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 9704–9715.

Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. 2023. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems 36 (2023), 22226–22246.

Jing Ma and Dongliang Zhang. 2023. TARig: Adaptive template-aware neural rigging for humanoid characters. Computers & Graphics 114 (2023), 158–167.

Paritosh Mittal, Yen-Chi Cheng, Maneesh Singh, and Shubham Tulsiani. 2022. Autosdf: Shape priors for 3d completion, reconstruction and generation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 306–315.

Charlie Nash, Yaroslav Ganin, SM Ali Eslami, and Peter Battaglia. 2020. Polygen: An autoregressive generative model of3d meshes. In International conference on machine learning. PMLR, 7220–7229.

Xiaoyu Pan, Jiancong Huang, Jiaming Mai, He Wang, Honglin Li, Tongkui Su, Wenjun Wang, and Xiaogang Jin. 2021. HeterSkinNet: A heterogeneous network for skin weights prediction. In Proceedings ofthe ACM on computer graphics and interactive techniques, Vol. 4. Association for Computing Machinery.

Rick Parent. 2012. Computer animation: algorithms and techniques. Newnes.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d difusion. arXiv preprint arXiv:2209.14988 (2022).

Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. 2024. Meshgpt: Generating triangle meshes with decoder-only transformers. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 19615–19625.

Chaoyue Song, Xiu Li, Fan Yang, Zhongcong Xu, Jiacheng Wei, Fayao Liu, Jiashi Feng, Guosheng Lin, and Jianfeng Zhang. 2025a. Puppeteer: Rig and animate your 3d models. arXiv preprint arXiv:2508.10898 (2025).

Chaoyue Song, Jianfeng Zhang, Xiu Li, Fan Yang, Yiwen Chen, Zhongcong Xu, Jun Hao Liew, Xiaoyang Guo, Fayao Liu, Jiashi Feng, et al. 2025b. Magicarticulate: Make your 3d models articulation-ready. In Proceedings of the Computer Vision and Pattern Recognition Conference. 15998–16007.

Andrea Tagliasacchi, Hao Zhang, and Daniel Cohen-Or. 2009. Curve skeleton extraction from incomplete point cloud. In ACM SIGGRAPH 2009 papers. 1–9.

Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. 2019. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog) 38, 5 (2019), 1–12.

Zhan Xu, Yang Zhou, Evangelos Kalogerakis, Chris Landreth, and Karan Singh. 2020. Rignet: Neural rigging for articulated characters. arXiv preprint arXiv:2005.00559 (2020).

Zhan Xu, Yang Zhou, Li Yi, and Evangelos Kalogerakis. 2022. Morig: Motion-aware rigging of character meshes from point clouds. In SIGGRAPH Asia 2022 conference papers. 1–9.

Xingguang Yan, Liqiang Lin, Niloy J Mitra, Dani Lischinski, Daniel Cohen-Or, and Hui Huang. 2022. Shapeformer: Transformer-based shape completion via sparse representation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 6239–6249.

Jia-Peng Zhang, Cheng-Feng Pu, Meng-Hao Guo, Yan-Pei Cao, and Shi-Min Hu. 2025. One model to rig them all: Diverse skeleton rigging with unirig. ACM Transactions on Graphics (TOG) 44, 4 (2025), 1–18.

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. 2024. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–20.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068 (2022).