# PPCAR-Net: Projection-Refined Parametric 3D Coronary Artery Reconstruction from Sparse X-ray Angiographic Views

Yu Ren<sup>1,2</sup>, Hwee Kuan Lee<sup>1</sup>, Tat-Jen Cham<sup>2</sup>, Jonathan Yap<sup>3</sup>, Khung Keong Yeo<sup>3</sup>

<sup>1</sup>Bioinformatics Institute, Agency for Science, Technology and Research (A\*STAR) <sup>2</sup>College of Computing and Data Science, Nanyang Technological University <sup>3</sup>National Heart Centre Singapore

## Abstract

Sparse-view 3D coronary reconstruction commonly relies on cross-view correspondence and triangulation, which are vulnerable to vessel overlap and foreshortening, or on volumetric prediction followed by vascular-graph extraction, which does not directly provide centrelines and radii. We introduce PPCAR-Net, a projection-refined parametric coronary artery reconstruction network that directly predicts a branch-structured centreline-and-radius representation without explicit point matching, triangulation, or an intermediate volume. Given a variable number of segmented views, a coarse predictor combines frozen VGGT features with learned branch queries to estimate branch presence, B-spline centreline trajectories, and dense radius profiles. Projection guided geometry and radius refiners then sample local evidence from the input views and apply residual corrections learned with 3D supervision. We evaluate representation fidelity and sparse view reconstruction quantitatively and qualitatively. On simulated angiographic masks generated from CT-derived coronary anatomy, PPCAR-Net produces better connected artery reconstructions and achieves strong centreline accuracy, particularly for RCA, while maintaining competitive volumetric overlap. Coarse-to-fine inference takes 121 ms, enabling real-time reconstruction. Code and models are publicly available at: https://github.com/G2304138H/PPCAR-Net.

Keywords: 3D coronary artery reconstruction; Sparse-view reconstruction; Learning-based medical image reconstruction; Structured anatomical representation; X-ray coronary angiography; Vascular modelling.

## 1 Introduction

X-ray coronary angiography is fundamental for diagnosing coronary artery disease and guiding intervention. Each angiogram, however, represents a two-dimensional projection of inherently threedimensional coronary anatomy. Foreshortening and vessel overlap can obscure spatial relationship between branches. In clinical practice, simultaneous biplane acquisitions are uncommon, and complementary monoplane views are generally acquired sequentially. Consequently, cardiologists must mentally integrate multiple views to infer three-dimensional vessel geometry (Klein et al., 1998). Conventional 3D quantitative coronary angiography partially automates this process by reconstructing anatomy from two calibrated, phase-matched projections through cross-view vessel correspondence and triangulation (Tu et al., 2010). However, sparse projections, overlapping branches, and foreshortening can make these correspondences ambiguous. We therefore investigate whether population-level patterns of coronary anatomy learned across training cases can support a plausible branch-structured 3D hypothesis when the projection evidence is ambiguous.

Existing learning-based methods commonly represent coronary arteries using neural fields, Gaussian primitives, or voxel volumes, rather than directly producing ordered coronary centrelines and lumen radii. Extracting vascular graphs from volumetric outputs can introduce errors that compromise downstream analysis and interpretation. We instead adopt the dense centreline-andradius representation (Iyer et al., 2023) as the model’s native output. We parameterise each dense branch centreline as a continuous cubic B-spline to ensure connectivity within each branch, while retaining dense radius values for localised radius variation. We refer to this branch-structured B-spline centreline-and-radius representation as vessel code. It is directly interpretable and editable, without requiring vascular-graph extraction from a volumetric prediction.

Table 1: Comparison of learning-based coronary artery reconstruction methods. Per-case runtime is the complete time required to reconstruct one 3D artery from two input views, including vasculargraph extraction when applicable.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Number of input views</td><td rowspan="2">Output representation</td><td colspan="3">Output</td><td rowspan="2">Per-case runtime ↓</td></tr><tr><td>3D volume</td><td>Connected centreline</td><td>Radius</td></tr><tr><td>Iyer et al. (2023)</td><td>2 or 3</td><td>Dense centreline points and radii</td><td>V</td><td>√</td><td>V</td><td></td></tr><tr><td>NeRF-CA (Maas et al., 2025)</td><td>≥4</td><td>Novel 2D views</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>NerT-CA (Maas et al., 2026)</td><td>M 3</td><td>Novel 2D views</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>SDF-CAR (Reda et al., 2026)</td><td>2</td><td>3D implicit SDF field</td><td></td><td>×</td><td>X</td><td>45 min</td></tr><tr><td>3DGR-CAR (Fu et al., 2024)</td><td>≥ 2</td><td>3D Gaussian primitives</td><td></td><td></td><td>X</td><td>63.7 s</td></tr><tr><td>DeepCA (Wang et al., 2025b)</td><td>2</td><td>3D voxel</td><td></td><td>× ×</td><td>X</td><td>2.69 s</td></tr><tr><td>AutoCAR (Zhu et al., 2025)</td><td>2</td><td>3D voxel</td><td></td><td>X</td><td>X</td><td>3.76 s</td></tr><tr><td>PPCAR-Net (ours)</td><td>≥1</td><td>B-spline centreline curves and dense radii</td><td></td><td></td><td></td><td>121 ms</td></tr></table>

We propose PPCAR-Net, a projection-refined parametric coronary artery reconstruction network that predicts vessel code through a coarse-to-fine process. Given a variable number of 2D coronary masks and their acquisition directions, a coarse predictor combines frozen VGGT features with learned branch queries to estimate branch existence, B-spline control points, and dense radius profiles. It infers a plausible branch-structured 3D hypothesis: the branch-existence predictions determine the number of active branches, the B-spline control points describe their trajectories, and the dense-radius predictions provide approximate radius profiles. Projection-guided geometry and radius refiners then sample local 2D evidence and apply learned residual corrections to the centreline geometry and radii. This feed-forward design empirically improves reconstruction accuracy without per-case optimisation, enabling sub-second inference. We train and evaluate using simulated angiographic masks generated from the ImageCAS CT angiography dataset (Zeng et al., 2023). Qualitative examples on diseased ASOCA anatomy (Gharleghi et al., 2023) and preprocessed real X-ray angiography explore anatomical and image-domain transfer, respectively. Table 1 compares the capabilities of our approach with representative learning-based methods.

The contributions of this work are:

• We demonstrate that integrating a coronary anatomical prior learned across the training cohort with projection-guided corrections based on case-specific evidence reconstructs accurate, anatomically structured 3D coronary arteries from sparse views, without explicit point correspondence or per-case optimisation. These results suggest that learned anatomical priors provide useful constraints under ambiguity from sparse X-ray angiographic views.

• PPCAR-Net, supported by branch-subset augmentation during training, achieves the best RCA structural and centreline reconstruction metrics among evaluated methods, with competitive volumetric accuracy and robust branch prediction. This is accomplished with sub-second inference, demonstrating both high accuracy and real-time capability.

• We enhance the explicit branch-structured centreline-and-radius representation by incorporating a smooth cubic B-spline parameterisation of dense polyline centrelines. This adaptation leads to more coherent and accurate vessel geometry reconstructions compared to dense point predictions, while maintaining interpretability and intra-branch continuity.

## 2 Related Work

Conventional 3D quantitative coronary angiography reconstructs coronary geometry from calibrated X-ray projections by segmenting target vessels, establishing cross-view centreline correspondence, and triangulating matched points (Klein et al., 1998; Tu et al., 2010). Related parametric approaches have used deformable B-splines for 2D vessel-boundary and centreline analysis (Klein et al., 1997), 3D B-spline curves for motion tracking in biplane angiograms (Shechter et al., 2003), and NURBS surfaces for structured coronary reconstruction and meshing (Vukicevic et al., 2018). These pipelines may require user interaction and remain sensitive to vessel overlap and foreshortening, motivating learning-based reconstruction from segmented angiograms.

Recently, learning-based methods have been used for automating the mapping of angiographic images to 3D vessels. AutoCAR backprojects image features into a sparse 3D volume and predicts vascular occupancy and centreness; post-processing converts this prediction into a vascular graph, while iterative deformation relates graphs reconstructed at contracted and relaxed cardiac states (Zhu et al., 2025). DeepCA follows a related volumetric strategy: it backprojects two segmented, nonsimultaneous projections into a 3D input volume and uses a conditional generative network to reconstruct a voxelised coronary tree while implicitly compensating for inter-view motion (Wang et al., 2025b). In contrast, Iyer et al. (2023) directly predict an ordered centreline-and-radius representation from segmented, uncalibrated angiograms and their model is trained and evaluated primarily on synthetically generated RCA trees with a fixed number of branches. More broadly, VesselDiffusion generates vascular point sets using a diffusion model (Guo et al., 2025). Because paired 2D projections and corresponding 3D artery labels are rarely available in clinical X-ray angiography, supervised methods commonly generate paired training data from coronary CT angiography and evaluate image-domain transfer on real X-ray angiograms.

Neural-field and Gaussian representations have also been applied to coronary reconstruction. NeRF-CA and NerT-CA model dynamic coronary appearance within an X-ray attenuation field and synthesise new composite X-ray-like views (Maas et al., 2025, 2026). 3DGR-CAR represents coronary occupancy with Gaussian primitives and combines a learned initialisation with projectionbased parameter optimisation (Fu et al., 2024), whereas SDF-CAR fits a patient-specific neural implicit field to two segmented projections (Reda et al., 2026). Although these methods recover or imply 3D structure, their native neural-field, Gaussian, and occupancy representations do not directly estimate explicit coronary artery geometries for downstream tasks such as blood-flow simulation; extracting arterial structure through post-processing may introduce additional errors and does not leverage learned structural priors.

## 3 Methods

## 3.1 3D Coronary Artery Reconstruction Task

Our task is to reconstruct an explicit 3D centreline-and-radius representation of a coronary artery from a variable number of calibrated 2D coronary mask projections. For each coronary case, the input is the variable-sized set

$$
\mathcal { X } = \{ ( I _ { k } , \theta _ { k } , \phi _ { k } ) \} _ { k = 1 } ^ { K } ,\tag{1}
$$

where K is the number of available views, $\scriptstyle { I _ { k } }$ is the coronary artery mask in view k, and $( \theta _ { k } , \phi _ { k } )$ specifies its acquisition direction. Each view follows a cone-beam projection model $\Pi _ { k } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 2 }$ whose pose is determined by $( \theta _ { k } , \phi _ { k } )$ and whose fixed intrinsic parameters map 3D coordinates to detector pixels. We use the same camera model to generate the input projections and to guide the refinement described in later subsections. The target centreline-and-radius representation is introduced next.

## 3.2 Parametric Vessel Representation and Dataset Construction

Here, we formalise vessel code, the explicit branch-structured centreline-and-radius representation predicted by PPCAR-Net, and describe how its training targets are constructed. A complete coronary tree is represented using at most M branches, indexed by $m \in \{ 1 , \ldots , M \}$ , with $e _ { m } \in \{ 0 , 1 \}$ indicating whether branch m is present. For each existing branch, we adopt the dense centrelineand-radius representation of Iyer et al. (2023). Branch m with $e _ { m } = 1$ is described by N ordered centreline-and-radius samples running from its proximal root or attachment point on its parent branch to its distal endpoint, as illustrated in Figure 1(C):

![](images/9aa8fcaeb4d3e06935a2844860aeb8c4f36fc28a070c887959c47ba696d3f778.jpg)  
Figure 1: Coronary trees and vessel code representation. (A) Schematic RCA tree showing the right coronary artery (RCA), posterolateral branch (PLB), posterior descending artery (PDA), and acute marginal branch (AM); (B) Schematic LCA tree showing the left main (LM), left anterior descending (LAD), and left circumflex (LCX) arteries; (C) Dense centreline $c _ { m , n }$ and radius $r _ { m , n }$ representation of a branch; (D) B-spline representation of the centreline of branch m with dark-blue points denoting B-spline control points $\pmb { p } _ { m , j }$ and the light-blue centreline that they define.

$$
V _ { m } = \{ ( \boldsymbol { c } _ { m , n } , \boldsymbol { r } _ { m , n } ) \} _ { n = 1 } ^ { N } , \qquad \boldsymbol { c } _ { m , n } \in \mathbb { R } ^ { 3 } , \quad \boldsymbol { r } _ { m , n } \in \mathbb { R } _ { > 0 } .\tag{2}
$$

Here, $\pmb { c } _ { m , n }$ denotes the n-th centreline point of branch m in 3D space, and $r _ { m , n }$ denotes the vessel radius associated with that centreline point. A circular lumen profile centred at $c _ { m , n }$ with radius $r _ { m , n }$ provides the corresponding tubular surface approximation.

To better reflect the continuous structure of an artery while reducing the output dimensionality, we parameterise the raw centreline coordinates by a cubic B-spline with $K _ { \mathrm { c p } }$ control points, and the reconstruction network predicts these control points directly. For branch $m ,$ , the control-point matrix $\boldsymbol { P _ { m } } \in \mathbb { R } ^ { K _ { \mathrm { c p } } \times 3 }$ and the decoded dense centreline points are

$$
\mathbf { \nabla } P _ { m } = ( \pmb { p } _ { m , 1 } , \dots , \pmb { p } _ { m , K _ { \mathrm { c p } } } ) ^ { \top } , \qquad \pmb { c } _ { m , n } = \sum _ { j = 1 } ^ { K _ { \mathrm { c p } } } B _ { j , 3 } ( t _ { n } ) \pmb { p } _ { m , j } , \quad t _ { n } = \frac { n - 1 } { N - 1 } .\tag{3}
$$

Here, $\pmb { p } _ { m , j } \in \mathbb { R } ^ { 3 }$ is the j-th control point, $t _ { n }$ is the normalised arc-length position of centreline point n, and $B _ { j , 3 } ( \cdot )$ is the j-th cubic B-spline basis function defined by the Cox–de Boor recursion. As shown in Figure 1(D), the control points determine the decoded centreline but do not need to lie on it. Because a B-spline is a continuous curve over its parameter domain, the decoded centreline of every branch is connected by construction.

Radius is deliberately not compressed by the centreline spline. The dense vector $\begin{array} { r l } { r _ { m } } & { { } = } \end{array}$ $( r _ { m , 1 } , \hdots , r _ { m , N } ) ^ { \sf T } \ \in \ \mathbb { R } _ { > 0 } ^ { N }$ retains one radius at every decoded centreline point. The complete vessel code is therefore $\{ ( e _ { m } , P _ { m } , r _ { m } ) \} _ { m = 1 } ^ { M }$ . This parameterisation provides a low-dimensional smooth prior for centreline geometry without imposing the same smoothness bottleneck on localised radius changes.

Dataset and target construction. Supervised training uses coronary annotations from the ImageCAS CT angiography dataset (Zeng et al., 2023) as 3D labels. Each annotation is separated into RCA and LCA components (Figure 1(A–B)). Each component mask is skeletonised to extract ordered branch centrelines, and the radius at each centreline point is estimated using the Euclidean distance transform, obtaining an ordered branch-wise vessel code. Branches follow deterministic appearance-based positions in the target sequence, allowing optional side branches without assigning every query a fixed anatomical name. Each retained branch contains N = 200 proximal-to-distal centreline-and-radius samples, represented by $K _ { \mathrm { c p } } = 2 0$ B-spline control points and dense radii in millimetres. The target control points are fitted offline by endpoint-constrained least squares, and $K _ { \mathrm { c p } } = 2 0$ yields an overall mean centreline fitting error of 0.164 mm and a centreline-based overlap score of 99.92% between the fitted and original vessels (Appendix Table 6). This process yields 750 RCA and 750 LCA components, with each cohort divided into 600 training, 75 validation, and 75 test cases. For every ImageCAS case, the cone-beam model $\Pi _ { k }$ generates seven $2 5 6 \times 2 5 6$ synthetic coronary-mask projections that mimic segmented X-ray angiograms at clinically motivated acquisition directions, from which the requested number of views is selected during training and evaluation. Separately, the 20 diseased CT angiography cases from ASOCA (Gharleghi et al., 2023) are used only for qualitative analysis; their annotations are not converted into vessel code targets or included in training or quantitative evaluation.

![](images/f1ca7b017b37a03579d2f017247adbcf8c916d07dc3b132558691409fe934ac7.jpg)  
Figure 2: Flexible-view coarse parametric predictor in PPCAR-Net: (A) Frozen VGGT features are concatenated (⊕) with encoded acquisition directions to form the multiview memory $\scriptstyle { E _ { \mathrm { v i e w } } }$ . (B) Trainable branch queries cross-attend to this memory, and the existence, control-point, and radius heads produce the coarse vessel code $\hat { V } ^ { ( 0 ) }$ . Fire and snowflake symbols denote trainable and frozen modules, respectively.

## 3.3 Flexible-View Coarse Parametric Predictor

The first stage of PPCAR-Net is a flexible-view coarse parametric predictor, summarised in Figure 2. The coarse predictor combines evidence from the available views with learned patterns of coronary anatomy to infer a plausible structured vessel code. The existence predictions determine the active number of branches, while the control-point and radius heads estimate their global trajectories and approximate radius profiles.

As illustrated in Figure $2 ( \mathrm { A } ) .$ , the frozen VGGT image backbone $f _ { \mathrm { b a c k b o n e } }$ extracts image tokens $\pmb { H } _ { k } ^ { I }$ from each input image $\scriptstyle { I _ { k } } .$ , while the view-direction encoder $f _ { \mathrm { v i e w } }$ encodes its acquisition direction $( \theta _ { k } , \phi _ { k } )$ as $h _ { k }$ . To obtain the view-direction-aware tokens $\mathbfit { T } _ { k }$ , the same vector $h _ { k }$ is repeated for every image token, concatenated with $\pmb { H } _ { k } ^ { I }$ along the feature dimension, and passed through a learned projection back to the model dimension $d .$ The tokens from all available views form the multiview memory $\pmb { E } _ { \mathrm { v i e w } } = [ \pmb { T } _ { 1 } ; \dots ; \pmb { T } _ { K } ]$ . The number of views K and their acquisition directions can vary between cases.

Figure 2(B) shows how M trainable branch queries $\mathbf { \delta q } _ { m }$ retrieve branch-specific evidence from this memory. The query matrix $Q _ { \mathrm { q u e r y } } = [ \pmb { q } _ { 1 } ; \ldots ; \pmb { q } _ { M } ]$ cross-attends to $\pmb { { \cal E } } _ { \mathrm { v i e w } }$ through $f _ { \mathrm { c r o s s - a t t n } } .$ with $Q _ { \mathrm { { q u e r y } } }$ serving as the queries and $\pmb { { E } } _ { \mathrm { v i e w } }$ as the keys and values. This produces branch tokens $[ { \pmb b } _ { 1 } ; \ldots ; { \pmb b } _ { M } ]$ , where $b _ { m } \in  { \mathbb { R } } ^ { d }$ . The branch queries correspond to the ordered branch entries: $\pmb q _ { m }$ retrieves image evidence for the m-th branch entry, while the existence head determines whether that branch is present in the current case. The resulting branch token $b _ { m }$ is mapped by the existence, control-point, and radius heads to the coarse vessel code $\hat { V } ^ { ( 0 ) } = \left\{ \left( \hat { e } _ { m } , \hat { P } _ { m } ^ { ( 0 ) } , \hat { \pmb { r } } _ { m } ^ { ( 0 ) } \right) \right\} _ { m = 1 } ^ { M }$ , where $\hat { e } _ { m } , \hat { P } _ { m } ^ { ( 0 ) }$ , and $\hat { \pmb { r } } _ { m } ^ { ( 0 ) }$ denote the predicted branch existence, B-spline control points, and dense radii, respectively. This stage learns the mapping from input appearance to 3D artery structure through paired training data. It therefore produces a plausible prior-guided 3D hypothesis but does not enforce direct projection-space consistency. The following refinement stage uses input view evidence to correct this hypothesis.

![](images/2673d3f31755735c3004a52bd847ea82f0cf9daac83823859f8968b8c7731779.jpg)  
Figure 3: Recursive projection-guided refinement in PPCAR-Net. The B-spline geometry refiner uses local multiview evidence to predict residual updates to the coarse control points. With the refined centreline fixed, the dense-radius refiner then updates the pointwise radii. The two refiners are trained sequentially and separately, using $\mathcal { L } _ { \mathrm { g e o m - r e f } }$ and $\mathcal { L } _ { \mathrm { r a d - r e f } }$ , respectively.

## 3.4 Projection-Guided Refinement

The refinement component of PPCAR-Net, summarised in Figure 3, progressively updates the coarse vessel code $\hat { V } ^ { ( 0 ) }$ using projection-guided geometry and radius corrections. At each refinement stage, the current centreline and radii are projected into every available input view using the same cone-beam operator $\Pi _ { k }$ employed to generate the input projections. Local 2D evidence is sampled around these projected positions and aggregated across views through cross-attention. A residual head then predicts a correction to the current 3D parameter.

For the geometry refiner, at refinement stage s, the query associated with control point $\hat { p } _ { m , j } ^ { ( s ) }$ crossattends to the multiview evidence sampled around its corresponding projected centreline position. The resulting geometry residual $\Delta \hat { p } _ { m , j } ^ { ( s ) }$ updates the control point as $\begin{array} { r } { \hat { \pmb { p } } _ { m , j } ^ { ( s + 1 ) } = \hat { \pmb { p } } _ { m , j } ^ { ( s ) } + \Delta \hat { \pmb { p } } _ { m , j } ^ { ( s ) } } \end{array}$ . The updated B-spline is decoded and reprojected before the next stage samples new evidence. We use $S _ { g } = 4$ shared-parameter geometry-refinement stages while holding the branch-existence predictions and coarse radii fixed. After geometry refinement, the dense-radius refiner applies the same residual principle at every decoded centreline point. A radius query cross-attends to the local multiview evidence and predicts $\Delta \hat { r } _ { m , n } ^ { ( s ) }$ , giving $\bar { r } _ { m , n } ^ { ( s + 1 ) } = \hat { r } _ { m , n } ^ { ( s ) } + \bar { \Delta { r } _ { m , n } ^ { ( s ) } }$ . The refined centreline geometry is held fixed throughout this module. We use $S _ { r } = 3$ shared-parameter radius-refinement stages, producing the final vessel code $\hat { V } _ { \mathrm { r e f i n e d } }$

## 3.5 Staged Training Objectives

PPCAR-Net is trained sequentially: first the coarse predictor, then the geometry refiner with the coarse predictor frozen, and finally the radius refiner with the preceding modules frozen. Because target branches have a deterministic ordering, prediction m is compared directly with ground-truth

branch m. At prediction stage s, the geometry loss supervises both the B-spline control points and the centreline decoded from them:

$$
\mathcal { L } _ { \mathrm { g e o m e t r y } } ^ { ( s ) } = \frac { \lambda _ { \mathrm { c p } } } { N _ { \mathrm { c p } } } \sum _ { m , j } e _ { m } \left\| \hat { \pmb { p } } _ { m , j } ^ { ( s ) } - \pmb { p } _ { m , j } \right\| _ { 2 } ^ { 2 } + \frac { \lambda _ { \mathrm { c l } } } { N _ { \mathrm { c l } } } \sum _ { m , n } e _ { m } \left\| \hat { \pmb { c } } _ { m , n } ^ { ( s ) } - \pmb { c } _ { m , n } \right\| _ { 2 } ^ { 2 } ,\tag{4}
$$

where $N _ { \mathrm { c p } }$ and $N _ { \mathrm { c l } }$ denote the total numbers of control points and decoded centreline points across existing branches, and $\lambda _ { \mathrm { c p } }$ and $\lambda _ { \mathrm { c l } }$ weight the corresponding supervision terms. While the B-spline parameterisation guarantees intra-branch connectivity, the attachment loss encourages connectivity between side branches and their parent branch. Let $\kappa _ { m } \in \{ 1 , \ldots , N \}$ denote the ground-truth index of this junction on the parent branch centreline. We regularise the predicted junction by

$$
\mathcal { L } _ { \mathrm { a t t a c h } } ^ { ( s ) } = \frac { 1 } { \sum _ { m } e _ { m } } \sum _ { m } e _ { m } \left\| \hat { c } _ { m , 1 } ^ { ( s ) } - \hat { c } _ { p , \kappa _ { m } } ^ { ( s ) } \right\| _ { 2 } ^ { 2 } ,\tag{5}
$$

where $p$ denotes the index of the parent branch. This loss directly encourages the predicted first point of each existing side branch to coincide with its corresponding predicted parent branch attachment point. The radius loss is the pointwise mean absolute error at corresponding dense centreline points:

$$
\mathcal { L } _ { \mathrm { r a d i u s } } ^ { ( s ) } = \frac { 1 } { N _ { \mathrm { c l } } } \sum _ { m , n } e _ { m } \left| \hat { r } _ { m , n } ^ { ( s ) } - r _ { m , n } \right| .\tag{6}
$$

Across these losses, $e _ { m }$ excludes absent branches. For the coarse prediction, $s = 0 ,$ and binary cross-entropy $\mathcal { L } _ { \mathrm { e x i s t } }$ supervises the predicted branch-existence labels:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o a r s e } } = \mathcal { L } _ { \mathrm { g e o m e t r y } } ^ { ( 0 ) } + \lambda _ { \mathrm { a t t a c h } } \mathcal { L } _ { \mathrm { a t t a c h } } ^ { ( 0 ) } + \lambda _ { \mathrm { r a d i u s } } \mathcal { L } _ { \mathrm { r a d i u s } } ^ { ( 0 ) } + \lambda _ { \mathrm { e x i s t } } \mathcal { L } _ { \mathrm { e x i s t } } , } \end{array}\tag{7}
$$

Absent branch entries therefore contribute to existence supervision but not to geometry, attachment, or radius supervision. For refinement, the losses used are

$$
\mathcal { L } _ { \mathrm { g e o m - r e f } } = \sum _ { s = 1 } ^ { S _ { g } } \alpha _ { s } ( \mathcal { L } _ { \mathrm { g e o m e t r y } } ^ { ( s ) } + \lambda _ { \mathrm { a t t a c h } } \mathcal { L } _ { \mathrm { a t t a c h } } ^ { ( s ) } ) , \qquad \mathcal { L } _ { \mathrm { r a d - r e f } } = \sum _ { s = 1 } ^ { S _ { r } } \beta _ { s } \mathcal { L } _ { \mathrm { r a d i u s } } ^ { ( s ) } ,\tag{8}
$$

where $\mathcal { L } _ { \mathrm { g e o m - r e f } }$ collects geometry and attachment losses over the recursive geometry stages, while ${ \mathcal { L } } _ { \mathrm { r a d - r e f } }$ collects radius losses over the radius stages. Here $\alpha _ { s }$ and $\beta _ { s }$ are loss weights. These objectives are applied separately: $\mathcal { L } _ { \mathrm { g e o m - r e f } }$ trains only the geometry refiner, whereas $\mathcal { L } _ { \mathrm { r a d - r e f } }$ trains only the radius refiner. The same existence and valid-point masks are applied throughout refinement.

Within-case branch-subset augmentation training. To model optional side branches, each training tree is paired with variants formed by successively removing side branches in reverse canonical order until only the required main path remains. Updated projections, existence labels, and geometry targets from all variants train the coarse predictor and geometry refiner, varying branch presence while retaining the same underlying anatomy.

## 3.6 Evaluation Metrics

We evaluate reconstruction accuracy using Dice, centreline Dice (clDice), Chamfer distance (CD), and connected-component count (CC count). Dice measures the volumetric overlap between the predicted and ground-truth 3D arteries and is reported as a percentage, making it sensitive to both vessel location and lumen extent. For volumetric evaluation, the prediction and ground-truth artery are represented on the same isotropic grid with a voxel spacing of 0.5 mm along each axis. clDice measures topology-aware agreement between the skeletons of the voxelised prediction and ground-truth artery. CD measures the geometric discrepancy in millimetres between the predicted and ground-truth centreline point sets. Each evaluated RCA or LCA tree forms one connected vascular structure because every retained side branch attaches to its corresponding main branch;

consequently, its ground-truth volume contains exactly one connected component. To evaluate prediction fragmentation, each method’s native output is converted to the same voxel grid and its connected foreground components are counted. The ideal CC count is therefore one, while larger values indicate disconnected predicted artery segments. Dice and clDice provide complementary volumetric and centreline-topology assessments, CD assesses centreline localisation, and CC count directly evaluates whether the prediction preserves the expected connected artery structure. Higher Dice and clDice values, a lower CD, and a lower CC count indicate better reconstruction. Every metric is first calculated per case and then summarised over the test cases with mean and standard error.

Additional data-construction and architectural details are provided in Appendix Sections A.1 and A.2, respectively.

## 4 Results

## 4.1 Comparison model selection

We quantitatively compare PPCAR-Net with 3DGR-CAR (Fu et al., 2024), AutoCAR (Zhu et al., 2025), and DeepCA (Wang et al., 2025b); the latter two are evaluated only in their fixed two-view setting. NeRF-CA (Maas et al., 2025) and NerT-CA (Maas et al., 2026) generate novel X-ray-like views rather than 3D arteries; the method of Iyer et al. (2023) operates on uncalibrated input views and assumes a fixed number of artery branches; therefore these methods are not evaluated. SDF-CAR reconstructs a 3D artery without cross-case pre-training, but its per-patient optimisation requires approximately 45 minutes per case (Reda et al., 2026), and thus is excluded from quantitative comparison.

## 4.2 RCA and LCA Reconstruction

Table 2 evaluates reconstruction from one, two, and four input views on the 75-case RCA test split described in Section 3.2, using Dice, clDice, Chamfer distance, and connected-component count. For methods without a native centreline graph, we extract the centreline using the post-processing approach of Zhu et al. (2025). All methods use the same view subsets and evaluation protocol.

Compared with the other methods in Table 2, our refined model outperforms 3DGR-CAR in the one-view setting. Under the conventional two-view setting, AutoCAR achieves the highest Dice, whereas our method achieves the highest clDice and lowest Chamfer distance, indicating stronger centreline-topology agreement and localisation despite the small difference in volumetric overlap. With four views, our refined model substantially outperforms 3DGR-CAR across Dice, clDice, and Chamfer distance. Dice alone does not fully capture vessel connectivity because disconnected foreground regions may still overlap with the target artery. As each branch centreline is connected by construction and attachment supervision encourages side branches to join the main branch, our refined predictions achieve the lowest CC count at every evaluated view setting and remain close to the ideal value of one, while the comparison methods produce more fragmented outputs.

Table 3 reports reconstruction from the same view counts on the 75-case LCA test split, using the same metrics and evaluation protocol. The LCA task is more challenging than RCA because its representation supports up to 13 branches, increasing anatomical complexity, projected branch overlap, and ambiguity under sparse views. In the conventional two-view setting, DeepCA achieves the highest Dice and clDice, while our method has lower Chamfer distance than AutoCAR and DeepCA and substantially fewer connected components (1.053 versus 11.36 and 9.97, respectively). AutoCAR and DeepCA backproject the observed views into volumetric representations, which can favour agreement with the input silhouettes and voxel-overlap metrics. In contrast, the connectedcomponent results and the qualitative examples in Figure 4 show that our explicit representation more consistently preserves a coherent artery-tree structure. Under one-view input, 3DGR-CAR achieves higher overlap scores but produces 46.43 connected components, compared with 1.197 for our method. Its predicted foreground volume is also substantially larger than the ground truth, indicating that part of its overlap arises from broad, fragmented coverage. Although a single projection does not determine depth, our model still produces a nearly connected artery-shaped 3D hypothesis rather than a fragmented volume. With four views, our method outperforms 3DGR-CAR in Dice, clDice, and connectivity, although its Chamfer distance remains higher.

Table 2: RCA reconstruction from one, two, and four input views. CD denotes Chamfer distance, and CC count denotes connected-component count. Values are mean ± standard error. The best result is shown in bold and the second-best result is underlined. – denotes a view-count setting that is not reported in the corresponding paper.
<table><tr><td rowspan="2">Method</td><td colspan="4">One view</td><td colspan="4">Two views</td><td colspan="4">Four views</td></tr><tr><td>Dice (%)↑</td><td>clDice (%)↑</td><td>CD (mm) ↓</td><td>CC count ↓</td><td>Dice (%)↑</td><td>clDice (%)↑</td><td>CD (mm) ↓</td><td>CC count ↓</td><td>Dice (%)↑</td><td>clDice (%)↑</td><td>CD (mm) ↓</td><td>CC count↓</td></tr><tr><td>3DGR-CAR (Fu et al., 2024)</td><td>22.18 (±0.65)</td><td>22.35 (±0.73)</td><td>10.11 (±0.44)</td><td>72.77 (±12.08)</td><td>37.32 (±0.91)</td><td>39.47 (±1.09)</td><td>7.72 (±0.48)</td><td>159.62 (±16.84)</td><td>44.33 (±1.12)</td><td>46.54 (±1.31)</td><td>7.37 (±0.49)</td><td>216.25 (±17.95)</td></tr><tr><td>AutoCAR (Zhu et al., 2025)</td><td></td><td></td><td></td><td></td><td>68.76 (±0.70)</td><td>79.13 (±1.37)</td><td>3.24 (±0.27)</td><td>3.35 (±0.31)</td><td></td><td></td><td></td><td></td></tr><tr><td>DeepCA (Wang et al., 2025b)</td><td></td><td></td><td></td><td></td><td>63.49 (±1.56)</td><td>78.00 (±1.73)</td><td>3.78 (±0.51)</td><td>4.41 (±0.60)</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>1.324</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.284</td></tr><tr><td>PPCAR-Net (coarse)</td><td>15.73 (±1.46)</td><td>16.43 (±1.84)</td><td>11.86 (±0.59)</td><td>(±0.072)</td><td>19.47 (±1.12)</td><td>20.40 (±1.56)</td><td>9.70 (±0.41)</td><td>1.324 (±0.064)</td><td>21.45 (±1.26)</td><td>22.88 (±1.70)</td><td>9.16 (±0.36)</td><td>(±0.059)</td></tr><tr><td>PPCAR-Net</td><td>27.37</td><td>32.38</td><td>9.39</td><td>1.081</td><td>66.92</td><td>80.62</td><td>3.16</td><td>1.054</td><td>73.05</td><td>88.54</td><td>2.12</td><td>1.014</td></tr><tr><td>(refined)</td><td>(±1.86)</td><td>(±2.56)</td><td>(±0.55)</td><td>(±0.032)</td><td>(±1.50)</td><td>(±1.92)</td><td>(±0.27)</td><td>(±0.026)</td><td>(±1.25)</td><td>(±1.45)</td><td>(±0.19)</td><td>(±0.014)</td></tr></table>

Table 3: LCA reconstruction from one, two, and four input views. CD denotes Chamfer distance, and CC count denotes connected-component count. Values are mean ± standard error. The best result is shown in bold and the second-best result is underlined. – denotes a view-count setting that is not reported in the corresponding paper.
<table><tr><td rowspan="2">Method</td><td colspan="4">One view</td><td colspan="4">Two views</td><td colspan="4">Four views</td></tr><tr><td>Dice (%)↑</td><td>clDice (%)↑</td><td>CD (mm) ↓</td><td>CC count ↓</td><td>Dice (%)↑</td><td>clDice (%)↑</td><td>CD (mm) ↓</td><td>CC count↓</td><td>Dice (%)↑</td><td>clDice (%)↑</td><td>CD (mm) ↓</td><td>CC count ↓</td></tr><tr><td>3DGR-CAR (Fu et al., 2024)</td><td>26.51 (±0.51)</td><td>26.48 (±0.56)</td><td>8.25 (±0.17)</td><td>46.43 (±3.54)</td><td>35.05 (±0.48)</td><td>37.16 (±0.54)</td><td>5.87 (±0.11)</td><td>76.10 (±8.06)</td><td>43.64 (±0.66)</td><td>47.81 (±0.75)</td><td>4.94 (±0.20)</td><td>100.88 (±11.59)</td></tr><tr><td>AutoCAR (Zhu et al., 2025)</td><td></td><td></td><td></td><td></td><td>46.89 (±1.29)</td><td>63.00 (±1.59)</td><td>6.13 (±0.38)</td><td>11.36 (±0.78)</td><td></td><td></td><td></td><td></td></tr><tr><td>DeepCA (Wang et al., 2025b)</td><td>一</td><td></td><td>一</td><td></td><td>56.56 (±0.81)</td><td>71.20 (±1.11)</td><td>6.66 (±0.24)</td><td>9.97 (±0.32)</td><td></td><td>一</td><td></td><td>一</td></tr><tr><td></td><td></td><td></td><td></td><td>1.829</td><td></td><td></td><td></td><td>1.605</td><td></td><td></td><td>12.46</td><td>1.618</td></tr><tr><td>PPCAR-Net (coarse)</td><td>11.15 (±0.76)</td><td>10.10 (±0.81)</td><td>14.08 (±0.51)</td><td>(±0.118)</td><td>14.92 (±0.77)</td><td>13.94 (±0.90)</td><td>12.37 (±0.51)</td><td>(±0.097)</td><td>14.59 (±0.73)</td><td>14.05 (±0.93)</td><td>(±0.49)</td><td>(±0.095)</td></tr><tr><td>PPCAR-Net</td><td>19.73</td><td>21.36</td><td>11.31</td><td>1.197</td><td>49.45</td><td>61.30</td><td>5.98</td><td>1.053</td><td>53.45</td><td>67.43</td><td>5.30</td><td>1.039</td></tr><tr><td>(refined)</td><td>(±1.00)</td><td>(±1.19)</td><td>(±0.45)</td><td>(±0.050)</td><td>(±1.32)</td><td>(±1.93)</td><td>(±0.47)</td><td>(±0.026)</td><td>(±1.28)</td><td>(±1.93)</td><td>(±0.42)</td><td>(±0.022)</td></tr></table>

## 4.3 Ablation Studies

Our ablation studies examine the contributions of projection-guided refinement, B-spline parameterisation, and branch-subset augmentation in PPCAR-Net. Starting from the same coarse prediction, direct per-scene optimisation improves reconstruction accuracy but underperforms learned refinement across all four reconstruction metrics for both RCA and LCA (Table 4). Direct optimisation fits only the observed 2D masks, which may admit geometrically implausible solutions under sparse-view ambiguity. In contrast, the learned refiners use 3D supervision across training cases to predict residual corrections from projection evidence. The combined geometry and radius refiner achieves 66.92% RCA Dice, compared with 41.02% for direct optimisation, while requiring only 61 ms rather than 11 s after coarse prediction.

B-spline parameterisation and branch-subset augmentation further improve reconstruction. Replacing direct dense-centreline prediction with B-spline control-point prediction increases two-view RCA refined Dice from 22.18% to 66.92% (Appendix Table 7). Branch-subset augmentation increases geometry-refined Dice from 46.57% to 65.89%, branch-existence accuracy from 92.95% to 94.63%, and F1 from 88.37% to 92.48%, supporting more reliable identification of optional side branches (Appendix Table 9).

Table 4: Refinement ablation in PPCAR-Net from a common two-view coarse initialisation. CD denotes Chamfer distance, and CC count denotes connected-component count. Values are mean ± standard error; post-coarse time excludes the shared coarse prediction. The best result is shown in bold and the second-best result is underlined.
<table><tr><td rowspan="2">Reconstruction stage</td><td colspan="4">RCA</td><td colspan="4">LCA</td><td rowspan="2">Post-coarse time ↓</td></tr><tr><td>Dice (%)↑</td><td>clDice (%) ↑</td><td>CD (mm) ↓</td><td>CC count ↓</td><td>Dice (%)↑</td><td>clDice (%)↑</td><td>CD (mm) ↓</td><td>CC count ↓</td></tr><tr><td>Coarse prediction</td><td>19.47 (±1.12)</td><td>20.40 (±1.56)</td><td>9.70 (±0.41)</td><td>1.324 (±0.064)</td><td>14.92 (±0.77)</td><td>13.94 (±0.90)</td><td>12.37 (±0.51)</td><td>1.605 (±0.097)</td><td></td></tr><tr><td>Direct per-scene optimisation</td><td>41.02 (±1.62)</td><td>50.02 (±2.24)</td><td>6.65 (±0.39)</td><td>1.243 (±0.057)</td><td>28.75 (±1.12)</td><td>33.16 (±1.51)</td><td>9.29 (±0.50)</td><td>1.421 (±0.073)</td><td>11 s</td></tr><tr><td>Learned geometry refiner</td><td>65.89 (±1.44)</td><td>80.84 (±1.90)</td><td>3.16 (±0.27)</td><td>1.068 (±0.029)</td><td>48.43 (±1.29)</td><td>61.38 (±1.93)</td><td>5.98 (±0.47)</td><td>1.053 (±0.026)</td><td>14 ms</td></tr><tr><td>Learned geometry and radius refiners</td><td>66.92 (±1.50)</td><td>80.62 (±1.92)</td><td>3.16 (±0.27)</td><td>1.054 (±0.026)</td><td>49.45 (±1.32)</td><td>61.30 (±1.93)</td><td>5.98 (±0.47)</td><td>1.053 (±0.026)</td><td>61 ms</td></tr></table>

![](images/c20ce365e2b0012bc3a102a0f6dc964d86553a9bd7af3397a467bc8bcf942b30.jpg)  
Figure 4: Qualitative two-view reconstruction. Columns show the input masks, 3D label, and the method outputs; rows show three RCA cases and three LCA cases, each rendered from two viewpoints. Arrows mark branches poorly visible in the input projections.

## 4.4 Qualitative Reconstruction and Transfer to Real X-ray Angiography

Figure 4 qualitatively illustrates these connectivity differences by showing gaps in voxel and implicit predictions. Subsequent graph extraction may fail to fill the gaps in the predicted artery. In contrast, our continuous branch representation preserves within-branch connectivity and exposes branch attachments and radii. The arrows in Figure 4 mark branches that are weakly visible in the inputs but recovered by our model, suggesting that learned anatomical patterns complement sparse projection evidence. Figure 5(A) shows that the coarse prediction captures the global artery shape, while projection-guided refinement uses 2D evidence to recover the corresponding radius reduction. This indicates sensitivity to visible narrowing, rather than clinical stenosis detection. Figure 5(B) demonstrates application to preprocessed real angiography, although the absence of paired 3D labels prevents accuracy assessment. Additional qualitative video results are available on our project page, accessible through the GitHub repository.

Implementation details and comparison-method adaptations are described in Appendix Sections A.3.1 and A.3.3, respectively. An additional ablation of the frozen image backbone and an inference-time breakdown for the comparison methods are reported in Appendix Tables 8 and 10.

![](images/440d7cbfc7bc56ea6be4e9bd1eaad917db76546a05812d16852adbbeebd5d68b.jpg)  
Figure 5: Qualitative reconstruction with PPCAR-Net on diseased CT anatomy and real X-ray angiography. (A) A diseased ASOCA (Gharleghi et al., 2023) case with a visually identified region of local artery narrowing. The 3D label, simulated masks, and radius-coloured coarse and refined predictions are shown. (B) A real two-view X-ray angiography case from the AutoCAR dataset (Zhu et al., 2025). The middle column overlays the prediction on each angiographic view, and the final column shows the radius-coloured 3D prediction.

## 5 Discussion and Conclusion

We introduce PPCAR-Net, a coarse-to-fine network that combines learned coronary anatomical patterns with projection-guided corrections to predict an explicit vessel code. Its explicit vessel-code representation guarantees a connected centreline within each branch, while attachment supervision produces predictions close to a single connected artery tree, a desirable property for clinical interpretation and downstream geometric analysis. The feed-forward pipeline enables sub-second reconstruction without per-case optimisation.

Figure 5(B) provides an initial qualitative application to real X-ray coronary angiography, but it does not constitute systematic clinical validation. Sequentially acquired clinical monoplane views can retain cardiac deformation even after phase matching. Future work should evaluate larger clinical cohorts and address inter-view misalignment caused by motion and gantry displacement (Tu et al., 2010) through motion-aware reconstruction and geometry calibration. A central open question is how to evaluate 3D reconstruction accuracy without a 3D artery label; paired acquisitions of CT and X-ray coronary angiography or expert assessment may therefore be required.

The learned coronary anatomy patterns can support a plausible reconstruction when projection evidence is incomplete, but they cannot reliably identify a specific stenosis that is concealed by overlap or foreshortening in every available view. Moreover, without clinical stenosis annotations, our current examples assess radius reconstruction rather than diagnostic stenosis detection. Further studies using expert lesion annotations and physiological measurements such as fractional flow reserve are therefore required to determine whether reconstructed radius changes correspond to clinically significant stenoses.

Finally, the present model requires known 3D artery labels for supervised training. Developing unsupervised or self-supervised objectives from real angiographic views could reduce this dependence; the current model may also provide initial pseudo-3D labels for datasets that contain only 2D X-ray coronary angiography images.

## References

Xueming Fu, Yingtai Li, Fenghe Tang, Jun Li, Mingyue Zhao, Gao-Jun Teng, and S. Kevin Zhou. 3DGR-CAR: Coronary artery reconstruction from ultra-sparse 2d x-ray views with a 3d gaussians representation. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, volume 15007 of Lecture Notes in Computer Science, pages 14–24. Springer Nature Switzerland, 2024. doi: 10.1007/978-3-031-72104-5\_2.

Ramtin Gharleghi, Dona Adikari, Katy Ellenberger, Michael Webster, Chris Ellis, Arcot Sowmya, Sze-Yuan Ooi, and Susann Beier. Annotated computed tomography coronary angiogram images and associated data of normal and diseased arteries. Scientific Data, 10(1):128, 2023. doi: 10.1038/s41597-023-02016-2.

Zhanqiang Guo, Zimeng Tan, Jianjiang Feng, and Jie Zhou. VesselDiffusion: 3d vascular structure generation based on diffusion model. IEEE Transactions on Medical Imaging, 44(9):3845–3857, 2025. doi: 10.1109/TMI.2025.3568602.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016. URL https://openaccess.thecvf.com/content\_cvpr\_ 2016/html/He\_Deep\_Residual\_Learning\_CVPR\_2016\_paper.html.

Kritika Iyer, Brahmajee K. Nallamothu, C. Alberto Figueroa, and Raj R. Nadakuditi. A multi-stage neural network approach for coronary 3d reconstruction from uncalibrated x-ray angiography images. Scientific Reports, 13:17603, 2023. doi: 10.1038/s41598-023-44633-2.

A. K. Klein, F. Lee, and A. A. Amini. Quantitative coronary angiography with deformable spline models. IEEE Transactions on Medical Imaging, 16(5):468–482, 1997. doi: 10.1109/42.640737.

J. L. Klein, J. G. Hoff, J. W. Peifer, R. Folks, C. D. Cooke, S. B. King, and E. V. Garcia. A quantitative evaluation of the three dimensional reconstruction of patients’ coronary arteries. International Journal ofCardiac Imaging, 14(2):75–87, 1998. doi: 10.1023/A:1005903705300.

Kirsten W. H. Maas, Danny Ruijters, Anna Vilanova, and Nicola Pezzotti. NeRF-CA: Dynamic reconstruction of x-ray coronary angiography with extremely sparse-views. IEEE Transactions on Visualization and Computer Graphics, 31(10):8782–8795, 2025. doi: 10.1109/TVCG.2025. 3579162.

Kirsten W. H. Maas, Danny Ruijters, Nicola Pezzotti, and Anna Vilanova. NerT-CA: Efficient dynamic reconstruction from sparse-view x-ray coronary angiography. In Reconstruction, Imaging, and Mixed Reality in Healthcare / Graphs in Biomedical Image Analysis, volume 16150 of Lecture Notes in Computer Science, pages 13–22, 2026. doi: 10.1007/978-3-032-06103-4\_2.

Ahmed Reda, Mohamed Ashraf, Mohamed G. Abdelkader, Yasser Abuouf, and Muhamed Albadawi. SDF-CAR: 3d coronary artery reconstruction from two views with a hybrid SDFoccupancy implicit representation. Full paper under review for MIDL 2026, 2026. URL https://openreview.net/pdf?id=bfjv51bxKJ.

Guy Shechter, Frédéric Devernay, Eve Coste-Manière, Arshed Quyyumi, and Elliot R. McVeigh. Three-dimensional motion tracking of coronary arteries in biplane cineangiograms. IEEE Transactions on Medical Imaging, 22(4):493–503, 2003. doi: 10.1109/TMI.2003.809090.

Shengxian Tu, Gerhard Koning, Wouter Jukema, and Johan H. C. Reiber. Assessment of obstruction length and optimal viewing angle from biplane x-ray angiograms. The International Journal of Cardiovascular Imaging, 26(1):5–17, 2010. doi: 10.1007/s10554-009-9509-3.

Arso M. Vukicevic, Serkan Cimen, Nikola Jagic, Goran Jovicic, Alejandro F. Frangi, and Nenad Filipovic. Three-dimensional reconstruction and NURBS-based structured meshing of coronary arteries from conventional x-ray angiography projection images. Scientific Reports, 8(1):1711, 2018. doi: 10.1038/s41598-018-19440-9.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5294–5306, 2025a. URL https://openaccess.thecvf.com/content/CVPR2025/html/Wang\_VGGT\_ Visual\_Geometry\_Grounded\_Transformer\_CVPR\_2025\_paper.html.

Yiying Wang, Abhirup Banerjee, Robin P. Choudhury, and Vicente Grau. DeepCA: Deep learningbased 3d coronary artery tree reconstruction from two 2d non-simultaneous x-ray angiography projections. In Proceedings ofthe Winter Conference on Applications ofComputer Vision, pages 337–346, 2025b. URL https://openaccess.thecvf.com/content/WACV2025/ html/Wang\_DeepCA\_Deep\_Learning-Based\_3D\_Coronary\_Artery\_Tree\_ Reconstruction\_from\_Two\_WACV\_2025\_paper.html.

An Zeng, Chunbiao Wu, Guisen Lin, Wen Xie, Jin Hong, Meiping Huang, Jian Zhuang, Shanshan Bi, Dan Pan, Najeeb Ullah, Kaleem Nawaz Khan, Tianchen Wang, Yiyu Shi, Xiaomeng Li, and Xiaowei Xu. ImageCAS: A large-scale dataset and benchmark for coronary artery segmentation based on computed tomography angiography images. Computerized Medical Imaging and Graphics, 109:102287, 2023. doi: 10.1016/j.compmedimag.2023.102287.

Yinheng Zhu, Yong Wang, Chunxia Di, Hanghang Liu, Fangzhou Liao, and Shaohua Ma. Sparse and transferable three-dimensional dynamic vascular reconstruction for instantaneous diagnosis. Nature Machine Intelligence, 7:730–742, 2025. doi: 10.1038/s42256-025-01025-7.

## A Appendix

## A.1 Dataset and Target-Construction Details

The ImageCAS coronary annotations (Zeng et al., 2023) are provided as NIfTI volumes. We use the $x \cdot , y \cdot$ , and z-axis spacing metadata to express all centreline coordinates, radii, and distances in millimetres. Starting from the 1,000 source cases, each labelled artery volume is automatically separated into RCA and LCA components and every separated component is inspected manually. A component is excluded if its RCA/LCA identity cannot be determined reliably, if the two artery systems cannot be separated, or if annotation defects prevent construction of a valid target. The resulting RCA and LCA collections are component cohorts rather than disjoint patient cohorts: retained RCA and LCA components may originate from the same ImageCAS case, so their counts do not represent completely different patients. These exclusions may bias the retained cohorts towards cases with clearer artery annotations and should be considered when interpreting generalisation to the complete clinical population.

Retained components are skeletonised in physical coordinates and converted into an initial centreline graph using local node connections and a minimum spanning tree, following the broad graph-construction procedure used by AutoCAR (Zhu et al., 2025). Because automatic conversion can fail for tortuous vessels, rapidly changing radii, and annotation defects, manual supervision removes extra branches, repairs erroneous connections, and verifies branch identity and ordering. Difficult geometry is retained whenever a reliable target can be produced after correction. For each side branch, the first centreline point is set to its attachment point on the corresponding main branch, explicitly encoding branch connectivity in the target vessel code. Radius is obtained from the distance transform of the CT artery mask, with each cross-section approximated by a circular lumen profile. Every branch is resampled from its proximal root or attachment to its distal endpoint to produce N = 200 ordered centreline-and-radius samples and is fitted with $K _ { \mathrm { c p } } = 2 0 { \bf B }$ -spline control points. This quality-control process produces 750 RCA and 750 LCA cases, split at patient level into 600 training, 75 validation, and 75 test cases.

For the qualitative narrowing example in Figure 5(A), we additionally use one diseased case from the ASOCA dataset (Gharleghi et al., 2023). ASOCA provides 40 expert-annotated CT coronary angiography volumes, comprising 20 healthy participants and 20 participants with confirmed coronary artery disease. This ASOCA case is used only for qualitative evaluation and is not included in model training or the quantitative test cohorts.

For RCA targets, the RCA trunk is the first branch and retained side branches are sorted by the normalised arc-length position of their attachment point while traversing the RCA trunk from proximal to distal. For LCA targets, the LM–LAD path is first, the LCX path is second, and the remaining side branches are sorted first along LAD and then along LCX, again by proximal-to-distal attachment position. Every retained ImageCAS LCA case contains a distinct LM artery. The ordering gives each optional side branch a deterministic target entry without assigning it a fixed anatomical name. For RCA, we retained a maximum of 7 branches; for LCA, we retained a maximum of 13 branches. For each case of RCA or LCA, unused side-branch entries are zero-padded and assigned an existence label of zero.

Seven clinically motivated projection anchors are defined separately for RCA and LCA, as listed in Table 5. Here, RAO and LAO denote the primary right and left anterior oblique rotations, while CRA and CAU denote the secondary cranial and caudal angulations; AP denotes zero primary rotation. These values are anchor directions rather than the only admissible acquisition angles, so nearby angular settings belong to the same clinical view family. For both RCA and LCA, the primary and secondary angles are independently varied within $\pm 1 0 ^ { \circ }$ of each artery-specific anchor.

Patient coordinates and projection centre. Vessel coordinates use the source-volume left–anterior– superior (LAS) patient frame: the positive x, y, and z axes point towards the patient’s left, anterior, and superior directions, respectively. No additional standing–supine rotation is applied. For each case, the projection isocentre is placed at the centroid of the reference artery surface. The reference is the main RCA surface for RCA and the combined LM–LAD and LCX surfaces for LCA. If this centroid is o in millimetres, the camera receives the centred point $\bar { \pmb { x } } = 1 0 ^ { - 3 } ( \pmb { x } - \pmb { o } )$ in metres, so the isocentre is the world origin ${ \boldsymbol { O } } = ( 0 , 0 , 0 ) ^ { \top }$

Table 5: Clinical acquisition-direction anchors used to generate the seven candidate projections for each artery. Each entry reports the primary RAO/LAO rotation followed by the secondary CRA/CAU angulation.
<table><tr><td>View</td><td>RCA anchor</td><td>LCA anchor</td></tr><tr><td>1</td><td> $\mathrm { L A O 4 0 ^ { \circ } , C R A 1 0 ^ { \circ } }$ </td><td> $\mathrm { R A O 2 5 ^ { \circ } , C A U 3 5 ^ { \circ } }$ </td></tr><tr><td>2</td><td> $\mathrm { R A O 7 5 ^ { \circ } , C R A 1 0 ^ { \circ } }$ </td><td> $\mathrm { L A O 5 ^ { \circ } , C A U 3 0 ^ { \circ } }$ </td></tr><tr><td>3</td><td> $\mathrm { A P , C R A 2 5 ^ { \circ } }$ </td><td> $\mathrm { A P , C A U 1 0 ^ { \circ } }$ </td></tr><tr><td>4</td><td> $\mathrm { R A O 2 0 ^ { \circ } , C R A 2 0 ^ { \circ } }$ </td><td> $\mathrm { R A O } 4 0 ^ { \circ } , \mathrm { C R A 2 5 ^ { \circ } }$ </td></tr><tr><td>5</td><td> $\mathrm { L A O 2 0 ^ { \circ } }$ </td><td> $\mathrm { L A O 2 0 ^ { \circ } , C R A 3 0 ^ { \circ } }$ </td></tr><tr><td>6</td><td> $\mathrm { R A O 7 5 ^ { \circ } , C A U 2 0 ^ { \circ } }$ </td><td> $\mathrm { L A O 4 0 ^ { \circ } , C R A 2 0 ^ { \circ } }$ </td></tr><tr><td>7</td><td> $\mathrm { L A O 6 0 } ^ { \circ }$ </td><td> $\mathrm { L A O 5 ^ { \circ } , C R A 4 0 ^ { \circ } }$ </td></tr></table>

Clinical angles and camera rotation. We use a positive clinical primary angle for LAO and a negative angle for $\mathrm { R A O ; \mathfrak { a } }$ positive secondary angle denotes CRA and a negative angle denotes CAU. AP has zero primary angle. The preprocessing converts these clinical angles $( a _ { k } , b _ { k } )$ to the stored projector angles $( \theta _ { k } , \phi _ { k } )$ . The conversion is $\theta _ { k } = \mathrm { w r a p } ( 9 0 ^ { \circ } - a _ { k } )$ ) and $\phi _ { k } = 9 0 ^ { \circ } - b _ { k }$ , where wrap maps an angle to $[ - 1 8 0 ^ { \circ } , 1 8 0 ^ { \circ } )$

Source, detector, and projection. The cone-beam projector uses a 256 × 256-pixel flat detector, a source-to-detector distance of 0.90 m, and a source-to-isocentre distance of 0.75 m. The detector pixel spacing is 0.55 mm/pixel for RCA and 0.65 mm/pixel for LCA.

## A.2 Model Details, Training, and Evaluation Protocols

This section provides additional architectural, target-fitting, and training details for PPCAR-Net, together with the test-time optimisation baseline.

## A.2.1 B-spline Representation and Target Fitting

For each artery branch $m ,$ the ordered dense centreline samples $\{ c _ { m , n } \} _ { n = 1 } ^ { N }$ are parameterised by a cubic B-spline with $K _ { \mathrm { c p } } = 2 0$ controls. The fixed basis matrix $\pmb { A } \in \mathbb { R } ^ { N \times K _ { \mathrm { c p } } }$ is evaluated at $t _ { n } = ( n - 1 ) / ( N - 1 )$ and has entries $A _ { n , j } = B _ { j , 3 } ( t _ { n } )$ . The basis uses the open-uniform clamped knot vector $\pmb { u } = ( 0 , 0 , 0 , 0 , 1 / 1 7 , . . . , 1 6 / 1 7 , 1 , 1 , 1 , 1 , 1 )$ . The branch control matrix is fitted offline by endpoint-constrained least squares:

$$
\begin{array} { c } { { P _ { m } ^ { \star } = \arg \underset { { P _ { m } } } { \operatorname* { m i n } } \left\| \boldsymbol C _ { m } - A { P _ { m } } \right\| _ { F } ^ { 2 } , } } \\ { { \mathrm { s u b j e c t \ t o } \quad A _ { 1 , : } P _ { m } = c _ { m , 1 } , } } \\ { { \quad \quad \quad A _ { N , : } P _ { m } = c _ { m , N } , } } \end{array}\tag{9}
$$

where $\boldsymbol { P _ { m } } \in \mathbb { R } ^ { K _ { \mathrm { c p } } \times 3 }$ is the control-point matrix, $C _ { m } \in \mathbb { R } ^ { N \times 3 }$ contains the dense target coordinates, and $\| \cdot \| _ { F }$ is the Frobenius norm. The decoded approximation is $\widetilde { C } _ { m } = A P _ { m } ^ { \star }$

Before training the image-to-artery model, we test whether the compact spline can represent the accepted ground-truth vessels. For each candidate control count, target controls are fitted and decoded to $N = 2 0 0$ centreline points. Fitting error is the mean paired Euclidean distance between decoded and original points at corresponding normalised-arc-length positions. Vessel clDice is computed using the unchanged dense pointwise radii, thereby isolating centreline-parameterisation fidelity. Table 6 quantifies the approximation–compactness trade-off. With $K _ { \mathrm { c p } } = 2 0$ , the overall fitting error is $0 . 1 6 4 \pm 0 . 1 0 8$ mm and vessel clDice is $9 9 . 9 2 \pm 0 . 6 2 \%$ , using 60 centreline parameters per branch instead of 600 dense coordinates. Table 7 complements this offline analysis by evaluating whether the B-spline parameterisation also benefits learned two-view coarse prediction and projection-guided refinement relative to directly predicting and refining all N = 200 dense centreline points.

Table 6: Offline B-spline representation fidelity. Entries are mean ± sample standard deviation. Centreline fitting error is reported in millimetres and vessel clDice as a percentage. The RCA and LCA statistics each use 750 cases. The overall statistic pools 1,500 source-case means, with each LCA case contributing the arithmetic mean of its LM–LAD and LCX branch metrics.
<table><tr><td rowspan="2"> $K _ { \mathrm { c p } }$ </td><td colspan="4">Centreline fitting error (mm) ↓</td><td colspan="4">Vessel clDice (%) ↑</td></tr><tr><td>RCA</td><td>LM-LAD</td><td>LCX</td><td>Overall</td><td>RCA</td><td>LM-LAD</td><td>LCX</td><td>Overall</td></tr><tr><td>8</td><td>1.18</td><td>0.65</td><td>0.81</td><td>0.95</td><td>86.21</td><td>96.98</td><td>92.78</td><td>90.55</td></tr><tr><td></td><td>(±0.72)</td><td>(±0.40)</td><td>(±0.38)</td><td>(±0.60)</td><td>(±17.91)</td><td>(±7.11)</td><td>(±9.66)</td><td>(±14.21)</td></tr><tr><td></td><td>0.56</td><td>0.31</td><td>0.46</td><td>0.47</td><td>97.35</td><td>99.47</td><td>98.01</td><td>98.04</td></tr><tr><td>12</td><td>(±0.41)</td><td>(±0.21)</td><td>(±0.26)</td><td>(±0.33)</td><td>(±6.61)</td><td>(±2.24)</td><td>(±4.45)</td><td>(±5.08)</td></tr><tr><td></td><td>0.19</td><td>0.11</td><td>0.17</td><td>0.16</td><td>99.88</td><td>100.00</td><td>99.94</td><td>99.92</td></tr><tr><td>20</td><td>(±0.13)</td><td>(±0.08)</td><td>(±0.11)</td><td>(±0.11)</td><td>(±0.84)</td><td>(±0.06)</td><td>(±0.49)</td><td>(±0.62)</td></tr><tr><td>Dense coordinates</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr></table>

Table 7: Effect of centreline representation on coarse and refined reconstruction using two input views. The dense-centreline variant directly predicts and refines the $N = 2 0 0$ ordered centreline points, whereas the B-spline variant predicts and refines $K _ { \mathrm { c p } } = 2 0$ control points before decoding them to the same number of dense points. All other model and evaluation settings are unchanged. CD denotes Chamfer distance, and CC count denotes connected-component count. Values are mean ± standard error; bold indicates the better representation within each matched prediction stage.
<table><tr><td rowspan="2">Stage</td><td rowspan="2">Centreline representation</td><td colspan="4">RCA</td><td colspan="4">LCA</td></tr><tr><td>Dice (%) ↑</td><td>clDice (%) ↑</td><td>CD (mm) ↓</td><td>CC count ↓</td><td>Dice (%) ↑</td><td>clDice (%) ↑</td><td>CD (mm) ↓</td><td>CC count ↓</td></tr><tr><td rowspan="3">Coarse</td><td rowspan="3">Dense centreline</td><td>10.22</td><td>10.23</td><td>13.81</td><td>1.284</td><td>8.97</td><td>8.29</td><td>16.09</td><td>2.276</td></tr><tr><td>(±0.74)</td><td>(±0.98)</td><td>(±0.46)</td><td>(±0.056)</td><td>(±0.53)</td><td>(±0.76)</td><td>(±0.59)</td><td>(±0.125)</td></tr><tr><td>19.47</td><td>20.40</td><td>9.70</td><td>1.324</td><td>14.92</td><td>13.94</td><td>12.37</td><td>1.605</td></tr><tr><td rowspan="3"></td><td rowspan="3">B-spline Dense centreline</td><td>(±1.12)</td><td>(±1.56)</td><td>(±0.41)</td><td>(±0.064)</td><td>(±0.77)</td><td>(±0.90)</td><td>(±0.51)</td><td>(±0.097)</td></tr><tr><td>22.18</td><td>26.87</td><td>10.36</td><td>1.189</td><td>15.21</td><td>15.77</td><td>13.87</td><td>1.526</td></tr><tr><td>(±1.07)</td><td>(±1.57)</td><td>(±0.47)</td><td>(±0.050)</td><td>(±0.68)</td><td>(±0.91)</td><td>(±0.58)</td><td>(±0.087)</td></tr><tr><td rowspan="2">Refined</td><td>B-spline</td><td>66.92</td><td>80.62</td><td>3.16</td><td>1.054</td><td>49.45</td><td>61.30</td><td>5.98</td><td>1.053</td></tr><tr><td></td><td>(±1.50)</td><td>(±1.92)</td><td>(±0.27)</td><td>(±0.026)</td><td>(±1.32)</td><td>(±1.93)</td><td>(±0.47)</td><td>(±0.026)</td></tr></table>

## A.2.2 Coarse Model Encoder Details

This section provides the implementation details of the frozen VGGT encoder used by the coarse predictor in Subsection 3.3.

VGGT backbone. The VGGT backbone (Wang et al., 2025a) processes each input view independently and produces visual features

$$
\mathbf { { \cal F } } _ { k } = f _ { \mathrm { V G G T } } ( { \cal I } _ { k } ) , \qquad { \cal F } _ { k } \in \mathbb { R } ^ { T _ { \mathrm { V G G T } } \times d _ { \mathrm { V G G T } } } .\tag{10}
$$

A learned projection and layer normalisation map these features to the common dimension:

$$
\pmb { H } _ { k } ^ { I } = \mathrm { L N } \left( \pmb { F } _ { k } \pmb { W } _ { \mathrm { V G G T } } + \beta _ { \mathrm { V G G T } } \right) \in \mathbb { R } ^ { T _ { \mathrm { V G G T } } \times d } ,\tag{11}
$$

where $W _ { \mathrm { V G G T } } \in \mathbb { R } ^ { d _ { \mathrm { V G G T } } \times d }$ and $\beta _ { \mathrm { V G G T } } \in \mathbb { R } ^ { d }$ are learned. Under the shared notation of the main paper, $T _ { \mathrm { i m g } } = T _ { \mathrm { V G G T } }$

To assess the effect of the image backbone, Table 8 compares the frozen VGGT backbone (Wang et al., 2025a) with ResNet-101 (He et al., 2016) using two input views. The ResNet alternative uses ResNet-101 without fine-tuning its backbone parameters.

## A.2.3 Branch Decoder Details

The parallel coarse decoder uses M trainable branch queries $q _ { m } \in \mathbb { R } ^ { d }$ and the query matrix $Q _ { \mathrm { q u e r y } } =$ $[ \pmb { q } _ { 1 } ; \dots ; \pmb { q } _ { M } ]$ . Every query attends to the multiview memory but not to another branch query. Using

Table 8: Two-view backbone comparison for RCA and LCA coarse and refined reconstruction. CD denotes Chamfer distance, and CC count denotes connected-component count. Values are mean ± standard error.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Output</td><td colspan="4">RCA</td><td colspan="4">LCA</td></tr><tr><td>Dice (%)↑</td><td>clDice (%)↑</td><td>CD (mm) ↓</td><td>CC count↓</td><td>Dice (%) ↑</td><td>clDice (%)↑</td><td>CD (mm) ↓</td><td>CC count↓</td></tr><tr><td>ResNet-101</td><td>Coarse</td><td>18.94 (±1.15)</td><td>19.94 (±1.43)</td><td>9.37 (±0.37)</td><td>1.428 (±0.097)</td><td>13.88 (±1.78)</td><td>13.69 (±0.89)</td><td>12.99 (±0.32)</td><td>1.878 (±0.097)</td></tr><tr><td>ResNet-101</td><td>Refined</td><td>60.62 (±1.29)</td><td>72.80 (±1.61)</td><td>4.06 (±0.47)</td><td>1.087 (±0.029)</td><td>45.19 (±1.71)</td><td>56.33 (±1.50)</td><td>6.71 (±0.44)</td><td>1.102 (±0.064)</td></tr><tr><td>VGGT</td><td>Coarse</td><td>19.47 (±1.12)</td><td>20.40 (±1.56)</td><td>9.70 (±0.41)</td><td>1.324 (±0.064)</td><td>14.92 (±0.77)</td><td>13.94 (±0.90)</td><td>12.37 (±0.51)</td><td>1.605 (±0.097)</td></tr><tr><td>VGGT</td><td>Refined</td><td>66.92 (±1.50)</td><td>80.62 (±1.92)</td><td>3.16 (±0.27)</td><td>1.054 (±0.026)</td><td>49.45 (±1.32)</td><td>61.30 (±1.93)</td><td>5.98 (±0.47)</td><td>1.053 (±0.026)</td></tr></table>

$\pmb { { \cal E } } _ { \mathrm { v i e w } }$ as the cross-attention keys and values, the shared decoder produces branch tokens

$$
[ b _ { 1 } ; \ldots ; b _ { M } ] = f _ { \mathrm { c r o s s - a t t n } } (  { Q _ { \mathrm { q u e r y } } } , E _ { \mathrm { v i e w } } ) , \qquad b _ { m } \in \mathbb { R } ^ { d } .\tag{12}
$$

The branch queries are decoded independently and in parallel and correspond to the ordered target entries defined in Subsection 3.2. Query m therefore retrieves evidence for branch entry m, but optional entries do not denote fixed, named anatomical side-branch categories. An existence prediction determines which optional branch outputs are active for an individual reconstruction. The prediction of side branch m is directly supervised by target entry m; no Hungarian matching is used.

The control head is

$$
g _ { \mathrm { c p } } : \mathbb { R } ^ { d } \to \mathbb { R } ^ { 3 K _ { \mathrm { c p } } } , \qquad \hat { P } _ { m } ^ { ( 0 ) } = \mathrm { r e s h a p e } ( g _ { \mathrm { c p } } ( b _ { m } ) ) .\tag{13}
$$

The raw radius head predicts N log-radii,

$$
g _ { \mathrm { r a d } } : \mathbb { R } ^ { d } \to \mathbb { R } ^ { N } , \qquad \hat { r } _ { m } ^ { ( 0 ) } = \exp ( g _ { \mathrm { r a d } } ( { \pmb b } _ { m } ) ) ,\tag{14}
$$

with the log values clamped to a finite numerical range before exponentiation. The branch-existence head produces one logit per query,

$$
g _ { \mathrm { e x i s t } } : \mathbb { R } ^ { d }  \mathbb { R } , \qquad z _ { m } = g _ { \mathrm { e x i s t } } ( \pmb { b } _ { m } ) .\tag{15}
$$

At inference, the main-branch entry is forced active and side-branch probabilities are $\hat { e } _ { m } = \sigma ( z _ { m } )$ The prediction heads are shared across all branch queries, and every output is decoded in the same world frame. Thus, M is a fixed maximum number of query outputs, while the predicted active branch count can vary between cases. We use maximum query counts of $M _ { \mathrm { R C A } } = 7$ and $M _ { \mathrm { L C A } } = 1 3$ The required primary branches are always active. An optional branch is retained at inference when $\sigma ( z _ { m } ) \geq 0 . 5$

## A.2.4 Projection-Guided Geometry and Radius Refiners

B-spline geometry refinement. The coarse predictor supplies a population-level artery estimate, but ambiguity in the sparse input views can leave case-specific centreline discrepancies. The geometry refiner corrects these discrepancies directly in the compact B-spline control-point space while holding the coarse radii $\hat { \pmb { r } } _ { m } ^ { ( 0 ) }$ fixed. Its inputs are the coarse controls $\hat { \cal P } ^ { ( 0 ) }$ , branch tokens, coronary mask images, acquisition directions, spatial image features extracted by the frozen backbone, and the known differentiable projection $\Pi _ { k } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 2 }$ for each view k. The same frozen-backbone features used by the coarse predictor are therefore reused during geometry refinement; the backbone itself remains frozen.

Because a B-spline control point does not generally lie on the decoded curve, each control index $j$ is associated with an on-curve anchor. The anchor index is the sampled centreline position at which the corresponding basis function is strongest:

$$
n _ { j } = \arg \operatorname* { m a x } _ { n } A _ { n , j } .\tag{16}
$$

At refinement stage s, the current decoded anchor is

$$
a _ { m , j } ^ { ( s ) } = \sum _ { \ell = 1 } ^ { K _ { \mathrm { c p } } } A _ { n _ { j } , \ell } \hat { p } _ { m , \ell } ^ { ( s ) } .\tag{17}
$$

Here, $s \in \{ 0 , \ldots , S _ { g } - 1 \}$ denotes the geometry-refinement stage. The anchor is projected to view k as ${ \pmb u } _ { m , j , k } ^ { ( s ) } = \Pi _ { k } ( { \pmb a } _ { m , j } ^ { ( s ) } )$ . Around $\pmb { u } _ { m , j , k } ^ { ( s ) }$ , the implementation samples an odd-sized input-mask patch, local learned image features, the mask distance transform, and its two-dimensional gradient. It also extracts the backbone feature at the corresponding image region: from the spatial VGGT patch-token map in the final model, or from the ResNet feature map in the backbone comparison. Collectively, these samples form the local multiview projection evidence for control point $( m , j )$ . Samples outside the detector or belonging to padded views are marked invalid.

The evidence encoder maps the local samples to dimension d. A refinement query combines the current control position, anchor position, branch token, control-index embedding, branch-index embedding, and recurrent state. Multi-head attention aggregates the encoded samples over valid views. The resulting state is passed to a residual head:

$$
\Delta \hat { p } _ { m , j } ^ { ( s ) } = \alpha _ { g } \operatorname { t a n h } \left( g _ { g } ( z _ { m , j } ^ { ( s ) } ) \right) \in \mathbb { R } ^ { 3 } ,\tag{18}
$$

where $\boldsymbol { z } _ { m , j } ^ { ( s ) } \in \mathbb { R } ^ { d }$ is the evidence-conditioned state, $g _ { g } : \mathbb { R } ^ { d }  \mathbb { R } ^ { 3 }$ is the learned residual head, and $\alpha _ { g }$ is the maximum update magnitude in millimetres. The controls are updated by

$$
\hat { P } ^ { ( s + 1 ) } = \hat { P } ^ { ( s ) } + \Delta \hat { P } ^ { ( s ) } .\tag{19}
$$

The output layer parameters are shared across the configured $S _ { g }$ stages. After each stage, the updated B-spline is decoded, its anchors are reprojected, and new image evidence is sampled for the next stage. The radius vector remains fixed throughout this module. Geometry refinement is therefore a fixed-depth learned computation at inference rather than per-case gradient-based optimisation.

The reported geometry-refiner configuration uses $S _ { g } ~ = ~ 4$ stages, $1 7 \times 1 7$ local patches, a 256-dimensional evidence state, and a maximum per-stage control update of $\alpha _ { g } = 7 \mathrm { m m }$

Dense-radius refinement. The radius refiner is enabled only with B-spline centrelines and raw dense-radius prediction. It receives the geometry-refined vessel, holds the refined centreline $\hat { C } _ { m } ^ { ( S _ { g } ) }$ fixed, and updates only the dense radii. At radius-refinement stage $u \in \{ 0 , \ldots , S _ { r } - 1 \}$ , the current centreline-and-radius model is differentiably rendered into every valid input view.

For projected centreline point $( m , n )$ , finite differences along the projected curve estimate a unit tangent. Rotating this tangent by $9 0 ^ { \circ }$ gives the transverse image-plane normal. Let L be the odd number of profile samples and $w _ { \mathrm { p r o f } }$ the profile half-width in pixels. The sampling grid is

$$
\begin{array} { r } { \pmb { u } _ { m , n , k , \ell } = \Pi _ { k } ( \hat { c } _ { m , n } ) + \delta _ { \ell } \pmb { n } _ { m , n , k } , \qquad \delta _ { \ell } \in [ - w _ { \mathrm { p r o f } } , w _ { \mathrm { p r o f } } ] , \quad \ell \in \{ 1 , \ldots , L \} , } \end{array}\tag{20}
$$

where ${ \mathbf { } } n _ { m , n , k }$ is the transverse unit normal. Four profiles are sampled: the observed input mask, the rendered vessel mask, their signed difference, and their absolute difference. These profiles expose whether the rendered vessel is too narrow or too wide around the current projected centreline while reducing interference from distant parts of the projected coronary tree. They are concatenated with the encoded view information and mapped to the model dimension.

For point $( m , n )$ , the query is formed from the branch token, branch-index and centreline-pointindex embeddings, encoded current radius, and recurrent point state. Multi-head attention aggregates the valid per-view profiles. Two one-dimensional convolutional blocks then propagate context along the ordered centreline. A zero-initialised scalar residual head predicts

$$
\Delta \hat { r } _ { m , n } ^ { ( u ) } = \alpha _ { r } \operatorname { t a n h } \left( g _ { r } ( z _ { m , n } ^ { ( u ) } ) \right) ,\tag{21}
$$

Table 9: Effect of branch-subset augmentation on two-view coarse prediction and geometry refinement. CD denotes Chamfer distance, and CC count denotes connected-component count. Values are mean ± standard error.
<table><tr><td rowspan="2">Prediction stage</td><td rowspan="2">Branch-subset augmentation</td><td colspan="4">RCA</td><td colspan="4">LCA</td></tr><tr><td>Dice (%) ↑</td><td>clDice (%)↑</td><td>CD (mm) ↓</td><td> $\mathbf { C C }$  count ↓</td><td>Dice (%) ↑</td><td>clDice (%) ↑</td><td>CD (mm) ↓</td><td>CC count ↓</td></tr><tr><td>Coarse prediction</td><td>X</td><td>11.38 (±0.95)</td><td>10.89 (±1.09)</td><td>13.42 (±0.52)</td><td>1.338 (±0.070)</td><td>9.82 (±0.54)</td><td>8.81 (±0.59)</td><td>14.70 (±0.49)</td><td>1.882 (±0.082)</td></tr><tr><td>Coarse prediction</td><td>√</td><td>19.47 (±1.12)</td><td>20.40 (±1.56)</td><td>9.70 (±0.41)</td><td>1.324 (±0.064)</td><td>14.92 (±0.77)</td><td>13.94 (±0.90)</td><td>12.37 (±0.51)</td><td>1.605 (±0.097)</td></tr><tr><td>Geometry refiner</td><td>X</td><td>46.57</td><td>58.45</td><td>6.13</td><td>1.068</td><td>37.52</td><td>44.57</td><td>8.01</td><td>1.211</td></tr><tr><td>Geometry refiner</td><td></td><td>(±1.45) 65.89</td><td>(±2.04) 80.84</td><td>(±0.43) 3.16</td><td>(±0.029) 1.068</td><td>(±1.04) 48.43</td><td>(±1.51) 61.38</td><td>(±0.45) 5.98</td><td>(±0.054) 1.053</td></tr><tr><td></td><td>√</td><td>(±1.44)</td><td>(±1.90)</td><td>(±0.27)</td><td>(±0.029)</td><td>(±1.29)</td><td>(±1.93)</td><td>(±0.47)</td><td>(±0.026)</td></tr></table>

where u is the radius-refinement stage, $z _ { m , n } ^ { ( u ) } \in \mathbb { R } ^ { d }$ is the evidence-conditioned point state, $g _ { r } : \mathbb { R } ^ { d } $ R is the residual head, and $\alpha _ { r }$ bounds one correction in millimetres. The update is

$$
\begin{array} { r } { \hat { r } _ { m , n } ^ { ( u + 1 ) } = \operatorname* { m a x } \left( r _ { \operatorname* { m i n } } , \hat { r } _ { m , n } ^ { ( u ) } + \Delta \hat { r } _ { m , n } ^ { ( u ) } \right) . } \end{array}\tag{22}
$$

The updated surface is rerendered and new transverse evidence is sampled before the next of the $S _ { r }$ stages. Because centreline coordinates are detached from this pathway, radius refinement cannot change vessel trajectory to reduce its loss. After the configured geometry and radius stages, the final refined vessel code is

$$
\hat { V } _ { \mathrm { r e f i n e d } } = \left\{ ( \hat { c } _ { m , n } ^ { ( S _ { g } ) } , \hat { r } _ { m , n } ^ { ( S _ { r } ) } ) \right\} _ { m , n } .\tag{23}
$$

The reported radius-refiner configuration uses $S _ { r } = 3$ stages, $L = 2 5$ transverse samples spanning $w _ { \mathrm { p r o f } } = 1 6$ pixels on either side of the projected centreline, a maximum update of $\alpha _ { r } = 1$ mm per stage, and minimum radius $r _ { \mathrm { m i n } } = 0 . 0 5$ mm.

## A.2.5 Within-Case Branch-Subset Training

Branch-count variation in the original dataset is entangled with differences in case-specific anatomy. To provide direct supervision for optional branch presence, we construct multiple branch-subset variants from each training case while holding its underlying anatomy fixed.

For a case with B ordered branches, every variant retains the required main path or paths: branch 1 for RCA and branches 1–2 for LCA. Starting from the complete artery tree, optional side branches are successively removed in reverse canonical order until only the required main path or paths remain. An RCA case therefore produces B variants, while an LCA case produces $B - 1$ variants, including the complete-tree and main-path-only versions. For every variant, we regenerate the projected masks and branch-existence labels while preserving the geometry targets and ordered identities of retained branches.

All variants from the same case are collated in one optimisation batch and use the same selected input views. The training loader visits every physical case once per epoch, and every corresponding branch-subset variant is included; no Bernoulli sampling probability is used. The variants are presented together rather than introduced sequentially as a curriculum. This protocol is applied when training both the coarse predictor and the geometry refiner.

Table 9 shows that within-case branch-subset training improves Dice, clDice, and Chamfer distance for both prediction stages and artery cohorts. For branch-existence prediction, it increases accuracy from 92.95% to 94.63% and F1 from 88.37% to 92.48%. The improvement is driven primarily by higher precision and specificity, indicating fewer false-positive side-branch predictions, although recall decreases modestly. These results suggest that varying branch presence while preserving patient anatomy helps the model learn that side branches are optional.

## A.2.6 Direct Per-Scene B-spline Optimisation Baseline

The direct-optimisation baseline starts from the same coarse control points $\hat { \pmb { P } } ^ { ( 0 ) }$ as the learned geometry refiner. For each test case, gradient-based optimisation updates only the B-spline controls to reduce the selected 2D projection objective against the input masks:

$$
P _ { \mathrm { o p t } } ^ { \star } = \arg \operatorname* { m i n } _ { P } \mathcal { L } _ { \mathrm { p r o j } } \left( \operatorname { R e n d e r } ( A P , \hat { r } ^ { ( 0 ) } ) , \{ I _ { k } \} _ { k = 1 } ^ { K } \right) .\tag{24}
$$

The coarse radius is fixed. The number of optimisation iterations and all hyperparameters are determined during method development and fixed before test-set evaluation. This baseline isolates whether the learned geometry refiner provides a better accuracy–runtime trade-off than test-time optimisation from the same initialisation.

Table 4 compares direct optimisation with the learned geometry and radius refiners from a common two-view coarse initialisation for RCA and LCA, including their post-coarse processing time.

## A.3 Implementation Details and Ablation Studies

## A.3.1 Experiment Details

PPCAR-Net is implemented in PyTorch and trained with one GPU per run. We use AdamW with an initial learning rate of $1 0 ^ { - 4 }$ , weight decay of $1 0 ^ { - 4 }$ , cosine learning-rate decay to $1 0 ^ { - 6 }$ , and a gradient-norm clipping threshold of 1.0. The coarse predictor, geometry refiner, and radius refiner all use a batch size of 16. The coarse predictor and geometry refiner are trained for at most 200 epochs, and the radius refiner for at most 100 epochs. The frozen image backbone is initialised from the pretrained facebook/VGGT-1B checkpoint, and its features are precomputed before predictor training. The test set is evaluated only after model and hyperparameter selection. All reported runtime measurements use a single NVIDIA GeForce RTX 4090 GPU.

## A.3.2 Evaluation Implementation Details

All volumetric metrics are computed on the common isotropic 0.5 mm evaluation grid defined in Subsection 3.6. The ground-truth mask is resampled by nearest-neighbour interpolation, while the predicted centreline and radii are rasterised as the union of filled, radius-varying tubular segments on this grid. The resulting voxelised prediction is used consistently for Dice, clDice, and connected component counting.

Let $\widehat { \mathcal { V } }$ and V denote the foreground-voxel sets of the predicted and ground-truth arteries, respectively. Their volumetric overlap is measured by

$$
\mathrm { D i c e } = \frac { 2 | \widehat { \mathcal { V } } \cap \mathcal { V } | } { | \widehat { \mathcal { V } } | + | \mathcal { V } | } .\tag{25}
$$

For clDice, let $\boldsymbol { \mathcal { S } } ( \cdot )$ denote the one-voxel 3D centreline obtained by skeletonisation. Topology precision, topology sensitivity, and clDice are

$$
\begin{array} { r } { T _ { \mathrm { p r e c } } = \frac { \displaystyle | { \cal S } ( \widehat { \mathcal { V } } ) \cap { \mathcal { V } } | } { \displaystyle | { \cal S } ( \widehat { \mathcal { V } } ) | } , } \\ { T _ { \mathrm { s e n s } } = \frac { \displaystyle | { \cal S } ( \mathcal { V } ) \cap \widehat { \mathcal { V } } | } { \displaystyle | { \cal S } ( \mathcal { V } ) | } , } \\ { \mathrm { c l D i c e } = \frac { \displaystyle 2 T _ { \mathrm { p r e c } } T _ { \mathrm { s e n s } } } { \displaystyle T _ { \mathrm { p r e c } } + T _ { \mathrm { s e n s } } } . } \end{array}\tag{26}
$$

Each ground-truth RCA or LCA tree contains exactly one connected component because all retained side branches attach to the corresponding main branch. Connected-component count evaluates whether a prediction preserves this expected structure. It is computed from the same 0.5 mm

voxelised prediction used for Dice and clDice. We apply 3D connected-component labelling to the foreground set $\widehat { \nu }$ and report

$$
N _ { \mathrm { C C } } = \left| \mathrm { C C } ( \widehat { \mathcal { V } } ) \right| ,\tag{27}
$$

where $\operatorname { C C } ( { \widehat { \mathcal { V } } } )$ is the set of connected foreground components. A value of one indicates a single connected prediction, while larger values indicate fragmentation.

Finally, let $\widehat { c }$ and C be the predicted and ground-truth centreline point sets in millimetre-valued world coordinates. We use the symmetric Chamfer distance

$$
\mathrm { C D } ( \widehat { \mathcal { C } } , \mathcal { C } ) = \frac { 1 } { | \widehat { \mathcal { C } } | } \sum _ { \widehat { c } \in \widehat { \mathcal { C } } } \operatorname* { m i n } _ { { c } \in \mathcal { C } } \| \widehat { c } - c \| _ { 2 } + \frac { 1 } { | \mathcal { C } | } \sum _ { c \in \mathcal { C } } \operatorname* { m i n } _ { { \widehat { c } } \in \widehat { \mathcal { C } } } \| c - \widehat { c } \| _ { 2 } .\tag{28}
$$

Dice and clDice are reported as percentages, Chamfer distance is reported in millimetres, and connected-component count is reported as a unitless count.

## A.3.3 Comparison-Method Adaptation, Evaluation, and Timing

We adapted the available implementations of 3DGR-CAR (Fu et al., 2024), AutoCAR (Zhu et al., 2025), and DeepCA (Wang et al., 2025b) to the same ImageCAS-derived data protocol used by our method. This experiment is not intended to reproduce the results reported on the original datasets of those papers. Instead, we train their models on our dataset and evaluate them under the same data split, input-view protocol, and metrics as our method to provide a controlled comparison. The available 3DGR-CAR and AutoCAR code did not contain every module required to train and evaluate their complete pipelines on our dataset. We therefore reimplemented the missing components from the algorithmic and implementation descriptions in the corresponding papers. The remaining adaptations principally concern the data-loading and evaluation interfaces. Each comparison-method loader reads the segmented two-dimensional coronary masks, their associated acquisition geometry, and the corresponding three-dimensional artery label, then converts these inputs to the tensor layout and spatial resolution expected by that method. The same training, validation, and test case assignments are used across methods. For each supported input-view count, all methods use the same selected views for a given test case so that differences are not caused by a different choice of projections.

Each comparison model is trained from scratch on our training split so that the experiment evaluates model capacity under the same data and supervision, without an advantage from external pre-training. Learning rate, training duration, loss weights, output threshold, and other methodspecific settings are selected using reconstruction performance on the validation split. The test split is not used for model selection or hyperparameter tuning. Apart from modules needed to load our data or complete unavailable output and evaluation steps from the descriptions in the corresponding papers, the published model structure and learning objective are retained.

All final predictions are evaluated by the same code. Volumetric outputs are resampled to the common isotropic 0.5 mm grid and binarised using the validation-selected threshold before Dice and clDice are calculated. The native outputs have different spatial dimensions and voxel resolutions: AutoCAR produces a $4 0 0 ^ { 3 }$ volume at 0.5 mm resolution, whereas DeepCA and 3DGR-CAR produce $1 2 8 ^ { 3 }$ volumes at approximately 0.9167 mm and 0.9239 mm resolution, respectively. Consequently, resampling each prediction to the common grid can introduce method-dependent discretisation effects, so the Dice and clDice comparison is not entirely independent of native output resolution. Nevertheless, these dimensions are part of the original output configuration of each model; we retain them rather than altering the model architectures or retraining them for a common output grid. When a method does not natively produce an ordered centreline, its predicted volume is converted to a vascular graph using the same graph-extraction post-processing, and Chamfer distance is then computed between its extracted centreline point set and the ground-truth centreline point set in millimetres. Reported values are the mean and standard error across the test cases. This shared evaluator keeps the metric definitions, physical coordinate convention, and test cases fixed across all methods.

Table 10: Two-view reconstruction-time comparison on one NVIDIA GeForce RTX 4090 GPU. Model inference produces each method’s native output, post-processing converts that output into a vascular graph/skeleton representation when required, and total time is their sum. The best result is shown in bold and the second-best result is underlined; post-processing times are ranked only among methods that require this stage. – denotes that post-processing is not required; N/R denotes that it was not recorded.
<table><tr><td>Method</td><td>Model inference time ↓</td><td>Post-processing time ↓</td><td>Total time ↓</td></tr><tr><td>SDF-CAR (Reda et al., 2026)</td><td>45 min</td><td>N/R</td><td>N/R</td></tr><tr><td>3DGR-CAR (Fu et al., 2024)</td><td>61 s</td><td>2.71 s</td><td>63.7 s</td></tr><tr><td>DeepCA (Wang et al., 2025b)</td><td>700 ms</td><td>1.99 s</td><td>2.69 s</td></tr><tr><td>AutoCAR (Zhu et al., 2025)</td><td>330 ms</td><td>3.43 s</td><td>3.76 s</td></tr><tr><td>PPCAR-Net (coarse)</td><td>60 ms</td><td>一</td><td>0.06 s</td></tr><tr><td>PPCAR-Net (refined)</td><td>121 ms</td><td>一</td><td>0.12 s</td></tr></table>

Inference-time breakdown. All runtimes in Table 10 use two input views and a single NVIDIA GeForce RTX 4090 GPU under the same software setup. We report model inference time, measured from supplying the 2D input views to obtaining the method’s native reconstruction output; postprocessing time, measured from the native output to a vascular graph/skeleton representation using the graph-extraction algorithm of AutoCAR (Zhu et al., 2025); and their sum as the total time. This post-processing is performed for every quantitatively evaluated case so that the final metrics use a graph/skeleton representation. Although the same graph-extraction algorithm is applied to all comparison methods, its runtime depends on the native input representation, evaluation-grid size, and vessel-mask complexity. DeepCA supplies a compact $1 2 8 ^ { 3 }$ binary uint8 mask without probability thresholding and therefore has the shortest post-processing time. The $1 2 8 ^ { 3 }$ binary output of 3DGR-CAR is accompanied by additional arrays, requiring more input/output validation and potentially presenting a more complex foreground. AutoCAR usually supplies a substantially larger 400<sup>3</sup> probability volume, which incurs additional decompression, validation, and thresholding and consequently has the longest post-processing time.

## A.3.4 Ablation Studies

With two input views, the branch-existence head achieves an accuracy of 94.63% and an F1 score of 92.48%, indicating reliable identification of the branches present in each case. Table 7 compares direct dense-centreline prediction with the B-spline parameterisation. The B-spline representation improves Dice, clDice, and Chamfer distance at both prediction stages for RCA and LCA. It also lowers the refined connected-component count for both arteries and the coarse count for LCA; the dense-centreline variant has a marginally lower RCA coarse count. Table 8 compares the frozen image encoders. After refinement, VGGT improves evaluation metrics over ResNet-101 for both RCA and LCA. At the coarse stage, VGGT improves Dice, clDice, and connected-component count for both arteries and Chamfer distance for LCA, while RCA Chamfer distance is slightly lower with ResNet-101. Table 9 evaluates within-case branch-subset augmentation. This augmentation improves evaluation metrics for both prediction stages and artery cohorts. These results support using the variants to learn optional side-branch structure. Table 10 separates native model inference from graph-extraction post-processing. Our native centreline-and-radius output requires no graphconversion stage and completes refined two-view reconstruction in 0.12 s. The refinement comparison is presented in Section 4.3 and Table 4.