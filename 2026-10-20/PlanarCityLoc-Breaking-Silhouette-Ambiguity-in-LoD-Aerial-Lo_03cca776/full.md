# PlanarCityLoc: Breaking Silhouette Ambiguity in LoD Aerial Localization via 3D Planes

Oussema Dhaouadi<sup>1,2,4,5,6</sup> Shao Jie Hu Chen<sup>1,2</sup> Jakob Gollreiter<sup>1,2</sup> Samir Jusufi<sup>1,3</sup> Nikolas Rieger<sup>1,2,7</sup> Caroline Wendlandt<sup>1,2</sup> Christian Wender<sup>1,8</sup> Jingfeng Mao<sup>1,2,4</sup> Guangming Wang<sup>1,4</sup>   
Zuria Bauer<sup>5</sup> Marc Pollefeys<sup>5,9</sup> Daniel Cremers<sup>2,6</sup> Olaf Wysocki<sup>1,4</sup>

<sup>1</sup>TUM.ai <sup>2</sup>Technical University of Munich <sup>3</sup>LMU Munich <sup>4</sup>University of Cambridge <sup>5</sup>ETH Zurich¨ <sup>6</sup>Munich Center for Machine Learning <sup>7</sup>Konrad Zuse School of Excellence in Reliable AI (relAI) <sup>8</sup>BMW Group <sup>9</sup>Microsoft

https://tum-vision.github.io/PlanarCityLoc/

## Abstract

Aerial visual localization is crucial for autonomous drone navigation in urban environments where GNSS signals are often degraded. Lightweight textureless Level-of-Detail (LoD) city models provide a geometric map alternative to textured assets. Existing silhouette-based methods rely on image-space alignment, which can leave different camera poses indistinguishable and favor incorrect poses when segmentation errors distort the building outlines. We present PlanarCityLoc, which moves beyond silhouette overlap by leveraging the planar geometry of building facades and roofs. Following geometry-aware initialization, we match image-side planar patches to LoD model planes and use the resulting 3D-to-2D correspondences to refine pose with Perspective-n-Point (PnP). The framework uses off-theshelf geometry and segmentation models without datasetspecific training or fine-tuning. Experiments on UAVD4L-LoDv2 and Swiss-EPFLv2 achieve state-of-the-art localization recall among the evaluated off-the-shelfLoD-based methods. By grounding image-space alignment in 3D planes, PlanarCityLoc enables more accurate aerial localization with lightweight, textureless city maps.

## 1. Introduction

Aerial visual localization for Unmanned Aerial Vehicles (UAVs), defined as estimating the precise 6-Degrees-of-Freedom (6DoF) camera pose from a single image, is essential for operations in Global Navigation Satellite System (GNSS)-denied urban environments [1, 37]. Camera pose estimates also support geometry-guided annotation of aerial imagery [15] and semantic mapping for autonomous landing [13]. While matching images to textured 3D meshes [31, 39] is established, maintaining these assets globally is expensive and privacy-restricted [23]. Textureless Level-of-Detail (LoD) city models provide a compact geometric reference without storing facade appearance [12, 26, 40, 46, 47].

One way to bridge the appearance gap between images and textureless maps is to compare their structural projections. LoD-Loc [46] follows this approach by predicting building wireframes in the query image and aligning them with edges projected from the LoD model. LoD-Loc v2 [47] instead compares building silhouettes, making alignment applicable to simpler LoD1 geometry (Fig. 1(a)). LoD-Loc v3 [26] uses instance masks to distinguish buildings, reducing ambiguity and broad score plateaus caused by merged semantic silhouettes in dense scenes.

These silhouette-based objectives still measure agreement in image space. Distinct camera poses can produce similar projections, and disjoint masks yield the same zero intersection-over-union over a range of poses. Segmentation errors introduce a further difficulty. Missing facades, displaced boundaries, or merged building instances change the target silhouette, so an incorrect pose can align better with the predicted mask than the true pose does. Consequently, even a distinct score maximum need not coincide with minimum pose error, as illustrated schematically in Fig. 1(c). Improving pose accuracy therefore calls for constraints beyond silhouette overlap alone.

![](images/52fb40219c18690fc3edb06e1a4c3f7f494b7bf2d9ab7765260dfd4051b0df6f.jpg)

![](images/c8b240ae516be674375539edc9a72145fd58f7580aa09c56b50c2ca61364b65d.jpg)  
(c) Optimization Landscape  
Figure 1. PlanarCityLoc overview. (a) Silhouette-based localization compares predicted building masks with renderings of the LoD model. (b) PlanarCityLoc grounds image-side planar patches in 3D through explicit correspondences with LoD planes. (c) Schematic score landscapes illustrate ambiguity in image-space overlap and a score maximum displaced from the correct pose. Plane-based refinement adds geometric constraints to move beyond image-space score maximization. Curves are illustrative, not measured method comparisons.

Our key observation is that building facades and roofs provide explicit 3D planar patches within LoD models. Off-the-shelf vision models [2, 11, 16, 20, 28, 38, 43] provide image-side geometry and segmentation without dataset-specific fine-tuning. Building on these cues, we propose PlanarCityLoc. Following geometry-aware coarse initialization, we match image-side planar patches to rendered LoD planes. This 3D planar grounding uses rendered depth to establish 3D-to-2D correspondences and iteratively refines pose with a robust Perspective-n-Point (PnP) solver [10] (Fig. 1(b)). The resulting geometric constraints allow refinement beyond an image-space score peak, addressing the gap between silhouette agreement and accurate camera pose illustrated in Fig. 1(c).

In summary, our main contributions are:

• We formulate zero-shot LoD-based aerial localization through explicit plane correspondences.

• We introduce iterative plane association between image patches and textureless map geometry for robust refinement of the camera pose.

• We evaluate on UAVD4L-LoDv2 and Swiss-EPFLv2, achieving state-of-the-art localization recall among the evaluated off-the-shelf LoD-based methods across all four splits, and analyze the contributions of geometric scoring and plane-based refinement through ablations.

## 2. Related Work

Visual localization. Visual localization estimates the 6DoF camera pose relative to a known map. Traditional approaches match local features to a 3D point cloud constructed via Structure-from-Motion [24, 31], pairing feature extractors [4] with robust matchers [8, 9, 30, 33]. Hierarchical pipelines [29] first retrieve relevant images before local feature matching. MeshLoc [25] localizes directly on 3D meshes, while UAVLoc [39] uses posed reference images rendered from textured meshes for aerial localization.

Render-and-compare approaches [42] offer another route to aerial pose estimation. MC-Loc [35] combines pretrained image features, a renderable scene representation, and particle filtering for pose refinement without task-specific training. Maintaining detailed appearance-bearing maps at city scale remains costly. AnyVisLoc [45], OrthoLoC [5], and OrthoTrack [6] instead explore localization using overhead imagery and 2.5D surface maps. Our work focuses on textureless LoD geometry, which provides a compact reference without storing facade appearance.

LoD-based localization. LoD-Loc [46] introduced the use of LoD city models for aerial localization through neural wireframe alignment. It predicts building edges in the query image, scores pose hypotheses using projected map wireframes, and refines the estimate through differentiable alignment. LoD-Loc v2 [47] shifts from wireframes to semantic building silhouettes, supporting simpler LoD1 models through coarse pose search and particle-filter refinement. LoD-Loc v3 [26] further separates buildings using instance segmentation trained on synthetic aerial imagery. Instancelevel alignment reduces the ambiguity of merged semantic masks in dense scenes, but remains sensitive to missing instances, incorrect boundaries, and merged or split predictions. Such errors can displace the overlap maximum from the true pose, even when the score has a distinct peak (Fig. 1(c)). Concurrent work, MatchCityLoc [7], explores an alternative through matching predicted and rendered surface normals. PlanarCityLoc complements silhouette alignment with explicit 3D plane correspondences. This grounding enables refinement beyond the image-space score peak, leveraging pretrained geometry and segmentation models without dataset-specific training or fine-tuning.

Monocular foundation models. Extracting structured planar patches from a single UAV image requires precise surface geometry and semantic segmentation. Recently, monocular geometry estimation has experienced a paradigm shift due to large-scale foundation models [11, 16, 18, 43]. Models such as MoGe-2 [38] can predict metric-scale 3D point maps and detailed surface normals from a single image without any retraining. Similarly, segmentation foundation models like SAM 3 [2] provide robust zero-shot building masks. The OccuFly benchmark [14] further highlights the challenges of applying pretrained metric-depth models to aerial imagery across different flight altitudes. We leverage these models to extract geometric and semantic cues from aerial images.

Plane-based localization. Planes provide structural constraints beyond individual keypoints and have been explored for augmented reality [32]. In indoor localization, PlanaRe-Loc [44] matches query-image and 3D-map planes through learned embeddings. PALMS [3] uses observed wall geometry and floorplans to initialize indoor particle-filter localization. Related primitive-based methods include $Z _ { - }$ FLoc [36], which matches lines and circles extracted from monocular reconstructions to floorplans, and outdoor approaches that align semantic or geometric primitives across views [22, 27]. These works establish the value of structural primitives, while their use for aerial localization against simplified city models remains less explored. We match facade and roof planes in LoD models to patches predicted from a single UAV image.

## 3. Method

Problem formulation. Given an LoD city model $\mathcal { M } ,$ a query image $I _ { q } \in \mathbb { R } ^ { H \times W \times 3 }$ with height H and width W, known intrinsics K, and a coarse GPS/IMU pose prior $\xi _ { p } ,$ our goal is to estimate the accurate 6-DoF camera pose ${ \pmb \xi } ^ { * } = ( x , y , z , \theta , \rho , r )$ , where $( x , y , z )$ is the world-space camera center and $( \theta , \rho , r )$ denotes yaw, pitch, and roll.

Overview. We show the pipeline in Fig. 2. We first extract geometric cues from the query image and group building surfaces into planar patches. Geometry-aware coarse initialization then brings the image and map into approximate alignment. Building on this estimate, our 3D planar grounding stage matches image patches to visible LoD planes and iteratively refines the pose through robust PnP. Finally, rank aggregation combines complementary alignment measures to select the output pose.

## 3.1. Geometry and Semantics Extraction

We extract query-side geometry and semantics using offthe-shelf foundation models. MoGe-2 [38] provides a surface-normal map $\mathbf { N } _ { q } \in \mathbb { R } ^ { H \times W \times 3 }$ and a depth map $D _ { q } \in \mathbb { R } _ { > 0 } ^ { H \times W }$ , while SAM 3 [2] provides a binary building mask $\tilde { M _ { q } } \in \{ 0 , 1 \} ^ { H \times W }$ . We derive a structural edge map $E _ { q }$ by applying a Sobel filter to $\mathbf { N } _ { q }$ within a dilated $M _ { q }$ These edges capture changes in surface orientation within and around buildings. We denote the unit normal at pixel u as ${ \bf n } _ { q } ( u ) = { \bf N } _ { q } ( u ) / \| { \bf N } _ { q } ( u ) \|$ .

## 3.2. Plane Extraction

The predicted geometry allows us to describe buildings through their individual facade and roof surfaces, rather than a single silhouette. These surfaces have counterparts in the planar geometry of LoD city models. We extract patches from the query image and rendered map views to establish correspondences for pose refinement.

Image-side patchification. Starting from the building mask $M _ { q } ,$ we group pixels into planar patches $\mathcal { P } _ { q } = \{ P _ { j } \}$ using 4-connected region growing. We merge neighboring pixels u and v when their normals agree,

$$
\mathbf { n } _ { q } ( u ) \cdot \mathbf { n } _ { q } ( v ) \geq \tau _ { n } ,\tag{1}
$$

where $\tau _ { n }$ is a normal similarity threshold. Furthermore, we require their relative depths to be consistent,

$$
\frac { | D _ { q } ( u ) - D _ { q } ( v ) | } { \operatorname* { m a x } ( D _ { q } ( u ) , D _ { q } ( v ) ) } < \tau _ { d } ,\tag{2}
$$

where $\tau _ { d }$ is a relative depth threshold. Depth continuity helps separate adjacent surfaces with similar normals but different ranges. Each patch $P _ { j } = \left( U _ { j } , \mathbf { p } _ { j } , \bar { \mathbf { n } } _ { j } \right)$ comprises its pixel support $U _ { j }$ , a robust 2D center $\mathbf { p } _ { j }$ at the median pixel location, and a normalized mean normal $\bar { \mathbf { n } } _ { j }$

Model-side patchification. Given a pose hypothesis $\xi ,$ we render the mask $\hat { M } ( \pmb { \xi } )$ , depth $\hat { D } ( \pmb { \xi } )$ , and normal map $\ddot { \bf N } ( \pmb { \xi } )$ and derive rendered edges $\hat { E } ( \pmb { \xi } )$ for scoring. We apply the same patch-extraction rule to obtain visible LoD patches $\mathcal { Q } ( \pmb { \xi } ) = \{ Q _ { i } \}$ . Each patch $Q _ { i } = \left( V _ { i } ( \pmb { \xi } ) , \mathbf { X } _ { i } , \hat { \mathbf { N } } _ { i } \right)$ contains its pixel support, a representative 3D center, and its cameraframe unit normal. We obtain $\mathbf { X } _ { i }$ by back-projecting the center of the visible patch using rendered depth and transforming it to world coordinates. Matching 3D centers to image locations supplies the correspondences for PnP.

## 3.3. Geometry-Aware Coarse Initialization

Reliable plane matching first requires approximate alignment between the query image and map. We obtain this initialization by rendering pose hypotheses around the sensor prior $\xi _ { p }$ and comparing them with the query predictions. Following the image-space search paradigm of LoD-Loc v2 and LoD-Loc v3 [26, 47], we use cost-volume (CV) search followed by particle-filter (PF) refinement, augmented with edge and depth cues. Both stages refine translation and yaw while keeping pitch and roll fixed at their prior estimates, as in LoD-Loc v2 [47]. The subsequent plane-based PnP stage refines all six pose degrees of freedom.

Cost-volume search. We first sample altitude hypotheses at the prior horizontal location to initialize height. We then search a translation-and-yaw grid G in two successive stages, narrowing the grid around the best hypothesis. The cost-volume alignment score $S _ { \mathrm { C V } }$ combines building-mask intersection-over-union $S _ { I } = | M _ { q } \cap \hat { M } | / | M _ { q } \cup \hat { M } |$ , edge agreement $S _ { E }$ , and depth consistency $S _ { D }$ . The edge term rewards rendered building boundaries that lie close to query normal-map edges, weighted by their coverage. The depth term combines similarity in depth variation with residual error after compensating for scale and offset differences. We express this cue fusion in the general weighted-sum form

![](images/54164a503a833fe9e8ad87f7d5ef7fe134c15c3efd513fb65552d10bd405077a.jpg)  
Figure 2. Overview of PlanarCityLoc. (a) We extract building masks, surface normals, depth, and structural edges from query image $I _ { q }$ with foundation models, then (b) form 2D planar patches. (c) A coarse pose $\xi _ { f }$ is obtained through altitude initialization, cost-volume search, and multi-scale particle filtering. (d) We iteratively match visible LoD planes to image patches and refine the pose using PnP RANSAC on the resulting 3D-to-2D correspondences. (e) Equal-weight rank aggregation of complementary metrics selects the final pose.

$$
{ \cal S } _ { \mathrm { C V } } ( \pmb \xi ) = w _ { I } ^ { c } ( \pmb \xi ) S _ { I } ( \pmb \xi ) + w _ { E } ^ { c } ( \pmb \xi ) S _ { E } ( \pmb \xi ) + w _ { D } ^ { c } ( \pmb \xi ) S _ { D } ( \pmb \xi ) .\tag{3}
$$

Here, the cue-dependent weights $w _ { I } ^ { c } , w _ { E } ^ { c } ,$ and $w _ { D } ^ { c }$ balance mask, edge, and depth agreement, respectively. On each grid, selection applies a weak proximity prior and a maskoverlap floor to favor plausible hypotheses. We denote the resulting coarse pose by $\pmb { \xi } _ { c } .$

Particle-filter refinement. Starting from $\pmb { \xi } _ { c } .$ , a multi-scale particle filter [17, 35] samples local pose perturbations and favors hypotheses with stronger rendered-to-query agreement. The perturbation scale decreases over successive steps to refine the pose. This stage uses the additive alignment score

$$
\begin{array} { r } { S _ { \mathrm { P F } } ( \pmb { \xi } ) = w _ { m } S _ { M } ( \pmb { \xi } ) + w _ { e } S _ { E } ( \pmb { \xi } ) + w _ { d } S _ { D } ( \pmb { \xi } ) . } \end{array}\tag{4}
$$

Here, $w _ { m } , w _ { e }$ , and $w _ { d }$ balance the three cues. Unlike the cost volume’s IoU term, $S _ { M }$ measures the fraction of the predicted building mask explained by the rendering:

$$
S _ { M } ( \pmb { \xi } ) = \frac { | M _ { q } \cap \hat { M } ( \pmb { \xi } ) | } { | M _ { q } | } .\tag{5}
$$

Mask recall tolerates building regions present in the rendered map but missing from the predicted mask, for example because of occlusion or incomplete segmentation. Unlike IoU, it does not penalize this extra rendered support. Edge and depth agreement provide complementary constraints, with a weak proximity prior discouraging drift during local search. The best pose retained after the final particle-filter stage, $\xi _ { f }$ , initializes plane-based refinement.

## 3.4. Iterative Plane-based PnP

The coarse estimate $\xi _ { f }$ aligns the rendered map with the image, but image-space agreement can still leave residual pose ambiguity. We refine this estimate by grounding image patches in the map’s 3D geometry. Each matched LoD patch provides a world-space point $\mathbf { X } _ { i }$ and a corresponding image location $\mathbf { u } _ { i }$ . Together, these 3D-to-2D correspondences constrain the camera pose through PnP, allowing refinement beyond the image-space score peak.

Iterative PnP. Starting from $\bar { \xi ^ { ( 0 ) } } = \xi _ { f } ,$ we re-render the LoD planes at each outer iteration t and construct a set of 3D-to-2D correspondences ${ \mathcal C } ^ { ( t ) } = \{ ( { \bf X } _ { i } , { \bf u } _ { i } ) \}$ . We update the pose as

$$
\boldsymbol { \xi } ^ { ( t + 1 ) } = \mathrm { P n P } \big ( \mathbf { K } , \boldsymbol { \mathcal { C } } ^ { ( t ) } \big )\tag{6}
$$

using PnP-RANSAC [10] followed by nonlinear refinement on the inliers. Patch centers can be imprecise because of segmentation errors or partial visibility. Given sufficient consistent correspondences, RANSAC rejects outlying matches so that every center need not be accurate. We repeat matching and pose estimation for up to N outer iterations, re-rendering each updated pose to renew correspondences and gather pose candidates.

Correspondence building. Image-side extraction can split a physical facade into several patches, while its rendered counterpart remains a single patch. To accommodate this fragmentation and then refine the associations, we apply two sequential matchers within each iteration.

Overlap voting associates each rendered patch with the query fragments that overlap it. Let $n _ { i j }$ count the overlapping pixels between rendered patch $Q _ { i }$ and query patch $P _ { j }$ We require a dominant overlap vote and consistent rendered and mean fragment normals:

$$
j = \arg \operatorname* { m a x } _ { j ^ { \prime } } n _ { i j ^ { \prime } } , \quad \frac { n _ { i j } } { \sum _ { j ^ { \prime } } n _ { i j ^ { \prime } } } \geq \tau _ { v } , \quad \hat { \mathbf { N } } _ { i } ^ { \top } \bar { \mathbf { n } } _ { \mathrm { m e r g e d } , i } \geq \tau _ { c } ,\tag{7}
$$

where $\tau _ { v }$ is the minimum vote fraction, $\tau _ { c }$ is the normal similarity threshold, and $\bar { \bf n } _ { \mathrm { m e r g e d } , 1 }$ <sub>i</sub> is the normalized mean normal of the supporting fragments. The target $\mathbf { u } _ { i }$ is the centroid of their combined overlapping pixels. This merges fragmented image evidence into a single correspondence for each rendered patch, from which we estimate the pose using iterative PnP within RANSAC.

Centroid consistency then seeks one-to-one associations between projected map-patch centers and query-patch centers. Using the perspective-projection operator π, we project ${ \bf X } _ { i } \mathrm { ~ t o ~ } \tilde { \bf p } _ { i } = \pi ( { \bf K } , { \pmb \xi } , { \bf X } _ { i } )$ and form mutual nearestneighbor pairs with $\mathbf { p } _ { j }$ , which become the targets ${ \bf { u } } _ { i } .$ . Distance and nearest-neighbor ratio tests reject distant or ambiguous pairs. We estimate the pose from these correspondences using SQPnP [34] within RANSAC.

## 3.5. Final Pose Selection via Rank Aggregation

The particle-filter pose and the successive PnP estimates form a candidate set V. A pose with high silhouette overlap need not have the most accurate geometric alignment, while prediction noise can lower the overlap of a well-aligned pose. We therefore combine eight complementary criteria: composite alignment score, mask recall, mask IoU, edge alignment, depth correlation, normal error, reprojection error, and matched-plane centroid distance. The composite alignment score follows the additive form in Eq. (4), but uses rendered normal-map edges to also capture internal facade and roof structure. Final selection balances all eight criteria instead of maximizing the composite score alone.

Depth correlation measures normalized cross-correlation between predicted and rendered depth over valid buildingmask overlap. Normal error averages angular differences over common building support. Reprojection error measures the mean image-space residual of the fitted inlier correspondences, while centroid distance averages the separation of matched image and rendered patch centers.

Rank aggregation balances these criteria without directly combining their different numerical scales. Let $r _ { k } ( \pmb { \xi } )$ be the candidate’s rank under criterion k, with higher ranks assigned to stronger agreement or smaller errors. We select

$$
\pmb { \xi } ^ { * } = \arg \operatorname* { m a x } _ { \pmb { \xi } \in \mathcal { V } } \frac { 1 } { 8 } \sum _ { k = 1 } ^ { 8 } r _ { k } ( \pmb { \xi } ) .\tag{8}
$$

Each criterion contributes equally, without learning aggregation weights.

## 4. Experiments

We evaluate PlanarCityLoc against published references and adapted LoD baselines. Additional results and evaluation details are provided in the supplementary.

## 4.1. Experiment Settings and Baselines

Datasets. We evaluate on the UAVD4L-LoDv2 and Swiss-EPFLv2 query splits [26, 47], inherited from UAVD4L-LoD and Swiss-EPFL [46], respectively. These dataset extensions introduce LoD1 maps while retaining the original queries. Each query benchmark has two splits: in-Traj. and out-of-Traj. for UAVD4L-LoDv2, and in-Place and out-of-Place for Swiss-EPFLv2. Swiss-EPFL imagery and poses originate from CrossLoc [41]. Our main experiments use LoD3 for UAVD4L-LoDv2 and LoD2 for Swiss-EPFLv2 unless stated otherwise. The released UAVD4L-LoDv2 priors are sensor-derived, whereas Swiss-EPFLv2 priors were synthesized from ground truth by the benchmark authors [46].

Baselines. Table 1 includes textured-map methods and LoD-based methods. We report published references alongside adapted LoD-Loc evaluations (†).

The adapted evaluations assess localization with oracle building masks and off-the-shelf segmentation, separately from published trained configurations. We use rasterized map renderings with ground-truth-rendered masks for LoD-Loc v2 and LoD-Loc v3, and additionally evaluate LoD-Loc v2 with SAM 3 [2] predictions. These variants contrast ideal masks with general-purpose predictions for localization.

Evaluation metrics. Following prior work [26, 47], we report median translation and rotation errors, and recall at 2m–2<sup>◦</sup>, 3m–3<sup>◦</sup>, and 5m–5<sup>◦</sup>. A query is recalled when both errors are strictly below the corresponding thresholds.

## 4.2. Implementation Details

The main method uses SAM 3 [2] for building masks and MoGe-2 ViT-S [38] for depth and normals. Plane extraction uses normal threshold $\tau _ { n } = 0 . 9 9 5$ and relative depth thresholds $\tau _ { d } ~ = ~ 0 . 0 5$ for image patches and 0.02 for rendered patches. We discard patches smaller than 50 pixels.

Coarse localization uses two translation-and-yaw grids followed by medium and fine particle-filter stages, with up to 15 and 6 iterations. Plane refinement uses up to $N = 5$ outer iterations. Standard correspondence filtering requires normal consistency $\tau _ { c } = 0 . 7$ and at least six matches, with a 20 px RANSAC threshold. Strongly fragmented image patches trigger stricter matching, while insufficient building coverage skips PnP. Search configurations and matching thresholds were established empirically during development and fixed for the reported evaluations.

<table><tr><td></td><td></td><td></td><td colspan="7">UAVD4L-LoDv2 query splits</td><td colspan="8">Swiss-EPFLv2 query splits</td></tr><tr><td></td><td></td><td></td><td colspan="3"></td><td colspan="3"></td><td colspan="3">out-of-Traj.</td><td colspan="3">in-Place</td><td colspan="3">out-of-Place</td></tr><tr><td>Method</td><td>Vision Model</td><td>Setup</td><td>2m-2°</td><td>3m-3°</td><td>5m-5°</td><td>T.e./R.e.</td><td>2m-2° 3m-3°</td><td>5m-5°</td><td>T.e./R.e.</td><td>2m-2°</td><td>3m-3°</td><td>5m-5°</td><td>T.e./R.e.</td><td>2m-2°</td><td>3m-3°</td><td>5m-5°</td><td>T.e./R.e.</td></tr><tr><td>Released priors</td><td></td><td></td><td>0.0</td><td>0.0</td><td>4.3</td><td>6.48/1.63</td><td>0.0</td><td>0.0 0.4</td><td></td><td>11.12/1.33 0.0</td><td>0.0</td><td>0.6</td><td>17.59/3.86</td><td>0.0</td><td>0.0</td><td>1.1</td><td>17.88/3.94</td></tr><tr><td>Textured Mesh-based Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">UAVLoc [39] LoFTR</td><td></td><td>84.98</td><td>88.09</td><td>88.90</td><td>0.81/0.29</td><td>91.56</td><td>92.02</td><td>92.11</td><td>0.79/0.20</td><td>26.40 46.21</td><td>62.22</td><td></td><td>3.06/1.86 68.87</td><td>81.00</td><td>84.96</td><td>1.12/1.08</td></tr><tr><td>UAVLoc [39]</td><td>RoMa</td><td>0</td><td>93.27</td><td>93.70</td><td>93.77</td><td>0.78/0.25</td><td>95.03</td><td>95.53 95.53</td><td>0.73/0.22</td><td>45.95</td><td>66.77</td><td>80.73</td><td>2.08/1.29</td><td>89.18</td><td>89.68</td><td>98.84</td><td>0.77/1.04</td></tr><tr><td colspan="2">LoD-based Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Requires Training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoD-Loc [46] LoD-Loc v2 [47]</td><td>Custom U-Net SAM2 + U-Net</td><td></td><td>49.56</td><td>71.82</td><td>89.09</td><td>3.32/1.48</td><td>54.20</td><td>75.05 89.51</td><td>3.33/1.18</td><td>36.79</td><td>50.56</td><td>69.77</td><td>2.87/1.78</td><td>14.24</td><td>31.39</td><td>59.89</td><td>8.73/2.78 2.64/0.73</td></tr><tr><td>LoD-Loc v3 [26]</td><td>SAM-Prompter</td><td></td><td>93.70 97.60</td><td>98.40</td><td>99.50 99.70</td><td>0.72/0.15 0.49/0.13</td><td>97.90</td><td>99.80 100.00</td><td>0.71/0.14</td><td>54.20</td><td>74.60 79.90</td><td>92.00</td><td>1.83/0.85</td><td>31.40</td><td>58.53 64.10</td><td>86.30 88.90</td><td>2.43/0.83</td></tr><tr><td>Oracle (GT masks)</td><td></td><td></td><td></td><td>98.90</td><td></td><td></td><td>97.40</td><td>99.00 99.40</td><td>0.60/0.12</td><td>58.60</td><td></td><td>95.40</td><td>1.61/0.77</td><td>36.90</td><td></td><td></td><td></td></tr><tr><td>LoD-Loc v2† [47]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoD-Loc v3† [26]</td><td></td><td>GT GT</td><td>57.2 54.5</td><td>78.8 78.7</td><td>92.8</td><td>1.85/1.23 1.87/1.23</td><td>64.9</td><td>94.5 98.3 96.4</td><td>1.76/0.94 2.15/0.95</td><td>74.9 56.3</td><td>90.6 72.2</td><td>93.5</td><td>1.36/0.85</td><td>43.3</td><td>75.2 73.6</td><td>93.9 93.9</td><td>2.15/0.78 2.27/0.79</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>90.6</td><td></td><td>41.3</td><td>83.8</td><td></td><td></td><td></td><td>78.9</td><td>1.79/0.91</td><td>40.1</td><td></td><td></td><td></td></tr><tr><td>Off-the-shelf MC-Loc [35]</td><td>DINOv2</td><td></td><td>1.20</td><td>4.10</td><td>17.40</td><td>8.29/2.58</td><td>2.40</td><td></td><td></td><td>7.02/2.29 0.90</td><td>4.40</td><td>17.50</td><td>8.18/2.54</td><td>2.90</td><td>9.00</td><td>30.20</td><td>6.23/1.97</td></tr><tr><td>MC-Loc [35]</td><td>RoMa</td><td>0 0</td><td>0.10</td><td>0.60</td><td>3.30</td><td>10.60/8.60</td><td>0.20</td><td>7.40 0.90</td><td>26.10 3.30</td><td>16.90/3.88 0.20</td><td>1.20</td><td>4.80</td><td>9.80/2.65</td><td>0.70</td><td>2.10</td><td>11.50</td><td>10.30/3.97</td></tr><tr><td>LoD-Loc v2† [47]</td><td>SAM3 [2]</td><td>0</td><td>35.7</td><td>67.8</td><td>81.5</td><td>2.33/1.27</td><td>64.3</td><td>93.3 98.2</td><td>1.77/0.95</td><td>9.4</td><td>19.9</td><td>39.7</td><td>6.51/1.83</td><td>17.4</td><td>33.0</td><td>66.2</td><td>3.90/1.33</td></tr><tr><td>PlanarCityLoc (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cost Vol. Selection</td><td>MoGe-2 + SAM3 [2]</td><td>0</td><td>27.6</td><td>59.9</td><td>79.4</td><td>2.62/1.49</td><td>34.9</td><td>74.1 95.3</td><td>2.35/1.22</td><td>15.4</td><td>30.2</td><td>48.7</td><td>5.04/1.72</td><td>11.3</td><td>23.2</td><td>57.5</td><td>4.42/1.41</td></tr><tr><td>+ Particle Filter</td><td>MoGe-2 + SAM3 [2]</td><td>0</td><td>36.0</td><td>67.7</td><td>82.8</td><td>2.36/1.30</td><td>60.8</td><td>90.6</td><td>98.5 1.81/0.98</td><td>24.9</td><td>41.6</td><td>61.8</td><td>3.54/1.54</td><td>14.5</td><td>31.1</td><td>70.2</td><td>3.99/1.23</td></tr><tr><td></td><td>+ Plane-based PnP MoGe-2 + SAM3 [2]</td><td>0</td><td>75.4</td><td>90.8</td><td>97.2</td><td>1.28/0.68</td><td>80.6</td><td>95.1 99.0</td><td></td><td>1.26/0.69 20.6</td><td>39.0</td><td>59.7</td><td>3.70/1.90</td><td>48.5</td><td>72.3</td><td>89.4</td><td>1.98/1.21</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1. Quantitative comparison on UAVD4L-LoDv2 and Swiss-EPFLv2. Setup: Off-the-shelf, In-domain training, Cross-dataset training, or GT oracle masks. $\mathbf { C } ^ { \ast }$ uses the InsLoD-Loc training set [26]. Recall is in %. T.e./R.e. are median translation (m) and rotation (<sup>◦</sup>) errors. For evaluated runs, recall requires both pose thresholds. Best/second-best distinct values per column within each setup group are bold/underlined. †: adapted re-runs (see Sec. 4.1).

## 4.3. Evaluation Results

Quantitative results. We report the results in Tab. 1. PlanarCityLoc outperforms the evaluated prior off-theshelf LoD-based methods in recall across all four splits and all three thresholds. Explicit plane correspondences substantially improve the particle-filter estimates on both UAVD4L-LoDv2 splits and Swiss-EPFLv2 out-of-Place. Textured-map UAVLoc remains stronger, benefiting from richer appearance cues unavailable in textureless geometry.

Swiss-EPFLv2 in-Place is more challenging for plane refinement, with $2 \mathrm { m } { - } 2 ^ { \circ }$ recall changing from 24.9% to 20.6%. This split has fewer visible map patches and a larger imbalance between predicted and rendered patch counts than out-of-Place, consistent with less reliable plane associations. Refinement also produces fewer successful PnP proposals, and candidate selection sometimes favors an inaccurate update. These observations highlight the importance of sufficiently rich, consistent planar evidence.

Qualitative analysis. Figure 3 shows that different methods can produce visually plausible alignments while yielding substantially different pose errors. Good image-space overlap alone is therefore insufficient to establish an accurate camera pose. The staged examples in Fig. 4 show how plane correspondences improve upon coarse alignment.

## 4.4. Ablation Studies

We ablate components on fixed 100-query subsets per UAVD4L-LoDv2 split. Pose-selection and LoD comparisons use the full splits.

<table><tr><td>Map</td><td> $2 \mathrm { m } { - } 2 ^ { \circ }$ </td><td> $3 \mathrm { m } { - } 3 ^ { \circ }$ </td><td> $5 \mathrm { m } { - } 5 ^ { \circ }$ </td><td> $\mathrm { T . e . ( m ) }$ </td><td> $\mathrm { R . e . } ( ^ { \circ } )$ </td></tr><tr><td>LoD1</td><td>23.8 / 44.8</td><td>54.1 / 74.3</td><td>82.2 / 93.0</td><td>2.85 / 2.13</td><td>1.26 / 0.95</td></tr><tr><td>LoD3</td><td>75.4 / 80.6</td><td>90.8 / 95.1</td><td>97.2 / 99.0</td><td>1.28 / 1.26</td><td>0.68 / 0.69</td></tr></table>

Table 2. LoD1 vs. LoD3. Full UAVD4L-LoDv2 splits (in-Traj. / out-of-Traj.). Joint recalls (%) and median errors.

<table><tr><td>Selection rule</td><td>2m-2°</td><td>3m-3°</td><td> $5 \mathrm { m } { - } 5 ^ { \circ }$ </td><td>T.e. (m)</td><td> $\mathbf { R . e . } ( ^ { \circ } )$ </td></tr><tr><td>Particle filter only</td><td></td><td></td><td>36.0 / 60.867.7 / 90.682.8 / 98.5 2.36 / 1.811.30 / 0.98</td><td></td><td></td></tr><tr><td>Max PnP inliers</td><td></td><td></td><td>65.8 / 74.884.1 / 91.1 95.4 / 98.21.49 / 1.370.82 / 0.76</td><td></td><td></td></tr><tr><td>Max mask IoU</td><td></td><td></td><td>67.4 / 80.788.7 / 95.396.8 / 99.11.54 / 1.320.79 / 0.75</td><td></td><td></td></tr><tr><td>Max edge score</td><td></td><td></td><td>76.6 / 79.6 90.6 / 94.6 97.1 / 99.01.25 / 1.330.69 / 0.77</td><td></td><td></td></tr><tr><td>Max composite score 72.3 / 81.289.7 / 96.0 97.2 / 99.0 1.39 / 1.30 0.72 / 0.74</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Equal-weight ranks</td><td></td><td></td><td>75.4 / 80.690.8 / 95.1 97.2 / 99.01.28 / 1.26 0.68 / 0.69</td><td></td><td></td></tr></table>

Table 3. Pose selection. Full UAVD4L-LoDv2 splits (in-Traj. / out-of-Traj.), using identical candidate pools. Best/second-best values at shown precision are bold/underlined.

Geometric abstraction (LoDs). Table 2 compares LoD1 and LoD3 on the complete UAVD4L-LoDv2 splits. Performance decreases with lower geometric detail. Coarser building shapes provide fewer faithful counterparts to image-side facade and roof planes, increasing the geometric gap and reducing correspondence accuracy.

Refinement stages. The staged results in Tab. 1 and the examples in Fig. 4 show how coarse image-space alignment provides a useful initialization for explicit 3D refinement. Plane-based proposals substantially improve three splits, showing the complementary value of geometric correspondences beyond silhouette scoring.

![](images/546182d15ee4738b4997d7858844cd0adb610edd730a3adde68637686f87ff25.jpg)

Figure 3. Qualitative comparison on UAVD4L-LoDv2 and Swiss-EPFLv2. Red overlays show LoD edges at each estimated pose Labels give translation (m) / rotation (<sup>◦</sup>) errors. Similar visual alignments can correspond to different pose errors.  
![](images/a971e2c7d41816b3421f3086c4aa7ecfe4f24d1ffc16ec2c568f8e4072eeadd1.jpg)  
Figure 4. Pose estimates through PlanarCityLoc on UAVD4L-LoDv2. Each row shows the query image and map overlays at the ground truth, sensor-prior, particle-filter, and final selected poses. Labels give translation (m) / rotation (<sup>◦</sup>) errors.

Pose selection. Table 3 compares selection rules on identical candidate pools. Equal-weight rank aggregation gives the lowest median rotation error on both UAVD4L-LoDv2 splits and the lowest median translation error on out-of-Traj., while maintaining near-best strict recall. No single alignment cue consistently gives the best pose. These results support balancing complementary criteria through ranks, which avoid dependence on their different numerical scales without learning aggregation weights.

![](images/164bcbe94f68583dc2668246d33c705dd24a97fa3f80925fe6cfb24b99e2b2dc.jpg)

![](images/f6169454601fa25d10f267d485242038c290a58a26ebed6a2875a55253a92bf8.jpg)  
Figure 5. Prior pose sensitivity on UAVD4L-LoDv2. Curves show means of split-wise median errors for the same 300 queries per split. Synthetic priors use expanded search. Native-prior points are separate references.

Sensitivity to prior pose noise. We investigate sensitivity to the initial pose by increasing translation and orientation noise, with search ranges expanded to match the tested perturbations (Fig. 5). Larger perturbations lead to higher final errors. Nevertheless, plane refinement reduces both translation and rotation errors relative to particle filtering at every tested noise level, measured by the mean of the split-wise medians. The experiment tests refinement under substantial initial uncertainty beyond the released priors.

Normal predictors. Table 4 compares geometry predictors on matched 100-query subsets. Each model supplies both depth and normals, while SAM 3 masks remain fixed. MoGe-2 outperforms the evaluated Metric3D v2 configurations in strict recall. Larger MoGe-2 backbones improve recall, especially on in-Traj. Main results use ViT-S.

## 4.5. Limitations

PlanarCityLoc requires a useful coarse pose and sufficient reliable plane correspondences. Performance depends on the accuracy of the foundation-model predictions and the quality of the LoD geometry. In particular, approximating complex building shapes with LoD1 can make plane assignment noisier and reduce localization accuracy. The multi-stage pipeline combines neural predictions with repeated rendering and cannot yet operate onboard in real time without further optimization. Future work targets fast initialization and hardware-aware real-time optimization.

## 5. Conclusion

We presented PlanarCityLoc, which grounds silhouettebased aerial localization in the 3D planar geometry of lightweight, textureless city models. By establishing plane correspondences and refining pose through PnP, our method achieves state-of-the-art localization recall among the evaluated off-the-shelf LoD-based methods without datasetspecific training. Our experiments demonstrate the value of planes for further improving pose estimation beyond image-space alignment. Effective refinement requires sufficient reliable plane correspondences. The current multistage pipeline is not yet real-time, motivating more efficient initialization and hardware-aware optimization for onboard deployment.

<table><tr><td>Model</td><td>Backbone</td><td>2m-2°</td><td>3m-3°</td><td>5m-5°</td><td>T.e.(m)</td><td>R.e.(°)</td></tr><tr><td>Metric3D v2 ViT-S</td><td></td><td>33.0 / 55.0 60.0 / 86.0</td><td></td><td>82.0 / 94.0</td><td>2.55 / 1.87 1.25 / 0.89</td><td></td></tr><tr><td>Metric3D v2 ViT-L</td><td></td><td></td><td>31.0 / 58.0 62.0 / 78.0</td><td>87.0 / 90.0</td><td>2.41 / 1.801.33 / 0.96</td><td></td></tr><tr><td>Metric3D v2 ViT-G2</td><td></td><td>58.0 / 59.081.0 / 81.0</td><td></td><td>95.0 / 89.0</td><td>1.70 / 1.73 0.91 / 0.90</td><td></td></tr><tr><td>MoGe-2</td><td>ViT-S</td><td></td><td></td><td>75.0 / 84.094.0 / 94.098.0 / 99.0</td><td>1.14 / 1.37 0.68 / 0.78</td><td></td></tr><tr><td>MoGe-2</td><td>ViT-B</td><td></td><td></td><td></td><td>82.0 / 83.0 97.0 / 93.098.0 / 98.01.05 / 1.230.65 / 0.65</td><td></td></tr><tr><td>MoGe-2</td><td>ViT-L</td><td></td><td></td><td></td><td>87.0 / 84.096.0 / 97.0 97.0 / 100.01.09 / 1.250.61 / 0.62</td><td></td></tr></table>

Table 4. Geometry predictors. Each model supplies both normals and depth. SAM3 masks are unchanged. All rows use the same 100 queries per UAVD4L-LoDv2 split (in-Traj. / out-of-Traj.). Recall uses joint thresholds. T.e./R.e. are medians. Bold/underline mark best/second-best at shown precision.

## Acknowledgements

Nikolas Rieger acknowledges support from the DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, funded by the Federal Ministry of Research, Technology and Space.

## References

[1] Yiming Cai, Yao Zhou, Hongwen Zhang, Yuli Xia, Peng Qiao, and Junsuo Zhao. Review of target geo-location al gorithms for aerial remote sensing cameras without control points. Appl. Sci., 2022. 1

[2] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoub hik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Radle, Triantafyllos Afouras, Effrosyni Mavroudi,¨ Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, RISHI HAZRA, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollar, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. SAM 3: Segment anything with concepts. In The Fourteenth International Con ference on Learning Representations, 2026. 2, 3, 5, 6, 11, 14

[3] Yunqian Cheng and Roberto Manduchi. PALMS: Planebased accessible indoor localization using mobile smartphones. In 2024 14th International Conference on Indoor Positioning and Indoor Navigation (IPIN), pages 1–7. IEEE, 2024. 3

[4] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on

computer vision and pattern recognition workshops, pages 224–236, 2018. 2

[5] Oussema Dhaouadi, Riccardo Marin, Johannes Meier, Jacques Kaiser, and Daniel Cremers. OrthoLoC: UAV 6-DoF localization and calibration using orthographic geodata. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. 2

[6] Oussema Dhaouadi, Zuria Bauer, Johannes Meier, Olaf Wysocki, Marc Pollefeys, and Daniel Cremers. OrthoTrack: Continuous 6-DoF UAV trajectory estimation anchored in public orthophotos. In Computer Vision – ECCV 2026, pages 55–73. Springer Nature Switzerland, 2026. 2

[7] Oussema Dhaouadi, Zuria Bauer, Johannes Meier, Marc Pollefeys, Daniel Cremers, and Olaf Wysocki. MatchCity-Loc: Aerial localization in LoD city models via surface normal matching. In IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2027. 2

[8] Johan Edstedt, Qiyu Sun, Georg Bokman, M¨ arten˚ Wadenback, and Michael Felsberg. RoMa: Robust Dense¨ Feature Matching. IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2

[9] Johan Edstedt, David Nordstrom, Yushan Zhang, Georg¨ Bokman, Jonathan Astermark, Viktor Larsson, Anders Hey-¨ den, Fredrik Kahl, Marten Wadenb˚ ack, and Michael Fels-¨ berg. RoMa v2: Harder Better Faster Denser Feature Matching. arXiv preprint arXiv:2511.15706, 2025. 2

[10] Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications ofthe ACM, 24(6):381–395, 1981. 2, 4, 13

[11] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In European Conference on Computer Vision, pages 241–258. Springer, 2024. 2, 3

[12] Gerhard Groger, Thomas H Kolbe, Claus Nagel, and Karl-¨ Heinz Hafele. OGC city geography markup language ¨ (CityGML) encoding standard. Open Geospatial Consortium: Wayland, 2012. 1

[13] Markus Gross, Andreas Greiner, Sai Bharadhwaj Matha, Felix Soest, Daniel Cremers, and Henri Meeß. SafeLand: Safe autonomous landing in unknown environments with Bayesian semantic mapping. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, 2026. 1

[14] Markus Gross, Sai B. Matha, Aya Fahmy, Rui Song, Daniel Cremers, and Henri Meess. OccuFly: A 3D vision benchmark for semantic scene completion from the aerial perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 3

[15] Markus Gross, Sai Bharadhwaj Matha, Rui Song, Viswanathan Muthuveerappan, Conrad Christoph, Julius Huber, and Daniel Cremers. SegFly: A dataset and 2D-3D-2D paradigm for aerial RGB-Thermal semantic segmentation at scale. In Proceedings of the European Conference on Computer Vision (ECCV), 2026. 1

[16] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):10579–10596, 2024. 2, 3

[17] Peter Karkus, David Hsu, and Wee Sun Lee. Particle filter networks with application to visual localization. In Conference on robot learning, pages 169–178. PMLR, 2018. 4

[18] Bingxin Ke, Kevin Qu, Tianfu Wang, Nando Metzger, Shengyu Huang, Bo Li, Anton Obukhov, and Konrad Schindler. Marigold: Affordable adaptation of diffusionbased image generators for image analysis. IEEE Transac tions on Pattern Analysis and Machine Intelligence, 2025. 3

[19] Tommie Kerssies, Niccolo Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, and Daan De Geus. Your vit is secretly an image segmentation model. In Proceedings of the computer vision and pattern recognition conference, pages 25303–25313, 2025. 14

[20] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings ofthe IEEE/CVF international confer ence on computer vision, pages 4015–4026, 2023. 2

[21] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance differentiable rendering. ACM Transac tions on Graphics, 39(6), 2020. 11

[22] Heping Li, Changliang Xue, Feng Wen, Hongbo Zhang, and Wei Gao. BSP-MonoLoc: Basic semantic primitives based monocular localization on roads. In 2021 IEEE/RSJ Interna tional Conference on Intelligent Robots and Systems (IROS), pages 5470–5475. IEEE, 2021. 3

[23] Francesco Nex, EK Stathopoulou, F Remondino, MY Yang, L Madhuanand, Yogender Yogender, B Alsadik, M Weinmann, B Jutzi, and R Qin. Usegeo-a uav-based multi-sensor dataset for geospatial research. ISPRS Open Journal ofPho togrammetry and Remote Sensing, page 100070, 2024. 1

[24] Linfei Pan, Daniel Bar´ ath, Marc Pollefeys, and Johannes L´ Schonberger. Global structure-from-motion revisited. In¨ European Conference on Computer Vision, pages 58–77. Springer, 2024. 2

[25] Vojtech Panek, Zuzana Kukelova, and Torsten Sattler. MeshLoc: Mesh-based visual localization. In Proceedings of the European Conference on Computer Vision, pages 589– 609, 2022. 2

[26] Shuaibang Peng, Juelin Zhu, Xia Li, Kun Yang, Yu Liu, Maojun Zhang, and Shen Yan. LoD-Loc v3: Generalized aerial localization in dense cities using instance silhouette alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12193– 12205, 2026. 1, 2, 3, 5, 6, 11

[27] Mason Peterson, Qingyuan Li, Yixuan Jia, Fernando Cladera, Carlos Nieto-Granda, Camillo Jose Taylor, and

Jonathan P How. Meridian: Metric-semantic primitive matching for cross-view geo-localization beyond urban environments. arXiv preprint arXiv:2606.06312, 2026. 3

[28] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Seg-¨ ment anything in images and videos. In International Conference on Learning Representations, pages 28085–28128, 2025. 2

[29] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and Marcin Dymczyk. From coarse to fine: Robust hierarchical localization at large scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12716–12725, 2019. 2

[30] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In CVPR, 2020. 2

[31] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016. 1, 2

[32] Fabien Servant, Eric Marchand, Pascal Houlier, and Isabelle Marchal. Visual planes-based simultaneous localization and model refinement for augmented reality. In 2008 19th International Conference on Pattern Recognition, pages 1–4. IEEE, 2008. 3

[33] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In CVPR, 2021. 2

[34] George Terzakis and Manolis Lourakis. A consistently fast and globally optimal solution to the perspective-n-point problem. In European Conference on Computer Vision, pages 478–494. Springer International Publishing, 2020. 5

[35] Gabriele Trivigno, Carlo Masone, Barbara Caputo, and Torsten Sattler. The unreasonable effectiveness of pretrained features for camera pose refinement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12786–12798, 2024. 2, 4, 6

[36] Ayumi Umemura, Toshinori Kuwahara, Marc Pollefeys, and Daniel Barath. Z-FLoc: Zero-shot floorplan localization via geometric primitives. arXiv preprint arXiv:2606.04788, 2026. 3

[37] Daniel KD Villa, Alexandre S Brandao, and Mario´ Sarcinelli-Filho. A survey on load transportation using multirotor uavs. Journal ofIntelligent & Robotic Systems, 98(2): 267–296, 2020. 1

[38] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. MoGe-2: Accurate monocular geometry with metric scale and sharp details. Advances in Neural Information Processing Systems, 38:35928–35959, 2025. 2, 3, 5, 11

[39] Rouwan Wu, Xiaoya Cheng, Juelin Zhu, Yuxiang Liu, Maojun Zhang, and Shen Yan. Uavd4l: A large-scale dataset for uav 6-dof localization. In 2024 International Conference on 3D Vision (3DV), pages 1574–1583. IEEE, 2024. 1, 2, 6

[40] Olaf Wysocki, Benedikt Schwab, Christof Beil, Christoph Holst, and Thomas H Kolbe. Reviewing open data seman-

tic 3d city models to develop novel 3d reconstruction meth ods. The International Archives ofthe Photogrammetry, Remote Sensing and Spatial Information Sciences, 48:493–500, 2024. 1

[41] Qi Yan, Jianhao Zheng, Simon Reding, Shanci Li, and Iordan Doytchinov. CrossLoc: Scalable aerial localization assisted by multimodal synthetic data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17358–17368, 2022. 5

[42] Shen Yan, Xiaoya Cheng, Yuxiang Liu, Juelin Zhu, Rouwan Wu, Yu Liu, and Maojun Zhang. Render-and-compare: Cross-view 6-dof localization from noisy prior. In 2023 IEEE International Conference on Multimedia and Expo (ICME), pages 2171–2176. IEEE, 2023. 2

[43] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiash Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10371–10381, 2024. 2, 3

[44] Hanqiao Ye, Yuzhou Liu, Yangdong Liu, and Shuhan Shen. PlanaReLoc: Camera relocalization in 3d planar primitives via region-based structure matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. 3

[45] Yibin Ye, Xichao Teng, Shuo Chen, Leqi Liu, Kun Wang, Xiaokai Song, and Zhang Li. Exploring the best way for UAV visual localization under low-altitude multi-view observation condition: A benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 1731–1741, 2026. 2

[46] Juelin Zhu, Shen Yan, Long Wang, Shengyue Zhang, Yu Liu, and Maojun Zhang. LoD-Loc: Aerial visual localization using LoD 3d map with neural wireframe alignment. Advances in Neural Information Processing Systems, 37: 119063–119098, 2024. 1, 2, 5, 6

[47] Juelin Zhu, Shuaibang Peng, Long Wang, Hanlin Tan, Yu Liu, Maojun Zhang, and Shen Yan. LoD-Loc v2: Aerial visual localization over low level-of-detail city models using explicit silhouette alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 26610–26621, 2025. 1, 2, 3, 5, 6, 11

# Supplementary Material

## A. Overview

As demonstrated in the main paper, PlanarCityLoc improves UAV localization through 3D planar constraints that address the ambiguities of silhouette alignment. Specifically, we first explain Computational Complexity in Sec. B and discuss Baseline Evaluation in Sec. C. We then present Extended Results in Sec. D, along with Extended Ablation Studies in Sec. E and a Robustness Analysis in Sec. F. Finally, we consider Limitations and Future Work in Sec. G and Ethical Considerations in Sec. H.

Experiments use off-the-shelf SAM 3 [2] masks and MoGe-2 ViT-S [38] geometry unless stated otherwise. We report joint localization recall and median translation and rotation errors, following the main paper.

## B. Computational Complexity

Our coarse stage follows the image-space search paradigm of prior work [26, 47]. PlanarCityLoc then introduces explicit 3D constraints through plane correspondences and PnP, avoiding another dense search over pose hypotheses.

On UAVD4L-LoDv2 in-Traj. and out-of-Traj., the rendering, matching, and PnP computations used in refinement take an average of 0.52 and 0.48 s per query on an NVIDIA H100 NVL. On Swiss-EPFLv2 in-Place, these components take 0.23 s on an NVIDIA A40. These component timings exclude one-time initialization, I/O, data transfers, plane extraction, and final pose selection.

Mean wall-clock latency is 6.42 s per query on UAVD4L-LoDv2 in-Traj. (H100 NVL) and 8.75 s on Swiss-EPFLv2 in-Place (A40), including image loading and transfers but excluding one-time initialization.

On the two UAVD4L-LoDv2 splits, refinement and final selection improve joint 2 m–2<sup>◦</sup> localization recall from 36.0% to 75.4% and from 60.8% to 80.6%, respectively (Tab. A1). This demonstrates the value of complementing coarse image-space alignment with correspondence-driven 3D pose updates.

## C. Baseline Evaluation

We distinguish published reference results from adapted evaluations in the main paper’s Tab. 1.

Our adapted evaluations share PlanarCityLoc’s nvdiffrast [21] rasterizer and start from the released benchmark priors, with coarse pose search followed by particle filtering [26, 47].

Ground-truth-rendered masks allow us to examine localization with ideal segmentation. For LoD-Loc v2 [47], SAM 3 [2] additionally tests off-the-shelf segmentation. Published references retain their original configurations.

Adapted evaluations of the public implementations use the same benchmark query splits with modified rendering and segmentation.

## D. Extended Results

Per-split breakdowns. We report the complete staged evaluation in Tab. A1. This evaluation details the performance of the cost volume, particle filter, and our plane matching approach across all benchmark splits. The fullsplit results show substantial gains from plane matching on both UAVD4L-LoDv2 splits and Swiss-EPFLv2 outof-Place, supporting the use of 3D constraints beyond silhouette agreement. We also report the full-split UAVD4L-LoDv2 LoD1 ablation and both Swiss-EPFLv2 splits with LoD2 maps. Swiss-EPFLv2 in-Place is more challenging for refinement (Tab. A1). Figure A1 compares scene examples and planar support. At the particle-filter pose, this split has fewer rendered map patches and a larger predicted-torendered patch-count ratio than out-of-Place. The examples contrast sparse building coverage and construction areas with multi-building scenes. Lower map support and greater patch-count imbalance suggest more challenging image-tomap plane association.

Plane extraction against LoD primitives. Figure A2 compares LoD planes rendered from the mesh with planes segmented from predicted normals. LoD planes follow the mesh tessellation and remain largely insensitive to the normal-consistency threshold. In contrast, predicted planes depend on normal and depth consistency as well as minimum region size. Fine settings can fragment a facade, while coarse settings can merge distinct surfaces. The examples illustrate this trade-off in the resulting planar representation.

Failure cases. Planar refinement requires useful imageto-map alignment. Figure A3 illustrates initialization failures. Off-the-shelf segmentation under domain shift can miss small buildings, misclassify lakes or parking lots, or produce noisy masks, favoring incorrect cost-volume hypotheses. Disjoint query and rendered masks then supply no overlap-based distinction between nearby hypotheses, weakening the initialization for plane matching. Beyond pure segmentation errors, matching can also degrade due to weak building visibility, ambiguous facade normals from repetitive architecture, or temporal mismatches between outdated LoD geometry (e.g., ground artifacts modeled as buildings) and the true UAV capture. When geometric and mask cues disagree strongly, reliable confidence estimation is needed to avoid incorrect pose updates.

<table><tr><td>Dataset / map</td><td>Split</td><td>Stage</td><td>Queries</td><td>1m-1°</td><td> $2 \mathrm { m } { - } 2 ^ { \circ }$ </td><td> $3 \mathrm { m } { - } 3 ^ { \circ }$ </td><td> $5 \mathrm { m } { - } 5 ^ { \circ }$ </td><td>T.e.</td><td>R.e.</td></tr><tr><td>UAVD4L-LoDv2 / LoD3</td><td>in-Traj.</td><td>Coarse search</td><td>1604</td><td>1.4</td><td>27.6</td><td>59.9</td><td>79.4</td><td>2.62</td><td>1.49</td></tr><tr><td>UAVD4L-LoDv2 / LoD3</td><td>in-Traj.</td><td>Particle filter</td><td>1604</td><td>3.1</td><td>36.0</td><td>67.7</td><td>82.8</td><td>2.36</td><td>1.30</td></tr><tr><td>UAVD4L-LoDv2 / LoD3</td><td>in-Traj.</td><td>PlanarCityLoc</td><td>1604</td><td>36.3</td><td>75.4</td><td>90.8</td><td>97.2</td><td>1.28</td><td>0.68</td></tr><tr><td>UAVD4L-LoDv2 / LoD3</td><td>out-of-Traj.</td><td>Coarse search</td><td>2192</td><td>2.7</td><td>34.9</td><td>74.1</td><td>95.3</td><td>2.35</td><td>1.22</td></tr><tr><td>UAVD4L-LoDv2 / LoD3</td><td>out-of-Traj.</td><td>Particle filter</td><td>2192</td><td>8.8</td><td>60.8</td><td>90.6</td><td>98.5</td><td>1.81</td><td>0.98</td></tr><tr><td>UAVD4L-LoDv2 / LoD3</td><td>out-of-Traj.</td><td>PlanarCityLoc</td><td>2192</td><td>35.5</td><td>80.6</td><td>95.1</td><td>99.0</td><td>1.26</td><td>0.69</td></tr><tr><td>Swiss-EPFLv2 / LoD2</td><td>in-Place</td><td>Coarse search</td><td>712</td><td>2.5</td><td>15.4</td><td>30.2</td><td>48.7</td><td>5.04</td><td>1.72</td></tr><tr><td>Swiss-EPFLv2 / LoD2</td><td>in-Place</td><td>Particle filter</td><td>712</td><td>5.2</td><td>24.9</td><td>41.6</td><td>61.8</td><td>3.54</td><td>1.54</td></tr><tr><td>Swiss-EPFLv2 / LoD2</td><td>in-Place</td><td>PlanarCityLoc</td><td>712</td><td>4.4</td><td>20.6</td><td>39.0</td><td>59.7</td><td>3.70</td><td>1.90</td></tr><tr><td>Swiss-EPFLv2 / LoD2</td><td>out-of-Place</td><td>Coarse search</td><td>379</td><td>1.1</td><td>11.3</td><td>23.2</td><td>57.5</td><td>4.42</td><td>1.41</td></tr><tr><td>Swiss-EPFLv2 / LoD2</td><td>out-of-Place</td><td>Particle filter</td><td>379</td><td>1.3</td><td>14.5</td><td>31.1</td><td>70.2</td><td>3.99</td><td>1.23</td></tr><tr><td>Swiss-EPFLv2 / LoD2</td><td>out-of-Place</td><td>PlanarCityLoc</td><td>379</td><td>7.9</td><td>48.5</td><td>72.3</td><td>89.4</td><td>1.98</td><td>1.21</td></tr><tr><td>UAVD4L-LoDv2 / LoD1</td><td>in-Traj.</td><td>Coarse search</td><td>1604</td><td>2.1</td><td>19.3</td><td>45.3</td><td>72.5</td><td>3.24</td><td>1.60</td></tr><tr><td>UAVD4L-LoDv2 / LoD1</td><td>in-Traj.</td><td>Particle filter</td><td>1604</td><td>2.7</td><td>22.0</td><td>51.1</td><td>73.8</td><td>2.94</td><td>1.38</td></tr><tr><td>UAVD4L-LoDv2 / LoD1</td><td>in-Traj.</td><td>PlanarCityLoc</td><td>1604</td><td>3.6</td><td>23.8</td><td>54.1</td><td>82.2</td><td>2.85</td><td>1.26</td></tr><tr><td>UAVD4L-LoDv2 / LoD1</td><td>out-of-Traj.</td><td>Coarse search</td><td>2192</td><td>2.3</td><td>24.3</td><td>56.9</td><td>86.4</td><td>2.75</td><td>1.25</td></tr><tr><td>UAVD4L-LoDv2 / LoD1</td><td>out-of-Traj.</td><td>Particle filter</td><td>2192</td><td>3.9</td><td>40.8</td><td>75.2</td><td>92.8</td><td>2.23</td><td>1.03</td></tr><tr><td>UAVD4L-LoDv2 / LoD1</td><td>out-of-Traj.</td><td>PlanarCityLoc</td><td>2192</td><td>8.3</td><td>44.8</td><td>74.3</td><td>93.0</td><td>2.13</td><td>0.95</td></tr></table>

Table A1. Full-split staged results. Recalls are joint percentages. Translation/rotation errors are medians in meters/degrees. Every stage is evaluated on the same queries within a split.

![](images/d35df33d4998b8c0aab834bd580a798c5b9c874a9459a6cade30791615945c43.jpg)  
Figure A1. Scene context and planar support on Swiss-EPFLv2. Left: views near each split’s median map-patch count and predicted-torendered ratio. Right: full-split distributions at the particle-filter pose. Boxes show interquartile ranges, lines show medians, and whiskers show the full range. The ratio axis is logarithmic.

## E. Extended Ablation Studies

We ablate core design choices on the same fixed 100-query subset per UAVD4L-LoDv2 split, holding other pipeline settings constant.

Coarse scoring components. Table A2 demonstrates the impact of different geometric cues during coarse search and particle filtering. Relying solely on mask intersection-overunion (IoU) provides a weak signal under occlusion. Edges add boundary constraints, while depth helps distinguish hypotheses with similar silhouettes but different spatial configurations. Combining all three cues gives the highest strict recall on both subsets, supporting their complementary role in initialization.

Plane extraction criteria. Table A3 explores the logic for grouping predicted surface normals into planar primitives. Grouping by normal consistency alone captures facades but can merge adjacent surfaces with similar orientations. Depth continuity helps separate these surfaces, but alone does not distinguish their orientation. Combining both criteria gives the highest strict recall and lowest median translation error on both subsets, while normal consistency alone remains competitive at looser thresholds.

![](images/e3ef66cf18edf5e0d5c5f61da1bdf4fdba42f3856b1b622c83e9402285c8208d.jpg)

Figure A2. Plane extraction qualitative comparison. LoD primitives against predicted plane extractions. Patch granularity varies with normal consistency and minimum region size. RQ and PQ denote Recognition Quality and Panoptic Quality, respectively.
<table><tr><td>Configuration</td><td> $2 \mathrm { m } { - } 2 ^ { \circ }$ </td><td> $3 \mathrm { m } { - } 3 ^ { \circ }$ </td><td> $5 \mathrm { m } { - } 5 ^ { \circ }$ </td><td> $\mathrm { T . e . ( m ) }$ </td><td> $\mathrm { R . e . } ( ^ { \circ } )$ </td></tr><tr><td>Mask only</td><td>62.0 / 66.0</td><td>87.0 / 82.0</td><td>95.0 / 94.0</td><td>1.39 / 1.50</td><td>0.81 / 0.81</td></tr><tr><td>Mask + edges</td><td>74.0 / 76.0</td><td>92.0 / 91.0</td><td>98.0 / 97.0</td><td>1.20 / 1.42</td><td>0.75 / 0.71</td></tr><tr><td>Mask + depth</td><td>60.0 / 71.0</td><td>80.0 / 89.0</td><td>92.0 / 97.0</td><td>1.66 / 1.49</td><td>0.95 / 0.74</td></tr><tr><td>All three cues</td><td>75.0 / 84.0</td><td>94.0 / 94.0</td><td>98.0 / 99.0</td><td>1.14 /1.37</td><td>0.68 / 0.78</td></tr></table>

Table A2. Coarse scoring components. Mask, edge, and depth cues in coarse search and particle filtering. Other stages are unchanged. Matched 100-query UAVD4L-LoDv2 subsets (in-Traj. / out-of-Traj.). Joint recalls (%) and median errors. Best/secondbest values are bold/underlined.
<table><tr><td>Configuration</td><td> $2 \mathrm { m } { - } 2 ^ { \circ }$ </td><td> $3 \mathrm { m } { - } 3 ^ { \circ }$ </td><td> $5 \mathrm { m } { - } 5 ^ { \circ }$ </td><td>T.e.(m)</td><td> $\mathbf { R . e . } ( ^ { \circ } )$ </td></tr><tr><td>Normal consistency</td><td></td><td></td><td>74.0 / 79.094.0 / 96.098.0 / 100.0</td><td>1.30 / 1.410.78 / 0.79</td><td></td></tr><tr><td>Depth continuity</td><td></td><td>31.0 / 64.072.0 / 92.0</td><td>93.0 / 97.0</td><td>2.36 / 1.671.25 / 0.77</td><td></td></tr><tr><td>Normals + depth</td><td></td><td>75.0 / 84.0 94.0 / 94.0</td><td>98.0 / 99.0</td><td>1.14 / 1.370.68 / 0.78</td><td></td></tr></table>

Table A3. Plane extraction criteria. Normal and depth consistency for predicted and rendered planes. Matched 100-query UAVD4L-LoDv2 subsets (in-Traj. / out-of-Traj.). Joint recalls (%) and median errors. Best/second-best values are bold/underlined.

PnP matching strategy. Table A4 evaluates our sequential alignment criteria during PnP optimization. Overlap voting accommodates fragmented image patches, while centroid consistency establishes more localized correspondences. Applying both matchers sequentially combines these complementary cues and gives the highest strict recall on both subsets. Overlap-only matching remains stronger at some looser out-of-Traj. thresholds.

<table><tr><td>Configuration</td><td> $2 \mathrm { m } { - } 2 ^ { \circ }$ </td><td> $3 \mathrm { m } { - } 3 ^ { \circ }$ </td><td> $5 \mathrm { m } { - } 5 ^ { \circ }$ </td><td> $\mathrm { T . e . ( m ) }$ </td><td> $\mathrm { R . e . } ( ^ { \circ } )$ </td></tr><tr><td>Overlap only</td><td>54.0 / 76.0</td><td>89.0 / 96.0</td><td>96.0 / 98.0</td><td>1.80 / 1.41</td><td>1.02 / 0.86</td></tr><tr><td>Centroid only</td><td>64.0 / 72.0</td><td>85.0 / 91.0</td><td>97.0 / 99.0</td><td>1.50 / 1.59</td><td>0.91 / 0.79</td></tr><tr><td>Overlap + centroid</td><td>75.0 / 84.0 94.0 / 94.0</td><td></td><td>98.0 / 99.0</td><td>1.14 /1.37</td><td>0.68 / 0.78</td></tr></table>

Table A4. PnP matching strategies. Overlap, centroid, and sequential correspondences. Matched 100-query UAVD4L-LoDv2 subsets (in-Traj. / out-of-Traj.). Joint recalls (%) and median errors. Best/second-best values are bold/underlined.

Outlier rejection. PnP-RANSAC [10] rejects inconsistent correspondences and refines the pose on the inliers. Updating correspondences as the pose changes allows subsequent iterations to use the improved image-to-map alignment.

Pose selection. Reprojection and centroid errors are obtained from PnP correspondences and are not computed for the particle-filter candidate. For each criterion, candidates without a measurement rank below those with valid measurements. Equal criterion values receive average ranks. Equal aggregate ranks retain the earlier candidate, with the particle-filter estimate considered first. These conventions also apply to the full-split pose-selection comparison in the main paper.

Building segmenters. Table A5 compares the impact of the upstream vision model, with visual examples shown in

![](images/a5262ac80d7d68da44b81dd39609fc551e7f3fe55e1a4fc4412a249dfc870bfc.jpg)  
Figure A3. Mask segmentation examples. Query images and predicted masks illustrate incomplete building coverage and non-building regions included by off-the-shelf segmentation. Such errors can weaken coarse alignment and the planar evidence available for refinement.

![](images/95c7391d552f7a50d7a9d372c2cee4c4ba5b113d3c6d25c47f835f46f788bcad.jpg)

$$
\mathrm { \Phi ^ { - } ~ C o a r s e ~ s e a r c h ~ \mathrm { \Phi ~ - \Phi ~ P a r t i c l e ~ f i l t e r ~ \hat { \Theta } ^ { + - } ~ P l a n a r C i t y L o c } }
$$

![](images/06cb200dadcd43de0f71c979df2fbdb36041e24bb10a913ee6ee7aaa5a11c943.jpg)

![](images/77cf7891e6b147169c0b325774a81e5b07bf4134c0ba5378e5c06ff4307b628a.jpg)  
Figure A4. Comprehensive prior noise evaluation. Joint recall and median errors under synthetic pose perturbations on UAVD4L LoDv2, using 300 queries per split and expanded search ranges. Native-prior points are separate references on the same queries.

Fig. A3. While the geometry prediction (MoGe-2) remains fixed, swapping the SAM3 [2] mask module for EoMT [19] severely degrades performance. Reliable building masks are therefore important for both coarse alignment and subsequent plane matching.

## F. Robustness Analysis

Sensitivity to prior pose noise. We summarize median pose error against injected prior noise in Fig. A4. We evaluate synthetic priors on 300 fixed queries per UAVD4L-LoDv2 split. Translation and angular components are uniformly perturbed within bounds of 10, 20, or 30 meters/degrees, with pitch and roll limited to ±5<sup>◦</sup>. Search ranges are expanded to accommodate the increased uncertainty. Native-prior results are shown separately.

![](images/81648167a7e75cb84462b61f19104773ccda25b8b6be4f9f71a7ff8af6938c6c.jpg)  
Figure A5. Visual examples of map completeness. Overlays of rendered LoD geometry for the intact map and representative remova conditions. Groups are selected by projected area, not physical building volume.

<table><tr><td>Configuration</td><td> $2 \mathrm { m } { - } 2 ^ { \circ }$ </td><td> $3 \mathrm { m } { - } 3 ^ { \circ }$ </td><td> $5 \mathrm { m } { - } 5 ^ { \circ }$ </td><td> $\mathrm { T . e . ( m ) }$ </td><td> $\mathbf { R . e . } ( ^ { \circ } )$ </td></tr><tr><td>SAM3</td><td>75.0 / 84.0</td><td>94.0 / 94.0</td><td>98.0 / 99.0</td><td>1.14 /1.37</td><td>0.68 / 0.78</td></tr><tr><td>EoMT</td><td>40.0 / 48.0</td><td>54.0 / 61.0</td><td>69.0 / 70.0</td><td>2.51 / 2.26</td><td>1.46 / 1.06</td></tr></table>

Table A5. Building segmenters. SAM 3 and EoMT with fixed MoGe-2 ViT-S geometry. Matched 100-query UAVD4L-LoDv2 subsets (in-Traj. / out-of-Traj.). Joint recalls (%) and median errors. Best/second-best values are bold/underlined.

<table><tr><td>Configuration</td><td>2m-2°</td><td> $3 \mathrm { m } { - } 3 ^ { \circ }$ </td><td> $5 \mathrm { m } { - } 5 ^ { \circ }$ </td><td>T.e. (m)</td><td> $\mathbf { R . e . } ( ^ { \circ } )$ </td></tr><tr><td>Intact map</td><td>75.0 / 84.0</td><td>94.0 / 94.0</td><td>98.0 / 99.0</td><td>1.14 /1.37</td><td>0.68 / 0.78</td></tr><tr><td>Remove smallest</td><td>79.0 / 82.0</td><td>94.0 / 98.0</td><td>98.0 / 99.0</td><td>1.18 / 1.41</td><td>0.68 / 0.73</td></tr><tr><td>Remove middle</td><td>71.0/ 77.0</td><td>89.0 / 95.0</td><td>94.0 / 99.0</td><td>1.41 / 1.46</td><td>0.72 / 0.81</td></tr><tr><td>Remove largest</td><td>52.0 / 69.0</td><td>70.0 / 84.0</td><td>84.0 / 95.0</td><td>1.87 / 1.47</td><td>1.02 / 0.76</td></tr></table>

Table A6. Map completeness. Matched 100-query UAVD4L-LoDv2 subsets (in-Traj. / out-of-Traj.). Groups of nearby mesh surfaces are removed by projected area. Queries without sufficient groups retain the intact map. Joint recalls (%) and median errors. Best/second-best values are bold/underlined.

Larger initial errors reduce localization accuracy. Planebased refinement nevertheless lowers median translation and rotation errors relative to particle filtering at every tested perturbation level. At the largest offsets, absolute recall remains low, highlighting the importance of a useful initialization even when 3D constraints improve the estimate.

Sensitivity to map completeness. We investigate the sensitivity of our pipeline to missing structures in the LoD model in Tab. A6. For each frame, we selectively remove a visible mesh-derived group of nearby surfaces before rendering the map prior. Removal size is defined by projected area at the initial pose, with each group approximating a visible structure rather than a labeled building instance. Removing small or medium groups has mixed effects, while removing the largest group lowers strict recall on both splits. This supports the importance of large visible structures as geometric anchors for localization. Figure A5 visualizes these scenarios.

## G. Limitations and Future Work

PlanarCityLoc depends on reliable coarse initialization. Poor masks or inaccurate GPS priors can lead to incorrect alignment that planar refinement cannot recover (Fig. A3). Localization also degrades with incomplete or poorly modeled LoD geometry (Tab. A6). Better upstream predictions and robust initialization could mitigate these limitations. Temporal filtering could disambiguate repetitive structures through motion cues. Efficient initialization and hardwareaware optimization remain necessary for onboard real-time operation.

## H. Ethical Considerations

Geographic bias and reliability. Our zero-shot evaluation is geographically constrained to specific urban environments, and performance may vary across different architectural styles. Lacking hard accuracy guarantees, the system requires rigorous fallbacks before deployment in safety-critical autonomous flights.

Privacy and data usage. Public datasets and textureless maps do not eliminate privacy risks. Query images and pretrained models may contain sensitive information, and downstream use must respect applicable data and model terms. Swiss-EPFL map data are provided by the Federal Office of Topography swisstopo.

Dual-use potential. Although intended for civilian inspection, aerial localization can enable surveillance and requires safeguards against misuse.