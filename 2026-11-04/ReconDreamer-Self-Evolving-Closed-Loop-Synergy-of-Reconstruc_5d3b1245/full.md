# ReconDreamer#: Self-Evolving Closed-Loop Synergy of Reconstruction and Generation for Dynamic Driving Scenes

Songling Liu<sup>∗</sup>, Guosheng Zhao<sup>∗</sup>, Xiaofeng Wang, Chaojun Ni, Guan Huang, Jiwen Lu, Xingang Wang<sup>†</sup> and Zheng Zhu<sup>†</sup>

Abstract—Dynamic driving scene reconstruction has made remarkable strides in recent years by leveraging generative priors, particularly in the quality of novel trajectory rendering. Nevertheless, a significant gap remains between the generated data and real-world sensor observations, severely hindering further improvements in reconstruction fidelity. In this paper, we present ReconDreamer#, a self-evolving framework that synergizes reconstruction and generation for dynamic driving scenes, which substantially bridges the gap between synthesized novel trajectory data and recorded real-world observations. Within this self-evolving closed-loop framework, the generative and reconstruction models mutually reinforce each other through iterative refinement, ultimately yielding a significant enhancement in the 3D representation capability of the reconstruction model. Additionally, we introduce a geometry-prior-based adaptive ground model to further enhance the rendering quality of the reconstruction model. To validate the effectiveness of the proposed method, we conduct comprehensive evaluations on Waymo, nuScenes, PandaSet, and EUVS benchmarks. Specifically on Waymo, at a 6-m lateral shift, ReconDreamer# improves NTA-IoU and NTL-IoU by 8.8% and 3.0%, respectively, and reduces FID by 15.1% relative to ReconDreamer++.

Index Terms—Driving scene reconstruction, 3D Gaussian Splatting, novel-view synthesis, video restoration.

## I. INTRODUCTION

R <sup>OBUST</sup> <sup>driving</sup> <sup>scene</sup> <sup>representation</sup> <sup>is</sup> <sup>of</sup> <sup>paramount</sup>importance for autonomous driving simulation systems. With advances in Neural Radiance Fields (NeRF) [1] and 3D Gaussian Splatting (3DGS) [2], [3], substantial progress has been achieved in reconstructing autonomous driving scenes along recorded trajectories. Surface-aware Gaussian representations further improve geometric fidelity by aligning primitives with reconstructed surfaces or planar structures [4], [5], [6], [7]. By incorporating generative priors [8], [9], recent methods [10], [11], [12] have significantly improved scene representation quality under novel trajectories, emerging as a new paradigm for dynamic driving scene reconstruction. Complementary view-augmentation approaches address extrapolation through inverse warping or view-prior guidance [13], [14], [15].

Central to this paradigm is the generative model’s ability to synthesize novel-trajectory data aligned with the recorded views. To train such a generative model, existing methods typically simulate degraded videos along the original trajectories, thereby forming training pairs with the recorded real data. For instance, our prior ReconDreamer series [10], [16] generates degraded-real data pairs by rendering original trajectories through deliberately under-optimized Gaussian Splatting models. FreeSim [17] simulates degradation through hand-crafted perturbation schemes, such as adding noise perturbations to Gaussian primitives and image blending. Nevertheless, these methods, while capable of generating degraded data, produce degradation patterns that are inconsistent with those encountered during actual novel-trajectory rendering. In practice, the degradation in novel-trajectory rendering primarily stems from artifacts induced by lateral viewpoint deviations in Gaussian Splatting models. This misalignment ultimately limits the 3D representation capacity of the reconstruction model.

To alleviate this misalignment, we present ReconDreamer#, a self-evolving framework that synergizes reconstruction and generation for dynamic driving scenes. Specifically, as illustrated in Fig. 1, our framework adopts a multi-round iterative self-evolution architecture. In the first iteration, we follow prior methods [10], [16] to construct misaligned degradedreal data pairs for training the generative model DriveRestorer, which is then incorporated into the reconstruction pipeline to enhance rendering quality. Subsequently, in the second iteration, the reconstruction model obtained from the first round is employed to render a novel-trajectory video with lateral viewpoint deviations. This video then serves as pseudo ground truth for training a reconstruction model [18] without DriveRestorer until convergence, after which the trained reconstruction model is applied to render the original trajectory. At this stage, degraded-real data pairs with aligned degradation patterns are obtained. Using such data pairs, we retrain DriveRestorer and subsequently embed it into the reconstruction pipeline. This iterative process is repeated, progressively improving both the generative and reconstruction models in a self-evolving manner. Beyond mitigating the degradation misalignment, we also introduce an Adaptive Ground Model (AGM) to further extend the representation capacity of the ground model in ReconDreamer++ [16]. Instead of relying solely on the hard geometric constraints imposed by LiDAR priors, our approach incorporates regularization-based soft planar constraints on the ground surface. To further validate the effectiveness of the proposed method, we conduct comprehensive evaluations on four widely-used benchmarks: Waymo [19], nuScenes [20], PandaSet [21], and EUVS [22]. Extensive experimental results demonstrate that our method significantly enhances the 3D representation capability of the reconstruction model.

![](images/a82c77856fc71943ff1008d0fe64b55a92cfd3e4698ac415d5c6824758070e91.jpg)  
Fig. 1. Overview of the self-evolving reconstruction–restoration loop. Starting from recorded video and an under-trained Gaussian scene, DriveRestorer improves novel-trajectory renderings used to train a new reconstruction. Rendering that reconstruction back at recorded poses yields new degraded–real pairs for the next restorer generation. The top rows compare renderings at 3-m and 6-m lane shifts across iterations.

The main contributions of this paper are summarized as follows:

• We propose ReconDreamer#, a self-evolving framework that synergizes reconstruction and generation for dynamic driving scenes, which substantially mitigates the pattern misalignment issue in generated data pairs.

• An Adaptive Ground Model (AGM) is introduced to enhance the representation capacity of the ground model by incorporating soft planar constraints.

• Extensive experiments on Waymo, nuScenes, PandaSet, and EUVS demonstrate the superior performance of ReconDreamer# along original and novel trajectories. On Waymo, ReconDreamer# improves NTA-IoU by up to 8.8% and NTL-IoU by up to 3.0%, while reducing FID by up to 15.1% compared with ReconDreamer++.

Extension of the conference versions. This manuscript extends our conference papers ReconDreamer [10] (CVPR 2025) and ReconDreamer++ [16] (ICCV 2025). It builds on their reconstruction backbone and online DriveRestorer pipeline, together with the scene decomposition, depth supervision, and NTDNet of ReconDreamer++. The journal version introduces two extensions: a self-evolving reconstruction– restoration framework that constructs new degraded–real pairs via auxiliary round-trip reconstruction and updates DriveRestorer across generations, and an Adaptive Ground Model that initializes ground Gaussians from a reconstructed surface and adapts their support, shape, and appearance under local soft planar constraints. We further provide controlled ablations of both extensions, multi-generation DriveRestorer comparisons, and analyses of ground-support coverage and local surface alignment, complementing the evaluation on Waymo, nuScenes, PandaSet, and EUVS. Tab. I summarizes the relationship among these versions.

## II. RELATED WORK

## A. Neural Reconstruction of Driving Scenes

Neural radiance fields represent scene appearance through a continuous volumetric function and enable novel-view synthesis from calibrated images [1]. Subsequent methods improve the treatment of large scenes, sampling, and computation. S-NeRF and UC-NeRF address street views and camera calibration [23], [24], while StreetSurf and deformable neural meshes investigate outdoor geometry [25], [26]. Mip-NeRF 360 handles unbounded environments, while Instant-NGP uses multiresolution feature grids to accelerate representation and optimization [27], [28]. Zip-NeRF combines grid-based representations with anti-aliasing, and NeO 360 investigates sparse-view synthesis of outdoor scenes [29], [30]. For driving scenes, Urban Radiance Fields incorporates sensor information into outdoor reconstruction and Block-NeRF partitions large environments into tractable components [31], [32]. These approaches demonstrate the importance of representation design and sensor alignment, but observations collected along one path still constrain some viewing directions much more strongly than others.

TABLE I  
COMPARISON OF THE CONFERENCE VERSIONS AND THE JOURNAL EXTENSION.
<table><tr><td rowspan="2">Method</td><td colspan="3">Base components</td><td rowspan="2">Ground support</td><td colspan="3">New mechanisms</td><td rowspan="2">New analyses Generation / geometry</td></tr><tr><td>Backbone /</td><td>Ground restoration decomposition NTDNet</td><td></td><td>Round-trip self-evolution initialization refinement</td><td>Surface</td><td>Soft planes /</td></tr><tr><td>ReconDreamer [10]</td><td>√</td><td></td><td>一</td><td>Backbone- dependent</td><td></td><td></td><td></td><td></td></tr><tr><td>ReconDreamer++ [16]</td><td>√</td><td>√</td><td>√</td><td>Fixed</td><td></td><td></td><td></td><td>一</td></tr><tr><td>ReconDreamer#</td><td>√</td><td>√</td><td>√</td><td>Adaptive</td><td>V</td><td>V</td><td>√</td><td></td></tr></table>

Dynamic environments require additional structure. Neural Scene Flow Fields and HyperNeRF model temporal changes through scene flow or higher-dimensional deformation, while K-Planes and NeRFPlayer explore factorized and decomposed dynamic radiance representations [33], [34], [35], [36]. ENeRF and HyperReel address efficient free-viewpoint video synthesis through geometry-guided or ray-conditioned sampling [37], [38]. Neural Scene Graphs and Panoptic Neural Fields explicitly associate scene content with objects, while MARS and UniSim study compositional scene modeling and neural sensor simulation in driving settings [39], [40], [41], [42]. HUGSIM builds a real-time closed-loop driving simulator from reconstructed Gaussian scenes, supporting ego and actor updates driven by control commands [43]. NeuRAD further develops neural rendering for autonomous-driving sensor configurations [44]. Other approaches incorporate sensor cues or decompose dynamic urban scenes [45], [46], [47]. ADGaussian predicts generalizable Gaussians from image and depth features, while DGGT reconstructs dynamic scenes and camera poses from unposed images [48], [49]. Such systems must model both appearance and the transformations that place agents in the world at each timestamp. A novel trajectory changes how these components occlude one another, even when their underlying motion is unchanged. Reliable simulation therefore requires coherent object placement and background reconstruction rather than independent frame-level plausibility.

3D Gaussian splatting offers an explicit, differentiable representation with efficient rasterization [2]. Mip-Splatting improves its behavior across changes in sampling scale through anti-aliasing [50]. Dynamic extensions model temporal changes with deformable or 4D Gaussians [3], [51]. GaussianPro guides densification through progressive geometric propagation, using patch matching to create Gaussians with accurate positions and orientations [52]. Other methods organize Gaussians for adaptive rendering and large scenes [53], [54], [55]. Street Gaussians separates static content and dynamic objects, enabling urban-scene rendering through object-aware Gaussian representations [18]. Periodic Vibration Gaussian, S3Gaussian, and OmniRe explore additional ways to represent dynamic urban content and scene decomposition [56], [57], [58], [59]. DrivingGaussian, AutoSplat, HO-Gaussian, and HUGS also study Gaussian representations for urban driving scenes [60], [61], [62], [63]. DrivingRecon investigates a feedforward reconstruction model for autonomous driving [64].

Their explicit primitives provide convenient interfaces for semantic grouping, geometric constraints, and support refinement. Our method uses this interface to assign a dedicated optimization policy to ground. It does not replace the Gaussian renderer or introduce an alternative dynamic-object model. Instead, it modifies the supervision loop and the way reliable road support is established and maintained.

## B. Generative Priors and Trajectory Extrapolation

Latent diffusion models provide learned appearance priors for image synthesis [8], [65]. Video diffusion extends these priors to temporal generation [9], [66], [67], [68], including highmotion synthesis [69]. Video Diffusion Models and Align Your Latents develop temporally coherent video generation with diffusion [70], [71]. Latte and CogVideoX explore transformerbased video diffusion [72], [73], while CogVideo, VideoGPT, and VideoPoet use transformer-based generative models [74], [75], [76]. Driving world models, including the DriveDreamer series, adapt generation to the structure of road environments [77], [78], [79]. GAIA-1, Vista, and Drive-WM also investigate driving video generation and world modeling [80], [81], [82]. WorldDreamer and Pandora study broader videobased world models [83], [84]. DriveArena and MagicDrive3D extend driving generation toward simulation and controllable 3D scene synthesis [85], [86]. MagicDrive, Panacea, and DriveScape explore controllable driving video generation [87], [88], [89]. GenAD and DrivingDiffusion study predictive or layout-guided generation for driving scenarios [90], [91]. The resulting videos can be used directly as visual simulations or as additional observations for a reconstructive model. These roles have different requirements: an attractive video is not necessarily consistent with the geometry, agent poses, or road layout of a particular reconstructed scene.

DriveDreamer4D investigates generated observations as data for driving-scene representation, and FreeVS studies generative view synthesis along free driving trajectories [11], [92]. DreamDrive generates visual references with video diffusion and lifts them into a hybrid Gaussian representation for 3Dconsistent dynamic driving-scene rendering [93]. StreetCrafter and methods using diffusion priors with Gaussian representations further illustrate the interaction between conditional generation and view extrapolation [12], [94], [95]. DriveX and RGE-GS further explore generative guidance for extrapolated driving views [96], [97]. DIFIX3D+ repairs novelview renderings with a single-step diffusion model and distills the corrected views back into the scene representation [98]. Generative supervision can fill appearance gaps, but errors can become persistent when they are repeatedly fitted by the scene model. The quality of supervision must consequently be assessed together with the representation into which it is incorporated.

Recent work explores complementary conditioning and fusion strategies. LidarPainter uses one-step diffusion with sparse LiDAR and corrupted renderings [99]. DriveFix models cross-camera and temporal consistency, while SPVC combines structured conditions with panoptic video repair [100], [101]. SymDrive uses symmetric views and autoregressive restoration, and DriveExplorer progressively restores extrapolated renderings [102], [103]. GenRe instead learns a generalizable enhancer that distills diffusion priors into pretrained Gaussian representations [104].

FaithFusion uses pixel-wise information gain to guide correction and distillation, while PRG-Fusion routes regions among preservation, repair, and generation [105], [106]. StreetNVS jointly conditions video diffusion on LiDAR, reference images, and camera poses, and GeoFlow uses geometryaligned priors for efficient driving video generation [107], [108]. VDEGaussian combines video diffusion with timestamp refinement and uncertainty-aware distillation for dynamic urban scenes [109]. A complementary application is ReconDreamer-RL, which incorporates diffusion-enhanced reconstruction into closed-loop simulation and reinforcement learning [110].

ReconDreamer introduces DriveRestorer, a restoration model trained from imperfect renders and recorded video targets, and uses it for online reconstruction improvement [10]. ReconDreamer++ addresses the gap between generated and recorded observations through scene decomposition, novelview depth supervision, and NTDNet [16]. Our extension focuses on the distribution used to train the restorer itself. Instead of treating that distribution as fixed, the current scene and restorer construct the inputs of the next training round. Real recorded images remain the restoration targets. The auxiliary reconstruction provides a way to exploit novel-view failures without treating generated videos as verified labels for those failures.

## C. Surface Reconstruction and Ground Constraints

Road geometry benefits from priors that are stronger than those appropriate for arbitrary background objects. LiDARbased ground segmentation identifies measurements that can support such priors [111]. Surface reconstruction then supplies connectivity and a more regular spatial discretization. We use Field-Aligned Online Surface Reconstruction as the initialization operator [112], followed by Butterfly-style subdivision to refine the mesh sampling [113]. These established geometric tools are components of our initialization, rather than new surface-reconstruction algorithms proposed by this work. RoGs places 2D Gaussian surfels on a regular meshgrid and initializes their heights and orientations from vehicle poses [7]. ExtraGS combines road-surface Gaussians with an SDF and uncertainty-guided generative supervision, while ViSE uses a dimension-reduced road SDF within a vision-only extrapolation pipeline [114], [115].

The distinction between surface construction and radiance representation is central. A reconstructed mesh can guide Gaussian locations without becoming the final rendering primitive. After initialization, Gaussian scales, orientations, opacity, and spherical-harmonic coefficients still determine the rendered image. Local plane constraints therefore operate at several levels: they restrict support movement, encourage thin covariance along the surface normal, and discourage noveltrajectory deformations from moving road content off the measured surface. A single position constraint does not by itself control all of these effects.

ReconDreamer++ already separates ground and fixes the base ground positions and primitive count [16]. AGM extends this treatment with explicit support adaptation and pruning protection. Its local planes accommodate road slope and changing orientation without assuming that the entire road lies on one horizontal plane. At the same time, the formulation retains the limits of the sensor data: incorrect segmentation, curb transitions, or missing measurements can still produce poor local neighborhoods. Our contribution concerns the coordinated use of reliable support, local geometry, and adaptive Gaussian refinement under generative supervision.

## III. METHOD

## A. Preliminary

ReconDreamer [10] proposes the incremental integration of world model knowledge, significantly enhancing the model’s performance for large maneuvers. The training process of ReconDreamer is divided into two stages. In the first stage, a partially trained reconstruction model [18] is utilized to render videos along the original trajectory. This process naturally introduces ghosting artifacts or blurriness, a direct consequence of the model’s underfitting. These degraded outputs are then paired with recorded sensor data to form training pairs, which are used to optimize the video restoration model, referred to as DriveRestorer [10]. In the second stage, degraded videos of novel trajectories are first rendered, and the finetuned DriveRestorer is subsequently employed to refine these rendering results, enhancing both their quality and spatial consistency. These restored outputs, combined with the original video data, are then integrated into the training pipeline to further enhance the performance and robustness of the reconstruction model. This two-stage approach ensures that ReconDreamer achieves superior performance across the novel trajectories. Additionally, ReconDreamer++ adheres to the two-stage paradigm while further optimizing the representation of dynamic driving scenes, thereby improving reconstruction quality.

## B. Overview of ReconDreamer#

As illustrated in Fig. 2, ReconDreamer# introduces a self evolving architecture based on multi-stage iterative optimization, which substantially reduces the mismatch between the degraded inputs seen during training and those encountered at inference time. Furthermore, we introduce an Adaptive Ground Model (AGM) to enhance the representational capacity for dynamic driving scenes. These two components are detailed in the following sections.

![](images/b4d5eeee068b91198dc48b07bffc076a6a93351b32874a6b8bdc6c0d0ed9e31c.jpg)  
Fig. 2. Trajectory round-trip construction for Self-Evolving DriveRestorer. Novel-trajectory rendering and restoration feed an auxiliary reconstruction, which returns to recorded poses to construct inputs paired with real video targets.

## C. Self-Evolving Framework

The proposed self-evolving framework consists of multiple rounds of iteration. Each round comprises three stages: pair construction, restoration model training, and reconstruction model updating. By alternating between updating the restoration model and the reconstruction model, the framework progressively aligns the degradation distribution faced by the restorer with the degradation patterns encountered in practical deployment, thereby expanding its failure coverage over successive rounds.

The first round. Following ReconDreamer [10] and Recon-Dreamer++ [16], we construct degraded training pairs from an under-converged Gaussian Splatting model $\mathcal { G ^ { \prime } } _ { 1 }$ [18]. The resulting renders along the original trajectories contain reconstruction artifacts and thus serve as degraded inputs for restoration training. These videos, together with the corresponding real observations, form the initial round-trip pair collection $\mathcal { D } _ { 1 }$ which is then used to train the restoration model DriveRestorer $\mathcal { R } _ { 1 }$ . After training, DriveRestorer is incorporated into the ReconDreamer pipeline to provide restored observations for novel trajectories and to participate in the optimization of the main reconstruction model $\mathcal { G } _ { 1 }$

The second round. The reconstruction model $\mathcal { G } _ { 1 }$ trained in the first round is used to render novel-trajectory videos, which serve as supervision to train a Gaussian Splatting model $\mathcal { G } ^ { \prime } { } _ { 2 }$ without a generative model. In contrast to the first stage, where the Gaussian Splatting model is deliberately left underconverged, this model $\mathcal { G } ^ { \prime } { } _ { 2 }$ is trained to convergence. Once converged, it is used to render original trajectory videos. Since this Gaussian Splatting model is trained on novel trajectories, rendering the original trajectories exposes it to a viewpoint shift, thereby producing rendering degradations that more closely resemble those encountered in practical deployment. The resulting low-quality rendered videos, together with the original captured data, constitute the second-round pair set $\mathcal { D } _ { 2 }$ Together with the first-round pair set $\mathcal { D } _ { 1 }$ , they form the training set $\{ \mathcal { D } _ { 1 } , \mathcal { D } _ { 2 } \}$ for the video restoration model $\mathcal { R } _ { 2 }$ at this stage, which is then incorporated into the reconstruction pipeline to optimize the reconstruction model $\mathcal { G } _ { 2 }$ of this round.

More rounds. On the basis of the above procedure, the framework can be extended to multiple rounds. In each round k, the reconstruction model $\mathcal { G } _ { k - 1 }$ from the previous round is used to render novel trajectory videos, which can be regarded as pseudo ground truth for the novel trajectories. These renderings are then used to train an auxiliary Gaussian splatting model ${ \mathcal { G } ^ { \prime } } _ { k } .$ . Since $\mathcal { G ^ { \prime } } _ { k }$ is trained on novel trajectories, rendering along the original trajectories exposes it to a viewpoint shift, thereby producing trajectory-shifted observations that constitute the pair set $\mathcal { D } _ { k }$ for round k. The accumulated pairs $\cup _ { j = 1 } ^ { k } { \mathcal { D } } _ { j }$ are then used to train the restoration model $\mathcal { R } _ { k }$ for the current round. Once $\mathcal { R } _ { k }$ is trained, it is incorporated into the reconstruction pipeline to optimize the reconstruction model $\mathcal { G } _ { k }$ of this round. This process keeps the degradation distribution modeled by the restorer synchronized with the current optimization state of the reconstruction model, thereby achieving continuous improvement in restoration performance and reconstruction quality over successive alternating updates. Furthermore, as the iteration progresses, we progressively increase the number of novel trajectories rendered by $\mathcal { G } _ { k - 1 }$ thereby expanding the training data available for $\mathcal { R } _ { k }$ and fostering mutual reinforcement between the generative and reconstruction models within the self-evolving framework.

## D. Adaptive Ground Model

To further enhance the 3D representation capability of the dynamic driving-scene reconstruction model, we introduce an Adaptive Ground Model (AGM). Unlike the ground model introduced in ReconDreamer++ [16], which relies strictly on the acquired ground point cloud and keeps the number and location parameters of ground Gaussians fixed during subsequent optimization, AGM imposes geometric constraints on the ground surface and optimizes all parameters of the ground model during training.

The overall procedure of AGM is summarized in Algo. 1. Specifically, let $P _ { g }$ be the segmented ground LiDAR points. We first reconstruct a surface from $P _ { g }$ and then downsample it to the target size [112], [113]. The resulting vertices of ground become Gaussian means to initialize the ground model ${ \mathcal { G } } _ { g } .$ Each vertex receives appearance, opacity, scale, and rotation parameters through inverse-distance interpolation from nearby members of $\mathcal { G } _ { g } ;$ quaternion results are normalized.

After initialization, the ground component is optimized under a geometric regularizer. Writing $r _ { i j }$ for a rotation axis, $q _ { i } =$ arg min<sub>j</sub> $s _ { i j }$ for the thinnest axis, and $b _ { i } \quad =$ arg $\operatorname* { m a x } _ { j } \vert n _ { i } ^ { \intercal } r _ { i j } \vert$ for the most normal-aligned axis, the ground regularizer is

$$
\begin{array} { c } { \displaystyle \mathcal { L } _ { g } = \frac { \lambda _ { p } } { N _ { g } } \sum _ { i } \big | n _ { i } ^ { \top } ( \widehat { \mu } _ { i } ( C ) - p _ { i } ) \big | } \\ { \displaystyle + \frac { \lambda _ { o } } { N _ { g } } \sum _ { i } ( 1 - \big | n _ { i } ^ { \top } r _ { i q _ { i } } \big | ) + \frac { \lambda _ { f } } { N _ { g } } \sum _ { i } s _ { i b _ { i } } . } \end{array}\tag{1}
$$

Here $\widehat { \mu } _ { i } ( C )$ is the rendered center under the enabled novelview deformation. A geometric projection maintains base support, whereas $\mathcal { L } _ { g }$ constrains rendered centers and Gaussian shape; the two operations are distinct.

For support refinement, AGM retains the backbone’s valid image-gradient and size criteria and applies different operations to ground primitives depending on their scale: large primitives are split, whereas small ones are duplicated. For large primitives, a selected parent is split by placing a child at

$$
\mu _ { i , l } ^ { \mathrm { c h i l d } } = \mu _ { i } + ( I - n _ { i } n _ { i } ^ { \top } ) u _ { i , l } .\tag{2}
$$

where $u _ { i , l }$ is an offset sampled from the parent ellipsoid. The projection removes the normal component of this offset, so that the child remains on the same local plane as its parent. Split children inherit the backbone’s attributes and scale shrinkage. For small primitives, AGM applies duplication, following the backbone’s ordinary duplication rule without additional geometric modification. Both operations alter the locations of ground primitives, so the updated means are no longer guaranteed to lie on the measured surface. We therefore introduce an additional plane-maintenance operation. For each current ground center $\mu _ { i } .$ , the centroid $p _ { i }$ and least-variance normal $n _ { i }$ are estimated from a local neighborhood of $P _ { g }$ [111]. With $\| n _ { i } \| = 1$ , an enabled plane-maintenance operation projects the base mean by

$$
\mu _ { i }  \mu _ { i } - n _ { i } n _ { i } ^ { \top } ( \mu _ { i } - p _ { i } ) .\tag{3}
$$

Algorithm 1 Adaptive Ground Model   
Require: Ground LiDAR $P _ { g } ;$ initial Gaussians $\mathcal { G } _ { g } ^ { 0 } ;$ scene-training observa  
tions O   
Ensure: Ground representation $\mathcal { G } _ { g }$   
1: Reconstruct a surface from $P _ { g } ^ { ^ { \scriptstyle * } }$ and downsample it to the target size   
2: Initialize ground means from the resulting surface vertices   
3: Interpolate appearance, opacity, scale, and rotation from $\mathcal { G } _ { g } ^ { 0 }$ by inverse   
distance; normalize quaternions   
4: Initialize the optimizer over all ground parameters, including means   
5: for each scene-optimization iteration t do   
6: Sample an observation with camera $C$ from $\scriptscriptstyle \mathcal { O }$   
7: Estimate local centroids $p _ { i }$ and unit normals $n _ { i }$ from $P _ { g }$ at current   
means   
8: Compute rendered centers $\widehat { \mu } _ { i } ( C )$ and axis indices $q _ { i } , b _ { i } ;$ evaluate $\mathcal { L } _ { g }$   
by Eq. (1)   
9: Optimize all ground parameters with $\mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { g }$   
10: Accumulate valid image-gradient and screen-size statistics   
11: if a backbone support-refinement step is due then   
12: Select large primitives to split and small primitives to duplicate   
using the backbone’s gradient and size criteria   
13: Split large primitives by $\operatorname { E q . } \left( 2 \right) ;$ inherit attributes and scale shrink   
age   
14: Duplicate small primitives using the backbone’s ordinary rule   
15: Synchronize ground attributes, optimizer state, and refinement   
statistics   
16: end if   
17: if plane maintenance is enabled at iteration t then   
18: Re-estimate $( p _ { i } , n _ { i } )$ from $P _ { g }$ at current ground means   
19: Project ground means onto the local planes by Eq. (3)   
20: end if   
21: end for

## IV. EXPERIMENTS

We organize evaluation around four complementary questions: whether the final scene preserves recorded-view fidelity, whether it remains coherent under trajectory extrapolation, whether AGM and self-evolution contribute independently, and how restoration behavior changes across generations. The following subsections first introduce the experimental setup, then report the main results, and finally present ablations and an efficiency analysis.

## A. Experimental Setup

Datasets. We follow ReconDreamer and ReconDreamer++ in dataset selection and conduct experiments on four datasets—Waymo [19], nuScenes [20], PandaSet [21], [42], [44], and EUVS [22]—using the same scenes as prior work to ensure fair comparison. Evaluations are conducted on the original trajectory and lateral shifts of 3 meters and 6 meters for Waymo and nuScenes; on the original trajectory, lateral shifts of 2 meters and 3 meters, and a vertical shift of 1 meter for PandaSet; and directly on the training and test sets for EUVS.

Metrics. On Waymo and nuScenes, recorded-view image quality is quantified by PSNR, SSIM [116], and LPIPS [117], all of which require aligned reference images. For synthesized trajectories lacking such references, we retain Novel Trajectory Agent IoU (NTA-IoU), Novel Trajectory Lane IoU (NTL-IoU), and FID [118] from the prior protocol [10], [11]. NTA-IoU measures the consistency of detected agents against projected annotated boxes, whereas NTL-IoU compares lane detections with geometrically transferred lane references. FID compares image distributions and, as such, does not certify the correctness of individual road markings or objects. PandaSet follows prior work [42], [44] in reporting FID only, while EUVS, given captured images for the extrapolated views, directly reports PSNR, SSIM, and LPIPS.

TABLE II  
QUANTITATIVE COMPARISON ON WAYMO. NTA AND NTL DENOTE NTA-IOU AND NTL-IOU, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td colspan="3">Recorded trajectory</td><td colspan="3">Lateral shift: 3 m</td><td colspan="3">Lateral shift: 6 m</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NTA↑</td><td>NTL↑</td><td>FID↓</td><td>NTA↑</td><td>NTL↑</td><td>FID↓</td></tr><tr><td>Street Gaussians [18]</td><td>36.50</td><td>0.9570</td><td>0.1149</td><td>0.4980</td><td>53.19</td><td>130.75</td><td>0.3740</td><td>49.27</td><td>213.04</td></tr><tr><td>FreeVS [92]</td><td>26.59</td><td>0.8057</td><td>0.1325</td><td>0.5050</td><td>56.84</td><td>104.23</td><td>0.4650</td><td>55.37</td><td>121.44</td></tr><tr><td>DriveDreamer4D [11]</td><td>34.37</td><td>0.9447</td><td>0.1321</td><td>0.4570</td><td>53.30</td><td>113.45</td><td>0.1590</td><td>50.09</td><td>261.81</td></tr><tr><td>ReconDreamer [10]</td><td>34.31</td><td>0.9431</td><td>0.1519</td><td>0.5390</td><td>54.58</td><td>93.56</td><td>0.4670</td><td>52.58</td><td>149.19</td></tr><tr><td>ReconDreamer++ [16]</td><td>36.29</td><td>0.9573</td><td>0.1076</td><td>0.5720</td><td>57.06</td><td>72.02</td><td>0.4890</td><td>56.57</td><td>111.92</td></tr><tr><td>ReconDreamer#</td><td>36.89</td><td>0.9638</td><td>0.0953</td><td>0.5984</td><td>58.75</td><td>66.42</td><td>0.5320</td><td>58.29</td><td>94.99</td></tr></table>

![](images/63250f13619a969acc53b8b134ea7c1e3e4f2e24045645989d6b50613079598f.jpg)  
Fig. 3. Qualitative comparison on Waymo. Columns show FreeVS, Street Gaussians, ReconDreamer, ReconDreamer++, and ReconDreamer#; rows show the recorded-view reference and renderings at 0-m, 3-m, and 6-m lateral shifts. Orange boxes enlarge vehicle and roadside details. Ground truth is shown onl for the recorded view.

Baselines. On Waymo, we compare against Street Gaussians [18], FreeVS [92], DriveDreamer4D [11], Recon-Dreamer [10], and ReconDreamer++ [16]. On nuScenes and EUVS, comparisons are made with Street Gaussians, ReconDreamer, and ReconDreamer++. On PandaSet, we additionally include UniSim [42] and NeuRAD [44]. Notably, FreeVS’s LiDAR-conditioned output does not cover the full upper image; we therefore follow the published Waymo crop convention. For each comparison, we adopt the corresponding scene, pose, masking, and metric protocols.

Implementation and model selection. We retain the reconstruction backbone, object decomposition, NTDNet, and sensor-depth supervision. AGM uses a target surface edge length of 0.10 m, a voxel scale of 0.03 m, eight neighbors for attribute interpolation, and local LiDAR neighborhoods for plane estimation. These are implementation settings, not constants built into the method or algorithm. The refinement schedule in Algorithm 1 separates the sensor-supported adaptation period from later pseudo-observation updates. For the generation study, we compare selected DriveRestorer generations using the same scene-reconstruction and evaluation protocol.

## B. Main Result

Waymo. Tab. II preserves the complete original- and noveltrajectory comparison. The results show that ReconDreamer# improves all nine reported Waymo measures over Recon-Dreamer++. On recorded views, PSNR increases from 36.29 to 36.89 and LPIPS decreases from 0.1076 to 0.0953, so extrapolation gains do not come at the expense of recordedview quality. At a three-meter shift, FID falls from 72.02 to 66.42; at six meters, it falls from 111.92 to 94.99. The relative FID reduction grows from 7.8% to 15.1% as the shift widens. At six meters, NTA-IoU also rises from 0.489 to 0.532 and NTL-IoU from 56.57 to 58.29, indicating that the distribution-level gain accompanies improved agent and lane consistency. Fig. 3 shows the corresponding renderings and enlarged vehicle and road details at matched shifts.

Street Gaussians  
TABLE III  
QUANTITATIVE COMPARISON ON NUSCENES.
<table><tr><td rowspan="2">Method</td><td colspan="3">Recorded trajectory</td><td colspan="3">Lateral shift: 3 m</td><td colspan="3">Lateral shift: 6 m</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NTA↑</td><td>NTL↑</td><td>FID↓</td><td>NTA↑</td><td>NTL↑</td><td>FID↓</td></tr><tr><td>Street Gaussians [18]</td><td>34.91</td><td>0.9528</td><td>0.1185</td><td>0.2190</td><td>48.72</td><td>149.50</td><td>0.1320</td><td>46.57</td><td>247.73</td></tr><tr><td>ReconDreamer [10]</td><td>33.85</td><td>0.9497</td><td>0.1235</td><td>0.3250</td><td>49.72</td><td>125.43</td><td>0.2460</td><td>47.08</td><td>197.52</td></tr><tr><td>ReconDreamer++ [16]</td><td>34.69</td><td>0.9533</td><td>0.1161</td><td>0.3650</td><td>51.33</td><td>118.27</td><td>0.2680</td><td>49.26</td><td>187.51</td></tr><tr><td>ReconDreamer#</td><td>35.62</td><td>0.9592</td><td>0.1111</td><td>0.3759</td><td>51.47</td><td>105.41</td><td>0.2772</td><td>49.53</td><td>133.99</td></tr></table>

![](images/d8c31a96272182d3c149147fa41adabff07c0c4854ad3de5035a6ee2d8d14d41.jpg)  
Fig. 4. Qualitative comparison on nuScenes. Columns show Street Gaussians, ReconDreamer, ReconDreamer++, and ReconDreamer#; rows show the recordedview reference and renderings at 0-m, 3-m, and 6-m lateral shifts. Orange boxes enlarge vehicles and nearby scene details. Ground truth is shown only for the recorded view.

nuScenes. Tab. III evaluates the same mechanisms under another dataset’s cameras and traffic patterns. Relative to ReconDreamer++, ReconDreamer# raises recorded-view PSNR from 34.69 to 35.62 and reduces LPIPS from 0.1161 to 0.1111. At a six-meter shift, NTA-IoU increases from 0.268 to 0.2772, NTL-IoU from 49.26 to 49.53, and FID decreases from 187.51 to 133.99. FID also decreases at three meters, from 118.27 to 105.41. The larger six-meter reduction is consistent with the method targeting errors exposed away from the recorded cameras. Fig. 4 shows the shifted-view comparison for vehicles and road appearance.

PandaSet. Tab. IV includes both lateral and vertical pose changes. The vertical test changes the viewing angle over road and object surfaces rather than merely moving the camera across lanes, making it a complementary check of viewpoint extrapolation. ReconDreamer# reduces FID relative to Recon-Dreamer++ from 13.8 to 12.81 without a shift, from 61.9 to 52.73 and 71.7 to 63.8 for two- and three-meter lateral shifts, and from 62.4 to 53.65 for a one-meter vertical shift. Fig. 5 illustrates the qualitative improvements under both shift directions.

TABLE IV  
FID COMPARISON ON PANDASET UNDER RECORDED, LATERAL-SHIFT, AND VERTICAL-SHIFT TRAJECTORIES.
<table><tr><td>Method</td><td>No shift Lane 2 m</td><td>Lane 3 m Vert. 1 m</td></tr><tr><td>UniSim [42]</td><td>74.70</td><td>97.50</td></tr><tr><td>NeuRAD [44]</td><td>25.00 72.30</td><td>93.90 76.30</td></tr><tr><td>Street Gaussians [18]</td><td>14.60 66.30</td><td>80.70 78.40</td></tr><tr><td>ReconDreamer [10]</td><td>16.10 65.40</td><td>74.90 67.70</td></tr><tr><td>ReconDreamer++ [16]</td><td>13.80 61.90</td><td>71.70 62.40</td></tr><tr><td>ReconDreamer#</td><td>12.81 52.73</td><td>63.80 53.65</td></tr></table>

![](images/17a2c258f58351f5f9e48c342d0a4269616dfb51bb4016c093ffbfe9adfbac90.jpg)  
Fig. 5. Qualitative comparison on PandaSet under recorded, lateral-shift, and vertical-shift trajectories. Columns compare Street Gaussians, ReconDreamer, ReconDreamer++, and ReconDreamer#. Orange boxes highlight fine structures in the distant scene.

TABLE V  
STATIC-REGION COMPARISON ON EUVS LEVEL 1.
<table><tr><td>Method</td><td>PSNR↑ SSIM↑ LPIPS↓</td></tr><tr><td>Street Gaussians [18] ReconDreamer [10]</td><td>15.08 0.6043 0.3428</td></tr><tr><td>ReconDreamer++ [16]</td><td>20.14 0.6914 0.3286 20.52 0.7033 0.3057</td></tr><tr><td>ReconDreamer#</td><td>21.59 0.7240 0.2995</td></tr></table>

EUVS. Tab. V measures extrapolated views against real images after foreground masking, complementing the perceptionbased proxies used for synthesized trajectories. Recon-Dreamer# improves on ReconDreamer++ in all three staticregion metrics: PSNR rises from 20.52 to 21.59, SSIM from 0.7033 to 0.724, and LPIPS falls from 0.3057 to 0.2995. Because the test view was independently captured across the lane, these image metrics check alignment to an actual observation rather than only the plausibility of a generated trajectory. Fig. 6 contrasts the training and test viewpoints; the masking keeps the comparison focused on their shared static content.

## C. Module Ablation Studies

We isolate the two extensions introduced by Recon-Dreamer#, including the AGM and the self-evolving reconstruction–restoration framework.

Tab. VI implements a two-factor comparison. The AGMonly variant uses the same fixed restoration model as the base, and the self-evolution-only variant retains the original ground treatment. AGM alone improves recorded-view LPIPS from 0.1076 to 0.1015 and six-meter NTL-IoU from 56.57 to 57.4. Self-evolution alone reduces six-meter FID from 111.92 to 100.11, although its six-meter NTL-IoU falls to 55.95. This contrast matters: better generated appearance does not by itself preserve lane structure at the larger shift. Combining both yields the best value in every reported column, including 0.0953 LPIPS, 58.29 six-meter NTL-IoU, and 94.99 six-meter FID.

For a higher-is-better metric $q ,$ complementarity can be summarized by the difference of differences,

$$
\Delta _ { \mathrm { i n t } } = q _ { \mathrm { b o t h } } - q _ { \mathrm { S E } } - q _ { \mathrm { A G M } } + q _ { \mathrm { b a s e } } .\tag{4}
$$

This is an analysis statistic, not an optimization objective or a guarantee of synergy. For lower-is-better metrics, the sign convention is reversed before interpretation. On sixmeter NTL-IoU, $\Delta _ { \mathrm { i n t } } = 1 . 5 1$ , indicating that the combined gain exceeds the sum of the isolated gains for this measure. Fig. 7 shows the four variants at matched poses, with enlarged vehicle and road-boundary regions.

![](images/655ade8bc193cffb2d76289f9dd0559a01c725fd919ed5a2fbdd4e7d9005331d.jpg)  
Fig. 6. Qualitative comparison on EUVS Level 1. The upper row shows a training-trajectory view (Train Set), while the lower row shows an independently captured cross-lane test view (Test Set). Columns compare ground truth, Street Gaussians, ReconDreamer, ReconDreamer++, and ReconDreamer#. Dynamic foreground is masked in the rendered comparisons, and orange boxes enlarge facade signs

TABLE VI  
NEW-MODULE ABLATION ON WAYMO. SE DENOTES SELF-EVOLVING DRIVERESTORER. ALL VARIANTS RETAIN THE INHERITED DEPTH AND NTDNET COMPONENTS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">SE</td><td rowspan="2">AGM</td><td colspan="3">Recorded</td><td colspan="3">3 m shift</td><td colspan="3">6 m shift</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NTA↑</td><td>NTL↑</td><td>FID↓</td><td>NTA↑</td><td>NTL↑</td><td>FID↓</td></tr><tr><td>ReconDreamer++</td><td></td><td></td><td>36.29</td><td>0.9573</td><td>0.1076</td><td>0.5720</td><td>57.06</td><td>72.02</td><td>0.4890</td><td>56.57</td><td>111.92</td></tr><tr><td>+ AGM</td><td></td><td>√</td><td>36.72</td><td>0.9608</td><td>0.1015</td><td>0.5780</td><td>58.00</td><td>71.72</td><td>0.5070</td><td>57.40</td><td>108.10</td></tr><tr><td>+ Self-Evolution</td><td>√</td><td></td><td>36.43</td><td>0.9568</td><td>0.1072</td><td>0.5830</td><td>57.75</td><td>70.45</td><td>0.5120</td><td>55.95</td><td>100.11</td></tr><tr><td>ReconDreamer#</td><td>√</td><td>了</td><td>36.89</td><td>0.9638</td><td>0.0953</td><td>0.5984</td><td>58.75</td><td>66.42</td><td>0.5320</td><td>58.29</td><td>94.99</td></tr></table>

TABLE VII

MULTI-ROUND DRIVERESTORER COMPARISON ON WAYMO. $\mathcal { R } _ { 1 }$ IS THE INITIAL STATIC RESTORER; $\mathcal { R } _ { 2 } - \mathcal { R } _ { 4 }$ ARE SUCCESSIVE SELF-EVOLUTION GENERATIONS.
<table><tr><td rowspan="2">DriveRestorer</td><td rowspan="2">Recorded LPIPS↓</td><td colspan="3">3 m shift</td><td colspan="3">6 m shift</td></tr><tr><td>NTA↑</td><td>NTL↑</td><td>FID↓</td><td>NTA↑</td><td>NTL↑</td><td>FID↓</td></tr><tr><td> $\mathcal { R } _ { 1 }$ </td><td>0.1015</td><td>0.5780</td><td>58.00</td><td>71.72</td><td>0.5070</td><td>57.40</td><td>108.10</td></tr><tr><td> $\mathcal { R } _ { 2 }$ </td><td>0.0985</td><td>0.5880</td><td>58.34</td><td>68.96</td><td>0.5180</td><td>57.84</td><td>101.94</td></tr><tr><td> $\mathcal { R } _ { 3 }$ </td><td>0.0974</td><td>0.5910</td><td>58.49</td><td>68.17</td><td>0.5240</td><td>57.99</td><td>99.45</td></tr><tr><td> $\mathcal { R } _ { 4 }$ </td><td>0.0953</td><td>0.5984</td><td>58.75</td><td>66.42</td><td>0.5320</td><td>58.29</td><td>94.99</td></tr></table>

## D. Multi-Round DriveRestorer Analysis

In this section, we analyze in detail the substantial gains brought by the self-evolving framework to DriveRestorer. As shown in Tab. VII, $\mathcal { R } _ { 1 }$ is the initial static restorer and $\mathcal { R } _ { 2 } \mathrm { - } \mathcal { R } _ { 4 }$ correspond to successive self-evolution generations, consistent with the notation in Sec. III. To ensure a fair and directly comparable assessment, each restorer generation is evaluated by integrating it into the complete reconstruction pipeline and measuring the quality of the final reconstructed scene, so that the reported results reflect the effect of the restorer itself rather than that of any auxiliary component.

With AGM held fixed, each successive restorer generation improves all reported aggregate measures. From $\mathcal { R } _ { 1 }$ to $\mathcal { R } _ { 4 } ,$ recorded-view LPIPS decreases from 0.1015 to 0.0953; at a six-meter shift, NTA-IoU increases from 0.507 to 0.532,

NTL-IoU from 57.4 to 58.29, and FID decreases from 108.10 to 94.99. The three-meter FID likewise drops from 71.72 to 66.42, indicating that the later restorers improve both tested shift distances under the fixed ground policy. These aggregate trends support reconstruction-mediated pair mining, while not implying that every scene or future generation must improve monotonically.

## E. Ground Representation Analysis

We further examine how AGM changes the spatial distribution and local surface alignment of the ground representation.

Fig. 8 complements rendered images with a direct view of ground support. Within the plotted region, the groundprimitive count rises from 458,569 to 781,687, while the shared top-view layout shows fewer holes and more even road coverage. More primitives alone would not establish a better ground surface: the matched local profile also checks whether their centers remain near the LiDAR plane. We quantify this property by the sensor-plane residual $\begin{array} { r } { e _ { \perp } = N _ { g } ^ { - 1 } \sum _ { i } \left| n _ { i } ^ { \top } ( \mu _ { i } - 1 ) \right. } \end{array}$ $p _ { i } ) |$ |. The concentrated residual profile alongside the increased coverage shows that AGM expands support without a corresponding spread away from the sensor-aligned road surface.

GroundTruth LaneShift@0mLaneShift@3mLaneShift@6m

![](images/07d8c5ae5183bfe85590364d838c2166a941e92c668b8659b955387d44cb4dc2.jpg)  
ReconDreamer++

![](images/545bc3d70bf307914610e9e138e57ecbff72f00a19915a792164f72dadc5cbe6.jpg)  
+ AGM

![](images/38a354dd4099ed7aa7e0657fa50e698f5590352be3a9950083ad3a3ca82e7d27.jpg)  
+ Self-Evolution

![](images/4e61742aa51a9a8547b003277e51496b16d6a6ccef0afe9580e51cbd9a76358b.jpg)  
ReconDreamer#

Fig. 7. Qualitative module ablation on Waymo. Columns compare ReconDreamer++, +AGM, +Self-Evolution, and the complete ReconDreamer# model. Rows show the reference view and renderings at 0 m, 3 m, and 6 m lateral shifts. Orange boxes highlight vehicle details and road-boundary artifacts.  
![](images/a7cfcc38947c72783c7183726a859d4ee0911e6869e147c9aebb2896b6447447.jpg)

(c) Plane residual r<sub>i</sub> (m)  
![](images/dee5be33126e214fa60cb97cd802cabd96b5c6d9bb8dd85062b808b2703e161e.jpg)  
Fig. 8. Ground-support comparison on Waymo scene 065. (a) The original ReconDreamer++ ground uses fixed support. (b) Complete AGM produces denser and more uniform support. Both top views share the same bounds, point size, and height colors; the cyan line marks the cross-section. (c) Signed residual $r _ { i } = n _ { i } ^ { \top } ( \mu _ { i } - p _ { i } )$ to the local LiDAR plane. The mean absolute residual e<sub>⊥</sub> and P95 show that AGM expands road support while maintaining surface alignment.

## F. Efficiency, Limitations, and Discussion

Self-evolution introduces offline computation for novelvideo restoration, auxiliary reconstruction, round-trip rendering, and restorer updating. AGM adds surface reconstruction and neighborhood estimation, and support preservation can increase the final primitive count. These training costs are distinct from the rendering throughput of the optimized Gaussian scene. The auxiliary reconstruction is not queried during ordinary final-scene rasterization, while any online restoration still incurs the cost of the generative model.

The method inherits several limitations. Round-trip mining can only supervise consequences visible from the recorded trajectory, and it cannot independently verify unseen geometry. Local ground planes can be unreliable near abrupt elevation changes or erroneous segmentation. Preserving low-opacity support may retain redundant primitives, and a surface-based initialization can bridge regions that should remain disconnected if the input geometry is wrong. Generative restoration may produce a coherent but incorrect appearance, especially for details not supported by the observations. These limitations motivate retaining real-image evaluation and sensor-derived geometry throughout the comparison.

The combined design provides a division of responsibility rather than a universal correction guarantee. The restorer adapts to encountered degradation, the auxiliary model makes some of that degradation supervisable, and AGM constrains how road support changes. The controlled ablations show that neither extension alone matches the complete model across recorded and shifted views, while the generation study shows consistent aggregate gains through $\mathcal { R } _ { 4 }$ . The remaining failure modes concern regions not observed by the round-trip path and geometry not reliably constrained by LiDAR.

## V. CONCLUSION

We presented ReconDreamer#, which extends generative– reconstructive driving-scene modeling through Self-Evolving DriveRestorer and an Adaptive Ground Model. The selfevolution procedure restores novel-trajectory renders, fits an auxiliary scene, and returns to recorded poses to construct restoration inputs with authentic image targets. This changes the degradation distribution available to successive restoration generations while preserving a supervised target domain. AGM complements that process by constructing surface-based road support, restricting split displacements to local tangential directions, projecting support onto LiDAR planes, and removing opacity-only ground deletion. Its formulation distinguishes adaptive support changes from continuous geometry and appearance optimization.

The resulting framework makes restoration learning responsive to the reconstruction it supports and keeps ground adaptation tied to sensor evidence. Its evaluation spans original trajectories, synthesized lane shifts, vertical perturbations, and independently captured cross-lane views. Relative to Recon-Dreamer++, the complete model reduces Waymo FID from 111.92 to 94.99 at a 6-m shift while increasing NTA-IoU from 0.489 to 0.532; it also reduces nuScenes FID from 187.51 to 133.99 at the same shift. The module and generation ablations show complementary gains from AGM and successive restorer updates. Remaining challenges include failures invisible to the round-trip path, inaccurate local ground estimates, and balancing support preservation against computational cost.

## REFERENCES

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing scenes as neural radiance fields for view synthesis,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 405– 421.

[2] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, “3D Gaussian splatting for real-time radiance field rendering,” ACM Trans. Graph., vol. 42, no. 4, pp. 1–14, 2023, art. no. 139.

[3] G. Wu et al., “4D Gaussian splatting for real-time dynamic scene rendering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 20 310–20 320.

[4] A. Guedon and V. Lepetit, “SuGaR: Surface-aligned Gaussian splatting for efficient 3D mesh reconstruction and high-quality mesh rendering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 5354–5363.

[5] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2D Gaussian splatting for geometrically accurate radiance fields,” in Proc. ACM SIGGRAPH Conf. Papers, 2024, pp. 1–11, art. no. 32.

[6] D. Chen et al., “PGSR: Planar-based Gaussian splatting for efficient and high-fidelity surface reconstruction,” IEEE Trans. Vis. Comput. Graph., vol. 31, no. 9, pp. 6100–6111, 2025.

[7] Z. Feng, W. Wu, T. Deng, and H. Wang, “RoGs: Large scale road surface reconstruction with meshgrid Gaussian,” 2024, arXiv:2405.14342.

[8] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 10 674– 10 685.

[9] A. Blattmann et al., “Stable Video Diffusion: Scaling latent video diffusion models to large datasets,” 2023, arXiv:2311.15127.

[10] C. Ni et al., “ReconDreamer: Crafting world models for driving scene reconstruction via online restoration,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 1560–1569.

[11] G. Zhao et al., “DriveDreamer4D: World models are effective data machines for 4D driving scene representation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 12 015–12 026.

[12] Y. Yan et al., “StreetCrafter: Street view synthesis with controllable video diffusion models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 822–832.

[13] J. Zhou et al., “FlexDrive: Toward trajectory flexibility in driving scene Gaussian splatting reconstruction and rendering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 1549–1558.

[14] J. Li, F. Wang, X. Qu, C. Wu, L. Liu, and T. Liu, “EVPGS: Enhanced view prior guidance for splatting-based extrapolated view synthesis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 16 398–16 407.

[15] H. Han, K. Zhou, X. Long, Y. Wang, and C. Xiao, “GGS: Generalizable Gaussian splatting for lane switching in autonomous driving,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 3, 2025, pp. 3329–3337.

[16] G. Zhao et al., “ReconDreamer++: Harmonizing generative and reconstructive models for driving scene representation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025, pp. 26 718–26 728.

[17] L. Fan, H. Zhang, Q. Wang, H. Li, and Z. Zhang, “FreeSim: Toward free-viewpoint camera simulation in driving scenes,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 12 004– 12 014.

[18] Y. Yan et al., “Street Gaussians: Modeling dynamic urban scenes with Gaussian splatting,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 156– 173.

[19] P. Sun et al., “Scalability in perception for autonomous driving: Waymo open dataset,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 2443–2451.

[20] H. Caesar et al., “nuScenes: A multimodal dataset for autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 11 618–11 628.

[21] P. Xiao et al., “PandaSet: Advanced sensor suite dataset for autonomous driving,” in Proc. IEEE Intell. Transp. Syst. Conf., 2021, pp. 3095– 3101.

[22] X. Han et al., “Extrapolated urban view synthesis benchmark,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025, pp. 28 718–28 728.

[23] Z. Xie, J. Zhang, W. Li, F. Zhang, and L. Zhang, “S-NeRF: Neural radiance fields for street views,” in Proc. Int. Conf. Learn. Represent., 2023.

[24] K. Cheng et al., “UC-NeRF: Neural radiance field for under-calibrated multi-view cameras in autonomous driving,” in Proc. Int. Conf. Learn. Represent., 2024.

[25] J. Guo et al., “StreetSurf: Extending multi-view implicit surface reconstruction to street views,” 2023, arXiv:2306.04988.

[26] F. Lu, Y. Xu, G. Chen, H. Li, K.-Y. Lin, and C. Jiang, “Urban radiance field representation with deformable neural mesh primitives,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2023, pp. 465–476.

[27] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 5460– 5469.

[28] T. Mueller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics primitives with a multiresolution hash encoding,” ACM Trans. Graph., vol. 41, no. 4, pp. 1–15, 2022, art. no. 102.

[29] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Zip-NeRF: Anti-aliased grid-based neural radiance fields,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2023, pp. 19 640–19 648.

[30] M. Z. Irshad et al., “NeO 360: Neural fields for sparse view synthesis of outdoor scenes,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2023, pp. 9153–9164.

[31] K. Rematas et al., “Urban radiance fields,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 12 922–12 932.

[32] M. Tancik et al., “Block-NeRF: Scalable large scene neural view synthesis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 8238–8248.

[33] Z. Li, S. Niklaus, N. Snavely, and O. Wang, “Neural scene flow fields for space-time view synthesis of dynamic scenes,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 6494–6504.

[34] K. Park et al., “HyperNeRF: A higher-dimensional representation for topologically varying neural radiance fields,” ACM Trans. Graph., vol. 40, no. 6, pp. 1–12, 2021, art. no. 238.

[35] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, “K-Planes: Explicit radiance fields in space, time, and appearance,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 12 479–12 488.

[36] L. Song et al., “NeRFPlayer: A streamable dynamic scene representation with decomposed neural radiance fields,” IEEE Trans. Vis. Comput. Graph., vol. 29, no. 5, pp. 2732–2742, 2023.

[37] H. Lin et al., “Efficient neural radiance fields for interactive freeviewpoint video,” in Proc. SIGGRAPH Asia Conf. Papers, 2022, pp. 1–9, art. no. 39.

[38] B. Attal et al., “HyperReel: High-fidelity 6-DoF video with rayconditioned sampling,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 16 610–16 620.

[39] J. Ost, F. Mannan, N. Thuerey, J. Knodt, and F. Heide, “Neural scene graphs for dynamic scenes,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 2855–2864.

[40] A. Kundu et al., “Panoptic neural fields: A semantic object-aware neural scene representation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 12 861–12 871.

[41] Z. Wu et al., “MARS: An instance-aware, modular and realistic simulator for autonomous driving,” in Proc. CAAI Int. Conf. Artif. Intell., 2024, pp. 3–15.

[42] Z. Yang et al., “UniSim: A neural closed-loop sensor simulator,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 1389– 1399.

[43] H. Zhou et al., “HUGSIM: A real-time, photo-realistic and closed-loop simulator for autonomous driving,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 4, pp. 4673–4691, 2026.

[44] A. Tonderski, C. Lindstroem, G. Hess, W. Ljungbergh, L. Svensson, and C. Petersson, “NeuRAD: Neural rendering for autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 14 895–14 904.

[45] S. Sun, B. Zhuang, Z. Jiang, B. Liu, X. Xie, and M. Chandraker, “LidaRF: Delving into LiDAR for neural radiance field on street scenes,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 19 563–19 572.

[46] H. Turki, J. Y. Zhang, F. Ferroni, and D. Ramanan, “SUDS: Scalable urban dynamic scenes,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 12 375–12 385.

[47] J. Yang et al., “EmerNeRF: Emergent spatial-temporal scene decomposition via self-supervision,” in Proc. Int. Conf. Learn. Represent., 2024.

[48] Q. Song, H. Lin, C. Li, S. Peng, and R. Huang, “ADGaussian: Generalizable Gaussian splatting for autonomous driving via multimodal joint learning,” in Proc. IEEE Int. Conf. Robot. Autom., 2026, pp. 519–526.

[49] X. Chen et al., “DGGT: Feedforward 4D reconstruction of dynamic driving scenes using unposed images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2026, pp. 1265–1276.

[50] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, “Mip-Splatting: Alias-free 3D Gaussian splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 19 447–19 456.

[51] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3D Gaussians for high-fidelity monocular dynamic scene reconstruction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 20 331–20 341.

[52] K. Cheng et al., “GaussianPro: 3D Gaussian splatting with progressive propagation,” in Proc. Int. Conf. Mach. Learn., vol. 235, 2024, pp. 8123–8140.

[53] T. Lu et al., “Scaffold-GS: Structured 3D Gaussians for view-adaptive rendering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 20 654–20 664.

[54] J. Lin et al., “VastGaussian: Vast 3D Gaussians for large scene reconstruction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 5166–5175.

[55] X. Du, Y. Wang, and X. Yu, “MVGS: Multi-view regulated Gaussian splatting for novel view synthesis,” in Proc. Eur. Conf. Comput. Vis., 2026, pp. 564–581.

[56] Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang, “Periodic Vibration Gaussian: Dynamic urban scene reconstruction and real-time rendering,” Int. J. Comput. Vis., vol. 134, no. 3, 2026, art. no. 83.

[57] N. Huang et al., “S3Gaussian: Self-supervised street Gaussians for autonomous driving,” in Proc. IEEE Int. Conf. Robot. Autom., 2026, pp. 10 283–10 290.

[58] Z. Chen et al., “OmniRe: Omni urban scene reconstruction,” in Proc. Int. Conf. Learn. Represent., 2025.

[59] T. Fischer, J. Kulhanek, S. Rota Bulo, L. Porzi, M. Pollefeys, and\` P. Kontschieder, “Dynamic 3D Gaussian fields for urban areas,” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 80 466–80 494.

[60] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, “DrivingGaussian: Composite Gaussian splatting for surrounding dynamic autonomous driving scenes,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 21 634–21 643.

[61] M. Khan et al., “AutoSplat: Constrained Gaussian splatting for autonomous driving scene reconstruction,” in Proc. IEEE Int. Conf. Robot. Autom., 2025, pp. 8315–8321.

[62] Z. Li, Y. Zhang, C. Wu, J. Zhu, and L. Zhang, “HO-Gaussian: Hybrid optimization of 3D Gaussian splatting for urban scenes,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 19–36.

[63] H. Zhou et al., “HUGS: Holistic urban 3D scene understanding via Gaussian splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 21 336–21 345.

[64] H. Lu et al., “DrivingRecon: Large 4D Gaussian reconstruction model for autonomous driving,” in Adv. Neural Inf. Process. Syst., vol. 38, 2025, pp. 186 133–186 152.

[65] D. Podell et al., “SDXL: Improving latent diffusion models for highresolution image synthesis,” in Proc. Int. Conf. Learn. Represent., 2024.

[66] R. Girdhar et al., “Factorizing text-to-video generation by explicit image conditioning,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 205– 224.

[67] J. Ho et al., “Imagen Video: High definition video generation with diffusion models,” 2022, arXiv:2210.02303.

[68] A. Gupta et al., “Photorealistic video generation with diffusion models,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 393–411.

[69] Y. Zeng et al., “Make pixels dance: High-dynamic video generation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 8850–8860.

[70] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” in Adv. Neural Inf. Process. Syst., vol. 35, 2022, pp. 8633–8646.

[71] A. Blattmann et al., “Align your latents: High-resolution video synthesis with latent diffusion models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 22 563–22 575.

[72] X. Ma et al., “Latte: Latent diffusion transformer for video generation,” Trans. Mach. Learn. Res., 2025.

[73] Z. Yang et al., “CogVideoX: Text-to-video diffusion models with an expert transformer,” in Proc. Int. Conf. Learn. Represent., 2025.

[74] W. Hong, M. Ding, W. Zheng, X. Liu, and J. Tang, “CogVideo: Largescale pretraining for text-to-video generation via transformers,” in Proc. Int. Conf. Learn. Represent., 2023.

[75] W. Yan, Y. Zhang, P. Abbeel, and A. Srinivas, “VideoGPT: Video generation using VQ-VAE and transformers,” 2021, arXiv:2104.10157.

[76] D. Kondratyuk et al., “VideoPoet: A large language model for zero-shot video generation,” in Proc. Int. Conf. Mach. Learn., vol. 235, 2024, pp. 25 105–25 124.

[77] X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu, “Drive-Dreamer: Towards real-world-driven world models for autonomous driving,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 55–72.

[78] G. Zhao et al., “DriveDreamer-2: LLM-enhanced world models for diverse driving video generation,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 10, 2025, pp. 10 412–10 420.

[79] ——, “UniDriveDreamer: A single-stage multimodal world model for autonomous driving,” in Proc. Eur. Conf. Comput. Vis., 2026, pp. 360– 377.

[80] A. Hu et al., “GAIA-1: A generative world model for autonomous driving,” 2023, arXiv:2309.17080.

[81] S. Gao et al., “Vista: A generalizable driving world model with high fidelity and versatile controllability,” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 91 560–91 596.

[82] Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang, “Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 14 749–14 759.

[83] X. Wang, Z. Zhu, G. Huang, B. Wang, X. Chen, and J. Lu, “World-Dreamer: Towards general world models for video generation via predicting masked tokens,” 2024, arXiv:2401.09985.

[84] J. Xiang et al., “Pandora: Towards general world model with natural language actions and video states,” 2024, arXiv:2406.09455.

[85] X. Yang et al., “DriveArena: A closed-loop generative simulation platform for autonomous driving,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025, pp. 26 933–26 943.

[86] R. Gao, K. Chen, Z. Li, L. Hong, Z. Li, and Q. Xu, “MagicDrive3D: Controllable 3D generation for any-view rendering in street scenes,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., 2026, pp. 5724– 5733.

[87] R. Gao et al., “MagicDrive: Street view generation with diverse 3D geometry control,” in Proc. Int. Conf. Learn. Represent., 2024.

[88] Y. Wen et al., “Panacea: Panoramic and controllable video generation for autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 6902–6912.

[89] W. Wu, X. Guo, W. Tang, T. Huang, C. Wang, and C. Ding, “DriveScape: High-resolution driving video generation by multi-view feature fusion,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 17 187–17 196.

[90] J. Yang et al., “Generalized predictive model for autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 14 662–14 672.

[91] X. Li, Y. Zhang, and X. Ye, “DrivingDiffusion: Layout-guided multiview driving scenarios video generation with latent diffusion model,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 469–485.

[92] Q. Wang, L. Fan, Y. Wang, Y. Chen, and Z. Zhang, “FreeVS: Generative view synthesis on free driving trajectory,” in Proc. Int. Conf. Learn. Represent., 2025.

[93] J. Mao et al., “DreamDrive: Generative 4D scene modeling from street view images,” in Proc. IEEE Int. Conf. Robot. Autom., 2025, pp. 367– 374.

[94] Z. Yu et al., “SGD: Street view synthesis with Gaussian splatting and diffusion prior,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., 2025, pp. 3812–3822.

[95] S. Hwang, M.-J. Kim, T. Kang, J. Kang, and J. Choo, “VEGS: View extrapolation of urban scenes in 3D Gaussian splatting using learned priors,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 1–18.

[96] Z. Yang, Z. Pan, Y. Yang, X. Zhu, and L. Zhang, “Driving view synthesis on free-form trajectories with generative prior,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025, pp. 28 083–28 092.

[97] S. Du, J. Liu, Q. Chen, H.-X. Chen, T.-J. Mu, and S. Yang, “RGE-GS: Reward-guided expansive driving scene reconstruction via diffusion priors,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025, pp. 25 756– 25 764.

[98] J. Z. Wu et al., “DIFIX3D+: Improving 3D reconstructions with singlestep diffusion models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 26 024–26 035.

[99] Y. Ji, K. Ma, H. Cai, A. Zhang, L. Ma, and X. Tan, “LidarPainter: One-step away from any lidar view to novel guidance,” in Proc. AAAI Conf. Artif. Intell., vol. 40, no. 7, 2026, pp. 5332–5340.

[100] H. Si et al., “DriveFix: Spatio-temporally coherent driving scene restoration,” 2026, arXiv:2603.16306.

[101] G. Li et al., “SPVC: Structured and panoptic video fixing for crossdataset driving scene rendering,” 2026, arXiv:2608.17420.

[102] Z. Liu et al., “SymDrive: Realistic and controllable driving simulator via symmetric auto-regressive online restoration,” 2025, arXiv:2512.21618.

[103] Y. Jia, J. Wang, J. Zhao, C. Li, S. Wang, and W. Gao, “DriveExplorer: Images-only decoupled 4D reconstruction with progressive restoration for driving view extrapolation,” 2025, arXiv:2512.23983.

[104] H. Che, J. Wang, Y. Chen, Z. Yang, S. Manivasagam, and R. Urtasun, “Diffusion-guided generalizable enhancer for urban scene reconstruction,” in Proc. IEEE Int. Conf. Robot. Autom., 2026, pp. 746–753.

[105] Y. Wang et al., “FaithFusion: Harmonizing reconstruction and generation via pixel-wise information gain,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2026, pp. 1198–1209.

[106] S. He, J. Chen, Z. Fang, D. Ren, and F. Zhao, “PRG-Fusion: Orchestrating generative priors with reconstruction evidence for driving view synthesis,” 2026, arXiv:2609.06948.

[107] Z. Kuang et al., “Effective multi-sensor conditioning for street-view novel-view synthesis,” 2026, arXiv:2606.01590.

[108] J. Liu et al., “GeoFlow: Efficient driving video generation via geometry-aligned priors,” in Proc. Eur. Conf. Comput. Vis., 2026, pp. 394–412.

[109] Y. Xiao et al., “VDEGaussian: Video diffusion enhanced 4D Gaussian splatting for dynamic urban scenes modeling,” 2025, arXiv:2508.02129.

[110] C. Ni et al., “ReconDreamer-RL: Enhancing reinforcement learning via diffusion-based reconstruction,” in Proc. Eur. Conf. Comput. Vis., 2026, pp. 677–696.

[111] D. Zermas, I. Izzat, and N. Papanikolopoulos, “Fast segmentation of 3D point clouds: A paradigm on LiDAR data for autonomous vehicle applications,” in Proc. IEEE Int. Conf. Robot. Autom., 2017, pp. 5067– 5073.

[112] N. Schertler, M. Tarini, W. Jakob, M. Kazhdan, S. Gumhold, and D. Panozzo, “Field-aligned online surface reconstruction,” ACM Trans. Graph., vol. 36, no. 4, pp. 1–13, 2017, art. no. 77.

[113] D. Zorin, P. Schroeder, and W. Sweldens, “Interpolating subdivision for meshes with arbitrary topology,” in Proc. 23rd Annu. Conf. Comput. Graph. Interact. Tech., 1996, pp. 189–192.

[114] K. Tan et al., “ExtraGS: Geometric-aware trajectory extrapolation with uncertainty-guided generative priors,” 2025, arXiv:2508.15529.

[115] K. Tan, Y. Shen, H. Sun, B. Wang, G. Chen, and H. Ye, “ViSE: A systematic approach to vision-only street-view extrapolation,” 2025, arXiv:2510.18341.

[116] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.

[117] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2018, pp. 586–595.

[118] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” in Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 6626–6637.