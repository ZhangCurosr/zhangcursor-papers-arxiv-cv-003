# Provisioning Robust and Efficient LiDAR Perception for Autonomous Driving

Tingxu Han<sup>∗</sup>, Zijie Shen<sup>∗</sup>, An Guo, Lyuke Wang, Chunrong Fang, Zhenyu Chen, and Wei Song

Abstract—LiDAR-based 3D object detection is a core perception service for autonomous driving, yet practical deployment requires maintaining robustness to point-cloud corruption while meeting latency and memory requirements. Existing approaches typically address robustness and resource efficiency separately, resulting in compact detectors that are fragile under corruption or robust detectors that remain costly to deploy. We therefore formulate LiDAR perception provisioning as constructing a detector that remains accurate under point-cloud corruptions while meeting constraints on inference latency and peak GPU memory. To achieve robustness, we first craft corrupted samples that mimic extreme weather conditions. Then we train candidate detectors offline by ordering clean and corrupted samples by perturbation sensitivity, through which detectors learn more from hard corrupted samples. We estimate perturbation sensitivity using an ensemble of seven heterogeneous 3D detectors by tracking changes in confidence, BEV overlap, and unmatched detections. To improve efficiency, we narrow the network and use offline neighborhood-density measurements to determine how many points and neighbors each sampling stage processes.

We evaluate our method using AP (Average Precision) under clean and corrupted conditions, inference latency, and peak allocated GPU memory. On KITTI for Car detection with PointRCNN, our method improves Moderate AP under corruption by 3.32 points over the original detector while reducing mean latency by 19.8% and peak allocated GPU memory by 53.9%. Across two datasets, two detector architectures, and three object classes, our method improves AP under corruption by 4.03 points on average over the corresponding original detectors, while reducing mean inference latency and peak allocated GPU memory by 14.4% and 38.0%, respectively. Our code is available at https://github.com/drunklibai/robust-efficient-lidar-perception.

Index Terms—Service provisioning, quality of service, LiDAR perception service, corruption robustness, resource efficiency.

## 1 INTRODUCTION

IDAR-BASED 3D object detection is a core perception point clouds into the locations, dimensions, and orientations of surrounding objects, and serves downstream consumers such as tracking, prediction, and motion planning that issue per-frame inference requests and expect detections within a response-time budget [1–3]. Onboard perception must process every incoming frame within its latency budget and available accelerator memory while remaining reliable under degraded input. [4, 5]. Extreme weather such as rain and snow, measurement noise, and point-density variations can corrupt LiDAR measurements and substantially degrade 3D detection performance [5, 6]. A deployable LiDAR perception service must therefore satisfy two coupled requirements: robustness to point-cloud corruption under extreme weather conditions such as rain, snow, noise, and point-density variation, and resource efficiency in terms of inference latency and memory use.

Existing research largely pursues these two objectives in isolation. Robustness-oriented approaches employ data augmentation, domain generalization or adaptation, and corruption-aware training to improve 3D detection under distribution shifts and adverse conditions [5, 7, 8]. Most of these methods modify only the training process, leaving the detector architecture and point-sampling workload unchanged. As a result, they improve robustness without reducing inference latency or memory use. Efficiency-oriented approaches instead rely on weight or channel pruning, knowledge distillation, lightweight detector design, or computationaware point sampling [9–13]. Their reduction decisions are typically guided by parameter magnitudes, channel statistics, or accuracy on clean data, with limited consideration of detector behavior under point-cloud corruption.

To address this gap, we cast the problem as offline provisioning of a LiDAR perception service: before deployment, a provider constructs a fixed detector that meets both requirements, so that no adaptation is needed while serving requests. We develop a provisioning framework that builds such a detector to improve robustness to pointcloud corruption while reducing inference latency and GPU memory use. For robustness, we first pair each clean training frame with a corrupted counterpart. An ensemble of seven 3D detectors then measures perturbation sensitivity from prediction discrepancies between each clean–corrupted pair, and we order all clean and corrupted samples from low to high sensitivity for training. For efficiency, we narrow the detector network and, for PointRCNN, use offline pointdensity statistics to reduce the numbers of points and neighbors processed at each sampling stage. The resulting detector is trained offline and then deployed for inference.

Experiments on KITTI [14] and ONCE [15] using PointR-CNN and SECOND show that our framework improves robustness under corruption while reducing inference cost. The main contributions are summarized as follows:

• We establish an offline provisioning setting for LiDAR perception services that considers robustness to pointcloud corruptions and deployment efficiency.

• We develop a provisioning framework that integrates two coordinated components. Specifically, we rank clean– corrupted pairs by perturbation sensitivity for robust training. We reduce inference cost through channelwidth reduction and data-informed budget allocation.

• We evaluate the framework on KITTI and ONCE using the point-based PointRCNN and voxel-based SECOND across three object classes. Compared with the corresponding original detectors, the provisioned detectors improve AP under corruption by 4.03 points on average while reducing mean inference latency and peak allocated GPU memory by 14.4% and 38.0%, respectively.

## 2 RELATED WORK

## 2.1 LiDAR Detection under Point-Cloud Corruptions

Research on robust LiDAR detection mainly follows two directions: common-corruption benchmarking and weatherspecific robustness enhancement. Dong et al. [6] and Kong et al. [5] introduce KITTI-C, nuScenes-C, and Waymo-C to evaluate 3D detectors under realistic point-cloud corruptions, while Robo3D covers adverse weather, external disturbances, and sensor failures. Weather-specific methods synthesize physically grounded fog and snowfall on clean LiDAR scans and retrain detectors under the corresponding conditions [16, 17]. These studies primarily aim to evaluate or improve detection robustness, without simultaneously reducing inference latency and memory use. Our offline provisioning framework instead combines perturbation-sensitivity-based data ordering with resource-efficient detector construction.

## 2.2 Data Prioritization for Robust Training

Curriculum learning presents training examples in a meaningful progression, typically from easy to hard, whereas selfpaced learning determines this progression from the learner’s current loss [18, 19]. Curricular Object Manipulation applies this idea to LiDAR detection by dynamically reweighting object-level losses and progressively inserting harder objects through ground-truth augmentation [20]. Other data-centric methods prioritize selected training examples: core-set active learning chooses representative samples for annotation, while online hard example mining retains high-loss regions during each update [21, 22]. Our method instead changes only the training order. It retains all clean and corrupted frames without changing their loss weights, assigns each clean–corrupted pair a fixed perturbation-sensitivity score based on the prediction discrepancy of an ensemble of pretrained reference detectors, and presents the complete mixed set from low to high perturbation-sensitivity.

SECOND uses sparse convolutions to avoid computation on empty voxels, while PointPillars organizes points into vertical pillars for efficient 2D convolution [2, 23]. For point-based detection, PointNet++-based detectors such as PointRCNN progressively downsample points and aggregate a bounded number of local neighbors [1, 24]. IA-SSD further improves efficiency by learning which foreground points to retain during downsampling [13]. Model-compression methods reduce network size through channel or structural pruning and knowledge distillation [10, 11, 25]. PointDistiller adapts structured knowledge distillation to point- and voxel-based 3D detectors [26]. These approaches reduce cost through representations, learned sampling, pruning, or distillation, but do not combine their efficiency mechanisms with training driven by detector behavior under point-cloud corruption. Our offline provisioning framework instead narrows network widths and, for PointRCNN, uses offline neighborhooddensity statistics to set fixed point and neighbor budgets for each sampling stage; for SECOND, we use offline data statistics to set a fixed voxel budget. It requires neither a learned sampler nor teacher–student distillation; ensemble predictions are used only to compute the perturbationsensitivity scores that determine training order.

## 2.3 Efficient LiDAR Detection and Point Processing

Efficient LiDAR detection has been studied through representation design, point processing, and model compression.

## 3 ROBUST AND RESOURCE-EFFICIENT LIDAR DE-TECTOR CONSTRUCTION

## 3.1 Problem Setting and Framework Overview

A deployable LiDAR perception for autonomous driving must remain accurate under point-cloud corruptions while limiting inference latency and GPU memory use. We therefore construct a fixed detector offline to preserve clean-data average precision (AP), improve AP under corruption, and reduce mean inference latency and peak allocated GPU memory relative to the original detector.

Let $\mathbf { \dot { \mathcal { D } } } = \{ ( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) \} _ { i = 1 } ^ { N }$ denote the clean training set, where $\mathbf { x } _ { i }$ is a LiDAR point cloud and $\mathbf { y } _ { i }$ contains its 3D object annotations. Let A denote the target detector architecture and $f _ { \theta _ { 0 } }$ the corresponding original detector with its default configuration. Given ${ \mathcal { D } } ,$ , a corruption operator $C ( \cdot ) , A ,$ and K pretrained reference detectors $\{ f _ { k } \} _ { k = 1 } ^ { k } ,$ our framework constructs a robust and resource-efficient detector $f _ { \theta ^ { \prime } }$

For any detector $f ,$ let APcln(f) and APcor(f) denote its AP on clean and corrupted data, and let Lat(f) and Mem(f) denote its mean inference latency and peak allocated GPU memory. The construction has two objectives:

$$
\begin{array} { r l } { \operatorname { m a x i m i z e } } & { \{ \operatorname { A P } _ { \mathrm { c l n } } ( f ) , \operatorname { A P } _ { \mathrm { c o r } } ( f ) \} , } \\ { \operatorname { m i n i m i z e } } & { \{ \operatorname { L a t } ( f ) , \operatorname { M e m } ( f ) \} . } \end{array}\tag{1}
$$

The objectives are interpreted jointly: relative to $f _ { 0 } ,$ we seek to improve AP under corruption and reduce latency and memory while limiting the loss of clean-data AP.

As shown in Fig. 1, our framework addresses the two objectives through two coordinated components. For robustness, it pairs each clean frame with a corrupted counterpart, uses an ensemble reference to measure their prediction discrepancy, and orders the complete 2N-sample training set from low to high perturbation sensitivity. For efficiency, it combines channel-width reduction with data-informed budget allocation. PointRCNN uses density-guided point- and neighbor-sampling budgets, whereas SECOND uses datainformed voxel-budget allocation. These architecture-specific budgets are determined offline. The resulting detector is trained from scratch on the ordered training set, with all construction decisions completed offline.

![](images/b648de8efe9fb4b3df78c1977c6bf71638b254cc00abca0a595706dc724b8741.jpg)  
Fig. 1. Offline profile-construction pipeline. Paired clean and corrupted point clouds are scored by an ensemble reference for clean-versus-corrupted prediction inconsistency. The reordered training data are combined with a density-guided sampling-budget allocation to train a compact detector.

## 3.2 Training Data Construction and Prioritization

We prepare the training data by generating a corrupted counterpart for each clean frame, scoring each pair by perturbation sensitivity, and ordering all clean and corrupted samples from low to high sensitivity. The intuition is that this progression may help the detector learn stable geometric and object-level patterns from less corruption-sensitive frames before learning from more challenging ones.

## 3.2.1 Corruption Injection

For each clean training sample $\left( \mathbf { x } _ { i } , \mathbf { y } _ { i } \right) \in \mathcal { D } _ { \cdot }$ , we generate and cache one corrupted counterpart $\tilde { \mathbf { x } } _ { i } = C _ { \tau _ { i } } ( \mathbf { x } _ { i } )$ , where $\tau _ { i }$ denotes the corruption type assigned to sample i. The corrupted point cloud retains the original annotation $\mathbf { y } _ { i } ,$ forming the paired corrupted set $\mathcal { D } ^ { c } = \{ ( \tilde { \mathbf { x } } _ { i } , \mathbf { y } _ { i } ) \} _ { i = 1 } ^ { N }$ . Each corrupted sample is generated once and reused for both sensitivity estimation and detector training, ensuring a consistent clean–corrupted pair throughout offline construction.

We consider six scene-level corruption types covering point-density changes, measurement noise, and adverse weather. Density decrease removes LiDAR points, whereas density increase introduces additional points. Gaussian noise perturbs point coordinates, and background noise injects randomly distributed clutter points. Rain and snow introduce precipitation-induced scattering and particle returns. All six corruption types are applied at severity level 4 of 5.

## 3.2.2 Perturbation-Sensitivity Scoring

For each clean–corrupted pair $\left( \mathbf { x } _ { i } , \tilde { \mathbf { x } } _ { i } \right)$ , we quantify how much the detection results change after corruption. The detection results on the clean frame serve as a same-scene reference, separating corruption-induced changes from the scene’s intrinsic difficulty. To reduce dependence on any single detector, we use ${ \dot { K } } = 7$ pretrained fixed detectors including PointRCNN [1], PointRCNN-IoU [1, 27], Part-$A ^ { 2 }$ [28], PointPillars [23], SECOND [2], SECOND-IoU [2, 27], and PV-RCNN [3], to produce ensemble-fused detections for both inputs. We then match the clean and corrupted detections by class and measure their confidence discrepancy, BEV overlap, and unmatched detections. We combine these discrepancies into a frame-level perturbation-sensitivity score $u _ { i } ,$ where a larger value indicates that the frame is more strongly affected by corruption and should appear later in the prioritized training order for better learning.

We first establish correspondences between the ensemblefused detections on the clean and corrupted point clouds. Within each object class, we greedily match boxes using a bird’s-eye-view (BEV) intersection-over-union (IoU) threshold of $0 . 5 ,$ with each box matched at most once. Let $\mathcal { M } _ { i }$ denote the resulting set of matched box pairs for frame i. Matched pairs are used to measure confidence and BEV localization changes, while unmatched detections are accounted for separately. For matched pairs $( a , b ) \in \mathcal { M } _ { i } ,$ , we measure the absolute confidence difference and the discrepancy in BEV overlap. Averaging over the matched pairs gives

$$
\Delta _ { i } ^ { \mathrm { s c o r e } } = \frac { 1 } { | \mathcal { M } _ { i } | } \sum _ { ( a , b ) \in \mathcal { M } _ { i } } | s ( a ) - s ( b ) | ,\tag{2}
$$

$$
\Delta _ { i } ^ { \mathrm { b o x } } = \frac { 1 } { | \mathcal { M } _ { i } | } \sum _ { ( a , b ) \in \mathcal { M } _ { i } } \left[ 1 - \mathrm { I o U } _ { \mathrm { B E V } } ( a , b ) \right] ,\tag{3}
$$

where $s ( a )$ and $s ( b )$ are the confidence scores of the clean and corrupted detections, respectively. The confidence discrepancy measures the magnitude of the change, regardless of whether confidence increases or decreases. Both terms are defined as zero when $\mathcal { M } _ { i }$ is empty.

Detections without a match capture changes that the two matched-box terms do not cover. Let $n _ { i , c }$ and $\tilde { n } _ { i , c }$ denote the numbers of clean and corrupted detections for class c, respectively, and define $\begin{array} { r } { Z _ { i } = \sum _ { c } \mathrm { \tilde { m a x } } ( n _ { i , c } , \tilde { n } _ { i , c } ) } \end{array}$ . We measure the unmatched-detection discrepancy as

![](images/87fd791493814e855fdbab0e4575e7ecf1873915ebc1361457f730777a97c9fb.jpg)  
Fig. 2. Case study of perturbation sensitivity. We compare ensemble predictions on an original point cloud and its corrupted version. Differences in false predictions, confidence scores, missing detections, and object locations are combined into a frame-level sensitivity score. A higher score indicates a harder sample that appears later in the prioritized training order.

$$
\Delta _ { i } ^ { \mathrm { m i s s } } = \left\{ \begin{array} { l l } { 1 - | \mathcal { M } _ { i } | / Z _ { i } , } & { Z _ { i } > 0 , } \\ { 0 , } & { Z _ { i } = 0 . } \end{array} \right.\tag{4}
$$

This term measures the deficit in matched detections relative to the sum of the larger detection count for each class. It captures unmatched predictions on either input. The framelevel perturbation-sensitivity score is then defined as

$$
u _ { i } = \Delta _ { i } ^ { \mathrm { s c o r e } } + \Delta _ { i } ^ { \mathrm { b o x } } + \lambda _ { \mathrm { m i s s } } \Delta _ { i } ^ { \mathrm { m i s s } } , \qquad \lambda _ { \mathrm { m i s s } } = 0 . 5 .\tag{5}
$$

Fig.2 intuitively illustrates the corruption-induced prediction discrepancies this score captures.

## 3.2.3 Prioritized Training Order

The perturbation-sensitivity score $u _ { i }$ is computed for a clean– corrupted pair and assigned to both the clean frame x<sub>i</sub> and its corrupted counterpart $\tilde { \mathbf { x } } _ { i }$ . The complete training list S contains all N clean samples and their N corrupted counterparts, each with its original annotation and assigned score. We obtain the prioritized list by sorting these 2N entries in ascending order of sensitivity:

$$
S _ { \pi } = \mathrm { s o r t } _ { \uparrow } ( S ; u ) .\tag{6}
$$

This ordering presents samples with smaller corruptioninduced prediction changes before those with larger changes.

Prioritization changes only the training-sample presentation order: all samples are retained, and the detection loss and sample weights remain unchanged. To preserve this order during training, we disable shuffling and traverse the same prioritized training list $S _ { \pi }$ in every epoch.

## 3.3 Resource-Efficient Detector Construction

While training data construction and prioritization target corruption robustness, we address deployment efficiency by reducing the detector’s inference workload. We reduce channel widths in both PointRCNN and SECOND, representing point-based and voxel-based architectures, respectively. For PointRCNN, we additionally use offline point-density statistics to determine point and neighborhood sampling budgets within the network. For SECOND, offline pointcount statistics additionally guide voxel-budget allocation. These modifications aim to reduce mean inference latency and peak allocated GPU memory. The channel widths and sampling budgets are specified before training and remain fixed throughout training and inference.

## 3.3.1 Point-Based Detectors

For PointRCNN, we reduce feature widths and sampling budgets for points and neighbors to decrease the inference workload. We scale the MLP and fully connected layer widths in the backbone, feature-propagation module, point head, and ROI head by 0.7, rounding to implementationcompatible values. For example, the point-head and ROIhead widths change from [256, 256] to [176, 176].

Sampling budgets are determined from point-density statistics computed offline over the training set, after filtering points to the detector’s input range. We measure the median valid-point count across training frames. and estimate the average neighbor count within each grouping radius using sampled query and reference points. These statistics guide the point and neighbor budgets, respectively.

For grouping branch b at layer l, let $\bar { c } _ { l , b }$ denote the average neighbor count and $S _ { l , b } ^ { \mathrm { o r i g } }$ the original neighbor budget. We use their ratio to estimate local redundancy:

$$
R _ { l , b } = \operatorname* { m a x } \left( \frac { \bar { c } _ { l , b } } { S _ { l , b } ^ { \mathrm { o r i g } } } , 1 \right) .\tag{7}
$$

The retention ratio for the neighbor-sampling budget and the corresponding integer-valued budget are given by

$$
\begin{array} { r l } & { \gamma _ { l , b } = \operatorname* { m i n } \left( \gamma _ { \mathrm { b u d g e t } } , \operatorname* { m a x } \left( \gamma _ { \mathrm { m i n } } , R _ { l , b } ^ { - 1 / 2 } \right) \right) , } \\ & { S _ { l , b } ^ { \mathrm { n e w } } = \operatorname* { m a x } \left( S _ { \mathrm { m i n } } , \mathrm { r o u n d } \left( \gamma _ { l , b } S _ { l , b } ^ { \mathrm { o r i g } } \right) \right) , } \end{array}\tag{8}
$$

where $0 < \gamma _ { \mathrm { m i n } } \leq \gamma _ { \mathrm { b u d g e t } } \leq 1$ bound the retained fraction, and $S _ { \mathrm { m i n } } ~ \le ~ S _ { l , b } ^ { \mathrm { o r i g } }$ prevents excessively small neighborsampling budget. Higher redundancy therefore permits a greater budget reduction, subject to these bounds.

For point sampling, let $\tilde { N }$ be the median valid-point count and $N _ { \mathrm { r e f } }$ the reference input size. We compute a shared scaling ratio and apply it to each layer:

$$
\begin{array} { r l } & { \gamma ^ { \mathrm { p o i n t } } = \operatorname* { m i n } \left( \gamma _ { \mathrm { b u d g e t } } ^ { \mathrm { p o i n t } } , \operatorname* { m a x } \left( \gamma _ { \mathrm { m i n } } ^ { \mathrm { p o i n t } } , \sqrt { \frac { \tilde { N } } { N _ { \mathrm { r e f } } } } \right) \right) , } \\ & { N _ { l } ^ { \mathrm { n e w } } = \operatorname* { m a x } \left( 1 , \mathrm { r o u n d } \left( \gamma ^ { \mathrm { p o i n t } } N _ { l } ^ { \mathrm { o r i g } } \right) \right) , } \end{array}\tag{9}
$$

where $0 < \gamma _ { \mathrm { m i n } } ^ { \mathrm { p o i n t } } \ \leq \ \gamma _ { \mathrm { b u d g e t } } ^ { \mathrm { p o i n t } } \ \leq \ 1$ . The same budget rules apply to applicable ROI point sampling and neighborhood aggregation; stages that aggregate all points retain that setting. The resulting network widths and sampling budgets remain fixed throughout training and inference.

Algorithm 1 Offline Robust and Resource-Efficient Detector   
Construction   
Require: Clean set $\mathcal { D } ;$ corruption operator $C ;$ reference detectors   
$\mathrm { \bar { \{ } }  f _ { k } \} _ { k = 1 } ^ { K } ;$ architecture A; construction settings; epochs T   
Ensure: Trained detector $f _ { \theta ^ { \prime } }$   
1: Initialize an empty list S   
2: for $i = 1$ to N do   
3: Generate and cache $\tilde { \mathbf { x } } _ { i } \gets C _ { \tau _ { i } } ( \mathbf { x } _ { i } )$   
4: Fuse reference detections separately on $\mathbf { x } _ { i }$ and $\tilde { \mathbf { x } } _ { i }$   
5: Match detections by class using BEV IoU $\geq 0 . 5$   
6: Compute $u _ { i }$ using (5)   
7: Append $( \mathbf { x } _ { i } , \mathbf { y } _ { i } , \bar { u _ { i } } )$ and $( \tilde { \mathbf { x } } _ { i } , \mathbf { y } _ { i } , u _ { i } )$ to $s$   
8: end for   
9: $S _ { \pi } \gets \mathrm { s o r t } _ { \uparrow } ( S ; u )$   
10: Reduce architecture-specific feature widths   
▷ point-processing or BEV layers   
11: if A uses point sampling and neighborhood aggregation then   
12: Estimate $\tilde { N }$ and $\bar { c } _ { l , b }$ over $\mathcal { D }$   
13: Set $S _ { l , b } ^ { \mathrm { n e w } }$ and $N _ { l } ^ { \mathrm { n e w } }$ using (8) and (9)   
14: end if   
15: Instantiate $f _ { \theta ^ { \prime } }$ with the resulting settings   
16: Train from scratch on $s _ { \pi }$ for $_ T$ epochs, preserving its order in every   
epoch   
17: return $f _ { \theta ^ { \prime } }$

## 3.3.2 Voxel-Based Detectors

SECOND groups points into voxels and extracts features through sparse 3D convolutions followed by a BEV backbone. For SECOND, we combine channel-width reduction in the BEV backbone and upsampling layers with data-informed voxel-budget allocation. Offline data statistics guide the voxel budget used by the detector. This voxel-based allocation complements the point- and neighbor-sampling budget allo cation used for PointRCNN. The resulting detector is trained from scratch on the prioritized training list $S _ { \pi }$ using the constructed channel-width and voxel-budget configuration.

## 3.4 Complete Procedure

Both detectors are trained from scratch on the prioritized training list $S _ { \pi } .$ , with the reduced channel widths and architecture-specific budgets fixed throughout training and inference. Algorithm 1 jointly targets corruption robustness and deployment efficiency. Given $\mathcal { D } , C , \{ f _ { k } \} _ { k = 1 } ^ { K } , \mathcal { A } ,$ construction settings, and $T ,$ it initializes the training list (line 1), generates corrupted counterparts (lines 2–3), fuses reference predictions (line 4), matches detections (line 5), and computes perturbation-sensitivity scores (line 6). Each pair is added with its annotation and shared score (line 7), and the complete list is sorted into $S _ { \pi }$ (line 9).

For efficiency, the algorithm reduces feature widths (line 10) and, where applicable, uses offline density statistics to allocate sampling budgets (lines 11–13); otherwise, spatial-processing settings are retained (lines 14–16). It then instantiates the reduced detector (line 17), trains it from scratch on $S _ { \pi }$ for $T$ epochs (line 18), and returns $f _ { \theta ^ { \prime } }$ (line 19).

## 4 EVALUATION

We evaluate our offline provisioning framework against its two objectives: robustness to point-cloud corruption and deployment efficiency. We also examine changes in clean-data accuracy and the contributions of the framework components. Our experiments address three research questions:

• RQ1: Corruption Robustness. How does our framework affect detection accuracy under point-cloud corruption and on clean data?

• RQ2: Deployment Efficiency. How much does our framework reduce mean inference latency and peak allocated GPU memory compared with the original detectors and baseline methods?

• RQ3: Component Contributions. How do perturbationsensitivity-based data prioritization and resourceefficient detector construction contribute to the observed results?

## 4.1 Experimental Setup

We evaluate our framework on KITTI [14] and ONCE [15] using PointRCNN [1] and SECOND [2], representing pointbased and voxel-based detector architectures, respectively. For both detectors, we evaluate Car, Pedestrian, and Cyclist on the clean validation sets and their corrupted counterparts.

Experiments use OpenPCDet v0.6 [29], PyTorch 1.13, CUDA 11.3, and one NVIDIA RTX 4090D.

We evaluate detection accuracy using average precision (AP) on KITTI and ONCE [14, 15]. Following previous work [6, 8], KITTI results use AP<sub>R40</sub>, evaluated at 40 recall positions, and are reported at the Easy, Moderate, and Hard difficulty levels. For ONCE, we also compute AP at 40 recall positions, using an R40 variant of the orientationaware evaluation procedure [15], and report results across the evaluated distance intervals. Both datasets therefore use the same recall sampling resolution, while retaining their dataset-specific matching and evaluation criteria.

## 4.2 RQ1: Corruption Robustness

We evaluate corruption robustness using average precision (AP) under corruption and report clean AP to assess the accompanying changes in clean-data accuracy. Using the dataset-specific protocols described in Section 4.1, we use KITTI Moderate AP for the main comparisons and report ONCE AP over three distance intervals: 0–30 m, 30–50 m, and beyond 50 m. For compact comparisons, we additionally report the arithmetic mean of the three ONCE distance-bin AP values, termed range-average AP.

Each clean frame has one fixed corrupted counterpart, generated with the toolkit of [30] using the six corruption types in Section 3.2.1 at severity level 4 of 5. For the proposed method, the training set contains all clean training frames and their cached corrupted counterparts in a 1:1 ratio. AP under corruption is evaluated in one pass over the mixed corrupted validation set, rather than averaged over six separate pertype evaluations. The results therefore characterize the evaluated mixed-corruption protocol. The complete baseline accuracy comparison focuses on Car, while the cross-class comparison includes Original, Network Slimming, and Ours. Original uses the released checkpoint and default detector configuration. Retrain is the full-size detector retrained on the 1:1 clean–corrupted mixture without channel-width or architecture-specific budget reallocation. GMP applies unstructured magnitude pruning, LTH/IMP applies iterative magnitude pruning with weight rewinding, and Network Slimming (Net. Slim. in the tables) reduces channels using batch-normalization scales. Ours combines perturbationsensitivity-based training-data prioritization with resourceefficient detector construction. For PointRCNN, it uses channel-width reduction and density-guided point- and neighbor-sampling budgets. For SECOND, it uses channelwidth reduction in the BEV backbone and upsampling layers together with data-informed voxel-budget allocation.

Table 1 shows that AP under corruption improves over Original in all six detector–class combinations on KITTI and five of six on ONCE. The mean gains are 3.26 points in KITTI Moderate AP and 4.80 points in ONCE rangeaverage AP. Across the twelve dataset–detector–class combinations, the descriptive mean gain is 4.03 points. The exception is PointRCNN Pedestrian on ONCE, whose AP under corruption decreases from 4.19 to 3.87, a loss of 0.32 points. A plausible explanation for the overall gains is the combination of corrupted-data training and perturbationsensitivity ordering: paired clean and corrupted frames expose the detector to the evaluated degradations, while the low-to-high-sensitivity order is intended to support progressive learning. RQ3 examines the contributions of the training order and ensemble reference.

TABLE 1  
Detection Accuracy across Datasets, Detectors, and Classes
<table><tr><td rowspan="2">Class</td><td rowspan="2">Method</td><td colspan="2">KITTI Moderate</td><td colspan="2">ONCE Range Avg.</td></tr><tr><td>Clean</td><td>Corr.</td><td>Clean Corr.</td><td></td></tr><tr><td colspan="6">PointRCNN</td></tr><tr><td rowspan="3">Car</td><td>Original</td><td>80.65</td><td>61.03</td><td>43.78</td><td>25.02</td></tr><tr><td>Net. Slim.</td><td>77.79</td><td>56.78</td><td>42.95</td><td>30.37</td></tr><tr><td>Ours</td><td>80.04</td><td>64.35</td><td>43.47</td><td>32.20</td></tr><tr><td rowspan="3">Pedestrian</td><td>Original</td><td>54.81</td><td>43.86</td><td>4.95</td><td>4.19</td></tr><tr><td>Net. Slim.</td><td>57.28</td><td>45.10</td><td>2.40</td><td>2.39</td></tr><tr><td>Ours</td><td>57.36</td><td>46.26</td><td>3.98</td><td>3.87</td></tr><tr><td rowspan="3">Cyclist</td><td>Original</td><td>71.09</td><td>47.94</td><td>26.41</td><td>16.45</td></tr><tr><td>Net. Slim.</td><td>68.40</td><td>49.00</td><td>20.79</td><td>16.34</td></tr><tr><td>Ours</td><td>69.50</td><td>52.81</td><td>23.95</td><td>19.90</td></tr><tr><td>SECOND</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Car</td><td></td><td>81.61</td><td>62.17</td><td></td><td></td></tr><tr><td>Original Net. Slim.</td><td>78.84</td><td>66.19</td><td>68.74 67.84</td><td>35.92 45.31</td></tr><tr><td>Ours</td><td>79.05</td><td>67.10</td><td>69.86</td><td>46.72</td></tr><tr><td rowspan="3">Pedestrian Net. Slim.</td><td>Original</td><td>51.15</td><td>36.89</td><td>27.64</td><td>16.64</td></tr><tr><td></td><td>43.75</td><td>36.60</td><td>26.26</td><td>17.55</td></tr><tr><td>Ours</td><td>45.01</td><td>37.70</td><td>28.92</td><td>18.46</td></tr><tr><td rowspan="3">Cyclist</td><td>Original</td><td>66.74</td><td>45.25</td><td>52.50</td><td>29.62</td></tr><tr><td>Net. Slim.</td><td>44.75</td><td>47.25</td><td>51.26</td><td>33.82</td></tr><tr><td>Ours</td><td>66.05</td><td>48.50</td><td>53.87</td><td>35.50</td></tr></table>

KITTI values are Moderate $\mathrm { { A P } _ { \mathrm { { R 4 0 } } } . }$ ONCE values are unweighted arithmetic means of the AP values in the 0–30 m, 30–50 m, and beyond 50 m bins. Clean and Corr. denote evaluation on clean and corrupted validation inputs, respectively.

Table 2 compares all six methods on Car detection. On KITTI, Ours raises Moderate AP under corruption from 61.03 to 64.35 for PointRCNN and from 62.17 to 67.10 for SECOND, compared with the corresponding Original baselines. Relative to Network Slimming, the gains are 7.57 and 0.91 points, respectively. Retrain achieves slightly higher values of 64.73 and 67.84. On ONCE, Ours and

Retrain both attain 32.20 range-average AP under corruption with PointRCNN after rounding, while Ours reaches 46.72 with SECOND compared with 46.32 for Retrain. Across these four dataset–detector comparisons, the reported APunder-corruption summaries for Ours differ from those for full-size Retrain by at most 0.74 points. RQ2 reports the corresponding latency and memory measurements. These comparisons reflect the complete configurations and do not isolate the effects of training sample order, channel widths, or architecture-specific budgets.

On KITTI, PointRCNN Car loses 0.61 points of clean Moderate AP while gaining 3.32 points under corruption. SECOND Pedestrian presents a larger trade-off: clean AP decreases from 51.15 to 45.01, while AP under corruption increases from 36.89 to 37.70. On ONCE, PointRCNN loses clean range-average AP in all three classes, by 0.31, 0.97, and 2.46 points, whereas SECOND gains 1.12, 1.28, and 1.37 points. PointRCNN Pedestrian on ONCE loses both clean AP and AP under corruption. These differences show that the benefits are not uniform across architectures and classes. The observed accuracy trade-offs reflect the performance of the complete configurations, which jointly incorporate training-data prioritization, channel-width reduction, and architecture-specific budget allocation. The results do not separate the effects of these possible effects, but show why clean AP must be considered alongside AP under corruption.

For ONCE Car detection, PointRCNN improves AP under corruption by 14.97, 6.55, and 0.02 points in the 0–30 m, 30– 50 m, and beyond-50 m intervals, respectively. SECOND improves by 17.26, 10.55, and 4.59 points. The PointRCNN gain is concentrated in the nearer intervals, whereas SEC-OND improves in all three. Perturbation-sensitivity ordering changes how training samples are presented, but does not restore missing measurements at inference; this may limit its benefits when the available geometric information is insufficient. The near-zero PointRCNN gain beyond 50 m also indicates that the aggregate improvement should not be interpreted as a uniform long-range benefit.

## 4.3 RQ2: Deployment Efficiency

We measure deployment efficiency using mean inference latency and peak allocated GPU memory, and report the 95th-percentile (p95) latency to characterize slower inference iterations. Measurements use batch size 1 on the hardware in Section 4.1. Each checkpoint undergoes 100 warm-up iterations followed by 500 timed iterations, with CUDA synchronization on every iteration. We repeat the measurement five times for each clean and corrupted input condition. Latency is reported in milliseconds; peak allocated GPU memory is measured in MB with torch.cuda.max\_memory\_allocated().

Table 3 shows that Ours reduces mean inference latency and peak allocated GPU memory relative to Original in all eight settings covering two datasets, two detectors, and clean and corrupted inputs. Averaging the relative reductions across these settings gives 14.4% lower mean latency and 38.0% lower peak allocated GPU memory. For clean KITTI inputs, PointRCNN latency decreases from 47.96 to 38.46 ms (19.8%), and memory from 408.1 to 188.2 MB (53.9%). On clean ONCE inputs, the corresponding reductions are 25.3% and 53.7%. SECOND achieves smaller reductions: 7.8% in latency and 22.5% in memory on clean KITTI inputs, and 2.6% and 22.7% on clean ONCE inputs. Both detectors combine channel-width reduction with data-informed budget allocation, but the budgeted operations differ: PointRCNN adjusts point and neighbor sampling, whereas SECOND adjusts the voxel budget. The reported measurements capture the combined effects of these modifications and do not isolate their individual contributions to the resource savings.

TABLE 2  
Car Detection Accuracy on Clean and Corrupted Data
<table><tr><td rowspan="3">Detector</td><td rowspan="3">Method</td><td colspan="6">KITTI</td><td colspan="6">ONCE</td></tr><tr><td colspan="3">Clean</td><td colspan="3">Corrupted</td><td colspan="3">Clean</td><td colspan="3">Corrupted</td></tr><tr><td>Easy</td><td>Moderate</td><td>Hard</td><td>Easy</td><td>Moderate Hard</td><td></td><td>0-30m</td><td>30-50m</td><td>&gt;50m</td><td></td><td>0-30m 30-50m</td><td>&gt;50m</td></tr><tr><td rowspan="6">PointRCNN</td><td>Original</td><td>91.62</td><td>80.65</td><td>78.0777.08</td><td></td><td>61.03</td><td>56.90</td><td>73.83</td><td>39.83</td><td>17.68</td><td>49.60</td><td>17.85</td><td>7.62</td></tr><tr><td>Retrain</td><td>88.71</td><td>80.01</td><td>77.35 82.13</td><td></td><td>64.73</td><td>60.08</td><td>73.21</td><td>39.80</td><td>17.44</td><td>64.55</td><td>24.31</td><td>7.74</td></tr><tr><td>GMP</td><td>74.02</td><td>62.84</td><td>60.37</td><td>62.53</td><td>49.16</td><td>43.84</td><td>57.34</td><td>27.63</td><td>12.65</td><td>45.92</td><td>16.46</td><td>5.52</td></tr><tr><td>LTH/IMP</td><td>89.11</td><td>77.85</td><td>75.15 79.56</td><td></td><td>62.31</td><td>57.72</td><td>70.96</td><td>37.13</td><td>14.62</td><td>60.02</td><td>21.83</td><td>5.31</td></tr><tr><td>Net. Slim.</td><td>89.29</td><td>77.79</td><td>73.14 73.52</td><td></td><td>56.78</td><td>50.59</td><td>72.95</td><td>39.36</td><td>16.55</td><td>61.88</td><td>22.03</td><td>7.20</td></tr><tr><td>Ours</td><td>89.56</td><td>80.04</td><td>77.53</td><td>81.71</td><td>64.35</td><td>60.01</td><td>73.05</td><td>39.86</td><td>17.50</td><td>64.57</td><td>24.40</td><td>7.64</td></tr><tr><td rowspan="6">SECOND</td><td>Original</td><td>90.55</td><td>81.61</td><td>78.6077.18</td><td></td><td>62.17</td><td>58.68</td><td>85.21</td><td>66.99</td><td>54.02</td><td>55.25</td><td>31.33</td><td>21.18</td></tr><tr><td>Retrain</td><td>89.23</td><td>78.02</td><td>75.3083.13</td><td></td><td>67.84</td><td>64.49</td><td>85.00</td><td>68.82</td><td>55.39</td><td>72.28</td><td>41.09</td><td>25.59</td></tr><tr><td>GMP</td><td>88.75</td><td>78.97</td><td></td><td>75.8581.57</td><td>65.32</td><td>61.86</td><td>80.33</td><td>57.05</td><td>42.77</td><td>65.97</td><td>33.88</td><td>19.47</td></tr><tr><td>LTH/IMP</td><td>88.88</td><td>78.73</td><td>75.52</td><td>81.22</td><td>65.05</td><td>61.63</td><td>83.14</td><td>63.44</td><td>50.01</td><td>69.62</td><td>37.95</td><td>22.71</td></tr><tr><td>Net. Slim.</td><td>88.18</td><td>78.84</td><td>75.7981.54</td><td></td><td>66.19</td><td>62.52</td><td>83.49</td><td>66.42</td><td>53.62</td><td>71.70</td><td>39.99</td><td>24.24</td></tr><tr><td>Ours</td><td>89.49</td><td>79.05</td><td>75.94 82.71</td><td></td><td>67.10</td><td>62.64</td><td>86.20</td><td>68.97</td><td>54.41</td><td>72.51</td><td>41.88</td><td>25.77</td></tr></table>

KITTI reports Easy/Moderate/Hard $\mathrm { A P _ { R 4 0 } ; }$ ONCE reports range-wise AP. Baselines are not budget-matched.

TABLE 3  
Inference Latency and Peak Allocated GPU Memory on the RTX 4090D
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Method</td><td colspan="3">KITTI</td><td colspan="3">ONCE</td></tr><tr><td>Mean latency</td><td>p95 latency</td><td>Peak memory</td><td>Mean latency</td><td>p95 latency</td><td>Peak memory</td></tr><tr><td rowspan="6">PointRCNN</td><td>Original</td><td>47.96/48.88</td><td>49.27/50.79</td><td>408.1/407.7</td><td>669.30/670.48</td><td>677.65/695.51</td><td>633.3/633.3</td></tr><tr><td>Retrain</td><td>46.92/46.91</td><td>48.71/48.74</td><td>408.1/407.7</td><td>669.17/670.52</td><td>677.04/696.29</td><td>633.3/633.3</td></tr><tr><td>GMP</td><td>47.03/47.65</td><td>48.74/49.97</td><td>440.4/440.4</td><td>669.92/670.40</td><td>677.27/687.38</td><td>666.0/666.0</td></tr><tr><td>LTH/IMP</td><td>47.50/47.59</td><td>50.03/49.77</td><td>440.4/440.4</td><td>669.80/670.23</td><td>677.86/678.99</td><td>666.0/666.0</td></tr><tr><td>Net. Slim.</td><td>45.55/45.82</td><td>47.56/48.34</td><td>331.4/331.4</td><td>666.93/666.26</td><td>675.05/673.82</td><td>468.8/468.8</td></tr><tr><td>Ours</td><td>38.46/38.46</td><td>40.52/40.42</td><td>188.2/188.2</td><td>499.99/499.74</td><td>507.20/506.53</td><td>293.5/293.5</td></tr><tr><td rowspan="6">SECOND</td><td>Original</td><td>12.74/12.65</td><td>13.38/12.94</td><td>292.1/297.3</td><td>16.52/16.73</td><td>19.77/20.31</td><td>355.1/367.8</td></tr><tr><td>Retrain</td><td>12.73/12.75</td><td>13.09/13.06</td><td>292.1/297.3</td><td>16.53/16.44</td><td>20.18/19.92</td><td>355.1/367.8</td></tr><tr><td>GMP</td><td>13.32/13.31</td><td>13.65/13.76</td><td>332.4/339.9</td><td>16.97/17.12</td><td>20.86/21.15</td><td>396.4/410.9</td></tr><tr><td>LTH/IMP</td><td>13.10/13.17</td><td>14.28/14.36</td><td>327.0/334.8</td><td>16.74/16.72</td><td>20.55/21.04</td><td>391.0/405.5</td></tr><tr><td>Net. Slim.</td><td>12.26/12.09</td><td>12.76/12.58</td><td>230.2/238.1</td><td>16.15/15.99</td><td>19.80/19.66</td><td>287.9/302.9</td></tr><tr><td>Ours</td><td>11.75/11.73</td><td>12.17/12.21</td><td>226.3/232.4</td><td>16.09/15.76</td><td>19.84/19.42</td><td>274.4/287.8</td></tr></table>

Each entry is clean/corrupted. Mean and p95 are milliseconds; memory is peak allocated GPU memory in MB.

Ours also achieves lower mean latency and peak allocated GPU memory than the evaluated baselines. On clean KITTI inputs, PointRCNN Retrain uses 46.92 ms and 408.1 MB, remaining close to Original. Network Slimming uses 45.55 ms and 331.4 MB; Ours reduces these values by 15.6% and 43.2%, respectively, while improving Moderate AP under corruption by 7.57 points (RQ1). GMP and LTH/IMP retain similar latencies of 47.03 and 47.50 ms and increase peak allocated GPU memory to 440.4 MB. These results are consistent with reducing the dimensions and sampling workload of the executed network: in the evaluated implementations, Ours achieves larger resource savings than GMP and LTH/IMP. The comparisons concern the reported configurations, not methods matched to a common resource budget.

The mean-latency reductions under corrupted inputs are 21.3% and 25.5% for PointRCNN on KITTI and ONCE, and 7.3% and 5.8% for SECOND. Relative to Original, the reported p95 latency is lower in seven of the eight evaluated settings. The only exception is SECOND on clean ONCE inputs, where p95 latency increases marginally from 19.77 to 19.84 ms, a difference of 0.07 ms. Overall, the results show lower p95 latency across the evaluated settings. These measurements demonstrate resource savings on GPU, and indicate the framework’s potential to facilitate real-time detection on resource-constrained onboard hardware.

## 4.4 RQ3: Component Contributions

We evaluate component contributions on KITTI using PointR-CNN, with clean AP and AP under corruption as the evaluation metrics. The training-order comparison uses the full-size detector and the 1:1 clean–corrupted mixture for Car detection. The component ablation study compares the full resource-efficient detector with variants without ensemble reference, data prioritization, or density-guided sampling across Car, Pedestrian, and Cyclist. We report Moderate AP in the text and all three difficulty levels in the tables.

Table 4 shows that perturbation-sensitivity ordering achieves the highest AP across all six clean and corrupted results. Compared with random order, clean Moderate AP increases from 79.59 to 82.12, and Moderate AP under corruption from 64.86 to 67.12, gains of 2.53 and 2.26 points. The corresponding gains over confidence-based ordering are 1.46 and 0.54 points. In the reported run, low-to-high perturbation-sensitivity ordering achieves higher AP than the two alternative ordering strategies. This pattern supports the use of corruption-induced prediction changes to determine the training order and are consistent with the intended progression from easier to more challenging samples.

TABLE 4  
Effect of Training Order on KITTI Car with Full-Size PointRCNN (AP<sub>R40</sub>, IoU 0.7)
<table><tr><td></td><td colspan="3">Clean</td><td colspan="3">Corrupted</td></tr><tr><td>Training order</td><td>E</td><td>M</td><td>H</td><td>E</td><td>M</td><td>H</td></tr><tr><td>Random order</td><td></td><td>88.05 79.59</td><td>77.06</td><td>82.07</td><td>64.86</td><td>61.98</td></tr><tr><td>Confidence-based</td><td></td><td>89.32 80.66 77.96 81.86 66.5862.20</td><td></td><td></td><td></td><td></td></tr><tr><td>Perturbation-sensitivity</td><td></td><td>91.27 82.12 79.50 84.42 67.12 62.52</td><td></td><td></td><td></td><td></td></tr></table>

TABLE 5

Component Ablation on KITTI with PointRCNN
<table><tr><td rowspan="2">Variant</td><td colspan="3">Clean</td><td colspan="3">Corrupted</td></tr><tr><td>E</td><td>M</td><td>H</td><td>E</td><td>M</td><td>H</td></tr><tr><td colspan="7">Car</td></tr><tr><td>Full pipeline</td><td></td><td>89.56 80.04 77.53 81.71 64.35 60.01</td><td></td><td></td><td></td><td></td></tr><tr><td>No ensemble reference</td><td></td><td>87.49 78.89 74.91 80.13 63.20 57.28</td><td></td><td></td><td></td><td></td></tr><tr><td>No data prioritization</td><td></td><td>87.38 77.22 76.55 78.77 63.20 58.93</td><td></td><td></td><td></td><td></td></tr><tr><td>No density-guided sampling</td><td></td><td>86.71 76.95 75.25 80.45 62.88 55.33</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7">Pedestrian</td></tr><tr><td>Full pipeline</td><td></td><td>64.51 57.36 48.96 53.74 46.26 39.72</td><td></td><td></td><td></td><td></td></tr><tr><td>No ensemble reference</td><td></td><td></td><td></td><td></td><td></td><td>53.58 45.65 39.46 44.93 37.38 32.74</td></tr><tr><td>No data prioritization</td><td></td><td></td><td></td><td></td><td></td><td>56.48 48.18 42.61 47.10 39.22 34.57</td></tr><tr><td>No density-guided sampling</td><td></td><td></td><td></td><td></td><td></td><td>63.32 54.91 48.14 52.24 40.97 36.81</td></tr><tr><td colspan="7">Cyclist</td></tr><tr><td>Full pipeline</td><td></td><td>89.01 69.50 65.84 76.07 52.81 48.88</td><td></td><td></td><td></td><td></td></tr><tr><td>No ensemble reference</td><td></td><td>85.58 66.83 62.4072.5748.3045.55</td><td></td><td></td><td></td><td></td></tr><tr><td>No data prioritization</td><td></td><td>85.37 67.28 63.81 73.55 49.97 47.11</td><td></td><td></td><td></td><td></td></tr><tr><td>No density-guided sampling</td><td></td><td>86.43 67.87 63.41 71.40 48.14 45.75</td><td></td><td></td><td></td><td></td></tr></table>

AP<sub>R40</sub> on clean data and under corruption. Results compare the full pipeline with the reported component variants.

Table 5 shows lower Moderate AP under corruption for the No ensemble reference variant than for the full pipeline, by 1.15, 8.88, and 4.51 points for Car, Pedestrian, and Cyclist, respectively. The No data prioritization variant loses 1.15, 7.04, and 2.84 points. Both variants also have lower clean Moderate AP in all three classes. These results support the roles of the ensemble reference and data prioritization in robust training, with the largest observed differences for Pedestrian. The ensemble reference is intended to reduce dependence on any single detector when ordering samples, although these results do not establish why the gains vary by class.

The No density-guided sampling variant achieves 62.88, 40.97, and 48.14 Moderate AP under corruption for Car, Pedestrian, and Cyclist, respectively, compared with 64.35, 46.26, and 52.81 for the full pipeline. Relative to this variant, the full pipeline improves Moderate AP by 1.47, 5.29, and 4.67 points under corruption and by 3.09, 2.45, and 1.63 points on clean data, respectively. These results are consistent with the intended role of density-guided allocation in retaining useful geometric information during sampling reduction. The improvements across all three classes under both clean and corrupted conditions provide empirical support for incorporating density-guided sampling-budget allocation into the resource-efficient detector construction process.

## 5 DISCUSSION AND LIMITATIONS

Our results suggest that corruption robustness and deployment efficiency can improve together through offline detector provisioning. Perturbation-sensitivity-based data prioritization guides robust training, while channel-width reduction and data-informed budget allocation target reduce inference workload. However, clean AP does not improve uniformly across detectors and classes. Deployment decisions should therefore consider clean AP and AP under corruption alongside mean inference latency and peak allocated GPU memory, rather than rely on any one metric.

Offline construction adds the cost of ensemble inference, sensitivity scoring, and density analysis to detector training. These operations are performed before deployment and do not enter the inference path of the constructed detector. This separates the cost of preparing training data and determining network widths and architecture-specific sampling budgets from the cost of processing each incoming frame. We have not quantified the total offline cost, so the reported inference savings alone do not quantify the overall computational savings across training and deployment.

The evaluation covers two datasets, two detector architectures, six synthetic corruption types at one severity level, and one desktop GPU. Generalization to real adverse weather, other architectures, and onboard hardware remains untested. Accuracy results are averaged over multiple training runs. The baselines are not matched to a common resource budget; small AP differences and the effects of individual reduction mechanisms therefore require further validation. AP measures detection quality rather than downstream driving safety, and peak allocated GPU memory excludes memory outside the measured PyTorch allocations. Matchedbudget comparisons, and evaluation under real corruptions on onboard hardware would strengthen the evidence.

## 6 CONCLUSION

We studied the offline provisioning of robust and resourceefficient LiDAR perception services for autonomous driving, where a provider constructs a fixed detector before deployment so that no adaptation is needed while serving inference requests. The proposed framework orders the complete clean and corrupted training set from least to most perturbation-sensitive, estimates sensitivity using an ensemble of seven reference detectors, and reduces inference cost by narrowing channel widths and allocating architecturespecific sampling budgets based on offline data statistics.

Across KITTI and ONCE, two detector architectures, and three object classes, the provisioned detectors improve AP under corruption by 4.03 points on average over the original detectors while reducing mean inference latency and peak allocated GPU memory by 14.4% and 38.0%, with AP under corruption improving in eleven of the twelve evaluated combinations and configuration-dependent trade-offs in clean-data AP. For Car detection on KITTI with PointRCNN, for example, AP under corruption increases by 3.32 points while mean latency and peak memory decrease by 19.8% and 53.9%. These results show that corruption robustness and deployment efficiency need not be traded against each other when both are addressed in one offline construction process. Future work includes comparing against baselines under matched resource budgets, measuring latency and memory on onboard hardware, and reporting per-corruption results over multiple training seeds.

## DATA AVAILABILITY

KITTI and ONCE are publicly available, and we generate corruptions as in [30]. Our code is available at https://github. com/drunklibai/robust-efficient-lidar-perception.

## ACKNOWLEDGEMENT

We thank the editors and anonymous reviewers for their time and comments. This research received partial support from the National Natural Science Foundation of China (Grants No. U24A20337, 62372228), Shenzhen-Hong Kong-Macau Technology Research Programme (Type C) (Grant No. SGDX20230821091559018). The opinions, findings, conclusions, or recommendations expressed in this work are solely those of the author(s) and do not necessarily reflect the views of the National Research Foundation.

## REFERENCES

[1] S. Shi, X. Wang, and H. Li, “PointRCNN: 3D object proposal generation and detection from point cloud,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 770–779.

[2] Y. Yan, Y. Mao, and B. Li, “SECOND: Sparsely embedded convolutional detection,” Sensors, vol. 18, no. 10, p. 3337, 2018.

[3] S. Shi, C. Guo, L. Jiang, Z. Wang, J. Shi, X. Wang, and H. Li, “PV-RCNN: Point-voxel feature set abstraction for 3D object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 10 529–10 538.

[4] W. Jiang, H. Han, D. Feng, L. Qian, Q. Wang, and X.-G. Xia, “Energy-efficient and accuracy-aware DNN inference with IoT device-edge collaboration,” IEEE Transactions on Services Computing, vol. 18, no. 2, pp. 784–797, March/April 2025.

[5] L. Kong, Y. Liu, X. Li, R. Chen, W. Zhang, J. Ren, L. Pan, K. Chen, and Z. Liu, “Robo3D: Towards robust and reliable 3D perception against corruptions,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 19 994–20 006.

[6] Y. Dong, C. Kang, J. Zhang, Z. Zhu, Y. Wang, X. Yang, H. Su, X. Wei, and J. Zhu, “Benchmarking robustness

of 3D object detection to common corruptions,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 1022–1032.

[7] A. Lehner, S. Gasperini, A. Marcos-Ramiro, M. Schmidt, M.-A. N. Mahani, N. Navab, B. Busam, and F. Tombari, “3D-VField: Adversarial augmentation of point clouds for domain generalization in 3D object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 17 295–17 304.

[8] Z. Li, J. Guo, T. Cao, B. Liu, and W. Yang, “GPA-3D: Geometry-aware prototype alignment for unsupervised domain adaptive 3D object detection from point clouds,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 8272–8282.

[9] S. Han, J. Pool, J. Tran, and W. J. Dally, “Learning both weights and connections for efficient neural networks,” Advances in Neural Information Processing Systems (NeurIPS), vol. 28, 2015.

[10] Z. Liu, J. Li, Z. Shen, G. Huang, S. Yan, and C. Zhang, “Learning efficient convolutional networks through network slimming,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2736– 2744.

[11] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[12] Z. Yang, Y. Sun, S. Liu, and J. Jia, “3DSSD: Point-based 3D single stage object detector,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 11 040–11 048.

[13] Y. Zhang, Q. Hu, G. Xu, Y. Ma, J. Wan, and Y. Guo, “Not all points are equal: Learning highly efficient pointbased detectors for 3D lidar point clouds,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 18 953–18 962.

[14] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? The KITTI vision benchmark suite,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2012, pp. 3354–3361.

[15] J. Mao, M. Niu, C. Jiang, H. Liang, J. Chen, X. Liang, Y. Li, C. Ye, W. Zhang, Z. Li, J. Yu, H. Xu, and C. Xu, “One million scenes for autonomous driving: ONCE dataset,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, vol. 1, 2021.

[16] M. Hahner, C. Sakaridis, D. Dai, and L. Van Gool, “Fog simulation on real LiDAR point clouds for 3D object detection in adverse weather,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 15 283–15 292.

[17] M. Hahner, C. Sakaridis, M. Bijelic, F. Heide, F. Yu, D. Dai, and L. Van Gool, “LiDAR snowfall simulation for robust 3D object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 16 364–16 374.

[18] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum learning,” in Proceedings of the 26th International Conference on Machine Learning (ICML), 2009, pp. 41–48.

[19] M. P. Kumar, B. Packer, and D. Koller, “Self-paced learning for latent variable models,” in Advances in

Neural Information Processing Systems (NeurIPS), vol. 23, 2010.

[20] Z. Zhu, Q. Meng, X. Wang, K. Wang, L. Yan, and J. Yang, “Curricular object manipulation in LiDAR-based object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 1125–1135.

[21] O. Sener and S. Savarese, “Active learning for convolutional neural networks: A core-set approach,” in International Conference on Learning Representations (ICLR), 2018.

[22] A. Shrivastava, A. Gupta, and R. Girshick, “Training region-based object detectors with online hard example mining,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 761–769.

[23] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom, “PointPillars: Fast encoders for object detection from point clouds,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 12 697–12 705.

[24] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “PointNet++: Deep hierarchical feature learning on point sets in a metric space,” in Advances in Neural Information Processing

Systems (NeurIPS), vol. 30, 2017.

[25] G. Fang, X. Ma, M. Song, M. B. Mi, and X. Wang, “Dep-Graph: Towards any structural pruning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 16 091–16 101.

[26] L. Zhang, R. Dong, H.-S. Tai, and K. Ma, “PointDistiller: Structured knowledge distillation towards efficient and compact 3D detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 21 791–21 801.

[27] J. Li, S. Luo, Z. Zhu, H. Dai, A. S. Krylov, Y. Ding, and L. Shao, “3D IoU-Net: IoU guided 3D object detector for point clouds,” arXiv preprint arXiv:2004.04962, 2020.

[28] S. Shi, Z. Wang, J. Shi, X. Wang, and H. Li, “From points to parts: 3D object detection from point cloud with partaware and part-aggregation network,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 8, pp. 2647–2664, 2020.

[29] O. D. Team, “OpenPCDet: An open-source toolbox for 3D object detection from point clouds,” https://github. com/open-mmlab/OpenPCDet, 2020.

[30] Castiel-Lee, “Robustness of 3D point cloud object detection,” https://github.com/Castiel-Lee/robustness pc detector, 2023.