# VPR-Space: A Benchmark for Visual Place Recognition in Space Analog Domains

Richard Fechner<sup>1</sup>, Peer Neubert<sup>1</sup>

Abstract— Visual Place Recognition (VPR) is an important means for loop closure detection in visual Simultaneous Localization and Mapping (SLAM). While state-of-the-art VPR and SLAM approaches show impressive results in urban and feature-rich natural environments, the vast, repetitive, and feature-poor environments of extraterrestrial surfaces like on the Moon or Mars, together with their challenging illumination conditions, pose severe challenges for existing approaches. This paper presents a novel benchmark, VPR-Space, to evaluate VPR capabilities on extraterrestrial surfaces. There are three main contributions: (1) We collected and curated sixteen VPR database-query datasets based on existing data from Space Analog Domains (SAD), terrestrial environments that resemble properties of the Moon or Mars. The collection includes a wide range of challenges regarding environment, viewpoint change, and illumination conditions. All datasets were manually corrected for ground-truth camera pose errors and unwanted camera artfacts. (2) To address the ill-posed nature of VPR, we propose a novel approach to generate the ground-truth information in VPR benchmarks. Instead of defining that a query and datbase image show the same place if and only if their camera poses are within fixed position and viewing direction thresholds, we propose to make the decision dependent on whether the match would be valuable in the downstream SLAM task. To avoid the overhead and restrictions of evaluating VPR in-the-loop within SLAM systems, we provide VPR ground-truth (GT) matrices based on the performance of an ensemble of state-of-the-art Relative Pose Estimators (RPE). Under this GT we require a VPR method to identify exactly those image pairs for which a subsequently applied RPE is expected to provide a valuable pose estimate in a SLAM context. (3) We evaluated nine state-of-the-art VPR approaches. The results show considerable room for improvement and indicate directions for future research. Detailed results, data, and code will be made available upon publication of the paper. We provide a pre-release version of the dataset on github: https://github.com/rfechner/VPR-Space

## I. INTRODUCTION

Reliable camera-based navigation is an important capability of autonomous mobile systems in GNSS-denied environments such as the surfaces of the Moon and Mars. The potential of, and challenges associated with, such navigation systems have recently been highlighted by the initial success and subsequent crash of the Mars helicopter Ingenuity. A likely cause being the inability of the visual navigation system to find features on the homogeneous surface of Mars [17]. In the context of camera-based navigation, Visual Place Recognition (VPR) addresses the problem of identifying matching image pairs between a query set and a database set of images depicting the same places. It is an important component of loop-closure detection in Simultaneous Localization and Mapping (SLAM) where it enables the correction of accumulated errors of odometry-based trajectory estimates. Existing SLAM and VPR approaches achieve impressive performance in urban and feature-rich natural environments like forests. However, the visual characteristics of extraterrestrial planetary surfaces, such as those of the Moon or Mars, differ fundamentally from these environments. They more closely resemble Earth’s stone and sand deserts or volcanic sites, exhibiting regolith and topographic structures such as craters and wrinkle ridges. The vast, often homogeneous and feature-poor landscapes, together with harsh illumination conditions can pose severe challenges for image descriptors. This is illustrated in Fig. 1, which shows the distributions of descriptor similarities for image pairs depicting either the same place (orange) or different places (blue). The two distributions are even more difficult to separate in the challenging Moon-like environment (right) than in the challenging urban environment (left). Combined with the low error tolerance of space missions, this motivates the need for careful analysis and evaluation of visual navigation algorithms under planetary-like conditions.

![](images/99e2da2a6baa6c99e5792e49a0c49037649410e299ff896c9eda0d8908474f88.jpg)

![](images/3cfc10f7429d8a2d6b41ba9b2e6974e34d64b829afc7077311e78cf1ad0e6062.jpg)  
Fig. 1: Image descriptor similarity histograms for Oxford RobotCar day–night [25] (left) and SPICE-HL³ noon–night [28] (right). We used Bag-of-Queries [2], a SotA VPR method, to compute the descriptors. The distributions in the space analog domain are even less separable than in this challenging urban domain (i.e., there is no similarity threshold which approximately separates the positive and negative matchings).

In this paper, we present VPR-Space, a benchmark for VPR approaches that provides (1) challenging combinations of curated database and query image sets with planetarylike conditions, and (2) carefully created ground-truth information about which image pairs depict the same place. Both parts are non trivial. Collecting large amounts of extraterrestrial on-site data is impractical. An established alternative for preparation of space missions is using Space Analog Domains (SADs), areas on the Earth that resemble properties of the Moon or Mars [27]. Table I lists available SAD-based datasets for the evaluation of SLAM systems. Sec. III will describe how we selected, manually curated, and preprocessed the available data in order to create representative database and query sets for the VPR-Space benchmark.

![](images/f118a3cc4aeec23a166c855e88356c4ddefe3c56d7f6809c9fe53cfa5964de5a.jpg)  
Fig. 2: Two images of the MADMAX A2–A4 dataset, captured 20m apart. Constructing the VPR-GT by pose thresholding or frustum overlap fails to capture this match, missing out on a potential match. RPEs however are able to estimate a low error relative pose change, making this a desired VPR match.

For the second part, creating the ground-truth (GT) information for evaluation of VPR approaches, a major challenge is the ill-posed nature of the VPR problem: There is no unique correct answer whether two images show the same place or not [29]. As will be discussed in Sec. II-C, the usual approach to create GT in existing VPR datasets is based on the ground truth pose difference between the two images (Pose thresholding) or overlap in field of view with a limited maximum distance (Frustum overlap). In this paper, we propose a different approach to VPR-GT generation: According to [10], the quality of a VPR result is driven by its value for the downstream task. To avoid the overhead and restrictions of evaluating VPR in-the-loop within a SLAM system, we propose to use the performance of an ensemble of Relative Pose Estimators (RPE) as an oracle that identifies valuable matches (i.e., the RPE quality is used to generate the GT matrices). This can address two problems of VPR-GT generation in large and open environments, and in environments with very harsh illumination conditions, both common properties of SADs:

• Problem 1: Fig. 2 illustrates how two images that can be considered to show the same place but which are taken far apart from one another, do not match when deciding correspondence based on pose distance or frustum overlap. The proposed RPE-GT includes this match if and only if we expect RPE approaches to find a valuable pose transformation.

• Problem 2: Fig. 3 illustrates an example image pair that is taken from the exact same camera pose but due to extreme lighting conditions, a low error relative pose estimate cannot be made. A GT based on pose distance or frustum overlap would include this match, although it would potentially introduce additional errors in a SLAM system.

Problem 2: Match may lead to wrong pose estimate  
![](images/88b3a5e29b1dbe55378ddf610e84abdf7dfe7e7babb02c450258ac751c823f8f.jpg)  
Fig. 3: Two images of the SPICE-HL³ SPAD dawn–night dataset, captured at the same location but with different lighting conditions. Based on pose thresholding or frustum overlap these images should match. Employing a set of SotA RPEs, the minimum rotation error is $> 3 1 ^ { \circ }$ ; matching both images is unlikely to result in a correct pose estimate. We argue to label this pair a non-match.

Sec. III-C will provide details on this RPE-based ground truth creation. Sec. IV will then evaluate state-of-the-art (SotA) VPR approaches on the proposed benchmark. In summary, the main contributions of this paper are:

1) VPR-Space, a challenging VPR benchmark for space analog domains, with 16 database-query sequence pairs derived from existing space analog SLAM datasets.

2) We propose and discuss a novel method to construct an application-driven ground truth for VPR. It leverages a set of SotA relative pose estimators to decide whether two images should match.

3) We evaluate nine SotA VPR methods on VPR-Space.

The benchmark with detailed results, evaluation code, as well as the code to reconstruct the benchmark from the raw SLAM datasets will be made available on the benchmark website upon paper publication.

## II. RELATED WORK

## A. VPR Datasets and Benchmarks

VPR is an intensively studied problem and the literature discusses typical challenges such as viewpoint changes (observing places from different camera poses) and appearance changes (e.g., due to varying illumination, weather, or season) [29], [10], [30]. To successfully perform VPR, extracted features must 1) be present in the environment, and 2) be processable (i.e., observable, extractable, describable, and matchable). Existing VPR datasets cover a broad spectrum of environments and visual dynamics. Datasets like Oxford RobotCar [25] test VPR in urban domains, capturing the streets of Oxford in different lighting and environmental conditions. The Tokyo 24/7 [34] dataset shows views of the streets of Tokyo at day and night, additionally introducing the challenge of changing environments. Semistructured VPR datasets include Nordland [32], a railway trip through Norway under changing seasons, or WildCross [20], a recent dataset addressing the gap of cross-modal VPR in natural domains. To evaluate general performance, VPR benchmarks aggregate these distinct domains. For example, VPR-Bench [40] combines a mix of indoor and outdoor (urban and semi-structured) VPR datasets to provide a balanced assessment across varied scenarios. However, existing benchmarks predominantly reflect terrestrial features (e.g., roads, vegetation, and man-made structures).

## B. SLAM Space Datasets

Although, to the best of our knowledge, no dedicated SAD VPR datasets are available, Table I provides properties of existing SLAM Space datasets that will be used for VPR benchmark construction in Sec. III. We want to highlight four particularly relevant datasets. The MADMAX dataset [26] captures 36 trajectories on 8 different sites near the city of Erfoud, Morocco. The collected trajectories depict dunes, coarse stone desert and traverses along and over rockstructures. The S3LI Etna dataset [13] was collected around the crest and in close proximity of Mt. Etna, Italy. The sequences collected along the crater ridge depict grayscale imagery of volcanic rock and ash, a very monotonous landscape. The LunarLoc dataset [33] publishes 17 sequences along a 27m by 27m simulated lunar landscape, consisting of images collected with a digital twin of the NASA IPEx rover. The recently published SPICE-HL³ dataset [28] was collected at the LunarLab in Luxembourg. It includes a particularly relevant traverse that follows a stop-and-capture routine with several sequences of the same fixed trajectory, each sequence providing differently illuminated images.

## C. Approaches to Construct a VPR-GT

Constructing the VPR-GT requires the critical decision of which images should match in the first place, a topic discussed at length by Schubert et al. [29]. Adjacently, Garg et al. [10] name “drivers” as the motivators for forming the VPR-GT. The established VPR benchmarks usually construct their VPR ground truths based on the distance in poses between two images or match individual images based on frame index. For example, the Tokyo 24/7 dataset [34] decides correspondence based on GPS distance of collected images; a found database image is a positive if it was captured at most 25m away from the query image. The Nordland dataset [32] matches image pairs based on their frame index, as database and query pairs are collected along the same trajectory and time-synchronized. A recent work on benthic VPR [23] uses an evaluation that is closer to the downstream task. The approach estimates 3D footprints on the seafloor and determines a top-down feature overlap to decide whether two images correspond to the same VPR-place. In a similar direction, the authors of the S3LI Vulcano dataset [12] mention the possibility to calculate VPR ground truths from point-cloud overlaps, when LiDAR or precise depth maps are available. They describe a VPR-GT construction built on shared visible information in the frustum but leave the construction of a decision rule to future work.

TABLE I: Sensor modalities relevant to VPR and attributes of SLAM datasets in space domains. The properties Viewpoint Change (VC) and Illumination Change (IC) denote the existence of hard viewpoint- and illumination changes within and between two sequences of the dataset respectively.
<table><tr><td>Dataset</td><td>Gracale</td><td>RGB</td><td>LIAR</td><td>IMUyro</td><td>GNSSPOS</td><td>Ha c</td><td>Har ric</td><td>Mas alog</td><td>Moln alog</td></tr><tr><td>Devon Island [9]</td><td></td><td>√√√×××√×√×××√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MADMAX [26]</td><td></td><td></td><td>X×√√√×××√×××√</td><td>XノXノノノノノノノノXノ</td><td>ノノノノノノノノノノノXノ</td><td>ノノノノノノノノノノ</td><td>××××××××××&gt;√√</td><td>ノノノノノノノノノXXX</td><td>×××××√×××&gt;√&gt;</td></tr><tr><td>S3LI Vulcano [12]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>S3LI Etna [13]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Erfoud [21]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ROBEX [35]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BASEPROD [11]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ENAV [22]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Katwijk [15]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LunarLoc [33]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SPICE-HL³ [28]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>POLAR Traverse [14]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TAILPlus [37]</td><td>X××√&gt;&gt;X&gt;X&gt;&gt;√X</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## III. BENCHMARK CONSTRUCTION

A VPR benchmark for SADs should include a diverse set of challenging scenarios and environments which can be used to gain a holistic understanding of the capabilities of a VPR method in SADs. We see four major design requirements for such a benchmark: (1) It should resemble relevant and established VPR challenges like Viewpoint Change (VC) and Illumination Change (IC) [30]. (2) It should include a diverse set of space analog environments and topologies, representing both, Mars- as well as Moon-analog domains. (3) It should have an application relevant VPR-GT which addresses the specific challenges of the domain. (4) It should challenge the SotA for VPR. In the following, we will show how VPR-Space meets these requirements.

## A. Choosing Database and Query Sequences

From the available datasets in Table I, we select 16 sequence pairs in total. Trajectory shapes and example images can be seen in Figure 4. We identify ten pairs of diverse and challenging database and query sequences falling into the VC category – six from MADMAX and two from S3LI and LunarLoc respectively. These pairs contain changing viewpoints of the same place. They are distinct sequences collected at the same site with crossover points and intervals. For the VPR challenge of IC we chose six pairs of sequences from SPICE-HL³’s A trajectory. The sequences follow a fixed trajectory but differ in lighting, camera sensor and exposure time. For the two cameras, one monochrome and one Single Photon Avalanche Diode (SPAD), we chose three main setups: dawn–night, noon–night and dawn–noon. We apply a sub-sampling routine to achieve an even distribution of data points across the trajectories. Details can be found in the Appendix. Overall, this selection satisfies our design requirements (1) and (2), as it consists of 10 VC and 6 IC setups, balances Mars- and Moon-analogs, has different trajectory topologies and geological environments.

DB: SPICE-HL³ SPAD dawn Q: SPICE-HL³ SPAD night  
![](images/37c360d05b24922c3143cc4db7ca2ab347ac6b51551a9156c75fc46a0bbe0234.jpg)  
DB: MADMAX A0 Q: MADMAX A2

![](images/ce0eb784606c46807fa46ab511a661d19b6bda728dbe9c4e9b500a6d86f683a4.jpg)  
DB: MADMAX C0 Q: MADMAX C1  
DB: MADMAX A2 Q: MADMAX A4

![](images/7982d273de548b97774578d6f8ddc59b078aa049f7078efc636022e26c469e6a.jpg)

![](images/a5f6826c3f9173596bc9acdbb9c9cea039871b920236f7ca9b3f9ca270a85899.jpg)  
DB: MADMAX E0 Q: MADMAX E1

![](images/d9dafe2a2f5598de553763498974eca75fce64d8ab125f0e2dbb5add10dad2ac.jpg)

![](images/5d4820ca922c4bd69ef484b961c822e408f7101cca34c57983cdee80d8760032.jpg)  
DB: MADMAX F0 Q: MADMAX F3

![](images/dd65ba6403e63007ebeb3fddd83888ad595373a0c495d11c64f502213c77c2ea.jpg)

![](images/f7046965843ea19871a5894156fdd2bd8c0fd94cd1d9e51c3c7059e48862b910.jpg)  
DB: MADMAX G0 Q: MADMAX G2  
DB: S3LI Etna crater Q: S3LI Etna crater inout

![](images/ab0a492d32582eb09278e190ca1217503dc1ebf5f59c22b17a39dfb0f66369e3.jpg)

![](images/2c308955aee79ca56a7d18f94e70a7e1af38cdcb471af5ab8a0b0a226897b4e6.jpg)

![](images/3e610b8557a7c06715bbb7d0332b97b81508ed08bbf106c496a77aded4dd5b88.jpg)

![](images/364162faa8a67cec641c55f73f89a635d6e71e5f4829250ac4a7c1903f07069e.jpg)  
DB: S3LI Etna traverse 1 Q: S3LI Etna loops

![](images/27389d0a31c3884606a727387e22c0751380cd9ee976eef4152163cf6bc5a929.jpg)

![](images/24086b01dd64cdc6efe4b877bb008b108f82e6b7c23f773d3224fc6bc93a483b.jpg)  
DB: LunarLoc 9 Q: LunarLoc 10

![](images/0f9500ded23980fbedf973e2df22049329028016a09eba181e1f543a920d3fb7.jpg)  
DB: LunarLoc 15 Q: LunarLoc 17

![](images/4f01e24193b208bef10e020de5145d2aee26f76abff64a001bb99dd32e61fa53.jpg)

![](images/51e475b0e19e7688d6aaf0806fed4b7e8ae708a0017fd7987ce66dc3ba925cd7.jpg)  
DB: SPICE-HL³ dawn Q: SPICE-HL³ night

![](images/ec0c17cb73f25644e989f1af677d31615b07adeff13808f94c956ec754696638.jpg)  
DB: SPICE-HL³ SPAD dawn Q: SPICE-HL³ SPAD noon

![](images/cc5ab715de6b6a86c717db3e19e2f83d54ef24b60f8fcecbb1bd9d2529e2f1b1.jpg)

![](images/c4e2b73d6e81b3e418da3b2344e51ecc40250fc969fe2d987557bdda6f5d0651.jpg)

![](images/9cc9f917b5e67b7eee684d075bfeeae98bd8b2689dc1883c1230c375cc5530ea.jpg)

![](images/82a8ff5997b3df94605123f85e38f62c89627708d3b44b39f7bf10b78088d303.jpg)  
DB: SPICE-HL³ dawn Q: SPICE-HL³ noon  
DB: SPICE-HL³ SPAD noon Q: SPICE-HL³ SPAD night

![](images/fd65939803dd7b341b18ab696ce59b254950a0f3fe980b56df058c6f7fb0d85d.jpg)

![](images/ad3ce8305acf134324d51bab3ada46ea77a97646a20cb28b315f54ec2dea2fc8.jpg)

![](images/5c829e2d7ac43d7385c33de256ba276f7cd46fdb9d5723503f67604ea300e5aa.jpg)

![](images/00f90340296ae24916a44e592abba60c24906586e745298f1c56d742351def60.jpg)  
DB: SPICE-HL³ noon Q: SPICE-HL³ night

![](images/dc4d075deb203d5869e4da081ee7839e9d340345f372c6a6e60b5a027d47fd8a.jpg)

![](images/91b028e6f06ed482aff46e400e44d1a33795ad8f7376d3c84a7de040c8d2de25.jpg)

![](images/16cd61a4fb46fae4dcb41776030627bc26edf56755db10ad21d6b0c4e79fc2c3.jpg)  
Fig. 4: An overview of the database (blue) and query (red) sequences of the VPR-Space benchmark. We show example matches from both sequences to illustrate the shape of the task. For SPICE-HL³, database and query sequence lie on top of each other.

## B. Data Cleaning

Our VPR-GT construction requires correct ground-truth camera poses in order to evaluate the result of an RPE. During initial data analysis, we noticed inconsistencies in the provided poses and imagery and decided to manually verify the selected sequences. In MADMAX and S3LI Etna we encountered GNSS and IMU errors, in the SPICE-HL³ sequences we observed some camera artifacts. We verified the data with three independent human annotators, using a shared labeling UI where position, orientation and associated image are visible. Any data point being (a) visually corrupted or (b) exceeding an apparent GT pose error beyond 1 meter in position or $1 0 ^ { \circ }$ in orientation was flagged. Out of about 9200 flagged data points, the three annotators agree on about 72%; at least two annotators agree on 81% of the flagged data points. We discarded every frame flagged by at least one annotator. The found errors are provided as part of the published data, so that they can be reused by others working with the raw data.

## C. Construction of the VPR Ground Truth

The VPR ground truth is usually represented by discrete sets of matching image pairs from query and database sets, e.g., in form of a ground-truth matrix [30]. We propose to create this matrix using an ensemble of available SotA Relative Pose Estimator approaches that serve as an oracle to tell whether a valuable pose estimate could be obtained from an image pair in a SLAM downstream task. The RPE ensemble is used in combination with ground-truth pose information. Instead of directly judging the ground-truth pose distance between a query and a database image, we evaluate the error between the output of the RPEs and the groundtruth pose transformation to decide whether an image pair is valuable in a SLAM context (and thus should be found by an VPR approach). We want to emphasize that, in contrast to an actual end-to-end evaluation in a SLAM system, this oracle has to be used only once for each image pair and not for each evaluated VPR approach.

We construct a majority voting scheme over multiple RPEs. We leverage off-the-shelf monocular metric-relative pose estimation pipelines that predict a 6DoF relative pose consisting of rotation and scaled translation. Where applicable, backbone architectures are provided with metric depth predictions from Depth Anything V2 [38] and refined using RANSAC [8]. Our evaluation ensemble comprises DUSt3R [36], MUSt3R [6], MapAnything [19], MADPose [39], and a SuperPoint [7] + LightGlue [24] combination.

Two images $I _ { 1 }$ and $I _ { 2 }$ become a match in the GT matrix if a sufficient number of RPEs produces a small relative pose error between the two images. An RPE $f ^ { ( i ) }$ takes two images

$$
\mathrm { G T s ; }
$$

![](images/8143bc6ab2c7e088d51c66f96f13106af96e0ccdb52e2656d0bae0b07e2dac2a.jpg)

![](images/fe9524ecc080d4e069399bedc2bd04c0c6934215bbeac01c249d5206c93c6d48.jpg)

![](images/c14be9fa4624e0a55a91e02a8b62d828b2c8cd8734d55a0f06f2a2a60fbfe5ba.jpg)  
Fig. 5: GT matrices (hard GT in dark, soft GT in light color) and Precision-Recall Curves for VPR-GTs of different constructions. To enable comparison, the number of positive matches is equalized across constructions. This example illustrates that established GT constructions tend to request harder matches. This pattern repeats across the dataset.

$I _ { 1 } , I _ { 2 } ,$ , associated depth images $D _ { 1 } , D _ { 2 }$ and the camera intrinsics matrix $K \in \mathbb { R } ^ { 3 \times 3 }$ . It outputs a relative rotation $\hat { R } ^ { ( i ) } \in S O ( 3 )$ and translation $\hat { t } ^ { ( i ) } \in \mathbb { R } ^ { 3 }$ . Since the ordering of input images can influence the results, each RPE produces two estimates: $f ^ { ( i ) } ( I _ { 1 } , I _ { 2 } , D _ { 1 } , D _ { 2 } , K ) = \hat { R } _ { 1 , 2 } ^ { ( i ) } , \hat { t } _ { 1 , 2 } ^ { ( i ) }$ and $f ^ { ( i ) } ( I _ { 2 } , I _ { 1 } , D _ { 2 } , D _ { 1 } , K ) = \hat { R } _ { 2 , 1 } ^ { ( i ) } , \hat { t } _ { 2 , 1 } ^ { ( i ) }$ From known ground truth relative rotation $R _ { \mathrm { G T } }$ and translation t<sub>GT</sub> we calculate the geodesic rotation error $\Delta R ^ { ( i ) } ~ \in ~ [ 0 , 1 8 0 )$ [°] in degrees and the euclidean distance $\Delta t ^ { ( i ) } \ \in \ [ 0 , \infty )$ [m] for the respective input ordering. In all experiments we relax $\Delta t ^ { ( i ) }$ to ∆t<sup>(i)</sup> := min $\| s \cdot \hat { t } ^ { ( i ) } - t _ { \mathrm { G T } } \| _ { 2 }$ , the minimal distance to the true translation $t _ { \mathrm { G T } }$ on the span of the estimated translation direction $\hat { t } ^ { ( i ) }$ . This reduces the required relative pose estimate to five degrees of freedom, making it easier for the RPEs to predict a correct translation. We say that RPE $f ^ { ( i ) }$ estimates the relative pose correctly if for both input orderings its corresponding estimate lands below fixed thresholds: $\hat { \Delta R _ { 1 , 2 } ^ { ( i ) } } < \tau _ { R } \hat { \land } \Delta t _ { 1 , 2 } ^ { ( i ) } < \tau _ { t }$ and $\Delta R _ { 2 , 1 } ^ { ( i ) } < \tau _ { R } \land \Delta t _ { 2 , 1 } ^ { ( i ) } < \tau _ { t }$ . We decide that the two images correspond if this condition holds for at least n RPEs. Setting $n > 1$ regularizes the RPE ensemble, reducing the risk to add image pairs where an RPE created an arbitrary transformation that is only correct by chance. Additionally, images taken more than $d _ { \mathrm { m a x } }$ meters apart are never positives as we observed that RPE errors grow too large with the distance between input images.

Building upon this majority voting mechanism, we can vary the error thresholds $\tau _ { R } , \tau _ { t }$ , and n to construct distinct soft- and hard- correspondence matrices required for a VPR evaluation. We follow the work of Schubert et al. [30] where the soft ground truth denotes matches outside the hard ground truth which should not be penalized when an algorithm retrieves them. Exact hyperparameter choices and resulting dataset statistics are provided in the Appendix. Figure 5 compares this approach to constructing the VPR-GT based on pose thresholding or frustum overlap.

TABLE II: AUPRC of SotA VPR methods on all sequence comparisons. Best per row in bold, runner-up underlined.
<table><tr><td>VPR chall. DB/Q pair</td><td></td><td>MeLo [4]</td><td>BoQ [2]</td><td>SLD [16]</td><td>AnyL [18]</td><td>DenV [34]</td></tr><tr><td rowspan="11">Viewwwoint chae lutto</td><td>MM A0 – A2</td><td>0.67</td><td>0.70</td><td>0.65</td><td>0.68</td><td>0.69</td></tr><tr><td>MM A2 – A4</td><td>0.36</td><td>0.41</td><td>0.39</td><td>0.29</td><td>0.32</td></tr><tr><td>MM C0 – C1</td><td>0.51</td><td>0.52</td><td>0.41</td><td>0.33</td><td>0.31</td></tr><tr><td>MM E0 – E1</td><td>0.33</td><td>0.30</td><td>0.23</td><td>0.26</td><td>0.30</td></tr><tr><td>MM F0 – F3</td><td>0.79</td><td>0.71</td><td>0.68</td><td>0.63</td><td>0.57</td></tr><tr><td>MM G0 – G2</td><td>0.57</td><td>0.49</td><td>0.48</td><td>0.50</td><td>0.41</td></tr><tr><td>S3E crater – crater_inout</td><td>0.52</td><td>0.33</td><td>0.32</td><td>0.18</td><td>0.35</td></tr><tr><td>S3E traverse_1 – loops</td><td>0.20</td><td>0.17</td><td>0.10</td><td>0.08</td><td>0.07</td></tr><tr><td>LL 9 - 10</td><td>0.62</td><td>0.56</td><td>0.36</td><td>0.44</td><td>0.41</td></tr><tr><td>LL 15 – 17</td><td>0.48</td><td>0.49</td><td>0.51</td><td>0.40</td><td>0.34</td></tr><tr><td rowspan="5">HL3 SPAD dawn – night HL3 SPAD dawn – noon Cchanne HL3 SPAD noon – night</td><td>0.18 0.76</td><td>0.21 0.79</td><td>0.13</td><td>0.23</td><td>0.02</td></tr><tr><td></td><td></td><td>0.69</td><td>0.66</td><td>0.80</td></tr><tr><td>0.21</td><td>0.35</td><td>0.19</td><td>0.35</td><td>0.05</td></tr><tr><td>HL3 FLIR dawn – night 0.33 HL3 FLIR dawn – noon</td><td>0.35</td><td>0.27</td><td>0.32</td><td>0.17</td></tr><tr><td>0.70 0.28</td><td>0.72</td><td>0.60</td><td>0.65</td><td>0.67</td></tr><tr><td>HL3 FLIR noon – night</td><td>0.41</td><td></td><td>0.31</td><td>0.33</td><td>0.12</td></tr><tr><td colspan="2">Worst case</td><td>0.18</td><td>0.17</td><td>0.10</td><td>0.08</td><td>0.02</td></tr><tr><td colspan="2">Avg. mean</td><td>0.47</td><td>0.47</td><td>0.39</td><td>0.40</td><td>0.35</td></tr><tr><td colspan="2">Avg. std</td><td>±0.20</td><td>±0.19</td><td>±0.19</td><td>±0.18</td><td>±0.24</td></tr></table>

TABLE III: Recall@k and recall@100% precision of SotA VPR methods. Best per row in bold, runner-up underlined

<table><tr><td>Case</td><td>Method</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@100p</td></tr><tr><td rowspan="5">Worst case</td><td>MeLo</td><td>0.24</td><td>0.46</td><td>0.53</td><td>0.00</td></tr><tr><td>BoQ</td><td>0.38</td><td>0.59</td><td>0.69</td><td>0.00</td></tr><tr><td>SLD</td><td>0.25</td><td>0.56</td><td>0.66</td><td>0.00</td></tr><tr><td>AnyL</td><td>0.31</td><td>0.54</td><td>0.64</td><td>0.001</td></tr><tr><td>DenV</td><td>0.06</td><td>0.23</td><td>0.33</td><td>0.00</td></tr><tr><td rowspan="4">土</td><td>MeLo</td><td>0.59±0.25</td><td> $0 . 7 5 { \pm } 0 . 1 9$ </td><td> $0 . 8 1 { \pm } 0 . 1 5$ </td><td> $0 . 0 7 { \scriptstyle \pm 0 . 0 7 }$ </td></tr><tr><td>BoQ</td><td>0.64±0.20</td><td> $\mathbf { 0 . 7 8 \pm 0 . 1 4 }$ </td><td> ${ \bf 0 . 8 4 \pm 0 . 1 1 }$ </td><td> ${ \bf 0 . 0 8 } \pm 0 . 0 8$ </td></tr><tr><td>SLD</td><td>0.59±0.21</td><td> $0 . 7 7 { \scriptstyle \pm 0 . 1 4 }$ </td><td> ${ \bf 0 . 8 4 } \pm 0 . 1 2$ </td><td> $0 . 0 5 { \pm } 0 . 0 7$ </td></tr><tr><td>AnyL</td><td>0.56±0.21</td><td> $0 . 7 3 { \pm } 0 . 1 6$ </td><td> $0 . 8 1 { \pm } 0 . 1 3$ </td><td> $0 . 0 4 \pm 0 . 0 4$ </td></tr><tr><td rowspan="2">AS</td><td>DenV</td><td>0.52±0.27</td><td> $0 . 6 8 { \pm } 0 . 2 4$ </td><td> $0 . 7 6 { \pm } 0 . 2 0 $ </td><td> $0 . 0 5 { \pm } 0 . 0 5$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

## IV. EVALUATION OF SOTA VPR APPROACHES

## A. Quantative Analysis

We evaluate MegaLoc [4] (MeLo), Bag-of-Queries [2] (BoQ), SALAD [16] (SLD), AnyLoc [18] with the global vocabulary (AnyL), AnyLoc with the unstructured vocabulary, MixVPR [1], CosPlace [5] as well as DenseVLAD [34] (DenV), NetVLAD [3] and DINOv3 (CLS token) [31] as baselines. We standardize with the database statistics and L2- normalize all descriptors before constructing the similarity matrices. We report the Area under the Precision-Recall Curve (AUPRC) in Table II and Recall@{1, 5, 10} as well as Recall@100p (recall at 100% precision) in Table III. The tables do not include AnyLoc with the unstructured vocabulary, MixVPR, CosPlace and DINOv3 as these methods didn’t score highest on any of the datasets; fully detailed results will be made available on the benchmark website.

The results indicate significant room for improvement for VPR methods on SAD. Averaged across all datasets, the best performing VPR approach achieves an AUPRC of 0.47 and recall@1 of 0.64 (with values of 0.18 and 0.38, respectively, for the most challenging dataset). Noticeable, in the worst case, a mode especially important for autonomous missions, R@100p is near zero across all methods.

Although being finetuned on urban data, Bag-of-Queries and MegaLoc demonstrate domain adaptation, as can be seen by their performance on the MADMAX $\mathbf { A } 0 \ - \ \mathbf { A } 2 ,$ MADMAX F0 – F3, and SPICE-HL³ dawn – noon pairs in Table II. Except for setups where outlines and shadows are quite similar, such as in SPICE-HL³’s dawn – noon pairings, current VPR methods struggle when the difference in lighting becomes too large. Notably, DenseVLAD, a non deep learning method, performs best on SPICE-HL³-SPADdawn-noon.

B. Qualitative Analysis: What are hard queries that the SotA cannot solve yet?

Three examples from the analysis of the BoQ results can be seen in Figure 6. One common pattern can be seen in Figure 6b: in SAD and other domains with large and open spaces, images along a straight sequence with a shared horizon tend to have high similarity. This is especially relevant as image pairs of this failure mode may pass the geometric verification stage and hence produce a relative pose estimate with large error.

## V. DISCUSSION

Both RPEs and VPR methods rely on visual overlap; RPEs to determine a relative pose and VPR methods to produce an embedding. They may share failure modes, which would exaggerate performance in SAD. To this end, Figure 3 can be interpreted as a possible limitation of our approach: Even though camera poses coincide, we do not require a match. However, we want to emphasize that our proposed VPR-GT is driven by the domain of interest and possible downstream application in a SLAM context. This results in a different notion of which matches should and should not be found.

Further, our VPR-GT is constructed to go beyond the capabilities of any one RPE, using a voting mechanism to increase robustness. As available RPEs improve or the expected characteristics of the downstream task change, RPE-based GT matrices may be updated. We refer to the GT used here as $^ {  } V P R \ – G T \ – R P E \ – \ O ^ { \ast }$ . If future work uses a modified GT, it should indicate this.

Although spatial verification stages and robust SLAM optimization back-ends reduce the influence of false-positive matchings [30], under the proposed GT, they gain relevance again (and so do AUPRC and R@100p metrics that explicitly incorporate false positives). Due to the GT construction approach, false positive image pairs are known to produce a large relative pose error. And due to the similar nature of RPE approaches and spatial verification stages, they are also likely to pass this verification stage.

![](images/c66d3cb0c1ea20b0f90fcc6605b686507d5582a1d56001d2dcbb038314119164.jpg)  
(a) MADMAX A0 – A2. Perceptual aliasing: The “general look” between the query (top) and best, but false positive match (bottom) is the same: a large boulder in the top half of the image, some grass below.

![](images/e40c9e6ec8681e191857d8ac98ab02f12654cebb92cae3fa4c121db269b7baec.jpg)  
(b) S3LI Etna traverse 1 – loops. Shared horizon: Best match database point is far away. The two images share the same horizon line, but an accurate relative pose estimation cannot be made – the best 5DoF pose estimate being 39m off.

![](images/6e99f425ec866f05815a1971520855ce7b353646ce218549494c4e0e1827d75f.jpg)  
(c) SPICE-HL³-SPAD dawn – night. Query and positive database image are taken in close proximity. An image with similar lighting conditions is matched instead; creating RPE translation- and rotation-errors of 4m and $8 7 ^ { \circ }$ respectively.  
Fig. 6: Examples of hard queries in VPR-Space, illustrating the challenges of space analog domains. A query image (dark blue) has highest similarity to a false positive image (orange) outside its VPR-GT. The best correct image (green) inside the hard ground truth is displayed in the center. The margin between the similarities indicates the severity of the example.

For brevity, we only evaluated holistic VPR descriptors. However, VPR-Space can also be used for evaluation of local-descriptor-based and hierarchical pipelines.

Finally, we want to make clear that the way we use RPEs for GT generation can not be used to solve the VPR problem itself, since the VPR-GT generation additionally requires GT pose information to evaluate the RPE results.

## VI. CONCLUSION

This work presented VPR-Space, a novel VPR benchmark for space analog domains. We selected and curated relevant sequence pairs from existing SLAM Space datasets to represent relevant viewpoint- and illumination-changes. To connect the VPR evaluation to the downstream SLAM task, we proposed to leverage an ensemble of SotA relative pose estimators to compute the VPR ground truth, rewarding VPR retrieval only if a low error pose estimate can be made. The evaluation of SotA VPR methods shows significant headroom for improvement and the qualitative analysis showed concrete failure modes of the VPR SotA. Our hope is that this benchmark sets new goals for VPR developments.

## APPENDIX

Sub-sampling routine: Due to different velocities and sampling rates of the sensor platforms, some sequences contain multiple images of the same place. We alleviate this issue by applying a rolling window approach for all but the SPICE-HL³ sequences which do not need this post-processing step. We briefly summarize the sub-sampling routine: The first image of the sequence is added to the result list and becomes the previously added image. Then, for every following image, we check whether the associated ∆ between the previous and current position exceeds 1m or the $\Delta$ in camera-yaw exceeds 10°. In which case we add the current image to the result list and the current image becomes the previous added image.

<table><tr><td rowspan="2">DB/Q pair</td><td colspan="2"># Images</td><td rowspan="2">% queries w/ match</td><td rowspan="2">% of pos. matches</td></tr><tr><td>DB</td><td>Q</td></tr><tr><td>MM A0 – A2</td><td>171</td><td>180</td><td>92.8</td><td>8.55</td></tr><tr><td>MM A2 – A4</td><td>180</td><td>412</td><td>20.9</td><td>0.74</td></tr><tr><td>MM C0 – C1</td><td>502</td><td>463</td><td>78.6</td><td>1.89</td></tr><tr><td>MM E0 – E1</td><td>517</td><td>507</td><td>43.2</td><td>0.62</td></tr><tr><td>MM F0 – F3</td><td>117</td><td>229</td><td>89.1</td><td>5.14</td></tr><tr><td>MM G0 – G2</td><td>258</td><td>290</td><td>51.7</td><td>1.75</td></tr><tr><td>S3E crater – crater_inout</td><td>1129</td><td>1683</td><td>18.2</td><td>0.15</td></tr><tr><td>S3E traverse_1 – loops</td><td>485</td><td>780</td><td>36.8</td><td>0.40</td></tr><tr><td>LL 9 - 10</td><td>119</td><td>183</td><td>71.0</td><td>3.19</td></tr><tr><td>LL 15 - 17</td><td>118</td><td>54</td><td>27.8</td><td>0.97</td></tr><tr><td>HL3 SPAD dawn – night</td><td>98</td><td>97</td><td>76.3</td><td>1.25</td></tr><tr><td>HL3 SPAD dawn – noon</td><td>98</td><td>97</td><td>100.0</td><td>6.00</td></tr><tr><td>HL3 SPAD noon – night</td><td>97</td><td>97</td><td>72.2</td><td>1.74</td></tr><tr><td>HL3 FLIR dawn – night</td><td>99</td><td>97</td><td>78.4</td><td>2.55</td></tr><tr><td>HL³ FLIR dawn – noon</td><td>99</td><td>99</td><td>96.0</td><td>5.93</td></tr><tr><td>HL³ FLIR noon – night</td><td>99</td><td>97</td><td>78.4</td><td>2.72</td></tr></table>

TABLE IV: Additional dataset statistics. For each database/query pair we report the number of images, the percentage of queries which have at least one match (% queries w/ match) and the percentage of positive matches (% of pos. matches) given the proposed VPR-GT-RPE-0.

## ACKNOWLEDGMENTS AND AI STATEMENT

We’d like to thank our colleagues Ida, Nick, Janine and Lena for proofreading. We’d like to thank Leon and Lorenz for helping with data annotation.

<table><tr><td rowspan="3"></td><td colspan="2">MADMAX</td><td colspan="2">S3LI Etna</td><td colspan="2">LunarLoc</td><td colspan="2">SPICE-HL3</td></tr><tr><td>soft</td><td>hard</td><td>soft</td><td>hard</td><td>soft</td><td>hard</td><td>soft</td><td>hard</td></tr><tr><td> $d _ { \mathrm { m a x } }$  [m]</td><td>35</td><td>35</td><td>35</td><td>35</td><td>35</td><td>35</td><td>1</td><td>1</td></tr><tr><td>τt [m]</td><td>4</td><td>3</td><td>4</td><td>3</td><td>4</td><td>3</td><td>0.4</td><td>0.2</td></tr><tr><td>τR [°]</td><td>30</td><td>20</td><td>30</td><td>20</td><td>30</td><td>20</td><td>20</td><td>10</td></tr><tr><td>n</td><td>2</td><td>4</td><td>2</td><td>4</td><td>2</td><td>4</td><td>1</td><td>1</td></tr></table>

TABLE V: Hyperparameters used for VPR-GT-RPE-0.

We used Claude Code to generate code to access datasets and existing approaches (RPEs as well as VPR), to support initial literature review as well as generation of figures, and to improve formal language.

## REFERENCES

[1] Amar Ali-Bey, Brahim Chaib-Draa, and Philippe Giguere. Mixvpr: Feature mixing for visual place recognition. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2997– 3006. IEEE, 2023.

[2] Amar Ali-Bey, Brahim Chaib-Draa, and Philippe Giguere. Boq: A place is worth a bag of learnable queries. In CVPR, 2024.

[3] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla, and Josef Sivic. Netvlad: Cnn architecture for weakly supervised place recognition. In CVPR, 2016.

[4] Gabriele Berton and Carlo Masone. Megaloc: One retrieval to place them all. In CVPR, 2025.

[5] Gabriele Berton, Carlo Masone, and Barbara Caputo. Rethinking visual geo-localization for large-scale applications. In CVPR, 2022.

[6] Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, and Vincent Leroy. Must3r: Multi-view network for stereo 3d reconstruction. In CVPR, 2025.

[7] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In 2018 IEEE/CVF conference on computer vision and pattern recognition workshops (CVPRW), pages 337–33712. IEEE, 2018.

[8] Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381–395, 1981.

[9] Paul Furgale, Pat Carle, John Enright, and Timothy D Barfoot. The devon island rover navigation dataset. The International Journal of Robotics Research, 31(6):707–713, 2012.

[10] Sourav Garg, Tobias Fischer, and Michael Milford. Where is your place, visual place recognition? In IJCAI, 2021.

[11] Levin Gerdes, Tim Wiese, Raul Castilla Arquillo, Laura Bielenberg,´ Martin Azkarate, Hugo Leblond, Felix Wilting, Joaqu´ın Ortega Cortes,´ Alberto Bernal, Santiago Palanco, et al. Baseprod: The bardenas semidesert planetary rover dataset. Scientific Data, 11(1):1054, 2024.

[12] Riccardo Giubilato, Marcus Gerhard Muller, Marco Sewtz, Laura Ale-¨ jandra Encinar Gonzalez, John Folkesson, and Rudolph Triebel. The s3li vulcano dataset: A dataset for multi-modal slam in unstructured planetary environments. arXiv preprint arXiv:2601.19557, 2026.

[13] Riccardo Giubilato, Wolfgang Sturzl, Armin Wedler, and Rudolph¨ Triebel. Challenges of slam in extremely unstructured environments: The dlr planetary stereo, solid-state lidar, inertial dataset. IEEE Robotics and Automation Letters, 7(4):8721–8728, 2022.

[14] Margaret Hansen, Uland Wong, and Terrence Fong. The polar traverse dataset: A dataset of stereo camera images simulating traverses across lunar polar terrain under extreme lighting conditions. arXiv preprint arXiv:2403.12194, 2024.

[15] Robert A Hewitt, Evangelos Boukas, Martin Azkarate, Marco Pagnamenta, Joshua A Marshall, Antonios Gasteratos, and Gianfranco Visentin. The katwijk beach planetary rover dataset. The International Journal of Robotics Research, 37(1):3–12, 2018.

[16] Sergio Izquierdo and Javier Civera. Optimal transport aggregation for visual place recognition. In CVPR, 2024.

[17] Jet Propulsion Laboratory. NASA Performs First Aircraft Accident Investigation on Another World. https://www.jpl.nasa.gov/news/ nasa-performs-first-aircraft-accident-investigation-on-another-world/, December 2024. Accessed: 2026-09-10.

[18] Nikhil Keetha, Avneesh Mishra, Jay Karhade, Krishna Murthy Jatavallabhula, Sebastian Scherer, Madhava Krishna, and Sourav Garg. Anyloc: Towards universal visual place recognition. IEEE Robotics and Automation Letters, 9(2):1286–1293, 2023.

[19] Nikhil Keetha, Norman Muller, Johannes Sch¨ onberger, Lorenzo Porzi,¨ Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulo, Christian Richardt, Deva Ramanan, Sebastian \` Scherer, and Peter Kontschieder. MapAnything: Universal feedforward metric 3D reconstruction. In International Conference on 3D Vision (3DV). IEEE, 2026.

[20] Joshua Knights, Joseph Reid, Kaushik Roy, David Hall, Mark Cox, and Peyman Moghadam. Wildcross: A cross-modal large scale benchmark for place recognition and metric depth estimation in natural environments. arXiv preprint arXiv:2603.01475, 2026.

[21] Simon Lacroix, Andrea De Maio, Quentin Labourey, Ellon Paiva Mendes, Pierre Narvor, Vincent Bissonette, Clement Bazerque, Fabrice´ Souvannavong, Raphael Viards, and Martin Azkarate. The erfoud¨ dataset: a comprehensive multi-camera and lidar data collection for planetary exploration. In 15th Symposium on Advanced Space Technologies in Robotics and Automation, 2020.

[22] Olivier Lamarre, Oliver Limoyo, Filip Maric, and Jonathan Kelly. The´ canadian planetary emulation terrain energy-aware rover navigation dataset. The International Journal of Robotics Research, 39(6):641– 650, 2020.

[23] Martin Kvisvik Larsen and Oscar Pizarro. Long-term visual localization in dynamic benthic environments: the sealoc dataset, footprintbased ground truth, and visual place recognition benchmark. Frontiers in Robotics and AI, 13, 2026.

[24] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Light-Glue: Local Feature Matching at Light Speed. In ICCV, 2023.

[25] Will Maddern, Geoffrey Pascoe, Chris Linegar, and Paul Newman. 1 year, 1000 km: The oxford robotcar dataset. The International Journal of Robotics Research, 36(1):3–15, 2017.

[26] Lukas Meyer, Michal Sm´ısek, Alejandro Fontan Villacampa, Lauraˇ Oliva Maza, Daniel Medina, Martin J Schuster, Florian Steidle, Mallikarjuna Vayugundla, Marcus G Muller, Bernhard Rebele, et al.¨ The madmax data set for visual-inertial rover navigation on mars. Journal of Field Robotics, 38(6):833–853, 2021.

[27] Preston, Louisa and Grady, Monica and Barber, Simeon. Tn2: The catalogue of planetary analogues. Technical note, The Planetary and Space Sciences Research Institute, 2012.

[28] David Rodr´ıguez-Mart´ınez, Dave Van Der Meer, Junlin Song, Abhishek Bera, Carlos Jesus P´ erez-del Pulgar, and Miguel Angel´ Olivares-Mendez. Spice-hl3: Single-photon, inertial, and stereo camera dataset for exploration of high-latitude lunar landscapes. Scientific Data, 13(1):374, 2026.

[29] Stefan Schubert and Peer Neubert. What makes visual place recognition easy or hard? arXiv preprint arXiv:2106.12671, 2021.

[30] Stefan Schubert, Peer Neubert, Sourav Garg, Michael Milford, and Tobias Fischer. Visual place recognition: A tutorial [tutorial]. IEEE Robotics & Automation Magazine, 31(3):139–153, 2023.

[31] Oriane Simeoni, Huy V Vo, Maximilian Seitzer, Federico Baldas-´ sarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, et al. Dinov3.¨ arXiv preprint arXiv:2508.10104, 2025.

[32] Niko Sunderhauf, Peer Neubert, and Peter Protzel. Are we there yet?¨ challenging seqslam on a 3000 km journey across all four seasons. In Proc. of ICRA Workshop on long-term autonomy, 2013.

[33] Annika Thomas, Aleksander Garbuz, Trevor Johst, Keerthana Srinivasan, Dami Thomas, Cormac O’Neill, Robaire Galliath, George Lordos, and Jonathan P How. Lunarloc: Robust global localization for autonomous surface operations on the moon. In 2026 IEEE Aerospace Conference, pages 1–11. IEEE, 2026.

[34] Akihiko Torii, Relja Arandjelovic, Josef Sivic, Masatoshi Okutomi, and Tomas Pajdla. 24/7 place recognition by view synthesis. In CVPR, pages 1808–1817, 2015.

[35] Mallikarjuna Vayugundla, Florian Steidle, Michal Smisek, Martin J Schuster, Kristin Bussmann, and Armin Wedler. Datasets of long range navigation experiments in a moon analogue environment on mount etna. In International Symposium on Robotics, 2018.

[36] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024.

[37] Zirui Wang, Chen Yao, Yangtao Ge, Guowei Shi, Ningbo Yang, Zheng Zhu, Kewei Dong, Hexiang Wei, Zhenzhong Jia, and Jing Wu. Are we ready for planetary exploration robots? the tail-plus dataset for slam in granular environments. arXiv preprint arXiv:2404.13600, 2024.

[38] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv:2406.09414, 2024.

[39] Yifan Yu, Shaohui Liu, Remi Pautrat, Marc Pollefeys, and Viktor´ Larsson. Relative pose estimation through affine corrections of monocular depth priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[40] Mubariz Zaffar, Sourav Garg, Michael Milford, Julian Kooij, David Flynn, Klaus McDonald-Maier, and Shoaib Ehsan. Vpr-bench: An open-source visual place recognition evaluation framework with quantifiable viewpoint and appearance change. International Journal of Computer Vision, 129(7):2136–2174, 2021.