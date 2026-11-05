# WayPOP: A Panoramic Open-Set Panoptic Tracking Benchmark

Rohit Mohan<sup>1,\*</sup> Maximilian Luz<sup>1,\*</sup> Swarnava Chowdhury<sup>1</sup> Yao Lu<sup>1</sup> Florian Drews<sup>2</sup> Thomas Nürnberg<sup>2</sup> Yakov Miron<sup>2,3</sup> Federico Tombari<sup>4,5</sup> Stefano Gasperini<sup>4,5,6</sup> Abhinav Valada<sup>1</sup>

<sup>1</sup>University of Freiburg <sup>2</sup>Bosch Research <sup>3</sup>University of Haifa <sup>4</sup>Technical University of Munich <sup>5</sup>Munich Center for Machine Learning <sup>6</sup>Visualais

<sup>\*</sup>Equal contribution.

## Abstract

Holistic perception for autonomous agents requires reliably segmenting and associating scene elements across time and camera views, including objects outside a fixed semantic taxonomy. Existing open-set panoptic benchmarks focus on individual images or views, while panoramic tracking benchmarks assume closed-set taxonomies. Consequently, these benchmarks do not evaluate whether unknown objects can be consistently segmented and tracked across synchronized multicamera sequences. We introduce WayPOP, a panoramic open-set panoptic tracking benchmark built on real Waymo PVPS sequences augmented with diverse out-oftaxonomy 3D assets from Objaverse. Inserted objects are placed at fixed locations in the 3D scene and rendered using calibrated vehicle-mounted cameras as the ego vehicle and surrounding traffic move, providing temporally and cross-view consistent panoptic annotations and auxiliary LiDAR range observations. WayPOP contains approximately 670 curated assets instantiated as 980 unique unknownobject tracks, yielding 53,452 unknown-object mask annotations across 21,300 camera images. Our evaluation protocol defines a hierarchy of tasks from viewindependent open-set panoptic segmentation to full panoramic open-set panoptic tracking, using known/unknown PQ and (w)STQ metrics. We benchmark representative open-set and open-vocabulary segmentation methods with temporal and cross-camera association. Our results reveal substantial limitations of current methods in truly open-set scenarios, highlighting robust unknown-object localization and identity association across time and views as major open challenges. The project page and benchmark are available at https://waypop.cs.uni-freiburg.de.

## 1 Introduction

Robots and autonomous agents require coherent scene understanding to operate safely in dynamic environments, with autonomous driving providing a concrete and safety-critical domain for this challenge. Perception is increasingly moving toward holistic spatiotemporal scene understanding [1], where thing instances are densely segmented and associated over time alongside per-pixel semantic labels for stuff classes [2–4]. Panoramic multi-camera setups further amplify this challenge: objects frequently leave one camera view and reappear in another, requiring both temporal and crosscamera instance association for unified scene interpretation. The Waymo Panoramic Video Panoptic Segmentation (Waymo PVPS) benchmark [5] exemplifies this setting by defining a panoramic panoptic tracking task in which instance identities remain consistent over time and across five synchronized cameras. However, existing benchmarks are largely restricted to fixed semantic taxonomies and therefore do not evaluate whether perception systems can robustly detect, segment, and associate previously unseen objects across time and views.

![](images/fb6ddd5472c7497d580d10ad104d123e997c9c7397108f39b1d0f7bc6b1850de.jpg)  
Figure 1: Overview of the WayPOP Dataset. Real Waymo PVPS driving sequences are augmented with 3D assets rendered consistently across synchronized camera views and time, enabling realistic evaluation of panoramic open-set panoptic segmentation and tracking under viewpoint changes, occlusions, and complex traffic conditions. Examples illustrate the diversity of inserted out-oftaxonomy objects, unknown-object densities, and scene contexts.

This limitation becomes critical in real-world deployment, where autonomous agents inevitably encounter long-tail and out-of-distribution (OOD) objects that fall outside a fixed training taxonomy. This has motivated a growing body of anomaly, open-set, and OOD perception benchmarks [6– 12]. However, constructing an open-set benchmark for panoramic video perception introduces additional data-generation challenges. Real-world anomalous events are rare and difficult to capture at scale, while staged collections [12] and simulation [11] trade realism for controllability and scalability. Image-level synthesis and copy-paste augmentation provide a scalable alternative, but are typically formulated as 2D image-plane operations. Extending these approaches to panoramic video is nontrivial, as inserted objects must maintain geometrically consistent projections, occlusions, and visible extents across synchronized camera views and time.

To address this gap, we introduce WayPOP, a benchmark for panoramic open-set panoptic segmentation and tracking in autonomous driving (Figure 1). WayPOP builds on Waymo PVPS [5] by augmenting real driving sequences with diverse out-of-taxonomy 3D assets sourced from Objaverse [13]. Unlike fully synthetic simulators [14, 15] or purely 2D copy-paste techniques, WayPOP preserves real-world scene context through geometry-grounded insertion of unknown objects into synchronized multi-camera observations, with auxiliary LiDAR range updates. Inserted objects remain static in the global 3D scene, while the ego vehicle and surrounding traffic continue to move. Despite this static placement, our experiments show that it remains challenging to detect, segment, and consistently track unknown objects under viewpoint changes, occlusions, and camera transitions. Our insertion pipeline is designed to maintain four key properties: geometric consistency (coherent object placement and occlusion relationships), photometric consistency (alignment with scene lighting and conditions), multi-view consistency (coherent projections across synchronized camera views), and temporal consistency (stable object identities over time). Together, these properties enable controlled evaluation over a highly diverse set of unknown objects while retaining the appearance, occlusion structure, and temporal continuity of real-world driving scenes.

WayPOP contains approximately 670 unique Objaverse assets instantiated as 980 unknown-object tracks, yielding 53,452 unknown-object mask annotations across 21,300 images from five synchronized camera streams with corresponding LiDAR range measurements. Building on this benchmark, we define multi-view open-set panoptic tracking (MV-OSPT), requiring panoptic predictions for known and unknown regions with temporally and cross-view consistent identities, together with a hierarchy of reduced task variants: view-independent open-set panoptic segmentation (VI-OSPS) and frontal-view open-set panoptic tracking (FV-OSPT). Finally, we benchmark representative camera-based open-set and open-vocabulary segmentation methods with temporal and cross-camera association, complemented by exploratory multimodal evaluations using LiDAR.

Table 1: Comparison of driving benchmarks for anomaly detection, out-of-distribution (OOD) recognition, open-set panoptic segmentation, and panoramic tracking. WayPOP combines unknown-object panoptic labels with temporal and cross-camera identity consistency in real driving sequences.
<table><tr><td colspan="4"></td><td colspan="3">Capabilities</td><td colspan="3">Size</td><td rowspan="2"></td><td rowspan="2">#Seq. Data origin</td></tr><tr><td>Dataset</td><td>Variant</td><td>LiDAR</td><td>Source</td><td>Temp.</td><td>Pano. Track.</td><td></td><td>#OOD Label</td><td>Test / Val</td><td></td></tr><tr><td rowspan="2">Fishyscapes [16]</td><td>FS Lost&amp;Found</td><td>x</td><td>Staged</td><td>x</td><td>x</td><td>x</td><td>7</td><td>SM</td><td>275 / 100</td><td>n/a</td><td>Lost and Found [6]</td></tr><tr><td>FS Static</td><td>x</td><td>2D aug.</td><td>x</td><td>x</td><td>x</td><td>12</td><td>SM</td><td>1,000 / 30</td><td>n/a</td><td>Cityscapes [17]</td></tr><tr><td rowspan="2">CAOS [18]</td><td>StreetHazards</td><td>x</td><td>Simulated</td><td>√</td><td>x</td><td>x</td><td>250</td><td>SM</td><td>1,500 / -</td><td>150</td><td></td></tr><tr><td>BDD-Anomaly</td><td>x</td><td>Val. set</td><td>x</td><td>x</td><td>x</td><td>3</td><td>SM</td><td>810 / -</td><td>n/a</td><td>BDD100K [19]</td></tr><tr><td rowspan="2">SMIYC [7]</td><td>RoadAnomaly21</td><td>x</td><td>Web</td><td>x</td><td>x</td><td>x</td><td>26</td><td>SM</td><td>100 / 10</td><td>n/a</td><td></td></tr><tr><td>RoadObstacle21</td><td>x</td><td>Staged</td><td>√</td><td>x</td><td>x</td><td>31</td><td>SM</td><td>382 /30</td><td>90</td><td></td></tr><tr><td rowspan="2">Wupp. OOD [20]</td><td>SOS</td><td>x</td><td>Staged</td><td>√</td><td>x</td><td>√</td><td>13</td><td>IM</td><td>1,129 / -</td><td>20</td><td></td></tr><tr><td>CWL</td><td>x</td><td>Simulated</td><td>√</td><td>x</td><td>√</td><td>18</td><td>IM</td><td>1,210/ -</td><td>26</td><td></td></tr><tr><td>Lost&amp;Found [6]</td><td></td><td>x</td><td>Staged</td><td>√</td><td>x</td><td>x</td><td>37</td><td>SM</td><td>1,068 / -</td><td>5</td><td></td></tr><tr><td>SOD [21]</td><td></td><td>√</td><td>Staged</td><td>x</td><td>x</td><td>x</td><td>9</td><td>SM</td><td>460 / 530</td><td>5</td><td></td></tr><tr><td>WD-Pascal [22]</td><td></td><td>x</td><td>Simulated</td><td>x</td><td>x</td><td>x</td><td>6</td><td>SM</td><td>-/70</td><td>n/a</td><td></td></tr><tr><td>Vistas-NP [23]</td><td></td><td>x</td><td>Class excl.</td><td>x</td><td>x</td><td>x</td><td>4</td><td>SM</td><td>11,167/-</td><td>n/a</td><td>Map. Vistas [24]</td></tr><tr><td>PANIC [9]</td><td></td><td>x</td><td>Real</td><td>x</td><td>x</td><td>x</td><td>58</td><td>PM</td><td>679 / 131</td><td>n/a</td><td>Cityscapes [17]</td></tr><tr><td>AnoVox [11]</td><td></td><td>√</td><td>Simulated</td><td>√</td><td>√</td><td>x</td><td>178</td><td>IM</td><td>n/a</td><td></td><td>1,228 CARLA [25]</td></tr><tr><td>WayPOP (Ours)</td><td></td><td>√</td><td>3D aug.</td><td>√</td><td>√</td><td>√</td><td>670</td><td>PM</td><td>19,665 / 1,635</td><td>80</td><td>Waymo PVPS [5]</td></tr></table>

All datasets provide RGB. Temp. = temporal; Pano. = panoramic; Track. = tracking. SM/IM/PM = semantic/instance/panoptic masks.

In summary, our main contributions are fivefold: (1) We introduce WayPOP, to our knowledge the first benchmark for panoramic open-set panoptic tracking in autonomous driving, enabling unknown-object evaluation across synchronized multi-camera video. (2) We develop a geometryguided, human-in-the-loop construction pipeline that augments real driving sequences with curated 3D assets, aiming to preserve geometric, photometric, multi-view, and temporal consistency. (3) We formalize multi-view open-set panoptic tracking (MV-OSPT) and define a hierarchical task suite spanning view-independent segmentation, monocular tracking, and full panoramic tracking. (4) We provide an open-set panoramic tracking evaluation protocol based on adapted PQ and (w)STQ metrics for separate evaluation of known and unknown regions. (5) We benchmark representative camera-based open-set and open-vocabulary methods combined with temporal and cross-camera association strategies, revealing substantial limitations in maintaining coherent unknown-object identities across time and views.

## 2 Related Work

Closed-set panoptic segmentation and tracking. Panoptic segmentation combines semantic segmentation of stuff regions with instance segmentation of thing objects [26], with benchmarks such as Panoptic COCO [27], Cityscapes [17], Mapillary Vistas [24], and Syn-Mediverse [14] driving substantial progress [28–30]. Panoptic tracking further requires temporally consistent instance identities [3, 4, 2, 31], while recent panoramic benchmarks such as WildPPS [32] and Waymo PVPS [5] additionally introduce cross-view consistency across synchronized cameras. However, these benchmarks remain restricted to closed-set taxonomies and therefore do not evaluate whether previously unseen objects can be segmented and consistently associated across time and views.

Open-set and OOD segmentation benchmarks. A complementary line of work studies anomaly, out-of-distribution (OOD), and open-set segmentation in driving scenes. Benchmarks such as Fishyscapes [16], Lost and Found [6], and RoadObstacle21 from SegmentMeIfYouCan [7] primarily provide semantic or binary anomaly masks, with the latter two addressing world-static road hazards observed from moving vehicles. SOS [12] additionally evaluates temporal identity persistence in this setting. Open-set panoptic benchmarks extend this setting by requiring unknown objects to be segmented as instances while preserving known-class panoptic predictions [8, 10, 9, 33], but remain predominantly single-view or frame-centric. JRDB-PanoTrack [34] is closest in spirit to our work, as it studies panoramic open-world panoptic tracking. However, it targets human-centric robotic environments rather than on-road autonomous driving with vehicle-mounted cameras and real traffic structures. As summarized in Table 1, existing driving-focused OOD benchmarks provide different combinations of unknown-object annotations, temporal sequences, and panoramic coverage, but none jointly supports open-set panoptic tracking across time and camera views. WayPOP addresses this missing intersection by combining unknown-object panoptic segmentation with temporally and cross-view consistent identities in real driving sequences. Furthermore, WayPOP introduces substantially greater OOD object diversity than prior work, enabling more thorough evaluation and better alignment with real-world variability.

OOD data generation for autonomous driving. Collecting diverse real-world OOD events in active traffic is difficult because anomalous objects are rare and their collection is constrained by safety and logistical considerations. Existing benchmarks therefore rely on staged capture [6, 12], simulation [11], validation-set mining [35, 36], or 2D image-level augmentation [16, 10], each trading off realism, controllability, and scalability. In particular, image-plane augmentation does not naturally provide geometry-consistent projections and occlusions across synchronized cameras and time. WayPOP instead inserts curated 3D assets into synchronized multi-camera and LiDAR observations, enabling temporally and cross-view consistent panoramic open-set evaluation in real driving sequences.

Open-set segmentation and association baselines. Recent open-set [37, 38, 10, 39–41] and openvocabulary panoptic segmentation methods [42–45] have improved frame-level recognition and segmentation of novel objects, but are typically applied independently to each image and therefore do not address the temporal and cross-camera consistency required for panoramic panoptic tracking. Conversely, modern association methods, including box-based trackers [46–49] and mask propagation approaches [50–53], are primarily evaluated in closed-set settings where unknown instances are not explicitly modeled. WayPOP bridges both lines of work by evaluating open-set panoptic predictions under progressively stronger constraints on temporal and cross-view consistency.

## 3 WayPOP Benchmark

## 3.1 Task Definition

Multi-View Open-Set Panoptic Tracking (MV-OSPT). We define multi-view open-set panoptic tracking, which requires full-scene panoptic predictions for both known classes and out-of-taxonomy objects, with temporally and cross-view consistent instance identities across synchronized camera streams. At each time step $t \in \{ 1 , \ldots , T \}$ , the input consists of synchronized multi-camera images $\{ I _ { t } ^ { v } \} _ { v = 1 } ^ { V }$ , and the goal is to produce panoptic predictions $P _ { t } ^ { v }$ assigning each pixel a semantic label y and instance identity k. The semantic label space is given by $y \in \mathcal { C } _ { \mathrm { k n o w n } } \cup$ {unknown}, where all outof-taxonomy objects are mapped to a single unknown category. As in standard panoptic segmentation, thing classes, including unknown, are segmented into instances with unique identities. Following Waymo PVPS [5], predictions across synchronized camera views are evaluated jointly, requiring consistent instance identities across cameras to form a unified panoramic scene representation.

MV-OSPT imposes two consistency constraints: temporal consistency, which requires stable identities over time, and cross-view consistency, which requires shared identities across camera views and view transitions. Together, these constraints require jointly reasoning over space, time, and camera geometry while handling both known and unknown objects. We derive two simpler task variants by selectively relaxing temporal and cross-view consistency constraints, yielding a hierarchy of tasks:

Frontal-View Open-Set Panoptic Tracking (FV-OSPT). By removing cross-view consistency while retaining temporal consistency, the task reduces to monocular open-set panoptic tracking in the front-facing camera stream. Methods must produce temporally consistent panoptic predictions with stable instance identities, including for unknown objects.

View-Independent Open-Set Panoptic Segmentation (VI-OSPS). Removing both temporal and cross-view consistency reduces the task to per-frame open-set panoptic segmentation across all camera views, with each image processed independently.

![](images/5e6badd6083d7cf680929ff293c85ec7f580275c95e41aef18d5d1c89831edbd.jpg)  
Figure 2: Overview of our geometry-guided data generation pipeline. Starting from real-world Waymo sequences and curated 3D assets from Objaverse, objects are inserted at scene-level locations sampled from LiDAR geometry. Inserted assets are rendered and composited into camera views and corresponding LiDAR measurements, with updated range images guiding occlusion masks. This unified process produces photorealistic RGB images, sensor-consistent LiDAR data, and temporally and cross-view consistent panoptic annotations with inserted objects labeled as unknown.

All task variants share the same open-set output space and differ only in the consistency constraints imposed on instance identities, forming a hierarchy from independent per-frame segmentation to monocular tracking and full panoramic multi-view tracking. VI-OSPS enables direct comparison to prior per-frame segmentation methods, FV-OSPT aligns with established monocular tracking benchmarks [17, 54, 19], and MV-OSPT represents the full setting requiring joint reasoning over temporal dynamics, cross-view geometry, and open-set instance segmentation.

## 3.2 Evaluation Metrics

We evaluate segmentation and tracking performance using established metrics adapted to the open-set and panoramic setting. For VI-OSPS, we report the mean Intersection-over-Union (mIoU) and Panoptic Quality (PQ) for semantic and panoptic segmentation quality, respectively. For FV-OSPT and MV-OSPT, we report Segmentation and Tracking Quality (STQ) [31] and the panoramic weighted STQ (wSTQ) metric from Waymo PVPS [5]. To reflect the open-set setting, all metrics are reported separately for known and unknown regions. For tracking evaluation, Association Quality (AQ) is computed in a class-conditioned manner by restricting ground-truth tracks to the respective subset while leaving predictions unfiltered, enabling evaluation of tracking consistency independently of semantic classification accuracy.

## 3.3 Geometry-Guided Data Generation Pipeline

WayPOP is constructed via a geometry-guided, human-in-the-loop pipeline that inserts synthetic 3D assets into synchronized multi-view camera and LiDAR observations from real driving sequences while preserving geometric, photometric, multi-view, and temporal consistency. The pipeline consists of three stages: (1) asset curation and filtering, (2) geometry-aware scene-level object placement, and (3) physically grounded rendering and sensor integration (Figure 2).

Source Data and Asset Curation. Our pipeline builds on real-world driving sequences from the PVPS subset [5] of the Waymo Open Dataset [55], providing synchronized multi-view camera streams, LiDAR observations, and panoptic annotations. We adopt the PVPS taxonomy as the set of known classes, treating all inserted objects as out-of-taxonomy. To populate the unknown object space, we curate 3D assets from Objaverse v1.0 [13] and Objaverse-XL [56], which provide comple mentary trade-offs between structured category annotations and object diversity. For Objaverse v1.0, we use LVIS labels [57] to remove assets that are semantically similar to known driving classes or are implausibly small for driving scenarios. For Objaverse-XL, which relies on unstructured metadata [58], we use Qwen3-14B [59] to filter semantically overlapping objects. Following automatic filtering, we manually remove degenerate or low-quality meshes, correct object scale, and verify driving-scene plausibility. This process yields a curated pool of approximately 670 unique assets for insertion. We refer to the supplementary material (Section B.1) for further details.

Scene-Level Insertion Point Sampling. Objects are inserted with a fixed pose in the global 3D coordinate frame of each driving sequence rather than independently per frame or camera view.

Inserted assets, therefore, remain static relative to the environment as the ego vehicle moves, and their projections across camera views and time follow directly from the recorded ego poses and sensor calibration. To identify valid insertion locations, we aggregate LiDAR observations over time and restrict candidates to visible static ground surfaces by excluding dynamic objects and non-ground semantic classes. While intersections with existing scene elements are generally avoided, we allow a small fraction of overlaps with annotated dynamic actors to avoid biasing insertions toward always-empty regions. Occasional actor-object intersection remains a limitation. For each candidate location, we estimate a local supporting surface using RANSAC [60] plane fitting and retain only sufficiently large, approximately planar regions. Candidate locations are prioritized based on visibility and spatial coverage, and a spatially distributed subset is selected in a bird’s-eye-view while maintaining a target distribution over surface types (e.g., road, sidewalk, and other ground). Objects are assigned from the curated asset pool while enforcing diversity and plausibility constraints. We avoid repeated object categories within a sequence and, for Objaverse-XL assets, derive coarse semantic categories for this from metadata using Qwen3-14B [59]. Finally, object scale is adapted to the spatial context by assigning smaller objects to nearby locations and larger objects to distant regions, while object pose is sampled randomly.

Physically Grounded Rendering and Sensor Integration. Given the selected object placements, we generate sensor-consistent observations by rendering inserted assets into all synchronized camera views and integrating them into the corresponding LiDAR observations. Since placement is defined in the global 3D frame, projections across views and time follow directly from the known sensor calibration. Inserted assets are rendered using the physically based Blender Cycles engine, accounting for view-dependent geometry and appearance. Scene lighting is configured to approximate the recording conditions using metadata-derived illumination cues, including sun position and weather information. Rendered objects are composited into the camera images using depth-based visibility masks derived from the inserted object’s geometry, with manual refinement as needed to correct incomplete occlusion relationships. We further improve visual coherence through photometric harmonization [61], adapting color and tone of the rendered object to the surrounding image. Ablations of lighting adaptation, occlusion handling, and harmonization are provided in the supplementary material (Section B.3).

To ensure geometric consistency across modalities, we update LiDAR range measurements by raycasting along the original Waymo sensor rays wherever inserted objects occlude existing returns. Operating directly in the range-image domain preserves the sensor sampling pattern and enforces consistent occlusion relationships between real and inserted content. Finally, panoptic annotations are generated from the rendering process. Instance identities are consistent across time and camera views by construction, and all inserted objects are mapped to the unknown class. This unified pipeline ensures joint consistency between RGB observations, LiDAR data, and annotations. We analyze temporal and geometric consistency under ego motion in the supplementary material (Section B.4).

## 3.4 Dataset Structure and Statistics

![](images/5ae07dc26db40429e6246ba7a19bd745c1392ab1f61606aeb5a1608c75189c15.jpg)  
(a) Unknown masks per camera

![](images/13a87a9d62d26d1e91abe1b93945b118e5805881f4eb6433e382c8ca682e7279.jpg)  
(b) Track duration distribution

![](images/c0c829c94c6f589018eaf43cbf220f5cc67980327faf662ab8fb925a57605fa5.jpg)  
(c) Visible cameras per track

![](images/fc868c5b7694a28e911136459a2062c8691145964c79f2978c5779a304b7fa06.jpg)  
(d) Unknown and known thing masks  
Figure 3: Dataset statistics for WayPOP. We report unknown mask distributions across cameras, temporal and cross-camera track visibility, and mask counts for known versus unknown objects. Mask statistics are per-image; track statistics use resolved visible tracks.

WayPOP is derived from the 20 validation sequences of Waymo PVPS [5]. Each sequence is split into two temporal clips, yielding 40 base clips, and rendered with two asset-source variants using Objaverse v1.0 and Objaverse-XL. This results in 80 sequence variants across the same real driving scenes, each with a different inserted out-of-taxonomy object. The benchmark consists of three splits: validation (6 V1 clips from 3 PVPS sequences), WayPOP-v1 (34 test clips from the remaining 17 sequences), and WayPOP-XL (40 test clips from all 20 sequences). Unlike Waymo PVPS, which evaluates 28 classes, WayPOP retains 22 known classes and excludes ego\_vehicle, bird, ground\_animal, static, dynamic, and ground. Inserted assets are added as instanceaware unknown objects. Each sequence variant provides five synchronized RGB streams, panoptic annotations with temporally consistent identities, and aligned LiDAR insertions, while our benchmark experiments focus on camera-based evaluation. Figure 3 summarizes the resulting mask distributions and temporal and cross-camera track visibility. Further asset and split statistics are provided in the supplementary material (Section B.2).

## 4 Baselines

We evaluate two complementary paradigms for handling out-of-distribution objects: open-set and open-vocabulary perception. Open-set methods explicitly model distributional uncertainty, while open-vocabulary methods leverage vision-language models to generalize beyond fixed training taxonomies. Since these methods operate on a per-image basis, we pair them with temporal and crosscamera association strategies for panoramic tracking evaluation. Our primary baselines use camera inputs. Exploratory RGB-LiDAR segmentation experiments are provided in the supplementary material (Section C.2).

## 4.1 Open-Set and Open-Vocabulary Baselines

Open-Set. We evaluate four open-set panoptic segmentation baselines covering complementary mechanisms for identifying unknown objects. U3HS [37] uses uncertainty-guided unknown prediction, PoDS [10] explicitly models unknown instances, Mask2Anomaly [39] provides a mask-centric anomaly formulation, and RbA [40] serves as a rejection-based baseline. Together, these methods span uncertainty estimation, anomaly segmentation, and rejection-based open-set inference. We refer to the supplementary material for further details (Section A.1).

Open-Vocabulary. We evaluate open-vocabulary panoptic segmentation methods spanning multiple design paradigms, including CLIP-based [62] approaches (FC-CLIP [42], MAFT+ [43], EOV-Seg [63]), diffusion-based segmentation (ODISE [45]), and prompt-driven segmentation (Open-WorldSAM [64]). All methods are evaluated using publicly available pretrained checkpoints without architectural modifications or additional fine-tuning. To obtain open-set predictions, all methods use a shared, fixed prompt vocabulary that includes synonyms for known classes and a small set of generic unknown prompts (e.g., obstacle, debris, road hazard). Since enumerating all possible unknown categories is infeasible, these prompts serve as a proxy for open-set recognition while avoiding overly broad terms such as object. Additional implementation details, the full vocabulary, and a prompt-sensitivity analysis for ODISE are provided in the supplementary material (Section A.2). The analysis shows that, across six prompt sets, ODISE’s unknown PQ varies substantially, whereas its known PQ remains comparatively stable.

## 4.2 Tracking and Panoramic Association Baselines

The segmentation baselines produce open-set panoptic predictions independently for each camera frame. To evaluate tracking, we pair each method with temporal and cross-camera association strategies that assign identities to predicted thing and unknown instances while leaving segmentation outputs unchanged. We first perform within-camera tracking using box-based [46, 47] and maskbased [50–52] methods, producing view-local tracklets. These tracklets are then projected into the panoramic space [5] and merged across overlapping cameras using IoU-based matching over short temporal windows. This associative formulation isolates tracking errors from segmentation quality, enabling separate analysis of temporal drift and cross-camera fragmentation. Additional analyses with ground-truth masks are provided in the supplementary material (Section C.4).

Table 2: VI-OSPS results on the WayPOP Objaverse v1.0 test split for open-vocabulary methods.
<table><tr><td></td><td></td><td colspan="3">Known</td><td colspan="3">Unknown</td></tr><tr><td>Method</td><td>Variant</td><td>PQ↑</td><td>SQ↑</td><td>RQ↑</td><td>PQ↑</td><td>SQ↑</td><td>RQ↑</td></tr><tr><td>MAFT+ [43]</td><td>Large</td><td>23.6</td><td>66.4</td><td>29.8</td><td>0.1</td><td>83.6</td><td>0.1</td></tr><tr><td>FC-CLIP [42]</td><td>ConvNeXt-Large</td><td>27.5</td><td>61.4</td><td>33.8</td><td>0.2</td><td>87.4</td><td>0.2</td></tr><tr><td>EOV-Seg [63]</td><td>ConvNeXt-Large</td><td>25.5</td><td>64.7</td><td>31.7</td><td>0.7</td><td>85.1</td><td>0.8</td></tr><tr><td>OpenWorldSAM [64]</td><td>Standard</td><td>10.8</td><td>59.7</td><td>14.2</td><td>1.0</td><td>83.8</td><td>1.2</td></tr><tr><td>ODISE [45]</td><td>Caption</td><td>24.1</td><td>67.8</td><td>29.1</td><td>30.9</td><td>86.2</td><td>35.8</td></tr></table>

Table 3: VI-OSPS results on the WayPOP test split for the Objaverse v1.0 and Objaverse-XL asset variants. Evaluating both asset sources enables comparison across different levels of OOD diversity and semantic structure.
<table><tr><td></td><td></td><td colspan="6">WayPOP-v1</td><td colspan="6">WayPOP-XL</td></tr><tr><td></td><td></td><td colspan="3">Known</td><td colspan="3">Unknown</td><td colspan="3">Known</td><td colspan="3">Unknown</td></tr><tr><td>Method</td><td></td><td>PQ↑</td><td>SQ↑</td><td>RQ↑</td><td>PQ↑</td><td>SQ↑</td><td>RQ↑</td><td>PQ↑</td><td>SQ↑</td><td>RQ↑</td><td>PQ↑</td><td>SQ↑</td><td>RQ↑</td></tr><tr><td>ΛO</td><td>ODISE [45] (Caption)</td><td>24.1</td><td>67.8</td><td>29.1</td><td>30.9</td><td>86.2</td><td>35.8</td><td>24.4</td><td>66.5</td><td>29.6</td><td>23.6</td><td>86.3</td><td>27.4</td></tr><tr><td>Open et</td><td>U3HS [37]</td><td>32.8</td><td>53.6</td><td>40.6</td><td>9.3</td><td>67.5</td><td>13.8</td><td>32.1</td><td>53.4</td><td>39.8</td><td>13.2</td><td>69.8</td><td>18.9</td></tr><tr><td></td><td>M2A [39]</td><td>46.9</td><td>76.7</td><td>57.0</td><td>27.5</td><td>79.6</td><td>33.8</td><td>45.8</td><td>76.9</td><td>55.7</td><td>26.1</td><td>79.5</td><td>32.3</td></tr><tr><td></td><td>RbA [40]</td><td>45.3</td><td>74.4</td><td>54.7</td><td>31.9</td><td>79.2</td><td>40.2</td><td>44.4</td><td>74.2</td><td>54.0</td><td>29.1</td><td>80.6</td><td>36.1</td></tr><tr><td></td><td>PoDS [10]</td><td>45.1</td><td>74.4</td><td>54.6</td><td>34.6</td><td>79.8</td><td>43.4</td><td>44.2</td><td>74.4</td><td>53.7</td><td>31.1</td><td>81.3</td><td>38.2</td></tr></table>

## 5 Experiments

We evaluate methods on both the Objaverse v1.0 and XL variants, with additional results provided in the supplementary material. In the following, we present results across tasks of increasing difficulty, progressively introducing temporal and multi-view consistency constraints.

## 5.1 Task1: View-Independent Open-Set Panoptic Segmentation (VI-OSPS)

We report results in Tables 2 and 3 for representative open-vocabulary and open-set methods on WayPOP-v1 and -XL. Current open-vocabulary methods exhibit systematic limitations in truly openset scenarios: Mask2Former-style [30] approaches often suppress OOD objects during panoptic inference due to taxonomy-driven foreground selection, while prompt-driven methods such as OpenWorldSAM struggle to localize OOD regions from generic prompts alone. Together, these findings suggest that robust OOD panoptic segmentation requires class-agnostic objectness beyond fixed taxonomies and explicit prompt specification. Further analysis is provided in the supplementary material (Section A.2.4). Open-set methods generally achieve higher unknown PQ than the evaluated open-vocabulary approaches, suggesting that explicit unknown-object modeling is beneficial in this setting. RbA and PoDS provide the strongest balance between known- and unknown-class performance, while M2A favors known-class accuracy at the expense of unknown detection and U3HS underperforms on both metrics. Open-set methods achieve comparable performance across WayPOP-v1 and WayPOP-XL, whereas open-vocabulary methods such as ODISE obtain higher unknown PQ on WayPOP-v1, likely due to recognizable LVIS category names enabling semantic matching. We therefore focus the main-text tracking evaluation on WayPOP-v1.0. The supplementary material provides a cross-dataset segmentation comparison (Section C.1) and additional frontal-view tracking results on WayPOP-XL (Section C.3).

## 5.2 Task2: Frontal-View Open-Set Panoptic Tracking (FV-OSPT)

Results for frontal-view tracking on the WayPOP Objaverse v1.0 test split are reported in Table 4, with additional results for the Objaverse XL test split provided in the supplementary material (Section C.3). We observe that open-set panoptic tracking performance is largely driven by the quality of the underlying segmentation, with open-set methods outperforming ODISE and PoDS achieving slightly higher unknown STQ than RbA. Among trackers, appearance-based methods such as MASA and DEVA achieve higher unknown AQ than the motion-based BoT-SORT, which may reflect the importance of appearance cues for associating unknown objects without reliable category priors. Interestingly, PoDS shows a larger gap between known and unknown STQ than RbA and ODISE, suggesting that its stronger unknown-class segmentation benefits unknown tracking, while known-class tracking remains limited by association errors.

Table 4: FV-OSPT results on the WayPOP Objaverse v1.0 test split. We combine image-level open-set and open-vocabulary segmentation with temporal association strategies for frontal-view open-set panoptic tracking.
<table><tr><td colspan="2"></td><td colspan="2">Known</td><td colspan="3">Unknown</td></tr><tr><td></td><td>Tracker</td><td>STQ↑ AQ↑</td><td>mIoU↑</td><td>STQ↑</td><td>AQ↑</td><td>mIoU↑</td></tr><tr><td></td><td>TrackTrack [46]</td><td>10.5</td><td>4.6</td><td>23.8</td><td>12.6 8.1</td><td>19.7</td></tr><tr><td rowspan="2">O IS45]</td><td>BoT-SORT [47]</td><td>10.9</td><td>5.0</td><td>23.8</td><td>11.5 6.8</td><td>19.7</td></tr><tr><td>MASA [50]</td><td>10.8</td><td>4.9</td><td>23.8 14.5</td><td>10.7</td><td>19.7</td></tr><tr><td></td><td>DEVA [51]</td><td>10.9</td><td>5.0</td><td>23.8</td><td>14.4 10.6</td><td>19.7</td></tr><tr><td>PD] 10]</td><td>TrackTrack [46]</td><td>16.7</td><td>5.6</td><td>49.6</td><td>26.4 25.4</td><td>16.0 43.5</td></tr><tr><td></td><td>BoT-SORT [47]</td><td>16.5</td><td>5.5</td><td>49.6</td><td>14.8 30.0</td><td>43.5</td></tr><tr><td></td><td>MASA [50]</td><td>15.8</td><td>5.0</td><td>49.6</td><td>20.7</td><td>43.5</td></tr><tr><td></td><td>DEVA [51]</td><td>19.1</td><td>7.4</td><td>49.6</td><td>29.0 19.3</td><td>43.5</td></tr><tr><td>Rb] 40]</td><td>TrackTrack [46]</td><td>18.4</td><td>6.8</td><td>49.4</td><td>26.3 18.1</td><td>38.3</td></tr><tr><td></td><td>BoT-SORT [47]</td><td>17.4</td><td>6.1</td><td>49.4</td><td>25.2 16.6</td><td>38.3</td></tr><tr><td></td><td>MASA [50]</td><td>16.2</td><td>5.3</td><td>49.4</td><td>29.6 22.9</td><td>38.3</td></tr><tr><td></td><td>DEVA [51]</td><td>20.6</td><td>8.5</td><td>49.4</td><td>28.7 21.5</td><td>38.3</td></tr></table>

## 5.3 Task3: Multi-View Open-Set Panoptic Tracking (MV-OSPT)

Results for panoramic multi-view tracking are reported in Table 5 with qualitative results for RbA (TrackTrack) presented in Fig. 4. Compared to the frontal-view setting, multi-view tracking substantially increases tracking difficulty due to large viewpoint changes, varying object visibility, and inconsistent appearance across cameras. Among the evaluated methods, PoDS combined with DEVA achieves the strongest unknown tracking performance, reaching 37.1 unknown wSTQ, while RbA with DEVA attains the highest unknown wAQ of 25.5. Consistent with the frontal-view results, this suggests that stronger OOD sensitivity remains beneficial when identities must persist across time and camera views. However, substantial association errors remain even with perfect segmentation: given ground-truth masks and classes (wmIoU = 100, marked Oracle), DEVA achieves only 43.5 known and 61.6 unknown wAQ.

Table 5: MV-OSPT results on the WayPOP Objaverse v1.0 test split. Methods combine image-level open-set and open-vocabulary segmentation with temporal and cross-camera association strategies for panoramic open-set panoptic tracking. Oracle results use ground-truth masks with the respective tracker to isolate association performance from segmentation errors.
<table><tr><td rowspan="2">Approach</td><td rowspan="2">Tracker</td><td colspan="3">Known</td><td colspan="3">Unknown</td></tr><tr><td>wSTQ↑</td><td>wAQ↑</td><td>wmIoU↑</td><td>wSTQ↑</td><td>wAQ↑</td><td>wmIoU↑</td></tr><tr><td>PoDS [10]</td><td>DEVA [51]</td><td>24.2</td><td>11.4</td><td>51.3</td><td>37.1</td><td>23.9</td><td>57.6</td></tr><tr><td>PoDS [10]</td><td>TrackTrack [46]</td><td>22.4</td><td>9.7</td><td>51.3</td><td>34.3</td><td>20.4</td><td>57.6</td></tr><tr><td>RbA [40]</td><td>DEVA [51]</td><td>23.4</td><td>10.6</td><td>51.5</td><td>36.9</td><td>25.5</td><td>53.3</td></tr><tr><td>RbA [40]</td><td>TrackTrack [46]</td><td>22.5</td><td>9.8</td><td>51.5</td><td>34.3</td><td>22.1</td><td>53.3</td></tr><tr><td>Oracle</td><td>DEVA [51]</td><td>65.9</td><td>43.5</td><td>100.0</td><td>78.5</td><td>61.6</td><td>100.0</td></tr></table>

## 6 Conclusion

In this work, we introduced WayPOP, a benchmark for panoramic open-set panoptic tracking in autonomous driving that enables evaluation of unknown-object segmentation and tracking in real driving sequences with long-tail and out-of-taxonomy objects. By augmenting real Waymo PVPS sequences with geometry-guided 3D object insertions, WayPOP supports a hierarchy of tasks ranging from open-set panoptic segmentation to full multi-view open-set tracking. Our experiments show that current open-set and open-vocabulary methods still struggle to robustly localize and maintain consistent unknown-object identities across time and camera views, highlighting panoramic open-set tracking as an important open challenge for autonomous perception. WayPOP nevertheless has severa limitations that motivate future work, including imperfect occlusion modeling, static inserted objects, and simplified lighting harmonization. Despite these limitations, we believe WayPOP provides a scalable and realistic foundation for future research on open-world driving perception, dynamic unknown-object insertion, and multimodal panoramic tracking.

![](images/228d44e99eb8b3ff4297212867a02b54e1979165e251c5d168dbbe5384192ebb.jpg)  
Figure 4: Qualitative results for MV-OSPT on RbA (TrackTrack). The front, front-left and front-right views are presented.

## Acknowledgments and Disclosure of Funding

This work was funded by the Bosch Research Collaboration on AI-Driven Automated Driving. Abhinav Valada was supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant 539134284. Stefano Gasperini was supported by the German Federal Ministry of Research, Technology and Space (BMFTR) through the Robotics Institute Germany (RIG). In addition, the authors would like to thank the German Federal Ministry of Research, Technology and Space and the German federal states (https://go-nhr.de/our-partners) for supporting this work with computing resources as part of the National High-Performance Computing (NHR) joint funding program.

Licensing. WayPOP is not affiliated with, sponsored by, or endorsed by Waymo LLC. WayPOP access requires Waymo Open Dataset registration and acceptance of its non-commercial license. Objaverse and Objaverse-XL collections use ODC-By v1.0, while individual assets retain their respective licenses. Original 3D assets are not redistributed; full terms and attribution notices accompany the release.

## Supplementary Material

## A Baseline Setup

## A.1 Open-Set Baselines

## A.1.1 Methods

We evaluate four open-set panoptic segmentation methods covering complementary approaches for unknown-object recognition, including uncertainty-guided prediction (U3HS [37]), explicit unknowninstance modeling (PoDS [10]), mask-based anomaly segmentation (Mask2Anomaly [39]), and rejection-based inference (RbA [40]). Together, these methods provide a broad basis for assessing how current open-set approaches generalize to unknown objects in WayPOP.

U3HS [37]. U3HS combines panoptic segmentation of known classes with uncertainty-based identification of unknown regions. It separates unknown objects into individual instances by clustering their learned instance-aware embeddings, without requiring unknown-category examples during training.

PoDS [10]. PoDS jointly predicts known semantic classes and unknown object instances using an OOD contextual module and dual decoders trained with an alignment-mismatch strategy.

Mask2Anomaly [39]. Mask2Anomaly combines mask classification with global masked attention, mask contrastive learning, and mask refinement to identify unknown objects while reducing false positives.

RbA [40]. RbA builds on mask classification and identifies unknown regions using an outlier score that measures rejection by all known classes.

We additionally evaluated Prior2Former [38], but, despite extensive experimentation and hyperparameter tuning, could not obtain meaningful results on WayPOP. We therefore omit it from the quantitative comparison.

## A.1.2 Implementation Details

All four baselines are trained on the Waymo PVPS training set using a batch size of 16, $6 4 0 \times 9 6 0$ crops $( H \times W )$ , and random horizontal flipping. U3HS, Mask2Anomaly, and RbA use ImageNetpretrained ResNet-50 backbones. Mask2Anomaly and RbA share an initial 60,000-iteration training setup with AdamW, an initial learning rate of $1 \stackrel { . } { \times } 1 0 ^ { - 4 } ( 1 \times 1 0 ^ { - 5 }$ for the backbone), weight decay of 0.05, and polynomial learning-rate decay with power 0.9. Both use random resizing with scale factors from 0.5 to 2.0, random cropping, and SSD color augmentation. They also incorporate COCO 2017 outliers [27] with probability 0.2 in their respective training stages. At inference, both use query confidence and overlap thresholds of 0.8.

U3HS [37] We use an output stride of 16 and 8-dimensional embeddings. We train for 120 epochs using Adam with initial learning rates of $1 \times 1 0 ^ { - 3 }$ for the model and $\mathsf { \overline { { 5 } } } \times 1 0 ^ { - 5 }$ for the backbone, a per-epoch decay factor of 0.98, and no weight decay. Data augmentation includes random zoom factors from 1.0 to 2.0 and SSD color augmentation. The semantic Dirichlet NLL, center MSE, prototype-instance, and discriminative losses have unit weights. At inference, unknown instances are obtained using DBSCAN with $\epsilon = 0 . 5$ , discarding instances smaller than 200 pixels. Center-based grouping uses a threshold of 0.1.

PoDS [10]. We initialize the HRNet-W32 backbone with pretrained weights and train the full model for 60,000 iterations on 22 known classes. We then train the OOD branches and contextual/dynamic modules for 40,000 additional iterations, keeping the backbone and base branches frozen. Both stages use Adam with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ and polynomial learning-rate decay with power 0.9. For OOD training, we apply COCO 2017 copy-paste augmentation [27] with probability 0.5, excluding categories overlapping with known Waymo classes. At inference, we use known and unknown center thresholds of 0.5 and 0.03, respectively.

Mask2Anomaly [39] We train the full Mask2Former-based model with 100 queries using the shared first-stage setup. COCO 2017 outliers are inserted during training, with pasted regions ignored in the panoptic supervision. The anomaly score is one minus the clamped maximum known-class response. Unknown instances are extracted from pixels with anomaly scores of at least 0.8 using eight-connected components with a minimum area of 1,024 pixels.

RbA [40] Starting from the final checkpoint of the shared first-stage setup, we fine-tune only the class and mask prediction heads for 5,000 additional iterations. The second stage uses the same AdamW settings, COCO 2017 outliers are inserted during this stage. The squared-hinge RbA loss has weight 1.0, with inlier and outlier margins of −1.0 and −0.1, respectively. At inference, we use an RbA rejection threshold of −0.2 and a minimum unknown-component area of 200 pixels (Table 6). We apply 3 × 3 morphological opening followed by closing to the unknown masks, then extract four-connected components to obtain unknown-object instances.

## A.1.3 Parameter Sensitivity

We examine the sensitivity of RbA [40] to two inference parameters: the unknown-mask threshold and the minimum connected-component area. Table 6 reports unknown PQ on the validation split across nine parameter combinations. Performance ranges from 18.73 to 22.67 PQ, with the highest value obtained at a threshold of −0.20 and a minimum area of 200.

Table 6: RbA unknown PQ on the validation split for different unknown-mask thresholds and minimum connected-component areas.
<table><tr><td></td><td colspan="2">Minimum component area</td></tr><tr><td>Threshold</td><td>100</td><td>200 500</td></tr><tr><td>-0.30</td><td>18.73</td><td>20.51 21.55</td></tr><tr><td>-0.20</td><td>21.09</td><td>22.67 21.40</td></tr><tr><td>-0.10</td><td>20.41</td><td>21.25 21.81</td></tr></table>

The best evaluated setting is highlighted in bold.

## A.2 Open-Vocabulary Baselines

## A.2.1 Methods

We evaluate a diverse set of open-vocabulary panoptic segmentation methods spanning multiple design paradigms, including CLIP-based [62] approaches (FC-CLIP [42], MAFT+ [43], EOV-Seg [63]), diffusion-based segmentation (ODISE [45]), and prompt-driven segmentation (OpenWorldSAM [64]). Together, these methods provide representative coverage of established and emerging approaches to open-vocabulary panoptic segmentation.

FC-CLIP [42]. FC-CLIP uses a frozen convolutional CLIP backbone for dense prediction, directly leveraging pretrained vision-language representations without task-specific fine-tuning. Throughout our evaluations, we use the ConvNeXt-Large [65] variant.

MAFT+ [43]. MAFT+ improves vision-language alignment through collaborative vision-text representation learning and mask-aware fine-tuning of region-level features. We use the Large variant with the released panoptic segmentation checkpoint.

EOV-Seg [63]. EOV-Seg is an efficient open-vocabulary panoptic segmentation method that unifies mask prediction and classification in a streamlined architecture optimized for scalable inference. We use the ConvNeXt-Large variant with the released panoptic segmentation checkpoint.

ODISE [45]. ODISE performs open-vocabulary panoptic segmentation using representations derived from pretrained text-to-image diffusion models. We use the caption-supervised checkpoint. The label-supervised variant exhibits the unknown-object suppression discussed in Section A.2.4.

OpenWorldSAM [64]. OpenWorldSAM extends SAM2 [66] with text-driven mask generation, combining promptable segmentation with language-based object specification. We use the standard segmentation checkpoint.

All methods are evaluated using publicly available pretrained checkpoints, without architectural modifications or additional fine-tuning on our dataset, reflecting the open-vocabulary setting in which models are expected to generalize without task-specific training.

## A.2.2 Prompt Set

We define a fixed set of textual prompts, selected on the validation split and shared across all openvocabulary methods during test-set evaluation, to ensure a fair, controlled comparison among different architectures and inference paradigms. Each class is represented by a set of synonyms to improve the robustness of language-based matching. Since it is not feasible to enumerate all possible outof-distribution object categories, we additionally include a small set of generic prompts intended to capture common real-world OOD objects (e.g., debris, road hazard, foreign object). These prompts are chosen to provide meaningful semantic grounding while avoiding overly broad terms (e.g., object) that could lead to trivial assignments. The first term (bold) corresponds to the canonical class name.

• Car: car, cars, automobile, minivan, sedan, SUV, hatchback, coupe

• Truck: truck, trucks, motortruck, pickup truck, cargo van, delivery van, sprinter van

• Bus: bus, buses, autobus, motorbus, motorcoach, omnibus

• Other large vehicle: other large vehicle, large vehicle, mobile crane, crane truck, construction vehicle, excavator, bulldozer, camper van, motorhome

• Bicycle: bicycle, bicycles, bike, bikes

• Motorcycle: motorcycle, motorcycles, motorbike, scooter, moped, motor scooter

• Trailer: trailer, semi-trailer, caravan

• Pedestrian: pedestrian, person, people, walker, child, girl, boy, woman, man, children, girls, boys, women, men, lady, guy, ladies, guys, clothes

• Cyclist: cyclist, bicyclist, bicyclists, bike rider

• Motorcyclist: motorcyclist, motorcyclists, motorcycle rider

• Construction cone: construction cone, traffic cone, road cone, pylon, construction cone pole

• Pole: pole, poles, post, utility pole, lamp post, light pole, street lamp, street light

• Stroller: stroller, pram, wheelchair, shopping cart, luggage, trolley

• Sign: sign, signs, signboard, road sign, traffic sign, street sign, street name sign, road name sign, warning sign, construction sign

• Traffic light: traffic light, traffic lights, traffic signal

• Building: building, buildings, edifice, edifices, house, shop, store, storefront, warehouse, garage

• Road: road, roads, street, streets, route, highway, asphalt, tarmac

• Lane marker: lane marker, lane marking, lane line

• Road marker: road marker, road marking, pavement marking

• Sidewalk: sidewalk, pavement, footpath, walkway

• Vegetation: vegetation, tree, trees, plant, plants, bushes, foliage, flora, grass, lawn, shrub, hedge

• Sky: sky, clouds

• Unknown/OOD: foreign object, road debris, debris, obstacle, fallen object, stray object, loose object, lost cargo, spilled cargo, road hazard

## A.2.3 Prompt Sensitivity

We evaluate sensitivity of ODISE [45] to unknown-object prompts on WayPOP-v1 by comparing six prompt sets, ranging from generic object descriptions to road-hazard terms and our full benchmark vocabulary. Table 7 reports the resulting known and unknown PQ. Unknown PQ varies substantially across prompt sets, while known PQ remains comparatively stable. The full vocabulary used in our benchmark achieves the highest unknown PQ among the evaluated alternatives, highlighting the influence of prompt choice on open-set performance.

## A.2.4 Failure Modes and Limitations

We observe consistent limitations of current open-vocabulary methods when applied to out-ofdistribution (OOD) objects: mask-first approaches (e.g., FC-CLIP [42], MAFT+ [43], ODISE [45]) often localize OOD regions in semantic segmentation but fail to produce corresponding segments in the panoptic setting, while prompt-driven methods such as OpenWorldSAM [64] fail to reliably capture OOD objects in both semantic and panoptic outputs.

This behavior stems from how Mask2Former-style [30] methods couple class prediction with panoptic inference. Their classification head is trained on a fixed set of in-distribution categories and a void class that absorbs everything outside this taxonomy, effectively deciding whether a region belongs to one of the known classes rather than whether it is foreground. Open-vocabulary prompts do not influence this foreground selection, and OOD regions are therefore typically assigned high void probability. In panoptic inference, this void probability is used to filter and remove queries before mask aggregation. Since open-vocabulary classification is applied only after this foreground selection, queries covering OOD regions are often removed beforehand and cannot be assigned to OOD prompts. Open-vocabulary reasoning is thus restricted to regions already considered foreground under the model’s in-distribution classifier. In contrast, semantic inference discards the void class and aggregates predictions across all open-vocabulary prompts, forcing each pixel to be assigned to the most likely foreground category. This allows generic OOD prompts (e.g., debris, obstacle) to capture OOD regions despite low confidence.

Table 7: Sensitivity of ODISE to unknown-object prompts on the WayPOP-v1 test split.
<table><tr><td>Unknown prompts</td><td>Known PQ ↑</td><td>Unknown PQ ↑</td></tr><tr><td>unknown object</td><td>24.1</td><td>7.7</td></tr><tr><td>foreign object</td><td>24.2</td><td>12.4</td></tr><tr><td>object, thing, item, entity</td><td>24.0</td><td>17.7</td></tr><tr><td>unknown object, unseen object, novel object, foreign object</td><td>24.3</td><td>17.4</td></tr><tr><td>road hazard, road debris, fallen object, obstacle</td><td>24.4</td><td>23.3</td></tr><tr><td>Ours (full vocabulary in Section A.2.2)</td><td>24.1</td><td>30.9</td></tr></table>

Notably, ODISE trained with caption-based supervision performs substantially better on OOD objects than label-based variants. Unlike label supervision, which ties foreground selection to a fixed taxonomy, caption-based training induces a class-agnostic objectness signal that is not restricted to predefined categories. This allows regions corresponding to arbitrary objects to be retained as foreground, increasing the likelihood that OOD regions survive the filtering stage and can be assigned open-vocabulary labels.

In contrast, prompt-driven methods such as OpenWorldSAM exhibit a different limitation. Since segmentation is conditioned directly on textual prompts, OOD objects can only be detected if they are well captured by the provided prompts. Generic OOD prompts (e.g., debris, foreign object) often provide weak or ambiguous grounding, leading to missing or low-confidence masks in both semantic and panoptic outputs.

Overall, these results indicate that current open-vocabulary methods remain fundamentally limited in truly open-set scenarios. Mask-first approaches suppress OOD objects due to taxonomy-driven foreground selection, while prompt-driven methods fail to localize them without sufficiently specific prompts. Together, this suggests that robust OOD panoptic segmentation requires a notion of classagnostic objectness that is decoupled from fixed training taxonomies and does not rely on explicit prompt specification.

## B Dataset Construction and Validation

## B.1 Asset Curation and Filtering

Asset curation aims to select valid 3D objects outside the known Waymo PVPS taxonomy that are suitable for insertion into driving scenes. While Objaverse v1.0 assets are filtered using LVIS category labels, Objaverse-XL requires metadata-based screening due to the absence of comparable structured annotations. We detail this screening and the subsequent visual verification below.

Metadata-based screening. We use Qwen3-14B [59] to screen Objaverse-XL assets based on their available metadata. The prompt includes the known Waymo PVPS categories, Qwen-generated synonyms, and examples of assets to retain or exclude. This supports identifying semantic overlap beyond exact keyword matches, including vehicle brands and subtypes. For each asset, the model predicts a primary identity, semantic and quality flags, known-class overlap indicators, and size and insertability estimates. It also returns a keep/review/drop recommendation, confidence, and a brief rationale. These outputs are intended to narrow the candidate pool for subsequent visual verification rather than determine final asset selection.

Visual Verification and Grouping. Retained candidates are rendered from three canonical viewpoints and grouped into coarse semantic families using a second metadata-based Qwen pass. These groupings facilitate browsing through a verification interface, where we manually inspect candidates for visual identity, mesh quality, and semantic overlap with known classes. Candidates with unsuitable or degenerate geometry, poor mesh quality, or known-class overlap are discarded. Final asset selection therefore remains human.

Limitations and Alternatives. Metadata-based screening depends on descriptions that may be incomplete or misleading. Since Qwen does not inspect mesh geometry, its predicted identities and size estimates can disagree with the rendered assets. To address these limitations, we investigated VLM-based captioning of three rendered viewpoints per asset, but found the generated captions insufficiently reliable for filtering. In addition, this alternative also requires rendering every candidate before screening, increasing the cost of curation. Fully manual filtering, on the other hand, is impractical at the scale of Objaverse-XL. We therefore combine metadata-based screening with manual visual verification.

## B.2 Dataset Composition and Statistics

## B.2.1 Dataset Structure and Splits

WayPOP is constructed from 20 Waymo PVPS validation sequences, each divided into two temporal clips and augmented with objects from Objaverse v1.0 and Objaverse-XL. This yields 40 base clips and 80 augmented clips. The benchmark consists of three splits: validation (6 V1 clips from 3 sequences), WayPOP-v1 (34 test clips from the remaining 17 sequences), and WayPOP-XL (40 test clips from all 20 sequences). The V1 validation and test splits therefore have no overlapping source sequences, while WayPOP-XL also includes the validation driving sequences with different inserted objects. Table 8 summarizes the resulting camera-image, unknown-mask, and unknown-track counts.

Table 8: Dataset statistics for the WayPOP validation and test splits.
<table><tr><td>Split</td><td>Partition</td><td>Camera Images</td><td>Unknown Masks</td><td>Unknown Tracks</td></tr><tr><td>Validation</td><td>Val</td><td>1,635</td><td>2,559</td><td>48</td></tr><tr><td>WayPOP-v1</td><td>Test</td><td>9,015</td><td>22,766</td><td>439</td></tr><tr><td>WayPOP-XL</td><td>Test</td><td>10,650</td><td>28,127</td><td>493</td></tr><tr><td>Total</td><td></td><td>21,300</td><td>53,452</td><td>980</td></tr></table>

Camera images count individual camera observations, unknown masks count per-image instance annota tions, and unknown tracks count distinct object identities associated across time and camera views.

## B.2.2 Asset Composition

Our curated object pool comprises approximately 670 unique assets: 270 from Objaverse v1.0 and 400 from Objaverse-XL. Objaverse v1.0 examples include alligators, cupboards, guitars, kegs, drones, and trampolines. For Objaverse-XL, identities inferred from metadata include geometric sculptures, four-legged robots, dragons, fantasy aircraft, wooden crates, and asteroids. Note that these descriptions are approximate and may not always match the rendered geometry.

Maximum object dimensions are centered near 1 m in both sources (Figure 5). For Objaverse v1.0, the median maximum dimension is 0.87 m (interquartile range: 0.51–1.36 m; 10th–90th percentiles: 0.35–2.04 m), compared with 1.06 m for Objaverse-XL (0.95–1.20 m and 0.75–1.40 m, respectively). Objaverse v1.0 exhibits a broader size distribution and a longer tail towards larger objects, while Objaverse-XL is more concentrated around 1 m.

## B.3 Visual Integration and Pipeline Ablations

To assess the contribution of the visual integration stages described in Section 3.3, we compare four cumulative pipeline variants: rendering without lighting adaptation or occlusion handling, lighting adaptation alone, lighting with occlusion handling, and the full pipeline with additional photometric harmonization. Table 9 compares the resulting images with the original Waymo observations using

![](images/e3b9fcee9c34b8b240c6ecfc5072e9fcff54761ca96d2a90379170919c929cb5.jpg)  
Figure 5: Distributions of maximum object dimensions by asset source, estimated from reconstructed object point clouds. Histograms are normalized separately for each source and use shared logarithmically spaced bins on a logarithmic horizontal axis.

Fréchet Inception Distance (FID) and Kernel Inception Distance (KID), while Table 10 reports the corresponding VI-OSPS performance of RbA [40] and Mask2Anomaly [39] on WayPOP-v1.

Table 9: Visual distribution comparison between pipeline variants and original Waymo images on WayPOP-v1.
<table><tr><td>Pipeline Variant</td><td>FID↓</td><td> $\mathbf { K I D } \times 1 0 ^ { 3 } \downarrow$ </td></tr><tr><td>No lighting adaptation, no occlusion</td><td>14.762</td><td> $0 . 5 4 4 \pm 0 . 1 1 5$ </td></tr><tr><td>Lighting adaptation, no occlusion</td><td>11.804</td><td> $0 . 2 0 6 \pm 0 . 0 5 6$ </td></tr><tr><td>Lighting + occlusion</td><td>11.676</td><td> $0 . 1 8 1 \pm 0 . 0 6 1$ </td></tr><tr><td>Full: lighting + occlusion + harmonization</td><td>9.745</td><td> $\mathbf { 0 . 0 5 7 \pm 0 . 0 5 6 }$ </td></tr></table>

Variants represent cumulative pipeline additions. Lower FID and KID as the mean ± standard deviation indicate smaller distribution differences. Best values are highlighted in bold.

Table 10: VI-OSPS performance across cumulative pipeline variants on a development version of WayPOP-v1.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pipeline Variant</td><td colspan="3">Known</td><td colspan="3">Unknown</td></tr><tr><td>PQ↑</td><td>SQ↑</td><td>RQ↑</td><td>PQ↑</td><td>SQ↑</td><td>RQ↑</td></tr><tr><td rowspan="4">RbA [40]</td><td>No lighting adaptation, no occlusion</td><td>44.1</td><td>69.8</td><td>53.7</td><td>38.0</td><td>83.4</td><td>45.6</td></tr><tr><td>Lighting adaptation, no occlusion</td><td>44.1</td><td>69.9</td><td>53.6</td><td>33.3</td><td>80.2</td><td>41.6</td></tr><tr><td>Lighting + occlusion</td><td>44.0</td><td>69.5</td><td>53.5</td><td>32.4</td><td>79.8</td><td>40.7</td></tr><tr><td>Full pipeline</td><td>43.7</td><td>69.7</td><td>53.0</td><td>29.9</td><td>79.1</td><td>37.8</td></tr><tr><td rowspan="4">Mask2Anomaly [39]</td><td>No lighting adaptation, no occlusion</td><td>47.4</td><td>77.3</td><td>57.5</td><td>38.4</td><td>81.4</td><td>47.2</td></tr><tr><td>Lighting adaptation, no occlusion</td><td>47.1</td><td>76.9</td><td>57.1</td><td>29.4</td><td>79.5</td><td>37.0</td></tr><tr><td>Lighting + occlusion</td><td>47.0</td><td>76.9</td><td>56.9</td><td>27.8</td><td>79.7</td><td>34.9</td></tr><tr><td>Full pipeline</td><td>46.8</td><td>76.8</td><td>56.8</td><td>24.3</td><td>78.7</td><td>30.9</td></tr></table>

The full pipeline includes lighting adaptation, occlusion handling, and harmonization.

Adding lighting adaptation, occlusion handling, and harmonization progressively reduces FID and KID, indicating a smaller distribution gap to the original Waymo images. At the same time, unknown PQ decreases from 38.0 to 29.9 for RbA and from 38.4 to 24.3 for Mask2Anomaly, while known PQ remains comparatively stable. The decrease in unknown performance is more pronounced in RQ than SQ, suggesting that the integration stages primarily affect unknown-object recognition rather than mask alignment. Together, these trends are consistent with a reduction in visual artifacts that could otherwise provide shortcuts for identifying inserted objects as unknown, making them more challenging for the evaluated segmentation methods.

Nevertheless, these aggregate comparisons do not establish artifact-free integration or identify the cues exploited by the segmentation methods. Inserted objects occupy only a small fraction of each image, limiting the sensitivity of whole-image FID and KID to local inconsistencies. Metadata-based lighting and photometric harmonization also cannot fully reproduce scene-specific shadows, indirect illumination, material appearance, or camera response, leaving a residual photometric gap.

## B.4 Temporal and Geometric Consistency

Since WayPOP inserts world-static objects, their geometry should remain consistent under the recorded ego motion, with residual measurement errors comparable to those of native static Waymo structures. To assess this, we transform mask-associated LiDAR points between consecutive frames using the recorded SE(3) ego poses and measure residual point-set velocity and surface mismatch. We compute the same statistics for traffic signs and independently verified stationary vehicles as a reference for measurement variability.

The results in Table 11 show broadly comparable residuals for inserted and native static objects across ego-vehicle speed ranges, supporting the temporal and geometric consistency of our insertions.

Table 11: Temporal and geometric consistency under ego motion.
<table><tr><td rowspan="2"></td><td colspan="2">WayPOP-v1.0</td><td colspan="2">WayPOP-XL</td></tr><tr><td>Inserted</td><td>Native reference</td><td>Inserted</td><td>Native reference</td></tr><tr><td>Ego speed 1-5 m/s</td><td>0.165/0.052</td><td>0.174/0.071</td><td>0.163/0.056</td><td>0.174/0.071</td></tr><tr><td>5-10m/s</td><td>0.250/0.078</td><td>0.191/0.079</td><td>0.246/0.082</td><td>0.187/0.077</td></tr><tr><td>&gt;10 m/s</td><td>0.227/0.083</td><td>0.220/0.100</td><td>0.286/0.108</td><td>0.239/0.104</td></tr></table>

Each entry reports residual velocity (m/s) / surface mismatch (m). Native references comprise traffic signs and independently verified stationary vehicles from Waymo. At ego speeds above 10 m/s, the size-comparable native stationary-vehicle subsets exhibit residual velocities of 0.245 m/s (v1.0) and 0.274 m/s (XL), closely matching the corresponding inserted-object measurements.

Future extensions with dynamic object insertions and occupancy annotations could support evaluation of semantic occupancy forecasting [67].

## C Additional Experiments

## C.1 Cross-Dataset Segmentation Comparison

To contextualize segmentation performance on WayPOP, we compare RbA [40] across PANIC, COCO with 20 % withheld classes, and WayPOP-v1.0 in Table 12. For PANIC, we use the official Cityscapes-trained RbA checkpoint. The COCO result follows the corresponding open-set protocol. RbA achieves unknown PQ of 56.5 on PANIC, 24.8 on COCO, and 31.9 on WayPOP-v1.

Table 12: RbA performance across open-set segmentation benchmarks.
<table><tr><td colspan="2"></td><td colspan="2">PQ↑</td></tr><tr><td>Dataset</td><td>Image domain</td><td>Known</td><td>Unknown</td></tr><tr><td>PANIC</td><td>Driving scenes</td><td></td><td>56.5</td></tr><tr><td>COCO (20 % withheld classes)</td><td>Generic images</td><td>47.0</td><td>24.8</td></tr><tr><td>WayPOP-v1.0</td><td>Driving scenes</td><td>45.3</td><td>31.9</td></tr></table>

PANIC reports unknown PQ only. Differences in image domains, class taxonomies, and unknown-object definitions preclude a controlled comparison of benchmark difficulty.

Although PANIC and WayPOP both depict driving scenes, their taxonomies and unknown-object definitions differ, while COCO additionally covers a broader image domain. These results suggest that WayPOP poses a meaningful challenge even for established open-set segmentation methods.

## C.2 RGB-LiDAR Segmentation

Additional sensing modalities provide informative cues for scene understanding beyond RGB [68, 69]. In particular, LiDAR-camera fusion combines geometric and visual information for panoptic segmentation [70]. To explore the utility of the aligned LiDAR measurements provided by WayPOP, we augment RbA [40] with RGB-LiDAR feature fusion. Separate ResNet-50 [71] encoders process RGB images and projected LiDAR range, height, and validity channels. We combine their features through addition at strides 4 and 8 and RGB-guided cross-attention at strides 16 and 32, retaining the original Mask2Former [30] decoder and RbA unknown-object scoring. LiDAR insertion updates range measurements but does not synthesize intensity or elongation for inserted surfaces. We therefore restrict the LiDAR inputs to range, height, and validity.

Table 13: RGB–LiDAR fusion for open-set panoptic segmentation on WayPOP-v1.
<table><tr><td colspan="2">Input</td><td colspan="2">PQ↑</td></tr><tr><td>RGB</td><td>LiDAR</td><td>Known</td><td>Unknown</td></tr><tr><td>√</td><td>x</td><td>45.3</td><td>31.9</td></tr><tr><td>√</td><td>√</td><td>46.8</td><td>34.1</td></tr></table>

Both configurations use RbA for unknown-object scoring.

As shown in Table 13, adding LiDAR improves known PQ from 45.3 to 46.8 and unknown PQ from 31.9 to 34.1. These exploratory results suggest that geometric information can complement camera inputs for both known and unknown segmentation.

## C.3 Additional Frontal-View Tracking Results

We compare frontal-view tracking performance for the strongest open-set segmentation methods across WayPOP-v1.0 and WayPOP-XL in Table 14. This comparison examines how the two asset sources affect segmentation and association performance.

Table 14: FV-OSPT results on open-set methods across WayPOP-v1 and WayPOP-XL.
<table><tr><td></td><td></td><td colspan="6">WayPOP-v1</td><td colspan="6">WayPOP-XL</td></tr><tr><td></td><td></td><td colspan="3">Known</td><td colspan="3">Unknown</td><td colspan="3">Known</td><td colspan="3">Unknown</td></tr><tr><td></td><td>Tracker</td><td></td><td>STQ↑ AQ↑</td><td>mIoU ↑</td><td>STQ↑ AQ↑</td><td></td><td>mIoU↑</td><td>STQ↑</td><td>AQ↑</td><td>mIoU↑</td><td>STQ↑ AQ↑</td><td></td><td>mIoU ↑</td></tr><tr><td></td><td>TrackTrack [46]</td><td>16.7</td><td>5.6</td><td>49.6</td><td>26.4</td><td>16.0</td><td>43.5</td><td>16.6</td><td>5.3</td><td>51.7</td><td>31.0</td><td>17.0</td><td>56.5</td></tr><tr><td></td><td>BoT-SORT [47]</td><td>16.5</td><td>5.5</td><td>49.6</td><td>25.4</td><td>14.8</td><td>43.5</td><td>16.7</td><td>5.4</td><td>51.7</td><td>29.7</td><td>15.6</td><td>56.5</td></tr><tr><td>0D] 10]</td><td>MASA [50]</td><td>15.8</td><td>5.0</td><td>49.6</td><td>30.0</td><td>20.7</td><td>43.5</td><td>15.3</td><td>4.5</td><td>51.7</td><td>33.8</td><td>20.3</td><td>56.5</td></tr><tr><td></td><td>DEVA [51]</td><td>19.1</td><td>7.4</td><td>49.6</td><td>29.0</td><td>19.3</td><td>43.5</td><td>18.5</td><td>6.6</td><td>51.7</td><td>32.2</td><td>18.4</td><td>56.5</td></tr><tr><td></td><td>TrackTrack [46]</td><td>18.4</td><td>6.8</td><td>49.4</td><td>26.3</td><td>18.1</td><td>38.3</td><td>18.1</td><td>6.4</td><td>51.5</td><td>32.1</td><td>18.9</td><td>54.4</td></tr><tr><td></td><td>BoT-SORT [47]</td><td>17.4</td><td>6.1</td><td>49.4</td><td>25.2</td><td>16.6</td><td>38.3</td><td>17.3</td><td>5.8</td><td>51.5</td><td>30.6</td><td>17.2</td><td>54.4</td></tr><tr><td>R4] 40]</td><td>MASA [50]</td><td>16.2</td><td>5.3</td><td>49.4</td><td>29.6</td><td>22.9</td><td>38.3</td><td>16.1</td><td>5.0</td><td>51.5</td><td>34.7</td><td>22.1</td><td>54.4</td></tr><tr><td></td><td>DEVA [51]</td><td>20.6</td><td>8.5</td><td>49.4</td><td>28.7</td><td>21.5</td><td>38.3</td><td>19.8</td><td>7.6</td><td>51.5</td><td>33.2</td><td>20.3</td><td>54.4</td></tr></table>

Known-class performance remains similar across the two variants, and MASA achieves the highest unknown STQ and AQ for both open-set models on both splits. PoDS achieves higher unknown STQ than RbA on WayPOP-v1.0 for every tracker, whereas RbA leads on WayPOP-XL. Unknown STQ is consistently higher on WayPOP-XL, accompanied by substantially higher unknown mIoU, while AQ changes are smaller and vary across trackers. For example, RbA with MASA achieves unknown STQ of 29.6 on v1.0 and 34.7 on XL, with AQ of 22.9 and 22.1, respectively. The higher unknown tracking scores on XL therefore primarily reflect improved segmentation rather than more accurate association.

## C.4 Perfect-Mask Tracking Analysis

To isolate association difficulty from segmentation errors, we evaluate frontal-view tracking with ground-truth masks and classes. Even with perfect frame-local predictions, identities must be maintained under viewpoint changes, occlusion, and image-space displacement induced by ego motion. For adjacent observations of the same unknown object, the median mask IoU is 0.1539 and the median centroid displacement is 13.84 pixels.

We additionally evaluate a privileged static-object prior that uses the known staticity of inserted objects to relink unknown-object tracklets. Ground-contact points are estimated from the bottom 2 % of each mask, backprojected onto the ground plane using camera calibration, and transformed into a common world coordinate frame using the recorded ego poses. Non-overlapping tracklets are relinked using Hungarian assignment, with a maximum temporal gap of 1 s, world-space separation of 1.5 m, reprojection error of 120 pixels per direction, and symmetric mask-area ratio of 4. Ambiguous assignments are rejected using a minimum cost margin of 0.15.

Table 15: Frontal-view tracking with ground-truth masks and classes on WayPOP-v1.
<table><tr><td colspan="2"></td><td colspan="2">Known</td><td colspan="2">Unknown</td></tr><tr><td>Tracker</td><td>Static prior</td><td>STQ↑</td><td>AQ↑</td><td>STQ↑</td><td>AQ↑</td></tr><tr><td rowspan="2">DEVA [51]</td><td>x</td><td>88.64</td><td>78.57</td><td>77.37</td><td>59.86</td></tr><tr><td>√</td><td>88.64</td><td>78.57</td><td>82.68</td><td>68.35</td></tr><tr><td rowspan="2">SAM 3.1 [72]</td><td>x</td><td>88.12</td><td>77.66</td><td>56.89</td><td>32.36</td></tr><tr><td>√</td><td>88.12</td><td>77.66</td><td>77.43</td><td>59.95</td></tr></table>

All configurations use perfect frame-local masks and classes (mIoU = 100). The static prior uses ego-motion compensation to relink unknown-object tracklets; known-class results remain unchanged by construction.

As shown in Table 15, unknown AQ improves from 59.86 to 68.35 for DEVA and from 32.36 to 59.95 for SAM 3.1 when using this prior. These gains indicate that static-scene geometry can help recover association continuity, while the remaining errors show that perfect masks and known staticity do not fully resolve tracking.

## References

[1] Maximilian Luz, Rohit Mohan, Thomas Nürnberg, Yakov Miron, Daniele Cattaneo, and Abhinav Valada. Latent gaussian splatting for 4d panoptic occupancy tracking. IEEE Robotics and Automation Letters, 2026.

[2] Whye Kit Fong, Rohit Mohan, Juana Valeria Hurtado, Lubing Zhou, Holger Caesar, Oscar Beijbom, and Abhinav Valada. Panoptic nuScenes: A large-scale benchmark for LiDAR panoptic segmentation and tracking. IEEE Robotics and Automation Letters (RA-L), 7(2):3795–3802, 2022.

[3] Juana Valeria Hurtado, Rohit Mohan, Wolfram Burgard, and Abhinav Valada. MOPT: Multi-object panoptic tracking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2020.

[4] Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. Video panoptic segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[5] Jieru Mei, Alex Zihao Zhu, Xinchen Yan, Hang Yan, Siyuan Qiao, Liang-Chieh Chen, and Henrik Kretzschmar. Waymo Open Dataset: Panoramic video panoptic segmentation. In European Conference on Computer Vision (ECCV), 2022.

[6] Peter Pinggera, Sebastian Ramos, Stefan Gehrig, Uwe Franke, Carsten Rother, and Rudolf Mester. Lost and Found: Detecting small road hazards for self-driving vehicles. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2016.

[7] Robin Chan, Krzysztof Lis, Svenja Uhlemeyer, Hermann Blum, Sina Honari, Roland Siegwart, Pascal Fua, Mathieu Salzmann, and Matthias Rottmann. SegmentMeIfYouCan: A benchmark for anomaly segmentation. In Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS D&B), 2021.

[8] Jaedong Hwang, Seoung Wug Oh, Joon-Young Lee, and Bohyung Han. Exemplar-based open-set panoptic segmentation network. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[9] Matteo Sodano, Federico Magistri, Jens Behley, and Cyrill Stachniss. Open-world panoptic segmentation. arXiv preprint arXiv:2412.12740, 2024. Accepted to the International Journal of Robotics Research (IJRR).

[10] Rohit Mohan, Kiran Kumaraswamy, Juana Valeria Hurtado, Kürsat Petek, and Abhinav Valada. Panoptic out-of-distribution segmentation. IEEE Robotics and Automation Letters (RA-L), 9(5):4075–4082, 2024.

[11] Daniel Bogdoll, Iramm Hamdard, Lukas Namgyu Rößler, Felix Geisler, Muhammed Bayram, Felix Wang, Jan Imhof, Miguel de Campos, Anushervon Tabarov, Yitian Yang, Martin Gontscharow, Hanno Gottschalk, and J. Marius Zöllner. AnoVox: A benchmark for multimodal anomaly detection in autonomous driving. In European Conference on Computer Vision Workshops (ECCVW), 2024.

[12] Alexey Nekrasov, Malcolm Burdorf, Stewart Worrall, Bastian Leibe, and Julie Stephany Berrio Perez. Spotting the unexpected (STU): A 3D LiDAR dataset for anomaly segmentation in autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[13] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3D objects. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[14] Rohit Mohan, Jose Arce, Sassan Mokhtar, Daniele Cattaneo, and Abhinav Valada. Syn-mediverse: a multimodal synthetic dataset for intelligent scene understanding of healthcare facilities. IEEE Robotics and Automation Letters, 9(8):7094–7101, 2024.

[15] Ahmed Rida Sekkat, Rohit Mohan, Oliver Sawade, Elmar Matthes, and Abhinav Valada. Amodalsynthdrive: A synthetic amodal perception dataset for autonomous driving. IEEE Robotics and Automation Letters, 9 (11):9597–9604, 2024.

[16] Hermann Blum, Paul-Edouard Sarlin, Juan Nieto, Roland Siegwart, and Cesar Cadena. Fishyscapes: A benchmark for safe semantic segmentation in autonomous driving. In IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2019.

[17] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes dataset for semantic urban scene understanding. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[18] Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou, Joseph Kwon, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. Scaling out-of-distribution detection for real-world settings. In International Conference on Machine Learning (ICML), 2022.

[19] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. BDD100K: A diverse driving dataset for heterogeneous multitask learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[20] Kira Maag, Robin Chan, Svenja Uhlemeyer, Kamil Kowol, and Hanno Gottschalk. Two video data sets for tracking and retrieval of out of distribution objects. In Asian Conference on Computer Vision (ACCV), 2022.

[21] Aasheesh Singh, Aditya Kamireddypalli, Vineet Gandhi, and K Madhava Krishna. LiDAR guided small obstacle segmentation. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020.

[22] Petra Bevandic, Ivan Krešo, Marin Orši ´ c, and Siniša Šegvi ´ c. Simultaneous semantic segmentation and ´ outlier detection in presence of domain shift. In German Conference on Pattern Recognition (GCPR), 2019.

[23] Matej Grcic, Petra Bevandi ´ c, and Siniša Šegvi ´ c. Dense open-set recognition with synthetic outliers ´ generated by Real NVP. In International Conference on Computer Vision Theory and Applications (VISAPP), 2021.

[24] Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulo, and Peter Kontschieder. The Mapillary Vistas dataset for semantic understanding of street scenes. In IEEE International Conference on Computer Vision (ICCV), 2017.

[25] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA: An open urban driving simulator. In Conference on Robot Learning (CoRL), 2017.

[26] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. Panoptic segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In European Conference on Computer Vision (ECCV), 2014.

[28] Rohit Mohan and Abhinav Valada. EfficientPS: Efficient panoptic segmentation. International Journal of Computer Vision (IJCV), 129(5):1551–1579, 2021.

[29] Bowen Cheng, Maxwell D Collins, Yukun Zhu, Ting Liu, Thomas S Huang, Hartwig Adam, and Liang-Chieh Chen. Panoptic-DeepLab: A simple, strong, and fast baseline for bottom-up panoptic segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[30] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[31] Mark Weber, Jun Xie, Maxwell Collins, Yukun Zhu, Paul Voigtlaender, Hartwig Adam, Bradley Green, Andreas Geiger, Bastian Leibe, Daniel Cremers, Aljoša Ošep, Laura Leal-Taixé, and Liang-Chieh Chen. STEP: Segmenting and tracking every pixel. In Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS D&B), 2021.

[32] Alexander Jaus, Kailun Yang, and Rainer Stiefelhagen. Panoramic panoptic segmentation: Insights into surrounding parsing for mobile agents via unsupervised contrastive learning. IEEE Transactions on Intelligent Transportation Systems (T-ITS), 24(4):4438–4453, 2023.

[33] Rohit Mohan, Julia Hindel, Florian Drews, Claudius Gläser, Daniele Cattaneo, and Abhinav Valada. Openset lidar panoptic segmentation guided by uncertainty-aware learning. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2224–2231. IEEE, 2025.

[34] Duy Tho Le, Chenhui Gou, Stavya Datta, Hengcan Shi, Ian Reid, Jianfei Cai, and Hamid Rezatofighi. JRDB-PanoTrack: An open-world panoptic segmentation and tracking robotic dataset in crowded human environments. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[35] Kaican Li, Kai Chen, Haoyu Wang, Lanqing Hong, Chaoqiang Ye, Jianhua Han, Yukuai Chen, Wei Zhang, Chunjing Xu, Dit-Yan Yeung, Xiaodan Liang, Zhenguo Li, and Hang Xu. CODA: A real-world road corner case dataset for object detection in autonomous driving. In European Conference on Computer Vision (ECCV), 2022.

[36] Zhongyu Xia, Jishuo Li, Zhiwei Lin, Xinhao Wang, Yongtao Wang, and Ming-Hsuan Yang. OpenAD: Open-world autonomous driving benchmark for 3D object detection. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track (NeurIPS D&B), 2025.

[37] Stefano Gasperini, Alvaro Marcos-Ramiro, Michael Schmidt, Nassir Navab, Benjamin Busam, and Federico Tombari. Segmenting known objects and unseen unknowns without prior knowledge. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[38] Sebastian Schmidt, Julius Körner, Dominik Fuchsgruber, Stefano Gasperini, Federico Tombari, and Stephan Günnemann. Prior2Former – evidential modeling of mask transformers for assumption-free open-world panoptic segmentation. In IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[39] Shyam Nandan Rai, Fabio Cermelli, Barbara Caputo, and Carlo Masone. Mask2Anomaly: Mask transformer for universal open-set segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 46(12):9286–9302, 2024.

[40] Nazir Nayal, Mısra Yavuz, João F. Henriques, and Fatma Güney. RbA: Segmenting unknown regions rejected by all. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[41] Yao Lu, Rohit Mohan, Florian Drews, Yakov Miron, and Abhinav Valada. Hyp2Former: Hierarchy-aware hyperbolic embeddings for open-set panoptic segmentation. arXiv preprint arXiv:2605.02580, 2026.

[42] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and Liang-Chieh Chen. Convolutions die hard: Openvocabulary segmentation with single frozen convolutional CLIP. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[43] Siyu Jiao, Hongguang Zhu, Jiannan Huang, Yao Zhao, Yunchao Wei, and Humphrey Shi. Collaborative vision-text representation optimizing for open-vocabulary segmentation. In European Conference on Computer Vision (ECCV), 2024.

[44] Xiaoyi Dong, Jianmin Bao, Yinglin Zheng, Ting Zhang, Dongdong Chen, Hao Yang, Ming Zeng, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, and Nenghai Yu. MaskCLIP: Masked self-distillation advances contrastive language-image pretraining. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[45] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Openvocabulary panoptic segmentation with text-to-image diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[46] Kyujin Shim, Kangwook Ko, Yujin Yang, and Changick Kim. Focusing on tracks for online multi-object tracking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[47] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. BoT-SORT: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022.

[48] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. ByteTrack: Multi-object tracking by associating every detection box. In European Conference on Computer Vision (ECCV), 2022.

[49] Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. Observation-centric SORT: Rethinking SORT for robust multi-object tracking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[50] Siyuan Li, Lei Ke, Martin Danelljan, Luigi Piccinelli, Mattia Segu, Luc Van Gool, and Fisher Yu. Matching anything by segmenting anything. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[51] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[52] Junjie Jiang, Zelin Wang, Manqi Zhao, Yin Li, and Dongsheng Jiang. SAM2MOT: A novel paradigm of multi-object tracking by segmentation. In AAAI Conference on Artificial Intelligence (AAAI), 2026.

[53] Maximilian Luz, Rohit Mohan, Ahmed Rida Sekkat, Oliver Sawade, Elmar Matthes, Thomas Brox, and Abhinav Valada. Amodal optical flow. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 14677–14684. IEEE, 2024.

[54] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? The KITTI vision benchmark suite. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

[55] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurélien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo Open Dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[56] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-XL: A universe of 10M+ 3D objects. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track (NeurIPS D&B), 2023

[57] Agrim Gupta, Piotr Dollár, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[58] Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3D captioning with pretrained models. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track (NeurIPS D&B), 2023.

[59] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[60] Martin A. Fischler and Robert C. Bolles. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM (CACM), 24(6): 381–395, 1981.

[61] Julian Jorge Andrade Guerreiro, Mitsuru Nakazawa, and Björn Stenger. PCT-Net: Full resolution image harmonization using pixel-wise color transformations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[62] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021.

[63] Hongwei Niu, Jie Hu, Jianghang Lin, Guannan Jiang, and Shengchuan Zhang. EOV-Seg: Efficient open-vocabulary panoptic segmentation. In AAAI Conference on Artificial Intelligence (AAAI), 2025.

[64] Shiting Xiao, Rishabh Kabra, Yuhang Li, Donghyun Lee, Joao Carreira, and Priyadarshini Panda. Open-WorldSAM: Extending SAM2 for universal image segmentation with language prompts. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[65] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[66] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. In International Conference on Learning Representations (ICLR), 2025.

[67] Riya Mohan, Juana Valeria Hurtado, Rohit Mohan, and Abhinav Valada. Forecastocc: Vision-based semantic occupancy forecasting. arXiv preprint arXiv:2602.08006, 2026.

[68] Julia Hindel, Rohit Mohan, Jelena Bratulic, Daniele Cattaneo, Thomas Brox, and Abhinav Valada. Label-´ efficient lidar semantic segmentation with 2d-3d vision transformer adapters. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4115–4122. IEEE, 2025.

[69] Juana Valeria Hurtado, Rohit Mohan, and Abhinav Valada. Hyperspectral adapter for semantic segmentation with vision foundation models. IEEE Robotics and Automation Letters, 2026.

[70] Rohit Mohan, Florian Drews, Yakov Miron, Daniele Cattaneo, and Abhinav Valada. Up-fuse: Uncertaintyguided lidar-camera fusion for 3d panoptic segmentation. arXiv preprint arXiv:2602.19349, 2026.

[71] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[72] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. SAM 3: Segment anything with concepts. In International Conference on Learning Representations (ICLR), 2026.