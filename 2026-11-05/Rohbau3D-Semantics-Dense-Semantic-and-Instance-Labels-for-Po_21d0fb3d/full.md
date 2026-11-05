# Rohbau3D-Semantics: Dense Semantic and Instance Labels for Point Clouds of Indoor Shell Construction Scenes

Lukas Rauch

Thomas Braml

University of the Bundeswehr Munich, Germany

{lukas.rauch, thomas.braml}@unibw.de

## Abstract

We present the Rohbau3D-Semantics dataset, which adds dense semantic and instance-level labels to laser scans of the previously released Rohbau3D domain data from shell construction sites. The dataset comprises 504 point cloud scenes from 14 survey sites, each captured with a terrestrial laser scanner and stored in a scanner-centered local coordinate system. Unlike point clouds sampled from reconstructed meshes or synthetic building models, the data preserve the characteristic properties of real laser scanning, including non-uniform sampling, measurement artifacts, reflections, and fine geometric detail. Combined with manual labels, expert oversight during pointwise annotation, and full-scene review, this yields a high-quality benchmark for semantic scene understanding in complex built environments. The dataset includes 17 class labels of building components and one background class, along with instance annotations for object-level analysis. In ad dition to the 3D point clouds, the release also provides scripts for deriving geometrically aligned panoramic and cube-map representations of color, LiDAR reflection inten sity, depth, surface normals, semantic labels, and instance labels from the 3D arrays. We describe the acquisition pipeline, annotation protocol, quality-control process, and panorama-generation procedure, and provide benchmark splits and baseline experiments for semantic segmentation. By combining precise terrestrial laser scans, dense anno tations, and a wide variety of shell-construction scenes, Rohbau3D-Semantics bridges the gap among synthetic indoor datasets, mesh-derived benchmarks, and smaller construction-specific resources. The dataset is intended to support research on 3D scene understanding, constructionsite analytics, and multimodal scene analysis in the built environment.

Access and download the full dataset via github.com/RauchLukas/rohbau3d.

## 1. Introduction

Spatial scene understanding increasingly depends on benchmark datasets and verified validation data that capture complex environments in faithful digital representations. In recent years, major progress in 3D scene understanding has been driven by large datasets of indoor spaces. However, the individual works primarily focus on furnished residential and office interiors, using synthetic, CAD-derived scenes [32], RGB-D reconstructions [2, 4], or high-fidelity laser-scan reconstructions with aligned imagery [28]. Although these benchmarks have substantially advanced semantic segmentation and related tasks in general, they do not address the actual challenges associated with raw data from terrestrial laser scanners and LiDAR measurements of structural-shell conditions.

This limitation is particularly relevant to investigations of shell construction sites. Such scenes are characterized by dynamically changing conditions, temporary structures, stored materials, construction debris, and inconsistent lighting. Fundamental building elements are often unfinished, not yet installed, partially occluded by supporting structures, or concealed by protective coverings. This setting is captured by the German term Rohbau ["ro:baU], which denotes the stage of a building project in which only the essential structural shell has been completed. Publicly available datasets for shell construction environments remain scarce, despite the importance of this construction phase for digital documentation and progress monitoring [25], building information modeling (BIM) [3, 23], and quality assurance for building components [30]. Resources with dense pointwise semantic and instance annotations for an engineeringbased understanding of the framework structure are rarer still [19]. Such annotations are particularly valuable in construction settings because many downstream tasks require not only recognizing object categories, but also distinguishing individual elements. While semantic labels support the identification of structurally relevant classes, instancelevel annotations are essential for object-wise analysis, such as tracking the installation state of individual components [33], comparing as-built conditions against BIM models [11, 25], localizing defects, or reasoning about navigable and obstructed space [5].

![](images/d1498cc04b7f015ecab744719c4f47675e753d0636cafc26d297e9cb24b1969b.jpg)  
Figure 1. Rohbau3D-Semantics: Dataset features in both 3D and 2D-projected form. A) color, B) semantic classes, C) class instances, D) color, E) depth, F) reflection intensity, G) surface normals, H) semantic classes, I) class instances.

To address this gap, we present Rohbau3D-Semantics, an extension of the previously released Rohbau3D dataset [20]. The original Rohbau3D release provided terrestrial laser scans and associated metadata from diverse shell construction sites, but no semantic annotations. Rohbau3D-Semantics extends the raw data with point-wise labels, aligned 2D panoramic projections, benchmark splits, and baseline results for supervised 3D semantic segmentation. The instance labels and linked 2D representations are released to support future instance-level and multimodal research, but these tasks are not evaluated in the present experiments. Although rooted in the construction domain, the dataset may also be relevant to broader work in automation and robotics involving cluttered or structurally degraded built environments [10, 13].

## 2. Related Datasets

While several datasets support indoor 3D scene understanding, none captures the same technical scope of shell construction sites. ScanNet v2 [4] and S3DIS [1] are central real-world indoor benchmarks based on RGB-D reconstructions or registered point samples. ScanNet++ [28] adds high-end laser scans, DSLR images, and iPhone RGB-D sequences. These datasets, however, primarily depict completed, furnished interiors rather than scanner-centered raw terrestrial scans of unfinished structural shells.

Structured3D [32] provides virtual indoor scenes rendered from CAD models, while VASAD [9] samples synthetic point clouds from digital building models. Although useful for scalable training, their point distributions arise from rendering or sampling rather than physical sensing. LiDAR-Net [6] instead supplies real, unfiltered mobile-LiDAR data with uneven density, noise, and registration errors, but its taxonomy mainly describes common architectural elements and furnishings rather than components specific to structural shells.

Table 1. Comparison of representative indoor and construction datasets. Domain distinguishes indoor scenes (Interior), building infor mation models (BIM), and construction sites (Constr.). Extent gives the number of sites, areas, rooms, scenes, or scans; Reported scale uses each dataset’s native metric. Our reported area is the sum of the floor-plan footprints reconstructed independently on each scan. Overlapping regions observed from multiple scanner positions are counted repeatedly; the value measures cumulative observations rather than unique site floor area. Sem./Inst. indicates semantic/instance annotation availability. Modalities: P = point cloud, M = mesh, RGB = color, D = depth, I = intensity, N = normals, and IMU = inertial measurements. Type: R = real and S = synthetic data. “–” indicates that the corresponding annotation is unavailable.
<table><tr><td>Dataset</td><td>Domain</td><td>Extent</td><td>Reported scale</td><td>Sensors</td><td>Classes</td><td>Sem./Inst.</td><td>Modalities</td><td>R/S</td></tr><tr><td>S3DIS [1]</td><td>Interior</td><td>6 areas / 271 rooms</td><td>6,000 m2 / 215M pts</td><td>RGB-D</td><td>13</td><td>Y/Y</td><td>P, RGB, D, N</td><td>R</td></tr><tr><td>ScanNet v2 [4]</td><td>Interior</td><td>1,513 scans</td><td>2.5M frames</td><td>RGB-D</td><td>20</td><td>Y/Y</td><td>M, RGB, D</td><td>R</td></tr><tr><td>ScanNet++ v2 [21]</td><td>] Interior</td><td>1006 scenes</td><td>1.1M DSLR+10M iPhone frames TLS+DSLR+RGB-D</td><td></td><td>3.4k</td><td>Y/Y</td><td>P, M, RGB, D</td><td>R</td></tr><tr><td>LiDAR-Net [6]</td><td>Interior</td><td>14 areas / 465 rooms</td><td>30,000 m2 / 3.6B points</td><td>Mobile LiDAR</td><td>24</td><td>Y/Y</td><td>P, RGB</td><td>R</td></tr><tr><td>VASAD [9]</td><td>BIM</td><td>6 models</td><td> $6 2 { , } 0 0 0 \mathrm { m ^ { 2 } }$ </td><td>Sim. LiDAR</td><td>11</td><td>Y/N</td><td>P, M</td><td>S</td></tr><tr><td>Hilti-Oxford [31]</td><td>Constr.</td><td>2 sites / 16 sequences</td><td> $3 { , } 0 0 0 \mathrm { m } ^ { 2 }$ </td><td>LiDAR+cam.</td><td>一</td><td>N/N</td><td>P, RGB, IMU</td><td>R</td></tr><tr><td>ConSLAM [24]</td><td>Constr.</td><td>1 site / 5 sequences</td><td>1.40 km trajectories</td><td>LiDAR+cam.</td><td>一</td><td>N/N</td><td>P, RGB, IMU</td><td>R</td></tr><tr><td>ConSite [29]</td><td>Constr.</td><td>3 projects / 55 rooms</td><td>2,641 m² / 45.3M pts</td><td>TLS</td><td>13</td><td>Y/N</td><td>P, RGB</td><td>R</td></tr><tr><td>Ours</td><td>Constr.</td><td>14 sites / 504 scans</td><td>66,000 m2 / 12.5B pts</td><td>TLS</td><td>17+1</td><td>Y/Y</td><td>P, RGB, D, I, N</td><td>R</td></tr></table>

Public construction-site data remains scarce. Hilti-Oxford [31] and ConSLAM [24] combine cameras, LiDAR, and inertial sensing in construction environments, but target SLAM, registration, or progress monitoring and provide no semantic annotations [22]. ConSite [29] has recently provided TLS data with semantic labels for 13 classes of structural and temporal components, though without instance annotations and only on a small scale.

Table 1 makes the different dataset roles explicit: mobile construction-site datasets target localization and mapping without semantic or instance labels, whereas ConSite provides construction-specific semantic labels but no instances. Against this background, Rohbau3D-Semantics combines real construction-site data with point-wise semantic and instance annotations and linked 2D representations. Among the compared site-related datasets with explicitly reported annotated point totals, it provides the largest collection, with 504 standalone scans and 12.5 billion points.

## 3. Dataset Description

Rohbau3D-Semantics builds upon the previously published Rohbau3D dataset, in which the underlying terrestrial laser scans were released as point clouds under a CC BY 4.0 license [20]. The present work extends these data with dense point-wise semantic and instance annotations and a reproducible pipeline for deriving corresponding panoramic and cube-map representations. The original publication documents the data acquisition and preprocessing in detail. For completeness, this section summarizes the relevant processing steps before describing the new annotation workflow and the resulting dataset.

## 3.1. Data Collection and Processing

The source data were collected over a 1-year period at multiple construction sites in the Munich area, Germany. The survey sites were selected in collaboration with four construction companies and the Munich State Building Authority to ensure diversity in building typologies, construction conditions, and the occurrence of objects. As a result, the dataset covers a broad range of unfinished indoor environments, with varying surface materials, architectural features, and stages of construction progress visible.

![](images/5797905fc78932414ddefdad1ea992cf6ecf2c02286297f3f6274bf96800086b.jpg)  
Figure 2. Illustration of the terrestrial laser scanner device that was used to record the data, and a representative construction site scene in the background. The laser sensor head rotates simultaneously around two axes (yaw and roll), thereby mapping the space in a spherical grid.

All point clouds were acquired using a steady FARO Focus M70 terrestrial laser scanner shown in Figure 2. This device can capture up to 1 million spherical points per second using LiDAR technology and produce colored point clouds by superimposing spherical high-dynamicrange (HDR) images in post-processing. Construction management and supervision use comparable business-grade laser scanners to measure, document, and evaluate ongoing construction processes. The laser scanner is tripodmounted, which improves measurement quality compared

![](images/c5ceb3c5908cb6b0ee444cafaa4d21deaccce5e024553a03914126891943e24f.jpg)  
Figure 3. Illustration of reflection noise and ghost points from LiDAR sensing. Reflections typically exhibit nonsensical color textures in RGB space. Ghost points typically appear suspended in the air or near sharp edges. Black polygons highlight such regions that are assigned to the None class (gray).

to handheld devices.

This setup yields high point density, high geometric accuracy, and consistent acquisition results, while preserving characteristic sensing imperfections, such as occlusions, reflections, and measurement artifacts, as illustrated in Figure 3. Scan positions were selected individually on site by a trained civil engineer to mimic the process of creating asbuilt documentation. The individual scans were not registered into a common global coordinate frame. Instead, each scan is stored in its local scanner-centered coordinate system, with the sensor origin located at $x , y , z = ( 0 , 0 , 0 )$ in the world coordinate system. This representation ensures a consistent, standardized structure across all samples, and a frame-like field of view, comparable to that of mobile Li-DAR devices.

Preprocessing was intentionally kept minimal to preserve the raw measurement characteristics. The only geometric filtering step removed points located outside a sphere of radius 25 m centered at the scanner origin to suppress unstable long-range measurements. No resampling, mesh reconstruction, or geometric regularization was applied. Consequently, the dataset retains the irregular sampling pattern and sensor-specific artifacts of LiDAR sensing, which pose a key challenge for 3D data analysis of actual sensor data [20]. Site descriptions and file-format details are provided in the supplementary material; scene-wise feature compendia are available in the dataset repository [18].

## 3.2. Data Annotation

The proposed Rohbau3D-Semantics dataset uses a semantic taxonomy of 17 foreground classes and one background class. The class set was defined as a condensed selection of object categories that are both relevant to building information modeling (BIM) and exist frequently enough in the data to be represented as independent classes. This design avoids overly fine-grained labels for rare objects while taking into account the important components for modeling the building. The list of annotation classes defines: (0) None, (01) Ceiling, (02) Slanted Ceiling, (03) Ceiling Cutout, (04) Floor, (05) Wall, (06) Drywall, (07) Wall Cutout, (08) Parapet, (09) Beam, (10) Column, (11) Staircase, (12) Window

Cutout, (13) Window, (14) Door Rough Opening, (15) Facade Element, (16) Elevator Doors, and (17) TGA (technical building equipment), often referred to as mechanical electrical plumbing (MEP). Based on the functional unity and geometric continuity, individual objects are further clustered into distinct instances within each semantic segment.

Annotations were created with a professional subcontractor specialized in 3D data labeling. Before large-scale annotation began, the authors trained the provider using detailed guidelines with class definitions, visual examples, and pilot scenes. A condensed version of the finalized annotation guidelines is provided within the supplementary material. An initial validation round on randomly selected samples was then used to identify errors and refine the procedure. Throughout the process, the authors supervised the annotation workflow, and ambiguous cases were resolved in collaboration with a construction-domain expert to ensure consistent labeling. Point-wise semantic ground-truth labels were produced through interactive point-level selection and iterative refinement. Unlike segment-based workflows in which precomputed clusters form the final labeling units [2, 4], component boundaries were checked and adjusted directly at the point level. Class-wise instanceobject segmentations were performed on this dataset using iterative polygon-based trimming in the 3D viewer. Point subsets were interactively selected and repeatedly refined from multiple viewing directions, enabling components to be separated in true 3D space with actual 3D geometric knowledge and reducing misinterpretations caused by occlusions or scan artifacts that may appear plausible from a single frontal view but become identifiable as noise from lateral perspectives. This is especially important near reflective surfaces and sharp edges, where LiDAR artifacts can appear plausible in 2D projections but reveal geometric deviations in the 3D representation.

Point-wise instance annotations were defined primarily by geometric continuity rather than by semantic class membership alone. In general, spatially separated elements of the same semantic class were labeled as distinct instances, for example, separate walls within the same room or adjacent windows separated by a narrow wall segment. Occlusion, limited visibility, or local scan artifacts caused an instance split only when the common physical connection remained unclear; disconnected observations were merged when their functional or geometric unity could be inferred from the scene context. Conversely, touching or intersecting elements of the same class were annotated as separate instances when they represented distinct structural components, such as walls meeting at a corner or beams intersecting at a joint. Cutouts and rough openings were treated as separate instances whenever they were spatially distinguishable. Technical building equipment was organized into local instances of related service elements, with close-by elements grouped when their individual boundaries were ambiguous. A special case is the background class None, for which each scene contains exactly one aggregated instance collecting all unassignable points, including clutter, artifacts, and ambiguous structures.

After completion of the annotation process, all 504 scenes were reviewed again by the authors, corrected where necessary, and approved for release. Finally, automated scripts were used to verify annotation completeness and consistency, ensuring that no points remained unlabeled and that each scene contained exactly one background instance of class 0 (None) with instance ID 0. The external annotation effort amounted to 3,818 hours. Supervision, review, and curation required an additional workload of roughly 10 % of the actual annotation time. The complete annotation and quality-control workflow is summarized in the supplementary material.

## 3.3. 2D Representations

In addition to the native 3D representation, the release provides scripts for deriving 2D panoramic and cube-map representations from each scene. These views support multimodal use cases and direct image-based inspection of the point-wise features and annotations. The panoramas use an equirectangular projection from the scanner-centered coordinate system onto a spherical image plane. Since each scan’s local origin coincides with the scanner position, the projection corresponds directly to the physical acquisition geometry.

Each 3D scan-point is mapped to spherical image coordinates using its azimuth and elevation angles. The horizontal pixel coordinate u is defined as

$$
u = { \frac { 1 } { 2 } } \left( 1 - { \frac { \mathrm { a t a n 2 } ( y , x ) } { \pi } } \right) ( w - 1 )\tag{1}
$$

where w denotes the panorama pixel width. The vertical coordinate v is computed from the elevation angle

$$
\phi = \arcsin \left( { \frac { z } { r } } \right)\tag{2}
$$

and mapped to the image grid as

$$
v = \left( 1 - \frac { \phi + f _ { \mathrm { d o w n } } } { f _ { \mathrm { u p } } + f _ { \mathrm { d o w n } } } \right) ( h - 1 ) ,\tag{3}
$$

where h is the panorama height and $f _ { u p } = f _ { d o w n } = 9 0 ^ { \circ }$ This results in an equirectangular projection with a horizontal field of view of 360<sup>◦</sup> and a vertical field of view of 180<sup>◦</sup>. In our implementation, the image resolution is controlled by an upscaling factor and a fixed aspect ratio. Because the point cloud is sparse, naturally, not all pixels are populated. The upscaling factor can be used to manually control the ratio of empty pixels.

The same projection pipeline is used to render multiple modalities from the point cloud, including color, intensity, depth, surface normals, semantic labels, and instance labels. For visualization, scalar attributes, such as intensity and depth, can be normalized over valid pixels and stored as grayscale images. Surface normals are linearly mapped from [−1, 1] to RGB values. Semantic and instance annotations are rendered as both colorized visualizations and machine-readable label-ID images.

For each point $p = ( x , y , z )$ , the radial distance to the sensor is computed as the Euclidean distance. If multiple points fall within the same pixel, the visibility is resolved by depth ordering. Points are sorted by radial distance and rasterized from far to near, such that the closest point overwrites all others at the same image location. This produces a z-buffer-like rendering [12] that is consistent with the scanner’s viewpoint. Pixels without a projected point remain invalid. This procedure establishes a direct correspondence between the 3D point cloud and its 2D panoramic representation, enabling multimodal use of geometry, appearance, and annotations within a common dataset.

In addition to the most common equidistant projection, this dataset also supports a cube map representation of the 3D space, projected into a collection of six square image faces with significantly reduced distortion, which may be preferred for 2D scene analysis [7, 8, 17]. Scripts for reproducibly generating both projection types from the released point-wise arrays are available in the Dataset Hub [18].

## 3.4. Dataset Statistics

Rohbau3D-Semantics comprises 504 terrestrial laser scan scenes from 14 construction sites, containing a total of 12,485,778,128 points and 20,786 annotated object instances. The dataset provides point-wise semantic labels for 17 foreground classes and one background class (None), together with class-wise instance annotations for object-level analysis.

Table 2 summarizes the overall point cloud statistics for the unprocessed full-scale dataset. Across all 504 scenes, the global averages are 24,773,369 points and 41.24 annotated instances per scene. Additionally, each full-resolution scan is accompanied by an index file that enables reproducible subsampling to a capped version with at most 15 million points per scene, thereby increasing homogeneity and reducing memory usage in large-scale learning pipelines. This capped version is referred to as ”15M”.

Figure 4 illustrates the semantic class distribution in terms of both points and instances. The class distribution is skewed at the point level, with floor-, wall-, and ceiling-related classes accounting for the majority (80%) of points. The inherent nature of indoor datasets implies that most data points belong to one of these three classes, since these components constitute the vast majority of the convex building envelope. Rohbau3D is no exception; it is subject to the same imbalance as comparable indoor data sources. However, the instance-level distribution is substantially less skewed, which indicates that many smaller object categories recur frequently across the dataset despite contributing comparatively few points per instance.

![](images/6ab0a492a47a650048fa7047365a3f7c94a05732d438a0257182f3e26f4764d2.jpg)  
Figure 4. The allocation of all 18 object classes in terms of point- and instance-frequency recurrence in the Rohbau3D-Semantics data.

Table 2. Dataset statistics by site.
<table><tr><td>Site</td><td>Points</td><td>Instances</td><td>mean Pts./scene</td><td>mean Inst./scene</td></tr><tr><td>site_00</td><td>629,986,448</td><td>1,227</td><td>17,499,620</td><td>34.08</td></tr><tr><td>site_01</td><td>224,578,587</td><td>562</td><td>210,694,220</td><td>26.76</td></tr><tr><td>site_02</td><td>118,823,353</td><td>622</td><td>10,802,120</td><td>56.55</td></tr><tr><td>site_03</td><td>1,549,189,153</td><td>3,366</td><td>27,664,090</td><td>60.11</td></tr><tr><td>site_04</td><td>378,528,127</td><td>998</td><td>27,037,720</td><td>71.29</td></tr><tr><td>site_05</td><td>1,771,487,032</td><td>2,134</td><td>27,253,650</td><td>32.83</td></tr><tr><td>site_06</td><td>1,804,683,438</td><td>3,452</td><td>27,764,360</td><td>53.11</td></tr><tr><td>site_07</td><td>135,670,833</td><td>513</td><td>27,134,170</td><td>102.60</td></tr><tr><td>site_08</td><td>1,664,353,250</td><td>1,963</td><td>27,739,220</td><td>32.72</td></tr><tr><td>site_09</td><td>1,009,946,470</td><td></td><td>1,064 28,054,070</td><td>29.56</td></tr><tr><td>site_10</td><td>151,315,713</td><td>221</td><td>25,219,290</td><td>36.83</td></tr><tr><td>site_11</td><td>1,122,217,527</td><td>1,242</td><td>26,098,080</td><td>28.88</td></tr><tr><td>site_12</td><td>1,111,902,857</td><td>1,575</td><td>27,797,570</td><td>39.38</td></tr><tr><td>site_13</td><td>813,095,340</td><td>1,847</td><td>17,675,990</td><td>40.15</td></tr><tr><td>Tot./Avg.</td><td>12,485,778,128</td><td>20,786</td><td>24,773,369</td><td>41.24</td></tr></table>

The None class plays a special role because all scene objects that cannot be assigned to any of the existing component classes are assigned to this None class, combining not only all scan artifacts, ghost points, and measurement noise, but also all objects that are irrelevant to technically understanding the structure of the building. Overall, the None class accounts for 7.3% of the total points, although this percentage exceeds 10% at some construction sites.

## 4. Training, Validation & Testing split

The 504 scans are not statistically independent: scans from different positions at the same construction site may capture overlapping areas and share distinctive geometry, materials, and construction conditions. A random scene-level split could therefore leak site-specific information and yield overly optimistic generalization estimates. Instead, we treat construction sites as independent units for testing and reserve all 96 test samples from site 00 and site 08 only for that. These two residential and high-rise office building projects were selected after examining the per-site instance distributions to obtain broad class coverage while retaining sufficient training data. Because no site contains every class and the number of scenes differs substantially among sites, the split prioritizes site-level independence and class coverage over equal scene counts or class frequencies.

The resulting allocation for our proposed standard split is 354/54/96 scenes (70.2%/10.7%/19.0%) for training, validation, and testing. The following list defines the proposed split; numbers in parentheses indicate the number of validation scenes selected from each training site. The full list of validation scene IDs is provided in metadata/data split.json in the Dataset Hub [18].

• Train / Val (70% + 11%): site 01 (3), site 02 (2), site 03 (7), site 04 (1), site 05 (9), site 06 (10), site 07 (1), site 09 (4), site 10 (1), site 11 (6), site 12 (5), site 13 (5);

• Test (19%): site 00, site 08.

Validation is intended for model selection rather than as an independent estimate of cross-site generalization. Holding out further complete sites for validation would reduce the already limited pool of 14 sites and could remove rare classes from training or validation. We therefore selected 54 validation scenes from the 12 training sites using their class and instance distributions to retain coverage of the full taxonomy, leaving 354 scenes for training. Consequently, the training and validation datasets may share similar location features, even though we ruled out overlaps between neighboring scans, while the test dataset does not overlap with either subset by location or local features. Results on site 00 and site 08 should thus be interpreted as a reproducible fixed-split reference on two unseen projects, not as comprehensive evidence across all construction-site diversity. Site-level cross-validation over alternative held-out groups would provide stronger generalization evidence but is left for future work.

## 5. Semantic Segmentation Baseline

We report results for 3D point cloud semantic segmentation, serving as the initial baseline for this dataset, obtained from fully supervised training on 17+1 object classes. This baseline is evaluated on the capped 15M version of Rohbau3D Semantics’ proposed standard data split. It does not benchmark instance segmentation or isolate the contributions of intensity, normals, depth, and the projected 2D representations. To provide a computationally tractable and consistent comparison across all evaluated backbones, the inputs were voxelized in the data loader using a 4x4x4 cm grid. This preprocessing kept training within the 32 GB memory budget of the available Tesla V100-SXM3 GPUs; evaluation at the full native point density is outside the scope of this initial baseline. The method followed a fully supervised training pipeline. The implementation leveraged the two established libraries for point cloud processing: PointNeXt [16] for MLP-based DNNs, and Pointcept (v.1.6.1)[14] for state of-the-art point-transformer models. We compare six backbones from the two libraries, including the standard Point-Net++ [15] for reference, PointNeXt [16] in the small (S) and large (L) model configurations, with varying network depth scale, Swin3D [27] in the small (S) and large (L) model configurations for a scaled window size of the sliding window mechanism, and the latest Point Transformer V3 [26] (PTv3) as the current benchmark for standard supervised learning. All training runs used geometric augmentation in the coordinate space and feature augmentation in the RGB color space, a cosine-scheduled learning rate, cross-entropy loss, and an AdamW optimizer. Table 3 tabulates the quantitative results of this baseline on the proposed dataset. Performance is calculated by the per-class intersection over union (IoU) metric. Figure 5 showcases qualitative results.

The evaluated baselines yield mIoU values between 0.44 and 0.72, with PTv3 achieving the highest value. Across all models, large and structurally dominant classes such as Ceiling, Floor, and Wall achieve consistently high IoU values, whereas smaller and semantically more specific categories, including Wall Cutout, Window Cutout, Window, and Elevator Doors, remain considerably more challenging. Given the pronounced class imbalance and the relative rarity of these categories, this pattern is expected. The results provide an initial quantitative reference for semantic segmentation difficulty across coarse and fine-grained classes under the proposed split; they do not establish performance for the dataset’s instance-level or multimodal use cases.

## 6. Conclusion

We presented a dataset of small- to large-scale indoor spaces from construction-site environments during the structural work phase, with high-detail semantic class and instance annotations in both 3D-space and projected 2D representations. In contrast to existing indoor 3D datasets, these data preserve the characteristic properties of physica laser sensing, including irregular sampling, reflections, and measurement artifacts. At the same time, terrestrial laser scanning provides exceptionally high point density and geometric accuracy, resulting in a benchmark that is realistic, precise, and challenging. The annotations required 3,818 hours of professional manual labeling and were supported by expert supervision, a second review of all 504 scenes, and automated completeness and consistency checks. Because the full dataset was annotated by one provider and subsequently reviewed and corrected by the authors, we do not report inter-annotator agreement or accuracy against an independently labeled subset; annotation quality is instead supported by full-scene expert review and automated consistency checks. The dataset’s scale further distinguishes it from previous construction-site datasets: it comprises 504 scans from 14 sites across multiple building types and construction settings.

Beyond extending the raw data with semantic and instance labels, we documented the acquisition pipeline, preprocessing steps, annotation workflow, panoramic projection procedure, and benchmark splits, and reported baseline results for supervised 3D point cloud semantic segmentation. The final dataset is relevant not only for constructionsite analytics, digital documentation, and BIM-related workflows, but also for robust 3D scene understanding in cluttered, partially occluded, and geometrically incomplete real-world environments with challenging lighting conditions and sensor noise.

The present evaluation is limited to supervised 3D semantic segmentation on one reference split. It therefore neither establishes an instance-segmentation benchmark nor quantifies the contributions of the available intensity, depth, normals, and projected 2D representations. The corresponding annotations and modalities are released as resources for these future evaluations, rather than presented as empirically validated contributions of the current baseline. Future work should establish instance-level baselines, modality ablations, and site-level cross-validation over alternative held-out groups. Recent developments in foundation models, promptable 3D segmentation, open-vocabulary 3D scene understanding, and self-supervised learning provide further opportunities to use the dataset beyond the present supervised setting.

Table 3. Quantitative results for supervised 3D semantic segmentation on the 96 test scans from the held-out sites site 00 and site 08.
<table><tr><td colspan="10">Siaid Cceiling Cin Cuout Ceing</td><td>Walll Ctot</td><td>parapet</td><td>Beamm</td><td>Colun</td><td>Staicase</td><td>Win utout</td><td>Windw</td><td>Dor y ppen.</td><td>Fa ment</td><td>Elevor Dors</td><td>TGA</td><td></td><td></td></tr><tr><td>PointNet++</td><td>model</td><td>None 0.58</td><td>0.45</td><td></td><td>0.42</td><td>L0r 0.95</td><td>Wal 0.71</td><td>Drywval 0.28</td><td>0.19</td><td>0.30</td><td>0.48</td><td>0.49</td><td>0.69</td><td>0.11</td><td>0.13</td><td></td><td></td><td></td><td></td><td>mIoU</td></tr><tr><td>PointNeXt-S</td><td>0.61</td><td>0.95 0.95</td><td>0.48</td><td>0.32</td><td>0.95</td><td>0.76</td><td>0.46</td><td>0.11</td><td>0.44</td><td>0.47</td><td>0.52</td><td></td><td>0.64</td><td>0.06</td><td>0.06</td><td>0.32 0.29</td><td>0.18 0.23</td><td>0.14 0.02</td><td>0.57 0.57</td><td>0.44 0.44</td></tr><tr><td></td><td>0.67</td><td>0.96</td><td>0.58</td><td>0.63</td><td>0.97</td><td>0.85</td><td>0.65</td><td>0.23</td><td>0.57</td><td>0.59</td><td></td><td>0.73</td><td>0.81</td><td>0.22</td><td>0.12</td><td>0.50</td><td>0.36</td><td>0.03</td><td>0.62</td><td>0.56</td></tr><tr><td>PointNeXt-L</td><td>0.75</td><td>0.98</td><td>0.73</td><td>0.83</td><td>0.97</td><td>0.87</td><td>0.68</td><td>0.37</td><td>0.76</td><td>0.64</td><td></td><td>0.81</td><td>0.93</td><td>0.43</td><td>0.32</td><td>0.60</td><td>0.59</td><td>0.45</td><td>0.69</td><td>0.69</td></tr><tr><td>Swin3D-S</td><td>0.77</td><td>0.97</td><td>0.72</td><td>0.75</td><td>0.97</td><td>0.91</td><td>0.78</td><td>0.30</td><td>0.68</td><td>0.69</td><td></td><td>0.78</td><td>0.93</td><td>0.45</td><td>0.56</td><td>0.63</td><td>0.59</td><td>0.46</td><td>0.66</td><td>0.70</td></tr><tr><td>Swin3D-L PTv3</td><td>0.77</td><td>0.98</td><td>0.73</td><td>0.80</td><td>0.98</td><td>0.89</td><td>0.72</td><td>0.46</td><td>0.62</td><td>0.67</td><td></td><td>0.82</td><td>0.92</td><td>0.48</td><td>0.43</td><td>0.63</td><td>0.64</td><td>0.71</td><td>0.68</td><td>0.72</td></tr></table>

![](images/56156306e48600e6633a2599c92581e1fa47821f765c9e4a831aab8ebf2805ec.jpg)  
Figure 5. Qualitative baseline 3D-point cloud semantic segmentation results from supervised learning on the Rohbau3D-Semantics data.

Finally, although the dataset is based on stationary laser scanning rather than mobile LiDAR, it may also serve as a useful reference for studying transfer to lower-density robotic sensing scenarios. We hope that Rohbau3D-Semantics contributes to more realistic, application-relevant, and data-efficient methods for understanding complex built spaces.

## 7. Licensing and Access

Rohbau3D-Semantics is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license, permitting reuse with attribution. The data are hosted in the university’s institutional repository (doi:10.60776/ZWJFI4); download and processing scripts are available on GitHub (github.com/RauchLukas/rohbau3d).

## References

[1] Iro Armeni, Ozan Sener, Amir Roshan Zamir, Helen Jiang, Ioannis Brilakis, Martin Fischer, and Silvio Savarese. 3D semantic parsing of large-scale indoor spaces. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1534–1543, 2016. 2, 3

[2] Iro Armeni, Sasha Sax, Amir Roshan Zamir, and Silvio Savarese. Joint 2D-3D-semantic data for indoor scene understanding, 2017. arXiv:1702.01105. 1, 4

[3] Jingdao Chen, Zsolt Kira, and Yong K. Cho. Deep learning approach to point cloud scene understanding for automated scan to 3D reconstruction. Journal of Computing in Civil Engineering, 33(4):04019027, 2019. 1

[4] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2432–2443, 2017. 1, 2, 3, 4

[5] Chunyong Feng, Junqi Yu, Wei Quan, Kai Wang, Jugang Guo, Yisheng Chen, and Zhenping Dong. Integration and evaluation of a 3D LiDAR SLAM system for construction robots in large-scale public building sites. Developments in the Built Environment, 26:100913, 2026. 2

[6] Yanwen Guo, Yuanqi Li, Dayong Ren, Xiaohong Zhang, Jiawei Li, Liang Pu, Changfeng Ma, Xiaoyu Zhan, Jie Guo, Mingqiang Wei, Yan Zhang, Piaopiao Yu, Shuangyu Yang, Donghao Ji, Huisheng Ye, Hao Sun, Yansong Liu, Yinuo Chen, Jiaqi Zhu, and Hongyu Liu. LiDAR-Net: A realscanned 3D point cloud dataset for indoor scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21989–21999, 2024. 2, 3

[7] Sandeep Inuganti, Hideaki Kanayama, Kanta Shimizu, Mahdi Chamseddine, Soichiro Yokota, Didier Stricker, and Jason Rambach. JOPP-3D: Joint open-vocabulary semantic segmentation on point clouds and panoramas, 2026. arXiv:2603.06168. 5

[8] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and´ Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3992–4003, 2023. 5

[9] Pierre-Alain Langlois, Yang Xiao, Alexandre Boulch, and Renaud Marlet. VASAD: A volume and semantic dataset for building reconstruction from point clouds. In Proceedings ofthe 26th International Conference on Pattern Recognition (ICPR), pages 4008–4015, 2022. 2, 3

[10] Nhut Le and Maryam Rahnemoonfar. 3D semantic segmentation for post-disaster assessment. In IGARSS 2025 – IEEE International Geoscience and Remote Sensing Symposium, pages 6488–6492, 2025. 2

[11] H. Macher, T. Landes, and P. Grussenmeyer. Point clouds segmentation as base for as-built BIM creation. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, II-5/W3:191–197, 2015. 2

[12] Alfred Nischwitz, Max Fischer, Peter Haberacker, and Gu-¨ drun Socher. Computergrafik: Band I des Standardwerks Computergrafik und Bildverarbeitung. Springer Fachmedien Wiesbaden, 2019. 5

[13] Xiao Pan, Sina Tavasoli, T. Y. Yang, and Sina Poorghasem. Post-disaster building indoor damage and survivor detection using autonomous path planning and deep learning with unmanned aerial vehicles, 2025. arXiv:2503.10027. 2

[14] Pointcept Contributors. Pointcept: A codebase for point cloud perception research, 2023. Accessed: 2026-10-26. 7

[15] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J. Guibas. PointNet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in Neural Informa tion Processing Systems, pages 5099–5108, 2017. 7

[16] Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. PointNeXt: Revisiting PointNet++ with improved training and scaling strategies. In Advances in Neural Infor mation Processing Systems, pages 23192–23204, 2022. 7

[17] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 5

[18] Lukas Rauch. Rohbau3D github repository, 2026. Link: https://github.com/RauchLukas/Rohbau3D; Accessed: 2026-10-26. 4, 5, 6, 11

[19] Lukas Rauch and Thomas Braml. Semantic point cloud segmentation with deep-learning-based approaches for the con struction industry: A survey. Applied Sciences, 13(16):9146, 2023. 1

[20] Lukas Rauch and Thomas Braml. Rohbau3D: A shell construction site 3D point cloud dataset. Scientific Data, 12(1): 1478, 2025. 2, 3, 4

[21] ScanNet++. ScanNet++ dataset changelog, 2024. Accessed: 2026-10-26. 3

[22] Tao Sun, Yan Hao, Shengyu Huang, Silvio Savarese, Konrad Schindler, Marc Pollefeys, and Iro Armeni. Nothing stands still: A spatiotemporal benchmark on 3D point cloud registration under large geometric and temporal change. ISPRS Journal of Photogrammetry and Remote Sensing, 220:799– 823, 2025. 3

[23] Shengjun Tang, Xiaoming Li, Xianwei Zheng, Bo Wu, Weixi Wang, and Yunjie Zhang. BIM generation from 3D point clouds by combining 3D deep learning and improved morphological approach. Automation in Construction, 141: 104422, 2022. 1

[24] Maciej Trzeciak, Kacper Pluta, Yasmin Fathy, Lucio Alcalde, Stanley Chee, Antony Bromley, Ioannis Brilakis, and Pierre Alliez. ConSLAM: Periodically collected real-world construction dataset for SLAM and progress monitoring. In Computer Vision – ECCV 2022 Workshops, pages 317–331. Springer, 2022. 3

[25] Girma Zewdie Tsige, Bashar Saleem Abbas Alsadik, Sander Oude Elberink, and Maarten Bassier. Automated scan-vs

BIM registration using columns segmented by deep learning for construction progress monitoring. The International Archives ofthe Photogrammetry, Remote Sensing and Spatial Information Sciences, XLVIII-G-2025:1455–1462, 2025. 1, 2

[26] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer V3: Simpler, faster, stronger. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4840–4851, 2024. 7

[27] Yu-Qi Yang, Yu-Xiao Guo, Jian-Yu Xiong, Yang Liu, Hao Pan, Peng-Shuai Wang, Xin Tong, and Baining Guo. Swin3D: A pretrained transformer backbone for 3D indoor scene understanding. Computational Visual Media, 11(1): 83–101, 2025. 7

[28] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A high-fidelity dataset of 3D indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12–22, 2023. 1, 2

[29] Hongzhe Yue, Qian Wang, Maggie Y. Gao, Tao Yang, Luping Li, Keyu Chen, and Mingzhu Wang. A point cloud dataset and deep learning framework for semantic segmentation of construction site point clouds. Computer-Aided Civil and Infrastructure Engineering, 45:100050, 2026. 3

[30] Hao Xuan Zhang and Zhengbo Zou. Quality assurance for building components through point cloud segmentation leveraging synthetic data. Automation in Construction, 155: 105045, 2023. 1

[31] Lintong Zhang, Michael Helmberger, Lanke Frank Tarimo Fu, David Wisth, Marco Camurri, Davide Scaramuzza, and Maurice Fallon. Hilti-Oxford dataset: A millimeter-accurate benchmark for simultaneous localization and mapping. IEEE Robotics and Automation Letters, 8(1):408–415, 2023. 3

[32] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3D: A large photo-realistic dataset for structured 3D modeling. In Computer Vision – ECCV 2020, pages 519–535. Springer, 2020. 1, 2

[33] Jia Zou, Xiongyao Xie, and Genji Tang. A method for detecting construction deviations in large and complex building structures utilizing synthetic point clouds for segmentation. Computer-Aided Civil and Infrastructure Engineering, 40(31):6597–6621, 2025. 2

## A. Construction-Site Overview

Table S1 summarizes the site characteristics most relevant to semantic diversity and LiDAR sensing difficulty. The complete machine-readable site metadata remain available through the accompanying Dataset Hub [18].

## B. Dataset Structure and Format

The released data follow a site–scan hierarchy, illustrated below in abbreviated form. Each scan is an independent, scanner-centered observation: its origin is the scanner position, and scans are not supplied in a common registered site coordinate system. Consequently, instance identifiers describe component observations within one scan and must not be interpreted as persistent object identities across scans. The repository-level JSON files provide the class registry, benchmark split, point-cloud file inventory, and site metadata, while the site PDF files provide rendered feature overviews.

rohbau3d/  
|-- metadata/  
| |-- class\_registry.json  
| |-- data\_split.json  
| |-- pointcloud\_file\_list.json  
| |-- site\_metadata.json  
| ’-- panorama/features/  
| ’-- site\_<id>.panorama.features.pdf  
|-- site\_00/  
| |-- scene\_00000/  
| | |-- coord.npy, color.npy  
| | |-- intensity.npy, normal.npy  
| | |-- class.npy, instance.npy  
| | |-- sample\_idx.npy  
| |-- inv\_sample\_idx.npy  
| |-- panorama/ (<sub>\*</sub>)  
| ’-- cube\_map/ (<sub>\*</sub>)  
| ’-- ...  
|-- site\_01/ ...  
’-- site\_13/

Within each scan directory, the feature and label arrays use a common point ordering: row i in each point-wise array describes the same measured point. Table S2 summarizes the released NumPy format. A redundant scene-level JSON record containing the annotation metadata for every instance can also be selected by the repository downloader.

Table S2. Per-scan point-cloud files. N denotes the number of points in the full-resolution scan.
<table><tr><td>File</td><td>Entries</td><td>Stored information</td></tr><tr><td>coord.npy</td><td>N × 3</td><td>Scanner-centered Cartesian coordinates (x, y, z).</td></tr><tr><td>color.npy</td><td>N × 3</td><td>Point-wise RGB color.</td></tr><tr><td>intensity.npy</td><td>N scalar</td><td>Laser reflection intensity.</td></tr><tr><td>normal.npy</td><td>N × 3</td><td>Reconstructed surface-normal vectors.</td></tr><tr><td>class.npy</td><td>N integer</td><td>Semantic label ID: 17 foreground classes and None.</td></tr><tr><td>instance.npy</td><td>N integer</td><td>Scene-local instance ID; not a cross-scan object identity.</td></tr><tr><td>sample_idx.npy</td><td>indices</td><td>Reproducible selection of at most 15 mil- lion full-resolution points.</td></tr><tr><td>inv_sample_idx.npy</td><td>indices</td><td>Index mapping from the 15M representa- tion to the full-resolution point set.</td></tr></table>

The full-resolution arrays remain the primary data; the two index files provide an optional computationally tractable 15M representation while retaining correspondence to the original observations.

The directories marked (\*) are derived representations generated locally from the 3D arrays and are not separate downloadable measurements. The supplied rendering script can produce color, radial depth, intensity, normal, semanticlabel, and instance-label panoramas or six-face cube maps. Panoramas default to 4096 × 2048 pixels and cube-map faces to 1024×1024 pixels; empty pixels are stored as transparent. The panoramic output additionally distinguishes visualized label images (class.png, instance.png) from machine-readable label-ID images (class id.png, instance id.png), while depth.npy retains numeric depth. Because all of these views are rendered from the scanner-centered point cloud using a common projection, they remain geometrically linked to the native 3D observations rather than constituting independent modalities.

## C. Annotation Review and Quality Control

Quality control proceeded as an iterative review–correction cycle. After provider delivery, the authors inspected each scene in 3D using its semantic and instance views. Review targeted missing labels, semantic inconsistencies across scenes, incorrect instance merges or splits, boundary leakage, and isolated outliers that could distort an instance bounding box. Minor errors were corrected directly by the authors; scenes with broader or repeated problems were rejected with visual feedback and returned to the provider for correction. Corrected scenes were reviewed again before acceptance, and ambiguous construction elements were resolved with domain knowledge and incorporated into subsequent annotation decisions.

After manual acceptance, automated checks verified that every point had valid semantic and instance identifiers, each scan contained exactly one None instance with ID 0, and no residual instance contained fewer than ten points. A failed check returned the scene to the correction loop. Figures S1 and S2 summarize the sequence and division of responsibilities.

![](images/6f118bb3430c79ed6da9e0c7f63156a0a81e12fca6f95b20f17ecb95a3461ba2.jpg)  
Figure S1. Annotation and quality-control pipeline. The workflow begins with taxonomy definition, annotation guidelines, provider training, and pilot scenes. Pilot review and construction-domain decisions provide feedback during annotation. After delivery, all 504 scenes undergo manual review, correction where necessary, and automated checks for complete semantic and instance identifiers, exactly one None instance per scene, and the absence of instances containing fewer than ten points. Failed checks trigger another correction and verification cycle before final approval.

Table S1. Site-level environmental diversity in Rohbau3D-Semantics. An asterisk marks a renovation site.
<table><tr><td>Site</td><td>Building and construction</td><td>Distinctive geometry and sensing conditions</td></tr><tr><td>site_00</td><td>Multi-story apartment building; brick masonry</td><td>Small-medium rooms; partly plastered walls; windows installed, doors absent.</td></tr><tr><td>site_01</td><td>Multi-story apartment building; brick masonry</td><td>Small-medium rooms; slanted ceilings; partly plastered walls; windows installed, doors absent.</td></tr><tr><td>site_02</td><td>Underground parking structure; reinforced concrete</td><td>Column grid; variable ceiling heights; low light; floor puddles.</td></tr><tr><td>site_03</td><td>Multi-story school; reinforced-concrete frame</td><td>Large rooms; well-lit areas; floor puddles.</td></tr><tr><td>site_04</td><td>Indoor swimming-pool hall; reinforced concrete</td><td>Large open volume; rounded ceiling components; large floor opening; facade not installed.</td></tr><tr><td>site_05</td><td>Multi-story school; drywall partitions Multi-story school; mixed solid and drywall</td><td>Variable room sizes; semi-transparent temporary facade covering.</td></tr><tr><td>site_06</td><td>partitions</td><td>Medium-large rooms; open facade areas; ceiling-mounted technical equipment.</td></tr><tr><td>site_07</td><td>Large sports hall; reinforced concrete High-rise office building; drywall</td><td>Large open volume and high ceiling.</td></tr><tr><td>site_08</td><td>construction</td><td>Small–large rooms; freestanding drywall components; glazed facade and ceiling services installed.</td></tr><tr><td>site_09*</td><td>Historic multi-story masonry building</td><td>Small rooms; narrow staircases; windows installed, doors absent; low light.</td></tr><tr><td>site_10*</td><td>Vaulted brick cellar</td><td>Small rooms; uneven floors; low light.</td></tr><tr><td>site_11</td><td>Two-story building with basement; masonry and precast concrete</td><td>Small–medium rooms; floor puddles; low light.</td></tr><tr><td>site_12</td><td>Multi-story apartment building with basement; precast reinforced concrete</td><td>Large window and door rough openings; some floor puddles and low-light scenes.</td></tr><tr><td>site_13*</td><td>Multi-story masonry building</td><td>Small rooms connected by corridors; partly plastered and partly exposed walls</td></tr></table>

![](images/a0057c3f7a6f106696dc2bbdd743b59ea749fe5f86b0fe1c3cb33c6c53285b34.jpg)  
Figure S2. Responsibilities in the annotation and quality-control workflow. The swimlanes distinguish tasks performed by the authors and construction-domain experts, the professional annotation provider, and the automated quality-assurance procedure. Solid arrows show the primary workflow, while dashed arrows show feedback and correction loops initiated by pilot errors, ambiguous cases, manual review, or failed automated checks.

## D. Condensed Annotation Guidelines

This section condenses the finalized instructions used to guide the professional annotation provider. The task was

to assign every measured point in each raw terrestrial laser scan to one of the 17 foreground classes or the None background class and to form scene-local object instances. The

foreground taxonomy targets installed shell-construction components and technical building equipment; its complete class order and definitions are given below.

## D.1. General Annotation Principles

1. Preserve the raw observations. Annotate each scan as delivered: do not filter, delete, move, or otherwise clean points. Outliers, reflections, and ghost geometry remain in the data and are labeled; occlusions and incomplete regions are not reconstructed.

2. Prioritize precise boundaries. Label large and small components at the point level and refine their boundaries as closely as the observations permit. Inspect geometry from multiple directions in 3D; use RGB appearance only as supporting evidence.

3. Segment objects, not only classes. Distinguishable physical components receive separate instance identifiers even when they share a semantic class; for example, five columns are five instances. The exceptions are the single None instance with ID 0 and the local grouping rule for related TGA elements described below.

4. Use None rather than guessing. Assign scan artifacts, stray points, objects outside the taxonomy, and geometry that cannot be classified reliably to None. No point may remain unlabeled.

5. Restrict foreground labels to installed components. Temporary supports, scaffolding, safety equipment, stored or not-yet-installed materials, loose cables, coverings, and comparable construction-site clutter belong to None, not to the corresponding structural or TGA class.

6. Apply decisions consistently. Use the same class and instance criteria across all scenes. Annotators were instructed to flag uncertain cases for review rather than resolve them by assumption; reviewed decisions were then applied to comparable cases.

7. Favor quality over throughput. Accurate object boundaries and fine structures take precedence over annotation speed. Each segment should be checked before completion, particularly around occlusions, intersections, reflective surfaces, and small components.

## D.2. Instance and Boundary Rules

Instance formation. Instance identifiers are local to one scan and do not encode cross-scan object identity. A separate instance was created for each distinguishable physical component; TGA is the exception and uses local groups of related elements as described below. Touching elements remained separate when they represented different construction components, such as walls meeting at a corner or beams intersecting at a joint. A gap caused by occlusion or missing measurements split an instance only when the common physical connection was unclear or ambiguous; fragments were merged when their continuation as one component could be inferred reliably from the surrounding geometry. Each spatially distinguishable cutout or rough opening formed a separate instance. All None points in a scan were collected in the single background instance with ID 0. Technical building equipment was grouped into local instances of related elements; close-by elements were combined when the scan did not support a reliable separation.

Cutout and rough-opening labels. These classes describe the narrow measured reveal or framing surfaces around an intentional opening—typically the two lateral jambs, the head or soffit, and the sill or lower reveal where present. They do not label empty space. If an otherwise empty opening was covered by temporary boards, foil, or comparable site material, points on that covering were assigned to None, not to the cutout or rough-opening class.

## D.3. Class-Specific Rules

The classes below follow the identifiers and order of the released class registry.

0 — None. Background for sensor noise, isolated outliers, reflections and ghost geometry, temporary supports or safety equipment, stored and not-yet-installed materials, objects outside the taxon omy, and ambiguous points. All such points share instance ID 0 within a scan.

1 — Ceiling. Horizontal surfaces forming the upper interior boundary of a space. Beams, cutouts, technical equipment, and temporary shoring are excluded. Continuous observations belong to one instance unless a true geometric break or an ambiguous occlusion prevents a reliable connection.

2 — Slanted Ceiling. Inclined surfaces that function as the upper interior boundary, including surfaces beneath pitched roofs. Inclined beams and the underside of a staircase are excluded when their structural identity can be determined from context. Instance formation follows the ceiling rule.

3 — Ceiling Cutout. Measured reveal surfaces framing an intentional opening through a ceiling, such as a shaft, stair opening, or service penetration. The void and any temporary covering are excluded. Each distinguishable cutout is a separate instance.

4 — Floor. Horizontal, walkable structural surfaces. Rubble, tools, stored material, temporary coverings, and other clutter are excluded. A floor continuing through an open transition remains one instance; true gaps, stairs, or elevation changes establish a new instance.

5 — Wall. Solid vertical elements made of concrete, brick, masonry, or comparable construction. Surface channels, protrusions, and local imperfections remain part of the wall rather than being discarded for deviating from a plane. Each straight wall section is a separate instance, with corners or direction changes defining component boundaries.

6 — Drywall. Non-load-bearing gypsum-board partitions, includ ing unfinished partitions with visible stud framing. Studs belonging to an incomplete drywall assembly are assigned to the adjoining drywall. Instance boundaries follow the wall rule, and RGB appearance may support the distinction from concrete or masonry. 7 — Wall Cutout. Reveal surfaces framing an intentional technical opening through a wall that is not intended for a window or door. Each spatially separate opening is an individual instance; empty space and temporary coverings are excluded.

8 — Parapet. Low, wall-like structural elements that do not extend to the ceiling, including protective barriers and low dividers. Full-height elements are labeled as walls. Each straight parapet section forms a separate instance, with corners starting a new instance.

9 — Beam. Permanent load-bearing elements that span between walls, columns, or other supports and exhibit a structural crosssection. Temporary supports and elongated non-structural objects are excluded. Distinct beams remain separate instances even where they meet or intersect.

10 — Column. Vertical load-bearing supports with a compact cross-section, including round, square, rectangular, freestanding, and partially wall-embedded columns. Every distinguishable column is a separate instance.

11 — Staircase. Permanent stair flights comprising treads, risers, and structurally associated landings. Temporary or permanent railings are excluded from this class. A connected flight and its associated landing may form one instance, whereas clearly separate flights form separate instances.

12 — Window Cutout. Reveal surfaces framing an opening intended for a window before installation. Each window location is a separate instance. The empty interior and temporary boards or foil covering the opening are excluded.

13 — Window. An installed or partially installed window unit, including its frame, sash, and glazing. Connected panes belonging to one assembled unit may share an instance; spatially distinct window units are separate. Reflection artifacts behind glazing are assigned to None.

14 — Door Rough Opening. Reveal surfaces framing an opening intended for a conventional door before its installation. Oversized open passages that are not purpose-made door openings remain part of the adjacent wall classification. Each doorway is a separate instance, while its void and temporary covering are excluded.

15 — Facade Element. Installed near-full-height facade units, commonly composed of glazed, wooden, plastic, or metal panels. Walls, columns, beams, and parapets that bound the unit retain their respective classes. Each structurally unified facade unit forms an instance; reflection artifacts are assigned to None.

16 — Elevator Doors. The complete installed elevator-doorway assembly, including the sliding door panels and metal framing. Each elevator entrance is a separate instance, while the surrounding wall or recess retains its applicable structural class.

17 — TGA. Installed technical building equipment that transports air, water, electricity, or data, including pipes, ventilation ducts, cable trays, and conduits. Related elements are organized into local instance groups, and close-by elements may share an instance when their individual boundaries are ambiguous. Loose cables, stored pipes, and temporary services are assigned to None.

Figure S3 complements the class definitions with representative decisions concerning instance boundaries, cutout surfaces, installed components, and the None class.

Instance boundaries  
![](images/f2444c85632e0e2db455dfdd0b16fb851f48353906dffed4c6132bcae37ca3aa.jpg)

Cutouts, recesses and rough openings  
![](images/d0d69b9fb0b6f7d2d4e903857c1b118e2c79e00b55dd16054b69e6517a837850.jpg)

The None class  
![](images/d9cc8e18bc9be9292c5c6102582ac33b114084a1da466b94c1491e12ecf76df7.jpg)  
Figure S3. Representative cases from the finalized annotation guidelines. Top: different colors indicate scene-local instance identifiers; distinct ceiling and wall components are separated, whereas connected floor observations remain one instance. Middle: door rough openings and window, wall, and ceiling cutouts comprise the measured reveal surfaces, while surrounding structural components retain their respective classes. Bottom: installed facade elements receive the corresponding foreground label, whereas reflections, temporary structures, loose cables, clutter, and uninstalled components are assigned to None. Arrows and labels identify the relevant semantic assignments.