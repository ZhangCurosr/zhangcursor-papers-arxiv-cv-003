# MultiFly: A Real-World Multimodal Aerial Dataset with Annotation-Efficient Label Transfer and Cross-Modal Semantic Consistency

Markus Gross<sup>1,2,3,4,\*</sup>, Andreas Greiner<sup>1,5,\*</sup>, Taehyoung Kim<sup>1,\*</sup>, Sivasubiramaniam Subbiah<sup>1</sup>, Tomaž Coticˇ<sup>1</sup>, Sai B. Matha<sup>1,5</sup>, Conrad Christoph<sup>1</sup>, Oussema Dhaouadi<sup>2,3,4</sup>, Simon Zieher<sup>1</sup>, Surya V. Kumar<sup>1</sup>, Gordon Elger<sup>5</sup>, Henri Meeß<sup>1</sup>, Olaf Wysocki<sup>4</sup>, Paul Spannaus<sup>1,†</sup>, Daniel Cremers<sup>2,3</sup>

![](images/f1eb058a492ac23c5e6379ae9bb36170a895d225414d90dd5bb98648e975c9c3.jpg)  
Fig. 1: Overview of MultiFly. A synchronized and calibrated UAV platform captures multimodal RGB, thermal, LiDAR, and radar data, while our geometry-driven pipeline propagates 0.67 % RGB-only annotations to all four modalities, generating 17, 272 cross-modally consistent semantic labels for four scenes, two altitudes, and 15 semantic categories.

Abstract— We introduce MultiFly, a real-world, low-altitude UAV dataset for semantic perception across RGB, thermal, LiDAR, and radar modalities. MultiFly provides 17,272 synchronized samples from four suburban scenes with framewise annotations for 15 semantic classes, together with calibration and GNSS-RTK/IMU measurements. To avoid costly and inconsistent modality-specific annotation, we propagate labels from only 115 manually annotated RGB images through shared geometric representations to all four modalities. This approach generates semantic labels for 17,157 additional RGB images, 17,272 thermal images, 840M LiDAR points, and 3.4M radar points. Transferred annotations achieve 89.93% average agreement with held-out manual annotations, and 90.94% average semantic consistency across all six modality pairs. We further establish semantic segmentation benchmarks for all four modalities, revealing distinct architectural behavior for dense LiDAR and sparse radar data. Taken together, MultiFly provides a scalable foundation for multimodal aerial perception and, to the best of our knowledge, the first public real-world low-altitude aerial benchmark that combines consistent framewise semantic annotations for RGB, thermal, LiDAR, and radar. Data at https://github.com/markus-42/multifly.

## I. INTRODUCTION

Semantic scene understanding is fundamental to aerial perception [7]–[9]. However, unmanned aerial vehicle (UAV) data remains substantially underrepresented in large-scale multimodal datasets, which increasingly drive modern foundation models. Notably, this is not simply a data-volume problem: UAV perception introduces distinct scene statistics, object scales, and viewing geometries, ranging from nadir to oblique perspectives, that are poorly represented in conventional ground-based datasets. Autonomous driving illustrates the contrast, where large-scale multimodal datasets have enabled systematic alignment across sensors, while similarly comprehensive resources for UAV perception remain limited.

This gap is particularly evident for semantic segmentation. Existing aerial datasets generally provide a single modality or limited modality combinations [5], [6], [10], while no public real-world UAV dataset, to the best of our knowledge, provides synchronized frame-wise semantic annotations across RGB, thermal, LiDAR, and radar. Moreover, these modalities differ substantially in geometry and sampling, making consistent annotation difficult and costly.

To address these challenges, we introduce MultiFly, a calibrated and time-synchronized four-modal aerial sensing platform and semantic dataset. Our annotation framework uses 3D scene geometry as a common semantic interface:

TABLE I: Comparison of real-world multimodal UAV datasets with frame-wise semantic labels (see Section II).
<table><tr><td rowspan="2">Dataset</td><td colspan="4">Perception Modalities</td><td colspan="2">Auxiliary Signals</td><td rowspan="2">Sample Count</td><td rowspan="2">Environments</td><td rowspan="2">Altitude [m]</td><td rowspan="2">Semantic Classes</td><td rowspan="2">Cross-Modal Semantic Consistency</td><td rowspan="2">Manual Correction Not Required</td></tr><tr><td>RGB</td><td>Thermal</td><td>LiDAR</td><td>Radar</td><td>IMU</td><td>GNSS</td></tr><tr><td>IndraEye [1]</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td>5,612</td><td>urban</td><td>7-30</td><td>13</td><td></td><td></td></tr><tr><td>MVUAV [2]</td><td>√</td><td>J</td><td></td><td></td><td></td><td></td><td>2,183</td><td>urban</td><td>5-20</td><td>36</td><td></td><td></td></tr><tr><td>CART [3]</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td>2,282</td><td>rural</td><td>40</td><td>11</td><td></td><td></td></tr><tr><td>Kust4K [4]</td><td>J</td><td>√</td><td></td><td></td><td></td><td></td><td>4,024</td><td>urban</td><td></td><td>8</td><td></td><td></td></tr><tr><td>SegFly [5]</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td>15,007</td><td>urban, industrial, rural</td><td>30, 40, 50</td><td>15</td><td>√</td><td>√</td></tr><tr><td>UAVScenes [6]</td><td>✓</td><td></td><td>√</td><td></td><td>=</td><td>√</td><td>120,000</td><td>rural, airport</td><td>80,90, 130</td><td>18</td><td>√</td><td></td></tr><tr><td>MultiFly (ours)</td><td>√</td><td>√</td><td>1</td><td>√</td><td>√</td><td>√</td><td>17,272</td><td>suburban</td><td>30,50</td><td>15</td><td>√</td><td>√</td></tr></table>

a sparse set of manually annotated RGB images is lifted into 3D and transferred to synchronized RGB, thermal, LiDAR, and radar measurements. This yields dense multimodal annotations from only 115 manually labeled RGB images, while maintaining a common taxonomy and geometric reference.

We summarize our main contributions as follows:

• MultiFly dataset release: 17,272 synchronized RGB, thermal, LiDAR, and radar samples across 4 suburban scenes, covering $5 6 { , } 7 0 0 \mathrm { m } ^ { 2 }$ and 15 semantic classes, with full sensor calibration and state information.

• Annotation-efficient label transfer: Semantic labels are propagated from only 115 manually annotated RGB images to additional 17,157 RGB frames, 17,272 thermal frames, $8 . 4 \times 1 0 ^ { 8 }$ LiDAR points, and $3 . 4 \times 1 0 ^ { 6 }$ radar points, achieving 89.93% average agreement with heldout manual annotations, and 90.94% average semantic consistency across all six modality pairs.

• Benchmarking: We establish semantic segmentation benchmarks on MultiFly for RGB, thermal, LiDAR, and radar modalities, and analyze architectural influences between dense LiDAR and sparse radar segmentation.

## II. RELATED WORK

## A. Multimodal Semantic Aerial Datasets

Semantic perception has evolved from single-sensor benchmarks [11], [12] toward multimodal datasets that enable learning across complementary sensors. In autonomous driving, early resources established core camera-LiDAR perception tasks, while second-generation datasets such as nuScenes [13] expanded to large-scale, synchronized multimodal sensing. This shift enabled not only sensor fusion, but also temporal and multi-task perception, prediction, and planning, and increasingly multimodal foundation models that unify these capabilities.

UAV perception has yet to undergo a comparable transition. Existing aerial datasets remain fragmented across sensing modalities and annotation formats: most are unimodal [10], [14], [15], while others combine only two modalities [3], [6], [16], rely on synthetic data [17], [18], or provide map-wise rather than frame-wise annotations [19], [20]. Consequently, the data required to develop and systematically evaluate unified multimodal UAV perception models remains largely unavailable. In particular, no public realworld low-altitude UAV dataset, to the best of our knowl edge, provides synchronized frame-wise semantics across RGB, thermal, LiDAR, and radar. MultiFly provides this missing substrate (Table I), combining synchronized and registered RGB, thermal, LiDAR, and radar data with inertial measurements from Inertial Measurement Units (IMUs) and global positioning from a Global Navigation Satellite System (GNSS) with real-time kinematic (RTK) corrections.

## B. Cross-Modal Semantic Consistency

Beyond sensor coverage, multimodal learning requires semantic supervision that is consistent across modalities. Independent annotation can introduce conflicting boundaries and labels, motivating geometry-based approaches that propagate semantics through a shared representation. KITTI-360 [21] transfers labels between images and 3D geometry, while WildScenes [22] projects multi-view image annotations into LiDAR. UAVScenes [6] similarly annotates reconstructed 3D maps and renders them into camera frames before projecting labels onto LiDAR, while SegFly [5] adopts a 2D-3D-2D pipeline to lift sparse RGB annotations into a reconstructed point cloud and renders them into RGB and thermal images. These approaches demonstrate the feasibility of geometrydriven annotation, but remain limited in modality coverage, and require manual verification and correction.

MultiFly extends this paradigm to four modalities through a unified geometry-driven pipeline, propagating sparse RGB annotations to RGB, thermal, LiDAR, and radar, while ensuring high semantic consistency without the necessity of manual correction. It therefore offers not only broader sensor coverage, but a shared frame-wise semantic reference for multimodal UAV perception and cross-modal evaluation.

## III. MULTIFLY PLATFORM

## A. Design

To acquire multimodal aerial data, we propose the Multi-Fly platform shown in Fig. 2, which integrates one forwardfacing RGB camera, one thermal camera, one spinning LiDAR, two forward-facing radars, and a GNSS-RTK-aided inertial navigation system (INS), with detailed specifications listed in Table II. One radar is pitched downward by 14<sup>◦</sup> to increase the effective vertical field of view (FOV), while the whole sensor rig is pitched downward by $4 5 ^ { \circ }$ to jointly observe the flight direction and ground below the UAV. All perception sensors are calibrated with respect to the LiDAR reference frame and synchronized to a common time base, while the INS provides the platform pose. Perception data are recorded at 10 Hz on an Aetina AIB MX22 edge computing unit using Docker and ROS 2 Humble, together with dedicated onboard power (DOGCOM 10000 mAh Li-Ion), networking (NetGear GSM110MX switch), and storage hardware. Example synchronized RGB, thermal, LiDAR, and radar measurements are shown in Fig. 4.

TABLE II: Sensor specifications of the MultiFly sensor suite. For more details, see Section III.
<table><tr><td rowspan="2">Sensor</td><td rowspan="2">Model</td><td rowspan="2">Count</td><td rowspan="2">Frequency</td><td rowspan="2">Data Type</td><td colspan="3">Field of View</td><td colspan="3">Resolution</td></tr><tr><td>Range</td><td>Azimuth</td><td>Elevation</td><td>Range</td><td>Azimuth</td><td>Elevation</td></tr><tr><td>RGB Camera and Lens</td><td>FLIR Blackfly S GigE BFS-PGE-31S4C-C</td><td>1</td><td>10 Hz</td><td>8-bit Bayer image</td><td></td><td>62°</td><td>48°</td><td></td><td>2048 px</td><td>1536 px</td></tr><tr><td>Thermal Camera</td><td>with Tamron M112FM06 IRCAM NIK III</td><td>1</td><td>20 Hz raw</td><td>16-bit monochrome</td><td></td><td>57.5°</td><td>47.39°</td><td></td><td>1280 px</td><td>1024 px</td></tr><tr><td>LiDAR</td><td>microbolometer camera Ouster OS1-128</td><td>1</td><td>10 Hz selected 10Hz</td><td>thermal image x, y, z, intensity,</td><td>90 m recommended</td><td>360°</td><td>45°</td><td></td><td>2048 cols</td><td>128</td></tr><tr><td>Radar</td><td>Continental ARS 548 RDI</td><td>2</td><td>10 Hz</td><td>ambient, reflectivity x, y, z, velocity, RCS,</td><td>200 m maximum 301 m</td><td>90° selected 100°</td><td>8°</td><td>0.22 m</td><td>per scan 1.2° near boresight</td><td>channels 2.3º</td></tr><tr><td></td><td></td><td></td><td></td><td>range, azimuth, elevation time, latitude, longitude, altitude,</td><td>1514 m extended</td><td>120° processed</td><td>28° below 100 m</td><td></td><td>1.68°at ±45°</td><td></td></tr><tr><td>/ INS</td><td>OxTS xRED with dual GNSS receivers</td><td>1</td><td>100 Hz</td><td>linear velocity, angular velocity, yaw, pitch, roll, status</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/600a06657d7650f907c03a4409fe3b859c169b96bf76883cf0e34e0ae43e1245.jpg)  
Fig. 2: MultiFly sensor rig (see Section III-A).

## B. Time Synchronization

All sensors are synchronized to a common GNSSdisciplined time base using the Precision Time Protocol (PTP). The INS acts as the PTP grandmaster, while the Aetina computer operates as a boundary clock and distributes the time reference to the sensor network. While PTP establishes a common time base, sensor-specific triggering and phase compensation align the actual acquisition times across modalities. More specifically, sensor acquisition is aligned with the LiDAR’s 10 Hz scan cycle, corresponding to a fixed rotation period of 100 ms. The rotating LiDAR generates a hardware trigger for the forward-facing RGB camera, with a configurable delay accounting for the exposure time and the camera’s orientation relative to the LiDAR scan, effectively aligning the acquisition with the LiDAR beam passing through the camera FOV. The radars operate at the same nominal frequency, but their measurement epochs have a deterministic phase offset relative to the LiDAR cycle, which is compensated using the driver’s cycle\_offset parameter, resulting in radar–LiDAR deviations within ±3 ms. Since the thermal camera does not support hardware triggering, its acquisition is controlled using timestamped GigE Vision Scheduled Action Commands, which are executed according to its PTP-synchronized clock.

![](images/887fdffbbdbd0591b9f8bb3e9c419c2182fb77ddbca261e9a9d2495ed5178daa.jpg)  
Fig. 3: Registration results (see Section III-D).

## C. Sensor Calibration and Registration

Extrinsics. We organize the sensor extrinsics as a calibration tree rooted at the LiDAR, which serves as the common reference frame. The tree consists of three edges:

a) RGB–LiDAR: From a static recording of a structured scene, we manually select natural landmarks that are sharp in both modalities and estimate the rigid transform by solving a perspective-n-point problem.

b) Radar–LiDAR: Corner reflectors are placed in the shared sensor FOV. Corresponding radar and LiDAR detections are paired and the rigid transform is recovered in closed form via Singular Value Decomposition.

c) RGB–Thermal: The two cameras are stereocalibrated on checkerboard corners observed by both sensors.

Composing the RGB–Thermal and RGB–LiDAR transforms places the thermal camera in the LiDAR frame.

Intrinsics. We calibrate the RGB camera and the thermal camera independently with Zhang’s planar method [23] from multiple checkerboard views. The RGB target is a conventional checkerboard, while the thermal target is a steel plate carrying plotter-cut vinyl squares, whose emissivity contrast against the exposed steel renders the corners detectable in the thermal band. LiDAR and radar use their factory intrinsics.

## D. Registration.

Radar detections are transformed into the LiDAR frame, and LiDAR-frame measurements are projected into either image using the calibrated extrinsics, intrinsics, and distortion models.

Furthermore, RGB–thermal registration requires scene geometry because the nonzero camera baseline induces depthdependent parallax. Consequently, the calibrated rigid extrinsic alone does not define a dense pixel-to-pixel mapping. In principle, the synchronized and calibrated LiDAR measurements could provide the required depth for reprojecting RGB observations into the thermal view. Instead, we follow SegFly [5] and use the metric RGB-based 3D reconstruction required by the label-transfer pipeline (Section IV-A). This reconstruction already provides scene-wide geometry and RGB image–point correspondences. Naturally, using the same geometric representation for both registration and semantic label transfer ensures consistency between both stages. Our choice therefore does not reflect a limitation of LiDAR, but avoids introducing a separate geometric representation for registration. Moreover, unlike SegFly [5], whose acquisition setting requires an independent thermal reconstruction and iterative closest point (ICP) alignment, our cameras are synchronized and rigidly calibrated. We therefore retain the fixed RGB–thermal extrinsic and require neither thermal reconstruction nor ICP.

Technically, for each synchronized RGB–thermal pair, the thermal camera pose is obtained by composing the corresponding RGB reconstruction pose with the fixed RGB– thermal extrinsic. Reconstructed landmarks jointly visible in both cameras are projected into their undistorted image planes using the calibrated camera models. Because each pair of projections represents the same physical 3D point, it identifies where scene content visible to the thermal camera appears in the corresponding RGB frame. Following SegFly, these projected landmarks determine the common image support used to warp the original RGB image. In particular, the RGB image is resampled from its distorted image plane into a common undistorted pinhole domain and subsequently mapped into the thermal distortion space. The valid overlap is cropped and resized to the thermal resolution, while pixels mapped outside the valid common support are masked. This process yields a registered RGB image on the pixel grid of every thermal image.

Registration results in Fig. 3 exhibit precise cross-modal alignment and high geometric fidelity.

## IV. ANNOTATION-EFFICIENT LABEL TRANSFER

Our goal is to generate dense semantic annotations for all four sensor modalities from only a sparse set of manually annotated RGB source views. With these, we aim to produce the complete collections of time-synchronized, framewise semantic labels $\widehat { \mathcal { V } } ^ { \mathrm { R G B } } , \widehat { \mathcal { V } } ^ { \mathrm { T h } } , \widehat { \mathcal { V } } ^ { \mathrm { L i } \widetilde { \mathrm { d } } }$ and $\widehat { \mathcal { V } } ^ { \mathrm { R a d } }$ for the RGB, thermal, LiDAR and radar modalities, respectively. The following subsections describe how these four sets are obtained via geometry-driven label transfer. While the individual transfer operations build on established geometrydriven methods [5], [21], [22], our contribution is their modality-specific adaptation and integration into a unified aerial framework that derives consistent RGB, thermal, Li-DAR, and radar labels from the same sparse set of manually annotated RGB views.

## A. RGB and Thermal Semantic Label Transfer

We build on SegFly’s geometry-driven 2D–3D–2D labeltransfer pipeline [5] and use the shared RGB reconstruction and calibrated RGB–thermal geometry introduced in Section III-D.

1) 2D-to-3D: The classical image-based RGB 3D reconstruction introduced in Section III-D provides the point cloud $\mathcal { P } ^ { \mathrm { R G B } }$ and the image–point correspondences. Manual semantic annotations from a sparse set of spatially stratified RGB source views are then lifted into the point cloud via the image–point correspondences that follow naturally from 3D reconstruction. Subsequent neighborhood-based completion then yields the RGB-derived semantic point cloud

$$
\mathcal { P } _ { \mathrm { s e m } } ^ { \mathrm { R G B } } = \{ ( \mathbf { x } _ { m } ^ { \mathrm { R G B } } , c _ { m } ) \} _ { m = 1 } ^ { M _ { \mathrm { R G B } } } .\tag{1}
$$

2) 3D-to-2D: The semantic point cloud $\mathcal { P } _ { \mathrm { s e m } } ^ { \mathrm { R G B } }$ is then rendered into all remaining RGB images and into the thermal images using SegFly’s visibility-aware Z-buffered rendering and the camera poses described in Section III-D. This produces complete sets of dense pseudo-label maps

$$
\begin{array} { r } { \widehat { \mathcal { V } } ^ { \mathrm { R G B } } = \{ \widehat { \mathbf { Y } } _ { n } ^ { \mathrm { R G B } } \} _ { n = 1 } ^ { N _ { \mathrm { R G B } } } \quad \mathrm { a n d } \quad \widehat { \mathcal { V } } ^ { \mathrm { T h } } = \{ \widehat { \mathbf { Y } } _ { n } ^ { \mathrm { T h } } \} _ { n = 1 } ^ { N _ { \mathrm { T h } } } . } \end{array}\tag{2}
$$

Because registration and semantic rendering use the same metric reconstruction and fixed RGB–thermal extrinsic, the resulting pseudo-labels are geometrically consistent across both modalities.

## B. LiDAR and Radar Semantic Label Transfer

The manually annotated RGB source views introduced in Section IV-A are selected in a spatially stratified manner and thus provide broad spatial coverage of each scene despite sparse manual supervision. This coverage allows their semantic labels to be transferred to the LiDAR and radar measurements without requiring additional modality-specific manual annotation.

1) LiDAR Label Transfer: We process each scene at each altitude independently. The individual LiDAR scans are transformed into the common scene frame and aggregated into the scene-level point cloud

$$
\mathcal { P } ^ { \mathrm { L i d } } = \{ \mathbf { x } _ { m } ^ { \mathrm { L i d } } \} _ { m = 1 } ^ { M _ { \mathrm { L i d } } } ,
$$

while retaining the association of every point with its original scan. To transfer the manually annotated semantics, we render $\mathcal { P } ^ { \mathrm { L i d } }$ into each annotated RGB source view using visibility-aware (Z-buffered) rendering and sample the corresponding labels, similar to the strategies of [21], [22]. A point may receive observations from multiple views. We therefore assign the majority class if its relative frequency exceeds $\tau _ { \mathrm { L i d } } = 0 . 7 5$ . Otherwise, the point is marked as unlabeled. This yields the semantic LiDAR point cloud

$$
\mathcal { P } _ { \mathrm { s e m } } ^ { \mathrm { L i d } } = \{ ( \mathbf { x } _ { m } ^ { \mathrm { L i d } } , \hat { c } _ { m } ^ { \mathrm { L i d } } ) \} _ { m = 1 } ^ { M _ { \mathrm { L i d } } } ,\tag{3}
$$

where ${ \widehat { c } } _ { m } ^ { \mathrm { L i d } } \in { \mathcal { C } } \cup \{ c _ { \mathrm { u n l } } \}$ . Using the retained point-to-scan associations, the labels are mapped back to the original scans. This produces the complete set of frame-wise semantic LiDAR annotations

$$
\widehat { \mathcal { V } } ^ { \mathrm { L i d } } = \{ \widehat { \mathbf { Y } } _ { k } ^ { \mathrm { L i d } } \} _ { k = 1 } ^ { K _ { \mathrm { L i d } } } ,\tag{4}
$$

where $K _ { \mathrm { L i d } }$ is the number of LiDAR scans. Points labeled $c _ { \mathrm { u n l } }$ are treated as ignore labels during supervised training.

2) Radar Label Transfer: We treat the two radar sensors as a single logical radar for clarity. Analogous to the LiDAR case, the individual radar scans are transformed into the common scene frame and aggregated into a scene-level point cloud

$$
\mathcal { P } ^ { \mathrm { R a d } } = \{ \mathbf { x } _ { m } ^ { \mathrm { R a d } } \} _ { m = 1 } ^ { M _ { \mathrm { R a d } } } ,\tag{5}
$$

while retaining the association of every point with its original scan. Because radar measurements contain a considerable number of ghost targets and other outliers (18.8% on average in our data), we apply two consecutive pre-processing steps before label transfer. First, a RANSAC-based plane fit $( \tau = 0 . 2 \mathrm { m } )$ estimates the ground surface. Points lying more than two standard deviations below the fitted plane are discarded to preserve genuine ground returns while removing clear outliers. Second, DBSCAN clustering $( \varepsilon \ = \ 1 . 5 \mathrm { m } ,$ min\_points = 25) is applied to remove remaining sparse ghost detections. Points removed by either step are marked as invalid $( c _ { \mathrm { i n v } } )$ . After pre-processing, semantic labels are transferred exactly as for LiDAR: the cleaned point cloud is rendered into the annotated RGB source views using visibility-aware (Z-buffered) rendering, and each point is assigned the majority class if its relative frequency exceeds $\tau _ { \mathrm { R a d } } = 0 . 7 5$ . Otherwise, it is marked unlabeled. This yields the semantic radar point cloud

$$
\mathcal { P } _ { \mathrm { s e m } } ^ { \mathrm { R a d } } = \{ ( \mathbf { x } _ { m } ^ { \mathrm { R a d } } , \hat { c } _ { m } ^ { \mathrm { R a d } } ) \} _ { m = 1 } ^ { M _ { \mathrm { R a d } } } ,\tag{6}
$$

where ${ \widehat { c } } _ { m } ^ { \mathrm { R a d } } \in { \mathcal { C } } \cup \{ c _ { \mathrm { u n l } } , c _ { \mathrm { i n v } } \}$ . Using the retained point-toscan associations, the labels are mapped back to the original

TABLE III: MultiFly dataset statistics (see Section V-B).
<table><tr><td rowspan="3">Statistic</td><td colspan="7">MultiFly Scenes and Altitudes</td><td rowspan="3">Total</td></tr><tr><td colspan="2">#1</td><td>#2</td><td colspan="2">#3</td><td>#4</td></tr><tr><td>30m</td><td>50m</td><td>30m 50m</td><td>30m 50m</td><td>30m</td><td></td><td>50m</td></tr><tr><td>Synchronized samples</td><td>2,209 2,903</td><td></td><td>1,987 2,166</td><td>1,766 2,214</td><td></td><td>2,097 1,930</td><td></td><td>17,272</td></tr><tr><td>IMU/GNSS-RTK pose</td><td>√</td><td>√</td><td>√ √</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>Covered area [m²]</td><td>17,400</td><td></td><td>11,100</td><td></td><td>13,000</td><td>15,200</td><td></td><td>56,700</td></tr><tr><td></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td></td><td></td></tr></table>

![](images/2e2e64a78d93ea9e5a84f53e686e61f8b9bf63ab0ad37279fbf350b9ef268ead.jpg)  
Fig. 4: MultiFly sample with semantic classes (Sec. V-B).

scans, producing the complete set of frame-wise semantic radar annotations

$$
\begin{array} { r } { \widehat { \mathcal { V } } ^ { \mathrm { R a d } } = \{ \widehat { \mathbf { Y } } _ { k } ^ { \mathrm { R a d } } \} _ { k = 1 } ^ { K _ { \mathrm { R a d } } } . } \end{array}\tag{7}
$$

Points labeled $c _ { \mathrm { u n l } }$ or $c _ { \mathrm { i n v } }$ can be treated as ignore labels during supervised training.

## V. MULTIFLY DATASET

## A. Data Acquisition and Source Annotation

We collect MultiFly using the EmQopter Q6500 UAV (Fig. 1) together with the calibrated and time-synchronized platform described in Section III. The acquisition campaign comprises four suburban scenes within the metropolitan area of Ingolstadt in Germany, recorded during summer under sunny and cloudy conditions. Each scene is surveyed at nominal flight altitudes of 30 m and 50 m above ground level. At each altitude, the UAV follows an automated doublegrid trajectory, in line with the acquisition strategy of Occu-Fly [24]. The flight speed is $\mathrm { 3 m s ^ { - 1 } }$ and the flight lines are planned with a nominal RGB side overlap of 40 %. During all flights the sensor stack is pitched downward by $4 5 ^ { \circ }$ relative to the horizontal to observe both the flight direction and the terrain below. To reduce temporal redundancy while retaining meaningful scene variation, the synchronized 10 Hz perception streams are subsampled by selecting every second frame at 30 m and every third frame at 50 m. The INS records at 100 Hz and its complete navigation stream is preserved at the original temporal resolution. The LiDAR records a full 360<sup>◦</sup> azimuth during acquisition. During offline preprocessing, each LiDAR scan is restricted to the $9 0 °$ sector facing forward and aligned with the RGB camera.

For label propagation, 115 RGB images are manually annotated using the 15 semantic classes adopted from SegFly [5], and constitute the only manual annotations used for data generation. No thermal images, LiDAR points, or radar points are manually annotated.

TABLE IV: Modal-wise label transfer evaluation (Sec. VI).
<table><tr><td>Altitude</td><td>RGB</td><td>Thermal</td><td>LiDAR</td><td>Radar</td><td>Average</td></tr><tr><td>30m</td><td>91.78%</td><td>90.51 %</td><td>91.80 %</td><td>85.92 %</td><td>90.00 %</td></tr><tr><td>50m</td><td>92.47%</td><td>90.11%</td><td>91.11 %</td><td>85.76 %</td><td>89.86%</td></tr><tr><td>Average</td><td>92.12%</td><td>90.31%</td><td>91.45%</td><td>85.84%</td><td>89.93 %</td></tr></table>

## B. Dataset Statistics

Table III summarizes the final MultiFly dataset by scene and acquisition altitude. MultiFly contains 17,272 synchronized multimodal samples. Each sample contains one RGB image, one corresponding thermal image, one LiDAR scan restricted to the forward 90<sup>◦</sup> sector, and one measurement set from each of the two radar sensors. Every sample is associated with a six degree of freedom pose obtained from the INS. The complete 100 Hz navigation stream is additionally provided at its native temporal resolution, preserving all measurements between consecutive 10 Hz perception samples.

MultiFly provides frame-wise semantic annotations in a shared taxonomy of 15 classes. RGB and thermal images are accompanied by pixel-wise semantic label maps, while LiDAR and radar measurements are accompanied by pointwise semantic labels. The complete semantic taxonomy and representative examples are shown in Fig. 4, and the dataset is organized hierarchically by scene, acquisition altitude, modality, and synchronized sample. Calibration parameters, including intrinsic and extrinsic parameters, as well as synchronization metadata and the complete navigation stream are provided alongside the perception data.

## VI. DATASET EVALUATION

## A. Modal-Wise Label Transfer Evaluation

RGB & Thermal. We follow SegFly’s RGB–thermal evaluation protocol [5] on MultiFly scene #4. The generated pseudo-labels are evaluated against 40 frames (10 for RGB and 10 for thermal at both altitudes) that were manually annotated solely for evaluation and held out from all stages of data generation (annotation cost: 29 min on average per frame). Results are quantified in Table IV. On average, MultiFly achieves 92.12 % accuracy for RGB and 90.31 % for thermal, on par with SegFly (91.27 % / 87.65 %).

LiDAR & Radar. We evaluate the transferred 3D labels by rendering the annotated LiDAR and radar scans into the same held-out RGB labels using the identical visibility-aware Z-buffered procedure employed during data generation. Agreement with the held-out manual annotations therefore measures both semantic label quality and geometric consistency of the LiDAR/radar point clouds with the RGB camera. MultiFly achieves high-quality label transfer accuracy with 91.45% for LiDAR and 85.84% for radar.

Across all four modalities, the average accuracy is 89.93%, confirming high semantic and geometric fidelity of our annotation-efficient label transfer.

## B. Cross-Modal Semantic Consistency Evaluation

Evaluation Protocol. We evaluate the semantic consistency of the final MultiFly labels across all modality pairs without relying on intermediate 3D representations or additional manual ground truth. Results are summarized in Table V. We distinguish three types of modality pairs:

TABLE V: Cross-modal semantic consistency across all modalities and scenes of MultiFly (see Section VI-B).
<table><tr><td rowspan="2">Modality Pair</td><td colspan="4">MultiFly Scene</td><td rowspan="2">Average</td></tr><tr><td>#1</td><td>#2</td><td>#3</td><td>#4</td></tr><tr><td colspan="6">2D-2D</td></tr><tr><td>RGB-Thermal</td><td>90.82%</td><td>89.83%</td><td>91.97%</td><td>91.47%</td><td>91.02 %</td></tr><tr><td colspan="6">2D-3D</td></tr><tr><td>RGB-LiDAR</td><td>91.09%</td><td>91.41%</td><td>90.96 %</td><td>92.79%</td><td>91.56%</td></tr><tr><td>RGB-Radar</td><td>91.89%</td><td>89.14%</td><td>85.84%</td><td>88.07%</td><td>88.73%</td></tr><tr><td>Thermal-LiDAR</td><td>91.37%</td><td>91.79%</td><td>91.21%</td><td>93.15%</td><td>91.88%</td></tr><tr><td>Thermal-Radar</td><td>92.16 %</td><td>89.52%</td><td>86.11%</td><td>88.44%</td><td>89.06 %</td></tr><tr><td colspan="6">3D-3D</td></tr><tr><td>LiDAR-Radar</td><td>95.40 %</td><td>93.16 %</td><td>92.55 %</td><td>92.58 %</td><td>93.42%</td></tr><tr><td>Average</td><td>92.12%</td><td>90.81%</td><td>89.77%</td><td>91.08%</td><td>90.94%</td></tr></table>

![](images/80034764f5083bb2ccbf055bc5e065edca784f474c2296e8f047e088249accff.jpg)  
Fig. 5: Cross-modal semantic consistency (Sec. VI-B).

2D–2D (RGB–Thermal): We reuse the geometry-driven 2D–3D–2D registration (Section III-D), but apply it to the RGB semantic labels instead of the RGB image. The labels are warped onto the thermal image grid via the recovered mapping, after which pixel-wise agreement with the thermal labels is measured.

2D–3D (RGB/Thermal – LiDAR/Radar): For each synchronized frame, we project the LiDAR or radar points into the corresponding RGB or thermal image using the calibrated extrinsics, intrinsics. The semantic label of each valid projected point is compared with the label of the projected pixel.

3D–3D (LiDAR–Radar): Since radar returns are substantially sparser than LiDAR point clouds, we establish crossmodal correspondences using nearest-neighbor association. For each radar point, we identify the closest LiDAR point within a fixed Euclidean distance threshold r and compare their semantic labels. We set r = 1 m, which provides a correspondence for 71 % of the radar points. Semantic agreement is then determined over these points. We further evaluate the sensitivity to the choice of r in the results.

For all modality pairs, only pixels/points that yield a valid semantic class (i.e., neither $c _ { \mathrm { u n l } } \ n o r \ c _ { \mathrm { i n v } } )$ are considered. We report consistency as the percentage of agreeing label pairs.

Results. As shown in Table V, MultiFly achieves an average cross-modal semantic consistency of 90.94 % across all six modality pairs and all four scenes. For the LiDAR–Radar pair, the consistency remains robust to the choice of association radius, with $r ~ = ~ \{ 0 . 1 , 0 . 2 5 , 0 . 5 , 1 . 0 , 2 . 0 \}$ m yielding {96, 95, 95, 93, 93} % agreement, respectively, while an unconstrained nearest-neighbor association (r = ∞) yields 86 %. Since all evaluations rely on the calibrated intrinsics, extrinsics, and the RGB–thermal registration, the high agreement also serves as a proxy for geometric fidelity.

TABLE VI: RGB and Thermal experiments (Section VII-B).
<table><tr><td></td><td>UPerNet [26]</td><td>SegFormer [27]</td><td>Firefly [5]</td></tr><tr><td colspan="4">RGB</td></tr><tr><td>Mean accuracy</td><td>53.17%</td><td>55.62%</td><td>58.21%</td></tr><tr><td>Mean intersection over union</td><td>39.96%</td><td>41.77%</td><td>44.05%</td></tr><tr><td colspan="4">Thermal</td></tr><tr><td>Mean accuracy</td><td>44.22%</td><td>47.81%</td><td>53.42%</td></tr><tr><td>Mean intersection over union</td><td>32.08%</td><td>34.71%</td><td>40.24%</td></tr></table>

![](images/aa2db619da64fecde90eeaef86f0c85afd63ae7614d629bbbf8327dcadfc3186.jpg)  
Fig. 6: RGB and Thermal experiments (see Section VII-B).

Notably, this performance is achieved although our approach employs static 3D reconstruction for label propagation and point-cloud aggregation, which could affect dynamic objects. While 4D reconstruction methods [25] are a promising direction for addressing this limitation in future work, our results confirm that our label transfer ensures both semantically consistent and geometrically coherent multimodal aerial data under this challenging setting.

## C. Annotation Efficiency

Only 115 (0.67%) of MultiFly’s 17,272 RGB frames serve as manually annotated source views, taking 29 min on average per frame. From this small subset, the geometrydriven pipeline automatically labels the remaining 17, 157 RGB frames, all 17, 272 thermal frames, $8 . 4 \times 1 0 ^ { 8 }$ LiDAR points, and $3 . 4 \times 1 0 ^ { 6 }$ radar points. No thermal frame or 3D point is manually annotated for data generation. The resulting labels exhibit high semantic quality within each modality (Table IV), and strong semantic consistency across modalities (Table V). Taken together, these results provide strong empirical evidence that sparse RGB annotations can be automatically propagated at scale to generate high-quality, semantically consistent 2D and 3D labels across four modalities without manual refinement.

## VII. BENCHMARK EXPERIMENTS

## A. Experimental Setup

We benchmark established semantic segmentation architectures to assess whether our dataset provides effective supervision for learning semantic representations from its annotated modalities. The RGB and thermal benchmarks follow the cross-modal training protocol of SegFly [5], including RGB-to-thermal adaptation using registered image pairs. LiDAR and radar models are trained independently for their respective modalities. The benchmark therefore evaluates both cross-modal image adaptation and modalityspecific 3D segmentation, while dataset-level semantic consistency is assessed separately in Section VI-B.

TABLE VII: LiDAR and Radar experiments (Section VII-C).
<table><tr><td></td><td>SpUNet [28]</td><td>LitePT [29]</td><td>PTv3 [30]</td></tr><tr><td colspan="4">LiDAR</td></tr><tr><td>Mean accuracy</td><td>49.49%</td><td>49.91%</td><td>51.76%</td></tr><tr><td>Mean intersection over union</td><td>34.72%</td><td>35.62%</td><td>37.09%</td></tr><tr><td colspan="4">Radar</td></tr><tr><td>Mean accuracy</td><td>25.95%</td><td>21.34%</td><td>22.68%</td></tr><tr><td>Mean intersection over union</td><td>14.50%</td><td>12.09%</td><td>12.90%</td></tr></table>

![](images/be8e7e32335158657cc263527e31f77d4627f123a786c02872dc920dde55d33a.jpg)  
Fig. 7: LiDAR and Radar experiments (see Section VII-C).

Leveraging MultiFly’s four distinct scenes, we adopt a strict held-out-scene protocol to assess scene-level generalization, using scenes 1–3 exclusively for training and scene 4 exclusively for final evaluation. All models are evaluated using their official configurations.

## B. RGB and Thermal Semantic Segmentation

We benchmark established UPerNet-Swin-S [26], SegFormer-MiT-B3 [27], and Firefly [5] for RGB and thermal semantic segmentation. All methods use their official implementations and configurations, including pretrained encoders or backbones, but without external taskspecific segmentation checkpoints. Following SegFly [5], training comprises three stages: RGB training, RGB-tothermal adaptation using registered MultiFly RGB–thermal pairs, and thermal fine-tuning. All models are trained for 30 epochs for RGB training, 10 epochs for RGB-to-thermal adaptation, and 30 epochs for thermal fine-tuning. We report the final-epoch results.

As shown in Table VI, Firefly achieves the strongest performance for both RGB and thermal semantic segmentation in terms of mean intersection over union (mIoU) and mean accuracy (mAcc). Qualitative predictions for a representative sample in Fig. 6 are consistent with the quantitative results, demonstrating that MultiFly supports effective semantic segmentation across both modalities.

## C. LiDAR and Radar Semantic Segmentation

We benchmark state-of-the-art SparseUNet (SpUNet) [28], LitePointTransformer-S (LitePT) [29], and PointTransformerV3 (PTv3) [30] independently on the LiDAR and radar labels. All methods use their official implementations and configurations unless stated otherwise. In addition to 3D coordinates, the LiDAR inputs include intensity, while the radar inputs include radar cross section (RCS) and Doppler velocity, in line with [14]. Each model is trained separately for LiDAR and radar semantic segmentation for 20 epochs, and we report the final-epoch results. For radar, the only configuration change is an increase in the voxelization gridsize from 0.05 m to 0.50 m, providing a coarser discretization to account for the lower density of radar points.

As shown in Table VII, PTv3 performs best on LiDAR, whereas SpUNet achieves the strongest radar performance. Prior work identifies sparse-convolutional approaches for sparser radar point clouds as an understudied area [14]. By benchmarking SpUNet against recent point-transformer architectures, we directly address this gap and find that established sparse convolutions can outperform recent attentionbased models. This result provides a promising initial benchmark for radar semantic segmentation, while controlled studies are required to disentangle the effects of individual architectural choices. Qualitative predictions for a representative sample in Fig. 7 further support the quantitative results.

## VIII. CONCLUSION

We introduce MultiFly, a real-world low-altitude UAV dataset with synchronized RGB, thermal, LiDAR, and radar semantic annotations. By propagating labels from only 115 manually annotated RGB images, MultiFly provides dense supervision across 17,272 multimodal samples with high annotation agreement and cross-modal consistency. Our benchmarks further establish initial semantic segmentation baselines for all four modalities.

## REFERENCES

[1] D. Manjunath, A. Sikdar, P. Gurunath, S. Udupa, and S. Sundaram, “Saga: Semantic-aware gray color augmentation for visible-to-thermal domain adaptation across multi-view drone and ground-based vision systems,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. IEEE, 2025, pp. 4578–4588.

[2] L. Cheng, W. Ji, H. Jin, J. Li, W. Li, and Y. Shen, “Unleashing multispectral video’s potential in semantic segmentation: A semisupervised viewpoint and new UAV-view benchmark,” in Advances in Neural Information Processing Systems 37, 2024, pp. 65 717–65 737.

[3] C. Lee et al., “Caltech aerial rgb-thermal dataset in the wild,” in European Conference on Computer Vision (ECCV). Springer, 2024.

[4] J. OuYang, Q. Wang, Y. Shang, P. Jin, H. Zhong, L. Zhou, and T. Shen, “An rgb-tir dataset from uav platform for robust urban traffic scenes semantic segmentation,” Sci Data, vol. 12, p. 1701, 2025.

[5] M. Gross, S. B. Matha, R. Song, V. Muthuveerappan, C. Christoph, J. Huber, and D. Cremers, “SegFly: A Dataset and 2D-3D-2D Paradigm for Aerial RGB-Thermal Semantic Segmentation at Scale,” in Proceedings of the European Conference on Computer Vision (ECCV), 2026.

[6] S. Wang, S. Li, Y. Zhang, S. Yu, S. Yuan, R. She, Q. Guo, J. Zheng, O. K. Howe, L. Chandra et al., “Uavscenes: A multi-modal dataset for uavs,” arXiv preprint arXiv:2507.22412, 2025.

[7] M. Gross, A. Greiner, S. B. Matha, F. Soest, D. Cremers, and H. Meeß, “Safeland: Safe autonomous landing in unknown environments with bayesian semantic mapping,” in Proceedings of the European Conference on Computer Vision (ECCV) Workshops, 2026.

[8] T. Nguyen, S. S. Shivakumar, I. D. Miller, J. Keller, E. S. Lee, A. Zhou, T. Özaslan, G. Loianno, J. H. Harwood, J. Wozencraft, C. J. Taylor, and V. Kumar, “Mavnet: An effective semantic segmentation micronetwork for mav-based tasks,” IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3908–3915, 2019.

[9] S. Bultmann, J. Quenzel, and S. Behnke, “Real-time multi-modal semantic fusion on unmanned aerial vehicles with label propagation for cross-domain adaptation,” Robotics and Autonomous Systems, vol. 159, p. 104286, 2023.

[10] I. Nigam, C. Huang, and D. Ramanan, “Ensemble knowledge transfer for semantic segmentation,” in 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), 2018, pp. 1499–1508.

[11] G. J. Brostow, J. Fauqueur, and R. Cipolla, “Semantic object classes in video: A high-definition ground truth database,” Pattern Recognition Letters, 2008.

[12] D. Munoz, J. A. Bagnell, N. Vandapel, and M. Hebert, “Contextual classification with functional max-margin Markov networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2009.

[13] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 11 621–11 631.

[14] H. Arroyo, P. Keir, D. Angus, S. Matalonga, S. Georgiev, M. Goli, G. Dooly, and J. Riordan, “Segmentation of drone collision hazards in airborne radar point clouds using pointnet,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 11, 2024.

[15] C. Lee, D. Ko, M. B. Maru, K. Jang, W. Choi, G. Cha, and S. Park, “Uav lidar bridge point cloud dataset and hybrid deep learning framework for robust semantic segmentation,” Automation in Construction, vol. 188, p. 107045, 2026.

[16] H. Li, Y. Zou, N. Chen, J. Lin, X. Liu, W. Xu, C. Zheng, R. Li, D. He, F. Kong, Y. Cai, Z. Liu, S. Zhou, K. Xue, and F. Zhang, “MARS-LVIG dataset: A multi-sensor aerial robots SLAM dataset for LiDAR-visualinertial-GNSS fusion,” The International Journal ofRobotics Research (IJRR), vol. 43, no. 8, pp. 1114–1127, Jul. 2024.

[17] G. Rizzoli, F. Barbato, M. Caligiuri, and P. Zanuttigh, “Syndronemulti-modal uav dataset for urban scenarios,” in Proceedings of the International Conference on Computer Vision, 2023, pp. 2210–2220.

[18] L. Zou, J. Wang, R. Liang, H. Wu, K. Chen, and Y. Wang, “Uavmm3d: A large-scale synthetic benchmark for 3d perception of unmanned aerial vehicles with multi-modal data,” arXiv preprint arXiv:2511.22404, 2025.

[19] M. Kölle, D. Laupheimer, S. Schmohl, N. Haala, F. Rottensteiner, J. D. Wegner, and H. Ledoux, “The hessigheim 3d (h3d) benchmark on semantic segmentation of high-resolution 3d point clouds and textured meshes from uav lidar and multi-view-stereo,” ISPRS Open Journal of Photogrammetry and Remote Sensing, vol. 1, p. 11, 2021.

[20] L. Lin, Y. Liu, Y. Hu, X. Yan, K. Xie, and H. Huang, “Capturing, reconstructing, and simulating: the urbanscene3d dataset,” in Proceedings of the European Conference on Computer Vision (ECCV), 2022.

[21] Y. Liao, J. Xie, and A. Geiger, “Kitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 3, pp. 3292–3310, 2023.

[22] K. Vidanapathirana, J. Knights, S. Hausler, M. Cox, M. Ramezani, J. Jooste, E. Griffiths, S. Mohamed, S. Sridharan, C. Fookes, and P. Moghadam, “Wildscenes: A benchmark for 2d and 3d semantic segmentation in large-scale natural environments,” The International Journal of Robotics Research, vol. 44, no. 4, pp. 532–549, 2025.

[23] Z. Zhang, “A flexible new technique for camera calibration,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 11, pp. 1330–1334, 2000.

[24] M. Gross, S. B. Matha, A. Fahmy, R. Song, D. Cremers, and H. Meess, “OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[25] S. Qian, G. Zhang, S. Wu, and D. Cremers, “Flow4r: Unifying 4d reconstruction and tracking with scene flow,” in Proceedings of the European Conference on Computer Vision (ECCV), 2026.

[26] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun, “Unified perceptual parsing for scene understanding,” in Proceedings of the European Conference on Computer Vision (ECCV), September 2018.

[27] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” in Neural Information Processing Systems, 2021.

[28] Pointcept Contributors, “Pointcept: A codebase for point cloud perception research,” https://github.com/Pointcept/Pointcept, 2023.

[29] Y. Yue, D. Robert, J. Wang, S. Hong, J. D. Wegner, C. Rupprecht, and K. Schindler, “LitePT: Lighter Yet Stronger Point Transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[30] X. Wu, L. Jiang, P.-S. Wang, Z. Liu, X. Liu, Y. Qiao, W. Ouyang, T. He, and H. Zhao, “Point transformer v3: Simpler, faster, stronger,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.