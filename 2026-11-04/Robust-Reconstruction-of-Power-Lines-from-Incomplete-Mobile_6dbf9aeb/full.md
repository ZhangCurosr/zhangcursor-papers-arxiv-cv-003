# Robust Reconstruction of Power Lines from Incomplete Mobile and Aerial LiDAR Data

Mohamed Gaha<sup>∗</sup>, Damos Ayobo Abongo<sup>∗†</sup>, Philippe Massicotte<sup>∗</sup>, Eya Ben Kacem<sup>∗†</sup>

Wael Jaafar<sup>†</sup>, Rami Langar<sup>†‡</sup>, and Christian Buteau<sup>∗</sup>

<sup>∗</sup>Hydro-Quebec, QC J3X 1S1, Canada

<sup>†</sup>Software and IT Engineering Department, École de Technologie Supérieure, Montreal, QC H3C 1K3, Canada <sup>‡</sup> LIGM-CNRS UMR 8049, University Gustave Eiffel, F-77420 Marne-la-Vallée, France

{gaha.mohamed, ayoboabongo.damos, massicotte.philippe, benkacem.eya, buteau.christian}@hydroquebec.com; {wael.jaafar, rami.langar}@etsmtl.ca

Preprint. This manuscript has been submitted to the IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (J-STARS) for peer review.

Abstract—Accurate vegetation-encroachment assessment and effective risk mitigation require electric utilities to precisely locate and characterize distribution-network assets, including poles and power lines, and to distinguish among line types such as mediumvoltage, low-voltage, neutral, and telecommunication lines. Mobile and aerial Light Detection and Ranging (LiDAR) point clouds provide high-resolution 3D representations of these assets and their surroundings, including vegetation, buildings, and roads. However, detecting and reconstructing power lines directly from LiDAR data remains challenging since conductor returns are often sparse, incomplete, noisy, or occluded by vegetation. These challenges are pronounced in distribution networks, where relatively low pole heights and complex line configurations place conductors in close proximity to trees. Moreover, mobile and aerial LiDAR point clouds differ substantially in point density, viewing geometry, spatial coverage, and occlusion patterns. In this work, we propose a robust and automated reconstruction method that combines scaled density-based clustering, robust polynomial fitting, and iterative fragment merging to recover individual conductors under severe data incompleteness and occlusion. Following semantic classification, the proposed reconstruction pipeline processes mobile and aerial LiDAR data using the same implementation and parameter configuration, without requiring modality-specific reconstruction code or parameter. We evaluate our method through controlled degradation experiments, comparative benchmarking, parameter sensitivity analysis, and large-scale validation on mobile and aerial LiDAR datasets. On the aerial dataset, the proposed method achieves a precision of 96.87%, a recall of 96.66%, an F<sub>1</sub>-score of 96.76%, and a geometric root mean square error (RMSE) of 0.127 m. On the mobile dataset, for which independent conductor-level reference annotations were unavailable, the method achieves a point-to-model RMSE of 0.045 m and a reconstruction coverage of 95.27%. To support reproducibility and facilitate further research, we publicly release the source code, documentation, and evaluation data available on https://github.com/varen-project.

Index Terms—mobile LiDAR, aerial LiDAR, power-line detection, curve fitting, vegetation risk.

## I. INTRODUCTION

Reconstructing power-line geometry is essential for electric utilities to assess vegetation encroachment and mitigate risks to distribution infrastructure. Light Detection and Ranging (Li-DAR) point clouds provide high-resolution 3D representations of utility assets and their surroundings, including vegetation, buildings, and roads [1]. Therefore, these data offer significant potential for automated power-network inspection, monitoring, and asset management. Nevertheless, reliable power-line detection and reconstruction from LiDAR point clouds remain challenging, particularly in distribution networks. Conductors and poles are often located close to trees and branches, resulting in frequent occlusions, sparse conductor returns, and confusion between power-line points and nearby vegetation.

Traditional reconstruction methods, including voxel-based approaches [2], often exhibit limited robustness to incomplete or noisy point clouds and may incur substantial computational costs on large datasets. More recent context-aware approaches [3], [4], [5], [6] exploit pole locations, span geometry, and mathematical conductor models to estimate powerline shapes. Although they represent important advances, their performance may deteriorate under severe occlusion or substantial data incompleteness. Recent reviews have shown that many existing power-line extraction and reconstruction methods were evaluated on relatively small datasets and that comparatively few studies addressed complex multi-conductor configurations [7]. Furthermore, most have been evaluated on relatively limited datasets, leaving their robustness and scalability for large-scale power-network reconstruction insufficiently investigated.

Existing methods are designed for a single LiDAR acquisition modality. Mobile LiDAR provides dense and detailed ground-level measurements, but its observations may be affected by roadside obstacles, uneven sampling, and limited visibility of elevated conductors. In contrast, aerial LiDAR provides broader spatial coverage from a predominantly topdown perspective, but conductor returns may be sparser and more difficult to distinguish from nearby vegetation. These differences in point density, viewing geometry, spatial coverage, noise, and occlusion patterns complicate the development of a reconstruction method that performs consistently across both modalities. Consequently, a method developed for one modality may require substantial parameter retuning or implementation changes before being applied to the other.

Recent learning-based studies have investigated power-line point extraction from both aerial and mobile LiDAR data [8], unified conductor-level geometric reconstruction across these acquisition modalities remains insufficiently investigated. To the best of our knowledge, no previous powerline reconstruction method has been specifically designed and extensively evaluated on both mobile and aerial LiDAR data using the same reconstruction implementation and parameter configuration. Addressing this limitation is important for utilities that acquire heterogeneous LiDAR datasets and require a scalable solution without modality-specific reconstruction code or extensive manual calibration. Hence, we propose a robust and automated reconstruction framework that combines adaptive coordinate prescaling, density-based clustering, robust polynomial fitting, and iterative fragment merging to recover individual conductors under severe occlusion and data incompleteness. The main contributions of this work are as follows:

1) We develop a unified reconstruction framework for mobile and aerial LiDAR point clouds. Following modalityspecific semantic classification, the framework uses the same reconstruction implementation and parameter configuration for both acquisition modalities.

2) We introduce a robust reconstruction method that combines adaptive coordinate prescaling, HDBSCAN clustering, Huber-weighted polynomial fitting, iterative fragment merging, and a rescue stage for rejected conductor candidates.

3) We conduct an extensive evaluation that includes controlled degradation experiments, comparisons with existing reconstruction pipelines, and a comprehensive parameter-sensitivity analysis. The large-scale evaluation covers 1,620 mobile LiDAR spans over 68 km and 10,870 aerial LiDAR spans over 382 km.

4) We publicly release the source code, documentation, and evaluation data. To the best of our knowledge, these resources constitute the first publicly available benchmark that includes power-line reconstruction data from both mobile and aerial LiDAR on https://github. com/varen-project.

The remainder of this paper is organized as follows. Section II reviews related work on LiDAR-based power-line detection and reconstruction. Section III presents the proposed reconstruction method. Section IV describes the controlled robustness experiments, large-scale evaluations, comparative benchmarking, and parameter sensitivity analysis. Finally, Section V summarizes the main findings, limitations, and directions for future research.

## II. RELATED WORK

This section reviews representative methods for detecting, extracting, and reconstructing power lines from LiDAR point clouds. Particular attention is given to their assumptions, methodological strengths, and limitations regarding vegetation-induced occlusions, incomplete conductor observations, closely spaced phases, and differences between LiDAR acquisition modalities.

Jwa et al. [2] introduced the Voxel-based Piece-wise Line Detector (VPLD) for the automatic 3D reconstruction of power lines from airborne laser scanning (ALS) LiDAR data. Their method partitioned the point cloud into sub-cubic voxels and grouped neighboring voxels likely to contain conductor points. The resulting components were connected, and the parameters of a catenary model were iteratively refined to produce continuous conductor reconstructions. This piecewise strategy recovered conductor geometry from locally detected observations, but its performance deteriorated when conductor returns were sparse or heavily occluded by vegetation. Its voxel-based iterative processing could also incur substantial computational costs on large datasets.

Liang and Zhang [9] proposed a method for extracting power lines from ALS LiDAR data. Their method identified candidate conductor points through geometric filtering and applied a Hough-like line detector to organize and connect the extracted observations. The resulting points were modeled in 3D space using a parabolic equation whose parameters were estimated by least-squares fitting. However, the method became less effective under substantial occlusion because large observation gaps prevented fragments belonging to the same conductor from being correctly associated.

Guo et al. [5] developed a contextual reconstruction method for ALS data. Their approach used the locations of two pylons delimiting a span to constrain the conductor search region. This region was partitioned into equally spaced cuboids, and candidate wire points were grouped according to their geometric characteristics and a neighborhood-based ratio. The conductors were then reconstructed using quadratic fitting and an iterative RANSAC procedure [10]. By incorporating span-level contextual information, their method reconstructed conductors even when portions of their LiDAR observations were missing. Nevertheless, it assumed that the conductor configuration remained sufficiently consistent throughout each span, which limited its applicability to spans containing substantial horizontal or vertical configuration changes.

For mobile laser scanning (MLS) data, Gaha et al. [4] proposed a contextual method for detecting and classifying utility poles and power lines. The search for conductor candidates was restricted to points located near a plane defined by two consecutive poles, thereby reducing the number of irrelevant observations and the processing complexity. RANSAC was used to extract approximately linear conductor segments, while RGB information was used to filter the resulting candidates. Subsequently, each conductor was modeled using three points, i.e., two points associated with consecutive poles and one point selected from the remaining candidates. The method was evaluated on 98 poles and achieved a recall of 98% and an accuracy of 84%. However, the experiments were limited to single-phase configurations, and the three-point model was sensitive to candidate-point misclassification, particularly under severe occlusion. In a subsequent study, Massicotte et al. [11] extended this approach by combining mobile LiDAR and image information to support utility-asset inventory. The reported accuracy ranged from 85% to 95%.

Also, Shi et al. [6] investigated power-line extraction from MLS data in urban environments. Their method combined geometric feature extraction and clustering to identify powerline points and support hazard analysis. Although designed for dense ground-level observations, the method was not evaluated on aerial LiDAR data.

More recently, Cao et al. [3] proposed an automated method for extracting power lines from ALS point clouds using connected-component segmentation and contextual geometric information. Their approach exploited the alignment of linear features with the span direction to distinguish conductor points from unrelated structures. To recover fragmented conductor segments, the authors introduced a graph-cut energyminimization algorithm and a slice-based segmentation procedure that separated power lines from nearby pylons, insulators, and jumper wires. The method was evaluated on nine datasets covering more than 80 km of transmission corridors with voltage classes ranging from 220 to 800 kV. It achieved an average precision of 99.77%, an average recall of 99.81%, and an average $\mathrm { F _ { 1 } }$ -score of 99.79%. Despite its high reported performance, the method of Cao et al. [3] relied on distancebased connectivity assumptions that are unreliable when gaps within a conductor are larger than the separation between neighboring conductors. This condition commonly occurs in distribution networks, where phases are closely spaced, and conductor observations may be fragmented by vegetation. Although these connectivity assumptions were suitable for transmission corridors with widely separated phases, they are less appropriate for dense distribution-line configurations.

Maset and Fusiello [12] proposed a complete airborne-LiDAR processing pipeline combining deep-learning-based semantic segmentation, DBSCAN-based pylon identification, and J-Linkage multi-model fitting for conductor vectorization. The method achieved an $\mathrm { F _ { 1 } }$ -score of 98.5% for vectorized cables on a 36-km transmission-line dataset. However, it was evaluated on airborne high-voltage infrastructure and required explicit catenary-based multi-model fitting.

The above studies demonstrated the value of combining contextual information with geometric conductor modeling. However, most methods are sensitive to severe occlusions, fragmented observations, or closely spaced conductors. When neighboring conductors are closely spaced, the extraction of individual conductors becomes challenging [13]. Generally, they were designed and evaluated for a single LiDAR acquisition modality. Consequently, their robustness and scalability across mobile and aerial LiDAR data remain insufficiently investigated. Table I summarizes the objectives, principal techniques, data sources, reported performance, and main limitations of the reviewed methods. The reported values should be interpreted with caution because the studies used different datasets, voltage classes, evaluation units, and metric definitions.

![](images/bb9e8d57c51ad2d42f72d97deb3c8fbd0fea726d1b463cb32099d5588f532729.jpg)  
Fig. 1: Example of a mobile LiDAR span classified using U-Next. Vegetation is shown in green, power lines in light blue, poles in dark blue, vehicles in orange, public-lighting assets in light green, and other objects in red.

## III. PROPOSED POWER-LINE DETECTION AND RECONSTRUCTION METHOD

The complete workflow consists of a modality-specific semantic classification stage followed by a unified four-stage reconstruction pipeline: 1) scaled clustering and geometric filtering, 2) robust parametric conductor modeling, 3) iterative fragment merging and conductor acceptance, and 4) rescue of rejected fragments. The semantic classification procedure differs between mobile and aerial LiDAR data, whereas all subsequent reconstruction stages use the same implementation and parameter configuration.

## A. Semantic Classification of Poles and Power Lines

LiDAR semantic classification remains challenging because of variations in point density and intensity, occlusions, measurement noise, and scanning artifacts. Learning-based methods can address some of these limitations by jointly exploiting geometric, contextual, and RGB information.

For mobile LiDAR data, we trained a U-Next semanticsegmentation model [14] on an internally annotated dataset. A small subset of point-cloud tiles was first manually annotated and used to train an initial model. The model was then applied to additional tiles, and its preliminary predictions were reviewed and corrected by a human annotator. The corrected annotations were incorporated into subsequent training iterations. This model-assisted annotation procedure was repeated until the classification performance stabilized. The final model achieved an average Intersection over Union (mIoU) of 87% over the critical power-line, vegetation, and pole classes. An example of the resulting semantic classification is shown in Fig. 1.

The trained model classified mobile LiDAR points into seven categories: vegetation, wires, poles, transformers, buildings, vehicles, and other objects. The annotated dataset contained approximately 154.5 million points collected in rural and semi-urban environments. As shown in Table II, the dataset was highly imbalanced. Indeed, transformers and poles represented approximately 0.05% and 1.23% of all annotated points, respectively. The data were divided geographically into mutually exclusive training, validation, and test regions using a 60%/15%/25% split. This geographic separation prevented points from the same span or acquisition segment from appearing in multiple subsets.

TABLE I: Comparison of LiDAR-based power-line detection and reconstruction methods.
<table><tr><td>Reference</td><td>Objective</td><td></td><td>Main technique</td><td>Context or model</td><td>Data</td><td></td><td>Reported per- formance</td><td>Main limitation</td></tr><tr><td>Jwa et al. [2]</td><td>3D line struction</td><td>power- recon-</td><td>Voxel-based piecewise detection and catenary fitting</td><td>Local voxel and iterative refinement</td><td>grouping ALS catenary</td><td>Accuracy: 93.8%</td><td></td><td>Sensitive to vegetation- induced occlusion and potentially computation- ally expensive</td></tr><tr><td>Liang and Zhang [9]</td><td>Power-line extraction and modeling</td><td>Geometric Hough-like detection,</td><td>filtering, and</td><td>Three-dimensional parabolic model</td><td>ALS conductor</td><td>Accuracy: 96.5%</td><td></td><td>Performance deteriorated when conductor obser- vations contained large</td></tr><tr><td>Guo et al. [5]</td><td>Robust conductor re- construction</td><td>Cuboid segmentation, iterative RANSAC</td><td>quadratic fitting, and</td><td>Span constrained by two ALS pylons</td><td></td><td>Accuracy: 94.9%</td><td>gaps consistent</td><td>Assumed a sufficiently conductor configuration within each span</td></tr><tr><td>Gaha et al. [4]</td><td>Pole power-line detection</td><td>and</td><td>RANSAC-based seg- ment extraction and RGB filtering</td><td>Plane defined by con- secutive poles and three- point modeling</td><td>MLS</td><td>Recall: 98%; ac- curacy: 84%</td><td></td><td>Limited to single-phase configurations and sen- sitive to candidate-point errors</td></tr><tr><td>Shi et al. [6]</td><td>Urban power-line extraction and hazard analysis</td><td>Geometric extraction clustering</td><td>feature Urban and context</td><td>distribution-network</td><td>roadside MLS</td><td>Accuracy: 91%</td><td>on aerial data</td><td>Designed for mobile Li- DAR and not evaluated</td></tr><tr><td>Massicotte et al. [11]</td><td>Utility-asset inventory</td><td>Combined image and LiDAR processing</td><td></td><td>Contextual detection of MLS utility assets</td><td></td><td>Accuracy: 95%</td><td></td><td>85- Limited implementation details restricted repro- ducibility and analysis</td></tr><tr><td>Cao et al. [3]</td><td>Automated extraction in transmission corridors</td><td>Connected components, cuts, slice-based segmentation, energy minimization</td><td>graph mation and</td><td>Span alignment and con- ALS textual geometric infor-</td><td></td><td>Precision: 99.77%; 99.81%; 99.79%</td><td>ductors</td><td>Distance-based connec- recall: tivity may fail for closely F1: spaced distribution con-</td></tr><tr><td>Maset and Fusiello [12]</td><td>Automatic power-line vectorization</td><td>Linkage multi-model fitting</td><td>DBSCAN and J- formation</td><td>FCN segmentation, contextual geometric in- ALS</td><td></td><td>F1: 98.5%</td><td></td><td>Performance on closely spaced distribution con- ductors was not assessed</td></tr><tr><td>Proposed method</td><td>Power-line detection and reconstruc- tion</td><td>Adaptive prescaling, polynomial and fragment merging</td><td>fitting, spans iterative</td><td>Unified reconstruction of HDBSCAN, robust mobile and aerial LiDAR</td><td>ALS MLS</td><td>ALS: precision 96.87%; recall annotations 96.66%; 96.76%</td><td>Independent F1: unavailable the mobile</td><td>conductor were for LiDAR</td></tr></table>

For aerial LiDAR data, we used either provider-supplied semantic labels or labels generated using commercial software such as Global Mapper [15]. Developing a new aerial semantic classification model was outside the scope of this work, which focused on conductor reconstruction after candidate wire points had been identified. Hence, the semantic preprocessing stage was modality-specific, i.e., mobile LiDAR data were classified using U-Next, whereas aerial LiDAR data used provider-supplied or commercially generated labels. Following this preprocessing stage, the same reconstruction implementation and parameter configuration were applied to both modalities.

TABLE II: Class distribution in the HQ-2025 mobile LiDAR dataset. Percentages indicate the proportion of points within each split.
<table><tr><td>Class</td><td>Train (%)</td><td>Val. (%)</td><td>Test (%)</td><td>Count (M)</td></tr><tr><td>Vegetation</td><td>89.79</td><td>72.54</td><td>89.98</td><td>134.6</td></tr><tr><td>Wire</td><td>1.31</td><td>3.24</td><td>1.54</td><td>2.6</td></tr><tr><td>Pole</td><td>0.98</td><td>2.25</td><td>1.16</td><td>1.9</td></tr><tr><td>Transformer</td><td>0.04</td><td>0.11</td><td>0.03</td><td>0.07</td></tr><tr><td>Building</td><td>5.71</td><td>17.29</td><td>6.26</td><td>11.9</td></tr><tr><td>Vehicle</td><td>1.38</td><td>1.69</td><td>0.83</td><td>2.0</td></tr><tr><td>Other</td><td>0.79</td><td>2.89</td><td>0.20</td><td>1.5</td></tr><tr><td>Total points</td><td colspan="4">91.0 M 24.4 M</td></tr></table>

## B. Scaled Clustering and Geometric Filtering

Semantic segmentation assigns all wire points within a span to the same class without distinguishing individual conductors, phases, voltage levels, or telecommunication cables. To separate these observations, we applied Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDB-SCAN) [16]. The latter was selected instead of the original DBSCAN algorithm [17] because it can identify clusters with varying densities while labeling isolated observations as noise. However, direct clustering in the original coordinate system remained difficult because a conductor could extend over several tens of meters while adjacent conductors could be separated by less than one meter. Consequently, we transformed the points into a scaled, span-aligned coordinate system before clustering. Let $\mathbf { p } _ { i } = \left[ x _ { i } \quad y _ { i } \quad z _ { i } \right] ^ { \top }$ denote the i-th candidate wire point, and $\mathbf { p } _ { i } ^ { x y } = \left[ x _ { i } y _ { i } \right] ^ { \mid }$ its horizontal coordinates. Principal Component Analysis (PCA) was applied to the horizontal point coordinates. The first principal direction, denoted by $\mathbf { u } \in \mathbb { R } ^ { 2 }$ , approximated the along-span direction. A perpendicular unit vector $\textbf { n } \in \mathbb { R } ^ { 2 }$ defined the cross-span direction as follows:

$$
\mathbf { u } ^ { \top } \mathbf { n } = 0 , \qquad \| \mathbf { u } \| _ { 2 } = \| \mathbf { n } \| _ { 2 } = 1 .\tag{1}
$$

Using the horizontal centroid $\bar { \mathbf { p } } ^ { x y }$ and mean elevation z¯, each point was projected into the local coordinate system as

$$
s _ { i } = \left( \mathbf { p } _ { i } ^ { x y } - \bar { \mathbf { p } } ^ { x y } \right) ^ { \top } \mathbf { u } ,\tag{2}
$$

$$
n _ { i } = \big ( \mathbf { p } _ { i } ^ { x y } - \bar { \mathbf { p } } ^ { x y } \big ) ^ { \top } \mathbf { n } ,\tag{3}
$$

$$
h _ { i } = z _ { i } - \bar { z } ,\tag{4}
$$

where $s _ { i } , \ n _ { i } ,$ and $h _ { i }$ represented the along-span, cross-span, and centered vertical coordinates, respectively. The transformed point used for clustering is

$$
\widetilde { \mathbf { p } } _ { i } = \left[ w _ { s } s _ { i } \quad w _ { n } n _ { i } \quad w _ { z } h _ { i } \right] ^ { \top } ,\tag{5}
$$

where $w _ { s } , \ w _ { n }$ , and $w _ { z }$ controlled the relative influence of the three coordinates. We used $w _ { n } ~ = ~ 2$ and $w _ { z } ~ = ~ 1$ to amplify lateral separations between neighboring conductors while preserving vertical separations. For example, setting $w _ { n } = 2$ caused a lateral separation of 1 m to contribute 2 m to the scaled distance. In contrast, $w _ { s }$ compressed distances along the span direction, thus facilitating the association of fragments belonging to the same conductor. Hence, the transformation reduced along-span gaps while increasing the separation between adjacent phases.

The along-span weight $w _ { s }$ was adapted to the span length and the HDBSCAN clust $\mathsf { e r \_ s e l }$ ection\_epsilon parameter. Let L denote the span length, ϵ<sub>HDBSCAN</sub> the HDB-SCAN cluster-selection epsilon, and $\rho _ { s } = 2 0 $ a fixed scaling coefficient. The target scaled length is defined as

$$
\ell _ { \mathrm { t a r g e t } } = \rho _ { s } \times \epsilon _ { \mathrm { H D B S C A N } } ,\tag{6}
$$

and the preliminary along-span weight is

$$
w _ { s } ^ { \star } = \frac { \ell _ { \mathrm { t a r g e t } } } { L } .\tag{7}
$$

The final weight was obtained by clipping this value within predefined bounds:

$$
w _ { s } = \mathrm { c l i p } \left( w _ { s } ^ { \star } , w _ { s , \operatorname* { m i n } } , w _ { s , \operatorname* { m a x } } \right) ,\tag{8}
$$

where ${ w _ { s , \mathrm { m i n } } = 0 . 0 5 }$ and $w _ { s , \mathrm { m a x } } = 1 . 0$ . Thus, the along-span coordinate was scaled by between 5% and 100% of its original value, with stronger compression applied to longer spans.

HDBSCAN was applied with ϵ<sub>HDBSCAN</sub> = 0.3, a minimum number of samples $k _ { \mathrm { m i n } } = 3 .$

After HDBSCAN clustering using the implementation of McInnes et al. [16], geometric and contextual criteria are used to remove likely classification errors. Clusters having a horizontal extent below 0.1 m are rejected. Also, PCA is applied to each cluster to assess its dominant orientation. Clusters whose principal direction is inconsistent with the span direction (absolute cosine below 0.85, i.e., a deviation larger than approximately 32<sup>◦</sup>) are considered unlikely to represent conductors and are removed. This filtering reduces vegetation points incorrectly assigned to the wire class. Fig. 2 shows the remaining clusters for the example in Fig. 1.

## C. Robust Parametric Conductor Modeling

After clustering, observations belonging to a physical conductor can remain divided among several spatially separated fragments, especially when vegetation occluded parts of the conductor or when the LiDAR data are sparse. A conductor model is therefore fitted to each cluster to identify geometrically compatible fragments. A catenary model can be fitted directly to the observed point coordinates without explicitly specifying the conductor’s physical properties such as conductor tension, temperature, and linear mass. We adopt a quadratic approximation for computational simplicity and stable fitting of highly fragmented observations. This choice is further supported by our experiments, in which quadratic fitting yielded slightly better reconstruction performance than catenary fitting. Accordingly, each conductor is represented by a span-aligned parametric model whose lateral and vertical components are modeled independently as linear and quadratic functions, respectively, of the local along-span coordinate s as follows:

$$
\widehat { n } ( s ) = \beta _ { 1 } s + \beta _ { 0 } ,
$$

$$
\widehat { h } ( s ) = \gamma _ { 2 } s ^ { 2 } + \gamma _ { 1 } s + \gamma _ { 0 } .\tag{9}
$$

(10)

The corresponding point in the original coordinate system is

$$
\widehat { \mathbf { p } } ( s ) = \left[ \bar { \mathbf { p } } ^ { x y } + s \mathbf { u } + \widehat { n } ( s ) \mathbf { n } \right] .\tag{11}
$$

Unlike methods that constructed a curve from three selected points [4], the proposed formulation uses all observations within a cluster. Note, that conductor fitting is performed using the unscaled, span-aligned local coordinates, denoted by $s _ { i } , n _ { i } , h _ { i }$ , and $z _ { i } ^ { \prime } = h _ { i }$ . The scaling in 8 is used exclusively for clustering. The quadratic coefficients are estimated by iteratively reweighted least squares using Huber weights [18], which reduces the influence of LiDAR noise and misclassified points.

For either the lateral or vertical model, let $q _ { i }$ denote the observed dependent variable and ${ \widehat { q } } ( s _ { i } )$ the corresponding model prediction. Specifically, $q _ { i } = n _ { i }$ for the lateral model and $q _ { i } = h _ { i }$ for the vertical model. The residual is defined as

![](images/3619fe2b4e748b95725af83a57e2262bb2162c151c3243d07f7d50d2af5c53ce.jpg)  
Fig. 2: Conductor fragments obtained using span-aligned prescaling, HDBSCAN clustering, and geometric filtering. Each color represents a different cluster.

$$
r _ { i } = q _ { i } - { \widehat { q } } ( s _ { i } ) .\tag{12}
$$

Each observation is assigned the Huber weight

$$
\omega _ { i } = \left\{ \begin{array} { l l } { 1 , } & { | r _ { i } | \leq c \widehat { \sigma } , } \\ { \displaystyle \frac { c \widehat { \sigma } } { | r _ { i } | } , } & { | r _ { i } | > c \widehat { \sigma } , } \end{array} \right.\tag{13}
$$

where $c = 1 . 3 4 5$ . Let $\widetilde { r } = \mathrm { m e d i a n } _ { j } ( r _ { j } )$ denote the median of the residuals over all observations j in the current cluster. The robust residual-scale estimate is then computed as

$$
\widehat { \sigma } = 1 . 4 8 2 6 \mathrm { \ m e d i a n } _ { i } \left( \left| \boldsymbol { r } _ { i } - \boldsymbol { \widetilde { r } } \right| \right) .\tag{14}
$$

The coefficients are updated by solving the corresponding weighted least-squares problem. Iterations continue until the weights converge or a maximum of $T ~ = ~ 1 0$ iterations is reached. For clusters containing at most 10 points, ordinary least squares is used because the Median Absolute Deviation (MAD)-based scale estimate could become unstable for very small samples.

## D. Iterative Fragment Merging and Conductor Acceptance

One of the primary contributions of this work is the way we iteratively merge the powerline segments to construct a more complete representation of each physical conductor. Our approach proceeds as follows: First, clusters are sorted by descending length, so that the longest and thus most geometrically reliable segments are processed first. Each unassigned cluster is then used in turn as a seed $\mathcal { M } _ { k } \ = \ \{ k \}$ , and the 3D parabola model of Eqs. (9) and (10) is fitted to $\mathcal { M } _ { k }$ using Huber-weighted regression. The compatibility of every cluster k<sup>′</sup> with this model is measured by a Huber root-mean-square error computed over all its points:

$$
\mathrm { R M S E } _ { k ^ { \prime } } = \sqrt { \frac { 1 } { | k ^ { \prime } | } \sum _ { { \bf p } _ { i } \in k ^ { \prime } } \rho _ { \delta _ { H } } ( r _ { i } ) } ,\tag{15}
$$

where the residual

$$
r _ { i } ^ { 2 } = \frac { { \left( n _ { i } - \widehat { n } ( s _ { i } ) \right) } ^ { 2 } } { 1 + \beta _ { 1 } ^ { 2 } } + { \left( h _ { i } - \widehat { h } ( s _ { i } ) \right) } ^ { 2 }\tag{16}
$$

combines the horizontal distance perpendicular to the fitted line and the vertical deviation, and

$$
\rho _ { \delta _ { H } } ( r ) = \left\{ { r ^ { 2 } } , \atop { 2 \delta _ { H } r - \delta _ { H } ^ { 2 } } , \quad r \leq \delta _ { H } , \right.\tag{17}
$$

is the Huber loss with $\delta _ { H } = 0 . 1$ m, so that isolated outlying points contribute linearly rather than quadratically. A seed whose own RMSE exceeds $\tau _ { \mathrm { R M S E } } = 0 . 3 0$ m is considered incoherent and is discarded. Otherwise, the unassigned cluster with the lowest RMSE is merged into $\mathcal { M } _ { k }$ if its RMSE does not exceed τ<sub>RMSE</sub>. The model is then re-fitted on the merged point set and compared against the remaining unmerged clusters; this process repeats until no additional merge occurs. Consequently, the best-aligned fragments are merged first, and each merge refines the model used to evaluate the remaining fragments. This best-first growth reduces the risk of prematurely merging adjacent conductors. After the iterative merging procedure, each candidate conductor is evaluated according to its along-span coverage:

$$
r _ { \mathrm { s p a n } } ( C _ { k } ) = \frac { \operatorname* { m a x } _ { \mathbf { p } _ { i } \in C _ { k } } s _ { i } - \operatorname* { m i n } _ { \mathbf { p } _ { i } \in C _ { k } } s _ { i } } { L } ,\tag{18}
$$

where L denotes the complete span length. Let $\tau _ { \mathrm { s p a n } }$ denote the minimum fraction of the complete span length that a candidate cluster must cover to be accepted as a reconstructed conductor. A candidate is accepted when

$$
r _ { \mathrm { s p a n } } ( C _ { k } ) > \tau _ { \mathrm { s p a n } } .\tag{19}
$$

We set $\tau _ { \mathrm { s p a n } } = 0 . 5 5$ . Candidates that do not satisfy this criterion are forwarded to the rescue stage.

## E. Rescue of Rejected Fragments

Rejected fragments are not discarded immediately since pairs of short fragments could jointly support a valid conductor model. Let $\mathcal { R } = \{ F _ { 1 } , \ldots , F _ { N _ { R } } \}$ denote the set of rejected fragments containing at least five points, having a horizontal extent of at least $\Delta s ,$ and whose Huber RMSE (Eq. (15)) with respect to their own model does not exceed τ . For each unordered pair $( F _ { i } , F _ { j } )$ , where $i < j$ , the two fragments must not overlap by more than 5% along the span. A robust spanaligned parametric quadratic model $\widehat { \mathcal { C } } _ { i j }$ is then fitted to the combined point set $F _ { i } \cup F _ { j }$ . The pair is scored using the worst per-fragment Huber RMSE with respect to $\widehat { \mathcal { C } } _ { i j }$ , denoted

$$
\eta _ { i j } = \mathrm { m a x } \left( \mathrm { R M S E } _ { F _ { i } } , \mathrm { R M S E } _ { F _ { j } } \right) .\tag{20}
$$

Pairs satisfying $\eta _ { i j } \leq$ τ<sub>RMSE</sub> are merged in ascending order of $\eta _ { i j } \colon$ the best pair is merged, the pairs involving either fragment are discarded, and the merged fragment is re-scored against the remaining fragments. Each resulting candidate, merged or not, is accepted when its span coverage satisfies $r _ { \mathrm { s p a n } } > \tau _ { \mathrm { r e s c u e } } ,$ , with $\tau _ { \mathrm { r e s c u e } } = 0 . 4 0$ . The rescue stage required $O ( N _ { R } ^ { 2 } )$ pairwise model evaluations. Algorithm 1 summarizes the complete reconstruction pipeline.

## IV. EXPERIMENTAL EVALUATION

We evaluate the proposed method from four complementary perspectives: 1) controlled robustness to point thinning and occlusion, 2) large-scale performance on aerial and mobile LiDAR datasets, 3) comparison with existing reconstruction pipelines, and 4) sensitivity to the main algorithmic parameters.

## A. Controlled Robustness Assessment

The controlled assessment evaluates the influence of data degradation on individual power-line spans. This assessment is challenging because spans differ in geometry, length, point density, and occlusion patterns, depending on the acquisition geometry and surrounding environment. To examine these effects under controlled conditions, we select three representative span architectures from the mobile LiDAR dataset. The mobile point clouds are acquired using a mapping unit equipped with two Velodyne VLP-32C LiDAR sensors, with a combined acquisition rate of two million points per second, an RTK-GPS system with a reported accuracy of ±2 cm, and an inertial measurement unit. The point clouds have an average density of approximately 1,200 points/m<sup>2</sup> and a reported positional accuracy of ±3 cm. The data are preprocessed and divided into individual pole-to-pole spans containing the conductors located between two consecutive poles. Further details regarding span extraction are provided in [11], [19], [20]. The selected architectures comprise: 1) a three-phase span, 2) a single-phase span, and 3) a horizontal-to-vertical span in which three horizontally arranged phases transition into a vertical three-phase configuration. The third architecture represents a complex case because the relative horizontal and vertical positions of the conductors vary along the span.

We evaluate each architecture under three degradation scenarios: uniform point thinning, multiple distributed gaps, and a single contiguous gap. The complete observations of each span define the 100% retention condition. For each architecture, we generate 20 evaluation files, corresponding to retainedpoint ratios ranging from 100% to 5% in decrements of 5 percentage points. Each file represents a single retention level; no repeated realizations are generated at the same level. For each evaluation file, reconstructed conductors are evaluated against individual reference conductors within the same span. Let $P = \{ \mathbf { p } _ { k } \} _ { k = 1 } ^ { N _ { P } }$ denote the points sampled along a predicted conductor curve, and let G denote the sampled reference geometry of a single reference conductor. The threedimensional Euclidean distance from each predicted sample to its nearest reference point is

$$
d _ { k } = \operatorname* { m i n } _ { \mathbf { g } \in G } \| \mathbf { p } _ { k } - \mathbf { g } \| _ { 2 } .\tag{21}
$$

The geometric match ratio is defined as

$$
\rho ( P , G ) = \frac { 1 } { N _ { P } } \sum _ { k = 1 } ^ { N _ { P } } \mathbb { I } ( d _ { k } \le d _ { \mathrm { m a t c h } } ) , \qquad d _ { \mathrm { m a t c h } } = 0 . 2 0 \mathrm { m } ,\tag{22}
$$

where I(·) equals one when its argument is true and zero otherwise. A prediction–reference pair is eligible for matching when

$$
\rho ( P , G ) \geq \tau _ { \mathrm { m a t c h } } , \qquad \tau _ { \mathrm { m a t c h } } = 0 . 7 0 .\tag{23}
$$

Thus, at least 70% of the predicted samples must lie within 0.20 m of the corresponding reference conductor. One-to-one matching is enforced within each realization: each prediction can be assigned to at most one reference conductor, and each reference conductor can be assigned to at most one prediction. Assigned prediction–reference pairs are counted as true positives (TP). Unmatched predictions are counted as false positives (FP), and unmatched reference conductors are counted as false negatives (FN).

1) Robustness to Uniform Point Thinning: Uniform point thinning evaluates the effect of progressively reducing the number of available observations without explicitly introducing contiguous occlusion intervals. Points are randomly retained according to the specified retention ratio. The original conductors contain approximately 25–30 points/m. Fig. 3 reports reconstruction F<sub>1</sub> score as a function of the retained-point ratio. All three span architectures achieve 100% performance at retention ratios of 5% or greater. At a 5% retention ratio, the effective linear density is approximately 1.25–1.5 points/m. When the retention ratio falls below 5%, the remaining observations no longer provide sufficient geometric support for reliable conductor reconstruction.

2) Robustness to Multiple Distributed Gaps: The second scenario introduces multiple gaps at randomly selected locations along each conductor to reproduce fragmented observations caused by vegetation occlusion. Unlike uniform point thinning, which distributes the retained observations across the complete span, this scenario removes contiguous groups of neighboring points while preserving the specified overall retention ratio. Fig. 3 shows that the proposed method achieves $\mathrm { F _ { 1 } }$ score of 100% for all three architectures at retention ratios of 5% or greater. Below this threshold, the remaining fragments become too short and too widely separated to support reliable model fitting and fragment association. Fig. 4 illustrates a successful reconstruction of a three-phase span using only 5% of the original conductor observations.

Algorithm 1: Power-Line Reconstruction from Mobile and Aerial LiDAR Data   
Input: Semantically classified span files containing candidate wire points   
Output: Set of reconstructed conductor models   
1 Set $\epsilon _ { { \mathrm { H D B S C A N } } } = 0 . 3 ~ { \mathrm { m } } , ~ \rho _ { s } = 2 0 , ~ w _ { n } = 2 . 0 , ~ w _ { z } = 1 . 0 , ~ w _ { s , \operatorname* { m i n } } = 0 . 0 5 , ~ w _ { s , \operatorname* { m a x } } = 1 . 0 , ~ k _ { \operatorname* { m i n } } = 3 , ~ m _ { \operatorname* { m i n } } = 3 ,$   
$m _ { \mathrm { m a x } } = 1 5 , \delta _ { H } = 0 . 1$ m, τ<sub>RMSE</sub> = 0.30 m, $\Delta s = 0 . 1$ m, $\tau _ { \mathrm { s p a n } } = 0 . 5 5 ,$ , and $\tau _ { \mathrm { r e s c u e } } = 0 . 4 0 ;$   
2 foreach classified span file do   
3 Extract candidate wire points and compute span length $L ;$   
$/ /$ Stage 1: Scaled clustering and geometric filtering   
4 Compute the span-aligned frame (u, n) using PCA on the horizontal coordinates;   
5 Project each point into local coordinates $( s _ { i } , n _ { i } , h _ { i } ) ;$   
6 Compute $w _ { s } \gets \mathrm { c l i p } \left( \rho _ { s } \epsilon _ { \mathrm { H D B S C A N } } / L , w _ { s , \mathrm { m i n } } , w _ { s , \mathrm { m a x } } \right) ;$   
7 Transform each point as $\widetilde { \mathbf { p } } _ { i } \gets [ w _ { s } s _ { i } , w _ { n } n _ { i } , w _ { z } h _ { i } ] ^ { \top } ;$   
8 Apply HDBSCAN with ϵ<sub>HDBSC</sub> $_ \mathrm { 4 N } = 0 . 3$ to $\{ \widetilde { \bf { p } } _ { i } \} ;$   
9 Remove clusters with insufficient along-span extent or orientation inconsistent with the span;   
$/ /$ Stage 2: Robust parametric conductor modeling   
10 Fit the lateral and vertical models in Eqs. (9) and (10) using Huber-weighted regression;   
$/ /$ Stage 3: Iterative fragment merging and acceptance   
11 foreach unassigned cluster k (longest first) do   
12 $\mathcal { M } _ { k } \gets \{ k \}$   
13 repeat   
14 Fit parabola via Huber on $\mathcal { M } _ { k } \colon \hat { n } ( s ) = \beta _ { 1 } s + \beta _ { 0 } \ \& \ \hat { h } ( s ) = \gamma _ { 2 } s ^ { 2 } + \gamma _ { 1 } s + \gamma _ { 0 } ;$   
15 foreach cluster k<sup>′</sup> (seed k and every unassigned cluster) do   
16 Compute the Huber RMSE RMSE ′ = $\begin{array} { r } { = \sqrt { \frac { 1 } { \left| \boldsymbol { k } ^ { \prime } \right| } \sum _ { \boldsymbol { p } \in \boldsymbol { k } ^ { \prime } } \rho _ { \delta _ { \boldsymbol { H } } } ( \boldsymbol { r } _ { \boldsymbol { p } } ) } } \end{array}$ , with $\begin{array} { r } { r _ { p } ^ { 2 } = \frac { ( n _ { p } - \hat { n } ( s _ { p } ) ) ^ { 2 } } { 1 + \beta _ { 1 } ^ { 2 } } + ( h _ { p } - \hat { h } ( s _ { p } ) ) ^ { 2 } } \end{array}$   
and $\rho _ { \delta _ { H } } ( r ) = r ^ { 2 } \ \mathrm { i f } \ r \leq \delta _ { H }$ , else $2 \delta _ { H } r \dot { - } \delta _ { H } ^ { 2 } ;$   
17 if $\mathcal { M } _ { k } = \{ k \}$ and RMSE<sub>k</sub> > τ<sub>RMSE</sub> then   
18 Discard seed k (incoherent) and continue with the next seed;   
19 k<sup>∗</sup> ← arg min<sub>k</sub>′ <sub>unassigned</sub> RMSE<sub>k</sub>′ ;   
20 if $\mathrm { R M S E } _ { k ^ { * } } \le$ τ<sub>RMSE</sub> then   
21 $\mathcal { M } _ { k }  \mathcal { M } _ { k } \cup \{ k ^ { * } \}$ and relabel $k ^ { * }$ to $k ;$   
22 until no cluster is added to $\mathcal { M } _ { k } ;$   
23 Sample the conductor model of $\mathcal { M } _ { k }$ along the span at spacing $\leq \Delta s$ to obtain the merged candidate $C _ { k } ;$   
24 Initialize the rejected set $\mathcal { R }  \emptyset ;$   
25 foreach merged candidate $C _ { k }$ do   
26 Compute $r _ { \mathrm { s p a n } } ( C _ { k } ) ;$   
27 if $r _ { \mathrm { s p a n } } ( C _ { k } ) > \tau _ { \mathrm { s p a n } }$ then   
28 Accept and save the reconstructed conductor model;   
29 else   
30 Add $C _ { k }$ to $\mathcal { R } ;$   
$/ /$ Stage 4: Rescue of rejected fragments   
31 Retain fragments in R containing at least five points, with an along-span extent of at least $\Delta s$ and a Huber RMSE   
to their own model of at most τ<sub>RMSE</sub>;   
32 foreach unordered pair $( F _ { i } , F _ { j } ) , i < j$ do   
33 if $F _ { i }$ and $F _ { j }$ overlap by at most 5% along the span then   
34 Fit $\widehat { \mathcal { C } } _ { i j }$ to $F _ { i } \cup F _ { j } { \mathrm { : } }$   
35 Compute the rescue score $\eta _ { i j } = \operatorname* { m a x } \left( \mathrm { R M S E } _ { F _ { i } } , \mathrm { R M S E } _ { F _ { j } } \right)$ with respect to $\widehat { \mathcal { C } } _ { i j }$   
36 if $\eta _ { i j } \leq$ τ<sub>RMSE</sub> then   
37 Store $( F _ { i } , F _ { j } , \eta _ { i j } )$ as a valid rescue pair;   
38 while valid rescue pairs remain do   
39 Merge the pair with the lowest $\eta _ { i j }$ into $F _ { i } \cup F _ { j }$ with model ${ \widehat { \mathcal { C } } } _ { i j } ;$   
40 Discard the pairs involving $F _ { i }$ or $F _ { j }$ and re-score the merged fragment against the remaining fragments;   
41 foreach rescued candidate $C _ { k } ,$ merged or not doPreprint submitted to IEEE J-STARS.   
42 if $r _ { \mathrm { s p a n } } ( C _ { k } ) > \tau _ { \mathrm { r e s c u e } }$ then   
43 Accept and save the reconstructed conductor model;

![](images/9db3e029c25ae444553081d015bb92d569850ddfb370463dce65017ad7be0e09.jpg)  
Fig. 3: Reconstruction $_ { \mathrm { F _ { 1 } - s c o r e } }$ for three conductor architectures under controlled degradation: uniform point thinning, multiple distributed gaps, and a single contiguous gap.

![](images/d264510412b18264d8fbd7eb5bb04789cf0b98d6097803a69e7e6e53a020d4ab.jpg)  
Fig. 4: Reconstruction of a three-phase span using 5% of the original conductor observations (red: retained LiDAR observations, white: reconstructed conductor).

3) Robustness to a Single Contiguous Gap: The third scenario evaluates reconstruction $\mathrm { F _ { 1 } }$ score in the presence of a single contiguous gap centered within a 58-m power-line span. This experiment examines whether the proposed method can associate conductor fragments separated by a long occluded interval. The gap is progressively enlarged toward both ends of the span, thus reducing the retained-point ratio and the length of the observed conductor fragments. As shown in Fig. 3, all three architectures achieve $\mathrm { F _ { 1 } }$ score of 100% at retention ratios of 35% or greater. At a retention ratio of 30%, our method no longer reconstructs the three-phase architecture successfully, although the other tested architectures remain detectable. At retention ratios below 10%, the remaining fragments are too short and separated to support reliable reconstruction for any of the three architectures. Fig. 5 illustrates a case in which 75% of the conductor span is occluded.

![](images/47c171806ff265a9fd4155a85dec633a5047b35b570e5a40be5161e0e10ac483.jpg)  
Fig. 5: Example of a single contiguous occlusion covering 75% of the conductor span (red: retained LiDAR observations, white: reconstructed conductor).

## B. Large-Scale Evaluation

The large-scale assessment evaluates the proposed method on heterogeneous aerial and mobile LiDAR datasets. Since independent conductor-level reference annotations are available only for the aerial dataset, the two modalities use different evaluation protocols. The aerial evaluation measures objectlevel detection and geometric reconstruction performance, whereas the mobile evaluation measures point-to-model geometric consistency and reconstruction coverage.

1) Aerial LiDAR Evaluation: The aerial dataset contains 10,870 pole-to-pole spans covering 382 km of distribution networks. It includes single-phase and three-phase systems, mixed horizontal and vertical configurations, and both mediumvoltage and low-voltage infrastructure. Individual span lengths range from 4 to 160 m. The point clouds are acquired using a RIEGL VQ-1460 laser scanner operating at a pulse repetition rate of up to 4.4 MHz. The point density ranges from 1 to $3 0 \mathrm { \ p o i n t s / m ^ { 2 } }$ , and the reported vertical RMSE over flat surfaces is 0.05 m. The LiDAR service provider supplies one 3D reference geometry for each physical conductor within each pole-to-pole span. The dataset contains 23,372 reference conductors. The reference geometries and predicted conductor models are expressed in the same coordinate reference system. Since the complete details of the provider’s proprietary annotation procedure are unavailable, we refer to these geometries as provider-supplied reference annotations rather than error-free ground truth.

Prediction-to-reference matching is performed independently within each span as defined in Subsection IV-A. For each successfully matched pair, the distance from each reference point to the predicted curve is computed. All resulting point-to-curve distances from successfully matched pairs are pooled before computing the reported geometric RMSE. Table III summarizes the large-scale aerial results. The proposed method reconstructs 22.592 conductors that matche to provider-supplied reference conductors. Our method achieves an $\mathrm { F _ { 1 } { \mathrm { - } } s c o r e }$ of 96.76%, and a geometric RMSE of 0.129 m.

Table IV summarizes the distribution of the per-span $\mathrm { F _ { 1 } - }$ score, processing time, and point-to-reference distance, defined as the distance between provider-supplied reference conductor points and the corresponding predicted parabola. Although the maximum point-to-reference distance reaches 3 m, a closer examination of the results revealed that this value corresponds to a single outlier located near the end of a span and is therefore not representative of the overall reconstruction accuracy. Indeed, 95% of the points exhibit a point-to-reference distance below 0.112 m, indicating a high level of geometric agreement between the predicted and reference conductors.

2) Mobile LiDAR Evaluation: The mobile dataset contains 1,620 power-line spans covering 68 km of distribution networks. It includes single-phase and three-phase systems, mixed horizontal and vertical layouts, and medium-voltage and low-voltage infrastructure. Individual span lengths range from 4 to 120 m. As a reminder, the 180 development spans are excluded from the 1,620 test spans. Independent conductor-level reference annotations are unavailable for this dataset. Consequently, the mobile evaluation does not measure object-level precision, recall, or $\mathrm { F _ { 1 } - s c o r e . }$ . Instead, it evaluates the geometric consistency between the candidate wire points produced by the semantic-classification stage and the reconstructed conductor models using reconstruction coverage and point-to-model RMSE. For each candidate wire point $\mathbf { p } _ { i }$ let $\begin{array} { r } { e _ { i } ~ = ~ \operatorname* { m i n } _ { \widehat { \mathbf { p } } \in \widehat { \mathcal { C } } } \| \mathbf { p } _ { i } - \widehat { \mathbf { p } } \| _ { 2 } } \end{array}$ denote its Euclidean distance between $\mathrm { L i D A R }$ conductor points and all reconstructed conductor curves $\widehat { \mathcal { C } }$ in the corresponding span. A candidate wire point is considered geometrically covered when

$$
e _ { i } \le d _ { \mathrm { c o v } } , \qquad d _ { \mathrm { c o v } } = 0 . 1 5 ~ \mathrm { m } .\tag{24}
$$

Let $N _ { \mathrm { w i r e } }$ denote the total number of candidate wire points and $N _ { \mathrm { c o v e r e d } }$ the number satisfying eq. (24). Reconstruction coverage is defined as $\begin{array} { r } { \mathrm { C o v e r a g e } = \frac { N _ { \mathrm { c o v e r e d } } } { N _ { \mathrm { w i r e } } } { \times } 1 0 0 \% } \end{array}$ . The pointto-model RMSE is computed over the covered candidate wire points as

$$
\mathrm { R M S E } _ { \mathrm { m o b i l e } } = \sqrt { \frac { 1 } { N _ { \mathrm { c o v e r e d } } } \sum _ { i : e _ { i } \le d _ { \mathrm { c o v } } } e _ { i } ^ { 2 } } .\tag{25}
$$

The proposed method achieves a reconstruction coverage of 95.27% and a point-to-model RMSE of 0.045 m. Thus, most points classified as wires are geometrically supported by a reconstructed conductor model within a maximum distance of 0.15 m.

## C. Comparative Benchmarking

We compare the proposed method with two existing reconstruction pipelines using the controlled dataset described in Subsection IV-A. The comparison includes the same three conductor architectures and three degradation scenarios as defined for Fig. 3.

The first baseline is a publicly available MATLAB implementation derived from the method of Shi et al. [6]. It detects and models power lines using geometric feature extraction, curve fitting, and plane-rotation operations. We refer to this implementation as EPL. The second quantitative baseline is our reimplementation of the method of Cao et al. [3], hereafter referred to as Cao-FEC. Because the original implementation is unavailable, the results reflect our implementation based on the methodological details reported in [3]. The original method was developed for transmission corridors and reported an $\mathrm { F _ { 1 } } \mathrm { - }$ score of 99.79% on its evaluation datasets. It relies on Fast Euclidean Clustering (FEC) with a distance threshold $D _ { \mathrm { F E C } }$ to separate individual conductors. When directly applied to the distribution-network benchmark, the Cao-FEC implementation achieves an $\mathrm { F _ { 1 } }$ -score of 47.7%. This performance difference is explained by the conflicting distance conditions within the evaluated distribution spans. To avoid merging neighboring conductors, the clustering threshold must satisfy

$$
D _ { \mathrm { F E C } } < d _ { \mathrm { i n t e r } } ^ { \mathrm { m i n } } \approx 1 . 0 ~ \mathrm { m } ,\tag{26}
$$

where $d _ { \mathrm { i n t e r } } ^ { \mathrm { m i n } }$ denotes the minimum observed separation between neighboring conductors. However, associating fragments belonging to the same conductor requires $D _ { \mathrm { F E C } } \ >$ $d _ { \mathrm { g a p } } ^ { \mathrm { m a x } } \approx 2 . 4 ~ \mathrm { m } .$ where $d _ { \mathrm { g a p } } ^ { \mathrm { m a x } }$ denotes the maximum observed gap between fragments of the same conductor. Since these conditions cannot be satisfied simultaneously in the original coordinate system, direct Euclidean clustering either fragments individual conductors or merges neighboring phases.

We also evaluate a modified Cao-FEC configuration using the span-aligned prescaling described in Subsection III-B. With $w _ { s } = 0 . 5 , w _ { n } = 2 . 0 , w _ { z } = 1 . 0 $ , and an RMSE threshold of $T _ { \mathrm { R M S E } } ~ = ~ 0 . 5 ~ \mathrm { m } .$ , the modified implementation achieves an $\mathrm { F _ { 1 } }$ -score of 89.7% on the controlled benchmark without changing the core FEC procedure. This result shows that spanaligned prescaling partially mitigates the conflicting distance constraints, although its performance remains below that of the proposed method.

Figs. 6–8 compare the proposed method, EPL, and the modified Cao-FEC (Cao-FEC mod.) implementation under the three degradation scenarios. The results are aggregated by scenario over the three conductor architectures and the corresponding degraded realizations.

Under the single contiguous gap scenario, the proposed method maintains an $_ \mathrm { F _ { 1 } - s c o r e }$ of 100% for retained-point ratios between 35% and 100%, as shown in Fig. 6. The performance of Cao-FEC (mod.) decreases sharply when the retained-point ratio reaches 60%, indicating sensitivity to increasing separation between fragments. EPL recovers some conductors under severe degradation and achieves $\mathrm { F _ { 1 ^ { - } } }$ scores between 90% and 95% for certain configurations at retention ratios between 25% and 35%. However, persistent FP predictions reduce its overall performance across the complete benchmark. Thus, EPL can outperform the proposed method in a small number of severely degraded configurations, but the proposed method provides more consistent performance across architectures and retention ratios. Figs. 7 and 8 present the results for uniform point thinning and multiple distributed gaps, respectively. The proposed method achieves the highest overall $\mathrm { F _ { 1 } }$ -scores across both scenarios. This advantage results primarily from its progressive fragment-merging procedure, which associates short but geometrically compatible fragments while limiting premature merging between adjacent phases.

TABLE III: Large-scale performance on the aerial LiDAR dataset.
<table><tr><td>Reference</td><td>Predicted</td><td>TP</td><td>FP</td><td>FN</td><td>Precision (%)</td><td>Recall (%)</td><td> $\overline { { \mathbf { F } _ { 1 } \mathbf { \Gamma } ( \% ) } }$ </td><td>RMSE (m)</td></tr><tr><td>23,372</td><td>23,323</td><td>22,592</td><td>731</td><td>780</td><td>96.87</td><td>96.66</td><td>96.76</td><td>0.129</td></tr></table>

TABLE IV: Distribution of the per-span F -score, processing time and geometric error on the aerial LiDAR dataset.
<table><tr><td>Metric</td><td>Mean</td><td>Std</td><td>Median</td><td>P95</td><td>Max</td></tr><tr><td> $\overline { { \mathrm { F } _ { 1 } } }$  per span (%)</td><td>96.31</td><td>15.82</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>Processing time per span (s)</td><td>0.153</td><td>0.342</td><td>0.100</td><td>0.350</td><td>0.47</td></tr><tr><td>Point-to-reference distance (m)</td><td>0.056</td><td>0.091</td><td>0.040</td><td>0.112</td><td>3.00</td></tr><tr><td>RMSE per conductor (m)</td><td>0.095</td><td>0.072</td><td>0.073</td><td>0.234</td><td>1.04</td></tr></table>

![](images/55eb1dd8dc969c0fd9fca383dcbd0a3621bbc75e1b182a54215a0adf9c5d56e2.jpg)  
Fig. 6: Comparison of the proposed method, EPL, and Cao-FEC (mod.) under a single contiguous gap for mobile LiDAR.

![](images/daf5957f7dd6dcbb6fbd80e769c66056c816c0be6fc0bbeeba76f9cc46bff054.jpg)  
Fig. 7: Comparison of the proposed method, $\mathrm { E P L , }$ and Cao-FEC (mod.) under uniform point thinning for mobile LiDAR.

## D. Sensitivity Analysis

A subset of 180 mobile development spans containing 420 conductors are used exclusively for parameter selection and sensitivity analysis. These spans are excluded from the

![](images/faafad0e153954a23d217e6e55f221986255c3ed002844e314f33396e94853c1.jpg)  
Fig. 8: Comparison of the proposed method, EPL, and Cao-FEC (mod.) under multiple distributed gaps for mobile Li-DAR.

1,620 mobile test set spans and are also excluded from the large-scale aerial evaluation set containing 10,870 spans. The resulting parameter configuration is subsequently applied to the large-scale dataset without further modification. We vary one parameter at a time while holding all remaining parameters at their selected values. Derived parameters are recomputed according to their defining equations. Performance is measured against the provider-supplied reference annotations using the matching protocol defined in Subsection IV-B1.

1) Sensitivity to HDBSCAN Cluster-Selection Epsilon: Table VI reports performance as ϵ<sub>HDBSCAN</sub> varies from 0.1 to 2.0. Across this range, the $_ \mathrm { F _ { 1 } - s c o r e }$ varies from 96.5% to 97.5%, indicating low sensitivity to this parameter, and no monotonic trend is observed. In fact, TP varies only between 399 and 403 and FP between 4 and 9. The selected value, $\epsilon _ { \mathrm { H D B S C A N } } = 0 . 3 .$ , lies within 0.4 points of the highest $\mathrm { F _ { 1 } - s c o r e }$ , obtained at $\epsilon _ { \mathrm { H D B S C A N } } ~ = ~ 0 . 9$ . This gain does not transfer to the airborne dataset, where $\epsilon _ { \mathrm { H D B S C A N } } =$ 0.9 reduces the $\mathrm { F } _ { 1 } \cdot$ -score from 96.76% to 96.56% and increases the reconstruction of RMSE from 0.12 to 0.17 m; $\epsilon _ { \mathrm { H D B S C A N } } = 0 . 3$ is therefore retained for both datasets. This robustness is partly explained by the subsequent fragmentmerging stage, which reassembles conductors that HDBSCAN splits into short fragments, and by the span-coverage criterion, which rejects most incoherent clusters. Since $w _ { s }$ depends on ϵ<sub>HDBSCAN</sub>, this experiment evaluates their coupled influence on reconstruction performance. In particular, the scaled alongwire extent $w _ { s } L \ = \ \rho _ { s } \epsilon _ { \mathrm { H D B S C A N } }$ grows with ϵ<sub>HDBSCAN</sub>. Further experiments with independently controlled scaling and clustering parameters are required to distinguish their individual effects.

TABLE V: Selected parameters of the proposed reconstruction pipeline.
<table><tr><td>Parameter</td><td>Description</td><td>Selected value</td><td>Type</td></tr><tr><td>€HDBSCAN</td><td>HDBSCAN cluster-selection epsilon in the scaled coordinate space</td><td>0.3</td><td>Fixed</td></tr><tr><td> $k _ { \mathrm { m i n } }$ </td><td>HDBSCAN minimum number of samples</td><td>3</td><td>Fixed</td></tr><tr><td> $m _ { c s }$ </td><td>HDBSCAN minimum cluster size depending on points density</td><td>[3,5]</td><td>Adaptive</td></tr><tr><td> $w _ { n }$ </td><td>Scale weight applied to the cross-span coordinate</td><td>2</td><td>Fixed</td></tr><tr><td> $w _ { z }$ </td><td>Scale weight applied to the vertical coordinate</td><td>1</td><td>Fixed</td></tr><tr><td> $w _ { s , \mathrm { m i n } }$ </td><td>Lower bound on the adaptive along-span scale weight</td><td>0.05</td><td>Fixed</td></tr><tr><td> $w _ { s , \mathrm { m a x } }$ </td><td>Upper bound on the adaptive along-span scale weight</td><td>1</td><td>Fixed</td></tr><tr><td> $\rho _ { s }$ </td><td>Scaling coefficient used to derive the adaptive along-span weight</td><td>20</td><td>Fixed</td></tr><tr><td> $w _ { s }$ </td><td>Adaptive along-span scale weight, clipped to  $[ w _ { s , \operatorname* { m i n } } , w _ { s , \operatorname* { m a x } } ]$ </td><td>[0.05, 1]</td><td>Adaptive</td></tr><tr><td> $d _ { \mathrm { s e g , m i n } }$ </td><td>Minimum accepted along-span extent of an initial cluster</td><td>0.1 m</td><td>Fixed</td></tr><tr><td> $N _ { \mathrm { w i r e , m i n } }$ </td><td>Minimum number of candidate wire points required to process a span</td><td>20</td><td>Fixed</td></tr><tr><td> $\delta _ { H }$ </td><td>Huber threshold of the RMSE used for merging and rescue</td><td>0.1 m</td><td>Fixed</td></tr><tr><td> $\tau _ { \mathrm { R M S E } }$ </td><td>Maximum Huber RMSE of a fragment with respect to a fitted 0.3 m conductor model for merging and rescue</td><td></td><td>Fixed</td></tr><tr><td> $\Delta s$ </td><td>Maximum sampling spacing of the conductor model and mini- mum extent of a rescue fragment</td><td>0.1 m</td><td>Fixed</td></tr><tr><td> $\tau _ { \mathrm { s p a n } }$ </td><td>Minimum fraction of the complete span length required for 0.55 acceptance</td><td></td><td>Fixed</td></tr><tr><td> $\tau _ { \mathrm { r e s c u e } }$ </td><td>Minimum fraction of the complete span length required for 0.40 accepting a rescued candidate</td><td></td><td>Fixed</td></tr></table>

TABLE VI: Sensitivity to HDBSCAN cluster-selection epsilon for mobile LiDAR.
<table><tr><td>€HDBSCAN</td><td>TP</td><td>FP</td><td>FN</td><td>Prec.</td><td>Rec.</td><td> $\overline { { \mathbf { F } _ { 1 } } }$ </td></tr><tr><td>0.1</td><td>402</td><td>4</td><td>18</td><td>99.0</td><td>95.7</td><td>97.3</td></tr><tr><td>0.2</td><td>400</td><td>7</td><td>20</td><td>98.3</td><td>95.2</td><td>96.7</td></tr><tr><td>0.3</td><td>402</td><td>6</td><td>18</td><td>98.5</td><td>95.7</td><td>97.1</td></tr><tr><td>0.4</td><td>402</td><td>6</td><td>18</td><td>98.5</td><td>95.7</td><td>97.1</td></tr><tr><td>0.5</td><td>400</td><td>9</td><td>20</td><td>97.8</td><td>95.2</td><td>96.5</td></tr><tr><td>0.7</td><td>399</td><td>7</td><td>21</td><td>98.3</td><td>95.0</td><td>96.6</td></tr><tr><td>0.9</td><td>403</td><td>4</td><td>17</td><td>99.0</td><td>96.0</td><td>97.5</td></tr><tr><td>1.0</td><td>401</td><td>5</td><td>19</td><td>98.8</td><td>95.5</td><td>97.1</td></tr><tr><td>1.5</td><td>402</td><td>6</td><td>18</td><td>98.5</td><td>95.7</td><td>97.1</td></tr><tr><td>2.0</td><td>401</td><td>7</td><td>19</td><td>98.3</td><td>95.5</td><td>96.9</td></tr></table>

2) Sensitivity to the Huber RMSE Merging Threshold: During iterative merging, a fragment is appended to a seed curve only if the Huber RMSE $( \delta _ { H } = 0 . 1$ m) of the joint fit satisfies $\mathrm { R M S E } _ { H } \le \tau _ { \mathrm { R M S E } }$ . This same threshold is applied to discard incoherent seeds and validate rescue candidates. Table VII reports the resulting performance as τ<sub>RMSE</sub> varies from 0.05 to 1.0 m.

The reconstruction performance remains stable within the interval $0 . 1 0 ~ \leq ~ \tau _ { \mathrm { R M S E } } ~ \leq ~ 0 . 3 5 ~ \mathrm { m }$ , yielding an $F _ { 1 }$ -score between 96.6% and 97.3%. The optimal result is achieved at τ<sub>RMSE</sub> = 0.20 m, and the adopted baseline of τ<sub>RMSE</sub> = 0.30 m lies well within this stable plateau. Strictly lower thresholds over-reject valid fragment associations, causing conductors to remain fragmented into multiple distinct curves and consequently increasing false positives (FPs). Conversely, excessively large thresholds erroneously permit the merging of fragments from physically distinct conductors. As a result, both precision and recall drop sharply for thresholds exceeding 0.4 m.

TABLE VII: Sensitivity to the Huber RMSE merging threshold for mobile LiDAR.
<table><tr><td>TRMSE</td><td>TP</td><td>FP</td><td>FN Prec.</td><td>Rec.  $\overline { { \mathbf { F } _ { 1 } } }$ </td></tr><tr><td>0.05</td><td>401</td><td>24 19</td><td>94.4</td><td>95.5 94.9</td></tr><tr><td>0.10</td><td>401</td><td>9 19</td><td>97.8</td><td>95.5 96.6</td></tr><tr><td>0.15</td><td>402</td><td>8 18</td><td>98.0</td><td>95.7 96.9</td></tr><tr><td>0.20</td><td>403</td><td>5</td><td>17 98.8</td><td>96.0 97.3</td></tr><tr><td>0.25</td><td>401</td><td>6 19</td><td>98.5</td><td>95.5 97.0</td></tr><tr><td>0.30</td><td>402</td><td>6 18</td><td>98.5</td><td>95.7 97.1</td></tr><tr><td>0.35</td><td>401</td><td>8 19</td><td>98.0</td><td>95.5 96.7</td></tr><tr><td>0.40</td><td>387</td><td>15 33</td><td>96.3</td><td>92.1 94.2</td></tr><tr><td>0.50</td><td>134</td><td>119 286</td><td>53.0</td><td>31.9 39.8</td></tr><tr><td>0.70</td><td>59</td><td>115 361</td><td>33.9</td><td>14.0 19.9</td></tr><tr><td>1.00</td><td>59</td><td>115 361</td><td>33.9</td><td>14.0 19.9</td></tr></table>

3) Influence of the Curve Model and Fitting Estimator: In the following experiments, a single component of the pipeline is modified at a time, while all other parameters are kept at the values listed in Table V. Table VIII compares the default Huber-weighted polynomial fitting with two alternatives: (i) the same line and parabola estimated by ordinary least squares (OLS) without re-weighting, and (ii) a catenary model as defined in equation 27:

$$
z ( t ) = z _ { 0 } + \frac { \cosh ( k ( t - t _ { m } ) + \varphi ) - \cosh ( \varphi ) } { k }\tag{27}
$$

![](images/53df80f9f3ebdd501aaaa4531b36f9c95a68a19b8d5a30b22edd081c608def82.jpg)  
Fig. 9: Challenging configurations involving intertwined conductors and nonlinear conductor trajectories.

TABLE VIII: Influence of the curve model and fitting estimator for mobile LiDAR.
<table><tr><td>Fitting</td><td>TP</td><td>FP</td><td>FN</td><td>Prec.</td><td>Rec.</td><td> $\mathbf { F } _ { 1 }$ </td><td>s/file</td></tr><tr><td>Polynomial (OLS)</td><td>394</td><td>17</td><td>26</td><td>95.9</td><td>93.8</td><td>94.8</td><td>0.11</td></tr><tr><td>Catenary (Huber IRLS)</td><td>400</td><td>9</td><td>20</td><td>97.8</td><td>95.2</td><td>96.5</td><td>1.27</td></tr><tr><td>Polynomial (Huber IRLS)</td><td>402</td><td>6</td><td>18</td><td>98.5</td><td>95.7</td><td>97.1</td><td>0.35</td></tr></table>

estimated with the same Huber IRLS scheme. The fitting model is used consistently in fragment merging, span validation and rescue.

Omission of the Huber re-weighting induces the most detrimental effect: the $F _ { 1 }$ -score decreases from 97.1% to 94.8%, accompanied by a simultaneous increase in both false positives (FPs) and false negatives (FNs). Without robust re-weighting, stray points bias the fitted curve, artificially inflating the RMSE of valid fragments. Consequently, these fragments fail to merge with their corresponding conductors, resulting in omitted conductors and residual fragments misclassified as FPs.

The catenary model yields comparable but slightly inferior performance relative to the parabolic model (F<sub>1</sub>-score of 96.5% versus 97.1%). Crucially, it exhibits a 3.6-fold increase in computational runtime. This overhead stems from the fact that each IRLS iteration requires a non-linear least-squares solution, whereas the parabolic model remains linear in its parameters. Given that the two curves differ by only a few millimeters under typical spans and sags, the parabola serves as a highly sufficient approximation. Nevertheless, it is worth noting that the synthetic conductors within the Damos dataset are generated as parabolas, which may inherently favor the parabolic model in this evaluation.

4) Standard versus Huber RMSE: To evaluate the merging criterion, Eq. (15) replaces the standard squared residual with the Huber loss $( \delta _ { H } = 0 . 1 \ \mathrm { m } )$ . Consequently, residuals exceeding $\delta _ { H }$ contribute only linearly to the total error. Table IX compares this approach with the standard RMSE across various thresholds (τ<sub>RMSE</sub>), while keeping all other pipeline stages identical.

At the selected threshold $( \tau _ { \mathrm { R M S E } } = 0 . 3 0$ m), both criteria yield nearly identical performance: the standard RMSE achieves an $F _ { 1 }$ -score of 97.3%, whereas the Huber RMSE achieves 97.1%, representing a difference of a single conductor. However, because the Huber loss grows more slowly than the squared loss, it produces lower values for identical residuals. For a given τ<sub>RMSE</sub>, the Huber criterion is therefore more permissive. This behavior explains why its performance degrades more rapidly at larger thresholds; at τ<sub>RMSE</sub> $= \ 0 . 4 0 \mathrm { ~ m ~ }$ , the $F _ { \mathrm { 1 } } { \mathrm { - s c o r e } }$ drops to 94.2% compared to 97.0% for the standard RMSE, as the algorithm begins erroneously merging fragments from neighboring conductors.

TABLE IX: Standard versus Huber RMSE merging criterion for mobile LiDAR. The adopted configuration is the Huber RMSE with $\tau _ { \mathrm { R M S E } } = 0 . 3 0 ~ \mathrm { m }$
<table><tr><td rowspan="2">TRMSE</td><td rowspan="2">Standard RMSE TP FP FN</td><td rowspan="2"> $\mathbf { F } _ { 1 }$ </td><td colspan="3">Huber RMSE</td></tr><tr><td>TP FP</td><td>FN</td><td> $\mathbf { F } _ { 1 }$ </td></tr><tr><td>0.10</td><td>401 9 19</td><td>96.6</td><td>401</td><td>9 19</td><td>96.6</td></tr><tr><td>0.15</td><td>401 8 19</td><td>96.7</td><td>402</td><td>8 18</td><td>96.9</td></tr><tr><td>0.20</td><td>403 5 17</td><td>97.3</td><td>403</td><td>5 17</td><td>97.3</td></tr><tr><td>0.25</td><td>403 5 17</td><td>97.3</td><td>401</td><td>6 19</td><td>97.0</td></tr><tr><td>0.30</td><td>403 5 17</td><td>97.3</td><td>402</td><td>6 18</td><td>97.1</td></tr><tr><td>0.40</td><td>401 6 19</td><td>97.0</td><td>387</td><td>15 33</td><td>94.2</td></tr></table>

On this synthetic dataset, which contains few outliers, the bounded influence of the Huber loss provides no measurable advantage. Nevertheless, its primary utility lies in processing real-world data, where it prevents a small number of stray points from causing the rejection of an otherwise coherent line fragment.

5) Effect of Span-Aligned Prescaling: Distribution network conductors in LiDAR data can extend over more than 50 m while being separated laterally by only 0.7–1.0 m. Applying HDBSCAN directly to the original coordinates treats the along-span and cross-span coordinates equally. Consequently, large longitudinal gaps can dominate the distance metric, while the smaller separation between neighboring conductors receives insufficient emphasis. In the unscaled baseline, all coordinate weights are set to one, i.e., $w _ { s } = w _ { n } = w _ { z } = 1$ which reduces the prescaling to a rotation of the point cloud. In the proposed configuration, $w _ { s } \in [ 0 . 0 5 , 1 . 0 ]$ is computed adaptively, $w _ { n } ~ = ~ 2 . 0$ and $w _ { z } ~ = ~ 1 . 0 $ ; all other parameters are unchanged. Table X compares both configurations. On the mobile LiDAR dataset, prescaling has a small effect: it recovers 2 additional TPs at the cost of 1 additional FP, increasing the $_ \mathrm { F _ { 1 } - s c o r e }$ by 0.1. The synthetic conductors are densely and regularly sampled, so that most fragments are already separated in the original coordinates. On the aerial dataset, where the point density is lower and longitudinal gaps are larger, the prescaling increases the $\mathrm { F _ { 1 } }$ -score and reduces the reconstruction RMSE.

6) Overall Parameter Sensitivity: Table XI summarizes the observed $\mathrm { F _ { 1 } }$ -score range when each parameter is varied independently, and all remaining parameters are held at their selected values $( \mathrm { F _ { 1 } } { = } 9 7 . 1 \% )$ . Among the evaluated parameters, the merging parameters τ<sub>RMSE</sub> and $\delta _ { H } .$ , and the span ratio $\tau _ { \mathrm { s p a n } }$ produce the largest performance variations. The parameters τ<sub>RMSE</sub> and $\delta _ { H }$ have the strongest overall influence because they directly control the precision–recall trade-off during fragment merging. Small values of τ<sub>RMSE</sub> prevent valid but separated fragments from being associated, whereas excessively large values permit fragments from neighboring phases to be merged (Section IV-D2). Since the Huber loss never exceeds the squared residual, reducing $\delta _ { H }$ lowers the RMSE of all candidates and has the same effect as a larger τ : for $\delta _ { H } = 0 . 0 2$ m, neighboring phases are merged in all multi-conductor spans, whereas all values $\delta _ { H } \geq 0 . 2$ m give identical results $( \mathrm { F _ { 1 } } { = } 9 7 . 3 \% )$ ). The span ratio $\tau _ { \mathrm { s p a n } }$ has no effect between 0.2 and 0.8 and reduces recall only when values close to 1 require conductors to cover almost the complete span. The clustering parameters ϵ<sub>HDBSCAN</sub>, $\rho _ { s } , \ w _ { s , \mathrm { { m i n } } } .$ , and $k _ { \mathrm { m i n } }$ change the $\mathrm { F _ { 1 } }$ -score by less than one percentage point, and $w _ { s , \mathrm { m a x } }$ has no effect, because $w _ { s }$ never exceeds 0.1 for the span lengths of this dataset. The settings that slightly improve the mobile LiDAR results, namely $w _ { s , \mathrm { m i n } } = 0 . 2 , \rho _ { s } = 5 0$ and $\delta _ { H } \geq 0 . 2 \textrm { m }$ , each change the outcome of a single file and were therefore also evaluated on the aerial dataset. None of them improves the aerial results consistently. The selected values therefore provide a balanced performance on both the mobile and the airborne LiDAR data.

TABLE X: Detection performance with and without spanaligned prescaling for mobile and aerial LiDAR. Precision, recall, and $\mathrm { F _ { 1 } }$ -score are reported as percentages.
<table><tr><td>Dataset</td><td>Configuration</td><td>TP</td><td>FP</td><td>FN</td><td>Prec.</td><td>Rec.</td><td> $\mathbf { F } _ { 1 }$ </td></tr><tr><td>Mobile</td><td>Prescaling No prescaling</td><td>402 400</td><td>6 5</td><td>18 20</td><td>98.5 98.8</td><td>95.7 95.2</td><td> $\overline { { 9 7 . 1 } }$  97.0</td></tr></table>

TABLE XI: Summary of the one-at-a-time parameter sensitivity analysis for mobile LiDAR.
<table><tr><td>Param.</td><td>Selected Val.</td><td>Test. range</td><td>Min.  $\mathbf { F } _ { 1 }$ </td><td>Max.  $\mathbf { F } _ { 1 }$ </td></tr><tr><td>TRMSE</td><td>0.30</td><td>[0.05, 1]</td><td>19.9 %</td><td>97.3 %</td></tr><tr><td> $\delta _ { H }$ </td><td>0.10</td><td>[0.02, 0.5]</td><td>19.9 %</td><td>97.3 %</td></tr><tr><td> $\tau _ { \mathrm { s p a n } }$ </td><td>0.55</td><td>[0.2, 1]</td><td>90.9 %</td><td>97.1 %</td></tr><tr><td> $w _ { s , \mathrm { m a x } }$ </td><td>1</td><td>[0.1, 2]</td><td>97.1 %</td><td>97.1 %</td></tr><tr><td> $w _ { s , \mathrm { m i n } }$ </td><td>0.05</td><td>[0.01,0.5]</td><td>97.1 %</td><td>97.5 %</td></tr><tr><td>€HDBSCAN</td><td>0.3</td><td>[0.1, 2]</td><td>96.5 %</td><td>97.5 %</td></tr><tr><td> $\rho _ { s }$ </td><td>20</td><td>[5,200]</td><td>96.9 %</td><td>97.5 %</td></tr><tr><td> $k _ { \mathrm { m i n } }$ </td><td>3</td><td>[2, 30]</td><td>96.5 %</td><td>97.3 %</td></tr></table>

## V. CONCLUSION AND FUTURE WORK

This paper presents a robust method for reconstructing power-line geometry from sparse and occluded mobile and aerial LiDAR point clouds. Following semantic classification, the proposed pipeline applies the same reconstruction implementation and parameter configuration to both modalities. It combines span-aligned prescaling, HDBSCAN clustering, Huber-weighted polynomial fitting, iterative fragment merging, and a rescue stage for rejected fragments. The controlled experiments demonstrate robustness to uniform point thinning, distributed gaps, and large contiguous occlusions. Under these conditions, the proposed method provides more consistent performance than the evaluated EPL and Cao-FEC pipelines. The sensitivity analysis identifies the maximum fragment-merging distance as the most influential parameter, while showing relatively stable performance across the tested HDBSCAN epsilon values. On 10,870 aerial spans containing 23,372 reference conductors, our method achieved a precision of 96.87%, a recall of 96.66%, an $\mathrm { F _ { 1 } }$ -score of 96.76%, and a geometric RMSE of 0.12 m. On 1,620 mobile spans, it achieved a reconstruction coverage of 95.27% and a point-tomodel RMSE of 0.045 m. The main limitations are the absence of independent conductor-level annotations for the mobile dataset. However, the controlled robustness experiments and the aerial evaluation use independent conductor references and therefore partially mitigate this limitation. Another limitation is the reliance on generated semantic labels for aerial and mobile LiDAR, and the limited capability of the quadratic model to represent nonlinear, crossing, or intertwined conductors such as in Fig. 9. Future work will investigate orientationaware hierarchical clustering and more flexible geometric models for these configurations. Moreover, it will strengthen the integration of conductor reconstruction with vegetationencroachment assessment and utility decision-support tools [19]. To support reproducibility, we make the source code, documentation, and evaluation data publicly available at https: //github.com/varen-project.

## ACKNOWLEDGMENT

This work is supported by the NSERC Alliance-Mitacs Accelerate research program CRSNG ALLRP 607469-25.

## REFERENCES

[1] Y. A. Waykar, “LiDAR technology: A comprehensive review and future prospects,” J. of Emerg. Technol. and Innov. Res., vol. 9, no. 7, 2022.

[2] Y. Jwa et al., “Automatic 3D powerline reconstruction using airborne LiDAR data,” in Int. Archives of the Photogram., Remote Sens. and Spatial Info. Sci., vol. XXXVIII-3/W8, Paris, France, Sep. 2009, pp. 105– 110.

[3] D. Cao et al., “An automatic method for powerline extraction from ALS point cloud of powerline corridors,” IEEE J. ofSel. Top. in Appl. Earth Observa. and Remote Sens., vol. 18, pp. 10 803–10 829, 2025. DOI: 10.1109/ JSTARS.2025.3555534

[4] M. Gaha et al., “A new LiDAR-based approach for poles and distribution lines detection and modelling,” Computer Sci. & Info. Technol., vol. 11, no. 6, pp. 85– 97, May 2021. DOI: 10.5121/csit.2021.110607

[5] B. Guo et al., “An improved method for power-line reconstruction from point cloud data,” Remote Sens., vol. 8, no. 1, p. 36, 2016, ISSN: 2072-4292. DOI: 10. 3390/rs8010036

[6] Z. Shi et al., “Extraction of urban power lines and potential hazard analysis from mobile laser scanning point clouds,” Int. J. of Remote Sens., vol. 41, no. 9, pp. 3411–3428, 2020. DOI: 10.1080/01431161.2019. 1701726

[7] N. Munir et al., “Power line extraction and reconstruction methods from laser scanning data: A literature review,” Remote Sensing, vol. 15, no. 4, p. 973, 2023. DOI: 10.3390/rs15040973

[8] V. Kumar et al., “Powerline extraction from aerial and mobile LiDAR data using deep learning,” Earth Science Informatics, vol. 17, no. 4, pp. 2819–2833, 2024. DOI: 10.1007/s12145-024-01310-w

[9] J. Liang et al., “A new power-line extraction method based on airborne LiDAR point cloud data,” in Proc. Int. Symp. Image Data Fus., IEEE, Aug. 2011, pp. 1–4. DOI: 10.1109/ISIDF.2011.6024293

[10] M. A. Fischler and R. C. Bolles, “Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography,” Commun. of the ACM, vol. 24, no. 6, pp. 381–395, 1981. DOI: 10.1145/358669.358692

[11] P. Massicotte et al., “Automated inventory of electrical distribution assets based on image recognition and ground LiDAR,” in Proc. Europ. Sig. Process. Conf., Lyon, France: IEEE, Aug. 2024, pp. 486–490.

[12] E. Maset and A. Fusiello, “Automatic vectorization of power lines from airborne lidar point clouds,” The Int. Archiv. of the Photogram., Rem. Sens. and Spat. Info. Sci., vol. XLVIII-2-2024, pp. 225–231, 2024. DOI: 10. 5194/isprs-archives-XLVIII-2-2024-225-2024

[13] Y. Shen et al., “A review and future directions of techniques for extracting powerlines and pylons from LiDAR point clouds,” Int. J. ofAppl. Earth Observ. and Geoinfo., vol. 132, p. 104 056, 2024. DOI: 10.1016/j. jag.2024.104056

[14] Z. Zeng et al., “Small but mighty: Enhancing 3D point cloud semantic segmentation with U-Next framework,” Int. J. of Appl. Earth Observ. and Geoinfo., vol. 136, p. 104 309, 2025, ISSN: 1569-8432. DOI: 10.1016/j.jag. 2024.104309

[15] Blue Marble Geographics, Global Mapper Pro, version 25.1, Comput. Softw. 2024. [Online]. Available: https: //www.bluemarblegeo.com/global-mapper/

[16] L. McInnes et al., “HDBSCAN: Hierarchical density based clustering,” J. of Op. Source Softw., vol. 2, no. 11, p. 205, 2017. DOI: 10.21105/joss.00205

[17] M. Ester et al., “A density-based algorithm for discovering clusters in large spatial databases with noise,” in Proc. Int. Conf. Knowl. Discov. and Data Mining (KDD), 1996, pp. 226–231.

[18] P. J. Huber, “Robust estimation of a location parameter,” The Annals of Mathema. Statist., vol. 35, no. 1, pp. 73– 101, 1964. DOI: 10.1214/aoms/1177703732

[19] P. Massicotte et al., “Field deployment of an automated pruning prescription method by analyzing vegetation using terrestrial LiDAR,” in IEEE Int. Instrument. and Measure. Technol. Conf., Accepted for presentation; proceedings forthcoming, 2026.

[20] L. Louis-Alexandre et al., “Inventaire automatisé des poteaux et conducteurs de distribution électrique basé sur la reconnaissance de LiDAR terrestre,” in Actes du Colloque GRETSI, France: GRETSI, 2025, pp. 1077– 1080.