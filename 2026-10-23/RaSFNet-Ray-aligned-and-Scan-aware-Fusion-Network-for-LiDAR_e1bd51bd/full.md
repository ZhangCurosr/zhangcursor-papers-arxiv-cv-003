# RaSFNet: Ray-aligned and Scan-aware Fusion Network for LiDAR Semantic Segmentation

Chuang Chen

Abstract—LiDAR point cloud semantic segmentation is a fundamental component of 3D perception for autonomous driving and robotics. Projection-based methods map 3D point clouds into 2D representations for feature extraction using 2D networks. However, aggregating multiple points during projection can obscure geometric differences within each frustum, while local feature extraction limits the modeling of spatial relationships across scanning regions. Moreover, proximity in the projected representation does not necessarily imply proximity in 3D space, and interactions between point and projected features may mix features from different ranges, reducing semantic discriminability. To address these challenges, we propose the Ray-aligned and Scan-aware Fusion Network (RaSFNet), which incorporates point cloud geometry and LiDAR scanning characteristics into projection-based semantic segmentation. Specifically, the Ray-Aligned Geometric Distribution Encoding (RAGDE) module characterizes point distributions within each frustum using a ray-aligned coordinate system. To model spatial context, a Scan-Topology-Aware Mamba (STAM) block organizes frustum features according to the LiDAR azimuth sampling pattern and captures long-range dependencies across scanning regions. In addition, a Range-Constrained Point-Frustum Fusion (RCPFF) mechanism uses shared range information to guide feature exchange between point and frustum representations while maintaining spatial correspondence. RaSFNet achieves mIoU scores of 70.2% and 77.7% on the SemanticKITTI and nuScenes validation sets, respectively, indicating competitive segmentation performance compared with the evaluated methods.

Index Terms—LiDAR semantic segmentation; 2D representation; geometric encoding; Mamba; point-frustum fusion.

## I. INTRODUCTION

IDAR point cloud semantic segmentation is a fundagrained environmental perception for autonomous driving, robotic navigation, and intelligent transportation systems, and serving as an essential foundation for outdoor autonomous decision-making [1], [2]. However, LiDAR point clouds acquired in real-world outdoor environments are inherently characterized by irregular spatial distributions, sparsity, and highly non-uniform point densities, which make it challenging for existing models to effectively capture fine-grained geometric structures [3], [4], [5]. Consequently, spatially adjacent ground classes with similar geometric characteristics, such as parking areas, sidewalks, and terrain, can be easily confused, limiting segmentation accuracy and the capability of existing models to meet the demands of complex outdoor perception [6], [7], [8], [9], [10]. Therefore, developing feature modeling methods tailored to the inherent characteristics of LiDAR point clouds, which can jointly capture fine-grained geometric details and global spatial dependencies, is crucial for improving semantic segmentation performance in complex outdoor scenes and remains a key challenge in 3D LiDAR perception [11], [12].

Considering that LiDAR acquires three-dimensional structural information by continuously scanning the surrounding environment with multiple laser beams along the horizontal direction, the resulting point clouds preserve regular spatial indexing relationships in the angular domain [13], [14]. Building on this property, range-image methods [15], [16], [17], [7] employ spherical projection to determine the row and column positions of each 3D point on a 2D grid according to its elevation and azimuth angles, transforming irregular point clouds into structured 2D representations. Such representations preserve the intrinsic scanning geometry and support the use of mature 2D backbone networks, providing structured inputs for LiDAR perception and making range-view methods a mainstream paradigm for point cloud semantic segmentation. However, projecting 3D point clouds onto a fixed-resolution 2D grid is inherently a lossy spatial compression process that cannot fully preserve the original three-dimensional structural information [18], [19]. On the one hand, multiple points at different ranges may be mapped to the same pixel, causing projection conflicts and the loss of local geometric details. On the other hand, sparse LiDAR sampling can lead to insufficient feature information in critical regions. These issues introduce discrepancies between 2D pixel adjacency and the 3D spatial topology, making it difficult to accurately capture geometric relationships in complex scenes and limiting the model's ability to effectively represent complete three-dimensional structures [7], [8], [9], [18], [19], [20].

Notably, the aforementioned projection issues are not limited to the representation of raw point clouds and may be further amplified by data augmentation strategies [9], [17]. A common strategy is to copy local point sets from other scenes into the current scene [21], [22], enriching scene compositions and object distributions in the training data while alleviating overfitting to specific point cloud distributions. Although such methods can generate richer scene layouts in 3D space, they typically focus on the geometric plausibility of insertion positions while overlooking LiDAR ray visibility and occlusion constraints [9], [20]. In a LiDAR system, only the first valid return along a given laser direction is generally retained [16], [17]. After external point sets are introduced, newly inserted points may share the same scanning direction as foreground points in the original scene. As illustrated in Fig. 1, when the augmented point cloud is projected onto the range image, points with different ranges and semantic labels may fall into the same image cell, disturbing the local distributions of range, intensity, and geometric features. To alleviate the information loss caused by projection conflicts, existing studies organize multiple points mapped to the same grid location into local frustum units and encode them as 2D frustum features [9], [12], [20]. A bidirectional fusion mechanism is established between 3D point features and 2D frustum features, allowing the original point information in overlapping projection regions to continuously participate in iterative feature updates and alleviating the information compression caused by single-pixel representations. Despite this improvement, the intrinsic characteristics of LiDAR point clouds remain insufficiently exploited in 3D structure encoding, frustum feature extraction, and feature fusion, limiting the modeling of geometric and topological relationships in complex scenes and the discrimination of spatially adjacent structures with similar geometry, as illustrated in Fig. 1.

![](images/a2baa69c5974119be9bb503335fa088b62932e66f992cade338cdc3f49ca7a06.jpg)  
Fig. 1. Visualization of projection conflicts caused by point cloud augmentation and their influence on semantic segmentation. The left part compares the original and augmented point clouds together with their corresponding range-view projections, while the right part shows the segmentation results of different methods in the affected regions. Red points indicate misclassified regions.

Motivated by the above limitations, we propose a new method for collaborative learning between 3D structural features and 2D frustum features, termed the Ray-aligned and Scan-aware Fusion Network (RaSFNet), which enhances finegrained 3D structure discrimination and improves the representation of geometrically similar objects in complex scenes. To distinguish different range distributions and geometric patterns within each frustum, we design a Ray-Aligned Geometric Distribution Encoding (RAGDE) module. Points are explicitly represented by their radial range and tangential offsets in a ray-aligned coordinate system, while multi-frequency sine and cosine functions are used for Fourier feature [23] mapping to enhance the representation differences among distinct spatial organization patterns. To enhance the global modeling capability of frustum features, we design a Scan-Topology-Aware Mamba (STAM) block. The module sequentially organizes features according to the LiDAR azimuth sampling pattern and leverages Mamba [24] to capture long-range dependencies across scanning regions, improving the representation of structural variations among different scan regions and global spatial relationships. To address the decoupling between 3D structural features and 2D frustum features during deep interaction, we introduce a a Range-Constrained Point-Frustum Fusion (RCPFF) mechanism that uses distance information as a shared cue linking 2D projections and 3D spatial structures, guiding the collaborative update of both feature types while enhancing spatial consistency and semantic complementarity.

Overall, our main contributions are summarized as follows:

1) We propose RaSFNet, a network for collaborative learning between 3D structural features and 2D frustum features. Unlike existing point-frustum interaction methods that lack persistent spatial constraints, RaSFNet explicitly models the geometric structures within frustums and LiDAR scanning relationships, while using range information to constrain the interaction between point and frustum features.

2) We propose RAGDE, which uses each frustum ray as a reference and encodes radial range, tangential offsets, and local variations with multi-frequency Fourier features to distinguish geometric distributions.

3) We propose a STAM Block, which reorganizes frustum features into sequences according to the LiDAR azimuth scanning pattern, extending feature extraction from fixed local neighborhoods to global scan-wise dependencies.

4) We propose RCPFF, which uses distance information to constrain the interaction between 3D structural and 2D frustum features, guiding their complementary updates while maintaining spatial correspondence.

5) Comprehensive experiments on the SemanticKITTI [13] and nuScenes [14] validation sets show that RaSFNet achieves mIoU scores of 70.2% and 77.7%, respectively.

## II. RELATED WORK

## A. Point Cloud Feature Encoding

Point cloud feature encoding aims to extract discriminative 3D geometric information from the spatial distribution of raw point clouds. PointNet [3] uses shared multilayer perceptrons (MLPs) for point-wise encoding, while PointNet++ [25] adds hierarchical local feature aggregation. RandLA-Net [4] strengthens neighborhood features through attentive local encoding. Cylinder3D [26] instead organizes points with cylindrical voxelization. Recent methods further improve point cloud representation from different perspectives. SFPNet [5] employs sparse focal point modulation to aggregate multilevel contextual information, PointGeo [27] jointly captures local and global geometric features, and MDSLCA [28] combines multi-scale local attention with dilated spatial attention to enhance structural representations over different spatial ranges. Unlike methods that directly process unordered point sets, RangeNet++ [15] projects range, 3D coordinates, and remission values onto a regular 2D grid, while SphereFormer [11] introduces radial windows to accommodate the distancedependent sparsity of LiDAR point clouds. Although these methods enhance geometric feature encoding, their representations remain largely focused on local neighborhood structures, with insufficient explicit modeling of differences in range distributions, geometric patterns, and spatial organizations. Consequently, point sets with distinct structural characteristics may still produce similar encoded representations. In this work, we introduce RAGDE, which characterizes radial and tangential structures along the LiDAR sensing ray, while multi-frequency Fourier mapping improves the separability of local point distributions.

## B. Projected Image Feature Extraction

Projection-image feature extraction transforms point clouds into structured image representations and leverages 2D networks to learn semantic features at different scales. RangeViT [17] divides the projected image into feature patches and employs Transformers [29] to integrate local details with global semantics, while RangeFormer [7] adopts a hierarchical Transformer architecture to strengthen feature interactions across different regions. Fast FMVNet [8] improves rangeimage representation through an efficient multi-scale architecture, whereas MINet [30] employs multiple feature paths at different scales with dense cross-scale interactions. RangeRet [31] introduces circular retention to exploit the 360<sup>◦</sup> continuity of range images and model spatial dependencies through distance-aware decay. Nevertheless, the projected features in these methods are still organized primarily on regular 2D grids, with limited consideration of range and point-density variations across different azimuthal regions. In this work, we propose STAM, which reorganizes feature sequences according to the LiDAR azimuth sampling pattern and performs differentiated modeling across scanning regions.

## C. Point Cloud-Projected Image Feature Fusion

To compensate for the information loss caused by a single projected representation, point-projection fusion methods retain the original point features and exploit the spatial correspondence between points and image representations for cross-representation interaction. PRNet [18] introduces rangeview projection and point-range convolution to dynamically map range-image features back to the original points. FRNet [9] employs point-frustum mappings to alternately propagate frustum features and aggregate point features across multiple network stages. HARP-NeXt [32] further develops a multiscale point–range fusion backbone to enable interaction between 3D geometric information and 2D semantic information at different feature levels. MFINet [12] constructs 3D pointview, bird's-eye-view, and range-view branches, enabling bidi rectional 3D-2D interaction through point-feature projection and 2D feature enhancement while integrating complementary information across different views. However, these methods mainly rely on fixed point–projection correspondences for feature fusion, without maintaining a shared geometric constraint during feature updates. In this work, we propose a RCPFF mechanism that treats the range information shared by both representations as a common geometric anchor and explicitly incorporates it into feature interaction and updating, preserving spatial correspondence while enhancing the complementary representation of 3D geometric and 2D semantic information.

## III. METHODOLOGY

RaSFNet enables collaborative learning between 3D structural representations and 2D frustum representations through multi-stage feature interaction, as illustrated in Fig. 2. Given an input point cloud, spatial correspondences between points and frustums are first established according to the projection relationship, producing the corresponding 3D and 2D representations. RAGDE, STAM, and RCPFF are then integrated into successive feature stages to encode, update, and fuse the two types of features. Finally, the fused features are mapped back to the point space through the prediction head to generate point-wise semantic labels.

## A. Problem Definition

Given an input LiDAR point cloud $\mathbf { \mathcal { P } } ~ = ~ \{ ( \mathbf { p } _ { i } , r _ { i } ) \} _ { i = 1 } ^ { N } ,$ where N denotes the number of points, $\mathbf { p } _ { i } = [ x _ { i } , y _ { i } , z _ { i } ] ^ { \mathrm { T } } \in \mathbb { R } ^ { 3 }$ represents the 3D coordinate of the i-th point, and $r _ { i }$ denotes the corresponding remission intensity. Each point is mapped onto a 2D grid through spherical projection:

$$
( u _ { i } , v _ { i } ) = \Pi ( \mathbf { p } _ { i } )\tag{1}
$$

where Π(·) denotes the LiDAR spherical projection function and $( u _ { i } , v _ { i } )$ represents the corresponding 2D grid coordinate. According to the projection coordinates, points mapped to the same grid cell are grouped into a frustum unit. Each frustum unit is associated with a 2D grid location and retains the indices of its constituent 3D points. The corresponding point and frustum features are represented as:

$$
\mathbf { F } ^ { 3 D } \in \mathbb { R } ^ { N \times C _ { 3 D } } \qquad \mathbf { F } ^ { 2 D } \in \mathbb { R } ^ { H \times W \times C _ { 2 D } }\tag{2}
$$

where $\mathbf { F } ^ { 3 D }$ denotes the 3D structural features associated with the original points, while $\mathbf { F } ^ { 2 D }$ denotes the 2D frustum features organized on an $H \times W$ projected grid.

## B. Ray-Aligned Geometric Distribution Encoding

Points projected to the same image cell may exhibit different range distributions and geometric structures, while conventional pooling operations tend to suppress such structural differences [9]. To address this issue, we propose the RAGDE module, as shown in Fig. 3, to enhance the discrimination of intra-frustum 3D structures.

![](images/7a67b4b0d6a4ed3a32ca14cf5d5e75ca57048ad9a07e4a3e451cb8a703a494b7.jpg)  
Fig. 2. Overall architecture of the proposed RaSFNet. The network consists of RAGDE, STAM, RCPFF, and a prediction head. The input point cloud is encoded into 3D structural features and 2D frustum features, which are progressively updated through hierarchical feature extraction and cross-representation interaction, followed by feature fusion and final semantic prediction.

![](images/d2ca6958132150a73a40f9e0177bdeb803c15a3e1dd5f6e04be26556147ab493.jpg)  
Fig. 3. Architecture of the RAGDE module for encoding ray-aligned point distributions.

For the g-th frustum point set $\mathcal { P } _ { g } = \{ ( \mathbf { p } _ { i } , r _ { i } ) \} _ { i = 1 } ^ { N _ { g } }$ , we first compute its spatial center $\bar { \mathbf { p } } _ { g } ,$ with the direction from the LiDAR origin to this center serving as the radial axis $\mathbf { e } _ { r } ^ { g } .$ . Two orthogonal tangential axes, $\mathbf { e } _ { t _ { 1 } } ^ { g }$ and ${ \bf e } _ { t _ { 2 } } ^ { g }$ , are further constructed on the plane perpendicular to the radial direction, under which the ray-aligned local coordinate of point $\mathbf { p } _ { i }$ is expressed as:

$$
\mathbf { q } _ { i } = \left[ ( \mathbf { p } _ { i } - { \bar { \mathbf { p } } } _ { g } ) ^ { \mathrm { T } } \mathbf { e } _ { r } ^ { g } , ( \mathbf { p } _ { i } - { \bar { \mathbf { p } } } _ { g } ) ^ { \mathrm { T } } \mathbf { e } _ { t _ { 1 } } ^ { g } , ( \mathbf { p } _ { i } - { \bar { \mathbf { p } } } _ { g } ) ^ { \mathrm { T } } \mathbf { e } _ { t _ { 2 } } ^ { g } \right] ^ { \mathrm { T } }\tag{3}
$$

To amplify subtle differences among similar local geometric features, logarithmic standard deviations $\sigma _ { \boldsymbol { q } } ^ { g }$ are used to preserve directional scale information, while frustum-wise standardization and fixed-range clipping convert $\mathbf { q } _ { i }$ into $\tilde { \mathbf { q } } _ { i }$ for multi-frequency encoding:

$$
\mathbf { q } _ { i } ^ { \mathrm { m f } } = \mathrm { C o n c a t } _ { b = 0 } ^ { B - 1 } \left[ \sin ( 2 ^ { b } \pi \tilde { \mathbf { q } } _ { i } ) , \cos ( 2 ^ { b } \pi \tilde { \mathbf { q } } _ { i } ) \right] , B = 4\tag{4}
$$

The point attributes are first augmented with range information and intra-frustum coordinate and intensity offsets. These augmented attributes are concatenated with the $\tilde { \mathbf { q } } _ { i } , \log ( 1 { + } \sigma _ { q } ^ { g } )$ and ${ \bf q } _ { i } ^ { \mathrm { m f } }$ processed by point-wise feature layers to obtain $\mathbf { F } ^ { 3 D }$

Given the point-level features $\mathbf { F } ^ { 3 D }$ , RAGDE constructs an initial frustum representation through distribution-aware aggregation. We compute the channel-wise mean and standard deviation to preserve both the overall feature response and its variability. To incorporate explicit spatial information, local coordinate statistics capture point locations and directional spread. These statistics are combined with range distribution, point count, and intensity statistics to form an auxiliary descriptor $\mathbf { s } _ { ( u _ { i } , v _ { i } ) }$ . Points sharing the same projected coordinates $( u _ { i } , v _ { i } )$ are aggregated into a single frustum feature, which is assigned to the corresponding grid location:

$$
\mathbf { F } ^ { 2 D } [ : , v _ { i } , u _ { i } ] = \phi _ { f } \left( \left[ \pmb { \mu } _ { ( u _ { i } , v _ { i } ) } , \pmb { \sigma } _ { ( u _ { i } , v _ { i } ) } , \mathbf { s } _ { ( u _ { i } , v _ { i } ) } \right] \right) ,\tag{5}
$$

where $\pmb { \mu } _ { ( u _ { i } , v _ { i } ) }$ and $\pmb { \sigma } _ { ( u _ { i } , v _ { i } ) }$ denote the channel-wise mean and standard deviation of the point-level features projected to $( u _ { i } , v _ { i } )$ , respectively. The operator [·] denotes concatenation, and $\phi _ { f }$ consists of a linear layer, BatchNorm, and SiLU activation.

## C. Scan-Topology-Aware Mamba

The horizontal axis of $\mathbf { F } ^ { 2 D }$ follows the $3 6 0 ^ { \circ }$ LiDAR azimuth scan, with its first and last columns corresponding to adjacent directions, whereas the vertical axis indexes different laser beams. To accommodate this anisotropic structure, STAM combines axis-aware local feature aggregation with sequence modeling over a subset of channels to capture feature dependencies along both axes, as shown in ${ \mathrm { F i g . } }$ . 4.

STAM maps the input frustum feature $\mathbf { F } ^ { 2 D }$ into the target feature space through a $3 \times 3$ convolution. To account for scale differences between the horizontal azimuth and vertical laserbeam axes, axis-specific local aggregation is performed using range-wise convolutions [33] with kernel sizes of $1 \times 5$ and $3 \times 1$ , respectively:

![](images/0cc4a7ade5ee65d7d7ada7eef47c208c323ad7884e18118651b4984bb92ed494.jpg)  
Fig. 4. Architecture of STAM for anisotropic feature modeling.

$$
\begin{array} { r l r } & { } & { \mathcal { A } ( X ) = \alpha _ { h } \mathcal { C } _ { 1 \times 5 } ( X ) + \alpha _ { v } \mathcal { C } _ { 3 \times 1 } ( X ) , } \\ & { } & { [ \alpha _ { h } , \alpha _ { v } ] = \mathrm { S o f t m a x } ( [ a _ { h } , a _ { v } ] ) , } \\ & { } & { \mathbf { F } _ { A x i s } = \mathcal { A } ( \mathcal { C } _ { 3 \times 3 } ( \mathbf { F } ^ { 2 D } ) ) , } \\ & { } & { \mathbf { F } _ { l } = \delta \left( \mathrm { N o r m } \left( \mathbf { F } ^ { 2 D } + \mathbf { F } _ { A x i s } \right) \right) } \end{array}\tag{6}
$$

where the learnable parameters $a _ { h }$ and $\boldsymbol { a } _ { v }$ balance the horizontal and vertical aggregation branches. STAM partitions the local feature $\mathbf { { F } } _ { l }$ along the channel dimension into a rowwise Mamba branch ${ \bf { F } } _ { m }$ and a retained branch $\mathbf { F } _ { \mathrm { r e t a i n } } \mathrm { : }$

$$
[ \mathbf { F } _ { m } , \mathbf { F } _ { r e t a i n } ] = \mathrm { S p l i t } _ { c } ( \mathbf { F } _ { l } ) .\tag{7}
$$

$\boldsymbol { \mathcal { A } } ( \cdot )$ injects neighborhood context into $\mathbf { F } _ { m } .$ , while $\boldsymbol { \mathcal { S } } ( \cdot )$ arranges the feature map into serpentine row-wise sequences and reverses the traversal direction between adjacent rows. STAM routes $\mathbf { F } _ { r e t a i n }$ through a bypass connection to preserve local features without sequence modeling. The Mamba branch updates ${ \bf { F } } _ { m }$ as:

$$
\mathbf { F } _ { m } ^ { \prime } = \mathbf { F } _ { m } + \gamma \mathcal { S } ^ { - 1 } \left( \operatorname { M a m b a } \left( \operatorname { L N } \left( \mathcal { S } \left( A ( \mathbf { F } _ { m } ) + \mathbf { F } _ { m } \right) \right) \right) \right)\tag{8}
$$

where $\boldsymbol { \mathcal { S } } ( \cdot )$ and $S ^ { - 1 } ( \cdot )$ denote scan-aware serialization and spatial restoration, respectively, while the learnable scalar $\gamma$ controls the Mamba update. STAM concatenates $\mathbf { F } _ { m } ^ { \prime }$ with ${ \bf F } _ { r e t a i n } .$ applies a $1 \times 1$ convolution for channel fusion, and adds the residual feature $\mathcal { R } ( \mathbf { F } ^ { 2 D } )$ to produce the output:

$$
\mathbf { F } _ { s t a m } = \delta \left( \mathrm { N o r m } \left( \mathcal { C } _ { 1 \times 1 } \left( \left[ \mathbf { F } _ { m } ^ { \prime } , \mathbf { F } _ { r e t a i n } \right] \right) \right) + \mathcal { R } ( \mathbf { F } ^ { 2 D } _ { } ) \right) ,\tag{9}
$$

where $\mathcal { R } ( \cdot )$ performs downsampling along the residual branch.

## D. Range-Constrained Point-Frustum Fusion

RCPFF uses range information as a shared geometric reference to constrain bidirectional updates between $\mathbf { F } ^ { 3 D }$ and $\mathbf { F } _ { s t a m }$ . Given the frustum feature $\mathbf { F } _ { s t a m }$ , the point-level feature $\mathbf { F } ^ { 3 D }$ , and the point-wise range range, the Frustumto-Point operation retrieves the frustum feature at $( u , v )$ and combines it with the corresponding point feature through range-guided weighted aggregation:

$$
\begin{array} { c } { { { \bf m } = \mathrm { F t P } \left( { \bf F } _ { s t a m } , ( u , v ) \right) , \nonumber } } \\  { \displaystyle { s _ { f } = \mathrm { S c o r e } ( d , { \bf m } ) , \qquad s _ { p } = \mathrm { S c o r e } ( d , { \bf F } ^ { 3 D } ) , } } \\ { { \displaystyle [ \alpha ^ { f } , \alpha ^ { p } ] = \mathrm { S o f t m a x } \left( [ s _ { f } , s _ { p } ] \right) , \ ~ } } \\ { { \displaystyle \hat { \bf F } ^ { 3 D } = \Phi _ { p } \left( \alpha ^ { f } { \bf m } + \alpha ^ { p } { \bf F } ^ { 3 D } \right) . } } \end{array}\tag{10}
$$

where $\mathrm { F t } \mathrm { P } ( \cdot )$ retrieves the frustum feature at the projected location, Score(·) computes the range-conditioned matching score, and $\Phi _ { p } ( \cdot )$ denotes the point-side fusion operation. The Softmax-normalized weights $\alpha ^ { f }$ and $\alpha ^ { p }$ determine the relative contributions of the frustum and point features, respectively, allowing contextual information from $\mathbf { F } _ { s t a m }$ to be integrated with the geometric information encoded in $\mathbf { F } ^ { 3 D }$

Point-to-Frustum interaction integrates the updated pointlevel geometry with the contextual representation produced by STAM. Projection correspondence organizes $\hat { \mathbf { F } } ^ { 3 D }$ in the frustum space, while the corresponding range information is encoded as a geometric feature. Joint fusion is formulated as

$$
\hat { \mathbf { F } } _ { s t a m } = \Phi _ { f } \left( \left[ \mathbf { F } _ { s t a m } , \psi _ { d } ( r a n g e ) , \mathrm { P t F } \left( \hat { \mathbf { F } } ^ { 3 D } \right) \right] \right) ,\tag{11}
$$

where PtF(·) denotes the projection-based organization of point features in the frustum space, $\psi _ { d } ( \cdot )$ encodes the range information, and $\Phi _ { f } ( \cdot )$ performs joint convolutional fusion. The resulting $\hat { \mathbf { F } } _ { s t a m }$ combines frustum-level context, pointlevel geometric structure, and range information within a unified representation.

## E. Fusion Head Module

Multi-stage point-frustum feature interaction progressively updates the point and frustum features, yielding hierarchical representations with varying spatial resolutions and receptive fields. These hierarchical representations encode spatial information at different granularities, with high-resolution features preserving finer spatial details and features from deeper stages capturing responses over broader receptive fields. The fusion head aggregates the multi-stage point and frustum features to integrate these complementary representations into unified point and frustum representations. Specifically, the point features are concatenated and fused according to point-wise correspondences, whereas the frustum features are first upsampled to a common spatial resolution and then concatenated and fused. The fused frustum features are subsequently mapped back to the 3D point domain and further integrated with the fused point features for final point-wise semantic prediction.

## IV. EXPERIMENTS

In this section, we conduct systematic experiments to evaluate the effectiveness of RaSFNet for outdoor LiDAR point cloud semantic segmentation. Comparisons with state-of-theart methods and ablation studies are conducted to assess its overall segmentation performance and the contributions of individual components, while qualitative results are provided to further demonstrate its segmentation performance.

## A. Datasets

We conduct comprehensive evaluations on two widely used outdoor LiDAR semantic segmentation datasets, SemanticKITTI [13] and nuScenes [14], covering diverse realworld autonomous driving scenarios. SemanticKITTI [13] was collected in Karlsruhe, Germany, using a Velodyne HDL-64E LiDAR sensor, with a vertical field of view (FoV) ranging from $- 2 5 ^ { \circ }$ to $3 ^ { \circ }$ . Following the official split, 19,130 LiDAR scans are used for training and 4,071 for validation. Each scan contains approximately 120,000 points, with 19 semantic categories considered for evaluation. nuScenes was collected using a 32-beam LiDAR sensor with a vertical FoV ranging from $- 3 0 ^ { \circ }$ to $1 0 ^ { \circ }$ . It contains 28,130 training samples and 6,019 validation samples from 1,000 driving scenes. The dataset provides annotations across 32 semantic categories, of which 16 categories are evaluated for semantic segmentation.

## B. Evaluation Metrics

We evaluate segmentation performance using the mean Intersection over Union (mIoU) and mean Accuracy (mAcc), which are commonly adopted for semantic segmentation. For each semantic class c, the IoU is defined as:

$$
\mathrm { I o U } _ { c } = \frac { T P _ { c } } { T P _ { c } + F P _ { c } + F N _ { c } } ,\tag{12}
$$

where $T P _ { c } , \ F P _ { c }$ , and $F N _ { c }$ denote the numbers of truepositive, false-positive, and false-negative points for class c, respectively. The mean IoU over $C$ evaluated classes is computed as:

$$
\mathrm { m I o U } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathrm { I o U } _ { c } .\tag{13}
$$

The class-wise accuracy is defined as:

$$
\mathrm { A c c } _ { c } = \frac { T P _ { c } } { T P _ { c } + F N _ { c } } ,\tag{14}
$$

and the mean Accuracy is given by:

$$
{ \mathrm { m A c c } } = { \frac { 1 } { C } } \sum _ { c = 1 } ^ { C } { \mathrm { A c c } } _ { c } .\tag{15}
$$

## C. Implementation Details

Although SemanticKITTI [13] and nuScenes [14] differ in LiDAR beam count (64 vs. 32), both utilize a full 360<sup>◦</sup> horizontal FoV. The range image resolutions were set to 64×1024 and 32×1024 for SemanticKITTI [13] and nuScenes [14], respectively. Each projected point is represented by its 3D coordinates and reflectance intensity. The feature encoder uses channel dimensions of (64,128,256,256) and compresses the output to 20 channels, while the four backbone stages each output 128 channels. The model is trained for 250,000 iterations using AdamW with an initial learning rate of 0.001, a batch size of 4, and the OneCycle learning-rate policy. All experiments were conducted on two NVIDIA RTX 4090 GPUs.

## D. Quantitative Results

We compare RaSFNet with LiDAR semantic segmentation methods based on different representations, including pointbased, sparse-voxel, bird's-eye-view, and range-view representations, on the SemanticKITTI [13] and nuScenes [14] validation sets. The results are presented in Tables I and II, respectively. RaSFNet achieves mIoU scores of 70.2% and 77.7% on the SemanticKITTI [13] and nuScenes [14] validation sets, respectively. On SemanticKITTI [13], RaSFNet performs well in the motorcycle, parking, sidewalk, and fence classes. On nuScenes [14], RaSFNet also obtains favorable results in the bicycle, motorcycle, bus, and pedestrian classes. The results suggest that RaSFNet maintains competitive overall accuracy and performs well across several semantic classes.

As the sensing distance increases, the spatial sampling density of LiDAR point clouds gradually decreases, accompanied by an overall decline in semantic segmentation performance across different methods. Near-range regions generally yield higher mIoU than far-range regions, indicating the performance variation under different point-density conditions. Table III provides a quantitative comparison of semantic segmentation performance and corresponding point density across different sensing-distance ranges on the SemanticKITTI [13] validation set. The average point density decreases from 923.8 to 13.1 points/m<sup>3</sup> with increasing sensing distance. Our method achieves mIoUs of 77.5%, 71.3%, 65.9%, 59.9%, and

TABLE I  
THE CLASS-WISE IOU SCORES OF DIFFERENT LIDAR SEMANTIC SEGMENTATION APPROACHES ON THE SEMANTICKITTI [13] VAL SET. ALL MIOU SCORES ARE GIVEN IN PERCENTAGE (%). THE BEST AND SECOND BEST SCORES FOR EACH CLASS ARE HIGHLIGHTED IN BOLD AND UNDERLINE.
<table><tr><td>Method</td><td>mIoU</td><td></td><td>car bi.cle</td><td>mt.cle</td><td>truck</td><td>oth.v.</td><td>pers.</td><td>b.clst</td><td>m.clst</td><td>road</td><td>park.</td><td>sidew.</td><td>oth.g</td><td>build.</td><td>fence</td><td>veget.</td><td>trunk</td><td>terra.</td><td>pole</td><td>traff.</td></tr><tr><td>RandLA-Net [4]</td><td>50.0</td><td>92.0 89.4</td><td>8.0</td><td>12.8</td><td>74.8</td><td>46.7</td><td>52.3</td><td>46.0</td><td>0.0</td><td>93.4</td><td>32.7</td><td>73.4</td><td>0.1</td><td>84.0</td><td>43.5</td><td>83.7</td><td>57.3</td><td>73.1</td><td>48.0</td><td>27.0</td></tr><tr><td>RangeNet++ [15]</td><td>51.2</td><td></td><td>26.5</td><td>48.4</td><td>33.9</td><td>26.7</td><td>54.8</td><td>69.4</td><td>0.0</td><td>92.9</td><td>37.0</td><td>69.9</td><td>0.0</td><td>83.4</td><td>51.0</td><td>83.3</td><td>54.0</td><td>68.1</td><td>49.8</td><td>34.0</td></tr><tr><td>SqueezeSegV2 [34]</td><td>40.8</td><td>82.7</td><td>15.1</td><td>22.7</td><td>25.6</td><td>26.9</td><td>22.9</td><td>44.5</td><td>0.0</td><td>92.7</td><td>39.7</td><td>70.7</td><td>0.1</td><td>71.6</td><td>37.0</td><td>74.6</td><td>35.8</td><td>68.1</td><td>21.8</td><td>22.2</td></tr><tr><td>SqueezeSegV3 [35]</td><td>53.3</td><td></td><td>87.1 34.3</td><td>48.6</td><td>47.5</td><td>47.1</td><td>58.1</td><td>53.8</td><td>0.0</td><td>95.3</td><td>43.1</td><td>78.2</td><td>0.3</td><td>78.9</td><td>53.2</td><td>82.3</td><td>55.5</td><td>70.4</td><td>46.3</td><td>33.2</td></tr><tr><td>SalsaNet [36]</td><td>59.4</td><td></td><td>90.5 44.6</td><td>49.6</td><td>86.3</td><td>54.6</td><td>74.0</td><td>81.4</td><td>0.0</td><td>93.4</td><td>40.6</td><td>69.1</td><td>0.0</td><td>84.6</td><td>53.0</td><td>83.6</td><td>64.3</td><td>64.2</td><td>54.4</td><td>39.8</td></tr><tr><td>PMF [37]</td><td>63.9</td><td></td><td>95.4 47.8</td><td>62.9</td><td>68.4</td><td>75.2</td><td>78.9</td><td>71.6</td><td>0.0</td><td>96.4</td><td>43.5</td><td>80.5</td><td>1.0</td><td>88.7</td><td>60.1</td><td>88.6</td><td>72.7</td><td>75.3</td><td>65.5</td><td>43.0</td></tr><tr><td>RangeVit [17]</td><td>60.9</td><td>94.7</td><td>44.1</td><td>61.4</td><td>71.9</td><td>37.7</td><td>65.3</td><td>75.5</td><td>0.0</td><td>95.5</td><td>48.4</td><td>83.1</td><td>0.0</td><td>88.3</td><td>60.0</td><td>86.3</td><td>65.3</td><td>72.7</td><td>63.1</td><td>42.7</td></tr><tr><td>MSF-CSCNet [38]</td><td>65.8</td><td>96.8</td><td>50.9</td><td>72.7</td><td>90.4</td><td>61.7</td><td>76.2</td><td>90.2</td><td>0.0</td><td>94.3</td><td>44.8</td><td>81.3</td><td>0.5</td><td>90.3</td><td>59.2</td><td>86.9</td><td>66.5</td><td>71.3</td><td>65.5</td><td>51.1</td></tr><tr><td>SphereFormer [11]</td><td>67.8</td><td>96.8</td><td>50.7</td><td>75.1</td><td>93.1</td><td>65.7</td><td>76.9</td><td>92.0</td><td>0.0</td><td>94.7</td><td>53.2</td><td>82.2</td><td>3.2</td><td>90.7</td><td>58.4</td><td>88.7</td><td>71.2</td><td>75.9</td><td>64.6</td><td>54.5</td></tr><tr><td>FRNet [9]</td><td>67.6</td><td>97.2</td><td>53.3</td><td>72.9</td><td>81.5</td><td>72.9</td><td>77.2</td><td>90.8</td><td>1 3</td><td>95.9</td><td>53.7</td><td>83.9</td><td>9.0</td><td>90.5</td><td>65.9</td><td>87.0</td><td>66.8</td><td>72.6</td><td>64.0</td><td>47.9</td></tr><tr><td>WaffleIron [39]</td><td>68.0</td><td>96.1</td><td>58.1</td><td>79.7</td><td>77.4</td><td>59.0</td><td>81.1</td><td>92.2</td><td></td><td>95.5</td><td>50.2</td><td>83.6</td><td>6.0</td><td>92.1</td><td>67.5</td><td>87.8</td><td>73.8</td><td>73.0</td><td>65.7</td><td>52.2</td></tr><tr><td>FARVNet [10]</td><td>68.5</td><td>97.0</td><td>54.2</td><td>75.9</td><td>89.6</td><td>72.6</td><td>76.0</td><td>90.1</td><td>0.0</td><td>95.7</td><td>56.9</td><td>83.4</td><td>22.7</td><td>89.8</td><td>62.3</td><td>87.0</td><td>65.9</td><td>73.0</td><td>63.2</td><td>47.1</td></tr><tr><td>PC-BEV [40]</td><td>66.4</td><td>94.3</td><td>67.6</td><td>71.4</td><td>75.4</td><td>67.5</td><td>77.4</td><td>81.3</td><td>0.1</td><td>95.2</td><td>39.7</td><td>83.0</td><td>0.1</td><td>86.3</td><td>49.7</td><td>85.1</td><td>70.2</td><td>68.2</td><td>70.2</td><td>64.2</td></tr><tr><td>RangeRet [31]</td><td>65.7</td><td>93.6 96.9</td><td>47.2</td><td>65.4</td><td>89.9</td><td>69.1</td><td>73.0</td><td>83.7</td><td>0.0</td><td>94.8</td><td>54.1</td><td>81.3</td><td>15.9</td><td>87.3</td><td>59.8</td><td>86.0</td><td>64.3</td><td>72.8</td><td>60.4</td><td>50.1</td></tr><tr><td>MFINet [12]</td><td>67.7</td><td></td><td>56.0</td><td>79.0</td><td>92.4</td><td>68.0</td><td>75.5</td><td>94.4</td><td>0.0</td><td>93.6</td><td>49.6</td><td>81.3</td><td>0.0</td><td>91.0</td><td>63.8</td><td>89.1</td><td>64.3</td><td>77.2</td><td>64.7</td><td>49.6</td></tr><tr><td>VaViT [41]</td><td>67.6</td><td>95.7</td><td>56.5</td><td>74.9</td><td>95.4</td><td>52.1</td><td>81.4</td><td>94.5</td><td>0.1</td><td>95.6</td><td>51.1</td><td>83.8</td><td>0.1</td><td>90.6</td><td>61.0</td><td>88.1</td><td>72.0</td><td>73.5</td><td>65.9</td><td>51.6</td></tr><tr><td>RaSFNet</td><td>70.2</td><td>97.1</td><td>58.9</td><td>80.8</td><td>84.6</td><td>70.7</td><td>81.0</td><td>92.8</td><td>0.0</td><td>96.0</td><td>57.6</td><td>85.4</td><td>20.1</td><td>92.0</td><td>71.1</td><td>87.9</td><td>68.6</td><td>74.2</td><td>65.5</td><td>49.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE II  
CLASS-WISE IOU (%) ON THE NUSCENES VALIDATION SET [14]. BEST AND SECOND-BEST DISTINCT SCORES IN EACH COLUMN ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY; TIES SHARE THE SAME RANK.
<table><tr><td>Method</td><td></td><td>mIoU</td><td>bar.</td><td>bi.cle</td><td>bus</td><td>car</td><td>con.v.</td><td>mt.cle</td><td>ped.</td><td>t.cone</td><td>trail.</td><td>truck</td><td>dr.s.</td><td>oth.f.</td><td>sidew.</td><td>terra.</td><td>man.</td><td>veget.</td></tr><tr><td>RangeNet++ [15]</td><td></td><td>65.5</td><td>66.0</td><td>21.3</td><td>77.2</td><td>80.9</td><td>30.2</td><td>66.8</td><td>69.6</td><td>52.1</td><td>54.2</td><td>72.3</td><td>94.1</td><td>66.6</td><td>63.5</td><td>70.1</td><td>83.1</td><td>79.8</td></tr><tr><td>PolarNet [42]</td><td></td><td>71.5</td><td>74.7</td><td>28.2</td><td>85.3</td><td>90.9</td><td>35.1</td><td>77.5</td><td>71.3</td><td>58.8</td><td>57.4</td><td>76.1</td><td>96.5</td><td>71.1</td><td>74.7</td><td>74.0</td><td>87.3</td><td>85.7</td></tr><tr><td>PCSCNet [43]</td><td></td><td>72.0</td><td>73.3</td><td>42.2</td><td>87.8</td><td>86.1</td><td>44.9</td><td>82.2</td><td>76.1</td><td>62.9</td><td>49.3</td><td>77.3</td><td>95.2</td><td>66.9</td><td>69.5</td><td>72.3</td><td>83.7</td><td>82.5</td></tr><tr><td>MinkowskiNet [44]</td><td></td><td>73.5</td><td>74.5</td><td>39.8</td><td>89.7</td><td>91.0</td><td>44.3</td><td>78.7</td><td>77.3</td><td>60.4</td><td>56.7</td><td>82.6</td><td>95.5</td><td>68.8</td><td>71.3</td><td>74.4</td><td>86.1</td><td>84.9</td></tr><tr><td>SalsaNext [45]</td><td></td><td>72.0</td><td>74.8</td><td>34.1</td><td>85.9</td><td>88.4</td><td>42.2</td><td>72.4</td><td>72.2</td><td>63.1</td><td>61.3</td><td>76.5</td><td>96.0</td><td>70.8</td><td>71.2</td><td>71.5</td><td>86.7</td><td>84.4</td></tr><tr><td>CENet [46]</td><td></td><td>73.3</td><td>75.8</td><td>36.4</td><td>87.7</td><td>88.5</td><td>44.6</td><td>73.4</td><td>75.2</td><td>65.1</td><td>62.3</td><td>77.5</td><td>96.2</td><td>71.8</td><td>72.3</td><td>74.5</td><td>86.8</td><td>85.4</td></tr><tr><td>SVASeg [47]</td><td></td><td>74.7</td><td>73.1</td><td>44.5</td><td>88.4</td><td>86.6</td><td>48.2</td><td>80.5</td><td>77.7</td><td>65.6</td><td>57.5</td><td>82.1</td><td>96.5</td><td>70.5</td><td>74.7</td><td>74.6</td><td>87.3</td><td>86.9</td></tr><tr><td>RangeViT [17]</td><td></td><td>75.2</td><td>75.5</td><td>40.7</td><td>88.3</td><td>90.1</td><td>49.3</td><td>79.3</td><td>77.2</td><td>66.3</td><td>65.2</td><td>80.0</td><td>96.4</td><td>71.4</td><td>73.8</td><td>73.8</td><td>89.9</td><td>87.2</td></tr><tr><td>MSF-CSCNet [38]</td><td></td><td>75.2</td><td>75.1</td><td>40.3</td><td>91.3</td><td>88.3</td><td>45.8</td><td>78.7</td><td>78.7</td><td>64.5</td><td>64.1</td><td>80.9</td><td>96.5</td><td>73.2</td><td>75.0</td><td>75.6</td><td>88.1</td><td>87.3</td></tr><tr><td>FRNet [9]</td><td></td><td>76.1</td><td>77.2</td><td>39.5</td><td>93.4</td><td>88.6</td><td>52.1</td><td>81.4</td><td>75.1</td><td>65.7</td><td>66.2</td><td>79.7</td><td>96.8</td><td>75.3</td><td>75.4</td><td>75.9</td><td>88.4</td><td>85.4 84.0</td></tr><tr><td>WaffleIron [39] HARP-NeXt [48]</td><td></td><td>76.1 77.1</td><td>77.8</td><td>45.8</td><td>93.7</td><td>85.3</td><td>50.2</td><td>81.9</td><td>75.8</td><td>66.2</td><td>68.7</td><td>83.2</td><td>96.7</td><td>72.9</td><td>74.9</td><td>73.7</td><td>86.3</td><td>88.0</td></tr><tr><td>FLARES (FIDNet) [49]</td><td></td><td>76.6</td><td>76.2 77.6</td><td>37.5</td><td>94.2</td><td>91.1</td><td>60.8</td><td>59.5</td><td>75.9</td><td>73.7</td><td>65.0</td><td>84.8</td><td>97.7</td><td>75.9</td><td>79.4</td><td>83.3</td><td>91.0</td><td>86.8</td></tr><tr><td></td><td>RaSFNet</td><td>77.7</td><td>77.7</td><td>43.5 46.2</td><td>92.9 94.2</td><td>88.1 87.9</td><td>56.5</td><td>79.5</td><td>77.7 78.7</td><td>65.3</td><td>67.0</td><td>83.1</td><td>96.6</td><td>72.8</td><td>75.0</td><td>74.5 74.3</td><td>88.5</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>58.6</td><td>87.2</td><td></td><td>68.1</td><td>68.1</td><td>78.2</td><td>97.0</td><td>73.6</td><td>75.6</td><td></td><td>89.5</td><td>87.5</td></tr></table>

45.3% in the 0-5 m, 5-10 m, 10-20 m, 20-40 m, and 40-50 m ranges, respectively, achieving the highest mIoU among the compared methods in all five ranges. For the 50-60 m range, our method achieves an mIoU of 25.4%, close to the highest value of 26.0%. Overall, favorable segmentation performance is maintained in both high-density near-range regions and low-density middle- and far-range regions. Distance-aware modeling of image feature extraction and point-image feature interaction facilitates adaptation to feature representation variations across different sensing distances.

TABLE III  
DISTANCE-WISE SEMANTIC SEGMENTATION PERFORMANCE (MIOU, %) ON THE SEMANTICKITTI [13] VALIDATION SET UNDER DIFFERENT POINT-DENSITY LEVELS. POINT DENSITY IS COMPUTED AS THE AVERAGE NUMBER OF POINTS PER NON-EMPTY 0.5 M<sup>3</sup> VOXEL WITHIN EACH DISTANCE RANGE ACROSS ALL FRAMES.
<table><tr><td>Method Density</td><td>0-5 m 923.8</td><td>5-10 m 331.9</td><td>10-20 m 88.2</td><td>20-40 m 29.8</td><td>40-50 m 16.2</td><td>50-60 m 13.1</td></tr><tr><td>SqueezeSegV2 [34]</td><td>43.6</td><td>43.1</td><td>38.3</td><td>27.7</td><td>19.5</td><td>9.3</td></tr><tr><td>FIDNet [50]</td><td>56.0</td><td>61.3</td><td>58.5</td><td>48.4</td><td>36.0</td><td>21.1</td></tr><tr><td>RangeVit [17]</td><td>58.6</td><td>62.7</td><td>60.6</td><td>50.5</td><td>40.7</td><td>23.3</td></tr><tr><td>SegNet4D [51]</td><td>65.4</td><td>62.8</td><td>61.7</td><td>56.0</td><td>43.3</td><td>26.0</td></tr><tr><td>Cylinder3D [26]</td><td>68.4</td><td>66.6</td><td>63.8</td><td>58.7</td><td>42.2</td><td>15.2</td></tr><tr><td>WaffleIron [39]</td><td>72.8</td><td>68.7</td><td>66.1</td><td>60.3</td><td>41.7</td><td>11.6</td></tr><tr><td>FRNet [9]</td><td>73.9</td><td>68.7</td><td>64.8</td><td>55.6</td><td>41.5</td><td>24.2</td></tr><tr><td>FARVNet [10]</td><td>73.7</td><td>68.4</td><td>65.6</td><td>57.7</td><td>40.3</td><td>22.6</td></tr><tr><td>RangeRet [31]</td><td>75.0</td><td>67.9</td><td>62.1</td><td>48.8</td><td>35.7</td><td>14.2</td></tr><tr><td>Ours</td><td>77.5</td><td>71.3</td><td>65.9</td><td>59.9</td><td>45.3</td><td>25.4</td></tr></table>

## E. Qualitative Results

Maintaining spatial consistency remains challenging in Li-DAR semantic segmentation, particularly within large-scale semantic regions. Existing methods may generate locally clustered errors that spread across semantic boundaries, resulting in fragmented predictions. Fig. 5 presents qualitative comparisons in three representative scenarios: a densely vegetated area, a road-terrain boundary, and a crossroad intersection. Across these scenarios, the compared methods exhibit clustered errors within vegetation or continuous misclassifications along road boundaries and connecting branches. RaSFNet produces fewer misclassified points, with more scattered errors and more coherent predicted regions. Its design considers both local geometric differences and global spatial relationships, allowing semantic discrimination to draw on fine-grained structural details and broader contextual information.

![](images/d5d43821978f018bdaad9a4f637fd1f8d5b21293d88851cea9dbbc0837ac305e.jpg)  
(a)  
(b)  
(c)  
Fig. 5. Segmentation errors in challenging scenes on the SemanticKITTI [13] val set. Red represents misclassified areas, and gray represents correctly classified areas.

To evaluate the ability of RaSFNet to distinguish semantically different classes with similar local geometric structures, Fig. 6 presents qualitative comparisons with FRNet [9], RangeRet [31], and VaViT [41] in three challenging scenes. Incomplete observations and occlusion obscure distinctive geometric characteristics, while similar planar geometry makes terrain and sidewalk difficult to distinguish. Across these scenes, the compared methods exhibit confusion between bus and car, terrain and vegetation, or terrain and sidewalk, whereas RaSFNet more accurately identifies the highlighted regions. RaSFNet enhances subtle geometric differences through encoding and guides the collaborative updating of 3D structural features and 2D frustum features under persistent spatial constraints, thereby improving semantic discrimination between regions with similar local geometric structures.

![](images/85e13e7d1b229f774a4d31362e57f3904bd1be5da6eff3615a8f192d6280f7c4.jpg)  
Fig. 6. Qualitative comparison of semantic segmentation results on the SemanticKITTI [13] validation set.

![](images/5082ea6d99238e01908f51eb32035c7e31cd65f2b047f8e3c353b4ec11c9ba8c.jpg)  
Fig. 7. Qualitative comparison of semantic segmentation results across a temporal sequence on the SemanticKITTI [13] validation set.

Long-range semantic perception remains challenging because increasing sensing distance substantially reduces the density of LiDAR measurements. Sparse measurements provide incomplete geometric structures and limited classdiscriminative cues, increasing the likelihood of recognition errors. As shown in Fig. 7, segmentation results are compared for the same bus at progressively greater distances across consecutive frames. FRNet [9] and RangeRet [31] produce incorrect labels within the bus region, with segmentation completeness deteriorating as the sensing distance increases. By contrast, RaSFNet preserves a more complete bus region across the examined distances, demonstrating greater robustness to the feature degradation associated with sparse longrange observations.

## F. Ablation Study

Component Ablation: Table IV evaluates the contributions of RAGDE, STAM, and RCPFF through leave-onecomponent-out experiments under identical training settings. Removing RCPFF reduces mIoU by 2.7 percentage points on SemanticKITTI and 2.2 percentage points on nuScenes, highlighting the importance of effectively integrating 3D structural information with 2D frustum features. Removing STAM also degrades performance on both datasets, suggesting that axis-aware aggregation and serpentine sequence modeling help capture contextual information from the encoded geometric features. Removing RAGDE reduces both mIoU and mAcc, supporting the contribution of ray-aligned encoding to local geometric representation. These consistent trends across the two datasets support the contributions of RAGDE, STAM, and RCPFF within the overall framework. The complete configuration outperforms all ablated variants in both mIoU and mAcc on both datasets.

TABLE IV  
ABLATION STUDY OF RAGDE, STAM, AND RCPFF ON THE SEMANTICKITTI AND NUSCENES VALIDATION SETS. ALL SCORES ARE REPORTED IN PERCENTAGE (%). W/O DENOTES THE REMOVAL OF THE CORRESPONDING MODULE.
<table><tr><td rowspan="2">Variant</td><td colspan="2">SemanticKITTI</td><td colspan="2">nuScenes</td></tr><tr><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td>w/o RAGDE</td><td>69.2</td><td>75.9</td><td>77.3</td><td>84.4</td></tr><tr><td>w/o STAM</td><td>68.5</td><td>74.7</td><td>76.7</td><td>84.3</td></tr><tr><td>w/o RCPFF</td><td>67.5</td><td>74.3</td><td>75.5</td><td>82.8</td></tr><tr><td>RaSFNet</td><td>70.2</td><td>76.6</td><td>77.7</td><td>85.2</td></tr></table>

Component Designs: Table V compares segmentation performance and parameter counts when the original modules are replaced with alternative encoding, sequence modeling, and fusion methods. The evaluated replacement configurations contain 8.1M-21.7M parameters and achieve mIoU scores of 66.1%-68.6%. In comparison, RaSFNet achieves 70.2% mIoU and 76.6% mAcc with 8.0M parameters, outperforming the evaluated alternatives. These differences are associated with how geometric information is modeled: the respective replacement methods do not explicitly characterize differences in range distributions and geometric organization along sensing rays, organize feature sequences according to LiDAR azimuthal sampling patterns, or maintain a shared range constraint throughout point-frustum feature updates. RaSFNet explicitly accounts for these geometric and sampling characteristics, enabling more discriminative feature encoding and spatially constrained point-frustum interaction.

TABLE V  
MODULE REPLACEMENT STUDY ON THE SEMANTICKITTI VALIDATION SET. EACH VARIANT REPLACES ONLY THE SPECIFIED MODULE WHILE RETAINING THE REMAINING COMPONENTS. ALL SCORES ARE REPORTED AS PERCENTAGES (%).
<table><tr><td>Module</td><td>Replacement</td><td>mIoU</td><td>mAcc</td><td>Params</td></tr><tr><td rowspan="3">RAGDE</td><td>PointNet++ [25]</td><td>68.2</td><td>74.3</td><td>8.2M</td></tr><tr><td>Cylinder3D [26]</td><td>68.6</td><td>74.7</td><td>8.2M</td></tr><tr><td>MDSLCA [28]</td><td>67.8</td><td>74.6</td><td>8.1M</td></tr><tr><td rowspan="2">STAM</td><td>Mamba [24]</td><td>67.9</td><td>74.0</td><td>10.0M</td></tr><tr><td>Vision Mamba [52]</td><td>67.1</td><td>73.3</td><td>12.4M</td></tr><tr><td rowspan="2">RCPFF</td><td>Concat-MLP Fusion</td><td>66.1</td><td>72.6</td><td>21.7M</td></tr><tr><td>HARP-NeXt [48]</td><td>68.5</td><td>74.5</td><td>8.4M</td></tr><tr><td colspan="2">RaSFNet (original modules)</td><td>70.2</td><td>76.6</td><td>8.0M</td></tr></table>

Range-Image Resolution: Range-image resolution deter-

mines the angular coverage of each frustum and the distribution of projected points. Table VI compares different resolutions on both datasets. A low resolution assigns more points to each frustum, which may increase projection conflicts and obscure local structures near semantic boundaries. Increasing the resolution separates projected points more effectively, whereas an excessively fine grid creates sparsely occupied frustums and limits feature aggregation. Consequently, neither mIoU nor mAcc improves monotonically with increasing image width. Among the evaluated configurations, SemanticKITTI achieves the highest mIoU and mAcc at $6 4 \times 1 0 2 4$ , while nuScenes achieves the highest values at $3 2 \times 4 8 0$

TABLE VI  
EFFECT OF RANGE-IMAGE RESOLUTION ON THE SEMANTICKITTI AND NUSCENES VALIDATION SETS. RESOLUTION IS EXPRESSED AS $H \times W .$ ALL SCORES ARE REPORTED AS PERCENTAGES (%).
<table><tr><td colspan="3">SemanticKITTI</td><td colspan="3">nuScenes</td></tr><tr><td>Resolution</td><td>mIoU</td><td>mAcc</td><td>Resolution</td><td>mIoU</td><td>mAcc</td></tr><tr><td> $6 4 \times 2 5 6$ </td><td>65.2</td><td>72.3</td><td> $3 2 \times 2 5 6$ </td><td>75.8</td><td>83.3</td></tr><tr><td> $6 4 \times 5 1 2$ </td><td>68.9</td><td>75.4</td><td> $3 2 \times 4 8 0$ </td><td>77.7</td><td>85.2</td></tr><tr><td> $6 4 \times 7 6 8$ </td><td>68.3</td><td>74.8</td><td> $3 2 \times 5 1 2$ </td><td>77.3</td><td>84.4</td></tr><tr><td> $6 4 \times 1 0 2 4$ </td><td>70.2</td><td>76.6</td><td> $3 2 \times 7 6 8$ </td><td>77.5</td><td>84.7</td></tr><tr><td> $6 4 \times 1 2 8 0$ </td><td>68.7</td><td>75.8</td><td> $3 2 \times 1 0 2 4$ </td><td>77.3</td><td>84.5</td></tr></table>

## V. CONCLUSION

This paper presented RaSFNet for projection-based LiDAR semantic segmentation. Guided by the physical acquisition characteristics of LiDAR, RaSFNet amplifies subtle differences among similar local point distributions through rayaligned encoding, serializes frustum features in a serpentine row-wise order along the azimuthal dimension, and uses range information shared by point and frustum representations to constrain their bidirectional updates. Across the evaluated datasets, RaSFNet produces more coherent semantic regions and clearer class boundaries, with less confusion between spatially adjacent classes that share similar geometric structures. Class-wise evaluations and ablation studies provide further analyses of the segmentation performance and the roles of the main components. Future work will explore extending RaSFNet to other 3D perception tasks, including 3D object detection and semantic occupancy prediction.

## REFERENCES

[1] H. Yan, A. Lau, and H. Fan, “Evaluating deep learning advances for point cloud semantic segmentation in urban environments,” KN-Journal of Cartography and Geographic Information, vol. 75, no. 1, pp. 3–22, 2025.

[2] Y. Ding, D. Biswas, K. Liang, K. Li, R. Qin, and X. Hu, “From clear to adverse: a survey of all-weather lidar semantic segmentation for autonomous driving,” Artificial Intelligence for Transportation, vol. 6, p. 100061, 2026.

[3] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 652–660.

[4] Q. Hu, B. Yang, L. Xie, S. Rosa, Y. Guo, Z. Wang, N. Trigoni, and A. Markham, “Randla-net: Efficient semantic segmentation of largescale point clouds,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 11 105–11 114.

[5] Y. Wang, W. Zhao, C. Cao, T. Deng, J. Wang, and W. Chen, “Sfpnet: Sparse focal point network for semantic segmentation on general lidar point clouds,” in European Conference on Computer Vision. Springer, 2024, pp. 403–421.

[6] R. Abbasi, A. K. Bashir, A. Rehman, and Y. Ge, “3d lidar point cloud segmentation for automated driving,” IEEE Intelligent Transportation Systems Magazine, vol. 17, no. 1, pp. 8–29, 2024.

[7] L. Kong, Y. Liu, R. Chen, Y. Ma, X. Zhu, Y. Li, Y. Hou, Y. Qiao, and Z. Liu, “Rethinking range view representation for lidar segmentation,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 228–240.

[8] B. Chen, C. Gong, and J. Röning, “Filling missing values matters for range image-based point cloud segmentation,” IEEE Transactions on Intelligent Vehicles, 2024.

[9] X. Xu, L. Kong, H. Shuai, and Q. Liu, “Frnet: Frustum-range networks for scalable lidar segmentation,” IEEE Transactions on Image Processing, vol. 34, pp. 2173–2186, 2025.

[10] C. Chen, L. Zhao, W. Guo, X. Yuan, S. Tan, J. Hu, Z. Yang, S. Wang, and W. Ge, “Farvnet: A fast and accurate range-view-based method for semantic segmentation of point clouds,” Sensors, vol. 25, no. 9, 2025.

[11] X. Lai, Y. Chen, F. Lu, J. Liu, and J. Jia, “Spherical transformer for lidar-based 3d recognition,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 17 545– 17 555.

[12] N. Ma, Z. Liu, and Y. Han, “Mfinet: Multi-view fusion and 2d–3d interaction enhancement for real-time lidar semantic segmentation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 10, 2026, pp. 7800–7808.

[13] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall, “Semantickitti: A dataset for semantic scene understanding of lidar sequences,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 9297–9307.

[14] W. K. Fong, R. Mohan, J. V. Hurtado, L. Zhou, H. Caesar, O. Beijbom, and A. Valada, “Panoptic nuscenes: A large-scale benchmark for lidar panoptic segmentation and tracking,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 3795–3802, 2022.

[15] A. Milioto, I. Vizzo, J. Behley, and C. Stachniss, “Rangenet++: Fast and accurate lidar semantic segmentation,” in 2019 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2019, pp. 4213–4220.

[16] T. Cortinhal, G. Tzelepis, and E. Erdal Aksoy, “Salsanext: Fast, uncertainty-aware semantic segmentation of lidar point clouds,” in International Symposium on Visual Computing. Springer, 2020, pp. 207–222.

[17] A. Ando, S. Gidaris, A. Bursuc, G. Puy, A. Boulch, and R. Marlet, “Rangevit: Towards vision transformers for 3d semantic segmentation in autonomous driving,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 5240–5250.

[18] X. Li, G. Zhang, T. Jiang, X. Cai, and Z. Wang, “Prnet: Point-range fusion network for real-time lidar semantic segmentation.” in IJCAI, 2022, pp. 1116–1122.

[19] J. Xu, R. Zhang, J. Dou, Y. Zhu, J. Sun, and S. Pu, “Rpvnet: A deep and efficient range-point-voxel fusion network for lidar point cloud segmentation,” in 2021 IEEE/CVF international conference on computer vision (ICCV). IEEE, 2021, pp. 16 004–16 013.

[20] C. Chen, L. Zhao, W. Guo, X. Yuan, S. Tan, J. Hu, Z. Yang, S. Wang, and W. Ge, “Farvnet: A fast and accurate range-view-based method for semantic segmentation of point clouds,” Sensors, vol. 25, no. 9, p. 2697, 2025.

[21] A. Xiao, J. Huang, D. Guan, K. Cui, S. Lu, and L. Shao, “Polarmix: A general data augmentation technique for lidar point clouds,” Advances in Neural Information Processing Systems, vol. 35, pp. 11 035–11 048, 2022.

[22] B. Chen, C. Gong, A. Tikanmäki, and J. Röning, “Copy-rotate-paste augmentation for point cloud segmentation,” IEEE Signal Processing Letters, vol. 32, pp. 2209–2213, 2025.

[23] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. Barron, and R. Ng, “Fourier features let networks learn high frequency functions in low dimensional domains,” Advances in neural information processing systems, vol. 33, pp. 7537– 7547, 2020.

[24] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in First Conference on Language Modeling, 2024. [Online]. Available: https://openreview.net/forum?id=tEYskw1VY2

[25] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” Advances in neural information processing systems, vol. 30, 2017.

[26] X. Zhu, H. Zhou, T. Wang, F. Hong, Y. Ma, W. Li, H. Li, and D. Lin, “Cylindrical and asymmetrical 3d convolution networks for lidar segmentation,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 9939–9948.

[27] L. An, P. Zhou, M. Zhou, Y. Wang, G. Geng, and Y. Liu, “Pointgeo: geometry transformer for point cloud analysis,” CAAI Transactions on Intelligence Technology, vol. 10, no. 6, pp. 1880–1892, 2025.

[28] J. Guang, Q. Zhang, and J. Liu, “Mdslca: Multi-scale dilated spatial and local channel attention for lidar point cloud semantic segmentation,” IEEE Transactions on Circuits and Systems for Video Technology, 2026.

[29] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[30] S. Li, X. Chen, Y. Liu, D. Dai, C. Stachniss, and J. Gall, “Multiscale interaction for real-time lidar data segmentation on an embedded platform,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 738–745, 2021.

[31] S. Mosco, D. Fusaro, W. Li, and A. Pretto, “Revisiting retentive networks for fast range-view 3d lidar semantic segmentation,” in 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2026, pp. 2499–2509.

[32] S. Abou Haidar, A. Chariot, M. Darouich, C. Joly, and J.-E. Deschaud, “Harp-next: high-speed and accurate range-point fusion network for 3d lidar semantic segmentation,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 5946– 5953.

[33] C. Scribano, M. Mahdi, N. Prisadnikov, Y. Fu, G. Franchini, D. P. Paudel, M. Bertogna, and L. Van Gool, “Accelerating vision foundation models with drop-in depthwise convolution,” in International Conference on Pattern Recognition. Springer, 2026, pp. 463–477.

[34] B. Wu, X. Zhou, S. Zhao, X. Yue, and K. Keutzer, “Squeezesegv2: Improved model structure and unsupervised domain adaptation for roadobject segmentation from a lidar point cloud,” in 2019 international conference on robotics and automation (ICRA). IEEE, 2019, pp. 4376– 4382.

[35] C. Xu, B. Wu, Z. Wang, W. Zhan, P. Vajda, K. Keutzer, and M. Tomizuka, “Squeezesegv3: Spatially-adaptive convolution for efficient point-cloud segmentation,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVIII 16. Springer, 2020, pp. 1–19.

[36] E. E. Aksoy, S. Baci, and S. Cavdar, “Salsanet: Fast road and vehicle segmentation in lidar point clouds for autonomous driving,” in 2020 IEEE intelligent vehicles symposium (IV). IEEE, 2020, pp. 926–932.

[37] Z. Zhuang, R. Li, K. Jia, Q. Wang, Y. Li, and M. Tan, “Perception-aware multi-sensor fusion for 3d lidar semantic segmentation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 16 280–16 290.

[38] Y. Bai, Y. Gong, J. Wang, and F. Wei, “Msf-cscnet: a supercomputingready 3d semantic segmentation network for urban point clouds via multi-scale fusion and context-aware channel modeling.” The Journal of Supercomputing, vol. 81, no. 11, p. 1200, 2025.

[39] G. Puy, A. Boulch, and R. Marlet, “Using a waffle iron for automotive point cloud semantic segmentation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3379–3389.

[40] S. Qiu, X. Li, X. Xue, and J. Pu, “Pc-bev: An efficient polar-cartesian bev fusion framework for lidar semantic segmentation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 6, 2025, pp. 6612–6620.

[41] G. Puy, N. Samet, A. Boulch, S. Gidaris, T.-H. VU, and R. Marlet, “Vanilla vit for automotive point cloud semantic segmentation,” arXiv preprint arXiv:2605.31177, 2026.

[42] Y. Zhang, Z. Zhou, P. David, X. Yue, Z. Xi, B. Gong, and H. Foroosh, “Polarnet: An improved grid representation for online lidar point clouds semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[43] J. Park, C. Kim, S. Kim, and K. Jo, “Pcscnet: Fast 3d semantic segmentation of lidar point cloud for autonomous car using point convolution and sparse convolution network,” Expert Systems with Applications, vol. 212, p. 118815, 2023.

[44] C. Choy, J. Gwak, and S. Savarese, “4d spatio-temporal convnets: Minkowski convolutional neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 3075–3084.

[45] T. Cortinhal, G. Tzelepis, and E. E. Aksoy, “Salsanext: Fast, uncertaintyaware semantic segmentation of lidar point clouds,” in Advances in Visual Computing. Springer, 2020.

[46] H.-X. Cheng, X.-F. Han, and G.-Q. Xiao, “CENet: Toward concise and efficient LiDAR semantic segmentation for autonomous driving,” in 2022 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2022, pp. 1–6.

[47] L. Zhao, S. Xu, L. Liu, D. Ming, and W. Tao, “Svaseg: Sparse voxelbased attention for 3d lidar point cloud semantic segmentation,” Remote Sensing, vol. 14, no. 18, p. 4471, 2022.

[48] S. Abou Haidar, A. Chariot, M. Darouich, C. Joly, and J.-E. Deschaud, “Harp-next: High-speed and accurate range-point fusion network for 3d lidar semantic segmentation,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2025, pp. 5946–5953.

[49] B. Yang and A. P. Condurache, “Flares: Fast and accurate lidar multirange semantic segmentation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2026, pp. 3451–3461.

[50] Y. Zhao, L. Bai, and X. Huang, “Fidnet: Lidar point cloud semantic segmentation with fully interpolation decoding,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 4453–4458.

[51] N. Wang, R. Guo, C. Shi, Z. Wang, H. Zhang, H. Lu, Z. Zheng, and X. Chen, “Segnet4d: Efficient instance-aware 4d semantic segmentation for lidar point cloud,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 15 339–15 350, 2025.

[52] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, “Vision mamba: efficient visual representation learning with bidirectional state space model,” in Proceedings of the 41st International Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.