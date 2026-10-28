# SAGE: Semantic-Augmented Geometric Experts for Point Cloud Registration

Tomoyasu Shimada Kazuhiko Murasaki Shoichiro Takeda Shogo Sato Riku Inoue Toshihiko Nishimura Ryuichi Tanida NTT, Inc.

## Abstract

Point cloud registration is an essential taskfor scene reconstruction, requiring the alignment ofpartially overlapping point clouds. Under low-overlap conditions, this task remains challenging because existing geometry-only methods cannot distinguish geometrically similar but semantically distinct structures (e.g., TV corners vs. desk corners and chair legs vs. table legs), resulting in incorrect correspondences. To address this geometric ambiguity, we present Semantic-Augmented Geometric Experts (SAGE), which integratesfrozen pre-trainedpoint cloud semantic representations into the registration pipeline without requiring additional sensors orfine-tuning the semantic encoder. To effectively leverage these semantic distinctions, SAGE employs a Semantic-Guided Mixture of Experts (SG-MoE) architecture. Through semantic-aware routing, SG-MoE enables each expert to specialize in distinct semantic-geometric patterns, reducing mismatches that confound geometry-only methods. With a structure-aware estimator, SAGE achieves the highest registration recall on 3DLoMatch while remaining competitive on 3DMatch. When evaluated with the same number ofrefinement iterations, SAGEfurther achieves the highest inlier ratio and registration recall on both benchmarks. These results indicate that frozen semantic representations provide effective cuesfor resolving geometric ambiguity in low-overlap registration.

## 1. Introduction

Point cloud registration is an essential task for 3D perception in robotics, autonomous driving, and augmented reality [18]. Despite progress, achieving robust registration under challenging conditions with low overlap remains an open problem. Current registration methods [17, 19, 26, 48, 50] generally follow a correspondence-based pipeline. This pipeline first establishes point-to-point correspondences between two point clouds using deep neural networks, and then estimates the transformation from these correspondences. Thus, this pipeline’s success depends on the quality of the correspondences. However, geometrically similar structures such as TV edges, desk edges, and furniture edges produce similar geometric features even though they differ semantically. In low-overlap scenarios where spatial context is severely limited, correspondences relying solely on geometric features become unreliable because there are insufficient points surrounding the target point to distinguish between semantically different features.

![](images/815fde29d1e1cf36bcf9ff42948a1f709f6a281b245d3a12be062c26702115b8.jpg)  
Figure 1. Limitation of geometric features in low-overlap point cloud registration. Point clouds are pseudo-colored by semantic regions for visualization. Top: GeoTransformer [26], relying solely on geometric features, is overwhelmed by dominant incorrect correspondences (red lines) due to similar geometric structures, failing to establish true inliers (no green lines visible). Bottom: SAGE (Ours) successfully disambiguates geometrically similar but semantically distinct structures. Green and red lines indicate inlier and outlier correspondences (threshold: 0.1 m).

Fig. 1 illustrates this fundamental limitation: widely adopted methods such as GeoTransformer [26] can produce entirely incorrect correspondences when faced with geometrically similar but semantically distinct structures in low-overlap scenarios.

Indeed, geometry-based methods [17, 48–50] have such limitations. Even recent Mixture of Experts (MoE) methods like PSReg [19], which use geometric feature-based routing to activate geometry-specialized experts, depend only on geometric representations. Although post-processing methods such as DecisionPCR [56] can boost results through iterative refinement, they still operate within the geometric domain, limited by the quality ceiling of geometry-based correspondence establishment. As one possible approach to overcome this limitation, several methods use semantic information for registration. However, they often require RGB-D sensors or calibrated cameras, while others rely on domain-specific models that generalize poorly across datasets [15, 22, 25, 45].

To overcome the fundamental limitations of relying solely on geometric features, we introduce Semantic-Augmented Geometric Experts (SAGE). SAGE integrates frozen pre-trained semantic representations into point cloud registration through a semantic-guided mixture-of-experts architecture that jointly exploits geometric and semantic features. This design requires neither additional sensors nor fine-tuning of the semantic encoder. First, SAGE employs the frozen semantic feature extractor of the recent pretrained model Sonata [43] to extract semantic features of the input source and target point clouds. Then, SAGE jointly encodes the semantic features with input point clouds of the source and target to extract their corresponding semanticgeometric features. After this feature extraction, SAGE refines the semantic-geometric features via our proposed Semantic-Guided MoE (SG-MoE).

Our main technical contribution is the design of SG-MoE. If Standard MoE [13, 31] is used for feature refinement, as in PSReg [19], geometric knowledge becomes dominant and semantic knowledge is neglected in the semantic-geometric features because the loss functions commonly adopted in registration tasks primarily optimize geometric fidelity. To prevent this domination, we design the expert routing in SG-MoE to be based solely on semantic features extracted from the frozen Sonata. This design enables each expert to refine the semantic-geometric features by explicitly considering semantic knowledge. As a result, SG-MoE performs the feature refinement process while preserving both semantic and geometric information.

Finally, SAGE estimates the transformation between the source and target point clouds using their refined semanticgeometric features following the superpoint matching in GeoTransformer [26]. As demonstrated in Fig. 1, while geometry-only methods like GeoTransformer fail to distinguish geometrically similar structures (producing incorrect correspondences shown in red), our SAGE successfully distinguishes them by leveraging semantic knowledge, achieving accurate registration even in low-overlap scenarios. Our contributions are:

• We propose SAGE, which integrates frozen pre-trained semantic representations into point cloud registration through a two-stage feature extraction without additional sensors or retraining, overcoming the limitations of purely geometric features.

• We design Semantic-Guided MoE (SG-MoE), where routing is determined solely by semantic features while experts process semantic-geometric features, enabling the distinction of geometrically similar but semantically different structures.

• Evaluation on 3DMatch [54], 3DLoMatch [17], and KITTI [16]<sup>1</sup> demonstrates that SAGE achieves the highest registration recall on 3DLoMatch among single-pass methods with a structure-aware estimator [26], the best feature matching recall across all sample counts and the best inlier ratio with 1000 or fewer samples under RANSAC, and the lowest translation error on KITTI, while performing on par with iterative methods [19, 51].

## 2. Related Work

Classical point cloud registration. Traditional methods laid the foundation for automatic point cloud alignment through coarse-to-fine pipelines [18]. These methods rely on handcrafted geometric descriptors [20, 29, 38] to encode local geometry, establish correspondences using RANdom SAmple Consensus (RANSAC) variants [1, 5, 10, 14], and refine alignment through Iterative Closest Point (ICP) [6, 9, 24, 28]. These methods successfully automated fundamental registration tasks and introduced the coarse-to-fine pipeline that remains widely adopted. Unfortunately, because handcrafted descriptors are not learned from data, these methods are sensitive to noise and partial observations, resulting in limited robustness.

Learning-based point cloud registration. Deep learning addressed the discriminative power limitations of handcrafted descriptors by introducing learnable feature representations. Pioneering works such as 3DMatch [54] for RGB-D scenes and FCGF [11] with sparse convolutions demonstrated improvements in both feature discriminability and robustness to noise and partial observations by learning local geometric features from large-scale datasets [3, 4, 12, 47]. However, a new challenge emerged in low-overlap scenarios where limited spatial context makes correspondence establishment particularly difficult. Unlike partial observations (which concern the quality and completeness of individual point clouds), low-overlap refers to the limited shared observable region between two point clouds. Even when each scan is individually complete, establishing reliable correspondences becomes difficult when the common area is small, as there is insufficient spatial context to disambiguate similar structures. Despite improved feature quality, methods relying on local geometric features alone struggle in such scenarios [18]. This is because they cannot leverage global scene structure to resolve ambiguities when similar local geometric patterns appear at multiple locations.

![](images/03df3f2a1f525c3d67658295c2f5d77d9a942bad82a86b091239ee10d7415d65.jpg)  
Figure 2. Overall pipeline of SAGE. Given two input point clouds, the Two-Stage Feature Extraction (Sec. 3.2) first applies a frozen Sonata encoder to extract point-level semantic features from each point cloud, and then feeds these semantic features together with the raw point coordinates into the KPConv encoder to produce semantic-geometric features. The Spatial Feature Aggregation (Sec. 3.3) then maps both the semantic and semantic-geometric features to the superpoint level via nearest-neighbor search and average pooling. The aggregated features are refined through N alternating iterations of Geometric Transformer layers (red box, self-attention) and Semantic-Guided Mixture of Experts (Sec. 3.4) layers (blue box, cross-attention), where semantic features guide expert routing to enable semantic-aware specialization. Finally, the framework adopts GeoTransformer’s coarse-to-fine matching for correspondence estimation and registration.

Global context modeling with transformers. Transformers addressed the locality limitation by enabling global context modeling through self-attention mechanisms [17, 40, 49]. GeoTransformer [26] pioneered this direction with geometric transformers and superpoint matching. Subsequent work explored end-to-end correspondence prediction [48], hierarchical modeling, diffusion-based approaches, multimodal fusion, and various architectural improvements [2, 8, 17, 22, 32–34, 39, 51, 56]. By incorporating global geometric relationships, these methods improved performance in low-overlap scenarios where spatial context is limited. However, we observe that purely geometric features struggle to disambiguate structures with similar geometric patterns but different semantic meanings.

Representation learning beyond geometry. Recent advances in 3D representation learning [41, 42, 44] have demonstrated the potential to learn semantic representations beyond pure geometry, offering a promising direction for addressing geometric ambiguity. Methods such as Point-BERT [53], masked autoencoders, and auto-regressive methods [7, 23] capture semantic patterns through pretraining with masked point modeling. Notably, Sonata [43] learns robust semantic representations by explicitly avoiding geometric shortcuts during training using a large dataset. This makes it possible to semantically distinguish between objects that have similar geometric characteristics,

such as walls and floors.

Semantic-enhanced registration. Several methods leverage semantic information for registration. S2Reg [45] acquires semantics via pre-trained 3D models or 2D-3D projection from images, while ML-SemReg [46] relies on semantic labels produced by external semantic segmentation models. ColorPCR [22], SemReg [15], and SARNet [25] employ multi-modal fusion or semantic constraints. However, these approaches either require additional modalities such as RGB-D data or calibrated images, or rely on explicit semantic labels produced by task-specific segmentation models. While these methods demonstrate the value of semantic information, directly integrating frozen pretrained semantic representations into point cloud registration without additional sensors or retraining remains unexplored. To address this, we propose an architecture that leverages frozen semantic representations solely from point clouds.

Mixture of experts. Mixture of Experts (MoE) offers an architectural approach to handle diverse input patterns efficiently. Specifically, this approach has a set of small modules, called experts, and each expert specializes in different input patterns [13, 31, 55]. PSReg [19] pioneered the application of MoE to registration. The method leverages GeoTransformer [26] to determine which regions overlap between point clouds, and uses this overlap information for routing inputs to experts. By enabling different experts to specialize in different geometric patterns (e.g., planar vs. curved surfaces, indoor vs. outdoor structures),

MoE provides adaptive processing capability. However, routing solely on geometric features fundamentally limits expert specialization; while experts can handle different geometric patterns, they cannot distinguish between semantically different structures that exhibit similar geometry. Our method addresses this limitation by incorporating pre-trained semantic representations into MoE routing, enabling experts to specialize according to both geometric and semantic characteristics.

## 3. Method

## 3.1. Problem Formulation

Given two partially overlapping point clouds, $\begin{array} { r } { \mathcal { P } = \{ { \bf p } _ { i } \} _ { i = 1 } ^ { N _ { \mathcal { P } } } } \end{array}$ and $\mathcal { Q } = \{ \mathbf { q } _ { j } \} _ { j = 1 } ^ { N _ { Q } }$ , where $\mathbf { p } _ { i } , \mathbf { q } _ { j } \in \mathbb { R } ^ { 3 }$ represent 3D point positions, the goal is to estimate the rigid transformation $\mathbf { T } = \left( \mathbf { R } , \mathbf { t } \right)$ that aligns Q to $\mathcal { P } _ { \cdot }$ , where $\mathbf { R } \in S O ( 3 )$ is the rotation in ${ \dot { \mathbb { R } } } ^ { 3 }$ and $\mathbf { t } \in \mathbb { R } ^ { 3 }$ is the translation.

As shown in Fig. 2, our proposed pipeline is based on GeoTransformer [26] but incorporates a newly developed semantic representation extractor and an SG-MoE Transformer. The pipeline proceeds in five stages: (1) semantic representation extraction using a frozen Sonata encoder in Sec. 3.2, (2) geometric feature extraction with semantic representations as input to KPConv [37] in Sec. 3.2, (3) spatial aggregation of semantic representations to the superpoint level in Sec. 3.3, (4) alternating geometric self-attention and semantic-guided MoE cross-attention for N iterations in Sec. 3.4, and (5) correspondence establishment and registration [26].

## 3.2. Two-Stage Feature Extraction

To efficiently extract semantic-geometric features from the inputs $\mathcal { P }$ and $\mathcal { Q } ,$ we adopt a two-stage strategy. In the first stage, we employ Sonata [43] to extract point-level semantic representations $\mathbf { F } _ { \mathrm { ~ s ~ } } ^ { \mathcal { P } } \in \mathrm { ~ \mathbb { R } ^ { \mathcal { N } _ { P } \times d _ { s } } ~ }$ from the input point cloud $\mathcal { P } _ { \cdot }$ , yielding $\mathbf { F } _ { s } ^ { \mathcal { \hat { P } } } = \operatorname { S o n a t a } ( \mathcal { P } )$ . Similarly, we obtain $\mathbf { F } _ { s } ^ { \mathcal { Q } } \in \mathbb { R } ^ { N _ { \mathcal { Q } } \times d _ { s } }$ from $\mathcal { Q }$ in the same manner. In the second stage, we use a trainable KPConv encoder [37] to obtain downsampled points $\hat { \mathcal { P } } = \{ \hat { \mathbf { p } } _ { i } \} _ { i = 1 } ^ { \hat { N } _ { \mathcal { P } } } ( \hat { N } _ { \mathcal { P } } \ll N _ { \mathcal { P } } )$ and their corresponding semantic-geometric features $\hat { \mathbf { F } } _ { s g } ^ { \mathcal { P } } \in$ $\mathbb { R } ^ { \hat { N } _ { \mathcal { P } } \times d _ { s g } }$ as $\big ( \hat { \mathcal { P } } , \hat { \mathbf { F } } _ { s g } ^ { \mathcal { P } } \big ) = \operatorname { K P C o n v } \big ( \mathcal { P } , \mathbf { F } _ { s } ^ { \mathcal { P } } \big )$ . Similarly, we get $\left( \hat { \mathcal { Q } } , \hat { \mathbf { F } } _ { s g } ^ { \mathcal { Q } } \right) = \operatorname { K P C o n v } \left( \mathcal { Q } , \mathbf { F } _ { s } ^ { \mathcal { Q } } \right)$

This two-stage design achieves semantic-geometric feature fusion by leveraging KPConv’s ability to learn conditional convolutions. By providing semantic representations $\mathbf { F } _ { s } ^ { \mathcal { P } }$ and $\mathbf { F } _ { s } ^ { \mathcal { Q } }$ as initial features, we enable KPConv’s kernel weights to adapt to the joint semantic-geometric space. The resulting features $\hat { \mathbf { F } } _ { s g } ^ { \mathcal { P } }$ and $\hat { \mathbf { F } } _ { s g } ^ { \mathcal { Q } }$ encode semantic-aware geometric patterns that can distinguish geometrically similar structures based on their semantic meaning.

## 3.3. Spatial Feature Aggregation

Following GeoTransformer [26]’s coarse-to-fine strategy, we perform transformer processing iteratively to refine the extracted features above $\mathbf { \bar { F } } _ { s } ^ { \mathcal { P } }$ and $\tilde { \mathbf { F } } _ { s g } ^ { \mathcal { P } }$ . To process these features with the transformer architecture, we aggregate $\mathbf { F } _ { s } ^ { \mathcal { P } }$ with $N _ { \mathcal { P } }$ points to match the $\hat { N } _ { \mathcal { P } }$ points of $\hat { \mathbf { F } } _ { s g } ^ { \mathcal { P } }$

For each point $\hat { \mathbf { p } } _ { k } \in \hat { \mathcal { P } }$ , we obtain its downsampled semantic representation $\hat { \mathbf { f } } _ { s , k } ^ { \mathcal { P } }$ by average-pooling the features of its $K$ nearest neighbors $( K = 1 6 )$ in the original point cloud $\begin{array} { r } { \mathcal { P } \left( \mathrm { i . e . , } \hat { \mathbf { f } } _ { s , k } ^ { \mathcal { P } } = \frac { 1 } { K } \sum _ { i \in \mathcal { I } _ { k } } \mathbf { f } _ { s , i } ^ { \mathcal { P } } \right) } \end{array}$ . This yields the downsampled representations $\hat { \mathbf { F } } _ { s } ^ { \mathcal { P } }$ with $\hat { N } _ { \mathcal { P } }$ points, matching the spatial dimension of $\hat { \mathbf { F } } _ { s g } ^ { \mathcal { P } } .$ . The same aggregation is applied to $\mathcal { Q }$ to obtain $\hat { \mathbf { F } } _ { s } ^ { \mathcal { Q } }$

## 3.4. Semantic-Guided Mixture of Experts

After the above aggregation, we perform the transformerbased refinement process iteratively for the semanticgeometric features $\hat { \mathbf { F } } _ { s g } ^ { \mathcal { P } }$ and $\hat { \mathbf { F } } _ { s g } ^ { \mathcal { Q } }$ . In this process, we use a new MoE module called Semantic-Guided Mixture of Experts (SG-MoE). Following GeoTransformer [26], we refine semantic-geometric features through N iterations of transformer blocks, where each block consists of a Geometric Transformer layer (self-attention) and an SG-MoE Transformer layer (cross-attention), followed by superpoint matching for correspondence establishment, a coarse-tofine strategy that has proven effective in recent transformerbased registration methods [17, 49, 51]. Geometric Transformer processes each point cloud independently with geometric positional encodings, while SG-MoE refines features through cross-attention with semantic-guided expert routing.

Although PSReg [19] pioneered MoE for registration, its geometry-only routing cannot distinguish regions with similar geometry but different semantics. Moreover, if we directly use semantic-geometric features $\hat { \mathbf { F } } _ { s g } ^ { \mathcal { P } }$ for routing, registration losses that optimize for geometric alignment cause these representations to collapse into purely geometric patterns, losing their semantic discriminability. To overcome this issue, we design our SG-MoE to route experts using superpoint-level semantic representations $\hat { \mathbf { F } } _ { s } ^ { \mathcal { P } }$ only.

As shown in Fig. 3, SG-MoE consists of (i) a crossattention layer, (ii) a semantic-guided router, and (iii) expert feed-forward networks (FFNs). Unlike Standard MoE (Fig. 3a) whose fused-feature routing suffers from geometric domination, our SG-MoE (Fig. 3b) strictly routes using pure semantic features. This design preserves semantic distinctions while experts refine the semantic-geometric features.

The refinement process in SG-MoE operates as follows. First, a cross-attention layer updates semantic-geometric features:

$$
\hat { \mathbf { F } } _ { \mathrm { a t t e n t i o n } } ^ { \mathcal { P } } = \mathrm { L a y e r N o r m } \left( \hat { \mathbf { F } } _ { s g } ^ { \mathcal { P } } + \mathrm { C r o s s A t t n } ( \hat { \mathbf { F } } _ { s g } ^ { \mathcal { P } } , \hat { \mathbf { F } } _ { s g } ^ { \mathcal { Q } } ) \right) ,\tag{1}
$$

![](images/e1c0acba7d8f8e43e0ffc95c33684380e17f1788c1d54e62025699e6c845feb5.jpg)  
(a) Standard MoE

![](images/dbf975f8ef587610a0d180a0a6db90bb76f1e91750dc05b1bb304218c0710459.jpg)  
Figure 3. Comparison of MoE architectures. (a) Standard MoE feeds the fused semantic-geometric features into both the router and the experts. (b) Our SG-MoE decouples the inputs, feeding pure semantic features to the router and semantic-geometric features to the experts. These processes apply symmetrically with P and Q swapped.

Similarly, we get $\hat { \mathbf { F } } _ { \mathrm { a t t e n t i o n } } ^ { \mathcal { Q } }$ from $\hat { \mathbf { F } } _ { s g } ^ { \mathcal { Q } }$ and $\hat { \mathbf { F } } _ { s q } ^ { \mathcal { P } }$ .

In a semantic-guided expert router network, expert routing is guided by frozen semantic representations. For each superpoint’s semantic representation $\hat { \mathbf { f } } _ { s , k } ^ { \mathcal { P } } \in \mathbb { R } ^ { d _ { s } }$ , a lightweight two-layer routing network computes expert selection probabilities:

$$
\mathbf { r } _ { k } = \mathrm { S o f t m a x } ( \pi ( \widehat { \mathbf { f } } _ { s , k } ^ { \mathcal { P } } ) ) \in \mathbb { R } ^ { N _ { \mathrm { e x p e r t } } }\tag{2}
$$

where π is a two-layer MLP with ReLU activation. Here, $\mathbf { r } _ { k } = [ r _ { k , 1 } , . . . , r _ { k , N _ { \mathrm { e x p e r t } } } ]$ represents the probability distribution over $N _ { \mathrm { e x p e r t } }$ experts for superpoint k, where $r _ { k , i }$ indicates the probability of assigning superpoint k to expert i. We employ a top-1 routing strategy where each superpoint is assigned to the expert with the highest probability, i.e., $i _ { k } ^ { * } = \arg \operatorname* { m a x } _ { 1 \leq i \leq N _ { \mathrm { e x p e r t } } } r _ { k , i } .$

For each point $\hat { \bf p } _ { k } ,$ let $\hat { \mathbf { f } } _ { \mathrm { a t t e n t i o n } , k } ^ { \mathcal { P } } \in \mathbb { R } ^ { d _ { s _ { g } } }$ denote the k-th transposed row of $\hat { \mathbf { F } } _ { \mathrm { a t t e n t i o n } } ^ { \mathcal { P } } .$ . The semantic-geometric features from cross-attention are processed by the assigned expert:

$$
\hat { \mathbf { f } } _ { \mathrm { o u t } , k } ^ { \mathcal { P } } = \mathrm { L a y e r N o r m } \left( \hat { \mathbf { f } } _ { \mathrm { a t t e n t i o n } , k } ^ { \mathcal { P } } + \psi _ { i _ { k } ^ { * } } ( \hat { \mathbf { f } } _ { \mathrm { a t t e n t i o n } , k } ^ { \mathcal { P } } ) \right) ,\tag{3}
$$

where $\psi _ { i _ { k } ^ { * } }$ is a two-layer feed-forward network with ReLU activation.

The router is regularized by the load-balancing objective, while the selected experts are optimized by the registration losses. Because routing is determined from frozen semantic features, different experts receive different subsets of semantic-geometric features and can consequently learn distinct refinement functions. This semantic-conditioned expert specialization helps reduce ambiguities that have been reported as a limitation of geometry-only registration methods [26, 50].

Following GeoTransformer [26], after N transformer block iterations, the refined features are used to establish correspondences through superpoint matching and point matching, followed by transformation estimation.

## 3.5. Training Objective

We train SAGE end-to-end using:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { c o a r s e } } \mathcal { L } _ { \mathrm { c o a r s e } } + \lambda _ { \mathrm { f i n e } } \mathcal { L } _ { \mathrm { f i n e } } + \lambda _ { \mathrm { m o e } } \mathcal { L } _ { \mathrm { m o e } } .\tag{4}
$$

The first two terms supervise correspondence quality at different spatial resolutions, while the third term regularizes expert utilization in the SG-MoE layer. We describe each term below. Note that the Sonata encoder remains frozen throughout training. Its semantic representations are never updated by gradients, preserving pre-trained knowledge. Therefore, the KPConv encoder, transformer layers, routing networks, and matching heads are trained by the loss (4).

Coarse-level supervision loss $( \mathcal { L } _ { \mathrm { c o a r s e } } )$ We adopt the overlap-aware circle loss from GeoTransformer [26] to supervise superpoint correspondences. The loss is computed bidirectionally as $\mathcal { L } _ { \mathrm { c o a r s e } } \overset { \cdot } { = } \mathcal { L } _ { \mathrm { o c } } ^ { \mathcal { P }  \mathcal { Q } } + \mathcal { L } _ { \mathrm { o c } } ^ { \mathcal { Q }  \mathcal { P } }$ , where ${ \mathcal { L } } _ { \mathrm { o c } }$ denotes this loss computed in each direction<sup>2</sup>.

Fine-level supervision loss $\left( \mathcal { L } _ { \mathrm { f i n e } } \right)$ . Following the optimal transport formulation in GeoTransformer [26], we supervise point-level matches within each superpoint pair using

Sinkhorn iterations. Here, $\tilde { \mathcal P }$ and $\tilde { \mathcal { Q } }$ denote intermediateresolution downsampled points obtained from the KPConv encoder. During training, we randomly sample $N _ { g } = 1 2 8$ ground-truth superpoint pairs. For each pair, we compute matching costs from point features at the $\tilde { \mathcal { P } } , \tilde { \mathcal { Q } }$ resolution and obtain soft assignments $\mathbf { A } _ { i }$ through the Sinkhorn algorithm, whose $( p , q )$ -th entry $a _ { p , q } ^ { i }$ represents the matching probability between two points. The negative log-likelihood loss is:

$$
\mathcal { L } _ { \mathrm { f i n e } } = \frac { 1 } { N _ { g } } \sum _ { i = 1 } ^ { N _ { g } } \left( - \sum _ { ( p , q ) \in \Gamma _ { i } } \log a _ { p , q } ^ { i } \right)\tag{5}
$$

where $\Gamma _ { i }$ includes ground-truth matches (within radius 0.05 m) and dustbin assignments for unmatched points<sup>3</sup>.

Expert load balancing loss $\left( \mathcal { L } _ { \mathrm { m o e } } \right)$ . To prevent the routing network from collapsing to a subset of experts, we adopt the load balancing loss [31]. Specifically, we compute the average routing probability $\begin{array} { r } { { \bar { r } _ { e } } \ = \ { \frac { 1 } { | B | } } \sum _ { k \in B } { r _ { k , e } } } \end{array}$ for each expert e across all superpoints in the batch B and minimize its KL divergence from the uniform distribution:

$$
\mathcal { L } _ { \mathrm { m o e } } = \sum _ { e = 1 } ^ { N _ { \mathrm { e x p e r t } } } \bar { r } _ { e } \log \frac { \bar { r } _ { e } } { 1 / N _ { \mathrm { e x p e r t } } } .\tag{6}
$$

## 4. Experiments

## 4.1. Implementation Details

Network architecture. We use $N = 3$ transformer block iterations. The number of experts is set to $N _ { \mathrm { e x p e r t } } ~ = ~ 4$ For the pair-wise distance embedding in geometric selfattention, we set $\sigma _ { d }$ to the voxel size at the superpoint level, and $\sigma _ { a } = 1 5 ^ { \circ }$ for the triplet-wise angular embedding. For local-to-global registration, only superpoint matches with at least 3 local point correspondences are used to compute transformation candidates. We iteratively recompute the transformation with the surviving inlier matches $N _ { r } ~ = ~ 5$ times.

Training and testing. We implement and train SAGE with PyTorch on an NVIDIA H100 GPU. Models are trained with the Adam optimizer for 45 epochs on 3DMatch with a batch size of 1. The learning rate starts from $1 0 ^ { - 4 }$ and is multiplied by 0.95 after each epoch. Loss weights are $\lambda _ { \mathrm { c o a r s e } } = 1 . 0 , \lambda _ { \mathrm { f i n e } } = 1 . 0$ and $\lambda _ { \mathrm { m o e } } = 0 . 0 1$ . We randomly sample $N _ { g } = 1 2 8$ ground-truth superpoint correspondences during training, and $N _ { c } = 2 5 6$ putative superpoint matches during testing. Data augmentation follows [17].

Baselines and reproduction. In the experiments, we evaluate two approaches: a single-pass approach and an iterative refinement approach. The former predicts the correspondences and the transformation directly from the extracted features without any feedback, and the latter iteratively updates the correspondences by embedding the estimate from the previous iteration as a prior to guide the subsequent matching. In the single-pass approach, we implemented several well-known methods, Predator [17], CoFiNet [49], GeoTransformer [26] and RoITr [50], to compare with our SAGE. For the iterative refinement approach, we extended our SAGE with the same iterative refinement used in PEAL [51] and PSReg [19]. We call this extension SAGE<sup>†</sup> in this paper based on the official implementation of PEAL [52]. On the other hand, the implementation of PSReg is not officially released, and several implementation hyperparameters are not reported, so we re-implemented PSReg from scratch. We restrict the comparison to methods that operate on point clouds alone, and therefore exclude semantic-enhanced methods such as SemReg [15], which require additional RGB images from calibrated cameras.

## 4.2. Evaluation on 3DMatch and 3DLoMatch

We evaluate SAGE on the 3DMatch [54] and 3DLo-Match [17] datasets to assess its performance in point cloud registration. We evaluate performance using three metrics following [17, 26]: Inlier Ratio $( I R ) ,$ which measures the fraction of putative correspondences whose residuals are below 0.1 m under the ground-truth transformation; Feature Matching Recall (FMR), which calculates the percentage of point cloud pairs with IR exceeding 5%; and Registration Recall $( R R ) ,$ which determines the fraction of pairs whose transformation error (RMSE) is below $0 . 2 \mathrm { m } ^ { 4 }$

Results with RANSAC-50K. Following standard protocols [18], we compare recent point cloud registration methods with our SAGE. We evaluate each method using RANSAC-50K with variable sample counts. Table 1 presents comprehensive results on both 3DMatch and 3DLoMatch benchmarks.

As shown in Table 1, among the single-pass methods the best and second-best scores are highlighted in boldface and underline, respectively. SAGE achieves the best FMR across all sample counts (98.3–98.6%), demonstrating a superior feature matching capability in high-overlap scenarios. For IR, SAGE achieves the best performance with 1000 or fewer samples, reaching 87.0% at 250 samples, and is second only to RoITr at 5000 and 2500 samples. For RR, SAGE achieves 92.0% at 2500 samples and 91.9% at 1000 samples, both of which are the highest among singlepass methods, while remaining competitive at other sample counts. The slight gap at 5000 samples (90.8% vs. Geo-Transformer’s 92.0%) is due to a limitation of RANSAC discussed below.

<table><tr><td rowspan="2"># Samples</td><td colspan="5">3DMatch</td><td colspan="5">3DLoMatch</td></tr><tr><td>5000</td><td>2500 1000</td><td></td><td>500</td><td>250</td><td>5000</td><td>2500</td><td>1000</td><td>500</td><td>250</td></tr><tr><td colspan="10">Feature Matching Recall (%) ↑</td></tr><tr><td>Predator [17]</td><td>96.6</td><td>96.6</td><td>96.5</td><td>96.3</td><td>96.5</td><td>78.6</td><td>77.4</td><td>76.3</td><td>75.7</td><td>75.3</td></tr><tr><td>CoFiNet [49]</td><td>98.1</td><td>98.3</td><td>98.1</td><td>98.2</td><td>98.3</td><td>83.1</td><td>83.5</td><td>83.3</td><td>83.1</td><td>82.6</td></tr><tr><td>GeoTrans. [26]</td><td>97.9</td><td>97.9</td><td>97.9</td><td>97.9</td><td>97.6</td><td>88.3</td><td>88.6</td><td>88.8</td><td>88.6</td><td>88.3</td></tr><tr><td>RoITr [50]</td><td>98.0</td><td>98.0</td><td>97.9</td><td>98.0</td><td>97.9</td><td>89.6</td><td>89.6</td><td>89.5</td><td>89.4</td><td>89.3</td></tr><tr><td>SAGE (ours)</td><td>98.4</td><td>98.5</td><td>98.6</td><td>98.6</td><td>98.3</td><td>89.6</td><td>89.7</td><td>89.9</td><td>90.0</td><td>89.4</td></tr><tr><td>PEAL [51]</td><td>97.8</td><td>98.1</td><td>98.0</td><td>98.0</td><td>97.8</td><td>85.8</td><td>86.0</td><td>85.5</td><td>85.1</td><td>85.0</td></tr><tr><td>PSReg [19]</td><td>97.1</td><td>97.4</td><td>97.7</td><td>97.7</td><td>97.7</td><td>85.3</td><td>85.7</td><td>85.7</td><td>85.8</td><td>85.3</td></tr><tr><td>SAGE† (ours)</td><td>97.2</td><td>97.5</td><td>97.7</td><td>97.8</td><td>97.7</td><td>85.4</td><td>85.9</td><td>85.4</td><td>85.2</td><td>85.3</td></tr><tr><td colspan="10">Inlier Ratio (%) ↑</td></tr><tr><td>Predator [17]</td><td>58.0</td><td>58.4</td><td>57.1</td><td>54.1</td><td>49.3</td><td>26.7</td><td>28.1</td><td>28.3</td><td>27.5</td><td>25.8</td></tr><tr><td>CoFiNet [49]</td><td>49.8</td><td>51.2</td><td>51.9</td><td>52.2</td><td>52.2</td><td>24.4</td><td>25.9</td><td>26.7</td><td>26.8</td><td>26.9 57.7</td></tr><tr><td>GeoTrans. [26]</td><td>71.9</td><td>75.2</td><td>76.0</td><td>82.2</td><td>85.1</td><td>43.5</td><td>45.3</td><td>46.2</td><td>52.9</td><td></td></tr><tr><td>RoITr [50]</td><td>82.6</td><td>82.8</td><td>83.0</td><td>83.0</td><td>83.0</td><td>54.3</td><td>54.6</td><td>55.1</td><td>55.2</td><td>55.3</td></tr><tr><td>SAGE (ours)</td><td>73.5</td><td>79.8</td><td>84.1</td><td>85.8</td><td>87.0</td><td>44.5</td><td>50.3</td><td>56.4</td><td>59.1</td><td>60.9</td></tr><tr><td>PEAL [51]</td><td>74.9</td><td>81.3</td><td>85.7</td><td>87.5</td><td>88.7</td><td>49.7</td><td>55.0</td><td>61.1</td><td>63.6</td><td>65.2</td></tr><tr><td>PSReg [19]</td><td>75.0</td><td>81.3</td><td>85.7</td><td>87.4</td><td>88.5</td><td>49.5</td><td>55.0</td><td>61.0</td><td>63.5</td><td>65.1</td></tr><tr><td>SAGE† (ours)</td><td>75.5</td><td>81.7</td><td>85.9</td><td>87.5</td><td>88.6</td><td>49.9</td><td>55.4</td><td>61.2</td><td>63.6</td><td>65.1</td></tr><tr><td colspan="10">Registration Recall (%) ↑</td><td></td></tr><tr><td>Predator [17]</td><td>89.0 89.3</td><td>89.9</td><td>90.6</td><td>88.5</td><td>86.6</td><td>59.8</td><td>61.2</td><td>62.4</td><td>60.8</td><td>58.1</td></tr><tr><td>CoFiNet [49] GeoTrans. [26]</td><td>92.0</td><td>88.9</td><td>88.4</td><td>87.4 91.4</td><td>87.0 91.2</td><td>67.5</td><td>66.2</td><td>64.2 74.2</td><td>63.1</td><td>61.0 73.5</td></tr><tr><td></td><td>91.9</td><td>91.8</td><td>91.8</td><td></td><td></td><td>75.0</td><td>74.8</td><td>74.8</td><td>74.1</td><td>73.6</td></tr><tr><td>RoITr [50]</td><td>90.8</td><td>91.7</td><td>91.8</td><td>91.4</td><td>91.0</td><td>74.7</td><td>74.8</td><td></td><td>74.2</td><td></td></tr><tr><td>SAGE (ours)</td><td>91.9</td><td>92.0</td><td>91.9</td><td>91.1</td><td>91.0</td><td>73.4</td><td>74.2</td><td>74.5</td><td>74.6</td><td>73.9</td></tr><tr><td>PEAL [51]</td><td></td><td>91.6</td><td>91.7</td><td>91.7</td><td>91.7</td><td>76.1</td><td>76.1</td><td>76.2</td><td>75.9</td><td>75.9</td></tr><tr><td>PSReg [19]</td><td>91.9</td><td>91.9</td><td>92.2</td><td>92.0</td><td>91.8</td><td>76.4</td><td>76.5</td><td>76.4</td><td>75.9</td><td>75.3</td></tr><tr><td>SAGE† (ours)</td><td>92.2</td><td>92.3</td><td>92.5</td><td>91.9</td><td>92.1</td><td>77.3</td><td>77.3</td><td>77.0</td><td>76.3</td><td>76.0</td></tr></table>

Table 1. Evaluation results on 3DMatch and 3DLoMatch using RANSAC-50K. Pink rows use iterative refinement. Boldface indicates the best result and underline indicates the second-best result within each group. Iterative PEAL and PSReg are reproduced under our setting and fall below their reported numbers.

On 3DLoMatch, SAGE achieves the best or tied-best FMR across all sample counts (89.6–90.0%). For IR, SAGE achieves the best performance with 1000 or fewer samples (56.4–60.9%), outperforming RoITr’s 55.3% by 5.6 points at 250 samples, and is second-best at 5000 and 2500 samples. This demonstrates SAGE’s robustness with limited correspondences. For RR, SAGE achieves 73.4–74.6%, which is competitive but slightly below the best methods in some configurations, due to the same limitation of RANSAC.

Specifically, the limitation of RANSAC in our setting is that its uniform random sampling treats all correspondences equally, failing to exploit the semantic structure that SAGE introduces.

Results with structure-aware estimators. We next evaluate SAGE under two protocols that overcome the limitation of RANSAC, and we use these methods to directly compare SAGE with the most relevant recent methods. The upper group of Table 2 reports single-pass results with LGR [26], a structure-aware estimator that exploits spatial relationships among correspondences rather than sampling them uniformly. Because PEAL refines its estimate over several iterations, we evaluate SAGE, PEAL, and PSReg under a common iterative protocol in which each method receives the same budget of six refinement iterations, which places them under identical input and evaluation conditions.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3DMatchFMR↑IR↑RR↑</td><td rowspan=1 colspan=1>3DLoMatchFMR↑IR↑RR↑</td></tr><tr><td rowspan=1 colspan=1>GeoTrans. [26]</td><td rowspan=1 colspan=1>98.270.992.5</td><td rowspan=2 colspan=1>87.1 43.574.089.454.473.4</td></tr><tr><td rowspan=2 colspan=1>RoITr [50]SAGE (ours)</td><td rowspan=1 colspan=1>98.182.090.9</td></tr><tr><td rowspan=1 colspan=1>98.271.791.0</td><td rowspan=1 colspan=1>89.444.376.3</td></tr><tr><td rowspan=1 colspan=1>PEAL [51]</td><td rowspan=1 colspan=1>97.873.591.5</td><td rowspan=1 colspan=1>85.8 49.576.9</td></tr><tr><td rowspan=1 colspan=1>PSReg [19]</td><td rowspan=1 colspan=1>97.1 73.492.1</td><td rowspan=1 colspan=1>85.449.577.4</td></tr><tr><td rowspan=1 colspan=1>SAGE† (ours)</td><td rowspan=1 colspan=1>97.2 73.792.2</td><td rowspan=1 colspan=1>85.4 49.777.6</td></tr></table>

Table 2. Evaluation on 3DMatch and 3DLoMatch with LGR [26].
<table><tr><td>Configuration</td><td>FMR↑[%]</td><td>IR↑[%]</td><td>RR↑ [%]</td></tr><tr><td>baseline</td><td>87.1</td><td>43.5</td><td>74.0</td></tr><tr><td>Two-Stage Feature Extraction only</td><td>88.5</td><td>40.4</td><td>74.7</td></tr><tr><td>Standard MoE only</td><td>87.5</td><td>42.6</td><td>74.7</td></tr><tr><td>SG-MoE only</td><td>87.5</td><td>43.8</td><td>75.3</td></tr><tr><td>Full SAGE</td><td>89.4</td><td>44.3</td><td>76.3</td></tr></table>

Table 3. Ablation study on 3DLoMatch with LGR.

Under LGR, SAGE attains the highest RR on 3DLo-Match (76.3%), surpassing GeoTransformer by 2.3 points and RoITr by 2.9 points, while remaining competitive on 3DMatch (91.0%). SAGE reaches this highest RR on 3DLoMatch despite a lower IR than RoITr, which indicates that its correspondences exhibit stronger spatial consistency that structure-aware estimators can effectively exploit.

Under this protocol, SAGE attains the highest IR and the highest RR on both 3DMatch and 3DLoMatch, reaching 92.2% and 77.6% RR respectively, whereas PEAL records the highest FMR. This shows that, when every method receives the same refinement budget, the semantic-guided routing in SAGE produces correspondences that are at least as reliable as those of the strongest recent MoE-based and iterative methods.

## 4.3. Ablation Studies

We analyze each component’s contribution on 3DLoMatch using LGR. Results are shown in Table 3.

Effect of SG-MoE. Removing SG-MoE decreases RR by 1.6 points and IR by 3.9 points. This demonstrates that semantic-guided expert routing directly improves correspondence quality. By routing based on frozen semantic features, experts naturally specialize for different semanticgeometric patterns, enabling effective disambiguation of geometrically similar but semantically distinct regions (e.g., walls vs. floors).

Effect of two-stage feature extraction. Without two-stage feature extraction, FMR drops 1.9 points and RR drops 1.0 point, while IR remains relatively stable. This indicates that semantic input primarily enables more discriminative geometric features by grounding them in semantic understanding, leading to better feature matching across more point cloud pairs.

![](images/c5ea49c4518c24ac02cbe7b28f61994333645c299862161727f7b175aa44c460.jpg)

![](images/2b6d79e2287758537a16bfe5d8a6d6b8500b0fd99365f81147cb96794f44d9b8.jpg)  
Figure 4. Qualitative comparison of correspondences on 3DLoMatch scenes. Each column shows a different point cloud pair. Top: GeoTransformer’s registration with correspondences (red for outliers, green for inliers). Bottom: SAGE’s registration and correspondences. SAGE effectively leverages semantic guidance to correctly identify inliers, even in low-overlap scenarios.

Complementary effects. Comparing w/ two-stage feature extraction (RR: 74.7%, IR: 40.4%) and w/ SG-MoE (RR: 75.3%, IR: 43.8%) reveals that both components contribute to different aspects. Two-stage feature extraction mainly improves FMR (+1.4 points) at the cost of IR (−3.1 points), whereas SG-MoE improves both IR and RR (+0.3 and +1.3 points). The complete SAGE achieves a 2.3-point RR improvement over the baseline, demonstrating the synergy of both components. Even with only one component, SAGE outperforms the baseline in RR, confirming the value of each design choice.

## 4.4. Qualitative Results

Fig. 4 shows qualitative comparisons on challenging 3DLo-Match scenes with geometrically similar structures. We visualize correspondences as lines colored green for inliers and red for outliers. In scenes where some furniture shares similar local geometry, GeoTransformer [26] (top row) produces predominantly outliers, resulting in low IR in some cases. In contrast, SAGE (bottom row) successfully distinguishes these geometrically similar but semantically distinct structures through semantic understanding, achieving high IR and accurate final registration. This visual evidence shows that semantic-guided expert routing effectively handles ambiguous geometric patterns that confound purely geometric methods. 5

Fig. 5 compares the expert activation patterns of Standard MoE and SG-MoE. Standard MoE produces fragmented assignments primarily driven by local geometric patterns, whereas SG-MoE assigns semantically similar regions to consistent experts, as illustrated by the coherent assignments of the highlighted furniture. This specialization emerges without explicit supervision of expert roles

Figure 5. Expert activation patterns. Top: Standard MoE produces fragmented expert assignments primarily driven by local geometry. Bottom: SG-MoE consistently assigns semantically similar regions, such as the furniture highlighted by red circles, to the same expert, demonstrating semantic-guided specialization.

and supports consistent feature processing across semantically related structures.

## 5. Conclusion

We presented SAGE, a framework that integrates frozen pre-trained semantic representations with a mixture of experts for point cloud registration, requiring neither additional sensors nor retraining of the semantic encoder. SAGE addresses a key limitation of geometryonly methods by distinguishing geometrically similar but semantically different structures, and achieves the highest registration recall among directly comparable methods on low-overlap 3DLoMatch while remaining competitive on 3DMatch and outdoor scenes.

## References

[1] Dror Aiger, Niloy J Mitra, and Daniel Cohen-Or. 4-Points Congruent Sets for Robust Pairwise Surface Registration. ACM Transactions on Graphics, 27(3):85:1–85:10, 2008. 2

[2] Li An, Pengbo Zhou, Mingquan Zhou, Yong Wang, and Guohua Geng. Diffusion Transformer for Point Cloud Registration: Digital Modeling of Cultural Heritage. Heritage Science, 12(1):198, 2024. 3

[3] Yasuhiro Aoki, Hunter Goforth, Rangaprasad Arun Srivatsan, and Simon Lucey. PointnetLK: Robust & Efficient Point Cloud Registration Using PointNet. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. 2

[4] Xuyang Bai, Zixin Luo, Lei Zhou, Hongbo Fu, Long Quan, and Chiew-Lan Tai. D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 2

[5] Daniel Barath and Jiˇr´ı Matas. Graph-Cut RANSAC. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018. 2

[6] Paul J Besl and Neil D McKay. A Method for Registration of 3-D Shapes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(2):239–256, 1992. 2

[7] Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, and Yufeng Yue. PointGPT: Auto-regressively Generative Pre-training from Point Clouds. In Advances in Neural Information Processing Systems, 2023. 3

[8] Guangyan Chen, Meiling Wang, Qingxiang Zhang, Li Yuan, and Yufeng Yue. Full Transformer Framework for Robust Point Cloud Registration With Deep Information Interaction. IEEE Transactions on Neural Networks and Learning Systems, 35(10):13368–13382, 2024. 3

[9] Yang Chen and Gerard Medioni. Object Modelling by Reg-´ istration of Multiple Range Images. Image and Vision Computing, 10(3):145–155, 1992. 2

[10] Sungjoon Choi, Qian-Yi Zhou, and Vladlen Koltun. Robust Reconstruction of Indoor Scenes. In IEEE Conference on Computer Vision and Pattern Recognition, 2015. 2

[11] Christopher Choy, Jaesik Park, and Vladlen Koltun. Fully Convolutional Geometric Features. In IEEE/CVF International Conference on Computer Vision, 2019. 2

[12] Haowen Deng, Tolga Birdal, and Slobodan Ilic. PPF-FoldNet: Unsupervised Learning of Rotation Invariant 3D Local Descriptors. In European Conference on Computer Vision, 2018. 2

[13] William Fedus, Barret Zoph, and Noam Shazeer. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022. 2, 3

[14] Martin A Fischler and Robert C Bolles. Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography. Communications ofthe ACM, 24(6):381–395, 1981. 2

[15] Sheldon Fung, Xuequan Lu, Dasith de Silva Edirimuni, Wei Pan, Xiao Liu, and Hongdong Li. SemReg: Semantics Constrained Point Cloud Registration. In European Conference on Computer Vision, 2024. 2, 3, 6

[16] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite. In IEEE Conference on Computer Vision and Pattern Recognition, 2012. 2, 15

[17] Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas Wieser, and Konrad Schindler. PREDATOR: Registration of 3D Point Clouds with Low Overlap. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 1, 2, 3, 4, 6, 7, 15

[18] Xiaoshui Huang, Guofeng Mei, Jian Zhang, and Rana Abbas. A Comprehensive Survey on Point Cloud Registration. arXiv preprint arXiv:2103.02690, 2021. 1, 2, 3, 6

[19] Xiaoshui Huang, Zhou Huang, Yifan Zuo, Yongshun Gong, Chengdong Zhang, Deyang Liu, and Yuming Fang. PSReg: Prior-Guided Sparse Mixture of Experts for Point Cloud Registration. In AAAI Conference on Artificial Intelligence, 2025. 1, 2, 3, 4, 6, 7, 22

[20] Andrew E Johnson and Martial Hebert. Using Spin Images for Efficient Object Recognition in Cluttered 3D Scenes. IEEE Transactions on Pattern Analysis and Machine Intel ligence, 21(5):433–449, 1999. 2

[21] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He,´ Bharath Hariharan, and Serge Belongie. Feature Pyramid Networks for Object Detection. In IEEE Conference on Computer Vision and Pattern Recognition, 2017. 12

[22] Juncheng Mu, Lin Bie, Shaoyi Du, and Yue Gao. ColorPCR: Color Point Cloud Registration with Multi-Stage Geometric-Color Fusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2, 3

[23] Yatian Pang, Wenxiao Wang, Francis E. H. Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked Autoencoders for Point Cloud Self-supervised Learning. In European Conference on Computer Vision, 2022. 3

[24] Helmut Pottmann, Qi-Xing Huang, Yong-Liang Yang, and Shi-Min Hu. Geometry and Convergence Analysis of Algorithms for Registration of 3D Shapes. International Journal ofComputer Vision, 67(3):277–296, 2006. 2

[25] Haobo Qin, Yinchang Zhou, Chao Liu, Xiaopeng Zhang, Zhanglin Cheng, and Jianwei Guo. SARNet: Semantic Augmented Registration of Large-Scale Urban Point Clouds. In International Conference on Computational Visual Media, 2024. 2, 3

[26] Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng, and Kai Xu. Geometric Transformer for Fast and Robust Point Cloud Registration. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 22

[27] Ignacio Rocco, Mircea Cimpoi, Relja Arandjelovic, Akihiko´ Torii, Tomas Pajdla, and Josef Sivic. Neighbourhood Consensus Networks. In Advances in Neural Information Pro cessing Systems, 2018. 13

[28] Szymon Rusinkiewicz and Marc Levoy. Efficient Variants of The ICP Algorithm. In International Conference on 3-D Digital Imaging and Modeling, 2001. 2

[29] Radu Bogdan Rusu, Nico Blodow, and Michael Beetz. Fast Point Feature Histograms (FPFH) for 3D Registration. In IEEE International Conference on Robotics and Automation, 2009. 2

[30] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural Networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 13, 14

[31] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In International Conference on Learning Representations, 2017. 2, 3, 6, 14

[32] Rui She, Qiyu Kang, Sijie Wang, Wee Peng Tay, Kai Zhao, Yang Song, Tianyu Geng, Yi Xu, Diego Navarro Navarro, and Andreas Hartmannsgruber. PointDifformer: Robust Point Cloud Registration with Neural Diffusion and Transformer. IEEE Transactions on Geoscience and Remote Sensing, 62:1–15, 2024. 3

[33] Rui She, Sijie Wang, Qiyu Kang, Kai Zhao, Yang Song, Wee Peng Tay, Tianyu Geng, and Xingchao Jian. PosDiffNet: Positional Neural Diffusion for Point Cloud Registration in a Large Field of View with Perturbations. In AAAI Conference on Artificial Intelligence, 2024.

[34] Tomoyasu Shimada, Kazuhiko Murasaki, Shogo Sato, Toshihiko Nishimura, Taiga Yoshida, and Ryuichi Tanida. Memory-Efficient Point Cloud Registration via Overlapping Region Sampling. In International Conference on Visual Communications and Image Processing, 2024. 3

[35] Richard Sinkhorn and Paul Knopp. Concerning Nonnegative Matrices and Doubly Stochastic Matrices. Pacific Journal of Mathematics, 21(2):343–348, 1967. 13

[36] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. LoFTR: Detector-free Local Feature Matching with Transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 13

[37] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, Franc¸ois Goulette, and Leonidas J Guibas. KPConv: Flexible and Deformable Convolution for Point Clouds. In IEEE/CVF International Conference on Computer Vision, 2019. 4

[38] Federico Tombari, Samuele Salti, and Luigi Di Stefano. Unique Signatures of Histograms for Local Surface Description. In European Conference on Computer Vision, 2010. 2

[39] Meiling Wang, Guangyan Chen, Yi Yang, Li Yuan, and Yufeng Yue. Point Tree Transformer for Point Cloud Registration. IEEE Transactions on Circuits and Systemsfor Video Technology, 35(7):6756–6772, 2025. 3

[40] Yue Wang and Justin M. Solomon. Deep Closest Point: Learning Representations for Point Cloud Registration. In IEEE/CVF International Conference on Computer Vision, 2019. 3

[41] Ziyi Wang, Yanran Zhang, Jie Zhou, and Jiwen Lu. UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 3

[42] Xiaoyang Wu, Xin Wen, Xihui Liu, and Hengshuang Zhao. Masked Scene Contrast: A Scalable Framework for Unsu-

pervised 3D Representation Learning. In IEEE/CVF Con ference on Computer Vision and Pattern Recognition, 2023. 3

[43] Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, and Julian Straub. Sonata: Self-Supervised Learning of Reliable Point Representations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 2, 3, 4, 12, 13, 18

[44] Yanhao Wu, Tong Zhang, Wei Ke, Congpei Qiu, Sabine Susstrunk, and Mathieu Salzmann. Mitigating Object De-¨ pendencies: Improving Point Cloud Self-Supervised Learn ing through Object Exchange. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 3

[45] Zongyi Xu, Xinyu Gao, Xinqi Jiang, Shiyang Cheng, Qianni Zhang, Weisheng Li, and Xinbo Gao. S2Reg: Structuresemantics Collaborative Point Cloud Registration. Pattern Recognition, 161:111290, 2025. 2, 3

[46] Shaocheng Yan, Pengcheng Shi, and Jiayuan Li. ML-SemReg: Boosting Point Cloud Registration with Multi level Semantic Consistency. In European Conference on Computer Vision, 2024. 3

[47] Yaoqing Yang, Chen Feng, Yiru Shen, and Dong Tian. FoldingNet: Point Cloud Auto-Encoder via Deep Grid Deforma tion. In IEEE/CVF Conference on Computer Vision and Pat tern Recognition, 2018. 2

[48] Zi Jian Yew and Gim Hee Lee. REGTR: End-to-End Point Cloud Correspondences With Transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 1, 3

[49] Hao Yu, Fu Li, Mahdi Saleh, Benjamin Busam, and Slobodan Ilic. CoFiNet: Reliable Coarse-to-fine Correspondences for Robust PointCloud Registration. In Advances in Neural Information Processing Systems, 2021. 3, 4, 6, 7

[50] Hao Yu, Zheng Qin, Ji Hou, Mahdi Saleh, Dongsheng Li, Benjamin Busam, and Slobodan Ilic. Rotation-Invariant Transformer for Point Cloud Matching. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 1, 5, 6, 7

[51] Junle Yu, Luwei Ren, Yu Zhang, Wenhui Zhou, Lili Lin, and Guojun Dai. PEAL: Prior-Embedded Explicit Attention Learning for Low-Overlap Point Cloud Registration. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2, 3, 4, 6, 7, 22

[52] Junle Yu, Luwei Ren, Yu Zhang, Wenhui Zhou, Lili Lin, and Guojun Dai. PEAL: Prior-embedded Explicit Attention Learning for Low-overlap Point Cloud Registration. https://github.com/Gardlin/PEAL/tree/ 82b382e9da456edc54fc43ba02d6f3aaf1c8e391, 2023. GitHub repository, commit 82b382e9da456edc54fc43ba02d6f3aaf1c8e391, accessed October, 2026. 6

[53] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-BERT: Pre-Training 3D Point Cloud Transformers with Masked Point Modeling. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 3

[54] Andy Zeng, Shuran Song, Matthias Nießner, Matthew Fisher, Jianxiong Xiao, and Thomas Funkhouser. 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions. In IEEE Conference on Computer Vision and Pattern Recognition, 2017. 2, 6

[55] Yaohua Zha, Tao Dai, Hang Guo, Yanzi Wang, Bin Chen, Ke Chen, and Shu-Tao Xia. Point Cloud Mixture-of-Domain-Experts Model for 3D Self-supervised Learning. In International Joint Conference on Artificial Intelligence, 2025. 3

[56] Yaojie Zhang, Tianlun Huang, Weijun Wang, and Wei Feng. Decision PCR: Decision Version of the Point Cloud Registration Task. arXiv preprint arXiv:2507.14965, 2025. 2, 3

# SAGE: Semantic-Augmented Geometric Experts for Point Cloud Registration Supplementary Material

## A. Network Architecture Details

## A.1. Semantic Representation Extractor (Sonata)

SAGE employs Sonata [43] as the semantic representation extractor. Sonata is a point cloud foundation model that is pre-trained through a contrastive learning objective designed to align point cloud representations with semantically meaningful structures while explicitly avoiding geometric shortcuts. Its encoder follows a sparse convolutional architecture that processes voxelized point clouds and produces per-point feature vectors of 1024 dimensions.

An important property of Sonata for our framework is that it outputs continuous, high-dimensional semantic representation vectors rather than discrete categorical labels. Unlike conventional semantic segmentation models that predict specific categories such as “chair” or “car,” Sonata captures rich semantic relationships in a continuous feature space where semantically similar structures are grouped together without being constrained by a predefined label vocabulary. This characteristic is essential for SAGE because it allows the SG-MoE router to exploit fine-grained semantic distinctions that go beyond coarse category boundaries. For both indoor and outdoor environments, we use the frozen Sonata model exclusively as a feature extractor, which ensures domain-agnostic operation that remains entirely free from the constraints of dataset-specific label spaces.

## A.2. Geometric Feature Extractor (KPConv) for 3DMatch and 3DLoMatch

Following GeoTransformer [26], we use a KPConv-FPN backbone for feature extraction. Prior to feeding the data into the backbone network, the input point clouds are processed with an initial voxelization using a sampling resolution of 2.5 cm. As the data progresses through successive downsampling stages, the voxel resolution is progressively coarsened by a factor of two at each level.

As shown in Table A, SAGE adopts the same KPConv-FPN backbone as GeoTransformer [26], with the only modification being the input dimension of the first KPConv layer. Specifically, we expand the input feature dimension from 1 (no input features) to 1088 to incorporate pre-trained semantic representations from Sonata [43]. This enables the backbone to learn semantic-geometric features from the initial stage.

Table A. Geometric Feature Extractor (KPConv) architecture. SAGE modifies only the input dimension of the first KPConv layer (red) to accommodate 1024-dimensional Sonata semantic features concatenated with 64-dimensional positional encodings.
<table><tr><td>Stage</td><td>GeoTransformer</td><td>SAGE</td></tr><tr><td rowspan="2">1</td><td>KPConv(1 → 64)</td><td>KPConv(1088 → 64)</td></tr><tr><td>ResBlock(64 → 128)</td><td>ResBlock(64 → 128)</td></tr><tr><td rowspan="3">2</td><td>ResBlock(128 → 128, strided)</td><td>ResBlock(128 → 128, strided)</td></tr><tr><td>ResBlock(128 → 256)</td><td>ResBlock(128 → 256)</td></tr><tr><td>ResBlock(256 → 256)</td><td>ResBlock(256 → 256)</td></tr><tr><td rowspan="3">3</td><td>ResBlock(256 → 256, strided)</td><td>ResBlock(256 → 256, strided)</td></tr><tr><td>ResBlock(256 → 512)</td><td>ResBlock(256 → 512)</td></tr><tr><td>ResBlock(512 → 512)</td><td>ResBlock(512 → 512)</td></tr><tr><td rowspan="3">4</td><td>ResBlock(512 → 512, strided)</td><td>ResBlock(512 → 512, strided)</td></tr><tr><td>ResBlock(512 → 1024)</td><td>ResBlock(512 → 1024)</td></tr><tr><td>ResBlock(1024 → 1024)</td><td>ResBlock(1024 → 1024)</td></tr><tr><td colspan="3">Upsampling (FPN)</td></tr><tr><td>5</td><td>NearestUpsampling UnaryConv(1536 → 512)</td><td>NearestUpsampling UnaryConv(1536 → 512)</td></tr><tr><td>6</td><td>NearestUpsampling UnaryConv(768 → 256)</td><td>NearestUpsampling UnaryConv(768 → 256)</td></tr></table>

## A.3. Geometric Feature Extractor (KPConv) for KITTI

For the KITTI odometry benchmark, we adapt the backbone architecture to handle larger-scale outdoor environments. The input point clouds are voxelized with a sampling resolution of 30 cm, significantly coarser than the 2.5 cm used for 3DMatch, reflecting the difference in spatial scale between indoor rooms and outdoor driving scenarios.

As shown in Table B, we employ a deeper 5-stage backbone compared to the 4-stage backbone used for 3DMatch. The additional downsampling stage (Stage 5) is necessary to capture long-range spatial relationships in outdoor scenes that can span over 100 meters, while maintaining computational efficiency. Without this additional stage, the receptive field would be insufficient to establish correspondences across the large spatial extents typical of KITTI scenes.

Similar to 3DMatch, we modify only the input feature dimension of the first KPConv layer from 1 to 1088 to incorporate semantic features from Sonata [43]. The deeper architecture with Feature Pyramid Network [21] connections enables effective multi-scale feature fusion across the extended spatial hierarchy required for outdoor registration.

## A.4. Semantic-Guided MoE Transformer

This subsection details the architecture of the SG-MoE Transformer introduced in Sec. 3.4 of the main paper, which refines the superpoint-level semantic-geometric features before matching. The module operates on $d _ { s g } = 2 5 6 \AA$ dimensional semantic-geometric features and stacks three blocks, each consisting of a geometric self-attention layer followed by an SG-MoE cross-attention layer (i.e., the block pattern [self, cross] repeated three times). The crossattention layer uses 4 heads.

Table B. KPConv-FPN architecture for KITTI. SAGE uses a $5 \mathrm { - }$ stage backbone with an additional downsampling stage (Stage 5) compared to 3DMatch to handle larger outdoor scenes. The input feature dimension of the first KPConv layer is modified from 1 to 1088 (highlighted in red) to incorporate semantic features.
<table><tr><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>GeoTransformer</td><td rowspan=1 colspan=1>SAGE</td></tr><tr><td rowspan=1 colspan=3>Downsampling</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>KPConv(1 → 64)            KPConv(1088 → 64)ResBlock(64 → 128)          ResBlock(64 → 128)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>ResBlock(128 → 128, strided)ResBlock(128 → 256)ResBlock(256 → 256)</td><td rowspan=1 colspan=1>ResBlock(128 → 128, strided)ResBlock(128 → 256)ResBlock(256 → 256)</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>ResBlock(256 → 256, strided)ResBlock(256 → 512)ResBlock(512 → 512)</td><td rowspan=1 colspan=1>ResBlock(256 → 256, strided)ResBlock(256 → 512)ResBlock(512 → 512)</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>ResBlock(512 → 512, strided)ResBlock(512 → 1024)ResBlock(1024 → 1024)</td><td rowspan=1 colspan=1>ResBlock(512 → 512, strided)ResBlock(512 → 1024)ResBlock(1024 → 1024)</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>ResBlock(1024 → 1024, strided)ResBlock(1024 → 2048)ResBlock(2048 → 2048)</td><td rowspan=1 colspan=1>ResBlock(1024 → 1024, strided)ResBlock(1024 → 2048)ResBlock(2048 → 2048)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Upsampling (</td><td rowspan=1 colspan=1>FPN)</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>NearestUpsamplingUnaryConv(3072 → 1024)</td><td rowspan=1 colspan=1>NearestUpsamplingUnaryConv(3072 → 1024)</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>NearestUpsamplingUnaryConv(1536 → 512)</td><td rowspan=1 colspan=1>NearestUpsamplingUnaryConv(1536 → 512)</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>NearestUpsamplingUnaryConv(768 → 256)</td><td rowspan=1 colspan=1>NearestUpsamplingUnaryConv(768 → 256)</td></tr></table>

Router. Routing is computed from the superpoint-level frozen semantic features $\hat { \bf f } _ { s , k } \in \mathbb { R } ^ { 1 0 2 4 }$ (Sonata [43] output, aggregated to the superpoint level as described in Sec. 3.3 of the main paper). The routing network π (Eq. (2) of the main paper) is a two-layer ML $\mathrm { P \ ( 1 0 2 4  2 5 6  } N _ { \mathrm { e x p e r t } } )$ with ReLU activation, followed by a softmax and top-1 selection. We apply a gate dropout of 0.1.

Experts. Each of the $N _ { \mathrm { e x p e r t } } ~ = ~ 4$ experts is a twolayer feed-forward network with ReLU activation and an inverted-bottleneck shape $( 2 5 6  5 1 2  2 5 6 )$ , with an expert dropout of 0.1. The experts process the semanticgeometric features produced by cross-attention, not the semantic features used for routing.

## A.5. Superpoint Matching Module

Following GeoTransformer [26], we adopt their superpoint matching strategy to establish coarse correspondences between downsampled point clouds. The module consists of three key operations: Gaussian correlation computation, dual-normalization, and top-k correspondence selection. Gaussian Correlation Matrix. Given the refined semanticgeometric features $\hat { \mathbf { F } } _ { \mathrm { o u t } } ^ { \mathcal { P } } \in \mathbb { R } ^ { \hat { N } _ { \mathcal { P } } \times d _ { s g } }$ and $\hat { \mathbf { F } } _ { \mathrm { o u t } } ^ { \mathcal { Q } } \in \mathbb { R } ^ { \hat { N } _ { \mathcal { Q } } \times d _ { s g } }$ from the SG-MoE Transformer, we first normalize them onto a unit hypersphere and compute a Gaussian correlation matrix $\mathbf { S } \in \mathbb { R } ^ { \hat { N } _ { \mathcal { P } } \times \hat { N } _ { \mathcal { Q } } }$

$$
s _ { i , j } = \exp \left( - \| \hat { \mathbf { f } } _ { \mathrm { o u t } , i } ^ { \mathcal { P } } - \hat { \mathbf { f } } _ { \mathrm { o u t } , j } ^ { \mathcal { Q } } \| _ { 2 } ^ { 2 } \right) ,\tag{7}
$$

where $\hat { \mathbf { f } } _ { \mathrm { o u t } , i } ^ { \mathcal { P } }$ and $\hat { \mathbf { f } } _ { \mathrm { o u t } , j } ^ { \mathcal { Q } }$ are the i-th and j-th rows of normalized feature matrices.

Dual-normalization. To suppress ambiguous matches from geometrically indiscriminative patches, we apply a dual-normalization operation [27, 36] on the correlation matrix:

$$
\bar { s } _ { i , j } = \frac { s _ { i , j } } { \sum _ { k = 1 } ^ { \hat { N } _ { \mathbb { Q } } } s _ { i , k } } \cdot \frac { s _ { i , j } } { \sum _ { k = 1 } ^ { \hat { N } _ { \mathcal { P } } } s _ { k , j } } .\tag{8}
$$

This operation normalizes the correlation scores both rowwise and column-wise, effectively penalizing matches where a superpoint has multiple high-scoring candidates.

Top-k Correspondence Selection. We select the $N _ { c }$ highest-scoring entries from the normalized correlation matrix S<sup>¯</sup> as the superpoint correspondences:

$$
\hat { \mathcal { C } } = \{ ( \hat { \mathbf { p } } _ { x _ { i } } , \hat { \mathbf { q } } _ { y _ { i } } ) ~ | ~ ( x _ { i } , y _ { i } ) \in \mathrm { t o p k } _ { x , y } ( \bar { s } _ { x , y } ) \} .\tag{9}
$$

Following GeoTransformer [26], we use $N _ { c } = 2 5 6$ putative superpoint matches during inference.

Empty Patch Removal. Before matching, we remove superpoints with empty local patches. A superpoint $\hat { { \bf p } } _ { i }$ is considered empty if its associated patch $\bar { \mathcal { G } } _ { i } ^ { \mathcal { P } }$ (i.e., the set of points assigned to pˆ<sub>i</sub> by the point-to-node grouping of Geo-Transformer [26]) contains no points. This preprocessing step improves matching quality by excluding uninformative superpoints.

## A.6. Point Matching Module

For each superpoint correspondence $( \hat { \mathbf { p } } _ { x _ { i } } , \hat { \mathbf { q } } _ { y _ { i } } ) \in \hat { \mathcal { C } } ,$ , we extract dense point correspondences from their local patches using an optimal transport layer [30].

Following the same procedure as in GeoTransformer [26], we compute a cost matrix $\mathbf { C } _ { i } \in \mathbb { R } ^ { n _ { i } \times m _ { i } }$ from intermediate-resolution point features $\mathbf { F } _ { x _ { i } } ^ { \mathcal { P } }$ and $\mathbf { F } _ { y _ { i } } ^ { \mathcal { Q } }$ :

$$
\mathbf { C } _ { i } = \mathbf { F } _ { x _ { i } } ^ { \mathcal { P } } ( \mathbf { F } _ { y _ { i } } ^ { \mathcal { Q } } ) ^ { T } / \sqrt { \tilde { d } } ,\tag{10}
$$

where $n _ { i } = | \mathcal { G } _ { x _ { i } } ^ { \mathcal { P } } | , m _ { i } = | \mathcal { G } _ { y _ { i } } ^ { \mathcal { Q } } |$ are patch sizes, and $\tilde { d }$ is the feature dimension.

This cost matrix is augmented to $\bar { \mathbf { C } } _ { i } \in \mathbb { R } ^ { ( n _ { i } + 1 ) \times ( m _ { i } + 1 ) }$ by appending dustbin entries (initialized with learnable parameter α) to handle unmatched points. The Sinkhorn algorithm [35] then produces soft assignments A<sub>i</sub>, from which we extract correspondences via mutual top-k selection after removing dustbin entries. We set k = 3 to obtain approximately 6000 correspondences for Local-to-Global Registration.

## A.7. Local-to-Global Registration

We adopt GeoTransformer’s Local-to-Global Registration (LGR) [26], a parameter-free and RANSAC-free method.

Local Phase: For each superpoint correspondence, we solve for transformation $\mathbf { T } _ { i } ~ = ~ ( \mathbf { R } _ { i } , \mathbf { t } _ { i } )$ using weighted SVD over local point correspondences $\mathcal { C } _ { i } \mathrm { : }$

$$
\mathbf { R } _ { i } , \mathbf { t } _ { i } = \arg \operatorname* { m i n } _ { \mathbf { R } , \mathbf { t } } \sum _ { ( \tilde { \mathbf { p } } _ { x _ { j } } , \tilde { \mathbf { q } } _ { y _ { j } } ) \in \mathcal { C } _ { i } } w _ { j } ^ { i } \| \mathbf { R } \mathbf { \cdot } \tilde { \mathbf { p } } _ { x _ { j } } + \mathbf { t } - \tilde { \mathbf { q } } _ { y _ { j } } \| _ { 2 } ^ { 2 } ,\tag{11}
$$

where weights $\boldsymbol { w } _ { j } ^ { i }$ are confidence scores from the optimal transport layer.

Global Phase: Select the transformation admitting the most inliers:

$$
\mathbf { R } , \mathbf { t } = \arg \operatorname* { m a x } _ { \mathbf { R } _ { i } , \mathbf { t } _ { i } } \sum _ { ( \tilde { \mathbf { p } } _ { x _ { j } } , \tilde { \mathbf { q } } _ { y _ { j } } ) \in \mathcal { C } } \mathbb { I } [ \| \mathbf { R } _ { i } \cdot \tilde { \mathbf { p } } _ { x _ { j } } + \mathbf { t } _ { i } - \tilde { \mathbf { q } } _ { y _ { j } } \| _ { 2 } < \tau _ { a } ] ,\tag{12}
$$

with acceptance radius $\tau _ { a } = 0 . 1 \mathrm { m }$ . We refine this transformation iteratively for $N _ { r } = 5$ iterations.

## B. Loss Function Details

## B.1. Overlap-Aware Circle Loss

We supervise superpoint matching with an overlap-aware circle loss adapted from GeoTransformer [26]. For anchor superpoints $\mathcal { A } ^ { \bar { P } }$ (those with at least one overlapping patch in $\hat { \mathcal { Q } } )$ , we define:

• $\mathcal { P } _ { k } ^ { + }$ : positive patches (overlap ≥ 10%)

• $\mathcal { P } _ { k } ^ { - }$ : negative patches (no overlap)

The loss for the direction $\mathcal { P }  \mathcal { Q }$ is:

$$
\mathcal { L } _ { \mathrm { o c } } ^ { \mathcal { P }  \mathcal { Q } } = \frac { 1 } { | \mathcal { A } ^ { \mathcal { P } } | } \sum _ { k \in \mathcal { A } ^ { \mathcal { P } } } \log ( 1 + \sum _ { j \in \mathcal { P } _ { k } ^ { + } } e ^ { w _ { k , j } \beta _ { k , j } ^ { + } } \cdot \sum _ { \ell \in \mathcal { P } _ { k } ^ { - } } e ^ { \beta _ { k , \ell } ^ { - } } ) ,\tag{13}
$$

where:

$d _ { k , j } = \lVert \hat { \mathbf { f } } _ { \mathrm { o u t } , k } ^ { \mathcal { P } } - \hat { \mathbf { f } } _ { \mathrm { o u t } , j } ^ { \mathcal { Q } } \rVert _ { 2 }$ (distance in the refined semanticgeometric feature space)

$w _ { k , j } = \sqrt { o _ { k , j } }$ (overlap-based weighting)

$\beta _ { k , j } ^ { + } = \gamma ( d _ { k , j } - \Delta _ { p } )$ and $\beta _ { k , \ell } ^ { - } = \gamma ( \Delta _ { n } - d _ { k , \ell } )$

The bidirectional coarse loss is $\mathcal { L } _ { \mathrm { c o a r s e } } ~ = ~ \mathcal { L } _ { \mathrm { o c } } ^ { \mathcal { P }  \mathcal { Q } } ~ + ~$ $ { \mathcal { L } } _ { \mathrm { o c } } ^ { \mathcal { Q }  \mathcal { P } }$

## B.2. Optimal Transport Loss

For each sampled ground-truth superpoint pair during training, we supervise point-level matches via negative loglikelihood loss [30].

Given cost matrix $\mathbf { C } _ { i }$ augmented to $\bar { \mathbf { C } } _ { i }$ with dustbin entries, Sinkhorn iterations produce soft assignment $\mathbf { A } _ { i }$ Ground-truth labels $\Gamma _ { i }$ include:

• Matched pairs: $( p , q )$ with $\| \mathbf { T } ^ { * } ( p ) - q \| _ { 2 } < 0 . 0 5 \mathbf { m }$

• Dustbin assignments for unmatched points

The fine loss is:

$$
\mathcal { L } _ { \mathrm { f i n e } } = \frac { 1 } { N _ { g } } \sum _ { i = 1 } ^ { N _ { g } } \left( - \sum _ { ( p , q ) \in \Gamma _ { i } } \log a _ { p , q } ^ { i } \right) ,\tag{14}
$$

where we sample $N _ { g } = 1 2 8$ ground-truth superpoint pairs per batch.

## B.3. Expert Load Balancing Loss

The SG-MoE layer introduces a set of trainable expert networks, which necessitates an auxiliary loss to ensure training stability and prevent expert saturation. This is a common failure mode in MoE models where the router learns to favor only one or a few popular experts, leaving the others untrained and wasting capacity. To enforce balanced expert utilization, we employ a load balancing loss ${ \mathcal { L } } _ { \mathrm { m o e } } \left[ 3 1 \right]$

This loss encourages the routing network to distribute the superpoints uniformly across all $N _ { \mathrm { e x p e r t } }$ experts on average over the batch. The routing network computes a probability distribution $\mathbf { r } _ { k } = [ r _ { k , 1 } , . . . , r _ { k , N _ { \mathrm { e x p e r t } } } ]$ for each superpoint $k .$ We first compute the average routing probability for each expert e by averaging its probability $r _ { k , e }$ across all superpoints in the batch B (including superpoints from both $\mathcal { P }$ and Q):

$$
\bar { r } _ { e } = \frac { 1 } { | \boldsymbol { \bar { B } } | } \sum _ { \boldsymbol { k } \in \mathcal { B } } { r _ { \boldsymbol { k } , e } } ,\tag{15}
$$

where |B| is the total number of superpoints in the batch. This gives us the batch-wise load distribution $\begin{array} { r c l } { \bar { \bf r } } & { = } & { \left[ \bar { r } _ { 1 } , \dots , \bar { r } _ { N _ { \mathrm { e x p e r t } } } \right] } \end{array}$ We want this distribution ¯r to be as close as possible to a uniform distribution $\mathbf { u } \ =$ $[ 1 / N _ { \mathrm { e x p e r t } } , \ldots , 1 / N _ { \mathrm { e x p e r t } } ]$ . We achieve this by minimizing the Kullback-Leibler (KL) divergence between ¯r and u:

$$
\mathcal { L } _ { \mathrm { m o e } } = K L ( \bar { \mathbf { r } } | | \mathbf { u } ) = \sum _ { e = 1 } ^ { N _ { \mathrm { e x p e r t } } } \bar { r } _ { e } \log \frac { \bar { r } _ { e } } { 1 / N _ { \mathrm { e x p e r t } } } .\tag{16}
$$

Minimizing this loss penalizes routing strategies that deviate significantly from uniformity, encouraging the model to leverage the full capacity of all $N _ { \mathrm { e x p e r t } }$ specialized experts.

## B.4. Training Configuration

Optimizer: Adam with learning rate $1 0 ^ { - 4 }$ , multiplied by 0.95 after each epoch, for 45 epochs.

Loss hyperparameters:

• Coarse loss: $\Delta _ { p } = 0 . 1 , \Delta _ { n } = 1 . 4 , \gamma = 8 0$

• Fine loss: matching radius $\tau = 0 . 0 5 \mathrm { m }$

• MoE load balancing: $\lambda _ { \mathrm { m o e } } = 0 . 0 1$ Random seed: 7351

## C. Evaluation Metrics

We provide detailed mathematical definitions of the evaluation metrics used in our experiments. The metrics differ between indoor (3DMatch/3DLoMatch) and outdoor (KITTI)

benchmarks due to different evaluation protocols and scene characteristics.

## C.1. 3DMatch and 3DLoMatch

Inlier Ratio (IR). The Inlier Ratio measures the quality of correspondences as the fraction of inlier matches among all putative correspondences:

$$
\mathrm { I R } = \frac { 1 } { \vert \mathcal { C } \vert } \sum _ { ( p _ { i } , q _ { j } ) \in \mathcal { C } } \mathrm { I I } [ \| T ^ { * } ( p _ { i } ) - q _ { j } \| _ { 2 } < \tau _ { \mathrm { i n l i e r } } ] ,\tag{17}
$$

where $\mathcal { C }$ denotes the set of putative correspondences, $T ^ { * }$ is the ground-truth transformation, I[·] is the indicator function, and $\tau _ { \mathrm { i n l i e r } } = 0 . 1$ m is the inlier threshold.

Feature Matching Recall (FMR). The Feature Matching Recall evaluates the percentage of point cloud pairs that have sufficient inliers for potential successful registration:

$$
\mathrm { F M R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } [ \mathbf { I R } _ { i } > \tau _ { \mathrm { F M R } } ] ,\tag{18}
$$

where $N$ is the total number of point cloud pairs in the dataset, and $\tau _ { \mathrm { F M R } } = 0 . 0 5 ( 5 \% )$ .

Registration Recall (RR). The Registration Recall measures the fraction of successfully registered point cloud pairs:

$$
\mathrm { R R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } [ \mathrm { R M S E } _ { i } < \tau _ { \mathrm { R R } } ] ,\tag{19}
$$

where the RMSE (Root Mean Square Error) is computed over ground-truth correspondences $\mathcal { C } ^ { * }$ under the estimated transformation $\hat { T } \colon$

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { | \mathcal { C } ^ { * } | } \sum _ { ( p _ { i } ^ { * } , q _ { j } ^ { * } ) \in \mathcal { C } ^ { * } } \| \hat { T } ( p _ { i } ^ { * } ) - q _ { j } ^ { * } \| _ { 2 } ^ { 2 } } ,\tag{20}
$$

and $\tau _ { \mathrm { R R } } = 0 . 2$ m is the success threshold.

## C.2. KITTI

For outdoor KITTI odometry benchmark, we adopt different metrics that focus on the accuracy of the estimated transformation itself rather than correspondence quality.

Relative Translation Error (RTE). The Relative Translation Error measures the Euclidean distance between the estimated translation vector and the ground-truth translation vector:

$$
\begin{array} { r } { \mathrm { R T E } = \lVert \hat { \mathbf { t } } - \mathbf { t } ^ { * } \rVert _ { 2 } , } \end{array}\tag{21}
$$

where $\hat { \mathbf { t } }$ is the estimated translation and $\mathbf { t } ^ { * }$ is the groundtruth translation. We report RTE in centimeters.

Relative Rotation Error (RRE). The Relative Rotation Error measures the geodesic distance between the estimated rotation matrix and the ground-truth rotation matrix:

$$
\mathrm { R R E } = \operatorname { a r c c o s } \left( \frac { \operatorname { t r a c e } ( \hat { \mathbf { R } } ^ { T } \mathbf { R } ^ { * } ) - 1 } { 2 } \right) ,\tag{22}
$$

where $\hat { \textbf { R } }$ is the estimated rotation matrix and $\mathbf { R } ^ { * }$ is the ground-truth rotation matrix. We report RRE in degrees. The geodesic distance represents the angle of rotation needed to align R<sup>ˆ</sup> with R<sup>∗</sup>.

Registration Recall (RR). Following the KITTI evaluation protocol [16], the Registration Recall measures the fraction of point cloud pairs whose transformation errors are below specified thresholds:

$$
\mathsf { R R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } [ \mathsf { R T E } _ { i } < \tau _ { t } \wedge \mathsf { R R E } _ { i } < \tau _ { r } ] ,\tag{23}
$$

where $\tau _ { t } = 2$ m is the translation error threshold, $\tau _ { r } = 5 ^ { \circ }$ is the rotation error threshold, and N is the total number of point cloud pairs. Note that both conditions must be satisfied simultaneously for a pair to be considered successfully registered.

Mean error computation. Following standard practice [17, 26], when reporting mean RTE and RRE values, we compute the average only over successfully registered pairs (those satisfying the RR criteria) to avoid including catastrophic failures that would dominate the statistics.

## D. Additional Experiments

Unless otherwise stated, all experiments in this section employ Local-to-Global Registration (LGR) [26] as the transformation estimator, which is a RANSAC-free method that leverages confidence-weighted correspondences for robust pose estimation. As discussed in the main paper, LGR is better suited to exploit the semantically structured correspondences produced by SAGE, and therefore serves as the default estimator throughout our supplementary evaluation.

## D.1. Outdoor Scene Performance on KITTI

To evaluate the generalization capability of SAGE to outdoor environments, we conduct experiments on the KITTI odometry benchmark [16]. Following the standard protocol, we measure Relative Translation Error (RTE), Relative Rotation Error (RRE), and Registration Recall (RR) as defined in Sec. C.2.

As shown in Table C, SAGE achieves competitive performance with RTE of 6.1 cm, RRE of $0 . 2 2 ^ { \circ }$ , and RR of 99.8%, demonstrating that our semantic-guided approach generalizes well to outdoor large-scale scenarios. The results confirm that semantic representations learned from indoor scenes transfer effectively to outdoor environments, validating the robustness of our method across diverse scene types.

## D.2. Scene-wise Analysis

To evaluate the generalization capability of SAGE across diverse environments, we conduct a scene-wise analysis spanning both the 3DMatch and 3DLoMatch benchmarks. Since these two benchmarks share the same set of test scenes but differ in the overlap ratio threshold used for selecting point cloud pairs, comparing results across both settings allows us to examine whether the benefit of semantic guidance is consistent regardless of the degree of overlap. This analysis also serves to clarify when and why the SG-MoE routing mechanism provides the most substantial gains.

<table><tr><td></td><td>RTE[cm]↓</td><td>RRE[°]↓</td><td>RR[%]↑</td></tr><tr><td>FCGF</td><td>9.5</td><td>0.3</td><td>96.6</td></tr><tr><td>D3Feat</td><td>7.2</td><td>0.3</td><td>99.8</td></tr><tr><td>FMR</td><td>~66</td><td>1.49</td><td>90.6</td></tr><tr><td>DGR</td><td>~32</td><td>0.37</td><td>98.7</td></tr><tr><td>SpinNet</td><td>9.9</td><td>0.47</td><td>99.1</td></tr><tr><td>Predator</td><td>6.8</td><td>0.27</td><td>99.8</td></tr><tr><td>HregNet</td><td>~12</td><td>0.29</td><td>99.7</td></tr><tr><td>CoFiNet</td><td>8.2</td><td>0.41</td><td>99.8</td></tr><tr><td>GeoTrans</td><td>6.8</td><td>0.24</td><td>99.8</td></tr><tr><td>DCATr</td><td>6.6</td><td>0.22</td><td>99.7</td></tr><tr><td>SAGE</td><td>6.1</td><td>0.22</td><td>99.8</td></tr></table>

Table C. Registration performance on KITTI.

Table D summarizes the Registration Recall (RR) for eight distinct scenes. We compare SAGE against GeoTransformer and RoITr.

The results demonstrate that SAGE achieves the high est overall RR of 87.9%, improving over RoITr (86.4%) and GeoTransformer (86.3%). A closer look at individual scenes reveals a clear pattern that is closely tied to the semantic diversity of each environment. SAGE achieves the largest gains in scenes that contain a rich variety of semantically distinct objects. For instance, in the Study scene, which typically contains diverse cluttered objects such as desks, bookshelves, monitors, and chairs, SAGE achieves 80.3% RR, significantly outperforming GeoTransformer (75.2%) and RoITr (74.3%) by over 5 percentage points. This substantial improvement can be attributed to the SG-MoE router, which assigns superpoints belonging to different object categories to specialized experts that can extract discriminative semantic-geometric representations for each category. In scenes with diverse objects, this expert specialization provides clear advantages because each expert can learn category-specific feature refinement strategies rather than relying on a single shared transformation. Similarly, in the Kitchen scene, which contains a variety of appliances, counters, and cabinets that are geometrically similar but semantically different, SAGE achieves 94.2% RR compared to GeoTransformer’s 91.7%.

On the other hand, SAGE shows relatively smaller gains or comparable performance in scenes dominated by semantically homogeneous structures. In the Hotel 3 scene, all three methods achieve similar results (85.4–86.4%), because the scene is largely composed of uniform corridors and walls that offer limited semantic diversity for the SG-MoE router to exploit. In the MIT Lab scene, SAGE (81.2%) slightly underperforms GeoTransformer (83.9%). This scene is characterized by long, narrow laboratory corridors with repetitive structural elements, where the overlapping regions are dominated by flat walls and floors that provide minimal semantic variation.

## D.3. Overlap-wise Analysis

In real-world scenarios, point cloud pairs often exhibit low overlapping areas, which significantly increases the difficulty of correspondence search. To assess the robustness of SAGE against such partiality, we evaluate the performance as a function of the overlap ratio.

We categorize the test pairs into ten bins based on their ground-truth overlap ratios, ranging from 0% to 100% with a bin width of 10%. The RR for each bin is reported in Table E.

As shown in Table E, SAGE demonstrates a clear and consistent advantage across the low-to-moderate overlap regime (10–50%), which represents the most challenging setting for correspondence-based registration. In the 10–20% bin, which contains the largest number of test pairs (1,040 pairs) and poses the greatest challenge, SAGE achieves 73.4% RR, outperforming RoITr (72.2%) and GeoTransformer (70.6%) by 1.2 and 2.8 percentage points, respectively. In the 20–30% bin, the improvement over RoITr is even more pronounced: SAGE reaches 88.7% RR compared to RoITr’s 85.5% (+3.2 points) and GeoTransformer’s 86.4% (+2.3 points). This pattern extends through the 30–50% range, where SAGE consistently achieves the highest RR. These results provide direct evidence for the effectiveness of the SG-MoE routing mechanism. In lowoverlap scenarios, the visible portion of each point cloud is limited, and the few geometric structures available for matching are often ambiguous. The SG-MoE router addresses this by leveraging semantic cues to assign each superpoint to a specialized expert, enabling the model to distinguish between geometrically similar but semantically different structures even when geometric context alone is insufficient.

In the 50–60% bin, SAGE (98.4%) is slightly below GeoTransformer (98.7%). At this overlap level, the registration problem becomes substantially easier because ample geometric context is available, and the benefit of semantic guidance diminishes accordingly. This marginal difference is consistent with our observation from the scene-wise analysis that semantic routing is most valuable when geometric cues alone are insufficient.

The relationship between overlap ratio and the relative advantage of SAGE further validates that the SG-MoE architecture is well suited for the most practically important operating regime, namely scenarios with limited overlap, which frequently arise in real-world applications such as large-scale SLAM and autonomous navigation.

\* Memory measured using PyTorch memlab. \*\* Inference time measured on NVIDIA H100.
<table><tr><td>Scenes</td><td>Kitchen</td><td>Home 1</td><td>Home 2</td><td>Hotel 1</td><td>Hotel 2</td><td>Hotel 3</td><td>Study</td><td>MIT Lab</td><td>All</td></tr><tr><td>#Pairs</td><td>1031</td><td>445</td><td>438</td><td>444</td><td>262</td><td>103</td><td>532</td><td>149</td><td>3404</td></tr><tr><td>GeoTrans.</td><td>91.7</td><td>84.9</td><td>85.4</td><td>94.1</td><td>80.2</td><td>86.4</td><td>75.2</td><td>83.9</td><td>86.3</td></tr><tr><td>RoITr</td><td>94.0</td><td>81.6</td><td>85.4</td><td>93.9</td><td>80.9</td><td>85.4</td><td>74.3</td><td>81.9</td><td>86.4</td></tr><tr><td>SAGE (Ours)</td><td>94.2</td><td>82.3</td><td>86.1</td><td>95.7</td><td>82.8</td><td>85.4</td><td>80.3</td><td>81.2</td><td>87.9</td></tr></table>

Table D. Registration Recall (%) on the 3DMatch and 3DLoMatch benchmark across different scene categories. #Pairs denotes the total number of point cloud pairs evaluated in each scene.
<table><tr><td>Overlap Ratio</td><td>0-10%</td><td>10-20%</td><td>20-30%</td><td>30-40%</td><td>40-50%</td><td>50-60%</td><td>60-70%</td><td>70-80%</td><td>80-90%</td><td>90-100%</td></tr><tr><td>#Pairs</td><td>3</td><td>1040</td><td>832</td><td>387</td><td>417</td><td>304</td><td>224</td><td>131</td><td>58</td><td>8</td></tr><tr><td>GeoTrans.</td><td>66.7</td><td>70.6</td><td>86.4</td><td>94.8</td><td>95.2</td><td>98.7</td><td>99.6</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>RoITr</td><td>66.7</td><td>72.2</td><td>85.5</td><td>94.6</td><td>94.5</td><td>98.0</td><td>98.7</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>SAGE (Ours)</td><td>66.7</td><td>73.4</td><td>88.7</td><td>95.3</td><td>96.2</td><td>98.4</td><td>99.6</td><td>100.0</td><td>100.0</td><td>100.0</td></tr></table>

Table E. Registration Recall (%) comparison across different overlap ratios. #Pairs denotes the number of point cloud pairs in each overlap bin.

<table><tr><td>Method</td><td>Params (M)</td><td>Mem. (GB)*</td><td>Time (ms)**</td><td>RR (%)</td></tr><tr><td>GeoTransformer</td><td>9.83</td><td>7.02</td><td>75</td><td>74.0</td></tr><tr><td>RoITr</td><td>10.10</td><td>9.28</td><td>140</td><td>73.4</td></tr><tr><td>SAGE (Ours)</td><td>121.71</td><td>16.69</td><td>179</td><td>76.3</td></tr></table>

Table F. Computational Efficiency Analysis

## D.4. Computational Efficiency

Table F shows that SAGE requires approximately 12.4× more model parameters and 2.4× more memory than Geo-Transformer, due to introducing Sonata and MoE. While inference time increases to 179ms, this overhead is comparable to RoITr (140ms), which also employs sophisticated architectures for improved performance.

SAGE has 121.71M total parameters, which is approximately 12.4 times larger than GeoTransformer (9.83M). However, the majority of this increase originates from the frozen Sonata encoder (108.41M parameters), which requires no gradient computation during training or inference. Excluding Sonata, SAGE contains only 13.3M parameters; the SG-MoE routing networks and expert FFNs add approximately 3.5M parameters beyond the GeoTransformer backbone, representing only a 35% increase.

Regarding inference time, SAGE requires 179ms per point cloud pair, comparable to RoITr (140ms). In terms of performance gain per unit of additional latency over GeoTransformer, SAGE achieves 2.3 percentage points of RR improvement for 104ms of additional latency, whereas RoITr adds 65ms without improving RR (73.4% vs. 74.0%), indicating that SAGE utilizes the additional computation more effectively.

It is worth noting that this computational overhead is not specific to SAGE, but rather reflects a fundamental cost of incorporating pre-trained semantic encoders into the registration pipeline. SAGE mitigates this inherent cost by freezing the encoder entirely and keeping the method-specific trainable overhead minimal. The peak memory consumption of 16.69 GB fits within commonly available GPUs such as the NVIDIA RTX 4090 (24 GB), making SAGE practical for offline applications that prioritize accuracy, such as 3D reconstruction for cultural heritage preservation and architectural surveying. As more lightweight point cloud foundation models emerge, the frozen encoder could be replaced with a more compact alternative without modifying the SG-MoE architecture itself.

## D.5. Additional Ablation Study

We conduct additional ablation studies to further validate the key design choices in SAGE. Specifically, we investigate three critical questions: (1) Should the semantic encoder be adapted during training? We examine whether fine-tuning Sonata with adapters improves performance or causes geometric shortcuts. (2) How should semantic features guide expert routing? We compare our query-side routing against cross-attention-based alternatives. (3) How many experts are optimal? We vary the number of experts from 2 to 6 to understand the capacity-efficiency trade-off. All experiments are conducted on 3DLoMatch with LGR, and results are summarized in Table G.

Sonata with Adapter. As shown in Table G(top), adding adapters to Sonata (w/ adapter) achieves 74.8% RR, showing a notable degradation (−1.5 points) compared to the fully frozen Sonata baseline (76.3% RR). This result demonstrates that allowing adaptation of the semantic encoder, even with lightweight adapter modules, causes the model to fall into a geometric shortcut [43]. Specifically, the registration loss functions, which optimize primarily for geometric alignment accuracy, drive the adapted semantic features to collapse toward purely geometric patterns that are more directly useful for minimizing registration errors. This geometric shortcut phenomenon causes the semantic representations to lose their ability to distinguish between geometrically similar but semantically different structures (e.g., TV corners vs. desk corners), which is precisely the discriminative power that Sonata was pre-trained to preserve.

<table><tr><td>Configuration</td><td>FMR IR</td><td>RR</td></tr><tr><td>Sonata Adaptation</td><td></td><td></td></tr><tr><td>w/o adapter w/ adapter</td><td>89.4 44.3 86.4 42.6</td><td>76.3 74.8</td></tr><tr><td></td><td>Routing Strategy</td><td></td></tr><tr><td>SG-MoE (ours)</td><td>89.4 44.3</td><td>76.3</td></tr><tr><td>SCAR</td><td>88.1 44.4</td><td>75.0</td></tr><tr><td></td><td>Number of Experts</td><td>74.9</td></tr><tr><td> $N _ { \mathrm { e x p e r t } } = 2$   $N _ { \mathrm { e x p e r t } } = 3$ </td><td>88.0 42.3 89.3 41.8</td><td>75.1</td></tr><tr><td> $N _ { \mathrm { e x p e r t } } = 4$ </td><td>89.4 44.3</td><td>76.3</td></tr><tr><td> $N _ { \mathrm { e x p e r t } } = 5$ </td><td>88.5 43.6</td><td>75.5</td></tr><tr><td> $N _ { \mathrm { e x p e r t } } = 6$ </td><td>88.9 43.0</td><td>75.7</td></tr></table>

Table G. Additional ablation studies on 3DLoMatch with LGR. All metrics in $\%$ . We evaluate: (1) adding adapters to Sonata, (2) alternative routing strategy using cross-attention (SCAR), and (3) varying the number of experts.

The degradation in FMR (89.4% → 86.4%) and IR (44.3% → 42.6%) further confirms that the adapted features become less effective at establishing reliable correspondences in low-overlap scenarios where semantic understanding is crucial. This validates our design choice of keeping Sonata completely frozen throughout training, ensuring that the semantic representations maintain their original semantic discriminability and do not degenerate into redundant geometric features.

Alternative Routing Strategy: SCAR. We explore an alternative routing mechanism called Semantic Cross-Attention Routing (SCAR), where expert selection is determined by cross-attention between source and target semantic features rather than using only the query point cloud’s semantic features.

As illustrated in Fig. A, SCAR computes routing probabilities by first performing cross-attention between the semantic features of the two point clouds. Specifically, for each superpoint in P, we compute cross-attention scores with all superpoints in Q using their frozen semantic representations $\hat { \mathbf { F } } _ { s } ^ { \mathcal { P } }$ and $\hat { \mathbf { F } } _ { s } ^ { \mathcal { Q } }$ :

![](images/135b44aa59787f162169c32bcb3cc8b5c8f9991b3d90c9f1cd918bfc77813544.jpg)  
Figure A. The Overview of SCAR. Semantic Cross-Attention Routing (SCAR) first computes cross-attention between source and target semantic features, then uses the attended features for routing. While SCAR incorporates inter-point-cloud information earlier, this premature soft matching introduces noise in lowoverlap scenarios, leading to inferior performance (76.3% vs. 75.0% RR on 3DLoMatch).

$$
\mathbf { A } _ { i , j } = \mathrm { S o f t m a x } \left( \frac { \hat { \mathbf { f } } _ { s , i } ^ { \mathcal { P } } ( \hat { \mathbf { f } } _ { s , j } ^ { \mathcal { Q } } ) ^ { T } } { \sqrt { d _ { s } } } \right) ,\tag{24}
$$

and then aggregate the attended features to compute routing probabilities:

$$
\mathbf { s } _ { i } ^ { \mathrm { { S C A R } } } = { \mathrm { S o f t m a x } } \left( { \mathrm { M L P } } _ { r } \left( \sum _ { j = 1 } ^ { { \hat { N } } _ { Q } } \mathbf { A } _ { i , j } \cdot { \hat { \mathbf { f } } } _ { s , j } ^ { \mathbf { \alpha } } \right) \right) .\tag{25}
$$

The key difference from our SG-MoE is that SCAR routes based on cross-attended features that already incorporate information from the target point cloud, while SG-MoE routes purely based on the query point cloud’s semantic features.

As shown in Table G(middle), SCAR achieves 75.0% RR, underperforming our vanilla SG-MoE by 1.3 points despite maintaining similar IR (44.4% vs. 44.3%). This degradation suggests that incorporating inter-point-cloud relationships into routing introduces premature correspondence assumptions before the experts have processed the features. The cross-attention operation in SCAR effectively performs a soft matching between source and target superpoints to guide routing, which may bias the routing decisions toward incorrect correspondences in challenging low-overlap scenarios.

<table><tr><td>Perturbation</td><td>10% 20% 30% 40% 50%</td></tr><tr><td>Gaussian noise</td><td>76.4 75.7 75.6 76.3 76.2</td></tr><tr><td>Feature dropout</td><td>76.4 76.1 75.0 75.8 76.2</td></tr><tr><td>Random replacement</td><td>75.8 75.1 74.8 74.8 73.5</td></tr></table>

Table H. Robustness to semantic feature degradation. Registration Recall (%).

In contrast, our SG-MoE design keeps routing decisions independent for each point cloud, based solely on its own semantic characteristics. This allows experts to first specialize in processing semantic-geometric patterns within each point cloud, and then establish correspondences through the subsequent cross-attention layers where both semantic and refined geometric information are available. Since correspondence information is already captured and refined in these subsequent cross-attention layers, introducing it prematurely in the routing stage provides no benefit and instead adds noise. Our design of using only query-side semantic features for routing is simpler, more robust, and more effective.

Number of Experts. We vary the number of experts $N _ { \mathrm { e x p e r t } }$ from 2 to 6 to study its impact on performance. As shown in Table G(bottom), performance generally improves as we increase the number of experts from 2 to 4, with the best results at $N _ { \mathrm { e x p e r t } } = 4 ( 7 6 . 3 \% \mathrm { R R } )$ . Beyond 4 experts, performance degrades (75.5% RR at $N _ { \mathrm { e x p e r t } } = 5 .$ , 75.7% RR at $N _ { \mathrm { e x p e r t } } = 6 )$ , likely due to over-specialization and insufficient training samples per expert.

With only 2 experts (74.9% RR), the model lacks sufficient capacity to capture diverse semantic-geometric patterns. Three experts (75.1% RR) provide some improvement but still underperform the optimal configuration. The optimal choice of $N _ { \mathrm { e x p e r t } } = 4$ strikes a balance between model capacity and training efficiency, allowing each expert to specialize in distinct semantic categories (e.g., planar structures like walls and floors, furniture, fixtures, and small objects) without over-fragmenting the feature space. Dependency on Semantic Quality. We evaluated SAGE’s robustness by injecting three types of perturbations into semantic features on 3DLoMatch.

As shown in Table H, SAGE maintains stable performance (RR >75%) under 50% noise or dropout. This robustness stems from the implicit regularization within our SG-MoE; through training, the router’s MLP learns to bypass feature noise and prioritize stable semantic cues.

Specifically, while Gaussian noise and feature dropout represent unstructured degradations that the router can learn to filter as out-of-distribution noise, random replacement introduces structured, semantically plausible but incorrect features. These “adversarial” semantic cues can mislead the router into assigning superpoints to experts specialized in irrelevant semantic contexts, providing them with a high but erroneous confidence. This structural confusion disrupts the correspondence search more severely than simple noise, as it actively guides the geometric refinement process toward semantically inconsistent regions, resulting in the observed gradual performance decline as the replacement ratio increases.

## D.6. Qualitative Visualization

To explicitly demonstrate the robustness of SAGE in challenging scenarios, we provide qualitative comparisons against the baseline GeoTransformer on the 3DLoMatch benchmark. Fig. B visualizes the registration results on lowoverlap pairs.

As observed in columns (b) and (c), GeoTransformer struggles to establish reliable correspondences in lowoverlap regions, often resulting in low Inlier Ratios (IR) and incorrect registrations (high RMSE). In contrast, SAGE successfully identifies correct correspondences (shown in green lines) by leveraging semantic cues from the Sonata foundation model, leading to accurate alignment even when the geometric overlap is limited $( \mathrm { e . g . , \textless 2 0 \% ) }$ . Specifically, SAGE effectively utilizes distinct semantic objects (e.g., furniture, fixtures) to anchor the registration, whereas the geometry-only baseline fails due to ambiguous geometric structures.

## D.7. Expert Routing Analysis

To further analyze the semantic specialization of SG-MoE, we visualize the expert assignment maps across six diverse scenes sampled from both 3DMatch and 3DLoMatch in Fig. D. Each superpoint is color-coded by its assigned expert (red, blue, green, yellow), and bounding boxes highlight semantically salient regions within each scene.

The visualization reveals that each expert develops a consistent functional role across all six scenes. Expert 4 (yellow) consistently handles large, semantically prominent structures such as sofas, beds, and countertops, which serve as strong anchors for registration due to their distinctive geometry and semantic identity. Expert 2 (blue) specializes in mid-scale objects such as chairs, shelves, and desks that provide complementary cues for correspondence establishment. Expert 1 (red) is assigned to smaller, distinctive regions that are less dominant in terms of spatial extent but carry unique semantic signatures. Expert 3 (green) primarily handles low-saliency background structures such as walls and floors, which are geometrically common across scenes and offer limited discriminative power for registration.

This consistent specialization pattern emerges without

RMSE:4.51

![](images/923a3a56ed7a7b990d6c97f262fd97b2dc67cdaddee5352e9e90516f7ef2953b.jpg)  
IR: 22.9%

![](images/74798bd0c95a227dca81b8b2d2fe2188c4345eaf7937f0349b031b5bc7f3ec10.jpg)  
IR:56.7%

![](images/8cb20b11dfa1a6e205805c3200c8dc3aa928b9e69b5d8e1c75bc538a0f3a4c8a.jpg)  
RMSE: 4.71

![](images/4782ec0e067dc3a5c186044600616c88050826d404c8e63ed417fc700195d638.jpg)  
RMSE : 0.03

![](images/77b681b1306aa28890581c80b5f3815db202bccec520b95df8a33014a4d6a9bf.jpg)  
Overlap : 24.4%  
IR:14.0%

![](images/81b1240cedc780851ec7e3b7e5a9b4597611f85b83b6b04118a6ea88eee82c91.jpg)

![](images/de4512413ae76dc9e50ac9b7bee7879ecd5079a970226cafe4c70601d7ce94d2.jpg)  
IR: 48.0%  
RMSE: 2.20  
RMSE : 0.02  
Overlap : 24.9%

![](images/ec3266fc48730be3f2a0334e664045ea0ff277dc48f9e908bef69959f4e7684e.jpg)

![](images/13451bdefebb320b6394d238b0c89befdfa05d0194ad6f2579f90b4d3778ecca.jpg)  
IR: 27.3%

![](images/e2c38c3420ce195668432d4afd6ac0af83d297c0d61e37261dfcced89f5c16b9.jpg)

![](images/92a13ce9f59a0e1e3675d95bb95c7953a1262d2373c1c805ddf729b7c17474be.jpg)

![](images/40b4e2d464bbc22842e49e261de003317d6d1717c3882b85e79cc53e95385c45.jpg)  
IR: 64.7%  
RMSE : 1.09  
RMSE: 0.01  
Overlap : 28.9%

![](images/0425f37e2be05dbc8fa8236906a6300b8270b104b711abeacf2c46777403c432.jpg)

![](images/5f47a5941bc9837ac9240de36de33f36ca103dda32b67c6a92b6fac408152a48.jpg)

![](images/87b4e2e4c1affdb0b83a9779d65f0217b5e47396a03f294f619e72875274f76e.jpg)

![](images/711a2f7623858745bf3790a988a26600d96cc78d2d696e4afcb20081c9564e33.jpg)

![](images/c87d5be1d98c6b9824e8bf3e435ac79c5598ce5cff1fc9ed47c49861f31b200b.jpg)  
IR : 2.2%

![](images/0294a4b4ec0d2e4b4f8bf0b638bf0a454bfa1f6592f4e2610dbb2c11f55bd177.jpg)  
IR: 39.1%

![](images/d4ded956c210672c98268163c473f7f393e3fe4f5d02317c6cd4e018f8a26a17.jpg)  
RMSE: 4.51

![](images/9afe6da2c6858f196a7e90f3b273fe4e998b8311de17b453da8217681d8e3021.jpg)  
RMSE: 0.05

![](images/8244f0ca46a6e913c45060e4543cbbfb974f48ca9caeea8033e3b087eb227ad6.jpg)  
IR: 2.2%

![](images/3de35ae8772140864042c20c098b132b4252072f82e17ac1aca3e46cbe9dc91d.jpg)

![](images/37f7a07e01d4d5d98a4cf19ea6d7e519211a32ddfbd3602c2acf55a0558fa4a9.jpg)  
Overlap : 14.2%  
IR:39.1%

![](images/b681ffea337ff36e3f4db4454c3a7cc4125e091d2e954781aee2e46dd5f9c20f.jpg)

![](images/758fa5aad3319c89d5fb80177362875349c93d8fd43a3a988096924d2b4fc848.jpg)  
RMSE: 0.05

![](images/4233aba2344b570eb20eff9dd11606103d5a420bebdcba52aa4037d54ac25478.jpg)

![](images/485549692fac2cd1a7c68fedefec7278cd5b0605d2c9555e27e3a0dc002fe36b.jpg)

![](images/5c44f8845b7979b2f192ec89979094ffe31970aeabb0a5d1ea18e52ec560c529.jpg)  
IR: 48.7%

![](images/cb51dab04c0e3de9b51e37591657ea6d6ba57a183eb19ff4a8df3362463728b1.jpg)  
Overlap : 14.2%  
RMSE:4.67  
RMSE : 0.04

![](images/9dfb73fbdd1f4ce1a763ce21839aaef3173cc942c958b41a1977329ee4f40c4c.jpg)

![](images/089741a260cb5ce6ffbc4daca71ac5892f44a6364ed1cedac4ba4bc0983eb8b7.jpg)  
Overlap : 19.0%

![](images/cfe3eabab269370dda5e6bcbbcafbbd2f87024cf919ed8fcd731471d8f9d3148.jpg)  
IR : 6.4%

![](images/f6b6925eb7c43a2f09296c694d95183ccc1d706b44c3253ded90c70a24c4fd61.jpg)  
IR: 49.1%

![](images/08e0d127674a4be8db672063dee2ec9aea8a783741336e3579eed81cd63746e0.jpg)

![](images/e9553d0c245a2cfd0270ae036f82a9aa42af309c15e3d7b6bd0644b8daf790de.jpg)  
RMSE: 3.81  
RMSE : 0.03

![](images/33748a2e92a3ea935be80cc925838ecb49e99ef47c9ed2d1b90057b00b9b6d69.jpg)

![](images/60287387201f42846a277584c8487bda2f8b4c31e0db3ceec63d1314b44a8120.jpg)  
(a) input

![](images/394a3c8b32525cee49560ff1df6f8cfd27887ff312bd8d4f3774000ff31e79f7.jpg)  
(b) GeoTransformer correspondences

![](images/2c7c50619ea5bcad7f3e2dc6e754594ced1ade191e7122c8f6750ccfde3993cc.jpg)  
Overlap : 21.2%

(c) SAGE (ours) correspondences  
![](images/88cf1818efcf2f6f755a07f44cf42a93e70324b2a059f1b12a4289713659d1c0.jpg)

(d) GeoTransformer results  
![](images/70714050868c4d61cd352937b13bb8d288f5bba4568164dd9a31b1710ea29896.jpg)  
(e) SAGE (ours) results

![](images/6a68117b3e306b0656cdc36d1fa8a754db5cd63a15e58f5985687460c9110708.jpg)  
(t) GT alignment

Figure B. Qualitative comparison on 3DLoMatch. We compare the registration quality between GeoTransformer (baseline) and SAGE (ours) on challenging low-overlap pairs. From left to right: (a) Input source (yellow) and target (blue) point clouds. (b) Correspondences generated by GeoTransformer with IR. (c) Correspondences generated by SAGE with IR. (d) Registration result of GeoTransformer with RMSE. (e) Registration result of SAGE with RMSE. (f) Ground truth alignment with overlap ratio. SAGE consistently achieves accurate registration (low RMSE) even in cases where GeoTransformer completely fails (e.g., 3rd and 4th rows).

![](images/745ab88867101be1ef2a7464eae7760abb04786888dba0ae615c16005d7ce6bf.jpg)

![](images/c72d06eb062ce978f0d7b4d205af7dbc60cdde0ecdc06876a252ea60a9a5eed3.jpg)

![](images/99fa8cfb5af589902a66cbefcc9f6ce8d10183c0969441c789e275d3eaa74c7b.jpg)  
(a) Source  
(b) Target

![](images/3993b6b157d04c8157821ab3229f8d0257dbc63ea858da699c02425e658ddb95.jpg)

![](images/f4f9bced91f35b54d82fecb604fb8f27e8e6fc26f181978965d388a458196632.jpg)  
(c) SAGE (ours) correspondences  
(d) SAGE (ours) results  
(e) GT alignment  
Figure C. Additional failure cases on 3DLoMatch. Each row shows a challenging scenario where SAGE fails to achieve accurate registration. From left to right: (a) Source point cloud (cyan), (b) Target point cloud (yellow), (c) SAGE correspondences with Inlier Ratio (red lines: outliers, green lines: inliers), (d) SAGE registration result, (e) Ground truth alignment. These failures occur due to: (1) severe occlusion reducing the overlapping area, (2) geometrically indiscriminative regions (uniform walls, floors, ceilings), and (3) severe truncation of semantic regions, preventing the semantic encoder from extracting reliable semantic representations. When semantic structures are incomplete (e.g., only partial views of furniture), the frozen Sonata encoder cannot provide discriminative semantic features, as it was trained on complete object instances.

any explicit supervision on expert roles; it arises purely from the frozen Sonata semantic features that guide the routing decisions. The fact that this functional division is preserved across scenes with substantially different layouts and object compositions demonstrates that SG-MoE routing generalizes beyond individual scene characteristics.

Importantly, within each scene pair, the source and target point clouds exhibit consistent expert assignments for the same semantic regions. For instance, when a sofa appears in both the source and target scans, it is routed to Expert 4 (yellow) in both cases, even though the two scans are captured from different viewpoints and contain different degrees of occlusion. This source-target consistency is essential for registration because the SG-MoE cross-attention layer processes the source and target features through their respective assigned experts. When both sides of a correspondence are handled by the same expert, the resulting feature representations share a common specialized subspace, which directly improves the quality of correspondence matching. Conversely, if source and target regions of the same object were routed to different experts, the resulting features would lie in incompatible subspaces, degrading the matching accuracy. The observed consistency confirms that the frozen semantic representations provide a stable routing signal that is invariant to the viewpoint from which a scene is observed.

![](images/f822397c33e575de98d7ad4bedfc36f2d714f30d5311ef57c23171f9baba0785.jpg)  
Figure D. Expert routing patterns of SG-MoE across six scenes from 3DMatch and 3DLoMatch. Each scene is shown as a source-target pair, where superpoints are color-coded by their assigned expert (red, blue, green, yellow) and bounding boxes highlight semantically salient regions. The same semantic regions in the source and target point clouds are routed to the same expert, ensuring compatible feature representations for correspondence matching. This consistent functional specialization emerges across diverse scenes without explicit supervision, confirming that the frozen semantic features enable scene-agnostic and view-invariant expert routing.

## E. Baseline Reproduction Details

To make our comparison transparent and reproducible, this section describes how we obtain the baseline results in the main paper. All methods compared under the matched protocol (PEAL [51], PSReg [19], and SAGE) are trained from scratch on the same NVIDIA H100 GPU and share the same training configuration as SAGE (Sec. B.4): the Adam optimizer with an initial learning rate of $1 0 ^ { - 4 }$ multiplied by 0.95 after each epoch, weight decay 10<sup>−6</sup>, batch size 1, and random seed 7351. They also share the KPConv-FPN backbone (2.5 cm initial voxel size, four stages), 128 coarse targets, 256 putative superpoint correspondences, and the geometric attention configuration (4 heads, $\sigma _ { d } = 0 . 2 , \sigma _ { a } =$ 15<sup>◦</sup>). Keeping these factors fixed isolates the effect of the routing design rather than confounding it with training or

backbone differences.

PEAL. The PEAL [51] code is publicly released, so we build our PEAL baseline directly on the official repository. PEAL conditions its attention on an overlap prior. Although the prior data is released, the procedure used to generate it is not documented, which prevents us from reproducing it exactly. We therefore regenerate the overlap prior from the publicly released GeoTransformer [26] pre-trained model and train PEAL in our own environment under the shared configuration above. We use a budget of six refinement iterations, matching the iterative protocol used for SAGE and PSReg.

PSReg. The official implementation of PSReg [19] is not publicly released, and several key hyperparameters, most notably the number of experts, are not reported in the paper. We therefore re-implement PSReg on the same framework as SAGE and train it from scratch. Following the design described in the PSReg paper, expert routing is driven by geometry-derived overlap information obtained from Geo-Transformer [26], with top-1 selection. To keep the comparison controlled, we set the number of experts to 4, identical to SAGE; all remaining settings follow the shared configuration above.

On the reproduced numbers. Our reproduced results for PEAL and PSReg fall below the values reported in their original papers. We attribute this to the absence of released pre-trained models, which forces retraining from scratch, and to the undocumented prior generation in PEAL and expert configuration in PSReg, which require our own implementation. Because these factors cannot be controlled by transcribing the originally reported numbers, we report all methods under a strictly matched protocol instead, so that the differences reflect the registration design rather than unspecified training conditions.

Difference from the reproduced PSReg. SG-MoE (Sec. A.4) and our PSReg re-implementation share the same backbone, attention, expert count (4), and top-1 routing; the only difference is the routing signal. PSReg fuses geometry-derived overlap information into the router input, whereas SG-MoE routes solely on the frozen semantic features while the experts still refine semantic-geometric features. The matched comparison in the main paper (Tables 1 and 2) therefore isolates the contribution of semanticguided routing.

## F. Limitations

While SAGE demonstrates significant improvements in low-overlap point cloud registration, several limitations warrant discussion for future work.

Dependency on Pre-trained Semantic Models. SAGE relies on the quality and generalization capability of the frozen Sonata encoder. The semantic degradation experiments in Table H quantify this dependency: while SAGE is robust to unstructured noise (76.2% RR even at 50% Gaussian noise), random replacement of semantic features with plausible but incorrect ones causes a more pronounced decline to 73.5% RR at 50% replacement. This indicates that the SG-MoE router is sensitive to structured semantic errors that mimic valid features, which could arise in practice when the encoder encounters object categories or scene types underrepresented in its training data (e.g., industrial equipment, medical instruments). In such domains, Sonata may produce semantically ambiguous representations that mislead the router into suboptimal expert assignments. As more diverse point cloud foundation models become available, replacing or complementing Sonata with encoders trained on broader data distributions could mitigate this limitation.

Limited Effectiveness in Semantically Homogeneous Scenes. The scene-wise analysis in Table D reveals that SAGE’s improvement over baselines is directly tied to the semantic diversity of each scene. In the MIT Lab scene, which consists of narrow corridors with repetitive walls and floors, SAGE (81.2%) slightly underperforms Geo-Transformer (83.9%), and in Hotel 3, all three methods achieve similar results (85.4–86.4%). The overlap-wise analysis in Table E further corroborates this trend: at 50– 60% overlap, where geometric context is already sufficient, SAGE (98.4%) is marginally below GeoTransformer (98.7%). These results indicate that when overlapping regions are dominated by uniform planar structures that lack semantic variation, the SG-MoE router cannot assign superpoints to meaningfully different experts, and the additional routing overhead provides no benefit. Fig. C illustrates concrete failure cases where this limitation manifests: severe occlusion leaves only featureless walls and floors in the overlapping area, and the truncation of semantic objects (e.g., only a partial view of a desk) prevents the frozen encoder from extracting discriminative representations. We note that these scenarios are inherently challenging for all correspondence-based registration methods, as the lack of distinctive features is a fundamental obstacle that is not specific to semantic approaches. Addressing robust registration in such low-texture, semantically homogeneous environments through temporal cues from sequential scans or multi-view constraints remains an important open problem for the community.

Despite these limitations, SAGE establishes a strong foundation for incorporating semantic understanding into point cloud registration, and the analyses presented in this work clarify the conditions under which semantic guidance is most and least effective, providing concrete guidance for future research in semantic-geometric fusion for 3D vision tasks.