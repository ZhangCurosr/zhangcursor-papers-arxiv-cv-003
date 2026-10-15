# Train to Stitch: Topology-Preserving Regularization for Interoperable Representations

Akshay Malhotra<sup>1⋆</sup>, Nicolas Zilberstein<sup>2∗</sup>, and Shahab Hamidi-Rad<sup>1</sup>

<sup>1</sup> InterDigital Communications Inc., Los Altos, CA, USA 2 Rice University, Houston, TX, USA

Abstract. As large, powerful pre-trained models become more prevalent, there is growing interest in developing methods that allow diferent neural networks to be seamlessly combined with little or no direct communication between them. Take autoencoder-based data compression as an example: the ability to independently choose architectures and train encoder and decoder models, while still ensuring they work together, could fundamentally transform how such models are built, deployed, and updated. In this work, we introduce a novel approach that leverages topological regularization to align the latent spaces of two independently trained autoencoder models, entirely eliminating the need for coordination during training. We provide a conditional theoretical analysis showing how the structural agreement encouraged by our regularizer can translate into approximate geometric compatibility between independently learned representations. Reconstruction experiments validate this approach empirically, substantially narrowing the interoperability gap across multiple datasets.

## 1 Introduction

Compressing data into compact, meaningful representations is a fundamental problem with broad applications spanning image compression [3, 27], audio compression [13], wireless communication [30, 15], and beyond. A common neural network approach trains an encoder-decoder pair end-to-end, where the encoder maps the data into a low-dimensional latent space, and the decoder reconstructs the original input from this latent representation. In many practical settings, however, encoders and decoders cannot be trained jointly. For example, a clientside encoder may face strict hardware or latency constraints while the server-side decoder is updated frequently, or separate organizations may build their own encoders and decoders without sharing model weights or training data. These scenarios call for interoperable designs: encoder and decoder components that are trained independently yet can be connected at deployment time with little or no coordination.

This challenge has been explored under a particular instantiation of this problem, termed zero-shot stitching [12]: given two independently trained autoencoders, the goal is to find a transformation between latent spaces from only a small calibration set of shared inputs. An early work [29] proposed mapping each autoencoder’s latent space into a shared relative representation. A key limitation, however, is that this requires training a dedicated decoder in the relative space, which demands substantial amounts of data. Other recent approaches [17, 25] have investigated whether a simple afine mapping can align independently learned latent spaces. While promising, a notable accuracy gap persists compared to jointly trained autoencoders, largely because each encoder’s nonlinear mapping is shaped by its architecture, optimization hyperparameters, and random initialization, producing latent geometries that can difer considerably across models. This efect is exacerbated in the small-data regime, and closing the gap requires large calibration sets, contradicting the premise of minimal interaction and limited shared data.

To close this gap, we take a complementary approach: rather than relying on the existence of a simple post-hoc, linear map, we introduce a training-time inductive bias that encourages independently trained models to produce geometrically compatible latent spaces. Our key insight is that many alignment assumptions can be understood through the lens of (approximate) isometry: if two point clouds are nearly distance-preserving copies of the same underlying structure, then an orthogonal alignment is well-justified (Section 3.2). This motivates regularizing training so that each latent space preserves the same geometric skeleton of the data.

To instantiate this idea, we build on topological autoencoders [28] and employ a diferentiable topological loss that aligns 0-dimensional persistence features between a target space and a learnable space.<sup>3</sup> Concretely, this topological loss encourages agreement of minimum spanning tree (MST) edge lengths between two spaces by leveraging the persistence pairings in each space. Intuitively, MST edges form a sparse topological backbone that captures global connectivity; matching this backbone across models yields latent spaces that are less sensitive to architectural idiosyncrasies and hence easier to stitch.

Building on this foundation, we propose a strategy that regularizes each autoencoder so that its latent space preserves the topological structure of the input space, requiring no information exchange between model owners (Fig. 1). After training, we fit a simple linear transformation using only a small calibration set. We further provide a theoretical analysis explaining why this regularization yields compatible latent spaces. We validate our approach through several numerical experiments spanning image compression and channel compression and reconstruction.

Contributions. The main contributions of this paper are summarized as follows:

1. Topology-preserving training for interoperability. We introduce a topological regularization framework for independently trained autoencoders that promotes latent-space compatibility using only 0-dimensional persistence features (MST edges). This loss requires no model-to-model interaction during training and enables post-training model stitching via a simple linear map learned from a small calibration set.

![](images/b43c502a1c4308d2bc867a365b32d0845178e97f6059238fa4c42dd3438dd14f.jpg)

(a) Model 1  
(b) Model 2  
![](images/385f9bec3069415a81205acf0e794f87c6aa91a4855abcf7fa3ede6e621ed860.jpg)  
(c) Latent Space Alignment  
Fig. 1: Proposed scheme: (a-b) Topological autoencoders. We regularize the training of each autoencoder $\mathrm { A E _ { 1 } , A E _ { 2 } }$ with the input data, preserving its topological structure in the latent space independently of the architecture of the encoder. We build graphs in both input and latent spaces, and we leverage the topological regularization to preserve the underlying connectivity. Notice that each autoencoder maps the input data to a diferent latent space, but both latent spaces preserve the topology up to a rotation and scaling. c) Stitching between two regularized autoencoders. The transformation is simplified, leading to an afine transformation.

2. Theoretical motivation for linear stitching. We show that reference-MST edge agreement controls disagreement between the induced tree-path metrics. Under the additional assumption of bounded tree stretch, we derive a bound on Euclidean pairwise-distance disagreement, providing a geometric motivation for fitting a low-complexity linear connector (Section 3.2).

3. Extensive multi-domain evaluation. We validate our approach on reconstruction and retrieval tasks across heterogeneous architectures. Our method demonstrates substantial reductions in stitching error compared to baseline direct alignment and relative representations, achieving near-optimal performance across multiple datasets.

## 2 Related Work

This work is primarily concerned with the interoperability of independently trained autoencoders, building on two lines of research: zero-shot stitching and geometric regularization of autoencoders.

## 2.1 Zero-shot stitching

Relative representation [29] is one of the most popular papers in zero-shot stitching. The work builds on the observation that the representations learned by different neural networks trained on the same data are related via conformal maps; in other words, the angles between latent embeddings are preserved. Based on this observation, they propose to map the latent space of each autoencoder to a pre-defined relative representation, which is invariant by construction to the transformations induced by stochastic factors in the training process. More precisely, they select at random a set of anchor elements $\mathcal { A } \in \mathcal { D } _ { t r a i n } ;$ every sample in the anchor set is represented concerning the embedded anchors $\begin{array} { r } { e _ { \pmb { a } ^ { ( j ) } } = \mathcal { E } \left( \pmb { a } ^ { ( j ) } \right) } \end{array}$ Then, the relative representation is defined as

$$
\begin{array} { r l } & { \pmb { r _ { x ^ { ( i ) } } } = \big ( \sin ( \pmb { e _ { x ^ { ( i ) } } } , \pmb { e _ { a ^ { ( 1 ) } } } ) , \sin ( \pmb { e _ { x ^ { ( i ) } } } , \pmb { e _ { a ^ { ( 2 ) } } } ) , } \\ & { \qquad \ldots , \sin ( \pmb { e _ { x ^ { ( i ) } } } , \pmb { e _ { a ^ { ( | A | ) } } } ) \big ) } \end{array}\tag{1}
$$

where sim : $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \to \mathbb { R }$ is a similarity function, yielding a scalar score r between two absolute representations r = sim $( e _ { x ^ { ( i ) } } , e _ { x ^ { ( j ) } } )$ . In particular, they consider the cosine similarity. Notice that the number of anchor points gives the dimensionality of this new similarity space; therefore, it is expected to improve the performance when increasing the number of anchor points. This proposed method requires a specialized decoder trained on the relative representation.

While in the original paper, the authors considered only the cosine similarity, other similarity functions can be incorporated [6, 5]. A recent work [11] proposes a method to do zero-shot stitching that leverages functional maps, i.e., by learning a linear transformation in the spectral domain.

A closely related line of work [17, 25] studies post-hoc alignment of independently learned latent spaces using simple parametric maps, enabling “zero-shot” translation between models without retraining the encoders/decoders.

## 2.2 Autoencoders via geometric regularization

This work focused on topological autoencoders; however, several previous works incorporate a geometric regularization [19]. Many of them build on the notion of curvature of a generator [2]. In [20], the authors propose first to build a neighborhood graph on the input data and use it to regularize the geometry and connectivity of the learned manifold. Remarkably, they propose a local approximation of the decoder instead of the encoder to extract local geometric information on the decoded manifold. In [31], a regularization based on conformal mapping, i.e., a mapping that preserves angles, is used to enforce the preservation of angles and relative distances between data and latent space. In [10], inspired by manifold learning, a regularization based on spectral embeddings is incorporated, encouraging the learned latent representation to align with the data geometry. Recently, a graph-preserving autoencoder was proposed in [21], which aims to preserve a pre-build Laplacian between the input data and the latent space.

## 3 Alignment with topological constraints

## 3.1 Problem formulation

We consider two autoencoders, $\mathrm { A E } _ { 1 } = \left( E _ { 1 } , D _ { 1 } \right)$ and $\mathrm { A E } _ { 2 } = \left( E _ { 2 } , D _ { 2 } \right)$ , where $E _ { i }$ and $D _ { i }$ denote the encoder and decoder of autoencoder i, respectively. Both autoencoders are trained independently on the same data distribution; that is, they may difer in architecture, initialization, and optimization trajectory. Given an input $x \in \mathbb { R } ^ { l }$ , each encoder produces a latent code $z _ { i } = E _ { i } ( x ) \in \mathbb { R } ^ { d }$ . Our goal is to enable interoperability between these models: connecting the encoder of one with the decoder of the other. Formally, we seek to perform zero-shot stitching; that is, cross-model decoding of the form $D _ { 2 } ( T ( E _ { 1 } ( x ) ) )$ , or symmetrically $D _ { 1 } ( T ( E _ { 2 } ( x ) ) )$ , where $T ( \cdot ) \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ is an afine transformation learned from only a small calibration set with minimal information exchange between the two models.

Direct alignment between latent spaces. Recent works [17, 25] proposed to directly align the latent spaces by fitting a transformation $T$ using a small dataset $D _ { T }$ of shared inputs. Denoting the latent embeddings of the calibration set as

$$
\mathbf { Z } _ { 1 } : = \{ E _ { 1 } ( x ) \mid x \in D _ { T } \} , \qquad \mathbf { Z } _ { 2 } : = \{ E _ { 2 } ( x ) \mid x \in D _ { T } \} ,
$$

the transformation is estimated by minimizing the mean squared discrepancy between the two point clouds:

$$
T ( \cdot ) = \arg \operatorname* { m i n } _ { T ( \cdot ) \in \mathcal { T } } \| \mathbf { Z } _ { 1 } - T ( \mathbf { Z } _ { 2 } ) \| ^ { 2 } .\tag{2}
$$

These methods typically restrict $\tau$ to afine or orthogonal maps, which offers simplicity but leaves a significant performance gap compared to end-to-end training. This suggests that the latent spaces produced by unconstrained, independent training are not well-aligned by such maps: each encoder induces a nonlinear geometry that is strongly influenced by architecture, optimization, and random initialization, making $Z _ { 1 }$ and $Z _ { 2 }$ poorly compatible under a simple afine fit.

Conversely, we view the existence of a low-complexity alignment map as a geometric consequence of the two latent point clouds being approximately isometric. Recall the classical result that if two centered point clouds have identical pairwise distance matrices, then they are related by an orthogonal transformation. This is formalized in Lemma 1.

Lemma 1. Let Z, $\mathbf { Y } \in \mathbb { R } ^ { n \times d }$ be matrices whose rows are the points $\{ z _ { i } \} _ { i = 1 } ^ { n }$ and $\{ y _ { i } \} _ { i = 1 } ^ { n }$ and assume they are centered: $\mathbf { Z } ^ { \top } \mathbf { 1 } = \mathbf { Y } ^ { \top } \mathbf { 1 } = 0$ . Let $\mathbf { \Delta } \varDelta ( \mathbf { Z } ) \doteq \bar { \mathbb { R } } ^ { \bar { n } \times \bar { n } }$ denote the squared distance matrix, $\Delta ( \mathbf { Z } ) _ { i j } = \| z _ { i } - z _ { j } \| _ { 2 } ^ { 2 }$ (and similarly for $\varDelta ( \mathbf { Y } ) )$ $I f \varDelta ( \mathbf { Z } ) = \varDelta ( \mathbf { Y } )$ , then there exists $\mathbf { Q } \in \mathbb { R } ^ { d \times d }$ with $\mathbf { \bar { Q } } ^ { \top } \mathbf { Q } = I$ such that $\mathbf { Y } = \mathbf { Z } \mathbf { Q } $

Proof. See Appendix A.1.

Lemma 1 motivates why a rigid alignment — either orthogonal or afine — is a natural choice when the two representations associated to each latent space $Z _ { 1 }$ and $Z _ { 2 }$ are approximately isometric. However, independent training does not guarantee this property. Our proposed framework aims to address this gap: we introduce a topological regularizer that explicitly encourages each encoder to produce latent spaces with compatible geometry, making a simple post-hoc alignment viable.

## 3.2 Topological regularization to promote alignability

We now introduce our proposed framework for training autoencoders with compatible latent spaces. The key idea is to promote geometric compatibility between independently trained latent spaces by adding a geometric constraint during training. Specifically, we encourage the latent space to preserve the topological structure of a reference point cloud via a diferentiable topological loss $L _ { \mathrm { t o p o } } ( \cdot , \cdot )$ based on persistent homology [28]. In particular, we focus on 0-dimensional persistence features, for which the persistence pairing boils down to the edges of the MST. Intuitively, the MST’s edges form a sparse “topological skeleton” capturing global connectivity of the point cloud; matching these edges across spaces is a lightweight way to encourage comparable geometry.

Topological loss. Given a target space $X : = \{ x _ { i } \} _ { i = 1 } ^ { n }$ and a learnable space $Z : =$ $\{ z _ { i } \} _ { i = 1 } ^ { n }$ , let $P$ be the set of all unordered pairs and $A _ { X }$ (resp. $A _ { Z } )$ the vector of all pairwise distances in X (resp. Z). Let $P _ { X }$ and $P _ { Z }$ denote the persistence pairing indices in the target and learnable spaces (MST edges when $d = 0 )$ Denoting

$$
\begin{array} { r } { D _ { X X } : = A _ { X } [ P _ { X } ] , \quad D _ { Z Z } : = A _ { Z } [ P _ { Z } ] , } \\ { D _ { Z X } : = A _ { Z } [ P _ { X } ] , \quad D _ { X Z } : = A _ { X } [ P _ { Z } ] , } \end{array}
$$

the topological loss is

$$
\begin{array} { r } { L _ { \mathrm { t o p o } } ( X , Z ) = \frac { 1 } { 2 } \| D _ { X X } - D _ { Z X } \| _ { F } ^ { 2 } + \frac { 1 } { 2 } \| D _ { X Z } - D _ { Z Z } \| _ { F } ^ { 2 } . } \end{array}\tag{3}
$$

MST-based topological regularization promotes linear alignability. Our loss (3) only constrains a sparse subset of distances (MST edges). We next show that while (3) does not, in general, guarantee isometry, it encourages geometric agreement along a shared MST backbone; under additional measurable conditions, this can be suficient to imply approximate near-isometry. The first term of (3) enforces agreement between target MST edge lengths and those same edges measured in the learnable space.

Let $T _ { X } = \left( V , P _ { X } \right)$ denote the MST graph on vertex set $V = \{ 1 , \ldots , n \}$ with edge set $P _ { X }$ . For a point cloud $Z ,$ define the tree-path metric induced by $T _ { X } ;$

$$
d _ { T _ { X } } ^ { Z } ( i , j ) : = \sum _ { ( u , v ) \in \mathrm { p a t h } _ { T _ { X } } ( i , j ) } \lVert z _ { u } - z _ { v } \rVert _ { 2 } ,
$$

where $\mathrm { p a t h } _ { T _ { X } } ( i , j )$ is the unique path between i and j in the tree (as it is an MST).

Let diam $( T _ { X } )$ be the tree diameter in number of edges: diam $\begin{array} { r } { \left( T _ { X } \right) : = \operatorname* { m a x } _ { i , j } \left| \mathrm { p a t h } _ { T _ { X } } ( i , j ) \right| } \end{array}$

Lemma 2 (Edge mismatch controls all tree-path distances). For any two point clouds $Z$ and Y on the same index set and any tree $T _ { X } = \left( V , P _ { X } \right)$

$$
\begin{array} { r l } & { \underset { i , j } { \operatorname* { m a x } } \left| d _ { T _ { X } } ^ { Z } ( i , j ) - d _ { T _ { X } } ^ { Y } ( i , j ) \right| \leq \sqrt { \mathrm { d i a m } ( T _ { X } ) } } \\ & { ~ \times \ \lVert A _ { Z } [ P _ { X } ] - A _ { Y } [ P _ { X } ] \rVert _ { 2 } . } \end{array}
$$

where $A _ { Z } [ P _ { X } ]$ is the vector of edge lengths $\{ \| z _ { u } - z _ { v } \| _ { 2 } : ( u , v ) \in P _ { X } \}$ (and similarly for $A _ { Y } [ P _ { X } ] $ .

Proof. See Appendix A.2.

Lemma 2 yields an unconditional form of compatibility. Specifically, if two latent spaces approximately match the same reference MST edge lengths on $P _ { X }$ (equivalently, the edge residuals $\varepsilon _ { 1 } , \varepsilon _ { 2 }$ are small), then they induce similar treepath metrics on $T _ { X } \colon$ for all pairs $( i , j ) , d _ { T _ { X } } ^ { Z _ { 1 } } ( i , j ) \approx d _ { T _ { X } } ^ { Z _ { 2 } } ( \bar { i } , j )$ , with an explicit bound provided by Lemma 2. This tree-metric agreement is the quantity most directly controlled by our topological regularizer.

Further, the approximate agreement of Euclidean pairwise distances $\| z _ { i } ^ { ( 1 ) } -$ $z _ { j } ^ { ( 1 ) } \| _ { 2 } \approx \| z _ { i } ^ { ( 2 ) } - z _ { j } ^ { ( 2 ) } \| _ { 2 }$ (near-isometry), requires an additional condition relating the tree-path metric to Euclidean distance in each latent space. We capture this via a bounded stretch assumption (Definition 1).

Definition 1 (Tree stretch with respect to $T _ { X } )$ . For a point cloud $Z ,$ define the stretch

$$
\tau ( Z ; T _ { X } ) : = \operatorname* { m a x } _ { i \neq j } \frac { d _ { T _ { X } } ^ { Z } ( i , j ) } { \| z _ { i } - z _ { j } \| _ { 2 } } \in [ 1 , \infty ] .
$$

Small $\tau ( Z ; T _ { X } )$ means the MST-path metric is a good proxy for Euclidean distances in $Z .$ This quantity is dataset-dependent but can be estimated by sampling pairs.

Bounded tree stretch is not guaranteed by the MST in the worst case; we therefore treat it as a measurable diagnostic. Under bounded stretch, tree-metric agreement can be lifted to approximate Euclidean distance agreement, quantified by a bound on $\| \varDelta ( \mathbf { Z } _ { 1 } ) - \varDelta ( \mathbf { Z } _ { 2 } ) \| _ { F }$ (Theorem 1).

Theorem 1 (MST/topological regularization promotes linear stitchability). Let $\mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } \in \mathbb { R } ^ { n \times d }$ be latent point clouds (rows are the points $\{ z _ { i } ^ { ( 1 ) } \} _ { i = 1 } ^ { n }$ and $\{ z _ { i } ^ { ( 2 ) } \} _ { i = 1 } ^ { n } )$ and assume they are centered: ${ \bf Z } _ { 1 } ^ { \top } { 1 } = { \bf Z } _ { 2 } ^ { \top } { 1 } = 0$ . Fix a reference tree $T _ { X } = \left( V , P _ { X } \right)$ with vertex set $V = \{ 1 , \ldots , n \}$ and edge set $P _ { X }$ (the 0-dimensional persistence pairing in the data space). For $k \in \{ 1 , 2 \}$ define the MST edge-mismatch residual

$$
\varepsilon _ { k } : = \big \| A _ { Z _ { k } } [ P _ { X } ] - A _ { X } [ P _ { X } ] \big \| _ { 2 } ,
$$

Assume a bounded tree-stretch condition holds for both clouds (Definition 1): $\tau ( Z _ { 1 } ; T _ { X } ) \le \tau$ and $\tau ( Z _ { 2 } ; T _ { X } ) \le \tau$

Then the squared-distance matrices satisfy,

$$
\begin{array} { r l r } & { } & { \| \varDelta ( \mathbf { Z } _ { 1 } ) - \varDelta ( \mathbf { Z } _ { 2 } ) \| _ { F } \leq 2 R \sqrt { n ( n - 1 ) } \Big ( 2 ( \tau - 1 ) R } \\ & { } & { + \sqrt { \mathrm { d i a m } ( T _ { X } ) } ( \varepsilon _ { 1 } + \varepsilon _ { 2 } ) \Big ) . } \end{array}
$$

Here, $\begin{array} { r } { R : = \operatorname* { m a x } _ { i \neq j } \operatorname* { m a x } \{ \| z _ { i } ^ { ( 1 ) } - z _ { j } ^ { ( 1 ) } \| _ { 2 } , \ \| z _ { i } ^ { ( 2 ) } - z _ { j } ^ { ( 2 ) } \| _ { 2 } \} . } \end{array}$

Proof. See Appendix Section A.3.

Combined with the result of Lemma 1, the approximate Euclidean distance agreement established by Theorem 1 provides geometric motivation for seeking a low-distortion orthogonal alignment, although it does not by itself provide a quantitative alignment guarantee. Overall, our analysis provides (i) an unconditional guarantee on the induced tree-metric alignment and (ii) a conditional route to Euclidean near-isometry in regimes where stretch is moderate.

Discussion and limitations. We want to emphasize that Theorem 1 provides only a conditional guarantee, and MST constraints alone do not uniquely determine a latent embedding, but they provably enforce agreement of induced tree metrics (Lemma 2). When the learned representations additionally have a small tree stretch (Definition 1), this implies approximate pairwise-distance agreement, which provides geometric motivation for seeking an orthogonal alignment.

## 3.3 Training methodology and stitching

We incorporate (3) into the training framework of the independently trained autoencoders, allowing them to produce more compatible latent spaces. Each autoencoder is regularized so that its latent space preserves the topological structure of the input space. Concretely, for $\boldsymbol { \mathrm { A E } } = ( E , D )$ , we train with

$$
L = \| x - D ( E ( x ) ) \| ^ { 2 } + \lambda L _ { \mathrm { t o p o } } ( x , E ( x ) ) .\tag{4}
$$

Because both $\mathrm { A E _ { 1 } }$ and $\mathrm { { A E _ { 2 } } }$ are trained to preserve the same reference topology (the data geometry induced by the input metric), their latent spaces are encouraged to share a common topological skeleton, despite being learned by diferent architectures. Note that this loss does not require any information to be exchanged between the two models $\mathrm { A E } _ { 1 }$ and $\mathrm { { A E } _ { 2 } }$ for training.

Stitching after training. After training, we fit an afine transformation T using a small calibration set $D _ { T }$ by solving (2) with $T ( z ) = T z$ . At inference time, stitching is performed by composing the encoder, the learned transformation, and the decoder, e.g.,

$$
{ \hat { x } } = D _ { 1 } ( T ( E _ { 2 } ( x ) ) ) ,
$$

and symmetrically for the opposite direction.

## 4 Experiments

We evaluate the performance of the proposed topology-preserving training and investigate its eficacy in achieving interoperability across independently trained models/representations when the models are connected post hoc by a lightweight afine connector. We report the performance across multiple diferent use cases and data domains: (i) image compression (section 4.1), (ii) wireless communication channel compression (section 4.2), (iii) sketch-based image retrieval performance (section 4.3). Additionally, in section 4.4 we present an empirical evaluation of the geometric diagnostics that probe the conditions used in our analysis in section 3.2.

## 4.1 Stitched image compression and reconstruction

We consider an autoencoder-based compression framework in which two architecturally diferent autoencoder models are first trained independently and then, post-training, cross-model performance is evaluated. The evaluation is performed across MNIST [9], CIFAR-10 [16], Tiny-ImageNet [18] and CelebA [24] datasets. As baseline metrics, we report the end-to-end performance of the two models $( \mathrm { L } _ { 1 1 }$ and L<sub>22</sub>, i.e. encoder and decoder coming from the same jointly trained autoencoder without any regularization) and the direct cross model performance $( \mathrm { L } _ { 1 2 }$ and $\mathrm { L _ { 2 1 } }$ , i.e. encoder and decoder coming from the diferent autoencoders and joined directly, without any form of alignment). Our method is indicated by $\mathrm { L ^ { T o p o } }$ . We compare the performance of our approach with the relative representation-based stitching $( \mathrm { L } ^ { \mathrm { R \bar { R } } } )$ [29], and the direct afine alignment-based stitching $( \mathrm { L } ^ { \mathrm { D A A } } )$ proposed in [25, 17]. To ensure that the latent space alignment across independently trained models is not an artifact of the model architecture, we choose completely diferent model architectures for the two autoencoders. For the MNIST and CIFAR-10 datasets, the first model is an MLP-based model, and the other has a convolutional neural network (CNN) based architecture. In the Tiny-ImageNet and CelebA datasets, the first model is a CNN based model and the other model is a vision transformer (ViT) based model. All experimental results are averaged over 5 independent training runs with diferent initializations. Further details on the model architecture and training can be found in Appendix B.

As seen from Table 1 the results for direct cross-model usage, $\mathrm { L _ { 1 2 } }$ and $\mathrm { L _ { 2 1 } }$ , are highly unreliable across all image datasets. This reafirms the observations from prior work that independently trained latent spaces are generally not directly compatible.

Fitting an $a f f i n e$ connector using a small calibration set substantially improves interoperability across all datasets. For example, on MNIST, the afinestitched errors drop to $\mathrm { L _ { 1  T  2 } ^ { D A A } = - 1 1 . 0 9 }$ and $\mathrm { L } _ { 2  \mathrm { T }  1 } ^ { \mathrm { D A A } } = - 1 1 . 1 6$ , representing more than an order-of-magnitude improvement over direct cross-decoding. Similar behavior holds on the other datasets too.

Topology-preserving training yields an additional consistent gain over afine stitching. On MNIST, topology reduces the afine-stitched error by 1.7 dB for

Table 1: The table shows the interoperability results on image datasets. We compare the reconstruction NMSE (dB) for diferent methods across diferent encoder-decoder configurations. $\mathrm { L _ { i j } }$ denotes the use of encoder $E _ { i }$ with decoder $D _ { j } .$
<table><tr><td>Dataset</td><td> $\mathbf { L _ { 1 1 } }$ </td><td> $\mathbf { L _ { 2 2 } }$ </td><td> $\mathbf { L _ { 1 2 } }$ </td><td> $\mathbf { L _ { 2 1 } }$ </td><td> $\bf { L _ { 1  T \cdot } ^ { D A A } }$ </td><td> $\bf { \Sigma } _ {  2 } \ \mathrm { \bf ~ L } _ { 2  \mathrm { T }  1 } ^ { \mathrm { D A A } }$ </td><td> $\mathbf { L _ { 1 2 } ^ { R R } }$ </td><td> $\mathbf { L _ { 2 1 } ^ { R R } }$ </td><td></td><td> $\mathbf { L _ { 1  T  2 } ^ { T o p o } L _ { 2  T  1 } ^ { T o p o } }$ </td></tr><tr><td>MNIST</td><td>-15.38-16.64</td><td></td><td>-0.11</td><td>1.19</td><td>-11.09</td><td>-11.16</td><td>-2.23</td><td>-2.98</td><td>-12.81</td><td>-12.42</td></tr><tr><td>CIFAR-10</td><td>-14.09-15.38</td><td></td><td>-2.99</td><td>-0.36 1</td><td>-13.80</td><td>-13.82</td><td>-5.87</td><td>-6.68</td><td>-14.34</td><td> $\mathbf { - 1 4 . 3 5 }$ </td></tr><tr><td>T-ImageNet -11.73 -10.42</td><td></td><td></td><td>0.05</td><td>-0.01</td><td>-10.22</td><td>-11.27</td><td>-7.59</td><td>-7.01</td><td>-10.48</td><td> $\mathbf { - 1 1 . 2 8 }$ </td></tr><tr><td>CelebA</td><td>-15.36-15.33</td><td></td><td>-2.57</td><td>-4.5</td><td>-14.38</td><td>-14.46</td><td>-7.92</td><td>-5.55</td><td>-14.93</td><td> $\mathbf { - 1 4 . 9 2 }$ </td></tr></table>

Table 2: Results on channel state information (CSI) compression. Same notation as Table 1.
<table><tr><td>Dataset</td><td> $\mathbf { L _ { 1 1 } }$   $\mathbf { L _ { 2 2 } }$   $\mathbf { L _ { 1 2 } }$ </td><td> $\mathbf { L _ { 2 1 } }$ </td><td> $\bf { L _ { 1  T  2 } ^ { D A A } }$ </td><td> $\mathbf { L _ { 2 \to T \to 1 } ^ { D A A } }$ </td><td> $\mathbf { L _ { 1 2 } ^ { R R } }$ </td><td> $\mathbf { L _ { 2 1 } ^ { R R } }$ </td><td> $\mathbf { L _ { 1  T  2 } ^ { T o p o } }$ </td><td> $\mathbf { L _ { 2  T  1 } ^ { T o p o } }$ </td></tr><tr><td></td><td></td><td>CDL-A,C -16.7 -30.0 5.85 28.28</td><td>-11.27</td><td>-11.58</td><td>-0.310.56</td><td></td><td>-14.88</td><td>-12.67</td></tr><tr><td>UMa</td><td></td><td>-4.48 -7.31 5.19 32.15</td><td>-2.68</td><td>-2.29</td><td>0.041.61</td><td></td><td>-3.54</td><td>-2.96</td></tr></table>

$L _ { 1  T  2 }$ and 1.25 dB for $L _ { 2  T  1 }$ . Across all datasets, the $\mathrm { L ^ { T o p o } }$ outperforms the existing alignment approaches, although in the Tiny-ImageNet data, the improvements are less pronounced, suggesting that in this setting, the afine connector already captures most of the alignable structure in that direction.

## 4.2 CSI compression: interoperability beyond vision

We next evaluate our approach on a compression task from wireless communications, where we compress the channel state information (CSI). CSI is a structured, high-dimensional representation of the wireless channel; in real systems, including 5G systems, it is communicated under strict bandwidth constraints, making compression essential. The use of autoencoders is being extensively studied for the upcoming releases of the wireless communication standards [22]. From an ML perspective, CSI compression is analogous to learned image compression: an encoder maps an input tensor to a compact code, which is quantized and a decoder reconstructs the original signal, with quality measured by reconstruction error.

We consider 3GPP standards-compliant CSI datasets CDL-A,C and UMa [1], where each sample is a complex-valued CSI tensor. We represent CSI using two real-valued channels (real and imaginary parts) and train the autoencoder with a quantization block (2-bit uniform scalar quantization [26]) after the encoding stage. Similar to the image experiments, 2 diferent model architectures (CsiNet based [30], and transformer based [26]) are trained independently. We evaluate interoperability by stitching encoders and decoders across model architectures.

Table 2 shows that direct cross-decoding fails substantially across all CSI datasets, indicating that independently trained representations are not immediately compatible. Fitting a lightweight afine connector improves performance, but a gap remains relative to within-model decoding. In contrast, topologypreserving training consistently improves stitched performance, reducing reconstruction error. Notably, the gains persist in the quantized setting, suggesting that the regularizer encourages a representation geometry that is robust to discretization artifacts.

Table 3: FS-COCO sketch-based image retrieval. Higher is better. Afine denotes fitting an afine connector on a calibration set.
<table><tr><td>Metric</td><td> $\mathbf { L _ { 1 1 } }$ </td><td> $\mathbf { L _ { 2 2 } }$ </td><td> $\bf { L _ { 1 2 } }$ </td><td> $\mathbf { L _ { 2 1 } }$ </td><td> $\bf { L _ { 1  T  2 } ^ { D A A } }$ </td><td> $\mathbf { L _ { 2 \to T \to 1 } ^ { D A A } }$ </td><td> $\mathbf { L _ { 1  T  2 } ^ { T o p o } }$ </td><td> $\mathbf { L _ { 2  T  1 } ^ { T o p o } }$ </td></tr><tr><td>mAP</td><td></td><td>0.636 0.524 0.002 0.002</td><td></td><td></td><td>0.348</td><td>0.317</td><td>0.378</td><td>0.388</td></tr><tr><td>Top-1</td><td>53.7</td><td>41.9</td><td>0.0</td><td>0.0</td><td>23.0</td><td>20.71</td><td>26.27</td><td>27.12</td></tr><tr><td>Top-5</td><td>75.6</td><td>64.5</td><td>0.1</td><td>0.1</td><td>47.3</td><td>43.12</td><td>50.54</td><td>51.52</td></tr><tr><td>Top-10</td><td>82.4</td><td>73.3</td><td>0.3</td><td>0.2</td><td>59.0</td><td>54.59</td><td>61.76</td><td>62.86</td></tr></table>

## 4.3 Sketch-based image retrieval

To test the robustness of our framework beyond autoencoder-style reconstruction, we evaluate the method on a cross-modal retrieval setting of sketch-based image retrieval (SBIR). The objective of SBIR is to embed sketches and images into a common latent space in which images semantically similar to a hand-drawn sketch query can be retrieved. We build our evaluation framework based on the recent SBIR implementation in [8] and leverage the FS-COCO [7] dataset. We first train 2 models end-to-end, independently. The first framework uses the ViT-B model for learning the sketch and image embeddings, and the second framework uses the ConvNeXt [23]. Based on the implementation in [8], models pre-trained on ImageNet-1K are utilized and then fine-tuned independently on the FS-COCO dataset with their proposed ICon loss function as the base loss metric. Post-training, one model produces sketch embeddings, and the other produces the image embedding. Table 3 reports mAP and Top-k retrieval accuracy.

Without alignment, cross-modal retrieval is essentially degenerate $( L _ { 1 2 } , L _ { 2 1 }$ ≈ 0 across metrics), indicating that independently trained sketch and image embedding spaces are not directly comparable. An afine connector yields strong gains (mAP 0.3486/0.3171; Top-10 0.5905/0.5459), and the proposed topologypreserving training improves the accuracies further. Especially, in the $L _ { 2  T  1 } ^ { T o p o }$ case, with the ViT-based sketch encoder and Conv-based image encoder, the top-1 to top-10 retrieval accuracies are improved by 6-8%. These results highlight the capability of topology-preserving regularization in promoting interoperability for applications beyond reconstruction.

## 4.4 Geometric diagnostics: tree-metric discrepancy

We next examine whether topology-preserving training moves representations toward the conditions used in our analysis and promotes agreement of the induced tree-path metric across independently trained latent spaces. To probe this mechanism empirically, we introduce a diagnostic that measures cross-model treemetric discrepancy. The tree-metric discrepancy measure is directly tied to the mechanism enforced by the topological loss: if both encoders preserve the MST edge lengths of the same reference tree, then their induced tree-path distances must agree for all pairs, up to accumulated edge-wise discrepancies (Lemma 2).

Tree-metric discrepancy. We first sample a fixed held-out set $\{ x _ { i } \} _ { i = 1 } ^ { n }$ and construct a reference tree $T _ { X } = \left( V , P _ { X } \right)$ by computing the minimum spanning tree (MST) in input space, where $V = \{ 1 , \ldots , n \}$ and $P _ { X }$ is the set of MST edges. For each encoder $E _ { k } \ ( k \in \{ 1 , 2 \} )$ , we compute latents $z _ { i } ^ { ( k ) } = E _ { k } ( x _ { i } )$ and define the induced tree-path distance

$$
d _ { T _ { X } } ^ { Z _ { k } } ( i , j ) : = \sum _ { ( u , v ) \in \mathrm { p a t h } _ { T _ { X } } ( i , j ) } \Vert z _ { u } ^ { ( k ) } - z _ { v } ^ { ( k ) } \Vert _ { 2 } .
$$

where, path $_ { \cdot T _ { X } } ( i , j )$ denotes the unique simple path between i and $j$ in the tree. We quantify the cross-model tree-metric discrepancy by

$$
\delta _ { t r e e } ( i , j ) : = \left| d _ { T _ { X } } ^ { Z _ { 1 } } ( i , j ) - d _ { T _ { X } } ^ { Z _ { 2 } } ( i , j ) \right| ,
$$

and report summary statistics (mean / 95th / 99th percentile) over 20k sampled pairs $( i , j )$ in Table 4.

Across datasets, topology-preserving training leads to a pronounced reduction in the cross-space tree-metric discrepancy $\delta _ { t r e e }$ . This pattern is consistent with the role of the topological loss: it encourages both models to induce similar path-length structure on the same MST skeleton, which is the quantity directly controlled by Lemma 2. A reduction in $\delta _ { t r e e }$ indicates that independently trained latent spaces encode a shared topological backbone more consistently, making them easier to interconnect with a lightweight afine connector. In our experiments, topology-preserving training substantially reduces $\delta _ { \mathrm { t r e e } }$ across datasets, and this reduction correlates with improved stitched performance.

## 4.5 Ablation studies

We conduct ablations to assess (i) sensitivity to the topology regularization weight λ and (ii) sample eficiency of the afine connector fitting stage (calibration set size $| D _ { T } | )$ . Unless stated otherwise, all other training and evaluation settings are identical to the main experiments.

Table 4: Tree-metric discrepancy diagnostics.
<table><tr><td rowspan="2">Dataset</td><td> $\delta _ { t r e e }$ </td><td>(no topo)</td><td colspan="2"> $\delta _ { t r e e }$  (topo)</td></tr><tr><td>Mean</td><td>95 99</td><td>Mean</td><td>95 99</td></tr><tr><td>MNIST</td><td>290.8</td><td>521.9 592.1</td><td>192.8</td><td>345.6 392.4</td></tr><tr><td>CIFAR</td><td>54.4</td><td>90.0 107.0</td><td>13.0</td><td>21.3 25.0</td></tr><tr><td>T-ImageNet</td><td>542.8</td><td>927.2 1109.8</td><td>84.9</td><td>151.3 185.2</td></tr><tr><td>CelebA</td><td>133.2</td><td>254.3298.7</td><td>99</td><td>188.9 218.9</td></tr></table>

![](images/3525ef3c1da67a32a07a413636052b349ccec2f863337123c56c29017f9d1364.jpg)  
(a) Efect of λ.

![](images/f5af883d62d2b2c46a83cfdccd1d3df80bc8d1a3aefc5fc20fc8a80bcd7b2c0f.jpg)  
(b) Efect of $| D _ { T } |$  
Fig. 2: MNIST ablations: sensitivity to topology weight λ and calibration size $| D _ { T } |$ |.

Efect of topology regularization weight. Figures 2 and 3 vary the topology regularization weight λ in (4). We observe that performance improves substantially over the $\lambda \approx 0$ regime (which reduces to standard training without topology preservation), and remains stable across a broad range of moderate λ values. For excessively large λ, performance can degrade, consistent with overregularization: enforcing topology too strongly can compete with reconstruction objectives and reduce representational flexibility. Overall, these ablations indicate that topology-preserving training is not brittle, and that selecting λ within a moderate range yields consistent gains.

Efect of calibration size for afine stitching. Figures 2 and 3 also vary the number of calibration examples used to fit the afine connector $T \ ( \mathrm { i . e . , } | D _ { T } |$ in (2)). Across both MNIST and CSI, topology-preserving training improves sample $e f f -$ ciency: for a fixed calibration budget, stitched performance is consistently better with topology than without. Moreover, the marginal benefit of increasing $| D _ { T } |$ diminishes beyond a moderate regime, suggesting that the principal role of the connector is to correct residual global misalignment once the training-time regularizer has aligned the two representations on a common geometric scafold.

![](images/eb5f946abbae5bec624d6723eb3f9ea11b65f05d02b76848bbaedf0b2a952091.jpg)  
(a) Efect of λ.

![](images/bc988c90cc8f6e8130da7b7db69d0916ffa27dc47e608397df2e5d2b846fd382.jpg)  
(b) Efect of $| D _ { T } |$  
Fig. 3: CSI ablations: sensitivity to topology weight λ and calibration size $| D _ { T } |$

## 5 Conclusion

We investigate the problem of zero-shot model stitching in autoencoders, demonstrating that topological regularization promotes a simplified, afine mapping between independent latent spaces. Our proposed framework introduces a training scheme aimed at preserving the topological structure of the input data within the latent space of an otherwise unconstrained autoencoder. We provide theoretical justification for how this topological loss controls agreement between induced tree-path metrics. Under the additional condition of bounded tree stretch, this agreement implies approximate Euclidean pairwise-distance agreement between representations. Through numerical experiments across heterogeneous domains, including both image and wireless datasets, we show that our method drastically simplifies latent-space alignment. Future work will explore alternative geometric losses to further refine the alignment of topological features.

## References

1. 3GPP: Study on channel model for frequencies from 0.5 to 100 GHz. Technical Report 38.901, Version 16.1.0, Release 16 (2020)

2. Arvanitidis, G., Hansen, L.K., Hauberg, S.: Latent space oddity: On the curvature of deep generative models. In: Intl. Conf. Learn. Repr. (ICLR) (2018)

3. Ballé, J., Laparra, V., Simoncelli, E.P.: End-to-end optimized image compression. In: Intl. Conf. Learn. Repr. (ICLR) (2017)

4. Borg, I., Groenen, P.J.: Modern multidimensional scaling: Theory and applications. Springer (2005)

5. Cannistraci, I., Fumero, M., Moschella, L., Maiorca, V., Rodolà, E.: Infusing invariances in neural representations. Workshop Topology, Algebra, and Geometry in Machine Learning TAG-ML at ICML (2023)

6. Cannistraci, I., Moschella, L., Fumero, M., Maiorca, V., Rodolà, E.: From bricks to bridges: Product of invariances to enhance latent space communication. In: Intl. Conf. Learn. Repr. (ICLR) (2024)

7. Chowdhury, P.N., Sain, A., Bhunia, A.K., Xiang, T., Gryaditskaya, Y., Song, Y.Z.: Fs-coco: Towards understanding of freehand sketches of common objects in context. In: European conference on computer vision. pp. 253–270. Springer (2022)

8. Demić, E., Zajc, L.Č.: Back to the drawing board: Rethinking scene-level sketchbased image retrieval. arXiv preprint arXiv:2509.06566 (2025)

9. Deng, L.: The MNIST database of handwritten digit images for machine learning research. IEEE Signal Process. Mag. 29(6), 141–142 (2012)

10. Duque, A.F., Morin, S., Wolf, G., Moon, K.R.: Geometry regularized autoencoders. IEEE Trans. Pattern Anal. Mach. Intell. 45(6), 7381–7394 (2023)

11. Fumero, M., Pegoraro, M., Maiorca, V., Locatello, F., Rodolà, E.: Latent functional map. arXiv preprint arXiv:2406.14183 (2024)

12. Gygli, M., Uijlings, J., Ferrari, V.: Towards reusable network components by learning compatible representations. In: AAAI Conf. Art. Intell. vol. 35, pp. 7620–7629 (2021)

13. Gârbacea, C., den Oord, A.v., Li, Y., Lim, F.S.C., Luebs, A., Vinyals, O., Walters, T.C.: Low bit-rate speech coding with VQ-VAE and a wavenet decoder. In: IEEE Intl. Conf. Acoust., Speech and Signal Process. (ICASSP). pp. 735–739 (2019). https://doi.org/10.1109/ICASSP.2019.8683277

14. Hensel, F., Moor, M., Rieck, B.: A survey of topological machine learning methods. Front. Artif. Intell. 4, 681108 (2021)

15. Huang, T.H., Malhotra, A., Hamidi-Rad, S.: A deep learning method for joint compression and unsupervised denoising of CSI feedback. In: IEEE Intl. Conf. on Commun. (ICC)). pp. 4150–4156 (2023)

16. Krizhevsky, A., Hinton, G., et al.: Learning multiple layers of features from tiny images. Tech. rep., University of Toronto (2009)

17. Lähner, Z., Moeller, M.: On the direct alignment of latent spaces. In: Workshop UniReps: the First Workshop on Unifying Representations in Neural Models at NeurIPS. pp. 158–169. PMLR (2024)

18. Le, Y., Yang, X.: Tiny imagenet visual recognition challenge. CS 231N 7(7), 3 (2015)

19. Lee, Y.: A geometric perspective on autoencoders. arXiv preprint arXiv:2309.08247 (2023)

20. Lee, Y., Kwon, H., Park, F.: Neighborhood reconstructing autoencoders. Advances in Neural Inf. Process. Syst. (NIPS) 34, 536–546 (2021)

21. Lim, J., Kim, J., Lee, Y., Jang, C., Park, F.C.: Graph geometry-preserving autoencoders. In: Intl. Conf. on Machine Learning (ICML) (2024)

22. Lin, X.: An overview of the 3gpp study on artificial intelligence for 5g new radio. arXiv preprint arXiv:2308.05315 (2023)

23. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11976–11986 (2022)

24. Liu, Z., Luo, P., Wang, X., Tang, X.: Deep learning face attributes in the wild. In: Proceedings of International Conference on Computer Vision (ICCV) (December 2015)

25. Maiorca, V., Moschella, L., Norelli, A., Fumero, M., Locatello, F., Rodolà, E.: Latent space translation via semantic alignment. Advances in Neural Inf. Process. Syst. (NIPS) 36 (2023)

26. Malhotra, A., Ibrahim, M.S., Patani, K.: Exploiting temporal priors for eficient real-time compression and feedback of wireless channels. In: Workshop on Machine Learning and Compression, NeurIPS. Vancouver, Canada (Dec 2024)

27. Mentzer, F., Agustsson, E., Tschannen, M., Timofte, R., Van Gool, L.: Conditional probability models for deep image compression. In: Proceedings of the IEEE/CVF Int. Conf. Comput. Vis. Pattern Recogn. (CVPR). pp. 4394–4402 (2018)

28. Moor, M., Horn, M., Rieck, B., Borgwardt, K.: Topological autoencoders. In: Intl. Conf. on Machine Learning (ICML). pp. 7045–7054. PMLR (2020)

29. Moschella, L., Maiorca, V., Fumero, M., Norelli, A., Locatello, F., Rodolà, E.: Relative representations enable zero-shot latent space communication. In: Intl. Conf. Learn. Repr. (ICLR) (2023)

30. Wen, C.K., Shih, W.T., Jin, S.: Deep learning for massive MIMO CSI feedback. IEEE Wireless Communications Letters 7(5), 748–751 (Mar 2018)

31. Yonghyeon, L., Yoon, S., Son, M., Park, F.C.: Regularized autoencoders for isometric representation learning. In: Intl. Conf. Learn. Repr. (ICLR) (2022)

## A Proofs

## A.1 Proof for Lemma 1

Proof. Let $\begin{array} { r } { \mathbf { H } = \mathbf { I } - \frac { 1 } { n } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ be the centering matrix. A classical identity in multidimensional scaling (MDS) states that the (centered) Gram matrix can be recovered from squared distances via double centering:

$$
\begin{array} { r } { { \bf Z } { \bf Z } ^ { \top } = - \frac { 1 } { 2 } { \bf H } \varDelta ( { \bf Z } ) { \bf H } , } \end{array}\tag{5}
$$

and similarly $\begin{array} { r } { \mathbf { Y } \mathbf { Y } ^ { \top } = - \frac { 1 } { 2 } \mathbf { H } \varDelta ( \mathbf { Y } ) \mathbf { H } } \end{array}$

Since $\varDelta ( \mathbf { Z } ) = \varDelta ( \mathbf { Y } )$ , applying (5) to both point clouds yields

$$
\begin{array} { r } { { \bf Z } { \bf Z } ^ { \top } \ : = \ : - \frac { 1 } { 2 } { \bf H } \Delta ( { \bf Z } ) { \bf H } \ : = \ : - \frac { 1 } { 2 } { \bf H } \Delta ( { \bf Y } ) { \bf H } \ : = \ : { \bf Y } { \bf Y } ^ { \top } . } \end{array}
$$

Denote this common Gram matrix by $\mathbf G \in \mathbb { R } ^ { n \times n } , \mathrm { ~ s o ~ } \mathbf G = \mathbf Z \mathbf Z ^ { \top } = \mathbf Y \mathbf Y ^ { \top }$

Assume Z has full column rank d (hence rank(G) = d). Take a thin SVD of Z:

$$
\mathbf { Z } = \mathbf { U } \pmb { \Sigma } \mathbf { V } ^ { \top } ,
$$

where $\mathbf { U } \in \mathbb { R } ^ { n \times d }$ and $\mathbf { V } \in \mathbb { R } ^ { d \times d }$ have orthonormal columns and $\pmb { \Sigma } \in \mathbb { R } ^ { d \times d }$ is diagonal with positive entries. Then

$$
\mathbf G = \mathbf Z \mathbf Z ^ { \top } = \mathbf U \Sigma \mathbf V ^ { \top } \mathbf V \Sigma \mathbf U ^ { \top } = \mathbf U \Sigma ^ { 2 } \mathbf U ^ { \top } .
$$

Since $\mathbf { Y } \mathbf { Y } ^ { \top } = G$ as well and $\operatorname { r a n k } ( \mathbf { Y } ) = \operatorname { r a n k } ( \mathbf { G } ) = d ,$ a thin SVD of Y can be written as

$$
\mathbf { Y } = \mathbf { U } \pmb { \Sigma } \mathbf { W } ^ { \top }
$$

for some orthogonal $\mathbf { W } \in \mathbb { R } ^ { d \times d }$ (the left singular vectors U and singular values Σ are determined by $\mathbf { G } = \mathbf { U } \pmb { \Sigma } ^ { 2 } \mathbf { U } ^ { \top }$ up to an orthogonal change of basis in R<sup>d</sup>). Therefore,

$$
\begin{array} { r l } & { \mathbf { Y } = \mathbf { U } \boldsymbol { \Sigma } \mathbf { W } ^ { \top } } \\ & { \quad = ( \mathbf { U } \boldsymbol { \Sigma } \mathbf { V } ^ { \top } ) ( \mathbf { V } \mathbf { W } ^ { \top } ) = \mathbf { Z } Q , } \\ & { \quad Q : = \mathbf { V } \mathbf { W } ^ { \top } . } \end{array}
$$

Since V and W are orthogonal, Q is orthogonal as well: $Q ^ { \top } Q = \mathbf { I }$

## A.2 Proof for Lemma 2

Proof. Fix $( i , j )$ and let path ${ \mathrm { ~ } _ { T _ { X } } } ( i , j ) = \left\{ e _ { 1 } , \dots , e _ { m } \right\}$ be the (ordered) edge set along the unique tree path from i to $j ,$ where $m = \vert \mathrm { p a t h } _ { T _ { X } } ( i , j )$ |. Write each edge as $e _ { \ell } = ( u _ { \ell } , v _ { \ell } ) \in P _ { X }$ , and set

$$
\delta _ { \ell } : = \| z _ { u _ { \ell } } - z _ { v _ { \ell } } \| _ { 2 } - \| y _ { u _ { \ell } } - y _ { v _ { \ell } } \| _ { 2 } .
$$

Then

$$
\begin{array} { r l r } & { } & { \left| d _ { T _ { X } } ^ { Z } ( i , j ) - d _ { T _ { X } } ^ { Y } ( i , j ) \right| = \left| \displaystyle \sum _ { \ell = 1 } ^ { m } \delta _ { \ell } \right| } \\ & { } & { \qquad \leq \displaystyle \sum _ { \ell = 1 } ^ { m } | \delta _ { \ell } | . } \end{array}
$$

Define $a \in \mathbb { R } ^ { | P _ { X } | }$ indexed by edges $e = ( u , v ) \in P _ { X }$ as

$$
\begin{array} { r l } & { a _ { e } : = \| z _ { u } - z _ { v } \| _ { 2 } - \| y _ { u } - y _ { v } \| _ { 2 } , } \\ & { ~ a : = A _ { Z } [ P _ { X } ] - A _ { Y } [ P _ { X } ] . } \end{array}
$$

The sum above is over the m coordinates $\{ | a _ { e \ell } | \} _ { \ell = 1 } ^ { m }$ , hence by Cauchy–Schwarz,

$$
\sum _ { \ell = 1 } ^ { m } | a _ { e _ { \ell } } | \leq \sqrt { m } \| a \| _ { 2 } = \sqrt { m } \| A _ { Z } [ P _ { X } ] - A _ { Y } [ P _ { X } ] \| _ { 2 } .
$$

Since $m \leq \dim ( T _ { X } )$ for all (i, j), taking the maximum over $( i , j )$ yields the claim.

## A.3 Proof for Theorem 1

Proof. For any pair (i, j), decompose

$$
\begin{array} { r l } & { \| z _ { i } ^ { ( 1 ) } - z _ { j } ^ { ( 1 ) } \| _ { 2 } - \| z _ { i } ^ { ( 2 ) } - z _ { j } ^ { ( 2 ) } \| _ { 2 } } \\ & { = \big ( \| z _ { i } ^ { ( 1 ) } - z _ { j } ^ { ( 1 ) } \| _ { 2 } - d _ { T _ { X } } ^ { Z _ { 1 } } ( i , j ) \big ) } \\ & { \quad + \big ( d _ { T _ { X } } ^ { Z _ { 1 } } ( i , j ) - d _ { T _ { X } } ^ { Z _ { 2 } } ( i , j ) \big ) } \\ & { \quad + \big ( d _ { T _ { X } } ^ { Z _ { 2 } } ( i , j ) - \| z _ { i } ^ { ( 2 ) } - z _ { j } ^ { ( 2 ) } \| _ { 2 } \big ) . } \end{array}\tag{6}
$$

The first and last terms are bounded by stretch since $\| z _ { i } - z _ { j } \| _ { 2 } \leq d _ { T _ { X } } ^ { Z } ( i , j ) \leq$ $\tau \| z _ { i } - z _ { j } \| _ { 2 } \leq \tau R$ , giving $0 \leq d _ { T _ { x } } ^ { Z } ( i , j ) - \| z _ { i } - z _ { j } \| _ { 2 } \leq ( \tau - 1 ) R$ for $Z \in \{ Z _ { 1 } , Z _ { 2 } \}$ The middle term is controlled by Lemma 2 together with the triangle inequality and the definitions of $\varepsilon _ { 1 } , \varepsilon _ { 2 }$ (both $Z _ { 1 }$ and $Z _ { 2 }$ match the same target edge lengths on $P _ { X } ~ \mathrm { u p ~ t o } ~ \varepsilon _ { k } )$ . This yields the uniform bound on absolute distance diferences. The squared-distance matrix bound follows from $| a ^ { 2 } - b ^ { 2 } | = | a - b | | a + b | \leq$ $2 R | a - b |$ and summing over all unordered pairs.

## B Experimental Details

## B.1 CIFAR-10

Data RGB images of size $3 \times 3 2 \times 3 2$ ; pixel intensities rescaled to [0, 1]. Models

Linear AE: flatten input to 3072. Encoder $\mathrm { M L P ~ 3 0 7 2 }  2 5 6  1 2 8  1 2 8  d$ with ReLU after each layer; decoder $d  1 2 8  1 2 8  2 5 6  3 0 7 2$ with ReLU in hidden layers and a sigmoid output (reshaped to $3 \times 3 2 \times 3 2 )$

Conv AE: 4 strided conv blocks with channels $3  1 6  3 2  6 4  d / 4$ (ReLU), then flatten to $z \in \mathbb { R } ^ { d } ;$ decoder uses 4 transposed-conv blocks $d / 4 $ $6 4 \to 3 2 \to 1 6 \to 3$ with ReLU and a sigmoid output.

Training Adam, learning rate $1 0 ^ { - 3 }$ , batch size 128, 200 epochs. The best validation checkpoint is retained. 5 independent runs. Latent size 64.

Stitching Afine transformation estimated using 1000 samples.

## B.2 MNIST

Data: Grayscale images of size $1 \times 2 8 \times 2 8 ;$ pixel intensities rescaled to [0, 1]. Models:

Linear AE: flatten input to 784. Encoder $\mathrm { M L P ~ 7 8 4 }  2 5 6  1 2 8  d$ with ReLU after each layer; decoder $d  1 2 8  2 5 6 $ 784 with ReLU in hidden layers and a sigmoid output (reshaped to $1 \times 2 8 \times 2 8 )$

Conv AE: stride-2 conv stack with channels $1  1 6  3 2  6 4  d / 4 \ ( \mathrm { R e L U } )$ then flatten $\mathrm { ~ t o ~ } z \in \mathbb { R } ^ { d }$ ; decoder uses transposed-conv stack $d / 4  6 4  3 2 $ $1 6 \to 1$ with ReLU and a sigmoid output.

Training: Adam, learning rate $1 0 ^ { - 3 }$ , batch size 128, 150 epochs, Kaiming initialization. The best validation checkpoint is retained. 5 independent runs. Latent size 64.

Stitching: Afine transformation estimated using 1000 samples.

## B.3 Tiny-ImageNet

Data: TinyImageNet-200 RGB images of size $3 \times 6 4 \times 6 4 ;$ ; rescaled to [0, 1]. Models:

Conv AE: encoder with 4 strided conv blocks $3 \to 3 2 \to 6 4 \to 1 2 8 \to 2 5 6$ using BN+GELU, followed by an FC layer to latent $z \in \mathbb { R } ^ { 6 4 }$ ; decoder uses $\mathrm { F C } + 4$ transposed-conv blocks $2 5 6 \to 1 2 8 \to 6 4 \to 3 2 \to 3$ with $\mathrm { B N + G E L U }$ , output is linear clamped to [0, 1].

ViT AE: patch embedding with patch size 8, 4 transformer encoder blocks, then features pooled together, then FC to $z ~ \in ~ \mathbb { R } ^ { 6 4 }$ ; transformer decoder reconstructs image patches. Embedding dim 256, 4 encoder transformer blocks, 2 decoder transformer blocks, 8 heads.

Training: Adam, learning rate $2 \times 1 0 ^ { - 4 }$ , batch size 512, 100 epochs, Kaiming initialization. The best validation checkpoint is retained. 5 independent runs. Latent size 64.

Stitching: Afine transformation estimated using 1000 samples.

## B.4 CelebA

Data: CelebA images resized to $3 \times 1 2 8 \times 1 2 8$ and converted pixel range [0, 1]);   
No augmentation or normalization.

## Models:

Conv AE: encoder with 5 strided conv blocks (GELU) with channels 3→32→ $6 4 \to 1 2 8 \to 2 5 6 \to$ 256 and spatial resolution $1 2 8 \to 6 4 \to 3 2 \to 1 6 \to 8 \to 4$ , then FC to latent $z \in \mathbb { R } ^ { d } ;$ decoder uses FC to $2 5 6 \times 4 \times 4$ followed by 5 transposed-conv blocks (GELU) with a final sigmoid output.

ViT AE: patch-based transformer encoder/decoder with patch size 8; Embedding dim 192, 4 encoder transformer blocks, 4 decoder transformer blocks, 6 heads; latent $z \in \mathbb { R } ^ { d }$

Training: Adam, learning rate $1 0 ^ { - 4 }$ , batch size 128, 100 epochs. The best validation checkpoint is retained. 5 independent runs. Latent size 64.

Stitching: Afine transformation estimated using 1000 samples.

## B.5 CDL-A,C and UMa

Data: CSI tensors are normalized using training-set mean/std, and represented as two channels (real/imag).

Models: End-to-end is performed as Encoder → Quantization → Decoder, with a straight-through quantizer using 2-bit uniform scalar quantization.

Training: Adam, initial learning rate $1 0 ^ { - 3 }$ , batch size 128, 150 epochs, with exponential LR decay every 5 epochs (factor 0.9). The best checkpoint is selected by the lowest held-out validation loss. 5 independent runs.

Stitching: Afine transformation estimated using 1000 samples.