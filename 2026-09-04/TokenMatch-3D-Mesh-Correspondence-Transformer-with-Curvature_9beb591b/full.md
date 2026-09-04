# TokenMatch: 3D Mesh Correspondence Transformer with Curvature-Guided Tokenisation

Adeela Islam<sup>1,2,6,∗</sup> Zorah Lähner<sup>3,4</sup> Vittorio Murino<sup>1,5</sup> Vladislav Golyanik<sup>6,†</sup>

<sup>1</sup>Italian Institute of Technology <sup>2</sup>University of Genoa <sup>3</sup>University of Bonn <sup>4</sup>Lamarr Institute <sup>5</sup>University of Verona <sup>6</sup>Max Planck Institute for Informatics, SIC

https://4dqv.mpi-inf.mpg.de/TokenMatch/

## Abstract

While data-driven 3D shape correspondence estimation has recently seen substantial progress, robust matching under partial observations and strong non-isometric deformations remains challenging. Existing learning-based approaches often rely on hand-crafted descriptors or template-based representations, whereas recent generative models over functional maps suffer from high inference cost, limited interpretability, and poor generalisation to partial shapes. In response to these limitations, this paper introduces TokenMatch, a new transformer-based unified model for estimating 3D shape correspondences. Our feed-forward approach trained exclusively on BeCoS, a challenging non-isometric partial-to-partial shape-matching dataset, can generalise to matching full shapes without retraining or fine-tuning. TokenMatch uses self- and cross-attention mechanisms to efficiently learn patchlevel and point-level relations as well as dense correspondences between shape pairs. Our core insight is that meshes can be adaptively tokenised into patches using shape curvature guidance, enabling effective learning of shape-specific geometric descriptors for correspondence estimation. We evaluate TokenMatch on standard benchmarks for partial and full shape matching, including CP2P, PSMAL, BeCoS, FAUST, SCAPE, and SHREC’19. Our method achieves consistently high performance, in most cases outperforming existing methods for partial and full shape matching in the mean geodesic error and intersection-over-union metrics, while also running faster at sub-second inference speeds.

## 1 Introduction

Establishing accurate correspondences between 3D shapes is a fundamental and open problem in geometry processing [11, 14, 27, 42, 47, 6, 64], with applications including texture transfer [18, 61], segmentation [58], deformation transfer [63], and statistical shape modelling [1, 32]. Given two shapes, the goal of shape correspondence estimation is to identify semantically consistent pointwise correspondences despite variability in pose, sampling, noise, and, critically, degree of partiality [74]. Among existing approaches, the functional map framework [47, 71] has emerged as a powerful paradigm, representing correspondences as compact linear operators in the Laplace–Beltrami spectral domain. Despite substantial progress in full-to-full shape matching [75, 27, 29], partial-shape matching remains a major open challenge: shapes may only partially overlap, increasing the solution space, making the problem particularly ill-posed, and often also requiring identification of the shared region. In addition, existing learning-based approaches often rely on assumptions of complete shapes or near-isometric deformations [3, 26, 72, 50], and are typically trained on small-scale and singlecategory datasets. As a result, they can encode dataset-specific biases that limit robustness to unseen deformations and artefacts and generalisation across partial and full shape matching settings [25].

![](images/3343f02271d459eb0221a9aee46883310dbdd3644a4639387e17a7d88e68b2b9.jpg)  
Figure 1: Overview of TokenMatch, our unified model for 3D mesh correspondence estimation. Given a pair of input meshes (bottom-left), we extract geometry-aware tokens (top-left; Sec. 4.1) and process them with a transformer encoder to obtain shape features (top-right; Secs. 4.2 and 4.4). Cross-attention enables efficient learning of inter-shape interactions, and the resulting representations are used to estimate functional maps. An additional overlap prediction module identifies shared regions between shapes (Sec. 4.4). Our method is first pre-trained in a self-supervised manner using a masked reconstruction objective and then trained on annotated partial-shape matching data using PointInfoNCE loss (Sec. 4.3) and functional map supervision (Sec. 4.5).

We address several open challenges in the field by introducing a new unified model that serves as a foundation for shape correspondence estimation across different settings; see Fig. 1. Our method, which we call TokenMatch, is based on a transformer encoder. Transformers have recently emerged as a dominant paradigm for representation learning due to their ability to model long-range dependencies and flexible interactions between input elements [69, 21, 73]. We argue that their ability to model relationships between variable-sized sets of elements makes them particularly well-suited for partial shape matching, where the number and arrangement of input regions are not known in advance. At the same time, while significant progress has been made in adapting transformers to 3D data, applying them to shape correspondence estimation is non-trivial due to the irregular structure of meshes and the absence of a natural and universal tokenisation.

Hence, the core ingredient of our mesh transformer for 3D shape correspondence is a new curvatureguided tokenisation scheme. Our idea is to represent shapes as collections of geometry-aware surface patches, which serve as transformer tokens. The construction of these patches is guided using local curvature and spectral shape cues, capturing both local geometric detail and intrinsic shape structure. Within our formulation, self-attention models intra-shape geometric relationships and interactions, while cross-attention explicitly captures dense correspondence between shapes. As a result, the designed mesh transformer architecture jointly reasons about geometry and correspondence through attention mechanisms. We choose functional maps [47], i.e., spectral representation for encoding and recovering pointwise correspondences from learned features.

Importantly, TokenMatch is template-free and does not rely on category-specific assumptions, making it applicable across diverse shape categories. Since our method accepts a varying number of tokens per shape, it generalises naturally to full shapes, while not necessarily requiring training or finetuning on them (though it can be done to improve performance). We train TokenMatch on the newly introduced large-scale, curated BeCoS dataset with multiple shape categories [23], which we consider a key factor behind its performance, alongside curvature-guided tokenisation. Notably, BeCoS provides diverse full and partial shape categories (humanoids and animals) with challenging, highly non-isometric deformations at realistic scales, encouraging robustness to deformation patterns that go beyond standard nearly isometric settings. It enables method training, evaluation and—as we demonstrate experimentally in Sec. 5—cross-category zero-shot generalisation of TokenMatch across more than 2500 unique shapes. Last but not least, TokenMatch is a feed-forward approach that achieves sub-second inference time, which is unprecedented in partial-shape matching, especially for combinatorial techniques and when no additional assumptions (e.g., on mesh discretisation) are made. To summarise, the main technical contributions of this paper are as follows:

• TokenMatch, the first mesh-transformer-based unified model for 3D shape correspondence estimation, combining functional maps with self- and cross-attention mechanisms to model geometric relationships and dense correspondences between shape pairs (Sec. 4). Our template-free formulation does not rely on shared templates or category-specific assumptions, enabling broader applicability across shape domains.

• A curvature-guided, geometry-aware, overlapping mesh tokenisation that enables transformer-based processing of irregularly sampled shapes, i.e., meshes with spatially varying sampling density (Sec. 4.1).

To further improve robustness and generalisation of TokenMatch, we employ a masked auto-encoder pre-training strategy, encouraging the model to learn geometry-aware features that are resilient to noise, partiality, and varying sampling densities (Sec. 4.2). To enable correspondence-aware reasoning, we employ cross-attention between features of shape pairs (Sec. 4.3), and we also predict overlaps between the shapes (Sec. 4.4). In the general case, our TokenMatch approach outputs partial functional maps (Secs. 4.5 and 4.6). We plan to release the source code and trained models for research purposes to facilitate reproducibility.

## 2 Related Work

This section reviews methods closely related to ours. For a broader overview of the field, we refer to the surveys on shape correspondence estimation and registration [74, 17, 67].

## 2.1 Shape Correspondence Estimation and 3D Transformer Models

Spectral methods for correspondence estimation between full meshes learn pointwise descriptors that induce functional maps [47, 48], yielding a compact low-rank representation in the Laplace–Beltrami eigenbasis. Deep learning has further enabled learning descriptor functions directly from data [39, 60], evolving from supervised formulations [20, 59] to unsupervised methods [14, 57]. This has led to a range of functional-map-based pipelines [75, 4, 31], often augmented with spatial information to improve robustness [27, 65, 4]. However, the global low-rank structure of functional maps limits fine-grained correspondence recovery, and many methods therefore rely on independently computed per-shape features followed by a separate alignment stage [55].

Partial-shape matching represents one of the central challenges in the field. This variant considers missing geometry and reduced global structure, requiring both pointwise correspondence estimation and identification of overlapping regions. Combinatorial approaches to partial-shape matching are effective under controlled settings, but do not scale well to high-resolution shapes due to their combinatorial computational complexity [56]. Unsupervised methods remain fragile due to limited regularisation and are particularly sensitive to topological noise such as self-intersections or reconstruction artefacts [53, 23, 43]. To improve efficiency, functional map-based methods extend spectral correspondence to partial settings. DPFM [3] introduces an overlap prediction module (vertex-wise) and EchoMatch [72] enforces cycle consistency of pointwise maps as a form of correspondence verification, while also demonstrating the effectiveness of pre-trained features such as DINOv2 [46] for partial matching. These methods often struggle to reliably infer overlapping regions. Their reliance on global spectral structure, vertex-level interactions or explicit overlap heuristics limits their ability to jointly reason about geometry and correspondence at scale.

Transformers [69] have emerged as a dominant architecture in modern deep learning due to their ability to act as a highly scalable, differentiable, general-purpose computational paradigm with minimal inductive bias. By relying on token interactions through self-attention, they naturally model long-range dependencies and global structure without imposing strong assumptions about the underlying signal. This property is particularly appealing for 3D learning, where geometric structure can vary significantly across resolution, topology, and sampling density. Consequently, transformer-based architectures have recently shown strong performance in a range of 3D tasks, including point cloud understanding [34] and mesh-based representation learning [38].

More recent advances further demonstrate the effectiveness of carefully designed transformer backbones for 3D data. In particular, Point Transformer V3 [70] has shown that incorporating geometryaware attention mechanisms while preserving the general-purpose nature of transformers leads to strong performance and scalability on large-scale point cloud tasks. However, many existing approaches for shape correspondence either rely on vertex-level tokenisation or operate in learned latent shape spaces [66], which can obscure fine-grained geometric structure and limit robustness under partial or non-uniform sampling. In contrast, we propose a curvature-driven mesh tokenisation that preserves intrinsic geometric cues while maintaining compatibility with a transformer backbone. The functional map framework combined with transformer-based modelling efficiently captures both spectral shape structure and cross-shape interactions. This enables the first mesh transformer for correspondence estimation that unifies full and partial 3D shape matching within a single framework, leveraging the flexibility and global reasoning capability of transformers while better respecting local surface geometry. Compared to partial-shape matching methods [24, 56, 3, 72], our feed-forward transformer-based formulation achieves the fastest inference among the evaluated methods, with substantial speed-up over the optimisation-based approaches [24, 56].

## 2.2 Data-Driven Methods with Generative Priors

Denoising diffusion probabilistic models (DDPMs) have emerged as a leading paradigm for generative modelling, formulating 2D image synthesis as an iterative denoising process that inverts a gradual corruption with Gaussian noise, often in a conditional setting [62, 33, 51]. The 2D structure of functional maps [47] has motivated connections to generative models, enabling DDPMs to operate on low-dimensional correspondence representations [15]. Recent works explore this direction in different ways and show alternatives to explicit functional map solvers. DenoisFM [75] directly generates functional maps and mitigates Laplacian eigenfunction sign ambiguity via an unsupervised selection strategy based on learned surface features, but relies on pre-defined category-specific shape templates. DiffuMatch [50] operates on absolute-valued functional maps to avoid sign ambiguity, replacing classical constraints such as orthogonality and Laplacian commutativity with learned regularisation. Despite their effectiveness, these approaches introduce significant computational overhead due to iterative sampling and rely on large-scale data to learn reliable priors over the functional map space. Their dependence on category-specific training distributions may limit diversity of the generated samples and cross-category generalisation.

Hence, our approach is distantly related to DenoisFM and DiffuMatch in that it is data-driven; in contrast to them, we leverage an efficient feed-forward transformer model and its inductive biases for estimating shape correspondences. Moreover, compared to DenoisFM, our approach is template-free and is orders of magnitude faster than slow guided DDPM generation.

## 3 Definitions and Background

This section provides core definitions used in this paper and reviews functional maps.

Definitions. Let $\mathcal { M }$ and $\mathcal { N }$ be two 3D shapes represented as triangular meshes, with vertex sets $V _ { \mathcal { M } }$ and $V _ { \mathcal { N } }$ of sizes $v _ { \mathcal { M } } = | V _ { \mathcal { M } } |$ and $v _ { \mathcal { N } } = | V _ { \mathcal { N } } |$ |, respectively. Our goal is to estimate a pointwise mapping $\Pi _ { \mathcal { N M } } : \mathcal { M }  \mathcal { N }$ , which assigns each point on $\mathcal { M }$ to a corresponding point on $\mathcal { N } ;$ in the partial setting, the mapping is defined only on a subset of $\mathcal { M }$ . Direct estimation of $\Pi _ { \mathcal { N M } } \in$ $\{ 0 , \dot { 1 } \} ^ { v _ { \mathcal { N } } \times v _ { \mathcal { M } } }$ is computationally expensive. Therefore, we adopt the functional map framework [47], where correspondences are represented in a reduced spectral basis.

Functional maps. For each shape $\pmb { S } \in \{ \mathcal { M } , \mathcal { N } \}$ , we compute the Laplace–Beltrami operator $\mathbf { L } _ { \mathcal { S } }$ and its first k eigenvectors $\Phi _ { S } \in \mathbb { R } ^ { v _ { S } \times k }$ so that $\mathbf { L } _ { \mathcal { S } } \boldsymbol { \Phi } _ { \mathcal { S } } = \boldsymbol { \Phi } _ { \mathcal { S } } \mathbf { L } _ { \mathcal { S } }$ , where $\Lambda _ { \mathcal { S } }$ contains the eigenvalues. The functional map $\mathbf { C } _ { \mathcal { M N } } \in \mathbb { R } ^ { k \times k }$ is defined as:

$$
\mathbf { C } _ { \mathcal { M N } } = \Phi _ { \mathcal { N } } ^ { \dagger } \Pi _ { \mathcal { N M } } \Phi _ { \mathcal { M } } ,\tag{1}
$$

where $^ { 6 6 } ( \cdot ) ^ { \dag , 9 }$ denotes Moore–Penrose pseudoinverse. The learned feature descriptors $\mathbf { F } _ { \mathcal { M } }$ and $\mathbf { F } _ { \mathcal { N } }$ are projected onto the spectral domain, resulting in $\mathbf { A } _ { \mathcal { M } }$ and $\mathbf { A } _ { \mathcal { N } } \mathbf { : }$

$$
{ \bf A } _ { \mathcal M } = \Phi _ { \mathcal M } ^ { \dagger } { \bf F } _ { \mathcal M } , \quad \mathrm { a n d } \quad { \bf A } _ { \mathcal N } = \Phi _ { \mathcal N } ^ { \dagger } { \bf F } _ { \mathcal N } .\tag{2}
$$

The functional map [47] is obtained as a solution to the following optimisation problem:

$$
\mathbf { C } _ { \mathcal { M N } } = \arg \operatorname* { m i n } _ { \mathbf { C } } \left\| \mathbf { C } \mathbf { A } _ { \mathcal { M } } - \mathbf { A } _ { \mathcal { N } } \right\| _ { F } ^ { 2 } + \lambda \left\| \mathbf { C } \mathbf { A } _ { \mathcal { M } } - \mathbf { A } _ { \mathcal { N } } \mathbf { C } \right\| _ { F } ^ { 2 } ,\tag{3}
$$

with a scalar weight λ and Frobenius norm denoted by $\| \cdot \| _ { F } \ , \|$ . The second term enforces commutativity with the Laplace–Beltrami operators [52]. Our goal is to learn robust feature descriptors that lead to accurate functional maps.

## 4 TokenMatch: Our 3D Mesh Correspondence Transformer

We propose a transformer-based, unified framework as a foundation for partial and full 3D shape correspondence estimation. Given two triangular meshes $\mathcal { M } = ( V _ { \mathcal { M } } , T _ { \mathcal { M } } )$ and $\mathcal { N } = ( V _ { \mathcal { N } } , T _ { \mathcal { N } } )$ our goal is to predict accurate point-wise correspondences between their vertices $V _ { \mathcal { M } } , V _ { \mathcal { N } }$ in a feed-forward manner. Unlike approaches that rely on fixed spectral representations or handcrafted descriptors [47, 52, 20], we operate directly on tokenised mesh geometry. This design avoids committing to a fixed representation or matching pipeline, while enabling scalable joint modelling of local surface geometry and global shape context. It also allows us to leverage a large-scale dataset.

To this end, we adapt the mesh representation into a transformer-friendly format inspired by masked mesh and point cloud auto-encoders [38, 35], enabling effective patch-wise processing of surface geometry. Our method learns expressive features to estimate functional maps, providing a robust solution for non-rigid and non-isometric shape correspondence. Fig. 1 provides an overview of our framework. We first perform geometry-aware tokenisation of the input mesh (Sec. 4.1). These tokens are then processed by a transformer encoder that learns contextualised shape features through pre-training (Sec. 4.2). As independent per-shape features lack cross-shape awareness, we introduce crossattention for feature exchange between shapes

![](images/81ea995425bbc085a220de347b78ba612ffe8c1313caf1a5da817250cdcc0693.jpg)  
Figure 2: Effect of increasing token density. The number of tokens increases from left to right (32, 64, 128, 256). Red dots indicate the token centres. Increasing the number of tokens improves coverage of critical regions, first capturing fingers and later extending to toes. Zoom recommended.

(Sec. 4.3), use the resulting features to predict overlap regions (Sec. 4.4) and, finally, estimate functional maps (Secs. 4.5 and 4.6).

## 4.1 Geometry-Aware Curvature-Guided Mesh Tokenisation

Given a triangular mesh $\begin{array} { r } { { \cal { S } } = ( { \cal { V } } _ { S } , { \cal { T } } _ { S } ) } \end{array}$ , where $V _ { S }$ denotes the set of vertices and $T _ { \mathcal { S } }$ denotes mesh connectivity, we construct g geometry-aware tokens for transformer processing. We partition the

![](images/e444629bc124a584793b98e3dad2c8ff49536e608b3898ef309d2e5c7fedcba3.jpg)  
Shape

![](images/2a59926ca415240fca015964de343fa6aeaf102ccb11c823da387f19ee7cae7a.jpg)  
Curvature

![](images/ac253c69f71d73528c73d738f8cb8c8d5da020feb8ffe35c330c95acc328c924.jpg)  
FPS

![](images/38750cb0e95e240bc76813517f3a771f40184e34f0bcaf0752acceea5a674c5a.jpg)  
Overlap

![](images/64ef048bd4f07117d4c5f9a1b2df63722cc3c5707aad781a701db6bc533e36c8.jpg)  
Tokens

Figure 3: The proposed geometry-aware mesh tokenisation: The process begins with a triangular input mesh representing a human pose. A spectral signal is computed and visualised as a heatmap. Points are sampled using curvature-weighted farthest point sampling (FPS). Soft overlapping regions are then defined using a geodesic influence radius, and mesh vertices are softly assigned to token regions. Final tokens are represented by their sampled centres.

mesh into overlapping regions to capture smoothly varying geometric structure. As discussed in App. B.3, moderate token overlap is important for accurate shape correspondence (see also Table 9).

To guide tokenisation, we compute a per-vertex signal combining local and spectral geometry:

$$
s ( i ) = \alpha \left\| \mathbf { H } ( i ) \right\| _ { 2 } + ( 1 - \alpha ) E ( i ) ,\tag{4}
$$

where $i \in V _ { S } , \mathbf { H } ( i )$ denotes the absolute mean curvature at vertex i, and is therefore non-negative. $\alpha \in [ 0 , 1 ]$ balances local geometric detail and global spectral structure, and $E ( i )$ denotes the spectral energy derived from the Laplace–Beltrami eigenfunctions. We set α to 0.6. Next, let $\phi _ { \ell } ( i )$ denote the value of the ℓ-th eigenfunction at vertex i. We compute the spectral energy as $\begin{array} { r } { E ( i ) = \sum _ { \ell = 5 } ^ { 1 6 } \phi _ { \ell } ( i ) ^ { 2 } } \end{array}$ This mid-frequency band captures stable structural details while excluding low-frequency modes $( \ell < 5 )$ , which mainly encode global shape information, and high-frequency modes $( \dot { \ell } > 1 \dot { 6 } )$ , which are more sensitive to remeshing and geometric noise.

Curvature-aware sampling. We select g token centres using farthest point sampling (FPS) [28] with a signal-weighted geodesic distance. Let $\mathcal { C } _ { t }$ denote the set of selected centres after iteration t, with $\mathcal { C } _ { 0 } = \emptyset$ . At iteration t, the next centre is chosen as follows:

$$
c _ { t } = \arg \operatorname* { m a x } _ { i \in V _ { S } } \left( \operatorname* { m i n } _ { c \in \mathcal { C } _ { t - 1 } } d _ { \mathrm { g e o } } ( i , c ) \cdot ( 1 + \beta s ( i ) ) \right) ,\tag{5}
$$

where $d _ { \mathrm { g e o } ( i , c ) }$ denotes geodesic distance between vertices i and $c , s ( i )$ is the geometry-aware vertex signal, and $\dot { \beta } > 0$ controls the influence of the geometry-aware signal. We choose $\beta \stackrel { \cdot } { = } 0 . 5$

Soft token assignment. Given sampled centres $\{ { c } _ { i } \} _ { i = 1 } ^ { g }$ and mesh element $x ,$ we define a soft assignment of mesh elements to tokens as follows:

$$
w _ { i } ( x ) = \frac { \exp { \left( - d _ { \mathrm { g e o } } ( c _ { i } , x ) ^ { 2 } / 2 \sigma ^ { 2 } \right) } } { \sum _ { k = 1 } ^ { g } \exp { \left( - d _ { \mathrm { g e o } } ( c _ { k } , x ) ^ { 2 } / 2 \sigma ^ { 2 } \right) } } ,\tag{6}
$$

where $w _ { i } ( x )$ denotes the soft assignment weight of mesh element x to token i, representing the degree to which x belongs to the token region centred at $c _ { i } .$ This formulation induces overlapping token regions governed by the geodesic Gaussian bandwidth and g. A broader bandwidth σ produces more diffuse tokens with stronger overlap, while increasing g reduces spacing between token centres, further enhancing overlap. According to Eq. (6), each x contributes to all tokens with $w _ { i } ( x )$ dependent on $d _ { \mathrm { g e o : } }$ , and we do not impose any truncation distance threshold<sup>3</sup>. In an ablation study in App. B.3, we analyse how these design choices affect the final performance (see Table 9). As shown in Figs. 2 and 3, the proposed soft tokenisation offers several advantages over hard surface partitioning: Its overlapping design improves robustness under varying mesh resolution and sampling density, since token representations arise from weighted aggregation rather than rigid assignments. As a result, the representation becomes less sensitive to irregular tessellations and discretisation noise. Conceptually, this can be viewed as a geometry-aware weighting scheme analogous to attention over intrinsic distances, where influence decays smoothly with geodesic distance [7]. Such a formulation enables the transformer to operate on more expressive and geometry-consistent inputs, which is especially beneficial for partial and full-shape correspondence estimation. We further analyse our design choices in $\mathrm { A p p }$ . B.1, where we provide an ablation study on different tokenisation strategies. We demonstrate that our curvature-guided mesh tokenisation yields the most consistent performance across multiple benchmarks, validating its effectiveness compared to alternative mesh tokenisation approaches.

![](images/99c47f482a68cc4a4401f8f46c50e85ee662ec62affcd8c5ecdd163f13037d99.jpg)  
Figure 4: Masked auto-encoding pre-training for mesh transformers. A random subset of tokens is masked out given a geometry-aware tokenised mesh. The transformer encoder $\mathcal { E }$ processes only visible tokens, and the model (including the decoder D) is trained to reconstruct the missing geometric information from the latent representation. This encourages learning of robust and semantically meaningful geometry-aware features that capture both local structure and global shape context.

## 4.2 Transformer Backbone

We now describe how the mesh tokens are processed by a transformer to obtain contextualised shape features. Given the geometry-aware tokenisation, each mesh element $x \in S$ is assigned to token regions via soft weights $w _ { i } ( \bar { x } )$ , as defined in Sec. 4.1 and Eq. (6). The weights $w _ { i } ( x )$ are used to aggregate element-wise descriptors into token features, producing geometric representations that capture both intrinsic and extrinsic surface cues.

Each token is associated with its centre coordinate $\mathbf { c } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ obtained via curvature-aware FPS (Sec. 4.1) and encoded as ${ \bf p } _ { i } = \mathrm { M L P } ( { \bf c } _ { i } )$ . The final token representation is given by $\mathbf { x } _ { i } = \mathbf { e } _ { i } + \mathbf { p } _ { i }$ forming the transformer input sequence ${ \bf H } ^ { ( 0 ) } = ( { \bf x } _ { 1 } , \ldots , { \bf x } _ { q } )$ . We use an $\bf M L P$ to project the aggregated feature vector into the token representation $\mathbf { e } _ { i }$ . The token sequence is then processed through L transformer layers H<sup>(ℓ+1)</sup> = TransformerBlock $( \mathbf { H } ^ { ( \ell ) } )$ , yielding the final contextualised token features $\mathbf { Z } \in \mathbb { R } ^ { g \times d }$ , which are then subsequently used for cross-shape feature interaction and correspondence estimation.

Masked auto-encoding pre-training. To learn geometry-aware representations, we pre-train the mesh transformer using a masked auto-encoding (MAE) objective, following the design of MeshMAE [38]; see Fig. 4. The pre-training encourages the model to recover missing geometric structures from partially observed token sequences, leading to robust and semantically meaningful representations. It simulates partiality during training and, thus, facilitates generalisation across partial-to-partial, full-to-full and partial-to-full matching scenarios.

Given the token embeddings $\mathbf { E } = \{ \mathbf { e } _ { i } \} _ { i = 1 } ^ { g }$ , we randomly sample a subset of token indices $\mathcal { I } _ { m } \subset$ $\{ 1 , \ldots , g \}$ with masking ratio r. Masked tokens are replaced by a learnable mask embedding e<sub>mask</sub>. The uncorrupted (visible) token sequence is processed by the transformer encoder. A lightweight decoder is then used to reconstruct the original geometric embeddings of the masked tokens.

Reconstruction loss. During the self-supervised MAE pre-training, we supervise reconstruction with geometric and feature-level objectives. The Chamfer Distance (CD) loss $\mathcal { L } _ { \mathrm { C D } }$ is computed between the reconstructed and ground-truth point sets as follows:

$$
\mathcal { L } _ { \mathrm { C D } } = \frac { 1 } { | P | } \sum _ { p \in P } \operatorname* { m i n } _ { q \in Q } \| p - q \| _ { 2 } + \frac { 1 } { | Q | } \sum _ { q \in Q } \operatorname* { m i n } _ { p \in P } \| q - p \| _ { 2 } ,\tag{7}
$$

where $P$ and $Q$ denote reconstructed and ground-truth surface samples, respectively. Additionally, we apply a feature reconstruction loss ${ \mathcal { L } } _ { \mathrm { f e a t } }$ on token embeddings for masked patches:

$$
\mathcal { L } _ { \mathrm { f e a t } } = \frac { 1 } { g } \sum _ { i = 1 } ^ { g } \| \hat { \mathbf { e } } _ { i } - \mathbf { e } _ { i } \| _ { 2 } ^ { 2 } ,\tag{8}
$$

where $\hat { \mathbf { e } } _ { i }$ and $\mathbf { e } _ { i }$ are, respectively, the reconstructed and original token embeddings from the encoder. Our final training objective for the transformer backbone reads as follows:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M A E } } = \mathcal { L } _ { \mathrm { f e a t } } + \lambda \mathcal { L } _ { \mathrm { C D } } . } \end{array}\tag{9}
$$

with the weight $\lambda = 0 . 5$ . After pre-training, we discard the decoder and retain only the encoder, as the decoder is required only for the reconstruction objective and not for correspondence estimation.

## 4.3 Cross-Shape Feature Interaction Learning

The encoded representations $\mathbf { Z } _ { \mathcal { M } } \in \mathbb { R } ^ { g _ { \mathcal { M } } \times d }$ and $\mathbf { Z } _ { \mathcal { N } } \in \mathbb { R } ^ { g _ { \mathcal { N } } \times d }$ capture rich intra-shape geometric context through self-attention. These features are computed independently for each shape. To enable correspondence-aware reasoning, we introduce a cross-attention module that allows information exchange between the two shapes. Specifically, we update the features of M by attending to ${ \mathcal { N } } .$ , and symmetrically for ${ \mathcal { N } } ,$ , as follows:

$$
\widetilde { \mathbf { Z } } _ { \mathcal { M } } = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } _ { \mathcal { M } } \mathbf { K } _ { \mathcal { N } } ^ { \top } } { \sqrt { d } } \right) \mathbf { V } _ { \mathcal { N } } , \quad \widetilde { \mathbf { Z } } _ { \mathcal { N } } = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } _ { \mathcal { N } } \mathbf { K } _ { \mathcal { M } } ^ { \top } } { \sqrt { d } } \right) \mathbf { V } _ { \mathcal { M } } ,\tag{10}
$$

where $\mathbf { Q } _ { \mathcal { M } } = \mathbf { Z } _ { \mathcal { M } } \mathbf { W } _ { Q } , \mathbf { K } _ { \mathcal { N } } = \mathbf { Z } _ { \mathcal { N } } \mathbf { W } _ { K }$ , and $\mathbf { V } _ { \mathcal { N } } = \mathbf { Z } _ { \mathcal { N } } \mathbf { W } _ { V }$ are learned projections.

This bidirectional cross-attention mechanism enables each token on one shape to aggregate geometryaware contextual information from the other shape, effectively conditioning both representations on potential correspondences. The resulting features $\tilde { \mathbf { Z } } _ { \mathcal { M } }$ and $\widetilde { \mathbf { Z } } _ { \mathcal { N } }$ encode both intra-shape structure and inter-shape alignment cues, and are, therefore, used for subsequent functional map estimation.

PointInfoNCE loss. To further improve correspondence-aware representation learning, we formulate a contrastive objective for point features. The loss $\mathcal { L } _ { \mathrm { n c e } }$ encourages features of corresponding points to be close in the embedding space, while non-corresponding points are pushed apart. We adopt the PointInfoNCE formulation of Attaiki et al. [3] and Xie et al. [72], which reads as follows:

$$
\mathcal { L } _ { \mathrm { n c e } } ( \widetilde { \mathbf { Z } } _ { \mathcal { M } } , \widetilde { \mathbf { Z } } _ { \mathcal { N } } ) = - \sum _ { ( i , j ) \in \mathcal { P } } \log \frac { \exp ( \widetilde { \mathbf { Z } } _ { \mathcal { M } } ( i ) \cdot \widetilde { \mathbf { Z } } _ { \mathcal { N } } ( j ) / \tau ) } { \sum _ { ( \cdot , k ) \in \mathcal { P } } \exp ( \widetilde { \mathbf { Z } } _ { \mathcal { M } } ( i ) \cdot \widetilde { \mathbf { Z } } _ { \mathcal { N } } ( k ) / \tau ) } .\tag{11}
$$

$\mathcal { P }$ is the set of matched points (ground-truth correspondences), $\tau = 0 . 0 7$ is the temperature parameter, and $\widetilde { \mathbf { Z } } \mathcal { M } ( i )$ is the feature vector of point i extracted from the cross-attended representation $\widetilde { \mathbf { Z } } \mathcal { M }$

## 4.4 Shape Overlap Prediction

Inspired by EchoMatch [72], we incorporate an overlap prediction module to estimate the shared region between M and N. In contrast to Xie et al. [72], which predicts overlap from independently extracted point features, our overlap prediction operates on transformer-based cross-attended features, explicitly allowing us to leverage cross-shape contextual interactions. Given cross-attended features $\widetilde { \mathbf { Z } } _ { \mathcal { M } }$ and $\widetilde { \mathbf { Z } } _ { \mathcal { N } }$ , we compute directional row-stochastic correspondences:

$$
\mathbf { I I } _ { \mathcal { M N } } = \operatorname { S o f t m a x } \left( \frac { \widetilde { \mathbf { Z } } _ { \mathcal { M } } \widetilde { \mathbf { Z } } _ { \mathcal { N } } ^ { \top } } { \tau } \right) \quad \mathrm { a n d } \quad \Pi _ { \mathcal { N M } } = \operatorname { S o f t m a x } \left( \frac { \widetilde { \mathbf { Z } } _ { \mathcal { N } } \widetilde { \mathbf { Z } } _ { \mathcal { M } } ^ { \top } } { \tau } \right) ,\tag{12}
$$

where $\tau = 0 . 1$ is a temperature parameter.

Overlap consistency is then estimated via cycle composition as $\mathbf { P } _ { \mathcal { M } } = \Pi _ { \mathcal { M N } } \mathbf { I I } _ { \mathcal { N M } }$ . The diagonal entries of $\mathbf { P } _ { \mathcal { M } }$ measure self-return probability after a round trip through ${ \mathcal { N } } ,$ , which is aggregated (with local neighbourhood smoothing) into per-vertex overlap scores $\mathbf { p } _ { \mathcal { M } } \in [ 0 , 1 ] ^ { v _ { \mathcal { M } } }$ and $\mathbf { \bar { p } } _ { \mathcal { N } } \mathbf { \bar { \Pi } } \mathbf { \Pi } \mathbf { \bar { e } } \left[ 0 , 1 \right] ^ { v _ { \mathcal { N } } }$ We supervise overlap prediction using ground-truth binary masks $\mathbf { m } _ { \mathcal { M } }$ and $\mathbf { m } _ { \mathcal { N } } .$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { o v } } = \mathrm { B C E } ( \mathbf { m } _ { \mathcal { M } } , \mathbf { p } _ { \mathcal { M } } ) + \mathrm { B C E } ( \mathbf { m } _ { \mathcal { N } } , \mathbf { p } _ { \mathcal { N } } ) , } \end{array}\tag{13}
$$

where $^ { 6  } \mathrm { B C E } ( \cdot , \cdot ) ^ { 5 }$ denotes the binary cross-entropy function.

![](images/ebbb979f0f841544119810789cd07668d1a7e649579825630244973fdc0ad4ec.jpg)  
Figure 5: Qualitative results on CP2P24, PSMAL and BeCoS datasets under varying levels of partiality and degree of non-isometric deformation. Matches are colour-coded columnwise. The full-full (F2F) setting shows results for the model trained on partial-partial (P2P). These examples offer a balanced visualisation of our overall results by TokenMatch. $\scriptstyle \cdots _ { \mathrm { G T } ^ { 3 } }$ denotes ground truth.

## 4.5 Functional Map Generation

Partial functional maps. When M and $\mathcal { N }$ only overlap partially, directly solving Eq. (3) can yield inaccurate correspondences due to mismatched descriptor support [3, 9]. Following prior work [72], we restrict descriptors on $\mathcal { N }$ to the overlapping region using a binary mask $\mathbf { m } _ { \mathcal { N } } \bar { \in } \ \{ 0 , 1 \} ^ { v _ { N } } , \mathrm { i . e . }$ $\dot { \mathbf { F } } _ { \mathcal { N } } = \mathbf { m } _ { \mathcal { N } } \odot \mathbf { F } _ { \mathcal { N } }$ , where $\because \odot ^ { \bullet }$ denotes element-wise multiplication. We then compute $\mathbf { A } _ { \mathcal { M } } =$ $\Phi _ { \mathcal { M } } ^ { \dag } \mathbf { F } _ { \mathcal { M } }$ and $\tilde { \mathbf { A } } _ { \mathcal { N } } = \Phi _ { N } ^ { \dagger } \tilde { \mathbf { F } } _ { \mathcal { N } }$ , and estimate ${ \bf C } _ { \mathcal M N }$ by replacing $\mathbf { A } _ { \mathcal { N } }$ with $\tilde { \mathbf { A } } _ { \mathcal { N } }$ in Eq. (3). Masking only N avoids rank-deficient systems under small overlap.

Functional map loss. We penalise the discrepancy between the predicted map $\mathbf { C } _ { \mathcal { M N } }$ and the ground-truth functional map $\mathbf { C } _ { \mathcal { M N } } ^ { \mathrm { g t } }$ . The functional map loss reads:

$$
\mathcal { L } _ { \mathrm { f m a p } } = \left. \mathbf { C } _ { \mathcal { M N } } - \mathbf { C } _ { \mathcal { M N } } ^ { \mathrm { g t } } \right. _ { F } ^ { 2 } .\tag{14}
$$

## 4.6 Final Training Objective

To summarise, $\mathcal { L } _ { \mathrm { M A E } }$ is used during MAE pre-training. After that, we proceed with supervised training of TokenMatch, jointly optimising for the correspondences, overlaps, and feature representations using a combination of equally-weighted $\mathcal { L } _ { \mathrm { f m a p } } , \mathcal { L } _ { \mathrm { o v } }$ and ${ \mathcal { L } } _ { \mathrm { n c e } } ,$ resulting in the total objective:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { f m a p } } + \mathcal { L } _ { \mathrm { o v } } + \mathcal { L } _ { \mathrm { n c e } } .\tag{15}
$$

## 5 Experimental Evaluation

This section presents a comprehensive evaluation of TokenMatch on both partial and full shape correspondence estimation tasks. We begin by describing the datasets, baseline methods, and evaluation metrics used throughout our experiments (Sec. 5.1). We then provide implementation details (Sec. 5.2) and report quantitative and qualitative results on challenging partial-to-partial matching benchmarks, followed by standard full shape correspondence benchmarks to assess performance under complete geometry (Sec. 5.3). Finally, we analyse the robustness and generalisation of our approach under more challenging settings, including anisotropic remeshing and cross-category transfer, highlighting the stability of TokenMatch across varying shape representations and deformation regimes.

## 5.1 Datasets, Baselines and Evaluation Metrics

Partial-shape matching datasets. We evaluate robustness under partiality using all major benchmarks for partial-to-partial shape correspondence, spanning both near-isometric and strongly nonisometric regimes. We consider (i) the CP2P (Cuts-Partial-to-Partial) [3, 24] dataset, which is derived from SHREC’16 [16] and TOSCA [10] and which contains scale-normalised, pre-aligned partial shapes generated via simulated cuts and primarily evaluates correspondence under controlled, nearisometric conditions across humanoid and animal models; (ii) the PSMAL (PARTIAL-SMAL) [24] dataset derived from SMAL [76], with substantial non-isometric deformations in animal shapes while maintaining normalised scale, thereby increasing geometric variability and task difficulty; and (iii) the BeCoS [23] dataset, a significantly more challenging benchmark than (i) and (ii), which aggregates partial humanoid and animal meshes from diverse datasets [2, 8, 10, 22, 41, 54, 76] and incorporates realistic scaling, strong non-isometric deformations, noise, and varying levels of incompleteness, reflecting more practical and unconstrained shape matching scenarios.

Table 1: Mean IoU (×100) on different partial-to-partial shape matching datasets: CP2P, PSMAL and BeCoS. Prior methods rely on predefined descriptors (XYZ [23] or DINOv2 [46]), while our approach learns features directly from mesh geometry. The best results are shown in bold.
<table><tr><td>Method</td><td>Feature Type</td><td>CP2P24 ↑</td><td>PSMAL ↑</td><td>BeCoS↑</td></tr><tr><td rowspan="2">SM-COMB [56]</td><td>XYZ</td><td>57.86</td><td>54.76</td><td>47.04</td></tr><tr><td>DINOv2</td><td>38.38</td><td>36.61</td><td>48.29</td></tr><tr><td rowspan="2">GC-PPSM [24]</td><td>XYZ</td><td>69.29</td><td>64.34</td><td>49.34</td></tr><tr><td>DINOv2</td><td>49.66</td><td>34.30</td><td>33.14</td></tr><tr><td rowspan="2">DPFM [3]</td><td>XYZ</td><td>63.86</td><td>67.04</td><td>48.18</td></tr><tr><td>DINOv2</td><td>74.15</td><td>73.67</td><td>51.02</td></tr><tr><td rowspan="2">EchoMatch [72]</td><td>XYZ</td><td>80.10</td><td>72.71</td><td>52.40</td></tr><tr><td>DINOv2</td><td>84.72</td><td>84.75</td><td>64.68</td></tr><tr><td>TokenMatch (Ours)</td><td>Learned (mesh)</td><td>85.56</td><td>85.21</td><td>65.25</td></tr></table>

Full-shape matching datasets. We evaluate TokenMatch on three widely used benchmarks for full shape matching: (i) the FAUST dataset [8] of human meshes in pose variations; (ii) the SCAPE dataset [2], consisting of a single subject in diverse poses; and (iii) the SHREC’19 dataset [44], which includes human shapes with substantial variation in body proportions and pose, making it particularly challenging. SHREC’19 is used solely for testing, and we exclude one outlier shape following common practice to keep results comparable [14]. We finetune a partial-shape model for Table 2 and Table 3; all other experiments use a model trained only on BeCoS

Table 2: Mean geodesic error (×100) on FAUST, SCAPE, and SHREC’19. The best results are shown in bold.
<table><tr><td>Method</td><td>FAUST↓ SCAPE↓</td><td>SHREC&#x27;19 ↓</td></tr><tr><td>3D-CODED [30]</td><td>2.50 16.10</td><td>17.30</td></tr><tr><td>TransMatch [66]</td><td>1.70 15.30</td><td>21.00</td></tr><tr><td>DUO-FMNet [19]</td><td>2.50 4.20</td><td>6.40</td></tr><tr><td>GeomFMaps [20]</td><td>1.90 2.40</td><td>7.90</td></tr><tr><td>AttentiveFMaps [37]</td><td>1.90 2.60</td><td>5.80</td></tr><tr><td>ConsistentFMaps [64]</td><td>2.30 2.60</td><td>3.80</td></tr><tr><td>SSL [12]</td><td>2.00 3.10</td><td>4.00</td></tr><tr><td>DiffZO [40]</td><td>1.90 2.40</td><td>4.20</td></tr><tr><td>ULRSSM [14]</td><td>1.60 2.20</td><td>5.70</td></tr><tr><td>SmS [13]</td><td>1.40 3.30</td><td>6.20</td></tr><tr><td>DenoisFM [75]</td><td>1.70 2.10</td><td>3.90</td></tr><tr><td>TokenMatch (Ours)</td><td>1.72 2.09</td><td>3.45</td></tr></table>

partial-to-partial data. Performance on the standard FAUST and SCAPE benchmarks has largely saturated [74]. We include them for completeness, as our focus is on more challenging correspondence estimation scenarios and cross-setting generalisability.

Baselines. Despite its high practical relevance, partial-to-partial shape matching remains underexplored. Existing approaches for this setting include learning-based (EchoMatch [72] and DPFM [3]) and axiomatic techniques (SM-COMB [56] and GC-PPSM [24]). In addition, we consider recent methods for non-rigid full shape correspondence, which can be broadly grouped into templatebased learning and descriptor-based methods. The first category includes large-scale templatebased models trained on synthetic datasets such as SURREAL [68], including 3D-CODED [30] and TransMatch [66], which learn dense correspondences across shapes in a canonical template framework using full-shape supervision. The second category consists of descriptor-based approaches that compute correspondences from learned or handcrafted shape features, including ULRSSM [14], DiffZO [40], ConsistentFMaps [64], GeomFMaps [20], AttentiveFMaps [37], DUO-FMNet [19], SSL [12], and SmS [13]. These methods rely on full-shape training or precomputed shape descriptors.

While all these methods are developed and evaluated in the full-shape setting, TokenMatch is trained exclusively on annotated partial-to-partial correspondences yet generalises to full-shape matching without additional training.

Evaluation metrics. Following established practices [72, 3], we report two evaluation measures: (1) mIoU (Mean Intersection over Union), which quantifies overlap prediction quality by computing the intersection over union between predicted and ground-truth binary masks over vertices, where higher values indicate better overlap estimation; and (2) GE (Geodesic Error) [36], which measures correspondence accuracy. In the partial-to-partial setting, GE is computed using geodesic distances normalised by the square root of the full-shape area to ensure scale invariance. We report mean GE, which is evaluated over the predicted overlapping region.

## 5.2 Implementation Details

For pre-training, following Liang et al. [38], we use ViT-Base, a transformer backbone with 12 layers, a hidden width of 768, an MLP dimension of 3072, and 12 attention heads, with $g = 2 5 6$ tokens used for all shapes. For overlap prediction, following Xie et al. [72], we use DiffusionNet [60] blocks with a hidden dimension of 16 and k = 50 Laplace–Beltrami eigenfunctions, such that the functional map $C \in \mathbb { R } ^ { 5 0 \times 5 0 }$ . We initialise the soft correspondence temperature to $\tau = 0 . 1$ and treat it as a learnable parameter during training. Final point-to-point correspondences within the predicted overlap region are obtained via nearest-neighbour matching in the feature space.

We conduct all experiments on a single node with two NVIDIA A100 80GB PCIe GPUs. The implementation is based on CUDA 13.1 [45]. No other processes are executed during training.

## 5.3 Experimental Results

We evaluate TokenMatch in partial-shape matching scenarios; see Table 1. Our approach outperforms all baselines across CP2P [3, 24], PSMAL [24], and BeCoS [23].

The improvements are pronounced on BeCoS, where strong geometric variability and missing regions make correspondence estimation highly challenging. Previous methods rely on predefined descriptors (XYZ and DINOv2 [46]). Among them, EchoMatch [72] achieves the second-best performance, whereas we learn geometry-aware representations and features directly from the mesh structure. This highlights the benefit of task-specific, geometry-native representations, together with masked pre-training, for robust correspondence learning under incomplete geometry across all benchmarks.

Fig. 5 shows qualitative results under partial and full shape matching. Table 2 reports results on FAUST [8], SCAPE [2], and SHREC’19 [44], i.e., standard benchmarks for full-shape matching. Our TokenMatch achieves consistently strong performance across all datasets, with particularly notable gains on the challenging (due to different mesh connectivities) SHREC’19 [44] benchmark. While some methods achieve competitive performance on individual datasets [50, 3], they often degrade under larger test distribution shifts. SmS [13] and ULRSSM [14] rank best on FAUST. Our approach is competitive with TransMatch [66] and DenoisFM [75] among the five most accurate methods, while also generalising from partial-to-partial training to full-shape matching without addi tional training. TokenMatch maintains stable performance across all settings, indicating improved generalisation to diverse non-rigid deformations. Additional ablation studies covering tokenisation design, geometric representation (mesh vs. point cloud), cross-shape interaction, token overlap degree and number of mesh tokens are included in Apps. B.1–B.4. Curvature guidance with soft token assignment, i.e., our main technical contribution, results in the strongest model.

Table 3: Results on anisotropic FAUST and SCAPE datasets. Bold highlights best results.
<table><tr><td>Model</td><td>FAUST_a ↓</td><td>SCAPE_a↓</td></tr><tr><td>3D-CODED [30] TransMatch [66]</td><td>2.90 2.70</td><td>16.90 16.20</td></tr><tr><td>DUO-FMNet [19]</td><td>3.00</td><td>4.40</td></tr><tr><td>GeomFMaps [20]</td><td>3.20 2.40</td><td>3.80</td></tr><tr><td>AttentiveFMaps [37] ConsistentFMaps [64]</td><td>2.60</td><td>2.80 2.70</td></tr><tr><td>ULRSSM [14]</td><td>1.90</td><td>2.40</td></tr><tr><td>DenoisFM [75]</td><td>2.20</td><td>2.20</td></tr><tr><td>TokenMatch (Ours)</td><td>1.90</td><td>1.70</td></tr></table>

Robustness to anisotropic remeshing. To assess the sensitivity of different methods to mesh connectivity, we evaluate our method on anisotropically remeshed versions of FAUST and SCAPE [2, 8].

![](images/9400400a15e2f5c2b4ecbc45bd337fce078686014c6bb09f09b26ed4c6cbe66f.jpg)  
Figure 6: Qualitative results on the BeCoS dataset [23]. Correspondences are shown under varying levels of partiality and deformation, with matches colour-coded columnwise. Notably, despite being trained exclusively on partial-to-partial (P2P) correspondences, the model generalises effectively to both full-to-full (F2F) and partial-to-full (P2F) settings. “GT” denotes ground truth.

These meshes exhibit non-uniform triangle sizes, with regions of both dense and coarse sampling, making them challenging for methods that depend on consistent connectivity or local neighbourhood structure [30, 66, 19]. The performance of most competing approaches degrades noticeably under these conditions, reflecting their sensitivity to irregular discretisation; see Table 3. In contrast, Token-Match remains stable and consistently accurate across both standard and anisotropic meshing settings. We believe this robustness stems from curvature-guided tokenisation, which captures geometry independent of mesh resolution, and the transformer operating on token-based representations rather than explicit connectivity, yielding stable and high accuracy across varying mesh structures.

Generalisation across categories. Table 4 reports results on SMAL [76] animal categories. Our model achieves the lowest mean GE across all but one category, including challenging cases such as hippo, horse and lion, where shape variability is high and local geometry is less consistent than in simpler shapes. Moreover, we evaluate a model (trained on the BeCoS [23] partial-to-partial shapes) on the BeCoS full-to-full scenario; see Figs. 5 and 6 for qualitative results. Table 5 and Fig. 6 demonstrate the strong generalisability of TokenMatch: Despite being trained exclusively on partial-to-partial correspondences, the model successfully handles full-to-full and partial-to-full matching while maintaining consistently accurate correspondences

Table 4: Results on SMAL [76].
<table><tr><td>Class</td><td>Ours</td><td>[75] [14]</td><td>[64]</td></tr><tr><td>wolf</td><td>3.40</td><td>3.50 3.50</td><td>4.40</td></tr><tr><td>dog</td><td>3.30</td><td>3.50</td><td>3.50 4.50</td></tr><tr><td>horse</td><td>3.60</td><td>3.70</td><td>3.80 4.40</td></tr><tr><td>cow</td><td>3.30</td><td>3.80</td><td>3.80 4.40</td></tr><tr><td>lion</td><td>3.60</td><td>4.00</td><td>3.80 4.90</td></tr><tr><td>fox</td><td>4.10</td><td>4.70</td><td>3.60 4.90</td></tr><tr><td>hippo</td><td>7.40</td><td>7.60</td><td>9.90 7.90</td></tr><tr><td>Mean</td><td>4.10</td><td>4.30</td><td>4.30 4.80</td></tr></table>

under severe partiality and deformation. There are only a few artefacts in the transferred colours.

Training and inference time. Our method requires ≈six hours on BeCoS (partial-to-partial) for training and 0.16 seconds per shape pair at inference (overlap prediction, functional map recovery and nearestneighbour recovery for point matches), measured on a single node with two NVIDIA A100 80GB PCIe GPUs. Table 6 shows that TokenMatch is comparable in training cost and faster at inference than

Table 5: Mean IoU results under different training and testing settings on the BeCoS dataset.
<table><tr><td>Setting</td><td>Protocol</td><td>Mean IoU (×100)↑</td></tr><tr><td>Partial-to-Partial</td><td>Trained and Tested</td><td>65.25</td></tr><tr><td>Partial-to-Full</td><td>Tested only</td><td>55.61</td></tr><tr><td>Full-to-Full</td><td>Tested only</td><td>71.85</td></tr></table>

DPFM and EchoMatch. It is also substantially faster than optimisation-based methods [24, 56].

Masked pre-training and robustness to noise. As we show in App. B.5, masked pre-training further enhances partial shape matching by improving the model’s ability to reason under missing or incomplete geometric observations. In App. B.6, we expose our method to different levels of added noise during inference to assess its robustness. We find that TokenMatch remains robust to moderate levels of noise without noise augmentation during training.

## 6 Conclusion

We demonstrate a generalisable and unified model, serving as a foundation for partial and full shapematching tasks. Our extensive experiments show consistent improvements over prior methods across standard benchmarks, along with strong robustness of our TokenMatch approach to partiality, non-rigid (non-isometric) deformations, and varying geometric complexity. Global reasoning enabled by transformer attention, when coupled with functional map supervision, consistently outperforms local descriptor-based approaches, particularly in challenging partial correspondence scenarios. This paper also demonstrates that geometry-native representations exhibit stronger transferability than fixed or generic pre-trained features such as raw XYZ embeddings or DINOv2 features, emphasising the importance of task-aligned geometric encoding.

Table 6: Training and inference times.
<table><tr><td>Method</td><td>Training (whole set)</td><td>Inference (per shape pair)</td></tr><tr><td>SM-COMB [56]</td><td>N/A</td><td>0.1 h</td></tr><tr><td>GC-PPSM [24]</td><td>N/A</td><td>3.1 h</td></tr><tr><td>DPFM [3]</td><td>4.8 h</td><td>0.18 s</td></tr><tr><td>EchoMatch [72]</td><td>5.4 h</td><td>0.20 s</td></tr><tr><td>Ours</td><td>6h</td><td>0.16 s</td></tr></table>

We experimentally validate our main technical contribution, i.e., curvature-guided mesh tokenisation with patch overlapping, which is necessary to achieve the reported accuracy of TokenMatch. Notably, training exclusively on partial correspondences is sufficient to achieve strong full-shape matching performance at inference time, without architectural modification or retraining. This effectively unifies full-to-full matching as a special case of partial-to-partial correspondence estimation, thereby reducing reliance on complete shape datasets. Along with its generalisability, TokenMatch is computationally efficient, requiring a fraction of a second per shape pair while maintaining high matching accuracy.

Future Work. The proposed TokenMatch approach relies on geodesic computations, which can become expensive at high mesh resolutions (e.g., higher than what is common in the current datasets). We include qualitative examples of observed inaccuracies: Attentive readers can recognise them with the help of the transferred colours in Figs. 5 and 6, as well as Figs. 11 and 13 in the Appendix. Future work will explore scalable alternatives to geodesic processing, improve robustness to noisy, unstructured geometry and extend the framework to large-scale real-world scan collections. We hope this work will open up new perspectives towards (i) a fully-fledged foundation model for shape correspondence estimation and towards (ii) addressing long-standing challenges of this research field.

## References

[1] B. Allen, B. Curless, and Z. Popovic. The space of human body shapes: reconstruction and parameterization´ from range scans. ACM Trans. Graph., 22(3), 2003.

[2] D. Anguelov, P. Srinivasan, D. Koller, S. Thrun, J. Rodgers, and J. Davis. Scape: shape completion and animation of people. In ACM SIGGRAPH. 2005.

[3] S. Attaiki, G. Pai, and M. Ovsjanikov. Dpfm: Deep partial functional maps. In International Conference on 3D Vision (3DV), pages 175–185. IEEE, 2021.

[4] L. Bastian, Y. Xie, N. Navab, and Z. Lähner. Hybrid functional maps for crease-aware non-isometric shape matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3313–3323, 2024.

[5] M. Belkin and P. Niyogi. Laplacian eigenmaps and spectral techniques for embedding and clustering. Advances in neural information processing systems, 14, 2001.

[6] M. S. Benkner, Z. Lähner, V. Golyanik, C. Wunderlich, C. Theobalt, and M. Moeller. Q-match: Iterative shape matching via quantum annealing. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 7586–7596, 2021.

[7] S. Bhardwaj, A. Vinod, S. Bhattacharya, A. Koganti, A. S. Ellendula, and B. Reddy. Curvature informed furthest point sampling. arXiv preprint arXiv:2411.16995, 2024.

[8] F. Bogo, J. Romero, M. Loper, and M. J. Black. Faust: Dataset and evaluation for 3d mesh registration. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3794–3801, 2014.

[9] A. Bracha, T. Dagès, and R. Kimmel. On unsupervised partial shape correspondence. In Proceedings of the Asian Conference on Computer Vision, 2024.

[10] A. M. Bronstein, M. M. Bronstein, and R. Kimmel. Numerical geometry of non-rigid shapes. Springer Science & Business Media, 2008.

[11] M. M. Bronstein and I. Kokkinos. Scale-invariant heat kernel signatures for non-rigid shape recognition. In IEEE computer society conference on computer vision and pattern recognition. IEEE, 2010.

[12] D. Cao and F. Bernard. Self-supervised learning for multimodal non-rigid 3d shape matching. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17735– 17744, 2023.

[13] D. Cao, M. Eisenberger, N. El Amrani, D. Cremers, and F. Bernard. Spectral meets spatial: Harmonising 3d shape matching and interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3658–3668, 2024.

[14] D. Cao, P. Roetzer, and F. Bernard. Unsupervised learning of robust spectral shape matching. ACM Transactions on Graphics, 2023.

[15] A. Cohen Rimon, M. Ben-Chen, and O. Litany. Fridu: Functional map refinement with guided image diffusion. Computer Graphics Forum, 44(5):e70203, 2025.

[16] L. Cosmo, E. Rodola, M. M. Bronstein, A. Torsello, D. Cremers, Y. Sahillioglu, et al. Shrec’16: Partialˇ matching of deformable shapes. In Eurographics Workshop on 3D Object Retrieval, EG 3DOR, pages 61–67. Eurographics Association, 2016.

[17] B. Deng, Y. Yao, R. M. Dyke, and J. Zhang. A survey of non-rigid 3d registration. In Computer Graphics Forum, pages 559–589. Wiley Online Library, 2022.

[18] H. Q. Dinh, A. Yezzi, and G. Turk. Texture transfer during shape transformation. ACM Transactions on Graphics (ToG), 24(2):289–310, 2005.

[19] N. Donati, E. Corman, and M. Ovsjanikov. Deep orientation-aware functional maps: Tackling symmetry issues in shape matching. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 742–751, 2022.

[20] N. Donati, A. Sharma, and M. Ovsjanikov. Deep geometric functional maps: Robust feature learning for shape correspondence. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8592–8601, 2020.

[21] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[22] R. M. Dyke, Y.-K. Lai, P. L. Rosin, S. Zappalà, S. Dykes, D. Guo, K. Li, R. Marin, S. Melzi, and J. Yang. Shrec’20: Shape correspondence with non-isometric deformations. Computers & Graphics, 92:28–43, 2020.

[23] V. Ehm, N. E. Amrani, Y. Xie, L. Bastian, M. Gao, W. Wang, L. Sang, D. Cao, T. Weißberg, Z. Lähner, et al. Beyond complete shapes: A benchmark for quantitative evaluation of 3d shape surface matching algorithms. In Computer Graphics Forum. Wiley Online Library, 2025.

[24] V. Ehm, M. Gao, P. Roetzer, M. Eisenberger, D. Cremers, and F. Bernard. Partial-to-partial shape matching with geometric consistency. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27488–27497, 2024.

[25] V. Ehm, P. Roetzer, F. Bernard, and D. Cremers. An integer linear programming approach to geometrically consistent partial-partial shape matching. In Proceedings ofthe International Conference on 3D Vision (3DV), 2026.

[26] V. Ehm, P. Roetzer, M. Eisenberger, M. Gao, F. Bernard, and D. Cremers. Geometrically consistent partial shape matching. In International Conference on 3D Vision (3DV). IEEE Computer Society, 2024.

[27] M. Eisenberger, Z. Lahner, and D. Cremers. Smooth shells: Multi-scale shape registration with functional maps. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12265–12274, 2020.

[28] Y. Eldar, M. Lindenbaum, M. Porat, and Y. Y. Zeevi. The farthest point strategy for progressive image sampling. IEEE Transactions on Image Processing (TIP), 6(9):1305–1315, 1997.

[29] D. Ezuz, B. Heeren, O. Azencot, M. Rumpf, and M. Ben-Chen. Elastic correspondence between triangle meshes. In Computer Graphics Forum, pages 121–134. Wiley Online Library, 2019.

[30] T. Groueix, M. Fisher, V. G. Kim, B. C. Russell, and M. Aubry. 3d-coded: 3d correspondences by deep deformation. In Proceedings ofthe european conference on computer vision (ECCV), 2018.

[31] O. Halimi, O. Litany, E. Rodola, A. M. Bronstein, and R. Kimmel. Unsupervised learning of dense shape correspondence. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4370–4379, 2019.

[32] N. Hasler, C. Stoll, M. Sunkel, B. Rosenhahn, and H.-P. Seidel. A statistical model of human pose and body shape. Computer Graphics Forum, 28(2):337–346, 2009.

[33] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[34] S.-M. Hu, Z.-N. Liu, M.-H. Guo, J.-X. Cai, J. Huang, T.-J. Mu, and R. R. Martin. Subdivision-based mesh convolution networks. ACM Transactions on Graphics (TOG), 41(3):1–16, 2022.

[35] L. Jiang, Z. Yang, S. Shi, V. Golyanik, D. Dai, and B. Schiele. Self-supervised pre-training with masked shape prediction for 3d scene understanding. In Computer Vision and Pattern Recognition (CVPR), 2023.

[36] V. G. Kim, Y. Lipman, and T. Funkhouser. Blended intrinsic maps. ACM transactions on graphics (TOG), 30(4):1–12, 2011.

[37] L. Li, N. Donati, and M. Ovsjanikov. Learning multi-resolution functional maps with spectral attention for robust shape matching. Advances in Neural Information Processing Systems, 35:29336–29349, 2022.

[38] Y. Liang, S. Zhao, B. Yu, J. Zhang, and F. He. Meshmae: Masked autoencoders for 3d mesh data analysis. In European Conference on Computer Vision, 2022.

[39] O. Litany, T. Remez, E. Rodola, A. Bronstein, and M. Bronstein. Deep functional maps: Structured prediction for dense shape correspondence. In Proceedings of the IEEE international conference on computer vision, pages 5659–5667, 2017.

[40] R. Magnet and M. Ovsjanikov. Memory-scalable and simplified functional map learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4041–4050, 2024.

[41] R. Magnet, J. Ren, O. Sorkine-Hornung, and M. Ovsjanikov. Smooth non-rigid shape matching via effective dirichlet energy optimization. In International Conference on 3D Vision (3DV), pages 495–504. IEEE, 2022.

[42] R. Marin, E. Corona, and G. Pons-Moll. Nicp: neural icp for 3d human registration at scale. In European Conference on Computer Vision, pages 265–285. Springer, 2024.

[43] G. Mei, F. Poiesi, C. Saltori, J. Zhang, E. Ricci, and N. Sebe. Overlap-guided gaussian mixture models for point cloud registration. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 4511–4520, 2023.

[44] S. Melzi, R. Marin, E. Rodolà, U. Castellani, J. Ren, A. Poulenard, P. Wonka, and M. Ovsjanikov. Shrec’19: matching humans with different connectivity. In Eurographics Workshop on 3D Object Retrieval. The Eurographics Association, 2019.

[45] NVIDIA Corporation. Cuda toolkit 13.1 - release notes, 2025.

[46] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2023.

[47] M. Ovsjanikov, M. Ben-Chen, J. Solomon, A. Butscher, and L. Guibas. Functional maps: a flexible representation of maps between shapes. ACM Transactions on Graphics (ToG), pages 1–11, 2012.

[48] M. Ovsjanikov, E. Corman, M. Bronstein, E. Rodolà, M. Ben-Chen, L. Guibas, F. Chazal, and A. Bronstein. Computing and processing correspondences with functional maps. In SIGGRAPH ASIA 2016 Courses, pages 1–60. 2016.

[49] E. Papadopoulou and D.-T. Lee. A new approach for the geodesic voronoi diagram of points in a simple polygon and other restricted polygonal domains. Algorithmica, 20(4):319–352, 1998.

[50] E. Pierson, L. Li, A. Dai, and M. Ovsjanikov. Diffumatch: Category-agnostic spectral diffusion priors for robust non-rigid shape matching. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 5745–5756, 2025.

[51] R. Po, W. Yifan, V. Golyanik, K. Aberman, J. T. Barron, A. Bermano, E. Chan, T. Dekel, A. Holynski, A. Kanazawa, et al. State of the art on diffusion models for visual computing. In Computer graphicsforum, page e15063. Wiley Online Library, 2024.

[52] J. Ren, M. Panine, P. Wonka, and M. Ovsjanikov. Structured regularization of functional map computations. In Computer Graphics Forum, pages 39–53. Wiley Online Library, 2019.

[53] E. Rodolà, L. Cosmo, M. M. Bronstein, A. Torsello, and D. Cremers. Partial functional correspondence. In Computer graphicsforum, pages 222–236, 2017.

[54] E. Rodola, S. Rota Bulo, T. Windheuser, M. Vestner, and D. Cremers. Dense non-rigid shape correspondence using random forests. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4177–4184, 2014.

[55] E. Rodolà, M. Moeller, and D. Cremers. Point-wise map recovery and refinement from functional correspondence. In Int’l Symposium on Vision, Modeling and Visualization (VMV), 2015.

[56] P. Roetzer, P. Swoboda, D. Cremers, and F. Bernard. A scalable combinatorial solver for elastic geometri cally consistent 3d shape matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 428–438, 2022.

[57] J.-M. Roufosse, A. Sharma, and M. Ovsjanikov. Unsupervised deep learning for structured shape matching. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 1617–1627, 2019.

[58] R. M. Rustamov. Laplace-beltrami eigenfunctions for deformation invariant shape representation. In Symposium on geometry processing, volume 257, pages 225–233, 2007.

[59] A. Sharma and M. Ovsjanikov. Weakly supervised deep functional map for shape matching. Advances in Neural Information Processing Systems, 33:19264–19275, 2020.

[60] N. Sharp, S. Attaiki, K. Crane, and M. Ovsjanikov. Diffusionnet: Discretization agnostic learning on surfaces. ACM Transactions on Graphics (ToG), pages 1–16, 2022.

[61] S. Shimada, V. Golyanik, E. Tretschk, D. Stricker, and C. Theobalt. Dispvoxnets: Non-rigid point set alignment with supervised learning proxies. In International Conference on 3D Vision (3DV), 2019.

[62] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.

[63] R. W. Sumner and J. Popovic. Deformation transfer for triangle meshes. ´ ACM Transactions on graphics (TOG), 23(3):399–405, 2004.

[64] M. Sun, S. Mao, P. Jiang, M. Ovsjanikov, and R. Huang. Spatially and spectrally consistent deep functional maps. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14497–14507, 2023.

[65] R. Sundararaman, N. Donati, S. Melzi, E. Corman, and M. Ovsjanikov. Deformation recovery: Localized learning for detail-preserving deformations. ACM Transactions on Graphics (TOG), 43(6):1–16, 2024.

[66] G. Trappolini, L. Cosmo, L. Moschella, R. Marin, S. Melzi, and E. Rodolà. Shape registration in the time of transformers. Advances in Neural Information Processing Systems, 34:5731–5744, 2021.

[67] O. Van Kaick, H. Zhang, G. Hamarneh, and D. Cohen-Or. A survey on shape correspondence. In Computer graphicsforum, pages 1681–1707. Wiley Online Library, 2011.

[68] G. Varol, J. Romero, X. Martin, N. Mahmood, M. J. Black, I. Laptev, and C. Schmid. Learning from synthetic humans. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 109–117, 2017.

[69] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[70] X. Wu, L. Jiang, P.-S. Wang, Z. Liu, X. Liu, Y. Qiao, W. Ouyang, T. He, and H. Zhao. Point transformer v3: Simpler faster stronger. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4840–4851, 2024.

[71] Y. Xie, L. Bastian, C. Deng, T. W. Mitchel, M. Gao, and D. Cremers. Deepshapematchingkit: Accelerated functional map solver and shape matching pipelines revisited. arXiv preprint arXiv:2604.10377, 2026.

[72] Y. Xie, V. Ehm, P. Roetzer, N. El Amrani, M. Gao, F. Bernard, and D. Cremers. Echomatch: Partial-topartial shape matching via correspondence reflection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11665–11675, 2025.

[73] Y. Yu, W. Xiong, W. Nie, Y. Sheng, S. Liu, and J. Luo. Pixeldit: Pixel diffusion transformers for image generation. arXiv preprint arXiv:2511.20645, 2025.

[74] A. Zhuravlev, L. Bastian, D. Cao, N. El Amrani, P. Roetzer, V. Ehm, R. Marin, H. Nishizawa, S. Morishima, C. Theobalt, N. Navab, D. Cremers, F. Bernard, Z. Lähner, and V. Golyanik. Non-rigid 3d shape correspondences: From foundations to open challenges and opportunities. Computer Graphics Forum, 2026.

[75] A. Zhuravlev, Z. Lähner, and V. Golyanik. Denoising functional maps: Diffusion models for shape correspondence. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26899–26909, 2025.

[76] S. Zuffi, A. Kanazawa, D. W. Jacobs, and M. J. Black. 3d menagerie: Modeling the 3d shape and pose of animals. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6365–6373, 2017.

# TokenMatch: 3D Mesh Correspondence Transformer with Curvature-Guided Tokenisation

Supplementary Material

This Appendix provides additional technical details and experimental results. App. A outlines dataset preprocessing and evaluation protocols across all benchmarks used in both partial and full-shape matching settings.

In App. B, we provide an in-depth analysis of our mesh tokenisation design space (App. B.1) and an ablation study (variant without cross-attention) (App. B.2) that motivate our curvature-guided overlapping formulation. We further investigate the sensitivity of key hyperparameters, such as degree of token overlap (App. B.3), the number of tokens (App. B.4), and masking ratio (App. B.5). We also evaluate the robustness and accuracy of TokenMatch to increasingly noisy inputs (App. B.6).

Finally, App. C provides additional qualitative results using the CP2P24 protocol and PSMAL dataset [24] to test our method under varying levels of partiality and non-isometric deformation. These visualisations show that TokenMatch produces accurate and geometrically consistent correspondences even in challenging partial-to-partial settings with limited overlap and incomplete local structure. We also report more results on BeCoS, which features realistic scenes with noise, missing regions, and strong deformations. Our model accurately matches full shapes across different degrees of partiality, with stable correspondences preserving semantic alignment under large geometric variations.

## A Dataset Split Details

We evaluate our method on both partial-to-partial and full shape matching tasks. For partial matching, we use three benchmark datasets: CP2P [3, 24], PSMAL [24], and BeCoS [23]. For full shape matching, we additionally consider the complete shape settings provided by full-to-full shape datasets. Below, we detail the dataset splits used in our experiments.

CP2P. The CP2P (Cuts-Partial-to-Partial) benchmark was introduced in Attaiki et al. [3] and is based on the SHREC16 [16] and TOSCA [10] datasets. In our main experiments, we follow the CP2P24 protocol [3, 24], which uses 120 shapes from the SHREC16 CUTS training set and evaluates on 100 test pairs sampled from 153 shapes in the CUTS24 test split of SHREC16.

PSMAL. The PSMAL dataset [26] is derived from SMAL [76] and consists of normalised nonisometric partial animal shapes across eight categories. We follow the standard evaluation protocol, which defines species-disjoint training and test splits over 49 shapes, resulting in 273 training pairs and 100 test pairs.

BeCoS. The large-scale BeCoS benchmark [23] contains realistically scaled non-isometric partial shapes of humans and animals. It provides 10,185 training instances, 137 validation instances, and 142 test instances.

FAUST. The FAUST dataset [8] consists of 100 watertight human meshes covering ten identities in ten poses. Following the standard remeshing protocol [20], an 80/20 split is used for training and testing, respectively.

SCAPE. The SCAPE dataset [2] contains 71 meshes of a single subject in different poses. The standard evaluation uses the first 51 shapes for training and the remaining 20 shapes for testing.

SHREC’19. The SHREC’19 benchmark [44] includes 44 shapes with diverse body types and poses, and is commonly used as a test-only setting. No training split is defined, and evaluation is performed on all available shapes except shape 40, which is excluded because it is non-watertight.

![](images/9d57b941868bb36f0822a749b126e68ecb9a776fd4e8c992f11acbd43c03d4de.jpg)  
(a) Spectral Laplacian

![](images/3fcee4ae415d87b6ba729e4692d0927de6c1b317cc27af8eec43a67b571edb82.jpg)  
(b) HKS clustering

![](images/45925228d74653405b55e1c6a7d70a54dde68d35206d0c19532ee0685fbc30a8.jpg)  
(c) Face splitting

![](images/adf92d0960178dc2fbf01fd48856241738b95aa03e7293ab196027f1ce0ddaf4.jpg)  
(d) Voronoi (FPS)

Figure 7: Tokenisation design exploration. We compare alternative mesh tokenisation strategies. Spectral clustering groups vertices in a global embedding space, HKS uses diffusion-based geometric descriptors, face-based splitting enforces local structure via hierarchical subdivision, and Voronoi partitioning provides uniform geodesic coverage. These highlight trade-offs between global structure, locality, and spatial continuity, motivating our curvature-guided overlapping tokenisation.

## B Ablation Studies

## B.1 Tokenisation Design Exploration

While the proposed TokenMatch relies on curvature-guided tokenisation, we explore several alternative mesh tokenisation strategies before arriving at our final curvature-guided overlapping design, each revealing different trade-offs between global structure, local geometry, and spatial continuity. We next refer to the elements of Fig. 7. Spectral Laplacian clustering [5] (a) computes Laplace– Beltrami eigenvectors and embeds each vertex into spectral space, where K-means is applied to form patches. While effective at capturing global geometric structure, it often produces spatially disconnected regions. Heat Kernel Signature (HKS) clustering [11] (b) replaces spectral embeddings with multi-scale HKS descriptors, yielding more deformation-robust and geometry-sensitive patches, but tends to produce overly diffuse regions with weak boundary adherence. Face-based hierarchical splitting [38] (c) recursively subdivides mesh faces in a top-down manner, representing each patch via local geometric features such as area, angles, and normals. This yields highly structured and locally consistent regions but lacks global shape awareness. Geodesic Voronoi partitioning [49] (d) uses farthest point sampling to select patch centres and assigns vertices via geodesic distance, producing uniformly distributed connected regions, though without explicitly encoding curvature or semantic structure. All these tokenisation policies provide a sub-optimal basis for attention mechanisms and correspondence estimation (e.g., to various irregularities or coarseness). These observations motivate our final design with curvature guidance, which combines curvature and spectral cues with overlapping regions to balance local detail preservation and global geometric consistency (Fig. 7).

Table 7 evaluates the impact of different mesh tokenisation strategies while keeping the transformer architecture and learned mesh feature representation fixed. The goal of this experiment is to isolate the contribution of the proposed curvature-guided tokenisation from other framework components. The proposed curvature-guided overlapping tokenisation consistently outperforms all alternative tokenisation schemes across the CP2P24, PSMAL, and BeCoS benchmarks.

These results indicate that the choice of tokenisation plays a critical role in the design of transformerbased shape correspondence estimation. While spectral and HKS-based approaches leverage intrinsic shape information, they partition the surface according to a fixed spectral decomposition that may not align with semantically meaningful correspondence regions. Similarly, hierarchical subdivision produces spatially regular tokens but is largely agnostic to the underlying geometric complexity of the surface. In contrast, our curvature-guided strategy allocates tokens according to local geometric variation, enabling the transformer to focus representational capacity on regions that are most informative for establishing correspondences. Furthermore, the use of overlapping tokens provides additional contextual continuity across neighbouring regions, reducing sensitivity to token boundaries and facilitating the modelling of correspondences that span multiple local surface structures. This is particularly beneficial for partial matching scenarios, where informative geometric cues may be fragmented or only partially observed.

![](images/0efc341a01f2609a59f8a01ae5964db5b24e7b26bd23eae3171bce6828cd5009.jpg)  
(a) Curvature-aware mesh tokenisation. Tokens con- (b) Point-cloud tokenisation. Token distribution over centrate around high-curvature and articulated regions. irregular surface samples without mesh connectivity.  
Figure 8: Token allocation on meshes and point clouds. (a): The proposed curvature-aware tokenisation adaptively concentrates token centres on structurally informative regions, providing finer coverage of articulations and boundaries. (b): For point cloud inputs, the absence of mesh connectivity leads to less structured token placement and reduced concentration around high-curvature regions.

Table 7: Comparison of different tokenisation strategies and feature representations across benchmarks. We evaluate classical tokenisation strategies and our proposed curvature-guided overlapping tokenisation with different feature encodings on the CP2P24 [3], PSMAL [24], and BeCoS [23] datasets. All features for meshes and point clouds are learned.
<table><tr><td>Method</td><td>Tokenisation Type</td><td>Feature Type</td><td>CP2P24↑</td><td>PSMAL ↑</td><td>BeCoS ↑</td></tr><tr><td rowspan="5">Ours</td><td>Curvature-Guided (overlapping)</td><td>Mesh</td><td>85.56</td><td>85.21</td><td>65.25</td></tr><tr><td>Spectral Laplacian [5]</td><td>Mesh</td><td>76.23</td><td>75.81</td><td>50.15</td></tr><tr><td>HKS clustering [11]</td><td>Mesh</td><td>75.01</td><td>74.65</td><td>49.02</td></tr><tr><td>Face splitting [38]</td><td>Mesh</td><td>75.62</td><td>75.21</td><td>51.23</td></tr><tr><td>Curvature-Guided (overlapping)</td><td>Point cloud</td><td>81.05</td><td>80.12</td><td>59.55</td></tr></table>

To better understand the local effects of the tokenisation, Fig. 8a provides a zoomed-in visualisation of token centres on the hand region. It shows that token placement is not uniform, but instead concentrates on geometrically informative structures such as fingertips, knuckles, and boundary regions. As token density increases, these high-curvature areas are progressively refined, indicating that the method adaptively allocates representational capacity to regions of higher geometric complexity. This adaptive behaviour is particularly advantageous for correspondence estimation, as it preserves fine-grained structure in regions that are most sensitive to deformation and partial observations. Overall, these findings validate our central design choice: incorporating geometry-aware tokenisation into the transformer leads to more discriminative and transferable shape representations, yielding consistently strong performance across diverse shape matching benchmarks.

Mesh vs. Point Cloud Representations. We find that our method supports point clouds with only minimal adjustments in tokenisation. Fig. 8b illustrates the tokenisation of point-cloud inputs, where the lack of mesh connectivity leads to less structured coverage of high-curvature regions. To evaluate the importance of the underlying geometric representation, we compare our proposed mesh-based framework against a point-cloud variant that uses the same transformer architecture, training protocol, and correspondence objective. As shown in Table 7 (last row), replacing the mesh representation with point clouds leads to a consistent decrease in performance across all benchmarks (still, this variant ranks second), reducing accuracy from 85.56% to 81.05% on CP2P24, from 85.21% to 80.12% on PSMAL, and from 65.25% to 59.55% on BeCoS. This result suggests that the gains achieved by our method are not solely due to the transformer backbone, but also stem from preserving the intrinsic surface structure available in meshes. Meshes explicitly encode local connectivity and neighbourhood relationships, providing richer geometric context for token construction and feature learning. Consequently, the transformer can reason over representations that better capture both local surface characteristics and global shape structure, leading to more reliable correspondence estimation under deformation, partiality, and geometric variability. These findings further support our choice of meshes for generalisable shape correspondence estimation models.

## B.2 Effect of Cross-Shape Interaction

We next analyse the impact of the cross-attention mechanism, which enables explicit interaction between source and target shape tokens. To isolate its effect, we remove cross-attention while keeping the curvature-guided tokenisation, mesh representation, and transformer backbone unchanged.

As shown in Table 8, removing cross-attention leads to a consistent degradation in performance across all benchmarks, with accuracy dropping from 85.56% to 83.94% on CP2P24, from 85.21% to 83.01% on PSMAL, and from 65.25% to 63.96% on BeCoS. Self-attention alone—while effective for intra-shape reasoning—is insufficient to model all nuances of correspondence associations (with the highest accuracy) between two distinct shapes. In contrast, cross-attention facilitates direct

Table 8: Effect of cross-attention in the proposed architecture.
<table><tr><td>Variant</td><td>CP2P24</td><td>PSMAL</td><td>BeCoS</td></tr><tr><td>Full model</td><td>85.56</td><td>85.21</td><td>65.25</td></tr><tr><td>w/o cross-attention</td><td>83.94</td><td>83.01</td><td>63.96</td></tr></table>

feature exchange between the source and target representations, allowing the model to establish correspondence hypotheses through joint reasoning rather than independent encoding. This becomes particularly important in partial or ambiguous settings, where local evidence alone is insufficient, and global cross-shape context is required to disambiguate matches. Overall, this experiment highlights that explicit inter-shape interaction is a key component for achieving accurate and robust shape correspondence estimation.

Table 9: Effect of geodesic Gaussian bandwidth σ on partial-to-partial shape matching. We vary σ to control token overlap. Moderate overlap (σ = 0.15) yields the best performance.
<table><tr><td>σ</td><td>CP2P24↑</td><td>PSMAL ↑</td><td>BeCoS ↑</td></tr><tr><td>0.00 (no-overlapping)</td><td>84.05</td><td>83.92</td><td>64.04</td></tr><tr><td>0.05</td><td>83.91</td><td>83.72</td><td>64.01</td></tr><tr><td>0.10</td><td>84.88</td><td>84.63</td><td>64.72</td></tr><tr><td>0.15</td><td>85.56</td><td>85.21</td><td>65.25</td></tr><tr><td>0.20</td><td>85.31</td><td>85.99</td><td>65.10</td></tr><tr><td>0.25</td><td>84.40</td><td>84.10</td><td>64.38</td></tr><tr><td>0.30</td><td>80.12</td><td>82.95</td><td>61.45</td></tr></table>

## B.3 Token Overlap Degree

We study the effect of the geodesic Gaussian overlap bandwidth σ, which controls how strongly neighbouring token regions overlap in our soft tokenisation scheme. Given token centres, we compute geodesic distances from each mesh element to all centres and convert them into soft assignment weights using a Gaussian decay. σ directly controls the spatial extent of each token. Low values produce sharply localised regions that behave similarly to hard partitions, while larger values generate more diffuse tokens with larger overlap between neighbouring regions. In practice, we observe that moderate overlap yields the most stable representations. Very small σ values can lead to brittle assignments that are sensitive to mesh irregularities, whereas overly large values reduce spatial discriminability by over-smoothing geometric structure. This confirms that a balanced level of overlap is important for robust and geometry-consistent tokenisation, as shown in Table 9.

Table 10: Effect of number of mesh tokens g on partial-shape matching. Increasing g improves coverage of fine geometric details but shows diminishing returns at higher resolutions.
<table><tr><td>g</td><td>CP2P24 ↑</td><td>PSMAL ↑</td><td>BeCoS ↑</td></tr><tr><td>32</td><td>75.20</td><td>70.75</td><td>55.91</td></tr><tr><td>64</td><td>78.15</td><td>72.60</td><td>61.45</td></tr><tr><td>128</td><td>83.72</td><td>81.32</td><td>62.14</td></tr><tr><td>256</td><td>85.56</td><td>85.21</td><td>65.25</td></tr></table>

## B.4 Number of Mesh Tokens

We analyse the effect of the number of mesh tokens g, which controls the spatial resolution of our tokenisation. A smaller number of tokens leads to coarse surface coverage, while increasing g allows the model to capture finer geometric details.

We evaluate our method with g ∈ {32, 64, 128, 256} while keeping all other components fixed. As shown in Table 10, increasing g consistently improves performance by enabling better coverage of local geometric structures. However, gains begin to saturate at higher resolutions, indicating a trade-off between representational granularity and redundancy.

## B.5 Impact of Token Masking Ratio

![](images/25ef3bb94afc2c083e88db9706108e01e501ff53f4bca5e17ba6bb5c24c332f6.jpg)

Our empirical findings show that masked autoencoder pre-training (Sec. 4.2) provides a substantial performance boost and improves the robustness and generalisation of the learned representations. To study the effect of the masking ratio in our masked auto-encoding pre-training strategy, we vary the proportion of mesh tokens

Figure 9: We show ground-truth shapes (left), reconstructions produced by our model (centre), and masked inputs (right). Our model recovers missing geometric structures from heavily occluded inputs while preserving fine-grained details and global shape consistency.

masked during training, which forces the model to reconstruct missing geometric information from partial shapes. We evaluate masking ratios of 0.5, 0.7, and 0.8 while keeping all other components fixed. As shown in Table 11, moderate masking provides the best balance between learning robust global structure and preserving sufficient context for reconstruction; see Fig. 9. Excessive masking reduces available geometric context, making learning overly difficult.

Table 11: Effect of token masking ratio in masked auto-encoding pre-training. Moderate masking (50%) yields the best performance by balancing reconstruction difficulty and geometric context.
<table><tr><td>Mask Ratio</td><td>CP2P24↑</td><td>PSMAL ↑</td><td>BeCoS ↑</td></tr><tr><td>0.10</td><td>81.24</td><td>83.01</td><td>62.15</td></tr><tr><td>0.30</td><td>82.75</td><td>84.18</td><td>64.21</td></tr><tr><td>0.50</td><td>85.56</td><td>85.21</td><td>65.25</td></tr><tr><td>0.70</td><td>83.15</td><td>80.22</td><td>59.71</td></tr><tr><td>0.80</td><td>80.21</td><td>82.02</td><td>60.04</td></tr></table>

Noise-level: 0

0.5

![](images/2fa123bed8d274b8425b52d72d98c936f1abbf43e4658c20dbfbad3b7f808875.jpg)

Noise-level: 0

0.1

![](images/df50bfc76006e8665c4addbca437af3ca45aef4b19c3958a8a13b134f95b63bc.jpg)

Figure 10: Robustness to geometric noise on the BeCoS [23] dataset. Shown are results on two different partial-to-partial shape pairs under increasing vertex noise levels (0.1, 0.5, 3.0 and 5.0). Despite strong geometric perturbations (both rightmost columns), correspondences estimated by TokenMatch degrade gradually; the method preserves structurally consistent correspondences and remains robust even under severe noise.

![](images/38d5532221cac93c5f2267f69e7c6d69e6f8bf9922160601cf4ca7e90a263712.jpg)  
Figure 11: Additional qualitative results for challenging scenarios on the CP2P24 and PSMAL datasets under varying levels of partiality and deformation. Matches are colour-coded columnwise.

![](images/0aa1b6949cb9c664aee3c72e9e49dfcf5d5a93c6c83d5697ae1c15fe2c0a10c3.jpg)  
Figure 12: Additional qualitative results on BeCoS [23]: The model is exclusively trained on partial shapes (partial-to-partial setting denoted by $\mathrm { ^ { 6 6 } P 2 P ^ { 9 } ) }$ and generalises well to full shapes (full-to-full setting denoted by “F2F”). Matches are colour-coded columnwise. “GT” stands for ground truth.

![](images/a9ca74045040d14ccac028dce24c0f4cb0cd044510a259b7f54ad464b6a352e7.jpg)  
Figure 13: Generalisation from partial-to-partial training to partial-to-full evaluation on the BeCoS [23] dataset. Our model produces consistent and semantically meaningful correspondences across complete shapes, demonstrating high robustness and transferability of the learned geometryaware representations.

## B.6 Robustness to Noise

We evaluate the robustness of our method under synthetic geometric perturbations applied at test time. Importantly, the model is trained exclusively on clean meshes, and noise is introduced only during evaluation to simulate real-world scanning artefacts and reconstruction imperfections.

We generate noisy meshes by perturbing vertex positions with Gaussian noise. Specifically, for each mesh, we compute the bounding box diagonal to normalise scale, and then add zero-mean Gaussian noise with standard deviation $\sigma _ { n }$ proportional to the chosen noise level. This ensures that the perturbation is scale-invariant across different shapes. We consider multiple noise levels by varying $\sigma _ { n }$ relative to the mesh scale. Table 12 shows that performance decreases by only 0.52 points (the third column reports the performance drop $\Delta )$ under mild noise and 2.87 points under moderate noise, indicating robustness to geometric perturbations without noise-specific training

Table 12: Method robustness under Gaussian noise $\sigma _ { n }$
<table><tr><td> $\sigma _ { n }$ </td><td>Mean IoU (×100)↑</td><td>Drop (∆)</td></tr><tr><td>0.0 (No noise)</td><td>65.25</td><td></td></tr><tr><td>0.1</td><td>64.73</td><td>-0.52</td></tr><tr><td>0.5</td><td>62.38</td><td>-2.87</td></tr><tr><td>3.0</td><td>48.22</td><td>-17.03</td></tr><tr><td>5.0</td><td>42.45</td><td>-22.80</td></tr></table>

Fig. 10 presents qualitative results under the same increasing geometric noise levels (0.1, 0.5, 3, and 5) on the partial-to-partial BeCoS [23] dataset. While severe noise causes expected degradation, TokenMatch remains robust under mild and moderate perturbations, highlighting an additional dimension of generalisability beyond the training data manifold.

## C Further Qualitative Results

We provide additional qualitative results on the CP2P24 and PSMAL datasets to further illustrate the performance of our method under varying levels of partiality and non-isometric deformation in Figs. 11 and 12. The visualisations demonstrate that our TokenMatch produces accurate and geometrically consistent correspondences even in challenging partial-to-partial settings, where shape overlap is limited and local structure is highly incomplete.

We further present results on BeCoS, which contains more realistic and unconstrained scenarios with significant noise, missing regions, and strong deformations. Notably, as shown in Fig. 13, our model effectively handles full shapes and settings with varying degrees of partiality. The predicted correspondences remain stable across large geometric variations, preserving semantic alignment even under substantial shape deformation.

Overall, these qualitative results highlight the robustness of our approach and its ability to learn transferable geometric representations that generalise beyond the training distribution.