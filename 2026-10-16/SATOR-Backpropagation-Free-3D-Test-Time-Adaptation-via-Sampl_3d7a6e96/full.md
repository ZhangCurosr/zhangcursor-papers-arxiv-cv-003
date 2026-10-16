# SATOR: Backpropagation-Free 3D Test-Time Adaptation via Sample-Adaptive Token Routing

Inseok Kong<sup>1</sup> Geunyoung Jung<sup>2</sup> Jiyoung Jung<sup>2\*</sup>

<sup>1</sup>Department of Geo Informatics, University of Seoul, South Korea

<sup>2</sup>Department of Artificial Intelligence, University of Seoul, South Korea

{kong7963, gyjung975, jyjung}@uos.ac.kr

## Abstract

Test-time adaptation (TTA) mitigates the performance degradation ofsource-pretrained models on unlabeled test data under distribution shifts. Recent backpropagation-free token-purging methods address 3D point cloud distribution shifts by filtering unreliable tokens. However, their reliance on predefined purge-size candidates requires multiple forward passes per sample and increases inference cost. We propose SATOR (Sample-Adaptive TOken Routing), a source-free and backpropagation-free 3D TTA framework that eliminates this search by deriving both the partitioning threshold and routing decision directly from each sample. SATOR computes token abnormality scoresfrom complementary attention and geometric relation graphs. It then derives a sample-specific partition into retained and pruned tokens via Otsu-based separability, without evaluating candidate sizes. Keep-ratio-calibrated routing then selects the pruned prediction only when its entropy, penalized by the keep ratio, is lower than that of the original prediction. Experiments on ModelNet40-C, ScanObjectNN-C, and ShapeNet-C demonstrate that SATOR achieves state-of-the-art accuracy on three 3D corruption benchmarks.

## 1. Introduction

Deep neural networks have recently achieved remarkable success in 3D point cloud classification, enabling various applications such as autonomous driving, robotics, and AR/VR [17, 18, 23, 26, 34, 36]. However, models trained on clean source data often suffer severe performance degradation when the test distribution differs from the training distribution. In 3D point clouds, such distribution shifts can arise from multiple factors, including sensor variations, occlusions, density fluctuations, and background perturbations. Because these shifts are inevitable in practice, they present significant challenges in deploying reliable 3D perception models to real-world environments.

Test-time adaptation (TTA) addresses this challenge by adapting a source-pretrained model to a target domain during inference, relying solely on unlabeled target samples. TTA has demonstrated notable efficacy in mitigating distribution shifts within the 2D image domain through a wide range of approaches, including entropy minimization [9, 13, 14, 24], pseudo-labeling [8], graph-based refinement [4], augmentation-based aggregation [35], classifier and batch normalization adjustments [6, 11], and testtime training [21]. Nevertheless, transferring these strategies directly to 3D point clouds is non-trivial. Unlike 2D images, 3D point clouds are inherently sparse, unordered, and non-Euclidean. Furthermore, corruptions in 3D point clouds rarely manifest uniformly across the spatial domain. Real-world 3D corruptions often introduce spurious points, remove informative regions, or disrupt local geometric structures, rendering 3D point cloud adaptation fundamentally distinct from that of 2D images.

In 3D point cloud classification, existing TTA methods span test-time training [1, 12], multi-view aggregation [2, 7], and diffusion-based input restoration [5, 20], but each either ties adaptation to a modified source-training procedure, scales inference cost with the number of generated views, or relies on a source-trained generative model. Although BFTT3D [27] eliminates backpropagation, it still depends on class-level source prototypes at test time. Purge-Gate (PG) [31] advances this paradigm by introducing token purging for source-free, backpropagation-free TTA. However, PG measures token divergence against a single global reference and selects the purge-size by searching over manually specified candidates. Consequently, the purge decision is not derived directly from the current sample, and the adaptation process requires multiple forward passes per sample.

Motivated by these limitations, we introduce SATOR (Sample-Adaptive TOken Routing), a source-free and backpropagation-free 3D TTA framework that derives its token partition and routing decision from each test sample. SATOR computes token abnormality scores from attention and geometric relation graphs within the current sample rather than against a stored global reference. These scores are adaptively fused via Otsu-based separability [15] and partitioned using sorted-split Otsu thresholding, yielding a sample-specific token partition without evaluating predefined purge-size candidates. A keep-ratio-calibrated routing mechanism then selects between predictions from the original and pruned token sequences, accounting for the keep ratio in the routing criterion. Overall, SATOR obtains its partition from within-sample token relations rather than from a search over predefined candidates, while preserving the deployment advantages of source-free and backpropagation-free TTA.

The main contributions of this work are summarized as follows:

• We propose SATOR, a source-free and backpropagationfree 3D TTA framework that replaces iterative purge-size search with sample-specific token partitioning and adaptive routing.

• We design an attention–geometry abnormality scoring module that computes per-token scores from two complementary relation graphs within the current sample, rather than from divergence to a stored global reference, and fuses them adaptively via Otsu separability. The fused scores directly induce a sample-specific partition threshold, so that the pruning level is directly tied to the current sample rather than to manually specified purge-sizes.

• We introduce keep-ratio-calibrated routing that penalizes predictions obtained after more aggressive pruning, selecting the pruned prediction only when its keep-ratio-adjusted entropy is lower than that of the original prediction.

• We demonstrate through extensive experiments that SATOR achieves state-of-the-art mean accuracy on the ModelNet40-C, ScanObjectNN-C, and ShapeNet-C corruption benchmarks.

## 2. Related Work

## 2.1. Point Cloud Test-Time Adaptation

Test-time adaptation (TTA) has been widely studied in the 2D image domain through various strategies, including pseudo-labeling, entropy minimization, test-time training, and augmentation-aware prediction aggregation [8, 21, 24, 35]. However, extending these ideas to 3D point clouds is non-trivial because point clouds are sparse, unordered, and non-Euclidean. Furthermore, 3D corruptions often manifest irregularly across spatial domains, rendering uniform adaptation strategies less effective.

Existing 3D TTA methods can be categorized by their adaptation mechanisms. TTT-based methods such as MATE [12] and SMART-PC [1] incorporate auxiliary selfsupervised objectives but tie adaptation to a modified sourcetraining procedure, limiting applicability to standard pretrained models. SVWA [2] and MAMVI [7] leverage multiple views generated through sampling or masking variations, improving robustness through multi-view aggregation at the cost of higher inference overhead. CloudFixer [20] and 3DD-TTA [5] rely on iterative diffusion-based input reconstruction and source-trained generative models, making them unsuitable for strictly source-free settings. GSDTTA [29] shifts adaptation to the graph spectral domain for efficiency, but still depends on backpropagation. BFTT3D [27] eliminates backpropagation but requires class-level source prototypes at test time. Moving toward efficient source-free adaptation, LN-TTA [32] and PG [31] require neither source data nor gradient updates. LN-TTA recalibrates the affine parameters of layer normalization and leaves the token sequence unchanged. PG proposes two variants: PG-SF (source-free) and PG-SP (source-prototype-based); since our setting is strictly source-free, we compare against PG-SF throughout. However, both PG variants rely on a manual candidate search for purge-size selection, requiring multiple forward passes per sample.

## 2.2. Token Scoring and Purging

Token pruning and merging methods reduce computational cost by identifying and removing redundant tokens [3, 19, 22, 30]. EViT [10] separates tokens into attentive and inattentive groups using CLS attention scores, while ToMe [3] merges similar tokens via bipartite matching. These methods score tokens by their global importance under clean distributions, making them less suited to detecting corruption-induced anomalies that manifest as disruptions in local token interactions. In 2D TTA, TCA [28] shows that token condensation can improve robustness without backpropagation. In 3D TTA, PG [31] applies token purging to source-free adaptation, but measures token divergence against a single global reference and selects the purge-size by searching over manually specified candidates, requiring multiple forward passes per sample. SATOR instead scores tokens through attention and geometric relation graphs within each sample, reflecting how each token relates to its local neighbors rather than a stored global reference, enabling search-free partitioning without source statistics.

## 2.3. Sample-Adaptive Inference

Instance-aware inference methods assign each input to a different computational path based on input-specific signals [25]. In vision transformers, A-ViT [33] extends this to the token level by learning per-token halting probabilities, enabling sample-specific computation graphs. However, such learned routing policies assume clean distributions and cannot be applied at test time under unknown corruptions. SATOR replaces learned routing with a parameter-free threshold derived from each sample’s token score distribution via sorted-split Otsu thresholding [15], which requires no predefined candidates. The routing decision is further calibrated by the keep ratio, which accounts for the differing token counts of the two branches.

![](images/a094075c3653ee5a9e7b1ca5ef0919437ff86740619c167a5da4db5a9a06e8b2.jpg)  
Figure 1. Overall pipeline of SATOR. Token abnormality scores are computed from attention and geometric relation graphs and adaptively fused based on their Otsu separability. A sample-specific token partition is then derived directly from the fused score distribution via sorted-split Otsu thresholding, eliminating iterative purge-size search. A keep-ratio-calibrated routing mechanism selects between the original and pruned token sequence predictions, accounting for the keep ratio in the routing criterion.

## 3. Method

## 3.1. Preliminaries

Point cloud tokenization. A point cloud is an unordered set of 3D points $X = \{ p _ { 1 } , p _ { 2 } , \dotsc , p _ { N _ { p } } \} \subset \mathbb { R } ^ { 3 }$ . Following common point cloud transformer pipelines [16], we tokenize X into M local patches using furthest point sampling (FPS) and k-nearest neighbors (KNN). For the t-th token, the patch center is selected as

$$
c _ { t } = \arg \operatorname* { m a x } _ { p \in X } \operatorname* { m i n } _ { c ^ { \prime } \in \{ c _ { 1 } , \ldots , c _ { t - 1 } \} } \| p - c ^ { \prime } \| _ { 2 } ,\tag{1}
$$

and its neighborhood is defined by

$$
\mathcal { N } _ { t } = \{ p \in X \mid p \in \mathrm { K N N } ( c _ { t } ) \} .\tag{2}
$$

The resulting token is written as $\boldsymbol { x } _ { t } = \left( c _ { t } , \mathcal { N } _ { t } \right)$ , yielding a token sequence $T = \{ x _ { t } \} _ { t = 1 } ^ { M }$ . This representation preserves local geometric structure while enabling transformer-style interactions across patches.

Source-free and backpropagation-free TTA setting. We consider a labeled source domain $\mathcal { D } _ { S } ~ = ~ \{ ( X _ { i } ^ { S } , y _ { i } ) \} _ { i = 1 } ^ { N _ { S } }$ and an unlabeled target domain $\mathcal { D } _ { T } ~ = ~ \{ X _ { j } ^ { T } \} _ { j = 1 } ^ { N _ { T } }$ , with $P _ { S } ( X ) \ne P _ { T } ( X )$ . The goal is to improve prediction reliability on target samples without access to source data or backpropagation-based updates at test time.

## 3.2. Token Relation Disruption under Distribution Shift

Distribution shifts in 3D point clouds—such as noise, density changes, and occlusions—alter local patch-token features at the earliest transformer block. Prior token-purging work has analyzed how noisy tokens degrade the attention mechanism under distribution shift, using normalization regularity and high-dimensional random-direction arguments [31]. Our analysis is complementary but takes a different perspective: rather than relying on a noise-dominant uniform-attention view or defining token divergence relative to a global reference, we focus on how distribution-induced patch-token feature deviations perturb early token relations. This motivates relation-based abnormality scoring from attention and geometric graphs, which is the basis of Sections 3.3–3.4.

Concretely, let $z _ { i } ^ { S } \in \mathbb { R } ^ { d }$ denote an ideal clean-domain counterpart of the Block-0 patch-token feature, and write the observed target-domain feature as $z _ { i } ^ { T } = z _ { i } ^ { S } + \delta _ { i }$ , where $\delta _ { i }$ denotes the feature-space deviation induced by the distribution shift. This notation does not assume a particular corruption model but represents the residual change observed at the patch-token feature level.

In a transformer attention block, pairwise token relations are determined by query-key similarities. For token pair $( i , j )$ , the attention logit is

$$
s _ { i j } = \frac { \left( \mathrm { L N } ( z _ { i } + { \bf e } _ { i } ) W _ { Q } \right) ^ { \top } \left( \mathrm { L N } ( z _ { j } + { \bf e } _ { j } ) W _ { K } \right) } { \sqrt { d _ { h } } } ,\tag{3}
$$

where $\mathbf { e } _ { i }$ and $\mathbf { e } _ { j }$ are positional embeddings and $d _ { h }$ is the head dimension. Even with frozen model parameters, patchtoken feature perturbations $\delta _ { i }$ can alter these attention logits and thus change early token relations. The following simple bound, included only to clarify the scoring motivation, formalizes this relation-perturbation view.

Proposition 1 (Attention-logit sensitivity to patch-token feature perturbations). Suppose that (i) layer normalization is Lipschitz with constant $L _ { \mathrm { L N } }$ on a compact domain where the token-wise standard deviation is bounded away from zero, (ii) the query and key projection matrices satisfy $\| W _ { Q } \| _ { 2 } \le B _ { Q }$ and $\| W _ { K } \| _ { 2 } \le B _ { K }$ , and (iii) the backbone is frozen at test time. Then for any token pair (i, j),

$$
\begin{array} { r l r } {  { | s _ { i j } ^ { T } - s _ { i j } ^ { S } | } \le \frac { L _ { \mathrm { L N } } } { \sqrt { d _ { h } } } \Big ( B _ { K } \| q _ { i } ^ { S } \| _ { 2 } \| \delta _ { j } \| _ { 2 } + B _ { Q } \| k _ { j } ^ { S } \| _ { 2 } \| \delta _ { i } \| _ { 2 } }  \\ & { } & { + L _ { \mathrm { L N } } B _ { Q } B _ { K } \| \delta _ { i } \| _ { 2 } \| \delta _ { j } \| _ { 2 } \Big ) , ~ } \end{array}\tag{4}
$$

where $q _ { i } ^ { S }$ and $k _ { j } ^ { S }$ denote the clean-domain query and key vectors:

$$
q _ { i } ^ { S } = \mathrm { L N } ( z _ { i } ^ { S } + { \bf e } _ { i } ) W _ { Q } , \quad k _ { j } ^ { S } = \mathrm { L N } ( z _ { j } ^ { S } + { \bf e } _ { j } ) W _ { K } .
$$

The proof is provided in Appendix A.2. Importantly, this bound is a local sensitivity result: it does not claim that attention collapses to a uniform distribution, nor does it require isotropic random-vector assumptions about the perturbations. It only shows that patch-token feature deviations can change pairwise attention relations even without parameter updates. SATOR therefore uses these perturbed early relations as signals for graph-based abnormality scoring, rather than estimating divergence from an explicit prototype or reference as in prior purging methods [31].

Moreover, through repeated attention mixing in deeper transformer blocks, initially localized perturbations can become diffuse across tokens, making token-level abnormality harder to separate at later layers. This motivates extracting token abnormality scores at Block-0, before repeated mixing obscures local token-level deviations. The block-depth ablation in Table 6 supports this choice.

## 3.3. Attention–Geometry Token Abnormality Scoring

The previous section motivates viewing early patch-token relations as sensitive to distribution-induced feature deviations. A single relation graph may not capture all such perturbations because attention relations reflect learned token interactions whereas geometric relations encode spatial neighborhood structure. We therefore compute token abnormality scores at Block-0 from two distinct relation graphs. Attention relation graph. Let $A ^ { ( h ) } \in \mathbb { R } ^ { M \times M }$ denote the Block-0 patch-token attention probabilities of head $h ,$ where $h = 1 , \ldots , N _ { h }$ . We aggregate the attention probabilities across heads and symmetrize them to obtain an undirected attention relation graph

$$
W ^ { a t t n } = \frac { 1 } { 2 } \left( \frac { 1 } { N _ { h } } \sum _ { h = 1 } ^ { N _ { h } } A ^ { ( h ) } + \left( \frac { 1 } { N _ { h } } \sum _ { h = 1 } ^ { N _ { h } } A ^ { ( h ) } \right) ^ { \top } \right)\tag{5}
$$

Geometric relation graph. The geometric relation graph complements the attention relation graph by measuring consistency among spatially neighboring patches. Let $d _ { i j } =$ $\| c _ { i } - c _ { j } \| _ { 2 } ^ { 2 }$ denote the squared Euclidean distance between patch centers. We connect each token to its $K _ { g }$ nearest neighboring patch centers and define the directed geometric affinity

$$
\tilde { W } _ { i j } ^ { g e o } = \left\{ \begin{array} { l l } { \exp \left( - d _ { i j } / \sigma _ { g } \right) , } & { j \in \mathcal { N } _ { K _ { g } } ^ { g e o } ( i ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{6}
$$

where $\mathcal { N } _ { K _ { q } } ^ { g e o } ( i )$ is the set of $K _ { g }$ nearest patch centers to $c _ { i } ,$ , and $\sigma _ { g }$ is the sample-wise mean $K _ { g } – \mathbf { N N }$ squared distance used as the bandwidth of the geometric affinity kernel. Throughout this section, $\epsilon = 1 0 ^ { - 6 }$ denotes a small constant for numerical stability. We then row-normalize and symmetrize the graph

$$
\bar { W } _ { i j } ^ { g e o } = \frac { \tilde { W } _ { i j } ^ { g e o } } { \sum _ { \ell = 1 } ^ { M } \tilde { W } _ { i \ell } ^ { g e o } + \epsilon } , \qquad W ^ { g e o } = \frac { 1 } { 2 } \left( \bar { W } ^ { g e o } + ( \bar { W } ^ { g e o } ) ^ { \top } \right) .\tag{7}
$$

Local graph energy as token abnormality. Given a relation graph W, we define the local graph energy of token i as

$$
\mathcal { E } _ { i } ( W ) = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { M } W _ { i j } \| z _ { i } - z _ { j } \| _ { 2 } ^ { 2 } ,\tag{8}
$$

where $z _ { i } , z _ { j } ~ \in ~ \mathbb { R } ^ { d }$ denote the Block-0 patch-token features. This energy increases when token i differs from strongly connected neighboring tokens, capturing both feature inconsistency and disruptions in the underlying relation graph. Applying this energy to the two relation graphs gives the attention-based and geometry-based token abnormality scores $\mathcal { E } _ { i } ^ { a t t n } = \mathcal { E } _ { i } ( W ^ { a t t n } )$ and $\mathscr { E } _ { i } ^ { g e o } = \mathscr { E } _ { i } ( W ^ { g e o } )$ , respectively. A formal decomposition of this graph energy under perturbation is provided in Appendix A.3.

Score normalization. Because the attention and geometric relation graphs are constructed from different relation spaces, their raw abnormality scores are not directly comparable. We therefore normalize each score map independently within the current sample

$$
\hat { \mathcal { E } } _ { i } ^ { a t t n } = \mathrm { N o r m a l i z e } ( \mathcal { E } _ { i } ^ { a t t n } ) , \quad \hat { \mathcal { E } } _ { i } ^ { g e o } = \mathrm { N o r m a l i z e } ( \mathcal { E } _ { i } ^ { g e o } ) ,\tag{9}
$$

where Normalize(·) denotes z-score standardization over the M tokens. The normalized scores are then used for sample-specific fusion and partitioning in the next stage.

## 3.4. Sample-Specific Token Partitioning

Given the attention- and geometry-based score maps, the next step is to determine how they should be fused and how the current sample should be partitioned into retained and pruned tokens. Search-based purging methods select from manually specified purge-sizes, leaving the pruning level only indirectly tied to the current sample. In contrast, SATOR derives both the fusion weight and the partitioning threshold directly from the sample’s score distribution.

Otsu-based separability for score-map reliability. A useful score map should not only identify high-abnormality tokens but also separate them from low-abnormality tokens. We measure this separation using Otsu-based separability [15]. For a score map q, we define

$$
\operatorname { S e p } ( q ) = { \frac { \operatorname* { m a x } _ { \tau } \sigma _ { B } ^ { 2 } ( \tau ) } { \sigma _ { \mathrm { t o t a l } } ^ { 2 } ( q ) + \epsilon } } ,\tag{10}
$$

where $\sigma _ { B } ^ { 2 } ( \tau )$ is the between-group variance induced by threshold $\tau ,$ and $\sigma _ { \mathrm { t o t a l } } ^ { 2 } ( q )$ is the total variance of the score map.

Adaptive fusion of attention and geometry scores. We compute the separability of the two normalized score maps as

$$
\eta _ { a t t n } = \mathrm { S e p } ( \hat { \mathcal { E } } ^ { a t t n } ) , \qquad \eta _ { g e o } = \mathrm { S e p } ( \hat { \mathcal { E } } ^ { g e o } ) ,\tag{11}
$$

and use them to weight each relation graph by how cleanly its own score map splits, so that the fusion does not depend on a globally fixed weight. The sample-adaptive fusion weight is

$$
\alpha ( X ) = \frac { \eta _ { a t t n } } { \eta _ { a t t n } + \eta _ { g e o } + \epsilon } ,\tag{12}
$$

and the fused token abnormality score is

$$
\begin{array} { r } { S _ { i } = \alpha ( X ) \hat { \mathcal { E } } _ { i } ^ { a t t n } + \big ( 1 - \alpha ( X ) \big ) \hat { \mathcal { E } } _ { i } ^ { g e o } . } \end{array}\tag{13}
$$

Thus, the fusion is determined by how separable each score map is for the current sample rather than by a fixed weight. Sorted-split Otsu token partition. After fusion, SATOR applies the same separability criterion to the fused score map ${ \cal { S } } = \{ { \cal { S } } _ { i } \} _ { i = 1 } ^ { M }$ to obtain a sample-specific threshold $\tau ^ { \star }$ . Rather than approximating the score distribution with a histogram, SATOR sorts the M token scores and directly evaluates all M − 1 candidate split points. This sorted-split approach avoids additional binning hyperparameters and directly matches the discrete nature of token partitioning.

Retained and pruned token sequences. Tokens with $S _ { i } \leq \tau ^ { \star }$ form the retained set $\kappa ,$ while tokens with $S _ { i } > \tau ^ { \star }$ form the pruned set. This directly determines both which tokens are removed and how many tokens are removed for the current sample, eliminating the need for iterative purge-size search. Unlike PG, which evaluates predefined purge-size candidates through multiple forward passes, SATOR performs search-free sample-specific partitioning by deriving the partition from the sample’s own score distribution. The retained token sequence is then forwarded through the remaining backbone to produce the pruned-branch prediction used by the routing stage.

## 3.5. Keep-Ratio-Calibrated Routing

After sample-specific token partitioning produces a pruned token sequence, SATOR decides whether to use its prediction or retain the prediction from the original token sequence.

Entropy and the fraction of retained tokens. Token removal can suppress corrupted evidence, but it can also discard useful context, so the pruned prediction is not always preferable. SATOR compares the two branches by their prediction entropy, but entropy alone can favor a prediction whose logits are more sharply peaked, regardless of how many tokens produced it. The following lemma states this relation; it is used here as motivation and does not identify token pruning with temperature scaling. The empirical relation between pruning and entropy is examined in Appendix C.6.

Lemma 1 (Entropy decreases under logit sharpening). Let $z ~ \in ~ \mathbb { R } ^ { C }$ be a logit vector with at least two distinct entries, and define the temperature-scaled prediction $p _ { \lambda , c } =$ $\exp ( \lambda z _ { c } ) / { \sum _ { c ^ { \prime } } \exp ( \lambda z _ { c ^ { \prime } } ) }$ . Then

$$
{ \frac { d } { d \lambda } } H ( p _ { \lambda } ) = - \lambda \operatorname { V a r } _ { p _ { \lambda } } ( z ) \leq 0 \ f o r \ a l l \lambda > 0 .\tag{14}
$$

The proof is provided in Appendix A.4. Entropy alone is therefore not a sufficient criterion when the two branches are computed from different numbers of tokens: the final routing decision must account for both prediction confidence and the fraction of retained tokens.

Keep-ratio calibration. We define the keep ratio $\rho ~ =$ $| \mathcal { K } | / M$ and the softmax entropy $\begin{array} { r } { H ( z ) = - \sum _ { c = 1 } ^ { C } p _ { c } \log p _ { c } , } \end{array}$ where $p _ { c } = \operatorname { s o f t m a x } ( z ) _ { c }$ and C is the number of classes. Since $\rho \in ( 0 , 1 ]$ , dividing the pruned entropy by $\rho$ penalizes predictions obtained after more aggressive pruning. This prevents low entropy from being favored solely because the prediction was made from a severely reduced token set.

Final routing rule. Let $z _ { \mathrm { o r i g } }$ and $z _ { \mathrm { p r u n e } }$ denote the logit vectors from the original and pruned token sequences, respectively. SATOR selects the pruned prediction only when its keep-ratio-calibrated entropy is lower than the entropy of

Table 1. Top-1 classification accuracy (%) across all 15 corruption types in ModelNet40-C. The highest mean accuracy is highlighted in bold and the second highest is underlined.
<table><tr><td>Method</td><td>Venue</td><td>UNI</td><td>GAUSS</td><td>BACKG</td><td>IMPUL</td><td>UPSAM</td><td>RBF</td><td>RBF-INV</td><td>DEN-DEC</td><td>DENS-INC</td><td>SHEAR</td><td>ROT</td><td>CUT</td><td>DISTORT</td><td>OCCL</td><td>LIDAR</td><td>MEAN</td><td>DIFF</td></tr><tr><td>SOURCE</td><td></td><td>66.53</td><td>59.12</td><td>7.21</td><td>31.69</td><td>74.72</td><td>67.71</td><td>69.77</td><td>62.28</td><td>75.08</td><td>74.35</td><td>38.13</td><td>58.63</td><td>70.02</td><td>38.53</td><td>18.64</td><td>54.16</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Test-Time Training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MATE [12]</td><td>ICCV&#x27;23</td><td>69.80</td><td>61.80</td><td>18.90</td><td>63.90</td><td>72.50</td><td>64.00</td><td>74.00</td><td></td><td>80.80</td><td>71.00</td><td>36.70</td><td>69.20</td><td>66.30</td><td>38.40</td><td>29.90</td><td>58.88</td><td>4.72</td></tr><tr><td>SMART-PC [1]</td><td>ICML&#x27;25</td><td>82.40</td><td>80.10</td><td>12.00</td><td>67.10</td><td>84.50 76.00</td><td>66.00 78.60</td><td>67.30</td><td></td><td>72.90</td><td>73.30</td><td>43.90</td><td>72.60</td><td>73.50</td><td>37.40</td><td>24.80</td><td>63.09</td><td>8.93</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Backpropagation Method</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PL [8]</td><td>ICML-W&#x27;13</td><td>73.74</td><td>63.70</td><td>10.53</td><td>55.23</td><td>62.56</td><td>67.18 68.35</td><td></td><td>74.15</td><td>78.28</td><td>69.69</td><td>52.35</td><td>72.45</td><td>65.11</td><td>43.40</td><td>37.12</td><td>59.59</td><td>5.43</td></tr><tr><td>TTT-ROT [21]</td><td>ICML&#x27;20</td><td>61.30</td><td>58.30</td><td>34.50</td><td>48.90</td><td>66.70</td><td>63.60</td><td>63.90</td><td>59.80</td><td>68.60</td><td>55.20</td><td>27.30</td><td>54.60</td><td>64.00</td><td>40.00</td><td>29.10</td><td>53.05</td><td>-1.11</td></tr><tr><td>SHOT [9]</td><td>ICML&#x27;20</td><td>29.60</td><td>28.20</td><td>9.80</td><td>25.40</td><td>32.70</td><td>30.30</td><td>30.10 65.03</td><td>30.90</td><td>31.20</td><td>32.10</td><td>22.80</td><td>27.30</td><td>29.40</td><td>20.80</td><td>18.60</td><td>26.61</td><td>-27.55</td></tr><tr><td>T3A [6]</td><td>NeurIPS&#x27;21</td><td>64.06</td><td>61.55</td><td>20.14</td><td>50.12</td><td>64.83</td><td>63.74</td><td></td><td>66.57</td><td>72.29</td><td>68.03</td><td>50.49</td><td>64.67</td><td>62.80</td><td>43.64</td><td>40.28</td><td>57.22</td><td>3.06</td></tr><tr><td>TENT [24]</td><td>ICLR&#x27;21</td><td>73.78</td><td>70.14</td><td>18.48</td><td>59.68</td><td>76.14</td><td>70.58</td><td>73.22</td><td></td><td>79.25</td><td>73.95</td><td>54.25</td><td>70.99</td><td>70.50</td><td>46.07</td><td>43.80</td><td>63.53</td><td>9.37</td></tr><tr><td>SAR [14]</td><td>ICLR&#x27;23</td><td>71.64</td><td>66.45</td><td>7.86</td><td>54.70</td><td>64.71</td><td>67.02</td><td>75.97</td><td></td><td>79.05</td><td>70.79</td><td>51.09</td><td>71.72</td><td>66.41</td><td>43.03</td><td>37.24</td><td>59.64</td><td>5.48</td></tr><tr><td>MEMO [35]</td><td>NeurIPS&#x27;22</td><td>67.87</td><td>57.21</td><td>4.05</td><td>28.24</td><td>65.64</td><td>64.75 65.68</td><td>76.99</td><td></td><td>83.51</td><td>69.77</td><td>38.13</td><td>74.07</td><td>66.29</td><td>35.78</td><td>29.42</td><td>55.16</td><td>1.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Backpropagation-Free Method</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DUA [11]</td><td>CVPR&#x27;22</td><td>65.00</td><td>58.50</td><td>14.70</td><td>48.50</td><td>68.80</td><td>62.80</td><td>63.20</td><td>62.10</td><td>66.20</td><td>68.80</td><td>46.20</td><td>53.80</td><td>64.70</td><td>41.20</td><td>36.50</td><td>54.73</td><td>0.57</td></tr><tr><td>LAME [4]</td><td>CVPR&#x27;22</td><td>67.63</td><td>58.55</td><td>4.09</td><td>31.81</td><td>63.86</td><td>64.83</td><td>65.76</td><td>76.34</td><td>82.09</td><td>68.19</td><td>37.40</td><td>71.35</td><td>65.28</td><td>36.18</td><td>29.13</td><td>54.83</td><td>0.67</td></tr><tr><td>BFTT3D [27] PG [31]</td><td>CVPR&#x27;24</td><td>66.53 78.12</td><td>59.20</td><td>7.01</td><td>31.85</td><td>65.60</td><td>67.79 73.58</td><td>69.81 75.81</td><td>77.96 79.09</td><td>85.25</td><td>74.35</td><td>38.17</td><td>74.80</td><td>70.02</td><td>38.61</td><td>30.51</td><td>57.16 68.68</td><td>3.00 14.52</td></tr><tr><td>LN-TTA [32]</td><td>ICCV&#x27;25</td><td></td><td>74.47</td><td>40.88</td><td>66.69</td><td>81.73</td><td></td><td></td><td></td><td>82.37</td><td>77.23</td><td>58.19</td><td>75.49</td><td>73.58</td><td>47.85</td><td>45.14</td><td></td><td></td></tr><tr><td>SATOR</td><td>WACV&#x27;26 Ours</td><td>80.92 79.79</td><td>76.14 75.97</td><td>28.57 34.78</td><td>68.92 68.02</td><td>81.36 81.78</td><td>75.28 76.18</td><td>75.77 76.58</td><td>80.71 82.27</td><td>83.67 84.17</td><td>79.13 78.33</td><td>60.25 58.08</td><td>77.11 80.28</td><td>75.41 75.49</td><td>50.89 49.92</td><td>51.50 52.27</td><td>69.71 70.26</td><td>15.55 16.10</td></tr></table>

Table 2. Top-1 classification accuracy (%) across all 15 corruption types in ShapeNet-C. The highest mean accuracy is highlighted in bold and the second highest is underlined.
<table><tr><td>Method</td><td>Venue</td><td>UNI</td><td>GAUSS</td><td>BACKG</td><td>IMPUL</td><td>UPSAM</td><td>RBF</td><td>RBF-INV</td><td>DEN-DEC</td><td>DENS-INC</td><td>SHEAR</td><td>ROT</td><td>CUT</td><td>DISTORT</td><td>OCCL</td><td>LIDAR</td><td>MEAN</td><td>DIFF</td></tr><tr><td>SOURCE</td><td></td><td>77.34</td><td>71.78</td><td>8.64</td><td>54.51</td><td>77.90</td><td>75.50</td><td>76.06</td><td>85.25</td><td>76.42</td><td>80.48</td><td>57.08</td><td>85.12</td><td>76.05</td><td>10.97</td><td>7.13</td><td>61.35</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Test-Time Training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MATE [12]</td><td>ICCV&#x27;23</td><td>77.80</td><td>74.70</td><td>4.30</td><td>66.20</td><td>78.60</td><td>76.30</td><td></td><td>86.10</td><td>86.60</td><td>79.20</td><td>56.10</td><td>84.10</td><td>76.10</td><td>12.30</td><td>13.10</td><td>63.12</td><td>1.77</td></tr><tr><td>SMART-PC [1]</td><td>ICML&#x27;25</td><td>80.80</td><td>78.90</td><td>8.90</td><td>60.40</td><td>81.80</td><td>81.10</td><td>75.30 81.70</td><td>84.80</td><td>78.40</td><td>80.80</td><td>63.70</td><td>84.90</td><td>79.80</td><td>11.50</td><td>8.80</td><td>64.42</td><td>3.07</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Backpropagation Method</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PL [8]</td><td>ICML-W&#x27;13</td><td>69.70</td><td>66.35</td><td>8.24</td><td>59.37</td><td>62.91</td><td>71.11</td><td>72.40</td><td>80.50</td><td>79.18</td><td>74.18</td><td>56.92</td><td>80.43</td><td>71.05</td><td>9.07</td><td>9.98</td><td>58.09</td><td>-3.26</td></tr><tr><td>TTT-ROT [21]</td><td>ICML&#x27;20</td><td>74.60</td><td>72.40</td><td>23.10</td><td>59.90</td><td>74.90</td><td>73.80</td><td>75.00</td><td>81.40</td><td>82.00</td><td>69.20</td><td>49.10</td><td>79.90</td><td>72.70</td><td>14.00</td><td>12.00</td><td>60.93</td><td>-0.42</td></tr><tr><td>SHOT [9]</td><td>ICML&#x27;20</td><td>44.80</td><td>42.50</td><td>12.10</td><td>37.60</td><td>45.00</td><td>43.70</td><td>44.20</td><td>48.40</td><td>49.40</td><td>45.00</td><td>32.60</td><td>46.30</td><td>39.10</td><td>6.20</td><td>5.90</td><td>36.19</td><td>-25.16</td></tr><tr><td>T3A [6] TENT [24]</td><td>NeurIPS&#x27;21</td><td>66.67</td><td>64.25</td><td>13.02</td><td>59.77</td><td>67.02</td><td>65.14</td><td>64.89</td><td>70.40</td><td>66.67</td><td>65.98</td><td>54.96</td><td>69.43</td><td>63.79</td><td>10.07</td><td>10.59</td><td>54.18</td><td>-7.17</td></tr><tr><td>SAR [14]</td><td>ICLR&#x27;21 ICLR&#x27;23</td><td>68.50</td><td>65.29</td><td>7.55</td><td>55.89 64.06</td><td>67.80</td><td>65.77</td><td>66.30</td><td>76.83</td><td>68.00</td><td>69.36</td><td>51.20</td><td>76.35</td><td>64.77</td><td>10.66</td><td>10.36</td><td>54.98</td><td>-6.37</td></tr><tr><td>MEMO [35]</td><td>NeurIPS&#x27;22</td><td>72.57 74.89</td><td>67.54 69.54</td><td>9.43 4.25</td><td>57.69</td><td>67.02 65.80</td><td>73.12 74.52</td><td>73.88</td><td>82.44</td><td>81.55</td><td>77.46</td><td>59.59</td><td>81.51</td><td>72.80</td><td>9.54</td><td>10.05</td><td>60.17</td><td>-1.18</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>75.72</td><td>85.32</td><td>84.49</td><td>79.60</td><td>55.07</td><td>84.35</td><td>76.29</td><td>9.79</td><td>7.48</td><td>60.32</td><td>-1.03</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Backpropagation-Free Method</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DUA [11]</td><td>CVPR&#x27;22</td><td>76.10</td><td>70.10</td><td>14.30</td><td>60.90</td><td>76.20</td><td>71.60</td><td>72.90</td><td>80.00</td><td>83.80</td><td>77.10</td><td>57.50</td><td>75.00</td><td>72.10</td><td>11.90</td><td>12.10</td><td>60.77</td><td>-0.58</td></tr><tr><td>LAME [4]</td><td>CVPR&#x27;22</td><td>72.87</td><td>66.92</td><td>4.89</td><td>58.42</td><td>64.50</td><td>73.52</td><td>74.63</td><td>84.40</td><td>83.59</td><td>78.42</td><td>53.45</td><td>83.34</td><td>75.26</td><td>10.36</td><td>8.47</td><td>59.54</td><td>-1.81</td></tr><tr><td>BFTT3D [27]</td><td>CVPR&#x27;24</td><td>72.58</td><td>65.91</td><td>10.57</td><td>60.86</td><td>67.78</td><td>73.45</td><td>74.38</td><td>84.55</td><td>83.52</td><td>78.58</td><td>57.96</td><td>83.71</td><td>75.17</td><td>11.95</td><td>10.00</td><td>60.73</td><td>-0.62</td></tr><tr><td>PG [31]</td><td>ICCV&#x27;25</td><td>79.21</td><td>75.43</td><td>22.46</td><td>68.38</td><td>78.95</td><td>76.78</td><td>77.12</td><td>83.52</td><td>80.73</td><td>79.24</td><td>64.91</td><td>83.55</td><td>76.08</td><td>10.15</td><td>9.55</td><td>64.40</td><td>3.06</td></tr><tr><td>LN-TTA [32]</td><td>WACV&#x27;26</td><td>79.37</td><td>76.26</td><td>14.44</td><td>68.66</td><td>79.56</td><td>78.01</td><td>78.36</td><td>84.27</td><td>82.59</td><td>79.66</td><td>65.24</td><td>83.63</td><td>76.95</td><td>8.99</td><td>9.32</td><td>64.35</td><td>3.00</td></tr><tr><td>SATOR</td><td>Ours</td><td>79.97</td><td>76.42</td><td>17.24</td><td>70.33</td><td>79.61</td><td>77.72</td><td>77.73</td><td>84.99</td><td>82.46</td><td>80.90</td><td>65.59</td><td>84.24</td><td>77.43</td><td>9.40</td><td>9.80</td><td>64.92</td><td>3.57</td></tr></table>

the original prediction

$$
\frac { H ( z _ { \mathrm { p r u n e } } ) } { \rho } < H ( z _ { \mathrm { o r i g } } ) .\tag{15}
$$

Otherwise, SATOR retains the prediction from the original token sequence. This makes the final decision adaptive to the current pruning outcome rather than treating token removal as a fixed replacement rule.

## 4. Experiments

Detailed dataset descriptions, baseline configurations, implementation settings, additional experiments, and supplementary ablations are provided in the Supplementary Material.

## 4.1. Main Results

ModelNet40-C. As shown in Table 1, SATOR achieves the highest mean accuracy of 70.26%, the largest gain over the source-only model (+16.10) among all backpropagation-free methods, ahead of LN-TTA (69.71%) and PG (68.68%). It exceeds PG on 13 of the 15 corruptions, with the largest margins on LiDAR distortion (+7.13), cutout (+4.79), and occlusion (+2.07). These corruptions involve spatial distortion or token removal, where within-sample scoring is most informative.

ShapeNet-C. ShapeNet-C is the benchmark on which adaptation is hardest to obtain: 10 of the 14 compared methods fall below the source-only model. As shown in Table 2, SATOR obtains the best mean accuracy of 64.92% and the largest improvement over the source-only model among all compared methods (+3.57), ahead of PG (+3.06) and LN-

Table 3. Top-1 classification accuracy (%) across all 15 corruption types in ScanObjectNN-C. The highest mean accuracy is highlighted in bold and the second highest is underlined.
<table><tr><td>Method</td><td>Venue</td><td>UNI</td><td>GAUSS</td><td>BACKG</td><td>IMPUL</td><td>UPSAM</td><td>RBF</td><td>RBF-INV</td><td>DEN-DEC</td><td>DENS-INC</td><td>SHEAR</td><td>ROT</td><td>CUT</td><td>DISTORT</td><td>OCCL</td><td>LIDAR</td><td>MEAN</td><td>DIFF</td></tr><tr><td>SOURCE</td><td></td><td>21.69</td><td>18.76</td><td>16.70</td><td>18.42</td><td>22.20</td><td>45.96</td><td>46.99</td><td>71.95</td><td>69.36</td><td>48.88</td><td>35.63</td><td>72.98</td><td>49.40</td><td>6.71</td><td>8.78</td><td></td><td>36.96 0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Test-Time Training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MATE [12]</td><td>ICCV&#x27;23</td><td>27.50</td><td>29.40</td><td>14.30</td><td>22.20</td><td></td><td>42.00</td><td>73.70</td><td></td><td>63.20</td><td>45.10</td><td>35.30</td><td>73.30</td><td>45.30</td><td>7.10</td><td>9.30</td><td>36.94</td><td>-0.02</td></tr><tr><td>SMART-PC [1]</td><td>ICML&#x27;25</td><td>27.50</td><td>39.10</td><td>19.30</td><td>21.50</td><td>25.60 29.80</td><td>40.80 44.20 48.90</td><td>68.30</td><td></td><td>60.20</td><td>49.40</td><td>45.40</td><td>70.10</td><td>49.10</td><td>8.40</td><td>12.20</td><td>39.56</td><td>2.60</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Backpropagation Method</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ICML-W&#x27;13</td><td>45.44</td><td>43.37</td><td>30.81</td><td>53.36</td><td></td><td>63.34</td><td>62.31</td><td></td><td></td><td></td><td>52.32</td><td></td><td></td><td></td><td>11.19</td><td></td><td>11.35</td></tr><tr><td>PL [8] TTT-ROT [21]</td><td>ICML&#x27;20</td><td>30.20</td><td>34.10</td><td>16.20</td><td>22.60</td><td>44.75 22.60</td><td>60.24 32.40</td><td>32.10</td><td>45.50</td><td>66.95 45.00</td><td>56.11 34.50</td><td>29.30</td><td>64.72 47.80</td><td>59.04 36.20</td><td>10.67 7.10</td><td>8.10</td><td>48.31 29.58</td><td>-7.38</td></tr><tr><td>SHOT [9]</td><td>ICML&#x27;20</td><td>29.50</td><td>31.60</td><td>17.60</td><td>24.80</td><td>27.20</td><td>31.00</td><td>32.40 53.18</td><td>40.70</td><td>35.00</td><td>30.20</td><td>26.60</td><td>36.60</td><td>29.30</td><td>10.50</td><td>12.40</td><td>27.69</td><td>-9.27</td></tr><tr><td>T3A [6]</td><td>NeurIPS&#x27;21</td><td>35.11</td><td>46.82</td><td>13.08</td><td>30.64</td><td>38.55</td><td>50.26</td><td>68.50</td><td></td><td>63.51</td><td>50.60</td><td>46.64</td><td>69.02</td><td>53.53</td><td>7.23</td><td>10.16</td><td>42.46</td><td>5.49</td></tr><tr><td>TENT [24]</td><td>ICLR&#x27;21</td><td>35.28</td><td>49.74</td><td>14.46</td><td>29.60</td><td>38.04</td><td>50.43</td><td>71.60</td><td></td><td>68.85</td><td>53.70</td><td>44.23</td><td>71.43</td><td>55.77</td><td>7.57</td><td>9.29</td><td>43.45</td><td>6.49</td></tr><tr><td>SAR [14] MEMO [35]</td><td>ICLR&#x27;23</td><td>46.99</td><td>42.00</td><td>30.98</td><td>53.70</td><td>43.37</td><td>60.59</td><td>62.48</td><td></td><td>67.64</td><td>59.55</td><td>53.01</td><td>65.58</td><td>61.62</td><td>10.84</td><td>11.53</td><td>48.78</td><td>11.82</td></tr><tr><td></td><td>NeurIPS&#x27;22</td><td>18.07</td><td>16.35</td><td>20.14</td><td>22.03</td><td>16.87</td><td>54.39 58.52</td><td>58.18</td><td></td><td>61.10</td><td>55.94</td><td>42.69</td><td>58.52</td><td>59.04</td><td>9.64</td><td>10.15</td><td>37.44</td><td>0.48</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Backpropagation-Free Method</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DUA [11] LAME [4]</td><td>CVPR&#x27;22</td><td>30.50</td><td>40.10</td><td>10.20</td><td>23.60</td><td>29.90</td><td>43.70 53.53</td><td>46.10</td><td>68.30</td><td>66.30</td><td>48.50</td><td>38.90</td><td>68.70</td><td>48.40</td><td>8.60</td><td>8.10</td><td>38.66</td><td>1.70</td></tr><tr><td>BFTT3D [27]</td><td>CVPR&#x27;22 CVPR&#x27;24</td><td>18.07 16.35</td><td>16.35 14.97</td><td>21.17 14.29</td><td>21.86 17.04</td><td>17.21</td><td>38.90</td><td>57.83 42.00</td><td>57.49 69.71</td><td>61.62 70.74</td><td>53.53 41.14</td><td>41.82 28.92</td><td>56.63</td><td>56.28</td><td>9.98 6.54</td><td>9.81 8.61</td><td>36.88 33.00</td><td>-0.08 -3.96</td></tr><tr><td>PG [31]</td><td>ICCV&#x27;25</td><td>39.59</td><td>52.67</td><td>20.83</td><td>44.58</td><td>13.43 40.62</td><td>53.70</td><td>56.11</td><td>68.85</td><td>72.12</td><td>54.56</td><td>48.88</td><td>69.19 70.91</td><td>43.20 59.04</td><td>10.16</td><td>9.12</td><td>46.78</td><td>9.82</td></tr><tr><td>LN-TTA [32]</td><td>WACV&#x27;26</td><td>43.55</td><td>58.52</td><td>25.30</td><td>46.47</td><td>43.89</td><td>56.45</td><td></td><td>71.95</td><td>73.32</td><td>55.08</td><td>48.71</td><td>71.95</td><td>57.14</td><td>9.12</td><td>8.95</td><td>48.50</td><td>11.54</td></tr><tr><td>SATOR</td><td>Ours</td><td>41.32</td><td>57.47</td><td>24.31</td><td>50.52</td><td>45.31</td><td>56.42</td><td>57.14 59.20</td><td>70.31</td><td>76.39</td><td>56.42</td><td>52.60</td><td>73.26</td><td>58.16</td><td>9.20</td><td>10.59</td><td>49.43</td><td>12.47</td></tr></table>

Table 4. Token-level noise localization on background corruption (ScanObjectNN-C).
<table><tr><td colspan="2">Score AUROC</td><td rowspan="2">AP</td></tr><tr><td></td><td></td></tr><tr><td>Attn</td><td>0.751</td><td>0.765</td></tr><tr><td>Geo</td><td>0.794</td><td>0.683</td></tr><tr><td>Adaptive</td><td>0.839</td><td>0.783</td></tr></table>

Table 5. Runtime and memory on ScanObjectNN-C. FPS: samples/s; Mem. in GB.
<table><tr><td>Method</td><td>Acc.</td><td>FPS</td><td>Mem.</td></tr><tr><td>BFTT3D</td><td>33.00</td><td>55.9</td><td>8.77</td></tr><tr><td>PG</td><td>46.78</td><td>174.1</td><td>1.38</td></tr><tr><td>LN-TTA</td><td>48.50</td><td>790.0</td><td>1.38</td></tr><tr><td>SATOR</td><td>49.43</td><td>483.2</td><td>1.38</td></tr></table>

TTA (+3.00). It exceeds PG on 13 of the 15 corruptions. ScanObjectNN-C. ScanObjectNN-C is the only benchmark derived from real-world scans, where corruptions compound with natural scan artifacts. As shown in Table 3, SATOR achieves the highest mean accuracy of 49.43%, again the largest gain over the source-only model (+12.47) among backpropagation-free methods, surpassing LN-TTA by 0.93 and SAR, the strongest backpropagation-based baseline, by 0.65. It exceeds PG on 13 of the 15 corruptions, with the largest margins on impulse noise (+5.94) and gaussian noise (+4.80).

## 4.2. Ablation and Analysis

Token-Level Noise Localization. We evaluate whether SATOR’s token abnormality scores localize corrupted tokens. We use background corruption in ScanObjectNN-C, where inserted background points provide a token-level proxy for corruption. As shown in Table 4, the fused score achieves the highest AUROC, outperforming both the attention-only and geometry-only scores. This suggests that attention and geometric relation graphs capture complementary corruption signals: the geometry score ranks abnormal tokens more reliably overall with higher AUROC, while the attention score localizes corrupted tokens more precisely with higher

AP. Their fusion achieves the best performance on both metrics, indicating that the two graphs provide complementary information for token partitioning.

Runtime and Memory Footprint. We compare SATOR with representative backpropagation-free 3D TTA methods. As shown in Table 5, SATOR achieves the best accuracy– efficiency trade-off among the compared backpropagationfree methods, reaching higher accuracy than PG (49.43% vs. 46.78%) while improving throughput by 2.8× (483.2 vs. 174.1 FPS). Compared with BFTT3D, SATOR achieves higher accuracy with 8.6× faster throughput (483.2 vs. 55.9 FPS) and substantially lower memory usage (1.38 vs. 8.77 GB). Against LN-TTA, SATOR achieves higher mean accuracy on all three benchmarks at the same memory footprint while remaining competitive in throughput, still ahead of PG and BFTT3D.

Scoring Block Ablation. We examine where token abnormality scoring should be applied within the transformer. Table 6 compares Block-0, 1, 2, 3, 6, and 11 on ScanObjectNN-C with the same routing rule and geometry neighborhood size. Block-0 gives the highest mean accuracy, and on this benchmark accuracy generally decreases as scoring moves to deeper blocks. This suggests that token-level corruption evidence is more separable before extensive token mixing, supporting our choice of Block-0 scoring. Per-corruptiongroup results for all three benchmarks are provided in Appendix C.8.

Routing Calibration Sensitivity. Table 7 evaluates the sensitivity of the routing criterion to the keep-ratio exponent. Entropy routing compares the two branches without a keepratio term. We generalize Eq. 15 to $H ( z _ { \mathrm { p r u n e } } ) / \rho ^ { \gamma }$ and vary the exponent $\gamma ;$ Eq. 15 corresponds $\mathrm { t o } \ \gamma = 1$ . Calibrated routing improves over uncalibrated entropy routing on all three benchmarks, by up to 0.57 points on ModelNet40-C.

Table 6. Scoring block depth ablation on ScanObjectNN-C.
<table><tr><td>Block</td><td>0</td><td>1</td><td>2</td><td>3</td><td>6</td><td>11</td></tr><tr><td>Acc.</td><td>49.43</td><td>48.13</td><td>48.23</td><td>48.16</td><td>47.66</td><td>47.36</td></tr></table>

Table 7. Routing calibration sensitivity across three benchmarks.
<table><tr><td>Dataset</td><td>Ent.</td><td> $\gamma = 0 . 5$ </td><td> $\gamma = 1$ </td><td> $\gamma = 1 . 5$ </td><td> $\gamma = 2$ </td></tr><tr><td>ModelNet40-C</td><td>69.69</td><td>69.85</td><td>70.26</td><td>70.07</td><td>70.02</td></tr><tr><td>ScanObjectNN-C</td><td>49.34</td><td>49.32</td><td>49.43</td><td>49.27</td><td>49.12</td></tr><tr><td>ShapeNet-C</td><td>64.79</td><td>64.82</td><td>64.92</td><td>64.87</td><td>64.86</td></tr></table>

![](images/b9af23fdbe375b3cc28f9cca5c54a22eba12271f2c46bece83674a8457e97ee5.jpg)

![](images/85b5ec662b74b72dab4a6ad48a254e531717e1097710940986f254e701ec4fba.jpg)  
Figure 2. Fused token abnormality scores on ModelNet40-C. The clean example (left) is unimodal under GMM-BIC; the corrupted example (right) is bimodal, with Otsu threshold $\tau ^ { \star }$

Accuracy peaks at $\gamma = 1$ and falls monotonically beyond it on every benchmark, so we use $\gamma = 1$ throughout without dataset-specific tuning.

Fused-Score Distribution Shape. We group samples by the modality of their fused score distribution, fitting Gaussian mixtures with $k \in \{ 1 , 2 , 3 \}$ and selecting k by BIC; 88.8– 96.7% of samples fall in the bimodal or multimodal groups across the three benchmarks (Appendix C.7). Fig. 2 contrasts a clean sample, which is unimodal, with a corrupted sample, which is bimodal with a clear Otsu threshold. Sorted-split Otsu does not assume a bimodal model, and on the unimodal group routing is also the most conservative: as shown in Table 8, it selects the pruned branch for 37.5% of unimodal samples against 42.8% of multimodal ones, and SATOR still improves over the pruned-only branch in every group. Binwise analyses over keep ratio and Otsu separability are in Appendix C.2. Together, these results show that keep-ratiocalibrated routing selectively trusts the pruned branch rather than applying it unconditionally.

Component Ablation. Table 9 reports final accuracy for each scoring variant. Attention alone, geometry alone, and $\alpha \ : = \ : 0 . 5$ all correspond to a globally fixed weight, and none is safe across datasets: without routing the best variant changes with the benchmark, with a gap of up to 6.75 points on ModelNet40-C. With routing, adaptive fusion is highest on all three, ahead of the fixed-weight variant on every one of them by up to 0.66 points. Both components contribute: on

Table 8. Fused-score shape diagnostic on ModelNet40-C. Groups from GMM-BIC; see Modality Classification in the supplementary. Accuracy and selection rate in %.
<table><tr><td>Metric</td><td>Unimodal</td><td>Bimodal</td><td>Multimodal</td></tr><tr><td>Fraction</td><td>7.51</td><td>57.28</td><td>35.21</td></tr><tr><td>Pruned</td><td>63.17</td><td>66.22</td><td>68.62</td></tr><tr><td>SATOR</td><td>65.25</td><td>70.04</td><td>72.06</td></tr><tr><td>Gain</td><td>+2.08</td><td>+3.82</td><td>+3.44</td></tr><tr><td>Selected</td><td>37.52</td><td>40.41</td><td>42.75</td></tr></table>

Table 9. Component ablation. Final classification accuracy averaged over all corruptions. The w/o Routing columns use the pruned prediction unconditionally and are reported as a diagnostic.
<table><tr><td rowspan="2">Scoring</td><td colspan="3">w/o Routing</td><td colspan="3">w/ Routing</td></tr><tr><td>MN40-C</td><td>ScanObj-C</td><td>Shape-C</td><td>MN40-C</td><td>ScanObj-C</td><td>Shape-C</td></tr><tr><td>Geo</td><td>61.87</td><td>48.57</td><td>58.19</td><td>69.29</td><td>48.53</td><td>64.82</td></tr><tr><td>Attn</td><td>68.62</td><td>48.62</td><td>63.96</td><td>69.84</td><td>48.37</td><td>64.69</td></tr><tr><td>Fixed α=0.5</td><td>66.48</td><td>49.02</td><td>62.17</td><td>69.85</td><td>48.77</td><td>64.79</td></tr><tr><td>Adaptive</td><td>66.73</td><td>49.16</td><td>62.60</td><td>70.26</td><td>49.43</td><td>64.92</td></tr></table>

ScanObjectNN-C routing improves only the adaptive score, and replacing our scores with random selection at the same keep ratio loses 1.3–2.4 points on every benchmark. The w/o Routing columns apply pruning unconditionally and isolate the scoring term rather than defining an operating point; under routing the two single scores fall within 0.55 points of each other. Per-corruption results, alternative combination rules, additional controls, and clean-benchmark evaluation are provided in the Supplementary Material.

## 5. Conclusion

We introduced SATOR, a source-free and backpropagationfree framework for 3D test-time adaptation that derives the partitioning threshold and routing decision from each test sample, integrating attention–geometry token abnormality scoring, Otsu-based sample-specific token partitioning, and keep-ratio-calibrated routing. The two scoring signals, attention inconsistency and geometric displacement, prove complementary and separable enough for sample-specific thresholding without a global reference. By deriving the partition from each sample, SATOR removes the repeated evaluations of search-based purging while preserving source-free deployment. Experiments on ModelNet40-C, ScanObjectNN-C, and ShapeNet-C demonstrate that SATOR achieves state-ofthe-art accuracy on three 3D corruption benchmarks.

## References

[1] Ali Bahri, Moslem Yazdanpanah, Sahar Dastani, Mehrdad Noori, Gustavo Adolfo Vargas Hakim, David Osowiechi, Farzad Beizaee, Ismail Ben Ayed, and Christian Desrosiers. SMART–PC: Skeletal model adaptation for robust test–time training in point clouds. Proceedings ofthe 42<sup>nd</sup> International Conference on Machine Learning (ICML 2025), 2025. 1, 2, $^ { 6 , }$ 7

[2] Ali Bahri, Moslem Yazdanpanah, Mehrdad Noori, Sahar Dastani, Milad Cheraghalikhani, David Osowiechi, Farzad Beizaee, Gustavo A Vargas Hakim, Ismail Ben Ayed, and Christian Desrosiers. Test-time adaptation in point clouds: Leveraging sampling variation with weight averaging. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 266–275. IEEE, 2025. 1, 2

[3] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022. 2

[4] Malik Boudiaf, Romain Mueller, Ismail Ben Ayed, and Luca Bertinetto. Parameter-free online test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8344–8353, 2022. 1, 6, 7

[5] Hamidreza Dastmalchi, Aijun An, Ali Cheraghian, Shafin Rahman, and Sameera Ramasinghe. Test-time adaptation of 3d point clouds via denoising diffusion models. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1566–1576. IEEE, 2025. 1, 2

[6] Yusuke Iwasawa and Yutaka Matsuo. Test-time classifier adjustment module for model-agnostic domain generalization. Advances in Neural Information Processing Systems, 34:2427– 2440, 2021. 1, 6, 7

[7] Inseok Kong, Geunyoung Jung, and Jiyoung Jung. Mamvi: 3d test-time adaptation via masked multi-view point clouds. arXiv preprint arXiv:2606.12939, 2026. 1, 2

[8] Dong-Hyun Lee et al. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In Workshop on challenges in representation learning, ICML, 2013. 1, 2, 6, 7

[9] Jian Liang, Dapeng Hu, and Jiashi Feng. Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation. In International conference on machine learning, pages 6028–6039. PMLR, 2020. 1, 6, 7

[10] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Not all patches are what you need: Expediting vision transformers via token reorganizations. arXiv preprint arXiv:2202.07800, 2022. 2

[11] M Jehanzeb Mirza, Jakub Micorek, Horst Possegger, and Horst Bischof. The norm must go on: Dynamic unsupervised domain adaptation by normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14765–14775, 2022. 1, 6, 7

[12] M Jehanzeb Mirza, Inkyu Shin, Wei Lin, Andreas Schriebl, Kunyang Sun, Jaesung Choe, Mateusz Kozinski, Horst Possegger, In So Kweon, Kuk-Jin Yoon, et al. Mate: Masked autoencoders are online 3d test-time learners. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16709–16718, 2023. 1, 2, 6, 7

[13] Zachary Nado, Shreyas Padhy, D Sculley, Alexander D’Amour, Balaji Lakshminarayanan, and Jasper Snoek. Evaluating prediction-time batch normalization for robustness under covariate shift. arXiv preprint arXiv:2006.10963, 2020. 1

[14] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. arXiv preprint arXiv:2302.12400, 2023. 1, 6, 7

[15] Nobuyuki Otsu. A threshold selection method from gray-level histograms. Automatica, 11:285–296, 1975. 2, 5

[16] Yatian Pang, Wenxiao Wang, Francis EH Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked autoencoders for point cloud self-supervised learning. In European conference on computer vision, pages 604–621. Springer, 2022. 3

[17] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652–660, 2017. 1

[18] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017. 1

[19] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems, 34:13937–13949, 2021. 2

[20] Hajin Shim, Changhun Kim, and Eunho Yang. Cloudfixer: Test-time adaptation for 3d point clouds via diffusion-guided geometric transformation. In European Conference on Computer Vision, pages 454–471. Springer, 2024. 1, 2

[21] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with selfsupervision for generalization under distribution shifts. In International conference on machine learning, pages 9229– 9248. PMLR, 2020. 1, 2, 6, 7

[22] Quan Tang, Bowen Zhang, Jiajun Liu, Fagui Liu, and Yifan Liu. Dynamic token pruning in plain vision transformers for semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 777– 786, 2023. 2

[23] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, and Leonidas J Guibas. Kpconv: Flexible and deformable convolution for point clouds. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6411–6420, 2019. 1

[24] Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726, 2020. 1, 2, 6, 7

[25] Xin Wang, Fisher Yu, Zi-Yi Dou, Trevor Darrell, and Joseph E Gonzalez. Skipnet: Learning dynamic routing in convolutional networks. In Proceedings ofthe European conference on computer vision (ECCV), pages 409–424, 2018. 2

[26] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic

graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 38(5):1–12, 2019. 1

[27] Yanshuo Wang, Ali Cheraghian, Zeeshan Hayder, Jie Hong, Sameera Ramasinghe, Shafin Rahman, David Ahmedt-Aristizabal, Xuesong Li, Lars Petersson, and Mehrtash Harandi. Backpropagation-free network for 3d test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23231–23241, 2024. 1, 2, 6, 7

[28] Zixin Wang, Dong Gong, Sen Wang, Zi Huang, and Yadan Luo. Is less more? exploring token condensation as trainingfree test-time adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 144–154, 2025. 2

[29] Xin Wei, Qin Yang, Yijie Fang, Mingrui Zhu, and Nannan Wang. 3d test-time adaptation via graph spectral driven point shift. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 26762–26771, 2025. 2

[30] Xinjian Wu, Fanhu Zeng, Xiudong Wang, and Xinghao Chen. Ppt: Token pruning and pooling for efficient vision transformers. arXiv preprint arXiv:2310.01812, 2023. 2

[31] Moslem Yazdanpanah, Ali Bahri, Mehrdad Noori, Sahar Dastani, Gustavo Adolfo Vargas Hakim, David Osowiechi, Ismail Ben Ayed, and Christian Desrosiers. Purge-gate: Backpropagation-free test-time adaptation for point clouds classification via token purging. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27640–27649, 2025. 1, 2, 3, 4, 6, 7

[32] Moslem Yazdanpanah, Ali Bahri, Mehrdad Noori, Sahar Dastani, Samuel Barbeau, David Osowiechi, Gustavo Adolfo Vargas Hakim, Ismail Ben Ayed, and Christian Desrosiers. Revisiting layer normalization for point cloud test time adaptation. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pages 537–546, 2026. 2, 6, 7

[33] Hongxu Yin, Arash Vahdat, Jose M Alvarez, Arun Mallya, Jan Kautz, and Pavlo Molchanov. A-vit: Adaptive tokens for efficient vision transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10809–10818, 2022. 2

[34] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud transformers with masked point modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19313–19322, 2022. 1

[35] Marvin Zhang, Sergey Levine, and Chelsea Finn. Memo: Test time robustness via adaptation and augmentation. Advances in neural information processing systems, 35:38629–38642, 2022. 1, 2, 6, 7

[36] Renrui Zhang, Ziyu Guo, Peng Gao, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, and Hongsheng Li. Point-m2ae: multiscale masked autoencoders for hierarchical point cloud pretraining. Advances in neural information processing systems, 35:27061–27074, 2022. 1