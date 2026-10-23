# RankAlign: Vision-Language Representation Alignment via Rank Transformation

Enzhe Zhao<sup>1,2∗</sup> Marco Fiorucci<sup>1∗</sup> Lamberto Ballan<sup>1</sup>

<sup>1</sup> University of Padova, Italy

<sup>2</sup> Harbin Institute of Technology, China

Project page: marcofiorucci.github.io/rankalign/

## Abstract

While the Platonic Representation Hypothesis posits that vision and language models converge toward compatible latent geometries, recovering class-level correspondences between unpaired modalities without cross-modal supervision remains challenging. Existing relational matching methods compare within-modality similarity kernels directly, although the same semantic structure may be represented through different numerical parameterizations across modalities. We propose RankAlign, a rank-based normalization strategy for class-level unpaired visionlanguage matching. RankAlign transforms each within-modality similarity kernel into an ordinal relational representation before permutation matching, while leaving the correspondence space and matching solver unchanged. The resulting representation is invariant to strictly increasing reparameterizations of similarity values while preserving the ordinal structure used for matching. We provide exact-recovery and stability guarantees that characterize correspondence recovery under both exact and partial ordinal agreement. Experiments across diverse frozen vision and language encoders and multiple datasets demonstrate improved and more stable class-level correspondence recovery, including under label corruption and domain shift.

## 1 Introduction

Recent advances in foundation models have fundamentally redefined representation learning across diverse modalities. While self-supervised vision encoders like DINO [1] and large-scale language models [3, 17] have matured within their respective domains, multimodal contrastive frameworks such as CLIP [15] and SigLIP2 [22] have further bridged the two modalities. These architectures induce sophisticated latent representations that capture intricate structural and semantic nuances. Yet, despite this progress, the geometric alignment between visual and language representation spaces remains poorly understood when encoders are trained independently on disparate datasets. This poses a significant challenge for class-level unpaired cross-modal matching, where category structure is available within each modality, while the correspondence between visual and textual categories remains unknown and must be recovered.

Representation similarity and the Platonic hypothesis. Independently trained visual and language models frequently exhibit a striking convergence toward compatible representational structures. This phenomenon is formalized by the Platonic Representation Hypothesis, which posits that scaling and data diversity drive neural networks toward shared latent geometries that encode the statistical regularities of the world [6]. In vision-language settings, visual and textual encoders learn compatible relational structures despite disparate training objectives [10, 19]. Although methods such as CCA [5],

SVCCA [16], and CKA [8] provide quantitative measures of representational similarity, they rely on direct numerical comparisons that are inherently sensitive to modality-specific scale distortions and biases.

Class-level unpaired vision-language matching. The emergence of a shared embedding space raises the fundamental question: can explicit correspondences between vision and language representations be recovered when independently trained encoders receive no paired supervision? Traditional alignment relies on image–caption pairs or manually verified correspondences to learn a cross-modal mapping. We instead study class-level unpaired vision-language matching with known withinmodality groups and an unordered candidate set of textual concepts, where visual groups and a set of candidate textual concepts are available but their cross-modal correspondence is unknown. Supervision is confined to constructing class-level representations within each modality, with no paired image–text samples, cross-modal anchors, or verified correspondences provided to the matching procedure. When reliable cross-modal anchors are available, supervised alignment is often simpler and more practical. Our goal is to determine whether independently trained encoders contain sufficient relational structure to recover correspondence without such anchors. This zero-anchor setting can arise when visual groups and a candidate textual ontology are collected or maintained independently, so that both sides are available but their correspondence has not been verified. In this regime, correspondence recovery can reduce the need for manually established cross-modal anchors and also serves as a diagnostic of structural compatibility between independently trained representation spaces. More broadly, it tests whether cross-modal correspondence can emerge from relational structure alone, without shared training data or explicit cross-modal supervision. Recent work frames this problem as matching pairwise relationships between embeddings [19]. Such formulations arise naturally in kernel alignment [2], graph matching via quadratic assignment [23], and structural optimal transport, most notably the Gromov-Wasserstein (GW) distance [14, 24].

Limitations of value-based relational alignment. Blind matching compares the within-modality similarity kernels $K _ { x }$ and $K _ { y }$ and searches for the class permutation π under which their relational structures agree best (Section 2.1). Existing relational alignment methods already operate at the level of pairwise relations rather than raw embeddings, but they still compare the numerical entries of the two kernels directly [11, 19, 12]. Even when vision and language encoders learn similar embeddings, the numerical entries of their within-modality kernels need not be directly comparable across modalities. The two modalities may agree on the ordering of concept similarities, while still assigning different kernel values to the same relations. Consequently, alignment objectives that rely directly on kernel-entry comparison can become weakly discriminative [8, 16].

This numerical mismatch weakens the discriminative value of direct kernel-entry comparison. At the same time, recent evidence suggests that cross-modal agreement can be more stable in relative and local relational structure than in absolute similarity values [4]. These observations raise a key question: can cross-modal correspondence be recovered from relational order when the numerical kernel values are not directly comparable across modalities?

Rank-based relational alignment. To address this issue, we introduce an rank-based formulation of relational alignment. Instead of matching kernels based on raw values, we align them by comparing their rank-based transformation, which uses the ordering induced by these entries. The goal is therefore not to compare $K _ { x } ( i , j )$ and $K _ { y } ( \pi ( i ) , \pi ( j ) )$ as numbers, but to ask whether the two modalities induce consistent relational order over concept pairs. Concretely, we replace each offdiagonal entry of a within-modality kernel with its rank among the off-diagonal entries and evaluate the same matching objective on the transformed kernels. The correspondence problem itself remains unchanged, while the matching signal is determined by relational order rather than raw kernel values.

Contributions. We study class-level unpaired vision-language matching with known categories. Class-level representations are available within each modality, while their cross-modal correspondence is unknown. Our contributions are summarized as follows:

• Solver-agnostic ordinal relational representation. We introduce a rank-based representation that plugs into existing blind-matching objectives and QAP solvers without changing the correspondence space or optimization procedure. RankAlign removes arbitrary strictly increasing reparameterizations of similarity values while retaining the ordinal structure used for matching.

• Theory of ordinal correspondence recovery. We formalize cross-modal agreement up to modality-specific monotone reparameterizations and show that the resulting ordinal representation is invariant to such distortions. We further provide exact-recovery and stability guarantees that separate cross-modal ordinal disagreement from the intrinsic distinguishability of class relational profiles.

![](images/b3eb7e121b4d82b71c2718dacb626ca9b1e8a1635d189e2f462680b929fdd510.jpg)  
Figure 1: Visual-language alignment pipeline. The RankAlign module provides the core ordinal representation of each within-modality similarity kernel. These representations are refined using local neighbor order before permutation matching, without changing the matching space or solver.

• Empirical validation. Experiments across diverse vision and language encoders and datasets demonstrate that RankAlign achieves more stable and reliable cross-modal correspondence alignment.

## 2 Method

Figure 1 summarizes our visual-language alignment pipeline, which illustrates its key steps: i) learning visual and textual embeddings and computing the corresponding prototypes; ii) RankAlign transforms the visual and textual similarity kernels into rank-based similarity kernels; and iii) permutation matching for cross-modal alignment. In the following sections, we will describe each component in detail.

## 2.1 Preliminary

We adopt the blind-matching setting of Schnaus et al. [19]. Let $f _ { v }$ and $f _ { l }$ denote frozen visual and language encoders, respectively. We consider N semantic classes shared across the two modalities. We construct class-level prototypes as

$$
x _ { i } = \frac { 1 } { | \mathcal { T } _ { i } | } \sum _ { I \in \mathcal { T } _ { i } } f _ { v } ( I ) , \qquad y _ { i } = \frac { 1 } { | \mathcal { T } _ { i } | } \sum _ { T \in \mathcal { T } _ { i } } f _ { l } ( T ) .\tag{1}
$$

Then, we center the prototypes independently within each modality and renormalize them before constructing the relational kernels

$$
\hat { x } _ { i } = \frac { x _ { i } - \bar { x } } { \| x _ { i } - \bar { x } \| } , \qquad \bar { x } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } x _ { j } ,\tag{2}
$$

and analogously for the language prototypes. This operation uses no cross-modal correspondence information and is permutation equivariant.

For notational simplicity, we use $x _ { i }$ and $y _ { i }$ below to denote the resulting normalized prototypes. The within-modality similarity kernels are

$$
K _ { x } ( i , j ) = \langle x _ { i } , x _ { j } \rangle , \qquad K _ { y } ( i , j ) = \langle y _ { i } , y _ { j } \rangle .\tag{3}
$$

A candidate cross-modal correspondence is represented by a permutation $\pi \in S _ { N }$ . Given a distortion function $\ell : \mathbb { R } \times \mathbb { R } \to \mathbb { R }$ , define

$$
D _ { \ell } ( K _ { x } , K _ { y } ; \pi ) = \sum _ { i \neq j } \ell ( K _ { x } ( i , j ) , K _ { y } ( \pi ( i ) , \pi ( j ) ) ) .\tag{4}
$$

Blind relational matching seeks

$$
\pi ^ { \star } \in \arg \operatorname* { m i n } _ { \pi \in S _ { N } } D _ { \ell } ( K _ { x } , K _ { y } ; \pi ) .\tag{5}
$$

Equivalently, we use the agreement score

$$
f ( \pi ) = \sum _ { i \neq j } K _ { x } ( i , j ) K _ { y } ( \pi ( i ) , \pi ( j ) ) .\tag{6}
$$

Lemma 1 (Distortion and agreement). For $\ell ( a , b ) = - a b$ , minimizing $D _ { \ell }$ is equivalent to maximizing $f .$ For the squared distortion $\ell ( a , b ) = ( a - b ) ^ { 2 }$

$$
D _ { \ell } ( K _ { x } , K _ { y } ; \pi ) = C _ { x } + C _ { y } - 2 f ( \pi ) ,\tag{7}
$$

where $C _ { x }$ and $C _ { y }$ do not depend on π. Hence both formulations have identical optimizers.

Hence, the objective function can be written as $\pi ^ { \star } \in$ arg max $\pi { \in } S _ { N } f ( K _ { x } , K _ { y } ; \pi )$ . And we use the maximization form throughout the remainder of the paper. The resulting problem is a quadratic assignment problem (QAP). A proof of Lemma 1 is given in Appendix $\mathrm { A . 1 }$

## 2.2 Limitations of value-based relational alignment

The objective in Eq. 6 compares numerical kernel values directly. This implicitly assumes that visual and language similarities are sufficiently calibrated across modalities. Such calibration is not guaranteed for independently trained encoders.

To formalize this distinction, suppose both modalities encode a shared latent relational structure $S = \{ s _ { i j } \}$ through modality-specific strictly increasing maps,

$$
K _ { x } ( i , j ) = \phi _ { x } ( s _ { i j } ) , \qquad K _ { y } ( \pi ^ { \star } ( i ) , \pi ^ { \star } ( j ) ) = \phi _ { y } ( s _ { i j } ) ,\tag{8}
$$

where $\pi ^ { \star }$ denotes the ground-truth correspondence.

Even under the correct correspondence, the numerical similarities in the two modalities may differ substantially whenever $\phi _ { x }$ and $\phi _ { y }$ have different parameterizations. Consequently, a value-based objective can penalize a correct correspondence despite the two modalities encoding the same underlying relational structure.

By contrast, strictly increasing transformations preserve relational order. For any two unordered class pairs $e , e ^ { \prime }$

$$
K _ { x } ( e ) < K _ { x } ( e ^ { \prime } ) \quad \Longleftrightarrow \quad K _ { y } ( \pi ^ { \star } ( e ) ) < K _ { y } ( \pi ^ { \star } ( e ^ { \prime } ) ) .\tag{9}
$$

Thus, the two modalities may share the same ordinal relational geometry without sharing the same numerical parameterization. This suggests representing relational structure through order rather than absolute similarity values, thereby making the matching objective insensitive to modality-specific monotone reparameterizations.

## 2.3 Rank-based relational alignment

Because the similarity kernels are symmetric, we rank unordered class pairs. Let

$$
{ \mathcal E } = \{ \{ i , j \} : 1 \le i < j \le N \} , \qquad M = | { \mathcal E } | = \frac { N ( N - 1 ) } 2 .\tag{10}
$$

For a symmetric kernel K, we define the empirical cumulative distribution function of its off-diagonal similarities as

$$
F _ { K } ( t ) = \frac { 1 } { M } \sum _ { \{ i , j \} \in \mathcal { E } } \mathbf { 1 } [ K ( i , j ) \le t ] ,\tag{11}
$$

and the corresponding rank transformation as

$$
R ( K ) ( i , j ) = { \left\{ \begin{array} { l l } { F _ { K } ( K ( i , j ) ) , } & { i \neq j , } \\ { 0 , } & { i = j . } \end{array} \right. }\tag{12}
$$

The transformation replaces the numerical spacing of pairwise similarities by their relative positions within each modality. If all M unordered off-diagonal similarities are distinct, the transformed values are $1 / M , \ldots , \bar { 1 }$ . If ties are present in the original kernel, they remain tied after the transformation. Rank transformation preserves the available relational ordering rather than creating additional relational information.

Lemma 2 (Ordinal invariance). For every strictly increasing function $\phi : \mathbb { R }  \mathbb { R } ,$

$$
R ( \phi \circ K ) = R ( K ) .\tag{13}
$$

Moreover, for every permutation $\sigma \in S _ { N }$

$$
R ( K \circ \sigma ) = R ( K ) \circ \sigma ,\tag{14}
$$

where $( K \circ \sigma ) ( i , j ) = K ( \sigma ( i ) , \sigma ( j ) )$

Lemma 2 establishes the central invariance of the rank representation. Modality-specific strictly monotone reparameterizations of similarity values are removed without changing the underlying correspondence structure. The transformation also preserves every strict pairwise ordering induced by the original kernel. The proof is provided in Appendix A.1.

Replacing the kernels in Eq. 6 by their rank representations gives the ordinal matching objective

$$
f _ { \mathrm { o r d } } ( \pi ) = \sum _ { i \ne j } \widetilde { K } _ { x } ( i , j ) \widetilde { K } _ { y } ( \pi ( i ) , \pi ( j ) ) , \qquad \widetilde { K } _ { x } = R ( K _ { x } ) , \quad \widetilde { K } _ { y } = R ( K _ { y } ) .\tag{15}
$$

The correspondence problem itself is unchanged. RankAlign changes only the relational representation supplied to the matching objective. The recovery analysis below focuses on the core rank representation $R ( K )$ , while the full ordinal refinement used in the experiments is described in Appendix A.1.

## 2.4 Recovery guarantees

We next give conditions under which ordinal relational matching recovers the ground-truth correspondence. We first establish exact recovery for the basic rank objective in Eq. 15, which isolates the contribution of ordinal information, and then study stability under partial ordinal disagreement.

Assumption 1 (Strict ordinal consistency). Let $\pi ^ { \star }$ denote the ground-truth correspondence. Assume that the unordered off-diagonal values of $K _ { x }$ are distinct and that the two modalities induce the same strict ordering over unordered class pairs under $\pi ^ { \star }$ . For any distinct $e , e ^ { \prime } \in \mathcal { E } ,$

$$
K _ { x } ( e ) < K _ { x } ( e ^ { \prime } ) \quad \Longleftrightarrow \quad K _ { y } ( \pi ^ { \star } ( e ) ) < K _ { y } ( \pi ^ { \star } ( e ^ { \prime } ) ) .\tag{16}
$$

Assumption 1 requires agreement in relational order, rather than agreement in numerical similarity values. The noise-free latent model of Eq. 8 satisfies this assumption whenever the latent similarities $s _ { i j }$ are distinct, independently of the strictly increasing modality-specific maps $\phi _ { x }$ and $\phi _ { y }$

Proposition 1 (Exact recovery). Under Assumption 1 and $N \geq 3 , \pi ^ { \star }$ is the unique maximizer of $f _ { \mathrm { o r d } }$ over $S _ { N }$ . Moreover, for every $\pi \neq \pi ^ { \star }$

$$
f _ { \mathrm { o r d } } ( \pi ^ { \star } ) - f _ { \mathrm { o r d } } ( \pi ) \geq \frac { 2 } { M ^ { 2 } } , \qquad M = \frac { N ( N - 1 ) } { 2 } ,\tag{17}
$$

when the objective is summed over ordered off-diagonal pairs. Equivalently, the corresponding margin over unordered pairs is at least $1 / M ^ { 2 }$

Proposition 1 shows that strict ordinal consistency is sufficient for unique correspondence recovery. The guarantee depends only on the relational ordering and the problem size, rather than on the numerical spacing of either kernel. The complete proof is given in Appendix $\mathrm { A } . 2 .$

Strict ordinal consistency is an idealized condition. In practice, some relational orders may disagree across modalities. We therefore state the stability result in a representation-independent form. Let

$Z ( K )$ denote any permutation-equivariant ordinal representation, with $Z _ { x } = Z ( K _ { x } )$ and $Z _ { y } =$ $Z ( K _ { y } )$ . We define the cross-modal residual as

$$
E _ { Z } = Z _ { y } \circ \pi ^ { \star } - Z _ { x } ,\tag{18}
$$

and the structural discrepancy induced by a permutation τ as

$$
d _ { Z } ( \tau ) = \lVert Z _ { x } - Z _ { x } \circ \tau \rVert _ { F } .\tag{19}
$$

Proposition 2 (Stability under partial ordinal disagreement). For any candidate permutation $\pi ,$ let ${ \tau = } \overline { { { ( \pi ^ { \star } ) ^ { - 1 } \circ \pi } } }$ . Then

$$
f _ { Z } ( \pi ^ { \star } ) - f _ { Z } ( \pi ) \geq d _ { Z } ( \tau ) \left( \frac { 1 } { 2 } d _ { Z } ( \tau ) - \| E _ { Z } \| _ { F } \right) ,\tag{20}
$$

where $\begin{array} { r } { f _ { Z } ( \pi ) = \sum _ { i \neq j } Z _ { x } ( i , j ) Z _ { y } ( \pi ( i ) , \pi ( j ) ) } \end{array}$ . Consequently, every maximizer $\widehat { \pi }$ satisfies

$$
d _ { Z } ( \widehat { \tau } ) \leq 2 \| E _ { Z } \| _ { F } , \qquad \widehat { \tau } = ( \pi ^ { \star } ) ^ { - 1 } \circ \widehat { \pi } .\tag{21}
$$

The result separates two sources of difficulty. The residual $\| E _ { Z } \| _ { F }$ measures cross-modal ordinal disagreement, while $d _ { Z } ( \tau )$ measures how distinguishable the relational structure is under an incorrect relabeling. Further specialization to the practical RankAlign representation, together with bounds on mismatched classes and approximate solver outputs, is provided in Appendix $\mathrm { A } . 2$

## 2.5 Computational complexity

We finally analyze the additional computational cost introduced by RankAlign. Constructing a dense similarity kernel from N class prototypes of dimension d requires $\mathrm { \dot { \cal O } } ( N ^ { 2 } d )$ time and $O ( N ^ { 2 } )$ memory. The kernel-level rank transformation sorts $M = N ( N - 1 ) \dot { / } 2 = O ( N ^ { 2 } )$ pairwise similarities, while the anchor-wise transformation ranks N rows of length $N - 1$ . Both operations require $O ( N ^ { 2 } \log N )$ time. The score transformation, symmetrization, and standardization are all $O ( N ^ { \mathbf { \bar { 2 } } } )$ .

Consequently, excluding feature extraction, RankAlign representation can be constructed in $O ( N ^ { 2 } d +$ $N ^ { 2 } \log N )$ time and $O ( \breve { N } ^ { 2 } + N d )$ memory. Since the $O (  { \bar { N } } ^ { 2 } d )$ kernel construction is already required by value-based relational matching, the additional asymptotic preprocessing overhead of RankAlign is $\dot { O } ( N ^ { 2 } \log N )$ time and $O ( N ^ { 2 } )$ memory. RankAlign does not change the subsequent correspondence problem. If $\overrightharpoon { C } _ { \cal S } ( N )$ denotes the running time of a chosen QAP solver S, the complete matching pipeline requires $C _ { S } ^ { ' } ( N ) + O ( N ^ { 2 } d + N ^ { 2 } )$ log N) time. Further details are provided in Appendix $_ { \mathrm { A } . 3 }$

## 3 Experiments

We measured performance using matching accuracy, defined as the fraction of correctly recovered class correspondences. A matching accuracy of 1.00 corresponds to exact recovery of the groundtruth class permutation. We compared against BlindMatch [19] using both Gromov–Wasserstein ((GW)) and Mutual kNN. RankAlign uses the same permutation solvers as the baselines but replaces the raw within-modality similarity kernels with the ordinal representation defined in Section 2.3 before matching. To isolate the contribution of this representation, we additionally compare against z-score normalization, quantile normalization, and row-wise ranking under an identical protocol. The controlled ablation in Appendix B.2 shows that the gains cannot be explained by generic normalization or local ranking alone. All experiments were conducted on a single NVIDIA RTX A6000 GPU with 48GB memory.

Pre-trained models. We evaluated RankAlign using a variety of frozen pre-trained visual and textual encoders to cover diverse representation geometries and supervision paradigms, including DINO/DINOv2 [1, 13], DeiT [21], and vision-language models such as CLIP [15], SigLIP [25], and SigLIP2 [22]. For language side, we primarily used SentenceTransformers models [17].

Prompt construction. Given an image classification dataset with N visual groups, we form classlevel prototypes independently in the two modalities. On the visual side, each image is embedded with the frozen visual encoder and the embeddings are averaged within each visual group. On the language side, we construct an unordered candidate set of $N$ textual concepts. For each candidate concept, we generate a small collection of semantically equivalent textual descriptions, embed them with the frozen language encoder, and average them to obtain a textual prototype. The correspondence between the visual groups and textual prototypes is withheld from the matching algorithm. The algorithm receives only the two unordered sets of class-level prototypes and recovers their cross-modal permutation from within-modality relational structure.

Dataset We conducted our experiments on CIFAR-100 [9], ImageNet-100 [18], the medical imaging dataset CheXpert [7], and a pseudo-class dataset derived from the LAION-Aesthetic subset [20]. More details are provided in Appendix B.1.

## 3.1 Computational Cost

Because encoders were frozen, no end-to-end training was required. RankAlign introduces only an ordinal preprocessing step before permutation matching. As analyzed in Section 2.5, constructing the complete RankAlign representation requires $O ( N ^ { 2 } \log N )$ additional preprocessing time and $\bar { O ( N ^ { 2 } ) }$ memory beyond the similarity-kernel construction. The transformed representations remain $N \times N$ so the overall computational cost is primarily determined by the QAP solver.

Table 1: Runtime and matching accuracy on CIFAR-100 across different problem sizes using CLIP as the vision encoder and Sentence-T as the language encoder. Results are reported as mean ± standard deviation over five random seeds. <sup>∗</sup> indicates that the solver reaches the 300s time limit. Actual wall-clock runtime may be slightly higher due to termination and post-processing overhead.
<table><tr><td>Solver</td><td>Method</td><td> $N = 5 0$ </td><td> $N = 7 0$ </td><td> $N = 1 0 0$ </td></tr><tr><td colspan="5">Matching accuracy</td></tr><tr><td>FAQ</td><td>BLINDMATCH</td><td> $0 . 2 6 \pm 0 . 1 0$ </td><td> $0 . 1 4 \pm 0 . 0 9$ </td><td> $0 . 0 4 \pm 0 . 0 2$ </td></tr><tr><td rowspan="2">OT</td><td>RANKALIGN</td><td> $0 . 9 8 \pm 0 . 0 2$ </td><td> $0 . 7 9 \pm 0 . 0 5$ </td><td> $0 . 6 6 \pm 0 . 2 1$ </td></tr><tr><td>BLINDMATCH</td><td> $0 . 0 4 \pm 0 . 0 1$ </td><td> $0 . 0 2 \pm 0 . 0 2$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td></tr><tr><td rowspan="2">FHG</td><td>RANKALIGN</td><td> $0 . 9 8 \pm 0 . 0 2$ </td><td> $0 . 7 9 \pm 0 . 0 5$ </td><td> $0 . 6 9 \pm 0 . 1 8$ </td></tr><tr><td>BLINDMATCH</td><td> $0 . 7 7 \pm 0 . 0 2$ </td><td> $0 . 4 2 \pm 0 . 0 7$ </td><td> $0 . 1 5 \pm 0 . 0 3$ </td></tr><tr><td></td><td>RANKALIGN</td><td> $0 . 9 8 \pm 0 . 0 2$  Runtime (s)</td><td> $0 . 9 1 \pm 0 . 0 4$ </td><td> $0 . 7 9 \pm 0 . 0 2$ </td></tr><tr><td colspan="5"></td></tr><tr><td rowspan="2">FAQ</td><td>BLINDMATCH</td><td> $0 . 5 4 \pm 0 . 1 8$ </td><td> $0 . 7 0 \pm 0 . 1 2$ </td><td> $1 . 4 4 \pm 0 . 2 1$ </td></tr><tr><td>RANKALIGN</td><td> $0 . 3 6 \pm 0 . 0 2$ </td><td> $1 . 1 0 \pm 0 . 0 3$ </td><td> $2 . 7 0 \pm 0 . 3 1$ </td></tr><tr><td rowspan="2">OT</td><td>BLINDMATCH</td><td> $0 . 1 2 \pm 0 . 0 3$ </td><td> $0 . 2 8 \pm 0 . 0 1$ </td><td> $0 . 8 2 \pm 0 . 0 1$ </td></tr><tr><td>RANKALIGN</td><td> $0 . 4 2 \pm 0 . 0 4$ </td><td> $1 . 2 0 \pm 0 . 0 3$ </td><td> $2 . 8 0 \pm 0 . 2 8$ </td></tr><tr><td rowspan="2">FHG</td><td>BLINDMATCH</td><td> $3 0 0 . 0 0 ^ { * }$ </td><td> $3 0 0 . 0 0 ^ { * }$ </td><td> $3 0 0 . 0 0 ^ { * }$ </td></tr><tr><td>RANKALIGN</td><td> $3 0 0 . 0 0 ^ { * }$ </td><td> $3 0 0 . 0 0 ^ { * }$ </td><td> $3 0 0 . 0 0 ^ { * }$ </td></tr></table>

Table 1 reports runtime and matching accuracy on CIFAR-100 with respect to the problem size N. When the same solver is used, RankAlign introduces only a modest runtime overhead while providing substantially higher matching accuracy. The OT-based solver remains computationally efficient across all tested problem sizes. In contrast, FHG reaches the runtime limit at larger problem sizes in our implementation. These results indicate that the main scalability bottleneck comes from permutation optimization rather than the rank transformation itself.

## 3.2 Scalability

We tested scalability on CIFAR-100 and ImageNet-100 by increasing the problem size N, where larger N requires finer-grained semantic discrimination and yields a more challenging matching problem. Following [19], we selected class subsets using a p-dispersion strategy, which favors subsets whose relational structures are well aligned across modalities. Specifically, let $S \subseteq \{ 1 , \dots , L \}$ be a subset of classes with $| S | = N$ . We can get a subset score

$$
A ( S ) = \sum _ { i , j \in S } \ell ( K _ { x } ( i , j ) , K _ { y } ( i , j ) ) ,
$$

which measures the agreement between visual and language relational kernels restricted to S. We then select subsets by approximately solving $S ^ { \star } \in$ arg max $S { : | S | } = N A ( S )$ , which is an instance of the pdispersion-sum problem. This procedure yielded class subsets where cross-modal relational structure is more consistent, making non-trivial matching feasible. For each problem size N, i.e., the number of classes to be matched, we constructed matching instances from the selected subsets and reported results across different visual encoders. We use the same solver of FHG [19] for a fair comparison. Fig. 2 shows that the matching problem becomes progressively more challenging as N increases, particularly at larger problem sizes. At smaller N, RANKALIGN reaches a matching accuracy of 1 for several backbone settings. Since matching accuracy measures the fraction of correctly recovered class correspondences, a value of 1 means that the recovered permutation exactly matches the ground-truth permutation, $\operatorname { i . e . , } \hat { \pi } = \pi ^ { \star }$ . This should be interpreted as exact correspondence recovery for the given matching instance rather than as an indication of overfitting. This observation is consistent with Proposition 1, which shows that under strict ordinal consistency, $\pi ^ { \star }$ is the unique maximizer of the ordinal matching objective. As N grows, RANKALIGN consistently outperforms BLINDMATCH(GW), with the advantage becoming increasingly pronounced at larger problem sizes. Most notably, at $N = 1 0 0$ , RANKALIGN retains approximately 30% − 60% matching accuracy across the representative backbones on CIFAR-100 and ImageNet-100, while BLINDMATCH(GW) falls to near-chance levels. This yields an improvement of roughly 45% − 70% at the largest evaluated problem size.

These observations are consistent with the recovery analysis in Section 2.4. Ordinal consistency yields unique recovery of the ground-truth permutation, while under partial disagreement recovery depends on the balance between cross-modal ordinal agreement and the distinguishability of class relational profiles. The widening performance gap suggests that ordinal relational structure remains informative as correspondence recovery requires increasingly fine-grained discrimination among classes. For clarity, Fig. 2 reports representative results for CLIP, DINOv2, and SigLIP2, while results across all evaluated vision backbones are provided in Appendix B.4. Additional comparisons in Appendix B.3 show that classical Orthogonal Procrustes and vanilla Gromov–Wasserstein optimal transport remain substantially below RankAlign under the same representations and matching instances.

![](images/33374cc3cb4fe11c3fdfa4063c0999df75a6c8a1ed0508273f20058ae4071501.jpg)

![](images/bb967d85fc1eb396fe3d18c2fe7ba6dac422ede36aaddb43098d388aedb6690f.jpg)  
Figure 2: Matching accuracy as a function of problem size on CIFAR-100 and ImageNet-100. Each color corresponds to a vision backbone. For each backbone, the darker solid line denotes RANKALIGN, while the lighter dashed line denotes BLINDMATCH with GW.

## 3.3 Robustness to label corruption

Setup. We conducted this experiment on CIFAR-100 and ImageNet-100 with $N = 1 0 0$ classes. For each corruption rate $r \in [ 0 , 1 ]$ , we randomly assigned a fraction r of image embeddings within each class to incorrect classes. Let $S _ { c }$ be the set of samples originally assigned to class $c ,$ and let $m _ { c } = \lfloor r \vert S _ { c } \vert \rfloor$ be the number of corrupted samples from class c. This produces corrupted labels $\tilde { y } _ { n } .$ while the language prototypes $\{ t _ { c } \} _ { c = 1 } ^ { C }$ remain unchanged. Using the corrupted labels, we recomputed the visual class centroids as

$$
\tilde { v } _ { c } = \frac { 1 } { | \{ n : \tilde { y } _ { n } = c \} | } \sum _ { n : \tilde { y } _ { n } = c } z _ { n } ,
$$

where $z _ { n }$ is the visual embedding of image n. We performed cross-modal matching between the corrupted visual prototypes $\{ \tilde { v } _ { c } \} _ { c = 1 } ^ { \tilde { C } }$ and the clean language prototypes.

Let π denote the recovered permutation. We report the class-level matching accuracy

$$
\mathrm { A c c } _ { \mathrm { m a t c h } } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathbf { 1 } [ \pi ( c ) = c ] .
$$

Each experiment was repeated over multiple random seeds and we report mean and standard deviation. Robustness is quantified by the relative drop from the clean setting:

$$
\mathrm { D r o p } ( r ) = \frac { \mathrm { A c c } ( 0 ) - \mathrm { A c c } ( r ) } { \mathrm { A c c } ( 0 ) } \times 1 0 0 \% ,
$$

where $\operatorname { A c c } ( r )$ denotes the mean matching accuracy at corruption rate r.

Table 2: Robustness under visual-side label corruption on CIFAR-100 and ImageNet-100. We report matching accuracy (%) as mean ± standard deviation over five random seeds and relative performance drop (%) with respect to the clean setting.
<table><tr><td>Dataset</td><td>Metric</td><td>0.00</td><td>0.10</td><td>0.20</td><td>0.30</td><td>0.40</td></tr><tr><td>CIFAR-100</td><td>Acc. (%) Drop (%)</td><td>78.7 ± 7.3 0.0</td><td>78.0 ± 5.2 0.9</td><td>77.6 ± 5.8 1.4</td><td>73.1 ± 15.9 7.1</td><td> $6 8 . 9 \pm 6 . 4$  12.5</td></tr><tr><td>ImageNet-100</td><td>Acc. (%) Drop (%)</td><td>74.4 ± 5.4 0.0</td><td>71.0 ± 3.9 4.6</td><td>71.4 ± 5.4 4.0</td><td>56.6 ± 12.0 23.9</td><td>41.5 ± 10.3 44.2</td></tr></table>

Table 2 reports robustness under increasing visual-side label corruption on CIFAR-100 and ImageNet-100. As the corruption rate increases, matching accuracy decreases on both datasets, reflecting the impact of noisy class assignments on prototype quality. The degradation remains gradual across a wide range of corruption levels, indicating that our method maintains reasonable performance even under moderate noise. Similar trends are observed on both datasets, suggesting that the approach is consistently robust across different data distributions.

## 3.4 LAION Pseudo-Class Matching

We further evaluate RankAlign on the LAION pseudo-class dataset, which provides a noisy image– text setting without manually defined class annotations. We compare RANKALIGN against BLIND-MATCH(GW) across different visual backbones.

Table 3: Matching accuracy on the LAION pseudo-class dataset. We compare BLINDMATCH(GW) with RANKALIGN across different visual backbones.
<table><tr><td>Vision model</td><td>BLINDMATCH</td><td>RANKALIGN</td></tr><tr><td>DINOv2</td><td> $0 . 5 9 \pm 0 . 2 9$ </td><td> $0 . 9 6 \pm 0 . 0 9$ </td></tr><tr><td>DeiT</td><td> $0 . 7 2 \pm 0 . 0 5$ </td><td> $0 . 7 6 \pm 0 . 0 9$ </td></tr><tr><td>CLIP</td><td> $0 . 7 7 \pm 0 . 1 3$ </td><td> $0 . 6 8 \pm 0 . 0 5$ </td></tr><tr><td>SigLIP</td><td> $0 . 5 4 \pm 0 . 1 3$ </td><td> $0 . 7 9 \pm 0 . 1 5$ </td></tr><tr><td>SigLIP2</td><td> $0 . 6 1 \pm 0 . 0 9$ </td><td> $0 . 7 4 \pm 0 . 1 1$ </td></tr></table>

As shown in Table 3, RANKALIGN achieves stronger overall matching performance than BLIND-MATCH(GW) across the evaluated visual backbones. The results indicate that rank-based relational alignment is generally more robust in this noisy pseudo-class setting, where class definitions are derived from weakly supervised image–text signals rather than manually curated labels. The consistent gains across different backbone families further suggest that the benefit of RankAlign is not tied to a specific visual encoder.

## 3.5 Medical Domain Evaluation

We also evaluate RankAlign on CheXpert, which introduces a substantial domain gap between medical images and language descriptions, together with subtle inter-class differences and class imbalance. Compared with natural-image benchmarks, CheXpert presents a more challenging crossmodal matching setting because different pathological categories may exhibit highly similar visual patterns, while the corresponding language descriptions encode fine-grained clinical distinctions. Moreover, the uneven class distribution further perturbs class-level prototype estimation, making reliable correspondence recovery more difficult. Table 4 reports matching accuracy across different visual backbones. We compare RANKALIGN against BLINDMATCH(GW) under the same evaluation setting. Across all evaluated backbones, RANKALIGN achieves substantially higher matching accuracy, indicating that the rank-based relational representation remains effective under pronounced domain shift and challenging medical-image semantics.

Table 4: Matching performance on the CheXpert subset. We report matching accuracy across different vision backbones for BLINDMATCH(GW) and RANKALIGN.
<table><tr><td>Vision model</td><td>BLINDMATCH</td><td>RANKALIGN</td></tr><tr><td>DeiT</td><td> $0 . 0 8 \pm 0 . 0 1$ </td><td> $0 . 6 5 \pm 0 . 0 1$ </td></tr><tr><td>DINOv2</td><td> $0 . 0 6 \pm 0 . 0 1$ </td><td> $0 . 6 2 \pm 0 . 3 0$ </td></tr><tr><td>CLIP</td><td> $0 . 0 8 \pm 0 . 0 1$ </td><td> $0 . 6 3 \pm 0 . 2 6$ </td></tr><tr><td>SigLIP</td><td> $0 . 1 0 \pm 0 . 0 2$ </td><td> $0 . 4 5 \pm 0 . 2 2$ </td></tr><tr><td>SigLIP2</td><td> $0 . 0 8 \pm 0 . 0 0$ </td><td> $0 . 2 3 \pm 0 . 0 4$ </td></tr></table>

## 4 Conclusions

This work revisits class-level unpaired vision-language matching from an ordinal perspective and introduces RankAlign, a rank-based normalization strategy for relational correspondence. Rather than modifying the underlying permutation-matching framework, RankAlign changes the relational repre sentation supplied to the matching objective. By emphasizing relational order over absolute similarity magnitudes, it provides a monotone-invariant signal for recovering cross-modal correspondences without cross-modal supervision. Beyond its empirical effectiveness, these findings offer strong support for the Platonic Representation Hypothesis, demonstrating that cross-modal convergence does not require rigid metric equivalence, but rather emerges naturally through shared relational topologies across independently trained representations. A fundamental property of rank-based normalization lies in its complete invariance to strictly increasing reparameterizations, providing provable guarantees against modality-specific scale distortions, domain shifts, and monotonic noise. These theoretical properties translate directly into empirical robustness: across diverse frozen encoders and matching settings, RankAlign delivers stable class-level correspondence recovery even under severe corruption conditions. By formulating cross-modal correspondence in terms of relational order rather than absolute similarity magnitudes, the approach ensures invariant alignment without relying on cross-modal anchors or paired supervision.

Limitations RankAlign relies on informative class-level prototypes in both modalities. When visual groups are estimated from noisy clustering rather than reliable class labels, prototype errors can distort the induced relational structure and reduce ordinal agreement, and our experiments suggest that prototype quality is a major source of degradation, although larger and more fine-grained matching problems may introduce additional difficulty. The permutation-based matching stage can also become computationally expensive as the number of classes grows. Moreover, RankAlign focuses on class-level zero-anchor correspondence and intentionally discards absolute similarity magnitudes, so instance-level, open-vocabulary, and hybrid ordinal–magnitude alignment remain directions for future work.

Broader Impact RankAlign may reduce the need for manually verified cross-modal correspondences in settings where paired multimodal supervision is limited or expensive. Potential applications include representation analysis, class-level retrieval, and domain-specific multimodal matching. As with other alignment methods, unreliable prototypes or biases in pretrained models may lead to incorrec correspondences, so applications in sensitive domains should include appropriate validation.

## Acknowledgments

We are grateful to Francis Engelmann for carefully reviewing the manuscript and providing valuable suggestions that helped improve the clarity of the paper.

## References

[1] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proc. of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 1, 6

[2] Nello Cristianini, John Shawe-Taylor, André Elisseeff, and Jaz Kandola. On kernel-target alignment. In Proc. ofAdvances in Neural Information Processing Systems (NeurIPS), 2001. 2

[3] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proc. ofthe Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2019. 1

[4] Fabian Gröger, Shuo Wen, and Maria Brbic. Revisiting the platonic representation hypothesis: An aristotelian view. In Proc. of the International Conference on Machine Learning (ICML), 2026. 2

[5] Harold Hotelling. Relations between two sets of variates. Biometrika, 28(3-4):321–377, 1936. 1

[6] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. Position: The platonic representation hypothesis. In Proc. of the International Conference on Machine Learning (ICML), 2024. 1

[7] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. In Proc. of the AAAI Conference on Artificial Intelligence, 2019. 7, 17

[8] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In Proc. ofthe International Conference on Machine Learning (ICML), 2019. 2

[9] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. 7, 17

[10] Mayug Maniparambil, Raiymbek Akshulakov, Yasser Abdelaziz Dahou Djilali, Mohamed El Amine Seddik, Sanath Narayan, Karttikeya Mangalam, and Noel E. O’Connor. Do vision and language encoders represent the world similarly? In Proc. ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1

[11] Luca Moschella, Valentino Maiorca, Marco Fumero, Antonio Norelli, Francesco Locatello, and Emanuele Rodolà. Relative representations enable zero-shot latent space communication. In Proc. ofthe International Conference on Learning Representations (ICLR), 2023. 2

[12] Antonio Norelli, Marco Fumero, Valentino Maiorca, Luca Moschella, Emanuele Rodolà, and Francesco Locatello. ASIF: Coupled data turns unimodal models to multimodal without training. In Proc. ofAdvances in Neural Information Processing Systems (NeurIPS), 2024. 2

[13] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning Robust Visual Features without Supervision. Transaction of Machine Learing Research (TMLR), 2024. 6

[14] Gabriel Peyré, Marco Cuturi, and Justin Solomon. Gromov-wasserstein averaging of kernel and distance matrices. In Proc. of the International Conference on Machine Learning (ICML), 2016. 2

[15] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proc. of the International Conference on Machine Learning (ICML), 2021. 1, 6

[16] Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. In Proc. ofAdvances in Neural Information Processing Systems (NeurIPS), 2017. 2

[17] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proc. ofthe International Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019. 1, 6

[18] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision, 115(3):211–252, 2015. 7, 17

[19] Dominik Schnaus, Nikita Araslanov, and Daniel Cremers. It’s a (blind) match! towards visionlanguage correspondence without parallel data. In Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 2, 3, 6, 7, 8

[20] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: an open large-scale dataset for training next generation image-text models. In Proc. ofAdvances in Neural Information Processing Systems (NeurIPS), 2022. 7, 17

[21] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In Proc. ofthe International Conference on Machine Learning (ICML), 2021. 6

[22] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Henaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv preprint arXiv:2502.14786, 2025. 1, 6

[23] Joshua T. Vogelstein, John M. Conroy, Vince Lyzinski, Louis J. Podrazik, Steven G. Kratzer, Eric T. Harley, Donniell E. Fishkind, R. Jacob Vogelstein, and Carey E. Priebe. Fast approximate quadratic programming for graph matching. PLOS One, 10(4), 2015. 2

[24] Hongteng Xu, Dixin Luo, Hongyuan Zha, and Lawrence Carin. Gromov-Wasserstein Learning for Graph Matching and Node Embedding. In Proc. of the International Conference on Machine Learning (ICML), 2019. 2

[25] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 6

## A Technical appendices and supplementary material

In this appendix, we provide further theoretical results and detailed proofs in §A. Additionally, we report supplementary empirical results and analyses in §B.

This appendix provides proofs and additional details for the theoretical results in the main text. We first establish the relation between relational distortion and agreement and prove the invariance properties of the rank representation. We then describe the full ordinal refinement used in the experiments. Next, we prove the exact-recovery and stability results and connect them to this refined representation and approximate solver outputs. Finally, we provide additional computational complexity details.

## A.1 Properties of the ordinal representation

We first prove the relation between the distortion formulation and the agreement score introduced in Section 2.1.

Proof of Lemma 1. For $\ell ( a , b ) = - a b$

$$
D _ { \ell } ( K _ { x } , K _ { y } ; \pi ) = - \sum _ { i \neq j } K _ { x } ( i , j ) K _ { y } ( \pi ( i ) , \pi ( j ) ) = - f ( \pi ) .
$$

Hence minimizing $D _ { \ell }$ is equivalent to maximizing $f .$

For $\ell ( a , b ) = ( a - b ) ^ { 2 }$

$$
D _ { \ell } ( K _ { x } , K _ { y } ; \pi ) = \sum _ { i \neq j } \left( K _ { x } ( i , j ) - K _ { y } ( \pi ( i ) , \pi ( j ) ) \right) ^ { 2 }\tag{22}
$$

$$
= \sum _ { i \neq j } K _ { x } ( i , j ) ^ { 2 } + \sum _ { i \neq j } K _ { y } ( \pi ( i ) , \pi ( j ) ) ^ { 2 }\tag{23}
$$

$$
- 2 \sum _ { i \neq j } K _ { x } ( i , j ) K _ { y } ( \pi ( i ) , \pi ( j ) ) .\tag{24}
$$

Because $( i , j ) \mapsto ( \pi ( i ) , \pi ( j ) )$ is a bijection over the ordered off-diagonal pairs,

$$
\sum _ { i \neq j } K _ { y } ( \pi ( i ) , \pi ( j ) ) ^ { 2 } = \sum _ { i \neq j } K _ { y } ( i , j ) ^ { 2 } .
$$

The first two terms are therefore independent of π, which proves the equivalence.

When the off-diagonal entries of both kernels have zero mean and unit variance, define

$$
\rho ( \pi ) = \frac { 1 } { N ( N - 1 ) } \sum _ { i \neq j } K _ { x } ( i , j ) K _ { y } ( \pi ( i ) , \pi ( j ) ) .
$$

Then

$$
f ( \pi ) = N ( N - 1 ) \rho ( \pi ) ,
$$

and

$$
D _ { \ell } ( K _ { x } , K _ { y } ; \pi ) = 2 N ( N - 1 ) ( 1 - \rho ( \pi ) ) .
$$

Thus maximizing the agreement score is equivalent to maximizing the correlation between the two relational structures.

Proof of Lemma 2. Let $\phi$ be strictly increasing. For any observed kernel entries,

$$
\phi ( K ( k , l ) ) \le \phi ( K ( i , j ) )
$$

if and only if

$$
K ( k , l ) \leq K ( i , j ) .
$$

Therefore

$$
F _ { \phi \circ K } ( \phi ( K ( i , j ) ) ) = F _ { K } ( K ( i , j ) ) ,
$$

which implies

$$
R ( \phi \circ K ) = R ( K ) .
$$

Now let $\sigma \in S _ { N }$ . Relabeling the classes induces a bijection on the unordered-pair set $\mathcal { E }$ and therefore leaves the multiset of off-diagonal kernel values unchanged. Hence

$$
R ( K \circ \sigma ) ( i , j ) = R ( K ) ( \sigma ( i ) , \sigma ( j ) ) ,
$$

and therefore

$$
R ( K \circ \sigma ) = R ( K ) \circ \sigma .
$$

The rank transformation also preserves strict pairwise order. For $e , e ^ { \prime } \in \mathcal { E }$

$$
K ( e ) < K ( e ^ { \prime } ) \quad \Longleftrightarrow \quad R ( K ) ( e ) < R ( K ) ( e ^ { \prime } ) .
$$

If all $\begin{array} { l } { \displaystyle {  { M } \ = \  { N } ( N \mathrm { ~ - ~ } 1 ) / 2 } } \end{array}$ unordered off-diagonal values are distinct, the rank values are $1 / M , 2 / M , \ldots , 1$ . If ties occur, equal values remain tied. Rank transformation therefore preserves the ordinal information already present in the kernel rather than creating additional relational distinctions.

The theoretical analysis in the main text is stated for the basic rank representation $R ( K )$ . In the experiments, we use a fixed ordinal refinement $T ( K ) = h ( R ( K ) )$ , which combines kernel-level ranks with anchor-wise neighbor order. For ranks computed over m candidate relations, we use the monotone reweighting

$$
g _ { m } ( u ) = - \log \left( 1 - u + \frac { 1 } { 2 m } \right) .\tag{25}
$$

Because $g _ { m }$ is strictly increasing, it changes the relative weighting of rank positions without changing their ordering. For each anchor i, define the within-row rank

$$
R _ { \mathrm { r o w } } ( K ) ( i , j ) = \frac { 1 } { N - 1 } \sum _ { l \neq i } \mathbf { 1 } \left[ R ( K ) ( i , l ) \leq R ( K ) ( i , j ) \right] , \qquad i \neq j ,\tag{26}
$$

with zero diagonal. Then let $\mathcal { C } ( A )$ denote standardization over the off-diagonal entries of a symmetric matrix $A ,$

$$
\mathcal { C } ( A ) ( i , j ) = \frac { A ( i , j ) - \mu _ { A } } { s _ { A } } , \qquad i \neq j ,\tag{27}
$$

with zero diagonal, where $\mu _ { A }$ and $s _ { A }$ are the mean and standard deviation of the off-diagonal entries. We define

$$
G ( K ) = { \mathcal { C } } { \big ( } g _ { M } ( R ( K ) ) { \big ) } ,\tag{28}
$$

and

$$
L ( K ) = { \mathcal { C } } ( \operatorname { s y m } g _ { N - 1 } ( R _ { \operatorname { r o w } } ( K ) ) ) ,\tag{29}
$$

where sym $\textstyle ( A ) = { \frac { A + A ^ { \top } } { 2 } }$

The RankAlign representation is then $T ( K ) = { \mathcal { C } } ( G ( K ) + L ( K ) )$

The two components are standardized before aggregation so that their relative contribution is not determined solely by numerical scale. Importantly, all operations in this refinement depend only on ordinal information. Consequently, $\bar { T ( \phi \circ K ) } = \bar { T ( K ) }$ for every strictly increasing $\phi ,$ and $T ( K \circ \sigma ) = T ( K ) \circ \sigma$ for every permutation $\sigma \in S _ { N }$ Thus, the refined representation inherits the monotone invariance and permutation equivariance of the basic rank representation without reintroducing the original similarity magnitudes.

## A.2 Recovery guarantees and approximate optimization

We first prove the exact-recovery result for the basic rank objective in Section 2.4.

Proof of Proposition 1. Let $X = R ( K _ { x } )$ , Under Assumption 1, the two kernels induce the same strict ordering under $\pi ^ { \star }$ . Hence

$$
R ( K _ { y } ) \circ \pi ^ { \star } = X .
$$

For an arbitrary candidate permutation π, define

$$
{ \boldsymbol { \tau } } = ( { \boldsymbol { \pi } } ^ { \star } ) ^ { - 1 } \circ { \boldsymbol { \pi } } .
$$

Then

$$
R ( K _ { y } ) \circ \pi = X \circ \tau .
$$

Therefore,

$$
f _ { \mathrm { o r d } } ( \pi ^ { \star } ) - f _ { \mathrm { o r d } } ( \pi ) = \| X \| _ { F } ^ { 2 } - \langle X , X \circ \tau \rangle\tag{30}
$$

$$
= { \frac { 1 } { 2 } } \| X - X \circ \tau \| _ { F } ^ { 2 } .\tag{31}
$$

The right-hand side is non-negative for every π, so $\pi ^ { \star }$ is a maximizer.

We next show uniqueness and the explicit margin. The permutation τ induces a permutation of the unordered-pair set E. If every unordered pair were fixed, then for any i and any two distinct $j , k \neq i$

$$
\tau ( i ) \in \{ i , j \} \cap \{ i , k \} = \{ i \} ,
$$

so $\tau = \mathrm { i d }$ . Thus every non-identity class permutation with $N \geq 3$ changes at least one non-trivial cycle of unordered pairs.

Under Assumption 1, the unordered rank values are $1 / M , \ldots , 1$ and are pairwise distinct. A nontrivial cycle contains at least two unordered pairs, and each changed rank differs by at least $1 / M$ Since every unordered pair contributes twice to the ordered off-diagonal objective,

$$
\| X - X \circ \tau \| _ { F } ^ { 2 } \geq \frac { 4 } { M ^ { 2 } } .
$$

Hence

$$
f _ { \mathrm { o r d } } ( \pi ^ { \star } ) - f _ { \mathrm { o r d } } ( \pi ) \geq \frac { 2 } { M ^ { 2 } } .
$$

Therefore $\pi ^ { \star }$ is the unique maximizer. If the objective is instead written over unordered pairs only, each relation appears once, giving the corresponding margin $1 / M ^ { 2 }$

The exact-recovery result above is stated for the basic rank representation. Under Assumption 1, the two modalities differ only by a strictly increasing reparameterization under the ground-truth correspondence. Since the refined representation $\bar { T ( K ) }$ is monotone invariant and permutation equivariant,

$$
T ( K _ { y } ) \circ \pi ^ { \star } = T ( K _ { x } ) .\tag{32}
$$

Hence $\pi ^ { \star }$ also maximizes the practical RankAlign objective. It is unique whenever the transformed representation has no non-trivial permutation symmetry. We now prove the stability result in the representation-independent form used in the main text. Let

$$
Z _ { x } = Z ( K _ { x } ) , \qquad Z _ { y } = Z ( K _ { y } ) ,
$$

and define $E _ { Z } = Z _ { y } \circ \pi ^ { \star } - Z _ { x }$ . For τ = (π<sup>⋆</sup>)<sup>−1</sup> ◦ π, let d<sub>Z</sub>(τ ) = ∥Z<sub>x</sub> − Z<sub>x</sub> ◦ τ∥<sub>F</sub>.

Proof of Proposition 2. Since $Z _ { y } \circ \pi ^ { \star } = Z _ { x } + E _ { Z }$ , we have $Z _ { y } \circ \pi = ( Z _ { x } + E _ { Z } ) \circ \tau$ . Therefore,

$$
f _ { Z } ( \pi ^ { \star } ) - f _ { Z } ( \pi ) = \langle Z _ { x } , Z _ { x } + E _ { Z } \rangle - \langle Z _ { x } , ( Z _ { x } + E _ { Z } ) \circ \tau \rangle\tag{33}
$$

$$
= \| Z _ { x } \| _ { F } ^ { 2 } - \langle Z _ { x } , Z _ { x } \circ \tau \rangle + \langle Z _ { x } - Z _ { x } \circ \tau ^ { - 1 } , E _ { Z } \rangle .\tag{34}
$$

The first two terms satisfy

$$
\| Z _ { x } \| _ { F } ^ { 2 } - \langle Z _ { x } , Z _ { x } \circ \tau \rangle = \frac { 1 } { 2 } d _ { Z } ( \tau ) ^ { 2 } .
$$

By Cauchy–Schwarz,

$$
\langle Z _ { x } - Z _ { x } \circ \tau ^ { - 1 } , E _ { Z } \rangle \geq - d _ { Z } ( \tau ) \Vert E _ { Z } \Vert _ { F } .
$$

Hence

$$
f _ { Z } ( \pi ^ { \star } ) - f _ { Z } ( \pi ) \geq d _ { Z } ( \tau ) \left( \frac { 1 } { 2 } d _ { Z } ( \tau ) - \| E _ { Z } \| _ { F } \right) .
$$

If πˆ maximizes $f _ { Z }$ , then we have

$$
f _ { Z } ( \pi ^ { \star } ) - f _ { Z } ( \hat { \pi } ) \leq 0 ,
$$

and

$$
d _ { Z } ( \hat { \tau } ) \leq 2 \| E _ { Z } \| _ { F } , \qquad \hat { \tau } = ( \pi ^ { \star } ) ^ { - 1 } \circ \hat { \pi } .
$$

The stability result also yields a bound on the number of mismatched classes. Define

$$
\operatorname { e r r } ( \pi ) = | \{ i : \pi ( i ) \neq \pi ^ { \star } ( i ) \} |
$$

and

$$
\kappa ( Z _ { x } ) = \operatorname* { m i n } _ { \tau \neq \mathrm { i d } } \frac { \| Z _ { x } - Z _ { x } \circ \tau \| _ { F } ^ { 2 } } { | \{ i : \tau ( i ) \neq i \} | } .
$$

Then

$$
\kappa ( Z _ { x } ) \mathrm { e r r } ( \hat { \pi } ) \leq d _ { Z } ( \hat { \tau } ) ^ { 2 } ,
$$

which gives

$$
\mathrm { e r r } ( \hat { \pi } ) \leq \frac { 4 \| E _ { Z } \| _ { F } ^ { 2 } } { \kappa ( Z _ { x } ) } .\tag{35}
$$

For the standardized representation T, define

$$
\rho ^ { \star } = \frac { \langle T _ { x } , T _ { y } \circ \pi ^ { \star } \rangle } { N ( N - 1 ) } .
$$

Because the off-diagonal entries are standardized to zero mean and unit variance,

$$
\| E _ { T } \| _ { F } ^ { 2 } = 2 N ( N - 1 ) ( 1 - \rho ^ { \star } ) .
$$

Substituting this expression into Eq. 35 gives

$$
\mathrm { e r r } ( \hat { \pi } ) \leq \frac { 8 N ( N - 1 ) ( 1 - \rho ^ { \star } ) } { \kappa ( T _ { x } ) } .
$$

This separates cross-modal ordinal disagreement, measured by $\rho ^ { \star }$ , from the intrinsic distinguishability of the relational profiles, measured by $\kappa ( T _ { x } )$

Finally, the previous result extends to an approximate QAP solver. Let πˆ denote an arbitrary solver output and define its objective deficit relative to the ground truth as

$$
\eta = \operatorname* { m a x } \{ 0 , f _ { Z } ( \pi ^ { \star } ) - f _ { Z } ( \hat { \pi } ) \} .
$$

From the stability inequality,

$$
d _ { Z } ( \hat { \tau } ) \left( \frac { 1 } { 2 } d _ { Z } ( \hat { \tau } ) - \| E _ { Z } \| _ { F } \right) \leq \eta .
$$

Solving the resulting quadratic inequality gives

$$
d _ { Z } ( \hat { \tau } ) \leq \| E _ { Z } \| _ { F } + \sqrt { \| E _ { Z } \| _ { F } ^ { 2 } + 2 \eta } .
$$

Hence,

$$
\mathrm { e r r } ( \hat { \pi } ) \leq \frac { \left( \| E _ { Z } \| _ { F } + \sqrt { \| E _ { Z } \| _ { F } ^ { 2 } + 2 \eta } \right) ^ { 2 } } { \kappa ( Z _ { x } ) } .
$$

The analysis therefore separates three sources of difficulty. Cross-modal disagreement is captured by $\| E _ { Z } \| _ { F }$ , structural ambiguity by $\kappa ( Z _ { x } )$ , and optimization error by η.

## A.3 Computational complexity

We finally provide a detailed accounting of the computational cost of RankAlign. Let N denote the number of classes and d the embedding dimension. The number of unordered class pairs is

$$
M = { \frac { N ( N - 1 ) } { 2 } } .
$$

Prototype centering and normalization require $O ( N d )$ time. Constructing each dense similarity kernel requires $O ( N ^ { 2 } d )$ time and $O ( N ^ { 2 } )$ memory. The kernel-level rank transformation sorts $M = { \overline { { O ( N ^ { 2 } ) } } }$ values and requires $O ( M$ log ${ \bf { \dot { M } } } ) ^ { ' } = O ( N ^ { \bf \dot { 2 } }$ log N) time.

The anchor-wise transformation ranks N rows of length $N - 1$ and also requires $O ( N ^ { 2 } \log N )$ time. The score transformation, symmetrization, and standardization each operate over at most $O ( N ^ { 2 } )$ entries and therefore require $O ( N ^ { 2 } )$ time. Consequently, the complete practical RankAlign representation can be constructed in $O ( N ^ { 2 } d + N ^ { 2 } \log N )$ time and $O ( N ^ { 2 } + N d )$ memory, excluding feature extraction.

The $O ( N ^ { 2 } d )$ kernel construction is already required by conventional value-based relational matching. The additional asymptotic preprocessing overhead introduced by RankAlign is $O ( N ^ { 2 } \log N )$ time and $O ( N ^ { 2 } )$ memory. Let $\dot { C } _ { \cal S } ( \dot { N } )$ denote the running time of a chosen $\mathsf { Q A P }$ solver S. The complete pipeline requires $\dot { C _ { S } } ( N ) + \dot { O } ( \dot { N ^ { 2 } } d + N ^ { 2 } \log N )$ time.

The transformed representation remains an $N \times N$ matrix. RankAlign therefore leaves the input structure of the QAP unchanged and can be combined with the same solver used by value-based relational matching. Evaluating the complete matching objective for one permutation requires $O ( N ^ { 2 } )$ operations. If a local-search solver swaps two assignments, only terms involving the affected classes change, so the corresponding objective update involves $O ( N { \bar { ) } }$ pairwise terms. We use an iterated local search (ILS) procedure to obtain a strong initial solution before applying the selected QAP solver. The ILS optimizes the rank-based objective and its solution is used to initialize FAQ, entropic OT, or FHG. The same initialization procedure is applied to BlindMatch for a fair comparison.

## B Additional experiments

In this section, we provide supplementary experimental results supporting the main paper.

## B.1 Datasets

We evaluate RankAlign on diverse benchmarks spanning web-scale pseudo-label data, object recognition datasets at different semantic granularities, and a domain-specific medical imaging dataset.

CIFAR-100 and ImageNet-100. To study scalability, we evaluated the matching accuracy as a function of the number of classes N using subsets of CIFAR-100 [9] and ImageNet-100 [18]. We varied N across matching instances to assess performance under increasing task difficulty.

LAION pseudo-class dataset. We evaluated RankAlign on a pseudo-class dataset derived from the LAION-Aesthetic subset [20]. Since LAION lacks predefined class labels, we defined $N = 2 0$ visual categories and assign pseudo-labels via caption matching. To ensure label quality, we classified each image using both its original caption (TEXT) and a Moondream-generated auxiliary caption, retaining only samples with a unique, consistent top-scoring assignment. The dataset contains approximately 7,000 samples, with 200–500 images per class. Visual and language prototypes are constructed by averaging embeddings within each pseudo-class and its associated captions, respectively.

Medical imaging dataset. We evaluated RankAlign on CheXpert [7], a large-scale chest X-ray benchmark with 220,000 radiographs. We extracted a 12-class single-label subset by retaining samples with one positive finding, totaling about 62,000 images. This setting is challenging due to subtle inter-class differences, class imbalance, and the medical image-text domain gap.

## B.2 Kernel Transformation Ablation

To determine whether the gains of RankAlign can be explained by generic normalization or by a single form of ranking, we conduct a controlled ablation on CIFAR-100. All variants use identical visual and language prototypes, class subsets, similarity kernels, encoders, and permutation solvers. Only the relational representation supplied to the solver is changed.

We compare the raw kernel without transformation (none), z-score normalization, row-wise ranking, quantile normalization, and the full RankAlign representation. Z-score normalization removes global mean and scale differences while preserving the relative spacing of kernel values. Row-wise ranking preserves local neighbor order but does not define a single kernel-level ordinal ordering. Quantile normalization aligns marginal value distributions.

Table 5 shows that the improvement cannot be explained by normalization alone. Z-score normalization provides almost no gain over the raw value-based kernel, consistent with the fact that affine normalization does not alter the optimizer of the inner-product matching objective. Quantile normalization also yields limited improvements. Row-wise ranking performs substantially better than value-based normalization, confirming that local ordinal information is useful. However, its advantage decreases as the problem size grows. RankAlign consistently achieves the highest matching accuracy across all problem sizes, indicating that the combination of kernel-level ordinal normalization and anchor-wise neighborhood order is more effective than generic normalization or local ranking alone.

Table 5: Kernel transformation ablation on CIFAR-100. All variants use the same prototypes, class subsets, encoders, kernels, and permutation solver, only the kernel transformation differs.
<table><tr><td>Transformation</td><td> $N = 3 0$ </td><td> $N = 5 0$ </td><td> $N = 7 0$ </td><td> $N = 1 0 0$ </td></tr><tr><td>None</td><td>0.03</td><td>0.01</td><td>0.00</td><td>0.00</td></tr><tr><td>Z-score</td><td>0.03</td><td>0.02</td><td>0.01</td><td>0.00</td></tr><tr><td>Row-wise ranking</td><td>0.32</td><td>0.14</td><td>0.09</td><td>0.06</td></tr><tr><td>Quantile normalization</td><td>0.06</td><td>0.07</td><td>0.04</td><td>0.00</td></tr><tr><td>RankAlign</td><td>1.00</td><td>0.98</td><td>0.91</td><td>0.79</td></tr></table>

## B.3 Comparison with Classical Structural Alignment Methods

To examine whether simpler classical alignment methods can recover the same cross-modal correspondence without RankAlign’s ordinal formulation, we compare against unsupervised Orthogonal Procrustes and standard entropic Gromov–Wasserstein optimal transport. All methods use the same class-level visual and language representations and are evaluated on the same CIFAR-100 matching instances, so the comparison isolates differences in the alignment objective rather than differences in the underlying encoders or prototypes.

Table 6: Comparison with classical structural alignment methods on CIFAR-100. All methods use the same class-level representations and matching instances. We report matching accuracy across problem sizes.
<table><tr><td>Model</td><td>Method</td><td> $N = 3 0$ </td><td> $N = 5 0$ </td><td> $N = 7 0$ </td><td> $N = 1 0 0$ </td></tr><tr><td rowspan="3">CLIP</td><td>Procrustes</td><td> $0 . 0 1 \pm 0 . 0 2$ </td><td> $0 . 0 3 \pm 0 . 0 1$ </td><td> $0 . 0 2 \pm 0 . 0 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Vanilla GW-OT</td><td> $0 . 0 3 \pm 0 . 0 3$ </td><td> $0 . 0 3 \pm 0 . 0 2$ </td><td> $0 . 0 3 \pm 0 . 0 3$ </td><td> $0 . 0 3 \pm 0 . 0 0$ </td></tr><tr><td>RankAlign</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 9 8 \pm 0 . 0 2$ </td><td> $0 . 9 1 \pm 0 . 0 4$ </td><td> $0 . 7 9 \pm 0 . 0 2$ </td></tr><tr><td rowspan="3">DeiT</td><td>Procrustes</td><td> $0 . 0 4 \pm 0 . 0 4$ </td><td> $0 . 0 2 \pm 0 . 0 1$ </td><td> $0 . 0 2 \pm 0 . 0 1$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td></tr><tr><td>Vanilla GW-OT</td><td> $0 . 0 7 \pm 0 . 0 4$ </td><td> $0 . 0 5 \pm 0 . 0 3$ </td><td> $0 . 0 7 \pm 0 . 0 5$ </td><td> $0 . 0 5 \pm 0 . 0 0$ </td></tr><tr><td>RankAlign</td><td> $0 . 9 3 \pm 0 . 0 1$ </td><td> $0 . 9 5 \pm 0 . 0 5$ </td><td> $0 . 7 2 \pm 0 . 0 5$ </td><td> $0 . 3 6 \pm 0 . 1 6$ </td></tr><tr><td rowspan="3">DINOv2</td><td>Procrustes</td><td> $0 . 0 5 \pm 0 . 0 4$ </td><td> $0 . 0 2 \pm 0 . 0 2$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td> $0 . 0 4 \pm 0 . 0 0$ </td></tr><tr><td>Vanilla GW-OT</td><td> $0 . 0 1 \pm 0 . 0 2$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>RankAlign</td><td> $0 . 9 5 \pm 0 . 0 6$ </td><td> $0 . 9 0 \pm 0 . 0 2$ </td><td> $0 . 9 0 \pm 0 . 0 4$ </td><td> $0 . 6 8 \pm 0 . 1 5$ </td></tr></table>

Table 6 shows that neither classical baseline approaches the performance of RankAlign across the evaluated backbones and problem sizes. Both Orthogonal Procrustes and vanilla GW-OT remain close to chance as N increases, whereas RankAlign retains substantially higher matching accuracy. Because all methods operate on the same class-level representations, this difference cannot be attributed to stronger encoders or higher-quality prototypes. These results indicate that a global linear alignment or direct matching of the original metric structures is insufficient to recover the cross-modal correspondence in these experiments, and RankAlign is not replacing a comparably accurate simpler baseline.

## B.4 Full larger-scale matching results

In the main paper, we report representative larger-scale matching results for a subset of vision backbones to improve readability. We provide the full results across all evaluated backbones, including CLIP, DINOv2, DeiT, SigLIP, and SigLIP2.

Figure 3 shows matching accuracy as a function of the problem size on CIFAR-100 and ImageNet-100. The overall trend is consistent with the representative results in the main text, RANKALIGN consistently outperforms BLINDMATCH with the GW objective across different backbone families, and its performance degrades more gradually as the number of matched classes increases. These results indicate that the improvement is not tied to a particular visual encoder, but holds across both vision-language and visual representation models.

![](images/05ed086963f96a03a9d6bf516e174ccfac27368137442eeb1173175ff03de6ba.jpg)  
Figure 3: Full larger-scale matching results on CIFAR-100 and ImageNet-100 across all evaluated vision backbones.Each color corresponds to a vision backbone. The darker solid line denotes RANKALIGN, while the lighter dashed line denotes BLINDMATCH with the GW.

## B.5 Existing Assets and Licenses

This work uses existing public datasets and pretrained models, including LAION-Aesthetic, CIFAR-100, ImageNet-100, CheXpert, DINO/DINOv2, DeiT, CLIP, SigLIP/SigLIP2, and SentenceTransformers. We cite the original sources for these assets and use them only for research evaluation. We will follow the corresponding licenses and terms of use for each dataset and pretrained model, and will include detailed license information in the released code and supplemental documentation.