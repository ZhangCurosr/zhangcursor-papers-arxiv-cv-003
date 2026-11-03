# QFacePlacement: From Arbitrary Tensorization to Semantic Spatial Correspondence in Modern Face Verification

Krishna Bhatia<sup>1</sup>, Kavita Choudhary<sup>2</sup>, Pruthwik Mishra<sup>2</sup>, and Praveen Kumar Chandaliya\*<sup>2</sup>

<sup>1</sup> QuantumAI Lab, Fractal Analytics, Mumbai, India

krishna.bhatia@fractal.ai

2 Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology, Surat, India

ds25ai002@aid.svnit.ac.in,pruthwikmishra@aid.svnit.ac.in,pkc@aid.svnit.ac.in\*

Abstract. Quantum-inspired computer vision often begins by reshaping a high-dimensional representation into a tensor and then attaching a tensor-network model to it. In face verification this can be misleading: modern embeddings are already optimized for a simple cosine geometry, and a formal tensorization does not guarantee the low-entanglement structure required by matrix-product operators or tensor trains. We study placement through two controlled interfaces that use the same modern face-recognition setting but expose very diferent mathematical objects. First, we examine cross-model embedding compatibility, in which a 512×512 alignment operator must map AdaFace embeddings into the ArcFace space. Dense Ridge and Procrustes maps remain strong, whereas deterministic TT-SVD/MPO compression fails even after basis transformations and low-rank residual hybrids: at the final 64K storage budget, dense Ridge averages 0.9297 directional AUC, rank-64 SVD 0.8952, the best low-rank-MPO hybrid 0.8685, and raw MPO approximately 0.649. Second, we move before global pooling and construct a genuine 7×7×7×7 probe–gallery correspondence tensor from local AdaFace descriptors. Here a geometry-aware symmetric TT with bond dimension 2 uses only 122 trainable parameters versus 2403 for a dense linear matcher and preserves LFW AUC (0.9802), showing that semantically meaningful tensor structure enables strong compression. However, it does not improve the stringent biometric tail: on CFP-FP, TAR at $\mathrm { F A R } { = } 1 0 ^ { - 3 }$ falls to 0.0102 versus 0.1734 for global cosine, and on CPLFW it reaches 0.7805 versus 0.8737. The resulting principle is sharper than “tensor networks help” or “tensor networks fail”: quantum-inspired factorization is credible only when the chosen interface exposes task-aligned axes and compressible structure, and aggregate AUC must be checked against operational low-FAR behaviour. This placement-oriented analysis provides a reusable design doctrine for quantum computer vision and biometric matching.

Keywords: Quantum computer vision · face verification · tensor networks · biometrics · representation structure · quantum-inspired learning

## 1 Introduction

Modern face verification is a dificult test bed for adding a new learning primitive. FaceNet [15], CosFace [21], ArcFace [2], and AdaFace [6] deliberately organize identity representations so that normalized cosine similarity already provides an efective verifier. Consequently, additional model capacity is useful only if the new module sees information or structure that the pooled embedding does not already expose. This observation is especially important for quantum and quantum-inspired computer vision. Quantum kernels and variational circuits require aggressive state preparation or dimensional compression, while tensornetwork (TN) models require a useful factorization of the object on which they operate. A vector of length 512 can always be reshaped because 512 = 2<sup>9</sup>, but algebraic reshaping alone does not imply that the resulting nine-site object has low operator entanglement or short-range dependencies. Conversely, a local correspondence tensor whose indices explicitly denote probe and gallery coordinates has a semantic product structure before any model is introduced.

Quantum face-recognition work spans multigate QCNNs [24], quantumenhanced eigenface/PCA formulations [19, 18], quantum optical acquisition [13], and quantum-inspired attention [3]. Our earlier near-term study likewise found that compact quantum and tensorized modules can be useful under restricted data or resource budgets while strong classical face representations remain dificult to displace [1]. These results motivate a question that precedes architecture choice: which interface actually has the structure a quantum-inspired factorization needs?.

We answer this question through two experiments that deliberately contrast arbitrary and semantic tensorization (Fig. 1). The first studies cross-model compatibility, where the object is a dense 512×512 linear operator between independently trained face-embedding spaces. The second moves before global pooling and operates on the full spatial similarity tensor between two 7 × 7 local descriptor grids. Both studies retain strong dense and low-rank classical controls and use external biometric protocols rather than random image splits.

Our main contributions are:

– We formulate semantic structure at the insertion point as the experimental variable, rather than selecting a Tensor Network (TN) architecture first and searching for an input shape.

– We show that a dense face-compatibility operator remains poorly represented by low-bond MPO/TT structure even under deterministic TT-SVD, basis transformations, and low-rank residual hybrids. A storage-matched truncated SVD is consistently stronger, indicating ordinary matrix-rank compressibility rather than TN locality.

– We show that a true local correspondence tensor is a substantially better structural fit: a relative-coordinate symmetric TT with only 122 parameters reaches 0.9802 LFW AUC and competitive stress-set AUC while providing roughly 20× compression relative to dense matching.

![](images/535465d889b4fa02ed1b3bd0959d3afff4a8a4e810e1b497f7f0286b5bb5c6f3.jpg)  
Fig. 1. QFacePlacement contrasts two interfaces. Compatibility alignment tensorizes an already-global embedding operator; local correspondence acts before global pooling on axes with explicit probe/gallery geometry. The same TN intuition is therefore tested on mathematically diferent objects.

We show that parameter eficiency and AUC are not suficient biometric claims. On the two pose-stress protocols, the learned TT reshapes the impostor tail unfavorably and loses substantial TAR at FAR=10<sup>−3</sup>. – We derive a placement doctrine for quantum computer vision: identify semantically factorized interfaces, measure whether the target operator is actually compressible under the proposed structure, compare against simple low-rank controls, and evaluate operational tails before attributing value to a quantuminspired model.

## 2 Related Work

## 2.1 Modern Face Verification and Local Evidence

Face recognition has progressively moved from metric-learning objectives to highly discriminative hyperspherical embeddings. FaceNet established triplet supervision [15]; ArcFace and CosFace improved angular separation [2, 21]; AdaFace adapts margins to recognizability [6]; and MagFace links embedding magnitude with face quality [8]. These approaches make it dificult to improve a global cosine score indiscriminately. Nevertheless, global pooling is not guaranteed to retain every cue useful under pose, occlusion, or domain shift. DeepFace-EMD, for example, operates on local descriptors and uses patch-wise optimal transport for out-of-distribution matching [12]. This distinction motivates our second placement: if local descriptors still retain spatially organized identity evidence, a structured matcher should be tested before that information is pooled away.

## 2.2 Quantum and Quantum-Inspired Vision

Quantum feature maps construct learned or fixed representations in Hilbert space [16, 4], while data re-uploading can increase the expressive power of shallow parameterized circuits [11]. However, near-term quantum models remain constrained by state preparation, limited qubit counts, measurement overhead, and trainability issues such as barren plateaus [7]. These constraints are particularly relevant to face recognition, where strong pretrained embeddings commonly contain hundreds of dimensions.

Tensor networks (TNs) provide a complementary quantum-inspired approach by representing high-order tensors through structured low-rank factorizations. Tensor Trains (TTs) and Matrix Product States (MPS) ofer compact representations of high-dimensional arrays [10], while tensorized neural networks can substantially reduce the parameterization of dense layers [9]. MPS-based models have also demonstrated supervised learning directly on product-structured feature representations [20]. Crucially, however, the efectiveness of TN compression depends on whether the target representation exhibits correlations that can be captured at a manageable bond dimension. We therefore treat this structural assumption as an empirical hypothesis and explicitly test whether the location and semantics of tensorization determine the efectiveness of TN representations for biometric matching.

## 3 Why Placement Is a Structural Question

The central motivation of this work is that the efectiveness of a Tensor Network (TN) representation depends not only on the chosen architecture, but also on where and how tensorization is introduced. A compact representation can arise for fundamentally diferent reasons. First, the underlying operator may possess low conventional matrix rank, in which case truncated SVD provides a natural and often highly competitive compression mechanism. Second, after a meaningful tensorization, the operator may exhibit low Schmidt rank across successive tensor bipartitions, making an MPO/TT representation structurally appropriate. These two notions of compressibility are distinct: low matrix rank does not imply TN locality, and a compact TN representation should not be credited with an advantage that is already explained by ordinary low-rank structure.

This distinction motivates our experimental design. Rather than selecting a TN architecture first and adapting the data to its required tensor shape, we treat tensorization placement and semantic structure as the primary experimental variables. We examine both a dense cross-model compatibility operator and a genuinely local correspondence tensor, thereby testing whether TN compression is favored by the structure of the object being represented. For each structured representation, we compare against dense, truncated-SVD, CP/Tucker, and parameter-free controls under appropriate storage and evaluation constraints. This design isolates whether observed gains arise from genuine TN-compatible structure or merely from generic low-rank compression, and establishes placement as a structural question rather than an architectural convenience.

Table 1. Cross-model compatibility at N = 5000 and the final 64k storage budget. Values are mean directional verification AUC
<table><tr><td>Adapter</td><td>Mean AUC ↑</td><td>Interpretation</td></tr><tr><td>Dense Ridge</td><td>0.9297</td><td>Reference map</td></tr><tr><td>Rank-64 SVD</td><td>0.8952</td><td>Strongest compact control</td></tr><tr><td>Low-rank + MPO hybrid</td><td>0.8685</td><td>Best TN salvage</td></tr><tr><td>Raw MPO / TT-SVD</td><td>~0.649</td><td>Severe mismatch</td></tr></table>

## 4 Placement I: Cross-Model Embedding Compatibility

## 4.1 Problem Definition

Model upgrades create a practical compatibility problem: a gallery encoded by a legacy face model may remain deployed while new probes are produced by a newer model. We use AdaFace IR101 trained on WebFace12M as the source representation and ArcFace R50 as the target. Given same-image source–target pairs $( x _ { i } , y _ { i } ) \in \mathbb { R } ^ { 5 1 2 } \times \mathbb { R } ^ { 5 1 2 }$ , the goal is to learn $f ( x ) \approx y$ such that probe-togallery verification remains meaningful across model spaces. Dense baselines include linear least squares, Ridge regression, and orthogonal Procrustes [14]. The structured family includes butterfly orthogonal maps, MPO/TT adapters, local unitary TNs, and spectral-unitary TNs. The decisive salvage experiment starts from the fitted dense Ridge operator and applies deterministic TT-SVD, eliminating gradient optimization as a confound. We additionally test signed permutations, Walsh–Hadamard basis changes, and low-rank-plus-MPO hybrids.

## 4.2 Storage-Matched Evaluation

We adopt a deliberately stringent, pre-registered success criterion: a structured candidate must (i) remain within 0.01 mean directional AUC of dense Ridge, (ii) remain within 0.002 AUC of a storage-matched truncated-SVD baseline, and (iii) require no more than 25% of the storage used by the dense operator. Under this criterion, a TN representation is considered successful only if it captures structure that cannot be explained by conventional low-rank compression at a comparable storage budget. The results in Table 1 and Fig. 2 show that the dense cross-model compatibility operator does not satisfy this requirement. Dense Ridge achieves a mean directional AUC of 0.9297, while rank-64 SVD retains 0.8952. In contrast, the strongest low-rank–MPO hybrid reaches only 0.8685, and the raw MPO performs substantially worse, with an AUC of approximately 0.649. The same ordering is observed under the earlier symmetric evaluation: Procrustes achieves AUCs of 0.980/0.828/0.965/0.979 on LFW/CFP/CASIA/VGGFace2, closely matched by Ridge at 0.980/0.827/0.965/0.978. By comparison, an MPO with bond dimension 8 achieves only 0.737/0.569/0.729/0.719, while the compact spectral TN obtains 0.668/0.585/0.646/0.632.

![](images/5ece2fc771e1efc9c9e431c969d16448db152f68ccde163cf76238c980d9bf06.jpg)  
Fig. 2. Final compatibility salvage. Conventional truncated SVD retains substantially more verification structure than the tested MPO/TT factorization at the same broad storage scale.

The structural diagnostics are consistent with this performance gap. Operator– Schmidt entropy remains high across the important tensor bipartitions, providing no evidence that the dense alignment operator admits a strongly localized lowbond representation in the tested factorizations. Importantly, this does not imply that the operator is incompressible: the SVD result demonstrates meaningful conventional matrix-rank compressibility. Rather, the evidence indicates that its useful compressibility is predominantly global matrix-rank structure, not the successive low-Schmidt-rank structure required for an eficient MPO/TT factorization. Thus, for this dense compatibility operator, tensor-network compression introduces an unfavorable structural constraint rather than exposing additional compressibility beyond what a conventional low-rank approximation already provides.

## 5 Placement II: Pre-Pooling Local Correspondence

## 5.1 Semantically Factorized Tensor

We next move the tensorization point before global pooling, where the spatial structure of the two images is still explicit. A frozen AdaFace IR101 backbone produces a $7 \times 7$ spatial grid of local 512-dimensional descriptors. Let $F _ { A } , F _ { B } \in$ R<sup>7×7×512</sup> denote the feature grids for probe image A and gallery image $B ,$ respectively. After $\ell _ { 2 }$ normalization of each local descriptor, we construct the

pairwise local-similarity tensor

$$
C _ { y _ { A } , x _ { A } , y _ { B } , x _ { B } } = \left. \widehat { F } _ { A } [ y _ { A } , x _ { A } ] , \widehat { F } _ { B } [ y _ { B } , x _ { B } ] \right. , \qquad C \in \mathbb { R } ^ { 7 \times 7 \times 7 \times 7 } ,\tag{1}
$$

where $y _ { A } , x _ { A } , y _ { B } , x _ { B } \in \left\{ 1 , \ldots , 7 \right\}$ . Thus, $C$ contains $7 ^ { 4 } = 2 4 0 1$ local correspondence scores, with its four modes explicitly encoding the probe row, probe column, gallery row, and gallery column. This construction difers fundamentally from reshaping a pooled 512-dimensional embedding: the tensor modes retain the spatial semantics of the two images and therefore expose a natural factorization over local correspondences.

The initial Born-rule TT classifier was consistently weaker than the corresponding linear models; we therefore use the linear TT/MPS family as the primary structured model and retain Born-rule variants as controls. We evaluate three tensor orderings:

$$
\pi _ { \mathrm { g r o u p } } = ( y _ { A } , x _ { A } , y _ { B } , x _ { B } )\tag{2}
$$

$$
\pi _ { \mathrm { i n t e r } } = ( y _ { A } , y _ { B } , x _ { A } , x _ { B } )\tag{3}
$$

$$
\pi _ { \mathrm { r e l } } = ( y _ { A } , x _ { A } , \Delta y , \Delta x )\tag{4}
$$

where $\varDelta y$ and $\varDelta x$ encode relative probe–gallery spatial ofsets. The grouped ordering preserves the within-image spatial coordinates, the interleaved ordering places corresponding probe and gallery coordinates adjacent, and the relative ordering explicitly represents translation-invariant spatial relationships. Primary TT/MPS models are constrained to satisfy pair symmetry, $s ( A , B ) = s ( B , A )$ so that the learned similarity is invariant to exchanging probe and gallery images. CP and Tucker factorizations are included as additional controls to determine whether any observed advantage arises from generic low-rank tensor approximation rather than from the sequential factorization imposed by TT/MPS.

## 5.2 Development Protocol

CASIA-WebFace [22] is used exclusively for model fitting and model selection. For each of five random seeds, we use 400 training identities and 100 identity-disjoint validation identities, with at most 20 images per identity. This yields 40,000 training pairs and 20,000 validation pairs per seed. All design choices, including tensor ordering, bond dimension, and model family selection, are determined solely by this development protocol.

LFW [5], CFP-FP [17], and CPLFW [23] are reserved as external biometric evaluation protocols and are never used for model or tensor-structure selection. We report AUC, EER, fold-calibrated accuracy, and TAR at $\mathrm { F A R } \in \{ 1 0 ^ { - 2 } , 1 0 ^ { - 3 } \}$ As a protocol integrity check, each evaluation fold must contain both genuine and impostor pairs. This check detected a mirror-specific CPLFW pair-ordering error during development and prevents class-pure folds from silently yielding undefined (NaN) metrics.

![](images/fc85b43da5431d323b25a474476220fbee2633482596bf59428b282bb496deeb.jpg)  
Fig. 3. External AUC for the pre-pooling correspondence study. Local evidence is especially useful on CFP-FP, but the parameter-free same-position matcher remains strongest there.

## 5.3 Parameter Eficiency and External AUC

The model selected exclusively on the CASIA validation set is the relativecoordinate, pair-symmetric TT with bond dimension $\chi = 2 .$ . It contains only 122 trainable parameters, corresponding to 5.1% of the 2,403 parameters in the dense linear correspondence head. As shown in Fig. 3 and Table 2, tensorization at the semantically structured local-correspondence interface is substantially more favorable than tensorization of the global compatibility operator. On LFW, the selected TT achieves an AUC of 0.980201, marginally exceeding the global cosine baseline. On CFP-FP, it reaches 0.800044, outperforming both global cosine and the dense linear correspondence head, making it the strongest learned correspondence-based matcher among the evaluated methods. Nevertheless, the parameter-free same-position local score achieves a higher AUC of 0.822771, demonstrating that part of the performance gain arises from the underlying local spatial correspondence itself rather than from the learned TT parameterization. On CPLFW, the global cosine baseline remains the strongest method, indicating that the advantage of semantic tensorization is protocol-dependent and does not uniformly translate across pose-variation conditions.

## 5.4 Low-FAR Performance Reveals Operational Limitations

The most consequential limitation emerges at the operational biometric threshold rather than in aggregate AUC. On CFP-FP, the TT achieves a TAR of only 0.010229 at $\mathrm { F A R } = 1 0 ^ { - 3 }$ , compared with 0.173429 for global cosine; on CPLFW, the corresponding TARs are 0.780533 and 0.873667, respectively. Thus, as shown in Fig. 4, an AUC-only assessment would give an overly favorable impression of the TT model. Although the relative-coordinate TT provides an efective and highly parameter-eficient compression of the local correspondence tensor and ofers a plausible geometry-aware representation, its unfavourable impostor-tail behaviour prevents it from being considered an operationally superior pose-robust verifier. These results emphasize that biometric model evaluation must include low-FAR operating points, where tail behaviour can difer substantially from aggregate ranking metrics such as AUC.

Table 2. External AUC and TAR at $\mathrm { F A R } { = } 1 0 ^ { - 3 }$ for key correspondence methods.
<table><tr><td>Dataset Method</td><td>AUC ↑</td><td> $\mathbf { T A R @ 1 0 ^ { - 3 } } \uparrow$ </td></tr><tr><td rowspan="3">CFP-FP Dense linear</td><td>Global cosine 0.796314</td><td>0.173429</td></tr><tr><td>Same-position local 0.822771 Relative TT</td><td>0.155714</td></tr><tr><td> $\chi = 2$  0.800044 0.794068</td><td>0.010229 0.038629</td></tr><tr><td rowspan="3"></td><td>Global cosine 0.966074</td><td>0.873667</td></tr><tr><td>CPLFW Same-position local 0.960581</td><td>0.743667</td></tr><tr><td>Relative TT  $\chi = 2$  0.961140</td><td>0.780533</td></tr><tr><td rowspan="3">LFW</td><td>Global cosine 0.979789</td><td>0.950667</td></tr><tr><td>Same-position local 0.979403</td><td>0.951333</td></tr><tr><td>Relative TT  $\chi = 2$  0.980201</td><td>0.951000</td></tr></table>

## 5.5 Structural Diagnostics and Failure Attribution

Our experiments are designed so that each negative result has a diagnostic interpretation rather than merely a lower score. Table 3 summarizes the structural assumptions and the evidence used to accept or reject them. In compatibility alignment, the decisive observation is not simply that MPO is worse than Ridge. The rank-64 SVD control proves that the operator contains compressible structure, while the much larger loss of MPO at a comparable storage scale indicates that the missing property is tensor locality. The operator-Schmidt entropy analysis then supplies a mechanism: important bipartitions remain close to maximally entangled under the tested site orderings and basis changes.

In local correspondence, the selected relative-coordinate TT gives the opposite structural signal. It is chosen independently on the CASIA validation set, uses only 122 parameters, and slightly exceeds dense linear matching on the two external AUC settings where both are available. This shows that a low-bond factorization can exploit the spatial object. Yet the same model fails badly at CFP-FP low FAR, so structural compressibility and biometric calibration are separate properties. A method can have the right inductive bias for average ranking while still reshaping the score tails in a harmful way.

![](images/55d69568067c1411847f97f3e97bcd1d26b46403989df0df2f254827c0d1a40b.jpg)  
Fig. 4. Operational TAR at FAR= $: 1 0 ^ { - 3 }$ . The TT preserves LFW but loses a substantial acceptance rate on the two pose-stress protocols.

## 6 Cross-Placement Analysis

## 6.1 Arbitrary Versus Semantic Tensorization

The two experiments difer primarily in the structure exposed to the same TN family. Compatibility alignment is a global operator whose tensor ordering is imposed after representation learning; its high operator Schmidt entropy indicates substantial cross-site dependence that low-bond TT/MPO factorizations cannot eficiently preserve. In contrast, local correspondence is naturally a fourindex geometric tensor, and relative coordinates align the factorization with spatial displacement structure. This structural diference explains why semantic tensorization enables strong compression whereas arbitrary tensorization does not. However, structural fit is not suficient for biometric superiority: the parameterfree local matcher still achieves higher CFP-FP AUC, and the learned TT degrades low-FAR performance. Thus, semantic tensorization is a prerequisite for efective TN compression, not a guarantee of operational robustness.

## 6.2 Matched Controls as a Placement Test

A useful quantum-inspired claim must survive simpler explanations. In compatibility alignment, truncated SVD investigate whether the ordinary matrix rank already explains the available compression. In local correspondence, CP/Tucker and dense linear heads investigate whether generic low rank or parameter count explains the result, while same-position, nearest-patch, and optimal-transport scores investigate whether learning is necessary at all. The resulting hierarchy prevents compactness from being mistaken for superiority.

Table 3. Structural assumptions and the controls used to test them.
<table><tr><td>Interface</td><td>Hypothesis</td><td>Diagnostic evidence</td></tr><tr><td>Compatibility map</td><td>A 512×512 map is low-SVD compresses moderately, but bond after tensorization TT/MPO remains much worse;</td><td>operator-Schmidt entropies stay</td></tr><tr><td>dence</td><td>Local correspon-Spatial probe/gallery axes Relative χ = 2 TT is selected support low-bond matching on disjoint validation and gives</td><td>high across tested bases. about 20× matcher compression</td></tr><tr><td></td><td></td><td>with competitive AUC. Operational veri- Better AUC implies a Rejected: CFP-FP TT AUC is</td></tr><tr><td>fication</td><td>better biometric operating competitive while</td><td> $\mathrm { T A R @ 1 0 ^ { - 3 } }$  col- lapses.</td></tr></table>

## 6.3 Why Biometric Tails Matter for Quantum Vision

Face verification difers from balanced image classification because it is often deployed at very low false-accept rates. A method can improve average ranking while moving a small number of impostors into the high-score tail. The P3 result demonstrates this directly. This issue is likely to be even more important for finite-shot or noisy quantum models, where measurement variation can perturb score tails before AUC changes appreciably. Quantum biometric work should therefore report TAR@FAR and calibration alongside AUC or accuracy.

## 6.4 Implications Beyond Face Recognition

The placement principle extends to other quantum-CV pipelines built on pretrained vision features. CNN feature maps, ViT token sequences, optical-flow fields, and patch-correlation volumes have diferent natural axes and factorization structures; pooled embeddings often do not. Before introducing an MPS, MPO, or qubit register, one should therefore specify the semantic meaning of each site and justify why the chosen ordering should yield compressible correlations.

This also calls for stronger evaluation of “quantum-inspired compression.” Parameter count alone is insuficient, as SVD, CP/Tucker, separable layers, or low-rank adapters may provide comparable compression. A meaningful quantuminspired claim requires that the proposed product structure matches the task and preserves performance at a favorable resource budget against these simpler controls. Our two experiments demonstrate both outcomes within the same biometric setting.

## 7 Placement Doctrine for Quantum Computer Vision

Our experiments suggest four practical principles: i)Start from the interface, not the circuit: Identify the available information and its natural factorization before selecting a quantum or TN primitive. ii) Test structural assumptions directly: For an MPO, compare against SVD and examine operator entanglement; for a local tensor, compare against parameter free local scores and alternative low-rank factorizations. iii) Separate compression from task performance: A 20× smaller matcher is a meaningful compression result even without an accuracy gain; the claim should remain appropriately limited to compression. iv) Evaluate operational performance: External identity-disjoint protocols and low FAR TAR, since these can reveal failures hidden by aggregate AUC.

These principles extend beyond face recognition to quantum-CV pipelines based on CNN features, token sequences, and attention maps. The key question is not whether a representation can be reshaped into a tensor, but whether its factorization reflects the task’s semantics and correlation structure.

## 8 Limitations and Reproducibility

Both experiments are quantum-inspired classical studies and do not involve execution on quantum hardware. The compatibility experiment evaluates TN/MPO representations of a cross-model face-space adapter, while the local experiment evaluates TT/MPS matchers over spatial correspondence tensors. Accordingly, we make no claims regarding quantum speedup, quantum advantage, or hardware eficiency. The two studies also use diferent face representations and are not intended as a single jointly optimized end-to-end system. Their purpose is instead to provide a controlled comparison of alternative tensorization placements. The local study provides the stronger statistical evaluation, using five CASIA identity disjoint splits, external LFW/CFP-FP/CPLFW protocols, fold-calibrated metrics, paired sign-flip tests, and hierarchical bootstrap comparisons. Future work should extend the placement doctrine to template-based IJB protocols, face-quality and uncertainty modeling, and trainable quantum encodings, while maintaining independent evaluation to avoid turning previously observed results into new placement claims.

## 9 Conclusion

QFacePlacement investigates when quantum-inspired factorization is structurally justified for modern face verification. Tensorizing a global 512-D compatibility operator is straightforward but structurally mismatched: deterministic MPO/TT compression remains substantially weaker than dense Ridge and storage-matched SVD. In contrast, moving the tensorization point before global pooling exposes a genuine probe–gallery correspondence tensor. A relative-coordinate, pair-symmetric TT then achieves approximately 20× parameter compression while preserving LFW performance and maintaining competitive aggregate AUC. However, low-FAR evaluation shows that this compression does not translate into operational superiority under cross-pose verification.

The central lesson is therefore placement before architecture. Quantum-inspired models are most credible when their factorization follows task semantics, demonstrates compression beyond simple low-rank controls, and preserves performance at operationally relevant thresholds. The positive and negative results provide a practical boundary for structured quantum computer vision: rather than tensorizing already-optimized representations arbitrarily, identify interfaces where the task itself exposes a meaningful and compressible factorization.

## References

1. Bhatia, K., Chandaliya, P.K., Devendrababu, S.: Evaluating near-term quantum machine learning for face recognition. In: AI for Forensics and Biometric Applications. Springer (2026)

2. Deng, J., Guo, J., Xue, N., Zafeiriou, S.: Arcface: Additive angular margin loss for deep face recognition. In: CVPR (2019)

3. Gauba, A.: Quantum-inspired attention for eficient face recognition. In: Quantum AI and NLP, CCIS, vol. 2659, pp. 58–66. Springer (2026)

4. Havlíček, V., Córcoles, A.D., Temme, K., Harrow, A.W., Kandala, A., Chow, J.M., Gambetta, J.M.: Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019)

5. Huang, G.B., Ramesh, M., Berg, T., Learned-Miller, E.: Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Tech. rep., University of Massachusetts Amherst (2007)

6. Kim, M., Jain, A.K., Liu, X.: Adaface: Quality adaptive margin for face recognition. In: CVPR (2022)

7. McClean, J.R., Boixo, S., Smelyanskiy, V.N., Babbush, R., Neven, H.: Barren plateaus in quantum neural network training landscapes. Nature Communications 9, 4812 (2018)

8. Meng, Q., Zhao, S., Huang, Z., Zhou, F.: Magface: A universal representation for face recognition and quality assessment. In: CVPR (2021)

9. Novikov, A., Podoprikhin, D., Osokin, A., Vetrov, D.: Tensorizing neural networks. In: NeurIPS (2015)

10. Oseledets, I.V.: Tensor-train decomposition. SIAM Journal on Scientific Computing 33(5), 2295–2317 (2011)

11. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., Latorre, J.I.: Data re-uploading for a universal quantum classifier. Quantum 4, 226 (2020)

12. Phan, H., Yin, Y., Manley, T., et al.: Deepface-emd: Re-ranking using patch-wise earth mover’s distance improves out-of-distribution face identification. In: CVPR (2022)

13. Salari, V., et al.: Quantum face recognition protocol with ghost imaging. Scientific Reports 13, 2401 (2023)

14. Schönemann, P.H.: A generalized solution of the orthogonal procrustes problem. Psychometrika 31, 1–10 (1966)

15. Schrof, F., Kalenichenko, D., Philbin, J.: Facenet: A unified embedding for face recognition and clustering. In: CVPR (2015)

16. Schuld, M., Killoran, N.: Quantum machine learning in feature hilbert spaces. Physical Review Letters 122(4), 040504 (2019)

17. Sengupta, S., Chen, J.C., Castillo, C., Patel, V.M., Chellappa, R., Jacobs, D.W.: Frontal to profile face verification in the wild. In: WACV (2016)

18. Singh, S., Thakur, A., Hussain, M.I., Hasan, M.: Accelerating face biometric verification via quantum pca and hybrid processing. In: ICIRCA (2025)

19. Singh, S., Thakur, A., Sinha, D.K., Sinha, G., Hasan, M.: Quantum-enhanced eigenface algorithm for face verification. Journal of Information Systems Engineering and Management 10(36s), 588–603 (2025)

20. Stoudenmire, E.M., Schwab, D.J.: Supervised learning with tensor networks. Advances in Neural Information Processing Systems 29 (2016)

21. Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., Li, Z., Liu, W.: Cosface: Large margin cosine loss for deep face recognition. In: CVPR (2018)

22. Yi, D., Lei, Z., Liao, S., Li, S.Z.: Learning face representation from scratch. arXiv preprint arXiv:1411.7923 (2014)

23. Zheng, T., Deng, W.: Cross-pose lfw: A database for studying cross-pose face recognition in unconstrained environments. Beijing University of Posts and Telecommunications Technical Report (2018)

24. Zhu, X., Bouridane, A., Celebi, M.E., Konar, D., Angelov, P., Ni, Q., Jiang, R.: Quantum face recognition with multigate quantum convolutional neural network. IEEE Transactions on Artificial Intelligence 5(12), 6330–6341 (2024)