# TopoField: Technical Report for ECCV 2026 SceneFun3D Challenge

Zhiwei Chen<sup>\*</sup> Zhiheng Fu<sup>\*</sup> Zixu Li<sup>†</sup> Mingyu Zhang Jiale Huang Yupeng Hu

Shandong University

{zivczw, fuzhiheng8, lizixu.cs, zhmingyu.cs}@gmail.com huangjiale.cs@outlook.com, huyupeng@sdu.edu.cn

Abstract. Functional segmentation requires locating small interactive parts, such as knobs, buttons, and sockets, in indoor point clouds and returning confidence-ranked instance masks. A point-wise semantic predictor produces a continuous probability field, whereas the benchmark evaluates a discrete set of scored instances. This creates a substantial field-toinstance gap. We present TopoField, a deterministic decoder built on top of VOLT. TopoField first converts 13 test-time forward passes into a semantic consensus field, then decodes instance candidates from the consensus geometry, and finally ranks the candidates in a class-aware manner. The augmentation stage retains the processed-point identity of every fragment prediction, accumulates probabilities in a common point array, normalizes by coverage count, and fuses the resulting caches. The decoding stage separates high-confidence cores from broader support regions, constructs a symmetric radius-pruned point graph, and uses a multisource graph-geodesic partition to split support regions that contain multiple seeds. On 561 ground-truth instances across 30 validation scenes, an IoU-0.5-valid candidate exists for 67.2% of instances, and top-1 coverage is 20.0%. These complementary measurements motivate joint attention to candidate construction and ranking. TopoField achieves 28.286% AP, 41.383% $\mathrm { A P _ { 5 0 } }$ , and 47.558% $\mathrm { A P _ { 2 5 } }$ . TopoField achieved second place on the ECCV 2026 SceneFun3D Challenge.

Keywords: 3D point clouds · functional interaction element segmentation · instance decoding · test-time augmentation · graph-geodesic partitioning

## 1 Introduction

SceneFun3D [1] asks a model to locate nine categories of functional interaction elements, such as knobs, buttons, handles, and sockets, in complete indoor laser scans. Each prediction requires a category, a point mask, and a confidence score.

These elements occupy only a small portion of their host objects, and adjacent elements can share material, color, and local geometry. The task therefore requires not only recognizing an interactable object, but also recovering the spatial extent of the part that afords an action. Related eforts emphasize fine-grained video understanding [2], region-focused visual representations [3], and entitylevel alignment in composed retrieval [4–6].

Two coupled dificulties arise after a point-wise semantic predictor such as VOLT [7]. Small and partly occluded parts make per-point probabilities noisy: a high threshold truncates weak boundaries, whereas a low threshold can merge the target with its host or a neighbor. In addition, the predictor returns a continuous probability field, while the benchmark evaluates discrete masks sorted by confidence. Even with reasonable semantics, proposal construction and ordering can therefore cause a large field-to-instance error.

We quantify this gap on 30 validation scenes containing 561 evaluable instances. The examined pool has an IoU-0.5-valid candidate for 67.2% of the instances, and top-1 coverage is 20.0%. These complementary statistics characterize candidate-pool coverage and the overlap achieved by top-ranked candidates.

We present TopoField, whose method follows the two stages in Fig. 1. Stage 1, TTA-based Semantic Consensus, applies the shared predictor to 13 test-time forward passes over 12 distinct rotation–scale outcomes. Each fragment retains a processed-point index that scatters its softmax output into a common point-by-class array; accumulated probabilities are divided by coverage count and the independent caches are renormalized and fused. Stage 2, Geometry-aware Instance Decoding, selects a class-specific operator from argmax connectivity, threshold-support connectivity, or a seeded graph basin. The basin branch separates high-confidence cores from broader supports, builds a symmetric radius-pruned 32-nearest-neighbor graph, and assigns support points to seed components by accumulated graph-path length. Candidates are then scored and suppressed class by class in the original scan point order.

Our contributions are as follows.

– We formulate the gap between point-wise semantics and the benchmark’s ranked instance-set output, and quantify it through complementary anycandidate and top-k coverage statistics.

– We organize the system as two explicit stages that mirror its implementation: index-aligned test-time semantic consensus and geometry-aware instance decoding. The first stage makes point identity and cache fusion explicit; the second exposes core selection, graph construction, geodesic seed assignment, and candidate ranking.

– The parameter-free decoder achieves 28.286% AP, 41.383% $\mathrm { A P _ { 5 0 } }$ , and 47.558% $\mathrm { A P _ { 2 5 } }$

## 2 Related Work

3D instance segmentation. PointGroup [8] groups point-wise semantics with center ofsets in two coordinate spaces, while SoftGroup [9] reduces hard-label error propagation with soft semantic grouping. Query-based methods such as Mask3D [10] and SPFormer [11] learn instance or superpoint queries and directly predict masks. These methods jointly learn representation, grouping, and refinement [12, 13]. In visual tracking, complementary approaches explore windowbased attention [14], correlative masked modeling [15], and query-gated feature selection [16]. TopoField converts a fixed semantic predictor’s output into a ranked instance set through deterministic grouping and confidence-based ranking.

Functional interaction element segmentation. SceneFun3D [1] extends indoor scene understanding from complete objects to functional parts such as handles, knobs, and buttons. These parts are small, often partly occluded, and geometrically similar to their neighbors [17,18]. Related work on composed video retrieval models shared and diferential semantics [19] and calibrates directional anchors in composed representations [20]. VOLT [7] supplies the point-wise semantics in our system; TopoField constructs and ranks instance proposals from these semantic predictions. In video tracking, semantic consistency strengthens point correspondence [21], while temporally coherent object flow supports multiframe object association [22].

Graph-based region partitioning. Watershed-like methods propagate labels from seeds under a local adjacency structure [23]. A sparse point graph constrains propagation to 3D neighborhoods and prevents unreachable regions from borrowing labels across empty space [24–26]. Graph-structured relations also support heterogeneous nearest-neighbor retrieval [27] and collaborative motion reasoning in multi-object tracking [28]. TopoField applies this idea to a classconditioned support: cores supply seeds, a radius-pruned symmetric kNN graph defines paths, and multi-source shortest paths assign support points to seeds. Connected components first isolate disconnected supports; geodesic propagation then handles connected supports with several cores.

Test-time augmentation. Test-time augmentation reduces variance by transforming the input and aggregating the corresponding outputs [29–32]. For perpoint prediction, point identity must be preserved across augmentation and testtime voxelization. Stage 1 therefore scatters probabilities through retained point indices, normalizes by coverage, and fuses the caches before Stage 2 consumes the consensus field. Complementary evidence is also integrated through coupled multimodal state-space models [33]. Egocentric video understanding systems further explore category-conditioned inference and test-time consistency [34], as well as capability-based reasoning routes [35].

Noise-aware visual matching. Reliable visual matching depends on identifying task-relevant evidence and estimating the confidence of candidate associations. Research on noisy correspondence studies these issues in image–text retrieval, where training pairs can contain incorrect semantic associations. Huang et al. [36] introduce the Noisy Correspondence Rectifier, which partitions training pairs according to their estimated reliability and adaptively rectifies their correspondence through co-teaching. Qin et al. [37] develop an evidential learning framework that models correspondence uncertainty and adjusts negative-sample learning through a robust dynamic hinge loss. For composed image retrieval [38], ConeSep [39] localizes noisy correspondences through geometric fidelity estimation and performs boundary-based targeted unlearning. Air-Know [40] integrates expert-prior arbitration with knowledge internalization to guide robust representation learning, while INTENT [41] combines visual invariant composition with discriminative learning to address cross-modal correspondence noise and modality-inherent interference. Together, these approaches highlight evidence selection and confidence estimation as central components of reliable matching. In TopoField, these considerations motivate the use of semantic consensus, confident core selection, and class-aware candidate scoring for constructing ranked instance predictions.

![](images/ca5cde6e35953c073aa1eeb9add2552a379a71056c45043e95ae097e6dd53469.jpg)

![](images/9ae446d890710eddc54f03d1b925a587fb23460dac8c2abce4ee21e58ea6d523.jpg)  
Fig. 1: TopoField overview. Stage 1 aggregates index-aligned point probabilities from 13 test-time forward passes into a semantic consensus field. Stage 2 applies class-wise initialization, sparse-graph geodesic propagation, and candidate scoring.

## 3 Method

TopoField is a deterministic decoder after a point-wise predictor. Stage 1 stabilizes the probability field and resolves index misalignment between augmented inputs; Stage 2 converts the field into instance masks and scores without updating predictor parameters.

## 3.1 Problem formulation

Given a scan with N points, the predictor produces P (c) for point i and class c ∈ $\{ 0 , \ldots , 9 \}$ , where class 0 is background and classes $1 , \ldots , 9$ are target categories.

The decoder returns

$$
\mathcal { V } = \{ ( M _ { \ell } , c _ { \ell } , s _ { \ell } ) \} _ { \ell = 1 } ^ { L } , \qquad M _ { \ell } \subseteq \{ 1 , \dots , N \} , \qquad c _ { \ell } \in \{ 1 , \dots , 9 \} ,\tag{1}
$$

where $M _ { \ell }$ uses the original point order and $s _ { \ell }$ orders candidates. Only foreground instances are emitted.

## 3.2 Stage 1: TTA-based Semantic Consensus

Augmented forward passes. The released profile runs 13 forward passes. Its first twelve are the Cartesian product of four vertical-axis rotations, $\rho \in \{ 0 , { \frac { 1 } { 2 } } , 1 , { \frac { 3 } { 2 } } \}$ , and three scales, $s \in \{ 0 . 9 5 , 1 . 0 0 , 1 . 0 5 \}$ . The thirteenth is a double-axis flip. Flipping both horizontal axes equals a 180-degree rotation, so the passes contain 12 distinct spatial transforms.

Index-aligned accumulation. For pass a and voxelized fragment point $j ,$ let $\pi _ { a } ( j )$ be its retained processed-point index. Test-time voxelization can place several fragments under the same index. We accumulate softmax probabilities in a common $N \times 1 0$ array

$$
{ \bf A } _ { i } ^ { ( a ) } = \sum _ { j : \pi _ { a } ( j ) = i } { \bf p } _ { j } ^ { ( a ) } , \qquad v _ { i } ^ { ( a ) } = \sum _ { j : \pi _ { a } ( j ) = i } 1 ,\tag{2}
$$

and divide by the number of contributions to each processed point:

$$
\tilde { \mathbf { P } } _ { i } ^ { ( a ) } = \frac { \mathbf { A } _ { i } ^ { ( a ) } } { v _ { i } ^ { ( a ) } } .\tag{3}
$$

Retained processed-point indices align predictions across coordinate and normal transformations, enabling direct accumulation in the common point array.

Cache fusion. Each pass writes an independent cache. For positive weights $w _ { a }$ and $\textstyle W = \sum _ { a } w _ { a }$ , we combine the caches and normalize each row:

$$
\bar { P } _ { i } ( c ) = \frac { \sum _ { a = 1 } ^ { 1 3 } ( w _ { a } / W ) \tilde { P } _ { i } ^ { ( a ) } ( c ) } { \sum _ { k = 0 } ^ { 9 } \sum _ { a = 1 } ^ { 1 3 } ( w _ { a } / W ) \tilde { P } _ { i } ^ { ( a ) } ( k ) } .\tag{4}
$$

All 13 passes receive equal weights. Since index 12 duplicates index 2 geometrically, their combined weight is $2 / 1 3$

## 3.3 Stage 2: Geometry-aware Instance Decoding

Class-conditioned route. The class-conditioned decoder selects among argmax connectivity, foreground-threshold connectivity, and a seeded graph basin for each category. The basin branch in Stage 2 of Fig. 1 uses confident cores and graph connectivity to partition support regions.

(a) Class-wise support and core. For each foreground class $c ,$ let

$$
\mathcal { F } _ { c } = \left\{ i : c = \arg \operatorname* { m a x } _ { c ^ { \prime } \in \{ 1 , . . . , 9 \} } \bar { P } _ { i } ( c ^ { \prime } ) \right\} .\tag{5}
$$

Given support threshold $\tau _ { s }$ and core threshold $\tau _ { h }$ , with $\tau _ { h } \geq \tau _ { s }$ , define

$$
S _ { c } = \{ i \in \mathcal { F } _ { c } : \bar { P } _ { i } ( c ) \geq \tau _ { s } \} , \qquad K _ { c } = \{ i \in S _ { c } : \bar { P } _ { i } ( c ) \geq \tau _ { h } \} .\tag{6}
$$

The support retains weak boundary evidence while the core supplies reliable seeds. The representative recipe uses $\tau _ { s } = 0 . 1 5 , \tau _ { h } = 0 . 4 0$ , at least five support points, and at least three core points.

(b) Sparse 3D graph. We build a graph $G _ { c } = ( S _ { c } , E _ { c } )$ only over the support points. Let $\mathcal { N } _ { k } ( i )$ be the k-nearest-neighbor set of i in $S _ { c } .$ We take the union of outgoing and incoming neighbor relations and remove long edges:

$$
E _ { c } = \{ \{ i , j \} : j \in \mathcal { N } _ { k } ( i ) { \mathrm { ~ o r ~ } } i \in \mathcal { N } _ { k } ( j ) , \ \| p _ { i } - p _ { j } \| _ { 2 } \leq r \} .\tag{7}
$$

The implementation uses $k = 3 2$ and weight $w _ { i j } = \| p _ { i } - p _ { j } \| _ { 2 }$ , bounded below by $1 0 ^ { - 1 2 }$ . The resulting sparse, symmetric graph constrains propagation to local 3D neighborhoods.

(c) Multi-source geodesic assignment. Connected components of $G _ { c }$ are formed first. Inside each support component, the core points induce seed components $Q _ { c j m }$ . If a support component contains more than one seed component, we run the multi-source Dijkstra step from all seeds and assign each support point to the seed with the smallest accumulated graph distance:

$$
a _ { c j } ( i ) = \arg \operatorname* { m i n } _ { m } \operatorname* { m i n } _ { q \in Q _ { c j m } } d _ { G _ { c } [ C _ { c j } ] } ( i , q ) .\tag{8}
$$

The resulting candidate is

$$
M _ { c j m } = \{ i \in C _ { c j } : a _ { c j } ( i ) = m \} .\tag{9}
$$

With one seed component, the whole support component forms a single candidate.

(d) Candidate ranking and suppression. For a materialized candidate M, the primary score is the mean consensus probability of its class:

$$
s ( M , c ) = { \frac { 1 } { | M | } } \sum _ { i \in M } \bar { P } _ { i } ( c ) .\tag{10}
$$

The decoder also supports quantile and hybrid pooling, and a class-wise size factor,

$$
s ^ { \prime } ( M , c ) = s ( M , c ) \left( \frac { | M | } { r _ { c } } \right) ^ { \alpha _ { c } } ,\tag{11}
$$

where $r _ { c }$ is a reference point count and $\alpha _ { c } = 0$ recovers the unscaled score. Exact duplicate masks keep the highest-scoring copy. For overlapping same-class candidates, soft NMS uses

$$
s _ { b } \gets s _ { b } \exp \left( - \frac { \mathrm { I o U } ( M _ { a } , M _ { b } ) ^ { 2 } } { \sigma } \right) .\tag{12}
$$

Candidates are filtered and suppressed class by class, and the final scene list is returned in original point order.

## 3.4 Implementation details

VOLT predicts at a 1 cm voxel resolution from XYZ, RGB, and normals. It has 12 transformer blocks, 768-dimensional embeddings, and 12 heads, initialized from ScanNet200 weights. Training uses AdamW $( 3 \times 1 0 ^ { - 4 }$ , weight decay 0.05), label-smoothed cross-entropy, multiclass Lovász loss, and EMA (0.999). Test inference uses frozen probability caches without gradient updates. Unless stated otherwise, Stage 1 is fixed while Stage 2 is varied.

## 4 Experiments

## 4.1 Setup

SceneFun3D [1] contains 160 training, 30 validation, and 30 test scenes over nine categories. We train VOLT with XYZ, RGB, and normals at 1 cm. Stage 1 is used only at inference. The oficial metric is AP over IoU 0.50–0.95, with $\mathrm { A P _ { 5 0 } }$ and $\mathrm { A P _ { 2 5 } }$ also reported. $R _ { \mathrm { a n y } }$ is the fraction of ground-truth instances whose candidate pool contains an IoU-0.5-valid mask; $R _ { k }$ restricts this test to the class-wise top-k candidates.

## 4.2 Candidate diagnosis

Using fixed epoch-80 EMA 13-TTA caches, Table 1 shows a 47.2-point gap between any-candidate and top-1 coverage. The two measurements distinguish candidate-pool coverage from top-ranked mask quality and provide complementary criteria for evaluating Stage 2.

Table 1: Candidate diagnosis on 30 validation scenes (561 ground-truth instances). Values are percentages.
<table><tr><td>Diagnostic Value</td></tr><tr><td>Foreground-point precision / recall 68.4 / 60.1</td></tr><tr><td> $R _ { \mathrm { a n y } } @ 0 . 5$  67.2</td></tr><tr><td> $R _ { 1 } @ 0 . 5$  20.0</td></tr><tr><td> $R _ { 3 } @ 0 . 5$  38.3</td></tr><tr><td> $R _ { 5 } @ 0 . 5$  46.3</td></tr></table>

## 4.3 Validation and test results

Table 2 reports validation results for successive decoder configurations. Support/core selection gives the largest improvement, and graph geodesic partitioning further separates adjacent candidates. Class-conditioned decoding raises AP from 37.60 to 38.45 on the development split and from 41.86 to 42.31 on the holdout split, reaching 39.67 AP across all 30 validation scenes.

Table 2: Validation configuration search (epoch-80, 13-TTA). AP values are percentages.
<table><tr><td>Configuration</td><td colspan="3">Development (20) Holdout (10) Full validation (30)</td></tr><tr><td>Baseline</td><td>35.21</td><td>39.47</td><td>36.58</td></tr><tr><td>+ Dual-threshold</td><td>36.89</td><td>41.02</td><td>38.14</td></tr><tr><td>+ Graph geodesic</td><td>37.60</td><td>41.86</td><td>38.92</td></tr><tr><td>+ Class-conditioned</td><td>38.45</td><td>42.31</td><td>39.67</td></tr></table>

Table 3 reports oficial submissions from two EMA checkpoints with robust and aggressive post-processing. Each checkpoint uses its associated saved seed configuration. The epoch-100 aggressive configuration achieves 28.286% AP, 41.383% $\mathrm { A P _ { 5 0 } }$ , and 47.558% $\mathrm { A P _ { 2 5 } }$

Table 3: Oficial-test submission results. AP, $\mathrm { A P _ { 5 0 } } .$ and $\mathrm { A P _ { 2 5 } }$ are percentages.
<table><tr><td>EMA epoch Post-processing</td><td>AP</td><td> $\mathrm { A P _ { 5 0 } }$ </td><td></td><td> $\mathrm { A P _ { 2 5 } }$ </td></tr><tr><td>80</td><td>Robust</td><td>27.919</td><td>41.373</td><td>45.878</td></tr><tr><td>80</td><td>Aggressive</td><td>27.640</td><td>41.103</td><td>46.364</td></tr><tr><td>100</td><td>Robust</td><td>28.270</td><td>40.847</td><td>46.761</td></tr><tr><td>100</td><td>Aggressive</td><td></td><td>28.286 41.383 47.558</td><td></td></tr></table>

## 4.4 Discussion

Decoder components. The successive configurations in Table 2 increase developmentsplit AP by 1.68 points with support/core selection, 0.71 points with graph geodesic partitioning, and 0.85 points with class-conditioned decoding. Together, these components raise full-validation AP from 36.58 to 39.67.

Method behavior. The two stages perform complementary operations. Stage 1 aggregates index-aligned point predictions into a semantic consensus field. Stage 2 constructs and ranks instance masks using class-conditioned connectivity and graph-geodesic partitioning. Support/core selection retains boundary evidence and supplies confident seeds, while geodesic assignment partitions multi-core support regions according to their 3D connectivity.

## 5 Conclusion

We presented TopoField, a two-stage deterministic decoder for functional interaction element segmentation. Stage 1 turns multiple test-time forward passes into a stable semantic field by retaining processed-point indices, accumulating softmax probabilities in a common point array, normalizing repeated fragment contributions by coverage, and fusing independent caches. Stage 2 uses classconditioned instance decoding; its seeded graph-basin branch separates a highconfidence core from a broader support region, constrains propagation with a symmetric radius-pruned point graph, and assigns support points to seed components by graph-geodesic distance. The validation analysis characterizes candidate coverage and ranking, and the combined decoder achieves 39.67% validation AP and 28.286% oficial-test AP. The decoder adds no learnable parameters to the point-wise predictor.

## References

1. Delitzas, A., Takmaz, A., Tombari, F., Sumner, R.W., Pollefeys, M., Engelmann, F.: SceneFun3D: Fine-Grained Functionality and Afordance Understanding in 3D Scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14531–14542 (2024)

2. Hu, Y., Song, Z., Feng, N., Luo, Y., Yu, J., Chen, Y.P.P., Yang, W.: Sf2t: Selfsupervised fragment finetuning of video-llms for fine-grained understanding. arXiv preprint arXiv:2504.07745 (2025)

3. Chen, Z., Hu, Y., Li, Z., Fu, Z., Song, X., Nie, L.: Ofset: Segmentation-based focus shift revision for composed image retrieval. In: ACM MM. p. 6113–6122 (2025)

4. Li, Z., Hu, Y., Fu, Z., Chen, Z., Li, Y., Nie, L.: Tema: Anchor the image, follow the text for multi-modification composed image retrieval. In: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 24421–24442 (2026)

5. Fu, Z., Li, Z., Chen, Z., Wang, C., Song, X., Hu, Y., Nie, L.: Pair: Complementarityguided disentanglement for composed image retrieval. In: ICASSP. pp. 1–5. IEEE (2025)

6. Huang, Q., Chen, Z., Li, Z., Wang, C., Song, X., Hu, Y., Nie, L.: Median: Adaptive intermediate-grained aggregation network for composed image retrieval. In: ICASSP. pp. 1–5. IEEE (2025)

7. Yilmaz, K., Kruse, A., Höfer, T., de Geus, D., Leibe, B.: Volume Transformer: Revisiting Vanilla Transformers for 3D Scene Understanding. arXiv:2604.19609 (2026)

8. Jiang, L., Zhao, H., Shi, S., Liu, S., Fu, C.W., Jia, J.: Pointgroup: Dual-set point grouping for 3d instance segmentation. In: CVPR (2020)

9. Vu, T., Kim, K., Luu, T.M., Nguyen, T., Yoo, C.D.: Softgroup for 3d instance segmentation on point clouds. In: CVPR (2022)

10. Schult, J., Engelmann, F., Kontogianni, T., Leibe, B.: Mask3d: Mask transformer for 3d semantic instance segmentation. In: ICRA (2023)

11. Sun, J., Qing, C., Tan, J., Xu, X.: Spformer: Semantic partition transformer for 3d instance segmentation. In: ECCV (2022)

12. Chen, S., Fang, J., Zhang, Q., Liu, W., Wang, X.: Hierarchical aggregation for 3d instance segmentation. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 15447–15456. IEEE (2021)

13. Kolodiazhnyi, M., Vorontsova, A., Konushin, A., Rukhovich, D.: Oneformer3d: One transformer for unified point cloud segmentation. In: 2024 IEEE/CVF Conference

on Computer Vision and Pattern Recognition (CVPR). pp. 20943–20953. IEEE (2024)

14. Song, Z., Yu, J., Chen, Y.P.P., Yang, W.: Transformer tracking with cyclic shifting window attention. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8791–8800 (2022)

15. Song, Z., Luo, R., Yu, J., Chen, Y.P.P., Yang, W.: Compact transformer tracker with correlative masked modeling. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 2321–2329 (2023)

16. Lv, M., Liu, Z., Shao, F., Chen, Y.P.P., Yu, J., Yang, W., Song, Z.: Gatemot: Qgated attention for dense object tracking. arXiv preprint arXiv:2604.26353 (2026)

17. Mo, K., Zhu, S., Chang, A.X., Yi, L., Tripathi, S., Guibas, L.J., Su, H.: Partnet: A large-scale benchmark for fine-grained and hierarchical part-level 3d object understanding. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 909–918. IEEE (2019)

18. Mo, K., Guibas, L.J., Mukadam, M., Gupta, A., Tulsiani, S.: Where2act: From pixels to actions for articulated 3d objects. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6813–6823 (2021)

19. Hu, Y., Li, Z., Chen, Z., Huang, Q., Fu, Z., Xu, M., Nie, L.: Refine: Composed video retrieval via shared and diferential semantics enhancement. ACM Transactions on Multimedia Computing, Communications and Applications 22(7), 1–24 (2026)

20. Li, Z., Hu, Y., Chen, Z., Huang, Q., Qiu, G., Fu, Z., Liu, M.: Retrack: Evidencedriven dual-stream directional anchor calibration network for composed video retrieval. In: AAAI. vol. 40, pp. 23373–23381 (2026)

21. Song, Z., Tang, Y., Luo, R., Ma, L., Yu, J., Chen, Y.P.P., Yang, W.: Autogenic language embedding for coherent point tracking. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 2021–2030 (2024)

22. Song, Z., Luo, R., Ma, L., Tang, Y., Chen, Y.P.P., Yu, J., Yang, W.: Temporal coherent object flow for multi-object tracking. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 6978–6986 (2025)

23. Vincent, L., Soille, P.: Watersheds in digital spaces: An eficient algorithm based on immersion simulations. IEEE TPAMI 13(6), 583–598 (1991)

24. Landrieu, L., Simonovsky, M.: Large-scale point cloud semantic segmentation with superpoint graphs. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4558–4567. IEEE (2018)

25. Wang, Y., Sun, Y., Liu, Z., Sarma, S.E., Bronstein, M.M., Solomon, J.M.: Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog) 38(5), 1–12 (2019)

26. Robert, D., Raguet, H., Landrieu, L.: Eficient 3d semantic segmentation with superpoint transformer. In: 2023 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 17149–17158. IEEE (2023)

27. Yang, Q., Chen, Z., Hu, Y., Li, Z., Fu, Z., Nie, L.: Stable: Eficient hybrid nearest neighbor search via magnitude-uniformity and cardinality-robustness. IEEE Transactions on Knowledge and Data Engineering (2026)

28. Song, Z., Yu, J., Chen, Y.P.P., Yang, W., Wang, X.: Hypergraph-state collaborative reasoning for multi-object tracking (2026), https://arxiv.org/abs/2604.12665

29. Ayhan, B., Kwan, C., Budavari, B., Kwan, L., Lu, Y., Perez, D., Li, J., Skarlatos, D., Vlachos, M.: Test-time augmentation for deep learning-based cell segmentation on microscopy images. In: Scientific Reports (2020)

30. Goyal, A., Law, H., Liu, B., Newell, A., Deng, J.: Revisiting point cloud classification with a simple and efective baseline. In: ICML (2021)

31. Qi, C.R., Yi, L., Su, H., Guibas, L.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems 30 (2017)

32. Shanmugam, D., Blalock, D., Balakrishnan, G., Guttag, J.: Better aggregation in test-time augmentation. In: 2021 IEEE/CVF international conference on computer vision (ICCV). pp. 1194–1203. IEEE (2021)

33. Li, W., Zhou, H., Yu, J., Song, Z., Yang, W.: Coupled mamba: Enhanced multimodal fusion with coupled state space model. Advances in Neural Information Processing Systems 37, 59808–59832 (2024)

34. Chen, Z., Hu, Y., Li, Z., Fu, Z., Qiu, G., Guan, W., Nie, L.: Egoadapt: A multiscene egocentric adaptation method for cvpr 2026 hd-epic vqa challenge. arXiv preprint arXiv:2605.24500 (2026)

35. Li, Z., Chen, Z., Fu, Z., Wang, W., Hu, Y., Guan, W., Nie, L.: Omniego-r<sup>2</sup>: A routed reasoning framework for the 1st cross-domain egocross challenge at cvpr 2026. arXiv preprint arXiv:2605.24481 (2026)

36. Huang, Z., Niu, G., Liu, X., Ding, W., Xiao, X., Wu, H., Peng, X.: Learning with noisy correspondence for cross-modal matching. Advances in Neural Information Processing Systems 34, 29406–29419 (2021)

37. Qin, Y., Peng, D., Peng, X., Wang, X., Hu, P.: Deep evidential learning with noisy correspondence for cross-modal retrieval. In: Proceedings of the 30th ACM International Conference on Multimedia. pp. 4948–4956 (2022)

38. Li, Z., Chen, Z., Wen, H., Fu, Z., Hu, Y., Guan, W.: Encoder: Entity mining and modification relation binding for composed image retrieval. In: AAAI. vol. 39, pp. 5101–5109 (2025)

39. Li, Z., Hu, Y., Chen, Z., Zhang, M., Fu, Z., Nie, L.: Conesep: Cone-based robust noise-unlearning compositional network for composed image retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16897–16909 (2026)

40. Fu, Z., Hu, Y., Yang, Q., Zhang, S., Chen, Z., Li, Z.: Air-know: Arbiter-calibrated knowledge-internalizing robust network for composed image retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2658–2670 (2026)

41. Chen, Z., Hu, Y., Fu, Z., Li, Z., Huang, J., Huang, Q., Wei, Y.: Intent: Invariance and discrimination-aware noise mitigation for robust composed image retrieval. In: AAAI. vol. 40, pp. 20463–20471 (2026)