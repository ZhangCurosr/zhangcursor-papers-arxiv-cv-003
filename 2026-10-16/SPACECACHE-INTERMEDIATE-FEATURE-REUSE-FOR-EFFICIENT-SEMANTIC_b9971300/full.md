# SPACECACHE: INTERMEDIATE FEATURE REUSE FOR EFFICIENT SEMANTIC SEGMENTATION ON POINT CLOUD STREAMS

Dongheon Yang<sup>∗</sup>, Beomseok Kang<sup>∗</sup>, Jae-Joon Kim

Seoul National University

{eddy49,beomseok,kimjaejoon}@snu.ac.kr

## ABSTRACT

Point-cloud streams are central to autonomous driving and robotics, yet existing segmentation networks process every frame from scratch, although consecutive frames largely overlap. We observe that the intermediate features of point-cloud transformers remain semantically consistent across frames for the same spatial region, especially in deep stages. Building on this, we present SpaceCache, a training-free feature cache that reuses these features across frames. Tokens of previously observed regions skip a sequence of transformer blocks and take their cached features, only the remaining tokens are computed, and the full token layout is restored for dense prediction. Lightweight cache management keeps reused features reliable by limiting each entry’s age and evicting low-confidence entries. Unlike token merging, SpaceCache selects tokens at negligible cost and runs its cache operations once per encoder stage. On both outdoor and indoor benchmarks, such as nuScenes, Waymo, and ScanNet, with two transformer backbones of Utonia and LitePT-B, SpaceCache achieves up to 1.65× speedup with less than 1- point mIoU drop and consistently outperforms token-merging methods. The code is available at: https://github.com/YangDongHeon/SpaceCache

## 1 INTRODUCTION

Point-cloud segmentation is a key component in real-time systems such as autonomous vehicles and robots, where low-latency inference is critical (Lang et al., 2019; Milioto et al., 2019). Much of the recent progress in this task has been driven by transformer-based backbones (Zhao et al., 2021; Wu et al., 2024; Yue et al., 2026), which treat individual points or local groups of points as tokens. However, a single LiDAR sweep often incorporates tens to hundreds of thousands of points (Sun et al., 2020; Caesar et al., 2020; Behley et al., 2019), causing these backbones to process an extremely large number of tokens per frame. This cost is further compounded in point-cloud streams, where such token-heavy inference must be repeated at every timestep. Despite this, accelerating inference in streaming point-cloud perception remains relatively underexplored.

In recent years, several studies have explored reducing inference latency in semantic segmentation, primarily for single-frame point cloud. For example, token-merging methods (Bolya et al., 2023; Norouzi et al., 2024; Tran et al., 2024; 2025) identify and merge similar, redundant tokens, thereby reducing the computation that would otherwise be performed on them. However, token merging requires similarity estimation and irregular aggregation operations such as gather/reduce to construct a new token sequence from the merged representations, introducing non-negligible GPU overhead (Lu et al., 2025). More recently, methods such as GitMerge3D (Tran et al., 2025) derive token similarities from layer-specific features, requiring merge and unmerge operations at every layer and causing this overhead to scale with network depth. Other approaches reduce computation through token pruning, which naturally avoids the repeated feature aggregation. Token Halting (Ye et al., 2023), for example, progressively prunes intermediate tokens but uses an additional learned token-selection module.

Encoder Stage (voxel size)  
![](images/f687838d76b55af71c10432d68a6e3db3ea96017adeb179d4185d0f4cc1e1bdc.jpg)  
Figure 1: Increasing temporal overlap across encoder stages. Visualization on the Waymo dataset using the stage-wise downsampling ratios of the Utonia and LitePT-B models. Both have five encoder stages (S0–S4). Each grid cell represents the voxel region corresponding to a single token, and its color indicates the ground-truth semantic class of that region. Magenta highlights voxels in frame t+1 that overlap with frame t. As progressive downsampling increases the voxel size at deeper stages, tokens represent larger spatial regions and exhibit higher cross-frame overlap.

Notably, temporal redundancy across multi-frame point clouds has received relatively limited attention. One such approach, EVE (Peng et al., 2021), prunes redundant input points using lightweight geometric proximity across frames and reuses their previous decoder outputs, but this inevitably reduces the contextual information available to the remaining tokens. As a result, EVE also relies on an additional learned module to compensate for the resulting errors. Consequently, existing approaches either incur non-negligible overhead in token merging or rely on additional training to mitigate the contextual information loss caused by token pruning.

In this paper, we introduce SpaceCache, an efficient intermediate-feature-reuse technique that skips computation for spatial regions observed in earlier frames and instead reuses their cached features. By reusing intermediate token representations, namely, those at deeper encoder stages, our method achieves the low overhead of token pruning while mitigating its contextual information loss in a training-free manner. This design is motivated by two observations: at deeper encoder stages, progressive downsampling leads to higher temporal token overlap (Figure 1), while cached features computed in earlier frames under different contexts become increasingly semantically consistent with their corresponding current-frame features (Sec. 2.2). SpaceCache targets only a few deeper encoder stages, where much of the encoder computation is concentrated, enabling substantial la tency reductions. Moreover, by caching features once per stage rather than at every layer within a stage, it keeps the cache footprint to only a few MiB. Our key contributions are as follows:

• Temporal redundancy grows with depth. We show that deeper encoder stages exhibit higher cross-frame token overlap and feature consistency (Sec. 2.2). This observation motivates SpaceCache, a training-free cache that lets tokens of previously observed regions bypass these stages.

• Reliability-aware cache management. Cached features can become stale on moving objects or unreliable at ambiguous regions, degrading accuracy when they are reused. We address this by combining confidence-based commit and eviction with lifetime-based eviction (Sec. 3.3).

• Better accuracy-efficiency trade-off. Across three benchmarks in both outdoor and indoor (nuScenes, Waymo, ScanNet) and two transformer backbones (Utonia, LitePT-B), SpaceCache consistently achieves a better accuracy-efficiency trade-off than token-merging methods, reaching up to 1.65× speedup within a 1-point mIoU drop (Sec. 4.1).

![](images/a2ffd861e1f0a31cae16643bed3a3f35a3a4b8cd629498ed52d12d9f61aab945.jpg)  
(a)

![](images/146fd7a3085d8c049c23c75dfcc9e05ec705df8f7bd76f5666dc6c5926f70bdd.jpg)  
(b)

![](images/8f1df5b3bc33903a49b35dbeec5455b2ce66e9b1fd0e1818103d7b9060264487.jpg)  
(c)  
Figure 2: Temporal-reuse opportunity in point-cloud encoders. (a) A multi-stage encoder, where stage l runs $B _ { l }$ blocks over a fixed token set and downsampling between stages shrinks the token set while enlarging the region each token covers. (b) The median fraction of stage-l overlapping tokens, whose region already appeared in the previous frame, grows substantially with depth. (c) The Reuse Agreement of stage-l overlapping tokens also increases with depth, indicating that the overlapped previous-frame token and the corresponding current-frame token predict the same class (Appendix A.2).

## 2 BACKGROUND AND PRELIMINARY STUDY

## 2.1 NETWORK PROPERTY

Point-cloud encoder architectures. Modern point-cloud encoders, such as Utonia (Zhang et al., 2026) and LitePT (Yue et al., 2026), typically adopt a hierarchical multi-stage architecture, as illustrated in Figure 2(a). Each stage consists of a downsampling layer that aggregates multiple spatially nearby tokens into a single token, followed by $B _ { l }$ transformer blocks that may include sparse convolution, attention, and MLP operations. An input embedding layer at the beginning of the network transforms the input point coordinates into high-dimensional feature vectors.

Token coordinates and features. We refer to each point or spatially aggregated representation processed by the encoder as a token. Each token is associated with a three-dimensional coordinate and a high-dimensional feature vector, which together form the input to the encoder blocks. At stage l, we denote the token coordinates of block b by $P _ { l } ^ { ( b ) } { = } \{ p _ { i } ^ { ( b ) } \} _ { i = 1 } ^ { N _ { l } }$ , where $N _ { l }$ is the number of tokens and $p _ { i } ^ { ( b ) } \in \mathbb { R } ^ { 3 }$ is the coordinate of token i. The output features are similarly denoted by $F _ { l } ^ { ( b ) } { = } \{ f _ { i } ^ { ( b ) } \} _ { i = 1 } ^ { N _ { l } } \backslash$ , where $f _ { i } ^ { ( b ) } \in \mathbb { R } ^ { D _ { l } }$ is the corresponding feature.

Consistent token coordinates within a stage. The token coordinate set changes only at downsampling layers. Since the transformer blocks within a stage preserve token coordinates, they all share the same coordinate set: $P _ { l } ^ { ( 0 ) } { = } P _ { l } ^ { ( 1 ) } { = } { \dots } { = } P _ { l } ^ { ( B _ { l } ) } { = } P _ { l }$ . Consequently, once tokens from different frames are matched to the same spatial region at a given stage, this correspondence remains valid throughout all blocks in that stage. This property enables intermediate features to be reused across the entire sequence of blocks, rather than only at an individual block.

## 2.2 FEATURE REUSE OPPORTUNITY

Our strategy for accelerating streaming point-cloud perception is to reuse intermediate features from previously observed spatial regions rather than repeatedly encoding them. Effective feature reuse depends on two questions: (1) how many tokens overlap across consecutive frames, and (2) whether the features of these overlapping tokens can be reliably reused. We examine these questions through temporal overlap and semantic consistency of intermediate features across frames.

Temporal overlap of intermediate tokens. Feature reuse is beneficial only when many tokens correspond to spatial regions already observed in previous frames. We therefore quantify their overlap across consecutive frames at each encoder stage. Because progressive downsampling changes the spatial granularity represented by each token, cross-frame overlap can vary across stages.

To quantify how often tokens from different frames correspond to the same physical regions, we transform their coordinates into a shared global coordinate system so that they can be compared in a common spatial reference frame (details; Sec. 3.2). Figure 2(b) shows that temporal overlap increases substantially with network depth: from 4% to 49% on nuScenes, 13% to 66% on Waymo, and 64% to 88% on ScanNet. Progressive downsampling makes deeper tokens represent larger spatial regions, increasing the likelihood that the same regions are observed across consecutive frames.

![](images/fef0748dfcb0972d9efd50bbd033409cb78acd0327840e433a59789f245169cb.jpg)  
Figure 3: Overview of SpaceCache. Inference pipeline with the Mask, Reuse, and Commit Gates. Cache-hit tokens bypass the selected transformer blocks, while cache-miss tokens are computed through the blocks. The Reuse Gate restores the full token layout, and the Commit Gate updates the cache after prediction. Each cached stage has its own SpaceCache.

Semantic consistency of overlapping features. Spatial overlap alone does not guarantee reliable feature reuse; features from overlapping regions must also preserve consistent semantic information across frames. In other words, a feature computed for a region in a previous frame should remain representative of the same semantic category when reused in the current frame.

To quantify this consistency, we define “Reuse Agreement”. For each semantic category c, we first construct a mean feature $\mu _ { c }$ by averaging the features $f _ { i }$ associated with that category. We then measure whether the reused feature and its corresponding current-frame feature are associated with the same semantic category (details; Appendix A.2). Figure 2(c) shows that Reuse Agreement increases with encoder depth, from approximately 75–83% at Stage 0 to 91–93% at Stages 3–4. In particular, the agreement begins to saturate from Stage 3, indicating that reused features at deeper stages largely preserve the semantic information of their corresponding current-frame features.

Design Takeaway. Figures 2(b) and (c) show that deeper encoder stages exhibit both higher temporal overlap and stronger semantic consistency across frames. These properties make deeper inter mediate representations increasingly suitable for temporal feature reuse.

## 3 METHOD

## 3.1 OVERVIEW OF SPACECACHE

Figure 3 illustrates the overall inference flow of a point-cloud encoder integrated with SpaceCache. SpaceCache stores reliable intermediate features from the last transformer block of selected encoder stages and reuses them when the corresponding spatial regions are observed again (cache-hit) in subsequent frames. The process is implemented through three gates with distinct roles: the Mask Gate, Reuse Gate, and Commit Gate.

The Mask Gate queries SpaceCache using the coordinates $p _ { i }$ of current-frame tokens to identify cache hits and misses. Cache-hit tokens (dotted bars in Figure 3) bypass the transformer blocks, while only cache-miss tokens are processed. After the transformer blocks, the Reuse Gate combines the newly computed features of cache-miss tokens (purple bars) with the cached features of cachehit tokens (orange bars) and restores the full-token layout for subsequent layers. Newly computed cache-miss features are also retained as candidates for the cache update. Once the frame prediction is produced, the Commit Gate determines which cached entries and candidate features remain valid and updates SpaceCache accordingly.

![](images/3e86ca2d2d2dfae5df61ee045a843aa0bde9be85d4a4fe6ea5ab31de5e254fef.jpg)  
Figure 4: Operation details of SpaceCache. Each entry stores a region index, cached feature, and age. Read: Current-frame tokens are mapped to global region indices and queried at the Mask Gate to identify cache hits; the corresponding cached features are then retrieved for reuse at the Reuse Gate. Write: Newly computed cache-miss features become candidate entries at the Reuse Gate; after prediction, the Commit Gate evicts invalid entries, commits valid candidates, and re-sorts the cache for the next-frame lookup.

## 3.2 OPERATION DETAILS OF SPACECACHE

SpaceCache components. Figure 4 illustrates the operations of SpaceCache in detail. Each entry in the cache consists of three components: a Regional Index $\mathbf { v } \in \mathbb { Z } ^ { 3 }$ recording a previously observed spatial region, a Feature $\mathbf { f } \in \mathbb { R } ^ { n }$ storing the intermediate representation computed for that region, and a scalar $A g e a \in \mathbb { N }$ counting the frames since the entry was inserted. The entries are maintained in arrays sorted by the index, allowing each lookup to be efficiently performed by binary search.

Transformation from token coordinate to region index. To reuse features across frames, Space-Cache must determine whether a token corresponds to a previously observed spatial region. However, token coordinates are generally expressed in each frame’s local coordinate system and hence cannot be directly compared across time. We therefore transform each token coordinate $p _ { i } ^ { t }$ into a shared global coordinate system using the ego pose $T _ { t }$ and then quantize it into a spatial cell whose size depends on the encoder stage, so that observations of the same physical region across frames are assigned the same region index: $\bar { p } _ { i } ^ { t } = T _ { t } p _ { i } ^ { t }$ and $v _ { i } ^ { t } = \left\lfloor \bar { p } _ { i } ^ { t } / s _ { l } \right\rfloor$ , where $s _ { l } = 2 ^ { l } g$ is the cell size at stage $l , g$ is the input voxel size, and $2 ^ { l }$ is the corresponding downsampling ratio. Finally, the three-dimensional region index $\boldsymbol { v } _ { i } ^ { t }$ is mapped into a scalar key $\phi ( v _ { i } ^ { t } )$ for efficient cache lookup on the GPU (Appendix B.1).

Step 1: cache read. At the Mask Gate, SpaceCache is queried using the region indices of the current-frame tokens. The lookup returns a cache-hit mask indicating which regions are already present in the cache (denoted as “Current region ∩ Cache region” in Figure 4), together with the cached features of the cache-hit tokens. The Mask Gate uses the hit mask to bypass redundant computation, while the Reuse Gate later retrieves the corresponding cached features (orange bars in “Read” of the figure). This operation is lightweight because it only requires checking whether each region index exists in the cache. Since the cache entries are sorted by region indices, we perform a binary search for each token in parallel on the GPU. Each lookup takes $\mathcal { O } ( \log M )$ operations, resulting in $\mathcal { O } ( N \log M )$ total work for N input tokens and M cached entries.

Step 2: candidate write. At the Reuse Gate, newly computed features of cache-miss tokens are paired with their region indices and temporarily stored as candidate entries (purple bars in “Write” of the figure). These candidates remain separate from the active cache until the frame-level semantic predictions become available for reliability assessment. Meanwhile, the newly computed features are combined with the cached features of cache-hit tokens to restore the original token layout. The candidates are then passed to the Commit Gate for validation and possible insertion into the cache.

Step 3: cache commit and eviction. Intermediate feature reuse does not guarantee that a cached feature remains valid for every subsequent frame. First, cached features may become stale as the scene changes, particularly in regions containing dynamic objects, where the semantic content of the same spatial region can change across frames. Second, a feature may be unreliable when it is first computed, particularly in ambiguous regions such as object boundaries or when reduced context leads to uncertain predictions. Repeatedly reusing such features can propagate errors across subsequent frames. Thus, SpaceCache evaluates on the fly whether newly generated candidate features should be committed to the cache and whether existing cached features should be evicted (Sec. 3.3).

## 3.3 CACHE MANAGEMENT IN SPACECACHE

Confidence filtering—dynamic commit and eviction. We use decoder prediction confidence as a proxy for feature reliability, filtering out low-confidence features from both new candidates and cached entries. As Figure 5 shows, Reuse Agreement (Sec. 2.2) increases with region confidence for both candidate and cached entries, indicating that higher-confidence features tend to remain more semantically consistent across frames.

The Commit Gate computes region confidence from the output logits of the current frame. For each point, we take its maximum class probability, and the confidence of a region v is the average of these probabilities over the input points in v:

$$
c ( v ) = \frac { 1 } { | \Omega _ { v } | } \sum _ { j \in \Omega _ { v } } \operatorname* { m a x } _ { k } \sigma ( z _ { j } ) _ { k } ,\tag{1}
$$

![](images/3cd92de7d045762258efb0dc212130bc7d23558036b90f0e20ec14d368db667c.jpg)  
(a)

![](images/fa9842376dc8811c3a8a35d6bd07eebb2589d7e75bd2eba7af498d7355639751.jpg)  
(b)  
Figure 5: Reuse Agreement of cache entries (Stage-3, Waymo with Utonia). (a) By region confidence $c ( v )$ at $L { = } 8$ for all, cached, and candidate entries; the dashed line marks θ=0.85. (b) By entry age with $L { = } \infty$

where $\Omega _ { v }$ is the set of input points associated with region $v , \ z _ { j }$ is the logit vector of point $j ,$ and σ is the softmax. With a pre-defined confidence threshold $\theta ,$ candidate entries with $c ( v ) { \leq } \theta$ are discarded, and cached entries at or below the same threshold are evicted.

Lifetime–static eviction. Confidence filtering does not explicitly bound how long a cached feature can persist, while Figure 5 shows that reuse agreement decreases with cache age. In particular, older entries are progressively less likely to remain semantically consistent with the corresponding current-frame features. SpaceCache therefore imposes a maximum lifetime $L$ on cache entries. Each newly committed entry is assigned an expiration age in $\{ 1 , \ldots , L \}$ from a hash of its region index, preventing all entries from expiring simultaneously (Appendix B.1). Once expired, the corresponding region is recomputed when observed again. A smaller L refreshes features more frequently at the cost of additional computation. We study this trade-off in Sec. 4.2.

## 3.4 COMPUTATION REDUCTION AND OVERHEAD

Computation reduction. Let $N _ { l }$ be the number of tokens at stage l and $h _ { l }$ the cache-hit rate; after the Mask Gate, the enclosed encoder blocks process only $( 1 - h _ { l } ) N _ { l }$ tokens. The major operations in point-cloud transformer blocks scale linearly with the number of tokens: windowed attention costs $\mathcal { O } ( W N _ { l } )$ for a fixed window size W, token-wise projections and MLPs cost $\mathcal { O } ( N _ { l } )$ , and submanifold sparse convolution costs $\mathcal { O } ( K N _ { l } )$ for a fixed kernel size $K$ . Hence, SpaceCache reduces the computation of these operations approximately in proportion to $h _ { l }$

Overhead. SpaceCache incurs a one-time per-frame overhead for region-index generation, cache lookup, feature assembly, and cache update. In our setting, all cache operations together take about 5 ms per frame, while the cache occupies approximately 70 MiB with single-precision features. The overhead remains small because index generation uses simple coordinate transformation and quantization, lookup performs parallel binary searches over scalar indices, and feature assembly only restores cached and computed features to their token positions. Since these costs are largely inde pendent of the number of enclosed blocks, they are better amortized as more blocks are bypassed. We provide a detailed breakdown in Sec. 4.3.

Stage selection for SpaceCache. Following the design takeaway from Sec. 2, among the five encoder stages (S0-S4), Stages 3 and 4 exhibit both high temporal overlap and high feature reusability. We therefore primarily apply SpaceCache to these deeper stages. SpaceCache remains applicable to any sequence of consecutive blocks within a stage that preserves token coordinates, and we evaluate different stage selections in Sec. 4.2.

Table 1: Comparison on outdoor streams. Each method is shown at its fewest-FLOPs setting within a 1-point mIoU drop from the backbone (rings in Figure 6). Latency (ms) and GFLOPs are measured per frame on representative scenes with dense attention. Dashes indicate that no setting satisfies the 1-point drop. <sup>∗</sup>: 2D method applied to point tokens. <sup>†</sup>: without the adaptive patch partition.
<table><tr><td></td><td colspan="4">Waymo</td><td colspan="4">nuScenes</td></tr><tr><td>Method</td><td>mIoU (%)↑</td><td>Latency↓</td><td>GFLOPs↓</td><td>Speedup↑</td><td>mIoU (%)↑</td><td>Latency↓</td><td></td><td>GFLOPs↓ Speedup↑</td></tr><tr><td>Utonia</td><td>67.17</td><td>721</td><td>4438</td><td>1.00×</td><td>77.29</td><td>226</td><td>1269</td><td>1.00×</td></tr><tr><td>+ ToMe*</td><td>66.35</td><td>691</td><td>3794</td><td>1.04×</td><td>76.31</td><td>255</td><td>1024</td><td>0.89×</td></tr><tr><td>+ ALGM*</td><td>66.24</td><td>633</td><td>3751</td><td>1.14×</td><td>76.36</td><td>209</td><td>1060</td><td>1.08×</td></tr><tr><td> $+ \mathrm { G i t M e r g e 3 D ^ { \dag } }$ </td><td>66.38</td><td>706</td><td>3864</td><td>1.02×</td><td>76.32</td><td>237</td><td>1123</td><td>0.95×</td></tr><tr><td> $+ \underbrace { \mathrm { S p a c e C a c h e } \phantom { . } \mathrm { s t a g e } } _ { \mathrm { ^ a } } .$ </td><td>66.25</td><td>563</td><td>3177</td><td>1.28×</td><td>76.47</td><td>192</td><td>982</td><td>1.18×</td></tr><tr><td> $+ \mathrm { S p a c e C a c h e } _ { \mathrm { S t a g e } - 3 , 4 }$ </td><td>66.50</td><td>537</td><td>2941</td><td>1.34×</td><td>76.38</td><td>187</td><td>918</td><td>1.21×</td></tr><tr><td>LitePT-B</td><td>72.82</td><td>307</td><td>2392</td><td>1.00×</td><td>80.04</td><td>98</td><td>683</td><td>1.00×</td></tr><tr><td>+ ToMe</td><td>72.00</td><td>245</td><td>1488</td><td>1.25×</td><td>79.33</td><td>101</td><td>492</td><td>0.97×</td></tr><tr><td>+ ALGM*</td><td></td><td></td><td></td><td></td><td></td><td>一</td><td></td><td></td></tr><tr><td>+  $\mathrm { G i t M e r g e 3 D ^ { \dag } }$ </td><td>72.05</td><td>287</td><td>2020</td><td>1.07×</td><td>79.05</td><td>101</td><td>605</td><td>0.97×</td></tr><tr><td>+ SpaceCache Stage-3</td><td>71.86</td><td>202</td><td>1374</td><td>1.52×</td><td>79.21</td><td>76</td><td>468</td><td>1.29×</td></tr><tr><td> $+ \mathrm { S p a c e C a c h e } _ { \mathrm { S t a g e } - 3 , 4 }$ </td><td>72.11</td><td>186</td><td>1226</td><td>1.65×</td><td>79.14</td><td>75</td><td>444</td><td>1.31×</td></tr></table>

![](images/07f7931a015963768003ccb84b5dfdb28b0863613f47f35717b9a7fe2c56b033.jpg)  
Figure 6: Accuracy-efficiency trade-off on outdoor streams. Curves sweep the merge ratio (ToMe, GitMerge3D), similarity threshold (ALGM), or confidence threshold θ (SpaceCache), plotting mIoU against FLOPs (top) and latency (bottom). Solid line and star–backbone; dashed lines–1- and 2-point mIoU drops; Rings–fewest-FLOPs point of each method within the 1-point drop.

## 4 EXPERIMENTAL RESULTS

Dataset and Implementation Details. We use the full validation sets of two outdoor LiDAR datasets, nuScenes (Caesar et al., 2020) (6,019 frames, 150 scenes) and Waymo (Sun et al., 2020) (5,976 frames, 202 scenes), whose labeled sweeps arrive at 2 Hz. For the indoor stream, we convert each ScanNet (Dai et al., 2017) RGB-D frame with its camera pose into a point-cloud stream subsampled to 3 Hz, using 10 validation scenes (2,221 frames). We integrate SpaceCache into two pretrained point-cloud networks, Utonia (Zhang et al., 2026) and LitePT-B (Yue et al., 2026). By default, SpaceCache is applied to encoder stages 3 and $\mathrm { 4 ~ ( S p a c e C a c h e _ { S t a g e - 3 , 4 } ) ; }$ we also evaluate a stage-3-only variant $\mathrm { ( S p a c e C a c h e _ { S t a g e - 3 } ) }$ We set the lifetime L to 8/4 frames for outdoor/indoor streams and the confidence threshold θ to 0.85. Outdoor/indoor latency is measured on an NVIDIA RTX A6000/Jetson AGX Orin. All models use a single forward pass without test-time augmentation.

Baselines. We compare SpaceCache with token-merging methods that aggregate redundant tokens within each frame: ToMe (Bolya et al., 2023) and ALGM (Norouzi et al., 2024), designed for 2D image transformers and applied to point tokens, and GitMerge3D (Tran et al., 2025) without its adaptive patch partition<sup>1</sup>. Implementation details are provided in Appendix B.

Table 2: Comparison on indoor streams. Each method is shown at its fewest-FLOPs setting within 1-point and 2-point mIoU drops. Kernel time (ms) is averaged over 50 representative frames. The notation follows Table 1.
<table><tr><td></td><td colspan="4">mIoU drop ≤ 1.0 pt</td><td colspan="4">mIoU drop ≤ 2.0 pt</td></tr><tr><td>Method</td><td>mIoU (%)↑</td><td>Kernel time↓</td><td>GFLOPs↓</td><td>Speedup↑</td><td>mIoU (%)↑</td><td>Kernel time↓</td><td>GFLOPs↓</td><td>Speedup↑</td></tr><tr><td>LitePT-B</td><td>50.63</td><td>43.3</td><td>39.9</td><td>1.00×</td><td>50.63</td><td>43.3</td><td>39.9</td><td>1.00×</td></tr><tr><td>+ ToMe*</td><td>49.72</td><td>41.1</td><td>28.0</td><td>1.06×</td><td>48.80</td><td>40.5</td><td>26.6</td><td>1.07×</td></tr><tr><td>+ ALGM*</td><td>49.80</td><td>39.9</td><td>32.1</td><td>1.09×</td><td>48.75</td><td>37.2</td><td>29.7</td><td>1.16×</td></tr><tr><td>+ GitMerge3D†</td><td>49.64</td><td>47.2</td><td>37.3</td><td>0.92×</td><td>48.65</td><td>46.8</td><td>37.1</td><td>0.93×</td></tr><tr><td>+ SpaceCache Stage-3</td><td>49.81</td><td>36.8</td><td>28.0</td><td>1.18×</td><td>48.68</td><td>35.5</td><td>25.6</td><td>1.22×</td></tr><tr><td>+ SpaceCache Stage-3,4</td><td>49.68</td><td>38.8</td><td>31.0</td><td>1.12×</td><td>48.80</td><td>36.2</td><td>26.2</td><td>1.20×</td></tr></table>

Table 3: Ablation study on Waymo with Utonia. (a) Lifetime control. (b) Stage selection. Gray: default setting; L=1 reproduces the backbone. Hit: cache-hit fraction of each cached stage. ∆Dyn./∆Sta.: accuracy change (points) on dynamic/static regions at stage-3 relative to L=1 (83.7/94.8%; Appendix C.3). ∆FLOPs: encoder FLOPs change relative to the backbone.

(a) Cache management (commit and eviction)
<table><tr><td rowspan="2">L</td><td colspan="4">Age only</td><td colspan="4">Age + Confidence</td></tr><tr><td>Speedup↑</td><td>mIoU↑</td><td>∆Dyn.</td><td>∆ Sta.</td><td>Speedup↑</td><td>mIoU↑</td><td>∆Dyn.</td><td>∆ Sta.</td></tr><tr><td>1</td><td>1.00×</td><td>67.18</td><td></td><td></td><td>1.00×</td><td>67.18</td><td></td><td></td></tr><tr><td>2</td><td>1.09×</td><td>67.14</td><td>-0.5</td><td>-0.1</td><td>1.10×</td><td>67.12</td><td>-0.5</td><td>0.0</td></tr><tr><td>4</td><td>1.24×</td><td>65.84</td><td>-1.9</td><td>-0.3</td><td>1.21×</td><td>67.11</td><td>-1.1</td><td>0.0</td></tr><tr><td>8</td><td>1.35×</td><td>63.99</td><td>-3.0</td><td>-0.8</td><td>1.25×</td><td>66.91</td><td>-1.8</td><td>0.0</td></tr><tr><td>16</td><td>1.44×</td><td>62.79</td><td>-4.1</td><td>-1.0</td><td>1.33×</td><td>66.68</td><td>-3.0</td><td>0.0</td></tr><tr><td>32</td><td>1.44×</td><td>62.37</td><td>-5.3</td><td>-0.9</td><td>1.37×</td><td>66.53</td><td>-3.9</td><td>-0.1</td></tr></table>

(b) Stage selection for SpaceCache
<table><tr><td>Stage</td><td>Hit (%)</td><td>∆FLOPs (%)</td><td>Speedup↑ mIoU↑</td><td></td></tr><tr><td>0</td><td>20</td><td>-0.7</td><td>1.00×</td><td>67.00</td></tr><tr><td>1</td><td>32</td><td>-2.1</td><td>1.01×</td><td>65.28</td></tr><tr><td>2</td><td>42</td><td>-5.0</td><td>1.03×</td><td>65.42</td></tr><tr><td>3</td><td>48</td><td>-34.7</td><td>1.21×</td><td>66.60</td></tr><tr><td>4</td><td>52</td><td>-6.5</td><td>1.03×</td><td>67.47</td></tr><tr><td>3+4</td><td>48/51</td><td>-41.1</td><td>1.25×</td><td>66.91</td></tr></table>

## 4.1 COMPARISON WITH PRIOR METHODS

Comparison in outdoor scenario. Table 1 compares SpaceCache with ToMe, ALGM, and Git-Merge3D on Waymo and nuScenes with Utonia and LitePT-B, each at its fewest-FLOPs setting within a 1-point mIoU drop. SpaceCache is the fastest in all four settings, reaching up to 1.65× on Waymo with LitePT-B (307→186 ms) by removing 49% of FLOPs.

Figure 6 extends this comparison over the full sweep of each method. Within the 2.0-point drop, SpaceCache achieves a better trade-off than token merging in all four settings, in both FLOPs (top) and latency (bottom). Two properties underlie this advantage: more FLOPs removed and better FLOPs-to-latency conversion. We analyze both in Sec. 4.3.

Comparison in indoor scenario. Table 2 reports each method using LitePT-B on ScanNet at its setting with the fewest FLOPs within 1-, 2-point mIoU drops. Since host-side launch overhead dominates the latency of small indoor frames, we report the average GPU kernel time over 50 representative frames. SpaceCache is the fastest with the fewest FLOPs at both drops, reaching 1.22× with 36% fewer GFLOPs within the 2-point drop. The speedup is smaller than outdoors because the denser indoor points are merged more aggressively by downsampling.

## 4.2 ABLATION STUDY

Cache management. Table 3(a) examines SpaceCache’s two control knobs, which trade accuracy for efficiency rather than having a single optimal value: the lifetime L and confidence filtering with threshold θ. With confidence filtering (“Age + Confidence”), increasing L from 1 to 32 raises the speedup to 1.37× but lowers the mIoU by 0.65 points, as older features survive longer in the cache; the drop is more pronounced on dynamic objects (∆ Dyn. of −3.9). Without filtering (“Age only”), the mIoU drops far more steeply (4.81 points at L=32), and even at a similar speedup it trails the filtered setting (65.84% at 1.24× for L=4 vs. 66.91% at 1.25× for L=8).

Stage selection. Table 3(b) shows how performance changes with the stage which SpaceCache is applied to. The speedup grows with the number of enclosed blocks, peaking at stage-3, which holds 12 of the 24 blocks. The accuracy loss shrinks with depth from stage-1 to 4, consistent with the semantic consistency in Figure 2; despite its low consistency, stage-0 loses little, as it reuses the fewest tokens and all later stages refine them. Stage-3, 4 together thus give the largest speedup at a small accuracy drop, which we use as the default.

![](images/e309fc29f833e5f15e3dc9aceb3bc987c0fdfcc79ee813984b7058ede213c775.jpg)

![](images/de2595d8f84d5411ad45443bef013763806bdf193584180b21e01bc68b62ddca.jpg)  
(a)

![](images/47735eb7225a7cf79b9a10ecdf33224750049e5147af9ee14aab7f551f3318f3.jpg)  
(b)  
(c)  
Figure 7: Latency and memory breakdown on the representative scenes of Waymo with Utonia, each method within a 2.0-point mIoU drop. (a) Latency per frame, split into the backbone layers and the policy overhead. (b) Policy overhead per frame, split into its components. (c) Peak memory with flash-attention, split into the model weights, the activation memory, and the cache.

## 4.3 EFFICIENCY ANALYSIS

Where the gain comes from. SpaceCache outperforms token merging at similar accuracy (Figure 6) for two reasons, decomposed on Waymo with Utonia within a 2-point drop. (1) More FLOPs removed. Cache-hit tokens skip every block of a stage at a negligible scoring cost, so SpaceCache shrinks the sparse convolution, attention, and FFN together, removing 37% of the total FLOPs (Appendix D). GitMerge3D instead merges tokens only within attention, and its per-block scoring leaves a net reduction of only 18%. (2) Better FLOPs-to-latency conversion. SpaceCache runs its cache operations once per stage, costing only 4.8 ms, mostly for sorting at the Commit Gate. Merging repeats scoring at every block and accumulates merged features through memory-bound gather/reduce, so GitMerge3D adds 136 ms of overhead (Figure 7). Even ToMe, the merging method that removes the most FLOPs here (23%; Appendix D), reduces the latency by only 7%, whereas SpaceCache<sub>Stage-3,4</sub> turns a 37% FLOPs reduction into a 25% latency reduction (Figure 7(a)).

Memory breakdown and overhead. The memory overhead of SpaceCache is also small. As shown in Figure 7(c), SpaceCache raises the peak memory by only 95 MiB (7.0%): 69 MiB for the cache itself and 26 MiB of activations for holding cache-miss features until the Commit Gate.

## 5 RELATED WORK

Efficient point-cloud backbones. Point-cloud backbones have moved from sparse convolution (Graham et al., 2018; Choy et al., 2019) to point transformers (Zhao et al., 2021; Wu et al., 2022), and many designs make them faster by sampling points randomly (Hu et al., 2020), grouping points into superpoints (Robert et al., 2023), or simplifying attention (Wu et al., 2024; Yue et al., 2026). Such designs, however, need training, which is costly now that backbones are pretrained at scale (Wu et al., 2025; Zhang et al., 2026). We instead speed up existing pretrained backbones on point-cloud streams without training.

Token reduction in vision. A common way to speed up transformer networks in vision is to process fewer tokens. Token merging combines similar tokens (Bolya et al., 2023; Liang et al., 2022a; Norouzi et al., 2024; Tran et al., 2025), pruning removes unimportant ones (Liang et al., 2022b; Chen et al., 2023; Zhao et al., 2023; Baiocchi et al., 2025; Li et al., 2025), and halting stops tokens at intermediate layers in 2D (Yin et al., 2022; Tang et al., 2023) and 3D (Ye et al., 2023). Some point networks instead group points into fewer tokens (Xu et al., 2021; Wang et al., 2025). These methods require training by design, except for merging. Merging, however, often searches for similar tokens again at every block and accumulates several tokens into one through slow memory accesses, which limits its speed-up (Lu et al., 2025). SpaceCache instead selects tokens cheaply and only once per stage, and copies each cached feature one-to-one without accumulating tokens.

Temporal reuse in point-cloud streams. While many methods use past frames for accuracy (Liu et al., 2019; Cao et al., 2020; Li et al., 2023; He et al., 2023; Dang et al., 2025), few use them to save computation. Of these, TempNet (Zhou et al., 2021) propagates key-frame features with trained modules, and SSCATeR (Dow et al., 2026) recomputes convolutions only where points changed. Closest to our work, EVE (Peng et al., 2021) prunes input points near the previous frame and trains a module to recover lost context. SpaceCache instead keeps all input points, reuses only deep-stage features, and needs no training.

## 6 CONCLUSION

We presented SpaceCache, a training-free intermediate-feature-reuse method for efficient pointcloud stream processing. By exploiting the high temporal overlap and feature reusability of deeper encoder stages, SpaceCache skips redundant computation while preserving dense token representations. Across outdoor and indoor semantic-segmentation benchmarks, SpaceCache consistently improves the accuracy-efficiency trade-off over existing token-reduction methods, with low runtime and memory overhead.

## REFERENCES

Alessandro Baiocchi, Indro Spinelli, Alessandro Nicolosi, and Simone Scardapane. Adaptive token selection for scalable point cloud transformers. Neural Networks, 188:107477, 2025.

Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and Jurgen Gall. SemanticKITTI: A dataset for semantic scene understanding of LiDAR sequences.¨ In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your ViT but faster. In International Conference on Learning Representations (ICLR), 2023.

Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Hanwen Cao, Yongyi Lu, Cewu Lu, Bo Pang, Gongshen Liu, and Alan Yuille. ASAP-Net: Attention and structure aware point cloud sequence segmentation. In Proceedings of the British Machine Vision Conference (BMVC), 2020.

Xuanyao Chen, Zhijian Liu, Haotian Tang, Li Yi, Hang Zhao, and Song Han. SparseViT: Revisiting activation sparsity for efficient high-resolution vision transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4D spatio-temporal ConvNets: Minkowski convolutional neural networks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

Chenxu Dang, Zaipeng Duan, Pei An, Xinmin Zhang, Xuzhong Hu, and Jie Ma. FASTer: Focal token acquiring-and-scaling transformer for long-term 3D object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Re. FlashAttention: Fast and´ memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Alexander Dow, Manduhu Manduhu, Matheus Santos, Ben Bartlett, Gerard Dooly, and James Riordan. SSCATeR: Sparse scatter-based convolution algorithm with temporal data recycling for real-time 3-D object detection in LiDAR point clouds. IEEE Sensors Journal, 26(6):7853–7874, 2026.

Benjamin Graham, Martin Engelcke, and Laurens van der Maaten. 3D semantic segmentation with submanifold sparse convolutional networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

Tong He, Pei Sun, Zhaoqi Leng, Chenxi Liu, Dragomir Anguelov, and Mingxing Tan. LEF: Lateto-early temporal fusion for LiDAR 3D object detection. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023.

Qingyong Hu, Bo Yang, Linhai Xie, Stefano Rosa, Yulan Guo, Zhihua Wang, Niki Trigoni, and Andrew Markham. RandLA-Net: Efficient semantic segmentation of large-scale point clouds. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Point-Pillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Enxu Li, Sergio Casas, and Raquel Urtasun. MemorySeg: Online LiDAR semantic segmentation with a latent memory. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Jianhao Li, Tianyu Sun, Xueqian Zhang, Zhongdao Wang, Bailan Feng, and Ke Xu. Efficient 3D perception on multi-sweep point cloud with Gumbel spatial pruning. In Proceedings ofthe IEEE International Conference on Robotics and Automation (ICRA), 2025.

Weicong Liang, Yuhui Yuan, Henghui Ding, Xiao Luo, Weihong Lin, Ding Jia, Zheng Zhang, Chao Zhang, and Han Hu. Expediting large-scale vision transformer for dense prediction without finetuning. In Advances in Neural Information Processing Systems (NeurIPS), 2022a.

Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Not all patches are what you need: Expediting vision transformers via token reorganizations. In International Conference on Learning Representations (ICLR), 2022b.

Xingyu Liu, Mengyuan Yan, and Jeannette Bohg. MeteorNet: Deep learning on dynamic 3D point cloud sequences. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

Wenbo Lu, Shaoyi Zheng, Yuxuan Xia, and Shengjie Wang. ToMA: Token merge with attention for diffusion models. In Proceedings of the International Conference on Machine Learning (ICML), 2025.

Andres Milioto, Ignacio Vizzo, Jens Behley, and Cyrill Stachniss. RangeNet++: Fast and accurate LiDAR semantic segmentation. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019.

Narges Norouzi, Svetlana Orlova, Daan de Geus, and Gijs Dubbelman. ALGM: Adaptive localthen-global token merging for efficient semantic segmentation with plain vision transformers. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Gao Peng, Bo Pang, and Cewu Lu. Efficient 3D video engine using frame redundancy. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2021.

Pointcept Contributors. Pointcept: A codebase for point cloud perception research. https:// github.com/Pointcept/Pointcept, 2023.

Damien Robert, Hugo Raguet, and Loic Landrieu. Efficient 3D semantic segmentation with Superpoint Transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo Open Dataset. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Quan Tang, Bowen Zhang, Jiajun Liu, Fagui Liu, and Yifan Liu. Dynamic token pruning in plain vision transformers for semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Philippe Tillet, H. T. Kung, and David Cox. Triton: An intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL), 2019.

Hoai-Chau Tran, Duy M. H. Nguyen, Duy M. Nguyen, TrungTin Nguyen, Ngan Le, Pengtao Xie, Daniel Sonntag, James Zou, Binh T. Nguyen, and Mathias Niepert. Accelerating transformers with spectrum-preserving token merging. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Tuan Anh Tran, Duy M. H. Nguyen, Hoai-Chau Tran, Michael Barz, Khoa D. Doan, Roger Wattenhofer, Ngo Anh Vien, Mathias Niepert, Daniel Sonntag, and Paul Swoboda. How many tokens do 3D point cloud transformer architectures really need? In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Xuying Wang, Yunsheng Zhang, Run Shao, Siyang Chen, Haifeng Li, and Xin Chen. SAT-Former: An efficient 3-D transformer with semantic aggregated point tokenizer for point cloud semantic segmentation in urban scenes. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 18:28326–28342, 2025.

Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point Transformer V2: Grouped vector attention and partition-based pooling. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point Transformer V3: Simpler, faster, stronger. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, and Julian Straub. Sonata: Self-supervised learning of reliable point representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Chenfeng Xu, Bohan Zhai, Bichen Wu, Tian Li, Wei Zhan, Peter Vajda, Kurt Keutzer, and Masayoshi Tomizuka. You Only Group Once: Efficient point-cloud processing with token representation and relation inference module. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021.

Mao Ye, Gregory P. Meyer, Yuning Chai, and Qiang Liu. Efficient transformer-based 3D object detection with dynamic token halting. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz, and Pavlo Molchanov. A-ViT: Adaptive tokens for efficient vision transformer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Yuanwen Yue, Damien Robert, Jianyuan Wang, Sunghwan Hong, Jan Dirk Wegner, Christian Rupprecht, and Konrad Schindler. LitePT: Lighter yet stronger point transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Yujia Zhang, Xiaoyang Wu, Yunhan Yang, Xianzhe Fan, Han Li, Yuechen Zhang, Zehao Huang, Naiyan Wang, and Hengshuang Zhao. Utonia: Toward one encoder for all point clouds. In Proceedings ofthe International Conference on Machine Learning (ICML), 2026.

Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip Torr, and Vladlen Koltun. Point Transformer. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

Tianchen Zhao, Xuefei Ning, Ke Hong, Zhongyuan Qiu, Pu Lu, Yali Zhao, Linfeng Zhang, Lipu Zhou, Guohao Dai, Huazhong Yang, and Yu Wang. Ada3D: Exploiting the spatial redundancy with adaptive inference for efficient 3D object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Yunsong Zhou, Hongzi Zhu, Chunqin Li, Tiankai Cui, Shan Chang, and Minyi Guo. TempNet: Online semantic segmentation on large-scale point cloud series. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

## A TOKEN REGIONS AND REUSE AGREEMENT

## A.1 TOKEN REGIONS

Pooling chain. Each downsampling layer merges the tokens that fall into the same voxel of the next stage into one token and records which token each of them went $\mathrm { t o } , \pi _ { l } ( i )$ for a token i of stage l−1. We call the chain of these records the pooling chain. Following it from the input points, the tokens of stage 0, every input point j reaches exactly one token of stage l, so each token v, which represents one spatial region, covers the input points

$$
\Omega _ { v } = \{ j : \pi _ { l } ( \pi _ { l - 1 } ( \cdot \cdot \cdot \pi _ { 1 } ( j ) ) ) = v \} .\tag{2}
$$

Token labels. The label of a token is the majority ground-truth label of its input points, $y _ { v } =$ arg max<sub>c</sub> $| \{ j \in \Omega _ { v } : y _ { j } = c \}$ |, ignoring the ignore label, and a token is pure if all of them carry the same label. The following sections use the same pooling chain and labels.

Token accuracy. Our accuracy measures score a group of tokens by the input points they cover through the pooling chain. For a group $G _ { t }$ of tokens in a frame $t ,$ the token accuracy is

$$
\operatorname { A c c } ( G _ { t } ) = \frac { \sum _ { v \in G _ { t } } \sum _ { j \in \Omega _ { v } } { \bf 1 } [ \hat { y } _ { j } = y _ { j } ] } { \sum _ { v \in G _ { t } } | \Omega _ { v } | } ,\tag{3}
$$

where $\hat { y } _ { j }$ is the predicted class, the argmax of the logits, and $y _ { j }$ is the ground-truth label of the voxel of point $j ;$ points with the ignore label are excluded. It is the recall of the points covered by the group within the frame, with each token weighted by its covered-point count. We then average $\operatorname { A c c } ( G _ { t } )$ over all frames, weighted by the covered-point count of each frame. Appendix C.3 uses this measure with different groups $G _ { t }$

## A.2 REUSE AGREEMENT

Reuse Agreement (Figures $2 ( \mathrm { c } )$ and 5) measures whether a reused feature predicts the same class as the feature the backbone computes for the same region in the current frame.

Class prototypes. At stage l, the prototype of class c is the mean current-frame feature over $F _ { c } ^ { l }$ , the pure tokens of class c:

$$
\mu _ { c } ^ { l } = \frac { 1 } { | F _ { c } ^ { l } | } \sum _ { f \in F _ { c } ^ { l } } f .\tag{4}
$$

Only classes with at least 20 such tokens in the frame get a prototype.

Agreement. Each feature $f$ is mapped to a probability distribution over the classes by the softmax of its cosine similarities to the prototypes,

$$
q ( f ) = \operatorname { s o f t m a x } \big ( \left[ \cos ( f , \mu _ { 1 } ^ { l } ) , \dots , \cos ( f , \mu _ { C } ^ { l } ) \right] / T _ { l } \big ) ,\tag{5}
$$

where the temperature $T _ { l }$ is calibrated per stage so that q has the same mean normalized entropy of $0 . 5$ at every stage, for a fair comparison between stages. The Reuse Agreement of a token is the overlapping area of the distributions of its current feature $f ^ { t }$ and its reused feature $\tilde { f }$ over the classes:

$$
\mathrm { R A } = \sum _ { c } \operatorname* { m i n } \big ( q _ { c } ( f ^ { t } ) , q _ { c } ( \tilde { f } ) \big ) \in [ 0 , 1 ] .\tag{6}
$$

RA is 1 when $q ( f ^ { t } )$ and $q ( \tilde { f } )$ are identical and 0 when they share no class.

For Figure 2(c), the reused feature is the one computed for the same region in frame $t { - } 1$ . We average RA over all overlapped tokens of 60 consecutive frame pairs per dataset and plot the mean over the three datasets for each backbone. For Figure 5, the reused feature is a cached or candidate feature.

## B IMPLEMENTATION DETAILS

This section describes how SpaceCache and the three baselines are implemented. All methods are implemented in the Pointcept codebase (Pointcept Contributors, 2023)<sup>2</sup> and use the same pretrained weights without training, and each baseline is swept over its main parameter (Figure 6; Table 5 lists the swept values). On LitePT-B (Yue et al., 2026), whose attention layers are only in the last two encoder stages, ToMe and GitMerge3D merge in those blocks.

All methods run with batch size 1 and dense attention instead of flash-attention (Dao et al., 2022). Note that SpaceCache itself is compatible with flash-attention, while GitMerge3D adds to the attention scores a bias for the number of tokens merged into each key, which flash-attention cannot express. The attention is computed in bfloat16 for Utonia (Zhang et al., 2026) and float16 for LitePT-B, as in each backbone, and all other layers in float32.

## B.1 SPACECACHE

Settings. The region of each stage is its own voxel, $s _ { l } = 2 ^ { l } g ,$ with $g = 5$ cm on Waymo (Sun et al., 2020) and nuScenes (Caesar et al., 2020) and 2 cm on ScanNet (Dai et al., 2017). Each cached stage keeps its own SpaceCache. When several tokens of a frame fall into the same region, their features are averaged into one entry. The cache is emptied at the start of every sequence. The lookup, the feature assembly, and the cache update run as fused Triton (Tillet et al., 2019) kernels on the RTX A6000 and in PyTorch on the Jetson AGX Orin.

Region key. ϕ shifts the three indices of a region $\boldsymbol { v } = ( v _ { x } , v _ { y } , v _ { z } )$ to non-negative values and packs them into separate bit fields of one 64-bit integer, 24 bits for $v _ { x }$ and $v _ { y }$ and 15 bits for $v _ { z } \colon$

$$
\phi ( v ) = \left( \left( v _ { x } + 2 ^ { 2 3 } \right) \ll 3 9 \right) | \left( \left( v _ { y } + 2 ^ { 2 3 } \right) \ll 1 5 \right) | ( v _ { z } + 2 ^ { 1 4 } ) ,\tag{7}
$$

where $\ll$ is the left bit shift and | the bitwise OR. It costs three additions, two shifts, and two ORs per token, and it is one-to-one for $| v _ { x } | , | v _ { y } | < 2 ^ { 2 3 }$ and $| v _ { z } | < 2 ^ { 1 4 }$ , i.e., within $\pm 2 ^ { 2 3 } s _ { l }$ horizontally and $\pm 2 ^ { 1 4 } s _ { l }$ vertically of the global origin. The range doubles with each stage:

<table><tr><td>Dataset</td><td>Stage</td><td>Cell sl</td><td>Horizontal</td><td>Vertical</td></tr><tr><td>Waymo, nuScenes</td><td>3</td><td>0.4 m</td><td>±3,355km</td><td>±6.5 km</td></tr><tr><td>ScanNet</td><td>3</td><td>0.16m</td><td>±1,342km</td><td>±2.6km</td></tr><tr><td>Waymo (placement ablation)</td><td>0</td><td>5 cm</td><td>±419km</td><td>±819m</td></tr></table>

All are far larger than any sequence in the benchmarks.

Lifetime. The Age of an entry is implemented as a due frame rather than a counter. Each region index v has a fixed phase, the linear hash $p _ { 1 } v _ { x } + p _ { 2 } v _ { y } + p _ { 3 } v _ { z }$ with large primes $p _ { i }$ , and an entry committed at frame t is due at the next frame congruent to that phase modulo L: $d _ { v } = t + 1 +$ $( \mathrm { p h a s e } ( v ) - ( t + 1 ) )$ mod $L .$ The lookup counts an entry as a cache hit only while $t < d _ { v }$ . Since the due frame is modular in the region index instead of counted from the insertion frame, every region is refreshed at a fixed L-frame cadence and roughly one L-th of the resident entries expires per frame, which keeps the cache-hit ratio stable across frames.

Regions without training labels. Waymo labels only its top LiDAR, so the preprocessing gives the ignore label to every point of its four short-range LiDARs around the vehicle. nuScenes labels the points that hit its own car as vehicle.ego, which the benchmark ignores. These points receive no training loss, so the Commit Gate does not insert a region that contains them. We take these points from the ignore label, which also covers the few points of classes that the benchmark ignores. At inference, the same points can be identified without labels, as the points of the short-range LiDARs on Waymo and the points inside a fixed box around the ego vehicle on nuScenes. The ignored point never count toward the mIoU, and we do not apply this exclusion on ScanNet.

## B.2 TOME

ToMe (Bolya et al., 2023)<sup>3</sup> merges tokens by bipartite matching between the attention and the MLP of every attention block. Within each serialized attention patch, the tokens are split alternately into two sets, each token of the first set is matched to its most similar token of the second set by the cosine similarity of their attention keys, and the best-matched pairs are averaged into one token, without proportional attention. The sweep parameter r is the merging ratio, the fraction of the tokens of each patch merged in a block. A token merged at stage l stays merged until the decoder unpooling from stage l to l−1, which copies its feature back to its original tokens. Stage 0 has no unpooling below it, so its merges are unmerged at the end of the decoder, before the segmentation head.

## B.3 ALGM

ALGM (Norouzi et al., 2024)<sup>4</sup> merges twice. Local merging (CLAP), after the attention of the first block of stage 0, merges each 4×4×4-voxel window into one token if the mean pairwise cosine similarity of its tokens exceeds τ. The windows are offset by two voxels from the downsampling grid so that the following downsamplings do not absorb the merge. Global merging (GBM), after the attention of the first block of stage 3, splits all tokens of the stage alternately into two sets and matches them by bipartite matching. Following the official implementation, it keeps a pair only if its similarity exceeds $\mu + \sigma / b ,$ where µ and σ are the mean and standard deviation of the best-match similarities and b is the block index, so the number of merges adapts to each frame. Both merges stay until the decoder returns to their stage and copies the merged features back: GBM at the first decoder block of stage 3, and CLAP at the first decoder block of stage 0. On LitePT-B, which has no attention in stage 0 and no decoder blocks, CLAP is applied after the first block of stage 0, and both merges are copied back at the corresponding unpooling.

## B.4 GITMERGE3D

GitMerge3D (Tran et al., 2025)<sup>5</sup> merges inside the attention of every attention block. Each serialized attention patch is divided into bins of $\left\lceil 1 / ( 1 - r ) \right\rceil$ ⌉ tokens, and one random token of each bin becomes a destination. A fraction r of the tokens in each patch are averaged into their most similar destinations, based on the cosine similarity of the value vectors in each attention head. The attention then runs on the merged queries, keys, and values, and its output is copied back to the original tokens right away.

We report GitMerge3D without the adaptive patch partition (<sup>†</sup> in the tables). The adaptive version splits the patches by a global energy score. Patches with a high energy are merged at ratio r as above, and the other patches are reduced from 1024 to 32 tokens each. At the same merge ratio, reducing a 1024-token patch to 32 tokens lowers the accuracy considerably on Waymo. On nuScenes and the indoor stream, the accuracy is preserved, but computing the energy and splitting the patches offset the savings from merging (Table 4).

Table 4: GitMerge3D with and without its adaptive patch partition at the same merge ratio r=0.5. Speedup is measured in latency on Waymo and nuScenes and in kernel time on ScanNet. <sup>†</sup>: without the adaptive patch partition.
<table><tr><td></td><td colspan="2">Waymo</td><td colspan="2">nuScenes</td><td colspan="2">ScanNet</td></tr><tr><td>Method</td><td>mIoU (%)↑</td><td>Speedup↑</td><td>mIoU (%)↑</td><td>Speedup↑</td><td></td><td>mIoU (%)↑ Speedup↑</td></tr><tr><td>Utonia</td><td>67.17</td><td>1.00×</td><td>77.29</td><td>1.00×</td><td>一</td><td></td></tr><tr><td>+ GitMerge3D† (r=0.5)</td><td>65.64</td><td>1.05×</td><td>74.03</td><td>1.00×</td><td></td><td></td></tr><tr><td>+ GitMerge3D (r=0.5)</td><td>63.64</td><td>0.89×</td><td>73.35</td><td>0.86×</td><td></td><td></td></tr><tr><td>LitePT-B</td><td>72.82</td><td>1.00×</td><td>80.04</td><td>1.00×</td><td>50.63</td><td>1.00×</td></tr><tr><td>+ GitMerge3D† (r=0.5)</td><td>72.05</td><td>1.07×</td><td>74.92</td><td>1.06×</td><td>49.44</td><td>0.92×</td></tr><tr><td>+ GitMerge3D (r=0.5)</td><td>55.37</td><td>1.08×</td><td>74.87</td><td>1.07×</td><td>49.48</td><td>0.92×</td></tr></table>

## B.5 SWEPT SETTINGS

Each method is swept over one parameter: the merge ratio r for ToMe and GitMerge3D, the similarity threshold τ of the local merge for ALGM, and the confidence threshold θ for SpaceCache. SpaceCache runs at $L { = } 8$ on the outdoor streams and L=4 on the indoor stream. Table 5 lists the swept values.

Table 5: Swept parameter values. The outdoor columns list every point of the Figure 6 curves; the ScanNet column lists the two reported operating points of Table 2. off disables the confidence gate. Bold marks the point picked within a 1.0-point mIoU drop.
<table><tr><td></td><td></td><td colspan="2">Waymo</td><td colspan="2">nuScenes</td><td>ScanNet</td></tr><tr><td>Method</td><td>Param. Utonia</td><td></td><td>LitePT-B</td><td>Utonia</td><td>LitePT-B</td><td>LitePT-B</td></tr><tr><td>ToMe*</td><td>r</td><td>0.01, 0.015, 0.025, 0.04, 0.0450.12, 0.16, 0.2</td><td>0.01, 0.05, 0.095,</td><td>0.01, 0.02, 0.025, 0.035, 0.045, 0.05</td><td>0.02, 0.04, 0.06, 0.09, 0.12</td><td>0.11, 0.13</td></tr><tr><td>ALGM*</td><td>T</td><td>0.96, 0.97, 0.98, 0.995, 1</td><td>0.6, 0.7, 0.75, 0.85, 0.95, 0.97</td><td>0.7, 0.95, 0.97, 0.995</td><td>0.9, 0.925, 0.95, 0.98, 1</td><td>0.9, 0.8125</td></tr><tr><td>GitMerge3D†</td><td>r</td><td>0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65,</td><td>0.5, 0.645, 0.7, 0.72</td><td>0.1, 0.2, 0.3, 0.35, 0.1, 0.2, 0.33, 0.45, 0.5, 0.6</td><td>0.45, 0.6</td><td>0.55, 0.645</td></tr><tr><td>SpaceCache Stage-3</td><td>θ</td><td>0.72 0.95,0.99</td><td>off, 0.55, 0.7, 0.8, off, 0.7, 0.8, 0.95, 0.8, 0.9, 0.97, 0.99</td><td>0.99</td><td>off, 0.5, 0.65, 0.8, 0.85, off 0.95, 0.99</td><td></td></tr><tr><td>SpaceCache Stage-3,4</td><td>θ</td><td>off, 0.5, 0.55, 0.8, 0.95,0.99</td><td>off, 0.6, 0.8, 0.95, 0.99</td><td>0.7, 0.85, 0.95, 0.985, 0.99</td><td>0.5, 0.7, 0.825, 0.95, 0.97, 0.99</td><td>0.99, 0.8</td></tr></table>

## C DATASET AND EXPERIMENTAL SETUP

## C.1 SCANNET VIDEO STREAM

Frames. We keep every tenth frame (3 Hz) of the 30 Hz RGB-D videos of 10 validation scenes, 2,221 frames in total, and drop the few frames without a valid camera pose. Each depth pixel closer than 8 m is back-projected with the depth intrinsics and takes the color of the nearest color pixel. The scenes are scene0025 00, scene0050 00, scene0169 00, scene0217 00, scene0426 00, scene0488 00, scene0565 00, scene0568 00, scene0580 00, and scene0583 00.

Labels. Each point takes the label of its pixel in the 2D label images of ScanNet (Dai et al., 2017), mapped to the 20 benchmark classes as for the fused scenes. All other labels are ignored.

Coordinates and normals. The backbone is pretrained on the fused ScanNet scenes, so we express each frame in the coordinates of its scene with the camera pose. All frames of a sequence thus share one voxel grid. LitePT-B also takes surface normals as input and is pretrained on the normals of the scene meshes, so each point takes the normal of its nearest mesh vertex instead of a normal estimated from the single depth frame.

## C.2 EFFICIENCY MEASUREMENT

Accuracy uses the full validation sets. Latency, FLOPs, and memory use representative scenes, whole sequences whose per-frame distribution of voxel counts matches the full set: 10 segments (299 frames) of Waymo, 8 scenes (321 frames) of nuScenes, and 2 scenes (395 frames) of ScanNet. On ScanNet, the FLOPs are counted on all 10 scenes.

Latency. For the outdoor scenes we measure on an RTX A6000 and report the mean frame latency. For the indoor scenes we measure on a Jetson AGX Orin and report the mean GPU kernel time per frame over 50 frames, every eighth frame of the scenes, summed over all GPU kernels of each frame. The scenes are streamed in full, so SpaceCache reaches these frames with its true cache state.

FLOPs. The FLOPs count the linear layers, the sparse convolutions with their actual input-output pairs, the attention, and the similarity computations of the merging methods, with one multiplyaccumulate as two FLOPs. Normalizations, activations, softmax, and gathers are not counted.

## C.3 METRICS OF THE ABLATION STUDY

In Table 3, the cache-hit fraction is averaged over all frames except the first of each sequence, where the cache is empty. ∆Dyn. and ∆Static group the third-stage tokens by whether the ground-truth label of their region (Appendix A.1) has changed since the region was last seen. A token is dynamic if the label has changed and either the old or the new label is a movable class (car, truck, bus, other vehicle, motorcyclist, bicyclist, pedestrian, bicycle, or motorcycle). A token is static if the label has not changed. For each group, we report the change in point accuracy from the backbone (L=1), where point accuracy is the fraction of points predicted correctly.

## D BREAKDOWN DETAILS

Figure 7 compares every method at its point within a 2.0-point mIoU drop on the representative scenes of Waymo with Utonia. This section explains where each method removes computation, where it spends its own overhead, and what it adds to memory (Table 6). FLOPs and latency use dense attention, and memory uses flash-attention.

## D.1 FLOPS SAVING AND SCORING OVERHEAD

Table 6 shows how many FLOPs each method removes from the network layers and how many its scoring, the step that decides which tokens to reduce, adds back. SpaceCache removes the most and adds the least. As in Section $3 . 4 , N _ { l }$ is the number of tokens at stage l, W the attention window size, K the kernel size of the sparse convolution, and $h _ { l }$ the cache-hit rate. D is the number of channels, and r is the merge ratio of ToMe and GitMerge3D.

FLOPs saving. A block at stage l costs $\mathcal { O } ( W D N _ { l } + K D ^ { 2 } N _ { l } )$ , where the first term comes from the attention matrix products $\scriptstyle Q K ^ { \ Y }$ and $A V$ and the second from the linear layers and the sparse convolution. SpaceCache removes $\mathcal { O } \big ( h _ { l } ( W D N _ { l } + K D ^ { 2 } N _ { l } ) \big )$ at every block of a cached stage. ToMe merges a fraction r of the tokens at every block and thus removes $\mathcal { O } \big ( ( 1 - ( 1 - r ) ^ { k } ) ( W D N _ { l } +$ $K D ^ { 2 } N _ { l } ) )$ at its k-th block. ALGM merges twice, locally at stage 0 and globally at stage 3. Each merging removes a fraction $\rho _ { l }$ of the tokens, set by the similarities in each frame, and thus $\mathcal { O } \big ( \rho _ { l } ( W \bar { D } N _ { l } + K D ^ { 2 } N _ { l } ) \big )$ at every following block. GitMerge3D merges the tokens only between the projections, so it removes $\mathcal { O } \big ( ( 1 - ( 1 - r ) ^ { 2 } ) W D N _ { l } \big )$ from the first term at every block.

Scoring overhead. SpaceCache adds only $\mathcal { O } ( N _ { l } )$ (9N<sub>l</sub> multiply-adds) once per stage, since it rotates the 3D coordinate of each token and never touches the D-dimensional features. The merging methods compare the features of the tokens instead. GitMerge3D adds $\mathcal { O } ( W D N _ { l } )$ at every block, the same order as the attention $\mathrm { F L O P s }$ it removes. ToMe also matches within each window but on the keys averaged over the H heads, which adds $\mathcal { O } ( W D N _ { l } / H )$ at every block. ALGM adds $\mathcal { O } ( N _ { 0 } D )$ for its local merging, which compares only the tokens within each small window, and $\mathcal { O } ( N _ { 3 } ^ { 2 } D )$ for its global merging, which compares all tokens of the third stage with each other.

## D.2 LATENCY OF THE OVERHEAD OPERATIONS

Table 6 splits the overhead latency of each method into scoring (Appendix D.1) and three operations that only move the token features: gather/reduce, scatter, and sort. The overhead thus consists of both computation and memory traffic, and its latency depends on how often each operation runs, not only on its FLOPs.

Data movement. Gather/reduce builds the reduced token set. The merging methods accumulate each group of source tokens into its destination with atomic additions, as in their official implementations (Bolya et al., 2023; Norouzi et al., 2024; Tran et al., 2025). Many source tokens add into the same destination at once, and the GPU serializes these conflicting writes. SpaceCache instead gathers the cache-miss tokens into a contiguous subset, reading each token once without any accumulation. Scatter restores the full token layout. The merging methods copy each merged feature back to its source tokens. SpaceCache writes the newly computed features of the cache-miss tokens and the cached features of the cache-hit tokens to their own positions. Sort maintains the cache and is used only by SpaceCache. The Commit Gate merges the new entries into the table, which is kept sorted by region key for the binary search, and evicts the expired and low-confidence entries, once per stage and frame.

Repetition across blocks. SpaceCache runs each operation once per stage. In contrast, Git-Merge3D repeats its scoring, gather/reduce, and scatter at every block. ToMe repeats its scoring and gather/reduce at every block but scatters only at the decoder unpooling. ALGM runs each operation only twice, for its local merging at stage 0 and its global merging at stage 3.

## D.3 MEMORY OVERHEAD

Table 6 splits the peak memory into the model weights (602 MiB for every method), the activation memory (the peak of the intermediate tensors of the forward pass), and the cache (the persistent table of SpaceCache). $\mathrm { S p a c e C a c h e _ { S t a g e - 3 } }$ and $\mathrm { S p a c e C a c h e } _ { \mathrm { S t a g e } - 3 , 4 }$ add 18 and 26 MiB of activations, because they hold the newly computed features of the cache-miss tokens until the Commit Gate after the decoder. Each cache entry holds the region key (64-bit integer), the feature (single precision, 432 channels at the third stage and 576 at the fourth), and the age of one region. The cache takes 47.7 MiB for the third stage and 21.7 MiB for the fourth (69.4 MiB in total). GitMerge3D restores the tokens right after each attention and keeps nothing across blocks, so its activation memory stays within 2 MiB of the backbone. ToMe and ALGM instead keep their merge maps and the unmerged tokens until the decoder restores them, which raises the activation memory by 175 and 58 MiB.

Table 6: FLOPs, latency, and memory breakdown on the representative scenes, each method at its point within a 2.0-point mIoU drop. The latency and memory blocks are plotted in Figure 7. FLOPs and latency are per frame and use dense attention, and memory uses flash-attention.
<table><tr><td rowspan="2"></td><td colspan="6">Waymo with Utonia</td></tr><tr><td>Utonia + ToMe*</td><td></td><td>+ ALGM*</td><td>+ GitMerge3D†</td><td>+ SpaceCache Stage-3</td><td>+ SpaceCache Stage-3,4</td></tr><tr><td>Sweep parameter</td><td></td><td>r=0.025</td><td>τ=0.98</td><td>r=0.60</td><td>θ=0.55</td><td>θ=0.55</td></tr><tr><td>Accuracy mIoU (%)↑</td><td>67.17</td><td>65.36</td><td>65.25</td><td>65.21</td><td>65.20</td><td>65.44</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GFLOPs Linear and SpConv</td><td></td><td></td><td>2622.8</td><td>3321.5</td><td></td><td></td></tr><tr><td>Attention products</td><td>3321.5</td><td>2485.2 928.5</td><td>951.8</td><td>178.9</td><td>2178.0 889.7</td><td>1955.5 857.3</td></tr><tr><td>Saving</td><td>1116.2</td><td>-1024.0</td><td>-863.1</td><td>-937.3</td><td>-1370.0</td><td>-1625.0</td></tr><tr><td>Overhead</td><td></td><td>13.4</td><td>87.2</td><td>124.0</td><td>&lt;0.001</td><td>&lt;0.001</td></tr><tr><td>Total↓</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>∆FLOPs (%)↓</td><td>4437.7</td><td>3427.1 -22.8</td><td>3661.8 -17.5</td><td>3624.4 -18.3</td><td>3067.7 -30.9</td><td>2812.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>-36.6</td></tr><tr><td>Latency (ms)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SpConv Attention</td><td>188.1</td><td>156.9</td><td>161.6</td><td>188.0</td><td>131.0</td><td>120.2</td></tr><tr><td>FFN + others</td><td>340.6</td><td>301.9</td><td>304.2</td><td>184.4</td><td>272.0</td><td>261.9</td></tr><tr><td></td><td>171.9</td><td>149.7</td><td>163.6</td><td>173.3</td><td>144.2</td><td>139.6</td></tr><tr><td>Overhead</td><td></td><td>40.0</td><td>11.6</td><td>136.4</td><td>2.6</td><td>4.8</td></tr><tr><td>Scoring</td><td></td><td>9.1</td><td>8.6</td><td>41.1</td><td>0.3</td><td>0.5</td></tr><tr><td>Gather/reduce</td><td></td><td>30.5</td><td>2.8</td><td>89.8</td><td>0.4</td><td>0.8</td></tr><tr><td>Scatter</td><td></td><td>0.4</td><td>0.2</td><td>5.5</td><td>0.1</td><td>0.2</td></tr><tr><td>Sort</td><td>一</td><td>一</td><td>一</td><td>一</td><td>1.9</td><td>3.4</td></tr><tr><td>Latency↓</td><td>700.6</td><td>648.4</td><td>641.0</td><td>682.0</td><td>549.9</td><td>526.5</td></tr><tr><td>Speedup (×)↑</td><td>1.00</td><td>1.08</td><td>1.09</td><td>1.03</td><td>1.27</td><td>1.33</td></tr><tr><td>Peak memory (MiB)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Weights</td><td>602</td><td>602</td><td>602</td><td>602</td><td>602</td><td>602</td></tr><tr><td>Activation</td><td>758</td><td>934</td><td>817</td><td>756</td><td>776</td><td>784</td></tr><tr><td>Cache</td><td></td><td>一</td><td></td><td></td><td>48</td><td>69</td></tr><tr><td>Peak↓</td><td>1360</td><td>1535</td><td>1419</td><td>1358</td><td>1426</td><td></td></tr><tr><td>∆Peak (%)↓</td><td></td><td>12.9</td><td>4.3</td><td>-0.1</td><td>4.8</td><td>1455 7.0</td></tr></table>