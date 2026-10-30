# Joint Rigid Alignment and Morphable Reconstruction from Partial 3D Face Scans

Eli A. Underwood<sup>1</sup> Nicolas Rondan<sup>2</sup> Angel V. Peterchev<sup>1</sup> J. Matias Di Martino<sup>1,2</sup> <sup>1</sup>Duke University <sup>2</sup>Universidad Catolica del Uruguay´

## Abstract

Aligning a partial, real-world 3D face scan to a canonical representation requires solving two coupled problems at once: estimating the rigid pose that brings the scan into a common frame, and completing the facial shape in regions the scan never observed. Existing pipelines typically decouple these steps, running a generic point-cloud registration front endfollowed by a separate non-rigidfitting or deformation stage. Building on the GeoTransformer superpointmatching architecture, we propose a single network that predicts both a rigidpose and a shape-complete point cloud in one forward pass: a cross-attention regressor, conditioned on transformer point features and predicted coarse correspondences, maps a partial scan to per-patch coefficients ofa 3D morphableface model learned offlinefrom a corpus of 8,000 identities. The resulting morphed template is then used, in the same forward pass, as the target for fine-grained Sinkhorn matching and rigid pose estimation, supervised purely by geometric alignment to ground-truth morphed geometry, so no hand-labeled coefficients are required. Against six classical and learned non-rigid registration baselines on held-out identities, our method leads the strongest ones by 3–4× on per-vertex error and baselines that collapse the head onto the visible region by more than an order of magnitude, and is simultaneously the most accurate and fastest method on ground-truth-free scans from two independent real-scan datasets, NoW and NPHM. Code will be made publicly available.

## 1. Introduction

3D scanners produce face geometry cheaply and at scale, but the resulting scans are rarely complete. A handheld or single-shot sensor typically observes one side of the face, misses the ears and much of the cranium, and returns a raw point cloud in an arbitrary, unknown pose. Turning such a scan into something usable for identity comparison, anthropometric measurement, prosthetic design, or animation retargeting requires solving two problems simultaneously. First, a rigid problem: recovering the rotation and translation that bring the scan into a canonical reference frame. Second, a non-rigid problem: completing the facial shape in the parts of the face the scan never saw, using prior knowledge of what faces look like, whether through a generic nonrigid solver or an iterative fit of a 3D morphable face model against dense correspondences.

We find this generic non-rigid prior is not just suboptimal but actively harmful on partial face scans. Benchmarking a representative set of non-rigid registration methods on partial views of a face database (Sec. 4), the majority satisfy their smoothness regularizer by collapsing the head onto the visible region (Fig. 1) rather than completing the occluded geometry, reaching per-vertex ground-truth error several times worse than methods that actually complete the shape.

Jointly predicting a rigid pose and a complete shape is not new, and neither is driving that completion with a 3D morphable face model. What is new here is coupling the two through superpoint-level, nonlocal geometric matching, rather than treating rigid alignment and morphable-model fitting as two stages where one waits on the other. We condition the morphable-coefficient regressor directly on Geo-Transformer’s coarse superpoint features and predicted correspondences, so the same self- and cross-attention representation that establishes where the scan overlaps the template also drives what facial shape fills the rest of it, and gradients from the completion signal flow back into that matching stage itself. Specifically (Fig. 2), a KPConv [21] backbone and GeoTransformer [15] produce coarse superpoint correspondences between the scan and a fixed meanface template. Then, a cross-attention regressor predicts per-patch coefficients of a 3D morphable model [3] from the pre- and post-transformer point features together with the predicted correspondences, deforming the template into a scan-specific, shape-complete morphed reference. This morphed reference is subsequently used, within the same forward pass, as the fine-matching target for Sinkhorn optimal transport [19] and rigid pose estimation. Because the morphed template is a full, complete point cloud rather than a partial, noisy scan, morphing is supervised directly by dense geometric alignment to ground-truth morphed geometry, so no hand-labeled morphable-model coefficients are required. Gradients from this geometric signal flow back through the transformer and backbone, letting the coarse matching stage itself become shape-aware.

## Contributions.

1. An end-to-end architecture that jointly predicts a 3D morphable face model’s coefficients and rigid pose from a single partial scan in one forward pass, sharing geometric features between the coarse-matching, morphing, and fine-matching stages rather than treating them as independent modules. Because completion is conditioned on GeoTransformer’s superpoint correspondences, the network localizes and adapts to whatever geometry is actually missing on a per-scan basis, while the same superpoint-level matching keeps global rigid alignment robust.

2. A cross-attention morphable-coefficient regressor, conditioned on pre-transformer backbone features, posttransformer matched features, and the predicted coarse correspondences, trained with a dense geometric loss against ground-truth morphed geometry rather than hand-labeled coefficients.

3. A short, scan-referenced flow-field refinement, warmstarted from our end-to-end prediction, that closes the remaining surface-fit gap in the visible region while leaving occluded geometry governed by the learned prior.

4. A rigorous evaluation protocol pairing per-vertex ground-truth error with Chamfer distance on a held-out validation set, exposing a critical collapse failure mode in standard non-rigid registration baselines that Chamfer distance alone conceals. We extend this evaluation to two externally sourced real-scan datasets, the Notquite-in-the-Wild (NoW) benchmark [18] and a subset from the Neural Parametric Head Models (NPHM) [7]. Across all settings, our method avoids the collapse failure and outperforms all baselines without external rigid alignment.

## 2. Related work

Rigid point cloud registration. Classical rigid registration alternates correspondence estimation and closed-form pose solving, from Iterative Closest Point (ICP) [2] to feature-based Random Sample Consensus (RANSAC) [4] pipelines built on descriptors such as Fast Point Feature Histograms (FPFH) [17]. Learned methods replace one or both stages with neural correspondence estimation: Predator [8] learns overlap-aware keypoint detection for lowoverlap scenes, while GeoTransformer [15] learns geometric self- and cross-attention over superpoints so accurate correspondences can be found without a RANSAC step. Dual Focus-Attention Transformer (DFAT) [6] focuses attention within local neighborhoods at both coarse and fine scales and Regor [24] regenerates rather than merely filters correspondences under high outlier ratios.

![](images/e47e283bfe7317c009a337852f4f0596afc1c6e01bc41b8e60c37f0b98967bdc.jpg)  
Figure 1. Mean-face template and partial validation scan (top), with the morphed references produced by three representative nonrigid registration baselines—OAReg [26], ClusterReg [25], and CPD [11]—and our method (bottom). The baselines collapse the head onto the visible region, whereas our method recovers a plausible full head from the same partial input.

Non-rigid registration and surface fitting. Non-rigid registration extends the rigid setting with a per-point or per-node deformation field, typically regularized for local smoothness or as-rigid-as-possible behavior. Optimalstep non-rigid ICP (NRICP) [1] and Coherent Point Drift (CPD) [11] are foundational classical approaches. More recent work replaces the deformation model or correspondence step with learned components. GraphSCNet [16] prunes outlier correspondences using local spatial consistency over a deformation graph before solving with embedded deformation. Test-time-optimization methods forgo learned priors entirely: Neural Scene Flow Prior (NSFP) [10] fits an unregularized flow MLP against a Chamfer objective, OAReg [26] adds an occlusion-aware correntropy term and a local linear-rigidity regularizer, and ClusterReg [25] reformulates registration as unsupervised clustering with a closed-form displacement field. None encode a prior over what a face specifically looks like, as their regularizers constrain local rigidity or smoothness, not global plausibility; Sec. 4 shows this matters, since smoothness or rigidity regularization alone does not prevent global shape collapse on partial face scans. Our refinement (Sec. 3.7) reuses Neural Scene Flow Prior’s [10] unregularized, per-scan flow-field optimizer directly, but warmstarted from our learned prediction rather than a generic initialization, improving accuracy with far fewer steps to

3D morphable face models. 3D morphable models (3DMMs) represent face and head shape as a lowdimensional basis fit to a corpus of registered scans, from the original 3DMM [3] and the Basel Face Model [12] to expressive models such as FLAME [9] and large-scale faceand-head models [13]. These are most commonly fit to an input via iterative optimization, or regressed from a single RGB image by a learned encoder [18], typically assuming a roughly frontal view and a known pose. Our regressor instead operates on 3D point features and jointly recovers the unknown SE(3) pose alongside completion. We instead use a morphable model as the target representation for a registration network. Concretely, the decoder regresses local, patch-wise coefficients directly from geometric point features, and the morphed points are used, in the same forward pass, as the target for classical fine-grained point matching, coupling morphable-model fitting to registration rather than treating it as a downstream step.

Learned shape completion priors. Generic point-cloud completion networks [22, 23] learn to hallucinate missing geometry from partial input, typically trained per-category on synthetic shape collections such as ShapeNet. Our regressor is a domain-specialized instance of this idea: rather than a generic latent shape code, the completion target is constrained to the linear span of a face-specific PCA basis, which regularizes the output to always be a plausible face and makes the completion interpretable as anatomically localized coefficients. Section 4 shows this domain-specific prior is precisely what separates methods that complete occluded facial geometry from methods that merely avoid destroying the visible region.

## 3. Method

Given a partial, arbitrarily-posed 3D point cloud ${ \textbf { S } } \in$ $\mathbb { R } ^ { N _ { s } \times 3 }$ of a face $( N _ { s }$ vertices), our goal is to jointly recover (i) the rigid transform $( R , t ) \in S E { ( 3 ) }$ that aligns S to a canonical frame, and (ii) a shape-complete point cloud M<sup>ˆ</sup> , in the same fixed topology as a mean-face template $\bar { \mathbf { M } } \in \mathbb { R } ^ { N _ { v } \times 3 } \left( N _ { v } \right.$ vertices), representing the scanned individual’s full facial geometry. Figure 2 gives an overview. Both the rigid pose (R, t) and the completed point cloud M<sup>ˆ</sup> come from a single network in one forward pass, conditioned on the same geometric point features, rather than independent rigid and non-rigid stages.

## 3.1. Preliminaries: a collection of local morphable patch models

Our shape-completion signal comes from a linear morphable basis, fit offline to a corpus of N=8,000 synthetic head identities from the Universal Head Model (UHM) model [5, 14] sharing the mean template’s fixed topology $( N _ { v } { = } 1 0 , 7 8 8$ vertices). Rather than fit one global basis over the whole head, which would let a coefficient’s effect spread across the entire face and make partial, patch-local evidence hard to exploit, we build P=32 independent local models, one per overlapping patch of M<sup>¯</sup> . This decomposition is what makes the coarse superpoint correspondences from Sec. 3.2’s GeoTransformer usable as completion evidence in the first place: because each patch has its own basis, Sec. 3.3’s regressor can localize completion to the parts of the facial shape that a given patch’s matched superpoints actually constrain, rather than entangling every prediction with the whole head at once. Patch p is the set of $K _ { n } { = } 7 0 0$ nearest template vertices to a fixed anchor, so a vertex may belong to several patches. Let $\mathbf { x } _ { p } ^ { ( n ) } \in \mathbb { R } ^ { 3 K _ { r } }$ be the flattened positions of patch p’s vertices in corpus identity n. We compute the patch mean $\begin{array} { r } { \mu _ { p } = \frac { 1 } { N } \sum _ { n } \mathbf { x } _ { p } ^ { ( n ) } } \end{array}$ and, by PCA over $\{ \mathbf { x } _ { p } ^ { ( n ) } - \mu _ { p } \} _ { n }$ , an orthonormal basis $\mathbf { B } _ { p } \in \mathbb { R } ^ { K \times 3 K _ { n } }$ of the leading K directions of shape variation $( K { = } 3 6 ;$ Tab. 1). A patch is reconstructed from coefficients $z _ { p } \in \mathbb { R } ^ { K }$ by

Algorithm 1 Forward pass: scan $\mathbf { \Psi }  ( \hat { \mathbf { M } } , \hat { R } , \hat { t } )$   
Require: partial scan S, template M<sup>¯</sup>   
1: Extract backbone + GeoTransformer features for M<sup>¯</sup> and S ▷   
Sec. 3.2   
2: Coarsely match superpoints between M<sup>¯</sup> and S ▷ Sec. 3.2   
3: Regress per-patch coefficients z from matched features ▷   
Sec. 3.3   
4: Reconstruct full point cloud $\hat { \mathbf { M } } = g ( z )$ ▷ Eq. (1)   
5: Update reference positions at all backbone levels to M<sup>ˆ</sup> ▷   
Sec. 3.4   
6: Fine-match via Sinkhorn $\mathrm { O T + L G R }$ on M<sup>ˆ</sup> to get $( \hat { R } , \hat { t } )$ ▷   
Sec. 3.4   
7: return M<sup>ˆ</sup> , $\hat { R } , \hat { t }$

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { p } = \mu _ { p } + z _ { p } ^ { \top } \mathbf { B } _ { p } , } \end{array}\tag{1}
$$

reshaped to $K _ { n } \times 3$ vertex positions. Assembling all P patches into a full point cloud $g ( z ) \in \mathbb R ^ { N _ { v } \times 3 }$ is a fixed, deterministic gather with no learned parameters. Each vertex takes its position from a designated covering patch. g is linear in z, so predicting the facial shape in occluded and visible regions alike reduces to predicting $z \in \mathbb { R } ^ { P \times K }$ , and Sec. 3.3 describes the network that regresses z from a partial scan. Algorithm 1 summarizes how this basis, the regressor, and the morphing step of Sec. 3.4 fit together in the full forward pass.

## 3.2. Backbone and coarse matching

We build a k-nearest-neighbor graph jointly over the raw mean-face template M<sup>¯</sup> and the raw source scan S and encode both with a shared KPConv feature-pyramid backbone [21] at L=4 resolution levels, giving per-point features at a fine (Stage-1) and coarse (Stage-4) resolution for both. The coarsest superpoint features are refined by a GeoTransformer [15], applying geometric self- and cross-attention (pairwise distances, embedded with Gaussian bandwidth $\sigma _ { d } .$ , and triplet angles, both invariant to the unknown rigid pose) within and between the point sets, producing posttransformer features ${ \bf F } _ { c } ^ { \mathrm { r e f } } , { \bf F } _ { c } ^ { \mathrm { s r c } }$ . A differentiable superpoint matching module then produces coarse correspondences from these features, supervised by a circle loss [20] against ground-truth overlap (Sec. 3.5).

![](images/1ab08ef71ee3f29d4dca518f9b7db9756bbe4b3ee1f9b7f7e810075af919afb4.jpg)  
Figure 2. Pipeline overview. A shared KPConv (Kernel Point Convolution) backbone and GeoTransformer produce coarse superpoint correspondences between the mean-face template and a partial scan; a cross-attention regressor predicts per-patch morphable-basis coeffi cients z (conditioned on post-transformer features, pre-transformer backbone features, and the coarse correspondences), which are decoded into a morphed template that updates the reference positions used by Sinkhorn optimal-transport (OT) fine matching and local-to-globa registration (LGR) to produce the estimated pose (R,<sup>ˆ</sup> t<sup>ˆ</sup>). Bottom: detail of the coefficient regressor, showing its various input feature sources and the per-patch decoding into z.

## 3.3. Morphable-coefficient regression

Coarse correspondences alone identify where the scan overlaps the template, but they say nothing about the occluded facial shape. We introduce a cross-attention regressor that predicts this separately, holding P learnable patch query tokens that cross-attend, via a 2-layer Transformer decoder, to a memory built from three feature sources: the post-transformer template and scan superpoint features $( \mathbf { F } _ { c } ^ { \mathrm { r e f } } , \mathbf { F } _ { c } ^ { \mathrm { s r c } } )$ , the pre-transformer scan backbone features F<sup>src</sup><sub>bb</sub> (raw local geometry, before it is entangled with the template), and the features at the predicted coarse correspondences $\mathbf { F } _ { \mathrm { c o r r } } ^ { \mathrm { r e f } } , \mathbf { F } _ { \mathrm { c o r r } } ^ { \mathrm { s r c } }$ (which parts of the scan are actually observed). A linear head maps each updated patch token to coefficients $z _ { p } \in \mathbb { R } ^ { K }$ (Sec. 3.1). Conditioning on both preand post-transformer features alongside the explicit correspondence set lets the regressor decouple direct geometric transfer in visible regions from learned completion in occluded areas.

## 3.4. Morphing the reference and fine matching

The predicted coefficients reconstruct a morphed template $\hat { \textbf { M } } = \ g ( z ) \ \in \ \mathbb { R } ^ { N _ { v } \times 3 }$ (Secs. 3.1 and 3.3). Because the pooled backbone points at each stage are exact copies of vertices of M<sup>¯</sup> , but KPConv pooling does not retain which vertex index each pooled point came from, we recover this correspondence via an exact-match nearest-neighbor lookup against the ordered vertex set of M<sup>¯</sup> , yielding a fixed index for every coarse- and fine-level template point. Since M<sup>ˆ</sup> preserves the same global vertex ordering as M<sup>¯</sup> , we use these indices to directly gather the morphed positions from M<sup>ˆ</sup> , so the morphed geometry, rather than the mean face, is what fine matching operates on. Groundtruth coarse correspondences for the circle loss are likewise computed after morphing. Fine matching then follows GeoTransformer’s local-to-global registration [15]: pointto-point scores within each coarse-matched patch pair are normalized with a differentiable Sinkhorn optimal-transport layer [19], dense correspondences are extracted, and a robust local-to-global procedure aggregates per-patch rigid hypotheses into (R,<sup>ˆ</sup> t<sup>ˆ</sup>). The rigid pose is thus estimated against a target that already reflects the network’s best guess at the scanned individual’s facial shape, not just the population mean.

## 3.5. Losses

The network is trained with a sum of three terms: a circle loss on coarse correspondences [20], a matching loss on the fine-level Sinkhorn scores (positive pairs pulled toward their ground-truth match, unmatched rows/columns pulled toward the dustbin), and a morph supervision term. Our generated training pairs (Sec. 3.6) provide full ground-truth morphed geometry, not just a partial, noisy scan, so the morph term is a direct, dense per-vertex loss,

$$
\mathcal { L } _ { \mathrm { m o r p h } } = \frac { 1 } { N _ { v } } \sum _ { i = 1 } ^ { N _ { v } } \left. \hat { \mathbf { M } } _ { i } - \mathbf { M } _ { i } ^ { \mathrm { g t } } \right. _ { 2 } ,\tag{2}
$$

where $\mathbf { M } ^ { \mathrm { g t } } = g ( z ^ { \mathrm { g t } } )$ is the full ground-truth morphed template, weighted ten times relative to the coarse and fine losses, dominating the training signal for the decoder. This is a geometric loss rather than a coefficient-regression loss. Because z is never supervised directly, gradients flow from a shape-level alignment signal back through Eq. (2) into the regressor and, because the memory it attends to includes post-transformer features, into the transformer and backbone as well.

## 3.6. Training data

Ground-truth coefficients and rigid transforms are not avail able for scans outside our own generated corpus, so we train entirely on pairs generated from the corpus of Sec. 3.1. The reference is the fixed mean face and per-patch bases computed over the full 8,000-identity corpus. Of those 8,000, $7 { , } 5 0 0$ generate training pairs (2 views each) and 250 are held out for evaluation (Sec. 4.1); the remaining 250 contribute only to the shared mean and per-patch bases, not to any pair. The source is a partial view of one identity, drawn with equal probability from one of two crop modes: Face-Cut, an axis-aligned crop with randomized bounds, or Self-Occlusion, a hidden-point-removal crop of the full head from a random viewpoint. A random SO(3) rotation and translation are applied to either crop mode, simulating an arbitrarily-posed handheld or single-shot scan. At train time only, we apply additive Gaussian noise (±0.005 per axis), a random subsample (keep ratio ∼ U(0.5, 1.0)), and a voxel-density normalization pass to better match real scanner density irregularity. Ground-truth patch coefficients $z ^ { \mathrm { g t } }$ are precomputed once per identity. The ground-truth transform is the actual (R, t) needed to bring the source crop back into the reference frame. Table 1 summarizes the resulting configuration.

## 3.7. Scan-referenced refinement

The forward pass above (Secs. 3.2 to 3.4) never looks at the scan surface directly when producing M<sup>ˆ</sup> : M<sup>ˆ</sup> is a function of learned coarse features, constrained to the morphablemodel basis, and is never pulled toward the scan by any perpoint data term. This is exactly what keeps it from the collapse failure of smoothness-regularized baselines (Sec. 4.4), but it also means M<sup>ˆ</sup> does not exploit fine, scan-specific surface detail that is available in the visible region and that a direct scan-fitting objective would recover.

<table><tr><td>Component</td><td>Value</td></tr><tr><td>Backbone</td><td>voxel 0.025, 4 stages</td></tr><tr><td>Transformer  $\sigma _ { d } ~ ( \mathrm { t r a i n } )$ </td><td>0.16</td></tr><tr><td>Coarse targets / correspondences</td><td>128 / 256</td></tr><tr><td>Coarse / fine matching radius</td><td>0.05 / 0.1</td></tr><tr><td>Sinkhorn iterations</td><td>100</td></tr><tr><td>Morphable patches × components</td><td> $3 2 \times 3 6$ </td></tr><tr><td>Loss weight (morph : coarse : fine)</td><td> $1 0 : 1 : 1$ </td></tr><tr><td>Optimizer</td><td> $\scriptstyle \mathrm { A d a m , l r } \ 1 0 ^ { - 4 } , \ x 0 . 9 5 / 1 0 \ \mathrm { e p } .$ </td></tr><tr><td>Weight decay / grad. clip</td><td> $1 0 ^ { - 4 } ( { \mathrm { e x c l . r e g r e s s o r } } ) / 1 . 0 ^ { \cdot }$ </td></tr><tr><td>Epochs / batch size</td><td> $5 0 0 / 8 { \times } 4 \mathrm { G P U s } ( \mathrm { e f f . ~ } 3 2 )$ </td></tr><tr><td>Noise / subsample aug.</td><td> $\pm 0 . 0 0 5 / \mathrm { k e e p } \sim \mathcal { U } ( 0 . 5 , 1 )$ </td></tr><tr><td>Voxel-density aug.</td><td> $\mathsf { d e d u p } v \in \{ 0 . 0 3 , \ldots , 0 . 0 6 \}$ </td></tr></table>

Table 1. Key configuration for the checkpoint used in all reported results. At real-scan (NoW and NPHM) inference only, $\sigma _ { d }$ is scaled by a fixed ×1.2 correction factor.

Our pipeline therefore closes with a short test-time refinement that recovers this detail without giving up the completion above. It takes the forward pass’s rigid pose (R,<sup>ˆ</sup> t<sup>ˆ</sup>) and morphed template M<sup>ˆ</sup> as initialization for a short run of a NSFP-style [10] per-scan flow-field optimization against the raw scan, the same unregularized flow-MLP objective NSFP uses from scratch, but warm-started from our prediction instead of the population mean and an ICP-derived pose. Because the flow field is fit against the observed scan, it can only meaningfully move points the scan actually constrains. In the occluded region, where no scan data exists to pull it anywhere, the geometry stays close to M<sup>ˆ</sup> , so the refinement inherits our shape-completion behavior there rather than overriding it. Unlike running NSFP from scratch, this refinement never pays for an external rigid-ICP pre-alignment stage (Sec. 4.2), so far fewer optimization steps are needed to converge. Sec. 4.7 isolates the forward pass alone.

## 4. Experiments

## 4.1. Datasets and protocol

We evaluate on three datasets: our own synthetic validation set, with ground truth, and two independent, external real-scan datasets, without ground truth. Validation (val) is drawn from a 250-identity pool of UHM-sampled heads [5, 14] held out from training, each identity rendered with two partial views (500 pairs total: 266 Self-Occlusion / 234 Face-Cut, grouped by the crop mode actually sampled for each view, Sec. 3.6), each paired with the corresponding full ground-truth morphed mesh, enabling direct, alignment-independent per-vertex accuracy (Sec. 4.3). We report Self-Occlusion and Face-Cut separately, since occlusion crops have systematically more missing geometry. NoW is the test split of the NoW benchmark [18], comprising scans of real human subjects. We use these raw scans directly as partial inputs, a different protocol from the benchmark’s official image-based challenge; NoW’s reference scans, ground truth for that challenge, are not usable as shape-completion ground truth here, so we treat NoW as having none available and test generalization beyond the training distribution. NPHM is a second, independent realscan dataset drawn from the NPHM head dataset [7]: 271 identities, one scan each. NPHM’s full-head scans do include hair, incompatible with our hairless UHM-based template, so we cannot evaluate on the full head or use it as shape-completion ground truth; instead we take a frontal crop matching NoW’s framing, downsampled to 5k points, and evaluate the same way as NoW (Sec. 4.5). Evaluation on both NoW and NPHM is limited to Chamfer distance (CD) as an accuracy proxy and qualitative inspection. CD is a surface-fit metric, not a shape-completion metric, a distinction Sec. 4.5 returns to. Every baseline deforms the same fixed-topology template, so a point’s index is preserved through deformation, making both metrics directly comparable with no correspondence search.

<table><tr><td></td><td colspan="2">GT err (mm) ↓</td><td colspan="2">CD (mm) ↓</td><td></td></tr><tr><td>Method</td><td>Self-Occlusion</td><td>Face-Cut</td><td>Self-Occlusion</td><td>Face-Cut</td><td>Time (s) ↓</td></tr><tr><td>NRICP [1]</td><td> $3 1 . 8 6 \pm 1 6 . 5 6$ </td><td> $2 9 . 6 2 \pm 1 0 . 3 6$ </td><td> $3 . 8 7 \pm 9 . 2 7$ </td><td> $1 . 7 1 \pm 6 . 6 3$ </td><td>64.45</td></tr><tr><td>CPD [11]</td><td> $4 6 . 6 5 \pm 1 6 . 5 1$ </td><td> $4 4 . 9 7 \pm 1 0 . 1 6$ </td><td> $1 . 5 5 \pm 0 . 1 9$ </td><td> $1 . 4 7 \pm 0 . 1 5$ </td><td>39.95</td></tr><tr><td>NSFP [10]</td><td> $9 . 9 9 \pm 4 . 2 7$ </td><td> $8 . 8 8 \pm 3 . 9 2$ </td><td> $0 . 5 0 \pm 0 . 2 5$ </td><td> $0 . 3 1 \pm 0 . 2 0$ </td><td>15.40</td></tr><tr><td>GraphSCNet [16]</td><td> $8 . 5 6 \pm 3 . 7 5$ </td><td> $7 . 7 7 \pm 3 . 6 3$ </td><td> $2 . 0 6 \pm 1 . 4 2$ </td><td> $1 . 3 5 \pm 0 . 8 2$ </td><td>40.60</td></tr><tr><td>ClusterReg [25]</td><td> $4 2 . 7 1 \pm 1 7 . 0 3$ </td><td> $3 8 . 2 2 \pm 1 1 . 0 2$ </td><td> $0 . 8 7 \pm 0 . 2 8$ </td><td> $0 . 9 1 \pm \ : 0 . 1 8$ </td><td>41.40</td></tr><tr><td>OAReg [26]</td><td> $3 6 . 6 7 \pm 1 1 . 6 3$ </td><td> $3 4 . 9 2 \pm 7 . 8 9$ </td><td> $2 . 1 5 \pm 0 . 8 8$ </td><td> $1 . 2 8 \pm 0 . 6 4$ </td><td>14.25</td></tr><tr><td>Ours</td><td> $\mathbf { 2 . 4 2 \ : \pm 1 . 0 5 }$ </td><td> ${ \bf 2 . 1 2 \pm 0 . 6 7 }$ </td><td> $\mathbf { 0 . 0 8 \ : \pm { \ : 0 . 0 4 } }$ </td><td> $\mathbf { 0 . 0 8 \ : \pm { \ : 0 . 0 2 } }$ </td><td>2.53</td></tr></table>

Table 2. Val set (250 identities × 2 views = 500 pairs: 266 Self-Occlusion / 234 Face-Cut pairs), GT vertex error and 1-way CD (partial scan→deformed-template, visible region only, same definition as Tab. 3), median ± std over pairs, all methods timed on the same hardware. Ours leads every column. Four of six baselines (NRICP, CPD, ClusterReg, OAReg) have GT error an order of magnitude worse than Ours despite ClusterReg posting some of the best CD in the table alongside 38–47mm GT error: CD’s blind spot to implausible completions, visible in single rows (Sec. 4.3).

## 4.2. Baselines

We compare against six non-rigid registration methods spanning classical optimization, correspondence-free clustering, test-time optimization, and learned correspondence pruning: NRICP [1], CPD [11], OAReg [26], Cluster-Reg [25], NSFP [10], and GraphSCNet [16]. None predicts pose jointly with shape, so each requires a rigid prealignment stage before it can run at all. We give every one the same strong initialization, a 24-start octahedral multistart point-to-point ICP, and report that cost as part of each baseline’s time in Tabs. 2 and 3. Ours predicts (R, t) end to end in one forward pass and never invokes this or any external rigid-alignment stage, including during its own refinement (Sec. 3.7). We do not include iterative 3DMM-fitting baselines: without a known or landmark-derived pose initialization, unavailable for an arbitrarily-posed raw scan, iterative fitting has no starting point short of the same external rigid-alignment stage our method avoids.

## 4.3. Metrics

Vertex error. Direct per-vertex ground-truth error $\begin{array} { r } { \frac { 1 } { N _ { v } } \sum _ { i } \lVert \hat { \mathbf { M } } _ { i } - \mathbf { M } _ { i } ^ { \mathrm { g t } } \rVert } \end{array}$ , exploiting the shared template topology for exact correspondence (no nearest-neighbor search). We report the median over pairs within each crop category rather than the mean to mitigate any potential outlier biases across baselines.

Chamfer distance (CD). We report 1-way CD from the partial input scan to the deformed template, restricted to the visible region the scan actually covers (Tabs. 2 and 3). CD re-searches nearest neighbors after deformation, so a method that collapses or flattens the whole shape onto the visible region can still post a good CD by landing near some scan point post-hoc, independent of whether the rest of the shape is plausible. Rather than filtering methods on a separate structural-validity metric before reporting accuracy, we report GT error and CD side by side for every method in the same table (Tab. 2): Sec. 4.4 shows several methods with competitive CD are simultaneously the worst by GT error, by an order of magnitude.

## 4.4. Results: accuracy and Chamfer’s blind spot

On the val set, ClusterReg and CPD, generic smoothness or correspondence-free regularizers with no face-specific prior, reach 38–47mm GT vertex error; ClusterReg in particular posts some of the lowest CD in the table despite this, while CPD’s CD is merely competitive, not best-in-table. NRICP and OAReg show the same pattern less extremely. This is not four separate failures: inspecting the geometry directly (Fig. 1) shows OAReg, ClusterReg, and CPD collapsing the head toward the visible region, which their regularizers have no mechanism to penalize, while CD only ever asks whether there is some nearby scan point. NSFP and GraphSCNet avoid outright collapse but still trail Ours by 3–4× on GT error, since their occluded-region behavior defaults to a generic guess rather than our learned, shapespecific completion (Sec. 3.3). Ours leads every baseline on GT error, CD, and time simultaneously. Sec. 4.7 shows the

<table><tr><td>Method</td><td>CD (mm) ↓</td><td>Time (s) ↓</td></tr><tr><td>NRICP [1]</td><td> $1 . 6 9 \pm 0 . 1 8$ </td><td>31.35</td></tr><tr><td>CPD [11]</td><td> $1 . 5 0 \pm 0 . 1 2$ </td><td>22.50</td></tr><tr><td>NSFP [10]</td><td> $0 . 6 1 \pm 0 . 0 5$ </td><td>11.80</td></tr><tr><td>GraphSCNet [16]</td><td> $1 . 6 1 \pm 0 . 0 9$ </td><td>37.45</td></tr><tr><td>ClusterReg [25]</td><td> $1 . 0 5 \pm 0 . 0 6$ </td><td>23.05</td></tr><tr><td>OAReg [26]</td><td> $1 . 6 1 \pm 0 . 1 4$ </td><td>10.90</td></tr><tr><td>Ours</td><td> $\mathbf { 0 . 5 9 \ : \pm { \ : 0 . 0 4 } }$ </td><td>2.26</td></tr></table>

Table 3. NoW benchmark test split, no ground truth: 1-way CD (partial scan→deformed-template) and end-to-end wall-clock time, median ± std over the test scans, all methods timed on the same cluster hardware. CD uses each scan’s own recovered scale factor rather than a fixed dataset-wide constant (Sec. 4.3). Ours leads on both CD and time.

<table><tr><td>Method</td><td>CD (mm) ↓</td><td>Time (s) ↓</td></tr><tr><td>NRICP [1]</td><td> $2 . 1 1 \pm 1 . 1 2$ </td><td>24.60</td></tr><tr><td>CPD [11]</td><td> $1 . 5 8 \pm 0 . 1 4$ </td><td>20.70</td></tr><tr><td>NSFP [10]</td><td> $0 . 8 0 \pm 0 . 1 1$ </td><td>13.60</td></tr><tr><td>GraphSCNet [16]</td><td> $1 . 6 8 \pm 0 . 3 5$ </td><td>20.10</td></tr><tr><td>ClusterReg [25]</td><td> $1 . 1 4 \pm 0 . 0 9$ </td><td>25.40</td></tr><tr><td> $\mathrm { O A R e g } \left[ 2 6 \right]$ </td><td> $1 . 5 6 \pm 0 . 3 4$ </td><td>12.60</td></tr><tr><td>Ours</td><td> $\mathbf { 0 . 7 2 \pm 0 . 0 4 }$ </td><td>2.08</td></tr></table>

Table 4. NPHM head dataset [7], face-only crops (full-head scans include hair, incompatible with our hairless template, so not usable as shape-completion ground truth): 1-way CD (partial scan→deformed-template) and end-to-end wall-clock time, median ± std over 271 scans, all methods timed on the same cluster hardware (a different cluster from Tabs. 2 and 3, so timings are comparable within this table but not across tables).

GT-error and time advantages already hold for the forward pass alone, before refinement is applied.

## 4.5. External dataset results

At real-scan (NoW and NPHM) inference only, we scale $\sigma _ { d }$ by a fixed ×1.2 correction factor (Tab. 1): real scans have different point density than our synthetic training distribution even after unit-scale rescaling, and this factor was set by directly comparing real scans’ and the template’s point density during preprocessing.

On NoW, Tab. 3 shows Ours is simultaneously the most accurate and fastest method in the table: it leads NSFP, the strongest baseline, on CD (0.59mm vs. 0.61mm), in roughly a fifth of NSFP’s time (2.26s vs. 11.80s), never paying NSFP’s own rigid-ICP tax.

In Ours, the forward pass alone trades tight scan-fit for a regularized, morphable-model-consistent facial shape rather than directly optimizing surface fit, and posts a substantially worse raw CD without it (Sec. 4.7). NSFP, by contrast, is a direct scan-fit optimizer initialized from an ICP pose, and wins CD for that reason regardless of whether its occluded-region completion is any good, which Tab. 2 shows it is not. Warm-starting the same objective from our own prediction closes this gap: the visible region snaps tightly to the scan surface, while the occluded region, which

![](images/66b3228270ef290c5423f53fde66ddca604da7957d7dff8758ab015537ab8402.jpg)

Figure 3. Accuracy (1-way CD, partial scan→deformed-template, median) vs. wall-clock time (log scale, includes rigid-ICP prealignment for every baseline that requires it) across each method’s own iteration/step budget, NoW benchmark test split. Baselines: NRICP [1], CPD [11], NSFP [10], GraphSCNet [16], Cluster-Reg [25], OAReg [26].

the refinement has no scan signal to correct, stays governed by our learned prior.

To check that our NoW results are not specific to that one benchmark, we repeat the real-scan evaluation on NPHM (Sec. 4.1). Table 4 shows the same pattern as Tab. 3: Ours leads every baseline on CD while being the fastest method by a wide margin.

## 4.6. Qualitative results

Figure 4 inspects individual Val, NoW, and NPHM examples directly. Each row is an independent example. The Morph column shows how far each row’s morphed reference (light blue) departs from the fixed template (dark blue). The Alignment column overlays that same morphed reference with the row’s input scan (orange) to show surface-fit quality.

## 4.7. Ablation Studies

Contribution of the refinement. The forward pass alone, before refinement, already leads every baseline on GT error and by two to three orders of magnitude on time (2.71/2.13mm at 0.12s per scan on Val), confirming the accuracy gain in Sec. 4.4 comes from the learned completion itself, not the refinement. Its contribution shows up in CD instead. Without it, raw CD on NoW is 2.29mm, worse than every method in Tab. 3, because the forward pass never fits the observed surface directly, by design (Sec. 3.7); adding it brings CD to 0.59mm (Tab. 3) at a cost of roughly two

![](images/8086da8647550989b0fa55ff2a3ad95e7cd94224096913e4dce8778156f7cada.jpg)  
Figure 4. Qualitative results on Val (left), NoW (middle), and NPHM (right) examples. Rows: Self-Occlusion / Face-Cut crops (Val) and two representative scans each for NoW and NPHM. Columns: the partial input scan; our morphed reference alone, viewed from the side to show its departure from the fixed template (dark blue); and the same morphed reference (light blue) overlaid with the input scan (orange) to show surface-fit quality.

additional seconds per scan.

Accuracy vs. latency. Figure 3 shows every method across its own step or iteration budget on NoW rather than a single operating point. Ours’ single leftmost point is the forward pass alone, before any refinement steps, matching the 2.29mm raw CD reported above. From its cheapest refinement setting onward, Ours Pareto-dominates every baseline at every budget tested. Even 500 steps already beats every baseline’s own highest-budget point except NSFP’s, and pushing further, up to 10,000 steps, keeps improving CD with no plateau. This confirms Tab. 3’s 500-step point is a genuine trade-off, not a ceiling, as more test-time compute keeps paying off but from a strictly better start.

## 5. Discussion and limitations

Evaluating completion quality without ground truth. Ours closes the practical, scan-referenced accuracy gap on both NoW and NPHM (Sec. 4.5 and Tabs. 3 and 4), but scan-referenced surface fit and occluded-region completion quality are different things, and we have no directly usable ground truth on either real-scan dataset to check the latter (NoW’s own reference scans serve a different task; NPHM’s full-head scans include hair, incompatible with our hairless template). Usable ground truth for NoW or NPHM would let those results be reported with the same confidence as our Val results.

Basis capacity and dataset scale. Our morphable prior is a linear PCA basis fit to 8,000 identities, patch-partitioned to keep local anatomical detail (Sec. 3.1). This bias both enables the collapse-avoidance and accuracy in Sec. 4.4 and caps expressiveness at whatever the training population spans, so under-represented identities, expressions, or ages are unlikely to be well reconstructed. A learned, non-linear basis could relax this at the cost of the plausibility guarantee shown in Tab. 2.

Broader impact. This work uses only point geometry, no color, texture, or identifying imagery, though improving recovery of complete facial geometry from partial scans is nonetheless dual-use, like biometric and facereconstruction research generally. Training identities are synthetic, sampled from the published UHM model [14] via the generation procedure of [5], not collected from rea subjects; both are used under their published terms, as are NoW [18] and NPHM [7]. We see no immediate deployment risk here.

## 6. Conclusion

We presented a single network that jointly predicts a morphable-model-based facial-shape completion and rigid pose for partial 3D face scans, closing with a short scanreferenced refinement. Our evaluation protocol reports pervertex ground-truth error alongside Chamfer distance, exposing a collapse failure mode Chamfer distance alone conceals. Our method avoids this failure, leading held-out Val identities by 3–4× over the strongest baselines (Tab. 2) and both NoW’s and NPHM’s ground-truth-free scans in accuracy and speed (Tabs. 3 and 4).

## References

[1] Brian Amberg, Sami Romdhani, and Thomas Vetter. Optimal step nonrigid ICP algorithms for surface registration. In CVPR, 2007. 2, 6, 7

[2] Paul J. Besl and Neil D. McKay. A method for registration of 3-D shapes. IEEE TPAMI, 14(2):239–256, 1992. 2

[3] Volker Blanz and Thomas Vetter. A morphable model for the synthesis of 3D faces. In Proceedings ofSIGGRAPH, pages 187–194, 1999. 1, 3

[4] Martin A. Fischler and Robert C. Bolles. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications ofthe ACM, 24(6):381–395, 1981. 2

[5] Simone Foti, Bongjin Koo, Danail Stoyanov, and Matthew J. Clarkson. 3D shape variational autoencoder latent disentanglement via mini-batch feature swapping for bodies and faces. In CVPR, 2022. 3, 5, 8

[6] Kexue Fu, Mingzhi Yuan, Changwei Wang, Weiguang Pang, Jing Chi, Manning Wang, and Longxiang Gao. Dual focusattention transformer for robust point cloud registration. In CVPR, 2025. 2

[7] Simon Giebenhain, Tobias Kirschstein, Markos Georgopoulos, Martin Runz, Lourdes Agapito, and Matthias Nießner.¨ Learning neural parametric head models. In CVPR, 2023. 2, 6, 7, 8

[8] Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas Wieser, and Konrad Schindler. Predator: Registration of 3D point clouds with low overlap. In CVPR, 2021. 2

[9] Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4D scans. ACM TOG, 36(6), 2017. Article 194. 3

[10] Xueqian Li, Jhony Kaesemodel Pontes, and Simon Lucey. Neural scene flow prior. In NeurIPS, 2021. 2, 5, 6, 7

[11] Andriy Myronenko and Xubo Song. Point set registration: Coherent point drift. IEEE TPAMI, 32(12):2262–2275, 2010. 2, 6, 7

[12] Pascal Paysan, Reinhard Knothe, Brian Amberg, Sami Romdhani, and Thomas Vetter. A 3D face model for pose and illumination invariant face recognition. In Advanced Video and Signal Based Surveillance (AVSS), 2009. 3

[13] Stylianos Ploumpis, Haoyang Wang, Nick Pears, William A. P. Smith, and Stefanos Zafeiriou. Combining 3D morphable models: A large scale face-and-head model. In CVPR, 2019. 3

[14] Stylianos Ploumpis, Evangelos Ververas, Eimear O’Sullivan, Stylianos Moschoglou, Haoyang Wang, Nick Pears, William A. P. Smith, Baris Gecer, and Stefanos Zafeiriou. Towards a complete 3D morphable model of the human head. IEEE TPAMI, 2020. 3, 5, 8

[15] Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng, and Kai Xu. Geometric transformer for fast and robust point cloud registration. In CVPR, pages 11143–11152, 2022. 1, 2, 4

[16] Zheng Qin, Hao Yu, Changjian Wang, Yuxing Peng, and Kai Xu. Deep graph-based spatial consistency for robust nonrigid point cloud registration. In CVPR, 2023. 2, 6, 7

[17] Radu Bogdan Rusu, Nico Blodow, and Michael Beetz. Fast point feature histograms (FPFH) for 3D registration. In IEEE International Conference on Robotics and Automation (ICRA), 2009. 2

[18] Soubhik Sanyal, Timo Bolkart, Haiwen Feng, and Michael J. Black. Learning to regress 3D face shape and expression from an image without 3D supervision. In CVPR, 2019. 2, 3, 5, 8

[19] Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics, 21(2):343–348, 1967. 2, 4

[20] Yifan Sun, Changmao Cheng, Yuhan Zhang, Chi Zhang, Liang Zheng, Zhongdao Wang, and Yichen Wei. Circle loss: A unified perspective of pair similarity optimization. In CVPR, 2020. 4

[21] Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, Franc¸ois Goulette, and Leonidas J. Guibas. KPConv: Flexible and deformable convolution for point clouds. In ICCV, 2019. 1, 3

[22] Xumin Yu, Yongming Rao, Ziyi Wang, Zuyan Liu, Jiwen Lu, and Jie Zhou. PoinTr: Diverse point cloud completion with geometry-aware transformers. In ICCV, 2021. 3

[23] Wentao Yuan, Tejas Khot, David Held, Christoph Mertz, and Martial Hebert. PCN: Point completion network. In Interna tional Conference on 3D Vision (3DV), 2018. 3

[24] Guiyu Zhao, Sheng Ao, Ye Zhang, Kai Xu, and Yulan Guo. Progressive correspondence regenerator for robust 3D regis tration. In CVPR, 2025. 2

[25] Mingyang Zhao, Jingen Jiang, Lei Ma, Shiqing Xin, Gaofeng Meng, and Dong-Ming Yan. Correspondence-free non-rigid point set registration using unsupervised clustering analysis. In CVPR, 2024. 2, 6, 7

[26] Mingyang Zhao, Gaofeng Meng, and Dong-Ming Yan. Occlusion-aware non-rigid point cloud registration via unsupervised neural deformation correntropy. In ICLR, 2025. 2, 6, 7