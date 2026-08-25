# What Remains Normal? Clean Images Miss Useful Near-Defect Normal Patches for Anomaly Detection

Joongwon Chae<sup>1,3</sup> Runming Wang<sup>1</sup> Peiwu Qin<sup>2</sup>

<sup>1</sup>Institute of Biopharmaceutical and Health Engineering,

Tsinghua Shenzhen International Graduate School, Tsinghua University, Shenzhen, China

<sup>2</sup>Guangdong Provincial Laboratory of Traditional Chinese Medicine,

Hengqin, Guangdong, China

<sup>3</sup>RatelSoft

## Abstract

Normal-only industrial anomaly detectors use patches from clean training images as normal references or reconstruction targets. This assumes that clean patches are sufficient for the normal regions encountered at test time. We test that assumption directly. On MVTec AD, admitting ground-truth-normal patches from real defect images to a DINOv2 memory candidate pool raises pixel average precision (P-AP) from 73.34 to 76.95 while keeping the encoder, test-time score, and number of stored references fixed. Patches within two patch cells of the annotated defect recover 94.70% of this gain. We then ask whether useful patches of this kind can be exposed using clean training images alone. BOUNDARYSUPPORT inserts a procedural synthetic defect to alter surrounding context, excludes every token intersecting the nominal insertion or a detected RGB change, and learns only from pixel-preserved neighboring patches. Across three paired seeds, the same principle improves P-AP in all six memory/reconstruction–dataset settings across MVTec, VisA, and Real-IAD. Matched controls identify the altered-context feature itself as the useful normal evidence: with synthetic input or selected positions fixed, altered-context features outperform their clean-view counterparts as both reconstruction targets and memory references. On MVTec memory, the final score change is also spatially selective, with larger reductions on normal patches next to defects than on mid-distance or far-normal patches in all 15 categories. Code is publicly available at https://github.com/jw-chae/boundary\_support.

## 1 Introduction

Industrial visual anomaly detection is commonly learned from defect-free images because real defects are sparse, diverse, and expensive to annotate. Memory-based detectors store features from clean training patches and compare test patches with these references (Roth et al., 2022; Chae et al., 2026). Reconstruction-based detectors instead learn to reproduce clean feature targets and use the residual as anomaly evidence (Deng & Li, 2022; Guo et al., 2025). Strong pretrained encoders improve both families (Oquab et al., 2024), yet the normal evidence they use still comes almost entirely from clean images.

This design hides a simple assumption: clean training patches are sufficientfor the normal regions that appear at test time. A detector can improve its search rule or reconstruction error only over references and targets that are actually available. If a normal test patch is poorly represented by the clean candidate set, the failure occurs before the final readout.

We therefore ask: do clean training images contain all normal patches needed to localize defects? Most patches in a defective image are still labeled normal, including patches immediately beside the defect. Their local appearance may remain normal while the surrounding image context differs from any clean training image. Because contextual encoders make dense features depend on surrounding content (Naseer et al., 2021; Darcet et al., 2024), these patches can occupy representations that clean-view features do not provide.

We test this with a matched fixed-budget diagnostic. For each donor/evaluation partition, the cleanonly memory and ground-truth-assisted oracle are scored on the identical held-out defect images; any image that supplies oracle candidates is excluded from both arms for that partition. The two arms use the same frozen DINOv2 representation, the same test-time scoring rule, and the same number of references. The only change is whether mask-labeled normal patches from real defective images are allowed into the candidate pool. On MVTec AD, the matched comparison raises P-AP from 73.34 to 76.95. Increasing the reference count by 5.3% on the same candidate union reaches 76.96, only 0.01 point above the fixed-budget result. The fixed-budget memory therefore retains essentially the entire measured gain.

The useful additional patches are concentrated near defects. Restricting the oracle pool to groundtruth-normal patches within two patch cells of the annotated defect reaches 76.76 P-AP, recovering 94.70% of the fixed-budget gain. Detailed distance sweeps in the two largest-gain categories fall from near to mid to far. The gain also persists after inpainting in those categories and transfers across the tested source/target defect types, reducing the explanatory range of visible-defect leakage and exact defect memorization.

This observation motivates BOUNDARYSUPPORT. Rather than learning the synthetic core as an anomaly, BOUNDARYSUPPORT uses it to change the context around a pixel-preserved normal patch. Tokens touched by the nominal insertion or by a directly measured RGB change are excluded. Only neighboring token positions that remain outside this changed set are used as additional normal references or reconstruction targets. In the memory branch, altered-context features compete with clean candidates under the original fixed bank budget. In the reconstruction branch, the same preserved positions receive an auxiliary normal target while the original clean branch is unchanged.

Across MVTec, VisA, and Real-IAD, both branches improve P-AP in all six detector–dataset settings over three paired seeds. The main question then becomes what part ofthe intervention is responsible? Two matched controls answer this directly. For reconstruction, fixing the synthetic input and ring while replacing the target $E ( \tilde { x } ) _ { p }$ with the clean-view feature $E ( x ) _ { p }$ reverses the gain. For memory, fixing the selected positions and bank budget while replacing $\boldsymbol { F } ( \boldsymbol { \tilde { x } } ) _ { p }$ with $F ( x ) _ { p }$ lowers P-AP in 26 of 27 categories across MVTec and VisA. A final distance-resolved analysis shows that the learned score change is strongest on normal patches adjacent to real defects.

Our contributions are:

• A fixed-budget diagnostic showing that clean-only memory can miss useful normal references: adding mask-labeled normal patches from real defective images raises MVTec P-AP from 73.34 to 76.95, and a two-cell near-defect band recovers 94.70% of the gain.

• BOUNDARYSUPPORT, a clean-only intervention that changes surrounding context, excludes modified tokens, and uses only preserved neighboring features as memory references or reconstruction targets.

• Cross-paradigm and matched-control evidence showing that altered-context features are more useful localization targets/references than their clean-view counterparts, together with a spatial test linking the final score change to normal regions near defects.

## 2 Related Work

Normal references and reconstruction targets. PatchCore compresses clean patch features into a representative coreset (Roth et al., 2022), while ProCon uses multiple memory anchors for non-parametric feature reconstruction (Chae et al., 2026). Reconstruction methods instead learn clean feature targets, from reverse distillation (Deng & Li, 2022) to Transformer reconstruction in Dinomaly (Guo et al., 2025). Across these formulations, normal references or targets are drawn primarily from clean training images.

Learning with contaminated training data. SoftPatch filters high-outlier-score patches before coreset construction and reweights retained references (Jiang et al., 2022). InReaCh identifies reliable nominal patches through recurrence across training realizations (McIntosh & Branzan Albu, 2023), while FUN-AD reconstructs a normal memory from potentially contaminated unlabeled data (Im et al., 2025). SoftPatch+ further strengthens patch-level noise identification with multiple discriminators (Wang et al., 2025). These methods suppress unreliable content already present in the training pool; we study normal features that are absent from an otherwise clean pool.

Synthetic anomaly learning. CutPaste learns from pasted transformations (Li et al., 2021); DRAEM reconstructs clean images while segmenting synthetic defects (Zavrtanik et al., 2021); NSA creates synthetic anomalies by Poisson blending (Schluter et al.¨ , 2022); and DeSTSeg distills clean teacher features from corrupted inputs (Zhang et al., 2023). GLASS and PGBL further synthesize difficult anomalies near normal or prototype boundaries (Chen et al., 2024, 2025). In these methods, the synthetic change serves as anomaly supervision, a localization signal, or a clean-restoration target.

Normal evidence inside defect-containing images. MuSc exploits recurrence across unlabeled test images (Li et al., 2024), while INP-Former and INP-Former++ extract intrinsic normal prototypes from the current image (Luo et al., 2025a,b). These methods show that useful normal evidence can remain inside images containing anomalies; our diagnostic instead asks whether such normal evidence is already represented by the clean training pool.

## 3 Clean Images Miss Useful Normal Patches Near Defects

We first perform a fixed-budget diagnostic that changes the candidate set while leaving the detector unchanged. Let $\mathcal { Z } _ { \mathrm { c l e a n } }$ be patch features extracted from clean training images and $\mathcal { Z } _ { \mathrm { d e f } , N }$ be patch features labeled normal by the ground-truth mask of real defective images. With bank size $B .$

$$
M _ { B } ^ { \mathrm { c l e a n } } = \mathrm { K C e n t e r } _ { B } ( \mathcal { Z } _ { \mathrm { c l e a n } } ) ,\tag{1}
$$

$$
M _ { B } ^ { \mathrm { o r a c l e } } = \mathrm { K C e n t e r } _ { B } ( \mathcal { Z } _ { \mathrm { c l e a n } } \cup \mathcal { Z } _ { \mathrm { d e f } , N } ) .\tag{2}
$$

The encoder, feature representation, final number of references, and test-time score are fixed. For each donor/evaluation partition, both the clean-only and oracle memories are evaluated on the same held-out images; images supplying oracle candidates are excluded from both arms. Reported values aggregate only these matched held-out evaluations.

## 3.1 Changing candidate identity at a fixed memory budget

On MVTec AD, the clean-only memory obtains 73.34 P-AP. The fixed-budget GT-normal oracle obtains 76.95, a gain of 3.62 points. An expanded oracle that uses the same candidate union but 5.3% more stored references reaches 76.96. The expanded and fixed-budget results differ by 0.01 point, while image AUROC changes from 99.76 to 99.71. At the tested budget, the main change is therefore the identity of the available references, and the resulting effect is concentrated in pixel localization.

## 3.2 Most of the gain comes from patches near defects

For a ground-truth-normal patch $p ,$ let ${ \mathcal { A } } _ { \mathrm { g t } }$ denote the annotated defect cells and let $d ( p , A _ { \mathrm { g t } } )$ be the Euclidean patch-grid distance to the nearest cell in ${ \mathcal { A } } _ { \mathrm { g t } }$ . Restricting the additional candidates to $0 < d \leq 2$ reaches 76.76 P-AP over all 15 categories and recovers 94.70% of the fixed-budget gain. In the two categories with the largest oracle gains, the same fixed-budget comparison gives $+ 1 6 . 2 9 / + 6 . 2 5 / + 0$ .55 points for near/mid/far patches in leather and $+ 1 3 . 3 3 / + 2 . 1 4 / + 0 . 6 7$ in tile. Thus the all-category statement is the near-band recovery; the full near–mid–far ordering is a two-category diagnostic.

![](images/25fdaf6782fc04fb300b34b161bab5bad1858f775ecc6d7c5132d0ad6022b16e.jpg)

![](images/d6d973cf847efb3d0e5201ddfbbfa77eecb08639b4e142972addc46fa6074d11.jpg)  
Figure 1: Clean images miss useful normal patches near defects. Left: two real-defect examples under the clean-only memory and the fixed-budget GT-normal oracle. Right: MVTec macro P-AP at a fixed reference count. Ground-truth-normal candidates within two patch cells recover 94.70% of the oracle gain. The oracle uses ground-truth masks only as a diagnostic.

## 3.3 Visible defect signal and exact defect identity are insufficient explanations

We test two simple alternatives in leather and tile. First, we dilate and inpaint the annotated defect before re-encoding the image. The resulting candidate features still improve over the clean memory by 11.72 and 6.11 P-AP points, respectively. Second, ground-truth-normal candidate features collected beside one defect type are evaluated only on other defect types. Three tested source-to-target directions remain positive: +7.69 points for leather cut → other types, +0.79 for tile crack → other types, and +1.46 for tile glue-strip → other types. These scoped controls narrow the explanation: the remaining gains after inpainting and cross-defect transfer make visible-defect signal and exact same-defect memorization insufficient on their own.

## 3.4 A small near-defect pool receives disproportionate memory capacity

The near-defect cohort is numerically small, yet approximate k-center selection repeatedly promotes it. In leather, near-defect GT-normal patches form only 0.23% of the oracle candidate pool but 2.68% of the selected anchors, an 11.73× enrichment. In tile, the corresponding shares are 0.70% and 3.44%, a 4.95× enrichment. The fixed-budget oracle improves 14 of 15 categories, and the two-cell near-only oracle improves 13 of 15. These observations explain how a small candidate cohort can materially change a fixed-size bank: the selection geometry allocates it more memory than its raw frequency would suggest. The downstream P-AP comparison, rather than enrichment alone, establishes that those replacements are useful.

## 4 BoundarySupport: Learning Preserved Patches under Altered Context

Section 3 identifies a useful population but uses real defective images and masks. BOUNDARYSUPPORT asks whether clean training images can expose useful normal features by changing only their surrounding context.

## 4.1 Exclude modified tokens; learn from preserved neighbors

For a clean image $x \in \{ 0 , \ldots , 2 5 5 \} ^ { H \times W \times 3 }$ , a procedural intervention g produces a synthetic view and nominal insertion mask,

$$
( { \tilde { x } } , A ) = g ( x ) .\tag{3}
$$

We use CutPaste-Scar (Li et al., 2021), same-category Poisson paste following NSA (Schluter et al.¨ , 2022), and a same-image local warp followed by Poisson paste. All three interventions are fixed procedural transformations.

Because interpolation and blending can modify pixels outside A, we compare the two uint8 RGB images directly:

$$
D ( u ) = \mathbf { 1 } \left[ \frac { 1 } { 3 } \sum _ { c = 1 } ^ { 3 } | \tilde { x } _ { c } ( u ) - x _ { c } ( u ) | > 1 \right] .\tag{4}
$$

Let $P$ max-pool a pixel mask to the encoder token grid. The excluded token set and retained ring are

$$
C = P ( A ) \lor P ( D ) ,\tag{5}
$$

$$
R = { \mathrm { D i l a t e } } ( P ( A ) , r ) \backslash C , \qquad r = 2 .\tag{6}
$$

Dilation is applied to $P ( A )$ rather than to $C .$ . The complete synthetic image is still encoded, so the synthetic insertion remains visible as context, while no token in $C$ can become a normal memory candidate or direct reconstruction target. Only positions in R are used as additional normal memory candidates or reconstruction targets. These positions are pixel-preserved, while their encoder features are computed from the complete altered-context view.

The detected-change term $P ( D )$ is operationally important because blending and interpolation can touch token footprints outside the nominal insertion. In the seed-0 reconstruction control, the strict construction above improves P-AP by 4.94 points on MVTec and 2.09 on VisA. Using the same synthetic view while excluding only the nominal mask yields +0.70 on MVTec and −2.40 on VisA. Thus the method does not treat an arbitrary geometric ring as preserved: it removes token footprints that the generated image actually changed according to Eq. 4.

## 4.2 Memory branch

The memory branch uses frozen DINOv2 features and the ProCon test-time readout (Chae et al., 2026). Candidate discovery uses a separate score. For layer $\ell ,$ view v, and matched position p,

$$
d _ { \ell } ( v , p ) = \mathrm { m e d i a n } _ { b = 1 , \ldots , 5 } \operatorname* { m i n } _ { m \in M _ { \ell , b } } \| F _ { \ell } ( v ) _ { p } - m \| _ { 2 } ,\tag{7}
$$

$$
s _ { \mathrm { c a n d } } ( v , p ) = \frac { 1 } { | \mathcal { L } | } \sum _ { \ell \in \mathcal { L } } d _ { \ell } ( v , p ) ,\tag{8}
$$

$$
\Delta _ { p } = s _ { \mathrm { c a n d } } ( \tilde { x } , p ) - s _ { \mathrm { c a n d } } ( x , p ) .\tag{9}
$$

Candidate discovery uses $k = 1$ Euclidean nearest distance in each bank; the deployed ProCon readout remains $k = 5$ . The two filters have distinct roles. The clean-view condition keeps positions already compatible with the clean memory, while the $\Delta _ { p }$ condition identifies positions whose representation moves away from that memory after context is altered. Quantiles are computed within each intervention type so that one intervention family cannot dominate through scale alone. The uniform policy used in the main results retains

$$
p \in R , \quad s _ { \mathrm { c a n d } } ( x , p ) \leq Q _ { 0 . 5 0 } , \quad \Delta _ { p } \geq Q _ { 0 . 9 0 } , \quad \Delta _ { p } > 0 .\tag{10}
$$

A shared location mask selects the altered-context feature at every layer. If $\mathcal { \widetilde { Z } } _ { \ell } ^ { \mathrm { s e l } }$ denotes the selected altered-context features and $\mathcal { Z } _ { \ell } ^ { \mathrm { c l e a n } }$ the clean candidates, each bank is rebuilt as

$$
M _ { \ell , b } ^ { \prime } = \mathrm { K C e n t e r } _ { B } ^ { ( b ) } \left( \mathcal { Z } _ { \ell } ^ { \mathrm { c l e a n } } \cup \widetilde { \mathcal { Z } } _ { \ell } ^ { \mathrm { s e l } } \right) .\tag{11}
$$

The target size B is exactly the clean-baseline bank size.

(a) Alter context while preserving neighboring patches  
![](images/a24a1b1e6728af25cb05b505414a54eb7530d5f20e765d719627a5c2771b77a9.jpg)  
Figure 2: BoundarySupport exposes normal features under altered context while learning only from pixel-preserved positions. (a) A procedural intervention produces an altered-context view x˜ and nominal insertion mask A. Tokens intersecting the nominal insertion or detected RGB changes form the excluded set C, while the surrounding pixel-preserved positions form the ring R. (b) At a pixel-preserved position $p \in R$ , the clean view and altered-context view yield the clean-view feature $F ( x ) _ { p }$ and altered-context feature $\begin{array} { r } { F ( \tilde { x } ) _ { p } , } \end{array}$ respectively. (c) The memory branch adds selected altered-context features to the clean candidate pool and selects exactly B references, preserving the original memory budget and test-time readout. (d) The reconstruction branch keeps the original clean objective and adds altered-context encoder targets only at positions in R. Test-time scoring uses the observed image without synthesis or masks.

## 4.3 Reconstruction branch

The reconstruction branch follows the frozen-encoder Dinomaly framework (Guo et al., 2025). The encoder provides fixed feature targets; only the bottleneck and decoder are trained. The original clean hard-mined objective ${ \mathcal { L } } _ { \mathrm { H M } } ( x )$ is unchanged. For the two fused encoder targets $E _ { j }$ and decoder outputs $G _ { j }$ , the synthetic branch adds

$$
\mathcal { L } _ { R } = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { 2 } \mathrm { m e a n } _ { p \in R } \left[ 1 - \cos ( \mathrm { s g } ( E _ { j } ( \tilde { x } ) _ { p } ) , G _ { j } ( \tilde { x } ) _ { p } ) \right] ,\tag{12}
$$

with total objective

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { H M } } ( x ) + 0 . 0 5 \mathcal { L } _ { R } .\tag{13}
$$

One of the three interventions is sampled uniformly for each training example. Positions in C receive no direct loss but remain visible to contextual attention. Test-time scoring is the original reconstruction residual on the observed image.

Table 1: Paired changes after adding BOUNDARYSUPPORT (mean±sample standard deviation over three seed-level category macros).
<table><tr><td>Branch</td><td>Dataset</td><td> $\Delta \mathrm { I - A U R O C }$ </td><td> $\Delta \mathrm { P - A U R O C }$ </td><td> $\Delta \mathrm { P \mathrm { - } A P }$ </td><td> $\Delta \mathrm { P - F 1 }$ </td><td> $\Delta \mathrm { \ A U P R O }$ </td></tr><tr><td>Memory MVTec</td><td></td><td> $- 0 . 0 3 \pm 0 . 0 2$ </td><td> $+ 0 . 0 7 \pm 0 . 0 1$ </td><td> $\mathbf { + 2 . 2 1 \pm 0 . 0 4 }$ </td><td> $+ 1 . 3 1 \pm 0 . 0 1$ </td><td> $+ 0 . 2 2 \pm 0 . 0 1$ </td></tr><tr><td>Memory VisA</td><td></td><td></td><td> $+ 0 . 0 2 \pm 0 . 0 1 - 0 . 0 2 \pm 0 . 0 1 + 1 . 8 2 \pm 0 . 0 3 + 1 . 0 3 \pm 0 . 0 2 - 0 . 0 3 \pm 0 . 0 1$ </td><td></td><td></td><td></td></tr><tr><td>Memory</td><td>Real-IAD</td><td> $- 0 . 1 1 \pm 0 . 0 2$ </td><td>−0.01 ± 0.01 +1.76 ± 0.04</td><td></td><td> $+ 1 . 3 1 \pm 0 . 0 2$ </td><td> $- 0 . 0 5 \pm 0 . 0 1$ </td></tr><tr><td>Recon.</td><td>MVTec</td><td> $- 0 . 0 8 \pm 0 . 0 3$ </td><td></td><td>+0.16 ± 0.02 +5.01 ± 0.07</td><td> $+ 2 . 9 7 \pm 0 . 1 4$ </td><td> $+ 0 . 5 5 \pm 0 . 0 4$ </td></tr><tr><td>Recon.</td><td>VisA</td><td> $+ 0 . 0 5 \pm 0 . 0 1$ </td><td> $- 0 . 1 6 \pm 0 . 0 3$ </td><td> ${ \bf + 2 . 2 7 \pm 0 . 3 8 }$ </td><td> $+ 1 . 1 6 \pm 0 . 2 8$ </td><td> $+ 0 . 2 1 \pm 0 . 1 8$ </td></tr><tr><td>Recon.</td><td>Real-IAD</td><td> $- 0 . 5 2 \pm 0 . 0 4$ </td><td> $- 0 . 0 3 \pm 0 . 0 1$ </td><td> ${ \bf + 4 . 2 0 \pm 0 . 1 2 }$ </td><td> $+ 2 . 5 6 \pm 0 . 0 9$ </td><td> $+ 0 . 1 6 \pm 0 . 0 3$ </td></tr></table>

Table 2: Matched feature-identity controls at seed 0. Reconstruction fixes the synthetic input and ring while changing only the reconstruction target; memory fixes the selected positions and bank construction while changing only the stored feature.
<table><tr><td>Branch</td><td></td><td>Dataset Clean-view target/reference</td><td>Altered-context target/reference</td><td>Altered – clean</td><td>Wins</td></tr><tr><td>Reconstruction MVTec</td><td></td><td>61.82</td><td>73.89</td><td>+12.07</td><td>15/15</td></tr><tr><td>Reconstruction</td><td>VisA</td><td>44.34</td><td>52.79</td><td>+8.45</td><td>11/12</td></tr><tr><td>Memory</td><td>MVTec</td><td>73.36</td><td>75.60</td><td></td><td>+2.23 15/15</td></tr><tr><td>Memory</td><td>VisA</td><td>52.28</td><td>54.11</td><td></td><td>+1.84 11/12</td></tr></table>

## 5 Experiments

## 5.1 Setup

We evaluate on MVTec AD (15 categories) (Bergmann et al., 2019), VisA (12) (Zou et al., 2022), and Real-IAD (30) (Wang et al., 2024). Memory banks use 5% of clean training patches on MVTec/VisA and 1% on Real-IAD; each BOUNDARYSUPPORT bank has exactly the corresponding clean-baseline count. Reconstruction trains for 5,000 iterations per category. Real-IAD synthesis donors are restricted to the same category and camera view. All six detector–dataset settings use paired seeds 0/1/2. Hyperparameters were developed on MVTec and VisA; the Real-IAD configuration was fixed before its outcomes were inspected.

P-AP from the full pixel ranking is our primary endpoint. Table 1 also reports I-AUROC, P-AUROC, P-F1, and AUPRO. The appendix provides seed-0 values for all seven metrics, three-seed P-AP summaries, and category results. Values are 0–100 category macros; uncertainty is over three paired seed macros, and category–seed counts report sign consistency.

## 5.2 P-AP improves in all six detector–dataset settings

Table 1 shows the same localization direction in all six settings. Memory improves all 171 category–seed pairs (45/45 MVTec, 36/36 VisA, 90/90 Real-IAD); reconstruction improves 158/171 (45/45, 29/36, 84/90). Overall, P-AP rises in 329/342 pairs. Real-IAD reconstruction gains 4.20 ± 0.12 P-AP while I-AUROC falls $0 . 5 2 \pm 0 . 0 4$ , so the measured effect is a pixel-localization ranking improvement.

## 5.3 Altered-context features are the effective normal evidence

We next isolate feature identity with two matched controls. In reconstruction, the synthetic input ${ \tilde { x } } ,$ ring $R ,$ frozen encoder, trainable network, auxiliary weight, training length, and seed are fixed. Only the ring target changes from $\mathrm { s g } ( E ( \tilde { x } ) _ { p } )$ to the clean-view feature $\operatorname { s g } ( E ( x ) _ { p } )$ . In memory, the selected locations $\mathcal { P } _ { \mathrm { s e l } }$ , clean candidate pool, absolute bank budget, joint k-center selection, projections, inference, and seed are fixed. Only the selected feature changes from $F ( x ) _ { p }$ to $F ( \tilde { x } ) _ { \boldsymbol { \mathscr { F } } }$ .

The reconstruction control is especially discriminative. Restoring the clean-view feature from the same synthetic input lowers P-AP by 7.13 points on MVTec and 6.36 on VisA relative to baseline;

Table 3: Distance-resolved MVTec memory score changes at seed 0.
<table><tr><td>Spatial cohort Mean  $1 0 0 \Delta s$ </td></tr><tr><td>Defect patches -0.61</td></tr><tr><td>Near-defect normal  $( 0 < d \leq 2 )$  -0.37</td></tr><tr><td>Mid-distance normal  $( 2 < d \leq 4 )$  +0.11</td></tr><tr><td>Far normal  $( d > 4 )$  +0.19</td></tr><tr><td>Clean-image normal +0.21</td></tr></table>

using the feature actually formed under altered context instead raises P-AP by 4.94 and 2.09. The direct altered-versus-clean target gap is 12.07 and 8.45 points. This matched comparison separates target identity from synthetic exposure.

The memory control separates spatial mining from feature identity. At exactly the same selected locations, altered-context features outperform the clean-view features by 2.23 points on MVTec and 1.84 on VisA, with positive category-level differences in 26 of 27 categories. AUPRO and image AUROC remain mixed or nearly unchanged in this comparison, while the P-AP localization difference is consistent.

## 5.4 Where does the final score change?

The discovery in Section 3 is spatial: useful oracle references concentrate near defects. We therefore ask whether the final detector changes the same region. For the MVTec memory branch, each ground-truthnormal test patch is assigned its distance d from the nearest defect patch, and we compute

$$
\Delta s ( p ) = s _ { \mathrm { B o U N D A R Y S U P P O R T } } ( p ) - s _ { \mathrm { b a s e l i n e } } ( p ) .\tag{14}
$$

Scores are reported below on a $1 0 0 \Delta s$ scale.

The score change is spatially selective. Defect and near-defect normal patches decrease on average, whereas mid-distance, far-normal, and clean-image normal patches increase slightly. More importantly, the within-category near-minus-far and near-minus-mid contrasts are negative in all 15 categories; their median 100∆s contrasts are −0.51 and −0.43, respectively. The final memory therefore changes normal responses preferentially near defects, the same spatial region identified by the fixed-budget oracle. Because far-normal and clean-image normal scores move slightly upward rather than downward, this pattern is not a uniform suppression of normal scores.

A complementary reconstruction diagnostic is shown in Figure 3. At seed 0, normal-region q95/q99 change by −1.91/ − 4.91 on MVTec, $- 0 . 2 8 / - 0 . 8 8$ on VisA, and $- 0 . 3 2 / - 1$ .13 on Real-IAD (0–100 scale). Mean defect scores also decrease by 4.01, 1.29, and 2.49 points, while P-AP rises by 4.94, 2.09, and 4.20. The full three-seed MVTec and VisA reconstruction runs repeat the joint decrease of these recorded score summaries. Localization therefore does not improve by simply increasing defect activation; it improves when the full ordering of defect and normal responses becomes more favorable.

Together, the oracle and matched controls identify the same object across training and inference: pixel-preserved normal patches acquire useful altered-context features, and the resulting detector changes normal responses most strongly near real defects.

## 5.5 Failure cases expose the same ranking trade-off

The aggregate localization gain does not require every category–seed pair to improve. Across three paired seeds, memory improves all 171 category–seed pairs, whereas reconstruction improves 158 of 171. At seed 0, reconstruction improves 52 of 57 categories; the remaining cases are informative because they occur in the same score direction as the successful cases. On VisA, cashew loses 2.79 P-AP points while its recorded normal-region q99 falls by 0.74 and its mean defect score falls by 1.79. PCB3 and fryum lose 2.42 and 1.26 points, respectively. On Real-IAD, rolled strip base loses 2.92 P-AP points even though its pixel AUROC and AUPRO change only slightly.

(a) P-AP gain decreases with distance from the defect  
(b) Normal-region and defect scores fall while P-AP rises  
![](images/2e2d0b7a65f28f0dc86de3b16677a30833a6eaca0a5a70a26d5ed3bb98620865.jpg)

![](images/969dfba5e37c92ddcb0adfea6a16d57e21574e11257c22444f452608275d7698.jpg)  
Figure 3: Complementary spatial and score diagnostics. Left: in leather and tile, the fixed-budget oracle gain decreases from near to mid to far candidate bands. Right: reconstruction at seed 0 reduces high anomaly scores on normal regions and also reduces mean defect scores while P-AP rises. These summaries diagnose where scores move; P-AP is determined by the full ranking.

These cases show why the method is evaluated by the complete pixel ranking rather than by one score summary. Lower scores on difficult normal regions are useful only when defect evidence is preserved enough to improve their ordering against defect pixels. This also explains the Real-IAD reconstruction result in Table 1: a 4.20-point P-AP gain coexists with a 0.52-point decrease in image AUROC. BoundarySupport changes which normal evidence is learned; it does not impose a monotone improvement on every metric or category.

## 6 Discussion

Reference and target identity are part of the detector. The fixed-budget oracle changes candidate identity while holding the representation, test-time score, and reference count fixed. The 3.62-point P-AP gain makes the composition of normal evidence a first-order part of the detector, alongside the readout applied after a bank or reconstruction target set is defined. Contaminated-training methods address the opposite failure mode by removing unreliable patches from a candidate pool. Our results show that even a clean pool can omit useful normal features. Memory construction therefore depends on both which candidates are excluded and which normal representations are available.

Normal-pool completeness is representation-dependent. Pixel-preserved patches can acquire different representations when their surrounding context changes, so image-level cleanliness does not imply feature-level coverage. For contextual encoders, normal evidence depends on which contextual states of normal patches are represented, not only on which source images are nominal.

The real-defect diagnostic and the clean-only intervention play different roles. The oracle uses real defects only to locate a deficit in clean references. BoundarySupport instead starts from a known-clean image, creates a controlled context change, measures which pixel footprints changed, and excludes the corresponding tokens before adding normal references or reconstruction targets. The oracle claim is therefore annotation-level, while the method neither reuses real defect pixels nor relies on physical purity at a real mask boundary. Its matched controls test the representational consequence directly: a pixel-preserved patch becomes more useful when encoded under altered context.

The same feature matters in two different computations. The matched controls narrow this statement further. Reconstruction performs better when it learns the preserved feature encoded under altered context rather than the matched clean-view feature. Memory shows the same ordering when location selection is fixed. These two computations use the feature differently—one as a training target, one as a stored reference—yet both identify the altered-context representation as the more useful localization evidence in the tested settings.

Localization gain and defect evidence can trade off. BOUNDARYSUPPORT improves P-AP without uniformly improving image-level metrics. Reconstruction can reduce true-defect scores together with high normal-region scores, and negative categories appear when defect suppression dominates. A natural next step is therefore to preserve defect ordering explicitly while exposing missing normal evidence. Another open problem is to identify deployment-relevant missing patches directly, without relying on a synthetic context intervention.

The current evidence covers DINOv2 ViT-B-family representations in one memory branch and one reconstruction branch. The two-paradigm agreement establishes transfer across different normal-modeling computations in this representation family; other encoders remain an empirical question.

## 7 Conclusion

At a fixed memory budget, ground-truth-normal patches from real defective images provide useful references absent from a clean-only pool: MVTec P-AP rises from 73.34 to 76.95, and patches within two cells recover 94.70% of the gain. BOUNDARYSUPPORT brings this observation to clean-only training by altering context, excluding modified tokens, and learning from preserved neighbors.

Across memory and reconstruction on three datasets and three paired seeds, BOUNDARYSUPPORT consistently improves pixel localization. Matched controls show that altered-context features outperform their clean-view counterparts as both targets and references. Normal-only detection therefore depends on which normal features are available, not only on how they are scored.

## References

Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. MVTec AD — a comprehensive Real-World dataset for unsupervised anomaly detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9592–9600, 2019. doi: 10.1109/CVPR.2019. 00982.

Joongwon Chae, Lihui Luo, Yang Liu, Dongmei Yu, Peiwu Qin, Runming Wang, and Ilmoon Chae. ProCon: Projection-Consistency memory for Training-Free anomaly detection. arXiv preprint arXiv:2607.04894, 2026. doi: 10.48550/arXiv.2607.04894.

Qiyu Chen, Huiyuan Luo, Chengkan Lv, and Zhengtao Zhang. A unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization. In Computer Vision – ECCV 2024, volume 15125 of Lecture Notes in Computer Science, pp. 37–54. Springer Nature Switzerland, 2024. doi: 10.1007/978-3-031-72855-6 3.

Qiyu Chen, Huiyuan Luo, Han Gao, Chengkan Lv, and Zhengtao Zhang. Progressive boundary guided anomaly synthesis for industrial anomaly detection. IEEE Transactions on Circuits and Systems for Video Technology, 35(2):1193–1208, 2025. doi: 10.1109/TCSVT.2024.3479887.

Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need´ registers. In International Conference on Learning Representations (ICLR), 2024. URL https: //openreview.net/forum?id=2dnO3LLiJ1.

Hanqiu Deng and Xingyu Li. Anomaly detection via reverse distillation from One-Class embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9737–9746, 2022. doi: 10.1109/CVPR52688.2022.00951.

Jia Guo, Shuai Lu, Weihang Zhang, Fang Chen, Huiqi Li, and Hongen Liao. Dinomaly: The less is more philosophy in Multi-Class unsupervised anomaly detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20405–20415, 2025. doi: 10.1109/CVPR52734.2025.01900.

Jiin Im, Yongho Son, and Je Hyeong Hong. FUN-AD: Fully unsupervised learning for anomaly detection with noisy training data. In Proceedings of the Winter Conference on Applications of Computer Vision (WACV), pp. 9429–9438, 2025. doi: 10.1109/WACV61041.2025.00915.

Xi Jiang, Jianlin Liu, Jinbao Wang, Qiang Nie, Kai Wu, Yong Liu, Chengjie Wang, and Feng Zheng. SoftPatch: Unsupervised anomaly detection with noisy data. In Advances in Neural Information Processing Systems, volume 35, pp. 15433–15445. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-1123.

Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, and Tomas Pfister. CutPaste: Self-Supervised learning for anomaly detection and localization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9664–9674, 2021. doi: 10.1109/CVPR46437.2021.00954.

Xurui Li, Ziming Huang, Feng Xue, and Yu Zhou. MuSc: Zero-Shot industrial anomaly classification and segmentation with mutual scoring of the unlabeled images. In International Conference on Learning Representations (ICLR), 2024. URL https://openreview.net/forum?id=AHgc5SMdtd.

Wei Luo, Yunkang Cao, Haiming Yao, Xiaotian Zhang, Jianan Lou, Yuqi Cheng, Weiming Shen, and Wenyong Yu. Exploring intrinsic normal prototypes within a single image for universal anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9974–9983, 2025a. doi: 10.1109/CVPR52734.2025.00932.

Wei Luo, Haiming Yao, Yunkang Cao, Qiyu Chen, Ang Gao, Weiming Shen, and Wenyong Yu. INP-Former++: Advancing universal anomaly detection via intrinsic normal prototypes and residual learning. arXiv preprint arXiv:2506.03660, 2025b. doi: 10.48550/arXiv.2506.03660.

Declan McIntosh and Alexandra Branzan Albu. Inter-Realization channels: Unsupervised anomaly detection beyond One-Class classification. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6285–6295, 2023. doi: 10.1109/ICCV51070.2023.00578.

Muhammad Muzammal Naseer, Kanchana Ranasinghe, Salman H. Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Intriguing properties of vision transformers. In Advances in Neural Information Processing Systems, volume 34, pp. 23296–23308. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ c404a5adbf90e09631678b13b05d9d7a-Abstract.html.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and´ Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=a68SUt6zFt.

Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Scholkopf, Thomas Brox, and Peter Gehler.¨ Towards total recall in industrial anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14318–14328, 2022. doi: 10.1109/CVPR52688. 2022.01392.

Hannah M. Schluter, Jeremy Tan, Benjamin Hou, and Bernhard Kainz. Natural synthetic anomalies¨ for Self-supervised anomaly detection and localization. In Computer Vision – ECCV 2022, volume

13691 of Lecture Notes in Computer Science, pp. 474–489. Springer Nature Switzerland, 2022. doi: 10.1007/978-3-031-19821-2 27.

Chengjie Wang, Wenbing Zhu, Bin-Bin Gao, Zhenye Gan, Jiangning Zhang, Zhihao Gu, Shuguang Qian, Mingang Chen, and Lizhuang Ma. Real-IAD: A Real-World Multi-View dataset for benchmarking versatile industrial anomaly detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22883–22892, 2024. doi: 10.1109/CVPR52733.2024.02159.

Chengjie Wang, Xi Jiang, Bin-Bin Gao, Zhenye Gan, Yong Liu, Feng Zheng, and Lizhuang Ma. Soft-Patch+: Fully unsupervised anomaly classification and segmentation. Pattern Recognition, 161:111295, 2025. doi: 10.1016/j.patcog.2024.111295.

Vitjan Zavrtanik, Matej Kristan, and Danijel Skocaj. DRAEM — a discriminatively trained reconstructionˇ embedding for surface anomaly detection. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8330–8339, 2021. doi: 10.1109/ICCV48922.2021.00822.

Xuan Zhang, Shiyu Li, Xi Li, Ping Huang, Jiulong Shan, and Ting Chen. DeSTSeg: Segmentation guided denoising Student-Teacher for anomaly detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3914–3923, 2023. doi: 10.1109/CVPR52729. 2023.00381.

Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. SPot-the-Difference Self-supervised Pre-training for anomaly detection and segmentation. In Computer Vision – ECCV 2022, volume 13690 of Lecture Notes in Computer Science, pp. 392–408. Springer Nature Switzerland, 2022. doi: 10.1007/978-3-031-20056-4 23.

## A Experimental Setup

## A.1 Datasets and evaluation

We use the standard normal-only training splits and labeled test splits of MVTec AD (Bergmann et al., 2019), VisA (Zou et al., 2022), and Real-IAD (Wang et al., 2024). MVTec and VisA were used to configure the method. For Real-IAD, the generator family, ring radius, reconstruction weight, and memory ratio were fixed before inspecting evaluation outcomes. All six detector–dataset settings were then evaluated with seeds 0, 1, and 2. Real-IAD synthesis donors are restricted to the same category and camera view.

Table 4: Evaluation settings.
<table><tr><td>Dataset</td><td>Categories</td><td>Memory ratio</td><td>Memory seeds</td><td>Reconstruction seeds</td></tr><tr><td>MVTec AD</td><td>15</td><td>5%</td><td>0,1,2</td><td>0,1,2</td></tr><tr><td>VisA</td><td>12</td><td>5%</td><td>0,1,2</td><td>0,1,2</td></tr><tr><td>Real-IAD</td><td>30</td><td>1%</td><td>0,1,2</td><td>0,1,2</td></tr></table>

The VisA memory result uses one uniform rule for all 12 categories and all three seeds: $q _ { \mathrm { c l e a n } } = 0 . 5 0$ $q _ { \Delta } = 0 . 9 0$ , all three interventions, joint candidate selection, and the original fixed 5% bank. Every layer and bank has the same target size as the clean baseline. The MVTec GT-assisted diagnostic uses seed 0. For each donor/evaluation partition, the clean-only and oracle variants are evaluated on the identical held-out images; donor images are excluded from both arms for that partition. Reported oracle diagnostics aggregate only these matched held-out evaluations.

## A.2 Input geometry and scoring

All encoders receive ImageNet-normalized RGB crops. For memory, MVTec images are aspectpreservingly resized to 448 and center-cropped to $3 9 2 \times 3 9 2$ ; VisA and Real-IAD use a square 448 resize followed by the same crop. Reconstruction uses the square resize/crop on all three datasets. Patch size 14 gives a $2 8 \times 2 8$ token grid.

The memory branch averages four layer maps, resizes to 392, applies Gaussian smoothing with $\sigma = 4$ , and uses the mean of the highest 0.5% patch scores as the image score. The reconstruction branch averages two fused residual maps, bilinearly resizes to $2 5 6 \times 2 5 6$ , applies a $5 \times 5$ Gaussian filter with $\sigma = 4$ , and uses the mean of the highest 1% pixel scores as the image score. BoundarySupport does not change the test-time input or scoring path.

## A.3 Metrics and aggregation

All metrics are reported on a 0–100 scale. Image metrics are I-AUROC, I-AP, and maximum I-F1 along the image precision–recall curve. Pixel metrics are P-AUROC, P-AP, and maximum P-F1 after flattening evaluated pixels within each category. AUPRO integrates connected-component overlap against falsepositive rate up to FPR 0.30 using 200 thresholds. Dataset results are arithmetic means over categories. Multi-seed entries first compute the category macro for each seed and then report the paired mean and sample standard deviation (ddof = 1). Category–seed counts summarize sign consistency and are not treated as independent replicates for uncertainty estimation.

## B Fixed-Budget GT-Normal Diagnostic

The diagnostic uses a frozen DINOv2-B/14 representation, the five-bank memory readout, a 5% memory ratio, and the evaluation geometry above. Ground-truth masks identify normal-labeled patches inside real defective images. Clean and oracle variants are always compared on the same held-out images within each donor/evaluation partition.

## B.1 Candidate identity and memory budget

The clean-only memory builds its candidate pool only from clean training patches. The expanded GT-normal oracle adds normal-labeled patches from defective images and applies the same percentage memory ratio, increasing the final bank size by about 5%. The fixed-budget GT-normal oracle uses the same enlarged candidate pool but restores the category-specific absolute bank count of the clean-only memory.

Table 5: MVTec fixed-budget diagnostic on matched held-out evaluation sets.
<table><tr><td>Candidate pool</td><td>I-AUROC</td><td>P-AUROC</td><td>P-AP</td><td>AUPRO</td><td>Rel. bank size</td></tr><tr><td>Clean candidates (matched eval.)</td><td>99.76</td><td>98.80</td><td>73.34</td><td>96.18</td><td>1.00</td></tr><tr><td>Clean + GT-normal, expanded</td><td>99.71</td><td>98.94</td><td>76.96</td><td>96.64</td><td>1.05</td></tr><tr><td> $\mathrm { { C l e a n + G T \mathrm { { \cdot n o r m a l } , } } }$  fixed budget</td><td>99.71</td><td>98.94</td><td>76.95</td><td>96.64</td><td>1.00</td></tr></table>

The fixed-budget candidate change raises P-AP by 3.62 points. Expanded and fixed-budget banks differ by only 0.01 point P-AP at this budget.

Table 6: MVTec category P-AP for the clean-only memory and GT-normal oracle variants.
<table><tr><td>Category</td><td>Clean (matched eval.)</td><td>Expanded GT-normal</td><td>Fixed-budget GT-normal</td><td>Near-defect GT-normal</td></tr><tr><td>bottle</td><td>89.02</td><td>90.29</td><td>90.30</td><td>90.21</td></tr><tr><td>cable</td><td>76.70</td><td>77.36</td><td>77.32</td><td>77.34</td></tr><tr><td>capsule</td><td>64.40</td><td>63.87</td><td>63.88</td><td>63.47</td></tr><tr><td>carpet</td><td>80.07</td><td>84.04</td><td>84.05</td><td>83.99</td></tr><tr><td>grid</td><td>63.74</td><td>66.38</td><td>66.45</td><td>66.26</td></tr><tr><td>hazelnut</td><td>81.64</td><td>84.70</td><td>84.62</td><td>84.67</td></tr><tr><td>leather</td><td>61.97</td><td>77.52</td><td>77.55</td><td>78.26</td></tr><tr><td>metal nut</td><td>83.07</td><td>87.38</td><td>87.36</td><td>87.09</td></tr><tr><td>pill</td><td>75.61</td><td>76.98</td><td>76.97</td><td>76.31</td></tr><tr><td>screw</td><td>65.89</td><td>67.00</td><td>67.01</td><td>66.12</td></tr><tr><td>tile</td><td>76.69</td><td>89.75</td><td>89.76</td><td>90.02</td></tr><tr><td>toothbrush</td><td>63.00</td><td>63.41</td><td>63.38</td><td>62.77</td></tr><tr><td>transistor</td><td>66.84</td><td>67.65</td><td>67.58</td><td>67.26</td></tr><tr><td>wood</td><td>79.57</td><td>82.76</td><td>82.78</td><td>83.08</td></tr><tr><td>zipper</td><td>71.83</td><td>75.28</td><td>75.26</td><td>74.58</td></tr><tr><td>Macro</td><td>73.34</td><td>76.96</td><td>76.95</td><td>76.76</td></tr></table>

The fixed-budget oracle improves 14 of 15 categories. The near-defect variant improves 13 of 15. The clean column here is the matched clean control evaluated under the oracle donor/evaluation partitions; Section K separately reports the ordinary seed-0 full-test category results.

## B.2 Distance from the annotated defect

For a ground-truth-normal patch $p$ and annotated defect-cell set ${ \mathcal { A } } _ { \mathrm { g t } }$ , define

$$
\mathcal { H } _ { \mathrm { n e a r } } = \{ p : y _ { p } = N , 0 < d ( p , A _ { \mathrm { g t } } ) \leq 2 \} ,\tag{15}
$$

$$
\mathcal { H } _ { \mathrm { m i d } } = \{ p : y _ { p } = N , 2 < d ( p , A _ { \mathrm { g t } } ) \leq 4 \} ,\tag{16}
$$

$$
\mathcal { H } _ { \mathrm { f a r } } = \{ p : y _ { p } = N , ~ d ( p ,  { A _ { \mathrm { g t } } } ) > 4 \} .\tag{17}
$$

The all-category near-defect oracle obtains 99.73 I-AUROC, 98.92 P-AUROC, 76.76 P-AP, and 96.57 AUPRO. It is 0.19 P-AP below the fixed-budget all-GT-normal oracle and recovers 94.70% of its gain over the clean-only memory.

## B.3 Inpainting and cross-defect transfer

We dilate the annotated defect by seven pixels, apply Telea inpainting with radius 3, re-encode the image, and retain candidate patches labeled normal by the original mask. The final memory count remains fixed.

Table 7: Distance-stratified fixed-budget P-AP in the two largest-gain categories.
<table><tr><td>Category</td><td>Clean</td><td>Fixed-budget GT-normal</td><td>Near</td><td>Mid</td><td>Far</td></tr><tr><td>leather</td><td>61.97</td><td>77.55</td><td>78.26</td><td>68.22</td><td>62.52</td></tr><tr><td>tile</td><td>76.69</td><td>89.76</td><td>90.02</td><td>78.83</td><td>77.36</td></tr></table>

Table 8: P-AP after removing the visible annotated defect by inpainting.
<table><tr><td>Category</td><td>Clean</td><td>Fixed-budget GT-normal</td><td>Inpainted-defect GT-normal</td></tr><tr><td>leather</td><td>61.97</td><td>77.55</td><td>73.69</td></tr><tr><td>tile</td><td>76.69</td><td>89.76</td><td>82.80</td></tr></table>

For cross-defect transfer, GT-normal candidates are collected next to one source defect type and evaluation is restricted to other defect types.

Table 9: Cross-defect transfer P-AP.
<table><tr><td>Source → evaluation defects</td><td>Clean</td><td>Cross-defect GT-normal</td><td>∆P-AP</td></tr><tr><td>leather: cut → color/fold/glue/poke</td><td>64.65</td><td>72.35</td><td>+7.69</td></tr><tr><td>tile: crack → other types</td><td>90.65</td><td>91.44</td><td>+0.79</td></tr><tr><td>tile: glue-strip → other types</td><td>73.98</td><td>75.44</td><td>+1.46</td></tr></table>

## B.4 Reference selection and oracle score changes

Near-defect patches occupy a small fraction of the candidate pool but a larger fraction of selected references.

The all-GT-normal fixed-budget oracle similarly raises the selected share of defective-image normal candidates from 4.95% to 15.98% in leather and from 4.20% to 8.65% in tile.

## C Synthetic Context Interventions

The three interventions are fixed, non-learned image operations. Each receives a clean uint8 RGB image x, a seeded random number generator, and, when needed, a clean donor $x ^ { \prime }$ from the same category. Real-IAD donors are also restricted to the same camera view. Each generator outputs a synthetic view x˜ and nominal insertion mask A.

## C.1 CutPaste-Scar

We use the scar geometry of CutPaste (Li et al., 2021). Let $s = \operatorname* { m i n } ( H , W ) / 2 2 4$ . The short side is sampled as an integer from

$$
\left[ \operatorname* { m a x } ( 2 , \operatorname { r o u n d } ( 2 s ) ) , \operatorname* { m a x } ( 3 , \operatorname { r o u n d } ( 1 6 s ) ) \right)
$$

and the long side from

$$
[ \operatorname* { m a x } ( { \mathrm { s h o r t } } + 1 , \operatorname { r o u n d } ( 1 0 s ) ) , \ \operatorname* { m a x } ( { \mathrm { s h o r t } } + 2 , \operatorname { r o u n d } ( 2 5 s ) ) ) .
$$

Orientation is swapped with probability 0.5. A same-image patch receives independent channel gains from U[0.9, 1.1], rotation from $U [ - 4 5 ^ { \circ } , 4 5 ^ { \circ } ]$ , bilinear interpolation, and reflected borders before being pasted at a valid location. The rotated binary support defines A.

Table 10: Near-defect candidate and selected-reference shares.
<table><tr><td>Category</td><td>Candidate near</td><td>Selected near</td><td>Enrichment</td><td>Selected anchors</td></tr><tr><td>leather</td><td>0.23%</td><td>2.68%</td><td>11.73×</td><td>258 / 9,604</td></tr><tr><td>tile</td><td>0.70%</td><td>3.44%</td><td>4.95×</td><td>310/9,016</td></tr></table>

Table 11: Raw oracle score summaries in leather and tile.
<table><tr><td>Category</td><td>Memory</td><td>Good normal  $\mathbf { q } 9 5$ </td><td>Defect-image normal  $\mathbf { q } 9 5$ </td><td> $\boldsymbol { \mathrm { q 9 9 } }$ </td><td>Defect mean</td></tr><tr><td>leather</td><td>Clean</td><td>0.40</td><td>0.48</td><td>0.61</td><td>0.65</td></tr><tr><td>leather</td><td>Fixed-budget GT-normal</td><td>0.40</td><td>0.44</td><td>0.53</td><td>0.60</td></tr><tr><td>tile</td><td>Clean</td><td>0.45</td><td>0.55</td><td>0.66</td><td>0.62</td></tr><tr><td>tile</td><td>Fixed-budget GT-normal</td><td>0.45</td><td>0.51</td><td>0.60</td><td>0.61</td></tr></table>

## C.2 Cross-image Poisson paste

For the NSA-style intervention (Schluter et al. ¨ , 2022), donor half-sizes are sampled independently as

$$
h _ { 1 / 2 } = \mathrm { c l i p } [ ( 0 . 0 3 + G ) H , 0 . 0 5 H , 0 . 2 0 H ] , \qquad G \sim \mathrm { G a m m a } ( 2 , 0 . 0 5 ) ,
$$

with the analogous rule for width. The donor crop is resized by $\mathrm { c l i p } ( Z , 0 . 7 , 1 . 3 )$ for $Z \sim \mathcal { N } ( 1 , 0 . 5 ^ { 2 } )$ and pasted with OpenCV NORMAL CLONE. The clone support excluding its one-pixel border defines A.

## C.3 Local warp and Poisson paste

A same-image crop spanning 10–25% of image height and width is transformed by rotation $U [ - 1 5 ^ { \circ } , 1 5 ^ { \circ } ]$ and isotropic scale U[0.82, 1.18] with bilinear interpolation and reflected borders, then inserted with the same Poisson paste and mask construction.

## C.4 Generation schedule

Memory materializes clean/synthetic feature pairs; reconstruction generates a synthetic view online. MVTec and Real-IAD memory use one cached pair per generator and clean image. VisA memory uses the same three intervention families under the uniform q50 policy. Reconstruction samples one of the three interventions uniformly for each training example. Donors are same-category, and Real-IAD additionally enforces same camera view.

## D Pixel-Preserved Patch Construction

The change mask in Eq. (4) is evaluated in signed arithmetic to avoid uint8 wraparound. Let

$$
A _ { P } = P ( A ) , \qquad D _ { P } = P ( D ) , \qquad C = A _ { P } \lor D _ { P } ,
$$

where $P$ is adaptive max pooling to the $2 8 \times 2 8$ token grid. At the final $3 9 2 \times 3 9 2$ crop, each token corresponds to a $1 4 \times 1 4$ pixel footprint. The strict retained set is

$$
R _ { \mathrm { s t r i c t } } = \operatorname { M a x P o o l } _ { 2 r + 1 } ( A _ { P } ) \land \lnot C , \qquad r = 2 .
$$

The dilation is applied to $A _ { P } . \mathrm { \ A }$ token changed outside the nominal insertion is excluded through $D _ { P }$ but does not expand the ring.

For the nominal-mask-only control, the retained set is

$$
R _ { \mathrm { n o m i n a l } } = \mathrm { M a x P o o l } _ { 2 r + 1 } ( A _ { P } ) \wedge \neg A _ { P } .
$$

This set can retain tokens intersecting detected RGB changes outside the nominal mask. Main memory candidates and reconstruction targets are restricted to $R _ { \mathrm { s t r i c t } }$ . The synthetic insertion remains part of the model input; the main reconstruction configuration does not mask its attention keys/values.

## E Memory Branch Details

The memory branch uses frozen plain DINOv2 ViT-B/14 (Oquab et al., 2024) without register tokens and extracts layers $[ - 3 , - 6 , - 8 , - 9 ]$ . Each 768-dimensional token is $\ell _ { 2 }$ -normalized and mapped through a fixed Gaussian projection (seed 42) to 512 dimensions. The projected feature is not normalized again.

For each layer, five banks are selected independently with approximate greedy k-center. Bank b uses seed $s + b ,$ a separate random 192-dimensional coreset projection, ten random starts, and target size

$$
B = \operatorname* { m a x } \{ 1 , \lfloor \rho N \rfloor \} .
$$

Caches and final banks are stored in FP16, while candidate distances, coreset selection, and inference distances are evaluated in FP32; TF32 is enabled.

## E.1 Candidate score

Candidate discovery uses the bank-wise nearest Euclidean distance

$$
d _ { \ell } ( z ) = \mathrm { m e d i a n } _ { b = 1 } ^ { 5 } \operatorname* { m i n } _ { m \in M _ { \ell , b } } \| z - m \| _ { 2 } , \qquad s _ { \mathrm { c a n d } } ( z ) = \frac { 1 } { 4 } \sum _ { \ell } d _ { \ell } ( z ) .
$$

For a clean/synthetic pair at the same position,

$$
\Delta _ { p } = s _ { \mathrm { c a n d } } ( \tilde { z } _ { p } ) - s _ { \mathrm { c a n d } } ( z _ { p } ) .
$$

Quantiles are computed separately within each intervention type. The uniform policy selects a shared four-layer position inside $R _ { \mathrm { s t r i c t } }$ when

$$
s _ { \mathrm { c a n d } } ( z _ { p } ) \leq Q _ { 0 . 5 0 } , \qquad \Delta _ { p } \geq Q _ { 0 . 9 0 } , \qquad \Delta _ { p } > 0 .
$$

Selected altered-context features are concatenated with all clean candidates, and the same k-center selection procedure selects exactly B final references per bank. MVTec/VisA use $\rho = 0 . 0 5$ and Real-IAD uses $\rho = 0 . 0 1$

## E.2 Test-time readout

The candidate-selection score is distinct from the deployed ProCon readout (Chae et al., 2026). Inference uses five nearest anchors per bank, soft reconstruction with a temperature derived from the median squared neighbor distance on at most 512 query tokens, median fusion across five banks, and averaging across four layers. The clean baseline and BoundarySupport share the same readout.

## F Reconstruction Branch Details

The reconstruction branch uses frozen dinov2reg vit base 14. Encoder blocks 2–9 provide eight targets, averaged into two groups over blocks 2–5 and 6–9. Trainable components are a 768–3072–768 bottleneck with dropout 0.2 and eight 12-head linear-attention decoder blocks.

Each category is trained for 5,000 iterations with batch size 16 using StableAdamW: learning rate $2 \times 1 0 ^ { - 3 }$ , betas (0.9, 0.999), weight decay $1 0 ^ { - 4 }$ , AMSGrad, gradient clipping at 0.1, 100 warm-up iterations, and cosine decay to $2 \times 1 0 ^ { - 4 }$ . Training is FP32. The clean hard-mining schedule is

$$
p _ { t } = \operatorname* { m i n } ( 0 . 9 t / 1 0 0 0 , 0 . 9 ) ,
$$

with gradients below the token-residual threshold scaled by 0.1.

For fused encoder target $E _ { j } ( v ) _ { p }$ and decoder output $G _ { j } ( v ) _ { p }$ , the pixel-preserved-patch objective is

$$
\mathcal { L } _ { R } = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { 2 } \mathrm { m e a n } _ { p \in R _ { \mathrm { s t r i c t } } } \left[ 1 - \cos { ( \mathrm { s g } ( E _ { j } ( \tilde { x } ) _ { p } ) , G _ { j } ( \tilde { x } ) _ { p } ) } \right] ,
$$

and

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { b a s e } } + 0 . 0 5 \mathcal { L } _ { R } .
$$

The two terms are backpropagated separately before one clipping and optimizer step, which is gradientequivalent to their sum. The main configuration uses $r = 2 .$ , auxiliary weight 0.05, and no core-key mask. Testing uses the baseline scoring path without synthesis or masks.

## G Full Dataset-Level Metrics

Table 12: Seed-0 category-macro metrics. Three-seed paired changes are reported in Table 1 and Appendix H.
<table><tr><td>Branch</td><td>Dataset</td><td>Variant</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>P-AUROC</td><td>P-AP</td><td>P-F1</td><td>AUPRO</td></tr><tr><td>Memory</td><td>MVTec</td><td>Baseline</td><td>99.76</td><td>99.91</td><td>99.38</td><td>98.80</td><td>73.34</td><td>71.04</td><td>96.18</td></tr><tr><td>Memory</td><td>MVTec</td><td>BoundarySupport</td><td>99.73</td><td>99.90</td><td>99.31</td><td>98.86</td><td>75.60</td><td>72.33</td><td>96.39</td></tr><tr><td>Memory</td><td>VisA</td><td>Baseline</td><td>99.17</td><td>99.28</td><td>97.50</td><td>99.08</td><td>52.23</td><td>54.71</td><td>97.06</td></tr><tr><td>Memory</td><td>VisA</td><td>BoundarySupport</td><td>99.19</td><td>99.30</td><td>97.54</td><td>99.07</td><td>54.05</td><td>55.75</td><td>97.03</td></tr><tr><td>Memory</td><td>Real-IAD</td><td>Baseline</td><td>94.89</td><td>94.33</td><td>89.00</td><td>99.17</td><td>52.02</td><td>53.11</td><td>97.60</td></tr><tr><td>Memory</td><td>Real-IAD</td><td>BoundarySupport</td><td>94.79</td><td>94.11</td><td>88.76</td><td>99.16</td><td>53.78</td><td>54.42</td><td>97.54</td></tr><tr><td>Recon.</td><td>MVTec</td><td>Baseline</td><td>99.75</td><td>99.91</td><td>99.24</td><td>98.35</td><td>68.95</td><td>68.79</td><td>94.69</td></tr><tr><td>Recon.</td><td>MVTec</td><td>BoundarySupport</td><td>99.68</td><td>99.86</td><td>99.07</td><td>98.53</td><td>73.89</td><td>71.89</td><td>95.20</td></tr><tr><td>Recon.</td><td>VisA</td><td>Baseline</td><td>98.93</td><td>99.04</td><td>96.80</td><td>98.97</td><td>50.70</td><td>54.29</td><td>94.77</td></tr><tr><td>Recon.</td><td>VisA</td><td>BoundarySupport</td><td>98.99</td><td>99.12</td><td>96.69</td><td>98.80</td><td>52.79</td><td>55.21</td><td>94.82</td></tr><tr><td>Recon.</td><td>Real-IAD</td><td>Baseline</td><td>94.27</td><td>93.55</td><td>88.15</td><td>99.29</td><td>49.46</td><td>52.06</td><td>96.24</td></tr><tr><td>Recon.</td><td>Real-IAD</td><td>BoundarySupport</td><td>93.75</td><td>92.87</td><td>87.26</td><td>99.26</td><td>53.66</td><td>54.62</td><td>96.40</td></tr></table>

## H Repeated Runs and Sign Consistency

Table 13: P-AP over three paired seeds (0, 1, 2).
<table><tr><td>Branch</td><td>Dataset</td><td>Baseline</td><td>BoundarySupport</td><td>Paired  $\Delta$ </td><td>Positive pairs</td></tr><tr><td>Memory</td><td>MVTec</td><td> $7 3 . 3 5 \pm 0 . 0 1$ </td><td> $7 5 . 5 6 \pm 0 . 0 3$ </td><td> $+ 2 . 2 1 \pm 0 . 0 4$ </td><td>45/45</td></tr><tr><td>Memory</td><td>VisA</td><td> $5 2 . 2 3 \pm 0 . 0 2$ </td><td> $5 4 . 0 5 \pm 0 . 0 3$ </td><td> $+ 1 . 8 2 \pm 0 . 0 3$ </td><td>36/36</td></tr><tr><td>Memory</td><td>Real-IAD</td><td> $5 2 . 0 2 \pm 0 . 0 2$ </td><td> $5 3 . 7 8 \pm 0 . 0 3$ </td><td> $+ 1 . 7 6 \pm 0 . 0 4$ </td><td>90/90</td></tr><tr><td>Recon.</td><td>MVTec</td><td> $6 8 . 9 6 \pm 0 . 1 1$ </td><td> $7 3 . 9 8 \pm 0 . 1 1$ </td><td> $+ 5 . 0 1 \pm 0 . 0 7$ </td><td>45/45</td></tr><tr><td>Recon.</td><td>VisA</td><td> $5 0 . 9 3 \pm 0 . 2 6$ </td><td> $5 3 . 2 0 \pm 0 . 3 9$ </td><td> $+ 2 . 2 7 \pm 0 . 3 8$ </td><td>29/36</td></tr><tr><td>Recon.</td><td>Real-IAD</td><td> $4 9 . 4 6 \pm 0 . 1 5$ </td><td> $5 3 . 6 6 \pm 0 . 1 8$ </td><td> $+ 4 . 2 0 \pm 0 . 1 2$ </td><td>84/90</td></tr></table>

Memory is positive in all 171 category–seed pairs, while reconstruction is positive in 158/171; overall, P-AP increases in 329/342 pairs. These counts describe sign consistency. The reported mean±sample standard deviation is computed from the three seed-level category macros.

## I Mechanism Controls

All controls in this section use seed 0. Controls 3 and 4 cover all 15 MVTec and 12 VisA categories.   
Control 5 covers all 15 MVTec categories.

## I.1 Control 3: Reconstruction target identity

At positions $p \in R _ { \mathrm { s t r i c t } }$ , BoundarySupport uses

$$
T _ { p } ^ { \mathrm { a l t e r e d } } = \mathrm { s g } [ E ( \tilde { x } ) _ { p } ] ,
$$

whereas the clean-target control keeps the same synthetic input and ring but uses

$$
T _ { p } ^ { \mathrm { c l e a n } } = \mathrm { s g } [ E ( x ) _ { p } ] .
$$

Table 14: Control 3 P-AP: target identity with synthetic input and preserved positions fixed.
<table><tr><td>Dataset</td><td>Clean-view target</td><td>Altered-context target</td><td>Altered – clean</td><td>Category wins</td></tr><tr><td>MVTec</td><td>61.82</td><td>73.89</td><td>+12.07</td><td>15/15</td></tr><tr><td>VisA</td><td>44.34</td><td>52.79</td><td>+8.45</td><td>11/12</td></tr></table>

The synthetic input, $R _ { \mathrm { s t r i c t } }$ , frozen encoder, bottleneck/decoder, auxiliary weight, training iterations, and seed are otherwise identical.

On MVTec, the clean target is 7.13 points below baseline and decreases P-AP in 15/15 categories; the altered-context target is 4.94 points above baseline and improves 15/15. The category-median alteredminus-clean difference is 11.84 points. On VisA, the clean target is 6.36 points below baseline; the altered-context target is 2.09 points above baseline, with an altered-minus-clean difference of 8.45 points and 11/12 category wins. AUPRO changes relative to baseline are $- 0 . 5 5 / + 0 . 5 1$ (clean/altered) on MVTec and $- 0 . 1 7 / + 0 . 0 5$ on VisA. I-AUROC changes remain within 0.08 point in these target variants.

## I.2 Control 4: Memory feature identity

Let $\mathcal { P } _ { \mathrm { s e l } }$ be the positions selected by the same candidate rule. We keep these positions, the clean candidate pool, absolute bank budget, joint k-center selection, bank count, projections, inference, and seed fixed, and compare

$$
{ \mathcal { Z } } _ { \mathrm { c l e a n - p o s } } = \{ F ( x ) _ { p } : p \in { \mathcal { P } } _ { \mathrm { s e l } } \} , \qquad { \mathcal { Z } } _ { \mathrm { a l t e r e d } } = \{ F ( { \tilde { x } } ) _ { p } : p \in { \mathcal { P } } _ { \mathrm { s e l } } \} .
$$

Only feature values differ before the same joint k-center selection constructs the final memory.

Table 15: Control 4 P-AP: stored-feature identity with selected positions and bank construction fixed. Deltas are computed from unrounded values.
<table><tr><td>Dataset</td><td>Clean-view feature</td><td>Altered-context feature</td><td>∆P-AP</td><td>Category wins</td></tr><tr><td>MVTec</td><td>73.36</td><td>75.60</td><td>+2.23</td><td>15/15</td></tr><tr><td>VisA</td><td>52.28</td><td>54.11</td><td>+1.84</td><td>11/12</td></tr></table>

The category-median P-AP differences are +1.78 on MVTec and +1.14 on VisA. MVTec AUPRO changes from 96.21 to 96.39, with 13/15 category wins, while I-AUROC is 99.74 versus 99.73. On VisA, AUPRO is 97.08 versus 97.03 and I-AUROC 99.17 versus 99.16. Across the two datasets, altered-context features yield higher P-AP in 26 of 27 categories. The only VisA reversal is candle at -0.23 point.

## I.3 Control 5: Distance-resolved final memory scores

For each anomalous MVTec test image, normal patches are partitioned by patch-grid distance to the nearest anomalous patch: near $0 < d \leq 2$ , mid $2 < d \leq 4 .$ , and far $d > 4$ . Defect patches and normal patches from clean images are aggregated separately. For each patch,

$$
\Delta s ( p ) = s _ { \mathrm { B o u n d a r y S u p p o r t } } ( p ) - s _ { \mathrm { b a s e l i n e } } ( p ) .
$$

Near-minus-far and near-minus-mid contrasts are negative in 15/15 categories. Their category-median contrasts on the $1 0 0 \Delta s$ scale are -0.51 and -0.43, respectively. Thus the final memory changes normal responses more strongly next to real defects than at mid or far distances.

## I.4 Additional reconstruction-object controls

The following controls change the supervised patch set or objective while retaining the same reconstruction architecture and auxiliary weight. The corresponding clean-view patch applies the auxiliary objective to

Table 16: Control 5 mean MVTec memory score change. Values are 100∆s.
<table><tr><td>Spatial cohort</td><td>Mean 100∆s</td></tr><tr><td>Defect patches</td><td>-0.61</td></tr><tr><td>Near-defect normal</td><td>-0.37</td></tr><tr><td>Mid-distance normal</td><td>+0.11</td></tr><tr><td>Far normal</td><td>+0.19</td></tr><tr><td>Clean-image normal</td><td>+0.21</td></tr></table>

Table 17: Seed-0 full-dataset reconstruction-object controls.
<table><tr><td></td><td>Dataset Supervised object</td><td>∆I-AUROC</td><td>∆I-AP</td><td>∆I-F1</td><td>∆P-AUROC</td><td>∆P-AP</td><td>∆P-F1</td><td>∆AUPRO</td><td>P-AP wins</td></tr><tr><td></td><td>MVTec Pixel-preserved surrounding patch</td><td>-0.07</td><td>-0.05</td><td>-0.18</td><td>+0.18</td><td>+4.94</td><td>+3.10</td><td>+0.51</td><td>15/15</td></tr><tr><td>MVTec</td><td>Corresponding clean-view patch</td><td>-0.02</td><td>0.00</td><td>+0.04</td><td>-0.01</td><td>-0.26</td><td>-0.15</td><td>-0.01</td><td>7/15</td></tr><tr><td>MVTec</td><td>Patch outside nominal mask</td><td>-0.23</td><td>-0.20</td><td>-0.71</td><td>-0.03</td><td>+0.70</td><td>+0.99</td><td>-0.09</td><td>8/15</td></tr><tr><td>MVTec</td><td>Synthetic core</td><td>-2.11</td><td>-1.44</td><td>-0.75</td><td>-20.86</td><td>-52.28</td><td>-45.18</td><td>-49.13</td><td>0/15</td></tr><tr><td>VisA</td><td>Pixel-preserved surrounding patch</td><td>+0.06</td><td>+0.08</td><td>-0.11</td><td>-0.17</td><td>+2.09</td><td>+0.92</td><td>+0.05</td><td>9/12</td></tr><tr><td>VisA</td><td>Corresponding clean-view patch</td><td>-0.27</td><td>-0.32</td><td>-0.65</td><td>-0.07</td><td>-0.30</td><td>-0.16</td><td>-0.28</td><td>4/12</td></tr><tr><td>VisA</td><td>Patch outside nominal mask</td><td>-0.14</td><td>-0.13</td><td>-0.30</td><td>-0.33</td><td>-2.40</td><td>-1.31</td><td>+0.09</td><td>5/12</td></tr><tr><td>VisA</td><td>Synthetic core</td><td>-1.10</td><td>-1.67</td><td>-0.90</td><td>-9.44</td><td>-42.67</td><td>-40.07</td><td>-42.02</td><td>0/12</td></tr></table>

matched clean-view positions. The nominal-ring control omits detected-RGB-change exclusion. The synthetic-core control applies a margin objective to the synthetic core.

## J Generator and Hyperparameter Analysis

These seed-0 reconstruction ablations use ten development categories: the five MVTec textures and VisA candle, capsules, chewinggum, macaroni2, and pipe-fryum.

Table 18: Paired changes on the ten-category development subset.
<table><tr><td>Setting</td><td>ΔI-AUROC</td><td>∆I-AP</td><td>∆I-F1</td><td>∆P-AUROC</td><td>∆P-AP</td><td>∆P-F1</td><td>ΔAUPRO</td></tr><tr><td>Three interventions, r = 2, λ = 0.05</td><td>+0.01</td><td>+0.01</td><td>-0.21</td><td>+0.23</td><td>+6.99</td><td>+4.18</td><td>+0.80</td></tr><tr><td>Scar only</td><td>+0.01</td><td>+0.01</td><td>-0.04</td><td>+0.26</td><td>+8.29</td><td>+4.58</td><td>+0.80</td></tr><tr><td>Poisson paste only</td><td>+0.12</td><td>+0.07</td><td>+0.40</td><td>+0.18</td><td>+4.45</td><td>+2.93</td><td>+1.09</td></tr><tr><td>Local warp + paste only</td><td>+0.04</td><td>+0.02</td><td>+0.05</td><td>+0.15</td><td>+4.60</td><td>+2.85</td><td>+0.70</td></tr><tr><td>r = 1</td><td>+0.05</td><td>+0.03</td><td>-0.04</td><td>+0.25</td><td>+7.55</td><td>+4.31</td><td>+0.99</td></tr><tr><td>r = 3</td><td>0.00</td><td>-0.01</td><td>-0.04</td><td>+0.23</td><td>+6.91</td><td>+4.11</td><td>+1.21</td></tr><tr><td>λ = 0.025</td><td>+0.04</td><td>+0.03</td><td>+0.02</td><td>+0.23</td><td>+6.66</td><td>+3.80</td><td>+1.17</td></tr><tr><td>λ = 0.1</td><td>+0.05</td><td>+0.03</td><td>+0.02</td><td>+0.23</td><td>+7.31</td><td>+4.12</td><td>+1.01</td></tr><tr><td>Core-key masking</td><td>+0.03</td><td>+0.01</td><td>-0.17</td><td>+0.23</td><td>+6.87</td><td>+4.23</td><td>+0.92</td></tr></table>

All three intervention families are positive on this subset. Positive P-AP also persists for $r \in \{ 1 , 2 , 3 \}$ and $\lambda \in \{ 0 . 0 2 5 , 0 . 0 5 , 0 . 1 \}$

## K Category-Level Results

All category tables below use seed 0 and the 0–100 scale. Cells with arrows report baseline → Boundary-Support unless noted otherwise.

Table 19: Category-level seed-0 results for MVTec Memory. Cells with arrows report baseline → BoundarySupport.
<table><tr><td>Category</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>P-AUROC</td><td>P-AP</td><td>P-F1</td><td>AUPRO</td></tr><tr><td>bottle</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>99.20→99.26</td><td>89.02→90.79</td><td>82.61→83.12</td><td>97.33→97.54</td></tr><tr><td>cable</td><td>99.81→99.78</td><td>99.86→99.85</td><td>98.14→98.14</td><td>98.66→98.66</td><td>76.70→77.68</td><td>72.22→72.54</td><td>95.66→95.71</td></tr><tr><td>capsule</td><td>98.75→98.75</td><td>99.67→99.67</td><td>98.48→98.48</td><td>98.98→98.99</td><td>64.40→65.03</td><td>61.17→61.53</td><td>96.66→96.70</td></tr><tr><td>carpet</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>99.58→99.61</td><td>80.07→81.90</td><td>76.78→77.68</td><td>98.61→98.70</td></tr><tr><td>grid</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>99.63→99.67</td><td>63.74→67.48</td><td>61.84→64.04</td><td>98.78→98.91</td></tr><tr><td>hazelnut</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>99.57→99.59</td><td>81.64→82.75</td><td>77.43→78.43</td><td>98.56→98.65</td></tr><tr><td>leather</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>99.59→99.68</td><td>61.97→68.55</td><td>60.39→65.28</td><td>98.63→98.95</td></tr><tr><td>metal_nut</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>97.90→97.94</td><td>83.06→83.62</td><td>87.67→87.75</td><td>93.05→93.19</td></tr><tr><td>pill</td><td>99.55→99.39</td><td>99.91→99.89</td><td>99.21→99.21</td><td>97.42→97.51</td><td>75.61→76.89</td><td>70.25→70.79</td><td>92.81→93.02</td></tr><tr><td>screw</td><td>98.66→98.52</td><td>99.46→99.41</td><td>97.59→96.62</td><td>99.46→99.46</td><td>65.89→66.13</td><td>61.42→61.49</td><td>98.37→98.35</td></tr><tr><td>tile</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>98.51→98.82</td><td>76.69→81.29</td><td>79.68→82.02</td><td>95.05→96.06</td></tr><tr><td>toothbrush</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>99.32→99.39</td><td>63.00→66.02</td><td>69.29→70.86</td><td>97.73→97.96</td></tr><tr><td>transistor</td><td>99.94→99.89</td><td>99.89→99.78</td><td>98.31→98.25</td><td>96.55→96.56</td><td>66.84→67.48</td><td>64.09→64.40</td><td>89.40→89.50</td></tr><tr><td>wood</td><td>99.66→99.66</td><td>99.86→99.86</td><td>98.95→98.95</td><td>99.03→99.16</td><td>79.57→83.14</td><td>72.66→75.34</td><td>96.78→97.21</td></tr><tr><td>zipper</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>98.57→98.59</td><td>71.83→75.20</td><td>68.04→69.71</td><td>95.37→95.44</td></tr></table>

Table 20: Category-level seed-0 results for VisA Memory, uniform q50. Cells with arrows report baseline → BoundarySupport.
<table><tr><td>Category</td><td>Baseline P-AP</td><td>Uniform ∆P-AP</td></tr><tr><td>candle</td><td>46.04</td><td>+0.33</td></tr><tr><td>capsules</td><td>68.93</td><td>+1.14</td></tr><tr><td>cashew</td><td>69.10</td><td>+2.75</td></tr><tr><td>chewinggum</td><td>73.90</td><td>+6.19</td></tr><tr><td>fryum</td><td>46.32</td><td>+0.42</td></tr><tr><td>macaroni1</td><td>27.64</td><td>+2.47</td></tr><tr><td>macaroni2</td><td>20.92</td><td>+0.93</td></tr><tr><td>pcb1</td><td>87.45</td><td>+0.98</td></tr><tr><td>pcb2</td><td>34.38</td><td>+1.66</td></tr><tr><td>pcb3</td><td>39.35</td><td>+0.58</td></tr><tr><td>pcb4</td><td>52.28</td><td>+0.24</td></tr><tr><td>pipe_fryum</td><td>60.46</td><td>+4.16</td></tr><tr><td>Macro</td><td>52.23</td><td>+1.82</td></tr></table>

Table 21: Category-level seed-0 results for Real-IAD Memory. Cells with arrows report baseline → BoundarySupport.
<table><tr><td rowspan=1 colspan=8>Category           I-AUROC         I-AP         I-F1   P-AUROC         P-AP         P-F1      AUPRO</td></tr><tr><td rowspan=1 colspan=2>audiojack       96.51→96.62</td><td rowspan=1 colspan=1>94.91→95.00</td><td rowspan=1 colspan=2>87.75→87.4499.83→99.84</td><td rowspan=1 colspan=1>60.97→68.74</td><td rowspan=1 colspan=2>61.49→64.6099.44→99.46</td></tr><tr><td rowspan=1 colspan=2>bottle_cap       96.27→96.29</td><td rowspan=1 colspan=1>96.54→96.61</td><td rowspan=1 colspan=1>90.62→90.91</td><td rowspan=1 colspan=1>99.64→99.60</td><td rowspan=1 colspan=1>35.04→36.86</td><td rowspan=1 colspan=1>38.59→40.24</td><td rowspan=1 colspan=1>98.80→98.68</td></tr><tr><td rowspan=1 colspan=2>button_battery   91.70→91.29</td><td rowspan=1 colspan=1>93.99→93.77</td><td rowspan=1 colspan=1>87.81→87.84</td><td rowspan=1 colspan=1>99.62→99.62</td><td rowspan=1 colspan=1>70.06→70.51</td><td rowspan=1 colspan=1>66.73→67.07</td><td rowspan=1 colspan=1>98.73→98.75</td></tr><tr><td rowspan=1 colspan=2>end_cap         91.12→90.72</td><td rowspan=1 colspan=1>90.85→90.58</td><td rowspan=1 colspan=1>88.39→88.01</td><td rowspan=1 colspan=1>99.38→99.34</td><td rowspan=1 colspan=1>14.10→14.77</td><td rowspan=1 colspan=1>26.00→26.64</td><td rowspan=1 colspan=1>98.01→97.86</td></tr><tr><td rowspan=1 colspan=1>eraser</td><td rowspan=1 colspan=1>95.25→95.14</td><td rowspan=1 colspan=1>94.39→94.32</td><td rowspan=1 colspan=1>86.46→86.05</td><td rowspan=1 colspan=1>99.79→99.79</td><td rowspan=1 colspan=1>49.00→50.48</td><td rowspan=1 colspan=1>51.47→52.09</td><td rowspan=1 colspan=1>99.30→99.31</td></tr><tr><td rowspan=1 colspan=1>fire_hood</td><td rowspan=1 colspan=1>96.09→95.72</td><td rowspan=1 colspan=1>92.27→91.51</td><td rowspan=1 colspan=1>87.28→86.55</td><td rowspan=1 colspan=1>99.66→99.63</td><td rowspan=1 colspan=1>51.23→52.77</td><td rowspan=1 colspan=1>51.29→51.81</td><td rowspan=1 colspan=1>98.88→98.81</td></tr><tr><td rowspan=1 colspan=1>mint</td><td rowspan=1 colspan=1>77.24→77.85</td><td rowspan=1 colspan=1>85.89→86.27</td><td rowspan=1 colspan=1>76.47→76.99</td><td rowspan=1 colspan=1>99.27→99.20</td><td rowspan=1 colspan=1>44.12→45.42</td><td rowspan=1 colspan=1>47.28→48.22</td><td rowspan=1 colspan=1>97.86→97.72</td></tr><tr><td rowspan=1 colspan=1>mounts</td><td rowspan=1 colspan=1>97.55→97.53</td><td rowspan=1 colspan=1>96.45→96.45</td><td rowspan=1 colspan=1>90.53→90.15</td><td rowspan=1 colspan=1>99.74→99.75</td><td rowspan=1 colspan=1>50.14→51.82</td><td rowspan=1 colspan=1>51.36→52.36</td><td rowspan=1 colspan=1>99.15→99.18</td></tr><tr><td rowspan=1 colspan=1>pcb</td><td rowspan=1 colspan=1>96.92→96.97</td><td rowspan=1 colspan=1>98.18→98.19</td><td rowspan=1 colspan=1>93.02→93.18</td><td rowspan=1 colspan=1>99.82→99.82</td><td rowspan=1 colspan=1>61.81→63.05</td><td rowspan=1 colspan=1>60.21→60.93</td><td rowspan=1 colspan=1>99.40→99.40</td></tr><tr><td rowspan=1 colspan=1>phone_battery</td><td rowspan=1 colspan=1>97.24→97.18</td><td rowspan=1 colspan=1>97.46→97.39</td><td rowspan=1 colspan=1>91.57→91.85</td><td rowspan=1 colspan=1>99.89→99.90</td><td rowspan=1 colspan=1>67.31→69.76</td><td rowspan=1 colspan=1>63.20→65.56</td><td rowspan=1 colspan=1>99.65→99.65</td></tr><tr><td rowspan=1 colspan=1>plastic_nut</td><td rowspan=1 colspan=1>95.92→95.89</td><td rowspan=1 colspan=1>88.24→87.92</td><td rowspan=1 colspan=1>84.58→82.91</td><td rowspan=1 colspan=1>99.87→99.84</td><td rowspan=1 colspan=1>50.41→52.57</td><td rowspan=1 colspan=1>48.66→50.31</td><td rowspan=1 colspan=1>99.56→99.49</td></tr><tr><td rowspan=1 colspan=1>plastic_plug</td><td rowspan=1 colspan=1>96.47→96.30</td><td rowspan=1 colspan=1>95.85→95.61</td><td rowspan=1 colspan=1>88.71→88.39</td><td rowspan=1 colspan=1>99.54→99.54</td><td rowspan=1 colspan=1>37.21→39.43</td><td rowspan=1 colspan=1>40.56→43.29</td><td rowspan=1 colspan=1>98.74→98.77</td></tr><tr><td rowspan=1 colspan=1>porcelain_doll</td><td rowspan=1 colspan=1>95.39→95.19</td><td rowspan=1 colspan=1>92.66→92.48</td><td rowspan=1 colspan=1>85.50→85.23</td><td rowspan=1 colspan=1>99.53→99.52</td><td rowspan=1 colspan=1>32.10→34.04</td><td rowspan=1 colspan=1>37.19→38.56</td><td rowspan=1 colspan=1>98.77→98.76</td></tr><tr><td rowspan=1 colspan=1>regulator</td><td rowspan=1 colspan=1>96.49→96.41</td><td rowspan=1 colspan=1>90.57→90.60</td><td rowspan=1 colspan=1>88.19→87.30</td><td rowspan=1 colspan=1>99.97→99.97</td><td rowspan=1 colspan=1>73.60→74.17</td><td rowspan=1 colspan=1>69.13→69.82</td><td rowspan=1 colspan=1>99.91→99.91</td></tr><tr><td rowspan=1 colspan=1>rolled_strip_base</td><td rowspan=1 colspan=1>99.69→99.61</td><td rowspan=1 colspan=1>99.84→99.80</td><td rowspan=1 colspan=1>99.02→98.93</td><td rowspan=1 colspan=1>99.78→99.80</td><td rowspan=1 colspan=1>44.67→47.19</td><td rowspan=1 colspan=1>45.64→47.64</td><td rowspan=1 colspan=1>99.27→99.32</td></tr><tr><td rowspan=1 colspan=1>sim_card_set</td><td rowspan=1 colspan=1>99.03→99.01</td><td rowspan=1 colspan=1>99.32→99.30</td><td rowspan=1 colspan=1>96.23→96.22</td><td rowspan=1 colspan=1>99.88→99.88</td><td rowspan=1 colspan=1>72.55→74.64</td><td rowspan=1 colspan=1>65.77→67.77</td><td rowspan=1 colspan=1>99.59→99.61</td></tr><tr><td rowspan=1 colspan=1>switch</td><td rowspan=1 colspan=1>99.14→99.09</td><td rowspan=1 colspan=1>99.55→99.53</td><td rowspan=1 colspan=1>97.59→97.59</td><td rowspan=1 colspan=1>99.38→99.38</td><td rowspan=1 colspan=1>65.09→65.18</td><td rowspan=1 colspan=1>63.88→64.05</td><td rowspan=1 colspan=1>97.94→97.94</td></tr><tr><td rowspan=1 colspan=1>tape</td><td rowspan=1 colspan=1>99.67→99.66</td><td rowspan=1 colspan=1>99.48→99.47</td><td rowspan=1 colspan=1>96.63→96.85</td><td rowspan=1 colspan=1>99.90→99.91</td><td rowspan=1 colspan=1>58.45→62.31</td><td rowspan=1 colspan=1>56.26→59.14</td><td rowspan=1 colspan=1>99.68→99.70</td></tr><tr><td rowspan=1 colspan=1>terminalblock</td><td rowspan=1 colspan=1>99.25→99.29</td><td rowspan=1 colspan=1>99.54→99.57</td><td rowspan=1 colspan=1>98.07→98.21</td><td rowspan=1 colspan=1>99.91→99.91</td><td rowspan=1 colspan=1>62.46→65.15</td><td rowspan=1 colspan=1>61.38→63.59</td><td rowspan=1 colspan=1>99.69→99.71</td></tr><tr><td rowspan=1 colspan=1>toothbrush</td><td rowspan=1 colspan=1>92.80→92.77</td><td rowspan=1 colspan=1>95.99→95.98</td><td rowspan=1 colspan=1>88.89→88.44</td><td rowspan=1 colspan=1>99.23→99.22</td><td rowspan=1 colspan=1>44.61→44.86</td><td rowspan=1 colspan=1>51.49→51.75</td><td rowspan=1 colspan=1>97.42→97.41</td></tr><tr><td rowspan=1 colspan=1>toy</td><td rowspan=1 colspan=1>90.76→91.09</td><td rowspan=1 colspan=1>95.76→95.92</td><td rowspan=1 colspan=1>86.85→87.33</td><td rowspan=1 colspan=1>88.59→88.53</td><td rowspan=1 colspan=1>24.99→25.58</td><td rowspan=1 colspan=1>35.41→35.62</td><td rowspan=1 colspan=1>69.69→68.73</td></tr><tr><td rowspan=1 colspan=1>toy_brick</td><td rowspan=1 colspan=1>84.61→83.67</td><td rowspan=1 colspan=1>85.22→83.60</td><td rowspan=1 colspan=1>75.98→74.85</td><td rowspan=1 colspan=1>97.79→97.74</td><td rowspan=1 colspan=1>52.78→54.58</td><td rowspan=1 colspan=1>53.93→55.57</td><td rowspan=1 colspan=1>93.54→93.37</td></tr><tr><td rowspan=1 colspan=1>transistor1</td><td rowspan=1 colspan=1>97.39→97.48</td><td rowspan=1 colspan=1>98.47→98.53</td><td rowspan=1 colspan=1>94.87→95.04</td><td rowspan=1 colspan=1>99.52→99.49</td><td rowspan=1 colspan=1>46.87→47.98</td><td rowspan=1 colspan=1>46.94→47.60</td><td rowspan=1 colspan=1>98.39→98.32</td></tr><tr><td rowspan=1 colspan=1>u.block</td><td rowspan=1 colspan=1>96.19→95.99</td><td rowspan=1 colspan=1>93.37→92.87</td><td rowspan=1 colspan=1>89.45→89.08</td><td rowspan=1 colspan=1>99.81→99.80</td><td rowspan=1 colspan=1>62.18→65.50</td><td rowspan=1 colspan=1>60.11→62.24</td><td rowspan=1 colspan=1>99.39→99.37</td></tr><tr><td rowspan=1 colspan=1>usb</td><td rowspan=1 colspan=1>97.46→97.36</td><td rowspan=1 colspan=1>96.90→96.83</td><td rowspan=1 colspan=1>92.54→92.68</td><td rowspan=1 colspan=1>99.83→99.84</td><td rowspan=1 colspan=1>52.29→52.45</td><td rowspan=1 colspan=1>53.75→54.33</td><td rowspan=1 colspan=1>99.45→99.46</td></tr><tr><td rowspan=1 colspan=1>usb_adaptor</td><td rowspan=1 colspan=1>93.42→93.08</td><td rowspan=1 colspan=1>91.24→90.88</td><td rowspan=1 colspan=1>84.97→85.09</td><td rowspan=1 colspan=1>99.81→99.81</td><td rowspan=1 colspan=1>31.58→33.31</td><td rowspan=1 colspan=1>35.70→38.34</td><td rowspan=1 colspan=1>99.37→99.38</td></tr><tr><td rowspan=1 colspan=1>vcpill</td><td rowspan=1 colspan=1>95.63→95.91</td><td rowspan=1 colspan=1>94.10→94.38</td><td rowspan=1 colspan=1>85.93→86.01</td><td rowspan=1 colspan=1>98.53→98.56</td><td rowspan=1 colspan=1>72.44→72.92</td><td rowspan=1 colspan=1>70.44→70.61</td><td rowspan=1 colspan=1>95.97→96.01</td></tr><tr><td rowspan=1 colspan=1>wooden_beads</td><td rowspan=1 colspan=1>89.56→89.58</td><td rowspan=1 colspan=1>92.64→92.41</td><td rowspan=1 colspan=1>83.59→83.49</td><td rowspan=1 colspan=1>99.00→98.98</td><td rowspan=1 colspan=1>49.79→51.04</td><td rowspan=1 colspan=1>52.40→53.44</td><td rowspan=1 colspan=1>97.26→97.17</td></tr><tr><td rowspan=1 colspan=1>woodstick</td><td rowspan=1 colspan=1>92.24→91.19</td><td rowspan=1 colspan=1>80.34→77.68</td><td rowspan=1 colspan=1>73.59→71.43</td><td rowspan=1 colspan=1>99.56→99.52</td><td rowspan=1 colspan=1>60.42→61.74</td><td rowspan=1 colspan=1>60.61→62.27</td><td rowspan=1 colspan=1>98.52→98.40</td></tr><tr><td rowspan=1 colspan=1>zipper</td><td rowspan=1 colspan=1>99.83→99.81</td><td rowspan=1 colspan=1>99.92→99.90</td><td rowspan=1 colspan=1>99.00→98.91</td><td rowspan=1 colspan=1>98.89→98.91</td><td rowspan=1 colspan=1>62.41→64.70</td><td rowspan=1 colspan=1>66.32→67.16</td><td rowspan=1 colspan=1>96.52→96.61</td></tr></table>

Table 22: Category-level seed-0 results for MVTec Reconstruction. Cells with arrows report baseline → BoundarySupport.
<table><tr><td>Category</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>P-AUROC</td><td>P-AP</td><td>P-F1</td><td>AUPRO</td></tr><tr><td>bottle</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>99.14→99.19</td><td>88.88→91.02</td><td>83.43→84.20</td><td>97.16→97.59</td></tr><tr><td>cable</td><td>99.94→100.00</td><td>99.97→100.00</td><td>99.46→100.00</td><td>98.25→98.28</td><td>76.09→77.57</td><td>73.70→74.75</td><td>93.67→93.30</td></tr><tr><td>capsule</td><td>98.96→98.44</td><td>99.77→99.65</td><td>99.08→99.08</td><td>98.68→98.59</td><td>60.32→61.60</td><td>58.54→59.16</td><td>97.74→97.30</td></tr><tr><td>carpet</td><td>99.96→99.84</td><td>99.99→99.95</td><td>99.44→99.44</td><td>99.34→99.42</td><td>68.56→74.44</td><td>71.60→75.10</td><td>97.36→97.96</td></tr><tr><td>grid</td><td>99.92→99.92</td><td>99.97→99.97</td><td>99.13→99.13</td><td>99.44→99.55</td><td>57.34→65.98</td><td>59.84→64.83</td><td>97.52→97.89</td></tr><tr><td>hazelnut</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>99.46→99.57</td><td>82.44→85.67</td><td>77.52→80.50</td><td>96.63→97.50</td></tr><tr><td>leather</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>99.34→99.63</td><td>52.80→65.53</td><td>54.14→66.06</td><td>98.24→98.60</td></tr><tr><td>metal_nut</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>96.99→96.99</td><td>80.23→82.05</td><td>86.16→86.52</td><td>94.78→95.09</td></tr><tr><td>pill</td><td>99.32→99.48</td><td>99.88→99.91</td><td>98.93→98.93</td><td>97.43→97.70</td><td>70.94→75.15</td><td>68.29→70.50</td><td>96.30→97.54</td></tr><tr><td>screw</td><td>98.93→98.61</td><td>99.64→99.53</td><td>97.07→96.23</td><td>99.67→99.64</td><td>62.42→62.51</td><td>60.22→59.88</td><td>98.44→98.30</td></tr><tr><td>tile</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>97.17→98.51</td><td>69.66→80.20</td><td>72.66→79.96</td><td>84.78→89.68</td></tr><tr><td>toothbrush</td><td>100.00→100.00</td><td>100.00→100.00</td><td>100.00→100.00</td><td>99.00→99.23</td><td>55.12→62.83</td><td>64.65→68.94</td><td>95.56→96.23</td></tr><tr><td>transistor</td><td>99.75→99.33</td><td>99.63→98.99</td><td>97.56→95.24</td><td>94.85→94.78</td><td>64.97→65.01</td><td>61.74→60.74</td><td>81.84→80.11</td></tr><tr><td>wood</td><td>99.56→99.65</td><td>99.86→99.89</td><td>98.36→98.36</td><td>97.49→97.99</td><td>70.61→79.80</td><td>68.17→72.67</td><td>93.53→94.18</td></tr><tr><td>zipper</td><td>99.95→99.97</td><td>99.99→99.99</td><td>99.58→99.58</td><td>99.01→98.92</td><td>73.83→79.03</td><td>71.15→74.57</td><td>96.77→96.67</td></tr></table>

Table 23: Category-level seed-0 results for VisA Reconstruction. Cells with arrows report baseline → BoundarySupport.
<table><tr><td>Category</td><td>I-AUROC</td><td>I-AP</td><td>I-F1</td><td>P-AUROC</td><td>P-AP</td><td>P-F1</td><td>AUPRO</td></tr><tr><td>candle</td><td>98.14→98.11</td><td>98.20→98.13</td><td>93.53→93.60</td><td>99.43→99.41</td><td>43.30→43.60</td><td>48.85→50.79</td><td>95.50→95.00</td></tr><tr><td>capsules</td><td>98.28→98.83</td><td>98.83→99.19</td><td>97.03→97.51</td><td>99.60→99.61</td><td>64.05→67.34</td><td>63.54→64.32</td><td>97.60→98.29</td></tr><tr><td>cashew</td><td>98.48→98.48</td><td>99.36→99.35</td><td>97.49→96.48</td><td>97.91→96.36</td><td>64.18→61.39</td><td>62.17→61.15</td><td>94.81→93.57</td></tr><tr><td>chewinggum</td><td>99.86→99.86</td><td>99.93→99.93</td><td>99.00→98.52</td><td>99.28→99.20</td><td>64.59→75.91</td><td>66.86→70.49</td><td>90.79→89.38</td></tr><tr><td>fryum</td><td>99.42→99.36</td><td>99.71→99.69</td><td>97.54→97.51</td><td>96.57→96.15</td><td>48.57→47.31</td><td>52.05→50.61</td><td>93.26→93.58</td></tr><tr><td>macaroni1</td><td>98.45→98.62</td><td>98.38→98.54</td><td>94.74→95.57</td><td>99.69→99.71</td><td>33.49→37.50</td><td>39.81→41.72</td><td>97.09→97.37</td></tr><tr><td>macaroni2</td><td>97.73→97.48</td><td>97.78→97.60</td><td>93.66→91.51</td><td>99.80→99.80</td><td>25.89→28.04</td><td>36.86→37.93</td><td>98.78→98.83</td></tr><tr><td>pcb1</td><td>98.97→99.11</td><td>98.91→99.08</td><td>96.55→96.55</td><td>99.64→99.61</td><td>85.71→86.48</td><td>79.10→79.76</td><td>95.67→95.87</td></tr><tr><td>pcb2</td><td>98.74→99.12</td><td>98.17→98.88</td><td>98.02→99.01</td><td>98.79→98.75</td><td>36.43→38.91</td><td>43.20→44.40</td><td>90.79→90.67</td></tr><tr><td>pcb3</td><td>99.39→99.33</td><td>99.42→99.37</td><td>96.55→96.55</td><td>99.12→99.08</td><td>35.62→33.20</td><td>43.33→43.10</td><td>94.96→94.18</td></tr><tr><td>pcb4</td><td>99.91→99.89</td><td>99.91→99.89</td><td>98.52→98.49</td><td>98.78→98.78</td><td>50.73→52.04</td><td>52.83→53.22</td><td>93.74→94.61</td></tr><tr><td>pipe_fryum</td><td>99.76→99.64</td><td>99.89→99.84</td><td>99.00→99.00</td><td>99.05→99.14</td><td>55.89→61.78</td><td>62.88→65.01</td><td>94.20→96.49</td></tr></table>

Table 24: Category-level seed-0 results for Real-IAD Reconstruction. Cells with arrows report baseline → BoundarySupport.
<table><tr><td rowspan=1 colspan=8>Category           I-AUROC         I-AP         I-F1    P-AUROC         P-AP         P-F1      AUPRO</td></tr><tr><td rowspan=1 colspan=2>audiojack       94.22→93.78</td><td rowspan=1 colspan=1>91.73→91.35</td><td rowspan=1 colspan=1>82.38→80.92</td><td rowspan=1 colspan=1>99.81→99.82</td><td rowspan=1 colspan=1>62.21→66.55</td><td rowspan=1 colspan=1>61.46→64.26</td><td rowspan=1 colspan=1>98.16→98.31</td></tr><tr><td rowspan=1 colspan=1>bottle_cap</td><td rowspan=1 colspan=1>95.78→95.49</td><td rowspan=1 colspan=1>95.75→95.34</td><td rowspan=1 colspan=1>88.93→87.78</td><td rowspan=1 colspan=1>99.56→99.57</td><td rowspan=1 colspan=1>34.92→38.60</td><td rowspan=1 colspan=1>38.69→41.58</td><td rowspan=1 colspan=1>97.14→96.95</td></tr><tr><td rowspan=1 colspan=1>button_battery</td><td rowspan=1 colspan=1>90.96→89.80</td><td rowspan=1 colspan=1>92.74→92.19</td><td rowspan=1 colspan=1>87.23→86.29</td><td rowspan=1 colspan=1>99.50→99.54</td><td rowspan=1 colspan=1>54.89→62.53</td><td rowspan=1 colspan=1>65.91→67.10</td><td rowspan=1 colspan=1>96.58→96.36</td></tr><tr><td rowspan=1 colspan=1>end_cap</td><td rowspan=1 colspan=1>87.06→84.72</td><td rowspan=1 colspan=1>87.12→84.88</td><td rowspan=1 colspan=1>85.65→84.36</td><td rowspan=1 colspan=1>99.29→99.28</td><td rowspan=1 colspan=1>17.66→17.46</td><td rowspan=1 colspan=1>26.22→26.72</td><td rowspan=1 colspan=1>96.86→97.17</td></tr><tr><td rowspan=1 colspan=1>eraser</td><td rowspan=1 colspan=1>93.78→93.72</td><td rowspan=1 colspan=1>93.00→92.74</td><td rowspan=1 colspan=1>85.28→84.72</td><td rowspan=1 colspan=1>99.78→99.82</td><td rowspan=1 colspan=1>51.88→58.67</td><td rowspan=1 colspan=1>53.61→58.13</td><td rowspan=1 colspan=1>98.05→98.23</td></tr><tr><td rowspan=1 colspan=1>fire_hood</td><td rowspan=1 colspan=1>95.96→94.89</td><td rowspan=1 colspan=1>92.61→89.27</td><td rowspan=1 colspan=1>87.32→84.59</td><td rowspan=1 colspan=1>99.77→99.78</td><td rowspan=1 colspan=1>57.42→63.05</td><td rowspan=1 colspan=1>58.13→59.93</td><td rowspan=1 colspan=1>98.09→97.88</td></tr><tr><td rowspan=1 colspan=1>mint</td><td rowspan=1 colspan=1>78.32→77.79</td><td rowspan=1 colspan=1>85.63→85.08</td><td rowspan=1 colspan=1>78.59→77.98</td><td rowspan=1 colspan=1>98.95→98.96</td><td rowspan=1 colspan=1>39.98→43.87</td><td rowspan=1 colspan=1>45.08→46.53</td><td rowspan=1 colspan=1>87.07→88.28</td></tr><tr><td rowspan=1 colspan=1>mounts</td><td rowspan=1 colspan=1>97.49→97.53</td><td rowspan=1 colspan=1>96.21→96.24</td><td rowspan=1 colspan=1>89.30→89.42</td><td rowspan=1 colspan=1>99.80→99.84</td><td rowspan=1 colspan=1>52.17→59.46</td><td rowspan=1 colspan=1>51.01→56.47</td><td rowspan=1 colspan=1>98.54→98.84</td></tr><tr><td rowspan=1 colspan=1>pcb</td><td rowspan=1 colspan=1>96.65→96.85</td><td rowspan=1 colspan=1>98.04→98.10</td><td rowspan=1 colspan=1>93.35→93.60</td><td rowspan=1 colspan=1>99.80→99.82</td><td rowspan=1 colspan=1>59.77→63.02</td><td rowspan=1 colspan=1>59.10→61.03</td><td rowspan=1 colspan=1>98.27→98.41</td></tr><tr><td rowspan=1 colspan=1>phone_battery</td><td rowspan=1 colspan=1>96.32→95.89</td><td rowspan=1 colspan=1>96.78→96.35</td><td rowspan=1 colspan=1>90.63→89.56</td><td rowspan=1 colspan=1>99.85→99.86</td><td rowspan=1 colspan=1>61.16→66.35</td><td rowspan=1 colspan=1>57.60→61.10</td><td rowspan=1 colspan=1>98.52→98.67</td></tr><tr><td rowspan=1 colspan=1>plastic_nut</td><td rowspan=1 colspan=1>95.27→95.27</td><td rowspan=1 colspan=1>87.90→87.58</td><td rowspan=1 colspan=1>84.62→82.67</td><td rowspan=1 colspan=1>99.79→99.78</td><td rowspan=1 colspan=1>47.37→50.52</td><td rowspan=1 colspan=1>47.22→49.72</td><td rowspan=1 colspan=1>97.66→97.68</td></tr><tr><td rowspan=1 colspan=1>plastic_plug</td><td rowspan=1 colspan=1>96.70→95.95</td><td rowspan=1 colspan=1>95.76→94.63</td><td rowspan=1 colspan=1>90.08→88.21</td><td rowspan=1 colspan=1>99.71→99.74</td><td rowspan=1 colspan=1>34.32→43.80</td><td rowspan=1 colspan=1>40.97→46.72</td><td rowspan=1 colspan=1>98.66→98.60</td></tr><tr><td rowspan=1 colspan=1>porcelain_doll</td><td rowspan=1 colspan=1>95.47→95.28</td><td rowspan=1 colspan=1>92.37→92.01</td><td rowspan=1 colspan=1>86.01→85.71</td><td rowspan=1 colspan=1>99.71→99.68</td><td rowspan=1 colspan=1>33.36→37.59</td><td rowspan=1 colspan=1>38.86→41.32</td><td rowspan=1 colspan=1>98.92→98.84</td></tr><tr><td rowspan=1 colspan=1>regulator</td><td rowspan=1 colspan=1>96.70→96.30</td><td rowspan=1 colspan=1>88.19→89.48</td><td rowspan=1 colspan=1>83.58→84.75</td><td rowspan=1 colspan=1>99.94→99.93</td><td rowspan=1 colspan=1>55.60→59.96</td><td rowspan=1 colspan=1>57.56→59.30</td><td rowspan=1 colspan=1>99.58→99.58</td></tr><tr><td rowspan=1 colspan=1>rolled_strip_base</td><td rowspan=1 colspan=1>99.71→99.66</td><td rowspan=1 colspan=1>99.87→99.85</td><td rowspan=1 colspan=1>98.83→98.93</td><td rowspan=1 colspan=1>99.76→99.77</td><td rowspan=1 colspan=1>50.65→47.72</td><td rowspan=1 colspan=1>50.68→50.55</td><td rowspan=1 colspan=1>98.78→98.84</td></tr><tr><td rowspan=1 colspan=1>sim_card_set</td><td rowspan=1 colspan=1>99.17→98.91</td><td rowspan=1 colspan=1>99.38→99.17</td><td rowspan=1 colspan=1>96.60→95.74</td><td rowspan=1 colspan=1>99.78→99.81</td><td rowspan=1 colspan=1>64.71→68.84</td><td rowspan=1 colspan=1>61.18→63.38</td><td rowspan=1 colspan=1>97.99→98.28</td></tr><tr><td rowspan=1 colspan=1>switch</td><td rowspan=1 colspan=1>99.11→99.00</td><td rowspan=1 colspan=1>99.52→99.46</td><td rowspan=1 colspan=1>96.62→96.35</td><td rowspan=1 colspan=1>99.55→99.29</td><td rowspan=1 colspan=1>63.06→65.66</td><td rowspan=1 colspan=1>62.96→63.36</td><td rowspan=1 colspan=1>98.38→97.80</td></tr><tr><td rowspan=1 colspan=1>tape</td><td rowspan=1 colspan=1>99.58→99.63</td><td rowspan=1 colspan=1>99.33→99.42</td><td rowspan=1 colspan=1>96.63→96.88</td><td rowspan=1 colspan=1>99.87→99.90</td><td rowspan=1 colspan=1>46.09→52.29</td><td rowspan=1 colspan=1>51.06→56.29</td><td rowspan=1 colspan=1>99.47→99.53</td></tr><tr><td rowspan=1 colspan=1>terminalblock</td><td rowspan=1 colspan=1>99.26→99.15</td><td rowspan=1 colspan=1>99.54→99.48</td><td rowspan=1 colspan=1>98.07→97.94</td><td rowspan=1 colspan=1>99.89→99.90</td><td rowspan=1 colspan=1>60.27→63.63</td><td rowspan=1 colspan=1>60.30→62.94</td><td rowspan=1 colspan=1>99.32→99.41</td></tr><tr><td rowspan=1 colspan=1>toothbrush</td><td rowspan=1 colspan=1>91.41→91.03</td><td rowspan=1 colspan=1>95.48→95.27</td><td rowspan=1 colspan=1>87.89→86.47</td><td rowspan=1 colspan=1>99.03→99.04</td><td rowspan=1 colspan=1>48.64→49.33</td><td rowspan=1 colspan=1>53.39→53.95</td><td rowspan=1 colspan=1>94.69→95.06</td></tr><tr><td rowspan=1 colspan=1>toy</td><td rowspan=1 colspan=1>88.94→88.12</td><td rowspan=1 colspan=1>94.96→94.43</td><td rowspan=1 colspan=1>86.17→85.58</td><td rowspan=1 colspan=1>92.28→91.21</td><td rowspan=1 colspan=1>23.18→24.24</td><td rowspan=1 colspan=1>33.36→35.01</td><td rowspan=1 colspan=1>87.59→86.17</td></tr><tr><td rowspan=1 colspan=1>toy_brick</td><td rowspan=1 colspan=1>82.53→79.90</td><td rowspan=1 colspan=1>83.28→79.00</td><td rowspan=1 colspan=1>73.80→71.40</td><td rowspan=1 colspan=1>97.92→97.93</td><td rowspan=1 colspan=1>48.33→54.91</td><td rowspan=1 colspan=1>53.59→56.43</td><td rowspan=1 colspan=1>82.54→84.16</td></tr><tr><td rowspan=1 colspan=1>transistor1</td><td rowspan=1 colspan=1>96.67→96.81</td><td rowspan=1 colspan=1>98.03→98.08</td><td rowspan=1 colspan=1>93.42→94.01</td><td rowspan=1 colspan=1>99.38→99.42</td><td rowspan=1 colspan=1>48.90→52.34</td><td rowspan=1 colspan=1>49.40→52.25</td><td rowspan=1 colspan=1>97.10→97.53</td></tr><tr><td rowspan=1 colspan=1>u_block</td><td rowspan=1 colspan=1>96.49→96.47</td><td rowspan=1 colspan=1>92.62→92.22</td><td rowspan=1 colspan=1>86.89→85.95</td><td rowspan=1 colspan=1>99.80→99.81</td><td rowspan=1 colspan=1>43.79→48.91</td><td rowspan=1 colspan=1>47.50→50.80</td><td rowspan=1 colspan=1>97.80→98.03</td></tr><tr><td rowspan=1 colspan=1>usb</td><td rowspan=1 colspan=1>96.92→96.92</td><td rowspan=1 colspan=1>96.35→96.35</td><td rowspan=1 colspan=1>93.04→92.75</td><td rowspan=1 colspan=1>99.83→99.84</td><td rowspan=1 colspan=1>52.28→54.60</td><td rowspan=1 colspan=1>54.05→55.55</td><td rowspan=1 colspan=1>99.13→99.12</td></tr><tr><td rowspan=1 colspan=1>usb_adaptor</td><td rowspan=1 colspan=1>93.04→92.48</td><td rowspan=1 colspan=1>90.68→90.23</td><td rowspan=1 colspan=1>84.42→83.13</td><td rowspan=1 colspan=1>99.72→99.75</td><td rowspan=1 colspan=1>29.59→33.06</td><td rowspan=1 colspan=1>35.63→38.72</td><td rowspan=1 colspan=1>97.73→98.27</td></tr><tr><td rowspan=1 colspan=1>vcpill</td><td rowspan=1 colspan=1>94.64→94.36</td><td rowspan=1 colspan=1>93.19→93.10</td><td rowspan=1 colspan=1>85.01→85.29</td><td rowspan=1 colspan=1>98.93→98.96</td><td rowspan=1 colspan=1>73.67→77.06</td><td rowspan=1 colspan=1>71.21→73.55</td><td rowspan=1 colspan=1>91.85→91.83</td></tr><tr><td rowspan=1 colspan=1>wooden_beads</td><td rowspan=1 colspan=1>88.80→87.52</td><td rowspan=1 colspan=1>92.19→90.81</td><td rowspan=1 colspan=1>82.92→81.78</td><td rowspan=1 colspan=1>98.90→98.91</td><td rowspan=1 colspan=1>46.64→50.52</td><td rowspan=1 colspan=1>50.16→53.42</td><td rowspan=1 colspan=1>90.71→92.07</td></tr><tr><td rowspan=1 colspan=1>woodstick</td><td rowspan=1 colspan=1>91.37→89.47</td><td rowspan=1 colspan=1>78.33→74.19</td><td rowspan=1 colspan=1>73.04→66.97</td><td rowspan=1 colspan=1>99.58→99.58</td><td rowspan=1 colspan=1>56.81→65.52</td><td rowspan=1 colspan=1>58.70→62.92</td><td rowspan=1 colspan=1>93.93→93.80</td></tr><tr><td rowspan=1 colspan=1>zipper</td><td rowspan=1 colspan=1>99.71→99.64</td><td rowspan=1 colspan=1>99.85→99.82</td><td rowspan=1 colspan=1>98.11→98.10</td><td rowspan=1 colspan=1>99.16→99.13</td><td rowspan=1 colspan=1>64.53→69.73</td><td rowspan=1 colspan=1>67.21→69.66</td><td rowspan=1 colspan=1>98.04→98.19</td></tr></table>

## L Score Changes and Failure Cases

## L.1 Reconstruction score changes

Table 25: Seed-0 reconstruction score changes.
<table><tr><td>Dataset</td><td>∆normal  $\mathsf { q } 9 5$ </td><td>∆normal  $\boldsymbol { \mathrm { q 9 9 } }$ </td><td>∆defect mean</td><td>∆P-AP</td></tr><tr><td>MVTec</td><td>-1.91</td><td>-4.91</td><td>-4.01</td><td>+4.94</td></tr><tr><td>VisA</td><td>-0.28</td><td>-0.88</td><td>-1.29</td><td>+2.09</td></tr><tr><td>Real-IAD</td><td>-0.32</td><td>-1.13</td><td>-2.49</td><td>+4.20</td></tr></table>

At seed 0, high anomaly scores on normal regions and mean defect scores decrease together on all three datasets while P-AP increases. The same macro direction repeats over all completed MVTec and VisA seeds.

## L.2 Memory score changes

For memory, we also report the excess normal score

$$
e _ { q } = Q _ { q } ( \mathrm { n o r m a l p a t c h e s i n d e f e c t i v e i m a g e s } ) - Q _ { q } ( \mathrm { p a t c h e s i n c l e a n i m a g e s } ) .
$$

Table 26: Seed-0 memory score changes under the uniform headline policies.
<table><tr><td>Dataset</td><td>Abs. q95</td><td>Abs. q99</td><td>Excess q95</td><td>Excess q99</td><td>Defect mean</td><td>∆P-AP</td></tr><tr><td>MVTec</td><td>-0.35</td><td>-1.02</td><td>-0.51</td><td>-1.14</td><td>-0.51</td><td>+2.26</td></tr><tr><td>VisA</td><td>+0.30</td><td>-0.43</td><td>-0.33</td><td>-0.92</td><td>-0.52</td><td>+1.82</td></tr><tr><td>Real-IAD</td><td>+0.55</td><td>+0.09</td><td>-0.11</td><td>-0.48</td><td>-0.21</td><td>+1.76</td></tr></table>

Absolute normal scores do not move in one direction on every dataset. The defect-image-minusclean-image excess q95/q99 decreases in all three settings.

## L.3 Representative failure cases

Table 27: Selected seed-0 negative and metric-trade-off cases.
<table><tr><td>Case</td><td>Observed change</td></tr><tr><td>Reconstruction / VisA / cashew</td><td>P-AP -2.79; P-AUROC -1.56; q99 -0.74; defect mean -1.79.</td></tr><tr><td>Reconstruction / VisA / pcb3</td><td>P-AP -2.42; AUPRO -0.78.</td></tr><tr><td>Reconstruction / VisA / fryum</td><td>P-AP -1.26; AUPRO +0.32.</td></tr><tr><td>Reconstruction / Real-IAD / rolled strip base</td><td>P-AP -2.92 while P-AUROC and AUPRO increase slightly.</td></tr><tr><td>Reconstruction / Real-IAD / end cap</td><td>P-AP -0.20; I-AUROC -2.33.</td></tr><tr><td>Memory / Real-IAD / u block</td><td>P-AP +3.32 while absolute q95 and q99 increase.</td></tr></table>

At seed 0, P-AP is positive in 57/57 memory categories and 52/57 reconstruction categories. Across all three seeds, the corresponding sign counts are 171/171 and 158/171. The clearest aggregate trade-off is Real-IAD reconstruction: P-AP/P-F1/AUPRO change by $+ 4 . 2 0 \pm 0 . 1 2 , + 2 . 5 6 \pm 0 . 0 9 , \mathrm { a n d } + 0 . 1 6 \pm 0 . 0 3 ,$ while I-AUROC changes by $- 0 . 5 2 \pm 0 . 0 4$

## M Uniform VisA Memory Comparison

The final VisA memory result applies one uniform q50/q90 rule to all 12 categories and all three seeds. A previously explored mixed policy was evaluated at seed 0 and reached 54.12 P-AP, compared with 54.05 for the uniform seed-0 policy and 52.23 for the clean baseline. The 0.07-point difference is small relative to the gain over the clean memory; the main paper therefore uses the uniform policy.

## N Compute and Reproducibility Details

Memory feature caches and banks are stored in FP16, while candidate, coreset, and readout distances are computed in FP32. Reconstruction uses FP32 training, 5,000 iterations per category, and batch size 16.

Table 28: Seed-0 reconstruction category-GPU hours. Totals are sums over categories, not parallel wall-clock time.
<table><tr><td>Dataset</td><td>Baseline hours</td><td>BoundarySupport hours</td><td>Ratio</td></tr><tr><td>MVTec</td><td>5.81</td><td>15.17</td><td>2.61×</td></tr><tr><td>VisA</td><td>4.99</td><td>12.71</td><td>2.55×</td></tr><tr><td>Real-IAD</td><td>12.60</td><td>31.21</td><td>2.48×</td></tr></table>

BoundarySupport reconstruction increases training cost in these runs while leaving the deployed architecture and scoring path unchanged. The memory branch likewise keeps the final bank size and readout fixed; its added work occurs during offline synthesis, candidate scoring, and bank reconstruction.

The release artifact will include category-level result files, fixed-budget and distance-diagnostic aggregation, intervention and mask construction, memory candidate selection, reconstruction target controls, and table/figure regeneration scripts. Reported tables are generated from category-level results using ROUND HALF UP to two decimals; algorithm-defining constants retain their stated precision.

## O Full 57-Category Qualitative Visualization

The following pages provide an all-category qualitative comparison across MVTec AD, VisA, and Real-IAD. Each row follows the same column order: input image, ground-truth mask, ProCon, ProCon + BoundarySupport, Dinomaly, and Dinomaly + BoundarySupport. The quantitative category-level results in Appendix K remain the basis for performance claims; these figures show the corresponding spatial predictions across all 57 benchmark categories.

MVTec AD.  
![](images/1c860c936153eaac2bf2ea395c7cd8e5f5648e0807631eeb5168ab6d22f5c093.jpg)

![](images/88318c21c67ee00acd130000a7b25f804823347ee13bded87e92988b6ff10f91.jpg)

![](images/bd7cb4edc1b1f708a13b57b44ef15ae78f1a1f1a306aa3fd14ad6c5d94fa1888.jpg)

VisA.  
![](images/73d45ea00bdeb2646245097d286dd21e0f2abd2cefe0bf6882732a4c4207c6d3.jpg)

![](images/6a2b5cb3e59b325700db5c5a80801dac7e56f589eaf937c8a8b374eb74359670.jpg)

![](images/4c761d3d4663ebb19d94b7fa50ec93c535a35176ca6bd2a5bda921c8b7ea5570.jpg)

Real-IAD.  
![](images/99cd46eb94998ba2d7976ab2c8d1a97c8a7e650f0f956e909e1f1fd37b4c560c.jpg)

![](images/b1ea7a6a1fa7ffec4408f771ba38a5aef5816788f80677a6867d8910d98d14da.jpg)

![](images/6e603ca83e69cd3785dcece1ba09ae6e437b870f9d2b27460e94489f09e4a8a4.jpg)

![](images/6c5fdfddcc99175202ef8c5f1452cbdf2c16a52801f3936610da40b909d69a47.jpg)

![](images/850e852fcb07805047fde4b7efe7c3c94eb0a676d7cf88056c037f23791b9e71.jpg)

![](images/b1aa9776445c8bbde03eb15b3650f36288032567223677cba4812e277d6bfb33.jpg)