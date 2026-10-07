# View Matters: Keyframe-Guided Text-Driven 3D Gaussian Editing

Kaizhe Zhang<sup>1</sup>, Yijie Zhou, Weizhan Zhang<sup>1\*</sup>, Xuanyu Wang, Feng Lei, Sha Gong

<sup>1</sup>School of Computer Science and Technology, MOE-KLINNS Lab

Xi’an Jiaotong University, Xi’an, Shaanxi 710049, China

zkz1081@stu.xjtu.edu.cn, zhangwzh@xjtu.edu.cn

## Abstract

Text-driven 3D Gaussian editing commonly does not distinguish the editing reliability of rendered views, although different viewpoints provide supervision of substantially different quality. Views that clearly show the scene and match the edit instruction provide reliable guidance, while less informative views may weaken the edit when all views are treated equally. We present View Matters, a viewimportance-aware framework that conducts editing around reliable keyframes. Keyframe Importance Estimation (KIE) identifies reliable views using geometric visibility, semantic distinctiveness, and edit relevance. Keyframe-Guided Editing (KGE) then propagates their editing signals asymmetrically to non-keyframes without noisy reverse influence, while Importance-Aware Optimization (IAO) preserves this reliability preference during 3DGS optimization. Across 23 scene–prompt pairs, View Matters achieves the highest average CLIP text–image similarity of 0.2822 and directional similarity of 0.2564 among the evaluated methods, with a four-minute editing time. Additional adjacent-view analysis indicates that the fidelity-oriented editing process maintains cross-view coherence.

## 1 Introduction

Text-driven 3D editing modifies reconstructed scenes from natural-language instructions. Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS) enable this task through differentiable rendering and pretrained image editors Haque et al. (2023); Wang et al. (2024). In particular, the explicit representation and efficient rendering of 3DGS make it well suited for iterative 3D editing Kerbl et al. (2023).

Most text-driven 3DGS editing methods follow a render– edit–optimize pipeline: a scene is rendered from multiple viewpoints, the rendered images are edited by a pretrained 2D model, and the edited observations are used to optimize the 3D representation Wang et al. (2024); Wu et al. (2024). A limitation of this paradigm is that editing supervision is not equally reliable across viewpoints. Informative views provide faithful editing signals, whereas ambiguous views may produce weak or inaccurate edits. Treating all views equally allows these unreliable signals to propagate into and accumulate during 3DGS optimization, reducing the fidelity of the edit.

Existing methods improve multi-view agreement through depth guidance, feature fusion, epipolar interaction, or shared attention Wu et al. (2024); Chen et al. (2024); Lee et al. (2025). Some methods already designate reference or key views for cross-view propagation. However, they do not explicitly determine which views should dominate the editing process. Symmetric interaction or uniform supervision, therefore, allows less reliable views to influence more reliable ones, making the final edit less faithful to the instruction. Consequently, the resulting views may agree with one another while expressing an edit that is weak or insufficiently faithful to the instruction.

Our key observation is that keyframe selection has a substantial impact on the final editing quality. This suggests that view-guided editing requires more than simply designating a set of anchors: the source–target roles should be assigned according to edit-specific reliability and maintained throughout both multi-view editing and subsequent 3D optimization. A view that provides clearer geometric, semantic, and editrelated evidence should provide guidance for a less informative view.

Based on this observation, we propose View Matters. Keyframe Importance Estimation (KIE) determines which views provide reliable evidence for the requested edit. Keyframe-Guided Editing (KGE) transfers editing signals from these keyframes to non-keyframes without noisy reverse influence. Importance-Aware Optimization (IAO) preserves the same reliability preference during 3DGS optimization, preventing reliable supervision from being diluted when edited views are projected back to the 3D representation. Shared keyframe guidance and local propagation maintain cross-view coherence throughout this process.

Our contributions are threefold:

• We formulate text-driven 3DGS editing as an unequalview-reliability problem and propose KIE to identify reliable editing anchors using geometric, semantic, and edit-related cues.

• We introduce KGE to propagate reliable editing signals from keyframes to non-keyframes without reverse contamination and IAO to preserve the same reliability prior during 3DGS optimization.

• Across 23 scene-prompt pairs, View Matters achieves the best average CLIP text–image and directional similarities among the evaluated methods with a four-minute editing time. Ablations verify the contribution of all three modules, while adjacent-view analysis indicates that improved fidelity does not sacrifice cross-view coherence.

## 2 Related Work

## 2.1 2D Image Editing

Pretrained diffusion models provide strong semantic priors for text-guided image editing. P2P Hertz et al. (2022) controls edits through cross-attention, while ControlNet Zhang et al. (2023) introduces additional structural conditions. Instruct-Pix2Pix (IP2P) Brooks et al. (2023) performs instructionbased image editing without requiring a user-provided target image. We adopt a frozen IP2P editor as the editing backbone and reorganize its attention for reliability-guided multi-view interaction.

## 2.2 Text-driven 3D Editing

Due to the scarcity of large-scale paired 3D editing data, most text-driven 3D editing methods transfer the priors of pretrained 2D diffusion models to 3D. Related 3D generation methods, such as DreamFusion Poole et al. (2022) and ConsistentDreamer ¸Sahin et al. (2025), synthesize new 3D content from text or a single image. Although they do not edit existing reconstructions, they demonstrate the effectiveness of diffusion guidance for 3D optimization. For editing existing scenes, NeRF-based approaches either optimize the underlying representation using score distillation or related diffusion objectives, as in Instruct 3D-to-3D Kamata et al. (2023), DreamEditor Zhuang et al. (2023), and Progressive3D Cheng et al. (2023), or follow a render-edit-optimize pipeline, as exemplified by Instruct-NeRF2NeRF Haque et al. (2023) and related follow-ups Liu et al. (2024); Wang et al. (2022, 2023). These methods establish the feasibility of transferring 2D editing priors to 3D, but NeRF optimization is often slow, while independently edited views may provide inconsistent supervision.

With the emergence of 3DGS, text-driven 3D editing methods increasingly adopt explicit Gaussian representations for more efficient rendering and optimization. GaussianEditor Wang et al. (2024) improves editing efficiency and controllability by leveraging the explicit structure of Gaussians. More recent methods, including GaussCtrl Wu et al. (2024), DGE Chen et al. (2024), EditSplat Lee et al. (2025), and D<sup>2</sup>Gaussian Sheng et al. (2025), further improve multi-view consistency or editing control through depth conditioning and latent alignment, epipolar-guided feature propagation, multiview fusion, and discretized view modeling, respectively. These methods primarily improve how editing information is aligned or propagated across views. View Matters instead explicitly estimates which views provide reliable evidence for the requested edit and preserves this view-reliability hierarchy throughout both multi-view editing and 3D optimization.

## 2.3 View Selection and Multi-View Interaction

View selection has been used to reduce redundancy and improve multi-view reconstruction Hornung et al. (2008). Neural rendering methods quantify useful observations through uncertainty reduction, Fisher information, or reconstructionaware criteria Pan et al. (2022); Jiang et al. (2024); Xiao et al. (2024). Their objective is to acquire or retain views that best reconstruct a scene. Text-driven editing poses a different question: a useful view must expose scene geometry and carry distinctive content and evidence relevant to a language instruction.

Multi-frame diffusion methods commonly reuse attention features from reference frames to stabilize generated content Ceylan et al. (2023). We adopt this mechanism but organize the interaction asymmetrically: a small set of keyframes interacts globally and then guides non-keyframes together with local context. This one-way guidance prevents less informative views from weakening reliable keyframe signals.

## 3 Method

## 3.1 Problem Formulation

Given a 3D Gaussian scene $G = \{ g _ { m } \} _ { m = 1 } ^ { M }$ and camera views $V = \{ v _ { i } \} _ { i = 1 } ^ { N }$ , the renderer produces $\mathbf { x } _ { i } = \mathcal { R } ( G , v _ { i } )$ . Given an editing instruction P, a pretrained image editor produces an edited observation $\tilde { \mathbf { x } } _ { i } = \mathcal { E } ( \mathbf { x } _ { i } , \mathbf { P } )$ . Our method partitions the view set into keyframes $V ^ { k }$ and non-keyframes $V ^ { n } =$ $V \setminus V ^ { k }$ , which provide different levels of supervision during editing and 3D optimization.

In addition to RGB images, the renderer also provides view-dependent geometric cues, such as depth and visibility, under each view. These cues support view-importance estimation.

Besides the edited image itself, we denote by

$$
\mathbf { F } _ { i } = \Phi ( \mathbf { x } _ { i } , \mathbf { P } )\tag{1}
$$

the intermediate feature representation extracted from the editor, where $\Phi ( \cdot )$ represents the feature extraction process.

## 3.2 Overview

As shown in Figure 1, given a 3D Gaussian scene and a text prompt, our goal is to produce an edited 3D scene that is both semantically faithful to the instruction and consistent across views. Instead of treating all rendered views equally, we perform the editing process under the guidance of a small set of representative keyframes. Our framework consists of three stages: 1) Keyframe Importance Estimation (KIE), designed to score rendered views and determine representative keyframes; 2) Keyframe-Guided Editing (KGE), developed to propagate editing signals from reliability-selected keyframes to non-keyframes through directed attention; and 3) Importance-Aware Optimization (IAO), introduced to prioritize representative keyframes via loss reweighting while retaining the original reconstruction objective.

![](images/e77ac18c4d901f08b41f2a4e955aa04dd7cb01cdb4136c1d6946de6a618d8a0c.jpg)  
Figure 1: Overview of View Matters. Given a 3DGS scene and an editing instruction, KIE estimates view reliability and selects representative keyframes. KGE lets these keyframes establish a shared context and asymmetrically guide less-reliable non-keyframes without reverse interference. IAO then preserves the reliability hierarchy by assigning stronger supervision to keyframes during 3DGS optimization. The resulting scene follows the requested edit faithfully while maintaining cross-view coherence.

## 3.3 Keyframe Importance Estimation (KIE)

Keyframes serve as the main source of editing guidance, a natural question is which rendered views should be selected as keyframes. We address this question by estimating view importance from three complementary perspectives: geometric visibility, semantic distinctiveness Radford et al. (2021), and edit relevance Fu et al. (2023). The normalized scores are then fused into a final importance score, based on which representative keyframes are selected.

Geometric Visibility. Because a 3DGS representation is reconstructed from multi-view observations, a view that observes more Gaussians covers a larger portion of the reconstructed geometry and can provide broader geometric evidence for editing. We therefore use the ratio of visible Gaussians as a geometry-based cue of view importance. Let M denote the total number of Gaussians, and let $\mathcal { k } ( \cdot )$ be the visibility indicator provided by the 3DGS renderer Kerbl et al. (2023). The geometry visibility score of view $v _ { i }$ is defined as

$$
s _ { i } ^ { \mathrm { g e o } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathcal { H } ( g _ { m } \mathrm { ~ i s ~ v i s i b l e ~ i n ~ } v _ { i } ) ,\tag{2}
$$

which corresponds to the ratio of visible Gaussians under view $v _ { i }$

Semantic Distinctiveness. Semantically distinctive views are more likely to be informative keyframes, as they capture unusual content that cannot be readily inferred from other views. We then measure how distinctive each view is in the semantic feature space. Let $\phi _ { \mathrm { i m g } } ( \cdot )$ denote the CLIP image encoder Radford et al. (2021). We extract normalized image features for all candidate views and compute their mean feature:

$$
\mathbf { f } _ { i } = \frac { \phi _ { \mathrm { i m g } } ( \mathbf { x } _ { i } ) } { \| \phi _ { \mathrm { i m g } } ( \mathbf { x } _ { i } ) \| _ { 2 } } , \qquad \bar { \mathbf { f } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { f } _ { i } .\tag{3}
$$

The semantic distinctiveness score is then defined as the Euclidean distance between the feature of view $v _ { i }$ and the mean feature:

$$
s _ { i } ^ { \mathrm { s e m } } = \| \mathbf { f } _ { i } - \bar { \mathbf { f } } \| _ { 2 } .\tag{4}
$$

A larger value indicates that the view is more distinctive relative to the overall view distribution.

![](images/dd8aeef666ffa9f72f626d503c8b04f0e134b5f4d7fa9b01ec42347360b7cc62.jpg)  
Figure 2: Illustration of the proposed keyframe-guided attention mechanism. Our method modifies the attention computation inside a frozen IP2P denoising network without introducing additional trainable parameters. Keyframes first perform global interaction to establish a shared editing context. Each non-keyframe then attends to its associated keyframe together with the immediately preceding frame, yielding an asymmetrical propagation scheme that uses keyframes as the dominant guidance source while maintaining local continuity across views.

Edit Relevance. We also consider views that are more closely aligned with the editing instruction to be more suitable keyframes, as they provide more direct guidance for the intended transformation. To estimate how relevant a view is to the target editing instruction, we compute the similarity between the image feature and the text feature in the CLIP embedding space. Let $\phi _ { \mathrm { t x t } } ( \cdot )$ denote the CLIP text encoder Radford et al. (2021), and let

$$
\mathbf { t } = \frac { \phi _ { \mathrm { t x t } } ( \mathbf { P } ) } { \| \phi _ { \mathrm { t x t } } ( \mathbf { P } ) \| _ { 2 } }\tag{5}
$$

be the normalized text feature of the editing prompt P. The edit relevance score is defined as

$$
s _ { i } ^ { \mathrm { e d i t } } = \mathbf { f } _ { i } ^ { \top } \mathbf { t } .\tag{6}
$$

Score Fusion. The three cues are complementary, as a view weak in one aspect may remain informative in others. We therefore fuse them for robust keyframe selection. Each cue is independently min-max normalized across all candidate views, with $\epsilon = 1 0 ^ { - 8 }$ for numerical stability. Denoting the normalized scores by $\hat { s } _ { i } ^ { ( \cdot ) }$ , we compute the importance of view $v _ { i }$ as

$$
s _ { i } = w _ { \mathrm { g e o } } \hat { s } _ { i } ^ { \mathrm { g e o } } + w _ { \mathrm { s e m } } \hat { s } _ { i } ^ { \mathrm { s e m } } + w _ { \mathrm { e d i t } } \hat { s } _ { i } ^ { \mathrm { e d i t } } .\tag{7}
$$

We use $( w _ { \mathrm { g e o } } , w _ { \mathrm { s e m } } , w _ { \mathrm { e d i t } } ) = ( 0 . 6 , 0 . 2 , 0 . 2 )$ by default.

Group-wise Keyframe Selection. To ensure broad spatial coverage, views are ordered along the rendering trajectory and partitioned into $K$ contiguous groups:

$$
\{ \mathcal { G } _ { k } \} _ { k = 1 } ^ { K } , \bigcup _ { k = 1 } ^ { K } \mathcal { G } _ { k } = V .\tag{8}
$$

For each group $\mathcal { G } _ { k }$ , we select the view with the highest importance score as the representative keyframe. The resulting keyframe subset is defined as

$$
V ^ { k } = \left\{ \left. v _ { k } ^ { * } \right| v _ { k } ^ { * } = \arg \operatorname* { m a x } _ { i : v _ { i } \in \mathcal { G } _ { k } } { s _ { i } } , \ k = 1 , \ldots , K \right\} ,\tag{9}
$$

and the remaining views form the non-keyframe subset

$$
V ^ { n } = V \setminus V ^ { k } .\tag{10}
$$

## 3.4 Keyframe-Guided Editing (KGE)

Given the selected keyframes, we perform multi-view editing by modifying the attention computation in a pretrained IP2P model. We do not introduce additional trainable parameters. Instead, the IP2P network is kept frozen, and keyframeguided multi-view interaction is imposed directly within its attention layers during the denoising process. Figure 2 illustrates the proposed keyframe-guided attention mechanism.

Global Interaction among Keyframes. We first let the selected keyframes interact globally to establish a coherent editing context. For each keyframe $v _ { k } \in \mathsf { V } ^ { k }$ , we use its query, key, and value features from the frozen IP2P attention layer, denoted by $\mathbf { Q } _ { k } , \mathbf { K } _ { k } , \mathbf { V } _ { k }$ . We then aggregate information across all keyframes by

$$
\begin{array} { r } { \bar { \mathbf { F } } _ { k } = \mathrm { A t t n } \big ( \mathbf { Q } _ { k } , \mathbf { K } ^ { k } , \mathbf { V } ^ { k } \big ) , } \end{array}\tag{11}
$$

where

$$
{ \bf K } ^ { k } = [ { \bf K } _ { 1 } , \ldots , { \bf K } _ { K } ] , \qquad { \bf V } ^ { k } = [ { \bf V } _ { 1 } , \ldots , { \bf V } _ { K } ] ,\tag{12}
$$

and

$$
{ \mathrm { A t t n } } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = { \mathrm { s o f t m a x } } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } } \right) \mathbf { V } .\tag{13}
$$

This step builds a globally consistent semantic context among representative keyframes.

Asymmetrical Propagation to Non-keyframes. For each non-keyframe $v _ { n } \in V ^ { n }$ , we modify the IP2P attention computation in the decoder layers Ceylan et al. (2023) so that it receives editing guidance asymmetrically from its associated keyframe, rather than interacting symmetrically with all views. Specifically, $\kappa ( n ) \in V ^ { k }$ denotes the keyframe selected from the group containing $v _ { n }$ . The group keyframe provides reliable editing guidance, whereas the previous frame offers stronger local correspondence because adjacent views typically share greater visual overlap. We therefore use $v _ { n - 1 } .$ when available, as an auxiliary guidance source to preserve local cross-view continuity. The propagated feature of $v _ { n }$ is computed as

$$
\begin{array} { r } { \hat { \bf F } _ { n } = \mathrm { A t t n } ( { \bf Q } _ { n } , { \bf K } _ { n } ^ { \mathrm { m e m } } , { \bf V } _ { n } ^ { \mathrm { m e m } } ) , } \end{array}\tag{14}
$$

where the memory bank is defined as

$$
\begin{array} { r } { \mathbf { K } _ { n } ^ { \mathrm { m e m } } = \left\{ \begin{array} { l l } { [ \mathbf { K } _ { \kappa ( n ) } , \mathbf { K } _ { n - 1 } ] , } & { n > 1 , } \\ { [ \mathbf { K } _ { \kappa ( n ) } ] , } & { n = 1 , } \end{array} \right. } \\ { \mathbf { V } _ { n } ^ { \mathrm { m e m } } = \left\{ \begin{array} { l l } { [ \mathbf { V } _ { \kappa ( n ) } , \mathbf { V } _ { n - 1 } ] , } & { n > 1 , } \\ { [ \mathbf { V } _ { \kappa ( n ) } ] , } & { n = 1 . } \end{array} \right. } \end{array}\tag{15}
$$

This design is inherently asymmetric: the associated keyframe provides the primary editing guidance, while the previous frame supplies additional local context for smoother propagation. For the first frame, the propagation naturally degenerates to keyframe-only guidance. Since the modification is applied directly to frozen IP2P attention layers, the editing process remains training-free.

Edited Multi-view Observations. After the modified denoising process, the frozen IP2P model directly produces the edited result for each view. We denote the final edited observation of view $v _ { i }$ by

$$
\mathbf { y } _ { i } = \mathcal { E } _ { \mathrm { I P 2 P } } ( \mathbf { x } _ { i } , \mathbf { P } ) ,\tag{16}
$$

where $\mathcal { E } _ { \mathrm { I P 2 P } } ( \cdot )$ denotes the frozen IP2P editor with our keyframe-guided attention modification. For keyframes, the editing result is obtained under global keyframe interaction. For non-keyframes, the result is further guided by asymmetrical propagation from the associated keyframe and local neighboring views.

## 3.5 Importance-Aware Optimization (IAO)

KIE identifies reliable keyframes, and KGE uses them to guide the remaining views. However, treating all edited views equally during 3DGS optimization would discard this reliability ordering and allow less reliable edits to weaken keyframe supervision. We therefore introduce IAO to preserve keyframe priority throughout scene optimization.

After obtaining edited multi-view observations, we optimize the underlying 3D Gaussian scene using a reconstruction-based objective, while assigning higher optimization priority to representative keyframes through a lightweight loss reweighting strategy. Instead of redesigning the entire optimization pipeline, we keep the original training objective and only increase the contribution of keyframe views during optimization.

In practice, training is performed with batch size 1. Let $v _ { i }$ denote the currently sampled view, and let

$$
\mathbf { x } _ { i } ^ { r } = \mathcal { R } ( G , v _ { i } )\tag{17}
$$

be the rendered image of the current 3D Gaussian scene. Let $\mathbf { y } _ { i }$ denote the corresponding edited supervision image cached for view $v _ { i } .$ . We first compute a reconstruction loss between the current rendering and the edited target:

$$
\begin{array} { r } { \mathcal { L } _ { i } ^ { \mathrm { r e c } } = \lambda _ { l 1 } \mathcal { L } _ { l 1 } ( \mathbf { x } _ { i } ^ { r } , \mathbf { y } _ { i } ) + \lambda _ { p } \mathcal { L } _ { \mathrm { p e r c } } ( \mathbf { x } _ { i } ^ { r } , \mathbf { y } _ { i } ) , } \end{array}\tag{18}
$$

where $\mathcal { L } _ { l 1 }$ is the pixel-wise $\ell _ { 1 }$ loss Isola et al. (2017) and $\mathcal { L } _ { \mathrm { p e r c } }$ is a perceptual loss Zhang et al. (2018).

To emphasize the role of representative keyframes, we apply a keyframe-aware weighting factor to the reconstruction loss of the current training step:

$$
{ \tilde { \mathcal { L } } } _ { i } = \omega _ { i } { \mathcal { L } } _ { i } ^ { \mathrm { r e c } } ,\tag{19}
$$

where

$$
\omega _ { i } = \left\{ \begin{array} { l l } { \lambda _ { k } , ( \lambda _ { k } > 1 ) } & { v _ { i } \in V ^ { k } , } \\ { 1 , } & { v _ { i } \in V ^ { n } , } \end{array} \right.\tag{20}
$$

and $\lambda _ { k }$ is the keyframe loss weight. In this way, the optimization process remains identical to the original reconstructionbased pipeline except that keyframe views receive stronger supervision.

## 4 Experiments

In this section, we first introduce the experimental setup and compare our method with representative approaches. We then directly examine how keyframe selection affects editing quality, analyze the contributions and sensitivity of the proposed components, and finally verify that the fidelity-oriented design preserves cross-view coherence.

## 4.1 Experimental Setup

## 4.1.1 Datasets and Metrics.

Following prior works, we select subsets from Mip-NeRF 360 Barron et al. (2022) and IN2N Haque et al. (2023) for editing evaluation. We adopt CLIP text-image similarity (T-I) Radford et al. (2021) and CLIP directional similarity (Direction) Gal et al. (2022) to measure the alignment between edited results and text instructions. In addi tion, we use adjacent-view CLIP temporal scores Haque et al. (2023) to evaluate cross-view consistency, which is largely overlooked in previous methods. The evaluation contains 23 unique scene-prompt pairs covering object identity and appearance, material transformation, global environment and style, and localized editing.

![](images/e3fa2816499bcfbd1062e29901d28ae289f942b5cc07850e31f518381e440581.jpg)  
Make the scene appear as though it's underwater

## 4.1.2 Baselines.

We compare our method with representative NeRF-based approaches, including IN2N Haque et al. (2023) and VICA Dong and Wang (2023), as well as representative 3DGS-based approaches, including GaussianEditor Wang et al. (2024), GaussCtrl Wu et al. (2024), DGE Chen et al. (2024), and EditSplat Lee et al. (2025). These 3DGS baselines cover iterative dataset updating, depth- and attentionbased alignment, epipolar-constrained key-view propagation, and multi-view fusion, respectively.

## 4.1.3 Implementation Details.

Our method is implemented in PyTorch and can be optimized on a single 48G RTX 4090 GPU. For both NeRF-based and 3DGS-based baselines, the initial 3D representations are reconstructed from the same multi-view images under identical training settings. For fair comparison, all methods use the same input scenes, text prompts, and initialization whenever applicable. Additional details on hyperparameters, scene selection, and prompts are provided in Appendix A.

## 4.2 Quantitative Analysis

Table 1 shows that View Matters achieves the best average T-I (0.2822) and Direction (0.2564) among the evaluated methods. The directional score improves by 0.0337 over GaussianEditor and 0.0412 over EditSplat, indicating more accurate instruction-following changes. Our method also completes editing in four minutes, faster than the evaluated baselines. These results show the overall benefit of organizing editing around reliable views.

Figure 3: The two examples span complementary editing regimes: a localized edit of a human subject (top) and a global transformation of an outdoor scene (bottom). Our method more faithfully follows both instructions while preserving editingirrelevant content and the underlying scene structure. Additional comparisons across scenes and edit instructions are provided in Appendix C.
<table><tr><td>Method</td><td>T-I↑</td><td>Direction ↑</td><td>Time ↓</td></tr><tr><td>IN2N</td><td>0.2599</td><td>0.1767</td><td>28 min</td></tr><tr><td>VICA</td><td>0.2439</td><td>0.1565</td><td>25 min</td></tr><tr><td>GaussianEditor</td><td>0.2769</td><td>0.2227</td><td>8min</td></tr><tr><td>GaussCtrl</td><td>0.2589</td><td>0.1572</td><td>10 min</td></tr><tr><td>EditSplat</td><td>0.2719</td><td>0.2152</td><td>7 min</td></tr><tr><td>DGE</td><td>0.2753</td><td>0.2179</td><td>5 min</td></tr><tr><td>Ours</td><td>0.2822</td><td>0.2564</td><td>4 min</td></tr></table>

Table 1: Quantitative comparison with representative textdriven 3D editing methods. Best results are in bold, and second-best results are underlined.

## 4.3 Qualitative Analysis

Figure 3 presents two representative cases selected to span complementary regimes in both content type and spatial extent: localized editing of a human subject and global appearance editing of an outdoor scene. The former tests whether a method can apply the requested semantic change without unnecessarily modifying the surrounding content, whereas the latter requires the edit to propagate coherently across the scene while preserving its underlying geometry and layout.

In these examples, IN2N distorts the source content, GaussianEditor modifies editing-irrelevant regions, and GaussCtrl only partially realizes the requested semantics. In contrast, our method handles both regimes by localizing the subjectlevel edit and coherently propagating the scene-level transformation, achieving a better balance between instruction fidelity and content preservation. Additional comparisons are provided in Appendix C.

![](images/d67b2c8f9c4e2999723aba7e53f8b555fc617136aba625c15832ea16dd3bb3b6.jpg)  
Figure 4: With the same candidate views and keyframe budget, random and importance-based selection yield different keyframes (top) and editing results (bottom). Our selection provides more reliable guidance and achieves a more faithful target transformation, highlighting the importance of view selection in multi-view editing.

![](images/04ef647495d66e4b6ada3ffd91113e6c53715dae018baf2abc1e20886b1570ca.jpg)  
Figure 5: Sensitivity of view-importance weights. The three cues are complementary, with Geo-dom. (0.6, 0.2, 0.2) adopted as the default for its best overall fidelity.

## 4.4 Analysis of View Importance

To directly test our central claim that views contribute unequally to multi-view editing, we use a controlled sparseview setting in which the effect of keyframe selection can be more clearly isolated. We compare importance-based selection with random selection while keeping the candidate views, keyframe budget, editing procedure, and 3DGS optimization identical. Thus, the only experimental variable is which views are selected as keyframes.

Figure 4 shows both the selected keyframes and their corresponding editing results. Random sampling may select views that provide less reliable geometric, semantic, or editability cues, thereby weakening the editing signal propagated to the remaining views. In contrast, our method identifies more informative keyframes and uses them to guide the other views, producing a more faithful target transformation while maintaining coherent appearance across viewpoints.

<table><tr><td>KIE E KGE IAO</td><td></td><td>T-I↑</td><td>Direction ↑</td></tr><tr><td rowspan="3">√</td><td>√</td><td>0.2702</td><td>0.2229</td></tr><tr><td></td><td>0.2754</td><td>0.2464</td></tr><tr><td>√</td><td>0.2761</td><td>0.2486</td></tr><tr><td>√</td><td>√ √</td><td>0.2822</td><td>0.2564</td></tr></table>

Table 2: Ablation study on View Matters. ✓ denotes the inclusion of each module. The full model consistently achieves the best results.

The results show that editing quality depends not only on how many views are used but also on which views provide the guidance. This result supports our central argument that views should not be treated equally in multi-view editing.

## 4.5 Ablation and Sensitivity Analysis

Following this controlled comparison, we progressively add KGE, KIE, and IAO to the per-view IP2P baseline. With KGE alone, the first view in each group serves as the keyframe. Table 2 shows that KGE improves performance through keyframe propagation, KIE provides more informative guidance, and IAO performs best by preserving keyframe priority during 3DGS optimization. These gains confirm their complementary roles. Qualitative ablations are provided in Appendix B.

Figure 5 analyzes the geometric visibility, semantic distinctiveness, and edit-relevance cues in KIE. Here, X-dom assigns 0.6 to cue X and 0.2 to the others; X-only retains only X; and No-X removes X while equally weighting the other two cues. Removing the geometric cue causes the largest degradation in editing fidelity. This cue is derived directly from the underlying 3DGS and captures the intrinsic viewdependent observability of the scene, providing a reliable basis for identifying informative views. Accordingly, Geo-dom, with weights (0.6, 0.2, 0.2), achieves the highest TI and Di rection while largely preserving adjacent-view consistency. This result supports our central design of explicitly selecting views according to their 3D editing evidence. Nevertheless, geometry alone underperforms Geo-dom: removing semantics weakens the intended edit direction, while removing edit relevance reduces semantic alignment and consistency. These results demonstrate that the three cues are complementary. We therefore adopt Geo-dom as the default setting, prioritizing 3D geometric evidence while retaining semantic and instruction-related cues for keyframe selection.

## 4.6 Cross-View Consistency Analysis

Although View Matters introduces no separate consistencyspecific module or loss, its reliability-guided design does not sacrifice cross-view coherence. Instead, shared keyframe guidance in KGE establishes a common editing context, while local propagation maintains continuity between adjacent views. Figure 6 presents two representative orderedview case studies rather than an aggregate consistency benchmark. Quantitatively, our method achieves higher CLIP temporal scores in both examples, reflecting stronger consistency across rendered views. Qualitatively, this improvement is evident in the highlighted regions, where our method produces more coherent local details and more stable visual patterns across viewpoints. In contrast, GaussCtrl suffers from greater inter-view variation, leading to inconsistent local appearance and reduced structural coherence.

![](images/d72b1474244ad38678e31d0f4d498616b8b58b4d4f2dcbc3b67720e05d0fa06f.jpg)  
Figure 6: Representative adjacent-view consistency analysis on two ordered rendering sequences.

## 5 Conclusion

View Matters organizes text-driven 3D Gaussian editing around unequal view reliability. KIE identifies reliable editing anchors, KGE propagates their information asymmetrically without noisy reverse influence, and IAO preserves the same priority during 3D optimization. The resulting framework improves instruction alignment and editing efficiency among the evaluated methods, while representative adjacentview analyses indicate that the fidelity-oriented design maintains cross-view coherence.

## References

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022.

Josef Bengtson, David Nilsson, Dong In Lee, and Fredrik Kahl. 3D-Consistent Multi-View Editing by Diffusion Guidance. arXiv preprint arXiv:2511.22228, 2025.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on com-

puter vision and pattern recognition, pages 18392–18402, 2023.

Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF international conference on computer vision, pages 23206–23217, 2023.

Minghao Chen, Iro Laina, and Andrea Vedaldi. Dge: Direct gaussian 3d editing by consistent multi-view editing. In European conference on computer vision, pages 74–92. Springer, 2024.

Xinhua Cheng, Tianyu Yang, Jianan Wang, Yu Li, Lei Zhang, Jian Zhang, and Li Yuan. Progressive3d: Progressively local editing for text-to-3d content creation with complex semantic prompts. arXiv preprint arXiv:2310.11784, 2023.

Jiahua Dong and Yu-Xiong Wang. Vica-nerf: Viewconsistency-aware 3d editing of neural radiance fields. Advances in Neural Information Processing Systems, 36: 61466–61477, 2023.

Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. arXiv preprint arXiv:2309.17102, 2023.

Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clipguided domain adaptation of image generators. ACM Transactions on Graphics (TOG), 41(4):1–13, 2022.

Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF international conference on computer vision, pages 19740–19750, 2023.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Alexander Hornung, Boyi Zeng, and Leif Kobbelt. Image selection for improved multi-view stereo. In 2008 IEEE Conference on Computer Vision and Pattern Recognition, pages 1–8. IEEE, 2008.

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125– 1134, 2017.

Wen Jiang, Boshu Lei, and Kostas Daniilidis. Fisherrf: Active view selection and mapping with radiance fields using fisher information. In European Conference on Computer Vision, pages 422–440. Springer, 2024.

Hiromichi Kamata, Yuiko Sakuma, Akio Hayakawa, Masato Ishii, and Takuya Narihira. Instruct 3d-to-3d: Text instruction guided 3d-to-3d conversion. arXiv preprint arXiv:2303.15780, 2023.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Dong In Lee, Hyeongcheol Park, Jiyoung Seo, Eunbyung Park, Hyunje Park, Ha Dam Baek, Sangheon Shin, Sangmin Kim, and Sangpil Kim. Editsplat: Multi-view fusion and attention-guided optimization for view-consistent 3d scene editing with 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11135–11145, 2025.

Xiangyue Liu, Han Xue, Kunming Luo, Ping Tan, and Li Yi. Genn2n: Generative nerf2nerf translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5105–5114, 2024.

Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang. Activenerf: Learning where to see with uncertainty estimation. In European Conference on Computer Vision, pages 230–246. Springer, 2022.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Onat ¸Sahin, Mohammad Altillawi, George Eskandar, Carlos Carbone, and Ziyuan Liu. Consistentdreamer: Viewconsistent meshes through balanced multi-view gaussian optimization. Pattern Recognition Letters, 190:118–125, 2025.

Yefei Sheng, Jie Wang, Ming Tao, and Bing-Kun Bao. D2gaussian: Dynamic control with discretized 3d view modeling for text-driven 3d gaussian splatting editing. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 9296–9305, 2025.

Zeng Tao, Zheng Ding, Zeyuan Chen, Xiang Zhang, Leizhi Li, and Zhuowen Tu. C3editor: Achieving controllable consistency in 2d model for 3d editing. arXiv preprint arXiv:2510.04539, 2025.

Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Clip-nerf: Text-and-image driven manipulation of neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3835–3844, 2022.

Can Wang, Ruixiang Jiang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Nerf-art: Text-driven neural radiance fields stylization. IEEE Transactions on Visualization and Computer Graphics, 30(8):4983–4996, 2023.

Junjie Wang, Jiemin Fang, Xiaopeng Zhang, Lingxi Xie, and Qi Tian. Gaussianeditor: Editing 3d gaussians delicately with text instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20902–20911, 2024.

Jing Wu, Jia-Wang Bian, Xinghui Li, Guangrun Wang, Ian Reid, Philip Torr, and Victor Adrian Prisacariu. Gaussctrl: Multi-view consistent text-driven 3d gaussian splatting editing. In European conference on computer vision, pages 55–71. Springer, 2024.

Wenhui Xiao, Rodrigo Santa Cruz, David Ahmedt-Aristizabal, Olivier Salvado, Clinton Fookes, and Leo Lebrat. Nerf director: Revisiting view selection in neural volume rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20742–20751, 2024.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

Jingyu Zhuang, Chen Wang, Liang Lin, Lingjie Liu, and Guanbin Li. Dreameditor: Text-driven 3d scene editing with neural fields. In SIGGRAPH Asia 2023 conference papers, pages 1–10, 2023.

## Appendix Overview

The appendices provide additional implementation details, ablation experiments, qualitative comparisons, and discussions that complement the main paper. They are organized as follows:

• Appendix A provides additional implementation details and hyperparameter settings.

• Appendix B presents additional ablation studies on the number of keyframes and the keyframe-loss weight in IAO, fol lowed by qualitative ablation results.

• Appendix C provides comparisons with recent methods without public implementations, evaluates generalization across human subjects, non-human objects, and complete scenes, and presents additional qualitative results.

• Appendix D discusses the limitations of the current framework and possible directions for future work.

## A Additional Implementation Details

We follow the parameter settings used in previous methods Haque et al. (2023); Wang et al. (2024); Chen et al. (2024). Unless otherwise specified, for most scenes, the guidance scales are set to 7.5 for the textual condition and 1.5 for the image condition. For each scene, we use 20 edited views, including 4 keyframes and 16 non-keyframes, and adopt a batch size of 5 during optimization. The selected views are updated every 500 iterations. The weight of the keyframe loss is set to 1.6. For evaluation, as shown in Table 4 we select 8 scenes and construct 23 scene-prompt pairs. For each prompt, we report the average score across all rendered views.

## B Additional Ablation Experiments

In this section, we conduct additional ablation studies to justify the chosen hyperparameter settings and to further verify the effectiveness of the three components in the keyframe selection module.

## B.1 Impact of the Number of Keyframes

We set the total number of edited views to 20 Wang et al. (2024); Chen et al. (2024), following previous work. With the total number of views fixed, varying the interval between selected keyframes is equivalent to varying the total number of keyframes. We therefore evaluate four settings, corresponding to 2, 4, 5, and 10 keyframes.

<table><tr><td>Number of keyframes</td><td>T-I↑</td><td>Dir ↑</td><td>Training Time (s)</td><td>Max Alloc. (GB)</td><td>Max Res. (GB)</td></tr><tr><td>2</td><td>0.2763</td><td>0.2781</td><td>263.46</td><td>17.78</td><td>25.38</td></tr><tr><td>4</td><td>0.2793</td><td>0.2840</td><td>266.04</td><td>17.78</td><td>25.37</td></tr><tr><td>5</td><td>0.2786</td><td>0.2821</td><td>268.34</td><td>17.78</td><td>25.39</td></tr><tr><td>10</td><td>0.2780</td><td>0.2819</td><td>324.58</td><td>24.54</td><td>37.25</td></tr></table>

Table 3: Effectiveness and resource consumption under different keyframe settings.

As shown in Tab. 3, increasing the number of keyframes from 2 to 4 improves both T-I and Dir, while the training time and memory consumption remain almost unchanged. This indicates that introducing a moderate number of keyframes can provide stronger supervision without noticeable additional computational overhead. When the number of keyframes is further increased to 5, the resource cost is still similar, but the performance shows a slight drop compared with the 4-view setting. Increasing the number of keyframes to 10 leads to a clear increase in both training time and GPU memory usage, while failing to bring further quantitative gains. These results suggest that too few keyframes provide insufficient supervision, whereas too many views introduce redundant information and unnecessary computational burden. Therefore, we adopt 4 keyframes as the default setting, since it achieves the best trade-off between editing performance and efficiency.

![](images/85770328b72c4d8c1f8901f2e74fdbfe0f2a159f4754690eeba06ea65d20d894.jpg)  
Baseline + KGE + KIE

## B.2 Effect of the Keyframe-Loss Weight in IAO

Figure 7 analyzes the sensitivity to the keyframe-loss weight λ<sub>k</sub>. The coarse sweep reveals a clear trade-off: increasing λ<sub>k</sub> generally improves Text–Image Similarity but gradually reduces Directional Similarity, suggesting that excessively strong keyframe supervision may favor semantic alignment at the expense of the intended edit direction. The fine-grained sweep further shows that $\lambda _ { k } = 1 . 6$ provides the best balance. Both metrics improve over $\lambda _ { k } = 1$ , whereas larger weights yield only marginal gains in Text–Image Similarity while decreasing Directional Similarity. We therefore set $\lambda _ { k } = 1 . 6$ as the default.

![](images/eeb39734fceee106e77ecc89e713e5d1215478ba3fcec0db52850cd71d709b59.jpg)  
Figure 7: Sensitivity to the keyframe-loss weight λ<sub>k</sub>. Relative changes in CLIP T–I and CLIP Directional Similarity are reported against $\lambda _ { k } = 1$ for the coarse and fine-grained sweeps.

## B.3 Qualitative Ablation Results

As shown in Fig. 8, Keyframe-Guided Editing (KGE) introduces richer target textures, suppresses undesired texture changes on the clothes, and mitigates the inconsistency around the neck region. Further incorporating Keyframe Importance Estimation enhances the details of the eye-mask region and removes part of the residual original eye contour. After additionally introducing Importance-Aware Optimization (IAO), the method fully suppresses the interference of the original eye contours and better preserves the appearance of non-edited regions, yielding the most reliable and consistent editing results. This improvement indicates that only when the 3D optimization process explicitly accounts for the different importance of views can the mode achieve the best editing quality, which also validates our core insight that view importance matters in 3D Gaussian editing.

![](images/a2084b3b3b95363e8055408d7687235871cdd2771257867c406b289e64a7ec0d.jpg)  
Figure 8: Qualitative ablation results. The progressive visual improvements illustrate the complementary effects of KGE, KIE, and IAO. Red and blue boxes highlight different local regions for comparison.

![](images/71b2dc396a237876c6480c355072b81d2e4ba95d167b1ec88aa216a85ffa5e0b.jpg)  
Figure 9: Qualitative comparison with C3Editor and 3D-consistent

## C Additional Experiment Results

## C.1 Comparison with Recent Methods without Public Implementations

Several recent text-driven 3D editing methods Tao et al. (2025); Bengtson et al. (2025) have not released their official implementations, preventing a controlled quantitative comparison under the same experimental setting. To provide a qualitative reference, we apply our method using the edit prompts reported in their papers and compare the resulting visual effects in Fig. 9. Results of the compared methods are taken from their original papers. Although this comparison does not constitute a strictly controlled evaluation, it illustrates how our method handles similar editing instructions while preserving the source content.

![](images/87ae8d87b0afb69849dbd9e22214cce10e6e61eb3d5bdedfc88da414b5bf226b.jpg)  
Add a few golden flowers

![](images/01d22200c8598e397b223fd56e33bfcd65660b9474b9dc88863e06cf872def9e.jpg)  
Customize the bench with a galaxy theme

![](images/91f47e8808491eebc94936adc72675f44a07c8bbe914db4d5e0f9a6e336a3a56.jpg)  
Turn the horse statue into wooden carving

## C.2 Generalization across Subjects and Edit Types

Figure 10 presents results covering human subjects, non-human objects, and complete scenes. These examples include both localized subject edits and global scene transformations, requiring different levels of semantic and structural modification. Our method produces faithful target changes while retaining editing-irrelevant content across these categories, supporting its generalization to diverse subjects and editing scopes.

![](images/1df8ca6ea9a5beb58f372c45fb387ca8005dfb4184c085bd44ebeb13ada435c4.jpg)  
Source Views “face”

![](images/02442d6abcbb91bfbcc162323d561dcfd74145282cd0c4cdf91f43c55be6266a.jpg)

![](images/2dd04f2f5b01aa555c21efc91371a56b1d84505c8e4618e4c8038be18ae28dda.jpg)  
Turn him into the Tolkien Elf  
Make his face resemble that of a marble sculpture

![](images/6f3bbb0b6c5711396c4010a813f07fa49ba5592d2a39e0a48d5f691242744142.jpg)  
Source Views “garden”

![](images/655da143a666ea1d3debe72fc465a60c6fdc54f62642db3a84f69ed7cef76d36.jpg)  
Make it van Gogh style

![](images/0e38e305e20550b7b6da2d604bcdf8cebee17de5483175263e2f5faabafe883b.jpg)

![](images/1b508da65e110f1264b3ccb1e7d8156aaa92746f19da4873035028a49e40b99f.jpg)  
Source Views “bear”

![](images/7ec5ed42f1d3e1edaa02463946a93a0297b1158c63e1a71c26e62c337c490bc9.jpg)  
Turn the vase into red  
Turn the bear into bronze statue

![](images/304f7fc01f46ef13433aabb4b4b83d47e9b41af35b5a74f6c35cbf375369d5e8.jpg)  
Turn the bear statue into a grizzly bear

Figure 10: Qualitative results across subjects and edit scopes. The examples cover human subjects, complete scenes, and nonhuman objects, including both localized edits and global transformations. For each instruction, two rendered viewpoints show that our method follows the target semantics while preserving editing-irrelevant content and the underlying scene structure.

## C.3 Additional Qualitative Results

Figure 11 provides additional results across different scenes and editing instructions. The results further show that our method can express the requested semantic changes while preserving the original scene structure and appearance. Renderings from multiple viewpoints also exhibit coherent edited content, indicating that reliability-guided propagation maintains stable editing effects across views.

![](images/a6e1f5e453301dbc39597b19c92722ed1c1a27b2b8696c5e58af3968e7224b2f.jpg)  
Bear

![](images/191c7220ab7f52038e1653096c4c89d5aa081a37d18343d3b609ab1bb83366c7.jpg)  
Turn the bear statue into an asiatic black bear

![](images/4c2877af13c95b215f6e907b3309f1105d6d66eec971f0eafe5191dc1b189add.jpg)  
Turn the bear statue into a polar bear

![](images/d7e8cbeebe326aa283ee0c3c8b348ce7e4fbc6c4afc155170002a559b0070302.jpg)

![](images/275421c2bed9df1897c67a7e7e7c55045827d68c374596ab63a7c70afdf8af96.jpg)

![](images/6b4b6e73c682889a8d9cac00907e42171b89cebd2c7a3f5cffd2839c697c5e88.jpg)  
Give him a mustache  
Face

![](images/fff7cb2fdf243a645ba0444590bfff41c75c5b44d09c46791eb14ff539293276.jpg)  
Make him look like Vincent Van Gogh  
Make him appear like he is made of paper with folded

![](images/93930649cb0e3759532cb05a54cb79243b623440a09a3e94c83c820645bb4250.jpg)

![](images/474ad441fb2a9baf4b46a10e307fe4945f784df3389430ad30d95ad054357cfd.jpg)  
Turn his face into a skull

![](images/55908dd5c084ce8f4cd438843e1b401451c13590894cc110547e8125e8309acd.jpg)  
Garden

![](images/59d06fb563a9ec600318c6f4b3d461626b36f05cd298354929e9ff27fd89ceb9.jpg)  
Make it van Gogh style

![](images/2907c07a14d8e4e3aba0884d82fc74120abe6ec7086a59f8cc3558c1b0727260.jpg)

![](images/abc6d610455399c288286f3dd30d733517e43adebefc437ec85924cfba265f61.jpg)  
Make it autumn

Person  
![](images/33323c7a9df3e59464a8899d713f8f78d06a07d7eb4fb575b01f4a1d466f127f.jpg)  
Make him wear a suit

![](images/af42d73230a0482098700b2cedbd99d2d7624078b6cc1e69176be9b11a91d8ef.jpg)

![](images/cdaccd8589589e088a2c707c12069dd80819217d0767fb8c5b3c6b68d4ee99bc.jpg)  
Make the man look like a mosaic sculpture  
Kitchen

![](images/76de919d16cda382fab6ccf16176857a3640b42524a70e1e5461dfca5326d0e8.jpg)

![](images/e60abd2fb7b1f5b47eadd5ed3955a91fd57bdea83df50d6930a8ad800b9c7d95.jpg)  
Turn the dozer into red

![](images/0044c56cc4237c4da75ab85193cb9f888899586431caca2b2ce0e7f05dea5188.jpg)

![](images/3af900aa7639907732e7fbace97c5db8bb174651d79242ee77b74a44cc556b90.jpg)  
Figure 11: Additional qualitative results across scenes and editing instructions. Each group presents the source scene and edited renderings from multiple viewpoints. Our method handles diverse identity, material, appearance, style, and color transforma tions while maintaining stable edited content across views.

## C.4 Evaluation Scenes and Prompts

Table 4: Detailed evaluation dataset: 23 scene–prompt pairs. The source and target prompts are used for computing CLIP and CLIP directional scores.
<table><tr><td>Scene</td><td>Edit Instruction</td><td>Target Prompt</td><td>Source Prompt</td></tr><tr><td>Face</td><td>“Turn him into the Tolkien Elf.&quot;</td><td>“A Tolkien Elf man with curly hair.&quot;</td><td>&quot;A man with curly hair in a grey jacket.&quot;</td></tr><tr><td>Face</td><td>“Turn him into an Einstein.&quot;</td><td>“Einstein with curly hair in a grey jacket.&quot;</td><td>“A man with curly hair in a grey jacket.&quot;</td></tr><tr><td>Face</td><td>“Turn his face into a skull.&quot;</td><td>“A skull with curly hair in a grey jacket.&quot;</td><td>“A man with curly hair in a grey jacket.&quot;</td></tr><tr><td>Face</td><td>“Turn him into spiderman with a mask.&quot;</td><td>“A spider man with a mask and curly hair.&quot;</td><td>“A man with curly hair in a grey jacket.&quot;</td></tr><tr><td>Person</td><td>“Turn the man into a clown.&quot;</td><td>“A clown standing next to a wall wearing a blue T-shirt.&quot;</td><td>“A man standing next to a wall wearing a blue T-shirt.&quot;</td></tr><tr><td>Person</td><td>“Turn him into a Super Mario.&quot;</td><td>“A photo of a Super Mario.&quot;</td><td>&quot;A photo of a person.&quot;</td></tr><tr><td>Person</td><td>“Turn him into a Minecraft character.&quot;</td><td>&quot;A photo of a person in Minecraft.&quot;</td><td>&quot;A photo of a person.&quot;</td></tr><tr><td>fangzhou</td><td>“Give him Chinese opera face paint.&quot;</td><td>“A photo of a face of a man with black hair and Chinese opera face paint.&quot;</td><td>&quot;A photo of a face of a man with black hair.&quot;</td></tr><tr><td>fangzhou</td><td>“Give him a pair of glasses.&quot;</td><td>&quot;A photo of a face of a man with black hair wearing a pair of glasses.&quot;</td><td>&quot;A photo of a face of a man with black hair.&quot;</td></tr><tr><td>Bear</td><td>&quot;Make the color of the bear look like rainbow color.&quot;</td><td>“A rainbow-colored bear in the forest.&quot;</td><td>“A stone bear in the forest.&quot;</td></tr><tr><td>Bear</td><td>“Turn the bear into a panda.&quot;</td><td>“A photo of a panda.&quot;</td><td>“A photo of a stone bear.&quot;</td></tr><tr><td>garden</td><td>“Make the scene appear as though it&#x27;s underwater.&quot;</td><td>“A photo of an outdoor garden in underwater.&quot;</td><td>“A photo of an outdoor garden.&quot;</td></tr><tr><td>garden</td><td>“&quot;Make it Van Gogh style.&quot;</td><td>“A table in an outdoor garden with Van Gogh style.&quot;</td><td>“A table in an outdoor garden.&quot;</td></tr><tr><td>garden</td><td>“&quot;Make it autumn.&quot;</td><td>“A photo of an outdoor garden in autumn.&quot;</td><td>&quot;A photo of an outdoor garden.&quot;</td></tr><tr><td>horse statue</td><td>“Turn the horse statue into a wooden carving.&quot;</td><td>&quot;A photo of a horse made of wood.&quot;</td><td>“A photo of a horse statue.&quot;</td></tr><tr><td>horse</td><td>“Make the stone horse a zebra.&quot;</td><td>“A photo of a zebra.&quot;</td><td>“A photo of a horse statue.&quot;</td></tr><tr><td>statue horse</td><td>“Turn the stone horse into a jade</td><td>“A photo of a horse made of jade.&quot;</td><td>“A photo of a horse statue.&quot;</td></tr><tr><td>statue bonsai</td><td>carving.&quot; &quot;Make the bonsai snowy.&quot;</td><td>“A photo of a snowy bonsai.&quot;</td><td>&quot;A photo of a bonsai.&quot;</td></tr><tr><td>bonsai</td><td>“Change the bonsai to look like it&#x27;s</td><td>&quot;A photo of a bonsai made of paper.&quot;</td><td>&quot;A photo of a bonsai.&quot;</td></tr><tr><td>bonsai</td><td>made of paper.&quot; “Add a few golden flowers.&quot;</td><td>“A photo of a bonsai with golden</td><td>&quot;A photo of a bonsai.&quot;</td></tr><tr><td>bicycle</td><td>“Turn the ground into a Namibian</td><td>flowers.&quot; “A photo of a Namibian desert.&quot;</td><td>“A photo of a park.&quot;</td></tr><tr><td>bicycle</td><td>desert.&quot; “Make the entire scene look as if it&#x27;s</td><td>“A watercolor style painting of a park.&quot;</td><td>&quot;A photo of a park.&quot;</td></tr><tr><td></td><td>painted in a watercolor style.&quot; “Customize the bench with a galaxy</td><td></td><td></td></tr><tr><td>bicycle</td><td>theme.&quot;</td><td>&quot;A photo of a bench with a galaxy theme.&quot; &quot;A photo of a bench.&quot;</td><td></td></tr></table>

## D Limitations and Discussion

Our framework has two main limitations. First, it employs a frozen 2D image editor and therefore inherits its failure modes. Edits involving substantial geometric changes or fine-grained local structures may be limited by the underlying editor. Since KGE, KIE, and IAO organize cross-view guidance and 3D supervision without retraining the editing backbone, they could potentially be adapted to stronger attention-based editors in future work.

Second, the benefit of view-importance modeling depends on the variation in editing reliability across candidate views. When densely sampled views provide similarly clear observations, the distinction between keyframes and other views becomes smaller, and the improvement over uniform treatment may be modest. Conversely, if none of the candidate views clearly observes the target region, view selection alone cannot provide a reliable editing anchor. Our controlled sparse-view experiment examine the former reliability differences, while more extreme visibility conditions remain an important direction for future evaluation.