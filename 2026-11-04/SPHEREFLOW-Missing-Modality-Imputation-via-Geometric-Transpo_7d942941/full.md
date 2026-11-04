# SPHEREFLOW: Missing Modality Imputation via Geometric Transport on Hypersphere

Huihan Wang<sup>1∗</sup> Juncheng Wang<sup>1∗</sup> Yuxiang Feng<sup>2,3</sup> Wenlong Hou<sup>1</sup> Guoqi Yu<sup>1</sup> Chao Xu<sup>3,4</sup> Yang Liu<sup>3,4</sup> Baigui Sun<sup>3,4</sup> Shujun Wang<sup>1,5†</sup>

<sup>1</sup>Department of Biomedical Engineering and Sports Technology (BEST), The Hong Kong Polytechnic University, Hong Kong, China <sup>2</sup>Zhejiang University, Hangzhou, China <sup>3</sup>IROOTECH TECHNOLOGY, China <sup>4</sup>Wolf 1069 b Lab, Sany Group, China <sup>5</sup>Department of Data Science and Artificial Intelligence, The Hong Kong Polytechnic University, Hong Kong, China <sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author. {wang2003huihan@gmail.com, wjc2830@gmail.com}<sup>∗</sup>, shu-jun.wang@polyu.edu.hk<sup>†</sup>

## Abstract

Missing modality imputation aims to synthesize unobserved modalities from observed ones. Existing methods treat each modality as living on its own latent manifold, forcing the generator to learn a long, unstructured jump between them. We present SPHEREFLOW, which takes a fundamentally different view: by projecting all modalities into a shared hyperspherical latent space via a frozen self-supervised encoder, cross-modal imputation reduces to short-range geometric transport between nearby points on the sphere. Realizing this idea with flow matching, however, exposes two geometric conflicts: the standard Gaussian noise source is catastrophically mismatched with the unit-norm data manifold, and the linear interpolation path departs the sphere surface into a region where the decoder has no training signal. We resolve both with simple, geometry-aware modifications: replacing the noise source with the observed-modality latent so that the velocity field learns only the cross-modal displacement, and training the decoder with a time-weighted loss that extends its domain to the near-surface region traversed by the flow. We prove that this data-as-source formulation reduces the worst-case off-manifold deviation from a quantity that grows with the latent dimension to a small, dimensionindependent bound governed by inter-modality similarity. Experiments on BraTS multi-modal MRI imputation and CT–MRI translation demonstrate state-of-the-art generation fidelity across all settings, while downstream brain tumor segmentation with our imputed data closes roughly eighty percent of the gap to the full-modality oracle. Code is available at https://github.com/WTHH031230/SphereFlow.

## 1 Introduction

Missing modality imputation—synthesizing an unobserved modality from observed ones—is critical in medical imaging [Wang et al., 2021], remote sensing, and autonomous driving, where acquiring all modalities is often infeasible [Chartsias et al., 2017, Sharma and Hamarneh, 2019]. Existing approaches treat this as a conditional generation problem, leveraging generative adversarial networks [Goodfellow et al., 2014, Isola et al., 2017, Zhu et al., 2017] or diffusion-based generative models [Ho et al., 2020, Song et al., 2020], with designs ranging from modality-specific architectures [Chartsias et al., 2017] and pairwise alignment losses [Zhang et al., 2025] to per-pair noise-to-data pipelines [Dar et al., 2019]. As illustrated in Fig. 1(b), all of these methods implicitly assume that each modality lives on its own latent manifold, so that imputation is a long, unstructured jump between separated manifolds.

![](images/f67a85a47742d25a6eb8bd51952ff8a24211451df95815e30523f0b9ae6ed2c0.jpg)  
(a) Missing Modality Imputation

![](images/bf5f0866c0098c0eec79107edc9de1349966f61992af8de92a21506be5dea553.jpg)  
(b) Modal-Specific Manifolds

![](images/c3e9d5341ebba7341d32a9faa3dc58665c011474b041e585a0c91ff0cccd6222.jpg)  
(c) Modal-Shared Hypersphere  
Figure 1: Two views of cross-modal imputation. (a) Task: given a subset of observed modalities, synthesize the missing one. (b) Conventional view: modalities are encoded into disjoint latent manifolds (pixel space or per-modality VAE), so cross-modal generation is a long jump between them. (c) Our view: a frozen DINOv2 encoder maps every modality to a single shared hypersphere $S ^ { d - 1 }$ , on which paired latents $\left( z _ { s } , z _ { t } \right)$ sit close together; imputation becomes short-range transport along the sphere.

Nevertheless, these methods scale poorly with the number of modalities and, more importantly, neglect a fundamental principle: different modalities of the same subject share the same semantic content and differ only in appearance. We argue that the bottleneck is representation, not generation. Current methods operate in pixel space or VAE latent spaces [Kingma and Welling, 2013, Rombach et al., 2022], where an MRI and a CT scan of the same brain land in unrelated regions because their pixel statistics differ entirely—the model must learn a separate appearance-level mapping for every source-target pair.

On the stark contrast, if all modalities were instead projected into a shared space organized by semantic content (Fig. 1(c)), cross-modal imputation would reduce to transport: a short-range displacement between nearby points. This raises a concrete question: what properties must such a latent space satisfy so that missing modality imputation can be formulated as geometric transport?

To answer this, we conduct pilot studies comparing three tokenizers—VAE [Kingma and Welling, 2013], MAE [He et al., 2022], and DINOv2 [Caron et al., 2021, Oquab et al., 2023]—as cross-modal latent spaces. Two properties emerge as essential: (i) cross-modal compactness relative to subject variability—different modalities of the same subject should be encoded much closer than different subjects of the same modality—and (ii) geometric regularity—latents should concentrate on a manifold with known geometry. VAE latents fail (i): different modalities land far apart, as the encoder captures pixel statistics rather than semantics. MAE satisfies (i) only by collapsing all subjects (regardless of modality) into a small region of latent space, sacrificing subject discriminability. DINOv2 uniquely satisfies both: its self-supervised contrastive training [Chen et al., 2020, He et al., 2020] yields features dominated by subject identity with small cross-modal displacement (i), and its LayerNorm constrains representations to a hypersphere $S ^ { d - 1 }$ (ii) [Chang et al., 2026, Kumar and Patel, 2026].

Guided by these findings, we propose SPHEREFLOW, which formulates cross-modal imputation as geometric transport on $S ^ { d - \mathrm { \hat { 1 } } }$ . We adopt DINOv2 as a frozen encoder within a Representation Autoencoder (RAE) [Zheng et al., 2025], mapping all modalities into the same $S ^ { d - 1 }$ where crossmodal pairs of the same subject land nearby. Imputation becomes short-range transport on $S ^ { d - 1 } \colon$ given a source-modality point $z _ { s }$ , find the target-modality point z<sub>t</sub>.

However, realizing this with flow matching [Lipman et al., 2022, Liu et al., 2022, Esser et al., 2024] exposes two geometric conflicts. First, standard flow matching generates a target latent z from Gaussian noise $\varepsilon \sim \mathcal { N } ( 0 , I ) { \ - } \mathrm { - a }$ paradigm designed for unconditional generation from scratch, not for cross-modal translation where a semantically meaningful source is already available. On a hypersphere this mismatch is severe: $\| \varepsilon \| \approx { \sqrt { d } }$ while $\| z \| { = } 1$ , so the interpolant must traverse a vast off-manifold void and the model wastes capacity on a radial collapse irrelevant to cross-modal transformation. Second, even with both endpoints on the sphere, linear interpolation produces intermediates with ∥z ∥ < 1—off-manifold points the decoder has never seen and cannot reconstruct. Riemannian flow matching [Chen and Lipman, 2023, Kumar and Patel, 2026] fixes the path via SLERP but still departs from noise; LV-RAE [Liu et al., 2026] augments the decoder but without geometric structure. No existing method addresses both.

![](images/7d0b54ba2a774988fe52112eb6dc687bca49b6b878d3b0c3c1c2ae0458b374cc.jpg)  
(a) Cross-Modal Compactness

![](images/9b4601359ef92f173ef85e7a243aabd7ef6409adf89070a31fadc06286c73ad9.jpg)  
(b) Geometric Regularity  
Figure 2: Pilot study: tokenizer comparison on BraTS. (a) Cross-modal compactness: 2-D projection of paired $\dot { T _ { 1 } }$ (blue) and $T _ { 1 c }$ (red) latents under three tokenizers, with thin lines connecting same-subject pairs; the title reports the ratio $\bar { d } _ { \mathrm { p a i r } } / \bar { d } _ { \mathrm { c r o s s } }$ (lower is better, see §2.1). (b) Geometric regularity: empirical distribution of per-token latent norms $\| z \|$

We present SPHEREFLOW to address both conflicts. For the first, we replace the Gaussian source with the paired source-modality feature—already on $S ^ { d - 1 } { \mathrm { - } } { \mathrm { s o } }$ the velocity field learns only the modalityspecific displacement. For the second, we train the decoder to reconstruct from any interpolant $z _ { \tau }$ with time-dependent weights $\left( 1 - \tau \right)$ and τ, extending its domain to the near-surface region the flow traverses. We validate on BraTS [Baid et al., 2021] (T1, T1ce, T2, FLAIR) and a paired CT–MRI benchmark, achieving state-of-the-art generation fidelity across all settings. The imputed modalities also improve downstream performance—brain tumor segmentation with our synthesized data approaches the accuracy of training with real complete data.

Our contributions:

• We formulate missing modality imputation as geometric transport on a hypersphere, identifying two essential latent space properties through pilot studies and showing that DINOv2- based RAE uniquely satisfies both.

• We identify two geometric conflicts when applying flow matching on $S ^ { d - 1 }$ and address both: replacing the noise source with the observed modality, and expanding the decoder’s domain with time-weighted reconstruction.

• We show that data-as-source reduces off-manifold deviation from $\Theta ( { \sqrt { d } } )$ to a small bounded quantity controlled by inter-modality cosine similarity.

• We achieve state-of-the-art results on BraTS and CT–MRI translation in both generation fidelity and downstream task performance.

## 2 Pilot Study: What Makes a Good Shared Latent Space?

In Section 1, we argued that transport-based imputation requires two latent space properties: crossmodal compactness (paired modalities closer than different subjects) and geometric regularity (concentration on a known manifold). We now empirically test these on three representative tokenizers— VAE [Kingma and Welling, 2013], MAE [He et al., 2022], and DINOv2 [Oquab et al., 2023]—by encoding paired $T _ { 1 } / T _ { 1 c }$ images from BraTS [Baid et al., 2021] (Fig. 2).

## 2.1 Cross-Modal Compactness

Setup. We extract paired $T _ { 1 } / T _ { 1 c }$ latents from 500 BraTS test slices (L2-normalized) and compute two cosine-distance statistics: $d _ { \mathrm { c r o s s } } ,$ the mean distance between modalities of the same slice, and $d _ { \mathrm { i n t r a } }$ , the mean distance between $d i f f e r e n t$ slices within the same modality. A good transport space needs a small ratio $d _ { \mathrm { c r o s s } } / d _ { \mathrm { i n t r a } }$ without $d _ { \mathrm { i n t r a } }$ itself collapsing (which would indicate loss of anatomical discriminability).

Findings. As shown in Fig. 2(a), VAE gives $d _ { \mathrm { c r o s s } } / d _ { \mathrm { i n t r a } } \approx 7 . 9 6 \colon$ : modalities land in unrelated regions, confirming that pixel-level encoders force long, unstructured jumps. MAE reduces the ratio to 0.47 but via representational collapse—all latents cluster into a tiny region, erasing anatomical identity. DINOv2 achieves the lowest ratio (0.41) while preserving well-separated slices: crossmodal displacement is small relative to anatomical variability. Takeaway: only DINOv2 provides cross-modal compactness without sacrificing discriminability, making short-range transport viable.

## 2.2 Geometric Regularity

Setup. We examine the distribution of latent norms $\| z \|$ for each tokenizer and compute the coefficient of variation $\mathbf { C V } _ { r } = \operatorname { s t d } ( \| z \| ) / \mathbb { E } [ \| z \| ]$ . A small $\mathrm { \dot { C } V _ { \it 1 } }$ indicates concentration on a hypersphere, enabling geometric analysis of interpolation paths and decoder domains.

Findings. Figure 2(b) shows the results. VAE latents spread broadly across $\mathbb { R } ^ { d }$ with no manifold structure. MAE latents show moderate radial concentration but lack a clean single-radius manifold. DINOv2 latents concentrate sharply at $\lVert z \rVert \approx 1$ with near-zero $\mathrm { C V } _ { r }$ , consistent with its output LayerNorm—confirming that they lie on a hypersphere $S ^ { d - 1 }$ . Takeaway: DINOv2’s hyperspherical geometry provides a well-characterized manifold for analyzing flow paths, decoder domains, and transport bounds.

Summary. Among the three tokenizers, only DINOv2 simultaneously satisfies both properties: cross-modal compactness without collapse and concentration on a known manifold $\dot { S } ^ { d - 1 }$ . We therefore adopt DINOv2 as a frozen encoder within a Representation Autoencoder (RAE), formulating imputation as short-range transport on the hypersphere. However, applying flow matching on $S ^ { d - \mathbf { \check { 1 } } }$ introduces geometric conflicts that we analyze and resolve in Section 3.

## 3 Method: SPHEREFLOW

The pilot study establishes DINOv2 as a shared hyperspherical latent space where cross-modal imputation reduces to short-range transport on $S ^ { d - 1 }$ . However, naïvely applying flow matching to this setting introduces two geometric conflicts that must be resolved. This section presents SPHEREFLOW. We begin with the baseline pipeline (§3.1), formalize these conflicts (§3.2), and then introduce our solutions: cross-modal flow matching (§3.3) and manifold-expanded decoder training (§3.4). Figure 3 provides a high-level overview.

## 3.1 Preliminaries

DINO Encoder. We adopt a frozen DINOv2 ViT-B/14 [Oquab et al., 2023] as our visual tokenizer. Given an input image $x \in \overset { \mathbf { 1 } } { \mathbb { R } } ^ { H \times W \times C }$ , we denote its output representation as $z = \mathcal { E } _ { \mathrm { D I N O } } ( x )$ . Due to the final LayerNorm, DINOv2 representations satisfy $\lVert \boldsymbol { z } \rVert \approx 1$ with near-zero radial variance [Chang et al., 2026, Kumar and Patel, 2026], so we treat $z \in S ^ { d - 1 }$ throughout. The encoder isfrozen and shared across all modalities, guaranteeing a common geometric space.

RAE Decoder. Following Zheng et al. [2025], we pair the frozen DINO encoder with modalityspecific decoders $\{ D _ { \phi } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ , one per modality, each reconstructing images of its respective modality from latent representations. Each decoder takes $z \in S ^ { d - 1 }$ as input and produces $\hat { x } = D _ { \phi } ^ { ( m ) } ( z ) \in$ $\mathbb { R } ^ { H \times W \times C }$ . They are trained with pixel-level and perceptual losses:

$$
\mathcal { L } _ { \mathrm { r e c } } = \Vert D _ { \phi } ^ { ( m ) } ( z ) - x \Vert _ { 1 } + \lambda _ { \mathrm { p e r c } } \cdot \mathcal { L } _ { \mathrm { L P I P S } } ( D _ { \phi } ^ { ( m ) } ( z ) , x ) + \lambda _ { \mathrm { a d v } } \cdot \mathcal { L } _ { \mathrm { a d v } } ( D _ { \phi } ^ { ( m ) } ( z ) )\tag{1}
$$

where ${ \mathcal { L } } _ { \mathrm { L P I P S } }$ denotes the perceptual loss and $\mathcal { L } _ { \mathrm { a d v } }$ a non-saturating GAN loss against a hinge-based discriminator. Using modality-specific decoders allows each to specialize in the pixel statistics of its modality (e.g., intensity ranges of T1 vs. T2), while sharing the same latent space. A critical observation for our later analysis: each decoder is trained exclusively on on-manifold points $z =$ $\mathcal { E } _ { \mathrm { D I N O } } ( x ) \in \mathcal { S } ^ { d - 1 }$ . Its behavior at off-manifold points $( \| z \| \neq 1 )$ ) is uncontrolled and, as we will show, problematic.

![](images/5f51391e75f01e8034556f2ce6cb490697e6799334e80c56d15e1caaa98b7ce8.jpg)  
Figure 3: Overview of SPHEREFLOW. (I) Tokenizer training (§3.4): a frozen DINOv2 encoder maps source and target images onto the hypersphere $S ^ { d - 1 }$ . Modality-specific decoders $D ^ { ( s ) } , D ^ { ( t ) }$ are trained to reconstruct both on-manifold latents $( z _ { s } , z _ { t } )$ and off-manifold interpolants $\left( z _ { \tau } \right)$ via the timeweighted expansion loss (Eq. 9). (II) Generator training (§3.3): a Diffusion Transformer learns the cross-modal velocity field $v _ { \theta } ( z _ { \tau } , \tau ) \mapsto u _ { \tau } = z _ { t } - z _ { s }$ with $z _ { \tau }$ as input, using the paired source latent as the flow origin instead of Gaussian noise. (III) Geometric conflict & SPHEREFLOW: standard flow matching (gray trajectory) departs from Gaussian noise at $\| \varepsilon \| \approx \sqrt { d } \gg 1$ 1 and must traverse a vast off-manifold void to reach the DINO hypersphere $( \lVert z \rVert = 1 )$ . SPHEREFLOW (orange trajectory) transports directly from $z _ { s } \mathrm { ~ t o ~ } z _ { t }$ along a short near-surface chord, with bounded off-manifold deviation (Proposition 1).

Diffusion Transformer (DiT). We employ a Diffusion Transformer (DiT) [Peebles and Xie, 2023], which adapts the Vision Transformer [Vaswani et al., 2017, Dosovitskiy et al., 2020] for diffusion modeling, as the velocity field backbone. The DiT takes a noisy latent $z _ { \tau }$ at timestep $\tau \in [ 0 , 1 ]$ and predicts the velocity $v _ { \theta } ( z _ { \tau } , \tau , c )$ , where c is a modality indicator specifying the target modality. This enables a single model to handle all imputation directions.

Standard Flow Matching. Flow matching [Lipman et al., 2022] learns a velocity field that defines a continuous-time ordinary differential equation (ODE) transporting a source distribution $p _ { 0 }$ to a target distribution $p _ { 1 }$ . In the standard formulation, the source is Gaussian noise $\varepsilon \sim \mathcal { N } ( 0 , I )$ and the target is data $z = \mathcal { E } _ { \mathrm { D I N O } } ( x )$ ). The conditional interpolant and velocity at timestep τ are:

$$
z _ { \tau } = \tau \cdot z + ( 1 - \tau ) \cdot \varepsilon , \quad u _ { \tau } = z - \varepsilon\tag{2}
$$

The DiT is trained to predict the velocity via:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { \tau \sim \mathcal { U } ( 0 , 1 ) , \varepsilon \sim \mathcal { N } ( 0 , I ) , z } \left\| v _ { \theta } ( z _ { \tau } , \tau ) - u _ { \tau } \right\| ^ { 2 }\tag{3}
$$

At inference, a sample ε is drawn and integrated forward via the learned ODE to obtain $\hat { z } \approx z ,$ which is then decoded as $\hat { x } = D _ { \phi } ( \hat { z } )$

## 3.2 Geometric Conflicts on $S ^ { d - 1 }$

With the shared hyperspherical latent space established, a natural next step is to apply flow matching to learn the cross-modal transport map on $S ^ { d - 1 }$ . However, directly applying the standard formulation (Eq. 2) to this geometry introduces two fundamental conflicts.

Conflict I: Source Distribution Mismatch. Data lies on $S ^ { d - 1 } \left( \left. z \right. = 1 \right)$ , while the Gaussian source concentrates at $\lVert \varepsilon \rVert \approx \sqrt { d } \ ( \mathrm { e . g . , \approx } 2 7 . 7$ for $d { = } 7 6 8 )$ . The interpolant $z _ { \tau } ~ ( \mathrm { E q . } ~ 2 )$ therefore traverses norms from $\sqrt { d } \tan 1$ , and the velocity field must simultaneously learn (a) a radial contraction by ${ \sqrt { d } } , ( \mathbf { b } )$ a directional rotation, and (c) the modality-specific displacement—three entangled tasks, only the last of which is relevant. In cross-modal imputation this is especially wasteful: the observed modality’s latent $z _ { s } \in \mathcal { S } ^ { d - 1 }$ is already a semantically meaningful source, yet the standard formulation discards it in favor of noise.

Conflict II: Off-Manifold Decoder Blindness. More broadly, flow matching with linear interpolation cannot guarantee that the entire trajectory stays on $\dot { S } ^ { d - 1 }$ . For any two distinct points $\bar { z } _ { 0 } , z _ { 1 } \in S ^ { d - 1 }$ , the linear interpolant $z _ { \tau } = \tau \cdot z _ { 1 } + ( 1 { - } \tau ) \cdot z _ { 0 }$ satisfies

$$
\| z _ { \tau } \| ^ { 2 } = \tau ^ { 2 } + ( 1 - \tau ) ^ { 2 } + 2 \tau ( 1 - \tau ) \langle z _ { 0 } , z _ { 1 } \rangle < 1 \quad \mathrm { w h e n e v e r } \ \langle z _ { 0 } , z _ { 1 } \rangle < 1\tag{4}
$$

That is, intermediate states inevitably dip inside the sphere. However, each RAE decoder ${ D } _ { \phi } ^ { ( m ) }$ is trained exclusively on on-manifold points $( \left. z \right. = 1 )$ and has never seen inputs with $\| z \| < 1$ During inference, the flow trajectory passes through this unseen region, and the decoders cannot provide meaningful reconstructions there—a mismatch between where the flow operates and where the decoders are valid.

## 3.3 Cross-Modal Flow Matching

To resolve Conflict I, we replace the Gaussian source with the paired source-modality latent. Let $z _ { s } = \mathcal { E } _ { \mathrm { D I N O } } ( x _ { s } ) \in \mathcal { S } ^ { d - 1 }$ and $z _ { t } = \mathcal { E } _ { \mathrm { D I N O } } ( x _ { t } ) \in S ^ { d - 1 }$ be a cross-modal pair. The interpolant and velocity are:

$$
z _ { \tau } = \tau \cdot z _ { t } + ( 1 { - } \tau ) \cdot z _ { s } , \quad u _ { \tau } = z _ { t } - z _ { s }\tag{5}
$$

The velocity $u _ { \tau }$ is constant along the path—it is simply the displacement from source to target latent. The DiT is trained to predict this displacement conditioned on the target modality indicator c:

$$
\mathcal { L } _ { \mathrm { C M F M } } = \mathbb { E } _ { \tau \sim \mathcal { U } ( 0 , 1 ) , ( z _ { s } , z _ { t } ) } \left\| v _ { \theta } ( z _ { \tau } , \tau , c ) - ( z _ { t } - z _ { s } ) \right\| ^ { 2 }\tag{6}
$$

Both endpoints now lie on $S ^ { d - 1 }$ , eliminating the norm mismatch, and the velocity field learns only the modality-specific displacement. Moreover, the interpolant remains close to the manifold, as quantified by the following result.

Proposition 1 (Off-manifold deviation bound). Let $z _ { s } , z _ { t } \in S ^ { d - 1 }$ with $\langle z _ { s } , z _ { t } \rangle =$ cos α for angular separation α. The maximum off-manifold deviation of the linear interpolant $z _ { \tau } \ ( E q . \ 5 )$ is:

$$
\Delta : = \operatorname* { m a x } _ { \tau \in [ 0 , 1 ] } ( 1 - \Vert z _ { \tau } \Vert ) = 1 - \sqrt { \frac { 1 + \cos \alpha } { 2 } } = 1 - \cos \frac \alpha 2\tag{7}
$$

attained at $\begin{array} { r } { \tau = \frac { 1 } { 2 } , B y } \end{array}$ contrast,for Gaussian-sourceflow matching, $\mathbb { E } \big [ \operatorname* { m a x } _ { \tau } \big | \big | z _ { \tau } \big | \big | - 1 \big | \big ] = \Theta ( \sqrt { d } )$

The proof is given in Appendix A.

Note that ∆ depends only on the angular separation α between the source and target latents, not on the ambient dimension d. In missing modality imputation, the two modalities depict the same underlying anatomy or scene, so their DINO representations naturally share substantial semantic overlap—α is small by the nature of the task. This contrasts sharply with the Gaussian-source deviation $\Theta ( { \sqrt { d } } )$ which grows with dimension. The bounded, dimension-independent deviation is what makes the decoder expansion in $\ S 3 . 4$ tractable.

At inference, given a source latent $z _ { s } = \mathcal { E } _ { \mathrm { D I N O } } ( x _ { s } )$ and a target modality indicator $c ,$ we integrate the learned velocity field starting from $z _ { s } \mathrm { : }$

$$
\hat { z } _ { t } = z _ { s } + \int _ { 0 } ^ { 1 } v _ { \theta } ( z _ { \tau } , \tau , c ) d \tau\tag{8}
$$

and decode with the target-modality decoder: $\hat { x } _ { t } = D _ { \phi } ^ { ( t ) } ( \hat { z } _ { t } )$

## 3.4 Manifold-Expanded Decoder Training

Cross-modal flow matching (§3.3) resolves the source distribution mismatch (Conflict I), but Conflict II remains: the linear interpolant $z _ { \tau }$ passes through the interior of the sphere, where the decoders have no training signal. We address this by expanding each decoder’s domain to cover the off-manifold region that the flow traverses.

Time-Weighted Reconstruction Loss. We train each modality-specific decoder to reconstruct its corresponding image from any intermediate point $z _ { \tau }$ along the interpolation path, with time-dependent weights:

$$
\mathcal { L } _ { \mathrm { e x p a n d } } = \mathbb { E } _ { \tau \sim \mathcal { U } ( 0 , 1 ) } \left[ \tau \cdot \ell \Big ( D _ { \phi } ^ { ( t ) } ( z _ { \tau } ) , x _ { t } \Big ) + ( 1 - \tau ) \cdot \ell \Big ( D _ { \phi } ^ { ( s ) } ( z _ { \tau } ) , x _ { s } \Big ) \right]\tag{9}
$$

where $\ell ( \cdot , \cdot )$ denotes the reconstruction loss (Eq. 1) and $z _ { \tau } = \tau \cdot z _ { t } + ( 1 { - } \tau ) \cdot z _ { s }$

The τ-weighting is principled: at $\tau { = } 0$ each decoder reconstructs its own modality (weight 1); at $\tau { = } 1$ the target decoder bears full responsibility. This matches supervision to the latent blend, avoiding conflicting gradients from a mismatched weighting.

Training Pipeline. SPHEREFLOW is trained in two independent stages:

Stage 1: Tokenizer training. We first train all modality-specific decoders $\{ D _ { \phi } ^ { ( m ) } \}$ with the DINO encoder frozen, using paired multi-modal data. Each decoder is trained with both the standard on-manifold reconstruction loss (Eq. 1) for its own modality, and the manifold-expanded loss over cross-modal pairs $( x _ { s } , x _ { t } )$

$$
\mathcal { L } _ { \mathrm { t o k e n i z e r } } = \mathcal { L } _ { \mathrm { r e c } } + \lambda \cdot \mathcal { L } _ { \mathrm { e x p a n d } }\tag{10}
$$

where λ balances the two terms. The $\mathcal { L } _ { \mathrm { r e c } }$ term ensures faithful on-manifold reconstruction for each modality, while $\mathcal { L } _ { \mathrm { e x p a n d } }$ extends all decoders’ domains into the off-manifold region that the flow will later traverse. Finally, since DINOv2 is a general-purpose encoder, we adapt it to the medical imaging modalities: we unfreeze the shared encoder $\mathcal { E } _ { \psi }$ (initialized from DINOv2) and train it jointly with all decoders, adding an alignment loss to a frozen copy ${ \mathcal { E } } _ { 0 }$ of the pretrained encoder:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a i l g n } } = \lambda _ { \mathrm { c o s } } \big ( 1 - \cos ( z , z ^ { \prime } ) \big ) + \lambda _ { \mathrm { m s e } } \big \| \mathrm { G A P } ( z ) - \mathrm { G A P } ( z ^ { \prime } ) \big \| _ { 2 } ^ { 2 } , \quad z = \mathcal { E } _ { \psi } ( x ) , z ^ { \prime } = \mathrm { s g } \big ( \mathcal { E } _ { 0 } ( x ) \big ) } \end{array}\tag{11}
$$

where $\cos ( \cdot , \cdot )$ is the token-level cosine similarity, GAP is global average pooling, and $\operatorname { s g } ( \cdot )$ is the stop-gradient. The first term anchors token directions and the second anchors global semantics, letting the encoder adapt to the reconstruction and expansion objectives while staying close to the pretrained representation. After this stage, the tokenizer (aligned encoder + trained decoders) is fixed.

Stage 2: Flow matching training. With the tokenizer frozen, we train the DiT independently with L<sub>CMFM</sub> (Eq. 6). Since the decoders were already expanded in Stage 1, they can reconstruct from any point along the flow trajectory at inference without further fine-tuning.

This decoupled design has a practical advantage: the tokenizer (encoder + all modality-specific decoders) is trained once, while the DiT can be trained or fine-tuned for specific modality pairs without retraining any decoder.

## 4 Experiments

We evaluate SPHEREFLOW on multi-contrast MRI synthesis (BraTS, §4.2), CT–MRI translation (SynthRAD2023, §4.3), downstream segmentation under missing modalities (§4.3), and ablations (§4.4). Figure 4 previews the headline results.

## 4.1 Experimental Setup

Datasets. BraTS [Baid et al., 2021] provides four paired brain MRI modalities $( T _ { 1 } , T _ { 1 c } , T _ { 2 } ,$ FLAIR/T ); SynthRAD2023 (brain track) [Thummerer et al., 2023] provides paired CT and $T _ { 1 } \cdot$ weighted MRI. We extract axial 256×256 slices after skull-stripping and per-volume normalization, and follow official splits.

![](images/3d1921ec54ebf9f0a9c4081189bc9542e773d81e5c13487307836a7c85a65f7b.jpg)  
LAMSYD MSMGAM AMMT ainurs

![](images/35d42236cb41e9317442fb4f8a3defcd5a533ff394bd9b1707db82365aaa588d.jpg)  
AMSVD AMSMGAM MMI 12DZhaeBBrain Ours

![](images/aad6646923ff469d270cf6b63f748341118c9bb0242b0b731811b2f993ab4d51.jpg)  
AMSYD MSYGAN AMMT 2 ZhaeBrrain ainurs

![](images/bbc27c992f95a8a82e13beeaf2a49a77e5f1cf1eddc0acd79ffbf096733cf32c.jpg)  
AMSVD AMSMGAM ANMI Ours

![](images/922caf97228b3704a6df899b7bc9b0624c2622ddc2b58a13232848b0782da015.jpg)  
Figure 4: Main-results overview. Top: PSNR/Dice (↑); bottom: rFID/HD95 (↓) across four generation tasks and segmentation. SPHEREFLOW achieves the best or near-best score on every panel.

Table 1: 1-to-1 generation on BraTS (all 12 directions).
<table><tr><td>Modalities</td><td>MMSyn</td><td>MMGAN</td><td>MMT</td><td>ModImp</td><td>M2DN</td><td>Zhang</td><td>CodeBrain</td><td>Ours</td><td>MMSyn</td><td>MMGAN</td><td>MMT</td><td>ModImp</td><td>M2DN</td><td>Zhang</td><td>CodeBrain</td><td>Ours</td></tr><tr><td>T1 → T1c</td><td>30.50</td><td>29.93</td><td>30.75</td><td>31.38</td><td>29.84</td><td>32.79</td><td>32.85</td><td>33.53</td><td>0.9185</td><td>0.9075</td><td>0.9077</td><td>0.9235</td><td>0.8840</td><td>0.9384</td><td>0.9405</td><td>0.9414</td></tr><tr><td>T1 → T2</td><td>27.91</td><td>29.68</td><td>29.62</td><td>29.29</td><td>28.87</td><td>31.76</td><td>30.77</td><td>32.68</td><td>0.8985</td><td>0.9035</td><td>0.8916</td><td>0.9010</td><td>0.8675</td><td>0.9326</td><td>0.9289</td><td>0.9390</td></tr><tr><td>T1 → T}</td><td>33.16</td><td>34.92</td><td>34.72</td><td>33.81</td><td>32.88</td><td>36.06</td><td>35.36</td><td>37.35</td><td>0.9220</td><td>0.9317</td><td>0.9244</td><td>0.9271</td><td>0.8937</td><td>0.9470</td><td>0.9458</td><td>0.9529</td></tr><tr><td>T1c → T1</td><td>34.82</td><td>36.81</td><td>33.65</td><td>33.83</td><td>33.50</td><td>37.21</td><td>36.34</td><td>37.87</td><td>0.9421</td><td>0.9517</td><td>0.9235</td><td>0.9392</td><td>0.9067</td><td>0.9586</td><td>0.9575</td><td>0.9612</td></tr><tr><td>T1c → T2</td><td>28.86</td><td>31.15</td><td>29.45</td><td>28.80</td><td>29.05</td><td>31.80</td><td>30.58</td><td>32.67</td><td>0.9019</td><td>0.9183</td><td>0.8872</td><td>0.8952</td><td>0.8683</td><td>0.9305</td><td>0.9252</td><td>0.9379</td></tr><tr><td>T1c → Tf</td><td>33.19</td><td>34.90</td><td>33.64</td><td>33.67</td><td>33.43</td><td>35.60</td><td>35.24</td><td>37.49</td><td>0.9147</td><td>0.9326</td><td>0.9148</td><td>0.9242</td><td>0.8972</td><td>0.9472</td><td>0.9463</td><td>0.9535</td></tr><tr><td>T2 → T1</td><td>30.57</td><td>32.42</td><td>31.82</td><td>28.86</td><td>30.49</td><td>34.64</td><td>31.72</td><td>35.63</td><td>0.9200</td><td>0.9251</td><td>0.9108</td><td>0.9097</td><td>0.8874</td><td>0.9509</td><td>0.9324</td><td>0.9562</td></tr><tr><td>T2 → T1c</td><td>26.06</td><td>26.51</td><td>28.57</td><td>26.01</td><td>26.80</td><td>30.81</td><td>27.50</td><td>32.05</td><td>0.8749</td><td>0.8908</td><td>0.8802</td><td>0.8795</td><td>0.8610</td><td>0.9260</td><td>0.9031</td><td>0.9349</td></tr><tr><td>T2 →Tf</td><td>30.48</td><td>35.20</td><td>34.25</td><td>31.87</td><td>33.02</td><td>37.15</td><td>33.57</td><td>38.11</td><td>0.9060</td><td>0.9323</td><td>0.9146</td><td>0.9047</td><td>0.8891</td><td>0.9555</td><td>0.9312</td><td>0.9584</td></tr><tr><td>Tj → T1</td><td>31.34</td><td>34.08</td><td>32.52</td><td>31.80</td><td>31.12</td><td>34.28</td><td>33.13</td><td>35.80</td><td>0.9243</td><td>0.9349</td><td>0.9174</td><td>0.9226</td><td>0.8931</td><td>0.9483</td><td>0.9438</td><td>0.9548</td></tr><tr><td>Tf → T1c</td><td>28.41</td><td>26.04</td><td>29.45</td><td>29.30</td><td>28.12</td><td>31.07</td><td>30.27</td><td>31.92</td><td>0.8966</td><td>0.8986</td><td>0.8957</td><td>0.8954</td><td>0.8718</td><td>0.9240</td><td>0.9233</td><td>0.9312</td></tr><tr><td>Tf →T2</td><td>25.61</td><td>31.30</td><td>29.30</td><td>29.12</td><td>29.19</td><td>31.50</td><td>30.78</td><td>32.83</td><td>0.8688</td><td>0.9107</td><td>0.8887</td><td>0.8851</td><td>0.8686</td><td>0.9285</td><td>0.9234</td><td>0.9381</td></tr><tr><td>AVERAGE</td><td>30.08</td><td>31.91</td><td>31.48</td><td>30.64</td><td>30.53</td><td>33.72</td><td>32.34</td><td>34.83</td><td>0.9074</td><td>0.9198</td><td>0.9047</td><td>0.9089</td><td>0.8824</td><td>0.9406</td><td>0.9334</td><td>0.9466</td></tr></table>

<table><tr><td>Modalities</td><td>MMSyn</td><td>MMGAN</td><td>MMT</td><td>ModImp</td><td>M2DN</td><td>Zhang</td><td>CodeBrain</td><td>Ours</td><td>MMSyn</td><td>MMGAN</td><td>MMT</td><td>ModImp</td><td>M2DN</td><td>Zhang</td><td>CodeBrain</td><td>Ours</td></tr><tr><td>T1 → T1c</td><td>52.25</td><td>118.12</td><td>112.16</td><td>112.75</td><td>134.75</td><td>27.49</td><td>56.12</td><td>19.70</td><td>2.97</td><td>2.47</td><td>2.52</td><td>2.73</td><td>2.29</td><td>2.90</td><td>2.91</td><td>3.09</td></tr><tr><td>T → T2</td><td>38.85</td><td>113.48</td><td>98.18</td><td>155.30</td><td>93.75</td><td>14.38</td><td>23.37</td><td>18.92</td><td>1.73</td><td>1.95</td><td>1.93</td><td>1.98</td><td>1.27</td><td>2.07</td><td>1.73</td><td>2.03</td></tr><tr><td>T1 →T}</td><td>39.16</td><td>129.69</td><td>132.42</td><td>141.37</td><td>94.71</td><td>31.41</td><td>34.16</td><td>19.47</td><td>3.05</td><td>2.88</td><td>2.68</td><td>2.30</td><td>2.39</td><td>3.15</td><td>3.20</td><td>2.93</td></tr><tr><td>T1c → T1</td><td>34.81</td><td>83.66</td><td>181.93</td><td>108.00</td><td>117.62</td><td>28.29</td><td>23.81</td><td>19.41</td><td>3.11</td><td>2.74</td><td>2.75</td><td>2.67</td><td>2.54</td><td>27</td><td>2.72</td><td>3.14</td></tr><tr><td>T1c → T2</td><td>43.23</td><td>98.51</td><td>115.51</td><td>160.10</td><td>94.79</td><td>29.70</td><td>33.17</td><td>16.90</td><td>1.72</td><td>2.00</td><td>2.16</td><td>1.91</td><td>1.25</td><td>2.11</td><td>1.64</td><td>2.01</td></tr><tr><td>T1c → Tf</td><td>61.80</td><td>154.50</td><td>154.57</td><td>160.60</td><td>102.35</td><td>35.56</td><td>35.45</td><td>19.39</td><td>2.92</td><td>2.90</td><td>2.33</td><td>2.39</td><td>2.26</td><td>3.03</td><td>3.02</td><td>2.93</td></tr><tr><td>T2 → T1</td><td>45.70</td><td>155.36</td><td>224.75</td><td>149.89</td><td>111.91</td><td>20.78</td><td>23.59</td><td>25.88</td><td>2.36</td><td>2.96</td><td>2.97</td><td>2.21</td><td>2.47</td><td>3.12</td><td>2.98</td><td>3.13</td></tr><tr><td>T2 2 → T1c</td><td>76.88</td><td>146.39</td><td>171.22</td><td>167.69</td><td>143.21</td><td>36.53</td><td>73.23</td><td>22.44</td><td>2.27</td><td>2.60</td><td>2.87</td><td>2.44</td><td>2.09</td><td>3.11</td><td>2.91</td><td>3.08</td></tr><tr><td>T2 → Tf</td><td>36.49</td><td>125.73</td><td>155.75</td><td>179.25</td><td>95.33</td><td>16.63</td><td>27.83</td><td>16.34</td><td>2.75</td><td>2.89</td><td>2.73</td><td>2.30</td><td>2.38</td><td>2.91</td><td>2.78</td><td>2.85</td></tr><tr><td>Tf → T1</td><td>43.33</td><td>131.76</td><td>179.63</td><td>183.45</td><td>121.98</td><td>55.86</td><td>58.35</td><td>22.50</td><td>3.33</td><td>2.76</td><td>3.13</td><td>2.40</td><td>2.73</td><td>3.12</td><td>3.00</td><td>3.26</td></tr><tr><td>Tf j → T1c</td><td>66.66</td><td>152.82</td><td>124.95</td><td>177.32</td><td>133.72</td><td>34.11</td><td>77.92</td><td>21.30</td><td>3.24</td><td>2.56</td><td>2.52</td><td>2.40</td><td>2.36</td><td>3.10</td><td>2.87</td><td>3.21</td></tr><tr><td>T →T2</td><td>43.79</td><td>129.58</td><td>105.97</td><td>178.85</td><td>94.69</td><td>29.17</td><td>53.95</td><td>16.90</td><td>1.67</td><td>2.01</td><td>2.04</td><td>1.79</td><td>1.29</td><td>1.72</td><td>1.43</td><td>2.01</td></tr><tr><td>AVERAGE</td><td>48.58</td><td>128.30</td><td>146.42</td><td>156.21</td><td>111.57</td><td>29.99</td><td>43.41</td><td>19.93</td><td>2.59</td><td>2.56</td><td>2.55</td><td>2.29</td><td>2.11</td><td>2.76</td><td>2.60</td><td>2.81</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Implementation. We use frozen DINOv2-B/14 [Oquab et al., 2023] with modality-specific ViT-G/14 RAE decoders [Zheng et al., 2025]. Stage 1 trains decoders in three steps: single-modality warm-up (Eq. 1), manifold-expansion fine-tuning (Eq. 9), and shared-encoder alignment. Stage 2 trains a DiTDH-XL [Peebles and Xie, 2023] with the cross-modal flow loss (Eq. 6) and logit-normal time sampling for up to 400 epochs. Inference uses 2-step Euler ODE on BraTS and 4-step on SynthRAD. For N-to-N, available source latents are averaged into a single $z _ { s } .$

Baselines and Metrics. We compare seven baselines: MMSyn [Chartsias et al., 2017], MM-GAN [Sharma and Hamarneh, 2019], MMT [Liu et al., 2023], M2DN [Meng et al., 2024], Mod-Imp [Qi et al., 2025], CodeBrain [Wu et al., 2025], and Zhang [Zhang et al., 2024], all trained on the same splits. We report PSNR, SSIM [Wang et al., 2004, Hore and Ziou, 2010], paired rFID [Heusel et al., 2017], IS [Salimans et al., 2016], and iFID [Xu et al., 2026] for generation quality, plus Dice, HD95, IoU, Sensitivity for downstream segmentation (3D U-Net trained on full-modality BraTS and evaluated on imputed inputs).

## 4.2 Multi-Modal MRI Imputation on BraTS

1-to-1 Translation. Table 1 reports all 12 directions among $\{ T _ { 1 } , T _ { 1 c } , T _ { 2 } , T _ { f } \}$ . SPHEREFLOW achieves the best PSNR on every direction (avg 34.83 dB, +1.11 dB over runner-up) and reduces rFID by 33.6% (29.99→19.93), with the largest gains on hard pairs $( T _ { 2 } {  } T _ { 1 c } \mathrm { : } + 1 . 2 \bar { 4 } { \bf d B } )$ .

N-to-N Synthesis. Table 2 (left) summarises average performance over all 14 input combinations (full per-combination results in Appendix Table 6). SPHEREFLOW leads on PSNR (35.88, +0.75 dB over runner-up), SSIM (0.9530), and rFID (19.68, −1.5% vs. Zhang), with gains largest in singlesource regimes. Simple latent averaging suffices for multi-source fusion because all modalities share the same hypersphere.

Table 2: N-to-N synthesis and segmentation averages on BraTS (14 input combinations). Full oracle = all ground-truth modalities. Per-combination results in Appendix Tables 6, 7.
<table><tr><td rowspan="2">Method</td><td colspan="4">Synthesis (N-to-N avg)</td><td colspan="4">Segmentation (avg)</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>rFID ↓</td><td>IS ↑</td><td>Dice ↑</td><td>HD95↓</td><td>IoU↑</td><td>Sens ↑</td></tr><tr><td>Zero-fill</td><td></td><td></td><td></td><td></td><td>0.3185</td><td>169.40</td><td>0.2693</td><td>0.3260</td></tr><tr><td>MMSyn</td><td>31.81</td><td>0.9213</td><td>37.12</td><td>2.64</td><td>0.5546</td><td>57.73</td><td>0.4641</td><td>0.5553</td></tr><tr><td>MMGAN</td><td>33.81</td><td>0.9339</td><td>93.12</td><td>2.51</td><td>0.5657</td><td>55.39</td><td>0.4877</td><td>0.5524</td></tr><tr><td>MMT</td><td>32.40</td><td>0.9125</td><td>135.77</td><td>2.48</td><td>0.5450</td><td>46.27</td><td>0.4633</td><td>0.5374</td></tr><tr><td>ModImp</td><td>32.23</td><td>0.9226</td><td>123.87</td><td>2.42</td><td>0.5627</td><td>59.44</td><td>0.4832</td><td>0.5574</td></tr><tr><td>M2DN</td><td>31.68</td><td>0.8936</td><td>104.17</td><td>2.18</td><td>0.5393</td><td>53.32</td><td>0.4507</td><td>0.5028</td></tr><tr><td>Zhang</td><td>35.13</td><td>0.9504</td><td>19.98</td><td>2.76</td><td>0.5080</td><td>53.06</td><td>0.4216</td><td>0.4872</td></tr><tr><td>CodeBrain</td><td>33.88</td><td>0.9446</td><td>32.45</td><td>2.63</td><td>0.5930</td><td>56.37</td><td>0.5174</td><td>0.5759</td></tr><tr><td>Ours</td><td>35.88</td><td>0.9530</td><td>19.68</td><td>2.80</td><td>0.7473</td><td>19.02</td><td>0.6511</td><td>0.7418</td></tr><tr><td>Full oracle</td><td></td><td></td><td></td><td></td><td>0.8554</td><td>11.41</td><td>0.7829</td><td>0.8583</td></tr></table>

Table 3: CT ↔ MRI translation on SynthRAD2023.
<table><tr><td rowspan="2">Methods</td><td colspan="4">CT → M RI</td><td colspan="4">MRI CT</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>rFID ↓</td><td>IS ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>rFID ↓</td><td>IS ↑</td></tr><tr><td>MMSyn</td><td>21.42</td><td>0.7666</td><td>107.89</td><td>2.98</td><td>22.56</td><td>0.8112</td><td>98.76</td><td>2.73</td></tr><tr><td>MMGAN</td><td>22.58</td><td>0.7543</td><td>182.83</td><td>2.95</td><td>25.69</td><td>0.7983</td><td>131.40</td><td>2.52</td></tr><tr><td>ModImp</td><td>16.40</td><td>0.6535</td><td>217.59</td><td>2.72</td><td>25.30</td><td>0.7553</td><td>140.33</td><td>2.46</td></tr><tr><td>MMT</td><td>23.68</td><td>0.7488</td><td>213.48</td><td>2.65</td><td>25.40</td><td>0.8105</td><td>145.04</td><td>2.32</td></tr><tr><td>CodeBrain</td><td>22.94</td><td>0.8083</td><td>54.82</td><td>3.08</td><td>25.26</td><td>0.8569</td><td>61.31</td><td>2.76</td></tr><tr><td>Zhang</td><td>25.51</td><td>0.8635</td><td>30.91</td><td>3.01</td><td>27.52</td><td>0.8944</td><td>38.61</td><td>2.92</td></tr><tr><td>M2DN</td><td>21.44</td><td>0.7497</td><td>117.02</td><td>2.42</td><td>23.79</td><td>0.8017</td><td>99.15</td><td>2.88</td></tr><tr><td>Ours</td><td>24.70</td><td>0.8452</td><td>13.65</td><td>2.71</td><td>27.82</td><td>0.9084</td><td>13.99</td><td>2.81</td></tr></table>

Table 4: Encoder comparison (Stage-1 reconstruction on $T _ { 1 c } ) .$
<table><tr><td rowspan="2">Encoders</td><td colspan="5">T1c</td></tr><tr><td>PSNR ↑</td><td>SSIM↑</td><td>rFID↓</td><td>IS↑</td><td>iFID↓</td></tr><tr><td>MedSAM</td><td>29.17</td><td>0.8642</td><td>140.82</td><td>1.96</td><td>48.58</td></tr><tr><td>MedMAE</td><td>31.23</td><td>0.8907</td><td>43.17</td><td>2.84</td><td>144.45</td></tr><tr><td>DINOv2</td><td>30.57</td><td>0.8824</td><td>35.29</td><td>2.91</td><td>40.64</td></tr></table>

## 4.3 Cross-Domain Translation: CT ↔ MRI

CT–MRI translation is harder than within-MRI synthesis due to fundamentally different imaging physics [Wang et al., 2024]. Table 3 shows that SPHEREFLOW achieves the best PSNR/SSIM/rFID on MRI→CT (27.82 dB, rFID 13.99). On CT→MRI, Zhang slightly leads PSNR/SSIM, but SPHEREFLOW halves rFID (30.91→13.65), reflecting the geometric advantage of decoding onmanifold latents.

Downstream Segmentation. Table 2 (right) evaluates downstream segmentation: a 3D U-Net trained on full-modality BraTS and tested on imputed inputs (full per-combination results in Appendix Table 7). SPHEREFLOW reaches avg Dice 0.7473 (+15.4 over runner-up CodeBrain, within 0.108 of the full-modality oracle 0.8554) and reduces HD95 from 46.27 to 19.02 (−59%). In single-source cases, baselines collapse below Dice 0.10–0.20, while SPHEREFLOW stays above 0.45.

## 4.4 Ablation Studies

Choice of Encoder. Table 4 compares encoders at Stage-1 reconstruction on $T _ { 1 c } .$ MedMAE wins PSNR (31.23) but suffers 3.6× higher iFID (144.45 vs. 40.64); DINOv2 trades 0.66 dB for the best rFID/IS/iFID, confirming that geometric regularity matters more than pixel fidelity for transport-based imputation.

Stage-wise RAE Ablation. We progressively enable each Stage-1 step and evaluate on-manifold $\scriptstyle ( z = \mathcal { E } _ { \mathrm { D I N O } } ( x _ { t } ) )$ and chord $( z _ { 1 / 2 } { = } \frac { 1 } { 2 } ( z _ { s } { + } z _ { t } ) )$ reconstruction.

Table 5: Stage-wise RAE ablation $( T _ { 1 } / T _ { 1 c } ) .$
<table><tr><td rowspan="2">Cascade stage</td><td colspan="3">On-manifold</td><td colspan="3">Chord (τ=0.5)</td></tr><tr><td>PSNR ↑</td><td>SSIM↑</td><td>rFID ↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>rFID ↓</td></tr><tr><td>Step 1: single-modality warm-up</td><td>33.14</td><td>0.8732</td><td>28.29</td><td>32.04</td><td>0.8731</td><td>41.66</td></tr><tr><td>+ Step 2: manifold expansion</td><td>33.83</td><td>0.8795</td><td>30.92</td><td>33.29</td><td>0.8784</td><td>35.06</td></tr><tr><td>+ Step 3: shared-encoder align</td><td>44.06</td><td>0.9545</td><td>16.55</td><td>41.40</td><td>0.9453</td><td>16.28</td></tr></table>

Table 5 shows that Step 2 (manifold expansion) provides a modest gain on its own (+0.69 dB onmanifold, +1.25 dB chord) by exposing the decoder to off-manifold inputs, but the chord rFID actually worsens slightly (28.29→30.92), indicating that the decoder generalizes to nearby off-manifold points without yet aligning modality-specific representations. Step 3 (shared-encoder alignment) is decisive: on-manifold PSNR jumps to 44.06 and the chord-vs-on-manifold rFID gap collapses from 13.4 to 0.3, confirming that once all modality decoders share a geometrically consistent latent neighborhood, the flow trajectory becomes nearly indistinguishable from the data manifold.

## 5 Conclusion

We presented SPHEREFLOW, which reframes missing modality imputation as short-range transport on a shared hypersphere $S ^ { d - 1 }$ , replacing the noise source with the observed-modality latent and expanding the decoder domain to cover the off-manifold region the flow traverses. These two geometry-aware modifications reduce the worst-case off-manifold deviation from $\Theta ( { \sqrt { d } } )$ to a dimension-independent bound $1 - \cos ( \alpha / 2 )$ , and yield state-of-the-art fidelity on BraTS and CT–MRI translation while closing roughly 80% of the segmentation gap to the full-modality oracle.

## Limitation

A current limitation is the reliance on DINOv2 providing cross-modal compactness; modalities without shared visual semantics (e.g., text and imaging) would require a domain-adapted encoder or a cross-modal pre-training stage. More broadly, the principle of aligning the latent geometry with the task structure should extend to other conditional generation settings where paired source data is available, such as image-guided speech synthesis or satellite multi-spectral fusion.

## Acknowledgments and Disclosure of Funding

This work was partially supported by the RGC Collaborative Research Fund (No. C5055-24G), the Seed Fund of the Research Institute for Smart Ageing (No. P0050946), the Tsinghua-PolyU Joint Research Initiative Fund (No. P0056509), and PolyU UGC funding (No. P0053716). The authors declare no competing interests.

## References

Ujjwal Baid, Satyam Ghodasara, Suyash Mohan, Michel Bilello, Evan Calabrese, Errol Colak, Keyvan Farahani, Jayashree Kalpathy-Cramer, Felipe C Kitamura, Sarthak Pati, et al. The rsnaasnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification. arXiv preprint arXiv:2107.02314, 2021.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

Hun Chang, Byunghee Cha, and Jong Chul Ye. Dino-sae: Dino spherical autoencoder for high-fidelity image reconstruction and generation. arXiv preprint arXiv:2601.22904, 2026.

Agisilaos Chartsias, Thomas Joyce, Mario Valerio Giuffrida, and Sotirios A Tsaftaris. Multimodal mr synthesis via modality-invariant latent representation. IEEE transactions on medical imaging, 37(3):803–814, 2017.

Ricky TQ Chen and Yaron Lipman. Flow matching on general geometries. arXiv preprint arXiv:2302.03660, 2023.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020.

Salman UH Dar, Mahmut Yurt, Levent Karacan, Aykut Erdem, Erkut Erdem, and Tolga Cukur. Image synthesis in multi-contrast mri with conditional generative adversarial networks. IEEE transactions on medical imaging, 38(10):2375–2388, 2019.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366–2369. IEEE, 2010.

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Amandeep Kumar and Vishal M Patel. Learning on the manifold: Unlocking standard diffusion transformers with representation encoders. arXiv preprint arXiv:2602.10099, 2026.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Jiang Liu, Srivathsa Pasumarthi, Ben Duffy, Enhao Gong, Keshav Datta, and Greg Zaharchuk. One model to synthesize them all: Multi-contrast multi-scale transformer for missing data imputation. IEEE transactions on medical imaging, 42(9):2577–2591, 2023.

Siyu Liu, Chujie Qin, Hubery Yin, Qixin Yan, Zheng-Peng Duan, Chen Li, Jing Lyu, Chun-Le Guo, and Chongyi Li. Improving reconstruction of representation autoencoder. arXiv preprint arXiv:2602.08620, 2026.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Xiangxi Meng, Kaicong Sun, Jun Xu, Xuming He, and Dinggang Shen. Multi-modal modalitymasked diffusion network for brain mri synthesis with random modality missing. IEEE Transactions on Medical Imaging, 43(7):2587–2598, 2024.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Liangce Qi, Yusi Liu, Yuqin Li, Weili Shi, Guanyuan Feng, and Zhengang Jiang. A unified missing modality imputation model with inter-modality contrastive and consistent learning. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 44–53. Springer, 2025.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

Anmol Sharma and Ghassan Hamarneh. Missing mri pulse sequence synthesis using multi-modal generative adversarial network. IEEE transactions on medical imaging, 39(4):1170–1183, 2019.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Adrian Thummerer, Erik Van der Bijl, Arthur Galapon Jr, Joost JC Verhoeff, Johannes A Langendijk, Stefan Both, Cornelis (Nico) AT van den Berg, and Matteo Maspero. Synthrad2023 grand challenge dataset: Generating synthetic ct for radiotherapy. Medical physics, 50(7):4664–4674, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Tonghe Wang, Yang Lei, Yabo Fu, Jacob F Wynne, Walter J Curran, Tian Liu, and Xiaofeng Yang. A review on medical imaging synthesis using deep learning and its clinical applications. Journal of applied clinical medical physics, 22(1):11–36, 2021.

Zhenbin Wang, Lei Zhang, Lituan Wang, and Zhenwei Zhang. Soft masked mamba diffusion model for ct to mri conversion. arXiv preprint arXiv:2406.15910, 2024.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

Yicheng Wu, Tao Song, Zhonghua Wu, Jin Ye, Zongyuan Ge, Zhaolin Chen, and Jianfei Cai. Codebrain: Imputing any brain mri via modality-and instance-specific codes. arXiv preprint arXiv:2501.18328, 2025.

Tongda Xu, Mingwei He, Shady Abu-Hussein, Jose Miguel Hernandez-Lobato, Haotian Zhang, Kai Zhao, Chao Zhou, Ya-Qin Zhang, and Yan Wang. Making reconstruction fid predictive of diffusion generation fid. arXiv preprint arXiv:2603.05630, 2026.

Xinzhe Zhang, Junjie Liang, Peng Cao, Jinzhu Yang, and Osmar R Zaiane. Structure-aware mri translation: Multi-modal latent diffusion model with arbitrary missing modalities. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 508–518. Springer, 2025.

Yue Zhang, Chengtao Peng, Qiuli Wang, Dan Song, Kaiyan Li, and S Kevin Zhou. Unified multimodal image synthesis for missing modality imputation. IEEE Transactions on Medical Imaging, 44(1):4–18, 2024.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223–2232, 2017.

## A Proof of Proposition 1

Proof. We proceed in three parts: (i) we derive $\| z _ { \tau } \| ^ { 2 }$ as a function of $\tau , ( \mathrm { i i } )$ we minimize it over $[ 0 , 1 ]$ to identify the point of maximum off-manifold deviation, and (iii) we compare with the Gaussiansource setting.

Part (i): Norm of the interpolant. Let $z _ { s } , z _ { t } \in \mathcal { S } ^ { d - 1 }$ with inner product $\langle z _ { s } , z _ { t } \rangle = \cos \alpha$ for angular separation $\alpha \in [ 0 , \pi ]$ . The linear interpolant is $z _ { \tau } = \tau z _ { t } + ( 1 - \tau ) z _ { s } \mathrm { f o r } \tau \in [ 0 , 1 ]$ . Expanding the squared norm:

$$
\begin{array} { r l } & { \| z _ { \tau } \| ^ { 2 } = \| \tau z _ { t } + \left( 1 - \tau \right) z _ { s } \| ^ { 2 } } \\ & { \qquad = \tau ^ { 2 } \| z _ { t } \| ^ { 2 } + ( 1 - \tau ) ^ { 2 } \| z _ { s } \| ^ { 2 } + 2 \tau ( 1 - \tau ) \langle z _ { s } , z _ { t } \rangle } \\ & { \qquad = \tau ^ { 2 } + ( 1 - \tau ) ^ { 2 } + 2 \tau ( 1 - \tau ) \cos \alpha } \end{array}\tag{12}
$$

where the last equality uses $\| z _ { s } \| = \| z _ { t } \| = 1$ . Expanding and collecting terms in τ:

$$
\| z _ { \tau } \| ^ { 2 } = 2 ( 1 - \cos \alpha ) \tau ^ { 2 } - 2 ( 1 - \cos \alpha ) \tau + 1\tag{13}
$$

This is a quadratic in τ with leading coefficient $a : = 2 ( 1 - \cos \alpha ) \geq 0 .$

Part (ii): Minimizing $\| z _ { \tau } \| ^ { 2 }$ over $[ 0 , 1 ]$ . Case $\alpha = 0 ( z _ { s } = z _ { t } )$ . The coefficient $a = 0$ , so $\| z _ { \tau } \| ^ { 2 } = 1$ for all τ, and $\Delta = 0$ trivially.

Case $\alpha \in ( 0 , \pi ]$ . Now $a > 0 ,$ , so (13) is a strictly convex quadratic in τ. Its unique global minimizer is

$$
\tau ^ { \star } = \frac { 2 ( 1 - \cos \alpha ) } { 2 \cdot 2 ( 1 - \cos \alpha ) } = \frac { 1 } { 2 }\tag{14}
$$

Since $\tau ^ { \star } = \frac { 1 } { 2 } \in [ 0 , 1 ]$ , the unconstrained minimizer lies within the domain and is therefore the constrained minimizer as well. Substituting $\begin{array} { r } { \tau = \frac { 1 } { 2 } } \end{array}$ into (13):

$$
\| z _ { 1 / 2 } \| ^ { 2 } = 2 ( 1 - \cos \alpha ) \cdot { \frac { 1 } { 4 } } - 2 ( 1 - \cos \alpha ) \cdot { \frac { 1 } { 2 } } + 1 = 1 - { \frac { 1 } { 2 } } ( 1 - \cos \alpha ) = { \frac { 1 + \cos \alpha } { 2 } }\tag{15}
$$

We now apply the half-angle identity $\cos ^ { 2 } ( \alpha / 2 ) = ( 1 + \cos \alpha ) / 2$ . Since $\alpha \in [ 0 , \pi ]$ we have $\alpha / 2 \in [ 0 , \pi / 2 ]$ , so cos $( \alpha / 2 ) \geq 0$ and hence

$$
\| z _ { 1 / 2 } \| = { \sqrt { \frac { 1 + \cos \alpha } { 2 } } } = { \sqrt { \cos ^ { 2 } { \frac { \alpha } { 2 } } } } = \cos { \frac { \alpha } { 2 } }\tag{16}
$$

Because $\| z _ { \tau } \| ^ { 2 }$ is minimized at $\begin{array} { r } { \tau = \frac { 1 } { 2 } } \end{array}$ , so is $\| z _ { \tau } \|$ (the square root is monotone), and the deviation $1 - \| z _ { \tau } \|$ is maximized there. To confirm that the deviation is non-negative, rewrite (12) as $\| z _ { \tau } \| ^ { 2 } =$ $1 - \ddot { 2 } \tau ( \ddot { 1 } - \tau ) ( 1 - \cos \alpha )$ . Since $\tau ( 1 { - } \tau ) \geq 0$ for $\tau \in [ 0 , 1 ]$ and $1 - \cos \alpha \geq 0$ , the subtracted term is non-negative, so $\| z _ { \tau } \| ^ { 2 } \leq 1$ and hence $\| z _ { \tau } \| \leq 1$ for all $\bar { \tau ( \cdot | ) } \in [ 0 , 1 ]$ . Therefore:

$$
\Delta : = \displaystyle \operatorname* { m a x } _ { \tau \in [ 0 , 1 ] } \bigl ( 1 - \| z _ { \tau } \| \bigr ) = 1 - \| z _ { 1 / 2 } \| = 1 - \cos \frac { \alpha } { 2 }\tag{17}
$$

Part (iii): Gaussian-source comparison. In standard flow matching the source is $\varepsilon \sim \mathcal { N } ( 0 , I _ { d } )$ with interpolant $z _ { \tau } = \tau z + ( 1 { - } \tau ) \varepsilon$ , where $z \in S ^ { d - 1 }$ . We show $\begin{array} { r } { \mathbb { E } \big [ \operatorname* { m a x } _ { \tau \in [ 0 , 1 ] } \big | \big \| z _ { \tau } \big \| - 1 \big | \big ] = \Theta ( \sqrt { d } ) } \end{array}$ Write $\textstyle \mu : = \operatorname { \mathbb { E } } [ \| \varepsilon \| ] = { \sqrt { 2 } } \Gamma { \left( { \frac { d + 1 } { 2 } } \right) } / \Gamma { \left( { \frac { d } { 2 } } \right) } = { \sqrt { d } } - { \frac { 1 } { 4 { \sqrt { d } } } } + O ( d ^ { - 3 / { \overline { { 2 } } } } )$

Lower bound. $\mathrm { A t } ~ \tau = 0 , \ z _ { 0 } = \varepsilon .$ , so max $r \vert \Vert z _ { \tau } \Vert - 1 \vert \geq \Vert \varepsilon \Vert - 1$ whenever $\| \varepsilon \| \geq 1$ . Since $\| \varepsilon \| ^ { 2 } \sim \chi _ { d } ^ { 2 } .$ , the event $\| \varepsilon \| < 1$ has probability at most $e ^ { - \dot { \Omega } ( d ) }$ for $d \geq 2$ . Therefore

$$
\mathbb { E } \Big [ \operatorname* { m a x } _ { \tau } \big | \left\| z _ { \tau } \right\| - 1 \big | \Big ] \geq \mu - 1 - O ( e ^ { - \Omega ( d ) } ) = \Omega ( \sqrt { d } )\tag{18}
$$

Upper bound. By the triangle inequality, $\Vert z _ { \tau } \Vert \leq \tau \Vert z \Vert + ( 1 - \tau ) \Vert \varepsilon \Vert = \tau + ( 1 - \tau ) \Vert \varepsilon \Vert$ for all $\tau .$ Whenever $\| \varepsilon \| \geq 2 ,$ two cases exhaust the possibilities for every $\tau \in [ 0 , 1 ] \colon \mathrm { i f } \ \| z _ { \tau } \| \geq 1$ , then $| \left\| z _ { \tau } \right\| - 1 | = \| z _ { \tau } \| - 1 \leq ( 1 - \tau ) ( \| \varepsilon \| - 1 ) \leq \| \varepsilon \| - 1 ; \operatorname { i f } \| z _ { \tau } \| < 1$ , then $\left| \left\| z _ { \tau } \right\| - 1 \right| = 1 - \left\| z _ { \tau } \right\| \leq$

$1 \leq \| \varepsilon \| - 1$ . Hence max $\tau | \left\| z _ { \tau } \right\| - 1 | \leq \| \varepsilon \| - 1$ whenever $\| \varepsilon \| \geq 2$ . Since $\operatorname* { P r } [ | | \varepsilon | | < 2 ] = e ^ { - \Omega ( d ) }$ for large d:

$$
\mathbb { E } \Big [ \operatorname* { m a x } _ { \tau } \big | \| z _ { \tau } \| - 1 \big | \Big ] \ \leq \ \mu - 1 + O ( e ^ { - \Omega ( d ) } ) = O ( \sqrt { d } )\tag{19}
$$

Combining: $\mathbb { E } [ \operatorname* { m a x } _ { \tau } \big | \| z _ { \tau } \| - 1 \big | \big ] = \Theta ( \sqrt { d } )$ , which grows with dimension. By contrast, the data-assource deviation $\Delta = 1 - \cos ( \alpha / 2 )$ depends only on the angular separation α and is independent of d. (The two notions of deviation coincide: as shown in Part (ii), $\| z _ { \tau } \| \leq 1$ when both endpoints lie on $S ^ { d - 1 }$ , so $1 - \| z _ { \tau } \| = \left| \ \| z _ { \tau } \| - 1 \right|$ throughout.) □

## B Evaluation Metrics

We employ two families of metrics: generation quality metrics that compare each synthesized image to its ground-truth counterpart, and downstream segmentation metrics that measure how well a pre-trained segmentation model performs when missing modalities are filled in by each imputation method.

## B.1 Generation Quality Metrics

Peak Signal-to-Noise Ratio (PSNR) Hore and Ziou [2010]. PSNR measures per-pixel reconstruction fidelity between a synthesized image xˆ and its ground truth $x ,$ defined as

$$
\mathrm { P S N R } ( \hat { x } , x ) = 1 0 \cdot \log _ { 1 0 } \frac { \mathrm { M A X } ^ { 2 } } { \mathrm { M S E } ( \hat { x } , x ) }\tag{20}
$$

where MAX is the maximum possible pixel value and $\begin{array} { r } { \mathrm { M S E } ( \hat { x } , x ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \hat { x } _ { i } - x _ { i } ) ^ { 2 } } \end{array}$ is the mean squared error over all N pixels. Higher is better. We report PSNR in decibels (dB), averaged over all test slices.

Structural Similarity Index (SSIM) Wang et al. [2004]. SSIM captures perceptual image quality by comparing local luminance, contrast, and structure between xˆ and x:

$$
\mathrm { S S I M } ( \hat { x } , x ) = \frac { ( 2 \mu _ { \hat { x } } \mu _ { x } + C _ { 1 } ) ( 2 \sigma _ { \hat { x } x } + C _ { 2 } ) } { ( \mu _ { \hat { x } } ^ { 2 } + \mu _ { x } ^ { 2 } + C _ { 1 } ) ( \sigma _ { \hat { x } } ^ { 2 } + \sigma _ { x } ^ { 2 } + C _ { 2 } ) }\tag{21}
$$

where $\mu$ and $\sigma ^ { 2 }$ denote local means and variances computed over a sliding window, $\sigma _ { \hat { x } x }$ is the local cross-covariance, and $C _ { 1 } , C _ { 2 }$ are small stabilizing constants. SSIM ranges in $[ - 1 , 1 ] ;$ ; higher is better. We use the default $1 1 \times 1 1$ Gaussian window and report the mean over all test slices.

Reconstruction Fréchet Inception Distance (rFID) Heusel et al. [2017]. rFID measures distributional similarity between the set of synthesized images and the set of ground-truth images in Inception-v3 feature space. It is computed as the Fréchet distance between two multivariate Gaussians fitted to the Inception features:

$$
\mathrm { r F I D } = \| \mu _ { \hat { x } } - \mu _ { x } \| ^ { 2 } + \mathrm { T r } \Big ( \Sigma _ { \hat { x } } + \Sigma _ { x } - 2 \big ( \Sigma _ { \hat { x } } \Sigma _ { x } \big ) ^ { 1 / 2 } \Big )\tag{22}
$$

where $( \mu _ { \hat { x } } , \Sigma _ { \hat { x } } )$ and $\left( \mu _ { x } , \Sigma _ { x } \right)$ are the mean and covariance of the Inception features for the synthesized and real sets, respectively. Lower is better. Following standard practice, we compute rFID in a paired fashion: the synthesized set and ground-truth set contain the same subjects, so rFID captures both sample quality and alignment fidelity. We compute rFID per imputation direction and report the average.

Inception Score (IS) Salimans et al. [2016]. IS evaluates the quality and diversity of generated images using the Inception-v3 classifier:

$$
\mathrm { I S } = \exp ( \mathbb { E } _ { \hat { x } } \left[ D _ { \mathrm { K L } } ( p ( y | \hat { x } ) \| p ( y ) ) \right] )\tag{23}
$$

where $p ( y | \hat { x } )$ is the conditional class distribution predicted by Inception for a synthesized image, and $p ( y ) = \mathbb { E } _ { \hat { x } } [ p ( y | \hat { x } ) ]$ is the marginal class distribution. A high IS indicates that individual images are confidently classified (sharp $p ( y | \hat { x } ) \big \rangle$ ) while the overall set is diverse (broad $p ( y ) )$ ). Higher is better.

Inception Fréchet Inception Distance (iFID) Xu et al. [2026]. iFID is a variant of FID designed to be more predictive of generation-stage FID when evaluated at the reconstruction stage. It uses the same formula as rFID but applies a learned feature transformation that better correlates reconstruction quality with downstream generation quality. Lower is better. We report iFID only in the encoder ablation (Table 4) where it provides additional discriminative power beyond rFID.

## B.2 Downstream Segmentation Metrics

For segmentation evaluation, we train a 3D U-Net on full-modality BraTS data (all four real modalities) and evaluate it on inputs where missing modalities have been replaced by each imputation method’s output. The segmentation model is never retrained or fine-tuned on imputed data. We evaluate on three tumor sub-regions defined by the BraTS protocol: whole tumor (WT), tumor core (TC), and enhancing tumor (ET), and report the average across all three.

Dice Similarity Coefficient (Dice). Dice measures volumetric overlap between the predicted segmentation $\hat { S }$ and ground-truth segmentation S:

$$
\mathrm { D i c e } ( \hat { S } , S ) = \frac { 2 | \hat { S } \cap S | } { | \hat { S } | + | S | }\tag{24}
$$

Dice ranges in [0, 1]; higher is better. A Dice of 1 indicates perfect overlap. We report the mean Dice across the three tumor sub-regions (WT, TC, ET), averaged over all test volumes.

95th-Percentile Hausdorff Distance (HD95). HD95 measures the boundary accuracy of the predicted segmentation by computing the 95th percentile of the symmetric surface distances:

$$
\mathrm { H D } 9 5 ( \hat { S } , S ) = \mathrm { p e r c e n t i l e } _ { 9 5 } \left( \left\{ \operatorname* { m i n } _ { s \in \partial S } d ( p , s ) : p \in \partial \hat { S } \right\} \cup \left\{ \operatorname* { m i n } _ { p \in \partial \hat { S } } d ( s , p ) : s \in \partial S \right\} \right)\tag{25}
$$

where $\partial \hat { S }$ and $\partial \cal S$ denote the surfaces of the predicted and ground-truth segmentations, and $d ( \cdot , \cdot )$ is the Euclidean distance. HD95 is reported in millimeters; lower is better. The 95th percentile is used instead of the maximum to reduce sensitivity to outlier voxels. We report the mean HD95 across the three sub-regions, averaged over all test volumes.

Intersection over Union (IoU). IoU, also known as the Jaccard index, measures volumetric overlap:

$$
\operatorname { I o U } ( { \hat { S } } , S ) = { \frac { | { \hat { S } } \cap S | } { | { \hat { S } } \cup S | } }\tag{26}
$$

IoU ranges in [0, 1]; higher is better. It is related to Dice by $\mathrm { I o U } = \mathrm { D i c e } / ( 2 - \mathrm { D i c e } )$ and penalizes both false positives and false negatives more heavily than Dice.

Sensitivity (Sens). Sensitivity, also known as recall or true positive rate, measures the fraction of ground-truth tumor voxels that are correctly detected:

$$
\mathrm { S e n s i t i v i t y } ( { \hat { S } } , S ) = { \frac { | { \hat { S } } \cap S | } { | S | } }\tag{27}
$$

Sensitivity ranges in [0, 1]; higher is better. High sensitivity is clinically important because missed tumor regions (false negatives) can lead to inadequate treatment planning.

## C Sampling-Step Efficiency

Figure 5 compares SPHEREFLOW against Gaussian-source flow matching (same backbone and tokenizer) as a function of Euler steps N on ${ T _ { 1 } {  } T _ { 1 c } . \mathrm { A t } N { = } 1 }$ the Gaussian baseline collapses to ∼17 dB—one step cannot bridge the $\Theta ( { \sqrt { d } } )$ off-manifold gap—while SPHEREFLOW already reaches 34.3 dB. Both saturate by N=2–8; SPHEREFLOW retains a persistent ∼1.7 dB lead, confirming the advantage is structural.

![](images/9de11b8ad1da077260c279832692a828ed461daa7140d2ab534c1d25d565fdbc.jpg)  
Figure 5: PSNR vs. Euler steps $( T _ { 1 } {  } T _ { 1 c } )$ . Gaussian FM collapses at $N { = } 1$ ; SPHEREFLOW starts at 34.3 dB. Both saturate by $N { = } 2 { - } 8 ;$ ; SPHEREFLOW retains a ∼1.7 dB lead.

## D Full N-to-N Synthesis Results

This section provides per-combination breakdowns for N-to-N synthesis on BraTS, complementing the averaged results in Table 2 (left).

Table 6: Full N-to-N synthesis results on BraTS (all 14 input combinations).
<table><tr><td>T1</td><td>Modalities</td><td>T1c</td><td>Tf MMSyn</td><td></td><td>MMGAN</td><td>ModImp</td><td>PSNR ↑ M2DN</td><td>Zhang</td><td>CodeBrain</td><td></td><td>MMSyn</td><td>MMGAN</td><td>MMT</td><td>ModImp</td><td>SSIM↑ M2DN</td><td></td><td></td><td>CodeBrain</td></tr><tr><td></td><td>T2 o</td><td>o</td><td>30.52</td><td>31.51</td><td>MMT</td><td></td><td>30.53</td><td>33.54</td><td>32.99</td><td>Ours 34.52</td><td>0.9130</td><td>0.9142</td><td>0.9076</td><td>0.9169</td><td>0.8818</td><td>Zhang 0.9393</td><td>0.9384</td><td>Ours 0.9444</td></tr><tr><td></td><td></td><td></td><td>29.03</td><td>31.38</td><td>31.65 31.49</td><td>31.33 28.88</td><td>30.10</td><td></td><td>30.93</td><td>35.26</td><td>0.9003</td><td>0.9161</td><td>0.9015</td><td>0.8978</td><td>0.8792</td><td>0.9441</td><td>0.9222</td><td>0.9498</td></tr><tr><td>O</td><td></td><td>o</td><td>32.29</td><td>34.29</td><td></td><td></td><td></td><td>34.20</td><td></td><td></td><td></td><td></td><td></td><td>0.9193</td><td>0.8907</td><td></td><td></td><td>0.9509</td></tr><tr><td>o</td><td>o o</td><td></td><td></td><td></td><td>32.21</td><td>32.07</td><td>31.99</td><td>34.87</td><td>34.05</td><td>36.01</td><td>0.9196</td><td>0.9342</td><td>0.9083</td><td></td><td></td><td>0.9454</td><td>0.9430</td><td></td></tr><tr><td>C</td><td></td><td></td><td>28.45</td><td>30.47</td><td>30.38</td><td>30.04</td><td>29.48</td><td>32.28</td><td>31.39</td><td>33.51</td><td>0.8959 0.9245</td><td>0.9147 0.9407</td><td>0.9003 0.9124</td><td>0.9013 0.9259</td><td>0.8778 0.8947</td><td>0.9336 0.9547</td><td>0.9302 0.9475</td><td>0.9414 0.9569</td></tr><tr><td>0</td><td></td><td></td><td>32.23 34.82</td><td>34.63</td><td>32.28</td><td>32.48</td><td>32.19</td><td>35.72</td><td>34.18</td><td>36.35</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.9647</td></tr><tr><td></td><td></td><td></td><td></td><td>36.99</td><td>34.79</td><td>34.86</td><td>34.38</td><td>38.11</td><td>37.01</td><td>38.86</td><td>0.9404</td><td>0.9506</td><td>0.9261</td><td>0.9395</td><td>0.9091</td><td>0.9632</td><td>0.9594</td><td>0.9550</td></tr><tr><td></td><td></td><td></td><td>32.56</td><td>34.87</td><td>33.46</td><td>33.24</td><td>32.59</td><td>35.82</td><td>35.11</td><td>36.60</td><td>0.9256</td><td>0.9389</td><td>0.9199</td><td>0.9295</td><td>0.8984</td><td>0.9535</td><td>0.9499</td><td></td></tr><tr><td></td><td></td><td></td><td>30.84</td><td>32.63</td><td>31.59</td><td>31.69</td><td>30.15</td><td>33.90</td><td>32.71</td><td>34.75</td><td>0.9197</td><td>0.9314</td><td>0.9115</td><td>0.9234</td><td>0.8898</td><td>0.9478</td><td>0.9438</td><td>0.9507</td></tr><tr><td></td><td></td><td></td><td>31.84</td><td>33.76</td><td>32.41</td><td>31.88</td><td>31.80</td><td>34.80</td><td>33.49</td><td>35.60</td><td>0.9183</td><td>0.9305</td><td>0.9093</td><td>0.9176</td><td>0.8878</td><td>0.9475</td><td>0.9418</td><td>0.9501</td></tr><tr><td></td><td></td><td></td><td>30.50</td><td>32.05</td><td>30.66</td><td>31.18</td><td>30.06</td><td>33.32</td><td>32.65</td><td>34.07</td><td>0.9159</td><td>0.9244 0.9505</td><td>0.9043</td><td>0.9184</td><td>0.8828</td><td>0.9443</td><td>0.9411</td><td>0.9475</td></tr><tr><td></td><td></td><td></td><td>35.26 35.36</td><td>37.38 37.49</td><td>35.87 34.76</td><td>35.49 35.22</td><td>35.52 34.26</td><td>38.62 38.35</td><td>37.47 36.93</td><td>39.20 38.80</td><td>0.9390 0.9474</td><td>0.9572</td><td>0.9291 0.9314</td><td>0.9360</td><td>0.9108</td><td>0.9638</td><td>0.9585</td><td>0.9639</td></tr><tr><td></td><td></td><td></td><td>31.41</td><td>33.18</td><td>31.33</td><td>32.08</td><td>30.15</td><td>34.08</td><td>33.28</td><td>34.40</td><td>0.9229</td><td>0.9370</td><td>0.9129</td><td>0.9488 0.9288</td><td>0.9135</td><td>0.9668 0.9501</td><td>0.9628</td><td>0.9670</td></tr><tr><td></td><td></td><td></td><td>30.27</td><td>32.68</td><td>30.72</td><td>30.75</td><td>30.38</td><td>34.15</td><td>32.05</td><td>34.35</td><td>0.9153</td><td>0.9340</td><td>0.9010</td><td>0.9138</td><td>0.9134 0.8801</td><td></td><td>0.9461</td><td>0.9486</td></tr><tr><td></td><td>Average</td><td></td><td>31.81</td><td>33.81</td><td>32.40</td><td>32.23</td><td>31.68</td><td>35.13</td><td>33.88</td><td>35.88</td><td>0.9213</td><td>0.9339</td><td>0.9125</td><td>0.9226</td><td>0.8936</td><td>0.9511 0.9504</td><td>0.9398</td><td>0.9509 0.9530</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.9446</td><td></td></tr><tr><td colspan="2">T2</td><td>Modalities Tf</td><td></td><td></td><td></td><td></td><td>rFID ↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>IS↑</td><td></td><td></td><td></td></tr><tr><td>T1</td><td></td><td>T1c</td><td>MMSyn</td><td>MMGAN</td><td>MMT</td><td>ModImp</td><td>M2DN</td><td>Zhang</td><td>CodeBrain</td><td>Ours</td><td>MMSyn</td><td>MMGAN</td><td>MMT</td><td>ModImp</td><td>M2DN</td><td>Zhang</td><td>CodeBrain</td><td>Ours</td></tr><tr><td></td><td>o</td><td>o</td><td>43.42</td><td>120.43</td><td>114.33</td><td>136.24</td><td>107.74</td><td>24.43</td><td>37.88</td><td>19.36</td><td>2.58</td><td>2.43</td><td>2.38</td><td>2.34</td><td>1.98</td><td>2.71</td><td>2.61</td><td>2.69</td></tr><tr><td>o</td><td></td><td>o</td><td>53.02</td><td>142.49</td><td>183.83</td><td>165.24</td><td>104.92</td><td>24.65</td><td>41.55</td><td>21.56</td><td>2.46</td><td>2.82</td><td>2.86</td><td>2.32</td><td>2.02</td><td>3.05</td><td>2.89</td><td>3.02</td></tr><tr><td>o</td><td></td><td></td><td>46.61</td><td>112.22</td><td>150.69</td><td>142.93</td><td>116.82</td><td>31.18</td><td>30.81</td><td>18.63</td><td>2.59</td><td>2.55</td><td>2.42</td><td>2.32</td><td>2.31</td><td>2.62</td><td>2.46</td><td>2.70</td></tr><tr><td>O</td><td></td><td></td><td>51.26</td><td>138.05</td><td>136.82</td><td>179.83</td><td>116.80</td><td>39.71</td><td>63.41</td><td>20.23</td><td>2.75</td><td>2.44 2.35</td><td>2.56 2.34</td><td>2.20 2.37</td><td>2.13 1.99</td><td>2.65 2.46</td><td>2.43 2.28</td><td>2.83 2.62</td></tr><tr><td></td><td></td><td></td><td>37.64 23.42</td><td>85.16 77.00</td><td>138.45 159.32</td><td>112.86 106.32</td><td>98.33 97.89</td><td>18.62 13.83</td><td>24.93 17.30</td><td>18.45 19.09</td><td>2.23 2.91</td></table>

Analysis by number of available sources. A clear trend emerges when the 14 combinations are grouped by how many modalities are observed.

Single-source (rows 1–4). This is the hardest regime: three out of four modalities must be hallucinated from a single observation. SPHEREFLOW achieves the best PSNR on all four single-source settings, with gains of +0.98 dB (from T<sub>1</sub>), +1.06 dB (from $T _ { 2 } ) , + 1 . 1 4 \ : \mathrm { d B }$ (from $\boldsymbol { T } _ { 1 c } )$ , and +1.23 dB (from $T _ { f } )$ over the runner-up Zhang. In rFID the advantage is even more pronounced: the single-source average rFID for SPHEREFLOW is 19.95, versus 29.99 for Zhang (−33.5%). This confirms that when only one modality is available, the short transport distance on the hypersphere matters most—the generator needs to cover three missing directions, and starting from an on-manifold semantic feature rather than noise provides a much stronger prior.

Notably, the FLAIR-only setting $( T _ { f } \mathrm { - o n l y } )$ is the weakest single source for all baselines (e.g., Zhang drops to 32.28 dB PSNR, the lowest among single-source rows), yet SPHEREFLOW still reaches 33.51 dB—demonstrating robustness even when the source modality carries the least contrast-specific information.

Two-source (rows 5–10). With two observed modalities, latent averaging places the source representation closer to every target, and all methods improve. SPHEREFLOW leads in PSNR on all six two-source settings and achieves the best rFID on three of six. On the remaining three,

Zhang holds a small rFID edge (e.g., 13.83 vs. 19.09 for $\{ T _ { 2 } , T _ { 1 c } \} )$ . This pattern—SPHEREFLOW dominating pixel-level fidelity while occasionally trailing on distributional metrics in multi-source regimes—is consistent with Zhang’s VAE decoder producing slightly sharper textures when ample source information is available, whereas SPHEREFLOW’s DINO-based decoder prioritizes semantic consistency.

Three-source (rows 11–14). When three modalities are observed, only one must be imputed, and the task becomes substantially easier for all methods. Performance differences narrow: SPHEREFLOW leads on PSNR in three of four settings and ties or slightly trails Zhang on SSIM in two. The average three-source PSNR for SPHEREFLOW is 36.69 dB versus 36.30 dB for Zhang (+0.39 dB), a smaller gap than in the single-source regime (+1.10 dB). This diminishing marginal gain is expected: with three sources, even simple averaging produces a high-quality latent near the target, so the geometric advantage of on-manifold transport is less decisive.

Per-metric observations. PSNR and SSIM are strongly correlated across all settings, and SPHERE-FLOW achieves the best average on both (35.88 dB, 0.9530). The SSIM advantage is most visible in single-source regimes where structural preservation matters most.

rFID. SPHEREFLOW achieves the best overall average (19.68 vs. 19.98 for Zhang), but the percombination picture is nuanced: SPHEREFLOW dominates in single-source rFID (4/4 best) while Zhang leads in several multi-source settings. We attribute this to the fact that Zhang’s per-modality VAE decoder can leverage multiple source cues to sharpen textures, whereas SPHEREFLOW’s decoder operates on a single averaged latent. Despite this, SPHEREFLOW’s average rFID remains the best across all 14 combinations.

IS. SPHEREFLOW achieves the highest average IS (2.80), indicating that synthesized images are both diverse and recognizable. The IS advantage is largest in single-source settings, consistent with the PSNR and rFID trends.

## E Full Segmentation Results

This section provides per-combination breakdowns for downstream brain tumor segmentation on BraTS, complementing the averaged results in Table 2 (right). The segmentation model is a 3D U-Net trained once on full-modality (all four real modalities) BraTS data. At test time, missing modalities are filled in by each imputation method; the segmentation model is never retrained or fine-tuned. This protocol isolates the quality of the imputed modalities: better imputation translates directly to higher segmentation accuracy.

Table 7: Full downstream segmentation results on BraTS (all 14 input combinations).
<table><tr><td>T1</td><td>T2</td><td>T1c</td><td>Zero</td><td>MMSyn</td><td>MMGAN</td><td>MMT</td><td>ModImp</td><td>M2DN</td><td>Zhang</td><td>CodeBrain</td><td>Ours</td><td>Zero</td><td>MMSyn</td><td>MMGAN</td><td>MMT</td><td>ModImp</td><td>M2DN</td><td>Zhang</td><td>CodeBrain</td><td>Ours</td></tr><tr><td></td><td>o</td><td>o</td><td>0.0094</td><td>0.0686</td><td>0.1852</td><td>0.1910</td><td>0.0851</td><td>0.1558</td><td>0.1570</td><td>0.2094</td><td>0.5710</td><td>316.87</td><td>237.74</td><td>167.35</td><td>94.73</td><td>232.55</td><td>140.21</td><td>114.11</td><td>169.75</td><td>35.75</td></tr><tr><td></td><td></td><td></td><td>0.0133</td><td>0.3025</td><td>0.2740</td><td>0.2696</td><td>0.2679</td><td>0.1796</td><td>0.2662</td><td>0.3637</td><td>0.6477</td><td>326.12</td><td>65.63</td><td>94.85</td><td>94.31</td><td>77.71</td><td>131.71</td><td>71.67</td><td>91.35</td><td>23.15</td></tr><tr><td></td><td></td><td></td><td>0.0257</td><td>0.6412</td><td>0.6029</td><td>0.5389</td><td>0.6125</td><td>0.5467</td><td>0.3932</td><td>0.6771</td><td>0.7811</td><td>301.14</td><td>35.79</td><td>66.93</td><td>58.71</td><td>43.21</td><td>62.99</td><td>75.44</td><td>38.97</td><td>22.09</td></tr><tr><td></td><td></td><td></td><td>0.3695</td><td>0.4096</td><td>0.3583</td><td>0.3395</td><td>0.4633</td><td>0.3508</td><td>0.4260</td><td>0.3946</td><td>0.6574</td><td>105.40</td><td>68.57</td><td>75.06</td><td>62.55</td><td>55.85</td><td>77.51</td><td>51.32</td><td>97.50</td><td>22.16</td></tr><tr><td></td><td></td><td></td><td>0.7386</td><td>0.8284</td><td>0.8348</td><td>0.8173</td><td>0.8254</td><td>0.7897</td><td>0.8195</td><td>0.8421</td><td>0.8512</td><td>28.89</td><td>13.31</td><td>13.07</td><td>17.19</td><td>14.66</td><td>20.98</td><td>14.49</td><td>13.71</td><td>12.05</td></tr><tr><td></td><td></td><td></td><td>0.0326</td><td>0.7345</td><td>0.7771</td><td>0.7497</td><td>0.7715</td><td>0.7488</td><td>0.5505</td><td>0.8088</td><td>0.8360</td><td>304.90</td><td>28.07</td><td>20.52</td><td>22.61</td><td>20.58</td><td>23.56</td><td>55.67</td><td>17.48</td><td>14.21</td></tr><tr><td></td><td></td><td></td><td>0.0645</td><td>0.2702</td><td>0.3634</td><td>0.3178</td><td>0.3285</td><td>0.3423</td><td>0.3321</td><td>0.3752</td><td>0.6706</td><td>260.65</td><td>90.64</td><td>66.87</td><td>69.16</td><td>105.50</td><td>61.82</td><td>63.14</td><td>82.26</td><td>22.89</td></tr><tr><td></td><td></td><td></td><td>0.4362</td><td>0.4498</td><td>0.4463</td><td>0.3922</td><td>0.4542</td><td>0.4578</td><td>0.4962</td><td>0.4452</td><td>0.6991</td><td>84.39</td><td>56.88</td><td>55.82</td><td>53.63</td><td>51.04</td><td>42.15</td><td>47.37</td><td>71.33</td><td>19.43</td></tr><tr><td></td><td></td><td></td><td>0.0147</td><td>0.7196</td><td>0.7021</td><td>0.6940</td><td>0.7007</td><td>0.6350</td><td>0.3675</td><td>0.7439</td><td>0.7991</td><td>345.43</td><td>26.04</td><td>40.16</td><td>32.35</td><td>31.11</td><td>43.96</td><td>105.77</td><td>29.84</td><td>20.33</td></tr><tr><td></td><td></td><td></td><td>0.4143</td><td>0.4325</td><td>0.4298</td><td>0.4109</td><td>0.4348</td><td>0.4107</td><td>0.4581</td><td>0.4532</td><td>0.6930</td><td>81.95</td><td>71.79</td><td>63.10</td><td>51.80</td><td>75.35</td><td>56.53</td><td>43.54</td><td>73.44</td><td>19.01</td></tr><tr><td></td><td></td><td></td><td>0.2744</td><td>0.7687</td><td>0.8018</td><td>0.7872</td><td>0.7955</td><td>0.7891</td><td>0.6800</td><td>0.8130</td><td>0.8411</td><td>126.01</td><td>22.48</td><td>22.11</td><td>19.47</td><td>18.75</td><td>18.62</td><td>33.03</td><td>18.58</td><td>13.67</td></tr><tr><td></td><td></td><td></td><td>0.7975</td><td>0.8392</td><td>0.8518</td><td>0.8413</td><td>0.8462</td><td>0.8440</td><td>0.8215</td><td>0.8501</td><td>0.8535</td><td>16.35</td><td>11.18</td><td>11.63</td><td>12.83</td><td>12.13</td><td>12.49</td><td>14.50</td><td>11.04</td><td>11.34</td></tr><tr><td></td><td></td><td></td><td>0.4468</td><td>0.4553</td><td>0.4433</td><td>0.4353</td><td>0.4456</td><td>0.4920</td><td>0.5016</td><td>0.4738</td><td>0.7085</td><td>60.15</td><td>67.39</td><td>65.76</td><td>45.09</td><td>81.65</td><td>37.92</td><td>40.47</td><td>62.34</td><td>17.68</td></tr><tr><td></td><td></td><td></td><td>0.8214</td><td>0.8441</td><td>0.8484</td><td>0.8453</td><td>0.8462</td><td>0.8085</td><td>0.8428</td><td>0.8520</td><td>0.8533</td><td>13.40</td><td>12.71</td><td>12.25</td><td>13.30</td><td>12.05</td><td>16.11</td><td>12.33</td><td>11.63</td><td>12.51</td></tr><tr><td></td><td>Average</td><td></td><td>0.3185</td><td>0.5546</td><td>0.5657</td><td>0.5450</td><td>0.5627</td><td>0.5393</td><td>0.5080</td><td>0.5930</td><td>0.7473</td><td>169.40</td><td>57.73</td><td>55.39</td><td>46.27</td><td>59.44</td><td>53.32</td><td>53.06</td><td>56.37</td><td>19.02</td></tr><tr><td colspan="4">Full</td><td></td><td></td><td></td><td>0.8554</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>11.41</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="14">Modalities</td><td></td><td>Zero MMSyn</td><td></td><td>MMGAN</td><td>MMT</td><td>Sensitivity ModImp</td><td></td><td></td><td>Zhang</td><td></td></tr><tr><td>T1</td><td>T2</td><td>T1c Tf</td><td>Zero</td><td>MMSyn</td><td>MMGAN</td><td>MMT</td><td>IoU ModImp</td><td>M2DN</td><td>Zhang</td><td>CodeBrain</td><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td>M2DN</td><td></td><td>CodeBrain</td><td>Ours</td></tr><tr><td>o</td><td>o</td><td>0</td><td>0.0094</td><td>0.0441</td><td>0.1308</td><td>0.1297</td><td>0.0576</td><td>0.1002</td><td>0.1044</td><td>0.1547</td><td>0.4489</td><td>0.0094</td><td>0.0792</td><td>0.1498</td><td>0.1867</td><td>0.0689</td><td>0.1185</td><td>0.1300</td><td>0.1764</td><td>0.5651</td></tr><tr><td></td><td></td><td></td><td>0.0116 0.0183</td><td>0.2138 0.5381</td><td>0.2012 0.5070</td><td>0.2013 0.4369</td><td>0.1945 0.5006</td><td>0.1323 0.4413</td><td>0.1884 0.2965</td><td>0.2898 0.5835</td><td>0.5310 0.6891</td><td>0.0130 0.0283</td><td>0.4367 0.6090</td><td>0.2637 0.5446</td><td>0.2683 0.4974</td><td>0.2668 0.5997</td><td>0.1545 0.4753</td><td>0.2410 0.3478</td><td>0.3458 0.6518</td><td>0.6551 0.7663</td></table>

Analysis by number of available sources. Single-source (rows 1–4). This is the most challenging and clinically relevant regime: only one modality is observed, and the segmentation model must operate on three imputed channels. All baselines collapse in at least some single-source settings—for example, with T -only input, the best baseline (CodeBrain) reaches only Dice 0.2094, far below clinical utility. SPHEREFLOW dramatically outperforms all baselines in every single-source case: Dice scores of 0.5710 (T -only), 0.6477 (T -only), 0.7811 $( T _ { 1 c } \mathrm { - o n l y ) }$ , and 0.6574 $( T _ { f } \mathrm { - o n l y } )$ . The gap over the runner-up CodeBrain ranges from +0.10 (T<sub>1c</sub>-only) to +0.36 (T<sub>1</sub>-only) in Dice, and HD95 improves by $3 { \overline { { - } } } 5 \times ( { \mathrm { e . g . } } , 1 6 9 . 7 { \overline { { 5 } } } \to 3 5 . 7 5$ for $T _ { 1 } \mathrm { - o n l y ) }$ . This result is particularly striking because the segmentation model was never exposed to imputed data during training—SPHEREFLOW’s synthesized modalities are realistic enough to be consumed directly by a model trained on real data.

Among the four single-source inputs, $T _ { 1 c }$ -only yields the highest Dice (0.7811) for SPHEREFLOW, which is expected: contrast-enhanced $T _ { 1 c }$ highlights the tumor core, providing the richest signal for segmentation. Conversely, T<sub>1</sub>-only is the hardest (0.5710), as non-contrast $\bar { T _ { 1 } }$ lacks tumor-specific signal. Even so, SPHEREFLOW with $T _ { 1 }$ -only surpasses every baseline’s performance with any single source.

Two-source (rows $5 \mathrm { - } I O )$ . With two observed modalities, segmentation accuracy improves across all methods. The key finding is that SPHEREFLOW remains the best on all six two-source combinations, with particularly large gains when $T _ { 1 c }$ is missing. For example, with $\{ T _ { 1 } , T _ { 2 } \}$ observed (no $\displaystyle T _ { 1 c } ,$ no $T _ { f } ) ,$ baselines plateau around Dice 0.33–0.38, while SPHEREFLOW reaches 0.6706 $( + 0 . 3 0 $ over CodeBrain). This indicates that SPHEREFLOW successfully synthesizes the contrast-enhanced features that the segmentation model relies on most.

When $T _ { 1 c }$ is available as one of the two sources (rows 5, 6, 9), all methods perform well (Dice $> 0 . 7 5 )$ , and the gap between SPHEREFLOW and baselines narrows. This is consistent with the interpretation that the segmentation model’s performance is bottlenecked by $T _ { 1 c }$ quality, and when the real $T _ { 1 c }$ is present, imputation quality of other modalities matters less.

Three-source (rows $\ l I { - } I 4 ) .$ . With three modalities observed, only one is imputed, and performance approaches the full-modality oracle for all methods. SPHEREFLOW still leads on 12 out of 14 Dice entries in this regime. The two settings where baselines are competitive are $\{ T _ { 2 } , T _ { 1 c } , T _ { f } \}$ (missing $T _ { 1 } )$ and $\{ T _ { 1 } , T _ { 1 c } , T _ { f } \}$ (missing $T _ { 2 } )$ —both cases where the missing modality is relatively redundant given the remaining three. Here, SPHEREFLOW achieves Dice 0.8535 and 0.8533 respectively, within 0.002–0.005 of the full oracle (0.8554), effectively solving the imputation problem.

Gap to the full-modality oracle. The full oracle (all four real modalities) achieves Dice 0.8554 and HD95 11.41. SPHEREFLOW reaches an average Dice of 0.7473 and HD95 of 19.02—closing 79.8% of the Dice gap and 95.2% of the HD95 gap relative to the zero-fill baseline (Dice 0.3185, HD95 169.40). By contrast, the best baseline CodeBrain closes only 51.1% of the Dice gap and 71.5% of the HD95 gap. The advantage is most dramatic in HD95: SPHEREFLOW’s average of 19.02 is 2.4× better than the best baseline (46.27 for MMT) and only 1.7× the oracle, indicating that imputed modalities preserve spatial accuracy of tumor boundaries.

Per-metric observations. IoU closely tracks Dice across all settings, with SPHEREFLOW leading on 14/14 combinations. The average IoU of 0.6511 is +0.1337 above the best baseline (0.5174 for CodeBrain), a 25.8% relative improvement.

Sensitivity measures the fraction of true tumor voxels correctly detected. SPHEREFLOW achieves the highest average sensitivity (0.7418), +0.1659 above CodeBrain (0.5759). High sensitivity is clinically critical—missed tumor regions can lead to under-treatment. The largest sensitivity gains appear in single-source settings (e.g., 0.5651 vs. 0.1764 for T<sub>1</sub>-only), where baselines catastrophically under-detect tumors.

Synthesis quality vs. segmentation performance. An interesting observation is the disconnect between synthesis fidelity and downstream utility for some baselines. Zhang achieves the second-best synthesis metrics (PSNR, rFID) but ranks only sixth in segmentation Dice (0.5080), well below CodeBrain (0.5930) and even MMSyn (0.5546). This suggests that Zhang’s VAE-based decoder produces images that look realistic but do not preserve the fine-grained diagnostic features (e.g., tumor boundaries, contrast enhancement patterns) that the segmentation model expects. SPHEREFLOW does not exhibit this disconnect: it leads on both synthesis fidelity and downstream segmentation, confirming that the DINO-based hyperspherical representation preserves semantically meaningful content through the imputation pipeline.

## F Qualitative Results

![](images/a22db10eacf47de8da8af1b9b44d2b7e6b6368cd9194b1d75f841cb892aeaf04.jpg)  
Figure 6: Qualitative comparison of synthesized images and error maps. Top two rows: $T _ { 1 } + T _ { 2 } + T _ { f }  T _ { 1 c }$ on BraTS. Row 1 shows synthesized $T _ { 1 c }$ images; row 2 shows the corresponding per-pixel absolute error (jet colormap, darker blue = lower error). Bottom two rows: MRI → CT on SynthRAD2023, with the same layout. Columns correspond to seven baselines, SPHEREFLOW, and the ground truth (GT).

Synthesis quality (Figure 6). On BraTS $T _ { 1 } + T _ { 2 } + T _ { f }  T _ { 1 c }$ (top), most baselines produce blurry outputs that lose fine anatomical detail—MMSyn and MMT wash out cortical sulci, while M2DN introduces intensity artifacts in the ventricles. Zhang and CodeBrain recover overall brain structure but exhibit elevated error in the tumor region, where contrast enhancement is hardest to hallucinate from non-contrast inputs. SPHEREFLOW produces the sharpest reconstruction with the lowest error across both healthy tissue and the tumor. The error map (row 2) confirms this quantitatively: SPHEREFLOW’s error is uniformly low (predominantly dark blue), whereas all baselines show concentrated high-error patches (red/orange) around the tumor core and cortical boundaries.

On SynthRAD2023 MRI → CT (bottom), the cross-domain gap is larger due to fundamentally different imaging physics. Several baselines (MMSyn, M2DN, ModImp) fail to reproduce bone density contrast, producing washed-out or inverted intensities. Zhang generates plausible CT contrast but with visible ring-like artifacts along the skull boundary. SPHEREFLOW best reproduces the char acteristic CT bone–soft-tissue contrast, and its error map shows the least residual error, particularly along the skull and internal structures where HU accuracy matters most for radiotherapy planning.

![](images/6fbc690378618b610d24d6234b7dbe37e6f2d5c1ecddf0c897fdc34e436cd865.jpg)  
Figure 7: Qualitative comparison of downstream segmentation. Setting: $T _ { 1 } + T _ { 2 } + T _ { f }  T _ { 1 c }$ on BraTS, then feeding all four modalities (three real + one imputed) to a 3D U-Net trained on full-modality data. Row 1: synthesized $T _ { 1 c }$ images. Row 2: segmentation overlays showing whole tumor (WT, yellow), tumor core (TC, blue), and enhancing tumor (ET, red). The rightmost column is the ground truth.

Downstream segmentation (Figure 7). The segmentation overlays reveal why synthesis fidelity translates (or fails to translate) into downstream accuracy. Most baselines produce segmentations that capture the rough whole-tumor region (WT, yellow) but catastrophically fail on the finer subregions: MMSyn, MMGAN, and M2DN either miss the tumor core (TC, blue) entirely or fragment it into disconnected patches; MMT and ModImp over-segment WT while under-detecting ET; Zhang produces a spatially displaced segmentation with missing ET. CodeBrain achieves a reasonable WT boundary but still under-segments TC and ET relative to the ground truth.

SPHEREFLOW produces the segmentation closest to the ground truth across all three sub-regions. The WT boundary closely matches the GT contour, TC is correctly localized within the tumor mass, and—critically—the enhancing tumor (ET, red) is detected with both high sensitivity and correct spatial extent. ET is the hardest sub-region to segment because it depends on contrast enhancement patterns that exist only in $T _ { 1 c } ;$ its accurate detection confirms that SPHEREFLOW’s imputed $T _ { 1 c }$ faithfully reproduces the contrast enhancement signal that the segmentation model relies on.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes].

Justification: The abstract and introduction claim state-of-the-art synthesis fidelity, a dimension-independent off-manifold bound, and closing ∼80% of the segmentation gap to the full-modality oracle. All are validated in Section 4 (Tables 1–5) and the bound is proved in Appendix A.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes].

Justification: The Conclusion identifies the reliance on DINOv2 cross-modal compactness as the main limitation, noting that modalities without shared visual semantics would require a domain-adapted encoder or cross-modal pre-training.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes].

Justification: Proposition 1 states its assumptions $( z _ { s } , z _ { t } \in S ^ { d - 1 }$ , angular separation α) and a complete three-part proof is provided in Appendix A, covering norm derivation, minimization, and the Gaussian-source comparison.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes].

Justification: Section 4.1 reports the encoder (DINOv2-B/14), decoder architecture (ViT-G/14 RAE), DiT variant (DiTDH-XL), training schedule (up to 400 epochs), inference steps (2-step and 4-step Euler), datasets (BraTS, SynthRAD2023), data preprocessing (axial 256×256 slices, skull-stripping, per-volume normalization), official splits, loss formulations (Eqs. 1–6), and all baselines.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes].

Justification: Code and trained checkpoints for both the tokenizer and the flow model are released at https://github.com/WTHH031230/SphereFlow. Both datasets (BraTS 2021, SynthRAD2023) are publicly available from their respective challenge organizers.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes].

Justification: Section 4.1 specifies the datasets, official splits, image resolution, preprocessing, encoder, decoder, DiT backbone, training stages, number of epochs, time sampling strategy (logit-normal), inference ODE steps, and N-to-N averaging strategy. All seven baselines are retrained on the same splits.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No].

Justification: Results are reported from single training runs without error bars or confidence intervals. We mitigate this by evaluating on the full official test splits across all 12 (1-to-1) and 14 (N-to-N) input combinations, providing a comprehensive view of performance variability across settings.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No].

Justification: The paper reports model architectures and training epochs but does not include GPU type, total wall-clock time, or memory requirements.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes].

Justification: The work uses publicly available medical imaging benchmarks (BraTS, SynthRAD2023) that have been de-identified and released for research. No new human-subject data is collected.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes].

Justification: The Conclusion discusses the positive clinical impact (practical deployment with 2–4 inference steps). A potential negative impact is that synthesized medical images could be mistaken for real acquisitions; however, the method is designed for modality imputation in clinical pipelines where the synthesized nature is known.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

## Answer: [N/A].

Justification: The method is a domain-specific medical image imputation model trained on public benchmarks. It does not produce general-purpose photorealistic images and poses minimal misuse risk beyond its intended clinical context.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes].

Justification: BraTS 2021, SynthRAD2023, DINOv2, RAE, and DiT are all cited. Both datasets are used under their official challenge licenses for research purposes.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes].

Justification: We release the code and trained checkpoints (tokenizer and flow model) with documentation and usage instructions at https://github.com/WTHH031230/ SphereFlow.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A].

Justification: The work does not involve crowdsourcing or new human-subject data collection.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [N/A].

Justification: No new human-subject experiments are conducted. Both datasets were previously de-identified and publicly released by their respective organizers.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A].

Justification: LLMs are not used as a component of the proposed method. Any LLM assistance in writing does not impact the core methodology or scientific rigor.