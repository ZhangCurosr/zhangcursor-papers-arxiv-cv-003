# Revisiting Cross-View Completion: Self-Supervised Pre-Training via Reconstruction Error Comparison

Thibaut Loiseau<sup>1</sup> Guillaume Bourmaud<sup>2</sup> Vincent Lepetit<sup>1</sup> <sup>1</sup> LIGM, Ecole Nationale des Ponts et Chaussées, IP Paris, Univ Gustave Eiffel, CNRS, France <sup>2</sup> Univ. Bordeaux, CNRS, Bordeaux INP, IMS, UMR 5218, France {thibaut.loiseau,vincent.lepetit}@enpc.fr guillaume.bourmaud@u-bordeaux.fr

![](images/2148bbcfa4763cbaebdf0636ebe2988c6ab5f300283dd534a32d53f873a79f36.jpg)  
Figure 1: Gekko: one network, three passes, no 3D annotation. A single network F is run three times on the same video frame pair: (1) it reconstructs a 90%-masked target with the reference view, (2) it reconstructs the same masked target without it, and (3) it regresses their per-pixel gap C(p) from the unmasked pair through one extra output channel C<sup>ˆ</sup>. That gap is large where a pixel is co-visible and vanishes where it is not, with no depth, pose or label anywhere in the pipeline. Regressing it is what turns cross-view completion’s blind spot into supervision: it supplies a binocular training signal for all masked regions, including the non-co-visible ones, where CroCo’s signal is implicitly monocular.

Project page: thibautloiseau.github.io/projects/gekko

## Abstract

Self-supervised pre-training via cross-view completion learns strong features for 3D vision from co-visible regions of image pairs. However, the reference view provides little information for reconstructing non-co-visible patches, implicitly yielding a monocular training signal in these regions. We introduce Gekko, which turns this limitation into a useful signal. The relative improvement of the crossview reconstruction error over a masked-autoencoder error is a self-supervised proxy for co-visibility: large improvements indicate co-visible regions, negligible ones non-co-visible areas. Gekko is a network, trained from scratch, that jointly performs cross-view completion, masked autoencoding, and per-pixel prediction of this relative improvement, providing an additional binocular signal for all masked regions without any ground-truth 3D annotation. Under identical architectures and training data, Gekko consistently outperforms CroCo on zero-shot correspondence estimation, relative pose estimation, and pointmap regression, with up to 6× higher accuracy at the strictest relative-pose threshold and a 22% drop in end-point error on ETH3D. The extra channel it learns is itself a strong co-visibility detector on unseen scenes, and Gekko’s frozen features outperform released cross-view backbones of comparable or larger size. It can also be trained directly from raw videos with a simple stride-based curriculum, removing the cumbersome 3D preprocessing prior methods require while matching models trained on curated data. Code and pre-trained models are publicly available.

## 1 Introduction

Learning meaningful representations for 3D vision from visual data is a fundamental challenge in com puter vision [12, 26, 10, 4, 27, 72, 39, 65], with applications spanning depth estimation [64, 34], visual localization [16], and 3D reconstruction [58, 22, 63, 55]. Self-supervised pre-training has emerged as a powerful paradigm for acquiring these representations without expensive manual annotations. Among recent approaches, cross-view completion [61] extends Masked Autoencoders (MAE) [27] to image pairs: patches in a target image are masked and reconstructed using both the remaining target patches and a reference view. The underlying principle is that reconstructing co-visible regions forces the network to establish correspondences and reason about geometry, yielding features that transfer effectively to downstream 3D tasks [2, 58, 31].

However, cross-view completion faces a critical limitation. When a masked patch is not co-visible (e.g., due to occlusion), the reference view provides little information in those regions, implicitly yielding a MAE-like monocular training signal. Consequently, cross-view completion pre-training does not benefit from image pairs with low overlap, where many masked patches lack a corresponding region in the reference view [61, 37].

We propose Gekko, a fully self-supervised pre-training framework that turns this limitation into a strength. Our key insight is that the relative improvement of the cross-view completion reconstruction error over the MAE reconstruction error is a reliable proxy for co-visibility (Figure 1). In co-visible regions, incorporating the reference image substantially improves reconstruction quality, yielding a large relative improvement. In non-co-visible regions, both reconstructions perform similarly, and the relative improvement is near zero. This proxy requires no ground-truth depth maps or camera poses, making it applicable to any collection of image pairs, including uncalibrated data.

Gekko is a network that jointly performs three tasks during pre-training: (i) cross-view completion, (ii) masked autoencoding, and (iii) prediction of the relative improvement map. The architecture is identical to CroCo [61] up to a single additional output channel, enabling a direct and fair comparison. The relative improvement prediction provides an additional dense binocular self-supervised signal for all masked regions. The cost of this third task is negligible: the encoder and the cross-attention decoder are byte-for-byte those of CroCo, and only the lightweight pixel head grows from three to four channels, so pre-training, fine-tuning and inference all keep CroCo’s architecture.

Under identical architectures and training data, Gekko consistently outperforms CroCo across three downstream tasks: up to 6× higher accuracy at the strictest threshold on relative metric pose estimation (ScanNet-1500), a 22% drop in end-point error on zero-shot correspondence estimation (ETH3D), and lower error on pointmap regression, with gains that scale to larger architectures. We also enable fully self-supervised pre-training from raw video with a simple stride-based curriculum, removing the expensive 3D preprocessing prior methods require [61, 37] while matching models trained on curated data. The extra channel is also directly interpretable: probed on its own, it detects co-visibility on scenes never seen during pre-training, and does so better than the cross-view reconstruction error it is derived from (Section 5.6).

Our contributions are as follows:

• We show that the relative improvement of a cross-view completion error over a masked autoencoding error is a reliable self-supervised proxy for co-visibility, substantially outperforming the cross-view completion error alone, and we propose Gekko, a network trained from scratch that jointly learns cross-view completion, masked autoencoding and relative improvement prediction. The relative improvement loss supplies a binocular signal for all masked regions without any ground-truth 3D labels.

• We demonstrate that Gekko consistently outperforms CroCo on three cross-view 3D vision tasks under identical architectures and training data, that its predicted channel is itself a strong co-visibility detector on scenes unseen during pre-training, and that its frozen features outperform released cross-view backbones of comparable or larger size.

• We show that Gekko can be trained from raw videos with a simple stride-based curriculum, removing the need for overlap-based data preprocessing and matching the performance of models trained on curated data, enabling fully self-supervised pre-training.

## 2 Related work

Self-supervised visual representation learning. Masked image modeling [27, 4, 72] and selfdistillation [10, 39, 3] have become dominant paradigms for learning visual features without labels. MAE [27] reconstructs randomly masked patches and learns rich visual features, while DINO [10] and I-JEPA [3] learn representations through self-distillation in pixel or latent space. DINOv2 [39] combines both strategies at scale. These methods operate on single images and therefore do not explicitly capture multi-view geometric relationships. Our work builds on masked image modeling but extends it to image pairs, where the comparison between monocular and binocular reconstruction errors provides a geometric training signal absent from single-image methods.

Pre-training on image pairs. CroCo [61] pioneered cross-view completion for self-supervised pre-training of 3D vision tasks by extending masked image modeling to image pairs. CroCo v2 [62] scaled this framework to large and diverse training data, demonstrating strong transfer to stereo matching and optical flow. P-Match [74] introduces a variant where both images are partially masked to pre-train an image matching model. Other variants include masked appearance transfer for object tracking [69, 50]. Notably, the foundational models DUSt3R [58] and MASt3R [31], and all subsequent works [63, 9, 75, 16, 68, 49, 22], build upon CroCo’s cross-view completion framework and have achieved remarkable success in dense 3D reconstruction and matching. All these methods share a common limitation: the training signal is implicitly monocular for non-co-visible regions. Our method addresses this by explicitly modeling the relative improvement of cross-view completion over monocular reconstruction, providing an additional binocular signal for all masked regions. A parallel line of work extends cross-view completion beyond image pairs: MuM [38] masks arbitrarily many views uniformly and decodes them with inter-frame attention, and Muskie [32] is a native multi-view backbone that learns view-invariant features by finding correspondences across views without 3D supervision. These extensions are orthogonal to our contribution (Gekko improves the training signal available within each pair), and combining the two is a promising direction, though not a straightforward one: with more than two views, the relative improvement might have to be defined against a set of references rather than a single one (Section 5.9).

Co-visibility in image matching and pre-training. Ground-truth co-visibility masks, derived from depth maps and camera poses, are widely used in image matching methods [40, 19, 35, 70, 51, 60, 20, 11, 57, 53, 18, 23, 25, 30, 24, 28, 52, 44] to compute overlaps and select training pairs. Several approaches [44, 20, 21, 54, 18, 24, 6] also leverage these masks to learn matchability scores at test time. Alligat0R [37] recently demonstrates that co-visibility prediction is an effective supervised pre-training objective for relative pose, training on ground-truth co-visibility labels derived from depth maps and camera poses. While effective, this approach requires ground-truth 3D annotations and expensive overlap-based data preprocessing. Our method pursues the same goal in a self-supervised manner: the relative improvement of cross-view completion over masked autoencoding serves as a proxy for co-visibility, eliminating the need for any ground-truth depth or pose information. Combined with a simple stride-based curriculum (Section 5.4), Gekko can be trained directly from raw videos, removing the cumbersome 3D data preprocessing required by both CroCo and Alligat0R, thereby enabling fully self-supervised pre-training.

## 3 Background and motivation

## 3.1 Background on CroCo and MAE

CroCo [61] reconstructs a target image $\ I _ { \mathrm { T } }$ from a masked version of itself and a reference image $I _ { \mathrm { R } }$ using a ViT encoder, a cross-attention decoder, and a pixel decoder (full formalism in Section B). The per-pixel reconstruction error on masked pixels is:

$$
\ell _ { \mathrm { C r o C o } } \left( \mathbf { p } \right) = \left. \mathtt { I } _ { \mathtt { T } } \left( \mathbf { p } \right) - \hat { \mathtt { I } } _ { \mathtt { T } | \mathtt { R } } \left( \mathbf { p } \right) \right. _ { 2 } ^ { 2 } .\tag{1}
$$

In co-visible regions CroCo transfers information from the reference, implicitly learning correspondences [2]; elsewhere the reference adds little and the reconstruction degrades (Figure 4).

A Masked Autoencoder (MAE) [27] reconstructs $\ I _ { \mathrm { T } }$ from a masked version of itself only, with per-pixel error:

$$
\ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) = \left. \mathbb { I } _ { \mathrm { T } } \left( \mathbf { p } \right) - \hat { \mathrm { I } } _ { \mathrm { T } } \left( \mathbf { p } \right) \right. _ { 2 } ^ { 2 } .\tag{2}
$$

MAE cannot exploit the reference view at all: it inpaints from unmasked target patches alone.

## 3.2 Relative improvement of CroCo over MAE

A natural question is whether co-visibility can be predicted from CroCo’s reconstruction error alone. Figure 2 (left) shows histograms of $\ell _ { \mathrm { C r o C o } } ( \mathbf { p } )$ split by ground-truth co-visibility labels. The distributions largely overlap, indicating that CroCo’s error alone is an unreliable predictor.

MAE provides a reconstruction error baseline that does not exploit cross-view cues and is independent of the co-visibility between the reference and target images.

This raises another question: can co-visibility be predicted by comparing CroCo’s and MAE’s reconstruction errors?

We define the relative improvement of CroCo over MAE:

$$
\mathbf { c } \left( \mathbf { p } \right) = \frac { \ell _ { \mathbf { M A E } } \left( \mathbf { p } \right) - \ell _ { \mathbf { C r o C o } } \left( \mathbf { p } \right) } { \ell _ { \mathbf { M A E } } \left( \mathbf { p } \right) } .\tag{3}
$$

In co-visible regions, $\ell _ { \mathrm { C r o C o } } \left( \mathbf { p } \right) \ll \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right)$ and consequently ${ \bf C } \left( { \bf p } \right) \approx 1 $ , because the network has access to the reference view. In non-co-visible regions, both networks perform similarly, leading to ${ \bf C } \left( { \bf p } \right) \approx 0$

Figure 2 (right) reports histograms of $\mathbf { C _ { \lambda } } ( \mathbf { p } )$ computed using ground-truth co-visibility labels. Ideally, co-visible regions would concentrate near 1. The distribution does spread over [0, 1], and the reason is structural: $\mathbf { C } ( \mathbf { p } )$ is a ratio whose denominator is the MAE error, so it is uninformative wherever that error is already small. This happens in uniform regions but equally in repeated or self-similar structure: the target alone already predicts the pixel well, the reference cannot improve on it, and $\mathbf { C } ( \mathbf { p } )$ stays near zero whether or not the pixel is co-visible. C(p) therefore approximates co-visibility rather than classifying it, and $\ell _ { \mathrm { M A E } } ( \mathbf { p } )$ is itself the detector of when it fails, which is what Section 4.3 exploits. Nevertheless, compared to CroCo’s error alone (Figure 2 (left)), the two distributions overlap substantially less, reaching an average precision of 0.74 compared to 0.57 for co-visibility prediction. The relative improvement C(p) is thus a more reliable indicator of co-visibility. In Section 5.6 we show that the channel Gekko actually learns is a substantially better co-visibility classifier still, on scenes never seen during pre-training.

Relative improvement, AP 0.74  
![](images/5d636be28e971a3659917f6bd53d9c2acb5bde33e4c8a5e25d1a8cd019ea22d0.jpg)  
$\ell _ { \mathrm { { C r o C o } } }$ (p)

![](images/895f97e789c499d0272f6edbebbbbbd427288ee506ff24597570fa1a543cc6d9.jpg)  
C (p)  
Figure 2: The relative improvement of CroCo over MAE separates co-visibility. Histograms split by ground-truth co-visibility, computed on the same 100 ScanNet-50 pairs (Section P). (left) CroCo’s reconstruction error $\ell _ { { \mathrm { C r o C o } } } \left( \mathbf { p } \right)$ (Eq. 1) alone. Errors are on average lower in co-visible regions (blue) than in non-co-visible ones (red), but the two distributions largely overlap, giving an average precision of only 0.57. (right) The relative improvement C (p) (Eq. 3). Values concentrate near zero for non-co-visible regions and spread across [0, 1] for co-visible ones, raising the average precision to 0.74.

## 4 Method

The relative improvement $\mathbf { C } ( \mathbf { p } )$ from Section 3.2 provides a pixelwise, albeit noisy, self-supervised co-visibility signal. Predicting co-visibility requires geometric reasoning, making it a useful training objective for 3D vision [37]. However, existing co-visibility objectives rely on ground-truth depth maps and camera poses, limiting their applicability.

We introduce Gekko, a self-supervised pre-training method that uses $\mathbf { C } ( \mathbf { p } )$ as pseudo co-visibility labels to improve CroCo’s pre-training without any 3D annotations (Figure 1).

## 4.1 Pre-training strategy

Gekko is a network $F _ { \mathrm { G e k k o } }$ , trained from scratch, and used in three complementary ways during pre-training.

Given a target image $\ I _ { \mathrm { T } }$ and a reference image I of the same scene from different viewpoints, Gekko performs three forward passes:

1. A cross-view completion pass that reconstructs $\ I _ { \mathrm { T } }$ from a masked version of itself and $\operatorname { I } _ { \mathrm { R } } { \mathrm { : } }$

$$
\hat { \mathrm { I } } _ { \mathrm { T | R } } = F _ { \mathrm { G e k k o } } \left( \mathbb { M } \odot \mathbb { I } _ { \mathrm { T } } , \mathbb { I } _ { \mathrm { R } } \right) ,\tag{4}
$$

where M masks out a random subset (90%) of the target input patches.

2. A masked autoencoder pass that reconstructs $\ I _ { \mathrm { T } }$ from a masked version of itself only:

$$
\hat { \mathrm { I } } _ { \mathrm { T } } = F _ { \mathrm { G e k k o } } \left( \mathbb { M } \odot \mathbb { I } _ { \mathrm { T } } \right) ,\tag{5}
$$

where M is the same mask as the one used for the cross-view completion pass.

3. A relative improvement prediction pass that predicts the relative improvement map C (Eq. 3) of CroCo’s reconstruction error (first pass) over MAE’s reconstruction error (second pass), from $\ I _ { \mathrm { T } }$ (unmasked) and $\operatorname { I } _ { \mathrm { R } } :$

$$
\begin{array} { r } { \hat { \boldsymbol { \mathsf { C } } } = F _ { \mathrm { G e k k o } } \left( \mathbb { I } _ { \mathrm { T } } , \mathbb { I } _ { \mathrm { R } } \right) . } \end{array}\tag{6}
$$

## 4.2 Network architecture

The architecture $F _ { \mathrm { G e k k o } }$ follows CroCo [61]: a ViT-based encoder, a cross-attention based decoder, and a lightweight pixel decoder. In its original version, the pixel decoder outputs three channels corresponding to the RGB values of the reconstructed target image. In Gekko, a fourth channel is added to predict the relative improvement map C<sup>ˆ</sup>. This architecture handles all three passes: in the MAE pass (second pass), since there is no reference image, the cross-attention layers operate as self-attention with the unmasked target patches and dedicated mask tokens, as in classical MAE. The fact that CroCo and Gekko share the same architecture (up to the extra output channel) and the same training data enables fair comparison in Section 5.

## 4.3 Training objective

The network $F _ { \mathrm { G e k k o } }$ is trained from scratch, in a fully self-supervised manner, minimizing:

$$
\mathcal { L } _ { \mathrm { G e k k o } } = \sum _ { \mathbf { p } \in \Omega _ { \mathtt { M } } } \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) + \ell _ { \mathrm { C r o C o } } \left( \mathbf { p } \right) + \ell _ { \mathrm { R I } } \left( \mathbf { p } \right)\tag{7}
$$

where $\Omega _ { \mathtt { M } }$ is the set of masked target pixel locations, $\ell _ { \mathrm { M A E } }$ and $\ell _ { \mathrm { { C r o C o } } }$ are defined in Eq. 2 and Eq. 1 respectively, and the relative improvement loss is:

$$
\ell _ { \mathrm { R I } } \left( \mathbf { p } \right) = \left( s g \left[ \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) - \ell _ { \mathrm { C r o C o } } \left( \mathbf { p } \right) \right] - s g \left[ \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) \right] \hat { \mathbf { C } } \left( \mathbf { p } \right) \right) ^ { 2 } ,\tag{8}
$$

where $s g \left[ \cdot \right]$ denotes the stop-gradient operator. By construction, when $\begin{array} { r } { \hat { \mathbf { c } } \left( \mathbf { p } \right) = \frac { \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) - \ell _ { \mathrm { C r o C o } } \left( \mathbf { p } \right) } { \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) } } \end{array}$ the loss satisfies $\ell _ { \mathrm { R I } } \left( \mathbf { p } \right) = 0$ . Note that Eq. 8 differs slightly from a direct regression of $\mathbf { C } ( \mathbf { p } )$ ; this formulation down-weights pixels where the MAE loss is low (see Section G), which often correspond to uniform regions that yield noisy pseudo-labels. This down-weighting leads to significantly improved performance and sharper co-visibility maps (see Section 5.8 and Figure 3). As in CroCo [61], reference and target images are normalized such that each patch has zero mean and unit variance.

In Figure 3, Gekko’s relative improvement maps ${ \hat { \mathsf { C } } } ,$ while less sharp in uniform regions, are strongly correlated with ground-truth co-visibility labels.

## 5 Experiments

We evaluate Gekko on three complementary tasks: zero-shot correspondence estimation directly after pre-training following [2], relative metric pose estimation after fine-tuning with a lightweight prediction head on top of frozen features, and pointmap regression with a DPT head [41]. All comparisons against CroCo use identical architectures and training data to ensure fairness. The appendix reports a broader set of experiments: generality across architectures (Section L), singleimage probes on depth, semantic segmentation and classification (Section M), full-network fine-tuning on pose and optical flow (Section N), complete pointmap metrics (Section O), data scaling on the raw-video mix (Section K), a per-row ablation analysis (Section F), and additional qualitative results (Section R).

## 5.1 Experimental setup

Pre-training datasets. All models are pre-trained on the indoor scenes ScanNet split of the Cub3 dataset [37], which consists of image pairs extracted from ScanNet [13]. Two overlap variants test whether Gekko benefits from low-overlap pairs, which standard cross-view completion does not: at least 50% overlap (ScanNet-50) and at least 5% (ScanNet-all). In Section 5.4, models are also pre-trained on DL3DV [36] to evaluate generalization to outdoor and diverse scenes. Section 5.6 and Section 5.7 instead use a 12-source mix of raw video, consumed with the stride curriculum and no 3D preprocessing (Section H); those models are written Gekko-B<sup>mix</sup> and Gekko-L<sup>mix</sup>.

Model architectures. Following CroCo [61], all models use a ViT-based architecture [17] in Base and Large sizes (details in Section D). As described in Section 4.2, Gekko adds a single output channel to CroCo’s architecture to predict the relative improvement map C<sup>ˆ</sup>, enabling direct and fair comparison. All pre-training and fine-tuning hyperparameters are given in Section C. Every CroCo result reported in this paper is our own pre-training from scratch under identical architecture, data, augmentation and schedule; no released checkpoint is used, with the single exception of the publicly released backbones probed in Table 7.

## 5.2 Zero-shot correspondence estimation

Features are first evaluated immediately after pre-training, without any fine-tuning, on dense correspondence estimation. Following [2], we evaluate on ETH3D [45] and report the Average End-Point Error (AEPE, ↓) for correspondences read from either the encoder or the decoder, both of which cross-view completion models implicitly learn. Gekko cuts AEPE by 22% for encoder features and 21% for decoder features over the CroCo baseline (Table 1), so the extra training signal improves features for downstream tasks relying on either part of the network.

Table 1: Zero-shot correspondences on ETH3D. AEPE (↓). All models are Base, pre-trained on ScanNet-50.
<table><tr><td>Model</td><td>AEPE (↓)</td></tr><tr><td>Encoder</td><td></td></tr><tr><td>CroCo-B</td><td>78.4</td></tr><tr><td>Gekko-B (ours)</td><td>60.8</td></tr><tr><td>Decoder CroCo-B</td><td></td></tr><tr><td>Gekko-B (ours)</td><td>101.9 80.8</td></tr></table>

Table 2: Curriculum pre-training on ScanNet-1500. Percentage of pairs (↑) within thresholds. “Curriculum” denotes training from raw videos with a stride-based schedule (Section 5.4). Best in bold.
<table><tr><td rowspan="2">Pre-training data</td><td rowspan="2">Model</td><td colspan="3">ScanNet-1500</td></tr><tr><td>10°/0.25m</td><td>10°/0.5m</td><td>10°/1m</td></tr><tr><td colspan="6">ScanNet (curriculum)</td></tr><tr><td rowspan="3">50</td><td>CroCo-B</td><td>20.1</td><td>34.4</td><td>40.2</td></tr><tr><td>Gekko-B</td><td>31.2</td><td>45.0</td><td>49.0</td></tr><tr><td>CroCo-B</td><td>18.9</td><td>35.1</td><td>43.9</td></tr><tr><td rowspan="2">all</td><td>Gekko-B</td><td>39.4</td><td>56.5</td><td>61.1</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td colspan="3">DL3DV (curriculum)</td><td></td><td></td></tr><tr><td rowspan="3">50</td><td>CroCo-B</td><td>8.5</td><td>19.9</td><td>27.2</td></tr><tr><td>Gekko-B</td><td>29.0</td><td>48.4</td><td>54.3</td></tr><tr><td>CroCo-B</td><td>15.6</td><td>29.3</td><td>36.8</td></tr><tr><td rowspan="2">all</td><td>Gekko-B</td><td>34.9</td><td>51.1</td><td>57.1</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

## 5.3 Relative metric pose estimation

We next fine-tune a lightweight MLP head on frozen features for relative metric pose estimation on ScanNet-1500 [44], a task that requires reasoning about both viewpoint and scene geometry. Gekko-B outperforms CroCo-B by nearly 6× at the strictest threshold (Table 3), and the gap widens on ScanNet-all (43.7% vs. 6.6%): Gekko benefits from low-overlap pairs where CroCo does not.

The advantage carries over to Large models: on ScanNet-all, Gekko-L reaches 35.9% at 10<sup>◦</sup>/0.25m against 7.5% for CroCo-L, a 4.8× improvement. Both Large models are pre-trained by us for 100k steps at the same global batch size, so the comparison isolates the training objective at both scales. The benefits of the relative improvement loss thus scale to larger architectures.

Table 3: Relative metric pose estimation on ScanNet-1500. Percentage of pairs (↑) within rotation/translation thresholds. Every row is our own pre-training from scratch; within each of the Base and Large groups the two rows share architecture, pre-training data, global batch size (768) and number of steps (100k), so the only difference is the training objective. Best result per column in bold.
<table><tr><td rowspan="2">Model</td><td colspan="3">ScanNet-50</td><td colspan="3">ScanNet-all</td></tr><tr><td>10°/0.25m</td><td>10°/0.5m</td><td>10°/1m</td><td>10°/0.25m</td><td>10°/0.5m</td><td>10°/1m</td></tr><tr><td>CroCo-B</td><td>5.5</td><td>12.9</td><td>18.5</td><td>6.6</td><td>15.7</td><td>23.0</td></tr><tr><td>Gekko-B (ours)</td><td>29.1</td><td>43.3</td><td>47.5</td><td>43.7</td><td>54.4</td><td>57.6</td></tr><tr><td>CroCo-L Gekko-L (ours)</td><td>一 一</td><td>一 一</td><td>一 一</td><td>7.5 35.9</td><td>16.1 50.6</td><td>23.0 56.3</td></tr></table>

## 5.4 Curriculum pre-training

3D preprocessing [62, 37] is expensive and limits scalability to new datasets. We therefore train directly from raw video with a simple stride-based curriculum.

Curriculum strategy. Pairs start at a stride of 1 between consecutive frames; every N steps the maximum stride $k _ { \mathrm { m a x } }$ grows by ∆k, and each pair draws its stride uniformly from $[ \dot { 1 } , k _ { \mathrm { m a x } } ]$ . Two variants mimic the overlap distributions of the preprocessed datasets: a -50 variant producing mostly high-overlap pairs, and an -all variant that reaches low-overlap pairs earlier. Schedule constants are given in Section C.

Results. Table 2 reports pose estimation on ScanNet-1500. Trained from raw video with no 3D annotation and no overlap-based preprocessing, Gekko matches its curated-data counterpart (31.2% vs. 29.1% at 10<sup>◦</sup>/0.25m; the exact training pairs differ between the two settings). The curriculum helps CroCo considerably on its own (20.1% vs. 5.5%), so progressive stride scheduling is useful independently of our objective. Yet Gekko keeps a clear margin under it, on DL3DV too. Better sampling does not substitute for the relative improvement signal.

## 5.5 Pointmap regression

Following DUSt3R [58], a DPT head on frozen features regresses per-pixel 3D pointmaps in the first image’s frame, evaluated in-domain (ScanNet) and out-of-domain (DL3DV, ETH3D). Gekko outperforms CroCo on all three benchmarks and both pre-training variants (Table 4), reducing Chamfer error by 10% in-domain and by 5–7% out-of-domain, so the relative improvement signal produces features that generalize beyond the fine-tuning distribution.

## 5.6 Does the predicted channel recover co-visibility?

Section 3.2 showed that the analytic relative improvement separates co-visible pixels better than $\ell _ { \mathrm { { C r o C o } } }$ alone. We now test what the network actually predicts (Tables 5 and 6): we run Gekko once per pair with no masking and threshold C<sup>ˆ</sup> as a per-pixel co-visibility classifier, against the same ground truth as Section 3.2. Both models are Base, 200k steps on the mix of Section H; neither benchmark contributes a pre-training scene. Protocol in Section I.

Table 4: Pointmap regression on ScanNet, DL3DV and ETH3D. Chamfer Overall (↓) for models pre-trained on DL3DV with curriculum and fine-tuned on ScanNet-all. Gekko outperforms CroCo on all three benchmarks, including out-of-domain evaluation on DL3DV and ETH3D. Accuracy and Completeness are broken out in Table 12. Best per group in bold.
<table><tr><td>Pre-training</td><td>Model</td><td>ScanNet</td><td>DL3DV</td><td>ETH3D</td></tr><tr><td rowspan="2">DL3DV-50</td><td>CroCo-B</td><td>0.126</td><td>1.430</td><td>0.399</td></tr><tr><td>Gekko-B</td><td>0.113</td><td>1.359</td><td>0.373</td></tr><tr><td rowspan="2">DL3DV-all</td><td>CroCo-B</td><td>0.122</td><td>1.409</td><td>0.398</td></tr><tr><td>Gekko-B</td><td>0.115</td><td>1.366</td><td>0.373</td></tr></table>

Table 6: AP by overlap tertile on ScanNet-1500. Chance differs per tertile.

Table 5: Co-visibility classification from the predicted channel C<sup>ˆ</sup>. One unmasked forward pass, scored against depth-andpose ground truth (↑; b.acc. = balanced accuracy at a single fitted threshold). Both models Base, 200k steps on the mix of Section H. Protocol in Section I.
<table><tr><td rowspan="2">Score</td><td colspan="3">ScanNet-1500 (chance AP 0.502)</td><td colspan="3">7-Scenes (chance AP 0.726)</td></tr><tr><td>AP</td><td>ROC-AUC</td><td>b.acc.</td><td>AP</td><td>ROC-AUC</td><td>b.acc.</td></tr><tr><td> $\ell _ { { \bf C r o C o } } ~ ( { \bf C r o C o - B ^ { \mathrm { m i x } } } )$ </td><td>0.576</td><td>0.577</td><td>0.539</td><td>0.799</td><td>0.603</td><td>0.554</td></tr><tr><td>Č (Gekko-Bmix, ours)</td><td>0.763</td><td>0.737</td><td>0.691</td><td>0.859</td><td>0.702</td><td>0.658</td></tr></table>

<table><tr><td>AP (↑)</td><td>low</td><td>mid</td><td>high</td></tr><tr><td>chance</td><td>0.285</td><td>0.506</td><td>0.714</td></tr><tr><td> $\ell _ { \mathrm { { C r o C o } } }$ </td><td>0.326</td><td>0.570</td><td>0.777</td></tr><tr><td>ê (ours)</td><td>0.609</td><td>0.760</td><td>0.849</td></tr></table>

C<sup>ˆ</sup> reaches 0.763 AP on ScanNet-1500 against 0.576 for $\ell _ { \mathrm { { C r o C o } } }$ (+0.26 over the 0.502 chance level against +0.07), and it separates the classes at one global threshold with 0.691 balanced accuracy (Table 5). On 7-Scenes, where 73% of pixels are co-visible and AP compresses, ROC-AUC is clearer: 0.702 against 0.603. The margin is widest where co-visibility is hardest: on the lowest-overlap tertile C<sup>ˆ</sup> gains +0.32 AP over chance against +0.04 (Table 6). It even beats the pseudo-label it was trained on: recomputing C(p) from the same model under its 90% mask gives only 0.555 AP, since a noisy per-mask estimate is a worse signal than the quantity regressed from it.

## 5.7 Comparison with released cross-view backbones

The comparisons above hold architecture, data and schedule fixed, which isolates the training signal but says nothing about publicly available models. We therefore probe four frozen Large backbones under one protocol, changing only the backbone (Table 7). The self-supervised rows differ in corpu and budget, so this is not a controlled comparison of objectives (Table 3 remains the matched evidence), and VGGT is a fully supervised ∼1B reference whose corpus contains ScanNet and DL3DV. Gekko-L<sup>mix</sup> leads both self-supervised checkpoints on every cross-view benchmark: +13.5 points of pose accuracy at 10<sup>◦</sup>/0.25m over CroCo v2-L, 29% lower Chamfer error in-domain and 20% on ETH3D, and less than half the frozen flow error. The gains remain specific to cross-view geometry (Section M).

## 5.8 Ablation studies

Ablation studies are conducted by pre-training several Base models on ScanNet to analyze the key design choices in Gekko. The different losses are defined in Section G. All models are evaluated on ScanNet-1500 for relative metric pose estimation. Results are summarized in Table 8.

Adding MAE without the relative improvement loss (row 2) barely moves the result, so the gain comes from predicting the relative improvement, not from multi-task training. A DUSt3R-style confidence loss [58] (row 3) does well on high-overlap pairs and collapses on low-overlap ones. The relative formulation beats the absolute one, patch normalization is critical on ScanNet-all, and down-weighting low-MAE pixels (Eq. 8) is worth 18.0 points while yielding sharper maps (Figure 3). Per-row analysis in Section F.

## 5.9 Limitations

Training cost. Gekko is more expensive to pre-train than CroCo. The extra full-resolution pass that predicts C<sup>ˆ</sup> roughly doubles activation memory and so halves the per-device batch (48 vs. 96 images for Base models), which means twice the GPUs for the same global batch of 768. A matched pair of 100k-step Base pre-trainings costs ∼2.4× more GPU-hours for Gekko; our reference run takes ∼25 h

Table 7: Frozen Large backbones under a single probing protocol. Identical head, fine-tuning data and schedule per column; only the frozen backbone changes. Pose: ScanNet-1500, % of pairs (↑) within 10<sup>◦</sup> and the given translation threshold. Pointmap: Chamfer Overall (↓). Flow: MPI-Sintel clean AEPE (↓), frozen, so not comparable to published full fine-tuning figures (Section N). The self-supervised rows are not data-matched: they use the released checkpoints, and Table 3 is the matched comparison. VGGT is afully supervised reference, not a baseline. Details in Section J. Best self-supervised in bold.
<table><tr><td rowspan="2">Frozen backbone</td><td rowspan="2">Sup.</td><td colspan="3">ScanNet-1500 pose (↑)</td><td colspan="3">Pointmap Overall (↓)</td><td>Sintel (↓)</td></tr><tr><td>0.25m</td><td>0.5m</td><td>1m</td><td>ScanNet</td><td>ETH3D</td><td>DL3DV</td><td>AEPE clean</td></tr><tr><td>Gekko-Lmix (ours)</td><td>self</td><td>33.2</td><td>53.3</td><td>60.1</td><td>0.077</td><td>0.317</td><td>1.122</td><td>6.17</td></tr><tr><td>CroCo v2-L [62]</td><td>self</td><td>19.7</td><td>36.2</td><td>45.3</td><td>0.109</td><td>0.394</td><td>1.359</td><td>13.08</td></tr><tr><td>MuM-L [38]</td><td>self</td><td>19.9</td><td>32.9</td><td>39.4</td><td>0.106</td><td>0.377</td><td>1.321</td><td>22.44</td></tr><tr><td>VGGT [56]</td><td>full</td><td>92.3</td><td>98.3</td><td>99.0</td><td>0.030</td><td>0.240</td><td>1.065</td><td></td></tr></table>

Table 8: Ablation studies on ScanNet-1500. Pose estimation accuracy (↑) at three thresholds. All models are Base, pre-trained on ScanNet. Best results in bold. Loss definitions in Section G.
<table><tr><td rowspan="2">Configuration</td><td colspan="3">ScanNet-50</td><td colspan="3">ScanNet-all</td></tr><tr><td>10°/.25m</td><td>10°%.5m</td><td>10°/1m</td><td>10°/.25m</td><td>10°/.5m</td><td>10°/1m</td></tr><tr><td> $\ell _ { { \mathrm { C r o C o } } } \left( { \mathrm { C r o C o } } - { \mathrm { B } } \right)$ </td><td>5.5</td><td>12.9</td><td>18.5</td><td>6.6</td><td>15.7</td><td>23.0</td></tr><tr><td> $\ell _ { \mathrm { C r o C o } } + \ell _ { \mathrm { M A E } }$ </td><td>6.7</td><td>15.3</td><td>20.9</td><td>6.2</td><td>14.5</td><td>21.2</td></tr><tr><td>lconf</td><td>26.7</td><td>38.7</td><td>45.8</td><td>2.2</td><td>6.1</td><td>11.2</td></tr><tr><td> $\ell _ { \mathrm { C r o C o } } + \ell _ { \mathrm { M A E } } + \ell _ { \mathrm { A I } }$ </td><td>14.3</td><td>23.0</td><td>25.2</td><td>23.3</td><td>33.6</td><td>39.4</td></tr><tr><td> $\ell _ { \mathrm { C r o C o } } + \ell _ { \mathrm { M A E } } + \ell _ { \mathrm { A I } } , \mathrm { n o n o r m } .$ </td><td>20.9</td><td>38.1</td><td>42.3</td><td>14.2</td><td>21.9</td><td>25.1</td></tr><tr><td> $\ell _ { \mathrm { { C r o C o } } } + \ell _ { \mathrm { { M A E } } } + \ell _ { \mathrm { { R I r a t i o } } }$ </td><td>27.2</td><td>39.0</td><td>42.7</td><td>25.7</td><td>39.6</td><td>44.4</td></tr><tr><td> $\ell _ { \mathrm { C r o C o } } + \ell _ { \mathrm { M A E } } + \ell _ { \mathrm { R I } } ( \mathrm { G e k k o - B } )$ </td><td>29.1</td><td>43.3</td><td>47.5</td><td>43.7</td><td>54.4</td><td>57.6</td></tr></table>

on 16 H100 GPUs, i.e. ∼400 H100-hours. The cost is confined to pre-training: the extra channel is one output dimension and the MAE branch is discarded afterwards, so fine-tuning and inference cost what CroCo costs.

Where the ratio is uninformative. C(p) is a ratio, and it is unreliable wherever its denominator is small. A low $\ell _ { \mathrm { M A E } } ( \mathbf { p } )$ means the pixel was already predictable from the target alone, in uniform regions but equally in repeated or self-similar structure, so the reference cannot improve on it and C(p) stays near zero regardless of true co-visibility. Eq. 8 makes $\ell _ { \mathrm { M A E } }$ its own detector of this failure by down-weighting exactly those pixels, which Table 8 shows is worth 18.0 points. It is a soft down-weighting, not an exclusion; explicitly masking such regions could help further.

Cross-view, frozen. Two scope limits are worth stating plainly. First, the relative improvement is a binocular signal and improves the tasks that need one: on single-image probes (NYUv2 depth, ADE20K segmentation, ImageNet classification), Gekko and CroCo are within about a point of each other (Section M), so we claim no general-purpose representation improvement. Second, every table here probes a frozen backbone, which measures what pre-training makes directly accessible rather than its value as an initialisation. Under full fine-tuning the advantage narrows unevenly: on 7-Scenes the position advantage survives and the rotation advantage does not, and on MPI-Sintel it does not survive at all against the released CroCo v2, pre-trained on 7.3M curated stereo and flow pairs (Section N). Our claim is accordingly the narrower one: Gekko gives better frozen cross-view features, not necessarily a better initialisation for full fine-tuning.

Data composition, dynamic scenes, and architecture. More data is not automatically better: swapping an indoor-only mix for the full 12-source mix costs Gekko five points on indoor ScanNet-1500 while leaving CroCo unchanged (Section K). We evaluate only static scenes; nothing in the objective assumes rigidity, but we have not tested it. Finally, Gekko inherits CroCo’s restriction to image pairs and a fixed patch size, and a multi-view extension [32, 38] would have to define the relative improvement against a set of references rather than a single one, which we see as an interesting and non-trivial research direction.

![](images/c61e7ca8195b054f3fe7ceda07ca6309fa0149aafcc97b64d190510201e53e28.jpg)  
Figure 3: Qualitative comparison of relative improvement maps C<sup>ˆ</sup> across ablation configurations. The relative improvement loss $\ell _ { \mathrm { R I } }$ (Eq. 8) produces sharper maps that better delineate co-visible regions, while other configurations yield unreliable predictions. The first row is ScanNet, the second 7-Scenes, which no model here saw during pre-training. More examples in Section R.

## 6 Conclusion

Cross-view completion learns strong features for 3D vision in co-visible regions, but where the refer ence adds little the training signal is implicitly monocular. We showed that the relative improvement of the cross-view reconstruction error over an MAE error is a self-supervised proxy for co-visibility needing no ground-truth depth or poses, and that a network trained to predict it recovers co-visibility better than the pseudo-label it was trained on. Building on this, Gekko jointly performs cross-view completion, masked autoencoding and relative improvement prediction, supplying a binocular signal for all masked regions at the cost of one extra output channel. At matched architecture and data it consistently outperforms CroCo on the cross-view tasks we evaluate, and it trains directly from raw video with a simple stride curriculum, removing the 3D preprocessing prior work requires [61, 37].

## Acknowledgments and Disclosure of Funding

This work was supported by the Bosch Research Foundation (Bosch Forschungsstiftung) and by the European Union (ERC Advanced Grant Explorer, Funding ID #101097259). It was granted access to the HPC resources of IDRIS under the allocation 2026-AD010617525R1 made by GENCI.

## References

[1] Adel Ahmadyan, Liangkai Zhang, Artsiom Ablavatski, Jianing Wei, and Matthias Grundmann. Objectron: A large scale dataset of object-centric videos in the wild with pose annotations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7822–7831, 2021.

[2] Honggyu An, Jin Hyeon Kim, Seonghoon Park, Jaewoo Jung, Jisang Han, Sunghwan Hong, and Seungryong Kim. Cross-view completion models are zero-shot correspondence estimators. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1103–1115, 2025.

[3] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a jointembedding predictive architecture. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15619–15629, 2023.

[4] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEit: BERT pre-training of image transformers. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=p-BhZSz59o4.

[5] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitScenes: A diverse real-world dataset for 3D indoor scene understanding using mobile RGB-D data. In NeurIPS Datasets and Benchmarks Track, 2021.

[6] Guillaume Bono, Leonid Antsfeld, Boris Chidlovskii, Philippe Weinzaepfel, and Christian Wolf. End-to-end (instance)-image goal navigation through correspondence as an emergent phenomenon. In International Conference on Learning Representations, 2024.

[7] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In European Conference on Computer Vision, pages 611–625, 2012.

[8] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual KITTI 2. arXiv preprint arXiv:2001.10773, 2020.

[9] Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, and Vincent Leroy. Must3r: Multi-view network for stereo 3d reconstruction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1050–1060, 2025.

[10] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

[11] Hongkai Chen, Zixin Luo, Lei Zhou, Yurun Tian, Mingmin Zhen, Tian Fang, David Mckinnon, Yanghai Tsin, and Long Quan. Aspanformer: Detector-free image matching with adaptive span transformer. In European Conference on Computer Vision, pages 20–36. Springer, 2022.

[12] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020.

[13] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

[14] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Scaling egocentric vision: The EPIC-KITCHENS dataset. In European Conference on Computer Vision, pages 720–736, 2018.

[15] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.

[16] Siyan Dong, Shuzhe Wang, Shaohui Liu, Lulu Cai, Qingnan Fan, Juho Kannala, and Yanchao Yang. Reloc3r: Large-scale training of relative camera pose regression for generalizable, fast, and accurate visual localization. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16739–16752, 2025.

[17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[18] Johan Edstedt, Ioannis Athanasiadis, Mårten Wadenbäck, and Michael Felsberg. Dkm: Dense kernelized feature matching for geometry estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17765–17775, 2023.

[19] Johan Edstedt, Georg Bökman, and Zhenjun Zhao. Dedode v2: Analyzing and improving the dedode keypoint detector. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4245–4253, 2024.

[20] Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, and Michael Felsberg. Roma: Robust dense feature matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19790–19800, 2024.

[21] Johan Edstedt, David Nordström, Yushan Zhang, Georg Bökman, Jonathan Astermark, Viktor Larsson, Anders Heyden, Fredrik Kahl, Mårten Wadenbäck, and Michael Felsberg. Roma v2: Harder better faster denser feature matching. In European Conference on Computer Vision, 2026.

[22] Sven Elflein, Qunjie Zhou, Sérgio Agostinho, and Laura Leal-Taixé. Light3r-sfm: Towards feedforward structure-from-motion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16774–16784, 2025.

[23] Miao Fan, Mingrui Chen, Chen Hu, and Shuchang Zhou. Occ<sup>2</sup>net: Robust image matching based on 3d occupancy estimation for occluded regions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9652–9662, 2023.

[24] Hugo Germain, Vincent Lepetit, and Guillaume Bourmaud. Visual correspondence hallucination. In International Conference on Learning Representations, 2022.

[25] Pierre Gleize, Weiyao Wang, and Matt Feiszli. Silk: Simple learned keypoints. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22499–22508, 2023.

[26] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.

[27] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

[28] Wei Jiang, Eduard Trulls, Jan Hosang, Andrea Tagliasacchi, and Kwang Moo Yi. Cotr: Correspondence transformer for matching across images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6207–6217, 2021.

[29] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.

[30] Shinjeong Kim, Marc Pollefeys, and Daniel Barath. Learning to make keypoints sub-pixel accurate. In European Conference on Computer Vision, pages 413–431. Springer, 2024.

[31] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71–91. Springer, 2024.

[32] Wenyu Li, Sidun Liu, Peng Qiao, Yong Dou, and Tongrui Hu. Muskie: Multi-view masked image modeling for 3d vision pre-training. arXiv preprint arXiv:2511.18115, 2025.

[33] Yiyi Liao, Jun Xie, and Andreas Geiger. Kitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3292–3310, 2022.

[34] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. In International Conference on Learning Representations, 2026.

[35] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching at light speed. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17627–17638, 2023.

[36] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024.

[37] Thibaut Loiseau, Guillaume Bourmaud, and Vincent Lepetit. Alligat0r: Pre-training through covisibility segmentation for relative camera pose regression. In Advances in Neural Information Processing Systems, 2025.

[38] David Nordström, Johan Edstedt, Fredrik Kahl, and Georg Bökman. Mum: Multi-view masked image modeling for 3d vision. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21736–21747, 2026.

[39] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[40] Guilherme Potje, Felipe Cadar, André Araujo, Renato Martins, and Erickson R Nascimento. Xfeat: Accelerated features for lightweight image matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2682–2691, 2024.

[41] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179– 12188, 2021.

[42] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3D: Large-scale learning and evaluation of real-life 3D category reconstruction. In IEEE/CVF International Conference on Computer Vision, pages 10901–10911, 2021.

[43] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In IEEE/CVF International Conference on Computer Vision, pages 10912–10922, 2021.

[44] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4938–4947, 2020.

[45] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3260–3269, 2017.

[46] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in RGB-D images. In IEEE Conference on Computer Vision and Pattern Recognition, pages 2930–2937, 2013.

[47] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from RGBD images. In European Conference on Computer Vision, pages 746–760, 2012.

[48] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, et al. DINOv3. Transactions on Machine Learning Research, 2026.

[49] Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912, 2024.

[50] Zikai Song, Run Luo, Junqing Yu, Yi-Ping Phoebe Chen, and Wei Yang. Compact transformer tracker with correlative masked modeling. Proceedings of the AAAI Conference on Artificial Intelligence, 37(2):2321–2329, 2023.

[51] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8922–8931, 2021.

[52] Dongli Tan, Jiang-Jiang Liu, Xingyu Chen, Chao Chen, Ruixin Zhang, Yunhang Shen, Shouhong Ding, and Rongrong Ji. Eco-tr: Efficient correspondences finding via coarse-to-fine refinement. In European Conference on Computer Vision, pages 317–334. Springer, 2022.

[53] Shitao Tang, Jiahui Zhang, Siyu Zhu, and Ping Tan. Quadtree attention for vision transformers. In International Conference on Learning Representations, 2022.

[54] Prune Truong, Martin Danelljan, Radu Timofte, and Luc Van Gool. Pdc-net+: Enhanced probabilistic dense correspondence network. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8):10247–10266, 2023.

[55] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. In International Conference on 3D Vision, pages 78–89, 2025.

[56] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.

[57] Qing Wang, Jiaming Zhang, Kailun Yang, Kunyu Peng, and Rainer Stiefelhagen. Matchformer: Interleaving attention in transformers for feature matching. In Proceedings of the Asian Conference on Computer Vision, pages 2746–2762, 2022.

[58] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697–20709, 2024.

[59] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. TartanAir: A dataset to push the limits of visual SLAM. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 4909–4916, 2020.

[60] Yifan Wang, Xingyi He, Sida Peng, Dongli Tan, and Xiaowei Zhou. Efficient loftr: Semi-dense local feature matching with sparse-like speed. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21666–21675, 2024.

[61] Philippe Weinzaepfel, Vincent Leroy, Thomas Lucas, Romain Brégier, Yohann Cabon, Vaibhav Arora, Leonid Antsfeld, Boris Chidlovskii, Gabriela Csurka, and Jérôme Revaud. Croco: Self-supervised pre-training for 3d vision tasks by cross-view completion. Advances in Neural Information Processing Systems, 35:3502–3516, 2022.

[62] Philippe Weinzaepfel, Thomas Lucas, Vincent Leroy, Yohann Cabon, Vaibhav Arora, Romain Brégier, Gabriela Csurka, Leonid Antsfeld, Boris Chidlovskii, and Jérôme Revaud. Croco v2: Improved cross-view completion pre-training for stereo matching and optical flow. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17969– 17980, 2023.

[63] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21924–21935, 2025.

[64] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In Advances in Neural Information Processing Systems, volume 37, pages 21875–21911, 2024.

[65] Lihe Yang, Shang-Wen Li, Yang Li, Xinjie Lei, Dong Wang, Abdelrahman Mohamed, Saining Xie, Hengshuang Zhao, Kaiming He, and Hu Xu. In pursuit of pixel supervision for visual pre-training. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 31974–31984, 2026.

[66] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A highfidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023.

[67] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In IEEE Conference on Computer Vision and Pattern Recognition, pages 3712–3722, 2018.

[68] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. In International Conference on Learning Representations, 2025.

[69] Haojie Zhao, Dong Wang, and Huchuan Lu. Representation learning for visual object tracking by masked appearance transfer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18696–18705, 2023.

[70] Xiaoming Zhao, Xingming Wu, Weihai Chen, Peter CY Chen, Qingsong Xu, and Zhengguo Li. Aliked: A lighter keypoint and descriptor extraction network via deformable transformation. IEEE Transactions on Instrumentation and Measurement, 72:1–16, 2023.

[71] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ADE20K dataset. In IEEE Conference on Computer Vision and Pattern Recognition, pages 633–641, 2017.

[72] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. Image BERT pre-training with online tokenizer. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=ydopy-e6Dg.

[73] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. ACM Transactions on Graphics, 37 (4), 2018.

[74] Shengjie Zhu and Xiaoming Liu. Pmatch: Paired masked image modeling for dense geometric matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21909–21918, 2023.

[75] Lojze Zust, Yohann Cabon, Juliette Marrie, Leonid Antsfeld, Boris Chidlovskii, Jerome Revaud, and Gabriela Csurka. Panst3r: Multi-view consistent panoptic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5856–5886, 2025.

## A Broader impact statement

This work contributes a self-supervised pre-training method for 3D vision. The primary applications are in 3D reconstruction, visual localization, and camera pose estimation, which benefit fields such as robotics, autonomous navigation, and augmented reality. As with any method that improves visual understanding, potential dual-use risks include surveillance applications. However, the method operates on pre-training and does not introduce capabilities beyond those already available in existing systems.

## B CroCo and MAE formalism

CroCo. Given a target image $\ I _ { \mathrm { T } }$ and a reference image $I _ { \mathrm { R } }$ of the same scene from different viewpoints, a Cross-view Completion network $F _ { \mathrm { C r o C o } }$ reconstructs the target image $\ I _ { \mathrm { T } }$ from a masked version of itself and the reference image $\tt { I } _ { R } \colon$

$$
\hat { \mathrm { I } } _ { \mathrm { T | R } } = F _ { \mathrm { C r o C o } } \left( \mathtt { M } \odot \mathtt { I } _ { \mathrm { T } } , \mathtt { I } _ { \mathrm { R } } \right) ,\tag{9}
$$

where M masks out a random subset (90% in practice) of the target input patches. A ViT encoder first encodes the reference patches and the unmasked target patches separately. A cross-attention based decoder then processes reference and target tokens to warp information from the reference into the target representation. A lightweight pixel decoder reconstructs the target image $\hat { \mathbb { I } } _ { \mathrm { T | R } }$ . CroCo is trained to minimize the total reconstruction error on masked target pixels: $\begin{array} { r } { \mathcal { L } _ { \mathrm { C r o C o } } = \sum _ { \mathbf { p } \in \Omega _ { \mathtt { M } } } \ell _ { \mathrm { C r o C o } } \left( \mathbf { p } \right) } \end{array}$ where $\Omega _ { \mathtt { M } }$ is the set of masked target pixel locations and $\ell _ { \mathrm { { C r o C o } } }$ is defined in Eq. 1.

MAE. A Masked Autoencoder $F _ { \mathrm { M A E } }$ reconstructs $\ I _ { \mathrm { T } }$ from a masked version of itself only:

$$
\hat { \mathrm { I } } _ { \mathrm { T } } = F _ { \mathrm { M A E } } \left( \mathrm { M } \odot \mathrm { I } _ { \mathrm { T } } \right) ,\tag{10}
$$

where M masks out a random subset (75% in monocular MAE, we perform 90% to get the same mask as CroCo and be able to compute the loss and the relative improvement) of the target input patches. MAE is trained to minimize $\begin{array} { r } { \bar { \mathcal { L } } _ { \mathrm { M A E } } = \sum _ { \mathbf { p } \in \Omega _ { \mathrm { M } } } \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) } \end{array}$ , where $\ell _ { \mathrm { M A E } }$ is defined in Eq. 2.

## C Training details

Stride curriculum. The two curriculum variants of Section 5.4 share $N { = } 5 \mathbf { k }$ steps between increments and differ only in the increment itself: $\Delta k { = } 2$ for the -50 variant and $\Delta k { = } 1 0$ for the -all variant. All other hyperparameters are identical to the preprocessed-data runs. The schedule is fixed rather than adaptive so that the comparison against preprocessed baselines is controlled; an adaptive per-sequence rule (for instance increasing the stride on plateau) would make the approach fully automatic.

Pre-training. All models are trained from scratch with a batch size of 768 and a learning rate of $1 . 5 \times 1 0 ^ { - 4 }$ for 100k steps. A cosine learning rate decay with a linear warmup of 5000 steps is used. Following CroCo [61], masking ratio is 90% and patches are normalized to zero mean and unit variance. Images are resized to a maximum dimension of 512 pixels while maintaining aspect ratio.

Fine-tuning for relative metric pose estimation. A simple MLP head is added on top of the frozen pre-trained decoder features, trained with a homoscedastic loss [29] for 20k steps with a batch size of 768 and a constant learning rate of $1 \times 1 0 ^ { - 4 }$ . Fine-tuning uses the all overlap variant to leverage the full range of viewpoint changes.

Fine-tuning for pointmap regression. A DPT head [41] is fine-tuned on top of frozen pre-trained features using the confidence-aware regression loss from DUSt3R [58]. Fine-tuning uses the ScanNetall dataset with the same batch size and learning rate as pose estimation.

## D Architecture details

Following CroCo [61], all models use a Vision Transformer (ViT) architecture [17] in two sizes:

• Base: 12-layer ViT encoder and 8-layer transformer decoder with embedding sizes of 768 and 512, pre-trained on both variants of ScanNet.

• Large: 24-layer ViT encoder and 12-layer transformer decoder with embedding sizes of 1024 and 768, pre-trained on ScanNet-all only.

## E CroCo qualitative reconstructions

![](images/53f44942bea2e35857f9c1b98f2c42d50c6d3b97f605ae1abd1ae20f9e7bd925.jpg)  
Figure 4: Qualitative CroCo reconstructions. Reconstructions are sharper in co-visible regions where the network warps information from the reference image. Conversely, reconstructions are blurred in non-co-visible regions where the reference provides little information (best viewed zoomedin).

## F Detailed ablation analysis

This section provides a detailed discussion of each ablation row in Table 8.

Impact of the relative improvement loss. Adding the MAE reconstruction branch without the relative improvement loss $\ell _ { \mathrm { R I } }$ (row 2) provides only marginal improvement over the CroCo baseline (20.9% vs. 18.5% at 10<sup>◦</sup>/1m). The key contribution of Gekko lies in the relative improvement prediction, not merely in jointly training with MAE.

Confidence prediction vs. relative improvement. The confidence-aware loss performs well on high-overlap pairs (45.8% at 10<sup>◦</sup>/1m on ScanNet-50) but fails on low-overlap pairs (11.2% on ScanNet-all). As illustrated in Figure 3, the confidence maps lack the sharp co-visibility boundaries that $\ell _ { \mathrm { R I } }$ achieves.

Relative vs. absolute improvement. Regressing the absolute improvement $\ell _ { \mathrm { M A E } } - \ell _ { \mathrm { C r o C o } }$ underperforms the relative formulation, particularly on ScanNet-all (39.4% vs. 57.6%). The relative improvement produces sharper maps that better capture co-visibility boundaries (Figure 3).

Impact of patch normalization. Without normalizing target patch tokens, performance drops sig nificantly on ScanNet-all (25.1% vs. 39.4%). Removing normalization leads to noisier predictions in uniform regions where MAE predicts a near-constant value, yielding unreliable relative improvement estimates.

Loss formulation details. Directly regressing the ratio underperforms the final formulation (Eq. 8) that multiplies the prediction by the denominator, improving from 25.7% to 43.7% at 10<sup>◦</sup>/.25m on ScanNet-all.

## G Ablation study: technical details

The different losses considered in the ablation study (Table 8) are defined as follows:

1.

$$
\ell _ { \mathrm { R I } } \left( \mathbf { p } \right) = \left( s g \left[ \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) - \ell _ { \mathrm { C r o C o } } \left( \mathbf { p } \right) \right] - s g \left[ \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) \right] \hat { \mathbf { C } } \left( \mathbf { p } \right) \right) ^ { 2 }\tag{11}
$$

is the relative-improvement loss of Gekko (see Eq. 3).

2.

$$
\ell _ { \mathrm { R I \ r a t i o } } \left( \mathbf { p } \right) = \left( s g \left[ \frac { \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) - \ell _ { \mathrm { C r o C o } } \left( \mathbf { p } \right) } { \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) } \right] - \hat { \mathbf { C } } \left( \mathbf { p } \right) \right) ^ { 2 }\tag{12}
$$

is a relative-improvement loss that reweights the previous loss as follows: $\begin{array} { r } { \left( \frac { 1 } { s g \left[ \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) \right] } \right) ^ { 2 } \ell _ { \mathrm { R I } } \left( \mathbf { p } \right) = \ell _ { \mathrm { R I r a t i o } } \left( \mathbf { p } \right) } \end{array}$ . This reweighting up-weights pixels where the MAE loss is low, which often correspond to uniform regions that yield noisy pseudo-labels. In practice, $\ell _ { \mathrm { R I } }$ significantly outperforms $\ell _ { \mathrm { { R I } \ r a t i o } }$

3.

$$
\ell _ { \mathrm { A I } } \left( \mathbf { p } \right) = \left( s g \left[ \ell _ { \mathrm { M A E } } \left( \mathbf { p } \right) - \ell _ { \mathrm { C r o C o } } \left( \mathbf { p } \right) \right] - \hat { \mathbf { C } } \left( \mathbf { p } \right) \right) ^ { 2 }\tag{13}
$$

is an absolute-improvement loss where $\hat { \mathbf { C } } \left( \mathbf { p } \right)$ learns to predict the difference between $\ell _ { \mathrm { { C r o C o } } }$ and $\ell _ { \mathrm { M A E } }$

4.

$$
\ell _ { \mathrm { c o n f } } \left( \mathbf { p } \right) = \left( 1 + \exp \left( \hat { \mathbf { C } } \left( \mathbf { p } \right) \right) \right) \left\| \mathbb { I } _ { \mathbf { T } } \left( \mathbf { p } \right) - \hat { \mathbb { I } } _ { \mathbf { T } \left| \mathbf { R } \right. } \left( \mathbf { p } \right) \right\| _ { 2 } ^ { 2 } - 0 . 2 \ln \left( 1 + \exp \left( \hat { \mathbf { C } } \left( \mathbf { p } \right) \right) \right)\tag{14}
$$

is a confidence-aware loss where ${ \hat { \mathbf { C } } } \left( \mathbf { p } \right)$ learns to predict the precision $( i . e .$ inverse variance) that is expected to be high in co-visible regions and low in non-co-visible regions.

## H The 12-source raw-video mix

Two experiments (Section 5.6 and Section 5.7) use a mix of twelve video sources rather than the curated Cub3 pairs: ScanNet [13], ScanNet++ [66], ARKitScenes [5], RealEstate10K [73] and EPIC-KITCHENS [14] (real indoor); Hypersim [43] (synthetic indoor); CO3Dv2 [42] and Objectron [1] (object-centric); DL3DV [36] and KITTI-360 [33] (real outdoor); and VirtualKITTI2 [8] and TartanAir [59] (synthetic outdoor).

Every source is consumed as raw video through the stride curriculum of Section 5.4: pairs are formed by sampling two frames of the same clip at a stride drawn from the current curriculum window. No structure-from-motion, depth, camera pose or overlap estimate is used at any point, so adding a source costs only the frames themselves. Sources are sampled with fixed weights, and validation uses a scene-disjoint held-out split of each source. Unless stated otherwise, mix models are trained with the same optimiser, resolution and global batch size as the ScanNet models of Section C.

## I Co-visibility probing protocol

For Table 5 we run each model once per pair with no masking and read the fourth output channel C<sup>ˆ</sup> directly; there is no post-processing, no test-time augmentation and no fitting of the network. Ground truth is computed on the fly: a target pixel is labelled co-visible if its back-projected 3D point reprojects inside the reference image and the two depths agree within 5%. Pixels whose reprojection falls on a hole in the reference depth map are excluded rather than counted as negatives, which would otherwise reward a model for predicting low co-visibility on missing data.

We evaluate on all 1500 pairs of ScanNet-1500 and on 558 retrieval pairs of 7-Scenes [46]. Neither benchmark contributes a scene to the mix of Section H. Average precision and ROC-AUC are threshold-free. For balanced accuracy we fit a single one-dimensional logistic regression per dataset and per score over all evaluated pixels and threshold it at $p = 0 . 5 ;$ the question is whether one global threshold separates the classes at all, so the threshold is fitted on the same pixels it is reported on. The CroCo baseline has no fourth channel, so it is scored by $- \ell _ { \mathbf { C r o C o } } ( \mathbf { p } )$ from a masked forward pass.

The tertile analysis of Table 6 sorts the 1500 pairs by their true co-visible fraction and splits them into three equal groups; chance average precision is the positive rate within each group and therefore differs per tertile.

Two remarks. First, the pseudo-label figure quoted in Section 5.6 (0.555 AP) is $\mathbf { C } ( \mathbf { p } )$ recomputed from the same trained model under its 90% training mask, averaged over mask draws. It is not directly comparable to the 0.74 of Figure 2, which is measured on a different set of 100 ScanNet-50 pairs using separately pre-trained CroCo and MAE models (Section P). Second, restricting the evaluation to SIFT keypoints raises every score, and raises the raw signals more than the prediction $( \hat { \mathbf { C } } 0 . 7 6 3  0 . 7 9 0 , \ell _ { \mathrm { C r o C o } } 0 . 5 7 6  0 . 5 8 \bar { 7 } )$ : the textureless-region limitation of Section 5.9 showing up exactly where it should.

## J Probing protocol for Table 7

All rows of Table 7 share the head architecture, the fine-tuning data and the step budget of the corresponding main-paper experiment; only the frozen backbone is swapped. Parameter counts differ substantially: ∼390M for Gekko-L and MuM-L, ∼420M for CroCo v2-L, and ∼1B for VGGT. Pose uses the MLP head of Section C on frozen decoder features; pointmap uses the DPT head; optical flow uses a DPT flow head with the tiled inference of CroCo-Flow at 0.9 overlap on the MPI-Sintel [7] subval split.

Two caveats. The self-supervised rows are not data-matched: CroCo v2 is pre-trained on 7.3M curated stereo and flow pairs and MuM on a corpus that includes ImageNet-1K, whereas Gekko-L<sup>mix</sup> sees only the raw video of Section H. And MuM’s frozen rows use a learning rate of $1 0 ^ { - 4 } .$ , since the rate used for the other rows does not train its head at all. VGGT is reported for scale only: it is trained with explicit camera and depth supervision, and both ScanNet and DL3DV are in its training corpus, so three of its four columns are in-distribution.

## K Data scaling on the raw-video mix

To test whether the relative improvement signal keeps paying off as data grows, we pre-train matched pairs of Base models on nested, seeded, scene-level subsets of the mix of Section H, from 1% to 100% of its scenes. Sources, sampling weights, resolution, global batch size (768) and step count (100k) are identical across all rows, as is the fine-tuning budget of every head.

Gekko leads at every scale, by between +7.4 and +24.1 points at the strictest pose threshold. Two honest caveats. Above 50% both methods dip on pose at a fixed 100k-step budget, which indicates the backbones are undertrained rather than that data hurts; extending the best mix model to 200k steps recovers $3 9 . 4 / 5 6 . 1 / 6 1 . 1$ and a pointmap Overall of 0.092. And composition matters more than size on an indoor benchmark: an indoor-only mix gives Gekko 39.4/59.2/65.5 against 34.4/51.5/56.5 for the full mix, while CroCo is essentially flat (13.7/29.9/38.6 against $1 5 . 1 / 3 0 . 5 / 3 9 . 2 )$ . Adding outdoor and object-centric video buys generality at a measurable in-domain cost.

Table 9: Data scaling. ScanNet-1500 pose (↑, at 10<sup>◦</sup>/0.25 0.5 1m) and ScanNet pointmap Chamfer Overall (↓) as a function of the fraction of mix scenes used for pre-training. Best per column in bold.
<table><tr><td rowspan="2">Pre-training data</td><td colspan="2">Pose (↑)</td><td colspan="2">Pointmap (↓)</td></tr><tr><td>CroCo-B</td><td>Gekko-B</td><td>CroCo-B</td><td>Gekko-B</td></tr><tr><td>1%</td><td>7.3 / 15.7 / 20.7</td><td>15.1 / 24.7 / 29.0</td><td>0.144</td><td>0.140</td></tr><tr><td>5%</td><td>17.7 / 32.9 / 39.3</td><td>25.1 / 40.8 / 46.3</td><td>0.126</td><td>0.121</td></tr><tr><td>10%</td><td>15.2 / 26.6 / 33.4</td><td>37.0 / 54.1 / 58.7</td><td>0.118</td><td>0.110</td></tr><tr><td>25%</td><td>17.4 / 33.7 / 42.0</td><td>41.5 / 57.5 / 63.2</td><td>0.108</td><td>0.103</td></tr><tr><td>50%</td><td>22.5 / 39.3 / 46.9</td><td>37.9 / 56.5 / 61.4</td><td>0.104</td><td>0.101</td></tr><tr><td>100%</td><td>15.1 / 30.5 / 39.2</td><td>34.4 / 51.5 / 56.5</td><td>0.102</td><td>0.096</td></tr></table>

## L Generality across architectures

The gain could conceivably be an artefact of CroCo’s specific encoder–decoder. We therefore pretrained six Base models from scratch on ScanNet-50 for 100k steps at global batch 768, varying the architecture and keeping everything else fixed.

Table 10: The advantage survives substituting either half of the architecture. ScanNet-1500 pose (↑) and ScanNet pointmap Chamfer Overall (↓). All models Base, ScanNet-50, 100k steps, matched head budgets.
<table><tr><td rowspan="2">Architecture</td><td colspan="2">Pose (↑)</td><td colspan="2">Pointmap (↓)</td></tr><tr><td>CroCo</td><td>Gekko</td><td>CroCo</td><td>Gekko</td></tr><tr><td>CroCo encoder + cross-attention decoder (main paper)</td><td>5.5 / 12.9 / 18.5</td><td>29.1 / 43.3 / 47.5</td><td>0.125</td><td>0.100</td></tr><tr><td>CroCo encoder + joint self-attention decoder</td><td>5.7 / 9.9 / 13.9</td><td>22.9 / 34.2 / 38.3</td><td>0.207</td><td>0.105</td></tr><tr><td>Frozen DINOv3 [48] encoder + cross-attention decoder</td><td>5.0 / 10.3 / 14.7</td><td>23.9 / 36.8 / 41.3</td><td>0.125</td><td>0.096</td></tr></table>

The joint self-attention decoder concatenates the two views’ tokens with a learned per-view embedding and uses no cross-attention anywhere; the DINOv3 variant freezes an off-the-shelf monocular encoder and trains only the decoder. The advantage survives both substitutions (+17.2 and +18.9 points against +23.6 for the paper’s architecture). It is also worth noting that the CroCo baseline barely moves across the three architectures (5.0 to 5.7), while Gekko changes substantially: the training signal, not the architecture, is what varies. As a control, a frozen DINOv3 encoder with no cross-view pre-training at all reaches 6.9/14.5/21.2 on pose and 0.142 on pointmap.

## M Single-image probes

The relative improvement is a binocular signal, so we do not expect it to help tasks that see one image. It does not, and it does not hurt either. Under the frozen protocol of the main paper with a DPT head, monocular depth on NYUv2 [47] (REL ↓ / δ<sub>1</sub> ↑) and semantic segmentation on ADE20K [71] (mIoU ↑ / pixel accuracy ↑) give:

Table 11: Single-image probes are a tie. Frozen backbone, DPT head, matched schedules. All models Base.
<table><tr><td rowspan="2">Pre-training</td><td colspan="2">NYUv2 depth</td><td colspan="2">ADE20K segmentation</td></tr><tr><td>CroCo-B</td><td>Gekko-B</td><td>CroCo-B</td><td>Gekko-B</td></tr><tr><td>ScanNet-50</td><td>0.181 / 73.5</td><td>0.191 / 72.3</td><td>17.6 / 71.2</td><td>17.6 / 71.5</td></tr><tr><td>ScanNet-all</td><td>0.200 / 70.9</td><td>0.200 / 70.5</td><td>17.6 / 71.3</td><td>16.8 / 70.9</td></tr><tr><td>12-source mix, 200k</td><td>0.147 / 80.8</td><td>0.143 / 82.2</td><td>20.8 / 73.6</td><td>20.2 / 73.8</td></tr></table>

The same holds for ImageNet-1K [15] classification with an attentive probe (50 epochs, batch 2048): 61.3 top-1 for Gekko-B<sup>mix</sup> against 61.0 for its matched CroCo-B control, a tie. On the eight dense Taskonomy [67] tasks under the standard 1k-image transfer protocol, Gekko-B<sup>mix</sup> wins three columns (2D texture edges 0.0029, occlusion edges 0.0006, principal curvature 0.0441) and loses the other five to the much larger released CroCo v2-L and MuM-L. We conclude that the relative improvement objective buys cross-view geometry and nothing else, and we make no claim beyond that.

## N Full-network fine-tuning

Every table in the main paper probes a frozen backbone. Fine-tuning the whole network measures a different property, its value as an initialisation, and it ranks models differently. We report it here because the distinction matters for how our claim should be read.

Optical flow. On MPI-Sintel, training the whole network at CroCo-Flow’s own sample budget (110k steps at effective batch 64, i.e. 7.0M of their 7.2M sample-passes) gives AEPE clean/final of 2.02/2.72 for Gekko-B<sup>mix</sup>, 2.21/2.59 for its matched CroCo-B control, 2.59/3.20 for the released MuM-L, and 1.48/2.15 for the released CroCo v2-L. The matched Base pair is a tie, and CroCo v2-L is clearly ahead of everything. This is consistent with the corpus rather than the objective driving this benchmark: CroCo v2 is pre-trained on 7.3M curated stereo and flow pairs, and the architecturematched figure it publishes is 1.76/2.30. It is also a validation of our pipeline: the released CroCo v2 checkpoint reaches 1.48 in our hands against 1.43 published, a 3% gap explained by four documented deviations (110k of 112.5k steps, effective batch 64 rather than 8 with the learning rate scaled linearly, bf16 rather than fp32, and the last checkpoint rather than the best on validation).

Relative pose on 7-Scenes. Fine-tuning the whole network at 224 × 224 for 6k steps gives median position/orientation errors of 6.16 cm/1.90<sup>◦</sup> for Gekko-B<sup>mix</sup> against $7 . 7 0 \mathrm { c m } / 2 . 0 8 ^ { \circ }$ for its matched CroCo-B control, i.e. 20% better on position and 9% on orientation. The released ViT-L models are ahead of both Base models (5.77 cm for CroCo v2-L, 5.29 cm for MuM-L) and we do not claim a win over them. Compared to the frozen numbers (23.2 cm for Gekko-B<sup>mix</sup> against 31.9 cm for CroCo-B), the position advantage survives fine-tuning while the rotation advantage largely does not.

Our pipeline reproduces CroCo v1’s published 5.0 cm on this benchmark to within 0.1 cm using the released CroCo v2 checkpoint, despite our own optimiser settings and a DINOv3 global descriptor in place of the original retriever. The retrieved pairs are computed once and are byte-identical across all rows.

## O Pointmap regression: full metrics

Table 12: Pointmap regression, full metrics. Accuracy, Completeness, and Overall metrics (↓) for models pre-trained on DL3DV with curriculum and fine-tuned on ScanNet-all. Gekko consistently outperforms CroCo across all benchmarks, including out-of-domain evaluation on DL3DV and ETH3D. Best result per group in bold.
<table><tr><td rowspan="2">Pre-training</td><td rowspan="2">Model</td><td colspan="3">ScanNet</td><td colspan="3">DL3DV</td><td colspan="3">ETH3D</td></tr><tr><td>Acc.</td><td>Comp.</td><td>Over.</td><td>Acc.</td><td>Comp.</td><td>Over.</td><td>Acc.</td><td>Comp.</td><td>Over.</td></tr><tr><td>DL3DV-50</td><td>CroCo-B Gekko-B</td><td>0.128 0.118</td><td>0.125 0.108</td><td>0.126 0.113</td><td>1.278 1.240</td><td>1.583 1.477</td><td>1.430 1.359</td><td>0.400 0.379</td><td>0.399 0.367</td><td>0.399 0.373</td></tr><tr><td>DL3DV-all</td><td>CroCo-B Gekko-B</td><td>0.123 0.119</td><td>0.121 0.110</td><td>0.122 0.115</td><td>1.268 1.230</td><td>1.549 1.503</td><td>1.409 1.366</td><td>0.399 0.376</td><td>0.396 0.369</td><td>0.398 0.373</td></tr></table>

## P Details on the computation of the histograms in Figure 2

The histograms in Figure 2 are computed on 100 image pairs from the ScanNet-50 set. For each pair, ground-truth co-visibility labels from Cub3 [37] are used: a pixel is considered co-visible if its 3D point projects within the image boundaries of the other view and the depth is consistent (within 5% tolerance). Both a pre-trained CroCo model and a pre-trained MAE model are run on each pair to obtain per-pixel reconstruction errors $\ell _ { \mathrm { C r o C o } } ( \mathbf { p } )$ and $\ell _ { \mathrm { M A E } } ( \mathbf { p } )$ , and the relative improvement $\dot { \mathbf { C } } ( \mathbf { p } )$ is computed according to Eq. 3.

## Q PCA visualization of encoder features

To qualitatively assess the learned representations, Figures 5 and 6 visualize the first three principal components of the encoder output tokens, mapped to RGB channels, for both CroCo and Gekko. The PCA is computed jointly over both views of each pair so that colors are directly comparable across images. Consistent colors in co-visible regions indicate that the encoder maps corresponding 3D points to similar feature vectors, a desirable property for downstream geometric tasks. Both models are the matched Base pair pre-trained on ScanNet-50. Figure 5 shows ScanNet pairs; Figure 6 adds pairs from DL3DV, Cambridge Landmarks, ETH3D and 7-Scenes, none of which either model saw during pre-training.

Image 1  
Image 2  
![](images/9e94045f0d7d13b0f0c028dfb38690648e16cdbeaada92e398623edfced9d31b.jpg)  
Figure 5: PCA visualization of encoder features for CroCo and Gekko. For each image pair, the three principal components of the encoder output tokens are mapped to RGB channels.

## R Additional qualitative results

![](images/d577d6dcc7207d33dcd372867bea4cb7f6cbd0fe441e0d4466d0e63e815e8ec6.jpg)  
Figure 6: PCA visualization of encoder features on other domains. Same visualization as Figure 5. The first five rows are pairs from DL3DV (two outdoor scenes), Cambridge Landmarks, ETH3D and 7-Scenes; the remaining rows are ScanNet. Both models here are pre-trained on ScanNet alone, so the first five rows are out of domain for each of them, yet Gekko still assigns matching colors to corresponding surfaces across the viewpoint change.

![](images/87b6fd1ec0518fed37fbf4f1acb942617e67e278919c3abb58d7af1fdeb98b87.jpg)  
Figure 7: Qualitative comparison of relative improvement maps $\hat { \mathsf { C } }$ across ablation configurations. The relative improvement loss $\ell _ { \mathrm { R I } }$ (Eq. 8) produces sharper maps that better delineate co-visible regions, while other configurations yield unreliable predictions.