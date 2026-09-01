# RealCAD: Towards Real-World Image-to-CAD Reconstruction under Domain Shift and Parameter Bias

Yihe Sun<sup>1</sup>, Ziyu Lu<sup>1</sup>, Kaihua Tang<sup>1∗</sup>, Xian-Sheng Hua<sup>1∗</sup>

<sup>1</sup>Tongji University

## Abstract

Reconstructing editable Computer-Aided Design (CAD) models from images is essential for downstream modification, manufacturing, and design reuse. However, existing imageto-CAD methods are developed predominantly on synthetic renderings and face two coupled obstacles: a substantial appearance domain gap between synthetic and real images, and a previously overlooked parameter bias in widely used CAD data. We show that the local normalization adopted by Deep-CAD concentrates several geometric parameters around a few discrete values while encoding substantial information in a single scale factor. Consequently, a model can achieve deceptively high parameter accuracy by exploiting these frequent values rather than inferring geometry from the input image. In this paper, we propose RealCAD, a unified framework that addresses these limitations at the representation, image, and feature levels. At the representation level, we redistribute scale information to the corresponding geometric parameters, producing less concentrated parameter distributions in a shared scale space. At the image level, geometry-constrained translation converts synthetic renderings toward the real-image domain while conditioning on object contours. At the feature level, a multi-positive contrastive objective aligns representations of the same CAD model across viewpoints and image domains, enabling CAD sequence prediction from each individual view. We further introduce OpenRealCAD, comprising four-view photographs of 392 3D-printed objects paired with ground-truth command sequences. Experiments show that the revised representation substantially reduces the accuracy attainable from parameter-frequency priors, making parameter accuracy a more reliable measure of image-conditioned geometric inference. RealCAD further improves real-domain command and parameter accuracy, while retaining competitive synthetic-domain performance. The code<sup>1</sup> and dataset<sup>2</sup> are publicly available.

## Introduction

Image-to-CAD reconstruction aims to recover editable computer-aided design (CAD) models from images, enabling downstream modification, manufacturing, and design reuse. Unlike surface-based representations such as meshes (Groueix et al. 2018; Nash et al. 2020; Wang et al. 2018) or point-based representations (Achlioptas et al. 2018;

![](images/283155bbbbc927dec390a5cd2f1fc67357184e3e132ed84fa9097dc6e5ce3296.jpg)  
(a) Domain Shift

(b) Parameter Bias  
![](images/eea7b68722df11994ccf637c1fd186e08873520c1648ef2d6cdb69c28cc9713c.jpg)  
Figure 1: Key challenges in image-to-CAD reconstruction. (a) Domain shift: a model trained on synthetic renderings reconstructs synthetic inputs accurately but degrades substantially on real-world images (e.g., 3D printed CAD models). (b) Parameter bias: the DeepCAD representation produces highly concentrated geometric-parameter distributions, encouraging reliance on dataset-specific shortcuts.

Cai et al. 2020; Mo et al. 2019; Yang et al. 2018), CAD models explicitly encode parametric construction histories as sequences of modeling operations. Recent studies have explored reconstructing parametric CAD models from text, point clouds, and images. However, textual descriptions are often insuficient to specify fine-grained geometry and spatial relationships, while point-cloud-based methods typically require dense and relatively complete 3D observations. Images, by contrast, are easier to acquire and therefore provide a more practical input modality for CAD reconstruction.

However, a key challenge to practical deployment is the simulation-to-reality (Sim2Real) gap (Ju et al. 2022): models trained on synthetically rendered images often exhibit substantial performance degradation when applied to realworld photographs, as illustrated in Figure 1(a). Nevertheless, systematic Sim2Real adaptation remains underexplored in image-to-CAD reconstruction. CADCrafter (Chen et al. 2025) provides preliminary evidence that models trained on synthetic images can generalize to captured photographs. Yet, its real-image evaluation set is not publicly available, and the model is trained primarily on synthetic renderings without an explicit mechanism for Sim2Real adaptation. Beyond the visual domain gap, the underlying CAD representation itself may also hinder reliable geometric inference. We identify a previously overlooked limitation in the normalization scheme of DeepCAD (Wu, Xiao, and Zheng 2021), a widely used benchmark and representation in sequence-based CAD modeling. Specifically, DeepCAD’s local normalization assigns a substantial portion of geometric scale information to a separate scale factor S, resulting in highly concentrated distributions for several normalized geometric parameters, as illustrated in Figure 1(b). This imbalance may encourage models to exploit dataset-specific statistical shortcuts rather than infer the underlying geometry from visual evidence. These observations suggest that reliable robust image-to-CAD reconstruction requires addressing both visual domain discrepancy and representation-induced parameter bias.

Collecting large-scale real-image datasets with paired CAD command-sequence annotations is expensive and dificult to scale, whereas synthetic images can be automatically rendered from existing CAD models with accurate annotations and controllable acquisition conditions. A practical strategy is therefore to leverage abundant synthetic data together with limited real data. To address the representation bias and domain gaps, we revise the DeepCAD parameter representation, construct the real-image benchmark Open-RealCAD, and propose the Sim2Real reconstruction framework RealCAD. Specifically, we redistribute the scale information encoded by DeepCAD’s local normalization factor S to the corresponding geometric parameters, thereby alleviating parameter concentration and reducing reliance on dataset-specific statistical shortcuts. We further construct OpenRealCAD, comprising real images of 392 3D-printed physical CAD objects captured from four fixed viewpoints under controlled conditions.

Building on the revised DeepCAD representation and OpenRealCAD, RealCAD leverages large-scale synthetic data together with limited real images to mitigate the synthetic-to-real gap at both the image and feature levels. At the image level, geometry-constrained synthetic-to-real translation adapts synthetic renderings toward the real domain while preserving geometric structure. At the feature level, multi-positive contrastive learning aligns representations of the same CAD model across viewpoints and image domains. The resulting representations are decoded into CAD command sequences and their associated parameters. Experiments show that RealCAD substantially improves reconstruction performance on real images while maintaining competitive performance in the synthetic domain.

The main contributions are summarized as follows:

• We identify an overlooked distributional limitation in the DeepCAD command-sequence representation and revise its parameterization by redistributing the scale information encoded by S to the corresponding geometric parameters, thereby mitigating parameter concentration and frequency-based shortcut learning.

• We construct OpenRealCAD, a real-image benchmark comprising 392 CAD models, four-view images of their 3D-printed physical objects, and paired ground-truth command sequences, enabling quantitative and reproducible evaluation of Sim2Real reconstruction.

• We propose RealCAD, a Sim2Real image-to-CAD reconstruction framework that combines geometry-constrained synthetic-to-real translation with multi-positive contrastive learning to reduce synthetic-to-real discrepancies at both the image and feature levels.

## Related Work

## Image-Based 3D Reconstruction

Image-based 3D reconstruction recovers 3D geometry or scene representations from one or multiple images using representations such as point clouds, meshes, voxels, neural fields, or 3D Gaussians (Scharstein and Szeliski 2002; Yao et al. 2018; Mildenhall et al. 2020; Kerbl et al. 2023; Zhang et al. 2026). Although these methods have achieved substantial progress in geometric reconstruction and novel-view synthesis, they primarily represent the geometry or appearance of the reconstructed object or scene and do not explicitly preserve parametric construction histories. In contrast, imageto-CAD reconstruction aims to recover executable command sequences containing modeling operations and parameters, thereby supporting history-aware editing and design reuse.

## Parametric CAD Generation and Reconstruction

DeepCAD (Wu, Xiao, and Zheng 2021) represents CAD models as sequences of sketch and extrusion commands and provides a large-scale dataset paired with construction histories. Based on this representation, subsequent methods reconstruct CAD sequences from point clouds (Khan et al. 2024a), text (Khan et al. 2024b), and images (Chen et al. 2025; Tan et al. 2026). However, the statistical properties of the widely used DeepCAD parameterization and their effects on model learning have received limited attention. We identify a representation-induced bias in the DeepCAD parameterization: its local normalization assigns a substantial portion of geometric scale information to a separate scale factor S, resulting in highly concentrated distributions for several normalized geometric parameters. We therefore revise the representation to produce more balanced parameter distributions and reduce reliance on frequency priors.

## Sim2Real Generalization for Image-to-CAD Reconstruction

Because paired real images and CAD command sequences are costly to collect, existing image-to-CAD methods are typically trained on synthetic renderings. The appearance gap between synthetic and real images, including diferences in materials, textures, illumination, and imaging conditions, limits real-world generalization. CADCrafter (Chen et al. 2025) promotes synthetic-to-real generalization by extracting texture-invariant geometric features from synthetic renderings, while Img2CADSeq (Tan et al. 2026) introduces an intermediate point-cloud representation and contrastive alignment. In contrast, our method performs adaptation at both the image and feature levels by combining geometryconstrained synthetic-to-real translation with multi-positive contrastive learning across viewpoints and image domains.

## Benchmark Construction

In this section, we first examine a previously underexplored distributional limitation in the parameter representation of

![](images/b56e959cc27d6e51b044e2c7fd72090f69b5db7d596a3dc6e6069126038d9cce.jpg)  
(a) CAD model (a) Previous CAD Command Sequence Encoding

![](images/bcfabb6d346d58214cc47e49df66073771a47193e5fcb0048f4d622470e0c0e3.jpg)  
(c) Modified CAD Command Sequence Encoding  
Figure 2: Revision of the DeepCAD parameter representation and its efect on parameter distributions: several parameters concentrate on very few valid values (e.g., most of x and y are concentrated at three specific values: 128, 176, and 223).

DeepCAD and introduce the revised DeepCAD representation. We then present OpenRealCAD, a real-image benchmark consisting of separate subsets for model adaptation and evaluation. Together, the revised DeepCAD representation and OpenRealCAD provide a unified benchmark for training, adapting, and evaluating image-to-CAD reconstruction models under Sim2Real settings.

## Preliminaries: CAD Command Sequence Encoding

A parametric CAD model is typically constructed through an ordered sequence of modeling operations. A common workflow begins by drawing a closed profile, referred to as a sketch, on a two-dimensional plane and subsequently extruding the profile along a specified direction to form a threedimensional solid. More complex models can be constructed by sequentially creating multiple sketch-extrusion units and combining them through Boolean operations such as union, subtraction, and intersection. We refer to the commands and associated parameters that describe this construction process as a CAD command sequence.

Although commercial CAD systems support a broad range of modeling operations, a compact set of commonly used commands, including sketching, extrusion, and Boolean operations, is suficient to represent a wide variety of mechanical parts. To obtain a structured representation suitable for neural networks, we follow the sketch-extrusion formulation introduced by DeepCAD and encode each command and its associated parameters as a 17-dimensional vector: $\mathbf { v } _ { i } =$ [Command, x $; , y , \alpha , f , r , \theta , \phi , \gamma , p _ { x } , p _ { y } , p _ { z } , s , e _ { 1 } , e _ { 2 } , b , u ]$ The first dimension specifies the command type, while the remaining dimensions store the corresponding geometric and operational parameters. Parameter entries that are not applicable to a given command are assigned a predefined invalid value. In addition to the modeling commands line, arc, circle, and extrusion, the representation contains two special tokens: ⟨SOL⟩ (Start of Loop), which marks the beginning of a closed sketch profile, and ⟨EOS⟩ (End of Sequence), which indicates the termination of the CAD command sequence. The command type is defined as ${ { C } _ { i } } \in \{ \langle \mathrm { S O L } \rangle , L , A , C , E , \langle \mathrm { E O S } \rangle$ } and the parameters associated with each command, together with their meanings, are summarized in Table 1, where b specifies the Boolean relationship between the current extrusion and the existing solid. It takes values from 0 to 3, corresponding to the creation of a new solid, union, subtraction, and intersection, respectively. The parameter u specifies the extrusion mode and takes values from 0 to 2, corresponding to one-sided, symmetric, and two-sided extrusion, respectively.

## Investigating Parameter Bias in DeepCAD

We first examine the parameter distributions induced by the DeepCAD command-sequence representation. Although continuous parameters are quantized into 256 discrete values, several geometric parameters occupy only a small subset and are strongly concentrated around a few dominant values. Figure 2(a) and (b) compare representative parameter distributions before and after our representation revision.

The concentration is most pronounced for sketch coordinates and radii. For x and y, the values 128, 176, and 223 collectively account for 61.62% of all observations, while radius values 47 and 48 account for 55.86% of the samples. Rotation parameters are also concentrated at four values corresponding to the principal orthogonal orientations; however, this pattern largely reflects the discrete orientation structure of the underlying CAD models rather than the same normalization-induced bias. By comparison, translation and extrusion-distance parameters are more dispersed after excluding the zero value encoded by 128.

This concentration mainly arises from the local normalization adopted by DeepCAD. Each sketch-extrusion unit is independently rescaled to fit the quantization range, and the corresponding scaling ratio is stored in the scale parameter s. In multi-unit models, independently normalizing geometrically diferent units repeatedly maps their local extrema to similar discrete values, while transferring much of their relative size information from the geometric parameters to s.

Figure 2(a) illustrates this efect using a composite model containing substructures with diferent dimensions. Under the original representation, geometrically distinct endpoints may be assigned identical quantized coordinates, while their relative dimensions are encoded primarily by s, resulting in concentrated geometric parameters and a comparatively dispersed scale distribution.

This information imbalance can afect both model learning and evaluation. A model may obtain high accuracy on strongly concentrated parameters by predicting dominant values rather than inferring geometry from the input image. At the same time, uniformly averaging parameter accuracies may understate errors in s, even though the scale factor carries substantial geometric information and controls the dimensions of an entire sketch–extrusion unit. Consequently, similar parameter accuracies do not necessarily imply comparable geometric fidelity.

To address this issue, we reparameterize all sketchextrusion units in a shared scale space while preserving their relative dimensions. Specifically, we redistribute the scale information stored in s to each afected geometric parameter. For each scale-dependent quantized parameter $p _ { \mathrm { o r i } } .$ , we absorb the unit-level scale factor s into the parameter as

$$
p _ { \mathrm { n e w } } = 1 2 8 + ( p _ { \mathrm { o r i } } - 1 2 8 ) \frac { s } { 1 9 2 } ,\tag{1}
$$

where 128 denotes the quantized origin and 192 is the reference value corresponding to the maximum scale used in the original representation.

This transformation is equivalent to expressing every sketch-extrusion unit under a common reference scale and encoding its original relative size directly in the corresponding geometric parameters. The explicit scale parameter s is therefore no longer required and is removed from the command vector. Because the transformation preserves the relative dimensions and spatial relationships of the original units, it retains the underlying CAD geometry while producing a more balanced parameterization.

We additionally canonicalize rotation and translation parameters. Since fewer than 0.01% of the samples contain non-orthogonal rotations, we remove these rare cases and represent rotations using a unified set of orthogonal orientations. For translation, we set the first sketch-extrusion unit as the reference origin and express all subsequent unit translations relative to it. The fixed translation of the reference unit is excluded from parameter-accuracy evaluation.

We refer to the resulting representation as the revised DeepCAD representation. As shown in Figure 2(b), after reparameterization, no individual value accounts for more than 3.5% of either coordinate distribution, while the maximum frequency for the radius parameter decreases to 14.6%. These results indicate that the revised DeepCAD representation substantially reduces parameter concentration while preserving the relative dimensions and spatial relationships of the retained CAD models.

## Details of OpenRealCAD

To support reproducible evaluation of Sim2Real image-to-CAD reconstruction, we construct OpenRealCAD, a realimage benchmark with paired ground-truth CAD command sequences. We randomly select 392 source models from DeepCAD and re-encode their construction sequences using the revised DeepCAD representation. This ensures that the synthetic and real domains share the same output parameterization and avoids representation mismatch during cross-domain training and evaluation.

We fabricate each model through 3D printing and capture it from four predefined viewpoints under controlled backgrounds and illumination. This setup limits pose and scene variation, enabling direct comparison between synthetic renderings and real photographs of the same geometry.

<table><tr><td>Command</td><td>Parameters and Descriptions</td></tr><tr><td>(SOL)</td><td>∅</td></tr><tr><td> $L$  (Line)</td><td> $x , y$  : Endpoint coordinates</td></tr><tr><td>A (Arc)</td><td>x, y : Endpoint coordinates α : Sweep angle f : Direction flag</td></tr><tr><td>C (Circle)</td><td>x, y : Center coordinates r : Radius</td></tr><tr><td>E (Extrude)</td><td> $\theta , \phi , \gamma$  : Plane rotation  $p _ { x } , p _ { y } , p _ { z }$  : Plane translation s : Scale factor  $e _ { 1 } , e _ { 2 }$  : Extrusion distances b : Boolean type</td></tr></table>

Table 1: CAD command types and their parameters.

OpenRealCAD is intended as a controlled real-image benchmark rather than an unconstrained in-the-wild dataset. By limiting background clutter, severe occlusion, and large viewpoint changes, the benchmark focuses evaluation on appearance diferences such as materials, surface textures, illumination, shadows, and imaging artifacts. It therefore provides an intermediate setting between synthetic-only evaluation and deployment in unconstrained real environments.

We split OpenRealCAD into disjoint training and test subsets at the CAD-instance level using a 7:3 ratio. To prevent instance-level leakage, all synthetic renderings associated with real-test models are excluded from synthetic training. The real training subset supports limited real-domain adaptation, while the held-out test subset evaluates generalization to unseen physical objects. Our dataset will be publicly released upon acceptance.

## Method

Building upon the revised DeepCAD representation, we propose RealCAD, an image-to-CAD reconstruction framework for Sim2Real generalization. RealCAD mitigates the synthetic-to-real discrepancy at both the image and feature levels. At the image level, geometry-constrained syntheticto-real translation reduces appearance diferences while preserving the underlying object structure. At the feature level, multi-positive contrastive learning promotes consistent representations of the same CAD model across viewpoints and image domains. Figure 3 illustrates the feature-learning pipeline, including multi-view sampling, visual encoding, and cross-view and cross-domain representation alignment. The learned representations are subsequently decoded by a Transformer decoder into CAD command sequences and their associated parameters.

![](images/dbaa842132fd7ec490c4c435d4c0138088649129d7408dbf18c0f55f97db1657.jpg)  
Figure 3: Overview of the feature-learning pipeline in RealCAD. Multi-view images from the synthetic and real domains are encoded by a frozen pretrained DINOv3 encoder and projected into a shared representation space, where multi-positive contrastive learning promotes cross-view and cross-domain consistency for the same CAD model.

## Geometry-Constrained Synthetic-to-Real Translation

To exploit abundant synthetic data while reducing the appearance gap to real photographs, we translate synthetic renderings toward the real-image domain. However, unconstrained image translation may alter object geometry and introduce inconsistent supervision for CAD reconstruction. We therefore employ an edge-conditioned ControlNet (Zhang, Rao, and Agrawala 2023) to constrain the translation process. Let $x _ { m , v } ^ { \mathrm { s y n } }$ denote the synthetic rendering of CAD model m from viewpoint v, and let $e _ { m , v }$ be its corresponding edge map. The translated image is generated as

$$
\boldsymbol { x } _ { m , v } ^ { \mathrm { t r } } = G _ { \mathrm { c t r l } } \left( x _ { m , v } ^ { \mathrm { s y n } } , e _ { m , v } , p ; \epsilon \right) ,\tag{2}
$$

where $G _ { \mathrm { c t r l } }$ denotes the edge-conditioned ControlNet, p is the text prompt, and ϵ denotes the random noise used during difusion sampling. The edge condition encourages the translated images to preserve object contours while adapting domain-specific appearance attributes such as materials, textures, illumination, and shadows. By retaining the underlying geometry while making synthetic appearances more similar to real photographs, the translated images narrow the imagelevel domain gap and facilitate Sim2Real generalization.

## Multi-Positive Contrastive Learning

For each CAD model, we render four fixed-view images and extract visual features using a pretrained DINOv3-ViT-7B encoder (Siméoni et al. 2025), whose parameters are kept frozen throughout training. A learnable projection layer maps the features into a shared latent space. The resulting representation is used both for contrastive alignment and as input to the Transformer decoder. Since diferent views and image-domain variants of the same CAD model share identical underlying geometry, we treat them as positive samples and regard images from diferent models as negatives.

For samples $( i , j )$ , let $\hat { \mathbf { z } } _ { i }$ and $\hat { \mathbf { z } } _ { j }$ denote the $\ell _ { 2 } \cdot$ -normalized latent representations. The similarity logit is defined as

$$
\begin{array} { r } { s _ { i j } = \alpha \hat { \mathbf { z } } _ { i } ^ { \top } \hat { \mathbf { z } } _ { j } + b , \qquad \alpha = \exp ( t ) , } \end{array}\tag{3}
$$

where t and b are learnable log-scale and bias parameters, respectively. This parameterization guarantees a positive logit scale $\alpha > 0$

We set $y _ { i j } = 1$ if the two samples are associated with the same CAD model and $y _ { i j } = - 1$ otherwise. The multipositive contrastive loss is formulated as

$$
\mathcal { L } _ { \mathrm { s i g l i p } } = - \frac { 1 } { | \Omega | } \sum _ { ( i , j ) \in \Omega } \log \sigma \left( y _ { i j } s _ { i j } \right) ,\tag{4}
$$

where Ω denotes the set of valid non-self sample pairs and $\sigma ( \cdot )$ is the sigmoid function. Because each CAD model may appear under multiple viewpoints and image domains, each sample can have multiple positives. This objective promotes cross-view and cross-domain consistency in the feature space while separating representations of diferent CAD models.

## CAD Sequence Reconstruction

Each latent representation is independently passed to a Transformer decoder to reconstruct the complete CAD command sequence. At each sequence position, the decoder predicts the command type and its associated discretized parameters. The reconstruction objective is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { c m d } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { p a r a } } , } \end{array}\tag{5}
$$

where $\mathcal { L } _ { \mathrm { c m d } }$ and $\mathcal { L } _ { \mathrm { p a r a } }$ denote the cross-entropy losses for command-type classification and discretized parameter classification, respectively, and $\lambda _ { 1 }$ and $\lambda _ { 2 }$ balance their contributions. The overall training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { s i g l i p } } + \mathcal { L } _ { \mathrm { r e c } } , } \end{array}\tag{6}
$$

which jointly optimizes cross-view and cross-domain representation alignment and CAD sequence reconstruction.

## Experiments

## Datasets Setup

We train and evaluate all models using three types of data: CAD command sequences encoded with the revised Deep-CAD representation, synthetic images rendered from these command sequences, and real images collected in Open-RealCAD. Figure 4 presents representative examples from OpenRealCAD, showcasing multiple physical CAD models captured from diferent viewpoints. For each CAD model, we render synthetic images from four fixed viewpoints using OpenCASCADE. The synthetic data are split into training and test sets at a ratio of 9:1, while OpenRealCAD is divided at the CAD-instance level into real training and test sets at a ratio of 7:3. To prevent instance leakage, synthetic renderings of models in the real test set are excluded from training. We additionally generate edge-conditioned translated images for image-level adaptation.

![](images/0b3a9987d0650cb589b549aa53ff090a1775ef98fa1af3b021714738e5ec8c61.jpg)  
(a)

![](images/72fccc528c730072af19dc068632a0e23b727605620895a7476835d03491cf72.jpg)  
(b)  
Figure 4: Examples from our OpenRealCAD dataset: (a) multi-view images of 3D-printed CAD models, and (b) additional 3D-printed models captured using smartphones.

## Implementation Details

We use a pretrained DINOv3-ViT-7B encoder to extract a 4096-dimensional feature from each image. All models are trained for 300 epochs with a batch size of 128 and an initial learning rate of $\mathbf { \bar { 1 } } \times 1 0 ^ { - 3 }$ on eight NVIDIA A100 GPUs. Unless otherwise specified, all methods use the same data splits and evaluation protocol.

## Baselines

We compare RealCAD with DeepCAD, Img2CAD, and CADCrafter. Since the original DeepCAD framework does not support image-conditioned reconstruction, we augment it with the same pretrained DINOv3 encoder used in RealCAD, enabling CAD sequence reconstruction from image features. For Img2CAD, we implement the method following its published architecture and training objective. For CADCrafter, we reproduce its supervised image-to-CAD reconstruction pipeline but omit the DPO stage. All methods are trained and evaluated using the same data splits and metrics, while unspecified implementation settings are kept consistent whenever possible.

## Experimental Results

Following prior image-to-CAD studies, we evaluate performance using Command Accuracy (Cmd.), Parameter Accuracy (Para.), median Chamfer Distance (mCD), and Invalidity Ratio (IR). Cmd. measures command-type accuracy at individual sequence positions, while Para. measures classification accuracy over valid command parameters. For successfully constructed predictions, mCD quantifies the geometric discrepancy between the reconstructed and ground-truth models. IR denotes the proportion of predicted sequences that cannot be parsed or constructed by the CAD kernel.

<table><tr><td rowspan="2">Parameter</td><td colspan="3">DeepCAD</td><td colspan="3">Revised DeepCAD</td></tr><tr><td>Model</td><td>Prior</td><td>Δ</td><td>Model</td><td>Prior</td><td>Δ</td></tr><tr><td>Line</td><td>57.71</td><td>16.77</td><td>40.94</td><td>50.27</td><td>3.14</td><td>47.13</td></tr><tr><td>Arc</td><td>51.44</td><td>31.31</td><td>20.13</td><td>50.61</td><td>22.93</td><td>27.68</td></tr><tr><td>Circle</td><td>72.23</td><td>26.13</td><td>46.10</td><td>57.81</td><td>6.41</td><td>51.40</td></tr><tr><td>Scale (s)</td><td>32.64</td><td>11.31</td><td>21.33</td><td></td><td></td><td></td></tr><tr><td>Rotation</td><td>88.48</td><td>54.15</td><td>34.33</td><td>87.05</td><td>54.15</td><td>32.90</td></tr><tr><td>Translation</td><td>50.07</td><td>7.15</td><td>42.92</td><td>70.34</td><td>1.64</td><td>68.70</td></tr><tr><td>Extrusion</td><td>34.38</td><td>14.78</td><td>19.60</td><td>33.07</td><td>14.78</td><td>18.29</td></tr></table>

Table 2: Comparison of the original DeepCAD representation and the revised DeepCAD representation. Model denotes the accuracy of the trained reconstruction model, while Prior denotes the accuracy obtained by always predicting the most frequent valid training-set value for each parameter type. ∆ = Model − Prior measures the gain over the frequencyonly baseline. All values are reported as percentages.

Higher Cmd. and Para. values are better, whereas lower mCD and IR values are preferred.

Efect of DeepCAD Representation Refinement. To quantify the extent to which parameter accuracy can be obtained from dataset-frequency priors without using visual evidence, we compare the trained reconstruction model with a frequency-only baseline. For each parameter type, the baseline always predicts the most frequent nonzero quantized value observed in the training set. The zero-valued bin 128 is excluded from both prior estimation and evaluation because it represents a legitimate geometric zero and remaines unafected by our reparameterization. The accuracy of the frequency-only baseline therefore provides an estimate of the performance attainable solely from the marginal parameter distribution. We train the same reconstruction architecture separately using the original DeepCAD representation and the revised DeepCAD representation.

As shown in Table 2, the frequency-only baseline achieves nontrivial accuracy under the original DeepCAD representation, particularly for Line, Arc, and Circle parameters, indicating that a considerable portion of parameter accuracy can be obtained without visual evidence. The revised Deep-CAD representation substantially reduces the prior accuracy for Line, Circle, and Translation from 16.77%, 26.13%, and 7.15% to 3.14%, 6.41%, and 1.64%, respectively, while retaining competitive model accuracy and enlarging the corresponding gains over the prior baseline. Rotation and Extrusion remain nearly unchanged because their distributions are not directly afected by the proposed scale redistribution. These results show that the revised DeepCAD representation reduces the accuracy attainable from marginal frequency priors and makes parameter accuracy more indicative of imageconditioned geometric inference.

Main Results.Table 3 shows that RealCAD achieves the best token-level performance on the real-domain test set, improving command and parameter accuracies over CADCrafter by 7.49 and 14.73 percentage points, respectively, supporting the efectiveness of combining image-level geometryconstrained synthetic-to-real translation with feature-level multi-positive contrastive learning for reducing the Sim2Real domain gap. However, these gains do not translate into a comparable improvement in mCD and are accompanied by a higher invalid ratio, since token-level metrics weight individual predictions similarly, whereas the final geometric quality is often dominated by a small number of geometry-sensitive parameters or structural operations. CAD parameters have highly heterogeneous geometric sensitivities: correcting a large number of low-impact parameters can improve tokenlevel accuracy without noticeably changing the reconstructed shape, whereas a single error in a topology-defining command or a geometry-sensitive parameter can dominate the final reconstruction error. Moreover, CAD execution is sequential and compositional, such that early prediction errors may alter the reference frame or feasible domain of subsequent operations. Therefore, geometry quality is determined not only by the number of correctly predicted parameters, but also by their semantic importance, cross-command consistency, and executability as a complete program.

<table><tr><td>Method</td><td>Cmd.↑</td><td>Para. ↑</td><td>mCD↓</td><td>IR↓</td></tr><tr><td colspan="5">Synthetic-Domain Test</td></tr><tr><td>DeepCAD* Img2CAD†</td><td>69.63</td><td>54.52</td><td>0.207</td><td>15.02 23.02</td></tr><tr><td>CADCrafter† Ours</td><td>73.02 67.82 72.94</td><td>57.20 45.80 61.35</td><td>0.226 0.246 0.220</td><td>22.92 26.85</td></tr><tr><td colspan="5">Real-Domain Test</td></tr><tr><td>DeepCAD*</td><td>43.34</td><td>38.35</td><td>0.293</td><td>15.38</td></tr><tr><td>Img2CAD†</td><td>61.80</td><td>43.08</td><td>0.302</td><td>30.77</td></tr><tr><td>CADCrafter†</td><td>57.36</td><td>37.30</td><td>0.309</td><td>17.09</td></tr><tr><td>Ours</td><td>64.85</td><td>52.03</td><td>0.308</td><td>48.72</td></tr></table>

Table 3: Quantitative comparison on the synthetic- and realdomain test sets. <sup>∗</sup> indicates our image-conditioned adaptation of the oficial DeepCAD code with a DINOv3 encoder. <sup>†</sup> indicates our reproduction based on the published paper, as the oficial code is unavailable. CADCrafter is reproduced without the DPO stage.

Ablation Studies. We evaluate geometry-constrained synthetic-to-real translation (ST) and multi-positive contrastive learning (CL) by independently removing each component. All variants use identical data splits, architectures, and optimization settings.

As shown in Table 4, CL alone increases real-domain command accuracy from 60.18% to 61.65%, although parameter accuracy decreases slightly. ST alone improves command and parameter accuracy to 62.61% and 51.48%, indicating that ST provides a larger individual gain than CL under this setting. Combining both modules yields the best real-domain results of 64.85% and 52.03%, improving over the baseline by 4.67 and 2.05 percentage points, respectively. The combined results suggest that ST and CL provide complementary benefits. ST narrows the appearance gap, while CL further promotes cross-view and cross-domain feature consistency.

The complete model shows a modest decrease in syntheticdomain command accuracy while maintaining comparable parameter accuracy. This result may reflect a trade-of between fitting synthetic-domain appearances and improving generalization to real images, which is consistent with the Sim2Real objective of RealCAD.

<table><tr><td>Modules</td><td colspan="2">Synthetic Images (%)</td><td colspan="2">Real Images (%)</td></tr><tr><td>CL ST</td><td>Cmd.</td><td>Para.</td><td>Cmd.</td><td>Para.</td></tr><tr><td>× ×</td><td>75.98</td><td>62.21</td><td>60.18</td><td>49.98</td></tr><tr><td>√ X</td><td>75.98</td><td>62.15</td><td>61.65</td><td>48.98</td></tr><tr><td>× √</td><td>72.51</td><td>60.90</td><td>62.61</td><td>51.48</td></tr><tr><td>√ √</td><td>72.94</td><td>61.35</td><td>64.85</td><td>52.03</td></tr></table>

Table 4: Efects of multi-positive contrastive learning (CL) and geometry-constrained synthetic-to-real translation (ST) on CAD reconstruction. Cmd. and Para. denote command and parameter accuracy, respectively. Bold values indicate the best result in each column.

Qualitative Results. Detailed comparisons of the qualitative results will be provided in the supplementary material.

## Limitations

Although this paper identifies and systematically analyzes several issues in recent CAD-related research, the current framework still has two main limitations. First, although the revised DeepCAD representation reduces parameter concentration caused by local normalization, CAD parameters remain intrinsically long-tailed because coordinates and geometric entities near the sketch origin occur more frequently. Long-tail-aware training and evaluation may further improve performance. Second, RealCAD mainly optimizes tokenlevel accuracy without explicitly modeling CAD grammar, command dependencies, or geometric feasibility. Consequently, token-level gains do not always yield comparable improvements in geometric quality or program validity, as indicated by the modest mCD improvement and higher IR. Future work could incorporate grammar-constrained decoding and geometry- or execution-aware objectives.

## Conclusion

In this paper, we focus on the robustness and reliability of Image-to-CAD methods in real-world application scenarios. We identify a previously overlooked parameter bias in the DeepCAD representation and introduce a revised parameterization that reduces geometric parameter concentration and frequency-based shortcuts. We also propose RealCAD, a Sim2Real image-to-CAD reconstruction framework combining geometry-constrained image translation with multipositive contrastive learning to improve cross-domain and cross-view feature consistency. Experiments show that the revised representation reduces dependence on marginal frequency priors, while RealCAD improves command and parameter accuracy on real images. However, the limited gains in geometric quality and program validity suggest that tokenlevel optimization alone is insuficient, motivating future work on geometry- and validity-aware learning.

## Acknowledgement

This work was supported by the Fundamental Research Funds for the Central Universities at Tongji University under Grant No. 22120260376.

## References

Achlioptas, P.; Diamanti, O.; Mitliagkas, I.; and Guibas, L. J. 2018. Learning Representations and Generative Models for 3D Point Clouds. In Proceedings of the 35th International Conference on Machine Learning, volume 80 ofProceedings of Machine Learning Research, 40–49. PMLR.

Cai, R.; Yang, G.; Averbuch-Elor, H.; Hao, Z.; Belongie, S.; Snavely, N.; and Hariharan, B. 2020. Learning Gradient Fields for Shape Generation. In Computer Vision – ECCV 2020, 364–381. Springer.

Carion, N.; Massa, F.; Synnaeve, G.; Usunier, N.; Kirillov, A.; and Zagoruyko, S. 2020. End-to-End Object Detection with Transformers. In Computer Vision – ECCV 2020, 213– 229. Springer.

Chen, C.; Wei, J.; Chen, T.; Zhang, C.; Yang, X.; Zhang, S.; Yang, B.; Foo, C.-S.; Lin, G.; Huang, Q.; and Liu, F. 2025. CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Chen, Z.; Tagliasacchi, A.; and Zhang, H. 2020. BSP-Net: Generating Compact Meshes via Binary Space Partitioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 45–54.

Dong, Y.; Zuo, Q.; Gu, X.; Yuan, W.; Zhao, Z.; Dong, Z.; Bo, L.; and Huang, Q. 2024. GPLD3D: Latent Difusion of 3D Shape Generative Models by Enforcing Geometric and Physical Priors. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 56–66. IEEE.

Girdhar, R.; Fouhey, D. F.; Rodriguez, M.; and Gupta, A. 2016. Learning a Predictable and Generative Vector Representation for Objects. In Computer Vision – ECCV 2016, 484–499. Springer.

Groueix, T.; Fisher, M.; Kim, V. G.; Russell, B. C.; and Aubry, M. 2018. A Papier-Mâché Approach to Learning 3D Surface Generation. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 216–224.

Gulrajani, I.; Ahmed, F.; Arjovsky, M.; Dumoulin, V.; and Courville, A. 2017. Improved Training ofWasserstein GANs. In Advances in Neural Information Processing Systems, volume 30, 5767–5777. Curran Associates, Inc.

Jones, R. K.; Barton, T.; Xu, X.; Wang, K.; Jiang, E.; Guerrero, P.; Mitra, N. J.; and Ritchie, D. 2020. ShapeAssembly: Learning to Generate Programs for 3D Shape Structure Synthesis. ACMTransactions on Graphics, 39(6): 234:1–234:20.

Ju, H.; Juan, R.; Gomez, R.; Nakamura, K.; and Li, G.-L. 2022. Transferring Policy of Deep Reinforcement Learning from Simulation to Reality for Robotics. Nature Machine Intelligence, 4(12): 1077–1087.

Kerbl, B.; Kopanas, G.; Leimkühler, T.; and Drettakis, G. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4): 1–14.

Khan, M. S.; Dupont, E.; Ali, S. A.; Cherenkova, K.; Kacem, A.; and Aouada, D. 2024a. CAD-SIGNet: CAD Language Inference from Point Clouds using Layer-wise Sketch Instance Guided Attention. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4713–4722.

Khan, M. S.; Sinha, S.; Sheikh, T. U.; Stricker, D.; Ali, S. A.; and Afzal, M. Z. 2024b. Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts. In Advances in Neural Information Processing Systems, volume 37, 7552–7579.

Koch, S.; Matveev, A.; Jiang, Z.; Williams, F.; Artemov, A.; Burnaev, E.; Alexa, M.; Zorin, D.; and Panozzo, D. 2019. ABC: A Big CAD Model Dataset for Geometric Deep Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9601–9611.

Li, C.; Pan, H.; Bousseau, A.; and Mitra, N. J. 2020. Sketch2CAD: Sequential CAD Modeling by Sketching in Context. ACM Transactions on Graphics, 39(6): 164:1– 164:14.

Li, J.; Xu, K.; Chaudhuri, S.; Yumer, E.; Zhang, H.; and Guibas, L. J. 2017. GRASS: Generative Recursive Autoencoders for Shape Structures. ACM Transactions on Graphics, 36(4): 52:1–52:14.

Li, P.; Guo, J.; Li, H.; Benes, B.; and Yan, D.-M. 2024. SFMCAD: Unsupervised CAD Reconstruction by Learning Sketch-Based Feature Modeling Operations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4671–4680.

Li, P.; Guo, J.; Zhang, X.; and Yan, D.-M. 2023. SECAD-Net: Self-Supervised CAD Reconstruction by Learning Sketch-Extrude Operations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16816–16826.

Liao, Y.; Donne, S.; and Geiger, A. 2018. Deep Marching Cubes: Learning Explicit Surface Representations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2916–2925.

Liu, M.; Xu, C.; Jin, H.; Chen, L.; T, M. V.; Xu, Z.; and Su, H. 2023a. One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization. In Advances in Neural Information Processing Systems, volume 36.

Liu, R.; Wu, R.; Van Hoorick, B.; Tokmakov, P.; Zakharov, S.; and Vondrick, C. 2023b. Zero-1-to-3: Zero-Shot One Image to 3D Object. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 9298–9309.

Long, X.; Guo, Y.-C.; Lin, C.; Liu, Y.; Dou, Z.; Liu, L.; Ma, Y.; Zhang, S.-H.; Habermann, M.; Theobalt, C.; et al. 2024. Wonder3D: Single Image to 3D Using Cross-Domain Difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9970–9980.

Ma, W.; Chen, S.; Lou, Y.; Li, X.; and Zhou, X. 2024. Draw Step by Step: Reconstructing CAD Construction Sequences from Point Clouds via Multimodal Difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 27154–27163.

Ma, W.; Xu, M.; Li, X.; and Zhou, X. 2023. MultiCAD: Contrastive Representation Learning for Multi-Modal 3D Computer-Aided Design Models. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, 1766–1776. ACM.

Mescheder, L.; Oechsle, M.; Niemeyer, M.; Nowozin, S.; and Geiger, A. 2019. Occupancy Networks: Learning 3D Reconstruction in Function Space. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4460–4470.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV.

Mo, K.; Guerrero, P.; Yi, L.; Su, H.; Wonka, P.; Mitra, N. J.; and Guibas, L. J. 2019. StructureNet: Hierarchical Graph Networks for 3D Shape Generation. ACM Transactions on Graphics, 38(6): 242:1–242:19.

Nash, C.; Ganin, Y.; Eslami, S. M. A.; and Battaglia, P. 2020. PolyGen: An Autoregressive Generative Model of 3D Meshes. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, 7220–7229. PMLR.

Nichol, A.; Jun, H.; Dhariwal, P.; Mishkin, P.; and Chen, M. 2022. Point-E: A System for Generating 3D Point Clouds from Complex Prompts. arXiv preprint arXiv:2212.08751.

Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H. V.; Szafraniec, M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; El-Nouby, A.; Assran, M.; Ballas, N.; Galuba, W.; Howes, R.; Huang, P.-Y.; Li, S.-W.; Misra, I.; Rabbat, M.; Sharma, V.; Synnaeve, G.; Xu, H.; Jégou, H.; Mairal, J.; Labatut, P.; Joulin, A.; and Bojanowski, P. 2024. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research.

Park, J. J.; Florence, P.; Straub, J.; Newcombe, R.; and Lovegrove, S. 2019. DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 165–174.

Parmar, N.; Vaswani, A.; Uszkoreit, J.; Kaiser, L.; Shazeer, N.; Ku, A.; and Tran, D. 2018. Image Transformer. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, 4055–4064. PMLR.

Scharstein, D.; and Szeliski, R. 2002. A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms. International Journal ofComputer Vision, 47(1–3): 7–42.

Sharma, G.; Goyal, R.; Liu, D.; Kalogerakis, E.; and Maji, S. 2018. CSGNet: Neural Shape Parser for Constructive Solid Geometry. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 5515–5523.

Sharma, G.; Liu, D.; Maji, S.; Kalogerakis, E.; Chaudhuri, S.; and Měch, R. 2020. ParSeNet: A Parametric Surface Fitting Network for 3D Point Clouds. In Computer Vision – ECCV 2020, 261–276. Springer.

Siméoni, O.; Vo, H. V.; Seitzer, M.; Baldassarre, F.; Oquab, M.; Jose, C.; Khalidov, V.; Szafraniec, M.; Yi, S.; Ramamonjisoa, M.; Massa, F.; Haziza, D.; Wehrstedt, L.; Wang, J.; Darcet, T.; Moutakanni, T.; Sentana, L.; Roberts, C.; Vedaldi, A.; Tolan, J.; Brandt, J.; Couprie, C.; Mairal, J.; Jégou, H.; Labatut, P.; and Bojanowski, P. 2025. DINOv3. arXiv preprint arXiv:2508.10104.

Tan, S.; Zhao, Z.; Gao, H.; Chen, Z.; Yin, X.; and Shen, E. 2026. Img2CADSeq: Image-to-CAD Generation via Sequence-Based Difusion. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems, volume 30, 5998–6008. Curran Associates, Inc.

Wang, N.; Zhang, Y.; Li, Z.; Fu, Y.; Liu, W.; and Jiang, Y.-G. 2018. Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images. In Proceedings of the European Conference on Computer Vision, 52–67.

Wei, J.; Wang, H.; Feng, J.; Lin, G.; and Yap, K.-H. 2023. TAPS3D: Text-Guided 3D Textured Shape Generation from Pseudo Supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16805–16815.

Wu, R.; Xiao, C.; and Zheng, C. 2021. DeepCAD: A Deep Generative Network for Computer-Aided Design Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 6772–6782.

Xu, X.; Peng, W.; Cheng, C.-Y.; Willis, K. D. D.; and Ritchie, D. 2021. Inferring CAD Modeling Sequences Using Zone Graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6062–6070.

Yang, Y.; Feng, C.; Shen, Y.; and Tian, D. 2018. FoldingNet: Point Cloud Auto-Encoder via Deep Grid Deformation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 206–215.

Yao, Y.; Luo, Z.; Li, S.; Fang, T.; and Quan, L. 2018. MVS-Net: Depth Inference for Unstructured Multi-view Stereo. In Proceedings of the European Conference on Computer Vision, 785–801.

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding Conditional Control to Text-to-Image Difusion Models. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision, 3836–3847.

Zhang, X.; Zheng, X.; Yin, Y.; Zhao, T.; Tang, K.; Mi, M. B.; Xu, Z.; and Chen, D. Z. 2026. AnchorSplat: Feed-Forward 3D Gaussian Splatting With 3D Geometric Priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 18924–18933.

## Supplementary Material

This supplementary material provides additional analysis and qualitative evaluations to complement the main paper. We first investigate the parameter distribution bias in the Deep-CAD dataset and provide detailed statistics for diferent CAD parameters. We further present qualitative reconstruction results under both synthetic and real-world image inputs.

## Detailed Analysis of the DeepCAD Dataset

The DeepCAD dataset contains noticeable distribution biases among several geometric parameters in its original CAD representation. Such biases may lead models to overfit frequent parameter values and reduce their ability to infer accurate geometric attributes from input images. To provide a detailed analysis, we report the observed frequency and percentage of dominant parameter values in the following tables.

<table><tr><td>Value</td><td>x</td><td>y</td></tr><tr><td>128</td><td>492273 (32.27%)</td><td>667450 (43.77%)</td></tr><tr><td>176</td><td>174920 (11.47%)</td><td>27623 (1.82%)</td></tr><tr><td>223</td><td>317427 (20.82%)</td><td>199585 (13.09%)</td></tr><tr><td>Others</td><td>317427 (35.44%)</td><td>199585 (41.32%)</td></tr></table>

Table 5: Distribution bias of parameters x and y in the Deep-CAD dataset.

<table><tr><td>Value</td><td>r</td></tr><tr><td>47</td><td>22448 (11.62%)</td></tr><tr><td>48</td><td>85071 (44.25%)</td></tr><tr><td>Others</td><td>84817 (44.12%)</td></tr></table>

Table 6: Distribution bias of parameter r in the DeepCAD dataset.

The above statistics demonstrate that the original Deep-CAD representation sufers from substantial distribution biases across diferent types of CAD parameters, including coordinates, geometric sizes, rotations, and translations.

For the coordinate parameters x and y, the distributions are highly imbalanced toward several specific values. In particular, the value 128 accounts for 32.27% of x occurrences and 43.77% of $y$ occurrences, respectively. Meanwhile, several other discrete values, such as 176 and 223, also appear frequently. This indicates that the sketch generation process in DeepCAD tends to favor specific coordinate locations rather than uniformly covering the spatial domain. Such bias may encourage models to rely on frequent coordinate priors instead of accurately inferring spatial positions from visual inputs.

The radius parameter r also exhibits a noticeable imbalance. The value 48 alone appears in 44.25% of all samples, while value 47 contributes another 11.62%. Together, these two values account for more than half of the observed radius distributions. This concentration suggests that the dataset contains a strong preference toward specific object scales, potentially limiting the model’s ability to generalize to objects with unseen or uncommon geometric sizes.

<table><tr><td>Value</td><td>θ</td><td>φ</td><td>γ</td></tr><tr><td>64</td><td>0 / 0%</td><td>145173 / 39.60%</td><td>14 / 0.00%</td></tr><tr><td>128</td><td>165852 / 45.21%</td><td>213205 / 58.18%</td><td>165913 / 45.28%</td></tr><tr><td>192</td><td>191242 / 52.15%</td><td>7842 / 2.15%</td><td>191791 /52.29%</td></tr><tr><td>255</td><td>8817/2.43%</td><td>0/0%</td><td>8847 /2.43%</td></tr><tr><td>Others</td><td>737 /0.21%</td><td>428/0.12%</td><td>83 /0.05%</td></tr></table>

Table 7: Distribution bias of rotation parameters θ, ϕ, and γ in the DeepCAD dataset.

<table><tr><td>Value</td><td> $p _ { x }$ </td><td> $p _ { y }$ </td><td> $p _ { z }$ </td></tr><tr><td>32</td><td>54523 /14.84%</td><td>15192/4.16%</td><td>7859 /2.15%</td></tr><tr><td>128</td><td>65157 /17.75%</td><td>190089 /51.87%</td><td>206149 /56.24%</td></tr><tr><td>Others</td><td>247450 /67.41%</td><td>161138/43.97%</td><td>152522/41.61%</td></tr></table>

Table 8: Distribution bias of translation parameters $p _ { x } , p _ { y } .$ and $p _ { z }$ in the DeepCAD dataset.

The rotation parameters θ, ϕ, and $\gamma$ demonstrate even stronger concentration patterns. For θ, the values 128 and 192 account for 45.21% and 52.15% of occurrences, respectively, covering 97.36% of all samples. Similarly, $\gamma$ is dominated by values 128 and 192, which together represent 97.57% of the distribution. For $\phi ,$ values 64 and 128 occupy 39.60% and 58.18% of samples, respectively. These results indicate that the original dataset contains only limited diversity in object orientations, causing the rotation representation to be strongly biased toward a few canonical poses.

The translation parameters $p _ { x } , p _ { y } ,$ , and $p _ { z }$ also exhibit considerable spatial bias. The value 128 dominates both $p _ { y }$ and $p _ { z }$ , accounting for 51.87% and 56.24% of samples, respectively. In contrast, $p _ { x }$ shows a relatively broader distribution, where the “Others” category contributes 67.41% of occurrences. This suggests that although horizontal translations are relatively diverse, vertical and depth translations remain concentrated around specific locations.

Overall, these observations reveal that the original Deep-CAD representation contains strong prior biases across multiple parameter dimensions. Such imbalanced distributions may introduce unintended shortcuts during training, where models can exploit frequent parameter patterns instead of learning generalizable geometric reasoning from image inputs. Therefore, addressing parameter distribution bias is important for improving robustness and generalization in image-to-CAD reconstruction.

## Qualitative Results

Figure 5 presents qualitative comparisons between our method and several reproduced baselines under both synthetic and real-world image inputs.

For synthetic images, our method successfully reconstructs all provided examples with accurate geometric structures. Existing approaches, including DeepCAD and Img2CAD, achieve comparable reconstruction quality on synthetic inputs. However, CadCrafter occasionally produces

Input Image

![](images/8bc287b152c826a8a19198b632fd2edb0897b2d78d859a995d43dd778fe66fe0.jpg)

Deepcad

![](images/e78e657bf2604878f750b607ee14d32277e2f1558b5e9717efcb0d83a18e9087.jpg)

Img2cad

![](images/cf944cada2455fe62464be649e6e727c977b408ca21dd6f0f35090d63062d57e.jpg)

Cadcrafter

X

Ours

![](images/58cf51233c096297a5dee74ed290a47565f85cbcc23be802af4853b735b465d0.jpg)

Ground Truth

![](images/ab6e610f3b0cc6f986a2363303404098e7aa8f9b43d98be94b186b6e2407269d.jpg)

![](images/b93e4a00354f6e0c8fe5648bbb6b29742947ffcdaec0ed31c852e606a7658be3.jpg)

![](images/15f38279149a78184a3ba5e93f80ebeeec7d1bb404af181fc3235395f8291765.jpg)

![](images/616c1f9ad8a62f0a533a0bbc9739ee3c724caf1ae967e3ebbb5580f69e7c703f.jpg)

![](images/1d8aa5ef74241d7b971db32a6efac3b6bc53f65476bf114be89bfcc8d24b8313.jpg)

![](images/36ddde84a11453873d1964956dd7a854dbabcaecf69ef3351f65feff2cbd4054.jpg)

![](images/18dd03e29c9cd2921392d68ca91e31acb54700b531393934999a4fbbd904980f.jpg)

![](images/0ffc99592b8c1aea1e5ea8454b2d6af64a3cdf8d33103e784e6f1bfaff57edf8.jpg)

![](images/0a65e53c585d09969cabd3a4e7f1cb2afe3fc3b2af5493ac2e3a97124c78aaaf.jpg)

![](images/38613b799056dc1df576a74a773bd7665c871308b4a902f1313021d676ed729e.jpg)

![](images/b1ed5f63bd2b558d01b42bc14fc938b84948cfc17a45bdb9fb66c4dcb34ae7cc.jpg)

![](images/904e32dee14bfec5d7cc5a8be365aec06c154e5dd48f6825a7f7fee2844acc7d.jpg)

![](images/5bba9ac82c25700cf7e4ddb3e328935accee4801b0e862b71119f6ed9bef31af.jpg)

![](images/3b4a8376cbd779d319098254d60f537638534ae0b8db8bd9931259dfaf028313.jpg)

![](images/486f0e45df49e320f06ef21f234f2c1fe5874151da5486902f1caafc5c6230c7.jpg)

![](images/bd3340ca26d26552c1961b80032c7a537cbb2dde8f1ee4bdbbfdef6c49e060d9.jpg)

![](images/f96efaa1bbc0c15e307b06ec3fc13ce3f646fcb3118d0fcb2332b779139c79d0.jpg)

![](images/871c632d1958bfc9fb3beb6a78c14c9905a09ae2b461abd8d73a732995206471.jpg)

![](images/c3450e0a6fcd699cf55d980eeb90d788e5f1dce7933d6deb32c4b8de96eed44f.jpg)

![](images/7550251548efaf9c9d9c93b68dabbf0e5e63f1103e265eb31a0337cafcb83892.jpg)

X

X

X

![](images/331de6d3fdc8ca498ef7af56a69ee743a09642da554743f6c9badf5a7e9e7b6a.jpg)

![](images/810c9ad7d066e26f41ff4ffdacbfaaef25800966206a633e88ecd0ddbfe3395f.jpg)

Figure 5: Qualitative comparison of image-to-CAD reconstruction results on synthetic and real-world image inputs. Our method produces more accurate geometric structures and dimensions compared with existing approaches, especially for complex objects under real-world conditions

inaccurate dimensions and spatial locations, resulting in incomplete reconstruction results.

For real-world image inputs, DeepCAD, Img2CAD, and CadCrafter are able to recover the overall topology of simple objects such as ring-like structures. Nevertheless, their predicted geometric dimensions are less accurate compared with our method. In particular, our approach is the only method capable of accurately reconstructing the triangular prism example, demonstrating stronger robustness when handling images with complex geometric structures and realistic variations.