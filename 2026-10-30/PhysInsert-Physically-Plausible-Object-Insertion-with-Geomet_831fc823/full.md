# PhysInsert: Physically Plausible Object Insertion with Geometric Priors and Bidirectional Object-Scene Interactions

Yue Gong<sup>1,2,</sup>∗, Le Zhang<sup>2</sup>, Rui Chen<sup>2</sup>, Song Chang<sup>2</sup>, Xiaochao Qu<sup>2</sup>, Luoqi Liu<sup>2</sup>, Ting Liu<sup>2,</sup>†, Yebin Liu<sup>3,</sup>†

<sup>1</sup>School of Computer Science and Engineering, Beihang University <sup>2</sup>MT Lab, Meitu Inc. <sup>3</sup>Department of Automation, Tsinghua University

## Abstract

We present PhysInsert, a framework for physically plausible reference-based object insertion that jointly adapts objects and their surrounding scenes. Existing methods often rely on masked inputs, generation-based pairs, or training pairs with limited viewpoint variation, leaving pose and illumination adaptation and bidirectional object–scene interactions insuficiently constrained. We construct the Insert World dataset with strictly aligned Blender-rendered counterfactual pairs and object references across viewpoints and lighting conditions. Building on these data, Geometry-Guided Adaptation couples a VGGT geometry stream with image and text streams un der target-scene feature supervision to guide spatially consistent insertion. Complementary 2D Object–Scene Spatial Supervision jointly supervises object occupancy and environmental changes while grounding spatial instructions in object regions through image–text attention. We introduce PhysInsert-Bench with dedicated cross-environment and cross-view cases requiring illumination and pose adaptation beyond appearance copying. Experiments on PhysInsert-Bench and Dream-Booth demonstrate improved image fidelity; A user study further show more accurate placement and improved lighting integration, perspective, and pose plausibility.

Project Page: https://arthuring.github.io/PhysInsert/

![](images/428e8d0d519ea6d6b77521cad1b4a4bf25ea047d1171a903c3cc56c678e4e53b.jpg)

![](images/59fa40c21c178dd392d2d1c5b47e293f8e12f2caf4ad03e60ebb5b2ea835e780.jpg)

![](images/ed2e300bf32c8a3b9701a1505a36fe865b458e7f02e03a1c238109ff066d1c2e.jpg)

![](images/b4b616ce9ecc11e209dad8b498707e4b286d95385bf37d11efcc60ea666d9961.jpg)

![](images/6a272b58da9d9e8eedb426e8be1e44f9f2c7a9e9965a89908bbd41f83c49d5dd.jpg)

![](images/067caac7d84046e613f94d2ab44026c128ca9b9ae369e4013d7f34efaf4cfc2e.jpg)

![](images/6c983c54201a05c6eae2768259f14c71a5455002878e75f86cea9efecb9263c9.jpg)

![](images/383875a5558aaa14a9ba2c32a9d56d007abdc08d518d5adfa2e373d39e42647c.jpg)

![](images/d289ad5b5068cbdce1eabad5dbe001f4db0f256c806aefe4ed05857c0f2adaa0.jpg)

![](images/c7d6e3eb2041646aa45457608b39c1b4bb01632b0cdef7ab159598b87b83bb42.jpg)  
Figure 1 (a–c) PhysInsert simultaneously supports cross-environment adaptation, bidirectional object–scene interactions, and cross-view adaptation. (d) InsertWorld ofers broad coverage of scenes, object properties, and interaction types.

## 1 Introduction

Reference-based object insertion aims to integrate a specified object into a target scene while preserving its identity. Physically plausible insertion further requires geometric and optical consistency, with the object’s perspective, pose, appearance, and efects on its surroundings all adapted to the target scene.

As illustrated in Figure 1, despite recent progress, three challenges remain in physically plausible object insertion. (a) Adaptation across environments. Existing methods (Chen et al., 2024a; Song et al., 2025; Tian et al., 2025) can insert objects into new scenes, but provide limited controlled supervision for adapting the same object across lighting environments. (b) Bidirectional object– scene interactions. Mask-and-fill methods (Gong et al., 2026; Yu et al., 2025) may adapt object illumination yet struggle to produce consistent scene changes; training pairs constructed through masking or generative editing (Chen et al., 2024a; Song et al., 2025) do not guarantee accurate supervision of shadows and reflections. Training examples with simplified backgrounds (Michel et al., 2023) also underrepresent complex occlusion, attachment, and spatial relationships with existing scene objects. (c) Adaptation across viewpoints. Even with consistent illumination, difering reference and target viewpoints still require perspective and pose adaptation; fixed-view training pairs (Yang et al., 2023; Wang et al., 2025b) ofer limited supervision for this change. Reliance on mask or box guidance also limits fine-grained text control of placement and pose (Chen et al., 2024a; Gong et al., 2026). Methods relying primarily on 2D image supervision (Chen et al., 2024a; Song et al., 2025) lack direct constraints on object–scene geometry, making it dificult to guide adaptation to the target perspective and pose.

We propose PhysInsert, a framework combining counterfactual scene data, geometric priors, and spatial guidance for physically plausible object insertion.

To provide counterfactual supervision for object–scene adaptation, we construct InsertWorld from strictly aligned pairs rendered by changing only object presence in complete scenes, capturing both scene-dependent object appearance and object-induced environmental changes. Object references rendered across lighting environments and viewpoints support cross-environment and cross-view adaptation. Paired with spatial instructions, these examples support text-guided control of placement and complex interactions with existing scene objects.

To align object perspective and pose with the scene’s 3D structure, Geometry-Guided Adaptation models post-insertion geometry as a supervised intermediate representation. Initialized from background VGGT features, a dedicated geometry stream evolves through joint attention with image and text tokens, with its feature readouts supervised against VGGT features of the with-object scene. The geometry stream remains active at inference to guide scene-consistent object adaptation.

To localize insertion-induced changes and ground spatial instructions in the image, we complement geometric adaptation with two forms of 2D supervision. Counterfactual Object–Scene Supervision uses object masks and aligned scene pairs to jointly supervise object occupancy and external efects. Soft-Mask Spatial Grounding aligns image–text attention with the object region to translate spatial instructions into placement control while retaining the full background input.

To evaluate scene adaptation beyond appearance copying, we construct PhysInsert-Bench with cases requiring illumination or pose adaptation and include text-guided and real-world evaluations.

Our main contributions are as follows:

• Dataset: InsertWorld combines strictly aligned counterfactual scene pairs, cross-view and cross-lighting object references, and spatial instructions to support learning object adaptation and complex object–scene interactions.

• Method: PhysInsert combines a background-initialized geometry stream supervised toward post-insertion scene features with 2D supervision, improving cross-environment and cross-view insertion and text-guided placement.

• Benchmark: PhysInsert-Bench tests illumination and pose adaptation beyond appearance copying through dedicated cross-environment, cross-view, and in-the-wild cases.

## 2 Related Work

Physics-aware Image Generation and Editing. Recent work has sought to improve the physical plausibility of image editing by modeling scene geometry and physical interactions. PhysicEdit (Zhao et al., 2026) learns video-based transition priors to model causal physical changes in instructionbased editing. PhyEdit (Xu et al., 2026) relocates existing objects within a scene using geometric guidance and depth supervision. Object3DIT (Michel et al., 2023) supports language-guided translation, rotation, insertion, and removal in procedurally rendered scenes; its insertion specifies objects through text rather than external reference images. Our task requires preserving a reference object’s identity while adapting its viewpoint, illumination, and interactions to a new scene.

Reference-based Object Insertion. For supervision, ObjectDrop (Winter et al., 2024) and Omni Paint (Yu et al., 2025) use real counterfactual pairs, supplemented by generated bootstrapping or insertion–removal cycles. ObjectMate (Winter et al., 2025) and Insert-Anything (Song et al., 2025) construct backgrounds through learned editing, whose accuracy depends on the editor; PS-Difusion (Wang et al., 2025b) uses rendered pairs. For control, mask-based methods (Yang et al., 2023; Chen et al., 2024a; Yu et al., 2025) discard background information within the editing region, while ObjectMate uses boxes, Insert-Anything also supports text guidance, and DIRECT (Gong et al., 2026) requires 3D proxy reconstruction and placement for pose control. For viewpoint adap tation, Paint by Example derives same-image references, PS-Difusion retains the target viewpoint, and ObjectDrop preserves pasted-object pose and illumination; AnyDoor, Insert-Anything, and ObjectMate incorporate multiview data. PhysInsert combines controlled counterfactual rendering and diverse references with supervised geometry states, object–efect supervision, and spatial-text grounding.

## 3 InsertWorld Dataset

This section compares InsertWorld with existing datasets, describes its construction pipeline, and summarizes its statistics of training–test split.

![](images/5ff58d8a2af689c9831799fc00e556208f5b29ac7f78ba31551a62e7b2d3c4b4.jpg)  
Figure 2 Overview of InsertWorld construction and PhysInsert-Bench composition.

## 3.1 Comparison with Existing Datasets

Table 1 Comparison of object-insertion datasets.
<table><tr><td>Dataset</td><td>CF</td><td>MV</td><td>ML</td><td>Ctrl.</td></tr><tr><td>AnyInsertion</td><td>X</td><td>Partial</td><td>X</td><td>Mask / Text</td></tr><tr><td>DIRECT</td><td>X</td><td>√</td><td>X</td><td>Mask + pose</td></tr><tr><td>InsertWorld</td><td>L</td><td>L</td><td>L</td><td>Text / Box</td></tr></table>

CF: controlled counterfactual pairs; MV: multiview references; ML: multilight references; Ctrl.: control inputs. ✓: provided; ✗: not specifically constructed; Partial: partial coverage.

As summarized in Table 1, AnyInsertion (Song et al., 2025) and DIRECT (Gong et al., 2026) do not explicitly provide controlled counterfactual scene pairs or dedicated multilight references. Multiview references cover only part of AnyInsertion. Our InsertWorld combines strictly aligned counterfactual pairs, multiview and multilight references to support adaptation across environments and viewpoints alongside object–scene interactions.

## 3.2 Dataset Construction

To reduce interpenetration and floating artifacts from random placement, we construct InsertWorld by removing existing objects from complete BlenderKit scenes (Figure 2). We render each scene with and without the object under fixed camera, lighting, and remaining scene settings, recomputing its shadows, mirror images, and reflections. These strictly aligned counterfactual pairs capture scene-dependent object appearance and object-induced environmental changes. Quality filtering and object deduplication retain suitable scene–object combinations.

We augment scene viewpoints and independently render each retained object across lighting environments and viewpoints to provide references for cross-environment and cross-view adaptation. Each example combines an object-free background, a with-object target, and reference images with Gemini-generated annotations comprising an editing instruction, a placement description, and a target-pose description. Filtering and sampling details are provided in Appendix I.

## 3.3 Dataset Overview

The InsertWorld training set contains 12,951 counterfactual scene image pairs and 275,310 reference images of 1,710 objects spanning diferent viewpoints and lighting conditions. For evaluation, PhysInsert-Bench comprises 400 cases from 103 scenes with no overlap with the training scenes and

![](images/44cb743b40e2680a18f10dbc13d3ef16036f53f9a42be37a70b7980bcf00d20f.jpg)  
Figure 3 PhysInsert combines a VGGT geometry stream with image and text streams. Target-scene features supervise geometry-token states, while 2D-OSS aligns object/efect predictions and spatial-text attention with image regions. Auxiliary supervision is used only during training.

object, including 180 Optical cases (154 with aligned reference–target viewpoints and 26 cross-view challenges) and 220 Structural cases with independent reference viewpoints. Benchmark construc tion is detailed in §4.2.

## 4 PhysInsert Model

This section introduces our geometry-conditioned object insertion model and then describes the construction of PhysInsert-Bench.

Given an object-free background I , reference-object images $\mathcal { T } _ { r } = \{ I _ { r } ^ { i } \} _ { i = 1 } ^ { N _ { r } } ,$ and an instruction $p ,$ we aim to generate the with-object image $I _ { + }$ while preserving object identity and adapting both the object and its environmental efects to the target scene.

## 4.1 Geometry-Conditioned Object Insertion

Our method combines Geometry-Guided Adaptation (GGA) for scene-consistent object adaptation with 2D Object–Scene Spatial Supervision (2D-OSS) for object regions, environmental changes, and spatial-text alignment (Figure 3). In our Qwen-Image-Edit-2511 backbone, the image stream $U _ { 0 }$ contains the noisy target latent and frozen-VAE encodings of the complete background and references; a frozen multimodal encoder produces text tokens $T _ { 0 }$ from the instruction and image context. Implementation and loss details are provided in Appendix $\mathrm { E }$

## 4.1.1 Geometry-Guided Adaptation

To guide scene-consistent object adaptation, GGA initializes a dynamic geometry stream from background features $F _ { - }$ extracted by frozen VGGT (Wang et al., 2025a). An MLP $A _ { \phi }$ maps these features to the DiT hidden space, with a learned geometry-modality embedding $e _ { G }$ . Geometry,

image, and text states interact through L three-stream DiT blocks:

$$
\begin{array} { r l } { G _ { 0 } = A _ { \phi } ( F _ { - } ) + e _ { G } , } & { { } } \\ { ( U _ { \ell + 1 } , T _ { \ell + 1 } , G _ { \ell + 1 } ) = B _ { \ell } ( U _ { \ell } , T _ { \ell } , G _ { \ell } ) , } & { { } \quad \ell = 0 , \ldots , L - 1 . } \end{array}\tag{1}
$$

At each block, the three streams compute their own queries, ke $^ { \prime \mathrm { { S } , } }$ and values, concatenate them for joint attention, and update through stream-specific modulation, output projections, and feed forward paths. The geometry states thus evolve through interaction with the target-image, background, reference, and text conditions throughout generation.

Background features describe the scene before insertion and alone do not specify its geometry after adding the object. We therefore extract target features $F _ { + }$ using the same frozen VGGT and train a shared readout $P _ { \psi }$ to predict them from geometry states at selected layers $ { s } _ { \mathrm { g e o } }$ . This supervision guides the dynamic geometry stream toward the target scene structure, complementing the initial background prior. The Geometry Alignment Loss is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { g e o } } = \displaystyle \sum _ { \ell \in S _ { \mathrm { g e o } } } \omega _ { \ell } \mathcal { D } _ { \mathrm { g e o } } ( P _ { \psi } ( G _ { \ell } ) , F _ { + } ) , } \\ & { \mathcal { D } _ { \mathrm { g e o } } ( A , B ) = \mathrm { S L } 1 _ { \beta _ { \mathrm { g e o } } } ( A , \mathrm { L N } ( B ) ) + \lambda _ { \mathrm { c o s } } \mathrm { C o s D i s t } ( A , \mathrm { L N } ( B ) ) . } \end{array}\tag{2}
$$

Here A and B denote predicted and target features, LN is layer normalization, $\beta _ { \mathrm { g e o } }$ controls the Smooth L1 transition, and $\lambda _ { \mathrm { { c o s } } }$ weights cosine distance; ω are normalized nonnegative layer weights. Target features are stop-gradient supervision only and are never injected into the generator.

## 4.1.2 2D Object–Scene Spatial Supervision

Complementing GGA’s geometric feature supervision, 2D-OSS directly constrains where the inserted object and its visible environmental efects should appear in the image, while aligning spatial instructions with the object region.

Counterfactual Object–Scene Supervision (COSS). We downsample the ground-truth object mask to the target-token grid to obtain a soft object occupancy map $M _ { o }$ . From the aligned counterfactual pair, we compute a smoothed diference map D and exclude the object region to construct the external-efect map $M _ { e }$

$$
\begin{array} { r l } & { \cal { D } = \mathrm { A v g P o o l } _ { \kappa \times \kappa } ( \mathrm { M e a n } _ { \mathrm { R G B } } | \mathrm { D o w n } ( I _ { + } ) - \mathrm { D o w n } ( I _ { - } ) | ) , } \\ & { \cal { M } _ { e } = \mathrm { c l i p } \bigg ( \frac { D - \delta _ { \mathrm { l o w } } } { \delta _ { \mathrm { h i g h } } - \delta _ { \mathrm { l o w } } } , 0 , 1 \bigg ) \left[ 1 - \mathrm { D i l a t e } _ { \rho _ { d } } \big ( { \bf 1 } [ M _ { o } > \delta _ { \mathrm { o b j } } ] \big ) \right] . } \end{array}\tag{3}
$$

Here Down maps images to the target-token grid, Mean<sub>RGB</sub> averages channels, and κ controls smoothing. Thresholds $\delta _ { \mathrm { l o w } } < \delta _ { \mathrm { h i g h } }$ map diferences to soft efect strengths; $\delta _ { \mathrm { o b j } }$ and dilation radius $\rho _ { d }$ exclude the object and its boundary neighborhood. Thus, $M _ { e }$ provides proxy labels for visible changes such as shadows and reflections, rather than a semantic decomposition of optical efects.

A shared spatial head operates on target-image tokens at layers $S _ { \mathrm { c f } } .$ averaging its logits across layers before producing object and efect probability maps. The Object Occupancy Loss $\mathcal { L } _ { o }$ and Scene Efect Loss $\mathcal { L } _ { e }$ each combine balanced binary cross-entropy and Dice losses, forming the Counterfactual Supervision Loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c f } } = \alpha _ { o } \mathcal { L } _ { o } + \alpha _ { e } \mathcal { L } _ { e } , \qquad \alpha _ { o } , \alpha _ { e } \geq 0 , \quad \alpha _ { o } + \alpha _ { e } = 1 . } \end{array}\tag{4}
$$

Soft-Mask Spatial Grounding (SSG). SSG grounds spatial instructions in the object region. The spatial description is a dedicated field of the generated caption (Figure 2); we use this field to identify its corresponding text-token subset $\mathcal { T } _ { s }$ in the instruction. At layer $\ell$ and head $h ,$ let $q _ { i } ^ { ( \ell , h ) }$ denote the query at target-image position i and $k _ { j } ^ { ( \ell , h ) }$ the key of spatial-text token $j .$ Their relevance is aggregated over phrase tokens using log-mean-exp and then averaged over heads:

$$
r _ { i } ^ { ( \ell ) } = \frac { 1 } { N _ { h } } \sum _ { h = 1 } ^ { N _ { h } } \log \left[ \frac { 1 } { | T _ { s } | } \sum _ { j \in \mathcal { T } _ { s } } \exp \left( \frac { q _ { i } ^ { ( \ell , h ) } \cdot k _ { j } ^ { ( \ell , h ) } } { \sqrt { d _ { h } } } \right) \right] ,\tag{5}
$$

where $N _ { h }$ is the number of heads and $d _ { h }$ the query/key dimension. With temperature $\tau _ { a } > 0$ , we normalize relevance over target-image positions and the soft object map over the same grid:

$$
P _ { r } ^ { ( \ell ) } ( i ) = \mathrm { s o f t m a x } _ { i } ( r _ { i } ^ { ( \ell ) } / \tau _ { a } ) , \qquad P _ { o } ( i ) = \frac { M _ { o } ( i ) } { \sum _ { i ^ { \prime } } M _ { o } ( i ^ { \prime } ) } .\tag{6}
$$

The Object Grounding Loss aligns these distributions over the supervised layer set $S _ { \mathrm { g r o u n d } }$

$$
\mathcal { L } _ { \mathrm { g r o u n d } } = \frac { 1 } { | S _ { \mathrm { g r o u n d } } | } \sum _ { \ell \in S _ { \mathrm { g r o u n d } } } D _ { \mathrm { K L } } \Big ( P _ { o } \Big \| P _ { r } ^ { ( \ell ) } \Big ) .\tag{7}
$$

Together with COSS, which supervises object occupancy and external efects, this alignment connects textual placement requirements to the region occupied by the inserted object.

Overall Objective. We combine COSS and SSG as the 2D Object–Scene Spatial Supervision Loss and optimize

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { 2 D } } = \lambda _ { \mathrm { c f } } \mathcal { L } _ { \mathrm { c f } } + \lambda _ { \mathrm { g r o u n d } } \mathcal { L } _ { \mathrm { g r o u n d } } , } \\ { \mathcal { L } = \mathcal { L } _ { \mathrm { f l o w } } + \lambda _ { \mathrm { g e o } } \mathcal { L } _ { \mathrm { g e o } } + \mathcal { L } _ { \mathrm { 2 D } } . } \end{array}\tag{8}
$$

with nonnegative weights $\lambda _ { \mathrm { g e o } } , \lambda _ { \mathrm { c f } } ,$ and $\lambda _ { \mathrm { g r o u n d } }$ . Text-guided inference uses $I _ { - } , \ z _ { r } .$ , and $p .$ For box-guided inference, we additionally supply a reference image annotated with the target bounding box as visual conditioning. All three streams remain active in both settings; target images, object masks, and auxiliary heads are used only during training.

## 4.2 Benchmark Construction

As shown in Figure 2(b), PhysInsert-Bench evaluates optical and spatial–pose adaptation using aligned counterfactual scene pairs. Optical cases emphasize illumination diferences and visible object-induced scene changes; their challenge cases and Structural cases additionally provide independent reference viewpoints to test pose adaptation. Each case is annotated for bidirectional interactions in lighting, spatial relationships, and pose, with physical-efect labels and evaluation focus tags identifying the efects emphasized by that case.

## 5 Experiments

## 5.1 Experimental Setup

Implementation Details. Our method builds upon Qwen-Image-Edit-2511, a DiT-based image edit ing model, retaining its pretrained Qwen2.5-VL (Bai et al., 2025) multimodal encoder and VAE and adding a VGGT geometry stream alongside the image and text streams. We fine-tune the DiT using LoRA (Hu et al., 2022) with a rank of 32. Training is performed with a batch size of 8 and a maximum training-image pixel count of 1024<sup>2</sup>. We use the AdamW optimizer (Loshchilov & Hutter, 2019) with a learning rate of 10−<sup>4</sup> and a weight decay of 0.01. All experiments are conducted on eight NVIDIA L20X GPUs. We train our model in a single stage on our proposed InsertWorld dataset for approximately 16,000 steps. At inference, we use 40 denoising steps and a CFG (Ho & Salimans, 2022) scale of 4.0. Full training and inference hyperparameters are provided in Appendix F.

Table 2 Text-guided results on PhysInsert-Bench using one object-reference image per case for all reported metrics. Phy-VLM obtains three valid ratings per output with GPT-5.5, retrying null responses, and reports mean ± 95% confidence-interval half-width. Unless otherwise specified, the following conventions apply to this and subsequent tables: +PhysW. denotes fine-tuning on our InsertWorld dataset with the same training budget as ours under the corresponding guidance setting; bold/underline mark the best/second-best across methods.
<table><tr><td>Method</td><td>DIoU (%)↑</td><td>Mask IoU (%)↑</td><td>DINO-Obj.↑</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>AbsRel↓</td><td>δ1.25 (%)↑</td><td>Phy-VLM↑</td></tr><tr><td colspan="9">Proprietary Models</td><td></td></tr><tr><td>Nano Banana Pro (Google DeepMind, 2025)</td><td>54.58</td><td>48.34</td><td>0.7023</td><td>22.85</td><td>0.8231</td><td>0.1527</td><td>46.55</td><td>0.2066</td><td>77.08</td><td>3.48 ± 0.09</td></tr><tr><td>GPT-Image-2 (OpenAI, 2026)</td><td>38.02</td><td>31.92</td><td>0.6746</td><td>19.39</td><td>0.7017</td><td>0.2949</td><td>53.46</td><td>0.3129</td><td>65.48</td><td>3.17 ± 0.11</td></tr><tr><td colspan="9">Open-Source Models</td><td></td></tr><tr><td>OmniPaint + PhysW. (Yu et al., 2025)</td><td>37.69</td><td>33.94</td><td>0.5828</td><td>20.67</td><td>0.8302</td><td>0.1733</td><td>73.54</td><td>0.2401</td><td>67.30</td><td>2.06 ± 0.13</td></tr><tr><td>Insert-Anything + PhysW. (Song et al., 2025)</td><td>22.66</td><td>21.46</td><td>0.6275</td><td>19.45</td><td>0.7711</td><td>0.3128</td><td>64.39</td><td>0.2651</td><td>67.79</td><td>2.36 ± 0.13</td></tr><tr><td>JoyAI-Image-Edit-Plus (Song et al., 2026)</td><td>46.37</td><td>39.21</td><td>0.6614</td><td>21.13</td><td>0.8211</td><td>0.1769</td><td>56.37</td><td>0.2308</td><td>75.70</td><td>3.04 ± 0.09</td></tr><tr><td>Qwen-Image-Edit-2511 (Wu et al., 2025)</td><td>48.32</td><td>44.14</td><td>0.6786</td><td>19.34</td><td>0.7840</td><td>0.2260</td><td>60.99</td><td>0.2207</td><td>73.35</td><td>2.94 ± 0.15</td></tr><tr><td>Ours</td><td>62.99</td><td>56.08</td><td>0.7276</td><td>24.40</td><td>0.8773</td><td>0.1118</td><td>41.96</td><td>0.1195</td><td>86.56</td><td>3.52 ± 0.11</td></tr></table>

Table 3 Box-guided results on PhysInsert-Bench. Ours is adapted from the text-guided model for an additional 4,000 box-guided steps; +PhysW. baselines use the same training budget with our method. † For PhysInsert-Bench, DIRECT uses GT-image-assisted pose fitting
<table><tr><td>Method</td><td>Mask IoU (%)↑</td><td>DINO-Obj.↑</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>AbsRel↓</td><td>δ1.25 (%)↑</td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>Nano Banana Pro (Google DeepMind, 2025)</td><td>52.94</td><td>0.7152</td><td>22.53</td><td>0.8056</td><td>0.1609</td><td>45.32</td><td>0.1846</td><td>81.03</td></tr><tr><td>GPT-Image-2 (OpenAI, 2026)</td><td>39.61</td><td>0.6798</td><td>21.45</td><td>0.7716</td><td>0.2049</td><td>50.68</td><td>0.2860</td><td>71.24</td></tr><tr><td colspan="9">Open-Source Models</td></tr><tr><td>AnyDoor (Chen et al., 2024a)</td><td>55.92</td><td>0.5379</td><td>20.42</td><td>0.8158</td><td>0.2413</td><td>63.50</td><td>0.2070</td><td>67.01</td></tr><tr><td>Insert-Anything (Song et al., 2025)</td><td>55.09</td><td>0.6507</td><td>22.70</td><td>0.8735</td><td>0.1608</td><td>52.23</td><td>0.1911</td><td>74.99</td></tr><tr><td>OmniPaint (Yu et al., 2025)</td><td>40.72</td><td>0.5441</td><td>20.60</td><td>0.8473</td><td>0.1402</td><td>59.95</td><td>0.2285</td><td>72.55</td></tr><tr><td>FreeCompose (Chen et al., 2024b)</td><td>55.62</td><td>0.5037</td><td>20.85</td><td>0.7666</td><td>0.3288</td><td>62.62</td><td>0.2497</td><td>62.70</td></tr><tr><td>PS-Diffusion (Wang et al., 2025b)</td><td>60.58</td><td>0.5098</td><td>20.41</td><td>0.7384</td><td>0.2511</td><td>60.35</td><td>0.2122</td><td>68.49</td></tr><tr><td>SHINE-Adapter (Lu et al., 2026)</td><td>52.07</td><td>0.5567</td><td>21.94</td><td>0.7910</td><td>0.1798</td><td>65.77</td><td>0.1976</td><td>78.18</td></tr><tr><td>DIRECT + GT Pose† (Gong et al., 2026)</td><td>65.86</td><td>0.6626</td><td>23.14</td><td>0.8957</td><td>0.1270</td><td>49.97</td><td>0.2180</td><td>68.05</td></tr><tr><td>Insert-Anything + PhysW. (Song et al., 2025) OmniPaint + PhysW. (Yu et al., 2025)</td><td>60.86 54.43</td><td>0.6991 0.5447</td><td>24.38 22.49</td><td>0.8759 0.8505</td><td>0.1554</td><td>43.80</td><td>0.1258</td><td>86.56</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.1290</td><td>57.85</td><td>0.1483</td><td>80.47</td></tr><tr><td>Qwen-Image-Edit-2511 (Wu et al., 2025)</td><td>37.39</td><td>0.5366</td><td>21.27</td><td>0.8605</td><td>0.1601</td><td>65.27</td><td>0.2654</td><td>71.85</td></tr><tr><td>Ours</td><td>68.62</td><td>0.7229</td><td>25.81</td><td>0.8876</td><td>0.1047</td><td>37.50</td><td>0.1064</td><td>89.41</td></tr></table>

Test Datasets. We evaluate on PhysInsert-Bench (§4.2), 40 DreamBooth (Ruiz et al., 2023) cases covering all objects with one reference per case, and 50 PhysInsert-Bench-Wild cases with multiview reference pools. The Wild set draws on ABO (Collins et al., 2022), NAVI v1.5 (Jampani et al., 2023), ORIDa (Kim et al., 2025), and real target images paired with reference views rendered from SAM 3D Objects (Chen et al., 2026) reconstructions. We favor reference poses that difer from the target pose to test cross-view adaptation; construction details are provided in Appendix I.2.

Table 4 Results on PhysInsert-Bench-Wild with one or three reference images, and DreamBooth with one reference image.
<table><tr><td rowspan="3">Method</td><td colspan="6">PhysInsert-Bench-Wild</td><td colspan="3">DreamBooth</td></tr><tr><td colspan="3">1-view</td><td colspan="3">3-view</td><td colspan="3">1-view</td></tr><tr><td>PSNR↑</td><td>FID↓</td><td>DINO-Obj.↑</td><td>PSNR↑</td><td>FID↓</td><td>DINO-Obj.↑</td><td>PSNR↑</td><td>FID↓</td><td>DINO-Obj.↑</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Nano Banana Pro (Google DeepMind, 2025) GPT-Image-2 (OpenAI, 2026)</td><td>15.78</td><td>123.2</td><td>0.7881</td><td>15.72</td><td>115.3</td><td>0.7909</td><td>15.58</td><td>83.74</td><td>0.8650</td></tr><tr><td></td><td>16.74</td><td>113.4</td><td>0.7457</td><td>16.88</td><td>112.9</td><td>0.7865</td><td>14.68</td><td>87.77</td><td>0.8821</td></tr><tr><td colspan="10">Open-Source Models</td></tr><tr><td>OmniPaint (Yu et al., 2025)</td><td>18.08</td><td>140.5</td><td>0.6393</td><td></td><td></td><td></td><td>15.10</td><td>115.8</td><td>0.7190</td></tr><tr><td>Insert-Anything (Song et al., 2025)</td><td>19.38</td><td>122.3</td><td>0.7536</td><td></td><td></td><td></td><td>16.95</td><td>90.53</td><td>0.8545</td></tr><tr><td>Qwen-Image-Edit-2511 (Wu et al., 2025)</td><td>12.87</td><td>208.1</td><td>0.5991</td><td>10.93</td><td>214.6</td><td>0.5787</td><td>16.58</td><td>86.17</td><td>0.8767</td></tr><tr><td>Ours</td><td>20.36</td><td>95.67</td><td>0.7784</td><td>20.60</td><td>87.90</td><td>0.7849</td><td>17.45</td><td>83.68</td><td>0.8856</td></tr></table>

Evaluation Metrics. We use PSNR and SSIM (Wang et al., 2004), LPIPS (Zhang et al., 2018), and FID (Heusel et al., 2017) to evaluate full-image fidelity, and DINO-Obj. to assess object identity preservation. Following PhyEdit (Xu et al., 2026), we use DIoU to measure object placement with a center-distance penalty, while Mask IoU measures visible-object overlap, including diferences in visible extent caused by occlusion. AbsRel and $\delta _ { 1 . 2 5 }$ assess estimated object-region depth consistency as a proxy for geometric plausibility. We also adapt the VLM-based physical plausibility evaluation of PhyEdit (Xu et al., 2026) as Phy-VLM, using GPT-5.5 with a fixed prompt template augmented by each case’s physical-efect annotations and evaluation-focus tags (§4.2). These tags direct attention to the physical efects emphasized by that case. Metric definitions and implementation details are provided in Appendix D.

## 5.2 Quantitative Results

Text-Guided Results. All metrics in Table 2 use one object-reference image per case. With placement specified through text instructions, PhysInsert achieves the highest DIoU and Mask IoU in Table 2, indicating more accurate object placement. It also leads the reported fidelity and identity metrics and achieves the best estimated depth consistency under the symmetric DA3 protocol. In the Phy-VLM evaluation, PhysInsert obtains the highest mean score.

Box-Guided Results. With one reference image per case, Table 3 shows that PhysInsert achieves the highest DINO-Obj. and PSNR, together with the lowest LPIPS and FID. Fine-tuning on InsertWorld (+PhysW.) improves all reported metrics for both Insert-Anything and OmniPaint under the same training budget, showing that our data also benefits existing insertion models.

Real-World Results. On PhysInsert-Bench-Wild and DreamBooth (Table 4), PhysInsert achieves the highest PSNR and lowest FID across the reported reference-view settings. Its DINO-Obj. scores lead the open-source methods throughout and all compared methods on DreamBooth.

User Study. In a blinded study, 27 participants rate 25 separate cases combining RealBokeh backgrounds and Distinctions-646 objects (Appendix K). PhysInsert receives the highest ratings for lighting integration, perspective, and pose, and the highest overall score of 4.04, while remaining competitive in identity and background preservation (Figure 4).

![](images/9ff1f6eda4dddc41193d01d9f8a8255f594c12e9b09ca80f21b79b6c7e81fdef.jpg)  
27 participants · 25 cases · Error bars: 95% participant-bootstrap C

Figure 4 User-study result. Error bars show 95% confidence intervals.  
![](images/8baccdbabc0934fbe469a19eb0c3282daffea8cdabd6604fe3f8b8365efa2836.jpg)  
Figure 5 Qualitative comparisons on PhysInsert-Bench (top), DreamBooth and PhysInsert-Bench-Wild (middle), and the user-study set (bottom).

## 5.3 Qualitative Results

Figure 5 compares PhysInsert-Bench optical and structural adaptation cases (top), DreamBooth and PhysInsert-Bench-Wild cases (middle), and user-study examples (bottom). PhysInsert better adapts the reflective containers and beverage can to scene lighting, including the can’s reflection in the adjacent glass. The tray and toy examples show orientation adaptation, with the toy’s cast shadow also matching the target scene more closely. User-study examples further illustrate foreground occlusion, perspective adaptation on an oblique window ledge, and object scale on a chair.

## 5.4 Ablation Studies

Quantitative Ablations. As Table 5, the full model improves PSNR, DIoU, and LPIPS over PhysW., demonstrating gains beyond dataset fine-tuning. Removing GGA, SSG, or COSS degrades all reported metrics, supporting each component’s contribution.

Qualitative Ablations. Figure 6(b) shows that SSG corrects the white vase’s placement, GGA adapts the car poster to the scene’s oblique perspective, and COSS strengthens its floor reflection compared with their respective ablations.

![](images/13faffbf7614afba0d28a0c312c7005bbc190bcec5fe1b6d51c4101e3cd92882.jpg)  
Figure 6 Mechanism probes and qualitative ablations. (a) Spatial-text attention, object/efect predictions, and VGGT feature evolution under identical GT-noised inputs (σ = 0.5). (b) Separate ablations of SSG, GGA, and COSS for placement and scale, perspective, and floor reflection. (c) Text-guided pose and placement control.

Table 5 Ablations on PhysInsert Bench.
<table><tr><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>DIoU (%)↑</td><td>DINO-Obj.↑</td><td>AbsRel↓</td><td> $\delta _ { 1 . 2 5 }$  (%)↑</td></tr><tr><td>Init.</td><td>19.34</td><td>0.7840</td><td>0.2260</td><td>60.99</td><td>48.32</td><td>0.6786</td><td>0.2207</td><td>73.35</td></tr><tr><td>PhysW.</td><td>22.53</td><td>0.8529</td><td>0.1442</td><td>48.61</td><td>60.19</td><td>0.7062</td><td>0.1372</td><td>84.18</td></tr><tr><td>w/o GGA</td><td>23.83</td><td>0.8702</td><td>0.1226</td><td>43.94</td><td>61.45</td><td>0.7206</td><td>0.1354</td><td>85.15</td></tr><tr><td>w/o SSG</td><td>23.73</td><td>0.8702</td><td>0.1203</td><td>44.97</td><td>62.51</td><td>0.7157</td><td>0.1287</td><td>86.06</td></tr><tr><td>w/o COSS</td><td>23.90</td><td>0.8722</td><td>0.1168</td><td>42.68</td><td>61.58</td><td>0.7232</td><td>0.1315</td><td>86.40</td></tr><tr><td>Full model</td><td>24.40</td><td>0.8773</td><td>0.1118</td><td>41.96</td><td>62.99</td><td>0.7276</td><td>0.1195</td><td>86.56</td></tr></table>

Spatial Attention. Under identical text, references, and GT-noised latents, the final model shows more object-focused spatial-text attention than SFT in Figure 6(a), supporting stronger alignment in this case.

Efect Localization. Under identical GT-noised inputs, the same frozen efect head yields a clearer response in the afected region from final-model features than from SFT features (Figure 6(a)).

Geometry Evolution. To inspect geometry-stream updates of GGA under the same GT-noised input, we compare auxiliary-head reconstructions from blocks 14, 29, and 59 with background and GT VGGT features in a shared PCA basis, revealing object-region feature changes alongside retained coarse scene structure in this example (Figure 6(a), bottom).

Text-Guided Control. Figure 6(c) further shows text-guided changes in laptop orientation and watering-can spout direction, together with placement behind a chair or on an upper shelf.

## 6 Conclusion

We presented PhysInsert, combining InsertWorld’s counterfactual scene pairs and references across viewpoints and lighting with a VGGT geometry stream and 2D object–scene supervision for joint object and scene adaptation. PhysInsert-Bench and PhysInsert-Bench-Wild evaluate controlled adaptation and real-world generalization. Experiments demonstrate improved image fidelity and placement, supporting joint modeling of objects and their surroundings for physically plausible insertion.

## References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025. URL https://arxiv.org/abs/2502.13923.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, et al. SAM 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025. URL https://arxiv.org/abs/2511.16719.

Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot objectlevel image customization. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6593–6602. IEEE, 2024a.

Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J. Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jia-Wei Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang, Jianing Yang, Bowen Zhang, Piotr Dollár, Georgia Gkioxari, Matt Feiszli, and Jitendra Malik. SAM 3D: 3Dfy anything in images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7220–7232, 2026. URL https://openaccess.thecvf.com/content/CVPR2026 /html/Chen\_SAM\_3D\_3Dfy\_Anything\_in\_Images\_CVPR\_2026\_paper.html.

Zhekai Chen, Wen Wang, Zhen Yang, Zeqing Yuan, Hao Chen, and Chunhua Shen. Freecompose: Generic zero-shot image composition with difusion prior. arXiv preprint arXiv:2407.04947, 2024b.

Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F. Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik. ABO: Dataset and benchmarks for real-world 3D object understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. URL https://amazon-berkeley-objec ts.s3.amazonaws.com/index.html.

Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum ?id=2dnO3LLiJ1.

Jingbo Gong, Yikai Wang, Yushi Lan, Yuhao Wan, Ziheng Ouyang, Rui Zhao, Ming-Ming Cheng, Qibin Hou, and Chen Change Loy. Direct 3d-aware object insertion via decomposed visual proxies. In ICML, 2026.

Google DeepMind. Gemini 3 Pro Image – Nano Banana Pro, 2025. URL https://deepmind.google/mode ls/gemini-image/pro/. Oficial model page.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://arxiv.org/abs/1706.08500.

Jonathan Ho and Tim Salimans. Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598, 2022. URL https://arxiv.org/abs/2207.12598.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, Andre Araujo, Ricardo Martin-Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, and Howard Zhou. NAVI: Category-agnostic image collections with high-quality 3D shape and pose annotations. In Advances in Neural Information Processing Systems, 2023. URL https://navidataset.github.io/.

Jinwoo Kim, Sangmin Han, Jinho Jeong, Jiwoo Choi, Dongyeoung Kim, and Seon Joo Kim. ORIDa: Objectcentric real-world image composition dataset. In Proceedings of the IEEE/CVF Conference on Computer

Vision and Pattern Recognition, pp. 3051–3060, 2025. URL https://openaccess.thecvf.com/content/ CVPR2025/html/Kim\_ORIDa\_Object-centric\_Real-world\_Image\_Composition\_Dataset\_CVPR\_2025\_p aper.html.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. URL https://arxiv.org/abs/1412.6980.

Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. URL https://arxiv.org/abs/2511.10647.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://op enreview.net/forum?id=PqvMRDCJT9t.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. In Computer Vision – ECCV 2024, pp. 38–55, 2024. doi: 10.1007/978-3-0 31-72970-6\_3. URL https://www.ecva.net/papers/eccv\_2024/papers\_ECCV/html/6319\_ECCV\_2024\_ paper.php.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://arxiv.org/abs/1711.05101.

Shilin Lu, Zhuming Lian, Zihan Zhou, Shaocong Zhang, Chen Zhao, and Adams Wai-Kin Kong. Does FLUX already know how to perform physically plausible image composition? In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2509.21278.

Oscar Michel, Anand Bhattad, Eli VanderBilt, Ranjay Krishna, Aniruddha Kembhavi, and Tanmay Gupta. Object 3dit: Language-guided 3d-aware image editing. Advances in Neural Information Processing Systems, 36:3497–3516, 2023.

OpenAI. GPT-Image-2, 2026. URL https://developers.openai.com/api/docs/models/gpt-image-2. Oficial model documentation.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. URL https://arxiv.org/abs/23 04.07193.

William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023. URL https://arxi v.org/abs/2212.09748.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dream-Booth: Fine tuning text-to-image difusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. URL https://arxiv.org/ abs/2208.12242.

Lin Song, Wenbo Li, Guoqing Ma, Wei Tang, Bo Wang, Yuan Zhang, Yijun Yang, Yicheng Xiao, Jianhui Liu, Yanbing Zhang, et al. JoyAI-Image: Awaking spatial intelligence in unified multimodal understanding and generation. arXiv preprint arXiv:2605.04128, 2026. URL https://arxiv.org/abs/2605.04128.

Wensong Song, Hong Jiang, Zongxing Yang, Ruijie Quan, and Yi Yang. Insert anything: Image insertion via in-context editing in dit. arXiv preprint arXiv:2504.15009, 2025.

Xueyun Tian, Wei Li, Bingbing Xu, Yige Yuan, Yuanzhuo Wang, and Huawei Shen. MIGE: Mutually enhanced multimodal instruction-based image generation and editing. arXiv preprint arXiv:2502.21291, 2025. URL https://arxiv.org/abs/2502.21291.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025a. URL https://arxiv.org/abs/2503.11651.

Weicheng Wang, Guoli Jia, Zhongqi Zhang, Liang Lin, and Jufeng Yang. PS-Difusion: Photorealistic subject-driven image editing with disentangled control and attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18302–18312, June 2025b. URL https://openaccess.thecvf.com/content/CVPR2025/html/Wang\_PS-Diffusion\_Photorealistic\_Su bject-Driven\_Image\_Editing\_with\_Disentangled\_Control\_and\_Attention\_CVPR\_2025\_paper.html.

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004. doi: 10.1109/TIP.2003.819861. URL https://www.cns.nyu.edu/\~lcv/ssim/.

Daniel Winter, Matan Cohen, Shlomi Fruchter, Yael Pritch, Alex Rav-Acha, and Yedid Hoshen. Objectdrop: Bootstrapping counterfactuals for photorealistic object removal and insertion. In European Conference on Computer Vision, pp. 112–129. Springer, 2024.

Daniel Winter, Asaf Shul, Matan Cohen, Dana Berman, Yael Pritch, Alex Rav-Acha, and Yedid Hoshen. ObjectMate: A recurrence prior for object insertion and subject-driven generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16281–16291, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-Image technical report, 2025. URL https://arxiv.org/abs/2508.02324.

Ruihang Xu, Dewei Zhou, Xiaolong Shen, Fan Ma, and Yi Yang. PhyEdit: Towards real-world object manipulation via physically-grounded image editing. arXiv preprint arXiv:2604.07230, 2026. URL https: //arxiv.org/abs/2604.07230.

Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with difusion models. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18381–18391. IEEE, 2023.

Yongsheng Yu, Ziyun Zeng, Haitian Zheng, and Jiebo Luo. Omnipaint: Mastering object-oriented editing via disentangled insertion-removal inpainting. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17324–17334. IEEE, 2025.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable efective ness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018. URL https://arxiv.org/abs/1801.03924.

Liangbing Zhao, Le Zhuo, Sayak Paul, Hongsheng Li, and Mohamed Elhoseiny. From statics to dynamics: Physics-aware image editing with latent transition priors. arXiv preprint arXiv:2602.21778, 2026. URL https://arxiv.org/abs/2602.21778.

Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren. Distance-IoU loss: Faster and better learning for bounding box regression. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020. URL https://arxiv.org/abs/1911.08287.

## Appendix for PhysInsert: Physically Plausible Object Insertion with Geometric Priors and Bidirectional Object-Scene Interactions

## A Limitations

PhysInsert does not explicitly optimize for object identity preservation. Its ability to preserve identity remains limited when the reference and target viewpoints difer substantially. Future work could better exploit complementary information across multiple reference views to preserve distinctive object features under large viewpoint changes.

## B Geometric Foundation Models

VGGT (Wang et al., 2025a) jointly predicts camera parameters, depth, and 3D scene structure, while Depth Anything 3 (Lin et al., 2025) recovers consistent geometry across views. Geometry alone does not determine insertion-induced shadows and reflections, motivating our combination of geometric priors and counterfactual supervision.

## C More Detailed Quantitative Results

Table S1 reports 180 Optical Adaptation cases, comprising 154 cases with aligned reference–target viewpoints and 26 cross-view Optical challenges, and 220 Pose Adaptation cases comprising 119 Spa tial and 101 Mechanical examples. Their single-reference outputs together form the 400-case evaluation in Table $^ { 3 ; }$ the three-reference setting covers the same 220 Pose Adaptation cases. +PhysW. baselines receive the same training budget. PhysInsert achieves the best PSNR, LPIPS, and FID on Optical Adaptation, and the best FID, DINO-Obj., and PSNR in both Pose Adaptation reference settings.

Table S1 Box-guided results on 180 Optical Adaptation and 220 Pose Adaptation cases.
<table><tr><td rowspan="3">Method</td><td colspan="4">Optical Adaptation</td><td colspan="6">Pose Adaptation</td></tr><tr><td colspan="3">1-view</td><td colspan="3">1-view</td><td colspan="3"></td></tr><tr><td>PSNR↑</td><td colspan="2">SSIM↑ LPIPS↓</td><td>FID↓ FID↓</td><td>DINO-Obj.↑</td><td>PSNR↑</td><td>FID↓</td><td>DINO-Obj.↑</td><td>PSNR↑</td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>Nano Banana Pro (Google DeepMind, 2025)</td><td>22.92</td><td>0.8153</td><td>0.1488</td><td>56.67 59.70</td><td>0.6886</td><td>22.20</td><td>60.85</td><td>0.6824</td><td>22.05</td></tr><tr><td>GPT-Image-2 (OpenAI, 2026)</td><td>22.01</td><td>0.8045 0.1785</td><td>66.55</td><td>64.31</td><td>0.6566</td><td>20.99</td><td>66.53</td><td>0.6404</td><td>21.06</td></tr><tr><td colspan="9">Open-Source Models</td></tr><tr><td>AnyDoor (Chen et al., 2024a)</td><td>21.08</td><td>0.8327</td><td>0.2148</td><td>82.09 84.99</td><td>0.5081</td><td>19.88</td><td></td><td></td><td></td></tr><tr><td>Insert-Anything (Song et al., 2025)</td><td>22.79</td><td>0.8768 0.1491</td><td>68.09</td><td>67.83</td><td>0.6197</td><td>22.62</td><td></td><td></td><td></td></tr><tr><td>Insert-Anything + PhysW. (Song et al., 2025)</td><td>24.44</td><td>0.8789 0.1460</td><td>57.10</td><td>55.32</td><td>0.6717</td><td>24.33</td><td>61.72</td><td>0.6192</td><td>24.07</td></tr><tr><td>OmniPaint (Yu et al., 2025)</td><td>20.65</td><td>0.8526</td><td>0.1337 80.98</td><td>77.46</td><td>0.5607</td><td>20.56</td><td></td><td></td><td></td></tr><tr><td>OmniPaint + PhysW. (Yu et al., 2025)</td><td>22.73</td><td>0.8617</td><td>0.1141</td><td>68.67 80.24</td><td>0.5138</td><td>22.29</td><td>86.20</td><td>0.4570</td><td>22.06</td></tr><tr><td>FreeCompose (Chen et al., 2024b)</td><td>20.88</td><td>0.7711</td><td>0.3176</td><td>83.33 79.59</td><td>0.4828</td><td>20.82</td><td></td><td></td><td></td></tr><tr><td>DIRECT + GT Pose (Gong et al., 2026)</td><td>23.19</td><td>0.8975</td><td>0.1142 67.29</td><td>61.85</td><td>0.6422</td><td>23.10</td><td>62.36</td><td>0.6338</td><td>23.45</td></tr><tr><td>Qwen-Image-Edit-2511 (Wu et al., 2025)</td><td>21.44</td><td>0.8674</td><td>0.1433</td><td>83.97 86.51</td><td>0.4951</td><td></td><td>21.13 146.9 44.42</td><td>0.4406</td><td>14.93</td></tr><tr><td>Ours</td><td>25.99</td><td>0.8942</td><td>0.09989</td><td>49.82</td><td>45.62</td><td>0.7021</td><td>25.67</td><td>0.7146</td><td>25.75</td></tr></table>

## D Evaluation Metric Definitions

Let $\widehat { I } _ { + }$ and $I _ { + }$ denote a generated image and its GT target. Table S2 summarizes the evaluation bscope and preferred direction. Except for dataset-level FID, metrics are computed per case and averaged equally over the corresponding evaluation set; DIoU, Mask IoU, and $\delta _ { 1 . 2 5 }$ are reported as

percentages by multiplying their mean scores by 100. Reference-view settings and evaluation sets follow the respective result tables. All metrics in Table 2 use one object-reference image per case.

Table S2 Evaluation metrics and their scope.
<table><tr><td>Metric</td><td>Evaluation scope</td><td>Preferred</td></tr><tr><td>PSNR</td><td>Full-image pixel fidelity</td><td>Higher</td></tr><tr><td>SSIM</td><td>Full-image structural similarity</td><td>Higher</td></tr><tr><td>LPIPS</td><td>Full-image perceptual distance</td><td>Lower</td></tr><tr><td>FID</td><td>Full-image feature distributions</td><td>Lower</td></tr><tr><td>DIoU (%)</td><td>Mask-box overlap and center distance</td><td>Higher</td></tr><tr><td>Mask IoU (%)</td><td>Visible foreground overlap</td><td>Higher</td></tr><tr><td>DINO-Obj.</td><td>Object-crop identity similarity</td><td>Higher</td></tr><tr><td>Phy-VLM</td><td>Physical plausibility with case-specific focus</td><td>Higher</td></tr><tr><td>AbsRel</td><td>Valid object-region depth error</td><td>Lower</td></tr><tr><td> $\delta _ { 1 . 2 5 } ~ ( \% )$ </td><td>Valid object-region depth agreement</td><td>Higher</td></tr></table>

## D.1 Full-Image Fidelity

PSNR. Peak signal-to-noise ratio measures pixel fidelity over the complete RGB image. For images of size $H \times W$ and pixel dynamic range $R ,$ it is

$$
\begin{array} { l } { { \mathrm { M S E } = \displaystyle \frac { 1 } { 3 H W } \sum _ { u = 1 } ^ { H } \sum _ { v = 1 } ^ { W } \sum _ { c = 1 } ^ { 3 } \left( \widehat { I } _ { + } ( u , v , c ) - I _ { + } ( u , v , c ) \right) ^ { 2 } , } } \\ { { \mathrm { P S N R } = 1 0 \log _ { 1 0 } \left( \displaystyle \frac { R ^ { 2 } } { \mathrm { M S E } } \right) . } } \end{array}\tag{S1}
$$

Our implementation uses RGB values in [0, 255] and $R = 2 5 5$

SSIM. Structural similarity (Wang et al., 2004) compares local luminance, contrast, and structure. For corresponding local windows x and y from the generated and GT images,

$$
\mathrm { S S I M } ( x , y ) = \frac { ( 2 \mu _ { x } \mu _ { y } + C _ { 1 } ) ( 2 \sigma _ { x y } + C _ { 2 } ) } { ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + C _ { 1 } ) ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + C _ { 2 } ) } ,\tag{S2}
$$

where $\mu , \sigma ^ { 2 } ,$ , and $\sigma _ { x y }$ are local means, variances, and covariance; $C _ { 1 }$ and $C _ { 2 }$ stabilize the ratios. We use the scikit-image implementation with RGB channel averaging and a dynamic range of 255, averaging local scores over the image.

LPIPS. Learned Perceptual Image Patch Similarity (Zhang et al., 2018) measures perceptual distance using deep features. Let $\tilde { \phi } _ { \ell } ( I )$ denote channel-normalized features at layer ℓ and $w _ { \ell }$ the learned channel weights. Then

$$
\mathrm { L P I P S } ( \widehat { I } _ { + } , I _ { + } ) = \sum _ { \ell } \frac { 1 } { H _ { \ell } W _ { \ell } } \sum _ { u , v } \Big \Vert w _ { \ell } \odot \Big ( \widetilde { \phi } _ { \ell } ( \widehat { I } _ { + } ) _ { u v } - \widetilde { \phi } _ { \ell } ( I _ { + } ) _ { u v } \Big ) \Big \Vert _ { 2 } ^ { 2 } .\tag{S3}
$$

We use the AlexNet-based LPIPS implementation on complete images, with RGB inputs mapped to [−1, 1].

FID. Fréchet Inception Distance (Heusel et al., 2017) compares the feature distributions of generated and GT full-image sets. With feature means $\mu _ { g } , \mu _ { t }$ and sample covariances $\Sigma _ { g } , \Sigma _ { t }$

$$
\begin{array} { r } { \mathrm { F I D } = \Vert \mu _ { g } - \mu _ { t } \Vert _ { 2 } ^ { 2 } + \operatorname { T r } \biggr ( \Sigma _ { g } + \Sigma _ { t } - 2 \bigr ( \Sigma _ { g } ^ { 1 / 2 } \Sigma _ { t } \Sigma _ { g } ^ { 1 / 2 } \bigr ) ^ { 1 / 2 } \biggr ) . } \end{array}\tag{S4}
$$

FID is computed once over all generated and GT images in each reported evaluation set, including by concatenating features across evaluation shards; it is not an average of per-image scores. The implementation evaluates the covariance cross term through an equivalent low-rank singular-value decomposition.

## D.2 Text-Guided Placement Accuracy

Foreground extraction for Table 2. We apply one fixed evaluator to all methods on the same N = 400 original text-guided cases. Qwen3.5-Plus receives, in order, the input background, the object-reference collage, and the generated image. Using the editing instruction and a component vocabulary fixed per case across methods, it localizes all visible newly inserted target instances and parts, excluding pre-existing background objects, cast shadows, and reflections. Reference panels depict views of the same target assembly rather than additional copies. The grounder returns component boxes in xyxy coordinates normalized to [0, 1000], with temperature 0 and a requested maximum of 8192 output tokens. Neither GT RGB images nor GT masks or boxes are provided to localization or segmentation.

RGB images are resized bilinearly to a shared canvas preserving the background aspect ratio, with longest side at most 1024 pixels. The reference collage preserves its own aspect ratio with the same size cap. VLM inputs are encoded as RGB JPEGs with quality 95 and no chroma subsampling. SAM 3 (Carion et al., 2025) (facebook/sam3) generates candidates for each component phrase, using score and mask-probability thresholds of 0.5 and discarding masks smaller than 16 pixels. For each VLM box, we select every same-component candidate whose tight-box center lies inside that box and at least half of whose mask pixels lie inside it. If no candidate meets both conditions, we select the existing same-component candidate with highest box DIoU to the VLM box, retaining negative match scores. If that component has no candidates, we retry its category head noun (preserving “bean bag” as a phrase); if still unmatched, SAM3’s box-prompt tracker supplies a single mask with logit threshold 0. Selected masks are united into $\widehat { M } _ { n }$ . Model weights remain in cfloat32 with bfloat16 autocasting and TF32 enabled. There is no GT-based mask selection, permethod tuning, or manual mask correction. Automatic localization and segmentation can still omit parts or select background regions.

Mask IoU. Let $M _ { n }$ be the GT visible-object mask, resized to the same canvas by nearest-neighbor interpolation and binarized at values greater than 127. We report

$$
\mathrm { M a s k I o U } ( \% ) = \frac { 1 0 0 } { N } \sum _ { n = 1 } ^ { N } \frac { \lvert \widehat { M } _ { n } \cap M _ { n } \rvert } { \lvert \widehat { M } _ { n } \cup M _ { n } \rvert + 1 0 ^ { - 6 } } .\tag{S5}
$$

This measures foreground overlap rather than enclosing-box overlap. The GT visible mask is used only at this scoring stage.

DIoU. Let ${ \widehat { B } } _ { n }$ and $B _ { n }$ be tight, axis-aligned boxes of $\widehat { M } _ { n }$ and $M _ { n } ,$ using half-open pixel coordinates.   
Let $\widehat { \mathbf { c } } _ { n }$ and $\mathbf { c } _ { n }$ denote their centers, and $d _ { n }$ cthe diagonal length of the smallest box enclosing both.

For a nonempty prediction, the distance-IoU score (Zheng et al., 2020) is

$$
\begin{array} { l } { { \displaystyle u _ { n } = \frac { | \widehat B _ { n } \cap B _ { n } | } { | \widehat B _ { n } \cup B _ { n } | + 1 0 ^ { - 8 } } , } } \\ { { \displaystyle s _ { n } = u _ { n } - \frac { \| \widehat \mathbf { c } _ { n } - \mathbf { c } _ { n } \| _ { 2 } ^ { 2 } } { d _ { n } ^ { 2 } + 1 0 ^ { - 8 } } , \quad \quad \mathrm { D I o U } ( \% ) = \frac { 1 0 0 } { N } \sum _ { n = 1 } ^ { N } s _ { n } . } } \end{array}\tag{S6}
$$

An empty predicted mask receives zero for both metrics and remains in the 400-case average. Negative DIoU values are retained: a displaced box can incur a center-distance penalty larger than its overlap. Higher values are better for both metrics. The JoyAI row uses the existing text400 outputs from JoyAI-Image-Edit-Plus-Difusers, as in the depth evaluation below.

Tables 2 and 5 use the same percentage-scaled DIoU definition in Eq. equation S6.

## D.3 Object Identity Preservation

DINO-Obj. DINO-Obj. measures identity preservation by comparing object crops from the generated and GT images. We use the evaluator’s GroundingDINO-selected (Liu et al., 2024) prediction box and the GT object box, expand both with the same crop factor, and crop the respective RGB images while retaining context inside each box. The crop evaluator uses detection confidence and the fraction of changed pixels relative to $I _ { - }$ to select its prediction box; the SAM3-based placement metrics above are evaluated separately and do not change these DINO-Obj. crops. The crop expansion factor is 1.1. Let $C _ { g , n }$ and $\boldsymbol { C } _ { t , n }$ denote the generated-image and GT object crops for case $n ,$ and let $f$ extract the CLS feature from frozen DINOv2 (Oquab et al., 2023) with registers (Darcet et al., 2024) (facebook/dinov2-with-registers-base). We define the per-case score $s _ { n } ^ { \mathrm { D I N O } }$ and the dataset average as

$$
\begin{array} { c } { { \displaystyle s _ { n } ^ { \mathrm { D I N O } } = \frac { f ( C _ { g , n } ) ^ { \top } f ( C _ { t , n } ) } { \| f ( C _ { g , n } ) \| _ { 2 } \| f ( C _ { t , n } ) \| _ { 2 } } , } } \\ { { \displaystyle { \mathrm { D I N O - O b j . } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { n } ^ { \mathrm { D I N O } } } . } } \end{array}\tag{S7}
$$

Here $n \in \{ 1 , \ldots , N \}$ indexes all evaluated cases, including missed detections, for which $s _ { n } ^ { \mathrm { D I N O } } = 0$ Features are ℓ -normalized before computing the similarity. The comparison uses the GT targetobject crop rather than the reference-view collage, and pixels outside the object silhouette but inside the crop are retained.

## D.4 Object-Region Depth Consistency

Following the depth-metric definitions and paired-input procedure described in PhyEdit (Xu et al., 2026), we use frozen Depth Anything 3 (DA3NESTED-GIANT-LARGE) (Lin et al., 2025). Let $I _ { - }$ be the background, $I _ { + }$ the GT target, and $\widehat { I } _ { + }$ the generated image. We estimate depths in two separate forward passes:

$$
\begin{array} { r } { ( D _ { - } ^ { g } , D _ { g } ) = \mathrm { D A } 3 ( [ I _ { - } , I _ { + } ] ) , } \\ { ( D _ { - } ^ { p } , D _ { p } ) = \mathrm { D A } 3 ( [ I _ { - } , \widehat { I } _ { + } ] ) . } \end{array}\tag{S8}
$$

Both branches use the same model weights, preprocessing, and per-case random seed, and we compare the second depth output from each pair. The generated-image branch receives only the background and generated image; the GT image is used to obtain the reference depth for scoring.

All RGB inputs are resized to a common canvas preserving the target aspect ratio, with a longest edge of 768 pixels; the GT object mask uses nearest-neighbor resizing. When needed, depth outputs are resized to this canvas by bilinear interpolation and validity masks by nearest-neighbor interpolation. For case n, let $d _ { n , i } = D _ { g , n } ( i )$ and $\hat { d } _ { n , i } = D _ { p , n } ( i )$ . The evaluation region $\Omega _ { n }$ is the GT object mask intersected with the non-sky validity masks from both outputs, retaining only pixels with finite $d _ { n , i } , \hat { d } _ { n , i } > 1 0 ^ { - 6 }$ . Corresponding pixel coordinates are compared directly, without additional depth-scale or ofset fitting.

AbsRel and $\delta _ { 1 . 2 5 }$ . For each case, absolute relative error measures the mean absolute depth diference relative to the estimated target depth, and threshold accuracy measures the fraction of valid object pixels passing the ratio threshold:

$$
\begin{array} { r l r } & { \mathrm { A b s R e l } _ { n } = \displaystyle \frac { 1 } { | \Omega _ { n } | } \sum _ { i \in \Omega _ { n } } \frac { | \hat { d } _ { n , i } - d _ { n , i } | } { d _ { n , i } } , } & \\ & { } & { \delta _ { 1 . 2 5 , n } = \displaystyle \frac { 1 } { | \Omega _ { n } | } \sum _ { i \in \Omega _ { n } } \mathbf { 1 } \left[ \operatorname* { m a x } \left( \frac { \hat { d } _ { n , i } } { d _ { n , i } } , \frac { d _ { n , i } } { \hat { d } _ { n , i } } \right) < 1 . 2 5 \right] . } \end{array}\tag{S9}
$$

The threshold is strict: pixels with a depth ratio equal to 1.25 do not pass. Lower AbsRel and higher $\delta _ { 1 . 2 5 }$ indicate better agreement.

Within each setting, every method is evaluated on the same N = 400 cases: the text-guided outputs for Table 2, and the single-view box-guided outputs for Table 3 (180 Optical Adaptation and 220 Pose Adaptation cases). All cases have valid object pixels. We report equally weighted per-image averages:

$$
{ \mathrm { A b s R e l } } = { \frac { 1 } { N } } \sum _ { n = 1 } ^ { N } { \mathrm { A b s R e l } } _ { n } , \qquad \delta _ { 1 . 2 5 } ( \% ) = { \frac { 1 0 0 } { N } } \sum _ { n = 1 } ^ { N } \delta _ { 1 . 2 5 , n } .\tag{S10}
$$

The JoyAI row uses the existing text400 outputs from JoyAI-Image-Edit-Plus-Difusers for depth evaluation. Both target and generated depths are DA3 estimates. These scores therefore measure estimated depth consistency and remain sensitive to depth-estimator errors and paired-image context.

## D.5 VLM-Based Physical Plausibility

We adapt the VLM-based physical plausibility assessment in PhyEdit (Xu et al., 2026) to PhysInsert Bench, using GPT-5.5 to evaluate the outputs of all compared methods on the same 400 cases. Each evaluation prompt combines a fixed template with the case’s annotated physical efects and evaluation-focus tags from Appendix I.1. The tags specify which physical efects warrant particular attention, such as illumination adaptation, cast shadows, reflections, or occlusion relationships, as applicable to the case. The same template and case-specific tag information are used across methods. For each method and case, we obtain three valid numerical ratings, retrying any evaluation that returns null until three valid ratings are available. No case is excluded because of a null response. We first average the three valid ratings within each case, then average these case-level scores equally over all 400 cases. We report the resulting Phy-VLM mean with the half-width of its 95% confidence interval; higher values indicate greater physical plausibility as judged by the VLM.

Scoring prompt. The following is an English translation of the scoring prompt used for Phy-VLM in Table 2. The placeholders contain the case-specific evaluation focus and the original insertion instruction.

Phy-VLM scoring prompt (English translation)

Evaluate the physical plausibility of the edited result relative to the ground truth (GT). Do not evaluate aesthetics, identity preservation, or pixel similarity, and do not speculate about the generation method.

Image 1: edited result; Image 2: GT; Image 3: original background; Image 4: reference object (multiple views still depict the same object). The background and reference object are provided only to aid understanding

First compare against the GT and separately verify the physical relationships in the GT and the result. Do not assume that a relationship present in the GT is also present in the result.

Evaluation focus: {{CASE\_FOCUS\_TEXT}}

Original insertion instruction (in this instruction, Image 1 refers to the background and Image 2 to the reference object, rather than the scoring-image order above): {{ORIGINAL\_INSTRUCTION}}

Incorrect object size alone does not constitute physical implausibility: do not deduct points merely because the object is larger or smaller than in the GT, relative to neighboring objects, or than specified by the instruction. This rule takes precedence over size requirements in the case annotations and the instruction.

Evaluate only gravitational relationships (support, contact, and hanging), optical relationships (illumination, environmental color, shadows, reflections, and transmission), and geometric relationships (perspective, orientation, occlusion, and interpenetration). Do not separately penalize object size or changes in shadow or reflection extent that naturally result from $\mathrm { i t } ;$ do not infer errors in these relationships solely from a size diference.

Compare with the GT according to the evaluation focus:

5: Key relationships are confirmed to be consistent or physically equivalent.

4: Key relationships are retained, with only minor local deviations.

3: The main relationships hold, but important efects clearly deviate.

2: Key relationships deviate substantially, with only partial consistency.

1: Core relationships are missing, reversed, or fundamentally broken.

Large discrepancies from the GT in key physical relationships must receive a low score; appearance or size diferences alone do not count.

A clearly visible, severe gravitational, optical, or geometric error outside the stated focus caps the score at 2. Assign 1 for an entirely unsupported floating object, a disconnected primary load-bearing contact, obvious interpenetration, or impossible occlusion. Minor issues outside the focus do not reduce the score. Proximity or overlapping contours do not establish load-bearing support; any deduction must be supported by visible evidence.

Return null when key evidence is unclear or the GT contains an obvious error that afects the judgment; do not automatically treat such cases as good or bad.

Return only JSON: {"score": an integer from 1 to 5 or null, "reason": "One sentence identifying the key physical diference between the result and the GT, or stating the evidence for consistency or the need for review."}.

## E Method Details and Notation

We use Z for latents, $U _ { \ell } , T _ { \ell } .$ and $G _ { \ell }$ for image-, text-, and geometry-stream states, and $F$ for VGGT features. $N _ { r }$ is the number of reference images, L the number of three-stream DiT blocks, and ℓ the layer/state index; $B _ { \ell }$ denotes the corresponding update block. The sets $\mathcal { S } _ { \mathrm { g e o } } , \mathcal { S } _ { \mathrm { c f } } .$ , and $S _ { \mathrm { g r o u n d } }$ select the layers used for geometry readout, counterfactual spatial prediction, and grounding supervision, respectively; |S| denotes a set’s cardinality.

## E.1 Latent Encoding and Flow Matching

A frozen VAE encodes the target image $I _ { + }$ , background $I _ { - }$ , and each reference $I _ { r } ^ { i }$ into $Z _ { 0 } , Z _ { - }$ , and $Z _ { r } ^ { i }$ , respectively. The reference latents are concatenated as $Z _ { r } = [ Z _ { r } ^ { 1 } ; \dots ; Z _ { r } ^ { N _ { r } } ]$ . A frozen multimodal encoder produces the initial text tokens $T _ { 0 }$ from instruction $p$ and the image context. The image stream is initialized as $U _ { 0 } = [ Z _ { t } ; Z _ { - } ; Z _ { r } ]$ . During training, we sample noise $\epsilon ,$ form the noisy target latent, and optimize

$$
\begin{array} { r } { \begin{array} { r l } & { \quad \sum _ { t } = ( 1 - \sigma _ { t } ) Z _ { 0 } + \sigma _ { t } \epsilon , } \\ & { \quad \quad \mathscr { L } _ { \mathrm { f l o w } } = \mathbb { E } _ { t , \epsilon } \Bigl [ w ( t ) \left\| v _ { \theta } ( Z _ { t } ; I _ { - } , \mathscr { T } _ { r } , p ) - ( \epsilon - Z _ { 0 } ) \right\| _ { 2 } ^ { 2 } \Bigr ] . } \end{array} } \end{array}\tag{S11}
$$

Here t is the sampled difusion timestep, $\sigma _ { t }$ the noise-mixing coeficient, $w ( t )$ the timestep-dependent loss weight, and $v _ { \theta }$ the predicted velocity field. The coeficients $\sigma _ { t }$ and w(t) follow the backbone’s flow-matching scheduler (Lipman et al., 2023). We adapt the image and text branches with LoRA and train the added geometry branch, while keeping the VAE and multimodal encoder frozen.

## E.2 Geometry Features and Alignment

The frozen VGGT (Wang et al., 2025a) extracts background features $F _ { - } = f _ { \mathrm { V G G T } } ^ { ( \ell _ { v } ) } ( I _ { - } )$ at layer $\ell _ { v } .$ The MLP $A _ { \phi }$ maps these $d _ { G }$ -dimensional features to the d<sub>D</sub>-dimensional DiT (Peebles & Xie, 2023) hidden space, and the learned geometry-modality embedding $e _ { G }$ gives $G _ { 0 } = A _ { \phi } ( F _ { - } ) + e _ { G }$ . Each stream computes its own queries, keys, and values, which are concatenated for joint attention, and updates its states through stream-specific modulation, output projections, and feed-forward paths. Motivated by similarities between depth and RGB features, the geometry branch reuses the image branch’s linear weights as a frozen base and applies independent low-rank adapters. The geometry states evolve alongside the image and text states through the $L$ three-stream blocks and their own residual updates.

The same frozen VGGT extracts target features $F _ { + } = \mathrm { s g } ( f _ { \mathrm { V G G T } } ^ { ( \ell _ { v } ) } ( I _ { + } ) )$ , where sg stops gradient propagation. A shared readout head $P _ { \psi }$ predicts these features from geometry states, with $\widehat F _ { + } ^ { ( \ell ) } =$ $P _ { \psi } ( G _ { \ell } )$ . The distance used in the main-text geometry objective is

$$
\begin{array} { r l } & { \displaystyle \mathcal { D } _ { \mathrm { g e o } } ( A , B ) = \mathrm { S L } 1 _ { \beta _ { \mathrm { g e o } } } ( A , \mathrm { L N } ( B ) ) } \\ & { \qquad \quad + \lambda _ { \mathrm { c o s } } \mathrm { C o s D i s t } ( A , \mathrm { L N } ( B ) ) , } \\ & { \displaystyle \mathcal { L } _ { \mathrm { g e o } } = \sum _ { \ell \in \mathcal { S } _ { \mathrm { g e o } } } \omega _ { \ell } \mathcal { D } _ { \mathrm { g e o } } \Big ( \widehat { F } _ { + } ^ { ( \ell ) } , F _ { + } \Big ) . } \end{array}\tag{S12}
$$

The projector and readout have trainable parameters $\phi$ and $\psi .$ . Here LN denotes layer normalization; $\beta _ { \mathrm { g e o } } > 0$ controls the transition between the quadratic and linear regions of Smooth L1, and $\lambda _ { \mathrm { c o s } }$ weights the cosine-distance term. The layer weights satisfy $\omega _ { \ell } \geq 0$ and $\begin{array} { r } { \sum _ { \ell \in S _ { \mathrm { g e o } } } \omega _ { \ell } = 1 } \end{array}$ . Target features serve only as training targets and are never injected into the generator.

## E.3 Counterfactual Targets and Spatial Prediction

We downsample the ground-truth object mask to the target-token grid to obtain the soft object occupancy map $M _ { o }$ . From aligned with/without images, the diference map $D$ and external-efect

map $M _ { e }$ are

$$
D = \mathop { \mathrm { A v g P o o l } _ { \kappa \times \kappa } } ( \mathop { \mathrm { M e a n _ { R G B } } } | \operatorname { D o w n } ( I _ { + } ) - \operatorname { D o w n } ( I _ { - } ) | ) ,\tag{S13}
$$

$$
M _ { e } = \mathrm { c l i p } \bigg ( { \frac { D - \delta _ { \mathrm { l o w } } } { \delta _ { \mathrm { h i g h } } - \delta _ { \mathrm { l o w } } } } , 0 , 1 \bigg ) \left[ 1 - \mathrm { D i l a t e } _ { \rho _ { d } } \big ( { \bf 1 } [ M _ { o } > \delta _ { \mathrm { o b j } } ] \big ) \right] .\tag{S14}
$$

All three maps are defined on the target-token grid. Down downsamples an image to this grid, Mean<sub>RGB</sub> averages the absolute diferences over RGB channels, and κ is the side length of the average-pooling kernel. The thresholds $\delta _ { \mathrm { l o w } } ~ < ~ \delta _ { \mathrm { h i g h } }$ map a diference magnitude to a soft efect strength: values below the lower threshold map to zero, values above the upper threshold map to one, and intermediate values are linearly rescaled. $\delta _ { \mathrm { o b j } }$ binarizes object occupancy, and $\mathrm { D i l a t e } _ { \rho _ { d } }$ expands that region by radius $\rho _ { d }$ in grid coordinates to exclude the object and its boundary neighborhood. $1 [ \cdot ]$ is an indicator and $\operatorname { c l i p } ( \cdot , 0 , 1 )$ clamps its argument to the unit interval. Thus, $M _ { e }$ is a proxy for visible changes outside the object, including cast shadows and reflections, rather than a semantic decomposition of individual optical efects.

An object–efect spatial head is shared across target-image tokens at layers in $\mathcal { S } _ { \mathrm { c f } }$ . After averaging its logits across layers, $\widehat { M _ { c } }$ denotes the predicted probability map for channel $c \in \{ o , e \}$ , corresponding cto object occupancy and external efects. Each channel combines balanced binary cross-entropy and Dice losses:

$$
\mathcal { L } _ { c } = \alpha _ { \mathrm { b c e } } \mathcal { L } _ { \mathrm { b c e } } ( \widehat { M } _ { c } , M _ { c } ) + \alpha _ { \mathrm { d i c e } } \mathcal { L } _ { \mathrm { d i c e } } ( \widehat { M } _ { c } , M _ { c } ) .\tag{S15}
$$

The nonnegative weights $\alpha _ { \mathrm { b c e } }$ and $\alpha _ { \mathrm { d i c e } }$ sum to one. The main-text counterfactual objective combines the two channels as $\mathcal { L } _ { \mathrm { c f } } = \alpha _ { o } \mathcal { L } _ { o } + \alpha _ { e } \mathcal { L } _ { e }$ , with $\alpha _ { o } , \alpha _ { e } \ge 0$ and $\alpha _ { o } + \alpha _ { e } = 1$

## E.4 Spatial-Text Attention and Mask Alignment

The generated caption contains a dedicated spatial-description field, shown in Figure 2. We use that field to identify the corresponding text-token subset $\mathcal { T } _ { s }$ in the instruction. Indices i and $i ^ { \prime }$ range over target-image token positions, whereas $j$ indexes $\mathcal { T } _ { s }$ . At layer ℓ and head h, $q _ { i } ^ { ( \ell , h ) }$ is the target-image query and $k _ { j } ^ { ( \ell , h ) }$ the spatial-text key; each has dimension $d _ { h }$ , and there are $N _ { h }$ heads. The spatial relevance score aggregates over the phrase tokens using log-mean-exp and then averages over heads:

$$
r _ { i } ^ { ( \ell ) } = \frac { 1 } { N _ { h } } \sum _ { h = 1 } ^ { N _ { h } } \log \left[ \frac { 1 } { | { \cal T } _ { s } | } \sum _ { j \in { \cal T } _ { s } } \exp \left( \frac { q _ { i } ^ { ( \ell , h ) } \cdot k _ { j } ^ { ( \ell , h ) } } { \sqrt { d _ { h } } } \right) \right] .\tag{S16}
$$

The positive temperature $\tau _ { a }$ controls the concentration of the attention distribution, whose softmax is taken over target-image positions only. We normalize the soft object mask on the same grid and compute

$$
P _ { r } ^ { ( \ell ) } ( i ) = \mathrm { s o f t m a x } _ { i } ( r _ { i } ^ { ( \ell ) } / \tau _ { a } ) , \qquad P _ { o } ( i ) = { \frac { M _ { o } ( i ) } { \sum _ { i ^ { \prime } } M _ { o } ( i ^ { \prime } ) } } ,\tag{S17}
$$

$$
\mathcal { L } _ { \mathrm { g r o u n d } } = \frac { 1 } { | S _ { \mathrm { g r o u n d } } | } \sum _ { \ell \in S _ { \mathrm { g r o u n d } } } D _ { \mathrm { K L } } \Big ( P _ { o } \Big \| P _ { r } ^ { ( \ell ) } \Big ) .\tag{S18}
$$

Here $D _ { \mathrm { K L } }$ denotes Kullback–Leibler divergence. This objective supervises the spatial distribution of image–text relevance. The object mask supplies training supervision and is not required at inference.

Overall loss weights. The overall coeficients $\lambda _ { \mathrm { g e o } } , \lambda _ { \mathrm { c f } }$ and $\lambda _ { \mathrm { g r o u n d } }$ weight the geometry, counterfactual, and grounding objectives relative to ${ \mathcal { L } } _ { \mathrm { f l o w } }$ . They are nonnegative and are not constrained to sum to one, unlike the normalized layer and channel weights within the auxiliary objectives. The 2D-OSS loss groups the weighted counterfactual and grounding terms as $\mathcal { L } _ { \mathrm { 2 D } } = \lambda _ { \mathrm { c f } } \mathcal { L } _ { \mathrm { c f } } + \lambda _ { \mathrm { g r o u n d } } \mathcal { L } _ { \mathrm { g r o u n d } }$ preserving these coeficients.

## F Training and Inference Hyperparameters

We use AdamW (Loshchilov & Hutter, 2019), which applies decoupled weight decay to the adaptivemoment updates of Adam (Kingma & Ba, 2015). Tables S3 and S4 report the training, inference, and auxiliary-supervision settings of the base PhysInsert model. The base PhysInsert model is trained in a single stage for approximately 16,000 optimizer updates using the joint objective in Eq. (8). The box-guided model is obtained by fine-tuning the text-guided model for an additional 4,000 steps. The +PhysW. baselines use the same training budget in each setting, including the additional adaptation for box-guided evaluation.

Table S3 Base-model training and inference hyperparameters. The additional box-guided training budget is listed separately.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td colspan="2">Base-model training</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Base learning rate</td><td>1 × 10−4</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Learning-rate scheduler</td><td>PyTorch ConstantLR (defaults)</td></tr><tr><td>Effective batch size</td><td>8</td></tr><tr><td>Batch size per GPU / accumulation steps</td><td>1/1</td></tr><tr><td>Training hardware</td><td>8× NVIDIA L20X</td></tr><tr><td>Mixed precision</td><td>BF16</td></tr><tr><td>LoRA rank (image, text, and geometry)</td><td>32</td></tr><tr><td>Maximum training-image pixels</td><td>1,048,576 (10242)</td></tr><tr><td>Training schedule</td><td>Single-stage joint optimization</td></tr><tr><td>Total base-model updates</td><td>Approximately 16,000</td></tr><tr><td colspan="2">Additional box-guided adaptation</td></tr><tr><td>Additional optimizer updates</td><td>4,000</td></tr><tr><td colspan="2">Inference</td></tr><tr><td>Classifier-free guidance (CFG) scale</td><td>4.0</td></tr><tr><td>Denoising steps</td><td>40</td></tr><tr><td>Sampler</td><td>Euler flow-matching solver</td></tr><tr><td>Noise schedule</td><td>Qwen-Image dynamic shift</td></tr><tr><td>Denoising strength</td><td>1.0</td></tr><tr><td>Negative prompt</td><td>Empty</td></tr><tr><td>Maximum output pixels</td><td>1,048,576 (10242)</td></tr></table>

We jointly optimize flow matching, geometry alignment, COSS, and SSG in a single training stage using Eq. (8). The auxiliary-loss weights are listed below; all layer indices are zero-based.

Table S4 Auxiliary-loss weights and module hyperparameters for single-stage training.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Flow Matching Loss coefficient</td><td>1</td></tr><tr><td>Geometry Alignment Loss,  $\lambda _ { \mathrm { g e o } }$ </td><td>0.02</td></tr><tr><td>Counterfactual Supervision Loss,  $\lambda _ { \mathrm { c f } }$ </td><td>0.01</td></tr><tr><td>Object Grounding Loss,  $\lambda _ { \mathrm { g r o u n d } }$ </td><td>0.005</td></tr><tr><td>Module settings</td><td></td></tr><tr><td>VGGT feature layer,  $\ell _ { v }$ </td><td>23</td></tr><tr><td>Feature / DiT dimensions,  $d _ { G } ~ / ~ d _ { D }$ </td><td>2048 / 3072</td></tr><tr><td>Projector / readout MLP hidden width</td><td>256 / 256</td></tr><tr><td>GGA readout layers,  $ { s } _ { \mathrm { g e o } }$ </td><td>{14, 29, 44, 59}</td></tr><tr><td>GGA normalized layer weights, ωe</td><td>(1, 2, 4, 10)/17</td></tr><tr><td>Smooth L1 transition,  $\beta _ { \mathrm { g e o } }$ </td><td>0.1</td></tr><tr><td>Cosine-distance weight,  $\lambda _ { \mathrm { c o s } }$ </td><td>0.25</td></tr><tr><td>COSS / SSG layers,  $S _ { \mathrm { c f } } = S _ { \mathrm { g r o u n d } }$ </td><td>{35, 47, 56}</td></tr><tr><td>Object / effect weights,  $\alpha _ { o } \mathrm { ~ / ~ } \alpha _ { e }$ </td><td>0.65 / 0.35</td></tr><tr><td>BCE / Dice weights,  $\alpha _ { \mathrm { b c e } } ~ / ~ \alpha _ { \mathrm { d i c e } }$ </td><td>0.5 / 0.5</td></tr><tr><td>Balanced-BCE positive-weight cap</td><td>20</td></tr><tr><td>Effect thresholds,  $\delta _ { \mathrm { l o w } } ~ / ~ \delta _ { \mathrm { h i g h } }$ </td><td>0.02 / 0.10</td></tr><tr><td>Object threshold,</td><td></td></tr><tr><td> $\delta _ { \mathrm { o b j } }$ </td><td>0.01</td></tr><tr><td>Smoothing-kernel side length, κ</td><td>3</td></tr><tr><td>Exclusion dilation radius,  $\rho _ { d }$  Grounding temperature,  $\tau _ { a }$ </td><td>1 token-grid cell 1.0</td></tr></table>

Efect thresholds operate on RGB values normalized to [0, 1], while smoothing and dilation operate on the target-token grid.

## G Inference Eficiency

Table S5 Inference eficiency on one NVIDIA L20X GPU at 1024 × 1024, batch size 1, 40 denoising steps, and CFG 4.0. Parameter counts are obtained from the loaded inference modules. Latency is the mean and sample standard deviation over five cases after two warmup calls; memory is the maximum PyTorch-allocated GPU memory over the timed calls. B denotes $1 0 ^ { 9 }$ parameters.
<table><tr><td>Method</td><td>Denoiser params (B)</td><td>Pipeline params (B)</td><td>Time/image (s)</td><td>Peak memory (GiB)</td></tr><tr><td>Qwen-Image-Edit-2511</td><td>20.430</td><td>28.849</td><td> $7 9 . 4 0 \pm 0 . 0 6$ </td><td>55.92</td></tr><tr><td>Ours (MLP)</td><td>20.815</td><td>30.143</td><td> $1 2 0 . 1 3 \pm 0 . 2 6$ </td><td>60.80</td></tr></table>

Parameter accounting. We count parameters in the loaded MLP-based PhysInsert model. The 2048 → 256 → 3072 geometry MLP contains 1,314,048 parameters; including input LayerNorm and the geometry-modality embedding gives 1,321,216 parameters. Denoiser counts include the DiT, its LoRA parameters, the geometry projector, and third-stream adapters. Pipeline counts additionally include the multimodal encoder, VAE, and frozen VGGT aggregator. Shared weights are counted once, and training-only feature readout and spatial heads are excluded. This gives approximately 20.815B denoiser and 30.143B pipeline parameters, a 4.48% pipeline increase over the backbone.

Measurement protocol. Both models use the same NVIDIA L20X GPU, BF16 weights, PyTorch 2.12.1, CUDA 12.6, and no model ofloading, with identical background, reference, and box-guidance images, prompts, and seeds. CUDA synchronization brackets each complete pipeline call, including encoding, denoising, and decoding; file I/O and model loading are excluded. PhysInsert measure ments include online VGGT extraction and unfused LoRA execution.

## H More Qualitative Results

Figures S1 and S2 show ten selected single-reference cases per guidance setting on PhysInsert-Bench, illustrating relative advantages in placement, viewpoint adaptation, and scene integration. Box-guided Ours uses the additional 4,000-step adaptation from Table 3; text-guided Ours uses the text-guided checkpoint without this adaptation. Object references are center-cropped for display only; inference inputs and generated outputs are unchanged.

Ours  
![](images/a83d6654deb346be742e8a95a9a2b16034b770f40ed5a20a7530a7472c902000.jpg)  
Figure S1 Selected box-guided comparisons on PhysInsert-Bench with one reference image. DIRECT addi tionally receives GT pose.

Background

GT

Nano Banana Pro

![](images/79813756f37b91194be33c89e97d1b0d0ee151fcd0cbf521ff19f3641bffaa02.jpg)  
Figure S2 Selected text-guided comparisons on PhysInsert-Bench with one reference image. References are center-cropped only for visualization.

Figures S3 and S4 present additional qualitative comparisons on 20 DreamBooth cases, with one reference image per case. Figure S5 presents further comparisons on 11 cases from the separate in-the-wild user-study set.

Background  
Object  
GT  
OmniPaint  
InsertAnything  
DIRECT  
GPT-Image2  
Nano-banana pro  
Ours  
![](images/20a3c28ad770b4aa178811cecb0983787044691e1616f125af2ff5fa0be779e9.jpg)  
Figure S3 Additional qualitative comparisons on DreamBooth (1/2). Columns show the background, object reference, ground-truth target (GT), OmniPaint, InsertAnything, DIRECT, GPT-Image2, Nano Banana Pro, and Ours. DIRECT uses a GT-pose RGB proxy; GPT-Image2 and Nano Banana Pro use text guidance.

GPT-Image2  
![](images/d559d87601ae82537e3a81e874702357c9990c24e630bfa48814c893fdb99b1b.jpg)  
Figure S4 Additional qualitative comparisons on DreamBooth (2/2). The column order and method settings are the same as in Figure S3.

![](images/dd6dff6cf9d6d320072c94b4225546b504d9edb4f3314e0fa89a7e67f05636ba.jpg)  
Figure S5 Additional qualitative comparisons on 11 cases from the in-the-wild user-study set. Columns show the object reference, background scene, OmniPaint, InsertAnything, DIRECT, Nano Banana Pro, and Ours. White boxes in the scene column indicate approximate insertion regions. OmniPaint is included for qualitative comparison only and was not part of the user-study ratings.

## I Dataset Construction Details

Sample Filtering and Deduplication. We randomly select candidate objects from complete BlenderKit scenes and render counterfactual pairs under fixed camera, lighting, and remaining scene settings, recomputing shadows, mirror images, and reflections after removal. Rule-based filtering first rejects pairs with excessively small or large image changes. Gemini then examines the scene pair and object views for object completeness, background plausibility after removal, image quality, and realism, excluding unsuitable scene structures, residual semantic parts, and visible rendering artifacts. We deduplicate retained objects using DINO feature similarity while retaining their associated scenes. The complete Gemini quality-scoring and interaction-classification prompts, including image order and JSON output examples, are reproduced in Appendix J.1.

Scene Augmentation. For each retained scene–object combination, we keep the object and scene configuration unchanged while randomly varying the camera position. At each viewpoint, we reren der a matched with/without pair and apply a second quality check.

Object Reference Rendering. Each deduplicated object is independently rendered under seven lighting conditions: fixed white light, fixed studio lighting, and five randomly sampled HDR environments. For each condition, we use 11 fixed and 12 random viewpoints while preserving object geometry and material, yielding $7 \times 2 3 = 1 6 1$ reference images per object. These references are associated with the corresponding scene pairs.

Dataset Comparison. Table 1 compares AnyInsertion (Song et al., 2025), DIRECT (Gong et al., 2026), and InsertWorld. CF denotes object-presence/absence pairs acquired through controlled capture or physical re-rendering under otherwise fixed settings; MV/ML denote same-object mul tiview/multilight data, and Ctrl. denotes control inputs. A blue check mark indicates that the corresponding data is provided; a red cross indicates that it is not specifically constructed; Partial indicates partial coverage.

## I.1 Controlled Benchmark Construction

Each controlled PhysInsert-Bench example contains an object-free background I , a reference set $\textstyle { \mathcal { T } } _ { r } .$ an insertion instruction $p ,$ the corresponding with-object target $I _ { + }$ , and the target object region. The target box is the 2D bounding box enclosing the object rendered in Blender at its target pose. For box guidance, an additional reference image annotated with this box is supplied as visual conditioning, alongside the object-reference images. The object-reference counts reported in the tables do not include this additional box-annotated image. Text guidance instead uses spatial instructions describing position, scale, orientation, and occlusion relationships without an inference time mask.

Each case also carries physical-efect labels and evaluation-focus tags, which specify the interactions emphasized by that example and are incorporated into the fixed Phy-VLM prompt. Interaction annotations describe scene-to-object requirements in illumination, visibility, and placement, and object-to-scene efects including cast shadows, reflections, mirror images, and occlusion changes. Optical cases combine substantial reference–target illumination diferences with visible environmental efects. Spatial & Mechanical cases and 26 Optical challenges provide independent reference viewpoints to test scene-dependent orientation or pose adaptation.

## I.2 Real-World Test Set Construction

DreamBooth. We construct 40 test cases covering all objects in DreamBooth. For each case, we randomly select one image as the target and use the remaining images of the same object as the reference pool; we report one-reference evaluation. We remove the object from the target image using Nano Banana Pro and manually review the resulting background.

PhysInsert-Bench-Wild. The 50 cases draw on Amazon Berkeley Objects (ABO) (Collins et al., 2022), NAVI v1.5 (Jampani et al., 2023), the training and validation splits of ORIDa (Kim et al., 2025), and real images with references generated using SAM 3D Objects (Chen et al., 2026). Refer ence pools contain multiview product images from ABO, multiview photographs of the same object from NAVI, and cross-scene observations of the same object from ORIDa. For the SAM 3D-based cases, we use real target images and render reference views from SAM 3D Objects reconstructions.

## J Dataset and Benchmark Samples

We visualize five examples from the InsertWorld training set and twenty examples from PhysInsert-Bench. The training examples show counterfactual scene pairs, isolated objects, and references under diferent viewpoints and illumination. The benchmark examples cover twenty distinct interaction tags, with one example per selected tag. These are illustrative data samples, not model predictions or a performance-ranked selection. Images retain their original aspect ratios and are resized for display. In the training examples, the original transparency of Removed-only renders is displayed over white; all other image contents are unchanged.

![](images/f2d4d3193982afe5c4a8562fabd45cd22c6280b3961e035a6798031645259fee.jpg)  
Figure S6 Five InsertWorld training examples. Each row contains the with-object scene (With), the corresponding object-free scene (Without), and the isolated target rendered from the scene viewpoint on a white display background (Removed-only). The right panel shows the same target under two lighting conditions (rows L1 and L2) and four corresponding fixed viewpoints (columns). Lighting conditions and viewpoints are selected separately for each example.

![](images/2f174b09c9a0898040b6ebe1b2a41384a37f78bbbac6da8f61b0e659d5c55955.jpg)  
Figure S7 PhysInsert-Bench samples (1/4). The original interaction tag appears at left in readable form, followed by the object-free scene (Without), with-object target (GT), and actual single-view object input. Case IDs refer to the benchmark manifest.

![](images/d3c209a7d77bba9ce2700b520e4f962d1d59838aba1ed4ac575752ffb0f9bcd6.jpg)  
Figure S8 PhysInsert-Bench samples (2/4). Column meanings follow Figure S7. The reference is the actual one-view benchmark input, including its original background; it is neither a multiview collage nor a newly rendered reference.

![](images/164fb88a26f3d25d5a191ab7cd1666cb374642e01cc60239dc182d7136ecda4a.jpg)  
Figure S9 PhysInsert-Bench samples (3/4). Column meanings follow Figure S7. The examples illustrate cloth, suspension, leaning, flat support, and depth ordering using the original benchmark annotations.

![](images/a9eb10d92d36e9c7aba8d35c8e97920c53662dcc63e5faf060ed05e0feffb332.jpg)  
Figure S10 PhysInsert-Bench samples (4/4). Column meanings follow Figure S7. These examples cover partial occlusion, placement relations, scale adaptation, perspective foreshortening, and viewpoint gaps.

## J.1 Gemini Prompts for Dataset Curation

We reproduce the quality-scoring and interaction-classification prompts used by the dataset-curation implementation. These prompts inspect rendered training-data candidates, not predictions from the methods compared in our result tables. Both produce a short insertion caption (insertion\_- description) as part of a structured JSON response; this caption is distinct from the full editing prompt used to condition the generation model. The wording and example responses below are preserved, with only typographic reformatting. The JSON examples are part of the prompts, not measured sample annotations.

Ordered visual inputs. Both stages receive the same first six images. Classification additionally receives an environment-diference visualization. A single prompt accompanies the ordered image list; changing this order would change the intended interpretation.

<table><tr><td></td><td>Index Input</td><td>Role</td></tr><tr><td>1</td><td>with</td><td>Scene with the candidate target inserted.</td></tr><tr><td>2</td><td>without</td><td>Matched scene after target removal.</td></tr><tr><td>3</td><td>removed_only</td><td>Isolated target/group in the scene view.</td></tr><tr><td>4-6</td><td>Front, right, top</td><td>Three canonical views of the target/group.</td></tr><tr><td>7</td><td>Environment difference</td><td>Classification only: signed changes outside the target; blue/- darker, red/brighter, green/chromatic.</td></tr></table>

Two complementary annotations. Stage 1 assigns six integer scores in [0, 5]: object validity, threeview quality, insertion-result validity, image realism, insertion dificulty, and overall quality. Difficulty rewards useful complexity but must not compensate for severe quality failures. Stage 2 retains an image-realism score, assigns relative scale and mechanical type, and records seven sceneinteraction attributes. Mechanical interaction and lighting/shadow complexity use integer levels 0/1/2; the other five interaction fields are Boolean. These are evidence-based annotations, not mutually exclusive benchmark-subset labels. In particular, a visible change in the amplified diference map is not suficient evidence of a reflection without confirmation in the original scene pair.

Version scope. The following templates correspond to the six-image quality-scoring and sevenimage interaction-classification implementations. Older prompt variants and batch-level model overrides may difer; reproducing a historical annotation also requires that batch’s recorded request configuration. The templates do not by themselves specify a universal dataset-retention threshold.

## J.1.1 Stage 1: Six-Dimension Quality Scoring

## Prompt S1: Quality scoring — complete request text

You are a senior quality-control expert for 3D object-insertion training data. You will receive 6 images and must judge whether this sample is suitable for object-insertion training.

\- Image 1: with, the result after inserting the object.

\- Image 2: without, the background before insertion.

\- Image 3: removed\_only, an isolated rendering of the object to be inserted.

\- Image 4: front view of the object.

\- Image 5: right view of the object.

\- Image 6: top view of the object.

## Prompt S1: Quality scoring (continued)

## [Task]

1. Describe the sample from the insertion perspective: ”insert [object] at [location]”. Treat Image 1 as the result of inserting the object shown in Images 3-6 into the scene in Image 2. If the object cannot be identified, write ”unclear insertion object”.

2. Score the following 6 dimensions from 0 to 5, where 5 is best. Use Images 3-6 especially for object\_validity and three\_view\_quality, and compare Images 1-2 especially for insertion\_result\_validity.

3. Output JSON keys must be English exactly as specified. Chinese labels below are only references.

If the inserted object is unclear, insertion\_description must be ”unclear insertion object”, and all six scores must be 0.

## [Dimensions]

## 1. object\_validity

\- Judge whether the target is a reasonable independent object or a tightly bound object group.

\- Reasonable examples: cups, chairs, flowerpots, lamps, curtains, wire mesh, hanging ornaments, decorative objects, or strongly bound small parts.

\- Unreasonable examples: walls, floors, ceilings, large background regions, large cabinet/room structures, or structural surfaces that change the whole scene.

\- If removed\_only or the three views show a wall, floor, ceiling, or huge background structure, score <= 1.

\- Curtains, rugs, wire mesh, railings, hanging screens, etc. may be large but can still be valid if they are semantically independent insertable objects.

\- Thin disk-like or sheet-like objects such as coasters, pads, washers, thin trays, dials, rugs, and similar items are valid when they have clear semantics or physical relations, such as a cup-coaster support/contact relation. Do not give a low score only because they are thin.

\- The inserted target should preferably be one object or a semantically coherent object group. Spatial separation is acceptable when the objects form one continuous semantic unit or a natural arrangement, such as a row of chairs. However, if the target contains multiple disconnected object groups placed in diferent unrelated regions, such as several groups of books in diferent compartments of a bookshelf, score <= 2.

## 2. three\_view\_quality

\- Judge whether Images 4-6 provide enough clear, learnable information about the inserted object.

\- At least one or more views should clearly show the object’s outline, scale, orientation, part relationship, or key material information. Penalize if all three views are blurry, overexposed, too dark, too small, heavily cropped, or nearly duplicate the same view.

\- Do not give a low score just because one or two views look like a line. Flat objects such as disks, coasters, pads, thin boards, trays, and dials naturally look line-like from front/side views. If the top view or another view clearly shows the shape, and the object has clear semantics or a physical relation, this dimension can be medium or high.

\- If at least one three-view image provides enough information, for example a coaster/disk/dial shape is clear in top view, the score should usually be >= 3.

\- Score <= 2 only when all three views provide little useful information, or when the object is extremely thin/long and lacks learnable shape, material, or semantic information, such as a very thin railing/rod/cable.

## 3. insertion\_result\_validity

\- Focus on Image 2 without: whether it is suitable as the pre-insertion background. The sample was constructed by deletion, so with is usually more reliable and without is more likely to contain artifacts.

\- Core check: whether Image 2 still contains parts that belong to the same semantic object, same object group, or strongly bound components of the inserted object. For a flowerpot, without should not leave leaves/soil/tray from the same plant; for a pendant lamp, without should not leave the lampshade/bulb/wire/rod of the same lamp.

\- If unrelated objects in without are floating, structurally odd, abstract, or physically unnatural, but they are not the same semantic object/group and not strongly bound to the inserted object, do not heavily penalize this dimension.

\- Penalize this dimension only when without has same-object residuals, same-group residuals, strongly bound residual parts, or an obvious hole/break at the insertion region that makes it unusable as a pre-insertion background.

## Prompt S1: Quality scoring (continued)

\- If Image 1 contains an obvious violation of physical plausibility in an otherwise highly realistic scene, such as a clearly floating object in an indoor environment, or if the inserted object has physically implausible mesh penetration with the scene, the score for this dimension should be <= 2.

\- If the diference between the Image 1 and Image 2 is too small, such as when the inserted object is heavily occluded or visually too similar to the surrounding environment, making the insertion result dificult to identify, the score for this dimension should be <= 2.

## 4. image\_realism

\- Judge whether Images 1 and 2 contain rendering or geometry quality problems that make the sample unusable. Do not simply penalize the scene for being stylized or abstract.

\- Acceptable: minimal abstract geometry scenes, simple 3D style, unrelated floating objects in without, unrelated physically unnatural context, or small black rendering artifacts outside the object insertion region that do not afect insertion quality.

\- Accept reasonable contact, support, attachment, or slight embedding between an object and its support, such as a cup on a coaster or an object attached to a pad/base. Do not label such physical contact as mesh penetration.

\- Penalize: severe visible mesh penetration, broken geometry, abnormal faces, wrong-normal black/bright faces, obviously wrong material rendering, abnormal exposure or lighting, large black/blank image regions, large unintended solid-color backgrounds, or rendering collapse.

\- If the image contains large black unrendered regions, or if the scene lighting is so low that key scene information is dificult to recognize, the score should be <= 2.

## 5. insertion\_dificulty

\- This is a bonus-like dimension. It measures whether the sample contains dificult but valuable insertion conditions. It must not hide severe failures in the basic dimensions above.

\- Dificulty has three categories: spatial complexity (occlusion, perspective deformation), physical complexity (size, support, tilt, suspension, contact), and lighting/material complexity (high contrast, strong shadows, reflection, self-emitting lamp-like objects, metal/transparent/glass/complex textures).

\- If the object is frontal, unoccluded, under simple lighting, with simple material and simple physical relation, dificulty should be low or medium.

## 6. overall\_quality

\- Overall suitability for object-insertion training, considering the five dimensions above.

\- If any basic dimension among object\_validity, three\_view\_quality, insertion\_result\_validity, and image realism has a severe error, overall\_quality must be low and cannot be rescued by high insertion\_dificulty.

\- If the four basic dimensions have no severe problem, insertion\_dificulty can boost overall\_quality. Highquality high-dificulty samples can receive 5. Low-dificulty samples usually tend toward 3-4 unless the basic dimensions are exceptionally good.

## [Scoring Discipline]

Be strict. Do not give 4 or 5 lightly; reserve high scores for samples with clearly strong evidence and no serious issues in the relevant dimensions.

For each dimension, the reason must be specific: point out the problem location and problem type, and avoid vague generic comments.

## [Output Requirements]

Return only one JSON object wrapped by “‘json and “‘. Do not output any extra text.

The JSON object must contain exactly these top-level keys: insertion\_description, object\_validity, three view\_quality, insertion\_result\_validity, image\_realism, insertion\_dificulty, overall\_quality, reason. reason must be a JSON object, not a string. It must contain exactly these six keys: object\_validity, three view\_quality, insertion\_result\_validity, image\_realism, insertion\_dificulty, overall\_quality. All six score values must be integers from 0 to 5.

## [Valid Output Template]

"insertion\_description": "insert a white vase on the tabletop",

"object\_validity": 5,

"three\_view\_quality": 4,

## Prompt S1: Quality scoring (continued)

"insertion\_result\_validity": 4,   
"image\_realism": 4,   
"insertion\_difficulty": 2,   
"overall\_quality": 4,   
"reason": {   
"object\_validity": "The target is an independent vase and is a reasonable insertable object.",   
"three\_view\_quality": "The three views clearly show the vase outline, volume, and material   
cues.",   
"insertion\_result\_validity": "The without image is usable and has no same-object residual parts   
around the insertion region.",   
"image\_realism": "The scene has no severe rendering collapse, broken geometry, or obvious   
material errors.",   
"insertion\_difficulty": "The object is mostly unoccluded with simple lighting and material, so   
the insertion difficulty is moderate to low.",   
"overall\_quality": "The four basic dimensions are acceptable, but the task is not very difficult,   
so it is a usable regular training sample."   
}   
}

## J.1.2 Stage 2: Object Properties and Scene Interactions

## Prompt S2: Interaction classification — complete request text

You are a senior analyst of 3D object-insertion data. You will receive seven images describing one insertion case.   
Classify the inserted object and its interactions with the scene. Also retain an image-realism quality score.

## [Input images]

\- Image 1: with, the scene containing the target object.

\- Image 2: without, the same scene after removing the target object.

\- Image 3: removed\_only, an isolated rendering of the complete removed object group.

\- Image 4: front view of the complete removed object group.

\- Image 5: right view of the complete removed object group.

\- Image 6: top view of the complete removed object group.

\- Image 7: signed environment diference map after masking the target object’s own pixels. Black means little or no detected change. Blue means the pre-existing scene became darker after insertion, red means it became brighter, and green means a mainly chromatic change with little luminance change. Intensity represents amplified change magnitude. Use it only to locate and type candidate changes, then verify their physical meaning in Images 1-2.

Use Images 3-6 to identify intrinsic object properties. Compare Images 1 and 2 to identify object-environment relations. Do not infer a relation merely because it is theoretically possible. For boolean fields, mark true only with visible evidence. For 0/1/2 fields, choose a positive level only with visible evidence supporting that level.

## [Tasks]

1. Write insertion\_description in the form ”insert [object] at [location]”.

2. Score image\_realism from 0 to 5, where 5 is best. The inserted target must be correctly rendered in Image 1 (with). Check Images 1 and 2 for rendering collapse, broken or missing geometry, wrong normals, severe penetration, abnormal exposure, missing textures, genuinely unrendered regions, abnormal black target parts, and clearly incorrect material rendering. Stylized but internally consistent scenes are acceptable.

\- Judge rendering and geometry quality only to determine whether the insertion sample is visually usable. Do not lower image\_realism merely because the scene is stylized, abstract, minimal, semantically unusual, or physically implausible in ways unrelated to the insertion.

\- Acceptable context includes minimal abstract geometry, a simple 3D rendering style, unrelated floating objects already present in without, unrelated physically unnatural scene content, and small isolated black rendering artifacts outside the target/insertion region when they do not afect insertion quality or interpretation.

\- Accept reasonable contact, support, attachment, and slight embedding between the target and its intended support, such as a cup resting on a coaster or an object attached to a pad/base. Do not call ordinary load-bearing contact, a tight attachment, or shallow support embedding severe mesh penetration. Penalize

## Prompt S2: Interaction classification (continued)

penetration only when geometry visibly passes through another surface by an implausible amount, produces broken/interleaved faces, or materially damages the target or support silhouette.

\- Quality problems include severe visible mesh penetration, broken or missing geometry, abnormal faces, wrongnormal black/bright faces, clearly incorrect material rendering, abnormal exposure or lighting, large black-/blank unrendered regions, large unintended solid-color backgrounds, and partial or complete rendering collapse.

\- Large-black/low-light score cap: if Image 1 contains a large genuinely unrendered black/blank region, or scene illumination is so low that the target or key scene structure is dificult to recognize, image\_realism must be 0, 1, or 2 and never higher. Use 2 when the sample remains usable despite the defect; use 0 or 1 when the blank region or darkness dominates, hides the target, or prevents reliable interpretation. This cap concerns missing content or unusable visibility, not physically plausible black materials or shadows.

\- Large unintended solid-color backgrounds of any color cap image\_realism at 2 when they visibly replace expected scene content or indicate rendering failure. Apply the stricter flat-gray gate below when the failed region is a substantial near-uniform neutral gray area.

\- Mandatory target-integrity gate: first identify the target’s complete expected structure, parts, silhouette, and materials from removed\_only and the front/right/top views. Then inspect every visible target region in with. The target must appear as a coherent rendered object: expected visible parts must exist, surfaces must be continuous, geometry must not look shattered, torn, erased, hollow, or accidentally clipped, and materials/textures must not be replaced by anomalous featureless black patches. Do not award 5 unles this target-integrity check clearly passes.

\- Mandatory target-visibility gate: compare with against removed\_only and determine whether a meaningful, recognizable portion of the inserted target is actually visible in the scene image. If the target is almost invisible because it is nearly entirely outside the frame, hidden by scene geometry, reduced to only a tiny sliver or scattered fragments, extremely underexposed, or merged with the background so strongly that its silhouette and material cannot be judged, image\_realism must be 0 or 1 and never 2 or higher. Use 0 when the target is efectively absent or cannot be reliably located in with; use 1 when only a tiny insuficient fragment or extremely weak trace remains. As a practical cue, visibility of roughly less than 10 percent of the expected projected target silhouette, or the absence of any coherent identifiable target part, normally triggers this rule. A physically small XS/S object is not a failure when it is still clearly resolved and recognizable.

\- Target-defect score caps: a clear but localized target rendering defect, such as a broken surface, missing small part, isolated invalid black patch, wrong-normal patch, texture dropout, or obvious unintended intersection, caps image\_realism at 2. A widespread defect, a missing major target part, severe geometry collapse, or a black/corrupt region that materially prevents recognizing or judging the inserted target requires image realism 0 or 1. A visible but milder suspicious artifact that does not clearly break geometry or materials caps the score at 3.

\- For target defects, one direct contradiction with removed\_only or the three object views is suficient evidence; do not require two independent indicators. Examples include an expected target surface becoming a featureless black hole, a target part abruptly disappearing without a foreground occluder, or a material region being replaced by black pixels whose boundary cuts across the target’s own geometry. This target-specific rule takes priority over the general black/gray-region rule below.

\- Mandatory flat-gray unrendered-region gate: if Image 1 contains a visually substantial region of nearly uniform neutral gray that is clearly empty/unrendered rather than scene geometry, image\_realism must be 0 or 1 and never 2 or higher. The decisive evidence is a pure-color-like gray area with almost no texture, material detail, geometry, shading gradient, lighting variation, or physically meaningful structure, often ending at an artificial boundary or replacing an area where scene content should exist. This gate applies even when the gray region does not overlap the target and takes priority over the general non-target two-indicator rule below. Use 0 when the blank gray region dominates or makes the image unusable; otherwise use 1.

\- Do not confuse the flat-gray gate with darkness caused by the inserted object or scene lighting. A cast/contact shadow, shaded wall, gray material, fog, or underexposed surface remains valid when it preserves scene texture, geometry, gradual illumination or penumbra, material variation, reflections, or boundaries consistent with physical objects. A dark region is not a pure-gray unrendered region merely because it is large or lowcontrast.

\- Dark pixels, a large black/gray area, low exposure, or contact with the image boundary are not by themselves rendering failures. Before lowering the score, first identify what geometry occupies that region using with, removed\_only, and the three object views. Dark target materials, dark scene objects, fabric folds, physically

## Prompt S2: Interaction classification (continued)

plausible cast/contact shadows, naturally dark rooms, foreground objects close to the camera, and ordinary image-frame cropping are valid rendered content.

\- Evidence of valid rendering includes a coherent object silhouette, continuous geometry, material texture, fabric detail or folds, soft shading gradients, plausible shadow transitions, highlights, and boundaries that follow visible scene geometry. A dark region with any of these cues must not be called flat, blank, or unrendered merely because its average brightness is low.

\- General non-target hard failure rule: assign image\_realism 0 or 1 for a black/gray/blank scene region only when there is positive visual evidence that rendering content is actually missing or corrupted and it materially prevents judging the target. For non-target/background regions, require at least two of these indicators: (a) the region is genuinely featureless with no geometry, texture, shading gradient, or plausible shadow structure; (b) it has an abrupt boundary that does not follow any object silhouette, occluder, image frame, or physical shadow; (c) it cuts through or replaces target parts that removed\_only or the three views show should be visible; (d) it is inconsistent with the corresponding scene geometry in with/without; (e) it produces impossible holes, erased surfaces, or broken topology. This two-indicator requirement does not apply to a directly verified target defect under the target-integrity gate. A dark region appearing only in with is valid only when it is visibly coherent target material or shading, not merely because it overlaps the target location.

\- Image-frame cropping is not a failure when enough of the target remains clearly visible and judgeable. However, cropping becomes a hard visibility failure under the target-visibility gate when almost the entire target lies outside the frame or only a tiny unjudgeable fragment remains. Similarly, foreground target parts may legitimately cover the bed, floor, support surface, or other background content.

\- Calibration example: dark pillows or other soft furnishings close to the camera may occupy and be cropped by a lower corner. If their silhouettes, fabric folds, shading, and mutual shadows remain coherent, this is valid rendering and must not receive 0 or 1 merely for being dark. It should normally receive 4; use 5 only if the entire image also satisfies the exceptional score-5 gate below.

\- The image\_realism reason must explicitly report both the target-visibility gate and the target-integrity gate. State whether the target is clearly visible and judgeable in with. If visibility fails, describe what little remains visible and assign 0 or 1. If integrity fails, name the damaged/missing target part or abnormal black region, compare it with removed\_only or a specific object view, and state the applied score cap. If both pass, explicitly state that a meaningful target portion is visible and its geometry and materials are coherent. Also state whether the large-black/low-light cap, unintended-solid-color cap, flat-gray gate, or any other general hard failure applies. Do not describe normal dark materials, shadows, foreground coverage, moderate frame cropping, reasonable support contact, attachment, or slight support embedding as corruption or penetration.

\- Score 5 is exceptional and must be rare. Award 5 only when the target is clearly visible and fully coherent, placement and contact are physically plausible, geometry/normals/materials/textures are clean, lighting and shadows are consistent, and no visible artifact, suspicious blank area, penetration, corruption, or meaningful quality concern remains anywhere in Image 1. Passing the hard-failure gates alone is not enough for 5. Use 4 for a strong, fully usable image with any minor imperfection, uncertainty, small artifact, slightly awkward integration, or merely ordinary rendering quality. When uncertain between 4 and 5, always choose 4.

3. Classify object\_properties:

\- relative\_scale: exactly one of ”XS”, ”S”, ”M”, ”L”, or ”XL”. Judge the complete removed object group’s physical scale relative to the visible scene and nearby objects in Images 1-2. Never use its enlarged crop size in removed\_only or the three views.

\- XS: an object component or sub-part rather than a normally complete standalone object, such as a handle, knob, bulb, small panel, fixture part, or one component of a larger assembly.

\- S: a small standalone or handheld/tabletop object, such as a cup, book, small ornament, small lamp, or compact container.

\- M: a normal standalone scene object larger than typical tabletop items but clearly below major furniture scale, such as a floor lamp, medium plant, stool, waste bin, or small appliance.

\- L: a large object at furniture or major appliance scale, such as a chair, table, sofa, bed, refrigerator, or substantial cabinet.

\- XL: an architectural or near-architectural-scale object occupying a broad scene region, such as a long wallspanning curtain, a large built-in wardrobe/cabinet wall, a room divider, or another object comparable to a wall opening or building element.

## Prompt S2: Interaction classification (continued)

Choose the closest scale class from visible scene evidence. Semantic identity is only an anchor; prioritize the object’s actual scale in this scene.

\- mechanical\_type: exactly ”rigid” or ”non\_rigid”. Use ”non\_rigid” for cloth, curtains, soft furnishings, flexible sheets, deformable vegetation, cables, and similar objects whose shape can deform. Use ”rigid” for objects whose shape is normally fixed. Judge the dominant property of a coherent object group.

4. Classify environment\_relations using the exact types and levels below.

Boolean fields:

\- has\_occlusion: true only when the target is meaningfully partially hidden by foreground scene geometry or participates in a nontrivial layered occlusion boundary. Use removed\_only as the primary reference: compare the target silhouette and visible parts in with against the isolated target in removed\_only, which represents the same target without scene occluders. If a part is visible in removed\_only but disappears or is cut of in with exactly where foreground scene geometry passes in front of it, that is strong evidence of occlusion. Use the front/right/top views as additional references for the target complete structure when needed. Do not classify image-frame cropping, an incomplete removed object, self-occlusion, shadows, or ordinary coverage of the background by a fully visible target as scene occlusion. The reason must identify the foreground occluder and which target part it hides, and should mention the removed\_only comparison when it provides the evidence.

\- has\_strong\_perspective\_depth: true when the scene has strong depth perspective around the target, such as large near-far scale variation, deep spatial placement, or substantial perspective deformation that makes insertion geometry harder. A normal perspective camera alone is not enough.

\- has\_reflective\_surface\_interaction: this field is specifically about the inserted target changing the appearance of a pre-existing reflective or transmissive scene surface. Use Image 7 only to locate scene regions changed by inserting the target, then inspect the same coordinates in the original with and without images. Image 7 colors mean: blue = the scene became darker, red = the scene became brighter, green = the scene mainly changed color. The color itself does not determine the label.

Return true only when all of the following are satisfied: (1) the changed region is outside the inserted target; (2) the receiving scene surface is visibly identifiable as glossy, polished, metallic, glass, water, or another genuinely reflective/refractive material; (3) the aligned with/without comparison shows a clear target-caused change on that surface; (4) the change has reflection, specular-highlight, or refraction characteristics rather than ordinary difuse illumination; and (5) the change is readily visible to a human at normal viewing scale. Examples that may qualify include the target or its color appearing in a glossy cabinet or polished floor, a target-caused specular highlight changing on metal or glass, or visible refraction through a transmissive scene surface.

Return false for ordinary difuse brightening or darkening of walls, ceilings, fabric, matte wood, or other nonreflective surfaces; generic bounce light or color bleeding; illumination cast by an emissive target; target cast/- contact shadows; ordinary light blocking; ambient occlusion; and any vague change for which the receiving surface cannot be confidently identified as reflective or transmissive. A recognizable mirror reflection belongs exclusively to has\_mirror\_reflection\_interaction and must be false here.

Material identity alone is not enough: the reflective/refractive content or highlight on that surface must visibly change between with and without. Conversely, a changed region alone is not enough: do not infer reflection merely from red, green, or blue pixels in Image 7. Image 7 is amplified and is never suficient evidence by itself. When the material, cause, or human visibility is uncertain, return false.

The reason must name the receiving scene surface and its visible reflective/transmissive material cue, identify the changed reflection/highlight/refraction, state whether Image 7 shows blue/red/green there, and confirm that the change remains readily visible in the original with/without images. For false, explicitly state which required condition is missing, such as no identifiable reflective receiver, only difuse illumination, only shadow/light blocking, only a mirror reflection, or insuficient human-visible change.

A. Scene-mirror change: a mirror or mirror-like scene surface contains recognizable reflected content that changes because of the target insertion. Compare the same mirror region at aligned coordinates in with and without. The reflected target, or another recognizable reflected scene object/structure whose mirror image is altered or occluded by the target, must appear, disappear, move, or visibly change between the two images. A mirror merely being present is false. An unchanged mirror image is false, even if it contains many recognizable scene objects.

B. Target-is-mirror: the inserted target itself is a mirror or contains a genuine mirror surface, and in with that target mirror visibly reflects recognizable objects or spatial structure from the surrounding environment. Use removed\_only and the three views to confirm that the reflective surface belongs to the target, then identify

## Prompt S2: Interaction classification (continued)

the specific reflected environment content visible after insertion. This path does not require a with/without comparison inside the target region because the target mirror is absent in without.

Mandatory decision procedure: first determine whether the candidate mirror belongs to the pre-existing scene or to the inserted target. For a pre-existing scene mirror, true requires a concrete aligned with/without change in its recognizable reflected content caused by the target. For a target mirror, true requires recognizable environment content inside the target mirror. Otherwise return false.

Do not use this field for ordinary glossy highlights, reflected color, illumination changes, vague silhouettes, blurry metallic reflections, dark screens, glass without recognizable reflected content, or changes outside the mirror region. Those may belong to has\_reflective\_surface\_interaction but are not mirror interaction. The reason must name the mirror owner, the reflected object or structure, and either the exact with/without mirror-content change for path A or the recognizable environment content reflected by the target mirror for path B.

\- has\_emissive\_large\_scale\_lighting\_change: true only when the target is self-emissive and its presence causes a visible large-scale illumination or color change in the surrounding environment. A glowing pixel or bright lamp surface without substantial environmental lighting change is false.

Three-level integer fields (must be JSON integers 0, 1, or 2, not booleans):

\- has\_mechanical\_interaction:

\- 0 (none): no clearly visible support, contact, suspension, attachment, or other meaningful force relation with scene geometry can be identified.

\- 1 (simple): ordinary static support and gravity balance through one simple support arrangement, such as an object resting on a table, shelf, floor, or a single regular base. A target simply mounted on or placed against a wall also counts as simple when no explicit load-bearing suspension mechanism or positional constraint is visible; an object that could plausibly occupy an arbitrary wall position must not be promoted to level 2 merely because it is described as hanging. Several feet touching the same flat floor still normally count as simple unless they create a genuinely nontrivial balance.

\- 2 (complex): nontrivial force balance involving multiple independent support/contact points or constraints from diferent directions, tension, clamping, leaning with multiple force-bearing contacts, or another mechanically coupled arrangement whose stability depends on more than ordinary single-surface support. A hanging or suspended target qualifies as level 2 only when the images show an explicit suspension dependency and load path, such as a visible hook, rope, cable, chain, bracket, slot, ceiling attachment, or multiple suspension points that constrain and support the target. Do not infer complex suspension from semantic identity or wall placement alone.

Mere proximity without visible force-bearing contact is 0. Assign level 2 only from visible mechanical evidence, not from an assumption about how the object is normally installed. State the visible support/contact points, suspension mechanism, and force relation in the reason.

\- has\_complex\_lighting\_shadows:

\- 0 (none): no clearly identifiable target-related shadow is visible; extremely weak generic ambient occlusion that cannot be confidently attributed to the target also counts as 0.

\- 1 (simple): exactly one dominant target-related contact/cast shadow direction or one coherent shadow lobe is visible under otherwise simple lighting. A soft shadow still counts, but level 1 is allowed only after checking all sides of the target and finding no second independently directed target shadow.

\- 2 (complex): assign level 2 whenever two or more distinct target-related cast-shadow lobes or projections are visible in diferent directions, on opposite sides of the target, or on diferent receiving surfaces. This is direct evidence of multiple or spatially varied light contributions. The shadows may be soft, faint, partially overlapping, or unequal in strength; they do not need to be crisp, high-contrast, intricate, or produced by obvious visible lamps. Also use level 2 for split illumination across the target, multiple light directions, overlapping shadows, strongly spatially varying light, or intricate/high-contrast shadow patterns.

Mandatory shadow inspection procedure:

1. Locate the complete target using with and removed\_only, then inspect the floor/support and the full perimeter on its left, right, front, and back image sides rather than stopping at the strongest shadow.

2. Compare the same coordinates in with and without. A dark or shaded lobe that appears with the target and disappears or weakens without it is target-related evidence. Use Image 7 to locate subtle non-target changes, then verify them in with/without.

3. Inspect shadows cast by nearby scene objects as supporting evidence for the scene light directions. If those shadows reveal illumination from multiple directions, actively search for the corresponding weaker target projections; scene-object shadows are clues, but level 2 still requires visible target-related shadow or split-light evidence.

## Prompt S2: Interaction classification (continued)

4. If two target shadows extend to the left and right, or otherwise diverge in direction, classify level 2 even when one is much softer or less visible than the other. A common level-2 case is one strong lobe on one side plus one broad faint lobe on the opposite side. Do not merge them into one soft shadow and do not reject the weaker lobe merely because its boundary is difuse.

5. Distinguish ambient occlusion from a weak cast shadow spatially. Ambient occlusion is restricted to a narrow contact rim or crevice that tightly hugs the target base with no directional outward extent. A dark region that spreads outward from the silhouette over the receiving surface, has a displaced or directional lobe shape, or occupies a broader area than the immediate contact rim is a shadow projection, even if it is faint and soft.

6. Use scene lighting evidence as a tie-breaker. When nearby objects show shadows or highlights consistent with two light directions, and the target has one strong lobe plus a weaker outward-spreading lobe in the corresponding second direction, count both target lobes and assign level 2. Do not call the weaker lobe ambient occlusion unless it is strictly confined to the narrow target-contact boundary.

Calibration example: a round chair on a rug with a visible projection to its right and a weaker but spatially extended projection to its left is level 2, not level 1. Softness and unequal intensity do not reduce the number of shadow directions.

Do not count truly generic ambient occlusion or texture darkening as an additional cast shadow. The reason must enumerate every detected target shadow by image-relative direction and receiving surface. A level-1 reason must explicitly state that every other dark region is confined to a narrow contact rim with no outward directional extent; saying only that it is faint or soft is not suficient to reject it.

## [Output requirements]

Return only one JSON object wrapped in a json code fence. Use exactly the English keys shown below. Every classification and every relation must have a concise, evidence-based reason.

Keep every reason to exactly one sentence of at most 20 English words. Keep the complete response below 700 English words; do not add commentary outside the JSON.

```csv
{
"insertion_description": "insert a table lamp on the side table",
"image_realism": 4,
"object_properties": {
"relative_scale": "S",
"mechanical_type": "rigid"
},
"environment_relations": {
"has_occlusion": false,
"has_strong_perspective_depth": false,
"has_mechanical_interaction": 1,
"has_complex_lighting_shadows": 1,
"has_reflective_surface_interaction": false,
"has_mirror_reflection_interaction": false,
"has_emissive_large_scale_lighting_change": true
},
"reason": {
"image_realism": "The scene is coherent and has no severe rendering defects.",
"relative_scale": "The lamp is a small tabletop object relative to the room and nearby
furniture.",
"mechanical_type": "The lamp has a fixed rigid body.",
"environment_relations": {
"has_occlusion": "The complete lamp is visible without meaningful foreground occlusion.",
"has_strong_perspective_depth": "The lamp is viewed in a locally shallow region without strong
near-far deformation.",
"has_mechanical_interaction": "Level 1: the lamp has a simple single-surface support relation
with the tabletop.",
"has_complex_lighting_shadows": "Level 1: one straightforward cast shadow from the lamp is
visible on the tabletop.",
"has_reflective_surface_interaction": "No nearby reflective surface visibly changes because of
the lamp.",
"has_mirror_reflection_interaction": "The lamp is not visibly reflected by a scene mirror and
does not itself contain a mirror reflecting recognizable scene objects.",
"has_emissive_large_scale_lighting_change": "The lamp visibly changes illumination over a broad
surrounding region."
```

<table><tr><td>Prompt S2: Interaction classification (continued)</td></tr><tr><td></td></tr><tr><td>}</td></tr><tr><td>メ</td></tr></table>

## K User Study Protocol

We collect complete responses from 27 distinct participants on 25 in-the-wild insertion cases, comparing PhysInsert, Insert-Anything, DIRECT, and Nano Banana Pro. Method identities are hidden, and case and method orders are randomized. These cases form a separate user-study set constructed by combining scenes from RealBokeh with foreground objects from Distinctions-646. With the back ground and object reference visible, participants rate side-by-side results from 1 to 5 (higher is better) on lighting integration (including object illumination and cast shadows), perspective, pose, identity preservation, and background preservation. We average over cases within each participant, then across participants; the overall score equally weights the five criteria.

The error bars in Figure 4 show 95% percentile confidence intervals estimated from 50,000 participant-level bootstrap resamples, with the 25 cases held fixed.