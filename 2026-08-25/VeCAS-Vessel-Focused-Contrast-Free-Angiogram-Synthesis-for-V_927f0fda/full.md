# VeCAS: Vessel-Focused Contrast-Free Angiogram Synthesis for Vascular Interventions

De-Xing Huang, Chen-Yu Wang, Hao Liang, Xiao-Hu Zhou, Mei-Jiang Gui, Tian-Yu Xiang, Qin-Yi Zhang, Chen Wang, Xiao-Liang Xie, Shi-Qi Liu, Ming-Yuan Liu, Zhen-Chang Wang, and Zeng-Guang Hou

Abstract— X-ray angiography relies on iodinated contrast agents to visualize vascular structures during imageguided interventions. However, contrast administration carries risks of adverse events, motivating the development of contrast-free alternatives. Generating X-ray angiograms directly from non-contrast X-ray images offers a potential solution, but existing approaches remain limited by (i) insufficient control over vascular localization and (ii) inefficient modeling of redundant background content. To address these challenges, we propose VeCAS, a two-stage Vessel-focused Contrast-free Angiogram Synthesis framework that separates vascular structure localization from angiographic appearance synthesis. In Stage I, a discriminative model localizes vascular structures in non-contrast X-ray images, while cross-modality latent distillation transfers vessel-sensitive knowledge from X-ray angiograms during training. In Stage II, a vessel-focused inpainting model synthesizes angiographic appearance within the localized vascular regions while preserving the non-vascular background. Experiments on an in-house lower-limb vascular intervention dataset show that VeCAS outperforms the comparison methods in terms of vascular structural fidelity and image quality. Visual Turing tests and physician assessments indicate the perceptual realism of the synthesized angiograms. In addition, robotic guidewire navigation experiments in vascular phantoms show that VeCAS guid-

ance reduces the time to target by 41.4% and the number of operation steps by 40.7% compared with non-contrast guidance. Together, these results suggest the potential of VeCAS to serve as “meta contrast agent” for vascular interventions.

Index Terms— Vascular intervention, X-ray angiography, iodinated contrast agent, angiogram synthesis.

## I. INTRODUCTION

MAGE-guided vascular interventions have become an important therapeutic option for peripheral artery disease because of their minimally invasive nature, faster postoperative recovery, and lower overall risk than conventional surgical procedures [1], [2]. During interventions, X-ray angiography provides real-time 2D imaging, enabling physicians to observe the spatial relationship between vascular structures and interventional instruments, such as guidewires, catheters, and balloons [3], [4]. Since vessels are poorly visible in noncontrast X-ray images, iodinated contrast agents are routinely injected to enhance vascular visibility, as shown in Fig. 1(Top).

Despite its clinical importance, the use of iodinated contrast agents is associated with several potential risks, including life-threatening allergic reactions [5] and acute kidney injury (AKI) [6]. Although these adverse events are multifactorial, the administered volume of iodinated contrast agents has been widely recognized as an important contributing factor [7]– [10]. A recent cross-sectional study involving 1.3 million patients undergoing vascular interventions reported that each additional 75 mL of contrast agent was associated with a 42% increase in the risk of AKI [11]. These findings have motivated increasing interest in ultra-low-contrast and non-contrast vascular interventions, which aim to reduce contrast-related risks while preserving effective image guidance [12], [13]. However, without contrast enhancement, vascular structures are difficult to observe, making instrument navigation more challenging and potentially reducing procedural efficiency.

To reduce the reliance on iodinated contrast agents, several strategies have been explored in clinical practice, including intravascular ultrasound- or physiology-guided interventions [14], [15], roadmap-based guidance [16], and $\mathrm { C O _ { 2 } }$ angiography [17]. However, these approaches usually require intravascular imaging, alternative contrast agents, or previously acquired angiographic information. Direct angiographic visualization from non-contrast X-ray images remains underexplored. Recent advances in artificial intelligence-generated content (AIGC) have demonstrated the ability to generate realistic visual content [18], [19]. This progress raises the question: Can AI serve as “meta contrast agent” to synthesize X-ray angiograms that approximate the vascular visualization provided by actual contrast agent? As illustrated in Fig. 1(Bottom), such a capability could provide a potential contrast-free strategy for vascular interventions.

![](images/3fc3a40ccd121bc3863aa68cca4a18622380efbba3a7c6b594ffcd4e1a7cda58.jpg)  
Fig. 1. Iodinated Contrast Agent vs. Meta Contrast Agent. Top: In conventional X-ray angiography, iodinated contrast agents are injected to enhance vessel visibility, but their use may introduce contrastrelated risks, such as kidney injury and allergic reactions. Bottom: The proposed meta contrast agent uses a trained deep neural network to synthesize X-ray angiograms from non-contrast X-ray images, providing a computational alternative with the potential to reduce contrast administration and improve procedural efficiency.

Nevertheless, developing a reliable contrast-free angiogram synthesis method remains challenging. Existing approaches mainly suffer from two limitations: (i) Insufficient control over vascular localization. Most existing methods formulate this task as image-to-image translation [20], [21] and learn a global mapping from the source domain to the target domain [13]. Consequently, these methods tend to emphasize the overall realism of synthesized images, while providing limited explicit control over whether the generated vessels are located at anatomically consistent positions. However, accurate vascular localization is essential for interventional guidance, because visually plausible but spatially misplaced vessels may provide misleading information for instrument navigation. (ii) Inefficient modeling of redundant background content. The objective of this task is not to regenerate the entire image, but to synthesize vascular appearance within anatomically relevant regions. The remaining non-vascular background can largely be preserved from the input non-contrast image. Since the background occupies the majority of pixels, directly modeling the whole image may allocate substantial model capacity to redundant background reconstruction and may weaken the learning of vascular appearance.

To address these challenges, we propose a two-stage Vesselfocused Contrast-free Angiogram Synthesis (VeCAS) framework. The core idea of VeCAS is to decouple vascular structure localization from angiographic appearance synthesis. In Stage I, vascular structures are localized from noncontrast X-ray images. Specifically, we formulate vascular structure localization as a discriminative problem and predict binary vessel masks, rather than relying on a generative model to implicitly determine vessel locations. This formulation provides explicit spatial supervision for vascular localization. Furthermore, we introduce a cross-modality latent distillation strategy that uses X-ray angiograms during training to transfer vessel-sensitive knowledge to the non-contrast image encoder, thereby improving vascular localization under contrast-free conditions. In Stage II, angiographic appearance is synthesized under the guidance of the predicted vessel mask and the non-contrast X-ray image. We develop a vesselfocused diffusion inpainting model that selectively modifies vascular regions while preserving the non-vascular background from the input image. In addition, a vessel-focused loss is introduced to restrict the learning objective to vascular regions, encouraging the model to concentrate on vascular content rather than unnecessary background reconstruction.

To evaluate the potential utility of VeCAS, we conduct experiments on an in-house dataset collected from patients undergoing lower-limb vascular interventions. In addition to quantitative metrics, we perform visual Turing tests, physician assessments, and phantom-based robotic navigation experiments. These evaluations assess the visual fidelity of the synthesized X-ray angiograms and provide a preliminary assessment of their potential to support interventional navigation.

Our main contributions are as follows:

• We propose VeCAS, a two-stage vessel-focused framework for contrast-free angiogram synthesis, providing a computational strategy to generate an angiogram-like vascular visualization from non-contrast X-ray images.

• A cross-modality latent distillation strategy is introduced to transfer vessel-sensitive knowledge from X-ray angiograms to non-contrast X-ray images, improving vascular localization under contrast-free conditions.

• A vessel-focused inpainting method is developed to synthesize angiographic appearance only within vascular regions while preserving the non-vascular background, thereby reducing unnecessary image modification.

• VeCAS outperforms existing methods in vascular structural fidelity and image quality. Reader studies further indicate that the synthesized angiograms exhibit perceptual realism, while phantom navigation experiments demonstrate their practical utility, with navigation time and operation steps reduced by 41.4% and 40.7%, respectively, compared with non-contrast guidance.

The remainder of this paper is organized as follows. Section II reviews related work. Section III describes the proposed method in detail. Section IV presents the experimental setup and results. Finally, Section V concludes the paper.

## II. RELATED WORK

In this section, we review studies related to this work from three perspectives: low-contrast and contrast-free interventions (Section II-A), medical image translation (Section II-B), and angiogram synthesis (Section II-C).

## A. Low-Contrast & Contrast-Free Interventions

Reducing contrast administration has become an important goal in image-guided vascular interventions, especially for patients with chronic kidney disease or other contrast-related risk factors [22]. Several clinical strategies have been explored to reduce or avoid iodinated contrast administration. In coronary interventions, intravascular ultrasound- or physiologyguided ultra-low-contrast and zero-contrast procedures have shown feasibility in patients with renal dysfunction [14], [15], [23]. Roadmap-based guidance has also been investigated to reduce the need for repeated contrast injections by overlaying previously acquired vascular information onto live fluoroscopy [16]. In peripheral vascular interventions, $\mathrm { C O _ { 2 } }$ angiography has been used as an alternative to reduce the use of iodinated contrast agents in high-risk patients [17], [24]. Although these strategies have demonstrated clinical value, they remain limited by their dependence on additional intravascular devices, physiological measurements, alternative contrast agents, or previously acquired angiographic roadmaps. This limitation motivates the development of computational approaches that can synthesize X-ray angiograms directly from non-contrast X-ray images.

![](images/0bee2fd420ef8ea8e4d90168b177867d666e229e5efce4de6be1b7406e1cf3e7.jpg)  
Fig. 2. Overview of VeCAS. VeCAS performs contrast-free angiogram synthesis using a two-stage framework. Stage I: A vascular structure localization model predicts the vessel mask from a non-contrast X-ray image. Stage II: A vessel-focused inpainting model synthesizes angiographic appearance within the predicted vascular regions while preserving the non-vascular background from the input image.

## B. Medical Image Translation

Medical image translation aims to transform images from one imaging domain to another and provides an important technical foundation for angiogram synthesis. Early methods were mainly based on generative adversarial networks (GANs) adapted from natural image translation [20], [21], [25]. These frameworks have been further extended to medical imaging by incorporating deformable registration [26] or improved network architectures [27] to better handle anatomical variations and modality differences. More recently, diffusion models have shown strong generative modeling capability and have been increasingly adopted for image synthesis tasks [28], [29]. Diffusion bridge models are particularly relevant to medical image translation because they explicitly model the transformation between two endpoint distributions [30], with representative methods including BBDM [31] and SelfRDB [32]. Another line of work adapts generative foundation models pre-trained on large-scale natural images [18], such as ControlNet [33] and T2I-Adapter [34], to conditional image translation tasks. Although these methods have achieved promising results, they usually learn a global mapping between image domains and couple vascular localization with angiographic appearance synthesis within a single model. This may result in inaccurate vessel localization and unnecessary background reconstruction, which are particularly undesirable for contrast-free angiogram synthesis. In contrast, VeCAS explicitly decouples localization and synthesis by first predicting a binary vessel mask and then performing vessel-focused inpainting.

## C. Angiogram Synthesis

Angiogram synthesis has attracted increasing attention because of its potential to reduce the reliance on contrast agents in clinical practice. Existing studies have mainly focused on synthesizing computed tomography angiography (CTA) or magnetic resonance angiography (MRA) from non-contrast CT or MRI using generative models [35], [36]. For example, Lyu et al. [37] proposed a GAN-based method for synthesizing CTA from non-contrast CT and demonstrated its potential for vascular disease assessment. Recent diffusion-based methods have further improved vascular structure modeling in angiogram synthesis. Wang et al. [38] introduced a 3D vascular tree state-space diffusion model to better capture vascular geometry in non-contrast 3D volumes, while Li et al. [39] proposed AngioDiff to improve vascular structure preservation and 3D consistency in CTA synthesis. Several studies have also explored generative models for synthesizing X-ray angiograms from non-contrast X-ray images. CAS-GAN [13] introduced a disentangled representation learning framework to model vessel and background features for contrast-free angiogram synthesis. Wang et al. [12] proposed an adversarial diffusion framework that incorporates a parametric vascular model and mask-guided adversarial learning to improve vascular geometric fidelity. However, these methods were mainly developed for coronary angiogram synthesis, where paired training data are difficult to obtain due to substantial cardiac and respiratory motion. In contrast, VeCAS focuses on peripheral angiogram synthesis, where motion-induced deformation is typically less pronounced, making paired learning more feasible.

## III. METHOD

## A. Overview

VeCAS is a two-stage framework for contrast-free angiogram synthesis, as shown in Fig. 2. Given a non-contrast X-ray image, VeCAS aims to synthesize angiogram-like vascular appearance while preserving the original non-vascular background. The key idea is to decouple vascular structure localization from angiographic appearance synthesis. In Stage I (Section III-B), a discriminative model predicts a binary vessel mask $\hat { m } \in \{ 0 , 1 \} ^ { 1 \times H \times W }$ from a non-contrast X-ray image $x _ { \mathrm { ~ N ~ } } \in \ \mathbb { R } ^ { C \times \hat { H } \times \hat { W } }$ , where C, H, and W denote the number of channels, height, and width, respectively. To exploit the complementary vascular information provided by the corresponding X-ray angiogram $\boldsymbol { x } _ { \mathrm { A } } \in \mathbb { R } ^ { C \times \hat { H } \times W }$ during training, we introduce a cross-modality latent distillation strategy based on feature prediction. In Stage II (Section III-C), the predicted vessel mask mˆ is used to guide a vessel-focused inpainting model to synthesize the X-ray angiogram $\hat { x } _ { \mathrm { A } } \in \mathbb { R } ^ { \hat { C } \times H \times W }$ Unlike conventional diffusion-based methods that model the entire image, VeCAS focuses on vascular regions during both training and inference. Specifically, the vessel-focused loss encourages the model to learn angiographic appearance within vascular regions, while the vessel-focused sampling strategy updates only the vascular regions and preserves the background from the input non-contrast X-ray image.

## B. Stage I: Vascular Structure Localization

In this stage, we localize vascular structures from a noncontrast X-ray image x<sub>N</sub> by predicting a binary vessel mask $\hat { m } .$ This step is formulated as a discriminative task rather than a generative modeling problem. This formulation provides explicit spatial supervision for vascular regions, thereby supporting anatomically consistent vessel localization. Given a non-contrast X-ray image, the vessel mask is obtained as:

$$
\hat { m } = \mathcal { D } \left( \mathcal { E } \left( x _ { \mathrm { N } } \right) \right)\tag{1}
$$

where $\mathcal { E } ( \cdot )$ and $\mathcal { D } ( \cdot )$ denote the encoder and decoder of the segmentation model, respectively.

Cross-Modality Latent Distillation. Learning vascular structures directly from non-contrast X-ray images is challenging because vessel cues are often weak and ambiguous without contrast enhancement. Cross-modality knowledge distillation [40]–[42] provides a natural way to exploit vascular priors from X-ray angiograms for improved localization. However, conventional strategies may impose overly restrictive supervision in this setting, since vessels are clearly highlighted in angiograms but may be barely visible in the corresponding non-contrast X-ray images. Under such a substantial modality gap, directly aligning features or logits across modalities may hinder optimization rather than facilitate knowledge transfer. To address this issue, we transfer vascular knowledge through latent feature prediction instead of direct feature or logit alignment, making the distillation process more flexible for vascular localization, as shown in Fig. 3.

Specifically, we first train an angiogram-domain segmentation model $\mathcal { D } _ { \mathrm { { A } } } \left( \mathcal { E } _ { \mathrm { { A } } } ( \cdot ) \right)$ using X-ray angiograms $x _ { \mathrm { A } }$ and their corresponding vessel masks m. Since this model is trained on contrast-enhanced images, it learns vessel-sensitive latent representations. We then freeze its parameters and train a dedicated encoder $\mathcal { E } _ { \mathrm { N } } ( \cdot )$ for non-contrast X-ray images. Following U-Net [43], $\mathcal { E } _ { \mathrm { A } } ( \cdot )$ and $\mathcal { E } _ { \mathrm { N } } ( \cdot )$ share the same architecture and extract multi-scale features from paired inputs $( x _ { \mathrm { A } } , x _ { \mathrm { N } } )$

$$
f _ { \mathrm { A } } ^ { ( i ) } = \mathcal { E } _ { \mathrm { A } } ^ { ( i ) } ( x _ { \mathrm { A } } ) , \quad f _ { \mathrm { N } } ^ { ( i ) } = \mathcal { E } _ { \mathrm { N } } ^ { ( i ) } ( x _ { \mathrm { N } } )\tag{2}
$$

where $i = 1 , 2 , \cdots , 5$ denotes the feature scale.

To bridge the modality gap between contrast-enhanced and non-contrast images, we introduce a multi-scale predictor $\mathcal { P } ( \cdot )$ to estimate angiogram-domain latent features from noncontrast features:

$$
\hat { f } _ { \mathrm { A } } ^ { ( i ) } = \mathcal { P } ^ { ( i ) } \left( f _ { \mathrm { N } } ^ { ( i ) } \right)\tag{3}
$$

![](images/9092f6d506b9a1f225a4957eb818f2ca30da604577c3ec1e8734451f3e86ec0f.jpg)  
Fig. 3. Stage I: Cross-Modality Latent Distillation for Vascular Structure Localization. A frozen angiogram-domain segmentation model $\pmb { \mathcal { D } } _ { \mathbf { A } } \left( \pmb { \mathcal { E } } _ { \mathbf { A } } ( \cdot ) \right)$ provides vessel-sensitive multi-scale features from X-ray angiograms. The trainable non-contrast encoder $\varepsilon _ { \mathbf { N } } ( \cdot )$ and multiscale predictor $\mathcal { P } ( \cdot )$ learn to predict angiogram-domain latent features from non-contrast X-ray images. The predicted features are then decoded by the frozen decoder $\mathcal { \bar { D } } _ { \mathbf { A } } ( \cdot )$ for vessel mask prediction.

where $\mathcal { P } ^ { ( i ) } ( \cdot )$ consists of two consecutive “Conv-BN-ReLU”. Since deeper features usually encode higher-level semantic information [44], [45], the predictor is applied only to highlevel feature maps, $i . e . , i = 3 , 4 , 5$ . An ablation study on this design is provided in the Supplementary Material Table S2.

The predicted multi-scale features are then fed into the frozen angiogram-domain decoder $\mathcal { D } _ { \mathrm { { A } } } ( \cdot )$ to obtain the vessel mask:

$$
\hat { m } = { \cal D } _ { \mathrm { A } } \left( \{ \tilde { f } _ { \mathrm { A } } ^ { ( i ) } \} _ { i = 1 } ^ { 5 } \right)\tag{4}
$$

where

$$
\tilde { f } _ { \mathrm { A } } ^ { ( i ) } = \left\{ \begin{array} { l l } { { f _ { \mathrm { N } } ^ { ( i ) } , } } & { { i = 1 , 2 } } \\ { { \hat { f } _ { \mathrm { A } } ^ { ( i ) } , } } & { { i = 3 , 4 , 5 } } \end{array} \right.\tag{5}
$$

To supervise the latent prediction process, we define the prediction loss as:

$$
\mathcal { L } _ { \mathrm { p r e d } } = \sum _ { i = 3 } ^ { 5 } \left. \hat { f } _ { \mathrm { A } } ^ { ( i ) } - f _ { \mathrm { A } } ^ { ( i ) } \right. _ { 2 } ^ { 2 }\tag{6}
$$

In addition, the predicted vessel mask is constrained by a segmentation loss combining binary cross-entropy loss and Dice loss:

$$
\mathcal { L } _ { \mathrm { s e g } } = 0 . 5 \mathcal { L } _ { \mathrm { B C E } } + 0 . 5 \mathcal { L } _ { \mathrm { D i c e } }\tag{7}
$$

The overall objective of Stage I is formulated as:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = \mathcal { L } _ { \mathrm { s e g } } + \lambda _ { \mathrm { p r e d } } \mathcal { L } _ { \mathrm { p r e d } }\tag{8}
$$

where $\lambda _ { \mathrm { p r e d } }$ controls the trade-off between segmentation supervision and latent prediction. A detailed analysis of $\lambda _ { \mathrm { p r e d } }$ is reported in the Supplementary Material Fig. S6.

Inference. During inference, only the non-contrast X-ray image $x _ { \mathrm { N } }$ is available. It is first passed through the noncontrast encoder $\mathcal { E } _ { \mathrm { N } } ( \cdot )$ to extract multi-scale features. The high-level features are then transformed by the predictor $\mathcal { P } ( \cdot )$ into angiogram-domain latent features. Finally, the vessel mask mˆ is obtained using the angiogram-domain decoder $\mathcal { D } _ { \mathrm { { A } } } ( \cdot )$ and subsequently used as the structural condition for Stage II.

![](images/e375b8085714f10a7be28d4e7bec4ea1c6c0cdd1a6afa410160e32a6cc93279e.jpg)  
Fig. 4. Stage II: Vessel-Focused Inpainting. (a) Training: A vessel-focused denoising loss restricts the diffusion learning objective to vascular regions, encouraging the model to learn angiographic vessel appearance rather than redundant background reconstruction. (b) Inference: At each reverse diffusion step, the denoised vascular output is fused with the forward-diffused non-contrast X-ray image according to the predicted vessel mask from Stage I, thereby synthesizing contrast-enhanced appearance while preserving the non-vascular background.

## C. Stage II: Vessel-Focused Inpainting

The vessel-focused inpainting model is built on diffusion models [28], [46], as illustrated in Fig. 4. The core idea is to learn a vessel-aware reverse diffusion process that synthesizes angiographic appearance within vascular regions while preserving the non-vascular background.

Preliminaries. Diffusion models learn the reverse of a predefined forward noising process, which is formulated as a Markov chain of length T. Given a clean image $x _ { 0 } = x ,$ , the forward process progressively perturbs the image by adding Gaussian noise over T timesteps:

$$
x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \quad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I )\tag{9}
$$

where $t \in \{ 1 , \cdots , T \}$ denotes the diffusion timestep, $\alpha _ { t } =$ $1 - \beta _ { t }$ , and $\begin{array} { r } { \bar { \alpha } _ { t } ~ = ~ \prod _ { i = 1 } ^ { t } \alpha _ { i } } \end{array}$ . Here, $\beta _ { t }$ follows a predefined variance schedule as in DDPM [28].

The reverse process aims to recover $x _ { 0 }$ from the noisy sample $x _ { t } .$ . In practice, a neural network $\epsilon _ { \theta } ( \cdot )$ is trained to estimate the noise ϵ conditioned on $x _ { t }$ and timestep t.

Vessel-Focused Denoising Loss. The standard diffusion objective supervises noise prediction over the entire image. However, for this task, the relevant content is concentrated in vascular regions. To encourage the model to focus on angiographic appearance, we introduce a vessel-focused denoising loss that restricts supervision to the vessel mask:

$$
\mathcal { L } _ { \mathrm { d e n o i s e } } = \mathbb { E } _ { { x } _ { \mathrm { A } } , t , \epsilon } \left[ \| m \odot ( \epsilon _ { \theta } ( x _ { \mathrm { A } , t } , t ) - \epsilon ) \| _ { 2 } ^ { 2 } \right]\tag{10}
$$

where ⊙ denotes element-wise multiplication.

Vessel-Focused Sampling. During inference, we follow diffusion-based inpainting strategies [47], [48] and decouple vessel synthesis from background preservation. Starting from Gaussian noise $x _ { T } \sim \mathcal { N } ( \mathbf { 0 } , I )$ , the model iteratively performs reverse denoising to synthesize vessel appearance within the masked region while preserving the background from the input non-contrast X-ray image. At each step, the intermediate denoised result is computed as:

$$
o _ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( \hat { x } _ { t } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( \hat { x } _ { t } , t ) \right) + \sigma _ { t } z\tag{11}
$$

where $\begin{array} { r } { \sigma _ { t } ^ { 2 } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } } \end{array}$ is the variance of the posterior Gaussian distribution $q \left( x _ { t - 1 } \middle | x _ { t } , x _ { 0 } \right)$ , and $z \sim \mathcal { N } ( \mathbf { 0 } , I )$ .

To preserve the original background while synthesizing only vascular regions, we fuse the denoised output with the input non-contrast image according to the vessel mask:

$$
\hat { x } _ { t - 1 } = o _ { t - 1 } \odot \hat { m } + x _ { \mathrm { N } , t - 1 } \odot ( 1 - \hat { m } )\tag{12}
$$

where $o _ { t - 1 }$ provides the synthesized vascular appearance, and $x _ { \mathrm { N } , t - 1 }$ denotes the forward-diffused non-contrast X-ray image at timestep t−1, which preserves the non-vascular background at the same noise level as the reverse diffusion process. After the final reverse step, the synthesized X-ray angiogram is obtained as $\hat { x } _ { \mathrm { A } } = \hat { x } _ { 0 }$

Training-Inference Input Discrepancy. During training, the diffusion model is supervised using X-ray angiograms x<sub>A</sub> and their corresponding vessel masks m. At inference time, ground-truth masks are unavailable, and the model instead uses the predicted mask mˆ from Stage I together with the noncontrast X-ray image $x _ { \mathrm { N } }$ . We further analyze the impact of mask quality on synthesis performance in Section IV-E.

## IV. RESULTS

## A. Clinical Dataset

The dataset used in this study was collected at Beijing Friendship Hospital, Capital Medical University<sup>1</sup>. It consists of 599 sequences from 102 patients who underwent lower-limb vascular interventions. For each sequence, one pre-contrast frame and one contrast-enhanced frame with clear vessel opacification were manually selected from the same sequence. Given the limited motion in lower-limb acquisitions, frame pairs without evident spatial displacement were treated as approximately aligned. This yielded 599 pairs of non-contrast Xray images and corresponding X-ray angiograms. The vascular regions in all X-ray angiograms were first annotated by a physician using LabelMe [49], and the annotations were then reviewed and corrected by a senior physician.

To evaluate VeCAS, we performed five-fold cross-validation at the patient level. In each fold, 80% of the patients were used for training, and the remaining 20% were used for testing. All sequences from the same patient were assigned to the same subset to avoid patient-level data leakage. The final performance is reported as “mean ± std” across the five folds.

Ethics Approval. This study was approved by the Ethics Committee of Beijing Friendship Hospital, Capital Medical University (Approval No. 2020-P2-073-02). Written informed consent was obtained from all participants before data collection. All images were de-identified during preprocessing, and all procedures were conducted in accordance with the Declaration of Helsinki.

## B. Implementation Details

All experiments were implemented using PyTorch (2.8.0) and Python (3.11.13) under Ubuntu (20.04.3). The input image size was set to $2 5 6 \times 2 5 6$ for all experiments.

Stage I. We adopted U-Net [43] as the backbone for vascular structure localization. The angiogram-domain segmentation model $\mathcal { D } _ { \mathrm { A } } \left( \mathcal { E } _ { \mathrm { A } } ( \cdot ) \right)$ was trained for 100 epochs using AdamW [50] with a learning rate of $3 \times 1 0 ^ { - 4 }$ . Subsequently, the angiogram-domain model was frozen, and the non-contrast encoder $\mathcal { E } _ { \mathrm { N } } ( \cdot )$ together with the multi-scale predictor $\mathcal { P } ( \cdot )$ was trained for 100 epochs using AdamW with a learning rate of $4 \times 1 0 ^ { - 4 }$ . The batch size was set to 16. Training was performed on a single NVIDIA RTX 4090 GPU with 24 GB of memory.

Stage II. The vessel-focused inpainting model was implemented using the Diffusers<sup>2</sup> library and operated in image space. The model was optimized using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 8 for 300 epochs. Training was performed on a single NVIDIA RTX 4090 GPU with 24 GB of memory. During inference, the number of sampling steps was set to 100. The influence of the number of sampling steps is analyzed in the Supplementary Material Table S4.

TABLE I  
RESULTS OF THE VISUAL TURING TEST. SENSITIVITY (SEN.), SPECIFICITY (SPE.), AND ACCURACY (ACC.) ARE REPORTED FOR EACH READER.
<table><tr><td>Metric</td><td>R1</td><td>R2</td><td>R3</td><td>R4</td><td>mean ± std</td></tr><tr><td>Sen. (%)</td><td>62.00</td><td>76.00</td><td>94.00</td><td>78.00</td><td> $7 7 . 5 0 \pm 1 3 . 1 0$ </td></tr><tr><td>Spe. (%)</td><td>50.00</td><td>52.00</td><td>62.00</td><td>78.00</td><td> $6 0 . 5 0 \pm 1 2 . 7 9$ </td></tr><tr><td>Acc. (%)</td><td>56.00</td><td>64.00</td><td>78.00</td><td>78.00</td><td> $6 9 . 0 0 \pm 1 0 . 8 9$ </td></tr></table>

Positives: Real X-ray angiograms.  
Negatives: Synthesized X-ray angiograms.

Evaluation Metrics. For Stage I, we report the Dice similarity coefficient (DSC) and centerline Dice (clDice) [51]. DSC measures the pixel-level overlap between the predicted and ground-truth vessel masks, while clDice better reflects the topological consistency of tubular vascular structures. For Stage II, DSC and clDice are also used to evaluate the vascular structural fidelity of synthesized X-ray angiograms. Specifically, the angiogram-domain segmentation model $\mathcal { D } _ { \mathrm { { A } } } \left( \mathcal { E } _ { \mathrm { { A } } } ( \cdot ) \right)$ trained on real X-ray angiograms, is used to extract vessel masks from synthesized angiograms. In addition, we report two commonly used image-quality metrics, including peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM), to evaluate whole-image fidelity.

## C. Perceptual and Physician Assessment

Visual Turing Test. To assess the perceptual realism of the synthesized X-ray angiograms, we conducted a Visual Turing Test [52]. Four independent readers with different levels of expertise evaluated 100 images, including 50 real X-ray angiograms and 50 synthesized X-ray angiograms, in a randomized and blinded setting. As reported in Table I, the readers achieved an average Sen. of 77.50%, Spe. of 60.50%, and Acc. of 69.00%. Because synthesized images were treated as negative samples, the relatively low Spe. indicates that readers often classified them as real, suggesting that some synthesized angiograms were visually difficult to distinguish from real angiograms.

Physician Assessment. To further assess the potential procedural usefulness of the synthesized angiograms, three experienced physicians independently evaluated 50 synthesized angiograms by comparing them with the corresponding real angiograms. The assessment used three criteria: vascular morphological realism (VMR), contrast opacification realism (COR), and procedural usability (PU).

• Vascular Morphological Realism evaluates whether the synthesized angiograms are consistent with the real angiograms in terms of vessel locations and trajectories, continuity, boundary definition, and major branches.

• Contrast Opacification Realism assesses the naturalness of the synthesized contrast appearance with respect to intravascular intensity distribution, contrast variation, and edge transitions.

• Procedural Usability measures whether the synthesized angiograms provide useful information about vessel trajectories and major branches for potential procedural guidance.

![](images/96b287e341af8f6bf8b43e44a16c72c04bf5663c5fe9918001565be874c735fc.jpg)  
Fig. 5. Physician Assessments of Synthesized Angiograms. Three physicians independently scored the synthesized angiograms using three criteria: vascular morphological realism (VMR), contrast opacification realism (COR), and procedural usability (PU).

A five-point Likert scale was used for each criterion, with higher scores indicating better perceived quality. Detailed scoring rules are provided in the Supplementary Material Section S1. As shown in Fig. 5, the synthesized angiograms received average scores above 3.00 from all three physicians across the three criteria. The pairwise linearly weighted Cohen’s κ coefficients ranged from 0.46 to 0.84, indicating moderate to high inter-physician agreement. Overall, these results suggest that the synthesized angiograms achieve generally acceptable quality and may provide useful angiographic visualization from non-contrast X-ray images.

TABLE II  
QUANTITATIVE COMPARISON OF DIFFERENT METHODS FOR STAGE I.RESULTS ARE REPORTED $\mathbf { A S \ ^ { * } m e a n } \pm \mathbf { s t d } ^ { \prime \prime }$ OVER FIVE-FOLDCROSS-VALIDATION. THE BEST AND SECOND-BEST RESULTS AREHIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY. STATISTICALSIGNIFICANCE AGAINST VECAS IS ASSESSED USING WILCOXONSIGNED-RANK TESTS ON PATIENT-LEVEL SCORES $( \ast ; p < 0 . 0 5$ , ∗∗:$p < 0 . 0 1 , * * * : p < 0 . 0 0 1 )$
<table><tr><td>Method</td><td>DSC (%) ↑</td><td>clDice (%) ↑</td></tr><tr><td>Generative Models</td><td></td><td></td></tr><tr><td>Pix2Pix [20] [CVPR&#x27;17]</td><td> $9 . 7 5 \pm 0 . 9 2 ^ { \ast \ast \ast }$ </td><td> $9 . 5 0 \pm 1 . 1 0 ^ { * * * }$ </td></tr><tr><td>CycleGAN [21] [ICCV&#x27;17]</td><td> $2 . 3 0 \pm 2 . 4 7 ^ { \ast \ast \ast }$ </td><td> $2 . 8 9 \pm 3 . 4 2 ^ { \ast \ast \ast }$ </td></tr><tr><td>ResViT [27] [TMI&#x27;22]</td><td> $3 7 . 0 3 \pm 7 . 0 9 ^ { \ast \ast \ast }$ </td><td> $4 2 . 6 9 \pm 7 . 9 3 ^ { \ast \ast \ast }$ </td></tr><tr><td>BBDM [31] [CVPR&#x27;23]</td><td> $5 . 4 3 \pm 1 . 6 7 ^ { \ast \ast \ast }$ </td><td> $5 . 3 8 \pm 1 . 8 6 ^ { \ast \ast \ast }$ </td></tr><tr><td>ControlNet [33] [IcCV&#x27;23]</td><td> $8 . 9 3 \pm 0 . 6 4 ^ { \ast \ast \ast }$ </td><td> $8 . 4 9 \pm 0 . 7 6 ^ { \ast \ast \ast }$ </td></tr><tr><td>Segmentation Models</td><td></td><td></td></tr><tr><td>U-Net [43] [MICCAI&#x27;15]</td><td> $5 1 . 2 9 \pm 2 . 9 1 ^ { \ast \ast }$ </td><td> $5 9 . 8 5 \pm 3 . 9 8$ </td></tr><tr><td>Attention U-Net [54] [MedIA&#x27;19]</td><td> $5 0 . 9 8 \pm 3 . 0 6 ^ { \ast \ast }$ </td><td> $5 9 . 4 9 \pm 3 . 8 4 ^ { \ast }$ </td></tr><tr><td>UNet++ [53] [TMr&#x27;20]</td><td> $5 1 . 3 9 \pm 3 . 5 2 ^ { \ast \ast \ast }$ </td><td> $5 9 . 8 7 \pm 4 . 3 9 ^ { \ast }$ </td></tr><tr><td>UCTransNet [55] [AAAI&#x27;22]</td><td> $\overline { { 5 1 . 0 7 \pm 3 . 4 8 } } { ^ { \ast \ast } }$ </td><td> $5 9 . 4 4 \pm 4 . 3 1 ^ { \ast }$ </td></tr><tr><td>VM-UNet [56] [ACM TOMM&#x27;25]</td><td> $4 7 . 6 4 \pm 5 . 6 5 ^ { \ast \ast \ast }$ </td><td> $5 4 . 9 5 \pm 7 . 5 2 ^ { \ast \ast \ast }$ </td></tr><tr><td>MambaLiteUNet [57] [CVPR&#x27;26]</td><td> $4 5 . 9 3 \pm 3 . 7 1 ^ { \ast \ast \ast }$ </td><td> $5 2 . 5 9 \pm 4 . 6 8 ^ { \ast \ast \ast }$ </td></tr><tr><td>Cross-modality Interaction Methods</td><td></td><td></td></tr><tr><td>Joint Learning</td><td> $5 0 . 8 6 \pm 3 . 6 6 ^ { * * }$ </td><td></td></tr><tr><td></td><td></td><td> $5 8 . 7 4 \pm 4 . 8 6 ^ { * * }$ </td></tr><tr><td>KD-Net [40] [MICCAr&#x27;20]</td><td> $5 1 . 2 6 \pm 2 . 9 1 ^ { \ast \ast }$ </td><td> $6 0 . 5 9 \pm 4 . 1 7$ </td></tr><tr><td>PMKL [41] [TMI&#x27;22]</td><td> $5 0 . 6 0 \pm 3 . 7 9 ^ { \ast \ast \ast }$ </td><td> $\overline { { 5 9 . 0 3 \pm 4 . 8 6 } } * *$ </td></tr><tr><td>ProtoKD [42] [ICASSP&#x27;23]</td><td> $4 9 . 9 2 \pm 3 . 0 9 ^ { \ast \ast \ast }$ </td><td> $5 7 . 9 2 \pm 3 . 9 1 ^ { \ast \ast \ast }$ </td></tr><tr><td>CroDiNo-KD [58] [ECML-PKDD&#x27;25]</td><td> $4 9 . 0 4 \pm 3 . 1 3 ^ { \ast \ast \ast }$ </td><td> $5 6 . 8 4 \pm 3 . 7 5 ^ { \ast \ast \ast }$ </td></tr><tr><td>VeCAS (Stage I) [Ours]</td><td> ${ \bf 5 2 . 5 6 \pm 3 . 3 2 }$ </td><td> ${ \bf 6 0 . 6 2 \pm 4 . 4 2 }$ </td></tr></table>

## D. Main Results

Stage I. Table II reports the quantitative results of vascular structure localization. The compared methods include generative models, segmentation models, and cross-modality interaction methods. Generative models, such as Pix2Pix [20], CycleGAN [21], BBDM [31], and ControlNet [33], show limited performance, indicating that direct image translation is not well suited for localizing vascular structures from non-contrast X-ray images. In contrast, segmentation models achieve substantially better results, supporting our formulation of vascular localization as a discriminative task. Among the compared segmentation models, UNet++ [53] obtains the best performance. Nevertheless, VeCAS further improves DSC and clDice by +1.17% and +0.75%, respectively, demonstrating the benefit of cross-modality latent distillation. Compared with existing cross-modality interaction methods, VeCAS also achieves the best overall performance, suggesting that latent feature prediction is more effective than direct feature or logit alignment given the large appearance gap between noncontrast X-ray images and X-ray angiograms. Qualitative results are provided in the Supplementary Material Fig. S3.

Stage I + II. Table III reports the quantitative comparison of the complete pipeline. VeCAS achieves the best vascular structural fidelity, with a DSC of 42.27% and a clDice of 47.70%. Compared with the second-best method for each metric, VeCAS improves DSC by +5.84% over ControlNet and clDice by +5.92% over RegGAN [26]. These results indicate that explicitly decoupling vascular structure localization from angiographic appearance synthesis improves vascular structural fidelity. Several baseline methods obtain competitive PSNR or SSIM, while their DSC and clDice remain much lower than those of VeCAS. This suggests that whole-image metrics are dominated by the large non-vascular background and may not fully reflect vascular structural fidelity. In contrast, VeCAS achieves a better balance between image fidelity and vascular structural fidelity by preserving the background while focusing synthesis on vascular regions. Fig. 6 provides qualitative comparisons of different methods. Most existing methods preserve the background appearance but generate weak, incomplete, or barely visible vascular structures. By contrast, VeCAS produces clearer and more continuous vessels while maintaining the non-vascular background.

QUANTITATIVE COMPARISON OF DIFFERENT METHODS FOR STAGE I + II. RESULTS ARE REPORTED AS “mean ± std” OVER FIVE-FOLD CROSS-VALIDATION. THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY. STATISTICAL SIGNIFICANCE AGAINST VECAS IS ASSESSED USING WILCOXON SIGNED-RANK TESTS ON PATIENT-LEVEL SCORES (∗: $\mathbf { \mathit { p } } < \mathbf { \mathit { 0 . 0 5 } }$ , ∗∗: $p < 0 . 0 1 , * * * : p < 0 . 0 0 1 )$  
TABLE III
<table><tr><td>Method</td><td>DSC (%) ↑</td><td>clDice (%) ↑</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td></tr><tr><td>Pix2Pix [20] [CVPR&#x27;17]</td><td> $1 3 . 1 1 \pm 2 . 2 6 ^ { \ast \ast \ast }$ </td><td> $\overline { { 1 4 . 7 8 \pm 2 . 4 6 ^ { * * * } } }$ </td><td> $3 6 . 0 6 \pm 1 . 2 0 ^ { * * }$ </td><td>0.9355 ± 0.0173***</td></tr><tr><td>CycleGAN [21] [ICCV 17]</td><td>26.41 ± 4.07***</td><td> $3 1 . 9 3 \pm 5 . 0 5 ^ { \ast \ast \ast }$ </td><td>35.71 ± 0.74***</td><td>0.9498 ± 0.0075***</td></tr><tr><td>CUT [25] [ECCV 20]</td><td> $2 6 . 7 2 \pm 2 . 9 6 ^ { \ast \ast \ast }$ </td><td> $3 2 . 1 3 \pm 3 . 7 1 ^ { \ast \ast \ast }$ </td><td> $3 5 . 3 4 \pm 0 . 5 1 ^ { \ast \ast \ast }$ </td><td> $\overline { { 0 . 9 0 1 5 \pm 0 . 0 5 7 4 } } * * *$ </td></tr><tr><td>RegGAN [26] [NeurIPS&#x27;21]</td><td> $3 5 . 6 0 \pm 4 . 7 5 ^ { \ast \ast \ast }$ </td><td> $\frac { 4 1 . 7 8 \pm 6 . 3 3 ^ { * * * } } { - }$ </td><td> $3 3 . 2 7 \pm 1 . 1 6 ^ { * * * }$ </td><td> $0 . 9 4 3 0 \pm 0 . 0 0 6 5 ^ { \ast \ast \ast }$ </td></tr><tr><td>ResViT [27] [7Mr22]</td><td> $3 1 . 0 6 \pm 3 . 9 6 ^ { \ast \ast \ast }$ </td><td>35.78 ± 5.20***</td><td> $3 3 . 5 6 \pm 1 . 0 \dot { 1 } ^ { \ast \ast \ast }$ </td><td> $0 . 9 4 0 0 \pm 0 . 0 0 8 0 ^ { \ast \ast \ast }$ </td></tr><tr><td>BBDM [31] [CVPR 23]</td><td> $3 0 . 8 0 \pm 3 . 9 5 ^ { \ast \ast \ast }$ </td><td> $3 6 . 5 4 \pm 5 . 2 3 ^ { \ast \ast \ast }$ </td><td> ${ \bf 3 7 . 6 7 \pm 0 . 5 5 }$ </td><td> $0 . 9 4 8 8 \pm 0 . 0 2 0 1 ^ { \ast \ast \ast }$ </td></tr><tr><td>ControlNet [33] [ICCV23]</td><td> $3 6 . 4 3 \pm 3 . 4 4 ^ { \ast \ast \ast }$ </td><td> $4 1 . 1 9 \pm 4 . 3 8 ^ { \ast \ast \ast }$ </td><td> $2 6 . 0 4 \pm 1 . 0 4 ^ { \ast \ast \ast }$ </td><td> $0 . 7 6 5 2 \pm 0 . 0 5 1 1 ^ { \ast \ast \ast }$ </td></tr><tr><td>SelfRDB [32] [MedlA·25]</td><td> $\overline { { 2 4 . 7 9 \pm 7 . 0 3 } } ^ { \ast \ast \ast }$ </td><td> $2 8 . 8 4 \pm 8 . 7 1 ^ { \ast \ast \ast }$ </td><td> $2 9 . 7 2 \pm 4 . 9 1 ^ { * * * }$ </td><td>0.8584 ± 0.0422***</td></tr><tr><td>VeCAS (Stage I + II) [Ours]</td><td> ${ \bf 4 2 . 2 7 \pm 4 . 9 9 }$ </td><td> ${ \bf 4 7 . 7 0 \pm 6 . 2 9 }$ </td><td>36.78 ± 0.91</td><td>0.9537± 0.0183</td></tr></table>

## E. Ablation Study

Distillation Loss ${ \mathcal { L } } _ { \mathrm { d i s t i l l } } .$ . Table IV reports the ablation results of the distillation loss in Stage I. Using only $\mathcal { L } _ { \mathrm { p r e d } }$ leads to poor performance, indicating that latent feature prediction alone is insufficient for accurate vascular localization.

![](images/9ea8452590b856964ade3fb865d054cbb8c7366065378a6ede9969f78ff2b0f1.jpg)  
Fig. 6. Qualitative Results of the Complete Pipeline (Stage $1 + \mathbb { I I } ) .$ Compared with baseline methods, VeCAS synthesizes clearer and more continuous vascular structures while preserving the non-vascular background. Most baseline methods retain the background appearance but produce weak, incomplete, or barely visible vessels.

![](images/dfc5a9778866a8008b1ef7170bd75da691fcc29d1fb7cc5d03c22a22e98be873.jpg)

![](images/cb426ee6a5882ca2dbd0e9cb7bbc7542a0852e4d74a926cb9ff5728403aaba17.jpg)

![](images/23f1127383af81a25dd35630c1790bff7adcc636fb31d8a272165bb66fa1b011.jpg)  
Fig. 7. Error Propagation Analysis under Different Mask Conditions. Stage II is evaluated using predicted masks from VeCAS, perturbed ground-truth masks, and oracle ground-truth masks to analyze how mask quality affects performance.

![](images/19a2e72597c6ba5ffef44b2069edc7ea7083c3ec3bf71b18a20531dbff82d65d.jpg)

TABLE IV  
ABLATION STUDY OF THE DISTILLATION LOSS $\pmb { \mathcal { L } } _ { \mathbf { d i s t i l l } }$ IN STAGE I. THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td> $\mathcal { L } _ { \mathrm { s e g } }$ </td><td> $\mathcal { L } _ { \mathrm { p r e d } }$ </td><td> $\mathrm { D S C } ~ ( \% ) ~ \uparrow$ </td><td> $\mathrm { c l D i c e } \ ( \% ) \ \uparrow$ </td></tr><tr><td>X</td><td>√</td><td> $2 9 . 4 9 \pm 3 . 5 3$ </td><td> $3 5 . 5 2 \pm 2 . 7 4$ </td></tr><tr><td>V</td><td>X</td><td> $5 1 . 4 9 \pm 3 . 4 7$ </td><td> $5 9 . 6 6 \pm 4 . 6 3 $ </td></tr><tr><td></td><td></td><td> ${ \bf 5 2 . 5 6 \pm 3 . 3 2 }$ </td><td> ${ \bf 6 0 . 6 2 \pm 4 . 4 2 }$ </td></tr></table>

TABLE V

ABLATION STUDY OF THE VESSEL-FOCUSED DESIGN IN STAGE II. THEBEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD ANDUNDERLINED, RESPECTIVELY.
<table><tr><td colspan="2">Vessel-focused</td><td rowspan="2"> $\mathrm { D S C } ~ ( \% ) ~ \uparrow$ </td><td rowspan="2"> $\mathrm { c l D i c e } \left( \% \right) \uparrow$ </td><td rowspan="2"> $\mathrm { P S N R \ ( d B ) \uparrow }$ </td><td rowspan="2">SSIM ↑</td></tr><tr><td>Loss</td><td>Sampling</td></tr><tr><td>××√</td><td>X</td><td> $1 1 . 4 3 \pm 3 . 7 6$ </td><td> $1 2 . 5 5 \pm 4 . 2 3$ </td><td> $2 9 . 6 9 \pm 0 . 3 4$ </td><td> $0 . 9 2 5 7 \pm 0 . 0 1 5 2$ </td></tr><tr><td></td><td></td><td> $1 1 . 3 8 \pm 3 . 0 3$ </td><td> $1 1 . 8 9 \pm 3 . 5 6$ </td><td> $3 6 . 1 9 \pm 0 . 9 3$ </td><td> $\underline { { 0 . 9 5 0 8 \pm 0 . 0 1 7 8 } }$ </td></tr><tr><td></td><td>√×</td><td> $2 0 . 5 1 \pm 4 . 3 2$ </td><td> $2 2 . 9 6 \pm 5 . 0 8$ </td><td> $\overline { { 3 1 . 6 6 \pm 0 . 5 8 } }$ </td><td> $\overline { { 0 . 9 3 2 2 \pm 0 . 0 1 6 9 } }$ </td></tr><tr><td></td><td></td><td> $\pm { \bf { 4 2 . 2 7 \pm 4 . 9 9 } }$ </td><td> $\mathbf { 4 7 . 7 0 \pm 6 . 2 9 }$ </td><td>一  ${ \bf 3 6 . 7 8 \pm 0 . 9 1 }$ </td><td> $\mathbf { 0 . 9 5 3 7 \pm 0 . 0 1 8 3 }$ </td></tr></table>

In contrast, $\mathcal { L } _ { \mathrm { s e g } }$ provides direct pixel-level supervision and substantially improves the results. Combining $\mathcal { L } _ { \mathrm { s e g } }$ and $\mathcal { L } _ { \mathrm { p r e d } }$ achieves the best performance, demonstrating that latent prediction provides complementary cross-modality supervision beyond standard segmentation training.

Vessel-Focused Design. Table V reports the ablation results of the vessel-focused design in Stage II. Removing both the vessel-focused loss and vessel-focused sampling results in poor performance across all metrics. Using only vesselfocused sampling improves PSNR and SSIM, but DSC and clDice remain low, suggesting that this strategy mainly benefits background preservation. In contrast, using only the vesselfocused loss improves DSC and clDice, indicating that emphasizing vascular regions during training is important for learning angiographic appearance. Combining both components yields the best overall performance. These results show that the vessel-focused loss and the vessel-focused sampling play complementary roles. Qualitative results are provided in the Supplementary Material Fig. S5.

Error Propagation from Stage I to Stage II. Since VeCAS adopts a two-stage design, we further analyze how mask quality affects the final synthesis results. Stage II is evaluated using the predicted masks from Stage I, the groundtruth masks, and several perturbed ground-truth masks. As shown in Fig. 7, using the ground-truth mask provides an oracle upper bound, achieving 60.22% DSC and 66.54% clDice. In comparison, the complete VeCAS pipeline using the predicted mask achieves 42.27% DSC and 47.70% clDice, indicating that mask quality directly affects vascular fidelity. Among different perturbations, random translation causes the largest performance degradation, suggesting that the inpainting model is sensitive to spatial misalignment because the mask determines where angiographic appearance is synthesized. Erosion and dilation also degrade the results, suggesting that accurate vessel boundaries and calibers are important for preserving vascular structural fidelity. In contrast, local vessel breaks and false-positive branches cause less performance degradation, likely because the global vessel location is still largely preserved. PSNR and SSIM remain relatively stable across different inputs, further confirming that whole-image fidelity metrics are dominated by the preserved background and may be insensitive to vascular structural errors.

![](images/5fab8384fa57c6b102e5ab9a41e40ec13b14f9a2f02b3fbe47c3121503bf306e.jpg)  
(a) Phantom Experimental Setup

![](images/af4bf6ddd03c2e7a86e4dbd9df409c1042c1c16d95da1f06d1dcd39eb58ba99d.jpg)

![](images/80128edb30604df444a585c2d7a6e17ff5f58c9901671e33e8bccebcb9862bdf.jpg)  
(b) Phantom Navigation Results  
Fig. 8. Phantom Experiment. (a) Phantom experimental setup. (b) Navigation efficiency under different guidance conditions, evaluated by time to target and the number of operation steps. ❶ non-contrast guidance, ❷ VeCAS guidance, and ❸ contrast guidance.

## F. Phantom Experiment

Experimental Setup. A simulated vascular intervention environment, as shown in Fig. 8(a), was constructed to evaluate the feasibility of VeCAS-guided guidewire navigation. The platform consists of a vascular phantom, a vascular robotic system [59], a guidewire, an overhead camera, and a local workstation. Each vascular phantom was fabricated based on the vessel annotation of a real X-ray angiogram, and the corresponding non-contrast X-ray image was spatially aligned with the phantom. During navigation, the guidewire was manipulated by the robotic system using an Xbox game controller, following the control scheme described in [60]. The guidewire was captured in real time, segmented using threshold-based preprocessing, and overlaid on the displayed image to simulate visual feedback during image-guided interventions. For VeCAS-guided navigation, the non-contrast X-ray image was sent from the local workstation to a remote server via SSH, and the synthesized X-ray angiogram was then transferred back and displayed on the local interface. More details are provided in the Supplementary Material Section S2.

Protocol. Three operators were invited to perform guidewire navigation on two vascular phantoms. For each phantom, each operator was asked to navigate the guidewire tip from the inlet to a predefined distal target point. Three guidance conditions were compared: ❶ non-contrast guidance, ❷ VeCAS guidance, and ❸ contrast guidance. For each operator, phantom, and guidance condition, the task was repeated five times.

Evaluation Metrics. Navigation performance was evaluated using two task-level metrics: time to target and the number of operation steps. Time to target was measured from the first control command to the arrival of the guidewire tip at the target region. The number of operation steps was used to quantify manipulation complexity, where one step was defined as one discrete control command, such as guidewire advancement, withdrawal, clockwise rotation, or counterclockwise rotation.

Main Results. As shown in Fig. 8(b), VeCAS guidance substantially improved robotic guidewire navigation efficiency compared with non-contrast guidance. Specifically, the time to target was reduced by 41.4%, and the number of operation steps was reduced by 40.7%. VeCAS guidance also achieved navigation performance comparable to contrast guidance. These results demonstrate the potential of VeCAS to provide contrast-free visual guidance for interventional navigation.

## V. CONCLUSION

In this paper, we present VeCAS, a Vessel-focused Contrastfree Angiogram Synthesis framework. VeCAS aims to provide vessel-enhanced visual guidance from non-contrast X-ray images by explicitly decoupling vascular structure localization from angiographic appearance synthesis. Cross-modality latent distillation is used to improve vascular localization by transferring vessel-sensitive knowledge from X-ray angiograms during training. The vessel-focused inpainting model then selectively synthesizes angiographic appearance within the localized vascular regions while preserving the non-vascular background. Quantitative experiments, physician assessments, and robotic guidewire navigation experiments in vascular phantoms collectively support the performance of VeCAS and suggest its potential to reduce the need for iodinated contrast administration in image-guided vascular interventions.

## REFERENCES

[1] A. K. Thukkani and S. Kinlay, “Endovascular intervention for peripheral artery disease,” Circ. Res., vol. 116, no. 9, pp. 1599–1613, 2015.

[2] J. S. Hiramoto, M. Teraa, G. J. de Borst, and M. S. Conte, “Interventions for lower extremity peripheral artery disease,” Nat. Rev. Cardiol., vol. 15, no. 6, pp. 332–350, 2018.

[3] H. Zhao et al., “Large-scale pretrained frame generative model enables real-time low-dose DSA imaging: An AI system development and multicenter validation study,” Med, vol. 6, no. 1, p. 100497, 2025.

[4] H. Zhao et al., “Generative AI-based low-dose digital subtraction angiography for intra-operative radiation dose reduction: A randomized controlled trial,” Nat. Med., vol. 32, pp. 288–296, 2026.

[5] O. Clement et al., “Immediate hypersensitivity to contrast agents: The French 5-year CIRTACI study,” Lancet Discov. Sci., vol. 1, pp. 51–61, 2018.

[6] M. Fahling, E. Seeliger, A. Patzak, and P. B. Persson, “Understanding¨ and preventing contrast-induced acute kidney injury,” Nat. Rev. Nephrol., vol. 13, no. 3, pp. 169–180, 2017.

[7] R. Mehran and E. Nikolsky, “Contrast-induced nephropathy: Definition, epidemiology, and patients at risk,” Kidney Int., vol. 69, pp. S11–S15, 2006.

[8] S. Tehrani, C. Laing, D. M. Yellon, and D. J. Hausenloy, “Contrastinduced acute kidney injury following PCI,” Eur. J. Clin. Invest., vol. 43, no. 5, pp. 483–490, 2013.

[9] D. Giacoppo et al., “Impact of contrast-induced acute kidney injury after percutaneous coronary intervention on short-and long-term outcomes: Pooled analysis from the HORIZONS-AMI and ACUITY trials,” Circ. Cardiovasc. Interv., vol. 8, no. 8, p. e002475, 2015.

[10] B. A. Bartholomew et al., “Impact of nephropathy after percutaneous coronary intervention and a method for risk stratification,” Am. J. Cardiol., vol. 93, no. 12, pp. 1515–1519, 2004.

[11] A. P. Amin, R. G. Bach, M. L. Caruso, K. F. Kennedy, and J. A. Spertus, “Association of variation in contrast volume with acute kidney injury in patients undergoing percutaneous coronary intervention,” JAMA Cardiol., vol. 2, no. 9, pp. 1007–1012, 2017.

[12] Z. Wang, R. Yi, X. Wen, C. Zhu, K. Xu, and K. He, “Angio-Diff: Learning a self-supervised adversarial diffusion model for angiographic geometry generation,” The Vis. Comput., vol. 41, no. 12, pp. 10 303– 10 315, 2025.

[13] D.-X. Huang et al., “CAS-GAN for contrast-free angiography synthesis,” in Proc. SSCI, 2025, DOI: 10.1109/CISM64958.2025.11060856.

[14] Z. A. Ali et al., “Imaging-and physiology-guided percutaneous coronary intervention without contrast administration in advanced renal failure: A feasibility, safety, and outcome study,” Eur. Heart J., vol. 37, no. 40, pp. 3090–3095, 2016.

[15] K. Shibata et al., “Feasibility, safety, and long-term outcomes of zerocontrast percutaneous coronary intervention in patients with chronic kidney disease,” Circ. J., vol. 86, no. 5, pp. 787–796, 2022.

[16] B. Hennessey et al., “Dynamic coronary roadmap versus standard angiography for percutaneous coronary intervention: The randomised, multicentre DCR4Contrast trial,” EuroIntervention, vol. 20, no. 3, p. e198, 2024.

[17] S.-R. Lee et al., “Carbon dioxide angiography during peripheral vascular interventions is associated with decreased cardiac and renal complications in patients with chronic kidney disease,” J. Vasc. Surg., vol. 78, no. 1, pp. 201–208, 2023.

[18] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proc. CVPR, 2022, pp. 10 684–10 695.

[19] T. Brooks et al., “Video generation models as world simulators,” OpenAI Blog, 2024.

[20] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proc. CVPR, 2017, pp. 1125– 1134.

[21] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proc. ICCV, 2017, pp. 2223–2232.

[22] M. Almendarez et al., “Procedural strategies to reduce the incidence of contrast-induced acute kidney injury during percutaneous coronary intervention,” JACC Cardiovasc. Interv., vol. 12, no. 19, pp. 1877–1888, 2019.

[23] T. Truong et al., “Ultra-low contrast IVUS-guided PCI in patients with severe chronic kidney disease: An observational prospective study,” Circ. Cardiovasc. Interv., vol. 18, no. 2, p. e014854, 2025.

[24] R. P. Wawer Matos Reimer et al., “Safety and evidence of CO2 as a vascular contrast agent as an alternative to iodine-based contrast media in vascular procedures: A systematic review by the ESUR Contrast Medium Safety Committee,” Eur. Radiol., vol. 35, no. 12, pp. 7680– 7687, 2025.

[25] T. Park, A. A. Efros, R. Zhang, and J.-Y. Zhu, “Contrastive learning for unpaired image-to-image translation,” in Proc. ECCV, 2020, pp. 319– 345.

[26] L. Kong et al., “Breaking the dilemma of medical image-to-image translation,” in Proc. NeurIPS, vol. 34, 2021, pp. 1964–1978.

[27] O. Dalmaz, M. Yurt, and T. C¸ ukur, “ResViT: Residual vision transformers for multimodal medical image synthesis,” IEEE TMI, vol. 41, no. 10, pp. 2598–2614, 2022.

[28] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Proc. NeurIPS, vol. 33, 2020, pp. 6840–6851.

[29] P. Dhariwal and A. Nichol, “Diffusion models beat GANs on image synthesis,” in Proc. NeurIPS, vol. 34, 2021, pp. 8780–8794.

[30] L. Zhou, A. Lou, S. Khanna, and S. Ermon, “Denoising diffusion bridge models,” in Proc. ICLR, 2024.

[31] B. Li, K. Xue, B. Liu, and Y.-K. Lai, “BBDM: Image-to-image translation with brownian bridge diffusion models,” in Proc. CVPR, 2023, pp. 1952–1961.

[32] F. Arslan, B. Kabas, O. Dalmaz, M. Ozbey, and T. C¸ ukur, “Selfconsistent recursive diffusion bridge for medical image translation,” MedIA, vol. 106, p. 103747, 2025.

[33] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proc. ICCV, 2023, pp. 3836–3847.

[34] C. Mou et al., “T2I-Adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models,” in Proc. AAAI, vol. 38, no. 5, 2024, pp. 4296–4304.

[35] Y. Xia, N. Ravikumar, T. Lassila, and A. F. Frangi, “Virtual highresolution MR angiography from non-angiographic multi-contrast MRIs: Synthetic vascular model populations for in-silico trials,” MedIA, vol. 87, p. 102814, 2023.

[36] A. Koch et al., “Cross-modality image synthesis from TOF-MRA to CTA using diffusion-based models,” MedIA, p. 103722, 2025.

[37] J. Lyu et al., “Generative adversarial network–based noncontrast CT angiography for aorta and carotid arteries,” Radiology, vol. 309, no. 2, p. e230681, 2023.

[38] Z. Wang, R. Yi, X. Wen, C. Zhu, and K. Xu, “VasTSD: Learning 3D vascular tree-state space diffusion model for angiography synthesis,” in Proc. CVPR, 2025, pp. 15 693–15 702.

[39] A. Li, W. Fang, G. Yang, and M. Xu, “AngioDiff: Structure-preserving and 3D-consistent diffusion for CT angiography synthesis,” in Proc. BIBM, 2025, pp. 2429–2434.

[40] M. Hu et al., “Knowledge distillation from multi-modal to mono-modal segmentation networks,” in Proc. MICCAI, 2020, pp. 772–781.

[41] C. Chen, Q. Dou, Y. Jin, Q. Liu, and P. A. Heng, “Learning with privileged multimodal knowledge for unimodal segmentation,” IEEE TMI, vol. 41, no. 3, pp. 621–632, 2022.

[42] S. Wang, Z. Yan, D. Zhang, H. Wei, Z. Li, and R. Li, “Prototype knowledge distillation for medical segmentation with missing modality,” in Proc. ICASSP, 2023, DOI: 10.1109/ICASSP49357.2023.10095014.

[43] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Proc. MICCAI, 2015, pp. 234–241.

[44] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in Proc. CVPR, 2015, pp. 3431–3440.

[45] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in Proc. CVPR, 2017, pp. 2117–2125.

[46] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in Proc. ICML, 2015, pp. 2256–2265.

[47] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool, “RePaint: Inpainting using denoising diffusion probabilistic models,” in Proc. CVPR, 2022, pp. 11 461–11 471.

[48] H. Zhang et al., “LeFusion: Controllable pathology synthesis via lesionfocused diffusion models,” in Proc. ICLR, 2025.

[49] B. C. Russell, A. Torralba, K. P. Murphy, and W. T. Freeman, “LabelMe: A database and web-based tool for image annotation,” IJCV, vol. 77, no. 1, pp. 157–173, 2008.

[50] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. ICLR, 2019.

[51] S. Shit et al., “clDice-a novel topology-preserving loss function for tubular structure segmentation,” in Proc. CVPR, 2021, pp. 16 560– 16 569.

[52] D. Geman, S. Geman, N. Hallonquist, and L. Younes, “Visual Turing test for computer vision systems,” PNAS, vol. 112, no. 12, pp. 3618–3623, 2015.

[53] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, “UNet++: Redesigning skip connections to exploit multiscale features in image segmentation,” IEEE TMI, vol. 39, no. 6, pp. 1856–1867, 2020.

[54] J. Schlemper, O. Oktay, M. Schaap, M. Heinrich, B. Kainz, B. Glocker, and D. Rueckert, “Attention gated networks: Learning to leverage salient regions in medical images,” MedIA, vol. 53, pp. 197–207, 2019.

[55] H. Wang, P. Cao, J. Wang, and O. R. Zaiane, “UCTransNet: Rethinking the skip connections in U-Net from a channel-wise perspective with transformer,” in Proc. AAAI, vol. 36, no. 3, 2022, pp. 2441–2449.

[56] J. Ruan, J. Li, and S. Xiang, “VM-UNet: Vision Mamba UNet for medical image segmentation,” ACM TOMM, 2025.

[57] M. M. Rahman, S. K. Jung, and T. Hammond, “MambaLiteUNet: Crossgated adaptive feature fusion for robust skin lesion segmentation,” in Proc. CVPR, 2026, p. 8556–8565.

[58] R. Ferrod, C. F. Dantas, L. Di Caro, and D. Ienco, “Revisiting crossmodal knowledge distillation: A disentanglement approach for RGBD semantic segmentation,” in Proc. ECML-PKDD, 2025, pp. 492–508.

[59] H.-L. Zhao et al., “Design and performance evaluation of a novel vascular robotic system for complex percutaneous coronary interventions,” in Proc. EMBC, 2021, pp. 4679–4682.

[60] H. Li, X.-H. Zhou, X.-L. Xie, S.-Q. Liu, Z.-Q. Feng, and Z.-G. Hou, “CASOG: Conservative actor–critic with smooth gradient for skill learning in robot-assisted intervention,” IEEE TIE, vol. 71, no. 7, pp. 7722–7731, 2024.