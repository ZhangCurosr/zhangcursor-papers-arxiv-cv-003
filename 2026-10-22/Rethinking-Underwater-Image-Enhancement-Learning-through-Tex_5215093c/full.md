# Rethinking Underwater Image Enhancement: Learning through Text-Guided Degradation Curriculum

Hongri Tian<sup>1†</sup>, Wenxue Li<sup>1†</sup>, Tian Ye<sup>1</sup>, Sixiang Chen<sup>1</sup>, Lujia Wang<sup>1</sup>, Xianying Wang<sup>2</sup>, Xinhu Zheng<sup>1</sup>, Gang Hua<sup>3</sup>, Lei Zhu<sup>1,4\*</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China. <sup>2</sup>China Geological Survey, Beijing, China.

<sup>3</sup>Director of Applied Science, Amazon.com, Inc., U.S..

<sup>4</sup>The Hong Kong University of Science and Technology, Hong Kong SAR, China.

\*Corresponding author(s). E-mail(s): leizhu@hkust-gz.edu.cn; <sup>†</sup>Hongri Tian and Wenxue Li are co-first authors.

## Abstract

Existing paired clean–degraded datasets have advanced learning-based underwater image enhancement, but they remain either limited in scale or dificult to acquire, and still fail to adequately cover the diverse degradations encountered in real-world underwater environments. Such scarce and insuficient data ultimately restrict the generalization of underwater image enhancement models, highlighting the need for realistic and controllable degradation modeling to better capture real underwater conditions and improve enhancement robustness. To this end, we propose a unified degradation-aware training framework for the underwater image enhancement task. A Text-Conditioned Degradation Augmentation Pipeline uses a text-guided difusion model to synthesize diverse and semantically consistent degraded samples, providing paired data to support the subsequent curriculum learning strategy. The proposed Progressive Degradation Curriculum organizes training across multiple attributes and difficulty levels to promote balanced learning, and a Transmission-Guided One-Step Difusion Model leverages this progressive process to better address the wide range of degradation severity and restoration ambiguity in underwater images. Extensive experiments on multiple underwater benchmarks demonstrate that our proposed method achieves superior performance. Our code and models will be released upon the publication of this work.

Keywords: Underwater image enhancement, difusion models, curriculum learning, data augmentation

## 1 Introduction

Underwater Image Enhancement (UIE) aims to recover clear, color-consistent, and detailpreserving images from degraded underwater observations. Unlike uniform degradations in common image restoration tasks, underwater degradations are highly heterogeneous and scenedependent, where they vary with water type, depth, illumination, and particle density, and often exhibit complex spatial coupling among these factors. Consequently, the underlying degradation distribution is extremely dificult to model or approximate with a fixed prior, making UIE an intrinsically ill-posed and data-sensitive problem.

![](images/0e378fcbeb92ed2ee916772a3ae0185e75f83d8368745c7c3b0904b74e2cd240.jpg)

(b)Text-guided degradation for data augmentation.  
![](images/1869ab0016f7f5f5e5ed3009d6f3a13194ff629f36cde42ccf5a2548a57cfbbe.jpg)

![](images/4e8565c320bff44988b5071dc6ebf88cf02103a6680e4a51ef18d55d6a2f81e7.jpg)  
Fig. 1 Top: Illustration of three common routes for constructing paired underwater datasets. Medium: Proposed text-driven degradation difusion paradigm for controllable underwater degradation synthesis. Bottom: Quantitative results of our method on performance improvement and synthesis quality.

Over the past decade, a wide range of enhancement approaches have been explored, spanning from physics-based restoration guided by analytical priors [1–3] to deep learning-based methods trained on paired datasets [4–9]. While the former provides physical interpretability, it fails to generalize across diverse water types and illumination conditions. The latter achieves stronger visual quality through elaborately designed networks but relies heavily on limited paired data.

Acquiring paired underwater data is inherently challenging because capturing the same scene under both degraded and clean conditions is almost impossible in dynamic environments. As illustrated in Figure 1-Top, such pairs are commonly obtained through (i) controlled watertank capture [10], (ii) pseudo-reference generation by manually selecting outputs from multiple UIE methods [9, 11], or (iii) synthetic rendering with simplified underwater formation models [4, 12– 14]. The first two routes provide visually realistic supervision but are costly and dificult to scale, whereas the third enables scalable generation but often fails to reproduce the complex lighting and scattering efects of natural underwater scenes.

These limitations reveal that the main bottleneck of UIE lies not only in model architecture but also in the lack of realistic and controllable degradation modeling for generating diverse training variants. Moreover, underwater degradations vary drastically in strength and composition—ranging from light color cast to dense scattering and mixed turbidity. This motivates us to rethink UIE — not as direct restoration, but as learning from degradation, where controllable, text-guided degradation modeling provides structured, degradation-aware supervision for enhancement learning. In addition, we argue that if degradations can be explicitly modeled and controlled, they can serve as a meaningful curriculum for enhancement, where the model gradually adapts to increasingly challenging degradations across multiple attributes and severity levels, which encourages more generalized enhancement performance.

To this end, we revisit underwater enhancement from a new perspective-leveraging the generative capability of the difusion model to explicitly model and control underwater degradations (Figure 1-Medium). Difusion-based generative models [15] provide an expressive latent space in which degradations can be synthesized and conditioned on semantic cues such as turbidity, illumination, or particle density. This textconditioned formulation enables fine-grained control over degradation attributes and supports large-scale data augmentation aligned with real underwater characteristics. The generated degradations further provide dificulty-aware cues that guide enhancement models to learn progressively under well-structured degradation distributions, yielding more stable optimization and visually coherent restoration across diverse underwater conditions. Moreover, we develop a Transmission-Guided One-Step Difusion enhancer that injects a transmission prior into the latent space and performs a single deterministic reverse update, providing transmission-guided restoration with lower sampling cost than multi-step difusion methods. Together, these designs establish a cohesive paradigm that unifies controllable degradation modeling and physically guided enhancement, ofering a new perspective on the UIE task.

Overall, our contributions can be summarized as follows:

• We introduce a Text-Conditioned Degradation Augmentation (TCDA) pipeline that leverages a text-guided difusion model to synthesize controllable degraded variants of clean images, enabling fine-grained attribute/severity control for task-aligned data augmentation.

• We employ a progressive degradation curriculum learning that gradually increases training dificulty, stabilizing optimization and improving the model’s robustness.

• We design a Transmission-Guided One-step Diffusion model (TG-Dif) that embeds a transmission prior into the latent space, achieving high-quality restoration with greater inference eficiency than multi-step difusion methods.

• Extensive experiments on multiple underwater benchmarks demonstrate that our method achieves state-of-the-art performance.

## 2 Related Work

## 2.1 Underwater Image Enhancement

Traditional UIE methods are typically built upon the underwater image formation model that characterizes light scattering and absorption. Early works such as underwater dark channel prior [1], red channel prior [2], and blurriness-based estimation [3] estimate the transmission map and background light to remove haze and recover true colors. More recent prior-based methods further exploit illumination characteristics to handle nonuniform lighting and color degradation [16, 17]. However, their performance heavily depends on accurate estimation of water parameters and scene depth, limiting their robustness across varying underwater conditions [18]. Deep learning has revolutionized UIE by providing strong nonlinear modeling and robust feature representation capabilities. Early convolutional methods [4–8] employed encoder–decoder or multi-branch architectures to learn direct mappings from degraded to clean images, achieving improved contrast and color fidelity through illumination modeling and multi-level feature learning. Adversarial and generative frameworks [19, 20] further enhanced perceptual realism but often introduced instability and color over-saturation. Moreover, some works use contrastive learning [21], domain priors, and external knowledge to improve UIE generalization and controllability. PhaseFormer [22] introduces phase-based attention to preserve structural cues, while CLIP-UIE [23] utilizes semantic guidance from vision-language models. Semantic-Guided Difusion [24] leverages multimodal cues from LLaVA and Zip-CLIP to generate semantic heatmaps for difusion guidance.

Recent studies highlight the role of generative priors and degradation modeling in difusionbased image restoration [25]. Underwater-specific approaches, including WF-Dif [26] and SLURPP [27], exploit difusion models for underwater enhancement. Related difusion-based restoration methods further explore degradationmodel optimization [28], global brightness alignment [29], and blur-decoupled learning [30].

Despite these advances, the efectiveness of related UIE methods remains largely constrained by the availability and diversity of training data.

## 2.2 Underwater Image Synthesis

The scarcity of paired and diverse underwater data has motivated the research on underwater image synthesis. Early physics-based methods simulate light attenuation using simplified water models but fail to reproduce complex scattering [4, 14]. Alternatively, some datasets create pseudo-references by applying multiple UIE methods to a degraded image and having a user select the best outcome as the “clean” target [9]. While visually plausible, this approach is fundamentally limited as it provides no insight into the forward degradation process itself. It merely ofers a single, fixed enhancement target for each input, failing to model the diverse and controllable degradation attributes (e.g., water type, turbidity) that are crucial for robust generalization. GAN-based approaches [12, 13] improve realism but sufer from domain bias and unstable training. Recent difusion-driven frameworks, including Atlantis++ [31] and TIDE [32], enable text- or depth-conditioned underwater scene synthesis. However, these methods focus on generating underwater scenes rather than modeling controllable degradations. Our work uniquely fills this gap by introducing a controllable, textdriven degradation difusion model that synthesizes diverse and realistic underwater degradations, providing richer supervision and significantly enhancing the generalization of UIE models.

![](images/2a80edb39ac5ba20e799d1d34f0e6c3bdd9953fdde7a3a75f58d06829c455157.jpg)  
Fig. 2 Overview of the proposed Text-Conditioned Degradation Augmentation (TCDA) pipeline. It consists of four stages: (1) Degradation Captioning via MLLMs for generating fine-grained textual descriptions of underwater degradations, (2) Flow-Based Degradation Modeling for learning text-conditioned mappings between clean and degraded representations, (3) Prompt-Space Augmentation for enriching degradation prompts with varying dificulty levels, and (4) Text-Conditioned Degradation Synthesis for controllable degradation generation to support curriculum learning.

## 3 Method

Our methodology addresses the challenges of underwater image enhancement through a twostage framework. First, to overcome the bottleneck of limited diversity and fixed degradation patterns in existing datasets, we introduce a Text-Conditioned Degradation Augmentation (TCDA) pipeline. This pipeline synthesizes a vast and diverse corpus of degraded images with controllable degradation attributes specified by finegrained text prompts. Second, leveraging this augmented data, we employ a progressive degradation curriculum to train a Transmission-Guided One-Step Difusion (TG-Dif) model, designed for transmission-guided image restoration with greater inference eficiency than multi-step difusion methods.

## 3.1 Text-Conditioned Degradation Augmentation

As illustrated in Figure 2, our TCDA pipeline establishes a controllable mapping between clean images, their degraded counterparts, and textual degradation prompts.

## 3.1.1 Degradation Captioning via MLLMs

We first collect a large-scale dataset of 12,483 degraded–clean image pairs from four representative underwater datasets: UIEB [9], EUVP [12], LSUI [11], and 3D TURBID [10]. To ensure fair evaluation, only the training subsets of UIEB and LSUI are used.

To create rich, structured textual representations of the visual degradations, we employ a multimodal large language model (Gemini 2.5- Flash [33]) to generate fine-grained captions. These captions describe six core degradation attributes: turbidity level, color-cast type and severity, density of suspended particles, lighting conditions, degree of haze or backscatter, and extent of contrast reduction.

To obtain more consistent captions, we guide the MLLM to describe degradation cues only, while excluding scene content. This process yields a dataset of triplets $\{ I _ { \mathrm { d e g } } ^ { i } , I _ { \mathrm { c l e a n } } ^ { i } , T _ { \mathrm { d e g } } ^ { i } \} _ { i = 1 } ^ { N }$ , where the text $T _ { \mathrm { d e g } } ^ { i }$ provides fine-grained supervision that directly links semantic attributes to visual degradation patterns, forming the foundation for our text-conditioned difusion modeling. Details about the prompt design are provided in Supplementary Sec. 2.1.

## 3.1.2 Flow-Based Degradation Modeling

Building upon the constructed triplets, we develop a text-conditioned difusion framework to learn the mapping from clean to degraded representations. To model this directed transformation, we adopt a flow-based difusion framework based on FLUX.1 Kontext [15].

Unlike traditional denoising-based models, flow-matching directly learns the vector field from a source to a target distribution, ofering a more stable and eficient training objective for this task. Specifically, the clean image $I _ { \mathrm { c l e a n } }$ and degraded image $I _ { \mathrm { d e g } }$ are encoded into compact latent representations $x _ { 0 }$ and $x _ { 1 }$ via a pre-trained VAE encoder. Concurrently, the textual description $T _ { \mathrm { d e g } }$ is encoded by frozen dual text encoders (T5- XXL [34] and CLIP-L [35]) to obtain an embedding c. The model learns a conditional velocity field $v _ { \theta } ( z _ { t } , x _ { 0 } , t , c )$ that drives the transformation of $x _ { 0 }$ towards $x _ { 1 }$ under text guidance. We adopt a flow-matching objective as

$$
\mathcal { L } _ { \mathrm { d e g r a d e } } = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } , c } \big [ \| v _ { \theta } ( z _ { t } , x _ { 0 } , t , c ) - ( \varepsilon - x _ { 1 } ) \| _ { 2 } ^ { 2 } \big ]\tag{1}
$$

where $z _ { t }$ denotes the intermediate latent state at timestep t, and $\varepsilon \sim \mathcal { N } ( 0 , 1 )$ represents the stochastic noise. This formulation enables the model to learn a text-conditioned degradation vector field that reflects both the deterministic transformations described by the text and the stochastic variations inherent to real-world degradations. To enhance training eficiency and reduce computational overhead, we adopt parameter-eficient fine-tuning with LoRA [36].

## 3.1.3 Controllable Degradation Synthesis

To fully leverage our text-conditioned degradation model, we construct a structured prompt bank. Specifically, instead of relying on unconstrained free-form prompts, we organize the prompt space around six degradation attributes, including turbidity, color cast, particles, lighting, haze/backscatter, and contrast. For each attribute, we define phrases with diferent severity levels and group them into three dificulty tiers, i.e., easy, moderate, and hard, so that the resulting prompts are both semantically coherent and explicitly controllable. Each prompt is a tuple $\begin{array} { r c l } { T _ { \mathrm { d e g } } } & { = } & { \left( a _ { 1 } , \dots , a _ { K } , d \right) } \end{array}$ , detailing key attributes $\{ a _ { k } \}$ and a dificulty level $d \in$ {easy, moderate, hard}. During synthesis, given a clean image $I _ { \mathrm { c l e a n } }$ and any such prompt $T _ { \mathrm { d e g } } .$ our model generates a corresponding degraded image $I _ { \mathrm { d e g } }$ . This text-driven paradigm not only enables fine-grained semantic control over degradation types and intensities but also provides the multi-dificulty data crucial for our progressive enhancement training, as detailed next. Representative prompt-conditioned degradation results are shown in Figure 4. Detailed design is described in Supplementary Sec. 2.2.

## 3.2 Progressive Degradation Curriculum Learning

Directly training a restoration model on the full, diverse spectrum of degradations generated by TCDA can lead to instability and slow convergence. To address this, we introduce a progressive degradation curriculum that leverages the dificulty levels defined in our prompt bank. This curriculum gradually increases the degradation complexity during training, allowing the model to first master fundamental color and tone restoration before tackling more complex scattering and contrast loss.

Specifically, for each dificulty level d ∈ {easy, moderate, hard}, we formulate a sampling schedule that dynamically adjusts the selection probability $P _ { \tau } ( d )$ as a function of the normalized training progress $\tau \in [ 0 , 1 ]$

$$
P _ { \tau } ( d ) = \frac { \exp \left( \boldsymbol { \alpha } _ { p } \cdot \boldsymbol { s } _ { d } ( \tau ) \right) } { \sum _ { d ^ { \prime } } \exp \left( \boldsymbol { \alpha } _ { p } \cdot \boldsymbol { s } _ { d ^ { \prime } } ( \tau ) \right) } ,\tag{2}
$$

where $\alpha _ { p }$ is a temperature factor controlling the transition sharpness, which we set to 5.0 in our

![](images/e1efee9f98b8cb1d3a84fe2ca641b6297d165c5d98478ae0aef5c5dc7fb407e9.jpg)  
Fig. 3 Overview of the proposed UIE framework. The model learns progressive degradation-aware enhancement through curriculum sampling and transmission-guided one-step difusion, combining physical priors and text-conditioned cues for controllable restoration with greater inference eficiency than multi-step difusion methods.

Table 1 Quantitative comparison on UIEB, C60, and U45. Each method is evaluated with its original training data and with TCDA augmentation. Parentheses show the performance change. Best results are in bold.
<table><tr><td rowspan="2">Methods</td><td colspan="5">UIEB</td><td colspan="2">C60</td><td colspan="2">U45</td></tr><tr><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>UCIQE↑</td><td>UIQM↑</td><td>UCIQE↑</td><td>UIQM↑</td><td>UCIQE↑</td><td>UIQM↑</td></tr><tr><td>UIEC2-Net [6] + TCDA</td><td>0.880 0.880 (+0.000)</td><td>17.961 19.020 (+1.059) 0.229 (-0.023)</td><td>0.252</td><td>0.526 0.550 (+0.024)</td><td>3.766 3.695 (-0.071)</td><td>0.487 0.525 (+0.038)</td><td>2.149 2.146 (-0.003)</td><td>0.535 0.554 (+0.019)</td><td>4.231 4.156 (-0.075)</td></tr><tr><td>SCNet [37] + TCDA</td><td>0.907 0.909 (+0.002)</td><td>19.560 20.998 (+1.438)</td><td>0.195 0.172 (-0.023)</td><td>0.585 0.593 (+0.008)</td><td>3.869 3.925 (+0.056)</td><td>0.566 0.551 (-0.015)</td><td>2.498 2.452 (-0.046)</td><td>0.570 0.578 (+0.008)</td><td>4.155 4.163 (+0.008)</td></tr><tr><td>FiveA+ [38] + TCDA</td><td>0.903 0.907 (+0.004)</td><td>18.671 20.516 (+1.845)</td><td>0.260 0.183 (-0.077)</td><td>0.558 0.586 (+0.028)</td><td>3.811 3.845 (+0.034)</td><td>0.531 0.548 (+0.017)</td><td>2.229 2.340 (+0.111)</td><td>0.546 0.563 (+0.017)</td><td>4.186 4.080 (-0.106)</td></tr><tr><td>WF-Diff [26] + TCDA</td><td>0.901 0.914 (+0.013)</td><td>19.984 20.622 (+0.638)</td><td>0.206 0.181 (-0.025)</td><td>0.590 0.592 (+0.002)</td><td>4.075 3.902 (-0.173)</td><td>0.551 0.553 (+0.002)</td><td>2.404 2.384 (-0.020)</td><td>0.568 0.570 (+0.002)</td><td>4.140 4.167 (+0.027)</td></tr><tr><td>CE-VAE [39] + TCDA</td><td>0.818 0.857 (+0.039)</td><td>20.032 19.786 (-0.246)</td><td>0.224 0.211 (-0.013)</td><td>0.598 0.601 (+0.003) 4.110 (+0.012)</td><td>4.098</td><td>0.555 0.559 (+0.004) 2.786 (+0.030)</td><td>2.756</td><td>0.590 0.595 (+0.005)</td><td>4.322 4.280 (-0.042)</td></tr><tr><td>Phaseformer [22] + TCDA</td><td>0.917</td><td>21.643 22.045 (+0.402)</td><td>0.161 0.152 (-0.009)</td><td>0.594</td><td>3.938</td><td>0.543</td><td>2.431</td><td>0.588</td><td>4.223</td></tr><tr><td>TG-Diff (Ours) + TCDA</td><td>0.923 (+0.006) 0.917</td><td>21.723</td><td>0.135</td><td>0.607 (+0.013) 0.598</td><td>3.946 (+0.008) 4.040</td><td>0.553 (+0.010) 0.557</td><td>2.483 (+0.052) 2.572</td><td>0.593 (+0.005) 0.603</td><td>4.219 (-0.004) 4.326</td></tr></table>

experiments to ensure a smooth yet efective curriculum, and $s _ { d } ( \tau )$ is the preference score for difficulty d. To realize a smooth shift, we instantiate

the scores as:

$$
\begin{array} { c } { { s _ { \mathrm { e a s y } } ( \tau ) = \cos ^ { 2 } \left( \displaystyle \frac { \pi \tau } { 2 } \right) , } } \\ { { s _ { \mathrm { m o d e r a t e } } ( \tau ) = \sin ( \pi \tau ) , } } \\ { { s _ { \mathrm { h a r d } } ( \tau ) = \sin ^ { 2 } \left( \displaystyle \frac { \pi \tau } { 2 } \right) , } } \end{array}\tag{3}
$$

![](images/b6f5f1b538ed2d2570d56fb73089e9d004622b4b1ba39462f000dcec8851e2f8.jpg)  
Prompt: Degrade this image with high turbidity, dominant yellowish-green cast, particle storm, deep water low light, severe haze, severe contrast reduction.

![](images/037820947922c63ee3b2459c178e51af911d5d71d2436f1ebdbf31a6dbf60f5b.jpg)

![](images/c8096c9e4fc47fa82899d0f205a77937c181633cc28921632b8c83240b9eb7fe.jpg)  
Prompt: Degrade this image with heavy milk-like turbidity, dominant blue-green cast, silt plume, artificial uneven lighting, severe haze, substantial contrast reduction.  
Fig. 4 Examples of learned degradation generation results. Given a clean image and a textual prompt explicitly describing degradation attributes (e.g., turbidity, color cast, lighting, haze), our model synthesizes degraded counterparts that accurately reflect the specified visual conditions and severity levels.

This formulation enables a smooth and interpretable evolution of the training data distribution, allowing the model to gradually adapt to increasingly complex degradations. As shown in Figure 3, the continuous evolution of sampling probabilities promotes progressive robustness across diverse underwater conditions.

## 3.3 Transmission-Guided One-Step Difusion

To bridge the gap between physics-agnostic diffusion models and restoration with greater inference eficiency than multi-step difusion methods, we propose Transmission-Guided One-Step Difusion (TG-Dif). Our framework builds upon the single-step SD-Turbo architecture [40], which we adapt to the UIE task. Runtime and modelcomplexity comparisons are provided in Supplementary Sec. 3.1. We freeze the pre-trained weights and apply LoRA [36] with a rank of 32 to both the U-Net and the VAE encoder.

To provide the difusion model with a dense spatial prior on attenuation and scattering, we condition it on a transmission map T estimated via the UDCP method [1]. This strategy provides the model with a physics-inspired cue about spatially varying water properties. To achieve this, the map is downsampled to the latent resolution and concatenated with the image latent $z _ { \mathrm { d e g } } ,$ , forming an expanded 5-channel input $x =$ $[ z _ { \mathrm { d e g } } , \mathcal { T } _ { \mathrm { l a t e n t } } ]$ where new channel weights are zeroinitialized.

Furthermore, to preserve high-frequency details during the single-step process, we augment the VAE with multi-scale skip connections. In this process, an intermediate skip feature $S _ { i }$ is first generated from the encoder feature $E _ { i }$ via a $1 \times 1$ convolution. This feature is then added to the corresponding upsampled decoder feature $U _ { i }$ as:

$$
S _ { i } = \mathrm { C o n v _ { 1 } } _ { \times 1 } ( E _ { i } ) , \quad U _ { i } \gets \mathrm { I n t e r p } ( U _ { i } ) + S _ { i } \ .\tag{4}
$$

## 4 Experiment

## 4.1 Experimental Setup

Implementation Details. In the TCDA pipeline, we utilize two NVIDIA A800 GPUs (80 GB) for training the degradation difusion model. We adopt a LoRA fine-tuning strategy with a rank of 8. All the images are resized to $1 0 2 4 \times 1 0 2 4$ in the training process. The model is optimized using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a per-GPU batch size of 1. We train the model for two epochs. During inference, we employ 28 denoising steps for text-conditioned degradation synthesis. For training the TG-Dif, we adopt the Stable Difusion-Turbo [40] as our base model and fine-tune it with LoRA of rank 32. During the training process, the samples are randomly cropped into 256 × 256 patches from the original images. We train the model for 50,000 iterations using the AdamW optimizer $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 )$ with a constant learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 12 per GPU. We optimize TG-Dif with the composite objective $\mathcal { L } _ { \mathrm { t o t a l } } ~ = ~ \mathcal { L } _ { \mathrm { L 1 } } + \alpha \mathcal { L } _ { \mathrm { L P I P S } }$ which combines the L1 reconstruction loss $( \mathcal { L } _ { \mathrm { L 1 } } )$ and the LPIPS perceptual loss $\left( \mathcal { L } _ { \mathrm { L P I P S } } \right)$ , with $\alpha ~ = ~ 3 . 3$ in all experiments.

Datasets and Evaluation Metrics. All images in the UIEB dataset [9] are first divided into 790 images for training and 100 images for testing.

![](images/78cbf3245efdb8f819e8428c333fcb5afe14063146b0e087b46e93c3e493b160.jpg)  
Fig. 5 Visual comparisons with state-of-the-art UIE methods on the UIEB dataset. Our method consistently outperforms state-of-the-art UIE methods for underwater images with varying degrees of degradation.

Based on 790 training images, we further utilize our text-conditioned degradation augmentation (TCDA) model to produce 10,270 augmented images. These augmented images are categorized into three dificulty levels (easy, moderate, and hard) with a ratio of 5:5:3. To assess the benefit of TCDA, we train all underwater image enhancement models with and without 10,270 augmented images, and then evaluate them on the same UIEB test set. For paired evaluations, we report PSNR [41], SSIM [42], and LPIPS [43] to measure reconstruction fidelity. For unpaired evaluations, we employ two no-reference underwater quality metrics: UCIQE [44] and UIQM [45]. In addition, we assess performance on the reference-free UIEB C60 subset [9] and U45 dataset [46], using UCIQE and UIQM to measure enhancement quality under unpaired conditions.

## 4.2 Comparison with State-of-the-Arts

We compare our approach with several state-ofthe-art UIE methods, including UIEC<sup>2</sup>-Net [6], SCNet [37], FiveA+ [38], WF-Dif [26], CE-VAE [39], and Phaseformer [22].

Quantitative Evaluation. The quantitative comparisons in Table 1 show that our method achieves the strongest overall performance across three underwater benchmarks. On the paired UIEB dataset, TG-Dif further achieves the best SSIM (0.927), PSNR (22.341), LPIPS (0.129), and UCIQE (0.613) after applying TCDA. On the reference-free C60 and U45 benchmarks, our method also obtains the highest UCIQE and UIQM scores in most cases, indicating improved performance on these reference-free metrics. TCDA improves most reported metrics across the compared methods. A discussion of occasional UIQM decreases is provided in Supplementary Sec. 3.2.

Qualitative Evaluation. Figure 5 presents visual comparisons on representative UIEB examples with diferent degradation levels. Existing methods often sufer from distinct artifacts: UIEC<sup>2</sup>-Net tends to leave residual haze and muted colors, SCNet and FiveA+ may introduce over-enhancement or unnatural color shifts, WF-Dif and CE-VAE sometimes fail to fully recover fine structures, and Phaseformer, although stronger, still shows less stable restoration under severe degradations. In contrast, our TG-Dif produces clearer visibility, more natural color balance, and better local detail recovery, as highlighted in the enlarged regions. After incorporating TCDA, “Ours+TCDA” further improves restoration fidelity, especially in heavily degraded scenes, where it removes haze more thoroughly and recovers clearer textures and structures. These results verify that the proposed data-centric curriculum not only improves objective performance but also leads to more visually reliable enhancement results.

Table 2 Ablation study on the key components of TG-Dif.
<table><tr><td>Backbone (Single-step)</td><td>Encoder-lora (Trainable)</td><td>Skip Conn.</td><td>UDCP</td><td>SSIM ↑ PSNR ↑ LPIPS↓</td><td></td><td></td></tr><tr><td>√</td><td rowspan="5">√</td><td rowspan="5"></td><td rowspan="5"></td><td>0.697</td><td>20.374</td><td>0.226</td></tr><tr><td>√</td><td>0.701</td><td>20.516</td><td>0.228</td></tr><tr><td>√</td><td>0.879</td><td>21.266</td><td>0.153</td></tr><tr><td>√</td><td>0.700</td><td>20.436</td><td>0.233</td></tr><tr><td>√</td><td>0.912</td><td>21.767</td><td>0.139</td></tr><tr><td>√</td><td>√ √</td><td>√ √</td><td>√</td><td>0.917</td><td>21.723</td><td>0.135</td></tr></table>

![](images/8acea5491989435883aaf4245732d50048aa71a106aacb14aa13f350e7fc7f0a.jpg)  
Fig. 6 Qualitative ablation study of the proposed TG-Dif components on the UIEB dataset. The full model restores clearer structure, more natural illumination, and balanced color rendition.

Table 3 Quantitative comparison on LSUI [11]. Best results are in bold.
<table><tr><td>Methods</td><td>SSIM↑</td><td>PSNR↑ LPIPS↓</td><td>UCIQE↑</td></tr><tr><td>UIEC2-Net [6]</td><td>0.857</td><td>20.570 0.255</td><td>0.552</td></tr><tr><td>SCNet [37]</td><td>0.872 20.737</td><td>0.215</td><td>0.584</td></tr><tr><td>FiveA+ [38]</td><td>0.876 21.655</td><td>0.241</td><td>0.570</td></tr><tr><td>WF-Diff [26]</td><td>0.895</td><td>23.346 0.165</td><td>0.590</td></tr><tr><td>CE-VAE [39]</td><td>0.851</td><td>25.535 0.190</td><td>0.578</td></tr><tr><td>Phaseformer [22]</td><td>0.891</td><td>25.483 0.135</td><td>0.579</td></tr><tr><td>TG-Diff (Ours)</td><td>0.904</td><td>25.734</td><td>0.130 0.592</td></tr></table>

![](images/ea959895859d8bff27d52090aa56413c1dfd516b17850a9a2f9566c6bd58c359.jpg)  
Fig. 7 Visualization of degradation generation across diferent dificulty levels. Conditioned on text prompts describing easy, moderate, and hard degradations, the model generates visually consistent degradation patterns with increasing severity.

## 4.3 Ablation Studies

Efectiveness of Transmission-Guided One-Step Difusion. We investigate each TG-Dif component through the quantitative and qualitative results in Table 2 and Figure 6. Starting from the single-step backbone, making the encoder trainable improves feature adaptability, while adding skip connections further enhances structural fidelity. Incorporating the UDCP prior improves SSIM and LPIPS while maintaining comparable PSNR, indicating complementary benefits from the transmission cue.

To further validate the efectiveness of our TG-Dif, we conduct additional comparison experiments on the LSUI dataset [11], which contains 3,879 training images and 400 testing images. As shown in Table 3, our TG-Dif consistently outperforms representative UIE networks across both reference-based and non-reference metrics, demonstrating the superiority of our network design and its strong adaptability to diverse underwater conditions. More visual comparisons are provided in Supplementary Sec. 4.1.

Efectiveness of Progressive Degradation Curriculum. We ablate the impact of the proposed Progressive Degradation Curriculum by comparing it with training on a single dificulty level. As shown in Table 4, integrating curriculum learning yields the best overall performance by enabling progressive adaptation to degradations of varying dificulty.

We further investigate the efect of the augmented-to-original data ratio $( R _ { \mathrm { A u g } } )$ on model performance. As shown in Table 5, enlarging the ratio from 1× to 3× and 5× generally improves restoration fidelity and perceptual quality.

Analysis of Controllable Degradation Generation. As shown in Figure 7, our difusionbased degradation model provides fine-grained control over degradation severity. Conditioned on prompts with increasing dificulty, it generates consistent degradation transitions from mild color distortion to severe scattering and contrast loss.

Table 4 Ablation of degradation-dificulty sampling. Curriculum learning yields the best balanced performance.
<table><tr><td>Methods</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>UCIQE↑</td></tr><tr><td>TG-Diff + Easy only</td><td>0.901</td><td>21.474</td><td>0.149</td><td>0.603</td></tr><tr><td>TG-Diff + Moderate only</td><td>0.905</td><td>21.392</td><td>0.147</td><td>0.608</td></tr><tr><td>TG-Diff + Hard only</td><td>0.907</td><td>21.404</td><td> $_ { 0 . 1 5 0 }$ </td><td>0.603</td></tr><tr><td>TG-Diff + Random Sample</td><td>0.925</td><td>22.248</td><td> $_ { 0 . 1 2 9 }$ </td><td>0.608</td></tr><tr><td>TG-Diff + Curriculum</td><td>0.927</td><td>22.341</td><td>0.129</td><td>0.613</td></tr></table>

Table 5 Ablation on the augmented-to-original data ratio $\left( R _ { \mathrm { A u g } } \right)$ . Increasing the ratio of synthetic degradations improves overall enhancement quality.
<table><tr><td>Methods</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>UCIQE↑</td></tr><tr><td> $R _ { \mathrm { A u g } } = 1$  (only moderate)</td><td>0.905</td><td>21.392</td><td>0.147</td><td>0.608</td></tr><tr><td> $R _ { \mathrm { A u g } } = 3$ </td><td>0.919</td><td>22.126</td><td>0.134</td><td>0.611</td></tr><tr><td> $R _ { \mathrm { A u g } } = 5$ </td><td>0.924</td><td>22.267</td><td>0.130</td><td>0.602</td></tr><tr><td> $\ R _ { \mathrm { A u g } } = 1 3$ </td><td>0.927</td><td>22.341</td><td>0.129</td><td>0.613</td></tr></table>

Table 6 Comparison between UWCNN-based augmentation and TCDA (moderate only).
<table><tr><td>Method</td><td colspan="3">SSIM↑ PSNR↑ LPIPS↓ UCIQE</td></tr><tr><td>w/ UWCNN</td><td>0.855 19.491</td><td>0.185</td><td>0.580</td></tr><tr><td>w/ TCDA (Ours)</td><td>0.905 21.392</td><td>0.147</td><td>0.608</td></tr></table>

To evaluate the distributional similarity of our synthesized degradations to real underwater images, we compare our text-driven degradation generation model with representative underwater image synthesis methods, including UWGAN [20], UWCNN [4], and UWNR [14]. We compute the Fr´echet Inception Distance (FID) between generated degraded images and real underwater samples from the UIEB test set (100 images) and the LSUI test set (400 images). As shown in Figure 1, our model achieves a lower FID score than the compared synthesis methods, suggesting closer alignment with real underwater images in the Inception feature space.

Validation of Degradation Dificulty Levels. We evaluate variations in the severity of our synthesized degradations using two non-reference underwater metrics: UCIQE, which reflects color fidelity and contrast, and UIQM, which measures perceptual sharpness and colorfulness. As shown in Figure 8, both scores decline consistently from easy to hard levels, indicating that the generated degradations exhibit controllable severity.

![](images/72639f9e879bfdb9fff1c0bf7d39a71dce0b330d26359698fed0c8932bcc3e1b.jpg)

![](images/898c8d235ef4571b67d3c6e6fdd7b4a1b526438ddbcd8e4877adc23482124d3e.jpg)  
Fig. 8 Trends of non-reference underwater image quality metrics under diferent degradation dificulty level settings. Both UCIQE and UIQM consistently decrease as the degradation dificulty increases from easy to hard.

Efectiveness of the Pipeline. As shown in Table 6, beyond FID, the TG-Dif model trained on our synthesized data achieves stronger enhancement performance than that trained on degradations generated by the representative baseline UWCNN.

## 5 Conclusion

In this work, we propose an explicit, controllable degradation modeling framework designed to alleviate the scale and diversity limitations of paired UIE data. We introduce TCDA to synthesize diverse underwater degradations with fine-grained textual control, a progressive degradation curriculum to organize these samples across attributes and dificulty levels, and TG-Dif to incorporate a transmission prior into one-step restoration. Together, these components convert controllable degradation modeling into structured supervision for robust UIE. Experiments on multiple benchmarks demonstrate improvements across most restoration and reference-free metrics, while the ablation studies examine the contributions of the curriculum, data augmentation, and transmission guidance. Future work will extend the framework to temporally consistent underwater video enhancement and broader real-world degradation conditions.

Data Availability Statement. In this work, we generate augmented training data for underwater image enhancement. The data is available in: https://github.com/ANTIT-Pixel/ Rethinking-Underwater-Image-Enhancement.

Acknowledgment. This work is supported by Guangdong Basic and Applied Basic Research Foundation (2025A1515140021), the Nansha Key Area Science and Technology Project (No. 2024ZD006), and the Program of Guangdong Education Department.

## References

[1] Drews, P., Nascimento, E., Moraes, F., Botelho, S., Campos, M.: Transmission Estimation in Underwater Single Images. In: Proceedings of the IEEE International Conference on Computer Vision Workshops, pp. 825–830 (2013)

[2] Galdran, A., Pardo, D., Pic´on, A., Alvarez-Gila, A.: Automatic Red-Channel Underwater Image Restoration. Journal of Visual Communication and Image Representation 26, 132–145 (2015)

[3] Chiang, J.Y., Chen, Y.-C.: Underwater Image Enhancement by Wavelength Compensation and Dehazing. IEEE Transactions on Image Processing 21(4), 1756–1769 (2011)

[4] Li, C., Anwar, S., Porikli, F.: Underwater Scene Prior Inspired Deep Underwater Image and Video Enhancement. Pattern Recognition 98, 107038 (2020)

[5] Li, C., Anwar, S., Hou, J., Cong, R., Guo, C., Ren, W.: Underwater Image Enhancement via Medium Transmission-Guided Multi-Color Space Embedding. IEEE Transactions on Image Processing 30, 4985–5000 (2021)

[6] Wang, Y., Guo, J., Gao, H., Yue, H.: UIEC<sup>2</sup>-Net: CNN-Based Underwater Image Enhancement Using Two Color Space. Signal Processing: Image Communication 96, 116250 (2021)

[7] Zhang, W., Zhuang, P., Sun, H.-H., Li, G., Kwong, S., Li, C.: Underwater Image Enhancement via Minimal Color Loss and Locally Adaptive Contrast Enhancement.

IEEE Transactions on Image Processing 31, 3997–4010 (2022)

[8] Wei, Y., Zheng, Z., Jia, X.: UHD Underwater Image Enhancement via Frequency-Spatial Domain Aware Network. In: Asian Conference on Computer Vision, pp. 21–36 (2022). Springer

[9] Li, C., Guo, C., Ren, W., Cong, R., Hou, J., Kwong, S., Tao, D.: An Underwater Image Enhancement Benchmark Dataset and Beyond. IEEE Transactions on Image Processing 29, 4376–4389 (2019)

[10] Duarte, A., Codevilla, F., Gaya, J.D.O., Botelho, S.S.: A Dataset to Evaluate Underwater Image Restoration Methods. In: OCEANS 2016-Shanghai, pp. 1–6 (2016). IEEE

[11] Peng, L., Zhu, C., Bian, L.: U-Shape Transformer for Underwater Image Enhancement. IEEE Transactions on Image Processing 32, 3066–3079 (2023)

[12] Islam, M.J., Xia, Y., Sattar, J.: Fast Underwater Image Enhancement for Improved Visual Perception. IEEE Robotics and Automation Letters 5(2), 3227–3234 (2020)

[13] Li, J., Skinner, K.A., Eustice, R.M., Johnson-Roberson, M.: WaterGAN: Unsupervised Generative Network to Enable Real-Time Color Correction of Monocular Underwater Images. IEEE Robotics and Automation letters 3(1), 387–394 (2017)

[14] Ye, T., Chen, S., Liu, Y., Ye, Y., Chen, E., Li, Y.: Underwater Light Field Retention: Neural Rendering for Underwater Imaging. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 487–496 (2022). IEEE

[15] Labs, B.F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., Lacey, K., Levi, Y., Li, C., Lorenz, D., M¨uller, J., Podell, D., Rombach, R., Saini, H., Sauer, A., Smith, L.: FLUX.1 Kontext:

Flow Matching for In-Context Image Generation and Editing in Latent Space. CoRR abs/2506.15742 (2025)

[16] Hou, G., Li, N., Zhuang, P., Li, K., Sun, H., Li, C.: Non-Uniform Illumination Underwater Image Restoration via Illumination Channel Sparsity Prior. IEEE Transactions on Circuits and Systems for Video Technology 34(2), 799–814 (2023)

[17] Zhou, J., Liu, Q., Jiang, Q., Ren, W., Lam, K.-M., Zhang, W.: Underwater Camera: Improving Visual Perception via Adaptive Dark Pixel Prior and Color Correction. International Journal of Computer Vision 133(11), 8215–8233 (2025)

[18] Liu, R., Fan, X., Zhu, M., Hou, M., Luo, Z.: Real-World Underwater Enhancement: Challenges, Benchmarks, and Solutions Under Natural Light. IEEE Transactions on Circuits and Systems for Video Technology 30(12), 4861–4875 (2020)

[19] Fabbri, C., Islam, M.J., Sattar, J.: Enhancing Underwater Imagery Using Generative Adversarial Networks. In: 2018 IEEE International Conference on Robotics and Automation, pp. 7159–7165 (2018). IEEE

[20] Wang, N., Zhou, Y., Han, F., Zhu, H., Zheng, Y.: UWGAN: underwater GAN for real-world underwater color restoration and dehazing. CoRR abs/1912.10269 (2019)

[21] Zhou, J., Sun, J., Li, C., Jiang, Q., Zhou, M., Lam, K.-M., Zhang, W., Fu, X.: HCLR-Net: Hybrid Contrastive Learning Regularization with Locally Randomized Perturbation for Underwater Image Enhancement. International Journal of Computer Vision 132(10), 4132–4156 (2024)

[22] Khan, R., Negi, A., Kulkarni, A., Phutke, S.S., Vipparthi, S.K., Murala, S.: Phaseformer: Phase-Based Attention Mechanism for Underwater Image Restoration and Beyond. In: Proceedings of the Winter Conference on Applications of Computer Vision, pp. 9600–9611 (2025)

[23] Liu, S., Li, K., Ding, Y., Qi, Q.: Underwater Image Enhancement by Difusion Model with Customized CLIP-Classifier. Pattern Recognition, 112232 (2025)

[24] Zhou, J., Liu, J., Zhang, D., Zhang, X., Lin, Z.: Semantic-Guided Difusion for Water-Related Image Enhancement. Pattern Recognition, 112520 (2025)

[25] Li, X., Ren, Y., Jin, X., Lan, C., Wang, X., Zeng, W., Wang, X., Chen, Z.: Difusion Models for Image Restoration and Enhancement: A Comprehensive Survey. International Journal of Computer Vision 133(11), 8078–8108 (2025)

[26] Zhao, C., Cai, W., Dong, C., Hu, C.: Wavelet-Based Fourier Information Interaction with Frequency Difusion Adjustment for Underwater Image Restoration. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8281–8291 (2024)

[27] Wu, J., Wang, T., Siddique, M.A.B., Islam, M.J., Fermuller, C., Aloimonos, Y., Metzler, C.A.: Single-Step Latent Difusion for Underwater Image Restoration. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

[28] Fei, B., Lyu, Z., Pan, L., Zhang, J., Yang, W., Luo, T., Wang, J., Dai, B., He, Y., Ouyang, W.: Generative Difusion Prior for Unified Image and Video Restoration & Enhancement. International Journal of Computer Vision 134(6), 269 (2026)

[29] Guo, R., Zhong, H., Pagnucco, M., Song, Y.: P-patchdif: Progressive patch difusion models for low-light image enhancement. International Journal of Computer Vision 134(9), 404 (2026)

[30] Cheng, J., Chen, W.-T., Lu, X., Yang, M.-H.: Bd-dif: Generative difusion model for image deblurring on unknown domains with blurdecoupled learning. International Journal of Computer Vision 134(10), 446 (2026)

[31] Zhang, F., You, S., Li, Y., Fu, Y.: Atlantis++: Enabling Underwater Depth

Estimation with Stable Difusion and Beyond. International Journal of Computer Vision 134(6), 260 (2026)

[32] Lin, H., Liang, D., Qi, Z., Bai, X.: A Unified Image-Dense Annotation Generation Model for Underwater Scenes. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 961–970 (2025)

[33] Team, G.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR abs/2507.06261 (2025)

[34] Rafel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the Limits of Transfer Learning with a Unified Text-To-Text Transformer. Journal of Machine Learning Research 21(140), 1–67 (2020)

[35] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning Transferable Visual Models from Natural Language Supervision. In: International Conference on Machine Learning, pp. 8748–8763 (2021). PMLR

[36] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-Rank Adaptation of Large Language Models. In: The Tenth International Conference on Learning Representations (2022)

[37] Fu, Z., Lin, X., Wang, W., Huang, Y., Ding, X.: Underwater Image Enhancement via Learning Water Type Desensitized Representations. In: IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 2764–2768 (2022)

[38] Jiang, J., Ye, T., Chen, S., Chen, E., Liu, Y., Jun, S., Bai, J., Chai, W.: Five A+ network: You only need 9k parameters for underwater image enhancement. In: 34th British Machine Vision Conference 2023, pp. 149–151 (2023)

[39] Pucci, R., Martinel, N.: CE-VAE: Capsule Enhanced Variational AutoEncoder for Underwater Image Enhancement. In:

IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2113–2123 (2025)

[40] Sauer, A., Lorenz, D., Blattmann, A., Rombach, R.: Adversarial Difusion Distillation. In: European Conference on Computer Vision, pp. 87–103 (2024). Springer

[41] Huynh-Thu, Q., Ghanbari, M.: Scope of Validity of PSNR in Image/Video Quality Assessment. Electronics Letters 44(13), 1 (2008)

[42] Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing 13(4), 600–612 (2004)

[43] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The Unreasonable Efectiveness of Deep Features as a Perceptual Metric. In: 2018 IEEE Conference on Computer Vision and Pattern Recognition, pp. 586–595 (2018). IEEE

[44] Yang, M., Sowmya, A.: An Underwater Color Image Quality Evaluation Metric. IEEE Transactions on Image Processing 24(12), 6062–6071 (2015)

[45] Panetta, K., Gao, C., Agaian, S.: Human-Visual-System-Inspired Underwater Image Quality Measures. IEEE Journal of Oceanic Engineering 41(3), 541–551 (2015)

[46] Li, H., Li, J., Wang, W.: A Fusion Adversarial Underwater Image Enhancement Network with a Public Test Dataset. arXiv preprint arXiv:1906.06819 (2019)