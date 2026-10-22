# Unlocking Spatial Grounding in Flow-Matching Models via SNR Tuning

Martin Thißen

Department of Computer Science

martin.thissen@h-da.de

Darmstadt University of Applied

Elke Hergenröther

Sciences

elke.hergenroether@h-da.de

Darmstadt, Germany

## Abstract

Despite the success of foundation models in image synthesis, achieving fine-grained spatial control remains a significant challenge, typically requiring auxiliary structural adapters or complex inference-time steering. While modern vision-language (VLM) text encoders possess the latent capacity to represent spatial coordinates, generative models often fail to leverage this information. We demonstrate that the signal-to-noise ratio (SNR) distribution is a critical determinant of spatial grounding in flow-matching models. Using a novel synthetic benchmark of geometric primitives, we isolate structural mechanics from textural complexity inherent in natural imagery and introduce a rigorous color-aware layout metric to quantify spatial fidelity. We show that training on a centered SNR distribution results in poor layout adherence (25.9%). Instead, we identify highnoise regimes as a "critical learning phase" for spatial grounding. Shifting the training focus toward these stages increases layout adherence to 80.8%, effectively reaching the 83.6% reconstruction ceiling of the underlying variational autoencoder. We find that the necessity for this shift scales with resolution and aspect ratio variance, and that the conservative noise schedules employed by models like FLUX.1 are insufficient for precise layout adherence. Finally, we show that these findings translate to real-world imagery via the COCO dataset (1.0 vs. 57.3 Average Precision), where state-of-the-art layout adherence is unlocked through distribution tuning alone, without architectural modifications.

## 1 Introduction

The rapid evolution of diffusion [17, 46] and flow-matching [29, 34] models is outpacing the adoption of these models for specialized use cases. In particular, while new state-of-the-art image synthesis models [4, 23, 47, 50] are frequently released, achieving fine-grained spatial control over their outputs typically requires adopting one of two paradigms. The first involves training-based methods, which utilize auxiliary structural adapters such as Control-Net [58], GLIGEN [27], or CreatiLayout [57]. The second relies on training-free methods, which steer the generation process of frozen backbones during inference to follow a target layout. However, adapters are often tightly coupled to specific model architectures, making them costly to retrain for each new release, while training-free methods often struggle with complex layouts or specific architectural constraints. This creates a critical need to understand why foundation models struggle to generate images that follow layouts from text alone.

Standard pre-trained models typically fail to interpret 2D coordinates embedded in text prompts, often treating them as non-spatial semantic tokens rather than geometric constraints. In contrast, recent advancements in vision-language models (VLMs), such as the Qwen-VL series [1, 2], demonstrate that multimodal embeddings can successfully encode spatial relationships and discrete coordinates for tasks like object detection and grounding. This suggests that their multimodal embedding space already possesses the latent capacity to represent spatial information. As state-of-the-art text-to-image models increasingly adopt VLM-based text encoders, it remains an open question why the generative process fails to leverage this innate grounding capability. Therefore, we investigate whether, and under what conditions, text-to-image models can learn to follow explicit 2D layout specifications encoded solely in text prompts.

To this end, we examine the grounding capabilities of flow-matching models by fine-tuning them to interpret 2D coordinates directly from text prompts without introducing structural modifications. To isolate the mechanisms of spatial grounding, we first construct a controlled synthetic dataset of geometric primitives, enabling exact, pixel-level assessment of layout fidelity (Fig. 1(a)). Our analysis reveals that the signal-to-noise ratio (SNR) distribution is the critical determinant of spatial grounding. We find that training on a balanced SNR distribution actually hinders the model from accurately learning coordinate-based constraints. Instead, biasing the training toward high-noise regimes, where global structure is established, allows the model to internalize the relationship between text tokens and spatial coordinates. Notably, we find that this distribution shift is necessary even at lower resolutions (256<sup>2</sup>), and that the conservative shifts employed by recent models like FLUX.1 [24] are insufficient for precise layout adherence. Finally, we demonstrate that these principles successfully translate to real-world imagery via the COCO dataset [28] (Fig. 1(b)), showing that state-of-the-art layout adherence can be unlocked in standard flow-matching models through distribution tuning alone, without the need for specialized architectural extensions.

Taken together, our results demonstrate that high-fidelity spatial control is a latent capability in state-of-the-art flow-matching models that can be unlocked by strategically shifting the SNR distribution during training, without requiring any architectural modifications. We show that the high-noise regime serves as a "critical learning phase" for spatial grounding, forcing the model to internalize geometric constraints when global structure is not yet observable. Our main contributions are as follows:

• Identification of SNR as a Key Facilitator: We provide a systematic study showing that biasing the SNR distribution toward high-noise regimes is essential for teaching models to interpret 2D coordinates embedded in text.

• A Synthetic Benchmark for Spatial Precision: We introduce a benchmark consisting of geometric primitives that enables the exact, pixel-level quantification of layout adherence, isolating grounding mechanics from the textural complexities of real-world images.

• Transfer to Real-World Imagery: We provide empirical evidence that these insights translate to complex real-world imagery via the COCO dataset, achieving state-of-theart layout adherence in standard flow-matching models through distribution tuning alone, bypassing the need for specialized structural adapters.

![](images/ab712567de516efdfd391a93d9888da8335ce182485ff230317b91aedf9e345b.jpg)  
(a) Synthetic Images

![](images/ebaf3589564ca46c5a4dd5dad6c9ff8548e6bb69f57101a49c8483ea048633dd.jpg)  
(b) Real-World Images  
Figure 1: From Synthetic Experiments to Real-World Transfer. Layout adherence is first studied in a controlled synthetic setting, using color-filled regions as exact ground truth (left). This allows an isolated validation before extending the approach to real images (right).

## 2 Related Work

Image Synthesis. Generative modeling has transitioned from Generative Adversarial Networks (GANs) [13] to Diffusion [8, 17, 46] and Flow Matching [29, 34] frameworks. Architectures have similarly evolved from convolutional U-Nets (e.g., Stable Diffusion [42], SDXL [39]) toward scalable Diffusion Transformers (DiT) [37] and Multimodal Diffusion Transformers (MMDiT) [10, 22, 24]. Recent models [23, 50] replace traditional text encoders (e.g., CLIP [40] or T5 [41]) with Vision-Language Models (VLMs) [1, 2, 30] to leverage pre-aligned latent spaces. However, while VLMs excel at spatial grounding tasks like detection [1, 2], generative models using them still struggle to natively interpret 2D coordinates within text prompts, necessitating external spatial control.

Training Dynamics and SNR Schedules. The performance of flow-matching and diffusion models heavily depends on the timestep sampling distribution $p ( t )$ , which determines the Signal-to-Noise Ratio (SNR) levels encountered during training [3, 10, 21]. While logit-normal distributions are typically tuned to improve visual fidelity based on resolution [10, 18, 24] or dimensionality [59], their impact on spatial grounding is only beginning to be examined. Recent work by Wu et al. [51] and Zhang et al. [57] provides evidence that noise schedules influence layout adherence. However, their findings are coupled with specialized regional or siamese architectures. In contrast, our work demonstrates that the SNR distribution alone serves as the primary lever for internalizing spatial-semantic mappings in standard, structurally unmodified flow-matching models. We show that the acquisition of spatial grounding from text prompts alone is highly sensitive to $p ( t )$ , revealing that highnoise regimes represent the essential window for anchoring textual coordinates to global structural composition.

Layout-to-Image (L2I) Generation. Layout-to-image approaches address the limitations of native spatial control by augmenting pre-trained synthesis models with explicit geometric conditioning (e.g., bounding boxes or masks). This is increasingly viewed as a multi-stage pipeline, beginning with upstream planning where Large Language Models (LLMs) or autoregressive VLMs translate linguistic descriptions into structural layouts [12, 14, 48]. Other approaches bridge the gap between text and pixels by synthesizing intermediate representations, such as depth maps, to guide the final generative process [9, 62, 63]. Once a spatial structure is established, the literature diverges into training-based and training-free strategies. Training-based L2I methods adapt model architectures to interpret spatial signals via finetuning or dedicated adapters [27, 54, 60]. These works have evolved to handle multi-instance scenarios by using spatial attention disentanglement [49, 61] or hierarchical composition [6] to minimize attribute leakage. Beyond image quality, these models often serve as data engines to synthesize paired training data for perception tasks [11, 64]. Recently, MMDiT architectures have been enhanced with siamese attention to isolate layout signals [57] and hybrid control frameworks to achieve data-efficient grounding [51]. To avoid the high cost of retraining, training-free L2I methods intervene in the sampling process of frozen backbones. These strategies primarily manipulate internal cross-attention maps to ensure semantic isolation of subjects [7, 53, 56] or employ optimization-based guidance to steer the denoising trajectory toward specified coordinates [5, 26, 31, 38, 52]. Beyond attention manipulation, compositional strategies decentralize the process by merging localized per-object noise estimates or synchronizing global and instance-specific denoising branches [25, 45]. Finally, probabilistic approaches like Psi-Sampler [55] reinterpret layout adherence as an inferencetime reward alignment problem, using Sequential Monte Carlo sampling to filter trajectories.

While existing L2I methods typically rely on external modules to bypass spatial reasoning limitations, this study investigates a more fundamental question: can flow-matching models acquire spatial grounding from text prompts alone, without any structural modifications?

## 3 Method

## 3.1 Synthetic Benchmark for Text-Layout Image Generation

To facilitate training and provide a controlled environment for evaluation, we construct a synthetic dataset comprising 100,000 training pairs. For rigorous quantitative evaluation, we hold out a curated benchmark set of 1,000 text-image pairs. Images are rendered at resolution $W \times H$ and contain $n \sim \mathcal { U } \{ 1 , 1 0 \}$ rectangles. In our standard benchmark, we set $W = H = 2 5 6$ . Notably, our generation pipeline is agnostic to resolution and aspect ratio, enabling us to evaluate scalability across higher resolutions $( \mathbf { e . g . } , 1 0 2 4 ^ { 2 } )$ and varying aspect ratios, as detailed in Section 4.2. By utilizing simple geometric primitives, this approach mitigates the textural complexity and photometric noise present in real-world images. This simplification enables us to disentangle spatial control from semantic rendering, allowing for a systematic and quantitative evaluation of a model’s geometric layout compliance.

Layout Synthesis. We adopt rectangles as spatial primitives to mirror standard boundingbox formats and enable precise evaluation. To facilitate a systematic study of spatial grounding, we sample geometric properties from an extensive distribution that spans diverse scales and aspect ratios. Specifically, we define aspect ratio r and area A through log-uniform sampling: l $\begin{array} { r } { \mathbf { \boldsymbol { \mathrm { \Omega } } } _ { 1 } ( r ) \sim \mathcal { U } ( \ln \frac { 1 } { 2 0 } } \end{array}$ ,ln20) and $\ln ( A ) \sim \mathcal { U } ( \ln A _ { m i n } , \ln A _ { m a x } )$ , where $A _ { m i n } = 0 . 0 5 W \times 0 . 0 5 H$ and $A _ { m a x } = 0 . 9 W \times 0 . 9 H$ . Dimensions are then derived as $w = \sqrt { A \cdot r }$ and $h = { \sqrt { A / r } }$ . This diverse geometric sampling ensures the benchmark provides a robust evaluation of spatial control, testing the model’s ability to generalize across a wide range of object shapes.

Scene Composition. We sample centers $\left( c _ { x } , c _ { y } \right)$ uniformly and clip the resulting rectangles to the image frame $[ 0 , W ] \times [ 0 , H ]$ . As rectangles are rendered sequentially, we remove fully obscured rectangles from the metadata and prompt to ensure semantic consistency. This guarantees that all described objects are at least partially visible, preventing a mismatch between the textual description and visual ground truth.

Semantic Mapping. Each rectangle is assigned a unique color from a fixed palette of ten distinct values. To ensure maximum discriminability, which is important for our layout adherence metric introduced in Section 3.2, these colors are selected to be maximally distant from one another in the CIELAB color space. Each color is mapped to its corresponding linguistic label using the XKCD color naming convention [36] (e.g., fluro green).

![](images/c13100ab7cb2396d97c0cb260486948d3db957c0d321a88d27c17612e58c6ea8.jpg)  
(a) Text Prompt

![](images/de502e3a9b902564907191f6d6289d728f967e8b9e8e59f996bc174583191123.jpg)  
(b) Underlying Occlusion

![](images/7689b7bface08429eabc005147b12e5c25a979d4041e91b4a9415e75393fedcf.jpg)  
(c) Final Image  
Figure 2: Example ⟨text prompt, image⟩ pair from our synthetic dataset: (a) the input text prompt used to condition the scene; (b) a visualization of the occlusion, where rectangles are rendered with semi-transparency to reveal depth ordering; and (c) the final rendered image.

Layout Encoding. Following Bai et al. [1], we normalize coordinates to [0,1000]. A typical object is formatted as: $[ x _ { m i n } , y _ { m i n } , x _ { m a x } , y _ { m a x } , \mathrm { c o l o r } ]$ . Crucially, rectangles are listed in the prompt in their exact rendering order, establishing a spatial hierarchy where the textual sequence dictates occlusion priority. To succeed, the model must not only solve the challenging problem of spatial grounding but also master the compositional reasoning required to render overlapping geometries where objects are frequently partially obscured. Figure 2 visualizes this hierarchy across the prompt, occlusion layers, and final rendered image.

## 3.2 Layout Adherence Metric

To assess the spatial fidelity of generated images relative to specified text layouts, we introduce a Layout Adherence Metric based on color-coded segmentation masks. This metric quantifies the alignment between the ground-truth image $I _ { \mathrm { { g t } } }$ and synthesized output $I _ { \mathrm { g e n } }$ by first transforming both into the perceptually uniform CIELAB color space. We then identify the set of unique foreground colors C present in $I _ { \mathrm { { g t } } }$ , where $| { \mathcal { C } } | \leq 1 0$ corresponds to the number of layout rectangles. For each color $c \in { \mathcal { C } }$ , we derive binary masks for both the ground-truth and generated images. Specifically, binary masks $M _ { c } ( I ; \tau )$ are derived for both images by thresholding the perceptual color distance between each pixel p and the target color c:

$$
M _ { c } ( I ; \tau ) = \left\{ \mathbf { p } \in \Omega \vert \Vert I ( \mathbf { p } ) - c \Vert _ { 2 } \leq \tau \right\} ,\tag{1}
$$

where $\Omega$ denotes the image spatial domain and <sub>τ</sub> is a predefined color-matching threshold. This masking process accounts for minor color deviations in the generative process while isolating the spatial extent of each layout element. The adherence for a specific color c is then calculated using the Intersection over Union (IoU) between the respective masks:

$$
\mathrm { I o U } _ { c } ( I _ { \mathrm { g t } } , I _ { \mathrm { g e n } } ; \tau ) = \frac { \lvert M _ { c } ( I _ { \mathrm { g t } } ; \tau ) \cap M _ { c } ( I _ { \mathrm { g e n } } ; \tau ) \rvert } { \lvert M _ { c } ( I _ { \mathrm { g t } } ; \tau ) \cup M _ { c } ( I _ { \mathrm { g e n } } ; \tau ) \rvert } .\tag{2}
$$

<sub>fl</sub>Finally, we compute the mean IoU (mIoU ) across all unique colors in the image:

$$
\mathrm { m I o U } _ { \tau } = \frac { 1 } { | \mathcal { C } | } \sum _ { c \in \mathcal { C } } \mathrm { I o U } _ { c } ( I _ { \mathrm { g t } } , I _ { \mathrm { g e n } } ; \tau ) .\tag{3}
$$

![](images/908433c1d6d251206dad550cce62b62001d1e32b444c66fe8381e88378dc1f8f.jpg)  
(a) Ground Truth

![](images/2b21e168b0aa6766d27f2c6826aab818a8a722afcf0abca9113c0b6941f4169f.jpg)  
Darkened (b) mIoU<sub>10</sub> =0 mIoU =1

![](images/91cdf4f56990c57627993cf126aa6cfc73f56f157f6231ca60b85cc5e7064e74.jpg)  
Misaligned (c) mIoU<sub>2:25</sub> ≈ 0.52

![](images/e4a3112f565aa655a363b9d9f95604964c5be32941b748e5cd52c3759b13359a.jpg)  
(d) Missing Object mIoU = 0.75

![](images/67944efa8b5d0b311e1b5466596136d09466340678a5d8785aeba8dc29158ce7.jpg)  
(e) Reversed Order mIoU<sub>2:25</sub> ≈ 0.75  
Figure 3: Layout Adherence Metric Sensitivity. We illustrate mIoU across different failure modes. The metric effectively penalizes spatial misalignment (c), missing elements (d), and incorrect occlusion ordering (e). Note that spatial misalignment can result in a lower score than an omission, reflecting the metric’s focus on precise spatial grounding.

As illustrated in Fig. 3, this metric effectively penalizes both spatial misalignments and the omission of required layout elements. To ensure the metric is not overly sensitive to the choice of the threshold <sub>τ</sub> (Fig. 3(b)), we report the mean performance over a discrete set of thresholds $\mathcal { T } = \{ 2 , 3 , \hdots , 2 5 \}$ , defined as:

$$
\mathrm { m I o U } _ { 2 : 2 5 } = \frac { 1 } { | T | } \sum _ { \tau \in \mathcal { T } } \mathrm { m I o U } _ { \tau } .\tag{4}
$$

This allows us to observe the trade-off between strict color matching and spatial alignment. Furthermore, we establish an empirical upper bound for this metric by calculating the adherence of the VAE-reconstructed ground truth $( I _ { \mathrm { v a e } } = \mathcal { D } ( \mathcal { E } ( I _ { \mathrm { g t } } ) ) )$ . This serves as a calibration point to isolate generative errors from the autoencoder’s reconstruction limitations.

## 3.3 Implementation Details

Fine-Tuning. While our primary experiments utilize FLUX.2-4B-base, we also scale our approach to larger models including FLUX.2-9B-base [23], FLUX.1-dev (12B) [22, 24], and Qwen-Image (20B) [50]. To enable fine-tuning on a single 80GB A100 GPU, we utilize the Adafactor optimizer [44] (with constant scale, disabling relative step and parameter scaling) alongside a fused backward pass and gradient checkpointing. This configuration, combined with BFloat16 precision, reduces the peak memory footprint of the 20B Qwen-Image model from 160GB to 44GB at 256 × 256 resolution. We fine-tune for 100,000 steps using a unit batch size, as layer-wise fused updates are mathematically incompatible with gradient accumulation. This approach leverages recent findings that small-batch training maintains competitive convergence and generalization for large generative models [35]. To ensure training stability and preserve generation fidelity, we employ an EMA model (0.9995 decay, $\gamma = 1 0$ Karras schedule [19]) and a cosine learning rate schedule $( 1 \times 1 0 ^ { - 5 } \mathrm { t o } 5 \times 1 0 ^ { - 6 } )$ following a 5,000-step warmup. Finally, 10% of prompts are nullified for classifier-free guidance (CFG). Inference. We generate images using 25 steps with a CFG scale of 3.0. We utilize a shifted logit-normal timestep scheduler $( \mu = 1 . 0 )$ , which provided the best empirical trade-off between speed and fidelity.

![](images/b2f9187e7a5110626a2bd39270e7acc648e8ada9288aac496568c0c8917f0039.jpg)  
Figure 4: Shifted logit-normal distribution $\mathbf { p } _ { \mathbf { l n } }$ for location parameters $\mu .$ Increasing <sub>µ</sub> shifts density toward higher noise levels (larger values of t), demonstrating how the distribution of training or sampling steps can be adjusted to prioritize specific denoising stages.

## 4 Experimental Results

Building on our setup, we systematically evaluate how SNR distribution, image resolution and aspect ratio, object properties, and model architecture influence the layout adherence of flow-matching models.

## 4.1 Signal-to-Noise Ratio Distribution

The SNR distribution during fine-tuning determines which noise levels the model encounters most frequently, directly influencing what it learns to denoise. This makes the timestep sampling distribution $t \sim p ( t )$ a hyperparameter with significant impact on final model performance [3, 21]. In the context of rectified flow models [34], the noised latent $x _ { t }$ at timestep $t \in [ 0 , 1 ]$ is defined by the interpolation:

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t \varepsilon , \quad x _ { 0 } \sim p _ { \mathrm { d a t a } } , \ \varepsilon \sim \mathcal { N } ( 0 , 1 ) , \ t \sim p ( t ) .\tag{5}
$$

We therefore investigate the effect of $p ( t )$ on layout adherence and hypothesize that the model primarily learns the mapping between text coordinates and image locations during the high-noise stages, where it establishes global structure. Following recent advancements in flow-matching scaling [3, 10, 18, 24], we shift the logit-normal distribution:

$$
p _ { l n } ( t ; \mu , \sigma ) = \frac { 1 } { \sigma \sqrt { 2 \pi } } \cdot \frac { 1 } { t ( 1 - t ) } \exp \left( - \frac { ( \mathrm { l o g i t } ( t ) - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{6}
$$

where logi $\begin{array} { r } { \mathrm { \Delta } t ( t ) = \log \frac { t } { 1 - t } } \end{array}$ . Following Batifol et al. [3, 24], we fix $\sigma = 1$ to isolate the effect of the mean SNR (<sub>µ</sub>) shift on layout adherence. By adjusting $\mu _ { : }$ , we can systematically shift the fine-tuning focus toward the high-noise regime $( \mu > 0 )$ , as illustrated in Fig. 4. For our experiment, we investigate $\mu \in \{ 0 . 0 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 , 2 . 5 , 3 . 0 \}$

Results. We find that the location parameter $\mu$ strongly influences spatial grounding (Tab. 1).   
Fig. 6 shows that lower values $( \mu \leq 0 . 5 )$ result in perceivable structural misalignments.

![](images/0d57141d23f7c9c50126b7cbb5745dbacc3c133bc65dc3e04df60bb0aa990bc4.jpg)  
Figure 5: Layout adherence evaluation. (Left) $\mathrm { m I o U } _ { 1 0 }$ evolution over 100k steps for $n = 1 6$ samples. (Right) Sensitivity to color match threshold <sub>τ</sub> across 1,000 text-image pairs. Higher $\mu$ values (except $\mu = 3 )$ drive convergence toward the empirical VAE upper bound (red).

<table><tr><td rowspan="2">Method</td><td colspan="3"> $\mathrm { m I o U } _ { 2 : 2 5 }$  ←</td></tr><tr><td> $2 5 6 ^ { 2 }$ </td><td> $5 1 2 ^ { 2 }$ </td><td> $1 0 2 4 ^ { 2 }$ </td></tr><tr><td> $p _ { l n } ( t ; \mu = 3 . 0 , \cdot )$ </td><td>69.4</td><td>79.1</td><td>80.1</td></tr><tr><td> $p _ { l n } ( t ; \mu = 2 . 5 , \cdot )$ </td><td>78.1</td><td>79.8</td><td>82.7</td></tr><tr><td> $p _ { l n } ( t ; \mu = 2 . 0 , \cdot )$ </td><td>79.4</td><td>83.8</td><td>81.4</td></tr><tr><td> $p _ { l n } ( t ; \mu = 1 . 5 , \cdot )$ </td><td>80.1</td><td>83.6</td><td>一</td></tr><tr><td> $p _ { l n } ( t ; \mu = 1 . 0 , \cdot )$   $p _ { l n } ( t ; \mu = 0 . 5 , \cdot )$ </td><td>80.8</td><td>48.1</td><td>一</td></tr><tr><td> $p _ { l n } ( t ; \mu = 0 . 0 , \cdot )$ </td><td>49.0 25.9</td><td>一 一</td><td>一 一</td></tr><tr><td>VAE (Recon.)</td><td>83.6</td><td>87.6</td><td>88.3</td></tr></table>

Table 1: Effect of logit-normal shift $\mu$ on layout adherence across resolutions.

<table><tr><td rowspan="2">Method</td><td colspan="3">mIoU2:25 ↑</td></tr><tr><td>1</td><td>2</td><td>3</td></tr><tr><td> $p _ { l n } ( t ; \mu = 3 . 0 , \cdot )$ </td><td>69.4</td><td>73.5</td><td>69.3</td></tr><tr><td> $p _ { l n } ( t ; \mu = 2 . 5 , \cdot )$ </td><td>78.1</td><td>75.2</td><td>75.1</td></tr><tr><td> $p _ { l n } ( t ; \mu = 2 . 0 , \cdot )$ </td><td>79.4</td><td>75.2</td><td>73.8</td></tr><tr><td> $p _ { l n } ( t ; \mu = 1 . 5 , \cdot )$ </td><td>80.1</td><td>74.4</td><td>一</td></tr><tr><td> $p _ { l n } ( t ; \mu = 1 . 0 , \cdot )$ </td><td>80.8</td><td>67.8</td><td>一</td></tr><tr><td> $p _ { l n } ( t ; \mu = 0 . 5 , \cdot )$   $p _ { l n } ( t ; \mu = 0 . 0 , \cdot )$ </td><td>49.0 25.9</td><td>一</td><td>一</td></tr><tr><td></td><td></td><td>一</td><td>一</td></tr><tr><td> $\mathrm { V A E } \left( \mathrm { R e c o n . } \right)$ </td><td>83.6</td><td>83.3</td><td>83.2</td></tr></table>

Table 2: Effect of logit-normal shift <sub>µ</sub> on layout adherence across aspect ratios $( r _ { \operatorname* { m a x } } )$

While increasing <sub>µ</sub> to 3.0 yields visually stable layouts, we observe a quantitative performance degradation in $\mathrm { m I o U } _ { 2 : 2 5 }$ (Tab. 1, Fig. 5). Our analysis (see Appendix A) reveals that this drop stems not from structural failure, but from a loss of pixel-level color precision. At this extreme, the model’s color assignments frequently fall just outside color-matching threshold <sub>τ</sub>, despite correct spatial placement. This observation highlights a fundamental trade-off in the $\mu { \mathrm { - s h i f t i n g } }$ schedule: while a high <sub>µ</sub> prioritizes global structural grounding, it may limit the model’s capacity to resolve high-frequency components. In the context of natural images, this would manifest as a loss of fine-grained textural details. Importantly, moderate shifts $( \mu \in \left[ 1 . 0 , 2 . 5 \right] )$ nearly reach the VAE reconstruction upper bound of 83.6. These results suggest that vanilla flow-matching architectures possess the latent capacity to internalize coordinate-based constraints directly from text prompts. Within our controlled synthetic environment, the model achieves near-optimal layout adherence through distribution shifting alone, demonstrating the architectural feasibility of precise spatial grounding without auxiliary modules or inference-time interventions.

![](images/2447409824d9912f4cf4a224902b1c971210a4b0800bce9a89a768205e2f1466.jpg)  
Figure 6: Qualitative comparison of generated images across varying $\mu .$ . We compare the ground truth (GT) and VAE baseline against our model for $\mu \in [ 0 , 3 . 0 ]$ . Notably, spatial inaccuracies are easily perceivable for lower values $( \mu \in [ 0 , 0 . 5 ] )$ . As $\mu$ increases, the generated images show noticeably higher fidelity and better spatial alignment with the GT.

## 4.2 Resolution & Aspect Ratio

To evaluate if this architectural feasibility generalizes beyond fixed square layouts, we extend our approach along two axes: resolution and aspect ratio, which we investigate progressively $( 2 5 6 ^ { 2 }  5 1 2 ^ { 2 }  1 0 2 4 ^ { 2 }$ and $r _ { \operatorname* { m a x } } : 1  2  3 )$ . For resolution experiments, we re-render our image data at $5 1 2 ^ { 2 }$ and $1 0 2 4 ^ { 2 }$ while keeping all geometric and textual properties fixed, isolating the effect of resolution from layout complexity. For aspect ratio experiments, we derive resolution buckets $( w , h )$ from a fixed area $A = 2 5 6 ^ { 2 }$ such that $w \cdot h \approx A$ , where w and $h$ are multiples of 16. We include all pairs satisfying $\frac { \operatorname* { m a x } ( w , h ) } { \operatorname* { m i n } ( w , h ) } \leq r _ { \operatorname* { m a x } }$ , ensuring portraitlandscape symmetry. Following prior work [10, 18, 24] suggesting that higher resolutions require increased timestep shifting for optimal image quality, we use the optimal $\mu$ from the previous configuration as a lower bound for subsequent searches. For instance, having found $\mu = 1$ optimal for $2 5 6 ^ { 2 }$ , we limit our search to $\mu \in [ 1 , 3 ]$ for the $5 1 2 ^ { 2 }$ scale. By applying this strategy, we can substantially reduce the search space compared to a full <sub>µ</sub>-sweep for each setting. Crucially, as bounding box coordinates remain normalized to [0,1000], the textual prompts are identical across all configurations. This ensures that layout, object count, and color assignments remain constant, allowing for a controlled comparison of spatial grounding performance across varying scales and geometries.

Resolution. Table 1 summarizes the impact of the logit-normal shift $\mu$ on layout adherence as resolution increases. We observe a clear trend: the optimal $\mu$ value is positively correlated with the image resolution. For the $2 5 6 ^ { 2 }$ base resolution, a moderate shift of $\mu = 1 . 0$ yields the highest performance $( 8 0 . 8 \mathrm { m I o U } _ { 2 : 2 5 } )$ , nearly saturating the VAE reconstruction bound of 83.6. However, this configuration fails to generalize to higher resolutions. $\mathrm { A t } 5 1 2 ^ { 2 } , \mu = 1 . 0$ results in a significant drop in layout adherence $( 4 8 . 1 \mathrm { m I o U } _ { 2 : 2 5 } )$ . Instead, higher resolutions require a more aggressive shift toward earlier timesteps (higher noise levels) to maintain spatial control. Specifically, peak performance shifts to $\mu = 2 . 0$ for $5 1 2 ^ { 2 }$ and $\mu = 2 . 5$ for $1 0 2 4 ^ { 2 }$ . Notably, with the appropriately tuned $\mu _ { : }$ , our model maintains high layout adherence across all scales, reaching 82.7 $\mathrm { m I o U } _ { 2 : 2 5 }$ at $1 0 2 4 ^ { 2 }$ , approaching the VAE’s empirical ceiling. This suggests that as the resolution increases, a stronger prior on the early denoising phase is essential for the acquisition of spatial grounding from text prompts alone.

Aspect Ratio. We further evaluate the impact of $\mu$ across varying aspect ratios $( r _ { \operatorname* { m a x } } )$ , as shown in Tab. 2. Consistent with the resolution experiments, we find that the base configuration $( \mu = 1 . 0 )$ is insufficient for handling non-square geometries, with performance dropping from 80.8 to 67.8 $\mathrm { m I o U } _ { 2 : 2 5 }$ when moving from $r _ { \operatorname* { m a x } } = 1 \mathrm { t o } r _ { \operatorname* { m a x } } = 2$ . By increasing $\mu$ to the 2.0–2.5 range, the model recovers strong layout adherence even for extreme aspect ratios $( r _ { \operatorname* { m a x } } = 3 )$ . While there is a slight decrease in absolute $\mathrm { m I o U } _ { 2 : 2 5 }$ as the aspect ratio becomes more extreme (likely due to the increased difficulty of placing objects in narrow or highly elongated canvases), the shift in $\mu$ effectively mitigates the drop in spatial control seen with centered timestep distributions. These results demonstrate that the relationship between the timestep distribution $p ( t )$ and spatial grounding is not static, but rather highly sensitive to the geometric complexity of the output space. Collectively, these findings indicate that spatial grounding is heavily dependent on the noise levels encountered during training, necessitating a higher noise prior as image resolution and aspect ratios increase.

We further find that layout adherence is strongly dependent on object size (tiny vs. large) but largely density-independent, remaining stable for $n \in [ 3 , 1 0 ]$ objects per image. For tiny objects $( < 1 6 ^ { 2 } \mathrm { p x } )$ , median mI $\scriptstyle \mathrm { { o U } } _ { 2 : 2 5 }$ is approximately 0.41, versus 0.85-0.95 for large (64 − $1 2 8 ^ { 2 } { \mathfrak { p } } \mathbf { x } )$ and huge $( > 1 2 8 ^ { 2 } ~ \mathrm { p x } )$ objects. However, a similar degradation appears in the VAE reconstruction ceiling itself, suggesting this size-dependence is a property of the underlying latent space rather than a model-specific limitation. Full results are in Appendix B.

## 4.3 Architectural Generalization

To assess if the relationship between spatial grounding and SNR distribution shift (parameterized by $\mu )$ generalizes across different architectural configurations and text-conditioning strategies, we extend our evaluation to three larger-scale models: FLUX.2-9B-base [23] and Qwen-Image (20B) [50], which both utilize a pre-aligned Vision-Language Model (VLM) encoder, and FLUX.1-dev (12B) [22, 24], which employs a dual-encoder setup consisting of CLIP-L [40] and T5-XXL [41]. Since these models utilize different VAE architectures with varying reconstruction fidelities, we measure performance relative to their respective VAE upper bounds. We use $\mu = 1 . 0$ , the optimal shift identified in Section 4.2 for square images at $2 5 6 ^ { 2 }$ resolution, as a common starting point across architectures and compare it against a centered SNR distribution $( \mu = 0 . 0 )$ as a baseline.

Generalization Across Architectures. The results in Tab. 3 demonstrate that shifting the timestep sampling distribution is a generalizable mechanism to unlock spatial grounding in flow-matching models using text prompts only. This effect remains consistent regardless of the specific transformer backbone or text-conditioning strategy. While our primary experiments use the FLUX.2-4B-base architecture, the ability to recover layout adherence in 9B, 12B, and 20B models by tuning the SNR shift suggests a fundamental property of how flow-matching models interpret spatial instructions across scales.

Impact of Text Conditioning. We observe a clear correlation between the choice of text encoder and the model’s sensitivity to the shifting parameter $\mu$ . At $\mu = 1 . 0 { \mathrm { . } }$ , models using Vision-Language Model (VLM) encoders, such as Qwen-Image and FLUX.2, demonstrate high layout adherence $( \geq 8 0 . 8 \mathrm { m I o U } _ { 2 : 2 5 } )$ . In contrast, FLUX.1-dev (using T5 and CLIP) requires a significantly stronger shift $( \mu = 3 . 0 )$ to achieve comparable grounding performance. This suggests that VLM-based conditioning provides a more integrated spatial-semantic representation that is inherently more "accessible" to the denoising process than the separately trained embeddings provided by T5 and CLIP encoders.

<table><tr><td colspan="4"></td><td colspan="3"> $\mathrm { m I o U } _ { 2 : 2 5 }$ </td></tr><tr><td>Model</td><td>Params</td><td>Text Encoder</td><td>μ</td><td>Model ↑</td><td>VAE↑</td><td> $\Delta ( G a p ) \downarrow$ </td></tr><tr><td rowspan="2">FLUX.2-4B-base</td><td>4B</td><td>Qwen3-VL-4B</td><td>0.0</td><td>25.9</td><td>83.6</td><td>57.7</td></tr><tr><td></td><td></td><td>1.0</td><td>80.8</td><td>83.6</td><td>2.8</td></tr><tr><td rowspan="2">FLUX.2-9B-base</td><td>9B</td><td>Qwen3-VL-8B</td><td>0.0</td><td>32.6</td><td>83.6</td><td>51.0</td></tr><tr><td></td><td></td><td>1.0</td><td>81.5</td><td>83.6</td><td>2.1</td></tr><tr><td rowspan="2">FLUX.1-dev</td><td>12B</td><td>CLIP + T5</td><td>1.0</td><td>7.5</td><td>88.6</td><td>81.1</td></tr><tr><td></td><td></td><td>3.0</td><td>78.7</td><td>88.6</td><td>9.9</td></tr><tr><td rowspan="2">Qwen-Image</td><td>20B</td><td>Qwen2.5-VL-7B</td><td>0.0</td><td>48.1</td><td>92.5</td><td>44.4</td></tr><tr><td></td><td></td><td>1.0</td><td>80.8</td><td>92.5</td><td>11.7</td></tr></table>

Table 3: Architectural Generalization. Strong layout adherence can be achieved across scales, architectures, and text-conditioning strategies through SNR shifting alone. $\mu = 1 . 0$ is used as a common starting point across all models, compared against a centered baseline $( \mu = 0 . 0 )$ . FLUX.1-dev instead requires $\mu = 3 . 0$ to reach comparable spatial grounding. "Gap" (∆) denotes the difference between VAE reconstruction and generated performance.

Generative Gap and VAE Learnability. While all four models achieve high mIoU<sub>2:25</sub> scores, their "generative gaps" (∆) reveal a fundamental trade-off between VAE reconstruction fidelity and latent-space learnability. The FLUX.2 VAE is specifically optimized for representation learnability [3], which is reflected in the remarkably small gaps of 2.8 (FLUX.2- 4B-base) and 2.1 (FLUX.2-9B-base), indicating that the FLUX.2 models can almost fully exploit the empirical VAE reconstruction limit. However, this optimization for learnability entails a trade-off in reconstruction precision (e.g., color fidelity), leading to a lower VAE upper bound compared to Qwen-Image or FLUX.1-dev. Conversely, while the latter models possess higher reconstruction ceilings, their latent spaces appear significantly harder to navigate for precise spatial grounding, as evidenced by their larger generative gaps. This suggests that recent advancements in improving VAE learnability [3, 59] not only benefit image quality but also play a critical role in a model’s ability to adhere to complex layout instructions from text prompts.

## 5 Real-World Images

In the previous section, we established that flow-matching models can acquire strong layout adherence through text prompts alone by optimizing the SNR distribution. While our synthetic benchmark allowed for a controlled mechanistic study, we now investigate whether these findings transfer to the semantically rich domain of real-world images.

Dataset and Evaluation. Following standardized protocols [27, 54, 57], we evaluate on MS-COCO 2017 [28], excluding images with crowds, dense clusters, or objects smaller than 2% of the canvas. This yields 61,986 training and 2,565 validation images. We assess layout adherence using a frozen YOLO-v11-x [20] object detector to report Average Precision (AP, $\mathsf { A P } ^ { 5 0 } )$ and Average Recall (AR), while reporting the detector’s performance on ground-truth images as an empirical ceiling. Additionally, we assess global image quality and semantic alignment by computing CLIP score [15], FID [16], and Inception Score (IS) [43].

<table><tr><td rowspan="2">Method</td><td colspan="3">Layout Adherence</td><td colspan="3">Image Quality</td></tr><tr><td>AP↑</td><td> $\mathsf { A P } ^ { 5 0 } \uparrow$ </td><td>AR↑</td><td>CLIP↑</td><td>FID↓</td><td>IS↑</td></tr><tr><td> $p _ { l n } ( t ; \mu = 3 . 0 , \cdot )$ </td><td>57.3</td><td>81.8</td><td>65.2</td><td>28.01</td><td>32.38</td><td>30.32</td></tr><tr><td> $p _ { l n } ( t ; \mu = 2 . 5 , \cdot )$ </td><td>52.8</td><td>78.9</td><td>60.5</td><td>27.92</td><td>32.44</td><td>29.62</td></tr><tr><td> $p _ { l n } ( t ; \mu = 2 . 0 , \cdot )$ </td><td>32.0</td><td>61.6</td><td>40.9</td><td>27.85</td><td>32.45</td><td>29.38</td></tr><tr><td> $p _ { l n } ( t ; \mu = 1 . 5 , \cdot )$ </td><td>14.9</td><td>36.3</td><td>23.1</td><td>27.65</td><td>32.00</td><td>29.82</td></tr><tr><td> $p _ { l n } ( t ; \mu = 1 . 0 , \cdot )$ </td><td>1.4</td><td>4.8</td><td>5.4</td><td>27.04</td><td>33.11</td><td>29.27</td></tr><tr><td> $p _ { l n } ( t ; \mu = 0 . 5 , \cdot )$ </td><td>1.0</td><td>3.1</td><td>4.2</td><td>26.97</td><td>34.34</td><td>27.84</td></tr><tr><td> $p _ { l n } ( t ; \mu = 0 . 0 , \cdot )$ </td><td>1.0</td><td>2.8</td><td>3.7</td><td>26.93</td><td>37.75</td><td>27.62</td></tr><tr><td>COCO val (GT)</td><td>61.6</td><td>77.2</td><td>74.9</td><td>30.47</td><td>0.00</td><td>29.06</td></tr></table>

Table 4: Effect of shift $\mu$ on layout adherence and image quality of real-world images. To validate generalization beyond our synthetic benchmark, FLUX.2-9B-base is fine-tuned for 5 epochs on MS-COCO 2017 [28] at $5 1 2 ^ { 2 }$ resolution and aspect ratios up to 3:1 $( r _ { \operatorname* { m a x } } { = } 3 )$ .

<table><tr><td rowspan="2">Method</td><td colspan="3">Layout Adherence</td><td colspan="3">Image Quality</td></tr><tr><td>AP↑</td><td> $\mathrm { A P } ^ { 5 0 } \uparrow$ </td><td>AR↑</td><td>CLIP↑</td><td>FID↓</td><td>IS↑</td></tr><tr><td>GLIGEN []</td><td>42.1</td><td>66.8</td><td>37.2</td><td>30.25</td><td>30.68</td><td>26.76</td></tr><tr><td>MIGC []</td><td>39.1</td><td>66.0</td><td>36.8</td><td>28.60</td><td>31.43</td><td>26.14</td></tr><tr><td>InstanceDiff [9]</td><td>51.9</td><td>71.9</td><td>43.7</td><td>30.04</td><td>29.94</td><td>27.23</td></tr><tr><td>Ranni []</td><td>14.7</td><td>22.3</td><td>17.0</td><td>30.02</td><td>31.94</td><td>24.44</td></tr><tr><td>CreatiLayout []</td><td>47.8</td><td>69.8</td><td>42.1</td><td>33.32</td><td>29.27</td><td>30.26</td></tr><tr><td>Hybrid Layout []</td><td>55.2</td><td>79.1</td><td>51.8</td><td>31.20</td><td>一</td><td>26.72</td></tr><tr><td>FLUX.2-9B-base  $\mu { = } 3 . 0 , \mathrm { e p o c h s } { = } 5$ </td><td>57.3</td><td>81.8</td><td>65.2</td><td>28.01</td><td>32.38</td><td>30.32</td></tr><tr><td> $\mu { = } 3 . 0 , \mathrm { e p o c h s { = } } 1 0$ </td><td>61.8</td><td>84.7</td><td>68.8</td><td>28.03</td><td>31.43</td><td>30.06</td></tr><tr><td>COCO val (GT)</td><td>61.6</td><td>77.2</td><td>74.9</td><td>30.47</td><td>0.00</td><td>29.06</td></tr></table>

Table 5: Quantitative comparison against specialized layout-to-image models. Evaluated on MS-COCO 2017, our fine-tuned FLUX.2-9B-base achieves state-of-the-art layout adherence and competitive image quality, without requiring architectural modifications.

Implementation Details. We select FLUX.2-9B-base for our real-world experiments, as it achieved the strongest layout adherence among the architectures evaluated in Tab. 3 (81.5 $\mathrm { m I o U } _ { 2 : 2 5 }$ at $\mu = 1 . 0 )$ . We fine-tune this model for 5 epochs at $5 1 2 ^ { 2 }$ resolution, supporting aspect ratios up to 3:1 $( r _ { \operatorname* { m a x } } = 3 )$ . As done for our synthetic benchmark (Tab. 1, 2), we investigate $\mu \in \{ 0 . 0 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 , 2 . 5 , 3 . 0 \}$ . To preserve the base model’s generative priors amidst complex real-world textures, we use a conservative learning rate of $1 \times 1 0 ^ { - 6 }$ , which we found yields more stable convergence than the higher rates used for geometric primitives. For real-world images, we adopt the standard FLUX.2 sampling configuration (50 steps, 4.0 guidance). While more efficient settings sufficed in our synthetic study, these default parameters are better suited to resolving complex real-world textures. Bounding boxes are normalized to [0,1000], sequenced by appearance in the COCO annotations, and formatted as: $[ x _ { m i n } , y _ { m i n } , x _ { m a x } , y _ { m a x }$ ,category name].

![](images/b4a74271af6021ed8aa225bf0d3e0dfc555ff5789627e834e292330a81d3784e.jpg)

C apple C bird car clock elephant 口 person 口 sports ball 口 tv baseball bat book cell phone cow frisbee potted plant 口 suitcase 日 umbrella baseball glove bowl chair dog giraffe sheep tennis racket 口 zebra

Figure 7: Qualitative results on COCO val. Bounding boxes indicate ground-truth layout conditions. Our model faithfully generates objects at specified locations across diverse scenes, compositions, and densities (rows 1–3). Row 4 shows representative failure cases.

Results. Table 4 demonstrates that spatial grounding via SNR distribution shifting (<sub>µ</sub> = 3.0) robustly transfers from our controlled synthetic environment to complex real-world imagery. Despite using only text prompts and a lightweight 5-epoch fine-tuning on MS-COCO, our model achieves state-of-the-art layout adherence compared to specialized layout-to-image models (Table 5). Specifically, our Average Recall (AR) of 65.2 significantly outperforms established models like GLIGEN (37.2), InstanceDiff (43.7), or Hybrid Layout (51.8), indicating that our model is remarkably effective at populating the canvas with all requested objects. Furthermore, our AP<sup>50</sup> of 81.8 exceeds the YOLO-v11-x detector’s empirical ceiling on ground-truth COCO images (77.2). Since ground-truth represents a theoretical upper bound, we hypothesize this reflects known limitations in the COCO dataset itself (e.g., bounding box noise, motion blur, and greater occlusion and background clutter than in our generated images) that lower the detector’s measured ceiling on real-world images. Overall, this shows that our model’s spatial precision is effectively saturating the empirical ceiling. While FID (32.38) and CLIP scores (28.01) trail CreatiLayout [57], which utilizes a more complex multi-stream architecture, our model achieves the highest Inception Score (30.32) among all compared methods, including CreatiLayout (30.26) and the ground-truth COCO images themselves (29.06), indicating that the rich generative priors of the FLUX.2-9B-base model are preserved during fine-tuning. To assess whether extended training yields further gains, we fine-tune FLUX.2-9B-base at $\mu = 3 . 0$ for 10 epochs while evaluating after each epoch, which further improves layout adherence across all metrics (AP: 61.8, AP<sup>50</sup>: 84.7, AR: 68.8; Tab. 5). The complete epoch-ablation is reported in Appendix C. Qualitative results (Fig. 7) of the FLUX.2-9B-base model after 10 epochs of fine-tuning show faithful spatial generation across diverse scenes, challenging compositions, and densities (rows 1–3). Observed failure modes (row 4) include compositional artifacts (e.g., merged or duplicated anatomical features) and high-frequency imprecision (e.g., clock face details). Preliminary experiments applying layout-conditioned denoising only during the first 40–50% of steps improved fine-grained details with only a marginal AP drop (1.5–5.1 points). While some compositional artifacts persist, also occasionally seen in the pre-trained FLUX.2-9B-base model prior to any fine-tuning (Appendix D), unifying these dynamics into a single, potentially scaled, foundation model remains future work. Ultimately, these findings validate the insights from our synthetic benchmark: the lack of native spatial control in foundation flow-matching models is not an architectural limitation. By shifting the SNR distribution to prioritize global structural dependencies, a standard flow-matching transformer can parse spatial coordinates directly from the text stream and compete with specialized architectures.

## 6 Conclusion

In this work, we investigated the latent spatial grounding capabilities of flow-matching models, addressing the discrepancy between the rich spatial embeddings of modern visionlanguage encoders and the poor native layout adherence of the generative models that leverage them. Unlike the prevailing paradigm of introducing auxiliary structural adapters or complex inference-time optimizations, we show that precise spatial control can be "unlocked" by shifting the timestep sampling distribution. Our systematic evaluation using a novel synthetic benchmark of geometric primitives allowed us to isolate the mechanics of spatial grounding from textural complexities. We identified the high-noise regime as a "critical learning phase", where the model is forced to internalize geometric constraints to establish global structure. By biasing the fine-tuning process toward these regimes, we enabled models to interpret 2D coordinates directly from text prompts with strong spatial accuracy. Furthermore, we demonstrated that these insights generalize to real-world imagery via the COCO dataset, achieving state-of-the-art layout adherence without any architectural modifications. The implications of our findings are twofold: first, they suggest that the perceived lack of spatial control in current models is not a limitation of architectural capacity. Second, our approach offers a highly efficient, architecture-agnostic path for enhancing spatial controllability, providing a recipe for integrating native layout adherence into future foundation models.

Limitations and future work. While our approach provides a lightweight and architectureagnostic path to spatial grounding, it reveals a trade-off: a strong bias toward high-noise regimes during fine-tuning, while essential for layout adherence, can compromise highfrequency textural synthesis. This suggests that the global structural alignment facilitated by high-noise training may compete with local generative refinement required for high-fidelity synthesis. Future research should investigate multi-stage or scheduled sampling distributions that transition from a high-noise bias (to lock in grounding) to a more uniform distribution (to recover textural quality). Furthermore, exploring Direct Preference Optimization (DPO) [33] or other reinforcement learning techniques [32] presents a promising avenue for aligning model outputs with both layout constraints and human-perceived image quality.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. https://arxiv.org/abs/2511.21631, 2025.

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. https://arxiv.org/abs/2502. 13923, 2025.

[3] Black Forest Labs. FLUX.2: Analyzing and enhancing the latent space of FLUX – representation comparison, 2025. URL https://bfl.ai/research/ representation-comparison.

[4] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. https://arxiv.org/abs/2509.23951, 2025.

[5] M Chen, I Laina, and A Vedaldi. Training-free layout control with cross-attention guidance. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 5331–5341, 2024.

[6] Bo Cheng, Yuhang Ma, Liebucha Wu, Shanyuan Liu, Ao Ma, Xiaoyu Wu, Dawei Leng, and Yuhui Yin. Hico: Hierarchical controllable diffusion model for layout-to-image generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[7] Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. Be yourself: Bounded attention for multi-subject text-to-image generation. In Computer Vision – ECCV 2024, pages 432–448, 2025.

[8] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 8780– 8794. Curran Associates, Inc., 2021.

[9] Abdelrahman Eldesokey and Peter Wonka. Build-a-scene: Interactive 3d layout control for diffusion-based image generation. In The Thirteenth International Conference on Learning Representations, 2025.

[10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 12606–12633. PMLR, 2024.

[11] Chengjian Feng, Yujie Zhong, Zequn Jie, Weidi Xie, and Lin Ma. Instagen: Enhancing object detection by training on synthetic dataset. In Proceedings of the IEEE / CVF Computer Vision and Pattern Recognition, 2024.

[12] Yutong Feng, Biao Gong, Di Chen, Yujun Shen, Yu Liu, and Jingren Zhou. Ranni: Taming text-to-image diffusion for accurate instruction following. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4744–4753, June 2024.

[13] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 27. Curran Associates, Inc., 2014.

[14] Runze He, Bo Cheng, Yuhang Ma, Qingxiang Jia, Shanyuan Liu, Ao Ma, Xiaoyu Wu, Liebucha Wu, Dawei Leng, and Yuhui Yin. Plangen: Towards unified layout planning and image generation in auto-regressive vision language models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 18143–18154, 2025.

[15] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 7514– 7528, 2021.

[16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems, 30, 2017.

[17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran Associates, Inc., 2020.

[18] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232, 2023.

[19] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24174–24184, June 2024.

[20] Rahima Khanam and Muhammad Hussain. Yolov11: An overview of the key architectural enhancements. https://arxiv.org/abs/2410.17725, 2024.

[21] Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems, 36: 65484–65516, 2023.

[22] Black Forest Labs. Flux. https://github.com/black-forest-labs/ flux, 2024.

[23] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/ blog/flux-2, 2025.

[24] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. https://arxiv.org/abs/2506.15742, 2025.

[25] Phillip Y. Lee, Taehoon Yoon, and Minhyuk Sung. Groundit: Grounding diffusion transformers via noisy patch transplantation. In Advances in Neural Information Processing Systems, 2024.

[26] Bonan Li, Yinhan Hu, Songhua Liu, and Xinchao Wang. Control and realism: Best of both worlds in layout-to-image without training. In Forty-second International Conference on Machine Learning, 2025.

[27] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22511–22521, June 2023.

[28] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

[29] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

[30] Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, Alexandre Sablayrolles, Amélie Héliou, Amos You, Andy Ehrenberg, Andy Lo, Anton Eliseev, Antonia Calvi, Avinash Sooriyarachchi, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Clémence Lanfranchi, Corentin Barreau, Cyprien Courtot, Daniele Grattarola, Darius Dabert, Diego de las Casas, Elliot Chane-Sane, Faruk Ahmed, Gabrielle Berrada, Gaëtan Ecrepont, Gauthier Guinet, Georgii Novikov, Guillaume Kunsch, Guillaume Lample, Guillaume Martin, Gunshi Gupta, Jan Ludziejewski, Jason Rute, Joachim Studnia, Jonas Amar, Joséphine Delas, Josselin Somerville Roberts, Karmesh Yadav, Khyathi Chandu, Kush Jain, Laurence

Aitchison, Laurent Fainsin, Léonard Blier, Lingxiao Zhao, Louis Martin, Lucile Saulnier, Luyu Gao, Maarten Buyl, Margaret Jennings, Marie Pellat, Mark Prins, Mathieu Poirée, Mathilde Guillaumin, Matthieu Dinot, Matthieu Futeral, Maxime Darrin, Maximilian Augustin, Mia Chiquier, Michel Schimpf, Nathan Grinsztajn, Neha Gupta, Nikhil Raghuraman, Olivier Bousquet, Olivier Duchenne, Patricia Wang, Patrick von Platen, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Pavankumar Reddy Muddireddy, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Quentin Torroba, Romain Sauvestre, Roman Soletskyi, Rupert Menneer, Sagar Vaze, Samuel Barry, Sanchit Gandhi, Siddhant Waghjale, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Teven Le Scao, Théo Cachet, Theo Simon Sorg, Thibaut Lavril, Thiziri Nait Saada, Thomas Chabal, Thomas Foubert, Thomas Robert, Thomas Wang, Tim Lawson, Tom Bewley, Tom Bewley, Tom Edwards, Umar Jamil, Umberto Tomasini, Valeriia Nemychnikova, Van Phung, Vincent Maladière, Virgile Richard, Wassim Bouaziz, Wen-Ding Li, William Marshall, Xinghui Li, Xinyu Yang, Yassine El Ouahidi, Yihan Wang, Yunhao Tang, and Zaccharie Ramzi. Ministral 3. https://arxiv.org/abs/2601.08584, 2026.

[31] Jiaqi Liu, Tao Huang, and Chang Xu. Training-free composite scene generation for layout-to-image synthesis. In Computer Vision – ECCV 2024, pages 37–53, 2025.

[32] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfe Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. Advances in Neural Information Processing Systems, 38:40783–40818, 2026.

[33] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. Advances in Neural Information Processing Systems, 38:82155– 82192, 2026.

[34] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.

[35] Martin Marek, Sanae Lotfi, Aditya Somasundaram, Andrew Gordon Wilson, and Micah Goldblum. Small batch size training for language models: When vanilla SGD works, and why gradient accumulation is wasteful. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

[36] Randall Munroe. Color survey results. https://blog.xkcd.com/2010/05/ 03/color-survey-results/, 2010. Accessed: 2026-04-30.

[37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, October 2023.

[38] Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7932–7942, June 2024.

[39] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for

high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024.

[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748– 8763. PMLR, 2021.

[41] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21 (140):1–67, 2020. URL http://jmlr.org/papers/v21/20-074.html.

[42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.

[43] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in Neural Information Processing Systems, 29, 2016.

[44] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pages 4596–4604. PMLR, 2018.

[45] Takahiro Shirakawa and Seiichi Uchida. Noisecollage: A layout-aware text-to-image diffusion model based on noise cropping and merging. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[46] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

[47] Z-Image Team. Z-image: An efficient image generation foundation model with singlestream diffusion transformer. https://arxiv.org/abs/2511.22699, 2025.

[48] Junsheng Wang, Nieqing Cao, Yan Ding, Mengying Xie, Fuqiang Gu, and Chao Chen. Ske-layout: Spatial knowledge enhanced layout generation with llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[49] Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. Instancediffusion: Instance-level control for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6232–6242, June 2024.

[50] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang

Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report. https://arxiv.org/abs/2508.02324, 2025.

[51] Keming Wu, Junwen Chen, Zhanhao Liang, Yinuo Wang, Ji Li, Chao Zhang, Bin Wang, and Yuhui Yuan. Hybrid layout control for diffusion transformer: Fewer annotations, superior aesthetics. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 17930–17940, 2025.

[52] Jiayu Xiao, Henglei Lv, Liang Li, Shuhui Wang, and Qingming Huang. R&b: Region and boundary aware zero-shot grounded text-to-image generation. In The Twelfth International Conference on Learning Representations, 2024.

[53] Han Xue, Zhiwu Huang, Qianru Sun, Li Song, and Wenjun Zhang. Freestyle layoutto-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[54] Binbin Yang, Yi Luo, Ziliang Chen, Guangrun Wang, Xiaodan Liang, and Liang Lin. Law-diffusion: Complex scene generation by diffusion with layouts. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 22669– 22679, October 2023.

[55] TaeHoon Yoon, Yunhong Min, Kyeongmin Yeo, and Minhyuk Sung. \$\psi\$-sampler: Initial particle sampling for SMC-based inference-time reward alignment in score models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

[56] Xiaohang Zhan and Dingming Liu. Larender: Training-free occlusion control in image generation via latent rendering. In International Conference on Computer Vision, 2025.

[57] Hui Zhang, Dexiang Hong, Yitong Wang, Jie Shao, Xinglong Wu, Zuxuan Wu, and Yu-Gang Jiang. Creatilayout: Siamese multimodal diffusion transformer for creative layout-to-image generation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 18487–18497, 2025.

[58] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to textto-image diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 3836–3847, October 2023.

[59] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. In The Fourteenth International Conference on Learning Representations, 2026.

[60] Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhongang Qi, Ying Shan, and Xi Li. Layoutdiffusion: Controllable diffusion model for layout-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22490–22499, June 2023.

[61] Dewei Zhou, You Li, Fan Ma, Xiaoting Zhang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6818–6828, June 2024.

[62] Dewei Zhou, Ji Xie, Zongxin Yang, and Yi Yang. 3dis-flux: Simple and efficient multi-instance generation with dit rendering. https://arxiv.org/abs/2501. 05131, 2025.

[63] Dewei Zhou, Ji Xie, Zongxin Yang, and Yi Yang. 3DIS: Depth-driven decoupled image synthesis for universal multi-instance generation. In The Thirteenth International Conference on Learning Representations, 2025.

[64] JingYuan Zhu, Shiyu Li, Yuxuan Liu, Jian Yuan, Ping Huang, Jiulong Shan, and Huimin Ma. ODGEN: Domain-specific object detection data generation with diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

## A Loss of Pixel-Level Color Precision at High $\mu$

To further understand the quantitative drop in $\mathrm { m I o U } _ { 2 : 2 5 }$ observed with shifting parameter $\mu = 3 . 0$ for square images at $2 5 6 \times 2 5 6$ resolution on our synthetic benchmark (Sec. 4.1), we conducted a manual visual analysis of generated images under this setting. We observed that the drop does not come from structural failure, but from a loss of pixel-level color precision. The model’s color assignments frequently fall just outside the color-matching threshold <sub>τ</sub>, despite correct spatial placement. While examples of correct spatial placement at $\mu = 3 . 0$ for square images at $2 5 6 \times 2 5 6$ resolution were already shown in Fig. 6, we provide additional examples in Fig. 8–10 that visualize the aggregated (union of) per-color binary intersection masks $\cup _ { c \in \mathcal { C } } \left[ M _ { c } ( I _ { \mathrm { g t } } ; \tau ) \cap M _ { c } ( I _ { \mathrm { g e n } } ; \tau ) \right]$ , which together visualize the intersection component of the per-color $\mathrm { I o U } _ { c } ( I _ { \mathrm { g t } } , I _ { \mathrm { g e n } } ; \tau )$ (Eq. 2) that is averaged into $\mathrm { m I o U } _ { \tau }$ (Eq. 3, Sec. 3.2). Aggregating across colors thus reveals which image regions are correctly matched at a given threshold <sub>τ</sub>. Each figure shows a grid of these aggregated intersection masks, with rows corresponding to color-matching threshold values $\tau \in \{ 2 , 5 , 1 0 , 1 5 , 2 0 , 2 5 \}$ and columns corresponding to the ground-truth image, its VAE reconstruction, and generated images (using the fine-tuned FLUX.2-4B-base model) at different values of $\mu .$ . In these figures, it can be seen that generated images for $\mu \geq 1$ are spatially and compositionally nearly identical. However, for the same <sub>τ</sub>, intersection coverage differs sharply across $\mu .$ For $\mu = 3 . 0 { \mathrm { : } }$ , color assignments frequently fall just outside the threshold (especially for low $\tau )$ , while for $\mu \leq 2 . 5$ they achieve higher color precision.

![](images/1e3e84491db898d83811ec6497b5a04f55eae24b6e1bc5cd12a72aea7eafa8a9.jpg)  
Figure 8: Loss of pixel-level color precision at $\mu = 3 . 0$ (Example 1/3). Aggregated per-color binary mask intersections across thresholds $\tau \in \{ 2 , 5 , 1 0 , 1 5 , 2 0 , 2 5 \}$ (rows) and GT/VAE/generated images at varying $\mu$ (columns). Intersection masks at $\mu = 3 . 0$ fade sharply for low $\tau \left( \leq 5 \right)$ , indicating loss of color precision rather than spatial misplacement.

![](images/ad4e615574814eb460b9a9e79ed9855b80c9a0735ab4b0b642313d0a7e392ebe.jpg)  
Figure 9: Loss of pixel-level color precision at $\mu = 3 . 0$ for $\tau \leq 1 5$ in a sparse scene (2/3).

![](images/37fc780956c9f9382ad72eb3a755a79586a7a3ad40ad13eb44a58e1cc2bc735a.jpg)  
Figure 10: Loss of pixel-level color precision at $\mu = 3 . 0$ for $\tau \leq 1 5$ in a dense scene (3/3).

![](images/97ded007a9584bbed9c2eeaaeac2ca8f5015de175f583aa4ff24fb9461989afd.jpg)

![](images/ae422a10a1a964b6c092e2183a1e2201f218dbe18422afbdf01e2b01a7eff8a2.jpg)  
Figure 11: Scale and density-dependent layout adherence. The generative model (blue) saturates the VAE’s capacity (gray). While localization is harder for smaller objects, performance is stable across scene densities, with near-constant median $\mathrm { m I o U } _ { 2 : 2 5 }$ for $n \in [ 3 , 1 0 ]$

## B Impact of Object Scale & Density on Layout Adherence

To assess how layout adherence varies with scene complexity and object scale, we evaluate the impact of object count and object size on the resulting spatial accuracy. For the object count analysis, we evaluate each value of $n \in \{ 1 , \ldots , 1 0 \}$ individually, corresponding to the full range of rectangles per image in our benchmark. For the size analysis, we categorize individual rectangles into five categories: Tiny $( < 1 6 ^ { 2 }$ px), Small $( 1 6 ^ { \overleftarrow { 2 } } - 3 2 ^ { 2 } ~ \mathrm { p x } )$ , Medium $( 3 2 ^ { 2 } { - } 6 4 ^ { 2 } { \mathrm { ~ p x } } )$ , Large $( 6 4 ^ { 2 } - 1 2 8 ^ { 2 } { \mathrm { p x } } )$ , and Huge $( > 1 2 8 ^ { 2 } \ : \mathrm { p x } )$

Object Scale. The relationship between object scale and layout adherence is illustrated in Fig. 11 (left). We observe a positive correlation between object size and spatial accuracy. As rectangle area increases, the model’s $\mathrm { m I o U } _ { 2 : 2 5 }$ consistently improves. For Large and Huge objects, the model achieves near-optimal spatial fidelity, with median $\mathrm { m I o U } _ { 2 : 2 5 }$ ranging from 0.85 to 0.95. While performance remains robust for Medium objects, we note an increase in variance, with the lower bound of the distribution dropping to 0.60. The model’s primary failure modes occur at smaller scales, where Tiny objects drop to a median $\mathrm { m I o U } _ { 2 : 2 5 }$ of approximately 0.41. Crucially, as shown in Fig. 11, our model’s median performance closely tracks the VAE reconstruction baseline across all size categories, indicating that the model effectively approaches the average spatial capacity of the underlying latent representation. However, we note a slight divergence in the upper quartiles (75th percentile) for Tiny and Small objects, where the VAE baseline maintains a higher performance ceiling. This suggests that while our model accurately predicts the expected location of Small objects, it does not yet fully capture the extreme precision occasionally preserved by the VAE reconstruction.

Object Density. We further analyze the impact of object density on layout adherence in Fig. 11 (right). Counter-intuitively, the model exhibits its highest variance when few rectangles are present $( n = 1 , 2 )$ , where the $\mathrm { m I o U } _ { 2 : 2 5 }$ distribution is broadest. Our investigation reveals that this is primarily driven by the object size sensitivity discussed previously. Performance in single-object images is highly dependent on whether that specific object is Tiny or Huge. As the number of objects increases, the statistics stabilize, with the median $\mathrm { m I o U } _ { 2 : 2 5 }$ remaining remarkably constant up to $n = 1 0$ . This suggests that the model is robust to scene complexity and effectively maintains layout adherence in crowded environments, consistently approaching the VAE reconstruction median regardless of object count.

<table><tr><td></td><td colspan="3">Layout Adherence</td><td colspan="3">Image Quality</td></tr><tr><td>Epoch</td><td>AP↑</td><td> $\mathsf { A P } ^ { 5 0 } \uparrow$ </td><td>AR↑</td><td>CLIP↑</td><td>FID↓</td><td>IS↑</td></tr><tr><td>0</td><td>0.3</td><td>0.9</td><td>1.6</td><td>22.90</td><td>78.99</td><td>18.29</td></tr><tr><td>1</td><td>17.5</td><td>40.1</td><td>26.0</td><td>27.83</td><td>33.78</td><td>29.46</td></tr><tr><td>2</td><td>36.4</td><td>66.1</td><td>45.1</td><td>27.91</td><td>32.89</td><td>29.57</td></tr><tr><td>3</td><td>45.5</td><td>73.7</td><td>53.7</td><td>28.06</td><td>31.55</td><td>29.19</td></tr><tr><td>4</td><td>51.1</td><td>78.3</td><td>59.1</td><td>28.06</td><td>31.88</td><td>29.69</td></tr><tr><td>5</td><td>54.6</td><td>80.5</td><td>62.5</td><td>28.06</td><td>32.21</td><td>30.06</td></tr><tr><td>6</td><td>57.0</td><td>82.5</td><td>64.7</td><td>28.06</td><td>31.47</td><td>30.47</td></tr><tr><td>7</td><td>58.7</td><td>83.8</td><td>66.2</td><td>28.08</td><td>31.34</td><td>30.20</td></tr><tr><td>8</td><td>59.9</td><td>84.5</td><td>67.2</td><td>28.06</td><td>31.11</td><td>29.78</td></tr><tr><td>9</td><td>60.7</td><td>85.0</td><td>68.1</td><td>28.02</td><td>31.06</td><td>29.36</td></tr><tr><td>10</td><td>61.8</td><td>84.7</td><td>68.8</td><td>28.03</td><td>31.43</td><td>30.06</td></tr><tr><td>COCO val (GT)</td><td>61.6</td><td>77.2</td><td>74.9</td><td>30.47</td><td>0.00</td><td>29.06</td></tr></table>

Table 6: Epoch-ablation of layout adherence and image quality on real-world images. FLUX.2-9B-base is fine-tuned on MS-COCO 2017 [28] at $\mu = 3 . 0$ and evaluated after each of 10 epochs. Results at $E = 5$ slightly differ from Tab. 4 and Tab. 5 due to training variance.

## C Extended Fine-Tuning on Real-World Images

To assess whether layout adherence continues to improve beyond the 5-epoch budget used in our <sub>µ</sub>-ablation (Tab. 4), we fine-tune FLUX.2-9B-base at $\mu = 3 . 0$ for 10 epochs, evaluating after each. Full epoch-by-epoch results are shown in Tab. 6. The 10-epoch endpoint is also reported in Tab. 5. Note that the 5-epoch checkpoint comes from this separate extended run and therefore slightly differs from the <sub>µ</sub>-ablation’s 5-epoch result (Tab. 4, Tab. 5) due to training variance. Average Precision (AP) and Average Recall (AR) improve continuously across all 10 epochs, with $\mathsf { A P } ^ { 5 0 }$ peaking at epoch 9, suggesting diminishing returns beyond this point. AP and $\mathsf { A P } ^ { 5 0 }$ exceed ground-truth COCO values after 10 epochs (see Sec. 5 for possible explanations), while AR approaches ground-truth. Image quality metrics vary more: CLIP peaks at epoch 7, remaining fairly stable between epochs 3–8 before declining slightly. The Inception Score fluctuates more but is generally highest from epoch 5 onward. FID, while also fluctuating, shows a general improving trend with extended training.

## D Compositional Failures in Pre-Trained Model

As noted in Sec. 5, compositional artifacts (e.g., merged or duplicated anatomical features) occasionally occur in generated images of our fine-tuned FLUX.2-9B-base model (Fig. 7). However, as can be seen in Fig. 12 and Fig. 13, such failure modes also occur in images generated using the pre-trained FLUX.2-9B-base model prior to any fine-tuning. The images were generated using the standard FLUX.2 sampling configuration (same as in Sec. 5), demonstrating that compositional failures such as merged elephant heads (Fig. 7, row 4) are already present in the base model rather than being introduced by our fine-tuning procedure. Each figure shows 8 samples (random integer seeds) for the fixed prompt given in the figure title, generated at $5 1 2 \times 5 1 2$ resolution with a guidance scale of 4.0 and 50 inference steps.

Prompt: Two elephants playing with each other and interlocking their tusks.

Seed: 3942341994

![](images/c0bd0fdbf102b9362d4c4e7778989624c5a29d7e38e4a1bcff9e482c7a7d7d81.jpg)

Seed: 4173077049

![](images/870189e4036c9ab38827fe9d036ac4a8ed2c18b956e091e8fc10984927753a10.jpg)

![](images/4eee5db308a5be0d26e56cdc781cc50dd0ee4a6f00ded33a21e112c85f2461d1.jpg)

Seed: 2686686013

![](images/517bc5e3588b8a895ffd6d09ef71c7519d45bf316c4f3d60b0f0b69efc18c64a.jpg)

![](images/be666509b2ded997e4e1e546319e020bdde2971a7318132a910e3f1a49c13931.jpg)

![](images/5a472d64bd9eae35511fd8af5fb3e801710e8e5338b61330413f99132f1bc0bc.jpg)

![](images/0be921bb7241b27bde9786e64df1751cf93a8b28611e8a291e4b1211c1441516.jpg)

![](images/626c8b097d8011d5a8e483f9daf0de6f56516b06b3df8e2919840448794404b8.jpg)  
Figure 12: Compositional failure example (1/2). The two elephants’ heads occasionally fuse into a single shared head (e.g., top-right image), a compositional artifact present in the pre-trained FLUX.2-9B-base model prior to any fine-tuning.

Prompt: Three seagulls huddled together in the water, facing in opposing directions.

![](images/bf7af453ef8dd685b311ac91f9ad31caffe2174cb7ee59df078e3d2de5046491.jpg)  
Figure 13: Compositional failure example (2/2). Across samples, the seagull is occasionally rendered with two beaks pointing in different directions (e.g., top-left image), or with multiple heads (e.g., the two bottom-right images), a compositional artifact already present in the pre-trained FLUX.2-9B-base model prior to any fine-tuning.