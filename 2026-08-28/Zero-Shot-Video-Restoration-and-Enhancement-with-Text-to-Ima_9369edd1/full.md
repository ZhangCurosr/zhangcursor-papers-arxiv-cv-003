# Zero-Shot Video Restoration and Enhancement with Text-to-Image Latent Diffusion Models and Multi-Modal References

Cong Cao<sup>1</sup>, Huanjing Yue<sup>1</sup>, Xin Liu<sup>2</sup>, Jingyu Yang

<sup>1</sup>School of Electrical and Information Engineering, Tianjin University, Tianjin, China   
<sup>2</sup>Computer Vision and Pattern Recognition Laboratory, School of Engineering Science, Lappeenranta-Lahti University of Technology LUT, Lappeenranta, Finland {caocong\_123@tju.edu.cn, huanjing.yue@tju.edu.cn, linuxsino@gmail.com, yjy@tju.edu.cn}

## Abstract

Zero-shot image restoration methods with text-to-image latent diffusion models have achieved great success in universal image restoration tasks without training. However, applying them to video restoration will result in severe temporal flickering. In this paper, we propose a novel framework for zero-shot video restoration and enhancement which uses a text-to-image latent diffusion model and multi-modal references. Through the proposed dual prompt tuning inversion and sampling, the inference time can be reduced to nearly 1/3 of the original. The performance and temporal consistency can be also significantly stregthened. By using the proposed texture-aware video token merging, the temporal correlation between frames can be further utilized to improve the temporal consistency. We futher propose the referenced self-attention and referenced token merging to support image reference. Experimental results demonstrate the superiority of the proposed method in restoring and enhancing temporally consistent videos.

## 1 Introduction

Recently, Denoising Diffusion Probabilistic Models (DDPMs) [1] have shown advanced generative capabilities beyond GANs, which inspires people to explore diffusion-based restoration methods. Different from using supervised learning and a diffusion framework to train a model for a specific restoration task [2], the works in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] utilize pretrained image diffusion models for universal zero-shot image restoration. Among them, [10, 11, 12, 13] work on latent space with text-to-image latent diffusion models. However, there is no temporal modeling in pretrained text-to-image latent diffusion models to process videos. Although these methods have achieved promising results on image restoration, directly applying them to video restoration will result in severe temporal flickering.

Along with the appearance of powerful pre-trained text-to-image diffusion models like Stable Diffusion [14], how to use text-to-image diffusion model for zero-shot video restoration/enhancement has garnered increasing attention. Generating temporally consistent results remains a key challenge in zero-shot video restoration/enhancement. In this work, we propose texture-aware video token merging to better balance temporal consistency and texture preservation. Another problem is accelerating the sampling process. For zero-shot video editing, DDIM inversion and DDIM sampling are common settings for acceleration. But they are not suitable for video restoration since the image for inversion is a degraded frame rather than a clean one. The combination of DDIM inversion and DDIM sampling biases the result towards the reconstructed degraded frame. Therefore we propose dual prompt tuning inversion and sampling to solve this problem. We optimize the conditional and unconditional embeddings in the inversion to better encode the information of degraded image and give a better initialization for sampling. Then we optimize the conditional and unconditional embeddings in the sampling to explore the potential of text-to-image latent diffusion model for video restoration. And our inversion and sampling can further improve temporal consistency by constraining the consistency of conditional and unconditional embeddings across frames.

On the other hand, existing zero-shot video restoration/enhancement methods did not consider introducing extra references [15, 16] to guide the restoration/enhancement. Extra references, such as text (describing the ground truth video or preferred style) or other clean images (captured from different angles or obtained by retrieval), can provide additional information to assist with video restoration and enhancement. Our inversion and sampling strategy has already supported the text reference. To better support image reference, we further propose referenced self-attention and referenced token merging. By utilizing referenced token merging, the textures of the image reference can be transferred to the input degraded videos in a temporally consistent manner.

Based on the above observations, we propose a novel framework for Zero-shot Video Restoration and enhancement using Text-to-Image latent diffusion model and Multi-modal references (ZVRM).

Our contributions are summarized as follows

• First, we propose a novel framework for zero-shot video restoration and enhancement using pretrained T2I latent diffusion model and supporting different multi-modal references, including no-reference, text-reference, image-reference, as well as both text and image references.

• Second, we propose dual prompt tuning inversion and sampling for acceleration and text reference, propose referenced self-attention and referenced token merging for image reference, and propose texture-aware video token merging to preserve temporal consistency in video restoration and enhancement.

• Extensive experiments demonstrate the effectiveness of our method in achieving temporalconsistent zero-shot video restoration and enhancement.

## 2 Related Work

## 2.1 Zero-Shot IR with Latent Diffusion Models

Zero-shot image restoration methods are training-free, utilizing a pre-trained diffusion model as the generative prior and constraining the content of the result in the sampling process to be consistent with degraded images. According to the type of pre-trained image diffusion model used, zero-shot image restoration (IR) can be divided into pixel-space zero-shot IR and latent-space zero-shot IR. Pixel-space zero-shot IR [3, 4, 5, 6, 7, 8, 9] utilizes pretrained unconditional image diffusion models working in pixel-space [1]. Latent-space zero-shot IR [10, 11, 12, 13] utilizes pretrained text-to-image latent diffusion models, such as Stable Diffusion [14]. PSLD [10] is the first framework to solve zero-shot image restoration using a text-to-image latent diffusion model. [11] proposes a prompt tuning method to jointly optimize the text embedding during sampling. [12] uses historical gradient information to guide the sampling process. However, all these methods are designed for image recovery tasks, and severe temporal flickering occurs when they are applied to degraded videos.

## 2.2 Zero-Shot Video Editing

With the recent success of text-to-image diffusion models like Stable Diffusion [14], many works apply a pre-trained text-to-image diffusion model for text-driven video editing. The key is how to solve the temporal consistency problem. Rerender-A-Video [17] proposes hierarchical cross-frame constraints to enforce temporal coherence in shapes, textures, and colors. TokenFlow [18] and VidToMe [19] enforce temporal consistency by unifying self-attention tokens across frames. Inspired by these works, we propose texture-aware video token merging, which better balances temporal consistency and texture preservation in video restoration and enhancement.

## 2.3 Zero-Shot Video Restoration and Enhancement

ZVRD [20] proposes to utilize the pre-trained image diffusion model for zero-shot video restoration, and introduces the short-long-range temporal attention layer, temporal consistency guidance, spatial-temporal noise sharing, and an early stopping sampling strategy. DiffIR2VR [21] proposes flow-guided video token merging to improve temporal consistency when applying diffusion-based image restoration models for video restoration. SVI [22] converts the time dimension to the batch dimension and introduces a batch-consistent sampling strategy that synchronizes the stochastic noise components. VISION-XL [23] further introduces a pseudo-batch consistent sampling strategy and pseudo-batch inversion to improve the performance. Although there are zero-shot video restoration and enhancement methods available, none of them consider the multi-modal reference [15, 16] for video restoration. Multi-modal references, such as text (describing the ground truth video or preferred style) or other clean images (captured from different angles or achieved from retrieve), can provide additional information to assist with video restoration and enhancement. Our method is the first zero-shot video restoration and enhancement method that supports different multi-modal references, including no-reference, text-reference, image-reference, as well as both text and image references.

## 3 Background

## 3.1 Latent Diffusion Models

Latent diffusion models (LDM) operate in the latent space with an autoencoder $\mathcal { E } , \mathcal { D }$ . First, an encoder $\mathcal { E }$ compresses the input RGB image x to a low-resolution latent $z = \mathcal { E } ( x )$ . Then, the forward and reverse diffusion processes work on the latent, where the latent can be reconstructed back into an image $\mathcal { D } ( z )$ ≈ x by the decoder $\mathcal { D } .$ . In the forward diffusion process, Gaussian noise is gradually added to $z _ { 0 }$ to obtain $z _ { t }$ through Markov transitions with the transition probability.

$$
q ( z _ { t } | z _ { t - 1 } ) = \mathcal { N } ( z _ { t } ; \sqrt { 1 - \beta _ { t } } z _ { t - 1 } , \beta _ { t } \mathrm { I } ) ,\tag{1}
$$

where $\beta _ { t }$ is the variance schedule for the timestep t. The backward process uses a trained U-Net $\varepsilon _ { \boldsymbol { \theta } }$ for denoising:

$$
p _ { \theta } ( z _ { t - 1 } | z _ { t } ) = \mathcal { N } ( z _ { t - 1 } ; \mu _ { \theta } ( z _ { t } , \tau , t ) , \Sigma _ { \theta } ( z _ { t } , \tau , t ) ) ,\tag{2}
$$

where τ denotes the textual prompt. $\mu _ { \theta }$ and $\Sigma _ { \theta }$ are computed by $\varepsilon _ { \theta }$

## 3.2 DDIM Sampling

Deterministic DDIM sampling [24] is employed to reverse diffusion process, which converts noisy latent $z _ { T }$ to a clean latent $z _ { \mathrm { 0 } }$ in a sequence of timestep:

$$
z _ { t - 1 } = \sqrt { \alpha _ { t - 1 } } \frac { z _ { t } - \sqrt { 1 - \alpha _ { t } } \varepsilon _ { \theta } } { \sqrt { \alpha _ { t } } } + \sqrt { 1 - \alpha _ { t - 1 } - \sigma _ { t } ^ { 2 } } \varepsilon _ { \theta } + \sigma _ { t } \epsilon _ { t }\tag{3}
$$

where $\alpha _ { t }$ and $\sigma _ { t }$ are parameters for noise scheduling $[ 2 4 ] , \epsilon _ { t } \sim \mathcal { N } ( 0 , 1 )$ . In practice, firstly $\hat { z } _ { 0 }$ is predicted from ${ \boldsymbol { z } } _ { t }$

$$
\hat { z } _ { 0 } = \frac { z _ { t } } { \sqrt { \bar { \alpha } _ { t } } } - \frac { \sqrt { 1 - \bar { \alpha } _ { t } } \varepsilon _ { \theta } } { \sqrt { \bar { \alpha } _ { t } } }\tag{4}
$$

where $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } } \end{array}$ . Then $z _ { t - 1 }$ is sampled using both $\hat { z } _ { 0 }$ and ${ \boldsymbol { z } } _ { t }$

$$
z _ { t - 1 } = \frac { \sqrt { \alpha _ { t } } { { \left( { 1 - { { \bar { \alpha } } } _ { t - 1 } } \right) } } } { { 1 - { { \bar { \alpha } } _ { t } } } } z _ { t } + \frac { { \sqrt { { { \bar { \alpha } } } _ { t - 1 } } { { \beta } _ { t } } } } { { 1 - { { \bar { \alpha } } _ { t } } } } { { { \hat { z } } } _ { 0 } } + { \sigma _ { t } } { \epsilon _ { t } }\tag{5}
$$

where $\beta _ { t } = 1 - \alpha _ { t }$

## 4 Method

## 4.1 Overall Framework

Given a degraded video with N frames $\{ I _ { i } \} _ { i = 0 } ^ { N - 1 }$ , our goal is to restore/enhance it to a clean video $\{ I _ { i } ^ { \prime } \} _ { i = 0 } ^ { N - 1 }$ . Our method leverages pretrained text-to-image latent diffusion models (LDM) for video restoration. In LDM, a U-Net removes the noise of latent in the reverse process, which is constructed from layers of 2D convolutional residual blocks and spatial self-attention blocks. We replace all $3 \times 3$ 2D convolutions with inflated $1 \times 3 \times 3 ~ 3 \mathrm { D }$ convolutions, so that the network can process video. For better temporal consistency, we propose texture-aware video token merging. For text-referenced video restoration/enhancement, we propose dual prompt tuning inversion and sampling. Furthermore, we propose referenced self-attention and referenced token merging for image-referenced video restoration/enhancement. The framework is illustrated in Fig. 1.

![](images/9af821922e3f696fb5dcfb7919c6754195ce74eab16186bc2b26e0ee1aaf99c8.jpg)  
Figure 1: Overview of our framework.

## 4.2 Texture-Aware Video Token Merging

Token merging has been proposed to speed up diffusion models [25] and improve the temporal consistency of generated videos [19] by merging similar tokens within a frame or between different frames. However, these methods use a unified merge ratio for all pixels. For image restoration, different areas have varying levels of difficulty in restoration [26, 27]. Smooth areas are easier to restore than areas with rich texture. We find that in video restoration, areas with rich texture suffer more visual quality damage compared to smooth areas when using a larger merge ratio. Therefore, we propose assigning different merge ratios to smooth and texture-rich areas.

We utilize the classifier in [27] to classify pixels of the input frame as belonging to either smooth or texture-rich areas. Since the input frame suffers from degradation, which affects the classification of different areas, we apply the classifier to the generated frame $\mathcal { D } ( \hat { z } _ { 0 } )$ in the second half of the sampling process every $N _ { c }$ steps $( N _ { c } { = } 2 5 )$ to update the classification of different areas. For low-light enhancement, we brighten the input frame using a gamma correction process for classification.

## 4.2.1 Texture-Aware Spatial Token Merging

We split the overall token chunk T of a frame into token sets $\mathbf { T } _ { s }$ and $\mathbf { T } _ { t } ,$ , based on the smooth and texture-rich areas, respectively. For each token set, we further partition the tokens into a source set $\mathbf { T } _ { s } ^ { s r c } \mathbf { \Gamma } ( \mathbf { T } _ { t } ^ { s r c } )$ and destination set $\mathbf { T } _ { s } ^ { d s t } ( \mathbf { T } _ { t } ^ { d s t } )$ , and compute the pair-wise cosine similarity between the source and destination sets. The $r _ { s } \left( r _ { t } \right)$ most similar paired source-destination tokens from $\mathbf { T } _ { s } \left( \mathbf { T } _ { t } \right)$ are merged. The merging ratio $r _ { s }$ is much larger than $r _ { t }$ . After self-attention, the merged tokens are unmerged to maintain the original shape. The spatial merging and unmerging operations are the same as in [25]. The unmerged $\mathbf { T } _ { s }$ and $\mathbf { T } _ { t }$ are finally assigned to their original positions.

## 4.2.2 Texture-Aware Temporal Token Merging

For temporal token merging, we select the first frame as the key frame and merge the other frames into the target frame based on token similarity between frames. Texture-rich areas often become very blurry after merging with a high temporal token merging ratio, while smooth areas remain almost unchanged. Therefore, we assign a higher merging ratio to the smooth areas of frames than to the texture-rich areas to further reduce computational costs. Specifically, we merge $\mathbf { T } _ { s }$ and $\mathbf { T } _ { t }$ of different frames, respectively. Then we unmerge the tokens after self-attention. The temporal merging and unmerging operations are the same as in [19].

## 4.3 Text-Referenced Strategy

## 4.3.1 Dual Prompt Tuning Inversion

PSLD [10] starts DDIM sampling from pure Gaussian noise latent and still requires 1000 steps of iteration for the reverse diffusion process. [28, 29] have proved that starting from a better initialization can significantly reduce the number of sampling steps. We apply the Distortion Adaptive

(DA) Inversion from [29] to convert $z _ { 0 } = \mathcal { E } ( A ^ { T } y )$ into $z _ { T ^ { \prime } }$ to provide a good initialization.

$$
\begin{array} { r l } & { z _ { t + 1 } = \sqrt { \alpha _ { t + 1 } } \frac { z _ { t } - \sqrt { 1 - \alpha _ { t } } \varepsilon _ { \theta } } { \sqrt { \alpha _ { t } } } } \\ & { ~ + \sqrt { 1 - \alpha _ { t + 1 } - \eta \beta _ { t + 1 } } \varepsilon _ { \theta } + \sqrt { \eta \beta _ { t + 1 } } \epsilon _ { t } ^ { \prime } } \end{array}\tag{6}
$$

where $\eta \left( 0 < \eta < 1 \right)$ is a hyper-parameter, $\boldsymbol { \epsilon } _ { t } ^ { \prime }$ is Gaussian noise and $\epsilon _ { t } ^ { \prime } \sim \mathcal { N } ( 0 , 1 )$ , and $T ^ { \prime }$ is set to 15. $z _ { T ^ { \prime } }$ is then employed for initialization. Although DDIM inversion is commonly used in image and video editing, it is not suitable for restoration. When inverting the latent from a degraded image into noisy latent, the predicted noise during the DDIM inversion process will deviate from the standard normal distribution, resulting in the deviation of the initialization from the predefined noise distribution, which generates unrealistic results.

Different from unconditional image diffusion models [29], text-to-image latent diffusion models utilize the classifier-free guidance technique [30] to amplify the effect of conditioned text.

$$
\varepsilon _ { \theta } ( z _ { t } , t , \mathcal { C } , \infty ) = w \cdot \varepsilon _ { \theta } ( z _ { t } , t , \mathcal { C } ) + ( 1 - w ) \cdot \varepsilon _ { \theta } ( z _ { t } , t , \mathcal { C } )\tag{7}
$$

For PSLD, $w = 7 . 5$ is the default parameter. Inspired by [31], we propose to optimize both conditional embeddings $\mathcal { C }$ and unconditional embeddings ∅ in the inversion to better encode the information of degraded images, namely dual prompt tuning inversion, which means automatically tuning the conditional text and unconditional null-text embeddings. The DA inversion with $w = 0$ outputs a sequence of noisy latent codes $z _ { T ^ { \prime } } ^ { * } , . . . , z _ { 0 } ^ { * }$ where $z _ { 0 } ^ { * } = z _ { 0 } = \mathcal { E } ( x _ { 0 } ^ { d e g r a d e d } )$ . We initialize $\tilde { z } _ { T ^ { \prime } } = z _ { T } ^ { * }$ <sub>′</sub> and perform the following optimization on the conditional embedding $\mathcal { C }$ with $w = 7 . 5$ for the timestamps $\mathbf { \bar { \Phi } } t = T ^ { \prime } , . . . , 1$

$$
\operatorname* { m i n } _ { \mathcal { C } _ { t } } | | z _ { t - 1 } ^ { * } - z _ { t - 1 } ( \tilde { z } _ { t } , t , \mathcal { C } _ { t } , \emptyset ) | | _ { 2 } ^ { 2 } + \gamma _ { 1 } | | \mathcal { C } _ { t } ^ { i } - \mathcal { C } _ { t } ^ { i - 1 } | | _ { 2 } ^ { 2 }\tag{8}
$$

the second term constrains the $\mathcal { C } _ { t }$ of frames i and $i + 1$ to have a similar value, which maintains temporal consistency. Through the above optimization, we can obtain the optimized conditional embeddings $\mathcal { C } .$ . Then we optimize for the unconditional embedding $\mathcal { O } ,$ where the DA inversion with $w = 1$ outputs a sequence of noisy latent codes $z _ { T ^ { \prime } } ^ { * * } , . . . , z _ { 0 } ^ { * * }$ . We initialize $\overline { { z } } _ { T ^ { \prime } } = z _ { T ^ { \prime } } ^ { * * }$ and perform the following optimization with $w = 7 . 5$ for the timestamps $\mathit { t } { = } T ^ { \prime } , { \ldots } , 1$

$$
\operatorname* { m i n } _ { \mathcal { O } _ { t } } | | z _ { t - 1 } ^ { * } - z _ { t - 1 } ( \overline { { z } } _ { t } , t , \mathcal { C } , \mathcal { O } _ { t } ) | | _ { 2 } ^ { 2 } + \gamma _ { 2 } | | \mathcal { O } _ { t } ^ { i } - \mathcal { O } _ { t } ^ { i - 1 } | | _ { 2 } ^ { 2 }\tag{9}
$$

where $\gamma _ { 1 }$ and $\gamma _ { 2 }$ are hyperparameters. Through the above optimization, we can get the optimized unconditional embeddings ∅. When applying inversion, we share the same noise $\epsilon _ { t } ^ { \prime }$ across all frames to further preserve temporal consistency. Through the proposed dual prompt tuning inversion, we can reduce sampling steps $\bar { \boldsymbol { T } }$ from 1000 to 250 with 60 additional inversion steps.

## 4.3.2 Dual Prompt Tuning Sampling

In the sampling process, we continue to optimize the conditional embeddings C and unconditional embeddings $\mathcal { D }$ to improve the performance. Although the conditional text and unconditional null-text embeddings have been optimized in the inversion, the optimization focuses on encoding the input degraded frames to give a better initialization to accelerate the sampling. The optimized conditional and unconditional embeddings are not most suitable for the sampling process and need further optimization. Firstly, we initialize conditional and unconditional embeddings with the optimization results in the inversion. Then we optimize the conditional embeddings, unconditional embeddings, and the reverse diffusion result alternately for each timestamp $t \stackrel { - } { = } T , . . . , 0$ . In every timestep t of sampling, a clean latent $\hat { z } _ { 0 }$ can be predicted from the noisy latent ${ \boldsymbol { z } } _ { t }$ by Eq. $^ { 4 , }$ which is also influenced by $\mathcal { C } _ { t }$ and $\scriptstyle { \mathcal { O } } _ { t }$ . For conditional embeddings $\mathcal { C } _ { t }$ , the optimization objective is:

$$
\operatorname* { m i n } _ { \mathcal { C } _ { t } } | | y - A ( \mathcal { D } ( \hat { z } _ { 0 } ( \mathcal { C } _ { t } , \mathcal { O } _ { t } ) ) ) | | _ { 2 } ^ { 2 } + \gamma _ { 1 } ^ { \prime } | | \mathcal { C } _ { t } ^ { i } - \mathcal { C } _ { t } ^ { i - 1 } | | _ { 2 } ^ { 2 }\tag{10}
$$

the first term aligns $\mathcal { C } _ { t }$ with the measurement $y ,$ the second term constrains the $\mathcal { C } _ { t }$ of different frames to maintain temporal consistency. We fix $\mathcal { C } _ { t }$ with the optimized result $\mathcal { C } _ { t } ^ { * }$ , and then optimize the unconditional embeddings $\scriptstyle { \mathcal { O } } _ { t }$

$$
\operatorname* { m i n } _ { \mathcal { O } _ { t } } | | y - A ( \mathcal { D } ( \hat { z } _ { 0 } ( \mathcal { C } _ { t } ^ { * } , \mathcal { D } _ { t } ) ) ) | | _ { 2 } ^ { 2 } + \gamma _ { 2 } ^ { \prime } | | \mathcal { O } _ { t } ^ { i } - \mathcal { O } _ { t } ^ { i - 1 } | | _ { 2 } ^ { 2 }\tag{11}
$$

the first term aligns $\scriptstyle { \mathcal { O } } _ { t }$ with the measurement $y ,$ , the second term constrains the $\scriptstyle { \mathcal { O } } _ { t }$ of different frames to maintain temporal consistency. $\gamma _ { 1 } ^ { \prime }$ and $\gamma _ { 2 } ^ { \prime }$ are hyperparameters. With fixed optimized results $\mathcal { C } _ { t } ^ { * }$ and $\boldsymbol { \mathcal { Q } } _ { t } ^ { * }$ , the reverse diffusion results are optimized by the loss in PSLD. Due to the balance between computation cost and performance, we only optimize C and ∅ in N iterations after $t < T _ { d p t s } . \ T _ { d p t s }$ is set to 5, and N is set to 5.

The Dual Prompt Tuning Inversion and Sampling support both no-text (null-text) reference and text reference. For null-text reference, conditional embeddings are generated from null text.

## 4.4 Image-Referenced Module

First, we apply the above inversion to the reference image to obtain the noisy latent of the reference image. Then, in the sampling process, we transfer the texture from the reference image to the current frame using the proposed modules.

## 4.4.1 Referenced Self-Attention

Inspired by cross-frame attention in zero-shot video editing [32, 17], we propose to replace the original self-attention layers in the U-Net with referenced self-attention layers to transfer the texture of the referenced image to the current frame. For each spatial self-attention layer, the query, key and value $Q ,$ K, V are obtained by linear projection of the feature $v _ { i }$ of $I _ { i } .$ . The corresponding spatial self-attention output is produced by $\begin{array} { r } { { S e l f A t t n } ( Q , K , V ) = { S o f t m a x } ( \frac { Q K ^ { T } } { \sqrt { d } } ) \cdot V . } \end{array}$

$$
Q = W ^ { Q } v _ { i } , K = W ^ { K } v _ { i } , V = W ^ { V } v _ { i } ,\tag{12}
$$

where $W ^ { Q } , W ^ { K } , W ^ { V }$ are pre-trained matrices that project the inputs to query, key and value, respectively. We propose referenced self-attention which uses the key $K ^ { \prime }$ and value ${ \check { V } } ^ { \prime }$ from both the current frame itself and the referenced image, allowing the current frame to pay attention to itself and the reference image simultaneously. The referenced self-attention output is produced by Ref\_SelfAttn(Q, K<sup>′</sup>, V <sup>′</sup>) = Softmax( QK<sup>′T</sup> ) · V<sup>′</sup>.   
√<sub>d</sub>

$$
Q = W ^ { Q } v _ { i } , K ^ { \prime } = W ^ { K } [ v _ { i } , v _ { r e f } ] , V ^ { \prime } = W ^ { V } [ v _ { i } , v _ { r e f } ] ,\tag{13}
$$

## 4.4.2 Referenced Token Merging

To further transfer the texture of the referenced image to the current frame and reduce computational costs, we propose referenced token merging. We select the referenced image as the key image and merge the current frame into the referenced image based on token similarity between images. We also utilize the classifier in [27] to classify pixels of the reference image as belonging to either smooth or texture-rich areas. Then we merge $\mathbf { T } _ { s } \left( \mathbf { T } _ { t } \right)$ from the referenced image and the current frame. After self-attention, we unmerge the merged tokens back into the original referenced image and current frame tokens. We alternately apply referenced self-attention and referenced token merging to the two self-attention modules in the transformer block of UNet, enhancing the fusion between the current frame and the referenced image for better texture transfer.

## 5 Experiments

## 5.1 Datasets

We evaluate our method on 2 video restoration tasks (video super-resolution, video deblur) and 1 video enhancement task (low-light video enhancement). For evaluation of video restoration, we collected 15 gt videos from commonly used test datasets for restoration (REDS4 [33], UDM10 [34], DAVIS [35]). For evaluation of video enhancement, we collected 10 paired low-light and normal-light videos from the DID dataset [36], which was captured in the real world. Due to the slow sampling speed and the fact that a test video contains a lot of frames, we first center crop the frames along the shorter edge, and then resize them to $5 1 2 \times 5 1 2$ , which matches the image size of the text-to-image diffusion model. For degraded videos of restoration tasks, we follow the settings of the linear degradation operator from [10] and [9]. For video enhancement, we utilize the same degradation model as in [9] and optimize the parameters of the degradation model during the sampling process. Due to page limitations, we provide more details in the supplementary materials.

![](images/db4c49346390cbf4d564e39e594b976324b31da336de0dc453522f2b516361ec.jpg)  
(a) Video Super-Resolution

![](images/60eac8cd64758f99f32575390051be636d8ecbe38c5404e36985fd404a89614c.jpg)  
(b) Low-Light Video Enhancement  
Figure 2: Visual quality comparison for video super-resolution and low-light video enhancement. Zoom in for better observation.

Table 1: Quantitative comparison with state-of-the-art methods for 4× video super-resolution. The best results are highlighted in bold.
<table><tr><td>Methods</td><td>Backbone</td><td>PSNR↑</td><td>SSIM↑</td><td>WE(10−2)↓</td></tr><tr><td>PSLD</td><td>SDv1.5</td><td>24.85</td><td>0.6313</td><td>2.0854</td></tr><tr><td>ZVRM (no Ref.)</td><td>SDv1.5</td><td>25.16</td><td>0.6599</td><td>0.5003</td></tr><tr><td>ZVRM (text Ref.)</td><td>SDv1.5</td><td>25.32</td><td>0.6642</td><td>0.4755</td></tr><tr><td>ZVRM (text Ref.+image Ref. 1)</td><td>SDv1.5</td><td>25.61</td><td>0.6785</td><td>0.3871</td></tr><tr><td>VISION-XL</td><td>SDXL</td><td>25.91</td><td>0.7487</td><td>1.5578</td></tr><tr><td>ZVRM (no Ref.)</td><td>SDXL</td><td>26.13</td><td>0.7594</td><td>0.9345</td></tr></table>

## 5.2 Comparison with State-of-the-art Methods

We utilize three metrics to evaluate the restoration and enhancement quality. Besides the commonly used metrics PSNR and SSIM, we utilize Warping Error (WE) [37] to evaluate temporal consistency. Due to the page limitation, we provide more analyses of inference time in the supplementary materials. For zero-shot video restoration and enhancement, we mainly compare our method with PSLD and VISION-XL, PSLD is also our backbone. Since VISION-XL utilizes Stable Diffusion XL 1.0 (SDXL) instead of Stable Diffusion v1.5 (SDv1.5), like PSLD, we change our SD backbone to SDXL to ensure a fair comparison when comparing our method with VISION-XL. It is worth noting that our method is also a plug-and-play module, which can be applied to any diffusion-based model. And our method can also be applied to blind restoration tasks with complex real-world degradation. We transfer our method to the backbone DiffBIR for blind video super-resolution, comparing it with DiffIR2VR and ZVRD.

![](images/0074f8d759094ca85db2a472f0395a63cb95a1e0894b831dfd736697f211ab21.jpg)  
(a)

![](images/5f43280b749551c1e4093ed4fe7118ec926d56f4ec83f0f63f212650849283f9.jpg)  
(b)  
Figure 3: Different kinds of reference images for zero-shot video super-resolution on the RealMCVSR dataset.

Table 2: Quantitative comparison with state-of-the-art methods for low-light video enhancement. The best results are highlighted in bold.
<table><tr><td>Methods</td><td>Backbone</td><td>PSNR↑</td><td>SSIM↑</td><td> $\mathrm { W E } ( 1 0 ^ { - 2 } ) \downarrow$ </td></tr><tr><td>PSLD</td><td>SDv1.5</td><td>18.17</td><td>0.8174</td><td>0.6895</td></tr><tr><td>ZVRM (no Ref.)</td><td>SDv1.5</td><td>18.40</td><td>0.8303</td><td>0.2170</td></tr><tr><td>ZVRM (text Ref.)</td><td>SDv1.5</td><td>18.45</td><td>0.8369</td><td>0.2032</td></tr><tr><td>ZVRM (text Ref.+image Ref. 1)</td><td>SDv1.5</td><td>18.79</td><td>0.8552</td><td>0.1924</td></tr></table>

Table 3: Quantitative comparison of 4× video super-resolution using different reference image sources on the RealMCVSR dataset [16]. The best results are highlighted in bold.
<table><tr><td>Methods</td><td>Backbone</td><td>PSNR↑</td><td>SSIM↑</td><td> $\mathrm { W E } ( 1 0 ^ { - 2 } ) \downarrow$ </td></tr><tr><td>PSLD</td><td>SDv1.5</td><td>22.91</td><td>0.6241</td><td>1.9198</td></tr><tr><td>ZVRM (no Ref.)</td><td>SDv1.5</td><td>23.14</td><td>0.6339</td><td>0.8215</td></tr><tr><td>ZVRM (image Ref. 1)</td><td>SDv1.5</td><td>23.46</td><td>0.6552</td><td>0.8069</td></tr><tr><td>ZVRM (image Ref. 2)</td><td>SDv1.5</td><td>23.33</td><td>0.6439</td><td>0.7720</td></tr><tr><td>ZVRM (image Ref. 3)</td><td>SDv1.5</td><td>23.37</td><td>0.6461</td><td>0.8013</td></tr></table>

For settings of no reference, we only give the null-text to our model. This is also the most common situation in the real world. Text references also exist in real world. For example, when people shoot video for sharing in the Internet, they sometimes also record a voiceover or introduction for the video. In addition, the videos on the Internet often have a description. For the restoration/enhancement task, users can also give a description for their desired output. To ease the evaluation, we follow [38] and generate text from ground truth for the settings of text reference. To be specific, we feed the full resolution ground truth frame to GPT4o to generate reference text. We can also generate reference text from degraded frames or user-preferred style. However, this will cause the enhanced result deviates from the GT, which makes it difficult to evaluate the restoration methods. For image reference, we follow the settings of [39] to ease the evaluation. To be specific, we select a GT frame that does not overlap with the test clip as the reference image. We name this kind of reference image Ref. 1. To explore the influence of different kinds of reference images, we also present results when using the clean image from retrieve (Ref. 2) and the image captured from another angle (Ref. 3) as reference image.

Table 1 and 2 list the quantitative results on the evaluation data for video super-resolution and lowlight video enhancement, respectively. It can be observed that our method ZVRM outperforms PSLD in all three metrics across the three tasks, even without any reference. For 4× video super-resolution, ZVRM (no Ref.) outperforms PSLD with a 0.31 dB gain in PSNR, has nearly only 1/4 of the WE value. For low-light video enhancement, ZVRM (no Ref.) outperforms PSLD with a 0.23 dB gain in PSNR, has nearly only 1/3 of the WE value. Text and image references (Ref. 1) can further enhance the performance. Our method also outperforms another zero-shot video restoration method, VISION-XL, in all three metrics. For 4× video super-resolution, ZVRM (no Ref.) outperforms VISION-XL with a 0.22 dB gain in PSNR, and has nearly half the WE value. This demonstrates the superiority of our method in terms of performance and temporal consistency. Due to page limitations, we provide the comparisons for video deblurring in the supplementary materials.

Fig. 2 presents the visual comparison results on the evaluation data for video super-resolution and low-light video enhancement. For video super-resolution, the top area of the bus has different textures in the two frames of PSLD results. Our method can restore more temporally consistent results. For low-light video enhancement, it can be observed that PSLD results in a loss of many details, whereas our method can maintain more details and preserve temporal consistency. Due to page limitations, we provide more comparisons in the supplementary materials.

We further explore the influence of different kinds of reference texts and images. Fig. 3 (a) shows the results when using a ground truth (gt) caption or a user-preferred style as reference text. When provided with a description of the user’s preferred style, such as “Monet,” as a reference text, our model can also generate the corresponding result with the desired style. Fig. 3 (b) shows the results when using three kinds of reference images. The Ref. 2 image from Fig. 3 (b) is retrieved from the

Table 4: Quantitative comparison with state-of-the-art methods for 4× blind video super-resolution on the DAVIS dataset. The best results are highlighted in bold.
<table><tr><td>Methods</td><td>Backbone</td><td>PSNR↑</td><td>SSIM↑</td><td>WE(10−2)↓</td></tr><tr><td>DiffBIR</td><td>SDv2.1</td><td>24.47</td><td>0.6727</td><td>0.6907</td></tr><tr><td>DiffIR2VR</td><td>SDv2.1</td><td>24.49</td><td>0.6718</td><td>0.6818</td></tr><tr><td>DiffBIR+ZVRD</td><td>SDv2.1</td><td>24.55</td><td>0.6799</td><td>0.4923</td></tr><tr><td>DiffBIR+ZVRM (no Ref.)</td><td>SDv2.1</td><td>24.71</td><td>0.6839</td><td>0.4397</td></tr><tr><td>DiffBIR+ZVRM (text Ref.)</td><td>SDv2.1</td><td>24.80</td><td>0.6860</td><td>0.4375</td></tr><tr><td>DiffBIR+ZVRM (text Ref.+image Ref. 1)</td><td>SDv2.1</td><td>24.96</td><td>0.6971</td><td>0.4288</td></tr></table>

Table 5: Ablation study for Dual Prompt Tuning Inversion (DPTI), Dual Prompt Tuning Sampling (DPTS), Texture-Aware Spatial Token Merging (TSTM), Texture-Aware Temporal Token Merging (TTTM), Referenced Self-Attention (RSA), and Referenced Token Merging (RTM) on 4× video super-resolution task.
<table><tr><td>DPTT</td><td>×</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>DPTS</td><td>×</td><td>×</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>TSTM</td><td>×</td><td>×</td><td>×</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>TTTM</td><td>×</td><td>×</td><td>×</td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>RSA</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>√</td><td>√</td></tr><tr><td>RTM</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>√</td></tr><tr><td>PSNR↑</td><td>24.85</td><td>24.88</td><td>25.03</td><td>25.05</td><td>25.16</td><td>25.31</td><td>25.49</td></tr><tr><td>SSIM↑</td><td>0.6313</td><td>0.6501</td><td>0.6585</td><td>0.6572</td><td>0.6599</td><td>0.6654</td><td>0.6726</td></tr><tr><td>WE(10−2)↓</td><td>2.0854</td><td>1.9646</td><td>1.0721</td><td>1.0710</td><td>0.5003</td><td>0.4622</td><td>0.4057</td></tr><tr><td>Steps↓</td><td>1000</td><td>310</td><td>360</td><td>360</td><td>360</td><td>360</td><td>360</td></tr></table>

DIV2K dataset by the retrieval system in [40]. Table 3 further provides the quantitative comparison results. It can be observed that every kind of reference image can improve the performance.

Following the settings of DiffIR2VR [21], we use DiffBIR as the backbone for blind video superresolution and evaluate it on the DAVIS testing sets. We compare our method with DiffIR2VR, ZVRD, and the backbone DiffBIR. Table 4 further provides the quantitative comparison results. It can be observed that our method achieves the best performance in all metrics. Due to page limitations, we provide visual quality comparison results in the supplementary materials.

## 5.3 Ablation Study

In this section, we perform an ablation study to demonstrate the effectiveness of the proposed dual prompt tuning inversion, dual prompt tuning sampling, referenced self-attention, referenced token merging, texture-aware spatial token merging, and temporal token merging. Taking video superresolution as an example, Table 5 lists the quantitative comparison results in the evaluation data by adding these modules one by one. Our proposed modules can bring a 0.64 dB gain for PSNR, a 0.0403 gain for SSIM, nearly a 4/5 reduction for WE, and reduce the diffusion steps from 1000 to 360 (60 inversion steps and 300 sampling steps). By adding the dual prompt tuning inversion (null-text reference), the diffusion steps are accelerated from 1000 to 310 (60 inversion steps and 250 sampling steps) while slightly increasing performance. By replacing plain sampling with dual prompt tuning sampling (null-text reference), the performance of three metrics can be significantly improved with an additional cost of 50 sampling steps (0.15 dB for PSNR, nearly a 1/2 reduction for WE). Referenced self-attention and referenced token merging can bring gains of 0.15 dB and 0.18 dB for PSNR, respectively. Due to page limitations, we provide more ablation studies in the supplementary materials.

## 6 Conclusion

In this paper, we propose a novel framework for zero-shot video restoration using a pretrained text-toimage latent diffusion model and multi-modal references. Through the proposed dual prompt tuning inversion and sampling, the inference time can be reduced to nearly one-third of the original. The performance and temporal consistency can also be significantly strengthened. By using the proposed texture-aware video token merging, the temporal correlation between frames can be further utilized to improve the temporal consistency. We further propose referenced self-attention and referenced token merging to support image reference. Experimental results demonstrate the superiority of the proposed method in speed, performance, and temporal consistency.

## References

[1] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

[2] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4713–4726, 2022.

[3] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems (NeurIPS), 32, 2019.

[4] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[5] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[6] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. In ICLR Workshop on Deep Generative Models for Highly Structured Data (ICLRW), 2022.

[7] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490, 2022.

[8] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022.

[9] Ben Fei, Zhaoyang Lyu, Liang Pan, Junzhe Zhang, Weidong Yang, Tianyue Luo, Bo Zhang, and Bo Dai. Generative diffusion prior for unified image restoration and enhancement. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9935–9946, 2023.

[10] Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, and Sanjay Shakkottai. Solving linear inverse problems provably via posterior sampling with latent diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

[11] Hyungjin Chung, Jong Chul Ye, Peyman Milanfar, and Mauricio Delbracio. Prompt-tuning latent diffusion models for inverse problems. arXiv preprint arXiv:2310.01110, 2023.

[12] Linchao He, Hongyu Yan, Mengting Luo, Kunming Luo, Wang Wang, Wenchao Du, Hu Chen, Hongyu Yang, and Yi Zhang. Iterative reconstruction based on latent diffusion model for sparse data reconstruction. arXiv preprint arXiv:2307.12070, 2023.

[13] Litu Rout, Yujia Chen, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Beyond first-order tweedie: Solving inverse problems using latent diffusion. arXiv preprint arXiv:2312.00852, 2023.

[14] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[15] Bo Zhang, Mingming He, Jing Liao, Pedro V Sander, Lu Yuan, Amine Bermak, and Dong Chen. Deep exemplar-based video colorization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8052–8061, 2019.

[16] Junyong Lee, Myeonghee Lee, Sunghyun Cho, and Seungyong Lee. Reference-based video superresolution using multi-camera video triplets. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17824–17833, 2022.

[17] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. arXiv preprint arXiv:2306.07954, 2023.

[18] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023.

[19] Xirui Li, Chao Ma, Xiaokang Yang, and Ming-Hsuan Yang. Vidtome: Video token merging for zero-shot video editing. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7486–7495, 2024.

[20] Cong Cao, Huanjing Yue, Xin Liu, and Jingyu Yang. Zero-shot video restoration and enhancement using pre-trained image diffusion model. AAAI, 2025.

[21] Chang-Han Yeh, Chin-Yang Lin, Zhixiang Wang, Chi-Wei Hsiao, Ting-Hsuan Chen, Hau-Shiang Shiu, and Yu-Lun Liu. Diffir2vr-zero: Zero-shot video restoration with diffusion-based image restoration models. arXiv preprint arXiv:2407.01519, 2024.

[22] Taesung Kwon and Jong Chul Ye. Solving video inverse problems using image diffusion models. ICLR, 2024.

[23] Taesung Kwon and Jong Chul Ye. Vision-xl: High definition video inverse problem solver using latent image diffusion models. arXiv preprint arXiv:2412.00156, 2024.

[24] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[25] Daniel Bolya and Judy Hoffman. Token merging for fast stable diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4599–4603, 2023.

[26] Xiangtao Kong, Hengyuan Zhao, Yu Qiao, and Chao Dong. Classsr: A general framework to accelerate super-resolution networks by data characteristic. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12016–12025, 2021.

[27] Jinho Jeong, Jinwoo Kim, Younghyun Jo, and Seon Joo Kim. Accelerating image super-resolution networks with pixel-level classification. arXiv preprint arXiv:2407.21448, 2024.

[28] Hyungjin Chung, Byeongsu Sim, and Jong Chul Ye. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12413–12422, 2022.

[29] Gongye Liu, Haoze Sun, Jiayi Li, Fei Yin, and Yujiu Yang. Accelerating diffusion models for inverse problems through shortcut sampling. arXiv preprint arXiv:2305.16965, 2023.

[30] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[31] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023.

[32] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023.

[33] Seungjun Nah, Sungyong Baik, Seokil Hong, Gyeongsik Moon, Sanghyun Son, Radu Timofte, and Kyoung Mu Lee. Ntire 2019 challenge on video deblurring and super-resolution: Dataset and study. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 0–0, 2019.

[34] Peng Yi, Zhongyuan Wang, Kui Jiang, Junjun Jiang, and Jiayi Ma. Progressive fusion video superresolution network via exploiting non-local spatio-temporal correlations. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 3106–3115, 2019.

[35] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017.

[36] Huiyuan Fu, Wenkai Zheng, Xicong Wang, Jiaxuan Wang, Heng Zhang, and Huadong Ma. Dancing in the dark: A benchmark towards general low-light video enhancement. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 12877–12886, 2023.

[37] Wei-Sheng Lai, Jia-Bin Huang, Oliver Wang, Eli Shechtman, Ersin Yumer, and Ming-Hsuan Yang. Learning blind video temporal consistency. In Proceedings of the European conference on computer vision (ECCV), pages 170–185, 2018.

[38] Jeongsol Kim, Geon Yeong Park, Hyungjin Chung, and Jong Chul Ye. Regularization by texts for latent diffusion inverse solvers. ICLR, 2025.

[39] Yuming Jiang, Kelvin CK Chan, Xintao Wang, Chen Change Loy, and Ziwei Liu. Reference-based image and video super-resolution via c2-matching. IEEE transactions on pattern analysis and machine intelligence, 45(7):8874–8887, 2022.

[40] Hang Guo, Tao Dai, Zhihao Ouyang, Taolin Zhang, Yaohua Zha, Bin Chen, and Shu-tao Xia. Refir: Ground ing large restoration models with retrieval augmentation. Advances in Neural Information Processing Systems, 37:46593–46621, 2024.

# Zero-Shot Video Restoration and Enhancement with Text-to-Image Latent Diffusion Models and Multi-Modal References – Supplementary Material –

Cong Cao<sup>1</sup>, Huanjing Yue<sup>1</sup>, Xin Liu<sup>2</sup>, Jingyu Yang<sup>1</sup>

<sup>1</sup>School of Electrical and Information Engineering, Tianjin University, Tianjin, China   
<sup>2</sup>Computer Vision and Pattern Recognition Laboratory, School of Engineering Science, Lappeenranta-Lahti University of Technology LUT, Lappeenranta, Finland {caocong\_123@tju.edu.cn, huanjing.yue@tju.edu.cn, linuxsino@gmail.com, yjy@tju.edu.cn}

This supplementary file provides details which were not presented in the main paper due to page limitations. In the following, we first give the details for zero-shot video enhancement and inference time analyses. Then we present more comparison results and ablation study. And a demo for video results comparison is given. Finally, limitations and societal impact of our work are given.

## 1 Learning degradation model for Zero-shot Video Enhancement

Linear inverse problems in image restoration (IR) can be formulated as

$$
y = A x + n\tag{1}
$$

where A is the linear degradation operator and n is additive white Gaussian noise, the task is to restore the ground-truth image x from the degraded image y. Following PSLD [1], a latent constraint is applied in the reverse diffusion process to preserve the content between the generated result and the degraded image. The constraint loss is formulated as

$$
\begin{array} { r l } & { \quad \mathcal { L } = \mathcal { L } _ { r e c } + \gamma _ { 1 } \mathcal { L } _ { r e g } } \\ & { \mathcal { L } _ { r e c } = \| y - A ( \mathcal { D } ( \hat { z } _ { 0 } ) ) \| _ { 2 } ^ { 2 } } \\ & { \mathcal { L } _ { r e g } = \| \hat { z } _ { 0 } - \mathcal { E } ( A ^ { T } y + ( I - A ^ { T } A ) \mathcal { D } ( \hat { z } _ { 0 } ) ) \| _ { 2 } ^ { 2 } } \end{array}\tag{2}
$$

where $\mathcal { L } _ { r e c }$ directly constrains the content, $\mathcal { L } _ { r e g }$ penalizes latents that are not fixed-points of the composition of the decoder-function with the encoder-function, ensuring that the generated sample remains on the manifold of real data. For zero-shot video enhancement, we utilize the degradation model in [2] and optimize the parameters of the degradation model in the sampling process. The degradation model can be formulated as follows:

$$
\begin{array} { r } { \pmb { y } = f \pmb { \mathcal { D } } ( \hat { z } _ { 0 } ) + \pmb { \mathcal { M } } , } \end{array}\tag{3}
$$

where the factor f is a scalar and the mask M is a matrix of the same dimension as $\mathcal { D } ( \hat { z } _ { \mathbf { 0 } } )$ . We optimize f and M using gradient descent during the sampling process.

## 2 Inference Time Analyses

The algorithm in 1 shows our entire reverse diffusion process. All experiments were conducted on a 48G A40 GPU. Taking the backbone PSLD as an example, each sampling step takes approximately 1.424 seconds without any reference. Our dual prompt tuning inversion and sampling can reduce the total sampling steps from 1000 to 360, reducing the total inference time from 23min44s to 8min32s. Our texture-aware video token merging can further reduce the cost of each sampling step from 1.424s to 1.184s, reducing the total inference time from 8min32s to 7min6s.

Algorithm 1 Sampling process of the proposed ZVRM (without reference).   
Input: Corrupted video $\overline { { \{ I _ { i } \} _ { i = 0 } ^ { N - 1 } } }$ , conditional embeddings C, unconditional embeddings ∅, latent z,   
autoencoder E and D   
Output: Restored video $\{ I _ { i } ^ { \prime } \} _ { i = 0 } ^ { N - 1 }$   
$= \mathcal { E } ( I )$   
2: for $t = 0$ to $T ^ { \prime } - 1$ do   
3: for $i = 0$ to $N - 1$ do   
4: OptimizeInversion $\tilde { \boldsymbol { z } } _ { t } , t , [ \mathcal { C } _ { t } ^ { i - 1 } , \mathcal { C } _ { t } ^ { i } ] , \boldsymbol { \emptyset } )$   
5: end for   
6: end for   
7: for $t = 0$ to $T ^ { \prime } - 1$ do   
8: for $i = 0$ to $N - 1$ do   
9: OptimizeInversion $( \overline { { z } } _ { t } , t , \mathcal { C } , [ \mathcal { O } _ { t } ^ { i - 1 } , \mathcal { O } _ { t } ^ { i } ] )$   
10: end for   
11: end for   
12: for $t = T - 1$ to 0 do   
13: for $i = 0$ to $N - 1$ do   
14: for $k = 1$ to K do   
15: OptimizeSampling( $z _ { t } , y , [ \mathcal { C } _ { t } ^ { i - 1 } , \mathcal { C } _ { t } ^ { i } ] )$   
16: end for   
17: for $k = 1$ to K do   
18: OptimizeSampling( $z _ { t } , y , \mathcal { C } _ { t } ^ { * } , [ \mathcal { O } _ { t } ^ { i - 1 } , \mathcal { O } _ { t } ^ { i } ] )$   
19: end for   
20: OptimizeSampling $( z _ { t } , y , \mathcal { C } _ { t } ^ { * } , \mathcal { O } _ { t } ^ { * } )$   
21: $z _ { t } ^ { i } =$ TextureAwareVideoTokenMerging $( z _ { t } ^ { i - 1 } , z _ { t } ^ { i } )$   
22: end for   
23: end for   
24: $I ^ { \prime } = \bar { \mathcal { D } } ( z _ { 0 } ^ { s a m p l i n g } )$

Table 1: Quantitative comparison with state-of-the-art methods for video deblurring. The best results are highlighted in bold.
<table><tr><td>Methods</td><td>Backbone</td><td>PSNR↑</td><td>SSIM↑</td><td> $\mathrm { W E } ( 1 0 ^ { - 2 } ) \downarrow$ </td></tr><tr><td>PSLD</td><td>SDv1.5</td><td>19.33</td><td>0.4755</td><td>5.4533</td></tr><tr><td>ZVRM (no Ref.)</td><td>SDv1.5</td><td>20.07</td><td>0.4852</td><td>0.8183</td></tr><tr><td>ZVRM (text Ref.)</td><td>SDv1.5</td><td>20.21</td><td>0.4831</td><td>0.7697</td></tr><tr><td>ZVRM (text Ref.+image Ref. 1)</td><td>SDv1.5</td><td>20.45</td><td>0.4910</td><td>0.7009</td></tr></table>

## 3 Comparison with State-of-the-art Methods

Table 1 lists the quantitative results on the evaluation data for zero-shot video deblurring. It can be observed that our method (ZVRM, no Ref.) outperforms PSLD by 0.74 dB in terms of PSNR, and has nearly only 1/8 of the WE value. Text and image references can further improve the performance.

Fig. 1 and 2 present the visual comparison results on the evaluation data for zero-shot video superresolution and deblurring. For video super-resolution, we change the stable diffusion backbone from SDv1.5 to SDXL, and compare our method with VISION-XL. It can be observed that there are unpleasant artifacts in the edges of VISION-XL results. Our method (ZVRM, no Ref.) can restore clearer edges. For video deblurring, it can be observed that the people in the PSLD results have different body parts on the two frames. Our method (ZVRM, no Ref.) has better temporal consistency for the people’s bodies and waves.

Fig. 3 shows more results when using different kinds of reference images. The Ref. 2 image in Fig. 3 is retrieved by Google’s image retrieval system.

Our method can also be applied to blind restoration tasks with complex real-world degradation. Fig. 4 gives the visual quality comparison results, and our method can restore sharp and temporal-consistent details. We also present a video demo to present the temporal consistency of our method (ZVRM, no Ref.).

![](images/9081dac0c553ba21835e603c0d61bb242bcc822f2e75393b427195d37b074c9e.jpg)  
Figure 1: Visual quality comparison for video super-resolution using an SDXL backbone. Zoom in for better observation.

## 4 Ablation Study

We provide more ablation studies on the modules of Dual Prompt Tuning Inversion (DPTI) and Dual Prompt Tuning Sampling (DPTS). Table 2 lists the quantitative comparison results in evaluation data by adding the modules one by one. Optimizing conditional and unconditional embeddings both brings gains.

## 5 Limitations

We would like to point out that our method also has some limitations. Taking the backbone PSLD as an example, although our method has reduced the total sampling steps from 1000 to 360, it still takes 7 minutes and 6 seconds to generate an image on an A40 GPU. We will further accelerate the sampling in future work.

![](images/f8cf4b0244428e885056aae7740ad05f77cd731994a192978ae235c25576bb06.jpg)  
Figure 2: Visual quality comparison for video deblurring. Zoom in for better observation.

Table 2: Ablation study for dual prompt tuning inversion, dual prompt tuning sampling, spatialtemporal self-attention and temporal latent fusion on 4× video super-resolution task. Cond. and Uncond. respectively represent the optimization of conditional and unconditional embedding.
<table><tr><td rowspan="2">DPTI</td><td></td><td>X</td><td>√</td><td> $\checkmark$ </td><td>√</td><td> $\checkmark$ </td></tr><tr><td> $\frac { \mathrm { C o n d . } } { \mathrm { U n c o n d . } }$ </td><td>×</td><td>X</td><td>√</td><td>√</td><td> $\widecheck { \widecheck { \widecheck { \mathbf { v } } } }$ </td></tr><tr><td rowspan="2">DPTS</td><td>Cond.</td><td>X</td><td>X</td><td>×</td><td>√</td><td>√</td></tr><tr><td>Uncond.</td><td>X</td><td>X</td><td>×</td><td>X</td><td>√</td></tr><tr><td>PSNR↑</td><td rowspan="4"></td><td>24.85</td><td>24.87</td><td>24.88</td><td>24.96</td><td>25.03</td></tr><tr><td>SSIM↑</td><td>0.6313</td><td>0.6457</td><td>0.6501</td><td>0.6552</td><td>0.6585</td></tr><tr><td> $\mathrm { W E ( 1 0 ^ { - 2 } ) \downarrow }$ </td><td>2.0854</td><td>2.0197</td><td>1.9646</td><td>1.3880</td><td>1.0721</td></tr><tr><td>Steps↓</td><td>1000</td><td>310</td><td>310</td><td>335</td><td>360</td></tr></table>

Ref. Image

![](images/df87e903f2977eb94dd56d3707e7703287bf751dd48bb3a1654d774c443d6a48.jpg)

![](images/11a986673f6bda7bd6a6a78367831bb8281d3f6f43e4c9bb032dc995ff27381d.jpg)

![](images/014d832f7cc18649e2254b33a912e8e1d48542ed4d9791de54d8326e98ebf492.jpg)

![](images/8f47c43f5909c39feeb78ea543f5ea707011b656edd016775fc16e60750de3ea.jpg)  
ZVRM

![](images/6eeb3109b7bc1bca560251e25bbc9228c53ade3c275023e6b8cf1f8fc09075de.jpg)  
ZVRM

![](images/d7bcefc22d50336af95f959992bbd1d8f2fe0975d54f870dcc2e38e1d35dceb3.jpg)

![](images/fd6f0573e6c0e4fabcae5020dc3cd2e8f3beb8b199e66c1996c98d787fe12e11.jpg)  
w/o Ref. Image w/ Ref. Image1 w/ Ref. Image2  
GT  
Figure 3: Different kinds of reference images for zero-shot video super-resolution.

## 6 Societal Impact

Video restoration and enhancement are widely used in video surveillance, video photography, etc. However, training a latent diffusion model for video restoration and enhancement requires substantial resources (several high-memory and fast GPUs). Therefore, in this paper, we propose a training-free method that utilizes a pre-trained text-to-image latent diffusion model. Our work can reduce the cost of experiments, accelerate research on video restoration and enhancement, and better serve relevant areas. At the same time, this kind of training-free video restoration and enhancement may bring more challenges in defending against attacks. We will further improve our method for deployment in critical security-sensitive systems.

## References

[1] Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, and Sanjay Shakkottai. Solving linear inverse problems provably via posterior sampling with latent diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

[2] Ben Fei, Zhaoyang Lyu, Liang Pan, Junzhe Zhang, Weidong Yang, Tianyue Luo, Bo Zhang, and Bo Dai. Generative diffusion prior for unified image restoration and enhancement. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9935–9946, 2023.

![](images/57d0e821594f3dcd7436d9084436aefa931bdc96aa479c4ff31a122326e632bf.jpg)  
Figure 4: Visual quality comparison for blind video super-resolution. Zoom in for better observation..