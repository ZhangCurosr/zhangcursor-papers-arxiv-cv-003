# PIXELMOE: SCALING PIXEL-SPACE DIFFUSION WITH LLM-STYLE MIXTURE-OF-EXPERTS

Ruixiao Dong<sup>1,2∗</sup> Jianmin Bao<sup>2†</sup> Bozhi Luan<sup>1,2∗</sup> Zijie Meng<sup>2,3∗</sup> Xin Tao<sup>2</sup> Pengfei Wan<sup>2</sup> Kun Gai<sup>2</sup> Li Li<sup>1†</sup> Houqiang Li<sup>1</sup>

<sup>1</sup> University of Science and Technology of China

<sup>2</sup> KlingAI Research

<sup>3</sup> Peking University

dongruixiaoyx@mail.ustc.edu.cn baojianmin.bjm@kuaishou.com {lil1,lihq}@ustc.edu.cn

## ABSTRACT

Pixel-space diffusion models provide an appealing alternative to latent-space diffusion models by directly modeling raw images without relying on a learned tokenizer. However, scaling pixel-space models has predominantly followed dense strategies, tightly coupling model capacity with active computation. In this work, we show that standard LLM-style Mixture-of-Experts (MoE) architectures can be effectively transferred to pixel-space diffusion. Building on this finding, we introduce PixelMoE, a pixel-space generative framework that sparsely scales model capacity with standard LLM-style MoE, without diffusion-specific routing mechanisms or auxiliary routing objectives. Specifically, PixelMoE combines standard Token-Choice (TC) routing with fine-grained routed experts for input-adaptive sparse computation, a shared expert for capturing common representations, and a bias-based load-balancing strategy. To complement sparse capacity scaling, PixelMoE further introduces Pixel-Space Hierarchical Flow Matching, which varies spatial resolution along the denoising trajectory from global structure formation to fine-grained detail refinement. At comparable active parameter counts, Pixel-MoE consistently outperforms its dense counterparts across model scales, demonstrating effective sparse capacity scaling in pixel space. On class-conditional ImageNet-1K, PixelMoE achieves FID scores of 1.47 at $2 5 6 \times 2 5 6$ and 1.53 at 512 × 512 without representation alignment or perceptual losses, comparing favorably with leading pixel-space and latent-space generative models. Code and models will be released soon.

![](images/88dcbd05178db8f2687d40be5a5fc43e450b8804719f80b9175659cd98aa5a7f.jpg)  
Figure 1: Progress in pixel-space image generation on ImageNet-1K at $2 5 6 \times 2 5 6$ . PixelMoE sparsely scales model capacity with MoE and compares favorably with leading approaches.

![](images/cb42292188f3f15205042e68581183fdc220babfafdf8274670eaf9308b23a74.jpg)  
(a) Dense vs. PixelMoE (100 Epochs)

![](images/98a7ad570051eb3e3929b5f9211a2bd971a2e7bc91827aa5a84f16523bab9c3f.jpg)  
(b) Pixel-Space Method Comparison  
Figure 2: ImageNet-1K generation with PixelMoE. Left: Dense vs. PixelMoE after 100 training epochs, where PixelMoE consistently achieves lower FID at comparable active parameter counts. PixelMoE-B/L/H abbreviate PixelMoE-B/L/H-E32A4, respectively, with 32 routed experts and top-4 activation. Right: Comparison with representative pixel-space methods. Marker size denotes active parameter counts.

## 1 INTRODUCTION

Recent advances in diffusion models, particularly Diffusion Transformers (DiTs) (Peebles & Xie, 2023; Ma et al., 2024) trained with flow matching objectives (Lipman et al., 2023; Liu et al., 2023), have substantially improved the quality and scalability of image generation. Most modern diffusion models (Rombach et al., 2022) denoise in latent spaces extracted by variational autoencoders (VAEs) (Kingma & Welling, 2013), whose lossy compression may discard high-frequency details and introduce structural artifacts, leading to a reconstruction bottleneck that limits achievable image fidelity. Recently, pixel-space diffusion models (Dhariwal & Nichol, 2021; Li & He, 2026) have regained attention by directly modeling raw images rather than learned latent representations. Advances in clean-image prediction (Li & He, 2026), architectural design (He et al., 2026; Yu et al., 2026), and training objectives (Ma et al., 2026b) have made raw-pixel modeling easier, enabling pixel-space models to achieve performance competitive with leading latent-space approaches.

Despite this progress, sparsely scaling pixel-space diffusion remains largely unexplored. Existing pixel-space models primarily scale their generators densely by increasing model width and depth, tightly coupling capacity growth with active computation. In contrast, Mixture-of-Experts (MoE) architectures (Shazeer et al., 2017) expand total capacity through sparse expert activation while controlling active computation, and have become a standard scaling paradigm in Large Language Models (LLMs) (Liu et al., 2024). However, directly transferring standard LLM-style MoE to dif fusion has not consistently yielded clear gains over dense counterparts (Fei et al., 2024; Liu et al., 2025). Subsequent approaches therefore adopt alternative routing paradigms such as Expert-Choice (EC) (Sun et al., 2025; Shi et al., 2025), or introduce diffusion-specific routing guidance and objectives (Wei et al., 2026). This naturally raises the question: Can standard LLM-style MoE be directly transferred to pixel-space diffusion?

We answer this question affirmatively. We propose PixelMoE, a Pixel-space Mixture-of-Experts generative framework that sparsely scales pixel-space diffusion transformers without diffusionspecific routing mechanisms or auxiliary routing objectives. Specifically, PixelMoE adopts standard Token-Choice (TC) routing, where each image token dynamically selects its top-A experts based on the routing scores. Each MoE layer combines fine-grained routed experts for input-adaptive sparse computation with a shared expert that captures common representations across image tokens. To maintain balanced expert utilization without auxiliary balancing objectives, we further adopt a biasbased load-balancing strategy. Beyond sparse capacity scaling, pixel-space representations can be directly downsampled and upsampled across resolutions while preserving meaningful visual structure. We exploit this property with Pixel-Space Hierarchical Flow Matching, organizing generation into a low-resolution coarse stage followed by full-resolution refinement.

Extensive experiments demonstrate strong sparse scaling behavior of PixelMoE. Under comparable active parameter counts, PixelMoE outperforms its dense counterparts across model scales. Furthermore, increasing the number of routed experts consistently improves generation quality. Standard Token-Choice routing also substantially outperforms Expert-Choice routing, without diffusionspecific routing guidance. Moreover, our routing analysis reveals clear expert specialization across semantic classes and along the denoising trajectory without explicit semantic or timestep guidance. Together, these results suggest that effective sparse scaling does not require diffusion-specific routing: with appropriately structured sparse capacity, standard Token-Choice routing can effectively exploit the rich semantic and denoising information already encoded in pixel-space representations. On ImageNet-1K (Deng et al., 2009), PixelMoE achieves FID scores of 1.47 at 256 × 256 and 1.53 at $5 1 2 \times 5 1 2$ without representation alignment (Yu et al., 2025; Shin et al., 2026) or perceptual losses (Ma et al., 2026b), comparing favorably with leading pixel-space and latent-space generative models. These results demonstrate that LLM-style sparse scaling transfers effectively to pixel-space diffusion.

## 2 RELATED WORK

Pixel-Space Generative Models. Early generative models, including GANs (Brock et al., 2019; Casanova et al., 2021) and diffusion models (Dhariwal & Nichol, 2021; Ho et al., 2022), synthesize images directly in pixel space. Latent-space models (Rombach et al., 2022; Esser et al., 2021) later became dominant by modeling compact representations extracted by variational autoencoders (VAEs) (Kingma & Welling, 2013), whose lossy compression introduces a reconstruction bottleneck that limits the achievable generation fidelity. This limitation has renewed interest in pixelspace generative models, which directly model raw images without lossy compression. Simple Diffusion (Hoogeboom et al., 2023) demonstrated competitive high-resolution pixel-space generation through architectural scaling, noise schedules, and spatial downsampling, while VDM++ (Kingma & Gao, 2023) studied diffusion-objective weighting and noise augmentation. JiT (Li & He, 2026) further showed that a large-patch Transformer can effectively model raw pixels through clean-image prediction. Subsequent approaches improve pixel-space generation through multi-resolution generation (Chen et al., 2025), neural-field decoding (Wang et al., 2026a), hierarchical architectures (Yu et al., 2026; He et al., 2026; Guo et al., 2026), frequency-decoupled modeling (Ma et al., 2026a), representation alignment (Shin et al., 2026), perceptual supervision (Ma et al., 2026b), and alternative flow parameterizations (Chen et al., 2026a). Despite these advances, most existing pixel-space models remain dense, coupling increased model capacity with increased active computation, while sparse MoE scaling in pixel-space diffusion remains underexplored.

Mixture-of-Experts in Visual Generation. The Mixture-of-Experts (MoE) paradigm was introduced to enable specialization among multiple expert networks (Jacobs et al., 1991) and later extended with sparse gating (Shazeer et al., 2017). Modern MoE architectures (Fedus et al., 2022; Dai et al., 2024) have become a key scaling mechanism for Large Language Models (LLMs) (Yang et al., 2025; Liu et al., 2024), motivating their adoption in visual generation. Early expert-based diffusion models, largely built on U-Net backbones, partition the denoising trajectory by noise level, denoising task, or generation path and assign specialized parameters to different regimes (Balaji et al., 2022; Feng et al., 2023; Xue et al., 2023; Lee et al., 2024; Ganjdanesh et al., 2024). As diffusion backbones shift toward Diffusion Transformers (DiTs) (Peebles & Xie, 2023), recent work instead explores sparse expert routing within Transformer blocks. However, directly applying conventional Token-Choice routing has not consistently yielded clear gains over dense models (Fei et al., 2024). Subsequent approaches have therefore introduced Expert-Choice routing (Sun et al., 2025), diffusion-aware routing or dynamic capacity allocation (Cheng et al., 2025; Shi et al., 2025), and specialized token-expert assignment with auxiliary routing objectives (Yuan et al., 2025; Wei et al., 2026). However, most existing sparse DiT approaches focus on latent-space diffusion. JiT-MoE (Liu et al., 2025) has recently extended MoE to pixel-space diffusion but reports only modest gains over its dense JiT counterparts. We therefore investigate whether standard LLM-style sparse MoE can provide systematic scaling gains in pixel-space generation.

## 3 PIXELMOE: PIXEL-SPACE DIFFUSION WITH MIXTURE-OF-EXPERTS

PixelMoE is a pixel-space generative framework designed to sparsely scale model capacity with standard LLM-style Mixture-of-Experts (MoE), while employing Pixel-Space Hierarchical Flow

![](images/f6054b7ee63c2df00758a2bca767c57f1bfd63e0e60324d281eb400c7ab1d29f.jpg)  
Figure 3: Overview of PixelMoE. Left: PixelMoE performs Pixel-Space Hierarchical Flow Matching using a shared PixelMoE backbone and DDT prediction head; the initial dense blocks are omitted for clarity. Right: Each PixelMoE block consists of self-attention followed by an LLM-style sparse MoE layer with Token-Choice routed experts and a shared expert. A balancing bias is employed to maintain balanced expert utilization. RMSNorm and timestep/class conditioning are omitted.

Matching for coarse-to-fine generation. As illustrated in the left panel of Figure 3, generation proceeds through two stages at different spatial resolutions. Stage 0 performs partial denoising at low resolution to establish global structure, while Stage 1 continues denoising at full resolution for detail refinement. The two stages share the PixelMoE backbone and Decoupled Diffusion Transformer (DDT) (Wang et al., 2026b) prediction head. The backbone consists of initial dense blocks followed by sparse PixelMoE blocks, while the DDT head performs stage-wise endpoint prediction.

As shown in the right panel of Figure 3, each PixelMoE block consists of a self-attention layer followed by an LLM-style sparse MoE layer. For each image token, standard Token-Choice routing selects a subset of fine-grained routed experts, while a shared expert is always activated to capture common representations across image tokens. To maintain balanced expert utilization without an auxiliary load-balancing objective, we further introduce a balancing bias during routing. Importantly, the router operates directly on pixel-token representations without explicit semantic or timestep routing guidance. We detail the sparse MoE design and the hierarchical flow formulation in the following subsections.

## 3.1 LLM-STYLE MIXTURE-OF-EXPERTS FOR PIXEL-SPACE DIFFUSION

The core design of PixelMoE is to directly transfer the standard LLM-style MoE paradigm to pixelspace diffusion. Following modern LLM-style MoE architectures (Liu et al., 2024; Kimi Team et al., 2026), we retain dense feed-forward networks (FFNs) in the early Transformer blocks and replace the FFNs in the remaining blocks with sparse MoE layers. Each MoE layer adopts standard Token-Choice routing over pixel-token representations, combines fine-grained routed experts with a shared expert, and employs a bias-based load-balancing strategy without an auxiliary objective.

Token-Choice Routing. For each image token, Token-Choice (TC) routing first computes its affinity to all routed experts. Let $\mathbf { h } _ { t , i } ^ { s } \in \mathbb { R } ^ { d }$ denote the hidden state of image token i at global timestep t in stage s, and let E denote the number of routed experts. We use a bias-free linear router with weight $\mathbf { \breve { W } } _ { r } \in \mathbb { R } ^ { E \times d }$ to produce sigmoid routing scores:

$$
\mathbf { r } _ { t , i } ^ { s } = \mathrm { s i g m o i d } \left( \mathbf { W } _ { r } \mathbf { h } _ { t , i } ^ { s } \right) ,\tag{1}
$$

where $r _ { t , i , e } ^ { s }$ denotes the routing score of token i for the e-th routed expert. To regulate expert utilization, each routed expert is associated with a balancing bias $b _ { e }$ , which is added to the routing

score only during expert selection. The set of top-A activated experts is

$$
\begin{array} { r } { S _ { t , i } ^ { s } = \mathrm { T o p A } _ { e } \left( r _ { t , i , e } ^ { s } + b _ { e } \right) , } \end{array}\tag{2}
$$

where A denotes the number of routed experts activated for each image token. After selection, the balancing bias is no longer used and the routing scores are normalized over the selected experts:

$$
g _ { t , i , e } ^ { s } = \frac { r _ { t , i , e } ^ { s } } { \sum _ { j \in S _ { t , i } ^ { s } } r _ { t , i , j } ^ { s } } , \qquad e \in S _ { t , i } ^ { s } ,\tag{3}
$$

where $g _ { t , i , e } ^ { s }$ denotes the routing weight of expert e for token i. Notably, the router relies solely on image-token hidden states, without explicit semantic or timestep guidance.

Expert Aggregation. Given the selected experts $\boldsymbol { S } _ { t , i } ^ { s }$ and routing weights $\begin{array} { r } { g _ { t , i , e } ^ { s } , } \end{array}$ PixelMoE aggregates the routed-expert outputs with the output of an always-active shared expert. Let $\mathcal { E } _ { e } ( \cdot )$ denote the e-th routed expert and $\bar { \mathcal { E } } _ { \mathrm { s h } } ( \cdot )$ the shared expert. Only the experts in $\boldsymbol { S } _ { t , i } ^ { s }$ process the image-token hidden state $\mathbf { h } _ { t , i } ^ { s }$ in the routed branch, while the shared expert is applied to every image token. Their outputs are combined as

$$
\mathbf { y } _ { t , i } ^ { s } = \lambda _ { r } \sum _ { e \in S _ { t , i } ^ { s } } { g _ { t , i , e } ^ { s } \mathcal { E } _ { e } \left( \mathbf { h } _ { t , i } ^ { s } \right) } + \mathcal { E } _ { \mathrm { s h } } \left( \mathbf { h } _ { t , i } ^ { s } \right) ,\tag{4}
$$

where $\lambda _ { r } ~ = ~ { \sqrt { A } }$ denotes the routing scale. Both routed and shared experts are implemented as FFNs with reduced intermediate dimensions, such that their combined active parameter count is comparable to that of the dense baseline.

Bias-Based Load Balancing. To maintain balanced expert utilization without introducing an auxiliary load-balancing objective, we dynamically update the expert-wise balancing biases during training. We collect the balancing biases for the routed experts into $\mathbf { b } = ( b _ { 1 } , \ldots , b _ { E } ) \in \mathbb { R } ^ { E }$ The expert-load distribution in the current training batch is $\textbf { F } \in \ \mathbb { R } ^ { E }$ , obtained by averaging image-token assignments, with each selected expert contributing $1 / A$ , while the uniform target is $\mathbf { Q } = ( 1 / E , \dots , \bar { 1 } / E )$ . We update the balancing biases as

$$
{ \bf b }  { \bf b } - \gamma \frac { { \bf F } - { \bf Q } } { \mathrm { R M S } ( { \bf F } - { \bf Q } ) } ,\tag{5}
$$

where $\gamma ~ = ~ 0 . 0 0 1$ controls the update magnitude. The detailed computation of $\mathbf { F }$ and the RMS normalization is provided in Section A.

## 3.2 PIXEL-SPACE HIERARCHICAL FLOW MATCHING

Pixel space naturally supports coarse-to-fine generation, as spatial resampling directly produces meaningful multi-resolution images. We therefore vary the spatial resolution along the denoising trajectory, using coarse representations during early denoising and full-resolution representations for later refinement. This hierarchical trajectory also requires the model to process heterogeneous pixel-token representations across resolutions and noise levels, which are naturally accommodated by the input-adaptive routing of PixelMoE. Specifically, Pixel-Space Hierarchical Flow Matching organizes generation into two stages: Stage 0 performs partial denoising at low resolution for coarse structure modeling, while Stage 1 continues at full resolution for detail refinement. As illustrated in Figure 4, both stages share the same PixelMoE model while denoising at different spatial resolutions and noise levels.

Hierarchical Flow Trajectory. Given a clean image $x _ { 0 } .$ , Stage 0 performs partial denoising at low resolution, starting from Gaussian noise $x ^ { s _ { 0 } }$ and terminating at a noisy coarse endpoint $x ^ { e _ { 0 } }$ :

$$
\begin{array} { l } { { x ^ { s _ { 0 } } = \epsilon _ { 0 } , \qquad \epsilon _ { 0 } \sim \mathcal { N } ( 0 , \delta _ { n } ^ { 2 } I ) , } } \\ { { x ^ { e _ { 0 } } = ( 1 - t _ { e _ { 0 } } ) \mathcal { D } ( x _ { 0 } ) + t _ { e _ { 0 } } \epsilon _ { 0 } , } } \end{array}\tag{6}
$$

where $\mathcal { D } ( \cdot )$ denotes $2 \times$ bilinear downsampling and $t _ { e _ { 0 } }$ specifies the Stage 0 endpoint. We use $\delta _ { n } = 1$ for 256 × 256 generation and $\delta _ { n } = 2$ for 512 × 512 generation.

![](images/0f082089fc74dafd94abdff12c3857b7e2babdcf9ebe5cf473fbf29a699d5295.jpg)  
Figure 4: Pixel-Space Hierarchical Flow Matching in PixelMoE. PixelMoE is jointly trained on a low-resolution coarse stage and a full-resolution refinement stage with stage-wise endpoint prediction. During sampling, generation first follows the low-resolution Stage 0 flow to obtain a coarse endpoint, which is then transitioned to the Stage 1 starting state for full-resolution refinement.

Stage 1 then continues denoising at full resolution from a noisy coarse state:

$$
\begin{array} { l l } { { x ^ { s _ { 1 } } = ( 1 - t _ { s _ { 1 } } ) \widetilde { x } _ { 0 } + t _ { s _ { 1 } } \epsilon _ { 1 } , \qquad \epsilon _ { 1 } \sim \mathcal { N } ( 0 , \delta _ { n } ^ { 2 } I ) , } } \\ { { x ^ { e _ { 1 } } = x _ { 0 } , } } \end{array}\tag{7}
$$

where $\widetilde { x } _ { 0 } = \mathcal { U } ( { \cal D } ( x _ { 0 } ) )$ is the coarse full-resolution image obtained by downsampling and upsampling $x _ { 0 } , \mathcal { U } ( \cdot )$ denotes 2× nearest-neighbor upsampling, and $t _ { s _ { 1 } }$ specifies the Stage 1 starting noise level. Accordingly, $t _ { s _ { 0 } } = 1$ and $t _ { e _ { 1 } } = 0$ correspond to the pure-noise start of Stage 0 and the cleanimage endpoint of Stage 1, respectively. Within each stage $s \in \{ 0 , 1 \}$ , we introduce a stage-relative timestep $\sigma \in [ 0 , 1 ]$ and linearly interpolate between the stage start and endpoint:

$$
\begin{array} { r } { x _ { \sigma } ^ { s } = ( 1 - \sigma ) x ^ { e _ { s } } + \sigma x ^ { s _ { s } } , } \end{array}\tag{8}
$$

where $x ^ { s _ { s } }$ and $x ^ { e _ { s } }$ denote the start and endpoint of stage s, respectively; $e . g . , x ^ { s _ { 0 } }$ and $x ^ { e _ { 0 } }$ for Stage 0. The corresponding target velocity is

$$
v ^ { s } = \frac { \partial x _ { \sigma } ^ { s } } { \partial \sigma } = x ^ { s _ { s } } - x ^ { e _ { s } } .\tag{9}
$$

Stage-Wise Endpoint Prediction. Inspired by the clean-data prediction in JiT (Li & He, 2026), we predict the endpoint of each stage. Specifically, given $x _ { \sigma } ^ { s }$ , PixelMoE predicts the stage endpoint $\hat { x } _ { \theta } ^ { e _ { s } } ( x _ { \sigma } ^ { s } , t , c )$ , where $t = t _ { e _ { s } } + \sigma ( t _ { s _ { s } } - t _ { e _ { s } } )$ is the corresponding global timestep used for conditioning the model and c denotes the class condition. The predicted endpoint induces the velocity

$$
v _ { \theta } ( x _ { \sigma } ^ { s } , t , c ) = \frac { ( x _ { \sigma } ^ { s } - \hat { x } _ { \theta } ^ { e _ { s } } ( x _ { \sigma } ^ { s } , t , c ) ) } { \sigma } .\tag{10}
$$

Substituting this parameterization into the velocity-matching objective gives

$$
\mathcal { L } _ { \mathrm { v - e n d p o i n t } } = \mathbb { E } \left[ \mathcal { L } _ { \mathrm { M S E } } ( v _ { \theta } ( x _ { \sigma } ^ { s } , t , c ) , v ^ { s } ) \right] = \mathbb { E } \left[ \frac { 1 } { \sigma ^ { 2 } } \mathcal { L } _ { \mathrm { M S E } } ( \hat { x } _ { \theta } ^ { e _ { s } } ( x _ { \sigma } ^ { s } , t , c ) , x ^ { e _ { s } } ) \right] ,\tag{11}
$$

where $\mathcal { L } _ { \mathrm { M S E } }$ denotes the mean squared error. For numerical stability, we clamp σ to max $( \sigma , \sigma _ { \mathrm { m i n } } )$ in the endpoint-prediction loss during training and in the velocity computation during sampling, with $\sigma _ { \mathrm { m i n } } = 0 . 0 5$ . This velocity-space endpoint objective is the only generation loss used to train the default PixelMoE models; we do not employ representation alignment or perceptual losses.

<table><tr><td>Method</td><td>#Params</td><td>#Active</td><td>FID↓</td></tr><tr><td>Dense-B</td><td>223M</td><td>223M</td><td>4.40</td></tr><tr><td>PixelMoE-B-E32A4</td><td>461M</td><td>225M</td><td>2.46</td></tr><tr><td>Dense-L</td><td>622M</td><td>622M</td><td>2.95</td></tr><tr><td>PixelMoE-L-E32A4</td><td>1.65B</td><td>632M</td><td>1.78</td></tr><tr><td>Dense-H</td><td>1.44B</td><td>1.44B</td><td>2.35</td></tr><tr><td>PixelMoE-H-E32A4</td><td>3.68B</td><td>1.47B</td><td>1.61</td></tr></table>

Table 1: Active parameter scaling: dense vs. MoE.

![](images/1f4d47eba1600ad0809ea34056668231c74f8e8ecd55bb26ccc2ee41bc9f7883.jpg)  
Figure 5: Convergence comparison between Dense-L and PixelMoE-L-E32A4.

Coarse-to-Fine Sampling. During sampling, generation proceeds sequentially from Stage 0 to Stage 1. Starting from low-resolution Gaussian noise, PixelMoE first follows the Stage 0 flow to obtain the coarse endpoint $\hat { x } _ { \theta } ^ { e _ { 0 } }$ . The upsampled Stage 0 endpoint cannot be directly used as the Stage 1 starting state $\bar { \hat { x } } _ { \theta } ^ { s _ { 1 } }$ , since both its signal coefficient and noise covariance differ from those of the predefined Stage 1 starting distribution. We therefore rescale the upsampled endpoint and inject an additional Gaussian perturbation:

$$
\hat { x } _ { \theta } ^ { s _ { 1 } } = \frac { 1 - t _ { s _ { 1 } } } { 1 - t _ { e _ { 0 } } } \mathcal { U } ( \hat { x } _ { \theta } ^ { e _ { 0 } } ) + \alpha \epsilon ^ { \prime } ,\tag{12}
$$

where the scaling factor aligns the signal component with the Stage 1 starting state, while $\epsilon ^ { \prime }$ follows a block-correlated Gaussian distribution and α is chosen to match the target covariance. The resulting state is constructed to match the predefined Stage 1 statistics while preserving the coarse structure generated in Stage 0. PixelMoE then follows the Stage 1 flow toward the clean full-resolution endpoint. The transition coefficients and block-correlated noise construction are derived in Section B.

Stage-Aware Timestep Sampling. The two stages denoise at different spatial resolutions and cover different portions of the denoising trajectory. We therefore adopt a stage-aware timestep sampling strategy that separately controls stage allocation and stage-relative timestep sampling. Stage 0 covers the global timestep interval $[ t _ { e _ { 0 } } , 1 ]$ , while Stage 1 covers $[ 0 , t _ { s _ { 1 } } ]$ . Specifically, a global LogitNorma $\mathrm { l } ( \mu _ { \mathrm { g } } , \delta _ { \mathrm { g } } )$ distribution determines the sampling probability of each stage, with the probability mass in the overlapping region divided equally between them. After sampling a stage s, we independently sample the stage-relative timestep σ ∼ LogitNorma $1 ( \mu _ { 1 } , \delta _ { 1 } )$ ) and map it to the corresponding global timestep as

$$
t = t _ { e _ { s } } + \sigma ( t _ { s _ { s } } - t _ { e _ { s } } ) .\tag{13}
$$

This formulation decouples stage allocation from stage-relative timestep sampling, allowing the timestep distribution within each stage to be controlled independently of how frequently that stage is selected.

## 4 EXPERIMENTS

We conduct all experiments on class-conditional ImageNet-1K generation (Deng et al., 2009).

Training Details. During training, we apply random horizontal flipping with probability 0.5. Models for comparison with previous methods are trained for 320 epochs, while other experiments use 100 epochs unless otherwise specified. For PixelMoE-H-E32A4, we apply additional regularization after epoch 200. We use a learning rate of $2 \times 1 0 ^ { - 4 }$ with a 5-epoch linear warm-up, followed by a constant schedule with cosine decay over the final 40 epochs. For optimization, we apply Gram Newton-Schulz Muon (gMuon) (Jordan et al., 2024; Zhang et al., 2026) with momentum 0.95 to eligible matrix-valued parameters, while the remaining parameters are optimized with AdamW (Loshchilov & Hutter, 2019) using $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ . We employ stage-aware Logit-Normal timestep sampling with $( \mu _ { \mathrm { g } } , \delta _ { \mathrm { g } } ) = ( 0 . 8 , 0 . 8 )$ for stage allocation and $( \mu _ { 1 } , \bar { \delta } _ { 1 } ) = ( 0 . 6 , \bar { 0 . 8 } )$ for stage-relative timestep sampling. Two exponential moving average (EMA) copies with decay rates 0.9996 and 0.9999 are maintained during training. Detailed training configurations are provided in Section E.

<table><tr><td>Method</td><td>#Params</td><td>#Active</td><td>FID↓</td></tr><tr><td>PixelMoE-L-E8A4</td><td>892M</td><td>632M</td><td>2.28</td></tr><tr><td>PixelMoE-L-E16A4</td><td>1.14B</td><td>632M</td><td>2.03</td></tr><tr><td>PixelMoE-L-E32A4</td><td>1.65B</td><td>632M</td><td>1.78</td></tr></table>

(a) Effect of scaling the number of experts.
<table><tr><td>Method</td><td>#PS</td><td>#Params</td><td>#Active</td><td>FID↓</td></tr><tr><td>Dense-L/16</td><td>16</td><td>625M</td><td>625M</td><td>3.15</td></tr><tr><td>PixelMoE-L-E32A4/16</td><td>16</td><td>1.65B</td><td>634M</td><td>2.89</td></tr><tr><td>Dense-L/8</td><td>8</td><td>622M</td><td>622M</td><td>2.95</td></tr><tr><td>PixelMoE-L-E32A4/8</td><td>8</td><td>1.65B</td><td>632M</td><td>1.78</td></tr></table>

(b) Computation scaling via token granularity.

Table 2: Sparse scaling analysis of PixelMoE. #PS denotes the patch size.  
![](images/fe190800fed58d71cd7335337d77efb5062ca055397ec842c255f3bb3b9f6283.jpg)  
(a) Expert Routing Across Semantic Classes

![](images/e4209fb691961c7ab5be4c60759ecd668084dc6a9c672a5c60cb9bed56ea666b.jpg)  
(b) Expert Routing Along the Denoising Trajectory  
Figure 6: Expert routing across semantic classes and along the denoising trajectory. The left panel shows expert routing across ImageNet-1K classes, while the right panel shows how routing evolves along the denoising trajectory.

Evaluation Protocol. We evaluate generation quality using Frechet Inception Distance´ (FID) (Heusel et al., 2017), Inception Score (IS) (Salimans et al., 2016), Precision, and Recall (Kynka¨anniemi et al., 2019). We sample 50,000 images with labels uniformly distributed over¨ the ImageNet-1K classes. We use a 50-point global timestep grid with the Heun solver and apply classifier-free guidance (CFG) (Ho & Salimans, 2021) with a CFG interval (Kynka¨anniemi et al.,¨ 2024). Detailed evaluation settings are provided in Section F.

## 4.1 SPARSE SCALING ANALYSIS

We first investigate whether sparse MoE scaling can effectively improve pixel-space generation. We study the sparse scaling behavior of PixelMoE from three perspectives: active parameter scaling, expert scaling, and computation scaling via token granularity.

Active Parameter Scaling. We first compare PixelMoE with its dense counterparts across model scales. For controlled comparisons, each dense baseline uses the same hierarchical flow setup and training recipe as its PixelMoE counterpart, with sparse MoE layers replaced by dense FFNs at comparable active parameter counts. As shown in Table 1, PixelMoE consistently achieves lower FID than dense models across Base, Large, and Huge scales. Beyond generation quality, we compare the convergence of Dense-L and PixelMoE-L-E32A4, extending the training to 200 epochs for this analysis. As shown in Figure 5, PixelMoE-L-E32A4 reaches the FID achieved by Dense-L at 200 epochs after only about 71 epochs, reducing the required training epochs by a factor of approximately 2.8. These results show that sparse MoE scaling improves generation quality while also accelerating convergence at comparable active parameter counts.

Expert Scaling. We next isolate the effect of sparse model capacity by varying the number of routed experts E while fixing the number of activated experts to A = 4. As shown in Table 2a, increasing E from 8 to 16 and then 32 improves FID from 2.28 to 2.03 and 1.78, respectively, while keeping the active parameter count fixed at 632M. This shows that PixelMoE effectively benefits from increased total model capacity without increasing the number of active parameters.

Computation Scaling via Token Granularity. We further study how PixelMoE benefits from increased computation by reducing the patch size, which increases the token sequence length while keeping the backbone scale fixed. As shown in Table 2b, PixelMoE consistently outperforms the dense models at both patch sizes, with a larger improvement at finer token granularity. A possible explanation is that smaller patches produce more spatially localized token representations and provide more token-level routing decisions, enabling the router to exploit finer-grained spatial variation.

## 4.2 EXPERT SPECIALIZATION

We analyze the routing behavior of PixelMoE-L-E32A4 on the ImageNet-1K validation set. For selected MoE layers, we collect token-level routing assignments and examine how expert preferences vary across semantic classes and along the denoising trajectory. We report the $\log _ { 2 }$ ratio of the observed assignment frequency to that under uniform routing, where positive and negative values indicate preferred and suppressed experts, respectively.

As shown in Figure 6, PixelMoE exhibits distinct routing patterns across semantic classes and along the denoising trajectory. Class-dependent routing is relatively weak in shallow layers, becomes clearer in intermediate layers, and is most pronounced around Layers 12–20, where different ImageNet-1K classes favor different subsets of experts. In contrast, routing along the denoising trajectory is already prominent in shallow layers, weakens in intermediate layers, and emerges strongly again in the final layer. The routing pattern also changes noticeably across the transition between the two hierarchical stages. Overall, class-dependent routing specialization is more pronounced in intermediate layers, whereas specialization along the denoising trajectory is stronger in shallow and final layers.

Balanced Global Utilization. The specialized routing patterns above could potentially arise from persistent over-utilization of a small subset of experts. We therefore examine the global routed-expert loads across different MoE layers of PixelMoE-L-E32A4. For each layer, we accumulate expert assignments over the ImageNet-1K validation set and report the resulting expert-load distribution. As shown in Figure 7, the routed expert loads remain close to the uniform distribution across all analyzed layers. This indicates that the biasbased load-balancing strategy effectively maintains globally balanced expert utilization while preserving input-adaptive expert specialization. Together, these results show

![](images/fe049868ae5edb259457d2ea7596edf5166008bfa0f3dfcfba005bfeb8d56e7b.jpg)  
Figure 7: Global expert utilization across layers. The routed expert loads of PixelMoE-L-E32A4 remain close to a uniform distribution across the analyzed MoE layers. Expert loads are normalized by their expected values under uniform routing, with the dashed line indicating 1.

that the observed routing patterns across semantic classes and along the denoising trajectory do not simply reflect persistent over-utilization of a small subset of experts.

## 4.3 BENCHMARKING AGAINST PREVIOUS SYSTEMS

Table 3 compares PixelMoE with representative latent-space and pixel-space generative systems on class-conditional ImageNet-1K at $2 5 6 \times 2 5 6$ resolution. PixelMoE-B-E32A4 achieves an FID of 1.88 with only 225M active parameters. As total model capacity increases, PixelMoE-L-E32A4 further improves the FID to 1.54, while PixelMoE-H-E32A4 reaches 1.47. Notably, these results are achieved without representation alignment (Yu et al., 2025; Shin et al., 2026) or perceptual losses (Ma et al., 2026b). Compared with leading dense pixel-space models, PixelMoE achieves better generation quality. PixelMoE also remains competitive with leading latent-space approaches without relying on a learned VAE. Together with the controlled scaling results in Tables 1 and 2a, these comparisons support LLM-style sparse MoE as an effective scaling paradigm for pixel-space generative modeling.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Epochs</td><td rowspan="2">#Params</td><td rowspan="2">#Active Params</td><td colspan="4">ImageNet-1K Generation @ 2562</td></tr><tr><td>FID↓</td><td>IS↑</td><td>Precision↑</td><td>Recall↑</td></tr><tr><td colspan="8">Latent-space models: Dense</td></tr><tr><td>LDM-4-G (Rombach et al., 2022)</td><td>170</td><td>400M+33M400M+33M</td><td></td><td>3.60</td><td>247.6</td><td>0.87</td><td>0.48</td></tr><tr><td>DiT-XL (Peebles &amp; Xie, 2023)</td><td>1400</td><td>675M+49M675M+49M</td><td></td><td>2.27</td><td>278.2</td><td>0.83</td><td>0.57</td></tr><tr><td>SiT-XL (Ma et al., 2024)</td><td>1400</td><td>675M+49M675M+49M</td><td></td><td>2.06</td><td>270.3</td><td>0.82</td><td>0.59</td></tr><tr><td>MaskDiT (Zheng et al., 2024)</td><td>1600</td><td>675M+49M675M+49M</td><td></td><td>2.28</td><td>276.5</td><td>0.80</td><td>0.61</td></tr><tr><td>REPA (Yu et al., 2025)</td><td>800</td><td></td><td>675M+49M 675M+49M</td><td>1.42</td><td>305.7</td><td>0.80</td><td>0.65</td></tr><tr><td>SVG-XL (Shi et al., 2026)</td><td>1400</td><td>675M+43M675M+43M</td><td></td><td>1.92</td><td>264.9</td><td></td><td></td></tr><tr><td>FlowDCN-XL (Wang et al., 2024)</td><td>400</td><td>618M+49M618M+49M</td><td></td><td>2.00</td><td>263.2</td><td>0.82</td><td>0.58</td></tr><tr><td colspan="8">Latent-space models: Sparse MoE</td></tr><tr><td>DiT-MoE-XL (Fei et al., 2024)</td><td></td><td>4.1B+49M</td><td>1.5B+49M</td><td>1.72</td><td>315.7</td><td>0.83</td><td>0.64</td></tr><tr><td>RaceDiT-XL (Yuan et al., 2025)</td><td>1450</td><td>2.8B+49M</td><td>707M+49M</td><td>2.06</td><td>318.6</td><td>0.83</td><td>0.60</td></tr><tr><td>DiffMoE (Shi et al., 2025)</td><td>1400</td><td>1.2B+49M</td><td>458M+49M</td><td>2.13</td><td>274.4</td><td>0.81</td><td>0.60</td></tr><tr><td>ProMoE-XL (Wei et al., 2026)</td><td>1000</td><td>1.6B+49M</td><td>675M+49M</td><td>2.59</td><td>265.6</td><td></td><td></td></tr><tr><td>SharpMoE-L (Deng et al., 2026)</td><td>1200</td><td>2.0B+49M</td><td>470M+49M</td><td>3.10</td><td>228.9</td><td>一</td><td>1</td></tr><tr><td colspan="8">Pixel-space models: Dense</td></tr><tr><td>ADM (Dhariwal &amp; Nichol, 2021)</td><td>400</td><td>554M</td><td>554M</td><td>4.59</td><td>186.7</td><td>0.82</td><td>0.52</td></tr><tr><td>RIN (Jabri et al., 2023)</td><td>480</td><td>410M</td><td>410M</td><td>3.42</td><td>182.0</td><td></td><td></td></tr><tr><td>Simple Diffusion (Hoogeboom et al., 2023)</td><td></td><td>2.0B</td><td>2.0B</td><td>2.44</td><td>256.3</td><td></td><td></td></tr><tr><td>JetFormer-L (Tschannen et al., 2025)</td><td>500</td><td>2.8B</td><td>2.8B</td><td>6.64</td><td></td><td>0.69</td><td>0.56</td></tr><tr><td>FractalMAR-H (Li et al., 2025)</td><td>600</td><td>848M</td><td>848M</td><td>6.15</td><td>348.9</td><td>0.81</td><td>0.46</td></tr><tr><td>PixelFlow-XL (Chen et al., 2025)</td><td>320</td><td>677M</td><td>677M</td><td>1.98</td><td>282.1</td><td>0.81</td><td>0.60</td></tr><tr><td>PixNerd-XL (Wang et al., 2026a)</td><td>320</td><td>700M</td><td>700M</td><td>1.93</td><td>298.0</td><td>0.80</td><td>0.60</td></tr><tr><td>DiP-XL (Chen et al., 2026b) JiT-G (Li &amp; He, 2026)</td><td>600</td><td>631M</td><td>631M 2.0B</td><td>1.79 1.82</td><td>281.9</td><td>0.80</td><td>0.63</td></tr><tr><td>PixelGen-XL (Ma et al., 2026b)</td><td>600 160</td><td>2.0B 676M</td><td>676M</td><td>1.83</td><td>292.6 293.6</td><td>0.79</td><td>0.62</td></tr><tr><td>PixelREPA-H (Shin et al., 2026)</td><td>600</td><td>953M</td><td>953M</td><td>1.81</td><td>317.2</td><td>0.79</td><td>0.63</td></tr><tr><td>DeCo-XL (Ma et al., 2026a)</td><td>600</td><td>682M</td><td>682M</td><td>1.69</td><td>304.0</td><td>0.79</td><td></td></tr><tr><td>PixelDiT-XL (Yu et al., 2026)</td><td>320</td><td>797M</td><td>797M</td><td>1.61</td><td>292.7</td><td>0.78</td><td>0.63</td></tr><tr><td>HyperDiT-H (He et al., 2026)</td><td>600</td><td>952M</td><td>952M</td><td>1.56</td><td>306.5</td><td>0.80</td><td>0.64</td></tr><tr><td>AsymFlow-H (Chen et al., 2026a)</td><td>600</td><td>953M</td><td>953M</td><td>1.57</td><td></td><td></td><td>0.64</td></tr><tr><td colspan="8">Pixel-space models: Sparse MoE</td></tr><tr><td>JiTMoE (Liu et al., 2025)</td><td>200</td><td>1.3B</td><td>465M</td><td>3.10</td><td>311.3</td><td></td><td></td></tr><tr><td>PixelMoE-B-E32A4</td><td>320</td><td>461M</td><td>225M</td><td>1.88</td><td>298.5</td><td>0.79</td><td>0.61</td></tr><tr><td>PixelMoE-L-E32A4</td><td>320</td><td>1.65B</td><td>632M</td><td>1.54</td><td>315.9</td><td>0.81</td><td>0.61</td></tr><tr><td>PixelMoE-H-E32A4</td><td>320</td><td>3.68B</td><td>1.47B</td><td>1.47</td><td>302.2</td><td>0.79</td><td>0.64</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Class-conditional generation results on ImageNet-1K at 256 × 256. For latent-space methods, parameter counts include the generative model and the tokenizer decoder used at inference.

We next examine whether this sparse scaling behavior of PixelMoE extends to higher-resolution pixel-space generation. As shown in Table 4, all PixelMoE models retain the same E32A4 sparse MoE design at 512×512 resolution. PixelMoE-B-E32A4 achieves an FID of 2.05, while PixelMoE-L-E32A4 and PixelMoE-H-E32A4 further improve the FID to 1.58 and 1.53, respectively. Despite the substantially increased dimensionality of the pixel-space generation problem, PixelMoE maintains strong generation performance across model scales and remains competitive with representative dense pixel-space and latent-space generative models. These results further demonstrate that the sparse scaling behavior of PixelMoE extends from 256 × 256 to higher-resolution pixel-space generation.

Qualitative results are shown in Figure 8. PixelMoE generates coherent images across diverse ImageNet-1K categories, with clear global structures and rich local textures. These samples further illustrate the strong visual fidelity of PixelMoE.

## 4.4 ABLATION STUDIES

We conduct ablation studies on the key components of PixelMoE. The sparse MoE design and explicit routing guidance are evaluated on PixelMoE-L-E32A4, while the hierarchical flow schedule, stage-aware timestep sampling, and DDT prediction head are evaluated on PixelMoE-B-E32A4. Unless otherwise specified, the experiments follow the default training recipe; for the hierarchical flow comparison, we adjust the number of training iterations to match the total number of processed image tokens.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Epochs</td><td rowspan="2">#Params</td><td rowspan="2">#Active Params</td><td colspan="4">ImageNet-1K Generation @ 5122</td></tr><tr><td>FID↓</td><td>IS↑</td><td>Precision↑</td><td>Recall↑</td></tr><tr><td>Latent-space models: Dense</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DiT-XL (Peebles &amp; Xie, 2023)</td><td>1400</td><td>675M+49M675M+49M</td><td></td><td>3.04</td><td>240.8</td><td>0.84</td><td>0.54</td></tr><tr><td>SiT-XL (Ma et al., 2024)</td><td>1400</td><td>675M+49M675M+49M</td><td></td><td>2.62</td><td>252.2</td><td>0.84</td><td>0.57</td></tr><tr><td>MaskDiT (Zheng et al., 2024)</td><td>1600</td><td>675M+49M675M+49M</td><td></td><td>2.50</td><td>256.2</td><td>0.83</td><td>0.56</td></tr><tr><td>REPA (Yu et al., 2025)</td><td>800</td><td>675M+49M</td><td>675M+49M</td><td>2.08</td><td>274.6</td><td>0.83</td><td>0.58</td></tr><tr><td>FlowDCN-XL (Wang et al., 2024)</td><td>400</td><td>618M+49M618M+49M</td><td></td><td>2.44</td><td>252.8</td><td>0.84</td><td>0.54</td></tr><tr><td>Latent-space models: Sparse MoE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DiT-MoE-XL (Fei et al., 2024)</td><td>一</td><td>4.1B+49M</td><td>1.5B+49M</td><td>2.30</td><td>298.4</td><td>0.85</td><td>0.57</td></tr><tr><td>Pixel-space models: Dense</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ADM (Dhariwal &amp; Nichol, 2021)</td><td>400</td><td>554M</td><td>554M</td><td>3.85</td><td>221.7</td><td>0.84</td><td>0.53</td></tr><tr><td>RIN (Jabri et al., 2023)</td><td>480</td><td>410M</td><td>410M</td><td>3.95</td><td>216.0</td><td></td><td></td></tr><tr><td>Simple Diffusion (Hoogeboom et al., 2023)</td><td></td><td>2.0B</td><td>2.0B</td><td>3.02</td><td>248.7</td><td></td><td></td></tr><tr><td>PixNerd-XL (Wang et al., 2026a)</td><td>320</td><td>700M</td><td>700M</td><td>2.84</td><td>245.6</td><td>0.80</td><td>0.59</td></tr><tr><td>DiP-XL (Chen et al., 2026b)</td><td>600</td><td>631M</td><td>631M</td><td>2.31</td><td>291.7</td><td>0.84</td><td>0.58</td></tr><tr><td>PixelDiT-XL (Yu et al., 2026)</td><td>320</td><td>797M</td><td>797M</td><td>1.81</td><td>278.6</td><td>0.78</td><td>0.67</td></tr><tr><td>JiT-G (Li &amp; He, 2026)</td><td>600</td><td>2.0B</td><td>2.0B</td><td>1.78</td><td>306.8</td><td></td><td></td></tr><tr><td>DeCo-XL (Ma et al., 2026a)</td><td>340</td><td>682M</td><td>682M</td><td>2.22</td><td>290.0</td><td>0.80</td><td>0.60</td></tr><tr><td>Pixel-space models: Sparse MoE (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PixelMoE-B-E32A4</td><td>320</td><td>463M</td><td>227M</td><td>2.05</td><td>317.3</td><td>0.80</td><td>0.61</td></tr><tr><td>PixelMoE-L-E32A4</td><td>320</td><td>1.65B</td><td>634M</td><td>1.58</td><td>330.5</td><td>0.81</td><td>0.61</td></tr><tr><td>PixelMoE-H-E32A4</td><td>320</td><td>3.68B</td><td>1.47B</td><td>1.53</td><td>322.2</td><td>0.80</td><td>0.63</td></tr></table>

Table 4: Class-conditional generation results on ImageNet-1K at $5 1 2 \times 5 1 2 .$ For latent-space methods, parameter counts include the generative model and the tokenizer decoder used at inference.  
![](images/1be42ffea2bfe01e860e485edb7ec4bfe02ad3ba7ff2796f7a7a138463695563.jpg)  
Figure 8: Qualitative results of PixelMoE. Samples at $2 5 6 \times 2 5 6$ and $5 1 2 \times 5 1 2$ resolutions.

The Shared Expert. We first examine the role of the shared expert in Table 5a. Under matched active parameter counts, introducing a shared expert improves FID from 2.22 to 1.78, while reducing the total parameter count from 2.00B to 1.65B. This suggests that combining shared and routed computation is more effective than allocating the entire active parameter budget to routed experts.

<table><tr><td>Exp.</td><td>#Params</td><td>#Active</td><td>FID↓</td></tr><tr><td>w/o Shared Expert</td><td>2.00B</td><td>632M</td><td>2.22</td></tr><tr><td>w/ Shared Expert</td><td>1.65B</td><td>632M</td><td>1.78</td></tr></table>

<table><tr><td>Exp.</td><td>#Expert Comb.</td><td>#Params</td><td>#Active</td><td>FID↓</td></tr><tr><td>PixelMoE-L-E8A1</td><td>8</td><td>1.65B</td><td>632M</td><td>1.95</td></tr><tr><td>PixelMoE-L-E32A4</td><td>35,960</td><td>1.65B</td><td>632M</td><td>1.78</td></tr></table>

(a) Effect of the shared expert.
<table><tr><td>Exp.</td><td>#Params</td><td>#Active</td><td>FID↓</td></tr><tr><td>Auxiliary Loss</td><td>1.65B</td><td>632M</td><td>2.05</td></tr><tr><td>Balancing Bias</td><td>1.65B</td><td>632M</td><td>1.78</td></tr></table>

(c) Comparison of load-balancing strategies.

(b) Ablation on fine-grained routed experts.
<table><tr><td>Exp.</td><td>Type</td><td>#Params</td><td>#Active</td><td>FID↓</td></tr><tr><td>Dense-L</td><td>Dense</td><td>622M</td><td>622M</td><td>2.95</td></tr><tr><td>DiffMoE-L-E32A4</td><td>EC</td><td>1.67B</td><td>653M</td><td>2.62</td></tr><tr><td>PixelMoE-L-E32A4</td><td>TC</td><td>1.65B</td><td>632M</td><td>1.78</td></tr></table>

(d) Token-Choice vs. Expert-Choice routing.

Table 5: Ablation studies of the sparse MoE design in PixelMoE-L-E32A4.
<table><tr><td>Routing Strategy</td><td>#Params</td><td>#Active</td><td>FID↓</td></tr><tr><td>+ Timestep Routing</td><td>1.65B</td><td>633M</td><td>2.10</td></tr><tr><td>+ Semantic Routing</td><td>1.65B</td><td>632M</td><td>1.85</td></tr><tr><td>PixelMoE-L-E32A4</td><td>1.65B</td><td>632M</td><td>1.78</td></tr></table>

(a) Explicit routing guidance.

<table><tr><td>Generation Schedule</td><td>#Params</td><td>#Active</td><td>FID↓</td></tr><tr><td>Single-stage flow</td><td>461M</td><td>225M</td><td>2.54</td></tr><tr><td>Hierarchical flow</td><td>461M</td><td>225M</td><td>2.46</td></tr></table>

<table><tr><td>Timestep Sampling</td><td></td><td>#Params #Active FID↓</td><td></td></tr><tr><td>Global Distribution</td><td>461M</td><td>225M</td><td>2.65</td></tr><tr><td>Stage-Aware Sampling</td><td>461M</td><td>225M</td><td>2.46</td></tr></table>

(c) Timestep sampling strategy.

(b) Hierarchical flow schedule.
<table><tr><td>Prediction Head</td><td>#Params</td><td>#Active</td><td>FID↓</td></tr><tr><td>w/o DDT Head</td><td>369M</td><td>133M</td><td>3.76</td></tr><tr><td>w/ DDT Head</td><td>461M</td><td>225M</td><td>2.46</td></tr></table>

(d) DDT prediction head.  
Table 6: Ablation studies of routing and generation components in PixelMoE. Explicit routing guidance is evaluated on PixelMoE-L-E32A4, while the remaining ablations are conducted on PixelMoE-B-E32A4.

Fine-Grained Routed Experts. We next study the effect of expert granularity under matched total and active parameter counts. As shown in Table 5b, changing the expert configuration from E8A1 to E32A4 expands the number of possible expert combinations from 8 to 35,960 without increasing either total or active parameter count. The corresponding FID improvement from 1.95 to 1.78 indicates that PixelMoE benefits from finer-grained expert composition.

Bias-Based Load-Balancing Strategy. The bias-based load-balancing strategy offers a simple but effective way to regulate expert utilization without introducing an auxiliary loss. As shown in Table 5c, the bias-based strategy improves FID from 2.05 to 1.78 compared with an auxiliary loadbalancing loss, with identical total and active parameter counts. Together with the expert-utilization analysis in Figure 7, this result indicates that PixelMoE can preserve flexible input-adaptive routing while maintaining balanced global expert utilization.

Token-Choice Routing Strategy. We compare Token-Choice and Expert-Choice routing to examine which routing paradigm is better suited to pixel-space diffusion. We implement DiffMoE (Shi et al., 2025), which adopts Expert-Choice routing, using the same pixel-space backbone and training recipe as PixelMoE while keeping the total and active parameter counts comparable. As shown in Table 5d, Token-Choice routing achieves a substantially lower FID. This suggests that tokencentric expert selection better accommodates the heterogeneous pixel-token representations across semantic classes and along the denoising trajectory.

Explicit Routing Guidance. A natural question is whether expert routing requires explicit timestep or semantic information. For timestep-aware routing, we provide the timestep embedding directly to the router together with the token hidden states; for semantics-aware routing, we follow the semantic routing design of ProMoE (Wei et al., 2026). As shown in Table 6a, neither variant improves upon the default PixelMoE router, which makes routing decisions solely from token hidden states. Together with the expert specialization observed across semantic classes and along the denoising trajectory in Figure 6, these results suggest that token hidden states in pixel-space diffusion already contain rich timestep and semantic cues that support effective routing without explicit guidance.

Hierarchical Flow Schedule. The pixel-space hierarchical flow schedule progressively changes the spatial granularity along the denoising trajectory. We compare it with a single-stage counterpart that performs the entire denoising process at full resolution. For a fair comparison, we adjust the number of training iterations such that both variants process the same total number of image tokens. As shown in Table 6b, hierarchical flow improves FID from 2.54 to 2.46. This result shows the benefit of adapting spatial granularity along the denoising trajectory by using coarser representations for early denoising and finer representations for later refinement.

Stage-Aware Timestep Sampling. The two hierarchical stages denoise at different spatial resolutions, motivating stage-aware timestep sampling. The baseline uses a single global Logit-Normal distribution $( \mu _ { \mathrm { g } } , \delta _ { \mathrm { g } } ) = ( 0 . 8 , 0 . 8 )$ to sample timesteps over the complete denoising trajectory. In contrast, our default strategy uses the same global distribution for stage allocation and independently samples the stage-relative timestep σ within each stage from a Logit-Normal distribution with $( \mu _ { 1 } , \dot { \delta _ { 1 } } ) = \mathsf { \bar { ( 0 . 6 , 0 . 8 ) } }$ . As shown in Table 6c, this stage-aware sampling strategy improves FID from 2.65 to 2.46, allowing the timestep distribution within each stage to be controlled independently of how frequently that stage is sampled.

DDT Prediction Head. We further ablate the Decoupled Diffusion Transformer (DDT) prediction head (Wang et al., 2026b) on PixelMoE-B-E32A4. Without DDT, the backbone features are mapped directly to the high-dimensional pixel-space prediction, whereas the default model introduces a wider prediction pathway after the backbone. As shown in Table 6d, the DDT head improves FID from 3.76 to 2.46. This result supports the use of the DDT prediction head in our final configuration.

## 5 CONCLUSION

This paper presents PixelMoE, a pixel-space image generation framework that sparsely scales model capacity by directly adopting an LLM-style Mixture-of-Experts (MoE) architecture. We further incorporate Pixel-Space Hierarchical Flow Matching to progressively increase spatial resolution along the denoising trajectory. Experiments on ImageNet-1K show that PixelMoE consistently outperforms its dense counterparts across model scales and achieves competitive performance relative to leading latent- and pixel-space generative models. Our routing analyses further reveal clear expert specialization across semantic classes and along the denoising trajectory under standard Token-Choice routing, without explicit semantic or timestep routing guidance. These results support LLMstyle sparse MoE as an effective scaling paradigm for pixel-space generative modeling.

## REFERENCES

Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.

Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=B1xsqj09Fm.

Arantxa Casanova, Marlene Careil, Jakob Verbeek, Michal Drozdzal, and Adriana Romero Soriano. Instance-conditioned GAN. Advances in Neural Information Processing Systems, 34: 27517–27529, 2021.

Hansheng Chen, Jan Ackermann, Minseo Kim, Gordon Wetzstein, and Leonidas Guibas. Asymmetric flow models. arXiv preprint arXiv:2605.12964, 2026a.

Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. PixelFlow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025.

Zhennan Chen, Junwei Zhu, Xu Chen, Jiangning Zhang, Xiaobin Hu, Hanzhen Zhao, Chengjie Wang, Jian Yang, and Ying Tai. DiP: Taming diffusion models in pixel space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 36136–36146, 2026b.

Kun Cheng, Xiao He, Lei Yu, Zhijun Tu, Mingrui Zhu, Nannan Wang, Xinbo Gao, and Jie Hu. Diff-moe: Diffusion transformer with time-aware and space-adaptive experts. In Forty-second International Conference on Machine Learning, 2025.

Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd annual meeting of the associationfor computational linguistics (volume 1: Long papers), pp. 1280–1297, 2024.

Haoyou Deng, Keyu Yan, Chaojie Mao, Xiang Wang, Yu Liu, Changxin Gao, and Nong Sang. Focusing on what matters: Saliency-harnessing accurate routing for diffusion moe. arXiv preprint arXiv:2606.26938, 2026.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. IEEE, 2009.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Scaling diffusion transformers to 16 billion parameters. arXiv preprint arXiv:2407.11633, 2024.

Zhida Feng, Zhenyu Zhang, Xintong Yu, Yewei Fang, Lanxin Li, Xuyi Chen, Yuxiang Lu, Jiaxiang Liu, Weichong Yin, Shikun Feng, et al. Ernie-vilg 2.0: Improving text-to-image diffusion model with knowledge-enhanced mixture-of-denoising-experts. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10135–10145. IEEE, 2023.

Alireza Ganjdanesh, Yan Kang, Yuchen Liu, Richard Zhang, Zhe Lin, and Heng Huang. Mixture of efficient diffusion experts through automatic interval and sub-network selection. In European Conference on Computer Vision, pp. 54–71. Springer, 2024.

Zipeng Guo, Lichen Ma, Yu He, Xiaolong Fu, Jingling Fu, Junshi Huang, and Yan Li. PixelU: A u-shaped transformer for efficient end-to-end pixel diffusion. arXiv preprint arXiv:2606.27760, 2026.

Yu He, Lichen Ma, Zipeng Guo, Xinyuan Shan, Jingling Fu, Dong Chen, Junshi Huang, and Yan Li. HyperDiT: Hyper-connected transformers for high-fidelity pixel-space diffusion. arXiv preprint arXiv:2605.15741, 2026.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. URL https://openreview. net/forum?id=qw8AKxfYbI.

Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022.

Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. Simple Diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pp. 13213–13232. PMLR, 2023.

Allan Jabri, David J. Fleet, and Ting Chen. Scalable adaptive computation for iterative generation. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 14569–14589. PMLR, 23–29 Jul 2023.

Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991.

Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. In International Conference on Learning Representations, volume 2025, pp. 23378–23402, 2025.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https: //kellerjordan.github.io/posts/muon/.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the ELBO with simple data augmentation. Advances in Neural Information Processing Systems, 36:65484–65516, 2023.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Tuomas Kynka¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved¨ precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019.

Tuomas Kynka¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. ¨ Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Advances in Neural Information Processing Systems, 37:122458–122483, 2024.

Yunsung Lee, JinYoung Kim, Hyojun Go, Myeongho Jeong, Shinhyeok Oh, and Seungtaek Choi. Multi-architecture multi-expert diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 13427–13436, 2024.

Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 36115– 36125, 2026.

Tianhong Li, Qinyi Sun, Lijie Fan, and Kaiming He. Fractal generative models. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/ forum?id=Qk9kn6lOlW.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Repre sentations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=XVjTT1nw5z.

Yahui Liu, Yang Yue, Jingyuan Zhang, Chenxi Sun, Yang Zhou, Wencong Zeng, Ruiming Tang, and Guorui Zhou. Efficient training of diffusion mixture-of-experts models: A practical recipe. arXiv preprint arXiv:2512.01252, 2025.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. SiT: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pp. 23–40. Springer, 2024.

Zehong Ma, Longhui Wei, Shuai Wang, Shiliang Zhang, and Qi Tian. DeCo: Frequency-decoupled pixel diffusion for end-to-end image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 43600–43610, 2026a.

Zehong Ma, Ruihan Xu, and Shiliang Zhang. PixelGen: Improving pixel diffusion with perceptual supervision. arXiv preprint arXiv:2602.02493, 2026b.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182. IEEE, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 10674–10685. IEEE, 2022.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. Advances in neural information processing systems, 29, 2016.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017. URL https://openreview. net/forum?id=B1ckMDqlg.

Minglei Shi, Ziyang Yuan, Haotian Yang, Xintao Wang, Mingwu Zheng, Xin Tao, Wenliang Zhao, Wenzhao Zheng, Jie Zhou, Jiwen Lu, et al. DiffMoE: Dynamic token selection for scalable diffusion transformers. arXiv preprint arXiv:2503.14487, 2025.

Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Jiwen Lu. Latent diffusion model without variational autoencoder. In International Conference on Learning Representations, volume 2026, pp. 154506–154537, 2026.

Jaeyo Shin, Jiwook Kim, and Hyunjung Shim. Representation alignment for just image transformers is not easier than you think. arXiv preprint arXiv:2603.14366, 2026.

Haotian Sun, Tao Lei, Bowen Zhang, Yanghao Li, Haoshuo Huang, Ruoming Pang, Bo Dai, and Nan Du. EC-DiT: Scaling diffusion transformers with adaptive expert-choice routing. In International Conference on Learning Representations, volume 2025, pp. 72383–72401, 2025.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Michael Tschannen, Andre Susano Pinto, and Alexander Kolesnikov. JetFormer: An autoregressive´ generative model of raw images and text. In International Conference on Learning Representations, volume 2025, pp. 85995–86016, 2025.

Shuai Wang, Zexian Li, Tianhui Song, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Exploring dcn-like architecture for fast image generation with arbitrary resolution. Advances in Neural Information Processing Systems, 37:87959–87977, 2024.

Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. PixNerd: Pixel neural field diffusion. In International Conference on Learning Representations, volume 2026, pp. 43559– 43580, 2026a.

Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. DDT: Decoupled diffusion transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 40633–40642, 2026b.

Yujie Wei, Shiwei Zhang, Hangjie Yuan, Yujin Han, Zhekai Chen, Jiayu Wang, Difan Zou, Xihui Liu, Yu Liu, Hongming Shan, et al. Routing matters in moe: Scaling diffusion transformers with explicit routing guidance. In International Conference on Learning Representations, volume 2026, pp. 23381–23405, 2026.

Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-to-image generation via large mixture of diffusion paths. Advances in Neural Information Processing Systems, 36:41693–41706, 2023.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=DJSZGGZYVi.

Yongsheng Yu, Wei Xiong, Weili Nie, Yichen Sheng, Shiqiu Liu, and Jiebo Luo. PixelDiT: Pixel diffusion transformers for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14273–14282, 2026.

Yike Yuan, Ziyu Wang, Zihao Huang, Defa Zhu, Xun Zhou, Jingyi Yu, and Qiyang Min. Expert race: A flexible routing strategy for scaling diffusion transformer with mixture of experts. In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview. net/forum?id=9W3CSuHoFA.

Jack Zhang, Noah Amsel, Berlin Chen, and Tri Dao. Gram Newton-Schulz, 2026. URL https: //dao-ailab.github.io/blog/2026/gram-newton-schulz/.

Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=vTBjBtGioE.

## A ADDITIONAL DETAILS OF BIAS-BASED LOAD BALANCING

We provide the detailed computation of the expert-load distribution used in the bias-based loadbalancing update. For image token i, the assignment to expert e is defined as

$$
f _ { t , i , e } ^ { s } = \left\{ \begin{array} { l l } { \frac { 1 } { A } , } & { e \in S _ { t , i } ^ { s } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{14}
$$

where $S _ { t , i } ^ { s }$ is the set of A routed experts selected for image token i. Averaging the assignments over all $N$ image tokens in the training batch gives the observed load fraction $F _ { e }$ for expert e:

$$
F _ { e } = \frac { 1 } { N } \sum _ { ( s , t , i ) \in \mathcal { T } } f _ { t , i , e } ^ { s } ,\tag{15}
$$

where $\tau$ denotes the image-token indices in the batch. Since each image token contributes a total assignment mass of one, $\begin{array} { r } { \sum _ { e = 1 } ^ { E } F _ { e } = 1 } \end{array}$

The uniform load target is $\mathbf { Q } = ( 1 / E , \dots , 1 / E )$ . For the bias update in Section 3.1, RMS normalization is defined as

$$
\mathrm { R M S } ( \mathbf { z } ) = \sqrt { \frac { 1 } { E } \sum _ { e = 1 } ^ { E } z _ { e } ^ { 2 } } .\tag{16}
$$

Accordingly, the balancing-bias vector is updated as

$$
\mathbf b \gets \mathbf b - \gamma \frac { \mathbf F - \mathbf Q } { \mathrm { R M S } ( \mathbf F - \mathbf Q ) } .\tag{17}
$$

We use $\gamma = 0 . 0 0 1$ throughout training. A small epsilon of $1 0 ^ { - 6 }$ is added to the denominator for numerical stability. The balancing bias is updated without gradient propagation and affects expert selection only; the original routing scores are used to compute the routing weights.

## B SCALE TRANSITION IN PIXEL-SPACE HIERARCHICAL FLOW MATCHING

During sampling, simply upsampling the low-resolution endpoint produced by Stage 0 does not yield a valid Stage 1 starting state, since nearest-neighbor upsampling replicates the noise associated with each low-resolution pixel and consequently introduces local correlations that are absent from the Stage 1 starting distribution used during training. We therefore follow Pyramidal Flow Matching (Jin et al., 2025) and construct a distribution-preserving transition from the Stage 0 endpoint $\hat { x } _ { \theta } ^ { e _ { 0 } }$ to the Stage 1 starting state $\hat { x } _ { \theta } ^ { s _ { 1 } }$ . Specifically, the Stage 0 endpoint and the predefined Stage 1 starting state are given by

$$
\begin{array} { r l } & { x ^ { e _ { 0 } } = ( 1 - t _ { e _ { 0 } } ) \mathcal { D } ( x _ { 0 } ) + t _ { e _ { 0 } } \epsilon _ { 0 } , } \\ & { x ^ { s _ { 1 } } = ( 1 - t _ { s _ { 1 } } ) \widetilde { x } _ { 0 } + t _ { s _ { 1 } } \epsilon _ { 1 } , \qquad \widetilde { x } _ { 0 } = \mathcal { U } ( \mathcal { D } ( x _ { 0 } ) ) , } \end{array}\tag{18}
$$

where $\epsilon _ { 0 } , \epsilon _ { 1 } \sim { \mathcal { N } } ( 0 , \delta _ { n } ^ { 2 } I )$

Assuming an ideal Stage 0 prediction, upsampling its endpoint gives

$$
\mathcal { U } ( x ^ { e _ { 0 } } ) = ( 1 - t _ { e _ { 0 } } ) \widetilde { x } _ { 0 } + t _ { e _ { 0 } } \mathcal { U } ( \epsilon _ { 0 } ) .\tag{19}
$$

To match the signal component of the predefined Stage 1 starting state, we first rescale the upsampled endpoint:

$$
\frac { 1 - t _ { s _ { 1 } } } { 1 - t _ { e _ { 0 } } } \mathcal { U } ( x ^ { e _ { 0 } } ) = ( 1 - t _ { s _ { 1 } } ) \widetilde { x } _ { 0 } + \beta \mathcal { U } ( \epsilon _ { 0 } ) ,\tag{20}
$$

where

$$
\beta = \frac { 1 - t _ { s _ { 1 } } } { 1 - t _ { e _ { 0 } } } t _ { e _ { 0 } } .\tag{21}
$$

The signal coefficient now matches that of $x ^ { s _ { 1 } }$ , while the remaining difference lies in the noise covariance.

For $2 \times$ nearest-neighbor upsampling, each low-resolution noise value is copied to a $2 \times 2$ block. Therefore, after factoring out the noise variance $\delta _ { n } ^ { 2 }$ , the within-block covariance of $\boldsymbol { \mathcal { U } } ( \boldsymbol { \epsilon } _ { 0 } )$ is

$$
\Sigma _ { \mathrm { u p } } = \left[ \begin{array} { l l l l } { 1 } & { 1 } & { 1 } & { 1 } \\ { 1 } & { 1 } & { 1 } & { 1 } \\ { 1 } & { 1 } & { 1 } & { 1 } \\ { 1 } & { 1 } & { 1 } & { 1 } \end{array} \right] .\tag{22}
$$

In contrast, the noise component $t _ { s _ { 1 } } \epsilon _ { 1 }$ of the Stage 1 starting state has covariance $t _ { s _ { 1 } } ^ { 2 } \delta _ { n } ^ { 2 } I _ { 4 }$ , where $I _ { 4 } \in \mathbb { R } ^ { 4 \times 4 }$ denotes the identity matrix over the four pixels within each $2 \times 2$ block. We therefore introduce a block-correlated Gaussian perturbation $\epsilon ^ { \prime } ,$ , sampled independently of the Stage 0 noise, and construct

$$
\hat { x } _ { \theta } ^ { s _ { 1 } } = \frac { 1 - t _ { s _ { 1 } } } { 1 - t _ { e _ { 0 } } } \mathcal { U } ( \hat { x } _ { \theta } ^ { e _ { 0 } } ) + \alpha \epsilon ^ { \prime } ,\tag{23}
$$

where

$$
\epsilon ^ { \prime } \sim \mathcal { N } ( 0 , \delta _ { n } ^ { 2 } \Sigma _ { \rho } ) ,\tag{24}
$$

and

$$
\Sigma _ { \rho } = \left[ { \begin{array} { l l l l } { 1 } & { \rho } & { \rho } & { \rho } \\ { \rho } & { 1 } & { \rho } & { \rho } \\ { \rho } & { \rho } & { 1 } & { \rho } \\ { \rho } & { \rho } & { \rho } & { 1 } \end{array} } \right] .\tag{25}
$$

For $\Sigma _ { \rho }$ to define a valid covariance matrix, it must be positive semidefinite. Its eigenvalues are $1 - \rho$ with multiplicity three and $1 + 3 \rho$ with multiplicity one, which requires $- { \frac { 1 } { 3 } } \leq \rho \leq 1$ . Since the additional noise is used to cancel the positive within-block correlations introduced by nearestneighbor upsampling, we further restrict $\rho \ : { \sf t o } - \frac { 1 } { 3 } \leq \rho \leq 0$

The actual within-block covariance of the resulting noise component in Equation (23) is

$$
\delta _ { n } ^ { 2 } \left( \beta ^ { 2 } \Sigma _ { \mathrm { u p } } + \alpha ^ { 2 } \Sigma _ { \rho } \right) .\tag{26}
$$

Matching this covariance to the target covariance $\delta _ { n } ^ { 2 } t _ { s _ { 1 } } ^ { 2 } I _ { 4 }$ of the predefined Stage 1 starting distribution and canceling the common factor $\delta _ { n } ^ { 2 }$ yields

$$
\beta ^ { 2 } \Sigma _ { \mathrm { u p } } + \alpha ^ { 2 } \Sigma _ { \rho } = t _ { s _ { 1 } } ^ { 2 } I _ { 4 } .\tag{27}
$$

Matching the diagonal and off-diagonal entries gives

$$
\beta ^ { 2 } + \alpha ^ { 2 } = t _ { s _ { 1 } } ^ { 2 } ,\tag{28}
$$

$$
\beta ^ { 2 } + \rho \alpha ^ { 2 } = 0 .\tag{29}
$$

Solving the two constraints yields

$$
\alpha ^ { 2 } = \frac { t _ { s _ { 1 } } ^ { 2 } } { 1 - \rho } , \qquad \beta ^ { 2 } = - \frac { \rho } { 1 - \rho } t _ { s _ { 1 } } ^ { 2 } .\tag{30}
$$

Within the admissible range $\rho \in [ - 1 / 3 , 0 ]$ , a more negative $\rho$ decreases $\alpha ^ { 2 }$ , i.e., the variance coefficient of the additionally injected noise, while increasing the proportion of noise inherited from the upsampled Stage 0 endpoint. To make the transition minimally perturbative to the Stage 0 prediction, we minimize the variance of the newly injected noise subject to the covariance-matching constraint. Following Pyramidal Flow Matching (Jin et al., 2025), we therefore choose the most negative admissible value, $\rho = - 1 / 3$ . Substituting it into Equation (30) gives

$$
\alpha = { \frac { \sqrt { 3 } } { 2 } } t _ { s _ { 1 } } , \qquad \beta = { \frac { 1 } { 2 } } t _ { s _ { 1 } } .\tag{31}
$$

Combining the expression for $\beta$ with its definition in Equation (21) determines the relation between the two stage-boundary noise levels:

$$
\frac { 1 - t _ { s _ { 1 } } } { 1 - t _ { e _ { 0 } } } t _ { e _ { 0 } } = \frac { t _ { s _ { 1 } } } { 2 } ,\tag{32}
$$

which gives

$$
t _ { e _ { 0 } } = \frac { t _ { s _ { 1 } } } { 2 - t _ { s _ { 1 } } } .\tag{33}
$$

Thus, the Stage 0 endpoint and Stage 1 starting noise levels cannot be chosen independently; specifying either one determines the other under the covariance-matching transition.

The resulting cross-stage transition used during sampling is therefore

$$
\hat { x } _ { \theta } ^ { s _ { 1 } } = \frac { 1 - t _ { s _ { 1 } } } { 1 - t _ { e _ { 0 } } } \mathcal { U } ( \hat { x } _ { \theta } ^ { e _ { 0 } } ) + \frac { \sqrt { 3 } } { 2 } t _ { s _ { 1 } } \epsilon ^ { \prime } , \qquad \epsilon ^ { \prime } \sim \mathcal { N } \big ( 0 , \delta _ { n } ^ { 2 } \Sigma _ { - 1 / 3 } \big ) .\tag{34}
$$

This transition preserves the coarse structure predicted by Stage 0 while matching the noise statistics of the predefined Stage 1 starting distribution and minimizing the amount of newly injected noise under the covariance-matching constraint.

Sampling the Correlated Noise. The required block-correlated Gaussian noise can be sampled directly. For each $2 \times 2$ spatial block and channel, we draw $\mathbf { z } \sim \mathcal { N } ( 0 , I _ { 4 } )$ and compute $\bar { z } = \textstyle { \frac { 1 } { 4 } } \mathbf { i } ^ { \top } \mathbf { z } ,$ where $\mathbf { 1 } \in \mathbb { R } ^ { 4 }$ denotes the all-ones vector. Then we set

$$
\epsilon ^ { \prime } = \delta _ { n } \frac { 2 } { \sqrt { 3 } } \left( { \bf z } - \bar { z } { \bf 1 } \right) .\tag{35}
$$

This construction satisfies

$$
\mathrm { C o v } ( \epsilon ^ { \prime } ) = \delta _ { n } ^ { 2 } \Sigma _ { - 1 / 3 } ,\tag{36}
$$

providing a simple implementation of the correlated re-noising used in the cross-stage transition.

## C PSEUDO-CODE OF PIXELMOE

We provide simplified PyTorch-like pseudo-code for the two key components of PixelMoE: the sparse MoE layer in Algorithm 1 and the two-stage pixel-space hierarchical flow matching schedule in Algorithm 2. In Algorithm 2, the heun solver evaluates velocity at global timesteps t but updates the state using stage-relative increments $\Delta \sigma = \Delta t / ( t _ { s _ { s } } - t _ { e _ { s } } )$

Algorithm 1 Sparse MoE Layer in PixelMoE: PyTorch-like Pseudo-code Algorithm 1 Sparse MoE Layer in PixelMoE: PyTorch-like Pseudo-code

```python
class PixelMoELayer(nn.Module):
def __init__(
self, dim, routed_dim, shared_dim,
E=32, A=4, gamma=1e-3
):
self.router = nn.Linear(dim, E, bias=False)
self.experts = nn.ModuleList(
[Expert(routed_dim) for _ in range(E)])
self.shared_expert = Expert(shared_dim)
self.router_bias = torch.zeros(E)
self.A = A
self.router_scale = math.sqrt(A)
self.gamma = gamma
def forward(self, x):
scores = torch.sigmoid(self.router(x))
# Balancing bias is used only for expert selection
expert_ids = torch.topk(
scores + self.router_bias,
self.A, dim=-1
).indices
# Normalize the original routing scores
weights = gather(scores, expert_ids)
weights = weights / weights.sum(
dim=-1, keepdim=True)
weights = self.router_scale <sub>*</sub> weights
routed = sparse_expert_forward(
x, self.experts, expert_ids, weights)
shared = self.shared_expert(x)
return routed + shared
@torch.no_grad()
def update_router_bias(self, load):
target = load.mean()
delta = load - target
self.router_bias -= (
self.gamma delta / rms(delta)
)
```

Algorithm 2 Pixel-Space Hierarchical Flow Matching: PyTorch-like Pseudo-code

```python
class HierarchicalFlow(nn.Module):
def __init__(self, model):
self.model = model # shared PixelMoE
def training_loss(self, x, c):
stage = sample_stage()
if stage == 0:
x_low = downsample(x)
noise = delta_n torch.randn_like(x_low)
x_start = noise
x_end = (1 - t_e0) <sub>*</sub> x_low + t_e0 <sub>*</sub> noise
t_start, t_end = 1.0, t_e0
else:
x_coarse = upsample(downsample(x))
noise = delta_n <sub>*</sub> torch.randn_like(x)
x_start = (1 - t_s1) <sub>*</sub> x_coarse + t_s1 <sub>*</sub> noise
x_end = x
t_start, t_end = t_s1, 0.0
# Stage-relative timestep
sigma = sample_timestep(stage)
x_sigma = sigma <sub>*</sub> x_start + (1 - sigma) <sub>*</sub> x_end
# Corresponding global timestep
t = t_end + sigma <sub>*</sub> (t_start - t_end)
# Stage-wise endpoint prediction
x_end_pred = self.model(x_sigma, t, c)
sigma_safe = sigma.clamp_min(sigma_min)
loss = ((x_end_pred - x_end) / sigma_safe) 2
return loss.mean()
def velocity(self, x, t, c, stage):
if stage == 0:
t_start, t_end = 1.0, t_e0
else:
t_start, t_end = t_s1, 0.0
# Convert global timestep to stage-relative timestep
sigma = (t - t_end) / (t_start - t_end)
# Convert endpoint prediction to velocity
x_end_pred = self.model(x, t, c)
sigma_safe = sigma.clamp_min(sigma_min)
return (x - x_end_pred) / sigma_safe
@torch.no_grad()
def sample(self, c):
# Stage 0: low-resolution generation
x = delta_n <sub>*</sub> torch.randn(low_res_shape)
x = heun_solver(self.velocity, x, c, stage=0)
# Distribution-preserving stage transition
x = stage_transition(x)
# Stage 1: full-resolution refinement
x = heun_solver(self.velocity, x, c, stage=1)
return x
```

## D MODEL CONFIGURATION

We implement PixelMoE at Base, Large, and Huge scales, with the detailed configurations summarized in Table 7. For 256 × 256 generation, all models use a patch size of 8; we use a patch size of 16 at 512 × 512. All models adopt the E32A4 MoE configuration unless otherwise specified, where each image token activates 4 experts from a pool of 32 routed experts. As the backbone scales from Base to Huge, we increase its depth and hidden dimension together with the capacity of both routed and shared experts. A wide DDT head is appended to each backbone for stage-wise endpoint prediction, using 2, 2, and 4 blocks for the Base, Large, and Huge models, respectively. Across all scales, the first four backbone blocks remain dense, while the remaining backbone blocks adopt sparse MoE layers; the DDT head retains dense FFNs.

Active Parameter Counting. In MoE blocks, image tokens are processed by the sparse MoE FFN, while class tokens use a separate dense FFN. We report the average active parameter count per token, weighted according to the respective numbers of image and class tokens during training.

## E ADDITIONAL TRAINING DETAILS

We summarize the detailed training configurations of PixelMoE in Table 8. Unless otherwise specified, the same training recipe is used across model scales. For $5 1 2 \times 5 1 2$ generation, we increase the patch size from 8 to 16 and the noise scale from 1.0 to 2.0, while keeping the remaining optimization, hierarchical flow, and MoE settings unchanged.

Controlled Dense Baselines. For controlled comparisons, each dense baseline follows the same backbone configuration, hierarchical flow schedule, DDT prediction head, optimizer, timestep sampling, and training recipe as the corresponding PixelMoE model. The sparse MoE layers are replaced by dense FFNs with comparable active parameter counts, while all other architectural and training settings are kept unchanged.

Regularization. For the final PixelMoE-H-E32A4 models, we introduce additional regularization after epoch 200. Following LlamaGen (Sun et al., 2024), we use a crop range of 1.1, where each training image is first resized to 1.1× the target resolution and then randomly cropped to the target size. We additionally apply dropout with a rate of 0.2 to the output of each self-attention module and feed-forward network. These modifications increase the diversity of training views and provide stronger regularization during the later stage of training.

Optimization. We use gMuon (Zhang et al., 2026) for eligible matrix-valued parameters, including the routed-expert matrices, while AdaLN layers, the final prediction layer, timestep and class embedders, and the remaining parameters are optimized with AdamW (Loshchilov & Hutter, 2019). We use a learning rate of $2 \times \bar { 1 } 0 ^ { - 4 }$ with a global batch size of 1024. After a 5-epoch warm-up, we keep the learning rate constant and apply cosine decay to $2 \times 1 0 ^ { - 5 }$ over the final 40 epochs. We train with bfloat16 mixed precision and maintain two EMA copies with decay rates 0.9996 and 0.9999.

Hierarchical Flow Training. We use the same two-stage pixel-space hierarchical flow matching schedule for both resolutions, with Stage 0 denoising at half spatial resolution and Stage 1 at full resolution. We set the Stage 0 endpoint to $t _ { e _ { 0 } } = 0 . 7 \bar { 5 }$ and use $\rho = - 1 / 3$ for the noise correlation. Under the cross-stage transition derived in Section B, these settings determine the Stage 1 starting noise level as $t _ { s _ { 1 } } ~ = ~ 6 / 7$ The relative sampling frequency of the two stages is determined by a global Logit-Normal distribution with $( \mu _ { \mathrm { g } } , \bar { \delta } _ { \mathrm { g } } ) \bar { ~ } = ~ ( \bar { 0 } . 8 , \bar { 0 . 8 } )$ , with the probability mass in the overlapping stage interval divided equally between the two stages. The stage-relative timestep is then independently sampled from a Logit-Normal distribution with $( \mu _ { 1 } , \delta _ { 1 } ) \stackrel { - } { = } ( 0 . 6 , 0 . 8 )$ . Under the resulting stage intervals, this yields sampling probabilities $p ( s = 0 ) \stackrel { . } { \approx } 0 . 2 3 1$ and $p ( s = 1 ) \approx 0 . 7 6 9$

Stage-wise Endpoint Prediction Objective. PixelMoE directly predicts the endpoint of each stage while using the velocity-space flow matching loss introduced in Section 3.2. For numerical stability near the stage endpoint, we clamp the inverse-square weighting using $\sigma _ { \mathrm { m i n } } = 0 . 0 5$ . The

<table><tr><td colspan="12">Model Depth Hidden Dim. Heads DDT Depth DDT Dim. Routed Dim. Shared Dim. #Dense</td></tr><tr><td>PixelMoE-B-E32A4</td><td>12</td><td>768</td><td>12</td><td>2</td><td>1536</td><td>384</td><td></td><td>632</td><td>4</td></tr><tr><td>PixelMoE-L-E32A4</td><td>24</td><td>1024</td><td>16</td><td>2</td><td>2048</td><td>512</td><td></td><td>836</td><td>4</td></tr><tr><td>PixelMoE-H-E32A4</td><td>32</td><td>1280</td><td>16</td><td>4</td><td></td><td>2560</td><td>640</td><td>1048</td><td>4</td></tr></table>

Table 7: PixelMoE model configurations. DDT Dim. represents the hidden state dimension in the DDT head. Routed Dim. and Shared Dim. denote the intermediate dimensions of the routed-expert and shared-expert feed-forward networks (FFNs), respectively. #Dense denotes the number of dense blocks in the PixelMoE backbone.

implemented objective is

$$
\mathcal { L } _ { \mathrm { v - e n d p o i n t } } = \mathbb { E } \left[ \frac { \mathcal { L } _ { \mathrm { M S E } } \left( \hat { x } _ { \theta } ^ { e _ { s } } , x ^ { e _ { s } } \right) } { \operatorname* { m a x } ( \sigma , \sigma _ { \mathrm { m i n } } ) ^ { 2 } } \right] ,\tag{37}
$$

where the expectation is taken over images, sampled stages, stage-relative timesteps, and Gaussian noise. The inverse-square weighting is not an additional loss-reweighting heuristic. Before applying the numerical clamp, the endpoint parameterization gives

$$
v _ { \theta } - v ^ { s } = \frac { x ^ { e _ { s } } - \hat { x } _ { \theta } ^ { e _ { s } } } { \sigma } ,\tag{38}
$$

so velocity-space MSE directly yields the $1 / \sigma ^ { 2 }$ weighting on endpoint-prediction error. The clamp at $\sigma _ { \mathrm { m i n } }$ only prevents this factor from becoming singular near the stage endpoint.

MoE Training. The router operates in FP32 and converts the routing logits into sigmoid scores. For each image token, we select the top-A experts according to the bias-adjusted scores, normalize the original sigmoid scores over the selected experts, and multiply the resulting routing weights by a scale factor of $\sqrt { A }$ before expert aggregation. The balancing bias only affects expert selection and is not used when computing the expert routing weights. After each optimizer update, expert loads accumulated during the corresponding forward passes are used to update the balancing bias with an RMS-normalized correction and a step size of 0.001. No auxiliary load-balancing objective is used for the default PixelMoE models.

Auxiliary Load-Balancing Baseline. For the auxiliary load-balancing loss baseline in Table ${ 5 \mathrm { c } } .$ we replace the bias-based balancing strategy with the Token-Choice load-balancing loss (Fedus et al., 2022; Liu et al., 2024). Specifically, let $r _ { i , e }$ denote the sigmoid routing score of image token i for expert e, and let $s _ { i }$ denote its selected top-A experts. We first normalize the routing scores over all experts as

$$
p _ { i , e } = \frac { r _ { i , e } } { \sum _ { j = 1 } ^ { E } r _ { i , j } } .\tag{39}
$$

For each expert, we define the mean routing probability and assignment fraction as

$$
P _ { e } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } p _ { i , e } , \qquad F _ { e } = \frac { 1 } { N A } \sum _ { i = 1 } ^ { N } { \bf 1 } [ e \in S _ { i } ] ,\tag{40}
$$

where $N$ is the number of image tokens. The load-balancing loss for the ℓ-th MoE layer is

$$
\mathcal { L } _ { \mathrm { b a l } } ^ { ( \ell ) } = E \sum _ { e = 1 } ^ { E } \mathrm { s g } ( F _ { e } ) P _ { e } ,\tag{41}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient. We average this loss across all MoE layers and add it to the training objective:

$$
\mathcal { L } _ { \mathrm { a u x } } = \lambda _ { \mathrm { a u x } } \cdot \frac { 1 } { L _ { \mathrm { M o E } } } \sum _ { \ell = 1 } ^ { L _ { \mathrm { M o E } } } \mathcal { L } _ { \mathrm { b a l } } ^ { ( \ell ) } ,\tag{42}
$$

where $L _ { \mathrm { M o E } }$ denotes the number of MoE layers. Following standard practice in prior MoE models (Fedus et al., 2022; Dai et al., 2024), we set $\lambda _ { \mathrm { a u x } } = 0 . 0 1$ . For this baseline, the training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { v - e n d p o i n t } } + \mathcal { L } _ { \mathrm { a u x } } . } \end{array}\tag{43}
$$

<table><tr><td rowspan="2">Configuration</td><td colspan="3">256 × 256</td><td colspan="3"> $5 1 2 \times 5 1 2$ </td></tr><tr><td>Base</td><td>Large</td><td>Huge</td><td>Base</td><td>Large</td><td>Huge</td></tr><tr><td colspan="7">Input and training</td></tr><tr><td>Training resolution</td><td> $2 5 6 ^ { 2 }$ </td><td> $2 5 6 ^ { 2 }$ </td><td> $2 5 6 ^ { 2 }$ </td><td> $5 1 2 ^ { 2 }$ </td><td> $5 1 2 ^ { 2 }$ </td><td> $5 1 2 ^ { 2 }$ </td></tr><tr><td>Patch size</td><td>8</td><td>8</td><td>8</td><td>16</td><td>16</td><td>16</td></tr><tr><td>Noise scale</td><td>1.0</td><td>1.0</td><td>1.0</td><td>2.0</td><td>2.0</td><td>2.0</td></tr><tr><td>Crop range</td><td>一</td><td>一</td><td> $1 . 1 ^ { \dagger }$ </td><td></td><td>一</td><td> $1 . 1 ^ { \dagger }$ </td></tr><tr><td>Dropout</td><td>0</td><td>0</td><td> $0 . 2 ^ { \dagger }$ </td><td>0</td><td>0</td><td> $0 . 2 ^ { \dagger }$ </td></tr><tr><td>Label dropout</td><td colspan="7">0.1</td></tr><tr><td>Epochs</td><td colspan="6">100 (default for scaling/ablation/routing), 320 (benchmarking)</td></tr><tr><td>Global batch size</td><td colspan="6">1024</td></tr><tr><td>Random horizontal flip</td><td colspan="6">0.5</td></tr><tr><td>Optimization Optimizer</td><td colspan="6"> $\mathrm { g M u o n + A d a m W }$ </td></tr><tr><td>Learning rate Warm-up epochs</td><td colspan="6"> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR schedule</td><td colspan="6">5 Constant + cosine tail</td></tr><tr><td>Cosine decay epochs</td><td colspan="6"> $4 0$ </td></tr><tr><td>Minimum learning rate</td><td colspan="6"> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay</td><td colspan="6">0</td></tr><tr><td>gMuon momentum</td><td colspan="6"></td></tr><tr><td></td><td colspan="6">0.95</td></tr><tr><td>AdamW  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td colspan="6">(0.9,0.95)</td></tr><tr><td>Precision</td><td colspan="6">bfloat16</td></tr><tr><td>EMA decay</td><td colspan="6">0.9999, 0.9996</td></tr><tr><td>Hierarchical flow training</td><td colspan="6"></td></tr><tr><td colspan="6">Number of stages</td></tr><tr><td>Stage resolutions</td><td colspan="6">2  $1 2 8 ^ { 2 } , 2 5 6 ^ { 2 } 1 2 8 ^ { 2 } , 2 5 6 ^ { 2 } 1 2 8 ^ { 2 } , 2 5 6 ^ { 2 } 2 5 6 ^ { 2 } 2 5 6 ^ { 2 } , 5 1 2 ^ { 2 } 2 5 6 ^ { 2 } , 5 1 2 ^ { 2 } 2 5 6 ^ { 2 } , 5 1 2 ^ { 2 }$ </td></tr><tr><td>Stage 0 endpoint  $t _ { e _ { 0 } }$ </td><td colspan="6">0.75</td></tr><tr><td>Stage 1 start  $t _ { s _ { 1 } }$ </td><td colspan="6"> $6 / 7$ </td></tr><tr><td>Noise correlation  $\rho$ </td><td colspan="6"> $- 1 / 3$ </td></tr><tr><td>Global timestep distribution</td><td colspan="6">Logit-Normal  $( \mu _ { \mathrm { g } } = 0 . 8 , \delta _ { \mathrm { g } } = 0 . 8 )$ </td></tr><tr><td>Stage-relative distribution</td><td colspan="6">Logit-Normal  $( \mu _ { 1 } = 0 . 6 , \delta _ { 1 } = 0 . 8 )$ </td></tr><tr><td>Endpoint-loss clamp  $\sigma _ { \mathrm { m i n } }$ </td><td colspan="6">0.05</td></tr><tr><td>MoE training</td><td colspan="6"></td></tr><tr><td>Router score</td><td colspan="6">Sigmoid</td></tr><tr><td>Router precision</td><td colspan="6">FP32</td></tr><tr><td>Routing scale</td><td colspan="6"> $\sqrt { A }$ </td></tr><tr><td></td><td colspan="6"></td></tr><tr><td>Load balancing</td><td colspan="6">Bias-based</td></tr><tr><td>Bias update rate</td><td colspan="6">0.001 RMS-normalized</td></tr></table>

Table 8: Training configurations of PixelMoE. For $5 1 2 \times 5 1 2$ generation, we increase the patch size from 8 to 16 and the Gaussian noise scale from 1.0 to 2.0. <sup>†</sup> For the final PixelMoE-H-E32A4 models, LlamaGen-style crop augmentation with a crop range of 1.1 and dropout with a rate of 0.2 are applied after epoch 200; the 100-epoch controlled experiments use the standard preprocessing without these additional regularization measures.

## F EVALUATION DETAILS

We follow the evaluation setup of JiT (Li & He, 2026) for FID and IS, including its ImageNet-1K reference statistics, and use the ADM (Dhariwal & Nichol, 2021) implementation for Precision and Recall. We summarize our sampling protocol in Table 9. During sampling, we construct a 50-point uniform global timestep grid, include the stage boundaries, and use the grid points within the timestep interval of each stage. For class-conditional ImageNet-1K generation, class labels are uniformly distributed over the 1,000 classes across distributed workers.

<table><tr><td>Configuration</td><td>Setting</td></tr><tr><td>Sampling solver Global timestep grid Global grid spacing</td><td>Heun 50 Uniform</td></tr><tr><td>Timestep shifting CFG scale CFG interval</td><td>None Swept per model Swept per model</td></tr><tr><td>EMA candidates Selection samples</td><td>0.9996, 0.9999 10,000</td></tr><tr><td>Final evaluation samples</td><td>50,000</td></tr><tr><td>Class sampling Metrics</td><td>Uniform over 1,000 classes FID, IS, Precision, Recall</td></tr></table>

Table 9: Evaluation configurations. CFG scale, CFG interval, and EMA decay are jointly selected for each model according to FID-10K, and the selected configuration is fixed for the final FID-50K evaluation.

Classifier-Free Guidance. We apply classifier-free guidance (Ho & Salimans, 2021) as

$$
v _ { \mathrm { c f g } } ( x _ { \sigma } ^ { s } , t , c ) = v _ { \theta } ( x _ { \sigma } ^ { s } , t , \phi ) + w ( t ) \left( v _ { \theta } ( x _ { \sigma } ^ { s } , t , c ) - v _ { \theta } ( x _ { \sigma } ^ { s } , t , \phi ) \right) ,\tag{44}
$$

where c and ϕ denote the class and null conditions, respectively, and $w ( t )$ is controlled by the CFG scale w and the CFG interval (Kynka¨anniemi et al., 2024). For an interval¨ $[ t _ { \operatorname* { m i n } } , t _ { \operatorname* { m a x } } ]$ , we define

$$
w ( t ) = { \left\{ \begin{array} { l l } { w , } & { t _ { \mathrm { m i n } } < t < t _ { \mathrm { m a x } } , } \\ { 1 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{45}
$$

Thus, CFG is applied only within the selected portion of the sampling trajectory. Outside this interval, sampling uses the standard conditional prediction without CFG amplification.

Model Selection. For every evaluated model, we jointly sweep the CFG scale, CFG interval, and the two EMA checkpoints. Each candidate configuration is first evaluated with 10,000 samples, and the configuration achieving the lowest FID-10K score is selected. We then regenerate 50,000 samples using the selected configuration and report all final generation metrics on this sample set.

## G MORE VISUALIZATION RESULTS

We provide additional uncurated samples of PixelMoE-H-E32A4 on ImageNet-1K. Figures 9 and 10 show samples at 256×256, while Figure 11 presents samples at $5 1 2 \times 5 1 2$ . All samples are generated conditioned on the specified classes using the same sampling settings as the reported quantitative results.

![](images/8cc9d9e41bb1e70a29c3fc2ee9950f0c98e6a3ae492c074dbf873ceeb8684e09.jpg)  
Class 012: House finch, Linnet, Carpodacus mexicanus

![](images/efd9d5ad48f24292ac28e6205ae4302e0dd6707bfd50630bd79e7ec67599aa7f.jpg)  
Class 042: Agama

![](images/fee084e59aaef3c9f7b4a0e6ffe8d0d87de0843a809ac2c2819a3817ff515b20.jpg)  
Class 107: Jellyfish

![](images/8c8d479dc819644132903ecc006bc56927835f7f595e2aa1c59b9fe52147e3e8.jpg)  
Class 110: Flatworm, Platyhelminth

![](images/0fce3b7a46cc5d4456c4a1e9e2dcefe69dd56c9404d1792ad8df97a95e237a1b.jpg)  
Class 130: Flamingo

![](images/1d8d1460bbff66f994bbd70357cf2eb1565017ff75b5c1230748de9825e7b939.jpg)  
Class 288: Leopard, Panthera pardus

![](images/5cf243a5f6d34b8190bd6f6fdfc0a8c5cba14bbb6e871cb5919fbdbde94ce8b9.jpg)  
Class 349: Bighorn, Bighorn sheep, Cimarron, Rocky Mountain

![](images/c462054c5fd8b8b18b7ec93021d3f316dec5554da4baf8519d8ad590ae8c4411.jpg)  
Class 425: Barn

bighorn  
![](images/b31374a639e183ab9a452a7da8ad2ac70fc9df93358dca4b1a5643a484e3fe0d.jpg)  
Class 453: Bookcase

![](images/d4d2edc62a9cb8a3af2bc22f1145cc3fada6cf56cfe8444bdd4b6e5e2446d170.jpg)  
Class 495: China cabinet, China closet  
Figure 9: Uncurated samples of PixelMoE-H-E32A4 on ImageNet-1K at 256 × 256. Samples are generated conditioned on the specified classes using the same sampling configuration as the reported quantitative results.

![](images/fe819a3b30c6a11d6ef0a42439b804d0f684da8b9440d63e4930f2463ea80f52.jpg)  
Class 658: Mitten

![](images/321c955ebd70e7bc8e933bdf549568f1bf44e904c49532d0dc74a9729510fc38.jpg)  
Class 718: Pier

![](images/d29175cbb2ab5f4b9ded4f44967378fc899a8e9ca7ebe5e0de7455a1fbf6d962.jpg)  
Class 725: Pitcher, Ewer

![](images/bd3d36ca2888d255e28c246a4edf4ee8efd1e25e983e59acda9c40476a488570.jpg)  
Class 779: School bus

![](images/ea4981652c6e577738e802e3e33ebf411dcec0f35dd5f0ffd339f043b22a0666.jpg)  
Class 829: Streetcar, Tram, Tramcar, Trolley, Trolley car

![](images/7f0c2da7063e21bfbe1e78b3005dd3441590d8cfc9cb094e46101b3ec43924f1.jpg)  
Class 873: Triumphal arch

![](images/a2ff9440cbe62c4ad257475bbc1fa0a859a9059b7ba84be5f85a17cb62b5c416.jpg)  
Class 911: Wool, Woolen, Woollen

![](images/8517e2ab3c573229932e4aca13e8dc9ec9dd35d6c8335d862d36e6febd032d20.jpg)

![](images/3dbdf99674b3c95024de264c18db7563b21c37d5d11e7e04f473094623c96f03.jpg)  
Class 946: Cardoon

Class 927: Trifle  
![](images/e969bb351828ceed85bb33b26fbeb7247e5e73488bff5688c83c0d6dbfeeb691.jpg)  
Class 975: Lakeside, Lakeshore

Figure 10: Uncurated samples of PixelMoE-H-E32A4 on ImageNet-1K at 256 × 256. Samples are generated conditioned on the specified classes using the same sampling configuration as the reported quantitative results.

![](images/f38b06bd8c803ddc4efbee30efa9ab2ffb9aa818c78afc991923474ea6ceba90.jpg)  
Class 014: Indigo bunting, Indigo finch, Indigo bird, Passerina  
cyanea

![](images/f1127bc55aee1fdcea1a9f3d8c140a12f59217c9482eaecf4d03fa3a4860568a.jpg)  
Class 117: Chambered nautilus, Pearly nautilus, Nautilus

![](images/849bd38eebc5903a91b3bdeef18a95e2829661c9630c6d7974296ee4233cb7eb.jpg)  
Class 309: Bee

![](images/dcb68a9f4e46621fab6b8e4150a57ec048ec5d4b5914fa7d7f2ac4c3b0bdf158.jpg)  
Class 458: Brass, Memorial tablet, Plaque

![](images/796aeac5317f11af2fba515bd28c5ad3f20d6d8b3268c7478bfb594737cb1269.jpg)  
Class 724: Pirate, Pirate ship

![](images/9f186c2b72451a902533f0b46586166a2d3c8ac776cf3b2237fa88486f6729ae.jpg)

![](images/af4bddaacbabb579fac75e848b5e6ff46db35e017cd9966a820417bc5476c0ce.jpg)  
Class 913: Wreck

Class 853: Thatch, Thatched roof  
![](images/0e37228495753da762a501325123406e040d5f561f1e085da7ad1ff2f6c25049.jpg)  
Class 989: Hip, Rose hip, Rosehip

Figure 11: Uncurated samples of PixelMoE-H-E32A4 on ImageNet-1K at 512 × 512. Samples are generated conditioned on the specified classes using the same sampling configuration as the reported quantitative results.